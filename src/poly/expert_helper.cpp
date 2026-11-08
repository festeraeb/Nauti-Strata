// src/poly/expert_helper.cpp - strata-expert-helper: routed experts in this machine's VRAM, computed for an engine
// on another machine (expert_wire.hpp).
//
//   strata-expert-helper --model FILE.gguf --listen [ADDR:]PORT [--devices 0,1] [--reserve-mib N]
//                        [--ram-mib N --rdma DEV:PORT:GID]
//
// The model is this machine's own copy of the engine's file (the same bytes; its path may differ).  Each device
// takes experts until its free VRAM, less the reserve, is used.  An engine's Hold names the experts it wants
// here, warmest first; the helper reads them from the file into VRAM and answers with those it holds.  A Compute
// is one layer's tokens: their inputs, and the held experts each was routed to with their weights; the answer is
// each token's weighted sum, computed with the kernels the engine's own VRAM experts use (native_mmvq over the
// GGUF blocks), so an expert gives here what it would give in the engine's VRAM.
//
// With --ram-mib the helper also keeps a RAM window: pinned host memory registered for RDMA READ (--rdma names the
// RoCE port).  A Hold that the devices cannot take, from an engine that asked for RAM (wire::kWantRam), goes into
// the window, read from the file once; the engine reads those experts' bytes over RoCE on a miss and computes them
// itself.  RAM costs less than VRAM, so the window holds far more: Le Chonk's misses came from t440's SSD at 72 ms
// an expert, and from its RAM by RDMA they take 8.4 ms (docs/REMOTE_TIER_PLAN.md).  `--devices ""` gives a helper
// without GPUs that only holds RAM (the plan's c1).
#include "expert_rdma.hpp"
#include "expert_wire.hpp"
#include "ram_window.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace {

using namespace strata;
namespace wire = strata::poly::wire;
using Clock = std::chrono::steady_clock;

constexpr int kSlots = 8;      // experts a launch multiplies (MmvqSlots)
constexpr int kSumRows = 32;   // experts one add_scaled adds (ScaledRows)
constexpr size_t kPad = 256;   // past an expert's down: the kernels may read a little beyond a matrix

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// A MoE layer's routed experts in the file.
struct LayerW {
    bool moe = false;
    uint32_t gu_type = 0, d_type = 0;
    size_t gu_bytes = 0, gu_stride = 0, d_bytes = 0;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
};

struct Model {
    std::unique_ptr<GgufModel> g;
    std::vector<LayerW> layers;
    int n_expert = 0, n_embd = 0, n_ff = 0;
    uint64_t bytes = 0;
    uint64_t key = 0;

    explicit Model(const std::string& path) {
        g = std::make_unique<GgufModel>(GgufModel::open(path));
        for (size_t s = 0; s < g->size(); ++s) bytes += g->shard(s).file_size();
        auto data = [&](const TensorInfo& t) {
            size_t s = 0;
            g->find(t.name, &s);
            return g->shard(s).tensor_data(t);
        };
        int max_layer = -1;
        for (size_t s = 0; s < g->size(); ++s)
            for (const TensorInfo& t : g->shard(s).tensors())
                if (t.name.rfind("blk.", 0) == 0) max_layer = std::max(max_layer, std::atoi(t.name.c_str() + 4));
        layers.resize((size_t) max_layer + 1);
        for (int il = 0; il <= max_layer; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const TensorInfo* td = g->find(b + "ffn_down_exps.weight");
            if (!td || td->shape.size() != 3) continue;
            LayerW& L = layers[(size_t) il];
            const int ne = (int) td->shape[2];
            L.moe = true;
            L.d_type = td->type;
            L.d_bytes = tensor_payload_bytes(*td) / (size_t) ne;
            L.down = data(*td);
            if (const TensorInfo* gu = g->find(b + "ffn_gate_up_exps.weight")) {
                L.gu_type = gu->type;
                L.gu_stride = tensor_payload_bytes(*gu) / (size_t) ne;
                L.gu_bytes = L.gu_stride / 2;
                L.gate = data(*gu);
                L.up = L.gate + L.gu_bytes;
                n_ff = (int) gu->shape[1] / 2;
            } else {
                const TensorInfo* tg = g->find(b + "ffn_gate_exps.weight");
                const TensorInfo* tu = g->find(b + "ffn_up_exps.weight");
                if (!tg || !tu || tg->type != tu->type)
                    throw std::runtime_error(b + "ffn_gate/up_exps: missing, or of two types");
                L.gu_type = tg->type;
                L.gu_stride = L.gu_bytes = tensor_payload_bytes(*tg) / (size_t) ne;
                L.gate = data(*tg);
                L.up = data(*tu);
                n_ff = (int) tg->shape[1];
            }
            n_expert = ne;
            n_embd = (int) td->shape[1];
            if (!kernels::native_mmvq_supported((int) L.gu_type) || !kernels::native_mmvq_supported((int) L.d_type))
                throw std::runtime_error(b + "ffn_*_exps: a type the GPU kernels do not take");
        }
        if (n_expert == 0) throw std::runtime_error(path + " has no routed experts");
        key = wire::model_key(bytes, n_expert, n_embd, n_ff);
    }
    size_t expert_bytes(int il) const { return 2 * layers[(size_t) il].gu_bytes + layers[(size_t) il].d_bytes; }
};

constexpr int kInRam = -1;   // Expert::dev of one in the RAM window

struct Expert {
    int dev = -1;           // a device's id, or kInRam
    uint8_t* w = nullptr;   // gate | up | down, as an engine's VRAM slot (kInRam: in the window, unpadded)
    size_t bytes = 0;       // what it takes of its device's budget, or its place in the window
    uint64_t off = 0;       // kInRam: its place's offset in the window
    int refs = 0;
};

// What a connection asked for and set up: RAM holds, and the queue pair it reads them through.
struct Session {
    bool want_ram = false;
    std::unique_ptr<poly::rdma::Qp> qp;
};

// A device: its budget, and one Compute at a time on its stream with its scratch.
struct Device {
    int id = 0;
    uint64_t budget = 0, used = 0;
    cudaStream_t st = nullptr;
    std::mutex mu;
    int rows_cap = 0;
    float *x = nullptr, *g = nullptr, *h = nullptr, *d = nullptr, *out = nullptr, *host = nullptr;
    void *xq = nullptr, *q8 = nullptr;

    void scratch(int rows, int n_embd, int n_ff) {
        if (rows <= rows_cap) return;
        for (void* p : {(void*) x, (void*) g, (void*) h, (void*) d, (void*) out, xq, q8})
            if (p) cudaFree(p);
        if (host) cudaFreeHost(host);
        const size_t e = (size_t) n_embd, f = (size_t) n_ff;
        ck(cudaMalloc(&x, (size_t) rows * e * 4), "scratch");
        ck(cudaMalloc(&xq, (size_t) rows * kernels::native_q8_1_bytes(n_embd)), "scratch");
        ck(cudaMalloc(&g, (size_t) kSlots * 2 * f * 4), "scratch");
        ck(cudaMalloc(&h, (size_t) kSlots * f * 4), "scratch");
        ck(cudaMalloc(&q8, (size_t) kSlots * kernels::native_q8_1_bytes(n_ff)), "scratch");
        ck(cudaMalloc(&d, (size_t) kSumRows * e * 4), "scratch");
        ck(cudaMalloc(&out, (size_t) rows * e * 4), "scratch");
        ck(cudaMallocHost(&host, (size_t) rows * e * 4), "scratch");
        rows_cap = rows;
    }
};

class Helper {
public:
    Helper(const std::string& model, const std::vector<int>& devices, int reserve_mib, int ram_mib,
           const std::string& rdma_port)
        : m_(model) {
        for (int id : devices) {
            auto d = std::make_unique<Device>();
            d->id = id;
            ck(cudaSetDevice(id), "cudaSetDevice");
            ck(cudaStreamCreateWithFlags(&d->st, cudaStreamNonBlocking), "cudaStreamCreate");
            d->scratch(8, m_.n_embd, m_.n_ff);
            size_t free_b = 0, total_b = 0;
            ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
            const uint64_t reserve = (uint64_t) reserve_mib << 20;
            d->budget = free_b > reserve ? free_b - reserve : 0;
            std::fprintf(stderr, "strata-expert-helper: device %d: %llu MiB for experts\n", id,
                         (unsigned long long) (d->budget >> 20));
            devs_.push_back(std::move(d));
        }
        std::fprintf(stderr, "strata-expert-helper: %d experts a layer, %d x %d, %llu MiB of model\n", m_.n_expert,
                     m_.n_embd, m_.n_ff, (unsigned long long) (m_.bytes >> 20));
        if (ram_mib > 0) open_window((uint64_t) ram_mib << 20, rdma_port);
    }

    ~Helper() {
        if (rdma_) poly::rdma::Device::dereg(window_mr_);
        if (window_) munmap(window_, win_.bytes());
    }

    void serve(wire::Conn c, std::string who) {
        std::vector<std::pair<int32_t, int32_t>> mine;   // this connection's holds, released when it ends
        std::vector<uint8_t> p, staging;
        Session s;
        bool spoke = false;   // a health probe connects and closes without a word: not worth a line
        try {
            for (;;) {
                const uint32_t t = c.recv(p);
                spoke = true;
                try {
                    if (t == wire::kHello) hello(c, p, s);
                    else if (t == wire::kHold) hold(c, p, mine, who, staging, s);
                    else if (t == wire::kRdma) rdma(c, p, s);
                    else if (t == wire::kRelease) release_some(c, p, mine, who);
                    else if (t == wire::kCompute) compute(c, p);
                    else throw std::runtime_error("unknown message " + std::to_string(t));
                } catch (const std::runtime_error& e) {
                    if (!c.open()) throw;
                    const std::string why = e.what();
                    c.send(wire::kErr, why.data(), why.size());
                }
            }
        } catch (const std::exception& e) {
            if (spoke) std::fprintf(stderr, "strata-expert-helper: %s: %s\n", who.c_str(), e.what());
        }
        s.qp.reset();   // no read of this engine's reaches the window once its places can be someone else's
        release(mine, who);
    }

private:
    // The window is allocated, faulted in and locked at start: a page that the kernel could still move or swap
    // cannot be registered, and a fault on the first Hold would cost a Compute its time.  The registration then
    // pins it for the NIC.  No --rdma: the window still holds, and its experts are reported where they are, but an
    // engine only asks for RAM when it can read it, so a window nobody can read is room nobody asks for.
    void open_window(uint64_t bytes, const std::string& rdma_port) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + std::to_string(bytes >> 20) + " MiB for the RAM window");
        if (mlock(p, bytes) != 0)
            std::fprintf(stderr, "strata-expert-helper: cannot lock the RAM window (%s); the kernel may swap it\n",
                         std::strerror(errno));
        window_ = (uint8_t*) p;
        win_ = poly::RamWindow(bytes);
        if (!rdma_port.empty()) {
            rdma_ = std::make_unique<poly::rdma::Device>(poly::rdma::parse_port(rdma_port));
            window_mr_ = rdma_->reg(window_, bytes, true);
        }
        std::fprintf(stderr, "strata-expert-helper: a RAM window of %llu MiB%s\n", (unsigned long long) (bytes >> 20),
                     rdma_ ? (", RDMA-readable on " + rdma_port).c_str() : ", not RDMA-readable (no --rdma)");
    }

    void hello(wire::Conn& c, const std::vector<uint8_t>& p, Session& s) {
        wire::Hello h{};
        if (p.size() != sizeof(h)) throw std::runtime_error("a Hello of the wrong size");
        std::memcpy(&h, p.data(), sizeof(h));
        if (h.model_key != m_.key || (int) h.n_expert != m_.n_expert || (int) h.n_embd != m_.n_embd ||
            (int) h.n_ff != m_.n_ff)
            throw std::runtime_error("this helper has another model (its files' bytes or its experts' shape differ)");
        s.want_ram = (h.flags & wire::kWantRam) && window_;
        wire::HelloOk ok{(uint32_t) devs_.size(), rdma_ ? wire::kHasRdma : 0u, 0, window_ ? win_.bytes() : 0};
        for (const auto& d : devs_) ok.capacity += d->budget;
        c.send(wire::kOk, &ok, sizeof(ok));
    }

    // One queue pair for this connection, on the window's protection domain, connected to the engine's.  The
    // depth is the device's limit of outstanding reads: the engine never has more in flight than it says.
    void rdma(wire::Conn& c, const std::vector<uint8_t>& p, Session& s) {
        if (!rdma_) throw std::runtime_error("this helper has no RDMA-readable RAM window (--ram-mib and --rdma)");
        wire::RdmaQp peer{};
        if (p.size() != sizeof(peer)) throw std::runtime_error("an Rdma of the wrong size");
        std::memcpy(&peer, p.data(), sizeof(peer));
        auto qp = std::make_unique<poly::rdma::Qp>(*rdma_, 64);
        qp->connect(peer);
        wire::RdmaOk ok{};
        ok.qp = qp->local();
        ok.rkey = window_mr_.rkey;
        ok.base = (uint64_t) (uintptr_t) window_;
        ok.bytes = win_.bytes();
        s.qp = std::move(qp);
        c.send(wire::kOk, &ok, sizeof(ok));
    }

    // {u32 n}, `skip` more bytes (a Hold's where), then n Pairs
    static std::vector<wire::Pair> pairs_of(const std::vector<uint8_t>& p, const char* what, size_t skip = 0) {
        uint32_t n = 0;
        if (p.size() < 4 + skip) throw std::runtime_error(std::string("a ") + what + " too short");
        std::memcpy(&n, p.data(), 4);
        if (p.size() != 4 + skip + (size_t) n * sizeof(wire::Pair))
            throw std::runtime_error(std::string("a ") + what + " of the wrong size");
        std::vector<wire::Pair> q(n);
        std::memcpy(q.data(), p.data() + 4 + skip, (size_t) n * sizeof(wire::Pair));
        return q;
    }

    // Hold: the experts asked for, warmest first, where the request allows and there is room: a device first
    // (kAny, kDeviceOnly), else the RAM window (kAny, kRamOnly) for a connection that asked for RAM.  One another
    // connection holds already is shared, wherever it is, unless this one may not have it there.
    void hold(wire::Conn& c, const std::vector<uint8_t>& p, std::vector<std::pair<int32_t, int32_t>>& mine,
              const std::string& who, std::vector<uint8_t>& staging, const Session& s) {
        const std::vector<wire::Pair> want = pairs_of(p, "Hold", 4);
        uint32_t where = 0;
        std::memcpy(&where, p.data() + 4, 4);
        if (where > wire::kRamOnly) throw std::runtime_error("a Hold's where is " + std::to_string(where));
        const bool dev_ok = where != wire::kRamOnly && !devs_.empty();
        const bool ram_ok = where != wire::kDeviceOnly && s.want_ram;
        std::vector<wire::Held> got;
        const auto t0 = Clock::now();
        uint64_t bytes = 0;
        size_t loaded = 0, in_ram = 0;
        for (const wire::Pair& q : want) {
            if (q.layer < 0 || q.layer >= (int) m_.layers.size() || !m_.layers[(size_t) q.layer].moe || q.expert < 0 ||
                q.expert >= m_.n_expert)
                continue;
            const auto key = std::make_pair(q.layer, q.expert);
            int have = 0;   // 1: on a device, 2: in RAM
            const Expert* ex = nullptr;
            bool exists = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = held_.find(key);
                if (it != held_.end()) {
                    exists = true;
                    if (it->second.dev == kInRam ? ram_ok : where != wire::kRamOnly) {
                        ++it->second.refs;
                        ex = &it->second;
                    }
                }
            }
            if (exists && !ex) continue;   // held where this connection may not have it: one copy an expert
            if (!ex) {
                if (!(dev_ok && load(q.layer, q.expert, staging)) && !(ram_ok && load_ram(q.layer, q.expert)))
                    continue;   // no room where it may go
                ++loaded;
                std::lock_guard<std::mutex> lk(mu_);
                ex = &held_.at(key);   // where it is: another connection may have loaded it meanwhile
            }
            have = ex->dev == kInRam ? 2 : 1;
            wire::Held h{q.layer, q.expert, wire::kOnDevice, 0, 0};
            if (have == 2) {
                h.kind = wire::kInRam;
                h.addr = (uint64_t) (uintptr_t) ex->w;   // a held expert's record stays put while it is held
                ++in_ram;
            }
            mine.push_back(key);
            got.push_back(h);
            bytes += m_.expert_bytes(q.layer);
        }
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        std::fprintf(stderr, "strata-expert-helper: %s: holds %zu of %zu experts asked for (%llu MiB, %zu of them in "
                             "RAM; %zu read from the file in %.1f s)\n",
                     who.c_str(), got.size(), want.size(), (unsigned long long) (bytes >> 20), in_ram, loaded, secs);
        const uint32_t m = (uint32_t) got.size();
        c.send(wire::kOk, &m, 4, got.data(), got.size() * sizeof(wire::Held));
    }

    // Release: the connection gives up some of its holds before it ends, so that its engine can take others in
    // their room (the set follows the answer).  Answers how many were its to give.
    void release_some(wire::Conn& c, const std::vector<uint8_t>& p, std::vector<std::pair<int32_t, int32_t>>& mine,
                      const std::string& who) {
        std::vector<std::pair<int32_t, int32_t>> drop;
        for (const wire::Pair& q : pairs_of(p, "Release")) {
            const auto key = std::make_pair(q.layer, q.expert);
            const auto it = std::find(mine.begin(), mine.end(), key);
            if (it == mine.end()) continue;   // not this connection's: nothing to give
            mine.erase(it);
            drop.push_back(key);
        }
        release(drop, who);
        const uint32_t m = (uint32_t) drop.size();
        c.send(wire::kOk, &m, 4);
    }

    // Reads an expert into the device with the most room for it, with a reference.  False: none has room.
    // The file's bytes come into `staging` first, outside any lock: a miss in the page cache is a disk read (77 ms
    // an expert from a SAS SSD), and a Compute on the device must not wait for it.  The copy to the device then
    // holds the device for its own time only.
    bool load(int il, int e, std::vector<uint8_t>& staging) {
        const LayerW& L = m_.layers[(size_t) il];
        const size_t bytes = m_.expert_bytes(il) + kPad;
        Device* best = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& d : devs_)
                if (d->used + bytes <= d->budget && (!best || d->budget - d->used > best->budget - best->used))
                    best = d.get();
            if (!best) return false;
            best->used += bytes;   // claimed before the copy, so that two connections do not both take the room
        }
        const uint8_t* gate = L.gate + (size_t) e * L.gu_stride;
        const uint8_t* up = L.up + (size_t) e * L.gu_stride;
        const uint8_t* down = L.down + (size_t) e * L.d_bytes;
        staging.resize(bytes);
        std::memcpy(staging.data(), gate, L.gu_bytes);
        std::memcpy(staging.data() + L.gu_bytes, up, L.gu_bytes);
        std::memcpy(staging.data() + 2 * L.gu_bytes, down, L.d_bytes);
        std::memset(staging.data() + 2 * L.gu_bytes + L.d_bytes, 0, kPad);
        uint8_t* w = nullptr;
        try {
            std::lock_guard<std::mutex> dl(best->mu);
            ck(cudaSetDevice(best->id), "cudaSetDevice");
            ck(cudaMalloc(&w, bytes), "cudaMalloc");
            ck(cudaMemcpy(w, staging.data(), bytes, cudaMemcpyHostToDevice), "upload");
        } catch (const std::runtime_error& err) {
            if (w) cudaFree(w);
            std::lock_guard<std::mutex> lk(mu_);
            best->used -= bytes;
            std::fprintf(stderr, "strata-expert-helper: blk.%d expert %d: %s\n", il, e, err.what());
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        auto [it, fresh] = held_.try_emplace(std::make_pair(il, e));
        if (!fresh) {   // another connection loaded it meanwhile: keep theirs
            ++it->second.refs;
            best->used -= bytes;
            std::lock_guard<std::mutex> dl(best->mu);
            cudaSetDevice(best->id);
            cudaFree(w);
            return true;
        }
        it->second = Expert{best->id, w, bytes, 1};
        return true;
    }

    // Reads an expert into a place in the RAM window, with a reference.  False: the window has no room.  The place
    // is claimed under the lock and filled outside it (a file read of up to 77 ms); nobody reads it before the Hold
    // answer names it.
    bool load_ram(int il, int e) {
        if (!window_) return false;
        const LayerW& L = m_.layers[(size_t) il];
        const uint64_t want = m_.expert_bytes(il);
        uint64_t off = 0, place = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!win_.take(want, off, place)) return false;
        }
        uint8_t* w = window_ + off;
        std::memcpy(w, L.gate + (size_t) e * L.gu_stride, L.gu_bytes);
        std::memcpy(w + L.gu_bytes, L.up + (size_t) e * L.gu_stride, L.gu_bytes);
        std::memcpy(w + 2 * L.gu_bytes, L.down + (size_t) e * L.d_bytes, L.d_bytes);
        std::lock_guard<std::mutex> lk(mu_);
        auto [it, fresh] = held_.try_emplace(std::make_pair(il, e));
        if (!fresh) {   // another connection loaded it meanwhile: keep theirs
            ++it->second.refs;
            win_.give(off, place);
            return true;
        }
        Expert x;
        x.dev = kInRam;
        x.w = w;
        x.bytes = place;
        x.off = off;
        x.refs = 1;
        it->second = x;
        return true;
    }

    void release(const std::vector<std::pair<int32_t, int32_t>>& mine, const std::string& who) {
        size_t freed = 0;
        uint64_t bytes = 0;
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& key : mine) {
            auto it = held_.find(key);
            if (it == held_.end() || --it->second.refs > 0) continue;
            if (it->second.dev == kInRam) {
                win_.give(it->second.off, it->second.bytes);
                bytes += it->second.bytes;
                ++freed;
                held_.erase(it);
                continue;
            }
            Device* d = dev(it->second.dev);
            {
                std::lock_guard<std::mutex> dl(d->mu);
                cudaSetDevice(d->id);
                cudaFree(it->second.w);
            }
            d->used -= it->second.bytes;
            bytes += it->second.bytes;
            ++freed;
            held_.erase(it);
        }
        if (!mine.empty())
            std::fprintf(stderr, "strata-expert-helper: %s: %zu experts freed (%llu MiB)\n", who.c_str(), freed,
                         (unsigned long long) (bytes >> 20));
    }

    Device* dev(int id) {
        for (auto& d : devs_)
            if (d->id == id) return d.get();
        throw std::runtime_error("no device " + std::to_string(id));
    }

    void compute(wire::Conn& c, const std::vector<uint8_t>& p) {
        wire::ComputeHead h{};
        if (p.size() < sizeof(h)) throw std::runtime_error("a Compute too short");
        std::memcpy(&h, p.data(), sizeof(h));
        const size_t e_row = (size_t) m_.n_embd;
        if (h.rows <= 0 || h.rows > 4096) throw std::runtime_error("a Compute of " + std::to_string(h.rows) + " rows");
        const size_t xb = (size_t) h.rows * e_row * 4;
        if (p.size() != sizeof(h) + xb + (size_t) h.n * sizeof(wire::Entry))
            throw std::runtime_error("a Compute of the wrong size");
        if (h.layer < 0 || h.layer >= (int) m_.layers.size() || !m_.layers[(size_t) h.layer].moe)
            throw std::runtime_error("layer " + std::to_string(h.layer) + " has no routed experts");
        const float* x = (const float*) (p.data() + sizeof(h));
        std::vector<wire::Entry> ents(h.n);
        std::memcpy(ents.data(), p.data() + sizeof(h) + xb, (size_t) h.n * sizeof(wire::Entry));

        // the entries by device, each with its expert's weights
        struct Job {
            const uint8_t* w;
            int row;
            float weight;
        };
        std::map<int, std::vector<Job>> by_dev;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const wire::Entry& en : ents) {
                auto it = held_.find(std::make_pair(h.layer, en.expert));
                if (it == held_.end())
                    throw std::runtime_error("blk." + std::to_string(h.layer) + " expert " + std::to_string(en.expert) +
                                             " is not held here");
                if (it->second.dev == kInRam)   // the engine reads those itself
                    throw std::runtime_error("blk." + std::to_string(h.layer) + " expert " + std::to_string(en.expert) +
                                             " is held in RAM, not on a device");
                if (en.row < 0 || en.row >= h.rows) throw std::runtime_error("an entry's row is out of range");
                by_dev[it->second.dev].push_back(Job{it->second.w, en.row, en.weight});
            }
        }
        // each device's part added up while its lock is held: its host buffer is the next Compute's after that
        std::vector<float> sum((size_t) h.rows * e_row, 0.f);
        for (auto& [id, jobs] : by_dev) {
            Device* d = dev(id);
            std::lock_guard<std::mutex> dl(d->mu);
            ck(cudaSetDevice(d->id), "cudaSetDevice");
            d->scratch(h.rows, m_.n_embd, m_.n_ff);
            run(*d, m_.layers[(size_t) h.layer], h, x, jobs.data(), (int) jobs.size(),
                [](const Job& j) { return j.w; }, [](const Job& j) { return j.row; },
                [](const Job& j) { return j.weight; });
            if (by_dev.size() == 1) std::memcpy(sum.data(), d->host, xb);
            else
                for (size_t i = 0; i < sum.size(); ++i) sum[i] += d->host[i];
        }
        c.send(wire::kOk, sum.data(), xb);
    }

    // The engine's launch() for these experts: kSlots a launch a stage, kSumRows weighted into their rows.
    template <class J, class W, class R, class S>
    void run(Device& d, const LayerW& L, const wire::ComputeHead& h, const float* x, const J* jobs, int n, W w_of,
             R row_of, S weight_of) {
        const int n_embd = m_.n_embd, n_ff = m_.n_ff;
        const size_t e_row = (size_t) n_embd, ff = (size_t) n_ff;
        const size_t xq_row = kernels::native_q8_1_bytes(n_embd), q8_row = kernels::native_q8_1_bytes(n_ff);
        ck(cudaMemcpyAsync(d.x, x, (size_t) h.rows * e_row * 4, cudaMemcpyHostToDevice, d.st), "input");
        for (int r = 0; r < h.rows; ++r)
            kernels::native_quantize_q8_1(d.x + (size_t) r * e_row, (uint8_t*) d.xq + (size_t) r * xq_row, n_embd, 1,
                                          d.st);
        ck(cudaMemsetAsync(d.out, 0, (size_t) h.rows * e_row * 4, d.st), "output");
        poly::ScaledRows sum{};
        for (int i0 = 0; i0 < n; i0 += kSlots) {
            const int m = std::min(kSlots, n - i0);
            kernels::MmvqSlots gate_up{}, down{};
            for (int j = 0; j < m; ++j) {
                const uint8_t* w = w_of(jobs[i0 + j]);
                gate_up.w[j] = w;
                gate_up.x[j] = (const uint8_t*) d.xq + (size_t) row_of(jobs[i0 + j]) * xq_row;
                gate_up.y[j] = d.g + (size_t) j * 2 * ff;
                down.w[j] = w + 2 * L.gu_bytes;
                down.x[j] = (const uint8_t*) d.q8 + (size_t) j * q8_row;
                down.y[j] = d.d + (size_t) ((i0 + j) % kSumRows) * e_row;
            }
            const bool one = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, n_embd, 2 * n_ff, d.st);
            for (int j = 0; j < m && !one; ++j)
                kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], n_embd, 2 * n_ff, 1,
                                     d.st);
            if (h.flags & wire::kGelu) poly::geglu_rows(d.g, d.h, n_ff, m, d.st);
            else poly::swiglu_clamp_rows(d.g, d.h, n_ff, m, h.limit, d.st);
            kernels::native_quantize_q8_1(d.h, d.q8, n_ff, m, d.st);
            if (!(m > 1 && kernels::native_mmvq_slots((int) L.d_type, down, m, n_ff, n_embd, d.st)))
                for (int j = 0; j < m; ++j)
                    kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff, n_embd, 1, d.st);
            for (int j = 0; j < m; ++j) {
                const int k = (i0 + j) % kSumRows;
                sum.x[k] = down.y[j];
                sum.acc[k] = d.out + (size_t) row_of(jobs[i0 + j]) * e_row;
                sum.w[k] = weight_of(jobs[i0 + j]);
                if (k == kSumRows - 1 || i0 + j == n - 1) poly::add_scaled(sum, k + 1, n_embd, d.st);
            }
        }
        ck(cudaMemcpyAsync(d.host, d.out, (size_t) h.rows * e_row * 4, cudaMemcpyDeviceToHost, d.st), "output");
        ck(cudaStreamSynchronize(d.st), "compute");
    }

    Model m_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::mutex mu_;   // held_, the budgets and win_
    std::map<std::pair<int32_t, int32_t>, Expert> held_;
    uint8_t* window_ = nullptr;   // the RAM window (--ram-mib); null: none
    poly::RamWindow win_;
    std::unique_ptr<poly::rdma::Device> rdma_;   // --rdma: the port the window is read through
    poly::rdma::Device::Mr window_mr_;
};

void usage() {
    std::fprintf(stderr,
                 "usage: strata-expert-helper --model FILE.gguf --listen [ADDR:]PORT [--devices 0,1] [--reserve-mib N]\n"
                 "                            [--ram-mib N [--rdma DEV:PORT:GID]]\n"
                 "  --model FILE     this machine's copy of the engine's model (the first file of a split one)\n"
                 "  --listen WHERE   the port engines connect to (--expert-helper HOST:PORT on their side)\n"
                 "  --devices LIST   the CUDA devices to hold experts on (default: all; \"\" none: a RAM-only helper)\n"
                 "  --reserve-mib N  VRAM each device keeps free beside the experts (default 768)\n"
                 "  --ram-mib N      a RAM window of N MiB, pinned, for the experts the devices cannot take, which the\n"
                 "                   engines read by RDMA (default 0: none)\n"
                 "  --rdma DEV:PORT:GID  the RoCE port the window is read through (rocep1s0f1:1:3 on t440)\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, listen;
    std::vector<int> devices;
    bool devices_given = false;
    int reserve_mib = 768, ram_mib = 0;
    std::string rdma_port;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model = val();
        else if (a == "--listen") listen = val();
        else if (a == "--reserve-mib") reserve_mib = std::atoi(val().c_str());
        else if (a == "--ram-mib") ram_mib = std::atoi(val().c_str());
        else if (a == "--rdma") rdma_port = val();
        else if (a == "--devices") {
            devices_given = true;
            const std::string s = val();
            for (size_t at = 0; at < s.size();) {
                size_t end = s.find(',', at);
                if (end == std::string::npos) end = s.size();
                devices.push_back(std::atoi(s.substr(at, end - at).c_str()));
                at = end + 1;
            }
        } else {
            usage();
            return 2;
        }
    }
    if (model.empty() || listen.empty()) {
        usage();
        return 2;
    }
    try {
        if (devices.empty() && !devices_given) {
            int n = 0;
            ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
            for (int d = 0; d < n; ++d) devices.push_back(d);
        }
        if (devices.empty() && ram_mib <= 0) throw std::runtime_error("no devices and no RAM window: nothing to hold");
        Helper helper(model, devices, reserve_mib, ram_mib, rdma_port);
        wire::Listener l(listen);
        std::fprintf(stderr, "strata-expert-helper: listening on port %d\n", l.port());
        std::atomic<int> next{0};
        for (;;) {
            wire::Conn c = l.accept();
            const std::string who = "engine " + std::to_string(++next);
            std::thread([&helper, who](wire::Conn conn) { helper.serve(std::move(conn), who); }, std::move(c)).detach();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-expert-helper: %s\n", e.what());
        return 1;
    }
}
