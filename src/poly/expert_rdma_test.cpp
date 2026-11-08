// src/poly/expert_rdma_test.cpp - the RAM window's read path over a real RoCE port, looped back to itself.
//
//   expert_rdma_test DEV:PORT:GID      (t440: rocep1s0f1:1:3)   or STRATA_RDMA_TEST_PORT=...
//   expert_rdma_test --serve DEV:PORT:GID [ADDR:]PORT        two machines: a 1 GiB window here,
//   expert_rdma_test --read DEV:PORT:GID HOST:PORT           read from there (queue pairs set up over TCP, as the
//                                                            helper's holder connection does)
//   expert_rdma_test --helper DEV:PORT:GID HOST:PORT MODEL.gguf   a running strata-expert-helper's RAM window:
//                                                            holds experts there, reads them as the engine does,
//                                                            and compares each with the model's own bytes
//
// A loop back through the BCM57414 runs at ~106 MB/s even in perftest's ib_read_bw (2026-11-08), so the loop back
// checks correctness only; the two-machine mode is what times a READ (t440 -> c4: ib_read_lat 7.88 ms a 23 MiB).
//
// Two queue pairs on the same port, connected to each other: one side's registered window (REMOTE_READ), the other
// side's staging buffer (LOCAL_WRITE), as the helper and an engine have them.  Checks that READs posted several at a
// time land whole and in the right places, that a READ with a wrong key fails instead of hanging (the engine's
// fallback), and times a Le Chonk-sized (23 MiB) READ.  Without a port named it says so and passes: ctest runs on
// machines without RoCE.
#include "expert_rdma.hpp"
#include "expert_wire.hpp"
#include "remote_helper.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::poly;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

static uint8_t pattern(size_t i) { return (uint8_t) ((i * 2654435761u) >> 13); }

// --serve: the helper's side, one connection, until it closes
static int serve(const std::string& where, const std::string& listen) {
    rdma::Device dev(rdma::parse_port(where));
    const size_t bytes = 1ull << 30;
    std::vector<uint8_t> window(bytes);
    for (size_t i = 0; i < bytes; ++i) window[i] = pattern(i);
    rdma::Device::Mr mr = dev.reg(window.data(), bytes, true);
    wire::Listener l(listen);
    std::printf("expert_rdma_test: serving a 1 GiB window on port %d\n", l.port());
    std::fflush(stdout);
    wire::Conn c = l.accept();
    std::vector<uint8_t> p;
    if (c.recv(p) != wire::kRdma || p.size() != sizeof(wire::RdmaQp)) throw std::runtime_error("expected an Rdma");
    wire::RdmaQp peer;
    std::memcpy(&peer, p.data(), sizeof(peer));
    rdma::Qp qp(dev, 64);
    qp.connect(peer);
    wire::RdmaOk ok{};
    ok.qp = qp.local();
    ok.rkey = mr.rkey;
    ok.base = (uint64_t) (uintptr_t) window.data();
    ok.bytes = bytes;
    c.send(wire::kOk, &ok, sizeof(ok));
    try { c.recv(p); } catch (const std::runtime_error&) {}   // the reader is done when it closes
    rdma::Device::dereg(mr);
    std::printf("expert_rdma_test: served\n");
    return 0;
}

// --read: the engine's side: single 23 MiB READs, then 4 and 16 at a time, each checked
static int read_from(const std::string& where, const std::string& at) {
    rdma::Device dev(rdma::parse_port(where));
    const uint32_t big = 23u << 20;
    const int slots = 16;
    std::vector<uint8_t> stage((size_t) slots * big);
    rdma::Device::Mr mr = dev.reg(stage.data(), stage.size(), false);
    rdma::Qp qp(dev, 64);
    wire::Conn c = wire::Conn::dial(at, 5000);
    const wire::RdmaQp mine = qp.local();
    c.send(wire::kRdma, &mine, sizeof(mine));
    std::vector<uint8_t> p;
    if (c.recv(p) != wire::kOk || p.size() != sizeof(wire::RdmaOk)) throw std::runtime_error("the Rdma was refused");
    wire::RdmaOk ok;
    std::memcpy(&ok, p.data(), sizeof(ok));
    qp.connect(ok.qp);
    uint64_t ids[32];
    auto run = [&](int n, int reps) {
        std::vector<double> ms;
        bool right = true;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                const uint64_t off = (uint64_t) ((r * n + i) % 40) * big + 4096;
                qp.post_read(stage.data() + (size_t) i * big, mr.lkey, ok.base + off, ok.rkey, big, (uint64_t) i);
            }
            for (int done = 0; done < n;) done += qp.poll(ids, 32, 10000);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            for (int i = 0; i < n && right; ++i) {
                const uint64_t off = (uint64_t) ((r * n + i) % 40) * big + 4096;
                for (size_t k = 0; k < big && right; k += 4099) right = stage[(size_t) i * big + k] == pattern(off + k);
            }
        }
        std::sort(ms.begin(), ms.end());
        check(right, std::to_string(n) + " READs at a time land whole");
        std::printf("expert_rdma_test: %d x 23 MiB at a time from %s: p50 %.2f ms (%.2f ms an expert, %.2f GB/s), "
                    "p95 %.2f ms\n", n, at.c_str(), ms[ms.size() / 2], ms[ms.size() / 2] / n,
                    (double) n * big / (ms[ms.size() / 2] * 1e6), ms[ms.size() * 95 / 100]);
    };
    run(1, 30);
    run(4, 20);
    run(16, 10);
    c.close();
    rdma::Device::dereg(mr);
    return 0;
}

// --helper: Hold in RAM only, an RDMA READ of each, compared with the file (gate | up | down, as the helper copies)
static int check_helper(const std::string& where, const std::string& at, const std::string& model) {
    strata::GgufModel g = strata::GgufModel::open(model);
    uint64_t file_bytes = 0;
    for (size_t s = 0; s < g.size(); ++s) file_bytes += g.shard(s).file_size();
    struct L {
        int il;
        size_t gu, gu_stride, d;
        const uint8_t *gate, *up, *down;
    };
    std::vector<L> layers;
    int n_expert = 0, n_embd = 0, n_ff = 0;
    auto data = [&](const strata::TensorInfo& t) {
        size_t s = 0;
        g.find(t.name, &s);
        return g.shard(s).tensor_data(t);
    };
    for (int il = 0; il < 256; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const strata::TensorInfo* td = g.find(b + "ffn_down_exps.weight");
        if (!td || td->shape.size() != 3) continue;
        const int ne = (int) td->shape[2];
        L x{il, 0, 0, strata::tensor_payload_bytes(*td) / (size_t) ne, nullptr, nullptr, data(*td)};
        if (const strata::TensorInfo* gu = g.find(b + "ffn_gate_up_exps.weight")) {
            x.gu_stride = strata::tensor_payload_bytes(*gu) / (size_t) ne;
            x.gu = x.gu_stride / 2;
            x.gate = data(*gu);
            x.up = x.gate + x.gu;
            n_ff = (int) gu->shape[1] / 2;
        } else {
            const strata::TensorInfo* tg = g.find(b + "ffn_gate_exps.weight");
            const strata::TensorInfo* tu = g.find(b + "ffn_up_exps.weight");
            x.gu = x.gu_stride = strata::tensor_payload_bytes(*tg) / (size_t) ne;
            x.gate = data(*tg);
            x.up = data(*tu);
            n_ff = (int) tg->shape[1];
        }
        n_expert = ne;
        n_embd = (int) td->shape[1];
        layers.push_back(x);
    }
    if (layers.empty()) throw std::runtime_error(model + " has no routed experts");
    const uint64_t key = wire::model_key(file_bytes, n_expert, n_embd, n_ff);
    RemoteHelper h(at, key, n_expert, n_embd, n_ff, true);
    std::printf("expert_rdma_test: %s: %d devices, %llu MiB of VRAM, %llu MiB RAM window, RDMA %s\n", at.c_str(),
                h.devices(), (unsigned long long) (h.capacity() >> 20), (unsigned long long) (h.ram_capacity() >> 20),
                h.has_rdma() ? "yes" : "no");
    rdma::Device dev(rdma::parse_port(where));
    rdma::Qp qp(dev, 64);
    const wire::RdmaOk ok = h.rdma(qp.local());
    qp.connect(ok.qp);
    // three experts each of the first, a middle and the last MoE layer
    std::vector<std::pair<int32_t, int32_t>> want;
    for (size_t li : {(size_t) 0, layers.size() / 2, layers.size() - 1})
        for (int e : {0, n_expert / 2, n_expert - 1}) want.emplace_back(layers[li].il, e);
    const std::vector<wire::Held> got = h.hold(want, wire::kRamOnly);
    check(got.size() == want.size(), "the helper holds all " + std::to_string(want.size()) + " in RAM");
    size_t most = 0;
    for (const L& x : layers) most = std::max(most, 2 * x.gu + x.d);
    std::vector<uint8_t> stage(most);
    rdma::Device::Mr mr = dev.reg(stage.data(), stage.size(), false);
    uint64_t ids[4];
    int same = 0;
    for (const wire::Held& w : got) {
        const L* x = nullptr;
        for (const L& y : layers)
            if (y.il == w.layer) x = &y;
        check(w.kind == wire::kInRam && w.addr >= ok.base && w.addr + 2 * x->gu + x->d <= ok.base + ok.bytes,
              "a RAM hold's address is in the window");
        const auto t0 = std::chrono::steady_clock::now();
        qp.post_read(stage.data(), mr.lkey, w.addr, ok.rkey, (uint32_t) (2 * x->gu + x->d), 0);
        qp.poll(ids, 1, 10000);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool eq = std::memcmp(stage.data(), x->gate + (size_t) w.expert * x->gu_stride, x->gu) == 0 &&
                        std::memcmp(stage.data() + x->gu, x->up + (size_t) w.expert * x->gu_stride, x->gu) == 0 &&
                        std::memcmp(stage.data() + 2 * x->gu, x->down + (size_t) w.expert * x->d, x->d) == 0;
        same += eq;
        std::printf("expert_rdma_test: blk.%d expert %d: %.1f MiB read in %.2f ms, %s the file's\n", w.layer, w.expert,
                    (double) (2 * x->gu + x->d) / 1048576.0, ms, eq ? "the same as" : "NOT the same as");
    }
    check(same == (int) got.size(), "every expert read from the window is the file's bytes");
    rdma::Device::dereg(mr);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::string(argv[1]) == "--helper") {
        try {
            check_helper(argv[2], argv[3], argv[4]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "expert_rdma_test: %s\n", e.what());
            return 1;
        }
        if (failures) return 1;
        std::printf("expert_rdma_test: ok\n");
        return 0;
    }
    if (argc >= 4 && (std::string(argv[1]) == "--serve" || std::string(argv[1]) == "--read")) {
        try {
            const int rc = std::string(argv[1]) == "--serve" ? serve(argv[2], argv[3]) : read_from(argv[2], argv[3]);
            if (failures) return 1;
            if (std::string(argv[1]) == "--read") std::printf("expert_rdma_test: ok\n");
            return rc;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "expert_rdma_test: %s\n", e.what());
            return 1;
        }
    }
    const char* env = std::getenv("STRATA_RDMA_TEST_PORT");
    const std::string where = argc > 1 ? argv[1] : env ? env : "";
    if (where.empty()) {
        std::printf("expert_rdma_test: skipped (name a RoCE port: expert_rdma_test DEV:PORT:GID)\n");
        return 0;
    }
    try {
        const rdma::Port port = rdma::parse_port(where);
        rdma::Device dev(port);
        const size_t win_bytes = 64ull << 20, stage_bytes = 64ull << 20;
        std::vector<uint8_t> window(win_bytes), stage(stage_bytes, 0);
        for (size_t i = 0; i < win_bytes; ++i) window[i] = pattern(i);
        rdma::Device::Mr wmr = dev.reg(window.data(), win_bytes, true);
        rdma::Device::Mr smr = dev.reg(stage.data(), stage_bytes, false);
        rdma::Qp helper(dev, 64), engine(dev, 64);
        helper.connect(engine.local());
        engine.connect(helper.local());

        // 16 reads of 1 MiB from scattered places, at most 4 outstanding, each into its own slot (the hairpin drops
        // packets under more: "transport retry counter exceeded" now and then with 8 x 4 MiB, 2026-11-08)
        const int n = 16, inflight = 4;
        const uint32_t len = 1u << 20;
        std::vector<uint64_t> from(n);
        for (int i = 0; i < n; ++i) from[i] = (uint64_t) ((i * 7) % 15) * len + (uint64_t) i * 4096;
        int posted = 0, done = 0;
        std::vector<int> seen(n, 0);
        uint64_t ids[32];
        while (done < n) {
            while (posted < n && posted - done < inflight) {
                engine.post_read(stage.data() + (size_t) posted * len, smr.lkey,
                                 (uint64_t) (uintptr_t) window.data() + from[posted], wmr.rkey, len, (uint64_t) posted);
                ++posted;
            }
            const int got = engine.poll(ids, 32, 5000);
            for (int k = 0; k < got; ++k) ++seen[ids[k]];
            done += got;
        }
        bool whole = true;
        for (int i = 0; i < n; ++i)
            whole = whole && seen[i] == 1 && std::memcmp(stage.data() + (size_t) i * len, window.data() + from[i], len) == 0;
        check(whole, "16 READs, 4 in flight: each completes once and its bytes are the window's");

        // a Le Chonk expert: 23 MiB, one READ, 5 times
        const uint32_t big = 23u << 20;
        std::vector<double> ms;
        for (int r = 0; r < 5; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            engine.post_read(stage.data(), smr.lkey, (uint64_t) (uintptr_t) window.data() + 4096, wmr.rkey, big, 1);
            engine.poll(ids, 1, 5000);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        check(std::memcmp(stage.data(), window.data() + 4096, big) == 0, "a 23 MiB READ lands whole");
        std::printf("expert_rdma_test: 23 MiB READ looped back on %s: p50 %.2f ms (the hairpin, not the link)\n",
                    where.c_str(), ms[2]);

        // a wrong key: the READ fails, and the poll says so rather than waiting forever
        bool failed = false;
        try {
            engine.post_read(stage.data(), smr.lkey, (uint64_t) (uintptr_t) window.data(), wmr.rkey ^ 0x5a5a, 4096, 2);
            engine.poll(ids, 1, 10000);
        } catch (const std::runtime_error& e) {
            failed = true;
            std::printf("expert_rdma_test: a READ with a wrong key: %s\n", e.what());
        }
        check(failed, "a READ with a wrong key fails");
        rdma::Device::dereg(wmr);
        rdma::Device::dereg(smr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        ++failures;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("expert_rdma_test: ok\n");
    return 0;
}
