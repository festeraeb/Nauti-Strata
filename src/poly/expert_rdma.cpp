// src/poly/expert_rdma.cpp - see expert_rdma.hpp.
#include "expert_rdma.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if STRATA_RDMA
#include <infiniband/verbs.h>

#include <cerrno>
#include <chrono>
#include <random>
#endif

namespace strata::poly::rdma {

Port parse_port(const std::string& s) {
    Port p;
    const size_t a = s.find(':');
    p.dev = s.substr(0, a);
    if (p.dev.empty()) throw std::runtime_error("RDMA port \"" + s + "\": expected DEV:PORT:GID");
    if (a == std::string::npos) return p;
    const size_t b = s.find(':', a + 1);
    char* end = nullptr;
    const std::string port = s.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
    p.port = (int) std::strtol(port.c_str(), &end, 10);
    if (port.empty() || *end || p.port < 1) throw std::runtime_error("RDMA port \"" + s + "\": bad port number");
    if (b == std::string::npos) return p;
    const std::string gid = s.substr(b + 1);
    p.gid = (int) std::strtol(gid.c_str(), &end, 10);
    if (gid.empty() || *end || p.gid < 0) throw std::runtime_error("RDMA port \"" + s + "\": bad GID index");
    return p;
}

#if STRATA_RDMA

bool built() { return true; }

namespace {
std::string err(const char* what, int code) { return std::string(what) + ": " + std::strerror(code); }
}  // namespace

Device::Device(const Port& p) : port_(p) {
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    if (!list) throw std::runtime_error(err("ibv_get_device_list", errno));
    ibv_device* found = nullptr;
    for (int i = 0; i < n; ++i)
        if (p.dev == ibv_get_device_name(list[i])) found = list[i];
    if (!found) {
        ibv_free_device_list(list);
        throw std::runtime_error("no RDMA device " + p.dev + " (" + std::to_string(n) + " seen)");
    }
    ibv_context* ctx = ibv_open_device(found);
    ibv_free_device_list(list);
    if (!ctx) throw std::runtime_error(err("ibv_open_device", errno));
    ctx_ = ctx;
    auto fail = [&](const std::string& why) {
        if (pd_) ibv_dealloc_pd((ibv_pd*) pd_);
        ibv_close_device(ctx);
        throw std::runtime_error(p.dev + ": " + why);
    };
    ibv_port_attr pa{};
    if (ibv_query_port(ctx, (uint8_t) p.port, &pa)) fail(err("ibv_query_port", errno));
    if (pa.state != IBV_PORT_ACTIVE) fail("port " + std::to_string(p.port) + " is not active");
    mtu_ = pa.active_mtu;
    lid_ = pa.lid;
    ibv_gid g{};
    if (ibv_query_gid(ctx, (uint8_t) p.port, p.gid, &g)) fail(err("ibv_query_gid", errno));
    std::memcpy(gid_, g.raw, 16);
    bool zero = true;
    for (uint8_t b : gid_) zero = zero && b == 0;
    if (zero) fail("GID index " + std::to_string(p.gid) + " is empty");
    ibv_device_attr da{};
    if (ibv_query_device(ctx, &da) == 0)
        max_rd_ = std::max(1, std::min({da.max_qp_rd_atom, da.max_qp_init_rd_atom, 64}));
    pd_ = ibv_alloc_pd(ctx);
    if (!pd_) fail(err("ibv_alloc_pd", errno));
}

Device::~Device() {
    if (pd_) ibv_dealloc_pd((ibv_pd*) pd_);
    if (ctx_) ibv_close_device((ibv_context*) ctx_);
}

Device::Mr Device::reg(void* addr, size_t len, bool remote_read) {
    const int access = remote_read ? IBV_ACCESS_REMOTE_READ : IBV_ACCESS_LOCAL_WRITE;
    ibv_mr* mr = ibv_reg_mr((ibv_pd*) pd_, addr, len, access);
    if (!mr) throw std::runtime_error(port_.dev + ": " + err("ibv_reg_mr", errno) + " (" + std::to_string(len >> 20) +
                                      " MiB; is the locked-memory limit high enough?)");
    return Mr{mr, mr->lkey, mr->rkey};
}

void Device::dereg(Mr& m) {
    if (m.mr) ibv_dereg_mr((ibv_mr*) m.mr);
    m = Mr{};
}

Qp::Qp(Device& d, int depth) : d_(d) {
    depth = std::max(depth, 1);
    ibv_cq* cq = ibv_create_cq((ibv_context*) d.ctx_, depth + 4, nullptr, nullptr, 0);
    if (!cq) throw std::runtime_error(d.port_.dev + ": " + err("ibv_create_cq", errno));
    cq_ = cq;
    ibv_qp_init_attr qa{};
    qa.send_cq = cq;
    qa.recv_cq = cq;
    qa.cap.max_send_wr = (uint32_t) depth;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.qp_type = IBV_QPT_RC;
    qa.sq_sig_all = 1;
    ibv_qp* qp = ibv_create_qp((ibv_pd*) d.pd_, &qa);
    if (!qp) {
        const int e = errno;
        ibv_destroy_cq(cq);
        throw std::runtime_error(d.port_.dev + ": " + err("ibv_create_qp", e));
    }
    qp_ = qp;
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = (uint8_t) d.port_.port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    if (const int rc = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        throw std::runtime_error(d.port_.dev + ": " + err("ibv_modify_qp INIT", rc));
    }
    std::random_device rd;
    psn_ = rd() & 0xffffff;
}

Qp::~Qp() {
    if (qp_) ibv_destroy_qp((ibv_qp*) qp_);
    if (cq_) ibv_destroy_cq((ibv_cq*) cq_);
}

wire::RdmaQp Qp::local() const {
    wire::RdmaQp q{};
    q.qpn = ((ibv_qp*) qp_)->qp_num;
    q.psn = psn_;
    std::memcpy(q.gid, d_.gid_, 16);
    q.lid = d_.lid_;
    q.mtu = (uint8_t) d_.mtu_;
    return q;
}

void Qp::connect(const wire::RdmaQp& peer) {
    ibv_qp* qp = (ibv_qp*) qp_;
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = (ibv_mtu) std::min<int>(peer.mtu, d_.mtu_);
    a.dest_qp_num = peer.qpn;
    a.rq_psn = peer.psn;
    a.max_dest_rd_atomic = (uint8_t) d_.max_rd_;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    std::memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t) d_.port_.gid;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.dlid = peer.lid;
    a.ah_attr.port_num = (uint8_t) d_.port_.port;
    if (const int rc = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                                 IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        throw std::runtime_error(d_.port_.dev + ": " + err("ibv_modify_qp RTR", rc));
    a = ibv_qp_attr{};
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;   // 4.096 us * 2^14 = 67 ms an attempt
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = psn_;
    a.max_rd_atomic = (uint8_t) d_.max_rd_;
    if (const int rc = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                                 IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
        throw std::runtime_error(d_.port_.dev + ": " + err("ibv_modify_qp RTS", rc));
}

void Qp::post_read(void* local, uint32_t lkey, uint64_t remote, uint32_t rkey, uint32_t len, uint64_t id) {
    ibv_sge sge{};
    sge.addr = (uintptr_t) local;
    sge.length = len;
    sge.lkey = lkey;
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote;
    wr.wr.rdma.rkey = rkey;
    if (const int rc = ibv_post_send((ibv_qp*) qp_, &wr, &bad))
        throw std::runtime_error(d_.port_.dev + ": " + err("ibv_post_send", rc));
}

int Qp::poll(uint64_t* ids, int max, int timeout_ms) {
    ibv_wc wc[32];
    max = std::min(max, 32);
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (unsigned spin = 0;; ++spin) {
        const int n = ibv_poll_cq((ibv_cq*) cq_, max, wc);
        if (n < 0) throw std::runtime_error(d_.port_.dev + ": ibv_poll_cq failed");
        if (n > 0) {
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS)
                    throw std::runtime_error(d_.port_.dev + ": RDMA READ: " + ibv_wc_status_str(wc[i].status) + " (" +
                                             std::to_string((int) wc[i].status) + ")");
                ids[i] = wc[i].wr_id;
            }
            return n;
        }
        if ((spin & 1023) == 0 && std::chrono::steady_clock::now() > until)
            throw std::runtime_error(d_.port_.dev + ": RDMA READ: no completion within " + std::to_string(timeout_ms) +
                                     " ms");
    }
}

#else   // built without libibverbs: nothing can be set up, and the callers run as before

bool built() { return false; }

namespace {
[[noreturn]] void none() { throw std::runtime_error("this build has no RDMA (configure with -DSTRATA_RDMA=ON)"); }
}  // namespace

Device::Device(const Port& p) : port_(p) { none(); }
Device::~Device() = default;
Device::Mr Device::reg(void*, size_t, bool) { none(); }
void Device::dereg(Mr& m) { m = Mr{}; }
Qp::Qp(Device& d, int) : d_(d) { none(); }
Qp::~Qp() = default;
wire::RdmaQp Qp::local() const { none(); }
void Qp::connect(const wire::RdmaQp&) { none(); }
void Qp::post_read(void*, uint32_t, uint64_t, uint32_t, uint32_t, uint64_t) { none(); }
int Qp::poll(uint64_t*, int, int) { none(); }

#endif

}  // namespace strata::poly::rdma
