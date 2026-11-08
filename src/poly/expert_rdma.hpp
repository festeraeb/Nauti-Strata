// src/poly/expert_rdma.hpp - the few libibverbs calls the expert helper's RAM window needs (expert_wire.hpp).
//
// The helper registers its RAM window for remote reads; an engine registers a staging buffer of its own and
// RDMA-READs a held expert's bytes into it on a miss, then computes it on its CPU from there.  One reliable-connected
// queue pair a connection, set up over the TCP link (wire::kRdma): the same calls as nauti-fabric's rdma_shim.c,
// here straight from C++.  Built with -DSTRATA_RDMA=ON (links libibverbs); without it every constructor throws, so
// the callers fall back to what they did before the window existed.
#pragma once

#include "expert_wire.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::poly::rdma {

/// A RoCE port: device name, port number, GID index ("rocep135s0f1:1:5"; the GID index names the address and the
/// RoCE version: on our BCM57414s index 3 or 5 is RoCE v2 over IPv4).
struct Port {
    std::string dev;
    int port = 1;
    int gid = 0;
};
/// "DEV:PORT:GID" or "DEV" (port 1, GID 0).  Throws std::runtime_error on anything else.
Port parse_port(const std::string& s);

/// True when this build can do RDMA at all.
bool built();

/// An opened device and its protection domain.  Throws std::runtime_error with the reason.
class Device {
public:
    explicit Device(const Port& p);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    struct Mr {
        void* mr = nullptr;
        uint32_t lkey = 0, rkey = 0;
    };
    /// Registers [addr, addr + len): remote_read, a peer may read it; else it is a local buffer reads land in.
    Mr reg(void* addr, size_t len, bool remote_read);
    static void dereg(Mr& m);

    const Port& port() const { return port_; }
    int max_rd() const { return max_rd_; }   ///< RDMA READs a queue pair may have outstanding (both ends)

private:
    friend class Qp;
    Port port_;
    void* ctx_ = nullptr;
    void* pd_ = nullptr;
    int mtu_ = 0;
    uint16_t lid_ = 0;
    uint8_t gid_[16] = {};
    int max_rd_ = 1;
};

/// A reliable-connected queue pair with a completion queue of its own.
class Qp {
public:
    /// depth: the work requests that may be outstanding (RDMA READs posted and not yet polled).
    Qp(Device& d, int depth);
    ~Qp();
    Qp(const Qp&) = delete;
    Qp& operator=(const Qp&) = delete;

    wire::RdmaQp local() const;
    /// Connects to the peer's queue pair and brings this one to ready-to-send.
    void connect(const wire::RdmaQp& peer);

    /// Posts one RDMA READ of len bytes from the peer's [remote, remote + len) (rkey) into local (lkey).  Its
    /// completion carries `id`.  Throws when the send queue refuses it.
    void post_read(void* local, uint32_t lkey, uint64_t remote, uint32_t rkey, uint32_t len, uint64_t id);
    /// Waits up to timeout_ms for at least one completion; up to max of their ids into ids.  Returns how many.
    /// Throws std::runtime_error when one failed (its status named) or none came in time.
    int poll(uint64_t* ids, int max, int timeout_ms);

private:
    Device& d_;
    void* cq_ = nullptr;
    void* qp_ = nullptr;
    uint32_t psn_ = 0;
};

}  // namespace strata::poly::rdma
