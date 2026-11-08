// src/poly/remote_helper.hpp - an engine's connection to a strata-expert-helper (expert_wire.hpp).
//
// One a card's engine.  A layer's Compute is sent before the CPU's own experts start and read back after them,
// so the round trip costs nothing while the CPU has work; the calls are the engine's thread's alone.
#pragma once

#include "expert_wire.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::poly {

class RemoteHelper {
public:
    /// Connects and checks that the helper has this model.  want_ram: this connection may be given RAM holds,
    /// which it reads by RDMA (wire::kWantRam).  Throws std::runtime_error with the reason.
    RemoteHelper(const std::string& addr, uint64_t model_key, int n_expert, int n_embd, int n_ff,
                 bool want_ram = false);

    const std::string& addr() const { return addr_; }
    uint64_t capacity() const { return capacity_; }   ///< expert bytes the helper's GPUs hold for all its engines
    uint64_t ram_capacity() const { return ram_capacity_; }   ///< and its RAM window
    bool has_rdma() const { return has_rdma_; }       ///< the window can be read by RDMA (rdma() can be set up)
    int devices() const { return devices_; }

    /// Asks the helper to hold `want` (warmest first) where `where` allows; returns those it holds, and where.
    std::vector<wire::Held> hold(const std::vector<std::pair<int32_t, int32_t>>& want,
                                 wire::Where where = wire::kDeviceOnly);
    /// Connects the engine's queue pair `mine` to one of the helper's on its RAM window: its numbers, the rkey.
    wire::RdmaOk rdma(const wire::RdmaQp& mine);
    /// Gives up `drop` of this connection's holds; returns how many the helper let go.
    uint32_t release(const std::vector<std::pair<int32_t, int32_t>>& drop);

    /// One layer's experts for `rows` tokens: x is rows * n_embd inputs, entries the held experts routed to.
    void send(int layer, int rows, float limit, bool gelu, const float* x, const wire::Entry* entries, int n);
    /// The answer to the send() before: rows * n_embd weighted sums into out.
    void recv(float* out, int rows);
    /// From another thread: ends the connection, so a blocked hold() or recv() returns with an error.
    void abort() { c_.shutdown(); }

private:
    std::string addr_;
    wire::Conn c_;
    int n_embd_ = 0;
    uint64_t capacity_ = 0, ram_capacity_ = 0;
    bool has_rdma_ = false;
    int devices_ = 0;
    std::vector<uint8_t> buf_;
};

}  // namespace strata::poly
