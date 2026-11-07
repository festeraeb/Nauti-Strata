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
    /// Connects and checks that the helper has this model.  Throws std::runtime_error with the reason.
    RemoteHelper(const std::string& addr, uint64_t model_key, int n_expert, int n_embd, int n_ff);

    const std::string& addr() const { return addr_; }
    uint64_t capacity() const { return capacity_; }   ///< expert bytes the helper holds for all its engines
    int devices() const { return devices_; }

    /// Asks the helper to hold `want` (warmest first); returns those it holds.
    std::vector<std::pair<int32_t, int32_t>> hold(const std::vector<std::pair<int32_t, int32_t>>& want);
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
    uint64_t capacity_ = 0;
    int devices_ = 0;
    std::vector<uint8_t> buf_;
};

}  // namespace strata::poly
