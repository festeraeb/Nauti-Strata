// src/poly/remote_helper.cpp - see remote_helper.hpp.
#include "remote_helper.hpp"

#include <cstring>
#include <stdexcept>

namespace strata::poly {

RemoteHelper::RemoteHelper(const std::string& addr, uint64_t model_key, int n_expert, int n_embd, int n_ff)
    : addr_(addr), n_embd_(n_embd) {
    c_ = wire::Conn::dial(addr, 5000);
    const wire::Hello h{model_key, (uint32_t) n_expert, (uint32_t) n_embd, (uint32_t) n_ff, 0};
    c_.send(wire::kHello, &h, sizeof(h));
    if (c_.recv(buf_) != wire::kOk) throw std::runtime_error(addr + ": " + wire::Conn::text(buf_));
    wire::HelloOk ok{};
    if (buf_.size() != sizeof(ok)) throw std::runtime_error(addr + ": a Hello answer of the wrong size");
    std::memcpy(&ok, buf_.data(), sizeof(ok));
    capacity_ = ok.capacity;
    devices_ = (int) ok.devices;
}

std::vector<std::pair<int32_t, int32_t>> RemoteHelper::hold(const std::vector<std::pair<int32_t, int32_t>>& want) {
    std::vector<wire::Pair> q;
    q.reserve(want.size());
    for (const auto& [l, e] : want) q.push_back({l, e});
    const uint32_t n = (uint32_t) q.size();
    c_.send(wire::kHold, &n, 4, q.data(), q.size() * sizeof(wire::Pair));
    if (c_.recv(buf_) != wire::kOk) throw std::runtime_error(addr_ + ": " + wire::Conn::text(buf_));
    uint32_t m = 0;
    if (buf_.size() < 4) throw std::runtime_error(addr_ + ": a Hold answer too short");
    std::memcpy(&m, buf_.data(), 4);
    if (buf_.size() != 4 + (size_t) m * sizeof(wire::Pair)) throw std::runtime_error(addr_ + ": a Hold answer of the wrong size");
    std::vector<std::pair<int32_t, int32_t>> got(m);
    for (uint32_t i = 0; i < m; ++i) {
        wire::Pair p;
        std::memcpy(&p, buf_.data() + 4 + (size_t) i * sizeof(p), sizeof(p));
        got[i] = {p.layer, p.expert};
    }
    return got;
}

uint32_t RemoteHelper::release(const std::vector<std::pair<int32_t, int32_t>>& drop) {
    if (drop.empty()) return 0;
    std::vector<wire::Pair> q;
    q.reserve(drop.size());
    for (const auto& [l, e] : drop) q.push_back({l, e});
    const uint32_t n = (uint32_t) q.size();
    c_.send(wire::kRelease, &n, 4, q.data(), q.size() * sizeof(wire::Pair));
    if (c_.recv(buf_) != wire::kOk) throw std::runtime_error(addr_ + ": " + wire::Conn::text(buf_));
    uint32_t m = 0;
    if (buf_.size() != 4) throw std::runtime_error(addr_ + ": a Release answer of the wrong size");
    std::memcpy(&m, buf_.data(), 4);
    return m;
}

void RemoteHelper::send(int layer, int rows, float limit, bool gelu, const float* x, const wire::Entry* entries, int n) {
    const wire::ComputeHead h{layer, rows, limit, gelu ? wire::kGelu : 0u, (uint32_t) n, 0};
    c_.send(wire::kCompute, &h, sizeof(h), x, (size_t) rows * n_embd_ * 4, entries, (size_t) n * sizeof(wire::Entry));
}

void RemoteHelper::recv(float* out, int rows) {
    if (c_.recv(buf_) != wire::kOk) throw std::runtime_error(addr_ + ": " + wire::Conn::text(buf_));
    const size_t want = (size_t) rows * n_embd_ * 4;
    if (buf_.size() != want) throw std::runtime_error(addr_ + ": a Compute answer of the wrong size");
    std::memcpy(out, buf_.data(), want);
}

}  // namespace strata::poly
