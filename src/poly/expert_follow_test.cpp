// src/poly/expert_follow_test.cpp - helper_trade(): the helper's set follows the answer (no CUDA).
#include "expert_follow.hpp"

#include <cstdio>
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

namespace {
// two layers of four experts; layer 1 is not this engine's
struct World {
    int n_layer = 3, n_expert = 4;
    std::vector<float> recent;
    std::vector<int64_t> prior;
    std::vector<uint8_t> vram, remote, pending;
    std::vector<uint64_t> bytes;
    World() : recent(12, 0.f), prior(12, 0), vram(12, 0), remote(12, 0), pending(12, 0), bytes{100, 0, 100} {}
    HelperTrade go(uint64_t room, int most = 32) const {
        return helper_trade(n_layer, n_expert, recent.data(), prior.data(), vram.data(), remote.data(),
                            pending.data(), bytes.data(), room, most);
    }
    size_t at(int il, int e) const { return (size_t) il * n_expert + e; }
};
bool has(const std::vector<std::pair<int32_t, int32_t>>& v, int il, int e) {
    for (const auto& p : v)
        if (p.first == il && p.second == e) return true;
    return false;
}
}  // namespace

int main() {
    {
        World w;
        const HelperTrade t = w.go(1000);
        check(t.in.empty() && t.out.empty(), "nothing used lately: nothing moves");
    }
    {
        World w;   // a hot miss and room for it: it comes over, nothing goes
        w.recent[w.at(0, 1)] = 3.f;
        const HelperTrade t = w.go(1000);
        check(t.in.size() == 1 && has(t.in, 0, 1) && t.out.empty() && t.in_bytes == 100, "a hot miss comes over");
    }
    {
        World w;   // used once: not hot enough (kSwapMin is 2)
        w.recent[w.at(0, 1)] = 1.f;
        check(w.go(1000).in.empty(), "one use is not a trend");
    }
    {
        World w;   // a hot miss, no room, a cold held one: they trade
        w.recent[w.at(0, 1)] = 4.f;
        w.remote[w.at(2, 3)] = 1;
        w.recent[w.at(2, 3)] = 0.5f;
        const HelperTrade t = w.go(0);
        check(t.in.size() == 1 && has(t.in, 0, 1) && t.out.size() == 1 && has(t.out, 2, 3), "a trade for a cold held one");
        check(t.in_bytes == 100 && t.out_bytes == 100, "the trade's bytes");
    }
    {
        World w;   // the held one is too warm: 4 < 2 * 1.5 + 2
        w.recent[w.at(0, 1)] = 4.f;
        w.remote[w.at(2, 3)] = 1;
        w.recent[w.at(2, 3)] = 1.5f;
        const HelperTrade t = w.go(0);
        check(t.in.empty() && t.out.empty(), "a warm held one stays");
    }
    {
        World w;   // the coldest held goes first
        w.recent[w.at(0, 1)] = 10.f;
        w.remote[w.at(2, 0)] = w.remote[w.at(2, 1)] = w.remote[w.at(2, 2)] = 1;
        w.recent[w.at(2, 0)] = 2.f;
        w.recent[w.at(2, 1)] = 0.f;
        w.recent[w.at(2, 2)] = 1.f;
        const HelperTrade t = w.go(0);
        check(t.out.size() == 1 && has(t.out, 2, 1), "the least used held one goes");
    }
    {
        World w;   // a held one that VRAM has too goes whatever its use, and its room serves a miss
        w.remote[w.at(2, 2)] = 1;
        w.vram[w.at(2, 2)] = 1;
        w.recent[w.at(2, 2)] = 50.f;
        w.recent[w.at(0, 0)] = 2.f;
        const HelperTrade t = w.go(0);
        check(t.out.size() == 1 && has(t.out, 2, 2), "a held one VRAM has is given up");
        check(t.in.size() == 1 && has(t.in, 0, 0), "its room takes a miss");
    }
    {
        World w;   // nothing hot: the redundant one still goes
        w.remote[w.at(2, 2)] = 1;
        w.vram[w.at(2, 2)] = 1;
        const HelperTrade t = w.go(0);
        check(t.out.size() == 1 && t.in.empty(), "a redundant held one goes even when nothing comes");
    }
    {
        World w;   // what VRAM has or is bringing is no miss; a pending Hold is not asked twice
        w.recent[w.at(0, 0)] = w.recent[w.at(0, 1)] = w.recent[w.at(0, 2)] = 5.f;
        w.vram[w.at(0, 0)] = 1;
        w.pending[w.at(0, 1)] = 1;
        const HelperTrade t = w.go(1000);
        check(t.in.size() == 1 && has(t.in, 0, 2), "VRAM's and pending ones are left alone");
    }
    {
        World w;   // a pending held one is not given up either
        w.recent[w.at(0, 0)] = 5.f;
        w.remote[w.at(2, 0)] = 1;
        w.pending[w.at(2, 0)] = 1;
        const HelperTrade t = w.go(0);
        check(t.out.empty() && t.in.empty(), "a held one with a Hold out is not given up");
    }
    {
        World w;   // another engine's layer takes no part
        w.recent[w.at(1, 0)] = 9.f;
        check(w.go(1000).in.empty(), "a layer this engine does not run is left alone");
    }
    {
        World w;   // `most` caps what comes over, hottest first; ties by the profile
        for (int e = 0; e < 4; ++e) w.recent[w.at(0, e)] = 3.f;
        w.prior[w.at(0, 2)] = 7;
        w.recent[w.at(2, 0)] = 8.f;
        const HelperTrade t = w.go(1000, 2);
        check(t.in.size() == 2 && t.in[0] == std::make_pair(2, 0) && t.in[1] == std::make_pair(0, 2),
              "the hottest two, the profile breaking the tie");
    }
    {
        World w;   // room runs out: the hottest come first, then it stops
        w.recent[w.at(0, 0)] = 3.f;
        w.recent[w.at(0, 1)] = 6.f;
        const HelperTrade t = w.go(150);
        check(t.in.size() == 1 && has(t.in, 0, 1), "with room for one, the hotter one comes");
    }
    {
        World w;   // two cold held ones make room for one hot miss when each is smaller
        w.bytes[2] = 50;
        w.recent[w.at(0, 0)] = 6.f;
        w.remote[w.at(2, 0)] = w.remote[w.at(2, 1)] = 1;
        const HelperTrade t = w.go(0);
        check(t.in.size() == 1 && t.out.size() == 2 && t.out_bytes == 100, "two small held ones go for one miss");
    }
    if (failures == 0) std::printf("expert_follow_test: OK\n");
    return failures == 0 ? 0 : 1;
}
