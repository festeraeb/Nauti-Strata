// src/poly/expert_follow.hpp - which experts the helper's set trades, so that it follows what is written.
//
// The engine's VRAM cache follows the answer under way (Engine::follow()): an expert the answer keeps routing to
// takes the slot of a cached one it uses less.  The helper's set (hold_remote()) was fixed at load from the
// profile, and a profile fits the average answer, not this one: measured on Le Chonk (2026-11-07), 1,338 experts
// on the helper served 10-15% of the routed experts while VRAM's 995 served 34-39%.  helper_trade() picks, once a
// token, the hot misses to send over and the cold held ones to give up; the engine asks the helper for them
// beside the answer, so the set moves with the prompt while the profile still decides where it starts.
//
// No CUDA, no engine: plain arrays over (layer, expert), so expert_follow_test checks it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::poly {

// Engine::follow(), and helper_trade() below, by one rule: an expert's recent use is the count of its uses, halved
// every kSwapHalf tokens.  An expert takes a cached one's slot when it was used twice lately and more than twice
// as much as that one.  Counts this small are noisy, and an upload gets in the way of the work: a slot goes to an
// expert that will clearly be used more.
constexpr float kSwapMin = 2.f;
constexpr int kSwapHalf = 128;

/// What the helper's set is to do: `in` come over (hottest first), `out` are given up (coldest first).
struct HelperTrade {
    std::vector<std::pair<int32_t, int32_t>> in, out;
    uint64_t in_bytes = 0, out_bytes = 0;
};

/// The experts per (layer, expert), i = layer * n_expert + expert, each array `n_layer * n_expert` long:
///   recent:   uses by the written tokens lately (Engine::recent_)
///   prior:    the profile's count, to break ties; may be null
///   vram:     1 when the engine's VRAM has it or it is on its way there: the helper need not
///   remote:   1 when the helper holds it for this engine
///   pending:  1 when a Hold for it is out already
///   layer_bytes: per layer, one expert's bytes (plus what the helper pads); 0 for a layer this engine does not
///                run, or that has no routed experts
/// `room` is what the helper has free for this engine now; `most` the most experts that may come over at once.
///
/// An expert comes over when it was used kSwapMin times or more lately and the helper has room, or when held ones
/// used at most half as much (less kSwapMin) can make the room: the same rule as the VRAM cache's, so the two never
/// thrash over the same expert.  A held expert that VRAM has too goes out whatever its use: the VRAM copy serves.
inline HelperTrade helper_trade(int n_layer, int n_expert, const float* recent, const int64_t* prior,
                                const uint8_t* vram, const uint8_t* remote, const uint8_t* pending,
                                const uint64_t* layer_bytes, uint64_t room, int most) {
    struct Pick {
        int il, e;
        float recent;
        int64_t prior;
    };
    HelperTrade t;
    if (most <= 0) return t;
    std::vector<Pick> hot, cold;
    for (int il = 0; il < n_layer; ++il) {
        if (layer_bytes[il] == 0) continue;
        for (int e = 0; e < n_expert; ++e) {
            const size_t i = (size_t) il * (size_t) n_expert + (size_t) e;
            const int64_t p = prior ? prior[i] : 0;
            if (remote[i]) {
                if (pending[i]) continue;
                if (vram[i]) {   // redundant: VRAM serves it, the helper's copy is room
                    t.out.emplace_back(il, e);
                    t.out_bytes += layer_bytes[il];
                    room += layer_bytes[il];
                } else {
                    cold.push_back({il, e, recent[i], p});
                }
            } else if (!vram[i] && !pending[i] && recent[i] >= kSwapMin) {
                hot.push_back({il, e, recent[i], p});
            }
        }
    }
    if (hot.empty()) return t;
    std::stable_sort(hot.begin(), hot.end(), [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    });
    std::stable_sort(cold.begin(), cold.end(), [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent < b.recent : a.prior < b.prior;
    });
    size_t next_cold = 0;
    for (const Pick& h : hot) {
        if ((int) t.in.size() >= most) break;
        const uint64_t b = layer_bytes[h.il];
        bool fits = room >= b;
        while (!fits) {
            if (next_cold >= cold.size()) break;
            const Pick& c = cold[next_cold];
            if (h.recent < 2.f * c.recent + kSwapMin) break;   // the coldest held one is too warm to give up for it
            room += layer_bytes[c.il];
            t.out.emplace_back(c.il, c.e);
            t.out_bytes += layer_bytes[c.il];
            ++next_cold;
            fits = room >= b;
        }
        if (!fits) break;   // the ones after it are colder still
        room -= b;
        t.in.emplace_back(h.il, h.e);
        t.in_bytes += b;
    }
    return t;
}

}  // namespace strata::poly
