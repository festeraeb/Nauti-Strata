// src/poly/ml4_layout_test.cpp - ml4_layout against Mistral Large 4's own metadata and ggml's rotary formulas.
//
// The layer arrays are the ones in Mistral-Large-4-1T-A52B's GGUF (72 layers).  The rotary expectations were
// computed independently in double from ggml-cuda's rope_yarn() and ggml_rope_yarn_corr_dims() (tools/ml4_rope_ref.py
// prints them); the tolerance is float rounding.
#include "ml4_layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace strata::poly::ml4;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void near(float got, double want, double tol, const std::string& what) {
    check(std::fabs((double) got - want) <= tol, what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

template <class F>
void throws(F&& f, const char* needle, const std::string& what) {
    try {
        f();
        check(false, what + ": did not throw");
    } catch (const std::runtime_error& e) {
        check(std::string(e.what()).find(needle) != std::string::npos,
              what + ": threw '" + e.what() + "', expected it to say '" + needle + "'");
    }
}

constexpr int kLayers = 72;

// attention.rope_pattern: rope,rope,rope,nope
std::vector<uint8_t> rope_pattern() {
    std::vector<uint8_t> v((size_t) kLayers);
    for (int il = 0; il < kLayers; ++il) v[(size_t) il] = il % 4 != 3;
    return v;
}

// attention.kv_source_map: every layer its own up to 38, then 39, 43, 47, ... read layer 35
std::vector<int32_t> kv_source_map() {
    std::vector<int32_t> v((size_t) kLayers);
    for (int il = 0; il < kLayers; ++il) v[(size_t) il] = (il >= 39 && il % 4 == 3) ? 35 : il;
    return v;
}

// attention.indexer.types: 17 full, then (shared, shared, full, full) repeating, the last one full
std::vector<uint8_t> indexer_types() {
    std::vector<uint8_t> v((size_t) kLayers, 1);
    for (int il = 17; il < kLayers; ++il) v[(size_t) il] = ((il - 17) % 4) >= 2;
    v[71] = 1;
    return v;
}

void test_plan() {
    const std::vector<LayerPlan> plan = layer_plan(kLayers, rope_pattern(), kv_source_map(), indexer_types(), 1);
    check(plan.size() == kLayers, "72 plans");
    check(plan[0].dense && !plan[1].dense && !plan[71].dense, "only layer 0 is dense");
    check(plan[0].has_rope && plan[2].has_rope && !plan[3].has_rope && !plan[7].has_rope && !plan[71].has_rope,
          "layers 3, 7, ... 71 have no rope");
    check(plan[3].owns_kv && plan[35].owns_kv && plan[38].owns_kv, "layers 3, 35 and 38 own their cache");
    check(!plan[39].owns_kv && plan[39].kv_source == 35 && !plan[71].owns_kv && plan[71].kv_source == 35,
          "layers 39 and 71 read layer 35's cache");
    check(plan[40].owns_kv && plan[40].kv_source == 40, "layer 40 owns its cache");
    int own = 0;
    for (const LayerPlan& p : plan) own += p.owns_kv;
    check(own == 72 - 9, "63 layers own a cache (9 read layer 35's)");
    check(plan[16].indexer_full && plan[16].indexer_source == 16, "layer 16 has a full indexer");
    check(!plan[17].indexer_full && plan[17].indexer_source == 16, "layer 17 shares layer 16's choice");
    check(!plan[18].indexer_full && plan[18].indexer_source == 16, "layer 18 shares layer 16's choice");
    check(plan[19].indexer_full && plan[19].indexer_source == 19, "layer 19 has a full indexer");
    check(plan[20].indexer_full && !plan[21].indexer_full && !plan[22].indexer_full && plan[23].indexer_full,
          "layers 20-23: full, shared, shared, full");
    check(!plan[70].indexer_full && plan[70].indexer_source == 68 && plan[71].indexer_full, "layers 70 and 71");
    int full = 0;
    for (const LayerPlan& p : plan) full += p.indexer_full;
    check(full == 44, "44 full indexers: 17, then 13 groups of (shared, shared, full, full), then shared, shared, full");

    // empty arrays: every layer turns, owns its cache, has a full indexer
    const std::vector<LayerPlan> plain = layer_plan(4, {}, {}, {}, 0);
    for (int il = 0; il < 4; ++il)
        check(plain[(size_t) il].has_rope && plain[(size_t) il].owns_kv && plain[(size_t) il].kv_source == il &&
                  plain[(size_t) il].indexer_full && plain[(size_t) il].indexer_source == il && !plain[(size_t) il].dense,
              "plain layer " + std::to_string(il));

    // two readers of one owner, and a reader of a later owner, are fine
    const std::vector<LayerPlan> shared = layer_plan(4, {}, {0, 1, 1, 1}, {}, 0);
    check(shared[2].kv_source == 1 && !shared[2].owns_kv && shared[3].kv_source == 1 && !shared[3].owns_kv,
          "layers 2 and 3 both read layer 1");
    check(layer_plan(4, {}, {0, 0, 2, 2}, {}, 0)[3].kv_source == 2, "layer 3 reads layer 2, which owns its cache");

    throws([] { layer_plan(4, {}, {0, 1, 3, 3}, {}, 0); }, "invalid KV source layer 3 for layer 2", "source ahead");
    throws([] { layer_plan(4, {}, {0, 1, 1, 2}, {}, 0); }, "invalid KV source layer 2 for layer 3",
           "source that reads another cache");
    throws([] { layer_plan(4, {1, 1, 0, 1}, {0, 1, 2, 2}, {}, 0); }, "invalid KV source layer 2 for layer 3",
           "rope mismatch between reader and source");
    throws([] { layer_plan(4, {}, {0, 1, 2, 7}, {}, 0); }, "invalid KV source layer 7 for layer 3", "source past the end");
    throws([] { layer_plan(4, {}, {}, {0, 1, 1, 1}, 0); }, "the first layer must have a full indexer", "first shared");
    throws([] { layer_plan(4, {1, 1}, {}, {}, 0); }, "attention.rope_pattern has 2 values for 4 layers", "short array");
    throws([] { layer_plan(0, {}, {}, {}, 0); }, "no layers", "no layers");
}

void test_rope() {
    // Mistral Large 4: base 10000, n_rot 64, YaRN factor 2048 over 512 tokens, beta 32 / 1, log multiplier 1
    float lo = 0, hi = 0;
    yarn_corr_dims(64, 512, 10000.0f, 32.0f, 1.0f, lo, hi);
    near(lo, 3.0, 0.0, "corr lo: floor(64 ln(512 / (32 2pi)) / (2 ln 10000)) = floor(3.25)");
    near(hi, 16.0, 0.0, "corr hi: ceil(64 ln(512 / 2pi) / (2 ln 10000)) = ceil(15.29)");
    const float attn = yarn_attn_factor(1.0f / 2048.0f, 1.0f);
    near(attn, 1.0 / (1.0 + 0.1 * std::log(2048.0)), 1e-6, "yarn attn factor cancels the magnitude");
    // the magnitude ggml then applies under YaRN is attn * (1 + 0.1 ln 2048) = 1
    float c = 0, s = 0;
    rope_pair(0, 0, 64, 10000.0f, 1.0f / 2048.0f, 1.0f, attn, lo, hi, nullptr, c, s);
    near(c, 1.0, 1e-6, "position 0 pair 0: cos 1");
    near(s, 0.0, 1e-6, "position 0 pair 0: sin 0");
    // position 100, pair 0: fully extrapolated (pair 0 < lo): theta = 100
    rope_pair(100, 0, 64, 10000.0f, 1.0f / 2048.0f, 1.0f, attn, lo, hi, nullptr, c, s);
    near(c, std::cos(100.0), 2e-5, "position 100 pair 0 cos");
    near(s, std::sin(100.0), 2e-5, "position 100 pair 0 sin");
    // position 100, pair 31: fully interpolated (pair 31 > hi): theta = 100 * 10000^(-62/64) / 2048
    const double th31 = 100.0 * std::pow(10000.0, -62.0 / 64.0) / 2048.0;
    rope_pair(100, 31, 64, 10000.0f, 1.0f / 2048.0f, 1.0f, attn, lo, hi, nullptr, c, s);
    near(c, std::cos(th31), 1e-6, "position 100 pair 31 cos");
    near(s, std::sin(th31), 1e-6, "position 100 pair 31 sin");
    // position 1000, pair 10: on the ramp; y = (10 - lo) / (hi - lo), mix = (1 - y)
    {
        const double extrap = 1000.0 * std::pow(10000.0, -20.0 / 64.0), interp = extrap / 2048.0;
        const double y = (10.0 - lo) / (hi - lo), mix = 1.0 - y;
        const double th = interp * (1 - mix) + extrap * mix;
        rope_pair(1000, 10, 64, 10000.0f, 1.0f / 2048.0f, 1.0f, attn, lo, hi, nullptr, c, s);
        near(c, std::cos(th), 2e-4, "position 1000 pair 10 cos");
        near(s, std::sin(th), 2e-4, "position 1000 pair 10 sin");
    }
    // LeRoPE: a factor divides the angle before YaRN; pair 0 of position 100 with factor 4: theta 25
    {
        float f[32];
        for (float& v : f) v = 4.0f;
        rope_pair(100, 0, 64, 10000.0f, 1.0f / 2048.0f, 1.0f, attn, lo, hi, f, c, s);
        near(c, std::cos(25.0), 1e-5, "factor 4 cos");
        near(s, std::sin(25.0), 1e-5, "factor 4 sin");
    }
    // no YaRN (ext_factor 0): plain rope with attn_factor as the magnitude
    rope_pair(7, 3, 64, 10000.0f, 1.0f, 0.0f, 0.5f, lo, hi, nullptr, c, s);
    near(c, 0.5 * std::cos(7.0 * std::pow(10000.0, -6.0 / 64.0)), 1e-6, "plain rope cos");
    near(s, 0.5 * std::sin(7.0 * std::pow(10000.0, -6.0 / 64.0)), 1e-6, "plain rope sin");
}

void test_ssmax() {
    near(ssmax(0, 2.0f, 0.5f), 0.5, 0.0, "ssmax at 0 is b");
    near(ssmax(1, 1.0f, 0.0f), std::log(2.0), 1e-6, "ssmax at 1 is ln 2");
    near(ssmax(99, 0.5f, -1.0f), 0.5 * std::log(100.0) - 1.0, 1e-6, "ssmax at 99");
}

}  // namespace

int main() {
    test_plan();
    test_rope();
    test_ssmax();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ml4_layout_test: ok\n");
    return 0;
}
