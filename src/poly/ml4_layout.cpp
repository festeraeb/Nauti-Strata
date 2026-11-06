// src/poly/ml4_layout.cpp - Mistral Large 4's layer map and position math.  The reference is llama.cpp:
// llama_model_mistral_large4::load_arch_hparams() for the checks, ggml_rope_yarn_corr_dims() and ggml-cuda's
// rope_yarn() for the rotary math, llama_context's YaRN setup for the magnitude.
#include "ml4_layout.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly::ml4 {

std::vector<LayerPlan> layer_plan(int n_layer, const std::vector<uint8_t>& rope_pattern,
                                  const std::vector<int32_t>& kv_source_map, const std::vector<uint8_t>& indexer_types,
                                  int leading_dense) {
    if (n_layer <= 0) throw std::runtime_error("layer_plan: no layers");
    auto sized = [&](size_t n, const char* what) {
        if (n != 0 && n != (size_t) n_layer)
            throw std::runtime_error(std::string(what) + " has " + std::to_string(n) + " values for " +
                                     std::to_string(n_layer) + " layers");
    };
    sized(rope_pattern.size(), "attention.rope_pattern");
    sized(kv_source_map.size(), "attention.kv_source_map");
    sized(indexer_types.size(), "attention.indexer.types");

    std::vector<LayerPlan> plan((size_t) n_layer);
    for (int il = 0; il < n_layer; ++il) {
        LayerPlan& p = plan[(size_t) il];
        p.has_rope = rope_pattern.empty() || rope_pattern[(size_t) il] != 0;
        p.kv_source = kv_source_map.empty() ? il : kv_source_map[(size_t) il];
        p.owns_kv = p.kv_source == il;
        p.indexer_full = indexer_types.empty() || indexer_types[(size_t) il] != 0;
        p.dense = il < leading_dense;
    }
    if (!plan[0].indexer_full) throw std::runtime_error("the first layer must have a full indexer");
    for (int il = 0; il < n_layer; ++il) {
        LayerPlan& p = plan[(size_t) il];
        const int src = p.kv_source;
        if (src < 0 || src > il || !plan[(size_t) src].owns_kv || plan[(size_t) src].has_rope != p.has_rope)
            throw std::runtime_error("invalid KV source layer " + std::to_string(src) + " for layer " +
                                     std::to_string(il));
        p.indexer_source = p.indexer_full ? il : plan[(size_t) il - 1].indexer_source;
    }
    return plan;
}

void yarn_corr_dims(int n_rot, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float& lo,
                    float& hi) {
    const float pi = 3.14159265358979323846f;
    auto corr = [&](float rot) {
        return (float) n_rot * logf((float) n_ctx_orig / (rot * 2 * pi)) / (2 * logf(freq_base));
    };
    lo = std::fmax(0.0f, floorf(corr(beta_fast)));
    hi = std::fmin((float) (n_rot - 1), ceilf(corr(beta_slow)));
}

void rope_pair(int64_t pos, int i, int n_rot, float freq_base, float freq_scale, float ext_factor, float attn_factor,
               float lo, float hi, const float* factors, float& c, float& s) {
    // ggml-cuda: theta_base = pos * theta_scale^(i0/2), theta_scale = base^(-2/n_rot), divided by freq_factors
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    const float theta_extrap = (float) pos * powf(theta_scale, (float) i) / (factors ? factors[i] : 1.0f);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp, mscale = attn_factor;
    if (ext_factor != 0.0f) {
        // rope_yarn_ramp over the pair's index 2i
        const float y = ((float) (2 * i) / 2 - lo) / std::fmax(0.001f, hi - lo);
        const float ramp_mix = (1.0f - std::fmin(1.0f, std::fmax(0.0f, y))) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
}

float yarn_attn_factor(float freq_scale, float log_mul) {
    auto mscale = [](float scale, float m) { return scale <= 1.0f ? 1.0f : (0.1f * m * logf(scale) + 1.0f); };
    const float factor = 1.0f / freq_scale;
    float f = log_mul != 0.0f ? mscale(factor, 1.0f) / mscale(factor, log_mul) : mscale(factor, 1.0f);
    f *= 1.0f / (1.0f + 0.1f * logf(factor));
    return f;
}

// llama.cpp's attention-temperature input is log(floor(pos / 1) + 1) * 1 + 1 in double, stored as float, and the
// graph takes 1 off again before s and b apply.
float ssmax(int64_t pos, float s, float b) {
    const float lg = (float) (std::log(std::floor((double) pos) + 1.0) * 1.0 + 1.0) - 1.0f;
    return lg * s + b;
}

}  // namespace strata::poly::ml4
