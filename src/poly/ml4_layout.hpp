// src/poly/ml4_layout.hpp - Mistral Large 4's layer map and position math, as llama.cpp reads them.
//
// The model's GGUF says, layer by layer, whether the attention turns its keys (attention.rope_pattern), which
// layer's cache it reads (attention.kv_source_map) and whether it has an indexer of its own or takes the token
// choice of the full one before it (attention.indexer.types).  This file turns those arrays into one plan a layer,
// with llama.cpp's checks, and holds the host side of the rotary and SSMax math the kernels repeat.  No CUDA: the
// engine (mistral_large4.cpp) and its tests both use it.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::poly::ml4 {

struct LayerPlan {
    bool has_rope = true;       ///< the last n_rot values of a query and key turn by position
    int kv_source = 0;          ///< the layer whose latent cache this one reads (itself when it owns one)
    bool owns_kv = true;        ///< kv_source == the layer: it projects and stores its own latents
    bool indexer_full = true;   ///< it scores and chooses the tokens to attend; else it takes indexer_source's choice
    int indexer_source = 0;     ///< the nearest full indexer at or before this layer
    bool dense = false;         ///< its FFN is one dense MLP, not routed experts
};

/// The plan of every layer.  `rope_pattern` and `indexer_types` are one value a layer or empty (every layer turns,
/// every indexer is full); `kv_source_map` is one value a layer or empty (every layer its own).  Throws
/// std::runtime_error with llama.cpp's wording when a source lies ahead of its layer, is not a layer with a cache
/// of its own, or turns its keys where the reader does not; when the first layer's indexer is not full; or when
/// an array is neither empty nor n_layer long.
std::vector<LayerPlan> layer_plan(int n_layer, const std::vector<uint8_t>& rope_pattern,
                                  const std::vector<int32_t>& kv_source_map, const std::vector<uint8_t>& indexer_types,
                                  int leading_dense);

/// ggml_rope_yarn_corr_dims(): the first and last rotary pair the YaRN ramp runs across.
void yarn_corr_dims(int n_rot, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float& lo, float& hi);

/// The cosine and sine ggml's rope gives rotary pair i (0 <= i < n_rot / 2) of a head at `pos`: the extrapolated
/// angle pos * freq_base^(-2i / n_rot) over factors[i] (LeRoPE; 1 without), YaRN's interpolation between lo and hi
/// when ext_factor != 0, and the magnitude attn_factor * (1 + 0.1 ln(1 / freq_scale)) under YaRN, attn_factor
/// else.  The same arithmetic as ggml-cuda's rope_yarn(), in float.
void rope_pair(int64_t pos, int i, int n_rot, float freq_base, float freq_scale, float ext_factor, float attn_factor,
               float lo, float hi, const float* factors, float& c, float& s);

/// llama.cpp's YaRN magnitude for this model: with rope.scaling.yarn_log_multiplier the context sets
/// attn_factor = mscale(factor, 1) / mscale(factor, log_mul), else mscale(factor, 1), times 1 / (1 + 0.1 ln factor),
/// where mscale(f, m) = 1 for f <= 1, else 0.1 m ln f + 1.  factor = 1 / freq_scale.
float yarn_attn_factor(float freq_scale, float log_mul);

/// SSMax: the per-head factor a query takes at `pos`, s * ln(pos + 1) + b.
float ssmax(int64_t pos, float s, float b);

}  // namespace strata::poly::ml4
