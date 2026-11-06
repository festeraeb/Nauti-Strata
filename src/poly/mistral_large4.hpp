// src/poly/mistral_large4.hpp - Mistral Large 4, "Le Chonk" (llama.cpp architecture "mistral-large4").
//
// 72 layers on one residual stream.  Every layer's attention is multi-head latent attention over a cache of
// 512-value latents with a 64-value rotary tail, chosen by a lightning indexer (DeepSeek's sparse attention) with
// 2,048 tokens a query at most; three layers in four turn their keys, the fourth does not, and from layer 39 on the
// layers that do not turn read layer 35's cache.  Indexers come in pairs: a layer scores and chooses, the shared
// ones after it take its choice.  Queries are scaled by SSMax, s ln(pos + 1) + b a head.  The first layer's FFN
// is dense; the others route 4 of 128 experts by softmax beside one shared expert.  The logits are soft-capped.
// The DSpark draft is not run here (--no-draft).
#pragma once

#include "engine.hpp"
#include "ml4_layout.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class MistralLarge4 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override { return 0; }
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void finish_logits(float* logits, int rows) override;
    bool batch_heads() const override { return true; }

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the caches.
    struct Block {
        ml4::LayerPlan plan;
        Dev wq, kv_a, kv_a_norm, k_b, v_b, wo, ssmax_s, ssmax_b;
        Dev idx_q_norm, idx_q_norm_b, idx_q_b, idx_k, idx_k_norm, idx_k_norm_b, idx_proj, idx_ssmax_s, idx_ssmax_b;
        uint16_t* lat = nullptr;   ///< max_context * row_, fp16: its own, or its KV source's
        float* pool = nullptr;     ///< max_context * idx_dim_: the indexer's keys (a pool of one token each)
        float* ring = nullptr;     ///< ring_ * 2 * idx_dim_: idx_store()'s rows on their way into pool
        Dev ffn_gate, ffn_up, ffn_down;   ///< the dense layer's
    };

    void attention(Block& B, int64_t pos, int n);
    void indexer(Block& B, int64_t pos, int n);
    void dense_ffn(Block& B, int il, int n);
    int dense_rows(int64_t pos, int n) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<ml4::LayerPlan> plan_;

    int n_head_ = 0, kv_rank_ = 0, mla_k_ = 0, mla_v_ = 0, n_rot_ = 0, row_ = 0;
    int idx_heads_ = 0, idx_dim_ = 0, idx_top_k_ = 0;
    int n_full_ = 0, n_own_ = 0;   ///< layers with an indexer of their own, and with a cache of their own
    int score_stride_ = 0, sel_stride_ = 0, score_rows_ = 0, ring_ = 0;
    float norm_eps_ = 0, w_scale_ = 1, softcap_ = 0;
    bool w_norm_ = true;
    Rope rope_{};   ///< the attention's, with LeRoPE factors; the indexer's is the same without the factors
    Dev rope_factors_;

    // device scratch
    float *q_ = nullptr, *qn_ = nullptr, *qabs_ = nullptr, *qc_ = nullptr;
    float *t2_ = nullptr, *kv_ = nullptr, *kvn_ = nullptr, *kr_ = nullptr;
    float *kqv_ = nullptr, *kqc_ = nullptr, *vout_ = nullptr;
    float *iqn_ = nullptr, *iq_ = nullptr, *iw_ = nullptr, *ikr_ = nullptr, *ik_ = nullptr, *ig_ = nullptr;
    float *ape_ = nullptr, *score_ = nullptr, *akeys_ = nullptr, *ascore_ = nullptr;
    size_t akeys_n_ = 0;
    int32_t* sel_ = nullptr;
};

}  // namespace strata::poly
