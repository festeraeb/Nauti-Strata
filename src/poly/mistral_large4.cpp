// src/poly/mistral_large4.cpp - Mistral Large 4's layers.  The reference for the math is llama.cpp's
// mistral-large4 graph (src/models/mistral-large4.cpp) and its sparse-attention cache (llama-kv-cache-dsa.cpp).
//
// The sparse attention runs on the GLM 5.3 kernels (kernels.cu's idx_* and dsa_attend) with a pool of one token:
// a pool's key is then the token's own indexer key, and the chosen pools are the chosen tokens.  The query the
// attention takes is a head's latent query (the nope part through k_b) with the turned tail beside it, row_ = 512
// + 64 values; a key is the stored latent with its own turned tail, or zeros where the layer does not turn.
#include "mistral_large4.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kPool = 1;            // a pool is one token
constexpr int kAttnRows = 32;       // tokens one pass of the sparse attention takes
constexpr size_t kScoreBytes = size_t(128) << 20;   // a prompt batch's indexer scores at most
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

std::vector<uint8_t> bools_of(const MetaValue* v) {
    std::vector<uint8_t> out;
    if (!v || v->type != MetaType::ARRAY) return out;
    out.reserve(v->items.size());
    for (const MetaValue& m : v->items) out.push_back(m.u != 0);
    return out;
}

std::vector<int32_t> ints_of(const MetaValue* v) {
    std::vector<int32_t> out;
    if (!v || v->type != MetaType::ARRAY) return out;
    out.reserve(v->items.size());
    for (const MetaValue& m : v->items) out.push_back((int32_t) m.u);
    return out;
}

// cudaMemcpy2DAsync between row-major float buffers of other pitches, on the engine's stream.
void copy_cols(float* dst, size_t dst_pitch, const float* src, size_t src_pitch, size_t width, size_t rows,
               void* stream) {
    if (rows == 0 || width == 0) return;
    ck(cudaMemcpy2DAsync(dst, dst_pitch * sizeof(float), src, src_pitch * sizeof(float), width * sizeof(float), rows,
                         cudaMemcpyDeviceToDevice, cs(stream)),
       "copy columns");
}

void zero_cols(float* dst, size_t dst_pitch, size_t width, size_t rows, void* stream) {
    if (rows == 0 || width == 0) return;
    ck(cudaMemset2DAsync(dst, dst_pitch * sizeof(float), 0, width * sizeof(float), rows, cs(stream)), "zero columns");
}

}  // namespace

void MistralLarge4::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_full_ = n_own_ = 0;
}

void MistralLarge4::check_draft_block() {
    throw std::runtime_error("the DSpark draft of this model is not run by this engine; --no-draft loads the model "
                             "without it");
}

void MistralLarge4::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_ff_ = shape_i("feed_forward_length");
    n_head_ = shape_i("attention.head_count");
    kv_rank_ = shape_i("attention.kv_lora_rank");
    mla_k_ = shape_i("attention.key_length_mla");
    mla_v_ = shape_i("attention.value_length_mla");
    n_rot_ = shape_i("rope.dimension_count");
    idx_heads_ = shape_i("attention.indexer.head_count");
    idx_dim_ = shape_i("attention.indexer.key_length");
    idx_top_k_ = shape_i("attention.indexer.top_k");
    const MetaValue* norm_eps = find("attention.layer_norm_epsilon");
    norm_eps_ = norm_eps ? (float) norm_eps->f : 1e-6f;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->f : 1.0f;
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = !norm || norm->u != 0;
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;
    const MetaValue* lead = find("leading_dense_block_count");
    const int leading_dense = lead ? (int) lead->u : 0;
    row_ = kv_rank_ + n_rot_;

    need(find("attention.q_lora_rank") == nullptr || find("attention.q_lora_rank")->u == 0,
         "a low-rank query projection, which this model has none of");
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || gating->u == 1, "the expert gate is not a softmax");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= mla_k_ && n_rot_ <= idx_dim_, "bad rope.dimension_count");
    need(mla_k_ == mla_v_, "the values are not as long as the keys");
    need(idx_dim_ == 128 && idx_heads_ > 0 && idx_heads_ <= 32, "the indexer's keys are not 128 values of 32 heads at most");
    need(idx_top_k_ > 0, "bad attention.indexer.top_k");
    need(kv_rank_ > 0 && kv_rank_ % 32 == 0 && row_ % 8 == 0, "bad attention.kv_lora_rank");
    need(n_expert_ > 0 || leading_dense >= n_layer_, "no routed experts");

    plan_ = ml4::layer_plan(n_layer_, bools_of(find("attention.rope_pattern")), ints_of(find("attention.kv_source_map")),
                            bools_of(find("attention.indexer.types")), leading_dense);

    // YaRN as llama_context sets it up for this model: the file's factor over its original context, beta_fast and
    // beta_slow, the log multiplier in the magnitude, which then cancels (apply_scale = false)
    const MetaValue* kind = find("rope.scaling.type");
    const MetaValue* factor = find("rope.scaling.factor");
    const MetaValue* orig = find("rope.scaling.original_context_length");
    const MetaValue* fast = find("rope.scaling.yarn_beta_fast");
    const MetaValue* slow = find("rope.scaling.yarn_beta_slow");
    const MetaValue* log_mul = find("rope.scaling.yarn_log_multiplier");
    const bool yarn = kind && kind->s == "yarn";
    const float freq_base = shape_f("rope.freq_base");
    const float freq_scale = yarn && factor && factor->f > 0 ? (float) (1.0 / factor->f) : 1.0f;
    const int n_ctx_orig = orig ? (int) orig->u : shape_i("context_length");
    rope_.theta_scale = powf(freq_base, -2.0f / (float) n_rot_);
    rope_.freq_scale = freq_scale;
    rope_.ext_factor = yarn ? 1.0f : 0.0f;
    rope_.attn_factor = yarn ? ml4::yarn_attn_factor(freq_scale, log_mul ? (float) log_mul->f : 0.f) : 1.0f;
    ml4::yarn_corr_dims(n_rot_, n_ctx_orig, freq_base, fast ? (float) fast->f : 32.0f, slow ? (float) slow->f : 1.0f,
                        rope_.corr0, rope_.corr1);

    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
}

void MistralLarge4::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    B.plan = plan_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) {
        return upload(tensor(il, n), blk(il, n), a, b, w);
    };
    const int nope = B.plan.has_rope ? mla_k_ - n_rot_ : mla_k_;   // a NoPE layer's whole head goes through k_b

    L.fixed = false;   // the launches take the position
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    if (il == lo_ || !rope_factors_.p) {   // one table for every layer (LeRoPE)
        const TensorInfo* f = gg_->find("rope_freqs.weight");
        if (f) rope_factors_ = upload(f, "rope_freqs.weight", n_rot_ / 2, 1, kF32);
    }

    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * mla_k_);
    if (B.plan.owns_kv) {
        B.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, kv_rank_ + (B.plan.has_rope ? n_rot_ : 0));
        B.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
        ++n_own_;
    }
    B.k_b = mat("attn_k_b.weight", nope, (int64_t) kv_rank_ * n_head_, kFloatMat);
    B.v_b = mat("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_, kFloatMat);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * mla_v_, n_embd_);
    B.ssmax_s = f32("attn_ssmax.weight", n_head_);
    B.ssmax_b = f32("attn_ssmax.bias", n_head_);
    if (B.plan.indexer_full) {
        B.idx_q_norm = f32("indexer.q_norm.weight", n_embd_);
        B.idx_q_norm_b = f32("indexer.q_norm.bias", n_embd_);
        B.idx_q_b = mat("indexer.attn_q_b.weight", n_embd_, (int64_t) idx_heads_ * idx_dim_);
        B.idx_k = mat("indexer.attn_k.weight", n_embd_, idx_dim_);
        B.idx_k_norm = f32("indexer.k_norm.weight", idx_dim_);
        B.idx_k_norm_b = f32("indexer.k_norm.bias", idx_dim_);
        B.idx_proj = mat("indexer.proj.weight", n_embd_, idx_heads_);
        B.idx_ssmax_s = f32("indexer.ssmax.weight", idx_heads_);
        B.idx_ssmax_b = f32("indexer.ssmax.bias", idx_heads_);
        ++n_full_;
    }

    if (B.plan.dense) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il);
}

// Per-token buffers hold rows_ tokens, row after row.
void MistralLarge4::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_;
    q_ = fbuf(heads * mla_k_);
    qn_ = fbuf(heads * mla_k_);
    qabs_ = fbuf(heads * kv_rank_);
    qc_ = fbuf(heads * row_);
    t2_ = fbuf(n * row_);
    kv_ = fbuf(n * kv_rank_);
    kvn_ = fbuf(n * kv_rank_);
    kr_ = fbuf(n * row_);
    kqv_ = fbuf(heads * row_);
    kqc_ = fbuf(heads * kv_rank_);
    vout_ = fbuf(heads * mla_v_);
    iqn_ = fbuf(n * n_embd_);
    iq_ = fbuf(n * idx_heads_ * idx_dim_);
    iw_ = fbuf(n * idx_heads_);
    ikr_ = fbuf(n * idx_dim_);
    ik_ = fbuf(n * idx_dim_);
    ig_ = fbuf(n * idx_dim_);
    ape_ = fbuf((size_t) idx_dim_);
    zero_vram(ig_, n * idx_dim_ * sizeof(float));   // a pool of one token has no gate: its key is the token's
    zero_vram(ape_, (size_t) idx_dim_ * sizeof(float));
    score_ = fbuf((size_t) score_rows_ * score_stride_);
    const size_t keys = (size_t) std::min(opt_.max_context, idx_top_k_ + kPool - 1);
    akeys_n_ = std::min(n, (size_t) kAttnRows) * keys * row_;
    akeys_ = fbuf(akeys_n_);
    ascore_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * n_head_);
}

int MistralLarge4::widest() const {
    return std::max({n_head_ * mla_k_, n_head_ * mla_v_, kv_rank_ * n_head_, row_, idx_heads_ * idx_dim_, n_ff_});
}

// A latent cache for every layer with one of its own, which the layers that read it share; the indexer's keys and
// ring for every layer that scores.
void MistralLarge4::alloc_caches() {
    ring_ = rows_ + kPool;
    const size_t ring_n = (size_t) ring_ * 2 * idx_dim_;
    float* ring = n_full_ ? (float*) seq_alloc(n_full_ * ring_n * sizeof(float), "the indexer cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.plan.owns_kv) {
            B.lat = (uint16_t*) kv_alloc((size_t) row_ * sizeof(uint16_t), 1);
        } else {
            const Block& S = blocks_[(size_t) B.plan.kv_source];
            if (!S.lat) throw std::runtime_error("layer " + std::to_string(il) + " reads the cache of layer " +
                                                 std::to_string(B.plan.kv_source) + ", which is on another card");
            B.lat = S.lat;
        }
        if (B.plan.indexer_full) {
            B.pool = (float*) kv_alloc((size_t) idx_dim_ * sizeof(float), kPool);
            B.ring = ring;
            ring += ring_n;
        }
    }
    score_stride_ = opt_.max_context / kPool + 1;
    score_rows_ = (int) std::clamp<size_t>(kScoreBytes / (score_stride_ * sizeof(float)), 1, (size_t) rows_);
    sel_stride_ = idx_top_k_ + kPool;
    if (!sel_) sel_ = (int32_t*) dmalloc((size_t) rows_ * sel_stride_ * sizeof(int32_t), "scratch");
}

// No recurrent state and no pools in progress: a layer's state is its cache rows, which are card rows.
void MistralLarge4::save_layer(int, int64_t, uint8_t*) {}

// A row: the latent each owning layer stores, and the indexer key each scoring layer stores.
size_t MistralLarge4::card_row_bytes() const {
    return (size_t) n_own_ * row_ * sizeof(uint16_t) + (size_t) n_full_ * idx_dim_ * sizeof(float);
}

void MistralLarge4::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t lat = (size_t) row_ * sizeof(uint16_t), key = (size_t) idx_dim_ * sizeof(float);
    for (const Block& B : blocks_) {
        if (n == 0) break;
        if (B.plan.owns_kv && B.lat) {
            ck(cudaMemcpy(dst, (const uint8_t*) B.lat + (size_t) first * lat, (size_t) n * lat, cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * lat;
        }
        if (B.plan.indexer_full && B.pool) {
            ck(cudaMemcpy(dst, (const uint8_t*) B.pool + (size_t) first * key, (size_t) n * key, cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * key;
        }
    }
}

void MistralLarge4::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t lat = (size_t) row_ * sizeof(uint16_t), key = (size_t) idx_dim_ * sizeof(float);
    const size_t skip = (size_t) (first - from), all = (size_t) count;
    for (const Block& B : blocks_) {
        if (n == 0) break;
        if (B.plan.owns_kv && B.lat) {
            ck(cudaMemcpy((uint8_t*) B.lat + (size_t) first * lat, src + skip * lat, (size_t) n * lat,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += all * lat;
        }
        if (B.plan.indexer_full && B.pool) {
            ck(cudaMemcpy((uint8_t*) B.pool + (size_t) first * key, src + skip * key, (size_t) n * key,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += all * key;
        }
    }
}

// While a token sees top_k tokens or fewer, every one is chosen and its attention is plain causal attention.  That
// holds up to a position, so the tokens of a call are dense first and sparse after.
int MistralLarge4::dense_rows(int64_t pos, int n) const {
    const int64_t last_dense = (int64_t) idx_top_k_ + kPool - 2;
    return (int) std::clamp<int64_t>(last_dense - pos + 1, 0, n);
}

// A scoring layer: the indexer's keys of the n tokens from `pos` into its cache, then for the tokens that see more
// than top_k the chosen tokens into sel_.  xn_ and xq_ hold the layer's input.
void MistralLarge4::indexer(Block& B, int64_t pos, int n) {
    // the query: the normed input through q_b, SSMax a head, turned on its first n_rot values (pairs i, i + n_rot/2)
    layer_norm(xn_, fp(B.idx_q_norm), fp(B.idx_q_norm_b), iqn_, n_embd_, n, norm_eps_, st_);
    lin(B.idx_q_b, iqn_, iq_, n);
    ssmax_scale(iq_, fp(B.idx_ssmax_s), fp(B.idx_ssmax_b), idx_dim_, idx_heads_, pos, n, st_);
    if (B.plan.has_rope) rope_ext(iq_, idx_dim_, idx_heads_, 0, n_rot_, rope_, nullptr, true, pos, n, st_);
    // the key: through attn_k, LayerNormed, turned the same way, into the pool (one token a pool: the key itself)
    lin(B.idx_k, xn_, ikr_, n, xq_);
    layer_norm(ikr_, fp(B.idx_k_norm), fp(B.idx_k_norm_b), ik_, idx_dim_, n, norm_eps_, st_);
    if (B.plan.has_rope) rope_ext(ik_, idx_dim_, 1, 0, n_rot_, rope_, nullptr, true, pos, n, st_);
    idx_store(ik_, ig_, B.ring, ring_, idx_dim_, pos, n, st_);
    idx_pool(B.ring, ring_, B.pool, ape_, idx_dim_, kPool, pos, n, st_);
    lin(B.idx_proj, xn_, iw_, n, xq_);

    const int dense = dense_rows(pos, n);
    if (dense >= n) return;
    const int n_top = idx_top_k_ / kPool;
    const int first = dense, m = n - dense;
    const int64_t at = pos + first;
    for (int t = 0, c = 0; t < m; t += c) {
        c = std::min(score_rows_, m - t);
        const float* q = iq_ + (size_t) (first + t) * idx_heads_ * idx_dim_;
        const float* w = iw_ + (size_t) (first + t) * idx_heads_;
        if (m >= kStageRows)
            idx_score_batch(blas_, q, w, B.pool, score_, score_stride_, idx_dim_, idx_heads_, kPool, at + t, c, akeys_,
                            akeys_n_, st_);
        else
            idx_score(q, w, B.pool, score_, score_stride_, idx_dim_, idx_heads_, kPool, at + t, c, st_);
        idx_select(score_, score_stride_, sel_ + (size_t) (first + t) * sel_stride_, sel_stride_, n_top, kPool, at + t,
                   c, st_);
    }
}

void MistralLarge4::attention(Block& B, int64_t pos, int n) {
    if (rows_of_) {
        // step(): a row at a time, the whole layer, with the row's sequence's caches
        const Row* rows = std::exchange(rows_of_, nullptr);
        float *const xn = xn_, *const ao = ao_;
        void* const xq = xq_;
        const size_t q8 = kernels::native_q8_1_bytes(n_embd_);
        for (int i = 0; i < n; ++i) {
            const int s = rows[i].seq;
            Block one = B;
            one.lat = (uint16_t*) seq_buf(s, B.lat);
            if (B.pool) one.pool = (float*) seq_buf(s, B.pool);
            if (B.ring) one.ring = (float*) seq_buf(s, B.ring);
            xn_ = xn + (size_t) i * n_embd_;
            xq_ = (uint8_t*) xq + (size_t) i * q8;
            ao_ = ao + (size_t) i * n_embd_;
            attention(one, rows[i].pos, 1);
        }
        xn_ = xn;
        xq_ = xq;
        ao_ = ao;
        rows_of_ = rows;
        return;
    }
    const bool rope = B.plan.has_rope;
    const int nope = rope ? mla_k_ - n_rot_ : mla_k_;
    const size_t heads = (size_t) n * n_head_;

    if (B.plan.indexer_full) indexer(B, pos, n);
    // else: sel_ holds the choice of the full indexer before this layer, for these same tokens

    // the queries: a head's nope part through k_b (per head: kv_rank values from nope), the turned tail beside it,
    // zeros there where the layer does not turn; then SSMax a head
    lin(B.wq, xn_, q_, n, xq_);
    const float* x = q_;
    if (rope) {
        copy_cols(qn_, (size_t) nope, q_, (size_t) mla_k_, (size_t) nope, heads, st_);
        x = qn_;
    }
    lin(B.k_b, x, qabs_, n, nullptr, kv_rank_);
    copy_cols(qc_, (size_t) row_, qabs_, (size_t) kv_rank_, (size_t) kv_rank_, heads, st_);
    if (rope) {
        copy_cols(qc_ + kv_rank_, (size_t) row_, q_ + nope, (size_t) mla_k_, (size_t) n_rot_, heads, st_);
        rope_ext(qc_, row_, n_head_, kv_rank_, n_rot_, rope_, fp(rope_factors_), false, pos, n, st_);
    } else {
        zero_cols(qc_ + kv_rank_, (size_t) row_, (size_t) n_rot_, heads, st_);
    }
    ssmax_scale(qc_, fp(B.ssmax_s), fp(B.ssmax_b), row_, n_head_, pos, n, st_);

    // the keys: the latent, normed, with its turned tail, stored as fp16 at the tokens' rows
    if (B.plan.owns_kv) {
        const int kvw = kv_rank_ + (rope ? n_rot_ : 0);
        lin(B.kv_a, xn_, t2_, n, xq_);
        copy_cols(kv_, (size_t) kv_rank_, t2_, (size_t) kvw, (size_t) kv_rank_, (size_t) n, st_);
        rms_norm(kv_, fp(B.kv_a_norm), kvn_, kv_rank_, n, rms_eps_, st_);
        copy_cols(kr_, (size_t) row_, kvn_, (size_t) kv_rank_, (size_t) kv_rank_, (size_t) n, st_);
        if (rope) {
            copy_cols(kr_ + kv_rank_, (size_t) row_, t2_ + kv_rank_, (size_t) kvw, (size_t) n_rot_, (size_t) n, st_);
            rope_ext(kr_, row_, 1, kv_rank_, n_rot_, rope_, fp(rope_factors_), false, pos, n, st_);
        } else {
            zero_cols(kr_ + kv_rank_, (size_t) row_, (size_t) n_rot_, (size_t) n, st_);
        }
        store_f16(kr_, B.lat, row_, pos, n, st_);
    }

    // the attention over the chosen rows: a prompt batch's tokens share their products, a few tokens get one each
    const int dense = dense_rows(pos, n);
    const int together = n < kStageRows ? 1 : std::min(rows_, kAttnRows);
    dsa_attend(blas_, qc_, B.lat, sel_, sel_stride_, idx_top_k_, kPool, pos, n, dense, akeys_, ascore_, together, kqv_,
               row_, n_head_, 1.f / std::sqrt((float) mla_k_), st_);
    // the values are the latents alone: a head's first kv_rank values, through v_b, then the output projection
    copy_cols(kqc_, (size_t) kv_rank_, kqv_, (size_t) row_, (size_t) kv_rank_, heads, st_);
    lin(B.v_b, kqc_, vout_, n, nullptr, mla_v_);
    lin(B.wo, vout_, ao_, n);
}

void MistralLarge4::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// What ggml's MoE graph does for this model: a softmax over every expert's score, the n_used most likely, their
// weights summed to one and scaled.
void MistralLarge4::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool MistralLarge4::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert.  It runs on the GPU while the host routes.
void MistralLarge4::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

// A layer from its input to its router's copy (the dense layer: to its end).
void MistralLarge4::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attention(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void MistralLarge4::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void MistralLarge4::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void MistralLarge4::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
}

}  // namespace strata::poly
