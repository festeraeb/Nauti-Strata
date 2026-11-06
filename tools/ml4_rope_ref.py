#!/usr/bin/env python3
"""Independent reference for src/poly/ml4_layout_test.cpp: ggml's YaRN rotary math in double.

Prints ggml_rope_yarn_corr_dims() and rope_yarn() for the cases the C++ test asserts, so the seeds in the
test can be checked against ggml's own formulas (ggml/src/ggml.c and ggml-cuda/rope.cu) rather than
against the C++ under test.
"""
import math


def corr_dim(n_rot, n_ctx_orig, base, rot):
    return n_rot * math.log(n_ctx_orig / (rot * 2 * math.pi)) / (2 * math.log(base))


def corr_dims(n_rot, n_ctx_orig, base, beta_fast, beta_slow):
    lo = max(0.0, math.floor(corr_dim(n_rot, n_ctx_orig, base, beta_fast)))
    hi = min(n_rot - 1, math.ceil(corr_dim(n_rot, n_ctx_orig, base, beta_slow)))
    return lo, hi


def ramp(lo, hi, i0):
    y = (i0 / 2 - lo) / max(0.001, hi - lo)
    return 1 - min(1.0, max(0.0, y))


def rope_yarn(pos, i, n_rot, base, freq_scale, ext_factor, attn_factor, lo, hi, factor=1.0):
    theta_scale = base ** (-2.0 / n_rot)
    theta_extrap = pos * theta_scale ** i / factor
    theta_interp = freq_scale * theta_extrap
    theta, mscale = theta_interp, attn_factor
    if ext_factor != 0.0:
        mix = ramp(lo, hi, 2 * i) * ext_factor
        theta = theta_interp * (1 - mix) + theta_extrap * mix
        mscale *= 1 + 0.1 * math.log(1 / freq_scale)
    return math.cos(theta) * mscale, math.sin(theta) * mscale


def attn_factor(freq_scale, log_mul):
    def mscale(scale, m):
        return 1.0 if scale <= 1.0 else 0.1 * m * math.log(scale) + 1.0
    factor = 1 / freq_scale
    f = mscale(factor, 1.0) / mscale(factor, log_mul) if log_mul else mscale(factor, 1.0)
    return f * 1 / (1 + 0.1 * math.log(factor))


def main():
    n_rot, orig, base = 64, 512, 10000.0
    fs = 1 / 2048
    lo, hi = corr_dims(n_rot, orig, base, 32.0, 1.0)
    print(f"corr dims lo={lo} hi={hi}")
    af = attn_factor(fs, 1.0)
    print(f"attn_factor={af!r}  magnitude under yarn={af * (1 + 0.1 * math.log(2048))!r}")
    for pos, i, fac in [(0, 0, 1.0), (100, 0, 1.0), (100, 31, 1.0), (1000, 10, 1.0), (100, 0, 4.0)]:
        c, s = rope_yarn(pos, i, n_rot, base, fs, 1.0, af, lo, hi, fac)
        print(f"pos={pos} pair={i} factor={fac}: cos={c!r} sin={s!r}")
    c, s = rope_yarn(7, 3, n_rot, base, 1.0, 0.0, 0.5, lo, hi)
    print(f"plain pos=7 pair=3 attn=0.5: cos={c!r} sin={s!r}")
    for pos, sc, b in [(0, 2.0, 0.5), (1, 1.0, 0.0), (99, 0.5, -1.0)]:
        print(f"ssmax pos={pos}: {sc * math.log(pos + 1) + b!r}")


if __name__ == "__main__":
    main()
