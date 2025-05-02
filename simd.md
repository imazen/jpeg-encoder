# SIMD Implementation Notes for Adaptive Quantization Math

This document outlines the process, decisions, and findings related to porting the SIMD-accelerated mathematical functions from `jpegli/lib/jpegli/adaptive_quantization.cc` (using the Google Highway library) to Rust (`src/jpegli/adaptive_quant_math.rs`) using the `wide` crate.

## Goal

Extract core mathematical, SIMD-accelerated functions used in adaptive quantization into a reusable Rust module, replacing Highway with `wide`.

## SIMD Library Choice

The `wide` crate was chosen over `std::simd` because `std::simd` is currently only available on nightly Rust.

## Ported Functions

The following functions were identified as pure mathematical operations suitable for extraction:

*   `masking_sqrt`: Computes `0.25 * sqrt(v * sqrt(K_MASKING_SQRT_MUL * 1e8) + K_MASKING_SQRT_LOG_OFFSET)`.
*   `eval_rational_polynomial`: Evaluates a ratio of two polynomials using Horner's method. Implemented as a macro (`eval_rational_polynomial!`) due to limitations with const generics on stable Rust.
*   `fast_log2f`: Approximates `log2(x)` using range reduction and the rational polynomial macro.
*   `fast_pow2f`: Approximates `2^x` using range reduction and polynomial evaluation.
*   `fast_pow2f_scalar`: Scalar version of `fast_pow2f`.
*   `ratio_of_derivatives_of_cubic_root_to_simple_gamma`: Computes a ratio related to gamma correction, parameterized by a `const bool INVERT`.
*   `compute_mask`: Applies a specific formula involving division, `mul_add`, and `max`.
*   `sort4`: Helper to sort 4 SIMD vectors using pairwise min/max.
*   `update_min4`: Helper to update the 4 smallest vectors with a new candidate.
*   `fast_reciprocal_nr`: Computes reciprocal using Newton-Raphson iteration, starting with `wide`'s `recip()`.

## Excluded Functions (Initially)

Initially, the following functions were considered too algorithm-specific for the math module, as they involve significant logic tied to accessing image buffers, handling 8x8 block structures, and specific AQ steps:

*   `GammaModulation`
*   `HfModulation`
*   `FuzzyErosion`
*   `ComputePreErosion`

However, per user request, the core SIMD processing loops from these functions were also ported and included in `adaptive_quant_math.rs` as helper functions (e.g., `compute_hf_metric_8x8`, `compute_gamma_sum_8x8`, `compute_diff_buffer_row`, `compute_fuzzy_erosion_row`). This makes the module contain both pure math utilities and some algorithm-specific SIMD kernels intended to be called by a higher-level AQ implementation (like `adaptive_quant_v2.rs`).

## `wide` Crate API Usage and Findings

Mapping from Highway to `wide` was generally straightforward.

**Key `wide` APIs Used (primarily `f32x8`, `i32x8`, `u32x8`):**

*   **Arithmetic:** `+`, `-`, `*`, `/` (via `Add`, `Sub`, `Mul`, `Div` traits)
*   **Methods:** `.abs()`, `.sqrt()`, `.recip()`, `.floor()`, `.max()`, `.min()`, `.mul_add()`, `.splat()`, `.new()`, `.to_array()`, `.round_float()`, `.trunc_int()`
*   **Bitwise:** `&`, `|`, `^`, `!`, `<<`, `>>` (via `BitAnd`, `BitOr`, `BitXor`, `Not`, `Shl`, `Shr` traits)
*   **Constants:** `::ZERO`, `::ONE`
*   **Comparison:** `.cmp_eq()`, `.cmp_ne()`, `.cmp_lt()`, etc. (via `CmpEq` etc. traits) - Note: These return masks of the *same* type (e.g., `f32x8`), suitable for use with `.blend()`.
*   **Blending:** `.blend(mask, true_vec, false_vec)`

**Missing APIs / Workarounds:**

*   **`reduce_max` for Floats:** Not implemented in `wide` for `f32x8`. An extension trait (`WideF32x8Ext`) was created in `adaptive_quant_math.rs` to provide this functionality using `bytemuck::cast` to access the underlying array.
*   **`ApproximateReciprocal`:** Highway's approximate version isn't available. `wide`'s `recip()` was used as the initial guess for `fast_reciprocal_nr`.
*   **`mul_neg_add` / `mul_neg_sub` style functions:** Specific fused operations like `c - (a*b)` (`mul_neg_add`) weren't directly available. Expanded forms like `c - a * b` were used.
*   **`ZeroIfNegative`:** Implemented using `v.max(f32x8::ZERO)`.

**Other Implementation Notes:**

*   **Const Generics:** Using const generics for array sizes in function arguments (e.g., `fn func<const N: usize>(data: &[T; N])`) requires the `generic_const_exprs` nightly feature when performing operations like `N + 1` in the signature. To keep the code on stable Rust, `eval_rational_polynomial` was converted to a macro.
*   **Constant Loading:** Highway's `LoadDup128` combined with `HWY_REP4` was used to load and splat coefficients. The Rust equivalent uses `wide::f32x8::splat()` with a single coefficient value, making the constant arrays smaller (e.g., 3 elements instead of 12 for `fast_log2f`).
*   **Bitcasting:** Reinterpreting vector types (e.g., `f32x8` <-> `i32x8`) requires the `bytemuck` crate (`bytemuck::cast`).
*   **Integer Conversion:** `wide` provides specific methods like `round_float()` (float to int) and `trunc_int()` (float to int, truncate), and `From` implementations (e.g., `i32x8::from(i16x8)`).
*   **Testing:** Tests were written for each function, often comparing against a scalar implementation or standard library functions (`f32::log2`, `f32::powf`). Approximate equality checking (`assert_vec_approx_eq`) was necessary for floating-point results. 