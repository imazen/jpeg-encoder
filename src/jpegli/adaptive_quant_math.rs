// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD and scalar mathematical functions used for adaptive quantization.
//! Ported from C++ Jpegli (`adaptive_quantization.cc`) and using the `wide` crate.
//! This module provides low-level math operations, often operating on 8-wide vectors.

#![allow(dead_code)] // TODO: Remove once integrated

use wide::*;
use std::ops::*;
// TODO: Add bytemuck dependency: cargo add bytemuck
use bytemuck;
use arrayref;

// Extension trait for wide::f32x8
trait WideF32x8Ext {
    fn reduce_max(self) -> f32;
}

impl WideF32x8Ext for f32x8 {
    #[inline]
    fn reduce_max(self) -> f32 {
        let arr: [f32; 8] = bytemuck::cast(self);
        // Use f32::max which handles NaNs correctly (propagates them)
        arr[0].max(arr[1]).max(arr[2].max(arr[3]))
            .max(arr[4].max(arr[5]).max(arr[6].max(arr[7])))
    }
}

// Constants from adaptive_quantization.cc
pub(crate) const K_INPUT_SCALING: f32 = 1.0 / 255.0;

// Constants for RatioOfDerivativesOfCubicRootToSimpleGamma
pub(crate) const K_SG_MUL: f32 = 226.0480446705883;
pub(crate) const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
pub(crate) const K_INV_LOG2E: f32 = 0.6931471805599453; // 1.0 / log2(E)
pub(crate) const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
pub(crate) const K_SG_V_OFFSET: f32 = 7.14672470003;
pub(crate) const K_GAMMA_EPSILON: f32 = 1e-2;
pub(crate) const K_GAMMA_NUM_OFFSET: f32 = K_GAMMA_EPSILON / K_INPUT_SCALING / K_INPUT_SCALING;
pub(crate) const K_GAMMA_NUM_MUL: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
pub(crate) const K_GAMMA_V_OFFSET: f32 =
    (K_SG_V_OFFSET * K_INV_LOG2E + K_GAMMA_EPSILON) / K_INPUT_SCALING;
pub(crate) const K_GAMMA_DEN_MUL: f32 =
    K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;

// Constant for GammaModulation
pub(crate) const K_GAMMA_MODULATION_BIAS: f32 = 0.16 / K_INPUT_SCALING;
pub(crate) const K_GAMMA_MODULATION_SCALE: f32 = K_INPUT_SCALING / 64.0;
// ln(2) folded in
pub(crate) const K_GAMMA_MODULATION_GAMMA: f32 = -0.15526878023684174 * K_INV_LOG2E;

// Constant for HfModulation
pub(crate) const K_HF_MODULATION_SUM_COEFF: f32 = -2.0052193233688884 * K_INPUT_SCALING / 112.0;

// Constant for ComputePreErosion
pub(crate) const K_MATCH_GAMMA_OFFSET: f32 = 0.019 / K_INPUT_SCALING;
pub(crate) const K_PRE_EROSION_LIMIT: f32 = 0.2;

// Constants for MaskingSqrt
pub(crate) const K_MASKING_SQRT_LOG_OFFSET: f32 = 28.0;
pub(crate) const K_MASKING_SQRT_MUL: f32 = 211.50759899638012;

/// Simulates `MaskingSqrt` from the C++ code.
/// ```cpp
/// template <typename D, typename V>
/// V MaskingSqrt(const D d, V v) {
///   static const float kLogOffset = 28;
///   static const float kMul = 211.50759899638012f;
///   const auto mul_v = Set(d, kMul * 1e8);
///   const auto offset_v = Set(d, kLogOffset);
///   return Mul(Set(d, 0.25f), Sqrt(MulAdd(v, Sqrt(mul_v), offset_v)));
/// }
/// ```
#[inline(always)]
pub(crate) fn masking_sqrt(v: f32x8) -> f32x8 {
    let mul_v = f32x8::splat(K_MASKING_SQRT_MUL * 1e8);
    let offset_v = f32x8::splat(K_MASKING_SQRT_LOG_OFFSET);
    f32x8::splat(0.25) * v.mul_add(mul_v.sqrt(), offset_v).sqrt()
}

// Macro version to avoid nightly features needed for generic const evaluation in functions.
macro_rules! eval_rational_polynomial {
    ($x:expr, $p:expr, $q:expr) => {
        {
            let x = $x;
            let p = $p;
            let q = $q;
            let deg_p = p.len() - 1;
            let deg_q = q.len() - 1;

            // Evaluate numerator polynomial p using Horner's method
            let mut yp = f32x8::splat(p[0]);
            for i in 1..=deg_p {
                yp = yp.mul_add(x, f32x8::splat(p[i]));
            }

            // Evaluate denominator polynomial q using Horner's method
            let mut yq = f32x8::splat(q[0]);
            for i in 1..=deg_q {
                yq = yq.mul_add(x, f32x8::splat(q[i]));
            }

            // Perform the division
            yp / yq
        }
    };
}

/// Computes base-2 logarithm like std::log2. Undefined if negative / NaN.
/// L1 error ~3.9E-6
/// Corresponds to `FastLog2f` in C++.
#[inline(always)]
pub(crate) fn fast_log2f(x: f32x8) -> f32x8 {
    // 2,2 rational polynomial approximation of std::log1p(x) / std::log(2).
    // Coefficients from C++ code (highest degree first for Horner's method)
    // Note: C++ used arrays of size 12 (4*3) with HWY_REP4 because it used
    // LoadDup128, which loads 4 elements and duplicates. Using splat here
    // with arrays of size 3 achieves the same result.
    const P: [f32; 3] = [7.4245873327820566E-01, 1.4287160470083755E+00, -1.8503833400518310E-06];
    const Q: [f32; 3] = [1.7409343003366853E-01, 1.0096718572241148E+00, 9.9032814277590719E-01];

    // Need bytemuck for this cast
    let x_bits: i32x8 = bytemuck::cast(x);

    // Range reduction to [-1/3, 1/3] - 3 integer, 2 float ops
    let exp_bits = x_bits - i32x8::splat(0x3f2aaaab); // 0x3f2aaaab = 2/3 in f32 bits

    // Shifted exponent = log2; also used to clear mantissa.
    let exp_shifted: i32x8 = exp_bits >> 23;
    let mantissa_bits = x_bits - (exp_shifted << 23);
    let mantissa: f32x8 = bytemuck::cast(mantissa_bits);

    // Use i32x8 -> f32x8 conversion
    let exp_val = exp_shifted.round_float(); // wide's i32->f32 conversion

    let poly_arg = mantissa - f32x8::splat(1.0);
    let poly_res = eval_rational_polynomial!(poly_arg, &P, &Q);

    poly_res + exp_val
}

/// Scalar version of fast_log2f.
/// Computes base-2 logarithm like std::log2. Undefined if negative / NaN.
#[inline(always)]
pub(crate) fn fast_log2f_scalar(x: f32) -> f32 {
    // Based on the SIMD version logic
    const P: [f32; 3] = [7.4245873327820566E-01, 1.4287160470083755E+00, -1.8503833400518310E-06];
    const Q: [f32; 3] = [1.7409343003366853E-01, 1.0096718572241148E+00, 9.9032814277590719E-01];

    let x_bits = x.to_bits();
    // Use wrapping_sub for integer arithmetic matching potential SIMD behavior
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab); // 0x3f2aaaab = 2/3 in f32 bits
    let exp_shifted = (exp_bits >> 23) as i32;
    let mantissa_bits = x_bits.wrapping_sub((exp_shifted as u32) << 23);
    let mantissa = f32::from_bits(mantissa_bits);
    let exp_val = exp_shifted as f32;

    let poly_arg = mantissa - 1.0;

    // Evaluate polynomials using Horner's method (scalar)
    let mut yp = P[0];
    for i in 1..=2 {
        yp = yp * poly_arg + P[i];
    }
    let mut yq = Q[0];
    for i in 1..=2 {
        yq = yq * poly_arg + Q[i];
    }

    let poly_res = if yq.abs() > 1e-9 { yp / yq } else { 0.0 }; // Avoid division by zero
    poly_res + exp_val
}

/// Scalar version of FastPow2f, computes 2^x.
/// max relative error ~3e-7
/// Needed for the final step in PerBlockModulations.
#[inline(always)]
pub(crate) fn fast_pow2f_scalar(x: f32) -> f32 {
    let floorx = x.floor();
    let exp_int = (floorx as i32) + 127;
    let exp_bits = (exp_int as u32) << 23;
    let exp = f32::from_bits(exp_bits);
    let frac = x - floorx;

    let mut num = frac + 1.01749063e+01;
    num = num * frac + 4.88687798e+01;
    num = num * frac + 9.85506591e+01;
    num *= exp;

    let mut den = frac * 2.10242958e-01 + -2.22328856e-02;
    den = den * frac + -1.94414990e+01;
    den = den * frac + 9.85506633e+01;

    num / den
}

/// Computes 2^x.
/// max relative error ~3e-7
/// Corresponds to `FastPow2f` in C++.
#[inline(always)]
pub(crate) fn fast_pow2f(x: f32x8) -> f32x8 {
    let floorx = x.floor();

    // Convert floorx to i32x8, add 127, shift, cast back to f32x8 via bits
    // Note: wide::i32x8::round_float() converts f32->i32 (truncates)
    let exp_int = floorx.trunc_int() + i32x8::splat(127);
    let exp_bits = exp_int << 23; // Shift logical left
    let exp: f32x8 = bytemuck::cast(exp_bits);

    let frac = x - floorx;

    let mut num = frac + f32x8::splat(1.01749063e+01);
    num = num.mul_add(frac, f32x8::splat(4.88687798e+01));
    num = num.mul_add(frac, f32x8::splat(9.85506591e+01));
    num *= exp;

    let mut den = frac.mul_add(f32x8::splat(2.10242958e-01), f32x8::splat(-2.22328856e-02));
    den = den.mul_add(frac, f32x8::splat(-1.94414990e+01));
    den = den.mul_add(frac, f32x8::splat(9.85506633e+01));

    num / den
}

/// Computes the ratio of the derivatives of the cubic root function to the simple gamma function.
/// This allows quantization to move from jxl's opsin space to butteraugli's log-gamma space.
/// Corresponds to `RatioOfDerivativesOfCubicRootToSimpleGamma` in C++.
#[inline(always)]
pub(crate) fn ratio_of_derivatives_of_cubic_root_to_simple_gamma<const INVERT: bool>(v: f32x8) -> f32x8 {
    // Clamp to zero to avoid issues with negative inputs (photons can't be negative)
    let v = v.max(f32x8::ZERO);

    let num_mul = f32x8::splat(K_GAMMA_NUM_MUL);
    let num_offset = f32x8::splat(K_GAMMA_NUM_OFFSET);
    let den_offset = f32x8::splat(K_GAMMA_V_OFFSET);
    let den_mul = f32x8::splat(K_GAMMA_DEN_MUL);

    let v2 = v * v;

    let num = num_mul.mul_add(v2, num_offset);
    let den = den_mul.mul(v).mul_add(v2, den_offset);

    if INVERT {
        num / den
    } else {
        den / num
    }
}

/// Corresponds to `ComputeMask` in C++.
#[inline(always)]
pub(crate) fn compute_mask(out_val: f32x8) -> f32x8 {
    let k_base = f32x8::splat(-0.74174993);
    let k_mul4 = f32x8::splat(3.2353257320940401);
    let k_mul2 = f32x8::splat(12.906028311180409);
    let k_offset2 = f32x8::splat(305.04035728311436);
    let k_mul3 = f32x8::splat(5.0220313103171232);
    let k_offset3 = f32x8::splat(2.1925739705298404);
    let k_offset4 = f32x8::splat(0.25) * k_offset3;
    let k_mul0 = f32x8::splat(0.74760422233706747);
    let k1 = f32x8::ONE;

    // Avoid division by zero.
    let v1 = (out_val * k_mul0).max(f32x8::splat(1e-3));
    let v2 = k1 / (v1 + k_offset2);
    let v3 = k1 / (v1 * v1 + k_offset3);
    let v4 = k1 / (v1 * v1 + k_offset4);

    k_base.add(k_mul4.mul_add(v4, k_mul2.mul_add(v2, k_mul3 * v3)))
}

/// Helper for FuzzyErosion: Sorts 4 vectors using pairwise min/max.
#[inline(always)]
pub(crate) fn sort4(min0: &mut f32x8, min1: &mut f32x8, min2: &mut f32x8, min3: &mut f32x8) {
    let tmp0 = min0.min(*min1);
    let tmp1 = min0.max(*min1);
    let tmp2 = min2.min(*min3);
    let tmp3 = min2.max(*min3);
    let tmp4 = tmp0.max(tmp2);
    let tmp5 = tmp1.min(tmp3);
    *min0 = tmp0.min(tmp2);
    *min1 = tmp4.min(tmp5);
    *min2 = tmp4.max(tmp5);
    *min3 = tmp1.max(tmp3);
}

/// Helper for FuzzyErosion: Updates the 4 minimum vectors with a new vector `v`.
#[inline(always)]
pub(crate) fn update_min4(v: f32x8, min0: &mut f32x8, min1: &mut f32x8, min2: &mut f32x8, min3: &mut f32x8) {
    let tmp0 = min0.max(v);
    let tmp1 = min1.max(tmp0);
    let tmp2 = min2.max(tmp1);
    *min0 = min0.min(v);
    *min1 = min1.min(tmp0);
    *min2 = min2.min(tmp1);
    *min3 = min3.min(tmp2);
}

/// Computes reciprocal using one Newton-Raphson iteration.
/// Based on `FastDivision<float, V>::ReciprocalNR` in C++.
/// `ApproximateReciprocal` is replaced with `wide::recip()`.
#[inline(always)]
pub(crate) fn fast_reciprocal_nr(x: f32x8) -> f32x8 {
    // Initial approximation
    let rcp = x.recip();
    // One Newton-Raphson iteration: result = rcp * (2.0 - x * rcp)
    // C++ used: NegMulAdd(x_rcp, rcp, Add(rcp, rcp)) = 2*rcp - (x*rcp)*rcp
    let two = f32x8::splat(2.0);
    let x_rcp = x * rcp;
    // rcp.mul_add(x.mul_neg(rcp), two) // FMA version: rcp * (2.0 - x*rcp)
    // rcp.mul_add(x.mul_neg(), two) // Simplified, check wide API if mul_neg exists
    // wide f32x8 doesn't have mul_neg. Let's use the expanded form:
    // rcp * (2.0 - x * rcp)
    let correction = two - x_rcp;
    rcp * correction
    // Alternative C++ form: 2.0 * rcp - x * rcp * rcp
    // two.mul_add(rcp, x.mul_neg().mul(rcp).mul(rcp)) // complex
    // rcp.mul_add(two, x_rcp.mul_neg().mul(rcp)) // Also complex
    // Let's stick to the simple (rcp * (2.0 - x*rcp)) expansion
}

/// Computes the sum of horizontal and vertical absolute differences in an 8x8 block using SIMD.
/// Corresponds to the core loop inside C++ `HfModulation`.
///
/// Assumes input slices are appropriately padded or accessed such that neighbor reads
/// (e.g., `x+1`, `x-1`) are valid for the processed range `x_start..x_start+8`.
///
/// Args:
/// * `rows`: A slice of 9 slices, representing the 8 rows of the block (y to y+7)
///           plus the row below (y+8). Each inner slice must contain at least
///           columns `x_start` to `x_start+8` (9 elements total starting at `x_start`).
/// * `x_start`: The starting column index within the slices.
///
/// Returns:
/// * The scalar sum of masked horizontal + vertical absolute differences for the 8x8 block.
#[inline(always)]
pub(crate) fn compute_hf_metric_8x8(rows: &[&[f32]], x_start: usize) -> f32 {
    assert!(rows.len() >= 9, "Need 9 rows for HfModulation (8 block + 1 below)");
    for row in rows {
        assert!(row.len() >= x_start + 9, "Rows need 9 elements starting at x_start");
    }

    let mut sum_vec = f32x8::ZERO;

    // Mask to zero out the difference calculation for the rightmost pixel (lane 7)
    // Since wide uses f32x8 for masks, we create a float mask.
    let h_mask_arr: [f32; 8] = [f32::from_bits(!0), f32::from_bits(!0), f32::from_bits(!0), f32::from_bits(!0),
                               f32::from_bits(!0), f32::from_bits(!0), f32::from_bits(!0), f32::from_bits(0)];
    let h_mask = f32x8::new(h_mask_arr);

    for dy in 0..8 {
        let row_in = &rows[dy][x_start..];      // Slice for row y+dy, starting at x
        let row_in_next = &rows[dy + 1][x_start..]; // Slice for row y+dy+1, starting at x

        // Assume row_in starts at an aligned boundary for 'p' and 'pd'.
        // This might require the caller to ensure alignment or use unaligned loads.
        // let p = f32x8::from_slice_unaligned(&row_in[0..8]);
        // let pd = f32x8::from_slice_unaligned(&row_in_next[0..8]);
        // Use arrayref for potentially unaligned loads
        let p = f32x8::new(arrayref::array_ref![row_in, 0, 8].clone());
        let pd = f32x8::new(arrayref::array_ref![row_in_next, 0, 8].clone());

        // Construct the right-neighbor vector 'pr'. This is inherently unaligned.
        // We load scalars [1..9] from the current row slice.
        let pr = f32x8::new(arrayref::array_ref![row_in, 1, 8].clone());

        // Vertical difference
        sum_vec += (p - pd).abs();

        // Horizontal difference (masked)
        let h_diff = (p - pr).abs();
        sum_vec += h_diff & h_mask; // Bitwise AND acts as select(mask, value, 0)
    }

    sum_vec.reduce_add()
}

/// Computes the gamma sum component for an 8x8 block using SIMD.
/// Corresponds to the core loop inside C++ `GammaModulation`.
///
/// Assumes input slices provide valid data for the range `x_start..x_start+8`.
///
/// Args:
/// * `rows`: A slice of 8 slices, representing the 8 rows of the block (y to y+7).
///           Each inner slice must contain at least columns `x_start` to `x_start+7` (8 elements).
/// * `x_start`: The starting column index within the slices.
///
/// Returns:
/// * The scalar sum of the ratio of derivatives, used in gamma modulation calculation.
#[inline(always)]
pub(crate) fn compute_gamma_sum_8x8(rows: &[&[f32]], x_start: usize) -> f32 {
    assert!(rows.len() >= 8, "Need 8 rows for GammaModulation");
    for row in rows {
        assert!(row.len() >= x_start + 8, "Rows need 8 elements starting at x_start");
    }

    let mut overall_ratio_vec = f32x8::ZERO;
    let bias_vec = f32x8::splat(K_GAMMA_MODULATION_BIAS);

    for dy in 0..8 {
        let row_in = &rows[dy][x_start..];
        // Assume alignment or use unaligned load
        // let in_vec = f32x8::from_slice_unaligned(&row_in[0..8]);
        // Use arrayref for potentially unaligned loads
        let in_vec = f32x8::new(arrayref::array_ref![row_in, 0, 8].clone());
        let iny = in_vec + bias_vec;
        let ratio_g = ratio_of_derivatives_of_cubic_root_to_simple_gamma::<true>(iny);
        overall_ratio_vec += ratio_g;
    }

    overall_ratio_vec.reduce_add()
}

/// Computes one row of the intermediate difference buffer used in ComputePreErosion using SIMD.
/// Corresponds to the inner loop of C++ `ComputePreErosion`.
///
/// Processes `xsize` elements. Assumes input rows are padded appropriately
/// (typically by 1 element left/right) so that neighbor accesses (x-1, x+1) are valid.
///
/// Args:
/// * `row_t`: Slice for the row above (y-1), logically starting at index -1 relative to `x=0`.
/// * `row_m`: Slice for the current row (y), logically starting at index -1.
/// * `row_b`: Slice for the row below (y+1), logically starting at index -1.
/// * `xsize`: The number of *output* elements to compute (0 to xsize-1).
/// * `diff_out`: Output slice to write results to, length >= `xsize`.
/// * `prev_diff_out`: Diff buffer row from previous iteration (iy&3 != 0 case), length >= `xsize`.
#[inline(always)]
pub(crate) fn compute_diff_buffer_row(
    row_t: &[f32], // Assumed padded, length >= xsize + 2
    row_m: &[f32], // Assumed padded, length >= xsize + 2
    row_b: &[f32], // Assumed padded, length >= xsize + 2
    xsize: usize,
    diff_out: &mut [f32],
    prev_diff_out: Option<&[f32]>, // Corresponds to x=0..xsize-1
) {
    // Check lengths assuming padding is included
    let required_len = xsize + 2; // Need access up to xsize+1 for right neighbor of xsize-1
    // Debugging print statements
    eprintln!("compute_diff_buffer_row: xsize={}, required_len={}, row_t.len={}, row_m.len={}, row_b.len={}",
              xsize, required_len, row_t.len(), row_m.len(), row_b.len());
    assert!(row_t.len() >= required_len, "Assertion failed for row_t");
    assert!(row_m.len() >= required_len, "Assertion failed for row_m");
    assert!(row_b.len() >= required_len, "Assertion failed for row_b");
    assert!(diff_out.len() >= xsize);
    if let Some(prev) = prev_diff_out {
        assert!(prev.len() >= xsize);
    }

    let match_gamma_offset_v = f32x8::splat(K_MATCH_GAMMA_OFFSET);
    let quarter = f32x8::splat(0.25);
    let limit_v = f32x8::splat(K_PRE_EROSION_LIMIT);

    let mut x = 0;
    let vector_width = 8;
    let limit = xsize.saturating_sub(vector_width - 1);

    // x loops from 0..xsize-1 (output index)
    // Access padded input slices relative to x. Slices start at logical index -1.
    while x < limit {
        // Indices relative to the start of the padded slice
        let current_x_idx = x + 1; // Logical x
        let left_x_idx = x;      // Logical x-1
        let right_x_idx = x + 2; // Logical x+1

        // Load vectors based on calculated indices
        let in_m = f32x8::new(arrayref::array_ref![row_m, current_x_idx, 8].clone());
        let in_r = f32x8::new(arrayref::array_ref![row_m, right_x_idx, 8].clone());
        let in_l = f32x8::new(arrayref::array_ref![row_m, left_x_idx, 8].clone());
        let in_t = f32x8::new(arrayref::array_ref![row_t, current_x_idx, 8].clone());
        let in_b = f32x8::new(arrayref::array_ref![row_b, current_x_idx, 8].clone());

        let base = quarter * (in_r + in_l + in_t + in_b);
        let gammacv =
            ratio_of_derivatives_of_cubic_root_to_simple_gamma::<false>(in_m + match_gamma_offset_v);

        let mut diff = gammacv * (in_m - base);
        let diff_sq = diff * diff;
        let diff_clamped = diff_sq.min(limit_v);
        let diff_sqrt = masking_sqrt(diff_clamped);

        // Debug print for x=0
        if x == 0 {
            eprintln!("-- x=0 SIMD --");
            eprintln!("in_m[0]: {:.8}", in_m.to_array()[0]);
            eprintln!("in_r[0]: {:.8}", in_r.to_array()[0]);
            eprintln!("in_l[0]: {:.8}", in_l.to_array()[0]);
            eprintln!("in_t[0]: {:.8}", in_t.to_array()[0]);
            eprintln!("in_b[0]: {:.8}", in_b.to_array()[0]);
            eprintln!("base[0]: {:.8}", base.to_array()[0]);
            eprintln!("gammacv[0]: {:.8}", gammacv.to_array()[0]);
            eprintln!("diff_presqrt[0]: {:.8}", diff_clamped.to_array()[0]);
            eprintln!("diff_final[0]: {:.8}", diff_sqrt.to_array()[0]);
        }

        // Restore original diff variable name for accumulation logic
        diff = diff_sqrt;

        if let Some(prev) = prev_diff_out {
             // prev_diff_out corresponds to logical x=0..xsize-1
             let prev_val = f32x8::new(arrayref::array_ref![prev, x, 8].clone());
             diff += prev_val;
        }

        // Store results - requires mutable slice
        diff_out[x..x + vector_width].copy_from_slice(&diff.to_array());
        x += vector_width;
    }

    // Handle remainder
    while x < xsize {
         let current_x_idx = x + 1;
         let left_x_idx = x;
         let right_x_idx = x + 2;

         let in_m = row_m[current_x_idx];
         let in_r = row_m[right_x_idx];
         let in_l = row_m[left_x_idx];
         let in_t = row_t[current_x_idx];
         let in_b = row_b[current_x_idx];

         let base = 0.25 * (in_r + in_l + in_t + in_b);
         let gammacv = scalar_ratio_of_derivatives::<false>(in_m + K_MATCH_GAMMA_OFFSET);
         let mut diff_s = gammacv * (in_m - base);
         diff_s *= diff_s;
         diff_s = diff_s.min(K_PRE_EROSION_LIMIT);
         let diff_clamped_s = diff_s;
         diff_s = scalar_masking_sqrt(diff_s);
         let diff_sqrt_s = diff_s;

         // Debug print for x=0
         if x == 0 {
             eprintln!("-- x=0 Scalar --");
             eprintln!("in_m: {:.8}", in_m);
             eprintln!("in_r: {:.8}", in_r);
             eprintln!("in_l: {:.8}", in_l);
             eprintln!("in_t: {:.8}", in_t);
             eprintln!("in_b: {:.8}", in_b);
             eprintln!("base: {:.8}", base);
             eprintln!("gammacv: {:.8}", gammacv);
             eprintln!("diff_presqrt: {:.8}", diff_clamped_s);
             eprintln!("diff_final: {:.8}", diff_sqrt_s);
         }

         if let Some(prev) = prev_diff_out {
             diff_s += prev[x];
         }
         diff_out[x] = diff_s;
        x += 1;
    }
}

/// Scalar version of ratio_of_derivatives_of_cubic_root_to_simple_gamma.
/// Computes the ratio of derivatives, potentially inverted.
#[inline(always)]
pub(crate) fn scalar_ratio_of_derivatives<const INVERT: bool>(v_scalar: f32) -> f32 {
    let v = v_scalar.max(0.0);
    let v2 = v * v;
    let num = K_GAMMA_NUM_MUL.mul_add(v2, K_GAMMA_NUM_OFFSET);
    let den = (K_GAMMA_DEN_MUL * v).mul_add(v2, K_GAMMA_V_OFFSET);
    if INVERT {
        num / den // Consider adding safety checks for den == 0?
    } else {
        den / num // Consider adding safety checks for num == 0?
    }
}

/// Scalar version of compute_mask.
/// Applies a formula based on masking theory.
#[inline(always)]
pub(crate) fn compute_mask_scalar(out_val: f32) -> f32 {
    let k_base = -0.74174993;
    let k_mul4 = 3.2353257320940401;
    let k_mul2 = 12.906028311180409;
    let k_offset2 = 305.04035728311436;
    let k_mul3 = 5.0220313103171232;
    let k_offset3 = 2.1925739705298404;
    let k_offset4 = 0.25 * k_offset3;
    let k_mul0 = 0.74760422233706747;
    let k1 = 1.0;

    // Avoid division by zero.
    let v1 = (out_val * k_mul0).max(1e-3);
    let v2 = k1 / (v1 + k_offset2);
    let v3 = k1 / (v1 * v1 + k_offset3);
    let v4 = k1 / (v1 * v1 + k_offset4);

    k_base + (k_mul4 * v4) + (k_mul2 * v2) + (k_mul3 * v3)
}

/// Scalar version of masking_sqrt.
#[inline(always)]
pub(crate) fn scalar_masking_sqrt(v: f32) -> f32 {
    // Match the SIMD implementation
    let mul_v = K_MASKING_SQRT_MUL * 1e8;
    let offset_v = K_MASKING_SQRT_LOG_OFFSET;
    // Use mul_add and ensure max(0.0) before sqrt
    0.25 * (v.mul_add(mul_v.sqrt(), offset_v)).max(0.0).sqrt()
}

/// Computes one row of the intermediate `tmp` buffer for FuzzyErosion using SIMD.
/// Processes a 3x3 neighborhood for each pixel using `sort4`/`update_min4`.
/// Corresponds to the main loop inside C++ `FuzzyErosion`.
///
/// Processes `xsize` elements. Assumes input rows are padded (typically by 1 element left/right).
///
/// Args:
/// * `row_t`: Pre-erosion row y-1 (padded left/right, length >= xsize + 2)
/// * `row_m`: Pre-erosion row y (padded left/right, length >= xsize + 2)
/// * `row_b`: Pre-erosion row y+1 (padded left/right, length >= xsize + 2)
/// * `xsize`: Number of *output* elements to compute.
/// * `tmp_out`: Output slice for the row y of the temporary buffer, length >= `xsize`.
#[inline(always)]
pub(crate) fn compute_fuzzy_erosion_row(
    row_t: &[f32], // Assumed padded, length >= xsize + 2
    row_m: &[f32], // Assumed padded, length >= xsize + 2
    row_b: &[f32], // Assumed padded, length >= xsize + 2
    xsize: usize,
    tmp_out: &mut [f32],
) {
    // Check lengths assuming padding is included
    let required_len = xsize + 2;
    assert!(row_t.len() >= required_len);
    assert!(row_m.len() >= required_len);
    assert!(row_b.len() >= required_len);
    assert!(tmp_out.len() >= xsize);

    let mul0 = f32x8::splat(0.125);
    let mul1 = f32x8::splat(0.075);
    let mul2 = f32x8::splat(0.06);
    let mul3 = f32x8::splat(0.05);

    let vector_width = 8;
    let limit = xsize.saturating_sub(vector_width - 1);
    let mut x = 0;

    // x loops 0..xsize-1 (output index)
    // Access padded input slices relative to x. Slices include border padding.
    while x < limit {
        let current_x_idx = x + 1;
        let left_x_idx = x;
        let right_x_idx = x + 2;

        // Load 3x3 neighborhood, center vector is row_m[current_x_idx]
        let mut min0 = f32x8::new(arrayref::array_ref![row_m, current_x_idx, 8].clone());
        let mut min1 = f32x8::new(arrayref::array_ref![row_m, left_x_idx, 8].clone());
        let mut min2 = f32x8::new(arrayref::array_ref![row_m, right_x_idx, 8].clone());
        let mut min3 = f32x8::new(arrayref::array_ref![row_t, left_x_idx, 8].clone());

        sort4(&mut min0, &mut min1, &mut min2, &mut min3);

        update_min4(f32x8::new(arrayref::array_ref![row_t, current_x_idx, 8].clone()), &mut min0, &mut min1, &mut min2, &mut min3);
        update_min4(f32x8::new(arrayref::array_ref![row_t, right_x_idx, 8].clone()), &mut min0, &mut min1, &mut min2, &mut min3);
        update_min4(f32x8::new(arrayref::array_ref![row_b, left_x_idx, 8].clone()), &mut min0, &mut min1, &mut min2, &mut min3);
        update_min4(f32x8::new(arrayref::array_ref![row_b, current_x_idx, 8].clone()), &mut min0, &mut min1, &mut min2, &mut min3);
        update_min4(f32x8::new(arrayref::array_ref![row_b, right_x_idx, 8].clone()), &mut min0, &mut min1, &mut min2, &mut min3);

        // Calculate linear combination
        let v = (mul0 * min0) + (mul1 * min1) + (mul2 * min2) + (mul3 * min3);

        // Store result
        tmp_out[x..x + vector_width].copy_from_slice(&v.to_array());
        x += vector_width;
    }

    // Handle remainder
    while x < xsize {
        let current_x_idx = x + 1;
        let left_x_idx = x;
        let right_x_idx = x + 2;
        let mut neighborhood = [
            row_m[current_x_idx],
            row_m[left_x_idx],
            row_m[right_x_idx],
            row_t[left_x_idx],
            row_t[current_x_idx],
            row_t[right_x_idx],
            row_b[left_x_idx],
            row_b[current_x_idx],
            row_b[right_x_idx],
        ];
        // Sort the 9 neighbors
        neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Get the 4 smallest values
        let min0_s = neighborhood[0];
        let min1_s = neighborhood[1];
        let min2_s = neighborhood[2];
        let min3_s = neighborhood[3];

        // Calculate linear combination
        let v_s = 0.125 * min0_s + 0.075 * min1_s + 0.06 * min2_s + 0.05 * min3_s;

        tmp_out[x] = v_s;
        x += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts;
    use arrayref;

    // Helper to compare f32x8 vectors with tolerance
    fn assert_vec_approx_eq(a: f32x8, b: f32x8, tol: f32) {
        let diff = (a - b).abs();
        let max_diff = diff.reduce_max();
        assert!(max_diff <= tol, "Vectors differ by more than {}: a={:?}, b={:?}", tol, a.to_array(), b.to_array());
    }

    #[test]
    fn test_masking_sqrt() {
        // Test cases derived from C++ output (values might differ slightly due to precision)
        let inputs = f32x8::new([
            0.0, 1e-4, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0,
        ]);
        let expected = f32x8::new([
            1.3228756, 1.6306306, 9.625259, 30.17791, 67.42797, 95.34837, 301.4919, 953.3929
        ]);

        let result = masking_sqrt(inputs);
        // Increased tolerance slightly due to potential fma/sqrt differences
        assert_vec_approx_eq(result, expected, 2e-5);
    }

    #[test]
    fn test_eval_rational_polynomial() {
        // Example: (2x^2 + 3x + 1) / (x + 2)
        let p: [f32; 3] = [2.0, 3.0, 1.0]; // Coeffs for 2x^2 + 3x + 1
        let q: [f32; 2] = [1.0, 2.0];       // Coeffs for x + 2

        let x = f32x8::new([0.0, 1.0, -1.0, 2.0, -3.0, 0.5, -1.5, 10.0]);
        // Expected: (2*x*x + 3*x + 1) / (x + 2)
        let expected_scalar: [f32; 8] = x.to_array().map(|xi| {
            let num = (2.0 * xi * xi) + (3.0 * xi) + 1.0;
            let den = xi + 2.0;
            num / den
        });
        let expected = f32x8::new(expected_scalar);

        let result = eval_rational_polynomial!(x, &p, &q);
        assert_vec_approx_eq(result, expected, 1e-6);

        // Test degree 0 case: 5 / 2 = 2.5
        let p0: [f32; 1] = [5.0];
        let q0: [f32; 1] = [2.0];
        let result0 = eval_rational_polynomial!(x, &p0, &q0);
        assert_vec_approx_eq(result0, f32x8::splat(2.5), 1e-7);
    }

    #[test]
    fn test_fast_log2f() {
        // Test cases covering different ranges
        let inputs = f32x8::new([
            1.0, 2.0, 4.0, 0.5, 0.25, 10.0, 1000.0, 0.01
        ]);
        // Use standard log2 for comparison
        let expected_scalar = inputs.to_array().map(|xi| xi.log2());
        let expected = f32x8::new(expected_scalar);

        let result = fast_log2f(inputs);
        // The C++ comment mentions L1 error ~3.9E-6. Let's use a slightly larger tolerance.
        assert_vec_approx_eq(result, expected, 5e-6);
    }

    #[test]
    fn test_fast_pow2f_scalar() {
        // Compare against standard powf(2.0, x)
        let inputs = [0.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0, 10.5, -5.2];
        for &x in &inputs {
            let expected = 2.0f32.powf(x);
            let result = fast_pow2f_scalar(x);
            let tolerance = (expected * 5e-7).max(1e-12); // Relative tolerance, with floor
            assert!((result - expected).abs() <= tolerance, "pow2f_scalar({}) failed: expected={}, got={}", x, expected, result);
        }
    }

    #[test]
    fn test_fast_pow2f() {
        let inputs = f32x8::new([
            0.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0, 10.5
        ]);
        let expected_scalar = inputs.to_array().map(|xi| 2.0f32.powf(xi));
        let expected = f32x8::new(expected_scalar);

        let result = fast_pow2f(inputs);
        // C++ comment mentions max relative error ~3e-7.
        // Compare element-wise with relative tolerance
        let rel_diff = ((result - expected) / expected).abs();
        let max_rel_diff = rel_diff.reduce_max();
        assert!(max_rel_diff <= 4e-7, "Max relative error too high: {}", max_rel_diff);
        // Also check absolute tolerance for small numbers near zero (input 0.0 -> expected 1.0)
        assert_vec_approx_eq(result, expected, 1e-6);
    }

    #[test]
    fn test_ratio_of_derivatives() {
        let inputs = f32x8::new([
            0.0, K_GAMMA_EPSILON / K_INPUT_SCALING, 0.1, 1.0, 10.0, 100.0, 1000.0, -1.0 // Test clamping
        ]);

        // Test invert=false
        let expected_false_scalar = inputs.to_array().map(|xi| scalar_ratio_of_derivatives::<false>(xi));
        let expected_false = f32x8::new(expected_false_scalar);
        let result_false = ratio_of_derivatives_of_cubic_root_to_simple_gamma::<false>(inputs);
        assert_vec_approx_eq(result_false, expected_false, 1e-6);

        // Test invert=true
        let expected_true_scalar = inputs.to_array().map(|xi| scalar_ratio_of_derivatives::<true>(xi));
        let expected_true = f32x8::new(expected_true_scalar);
        let result_true = ratio_of_derivatives_of_cubic_root_to_simple_gamma::<true>(inputs);
        assert_vec_approx_eq(result_true, expected_true, 1e-6);
    }

    #[test]
    fn test_compute_mask() {
        let inputs = f32x8::new([
            0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, -1.0, 5.0
        ]);
        let expected_scalar = inputs.to_array().map(scalar_compute_mask);
        let expected = f32x8::new(expected_scalar);

        let result = compute_mask(inputs);
        assert_vec_approx_eq(result, expected, 1e-6);
    }

    #[test]
    fn test_sort4() {
        let mut v0 = f32x8::new([8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]);
        let mut v1 = f32x8::new([0.0, 9.0, 2.0, 8.0, 3.0, 7.0, 1.0, 6.0]);
        let mut v2 = f32x8::new([4.0, 5.0, 3.0, 6.0, 1.0, 9.0, 0.0, 7.0]);
        let mut v3 = f32x8::new([2.0, 7.0, 1.0, 9.0, 0.0, 8.0, 6.0, 5.0]);

        let mut all_vecs = [v0, v1, v2, v3];
        // Perform a reference sort lane-wise
        let mut expected_arrays = [[0f32; 8]; 4];
        for i in 0..8 {
            let mut lane_vals = [all_vecs[0].to_array()[i], all_vecs[1].to_array()[i], all_vecs[2].to_array()[i], all_vecs[3].to_array()[i]];
            lane_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            expected_arrays[0][i] = lane_vals[0];
            expected_arrays[1][i] = lane_vals[1];
            expected_arrays[2][i] = lane_vals[2];
            expected_arrays[3][i] = lane_vals[3];
        }

        sort4(&mut v0, &mut v1, &mut v2, &mut v3);

        assert_eq!(v0.to_array(), expected_arrays[0]);
        assert_eq!(v1.to_array(), expected_arrays[1]);
        assert_eq!(v2.to_array(), expected_arrays[2]);
        assert_eq!(v3.to_array(), expected_arrays[3]);
    }

    #[test]
    fn test_update_min4() {
        // Initial sorted values (ascending)
        let mut min0 = f32x8::splat(1.0);
        let mut min1 = f32x8::splat(3.0);
        let mut min2 = f32x8::splat(5.0);
        let mut min3 = f32x8::splat(7.0);

        // Case 1: New value smaller than all
        let v_small = f32x8::splat(0.5);
        update_min4(v_small, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0.to_array()[0], 0.5);
        assert_eq!(min1.to_array()[0], 1.0);
        assert_eq!(min2.to_array()[0], 3.0);
        assert_eq!(min3.to_array()[0], 5.0);

        // Case 2: New value between min1 and min2
        let mut min0 = f32x8::splat(1.0);
        let mut min1 = f32x8::splat(3.0);
        let mut min2 = f32x8::splat(5.0);
        let mut min3 = f32x8::splat(7.0);
        let v_mid = f32x8::splat(4.0);
        update_min4(v_mid, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0.to_array()[0], 1.0);
        assert_eq!(min1.to_array()[0], 3.0);
        assert_eq!(min2.to_array()[0], 4.0);
        assert_eq!(min3.to_array()[0], 5.0);

        // Case 3: New value larger than all
        let mut min0 = f32x8::splat(1.0);
        let mut min1 = f32x8::splat(3.0);
        let mut min2 = f32x8::splat(5.0);
        let mut min3 = f32x8::splat(7.0);
        let v_large = f32x8::splat(10.0);
        update_min4(v_large, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0.to_array()[0], 1.0);
        assert_eq!(min1.to_array()[0], 3.0);
        assert_eq!(min2.to_array()[0], 5.0);
        assert_eq!(min3.to_array()[0], 7.0);
    }

    #[test]
    fn test_fast_reciprocal_nr() {
        let inputs = f32x8::new([1.0, 2.0, 4.0, 0.5, 0.1, -1.0, -5.0, 100.0]);
        let expected_scalar = inputs.to_array().map(|xi| 1.0 / xi);
        let expected = f32x8::new(expected_scalar);

        let result = fast_reciprocal_nr(inputs);

        // Newton-Raphson improves precision, should be close to actual reciprocal
        assert_vec_approx_eq(result, expected, 1e-6);

        // Test edge case near zero (behavior might differ from ieee754 division)
        let near_zero = f32x8::splat(1e-10);
        let result_nz = fast_reciprocal_nr(near_zero);
        let expected_nz = f32x8::splat(1.0 / 1e-10);
        // Tolerance needs to be large here due to large magnitude
        let diff = (result_nz - expected_nz).abs() / expected_nz;
        assert!(diff.reduce_max() < 1e-6, "Near zero failed");
    }

    #[test]
    fn test_compute_hf_metric_8x8() {
        // Create dummy input data (9 rows, 17 columns for easy indexing x=0..8, x+1=1..9)
        let mut data = [[0.0f32; 17]; 9];
        let mut expected_sum = 0.0;
        for r in 0..9 {
            for c in 0..17 {
                // Simple gradient pattern
                data[r][c] = (r * 17 + c) as f32;
            }
        }

        // Manually calculate expected sum for the 8x8 block at (0,0)
        for dy in 0..8 {
            for dx in 0..8 {
                let p = data[dy][dx];
                let pd = data[dy+1][dx];
                expected_sum += (p - pd).abs();
                if dx < 7 {
                    let pr = data[dy][dx+1];
                    expected_sum += (p - pr).abs();
                }
            }
        }

        // Create slices for the function
        let row_slices: Vec<&[f32]> = data.iter().map(|row| &row[..]).collect();

        let result = compute_hf_metric_8x8(&row_slices, 0);

        assert!((result - expected_sum).abs() < 1e-5, "HF metric mismatch: expected={}, got={}", expected_sum, result);
    }

    #[test]
    fn test_compute_gamma_sum_8x8() {
        // Create dummy input data (8 rows, 8 columns)
        let mut data = [[0.0f32; 8]; 8];
        let mut expected_sum = 0.0;
        for r in 0..8 {
            for c in 0..8 {
                data[r][c] = (r * 8 + c) as f32 * K_INPUT_SCALING; // Scale input
            }
        }

        // Manually calculate expected sum
        for dy in 0..8 {
            for dx in 0..8 {
                let iny = data[dy][dx] + K_GAMMA_MODULATION_BIAS;
                expected_sum += scalar_ratio_of_derivatives::<true>(iny);
            }
        }

        let row_slices: Vec<&[f32]> = data.iter().map(|row| &row[..]).collect();
        let result = compute_gamma_sum_8x8(&row_slices, 0);

        assert!((result - expected_sum).abs() / expected_sum.abs() < 1e-6, "Gamma sum mismatch: expected={}, got={}", expected_sum, result);
    }

    // Scalar version for testing compute_diff_buffer_row
    fn scalar_compute_diff(
        row_t_padded: &[f32],
        row_m_padded: &[f32],
        row_b_padded: &[f32],
        x: usize, // Logical index 0..xsize-1
        prev_diff: Option<f32>
    ) -> f32 {
        let border = 1;
        // Indices relative to start of *padded* slice
        let current_x_idx = x + border;
        let left_x_idx = x + border - 1;
        let right_x_idx = x + border + 1;

        let in_m = row_m_padded[current_x_idx];
        let in_r = row_m_padded[right_x_idx];
        let in_l = row_m_padded[left_x_idx];
        let in_t = row_t_padded[current_x_idx];
        let in_b = row_b_padded[current_x_idx];

        let base = 0.25 * (in_r + in_l + in_t + in_b);
        let gammacv = scalar_ratio_of_derivatives::<false>(in_m + K_MATCH_GAMMA_OFFSET);
        let mut diff_s = gammacv * (in_m - base);
        diff_s *= diff_s;
        diff_s = diff_s.min(K_PRE_EROSION_LIMIT);
        let diff_clamped_s = diff_s;
        diff_s = scalar_masking_sqrt(diff_s);
        let diff_sqrt_s = diff_s;

        // Debug print for x=0
        if x == 0 {
            eprintln!("-- x=0 Scalar --");
            eprintln!("in_m: {:.8}", in_m);
            eprintln!("in_r: {:.8}", in_r);
            eprintln!("in_l: {:.8}", in_l);
            eprintln!("in_t: {:.8}", in_t);
            eprintln!("in_b: {:.8}", in_b);
            eprintln!("base: {:.8}", base);
            eprintln!("gammacv: {:.8}", gammacv);
            eprintln!("diff_presqrt: {:.8}", diff_clamped_s);
            eprintln!("diff_final: {:.8}", diff_sqrt_s);
        }

        if let Some(prev) = prev_diff {
            diff_s += prev;
        }
        diff_s
    }

    #[test]
    fn test_compute_diff_buffer_row() {
        let xsize = 20;
        let border = 1;
        let padded_len = xsize + 2 * border;

        // Create rows with padding and values designed to avoid exact cancellation at x=0
        let row_t_padded: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 0.11 + 0.01)).collect();
        let row_m_padded: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 0.21 + 0.02)).collect();
        let row_b_padded: Vec<f32> = (0..padded_len).map(|i| (i as f32 * 0.31 + 0.03)).collect();
        let prev_diff_out: Vec<f32> = (0..xsize).map(|i| (i as f32 * 0.04 + 0.04)).collect();
        let mut diff_out = vec![0.0f32; xsize];
        let mut expected_diff_out = vec![0.0f32; xsize];

        // Calculate expected scalar results using padded rows
        for x in 0..xsize {
            // Pass the full padded slices to the scalar helper
            expected_diff_out[x] = scalar_compute_diff(
                &row_t_padded,
                &row_m_padded,
                &row_b_padded,
                x,
                Some(prev_diff_out[x])
            );
        }

        compute_diff_buffer_row(
            &row_t_padded[border-1..border+xsize+1],
            &row_m_padded[border-1..border+xsize+1],
            &row_b_padded[border-1..border+xsize+1],
            xsize,
            &mut diff_out,
            Some(&prev_diff_out)
        );

        for x in 0..xsize {
            assert!((diff_out[x] - expected_diff_out[x]).abs() < 1e-5, "Diff buffer mismatch at x={}: expected={}, got={}", x, expected_diff_out[x], diff_out[x]);
        }

        let mut diff_out_no_prev = vec![0.0f32; xsize];
        let mut expected_diff_out_no_prev = vec![0.0f32; xsize];
        for x in 0..xsize {
            expected_diff_out_no_prev[x] = scalar_compute_diff(
                &row_t_padded,
                &row_m_padded,
                &row_b_padded,
                x,
                None
            );
        }
        compute_diff_buffer_row(
            &row_t_padded[border-1..border+xsize+1],
            &row_m_padded[border-1..border+xsize+1],
            &row_b_padded[border-1..border+xsize+1],
            xsize,
            &mut diff_out_no_prev,
            None
        );
        for x in 0..xsize {
            assert!((diff_out_no_prev[x] - expected_diff_out_no_prev[x]).abs() < 1e-5, "Diff buffer (no prev) mismatch at x={}: expected={}, got={}", x, expected_diff_out_no_prev[x], diff_out_no_prev[x]);
        }
    }

    #[test]
    fn test_compute_fuzzy_erosion_row() {
        let xsize = 18;
        let border = 1;
        let padded_len = xsize + 2 * border;

        // Create rows with padding and distinct values
        let row_t: Vec<f32> = (0..padded_len).map(|i| (i % 7) as f32).collect();
        let row_m: Vec<f32> = (0..padded_len).map(|i| (i % 5) as f32 + 10.0).collect();
        let row_b: Vec<f32> = (0..padded_len).map(|i| (i % 9) as f32 + 20.0).collect();
        let mut tmp_out = vec![0.0f32; xsize];
        let mut expected_tmp_out = vec![0.0f32; xsize];

        // Calculate expected scalar results
        for x in 0..xsize {
            let mut neighborhood = [
                row_m[x + border],
                row_m[x + border - 1],
                row_m[x + border + 1],
                row_t[x + border - 1],
                row_t[x + border],
                row_t[x + border + 1],
                row_b[x + border - 1],
                row_b[x + border],
                row_b[x + border + 1],
            ];
            neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap());
            expected_tmp_out[x] = 0.125 * neighborhood[0] + 0.075 * neighborhood[1] + 0.06 * neighborhood[2] + 0.05 * neighborhood[3];
        }

        // Run the SIMD function - Pass padded slices including necessary borders
        compute_fuzzy_erosion_row(
            &row_t, // Full padded row
            &row_m,
            &row_b,
            xsize, // Number of elements to compute
            &mut tmp_out
        );

        // Compare results
        for x in 0..xsize {
            assert!((tmp_out[x] - expected_tmp_out[x]).abs() < 1e-6, "Fuzzy erosion mismatch at x={}: expected={}, got={}", x, expected_tmp_out[x], tmp_out[x]);
        }
    }

    // Helper for scalar calculation for compute_mask test
    fn scalar_compute_mask(ov: f32) -> f32 {
        let k_base = -0.74174993;
        let k_mul4 = 3.2353257320940401;
        let k_mul2 = 12.906028311180409;
        let k_offset2 = 305.04035728311436;
        let k_mul3 = 5.0220313103171232;
        let k_offset3 = 2.1925739705298404;
        let k_offset4 = 0.25 * k_offset3;
        let k_mul0 = 0.74760422233706747;
        let k1 = 1.0;

        let v1 = (ov * k_mul0).max(1e-3);
        let v2 = k1 / (v1 + k_offset2);
        let v3 = k1 / (v1 * v1 + k_offset3);
        let v4 = k1 / (v1 * v1 + k_offset4);

        k_base + (k_mul4 * v4) + (k_mul2 * v2) + (k_mul3 * v3)
    }
} 