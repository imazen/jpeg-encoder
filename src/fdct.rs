/*
 * Ported from mozjpeg to rust
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, 2020, D. R. Commander.
 *
 * Conditions of distribution and use:
 * In plain English:
 *
 * 1. We don't promise that this software works.  (But if you find any bugs,
 *    please let us know!)
 * 2. You can use this software for whatever you want.  You don't have to pay us.
 * 3. You may not pretend that you wrote this software.  If you use it in a
 *    program, you must acknowledge somewhere in your documentation that
 *    you've used the IJG code.
 *
 * In legalese:
 *
 * The authors make NO WARRANTY or representation, either express or implied,
 * with respect to this software, its quality, accuracy, merchantability, or
 * fitness for a particular purpose.  This software is provided "AS IS", and you,
 * its user, assume the entire risk as to its quality and accuracy.
 *
 * This software is copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
 * All Rights Reserved except as specified below.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * software (or portions thereof) for any purpose, without fee, subject to these
 * conditions:
 * (1) If any part of the source code for this software is distributed, then this
 * README file must be included, with this copyright and no-warranty notice
 * unaltered; and any additions, deletions, or changes to the original files
 * must be clearly indicated in accompanying documentation.
 * (2) If only executable code is distributed, then the accompanying
 * documentation must state that "this software is based in part on the work of
 * the Independent JPEG Group".
 * (3) Permission for use of this software is granted only if the user accepts
 * full responsibility for any undesirable consequences; the authors accept
 * NO LIABILITY for damages of any kind.
 *
 * These conditions apply to any software derived from or based on the IJG code,
 * not just to the unmodified library.  If you use our work, you ought to
 * acknowledge us.
 *
 * Permission is NOT granted for the use of any IJG author's name or company name
 * in advertising or publicity relating to this software or products derived from
 * it.  This software may be referred to only as "the Independent JPEG Group's
 * software".
 *
 * We specifically permit and encourage the use of this software as the basis of
 * commercial products, provided that all warranty or liability claims are
 * assumed by the product vendor.
 *
 * This file contains a slower but more accurate integer implementation of the
 * forward DCT (Discrete Cosine Transform).
 *
 * A 2-D DCT can be done by 1-D DCT on each row followed by 1-D DCT
 * on each column.  Direct algorithms are also available, but they are
 * much more complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 */

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

const DCT_SIZE: usize = 8;

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    // right shift with rounding
    (x + (1 << (n - 1))) >> n
}

#[inline(always)]
fn into_el(v: i32) -> i16 {
    v as i16
}

#[allow(clippy::erasing_op)]
#[allow(clippy::identity_op)]
pub fn fdct(data: &mut [i16; 64]) {
    /* Pass 1: process rows. */
    /* Note results are scaled up by sqrt(8) compared to a true DCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */

    let mut data2 = [0i32; 64];

    for y in 0..8 {
        let offset = y * 8;

        let tmp0 = i32::from(data[offset + 0]) + i32::from(data[offset + 7]);
        let tmp7 = i32::from(data[offset + 0]) - i32::from(data[offset + 7]);
        let tmp1 = i32::from(data[offset + 1]) + i32::from(data[offset + 6]);
        let tmp6 = i32::from(data[offset + 1]) - i32::from(data[offset + 6]);
        let tmp2 = i32::from(data[offset + 2]) + i32::from(data[offset + 5]);
        let tmp5 = i32::from(data[offset + 2]) - i32::from(data[offset + 5]);
        let tmp3 = i32::from(data[offset + 3]) + i32::from(data[offset + 4]);
        let tmp4 = i32::from(data[offset + 3]) - i32::from(data[offset + 4]);

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data2[offset + 0] = (tmp10 + tmp11) << PASS1_BITS;
        data2[offset + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data2[offset + 2] = descale(
            z1 + (tmp13 * FIX_0_765366865),
            CONST_BITS - PASS1_BITS,
        );
        data2[offset + 6] = descale(
            z1 + (tmp12 * -FIX_1_847759065),
            CONST_BITS - PASS1_BITS,
        );

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; /* sqrt(2) * c3 */

        let tmp4 = tmp4 * FIX_0_298631336; /* sqrt(2) * (-c1+c3+c5-c7) */
        let tmp5 = tmp5 * FIX_2_053119869; /* sqrt(2) * ( c1+c3-c5+c7) */
        let tmp6 = tmp6 * FIX_3_072711026; /* sqrt(2) * ( c1+c3+c5-c7) */
        let tmp7 = tmp7 * FIX_1_501321110; /* sqrt(2) * ( c1+c3-c5-c7) */
        let z1 = z1 * -FIX_0_899976223; /* sqrt(2) * ( c7-c3) */
        let z2 = z2 * -FIX_2_562915447; /* sqrt(2) * (-c1-c3) */
        let z3 = z3 * -FIX_1_961570560; /* sqrt(2) * (-c3-c5) */
        let z4 = z4 * -FIX_0_390180644; /* sqrt(2) * ( c5-c3) */

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        data2[offset + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data2[offset + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data2[offset + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data2[offset + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    /* Pass 2: process columns.
     * We remove the PASS1_BITS scaling, but leave the results scaled up
     * by an overall factor of 8.
     */

    for x in 0..8 {
        let tmp0 = data2[DCT_SIZE * 0 + x] + data2[DCT_SIZE * 7 + x];
        let tmp7 = data2[DCT_SIZE * 0 + x] - data2[DCT_SIZE * 7 + x];
        let tmp1 = data2[DCT_SIZE * 1 + x] + data2[DCT_SIZE * 6 + x];
        let tmp6 = data2[DCT_SIZE * 1 + x] - data2[DCT_SIZE * 6 + x];
        let tmp2 = data2[DCT_SIZE * 2 + x] + data2[DCT_SIZE * 5 + x];
        let tmp5 = data2[DCT_SIZE * 2 + x] - data2[DCT_SIZE * 5 + x];
        let tmp3 = data2[DCT_SIZE * 3 + x] + data2[DCT_SIZE * 4 + x];
        let tmp4 = data2[DCT_SIZE * 3 + x] - data2[DCT_SIZE * 4 + x];

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[DCT_SIZE * 0 + x] = into_el(descale(tmp10 + tmp11, PASS1_BITS));
        data[DCT_SIZE * 4 + x] = into_el(descale(tmp10 - tmp11, PASS1_BITS));

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[DCT_SIZE * 2 + x] = into_el(descale(
            z1 + tmp13 * FIX_0_765366865,
            CONST_BITS + PASS1_BITS,
        ));
        data[DCT_SIZE * 6 + x] = into_el(descale(
            z1 + tmp12 * -FIX_1_847759065,
            CONST_BITS + PASS1_BITS,
        ));

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; /* sqrt(2) * c3 */

        let tmp4 = tmp4 * FIX_0_298631336; /* sqrt(2) * (-c1+c3+c5-c7) */
        let tmp5 = tmp5 * FIX_2_053119869; /* sqrt(2) * ( c1+c3-c5+c7) */
        let tmp6 = tmp6 * FIX_3_072711026; /* sqrt(2) * ( c1+c3+c5-c7) */
        let tmp7 = tmp7 * FIX_1_501321110; /* sqrt(2) * ( c1+c3-c5-c7) */
        let z1 = z1 * -FIX_0_899976223; /* sqrt(2) * ( c7-c3) */
        let z2 = z2 * -FIX_2_562915447; /* sqrt(2) * (-c1-c3) */
        let z3 = z3 * -FIX_1_961570560; /* sqrt(2) * (-c3-c5) */
        let z4 = z4 * -FIX_0_390180644; /* sqrt(2) * ( c5-c3) */

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        data[DCT_SIZE * 7 + x] = into_el(descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 5 + x] = into_el(descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 3 + x] = into_el(descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 1 + x] = into_el(descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS));
    }
}

#[cfg(test)]
mod tests {

    // Inputs and outputs are taken from libjpegs jpeg_fdct_islow for a typical image

    use super::fdct;

    const INPUT1: [i16; 64] = [
        -70, -71, -70, -68, -67, -67, -67, -67, -72, -73, -72, -70, -69, -69, -68, -69, -75, -76,
        -74, -73, -73, -72, -71, -70, -77, -78, -77, -75, -76, -75, -73, -71, -78, -77, -77, -76,
        -79, -77, -76, -75, -78, -78, -77, -77, -77, -77, -78, -77, -79, -79, -78, -78, -78, -78,
        -79, -78, -80, -79, -78, -78, -81, -80, -78, -76,
    ];

    const OUTPUT1: [i16; 64] = [
        -4786, -66, 2, -18, 12, 12, 5, -7, 223, -37, -8, 21, 8, 5, -4, 6, 60, 6, -10, 5, 0, -2, -1,
        5, 21, 21, -15, 12, -2, -7, 1, 0, -2, -5, 16, -15, 0, 5, -4, -8, 0, -7, -4, 6, 7, -4, 5, 4,
        3, 0, 1, -5, 0, -1, 4, 1, -5, 7, 0, -3, -6, 1, 1, -4,
    ];

    const INPUT2: [i16; 64] = [
        21, 28, 11, 24, -45, -37, -55, -103, 38, -8, 31, 17, -19, 49, 15, -76, 22, -48, -36, -31,
        -23, 35, -23, -72, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20,
        -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30,
        -45, -42, -44, -15, -20, -44,
    ];

    const OUTPUT2: [i16; 64] = [
        -1420, 717, 187, 910, -244, 579, 222, -191, 461, 487, -497, -29, -220, 179, 63, -95, 213,
        414, -235, -187, -108, 74, -73, -70, -63, 311, 13, -290, 17, -38, -180, -47, -254, 201,
        116, -247, 102, -109, -185, -36, -310, 107, 73, -91, 126, -121, -99, -37, -253, 43, -15,
        53, 101, -91, -3, -37, -136, 12, -44, 81, 53, -45, 31, -24,
    ];

    #[test]
    pub fn test_fdct_libjpeg() {
        let mut i1 = INPUT1.clone();
        fdct(&mut i1);
        assert_eq!(i1, OUTPUT1);

        let mut i2 = INPUT2.clone();
        fdct(&mut i2);
        assert_eq!(i2, OUTPUT2);
    }

    #[test]
    #[ignore] // Requires reference values from C++ jpegli
    fn test_float_dct_basic() {
        // Sample 8x8 block data (level-shifted: pixel_u8 as f32 - 128.0)
        // Example: A simple gradient or constant block
        let input_pixels: [f32; 64] = [
            -128.0, -112.0, -96.0, -80.0, -64.0, -48.0, -32.0, -16.0,
            -112.0, -96.0, -80.0, -64.0, -48.0, -32.0, -16.0,   0.0,
            -96.0, -80.0, -64.0, -48.0, -32.0, -16.0,   0.0,  16.0,
            -80.0, -64.0, -48.0, -32.0, -16.0,   0.0,  16.0,  32.0,
            -64.0, -48.0, -32.0, -16.0,   0.0,  16.0,  32.0,  48.0,
            -48.0, -32.0, -16.0,   0.0,  16.0,  32.0,  48.0,  64.0,
            -32.0, -16.0,   0.0,  16.0,  32.0,  48.0,  64.0,  80.0,
            -16.0,   0.0,  16.0,  32.0,  48.0,  64.0,  80.0,  96.0,
        ]; // Replace with actual test data
        let mut coeffs = [0.0f32; 64];
        let mut scratch = [0.0f32; 64];

        super::forward_dct_float(&input_pixels, &mut coeffs, &mut scratch);

        // Reference coefficients obtained from running jpegli C++
        // on the same input_pixels data.
        let expected_coeffs: [f32; 64] = [
            // Fill with reference values...
            0.0; 64 // Placeholder
        ];

        // Compare coeffs with expected_coeffs (allow for small floating point differences)
        let epsilon = 1e-4;
        for i in 0..64 {
            assert!((coeffs[i] - expected_coeffs[i]).abs() < epsilon,
                "Mismatch at index {}: expected {}, got {}",
                 i, expected_coeffs[i], coeffs[i]);
        }
    }

}

// Constants and functions for Jpegli Float DCT implementation
// Based on lib/jpegli/dct-inl.h

// Constants for DCT implementation. Generated by the following snippet:
// for i in range(N // 2):
//    print(1.0 / (2 * math.cos((i + 0.5) * math.pi / N)), end=", ")
mod float_dct_constants {
    // WcMultipliers<8>
    pub const WC_MULTIPLIERS_8: [f32; 4] = [
        0.5097955791041592,
        0.6013448869350453,
        0.8999762231364156,
        2.5629154477415055,
    ];
    pub const SQRT2: f32 = 1.41421356237f32;
    pub const INV_8: f32 = 1.0 / 8.0;
}

#[inline(always)]
fn transpose_8x8_block(input: &[f32; 64], output: &mut [f32; 64]) {
    // Basic scalar transpose
    for i in 0..8 {
        for j in 0..8 {
            output[j * 8 + i] = input[i * 8 + j];
        }
    }
}

#[inline(always)]
fn add_reverse<const N_HALF: usize>(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32]) {
    // N_HALF corresponds to N / 2 in the C++ code
    // The C++ code operates on 8-element vectors per loop iteration.
    // This scalar version processes element by element within the rows.
    for i in 0..N_HALF {
        for k in 0..8 {
            // Access corresponding rows and reverse access for a_in2
            a_out[i * 8 + k] = a_in1[i * 8 + k] + a_in2[(N_HALF * 2 - 1 - i) * 8 + k];
        }
    }
}


#[inline(always)]
fn sub_reverse<const N_HALF: usize>(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32]) {
    // N_HALF corresponds to N / 2 in the C++ code
    for i in 0..N_HALF {
        for k in 0..8 {
             // Access corresponding rows and reverse access for a_in2
            a_out[i * 8 + k] = a_in1[i * 8 + k] - a_in2[(N_HALF * 2 - 1 - i) * 8 + k];
        }
    }
}

#[inline(always)]
fn multiply<const N_HALF: usize>(
    coeff_second_half: &mut [f32],
    multipliers: &[f32]
) {
    // Check if multipliers array has enough elements
    assert!(multipliers.len() >= N_HALF);
    for i in 0..N_HALF {
         for k in 0..8 {
            // Indexing relative to the start of the second half slice
            coeff_second_half[i * 8 + k] *= multipliers[i];
        }
    }
}

#[inline(always)]
fn b<const N: usize>(coeff: &mut [f32]) {
    // N here corresponds to N/2 in the C++ B<N/2> call site context
    // (e.g. called with N=4 when processing N=8 DCT)
    let sqrt2 = float_dct_constants::SQRT2;
    for k in 0..8 {
        coeff[0 * 8 + k] = coeff[0 * 8 + k] * sqrt2 + coeff[1 * 8 + k];
    }
    for i in 1..(N - 1) {
        for k in 0..8 {
            coeff[i * 8 + k] += coeff[(i + 1) * 8 + k];
        }
    }
}


#[inline(always)]
fn inverse_even_odd<const N: usize>(a_in: &[f32], a_out: &mut [f32]) {
    for i in 0..(N / 2) {
        for k in 0..8 {
             a_out[2 * i * 8 + k] = a_in[i * 8 + k];
        }
    }
    for i in (N / 2)..N {
         for k in 0..8 {
            a_out[(2 * (i - N / 2) + 1) * 8 + k] = a_in[i * 8 + k];
        }
    }
}

// Recursive DCT implementation structure
trait DCT1DImplTrait {
    fn compute(mem: &mut [f32]);
}

struct DCT1DImpl<const N: usize>;

impl DCT1DImplTrait for DCT1DImpl<1> {
    #[inline(always)]
    fn compute(_mem: &mut [f32]) {
        // Base case: N=1 DCT is identity
    }
}

impl DCT1DImplTrait for DCT1DImpl<2> {
    #[inline(always)]
    fn compute(mem: &mut [f32]) {
        // Operates on two rows (16 elements total)
        for k in 0..8 {
            let in1 = mem[0 * 8 + k];
            let in2 = mem[1 * 8 + k];
            mem[0 * 8 + k] = in1 + in2;
            mem[1 * 8 + k] = in1 - in2;
        }
    }
}

// Generic implementation for N > 2 (power of 2)
// Needs explicit specialization or handling for N=4, N=8 etc.
// Let's implement N=8 directly based on the recursive calls.
impl DCT1DImplTrait for DCT1DImpl<8> {
    #[inline(always)]
    fn compute(mem: &mut [f32]) { // mem is [f32; 64]
        let mut tmp = [0.0f32; 64];
        {
            let (tmp_first_half, tmp_second_half) = tmp.split_at_mut(32);

            // First level recursion (N=8 -> N=4)
            let (mem_first_half, mem_second_half) = mem.split_at_mut(32);

            add_reverse::<4>(mem_first_half, mem_second_half, tmp_first_half);
            DCT1DImpl::<4>::compute(tmp_first_half); // Operates on tmp_first_half

            sub_reverse::<4>(mem_first_half, mem_second_half, tmp_second_half);

            // Call multiply only on the second half, passing N/2 and multipliers
            multiply::<4>(tmp_second_half, &float_dct_constants::WC_MULTIPLIERS_8);

            // Second recursion operates on tmp_second_half
            DCT1DImpl::<4>::compute(tmp_second_half);
            b::<4>(tmp_second_half);
        }
        // tmp is fully available again here
        inverse_even_odd::<8>(&tmp, mem);
    }
}

// We need DCT1DImpl<4> for the recursion
impl DCT1DImplTrait for DCT1DImpl<4> {
     #[inline(always)]
     fn compute(mem: &mut [f32]) { // mem is [f32; 32]
        let mut tmp = [0.0f32; 32];

        // Second level recursion (N=4 -> N=2)
        let (mem_first_half, mem_second_half) = mem.split_at_mut(16);
        let (tmp_first_half, tmp_second_half) = tmp.split_at_mut(16);

        add_reverse::<2>(mem_first_half, mem_second_half, tmp_first_half);
        DCT1DImpl::<2>::compute(tmp_first_half); // Base case N=2

        sub_reverse::<2>(mem_first_half, mem_second_half, tmp_second_half);
        DCT1DImpl::<2>::compute(tmp_second_half); // Base case N=2
        b::<2>(tmp_second_half); // N=2 means N/2 from C++

        inverse_even_odd::<4>(&tmp, mem);
    }
}


#[inline(always)]
fn dct_1d(pixels: &[f32], output: &mut [f32]) {
    // Assumes pixels is 64 element row/column block, output is 64 element block
    // Performs 8 parallel 1D DCTs of size 8
    let mut tmp = [0.0f32; 64];

    // Load - In C++ this loads columns into tmp using LoadFromBlock
    // In Rust, assuming input `pixels` is already arranged correctly for the first 1D pass (row-wise)
    // For the second pass (column-wise), the transposed data is used.
    // Let's simplify and assume `pixels` is the 64-element block to process.
     tmp.copy_from_slice(pixels);

    // Process 8 columns/rows in place within tmp
    DCT1DImpl::<8>::compute(&mut tmp);

    // Scale and Store - In C++ uses StoreToBlockAndScale
    let mul = float_dct_constants::INV_8;
    for i in 0..64 {
         output[i] = tmp[i] * mul;
    }
}


/// Performs a forward DCT on an 8x8 block using floating-point arithmetic
/// based on the jpegli implementation.
///
/// * `pixels`: Input 8x8 block (64 elements, row-major). Should be level-shifted (centered around 0.0).
/// * `coefficients`: Output 8x8 block (64 elements, row-major) for DCT coefficients.
/// * `scratch_space`: A temporary buffer of 64 f32 elements.
pub fn forward_dct_float(pixels: &[f32; 64], coefficients: &mut [f32; 64], scratch_space: &mut [f32; 64]) {
    // 1. DCT1D on rows
    dct_1d(pixels, scratch_space);

    // 2. Transpose
    transpose_8x8_block(scratch_space, coefficients);

    // 3. DCT1D on columns (using the transposed block as input)
    dct_1d(coefficients, scratch_space);

    // 4. Transpose back
    transpose_8x8_block(scratch_space, coefficients);
}
