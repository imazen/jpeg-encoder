// Ported from lib/extras/xyb_transform.cc and lib/cms/opsin_params.h

#![allow(dead_code)] // Allow dead code initially

use arrayref::array_ref;
use wide::f32x8;

use super::{c_structs::div_ceil};
// --- Constants --- //

// From opsin_params.h
mod opsin_consts {
    pub const K_M00: f32 = 0.30;
    pub const K_M02: f32 = 0.078;
    pub const K_M01: f32 = 1.0 - K_M02 - K_M00; // 0.622
    pub const K_M10: f32 = 0.23;
    pub const K_M12: f32 = 0.078;
    pub const K_M11: f32 = 1.0 - K_M12 - K_M10; // 0.692
    pub const K_M20: f32 = 0.24342268924547819;
    pub const K_M21: f32 = 0.20476744424496821;
    pub const K_M22: f32 = 1.0 - K_M20 - K_M21; // 0.55180986651

    pub const K_OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] = [
        [K_M00, K_M01, K_M02],
        [K_M10, K_M11, K_M12],
        [K_M20, K_M21, K_M22],
    ];

    pub const K_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [
        0.0037930732552754493, // Bias0
        0.0037930732552754493, // Bias1
        0.0037930732552754493, // Bias2
    ];

    // These seem to be pre-calculated -cbrtf(bias)
    pub const K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT: [f32; 3] = [
        -0.15593413, // -cbrt(Bias0)  -- Note: C++ uses cbrtf, Rust f32 has cbrt
        -0.15593413, // -cbrt(Bias1)
        -0.15593413, // -cbrt(Bias2)
    ];

    pub const K_SCALED_XYB_OFFSET: [f32; 3] = [
        0.015386134, // Offset0
        0.0,         // Offset1
        0.27770459,  // Offset2
    ];

    pub const K_SCALED_XYB_SCALE: [f32; 3] = [
        22.995788804, // Scale0
        1.183000077,  // Scale1
        1.502141333,  // Scale2
    ];
}

// --- SIMD Helper Functions (Mirroring C++ Highway functions) --- //

/// Applies 3x3 Opsin Absorbance matrix and bias using SIMD.
/// Corresponds to C++ `OpsinAbsorbance`.
#[inline(always)]
fn opsin_absorbance_simd(
    r: f32x8, g: f32x8, b: f32x8,
    premul_absorb: &[f32x8; 9], // Matrix elements as vectors
    bias: f32x8, // Bias as a vector
) -> [f32x8; 3] {
    let mixed0 = premul_absorb[0].mul_add(r, premul_absorb[1].mul_add(g, premul_absorb[2].mul_add(b, bias)));
    let mixed1 = premul_absorb[3].mul_add(r, premul_absorb[4].mul_add(g, premul_absorb[5].mul_add(b, bias)));
    let mixed2 = premul_absorb[6].mul_add(r, premul_absorb[7].mul_add(g, premul_absorb[8].mul_add(b, bias)));
    [mixed0, mixed1, mixed2]
}

/// Stores XYB vectors to output pointers.
/// Corresponds to C++ `StoreXYB`.
#[inline(always)]
fn store_xyb_simd(
    r: f32x8, g: f32x8, b: f32x8,
    valx: &mut f32x8, valy: &mut f32x8, valz: &mut f32x8,
) {
    let half = f32x8::splat(0.5);
    let valx_vec = half * (r - g);
    let valy_vec = half * (r + g);
    let valz_vec = b;

    // Use unaligned stores by copying to slices
    *valx = valx_vec;
    *valy = valy_vec;
    *valz = valz_vec;
}

/// Converts one RGB vector to XYB using SIMD.
/// Corresponds to C++ `LinearRGBToXYB`.
#[inline(always)]
fn linear_rgb_to_xyb_simd(
    r: f32x8, g: f32x8, b: f32x8,
    premul_absorb: &[f32x8; 12], // Premultiplied matrix and bias_cbrt as vectors
    valx: &mut f32x8, valy: &mut f32x8, valz: &mut f32x8,
) {
    let premul_matrix: &[f32x8; 9] = array_ref!(premul_absorb, 0, 9);
    let bias_cbrt: &[f32x8; 3] = array_ref!(premul_absorb, 9, 3);

    // OpsinAbsorbance expects the *original* bias, not the negated cbrt version
    let bias_orig = f32x8::splat(opsin_consts::K_OPSIN_ABSORBANCE_BIAS[0]); // Assuming all biases are same

    let [mixed0, mixed1, mixed2] = opsin_absorbance_simd(
        r, g, b,
        premul_matrix,
        bias_orig,
    );

    // mixed* should be non-negative even for wide-gamut, so clamp to zero.
    let mixed0 = mixed0.max(f32x8::ZERO);
    let mixed1 = mixed1.max(f32x8::ZERO);
    let mixed2 = mixed2.max(f32x8::ZERO);

    // Cube root and add bias component (which is -cbrt(original_bias))
    let cr0 = mixed0.powf(1.0/3.0) + bias_cbrt[0];
    let cr1 = mixed1.powf(1.0/3.0) + bias_cbrt[1];
    let cr2 = mixed2.powf(1.0/3.0) + bias_cbrt[2];

    store_xyb_simd(cr0, cr1, cr2, valx, valy, valz);
}


// --- Row Processing Functions (Mirroring C++ row functions) --- //

/// Converts Linear RGB row to XYB row (in-place) using SIMD acceleration.
/// Corresponds to C++ `LinearRGBRowToXYB`.
pub fn linear_rgb_row_to_xyb(
    row0: &mut [f32], row1: &mut [f32], row2: &mut [f32],
    premul_absorb: &[f32x8; 12], // Expects a flat slice of 12 * Lanes<f32x8>
    xsize: usize,
    intensity_target: f32, // Added to be available in scalar fallback
) {
    // Ensure premul_absorb is correctly sized for loading f32x8 vectors
    let vector_len = 8;
   
    let vectorized_length = div_ceil(xsize, vector_len);
    let row0_vec: &mut [f32x8] = bytemuck::cast_slice_mut(&mut row0[..vectorized_length * vector_len]);
    let row1_vec: &mut [f32x8] = bytemuck::cast_slice_mut(&mut row1[..vectorized_length * vector_len]);
    let row2_vec: &mut [f32x8] = bytemuck::cast_slice_mut(&mut row2[..vectorized_length * vector_len]);

    for i in 0..vectorized_length {
        linear_rgb_to_xyb_simd(
            row0_vec[i], row1_vec[i], row2_vec[i],
            premul_absorb,
            &mut row0_vec[i], &mut row1_vec[i], &mut row2_vec[i],
        );
    }

    let mut x = vectorized_length * vector_len;

    // Handle remainder using scalar
    while x < xsize {
        let r = row0[x];
        let g = row1[x];
        let b = row2[x];

        // Replicate scalar opsin absorbance
        let mut mixed = [0.0f32; 3];
        let bias_orig = opsin_consts::K_OPSIN_ABSORBANCE_BIAS;
        let premul_matrix = opsin_consts::K_OPSIN_ABSORBANCE_MATRIX;
        let mul = intensity_target / 255.0f32; // Access intensity_target

        mixed[0] = (premul_matrix[0][0] * mul) * r + (premul_matrix[0][1] * mul) * g + (premul_matrix[0][2] * mul) * b + bias_orig[0];
        mixed[1] = (premul_matrix[1][0] * mul) * r + (premul_matrix[1][1] * mul) * g + (premul_matrix[1][2] * mul) * b + bias_orig[1];
        mixed[2] = (premul_matrix[2][0] * mul) * r + (premul_matrix[2][1] * mul) * g + (premul_matrix[2][2] * mul) * b + bias_orig[2];


        // Clamp to zero before cube root
        let m0 = mixed[0].max(0.0);
        let m1 = mixed[1].max(0.0);
        let m2 = mixed[2].max(0.0);

        // Cube root and add bias component (Rust f32 has cbrt)
        let cr0 = m0.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[0];
        let cr1 = m1.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[1];
        let cr2 = m2.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[2];

        // Final XYB conversion (scalar)
        row0[x] = 0.5 * (cr0 - cr1); // X
        row1[x] = 0.5 * (cr0 + cr1); // Y
        row2[x] = cr2;               // B

        x += 1;
    }
}

/// Computes premultiplied Opsin Absorbance matrix and combined bias/cbrt constants.
/// Stores results in a flat f32 slice. Corresponds to C++ `ComputePremulAbsorb`.
///
/// The output buffer `premul_absorb` must have size `12 * f32x8::LANES`.
pub fn compute_premul_absorb(intensity_target: f32, premul_absorb: &mut [f32x8; 12]) {
    let vector_len = 8;
    assert!(premul_absorb.len() >= 12 * vector_len, "premul_absorb buffer is too small");

    let mul = intensity_target / 255.0f32;

    // Compute and store matrix vectors using as_array().copy_to_slice()
    for j in 0..3 {
        for i in 0..3 {
            let absorb_vec = f32x8::splat(opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[j][i] * mul);
            premul_absorb[j * 3 + i] = absorb_vec;
        }
    }

    // Compute and store bias_cbrt vectors using as_array().copy_to_slice()
    for i in 0..3 {
        let neg_bias_cbrt_vec = f32x8::splat(opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[i]);
        premul_absorb[9 + i] = neg_bias_cbrt_vec;
    }
}
pub fn create_premul_absorb(intensity_target: f32) -> [f32x8; 12] {
    let mut premul_absorb = [f32x8::ZERO; 12];
    compute_premul_absorb(intensity_target, &mut premul_absorb);
    premul_absorb
}

/// Scales XYB values in-place using affine transform (scalar implementation).
/// Corresponds to C++ `ScaleXYBRow` (scalar version in HWY_ONCE).
///
/// This function is implemented using scalar operations for simplicity, as
/// the C++ version in HWY_ONCE was also scalar. Modern Rust compilers
/// should be able to auto-vectorize this loop effectively.
pub fn scale_xyb_row(
    row0: &mut [f32], // X
    row1: &mut [f32], // Y
    row2: &mut [f32], // B
    xsize: usize,
) {
    use opsin_consts::{K_SCALED_XYB_OFFSET, K_SCALED_XYB_SCALE};
    assert!(row0.len() >= xsize);
    assert!(row1.len() >= xsize);
    assert!(row2.len() >= xsize);

    for x in 0..xsize {
        let original_y = row1[x]; // Need original Y for B calculation
        row2[x] = (row2[x] - original_y + K_SCALED_XYB_OFFSET[2]) * K_SCALED_XYB_SCALE[2]; // B
        row0[x] = (row0[x] + K_SCALED_XYB_OFFSET[0]) * K_SCALED_XYB_SCALE[0];             // X
        row1[x] = (original_y + K_SCALED_XYB_OFFSET[1]) * K_SCALED_XYB_SCALE[1];             // Y
    }
}

// --- Tests --- //
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TOLERANCE: f32 = 1e-5; // Use a reasonable tolerance for float comparisons

    fn assert_approx_eq_slice(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Slice lengths differ");
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            // Check for NaNs before comparison
            if va.is_nan() || vb.is_nan() {
                 assert!(va.is_nan() && vb.is_nan(), "Mismatch at index {}: one is NaN, other is not ({} vs {})", i, va, vb);
                 continue;
            }
            assert!((va - vb).abs() <= tolerance, "Mismatch at index {}: {} vs {}", i, va, vb);
        }
    }

    // Helper to compute Opsin Absorbance scalar
    fn scalar_opsin_absorbance(
        r: f32, g: f32, b: f32,
        premul_matrix: &[f32; 9],
        bias: &[f32; 3],
    ) -> [f32; 3] {
        let mixed0 = premul_matrix[0] * r + premul_matrix[1] * g + premul_matrix[2] * b + bias[0];
        let mixed1 = premul_matrix[3] * r + premul_matrix[4] * g + premul_matrix[5] * b + bias[1];
        let mixed2 = premul_matrix[6] * r + premul_matrix[7] * g + premul_matrix[8] * b + bias[2];
        [mixed0, mixed1, mixed2]
    }


    #[test]
    fn test_compute_premul_absorb() {
        let intensity_target = 255.0;
        let premul_absorb = create_premul_absorb(intensity_target);

        let mul = intensity_target / 255.0f32;

        // Verify matrix part
        for j in 0..3 {
            for i in 0..3 {
                let expected_scalar = opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[j][i] * mul;
                let loaded_vector = premul_absorb[j * 3 + i];
                assert_approx_eq_slice(&loaded_vector.to_array(), &[expected_scalar; 8], 1e-6);
            }
        }

        // Verify bias_cbrt part
        for i in 0..3 {
            let expected_scalar = opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[i];
            let loaded_vector = premul_absorb[9 + i];
            assert_approx_eq_slice(&loaded_vector.to_array(), &[expected_scalar; 8], 1e-6);
        }
    }

    #[test]
    fn test_linear_rgb_row_to_xyb() {
        let intensity_target = 255.0;
        let xsize = 20; // Test with a size that is not a multiple of vector_len
        let premul_absorb_buffer = create_premul_absorb(intensity_target);

        let mut r_row: Vec<f32> = (0..xsize).map(|i| (i as f32 / xsize as f32).sin().abs()).collect();
        let mut g_row: Vec<f32> = (0..xsize).map(|i| (i as f32 / xsize as f32).cos().abs()).collect();
        let mut b_row: Vec<f32> = (0..xsize).map(|i| (i as f32 / xsize as f32) * 0.5 + 0.25).collect();

        let r_row_orig = r_row.clone();
        let g_row_orig = g_row.clone();
        let b_row_orig = b_row.clone();

        linear_rgb_row_to_xyb(
            &mut r_row, &mut g_row, &mut b_row,
            &premul_absorb_buffer,
            xsize,
            intensity_target,
        );

        // Manually compute expected scalar results
        let mut expected_x = vec![0.0f32; xsize];
        let mut expected_y = vec![0.0f32; xsize];
        let mut expected_b = vec![0.0f32; xsize];

        let mul = intensity_target / 255.0f32;
        let premul_matrix_scalar: [[f32; 3]; 3] = [
            [opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[0][0] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[0][1] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[0][2] * mul],
            [opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[1][0] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[1][1] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[1][2] * mul],
            [opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[2][0] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[2][1] * mul, opsin_consts::K_OPSIN_ABSORBANCE_MATRIX[2][2] * mul],
        ];
        let premul_matrix_flat: [f32; 9] = [
            premul_matrix_scalar[0][0], premul_matrix_scalar[0][1], premul_matrix_scalar[0][2],
            premul_matrix_scalar[1][0], premul_matrix_scalar[1][1], premul_matrix_scalar[1][2],
            premul_matrix_scalar[2][0], premul_matrix_scalar[2][1], premul_matrix_scalar[2][2],
        ];


        for x in 0..xsize {
            let r = r_row_orig[x];
            let g = g_row_orig[x];
            let b = b_row_orig[x];

            let mixed = scalar_opsin_absorbance(
                r, g, b,
                &premul_matrix_flat,
                &opsin_consts::K_OPSIN_ABSORBANCE_BIAS,
            );

            // Clamp to zero before cube root
            let m0 = mixed[0].max(0.0);
            let m1 = mixed[1].max(0.0);
            let m2 = mixed[2].max(0.0);

            // Cube root and add bias component
            let cr0 = m0.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[0];
            let cr1 = m1.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[1];
            let cr2 = m2.cbrt() + opsin_consts::K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[2];

            // Final XYB conversion (scalar)
            expected_x[x] = 0.5 * (cr0 - cr1);
            expected_y[x] = 0.5 * (cr0 + cr1);
            expected_b[x] = cr2;
        }

        assert_approx_eq_slice(&r_row, &expected_x, TOLERANCE);
        assert_approx_eq_slice(&g_row, &expected_y, TOLERANCE);
        assert_approx_eq_slice(&b_row, &expected_b, TOLERANCE);
    }

     #[test]
    fn test_scale_xyb_row() {
        let xsize = 20;
        let mut x_row: Vec<f32> = (0..xsize).map(|i| (i as f32 * 0.1)).collect();
        let mut y_row: Vec<f32> = (0..xsize).map(|i| (i as f32 * 0.2)).collect();
        let mut b_row: Vec<f32> = (0..xsize).map(|i| (i as f32 * 0.3)).collect();

        let x_row_orig = x_row.clone();
        let y_row_orig = y_row.clone();
        let b_row_orig = b_row.clone();

        scale_xyb_row(&mut x_row, &mut y_row, &mut b_row, xsize);

        let mut expected_x = vec![0.0f32; xsize];
        let mut expected_y = vec![0.0f32; xsize];
        let mut expected_b = vec![0.0f32; xsize];

         use opsin_consts::{K_SCALED_XYB_OFFSET, K_SCALED_XYB_SCALE};

        for x in 0..xsize {
            let original_y = y_row_orig[x];
            let original_x = x_row_orig[x];
            let original_b = b_row_orig[x];

            expected_b[x] = (original_b - original_y + K_SCALED_XYB_OFFSET[2]) * K_SCALED_XYB_SCALE[2];
            expected_x[x] = (original_x + K_SCALED_XYB_OFFSET[0]) * K_SCALED_XYB_SCALE[0];
            expected_y[x] = (original_y + K_SCALED_XYB_OFFSET[1]) * K_SCALED_XYB_SCALE[1];
        }

        assert_approx_eq_slice(&x_row, &expected_x, TOLERANCE);
        assert_approx_eq_slice(&y_row, &expected_y, TOLERANCE);
        assert_approx_eq_slice(&b_row, &expected_b, TOLERANCE);
    }
} 