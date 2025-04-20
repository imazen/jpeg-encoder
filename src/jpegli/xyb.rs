// Ported from lib/extras/xyb_transform.cc and lib/cms/opsin_params.h

use alloc::vec::Vec;
use arrayref::array_ref;

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

    pub const K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT: [f32; 3] = [
        -0.15593413, // -cbrt(Bias0)
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

// --- XYB Conversion Functions (Scalar) --- //

// Applies 3x3 Opsin Absorbance matrix (optionally premultiplied) and bias.
#[inline(always)]
fn opsin_absorbance(
    r: f32, g: f32, b: f32,
    premul_absorb: &[f32; 9], // Matrix elements (row-major)
    bias: &[f32; 3],
) -> [f32; 3] {
    let mixed0 = premul_absorb[0] * r + premul_absorb[1] * g + premul_absorb[2] * b + bias[0];
    let mixed1 = premul_absorb[3] * r + premul_absorb[4] * g + premul_absorb[5] * b + bias[1];
    let mixed2 = premul_absorb[6] * r + premul_absorb[7] * g + premul_absorb[8] * b + bias[2];
    [mixed0, mixed1, mixed2]
}

// Computes premultiplied Opsin Absorbance matrix and combined bias/cbrt constants.
// Stores 12 floats: 9 for matrix, 3 for bias_cbrt.
pub fn compute_premul_absorb(intensity_target: f32) -> [f32; 12] {
    use opsin_consts::*;
    let mut premul_absorb = [0.0f32; 12];
    let mul = intensity_target / 255.0;
    for r in 0..3 {
        for c in 0..3 {
            premul_absorb[r * 3 + c] = K_OPSIN_ABSORBANCE_MATRIX[r][c] * mul;
        }
    }
    for i in 0..3 {
        premul_absorb[9 + i] = K_NEG_OPSIN_ABSORBANCE_BIAS_CBRT[i];
    }
    premul_absorb
}

// Converts Linear RGB row to XYB row (in-place).
pub fn linear_rgb_row_to_xyb(
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
    premul_absorb: &[f32; 12], // Output of compute_premul_absorb
    xsize: usize,
) {
    debug_assert!(row_r.len() >= xsize);
    debug_assert!(row_g.len() >= xsize);
    debug_assert!(row_b.len() >= xsize);

    let premul_matrix = array_ref!(premul_absorb, 0, 9); // Use local macro directly
    let bias_cbrt = array_ref!(premul_absorb, 9, 3);     // Use local macro directly

    for x in 0..xsize {
        let r = row_r[x];
        let g = row_g[x];
        let b = row_b[x];

        let [mixed0, mixed1, mixed2] = opsin_absorbance(
            r, g, b,
            premul_matrix,
            &opsin_consts::K_OPSIN_ABSORBANCE_BIAS, // Use original bias here
        );

        // Clamp to zero before cube root
        let m0 = mixed0.max(0.0);
        let m1 = mixed1.max(0.0);
        let m2 = mixed2.max(0.0);

        // Cube root and add bias component (which is -cbrt(original_bias))
        let cr0 = m0.cbrt() + bias_cbrt[0];
        let cr1 = m1.cbrt() + bias_cbrt[1];
        let cr2 = m2.cbrt() + bias_cbrt[2];

        // Final XYB conversion
        row_r[x] = 0.5 * (cr0 - cr1); // X
        row_g[x] = 0.5 * (cr0 + cr1); // Y
        row_b[x] = cr2;               // B
    }
}

// Scales XYB values in-place using affine transform.
pub fn scale_xyb_row(
    row_x: &mut [f32],
    row_y: &mut [f32],
    row_b: &mut [f32],
    xsize: usize,
) {
    use opsin_consts::{K_SCALED_XYB_OFFSET, K_SCALED_XYB_SCALE};
    debug_assert!(row_x.len() >= xsize);
    debug_assert!(row_y.len() >= xsize);
    debug_assert!(row_b.len() >= xsize);

    for x in 0..xsize {
         // Original C++ order seems different from typical X,Y,B - matching it here:
        // row2[x] = (row2[x] - row1[x] + jxl::cms::kScaledXYBOffset[2]) * jxl::cms::kScaledXYBScale[2]; // B (depends on Y)
        // row0[x] = (row0[x] + jxl::cms::kScaledXYBOffset[0]) * jxl::cms::kScaledXYBScale[0];             // X
        // row1[x] = (row1[x] + jxl::cms::kScaledXYBOffset[1]) * jxl::cms::kScaledXYBScale[1];             // Y

        let original_y = row_y[x]; // Need original Y for B calculation
        row_b[x] = (row_b[x] - original_y + K_SCALED_XYB_OFFSET[2]) * K_SCALED_XYB_SCALE[2];
        row_x[x] = (row_x[x] + K_SCALED_XYB_OFFSET[0]) * K_SCALED_XYB_SCALE[0];
        row_y[x] = (original_y + K_SCALED_XYB_OFFSET[1]) * K_SCALED_XYB_SCALE[1];
    }
}

// --- Helpers --- //

// --- Tests --- //
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TOLERANCE: f32 = 1e-6;

    fn assert_approx_eq_slice(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < tolerance, "{} vs {}", va, vb);
        }
    }

    #[test]
    fn test_xyb_conversion_srgb_white() {
        let intensity_target = 255.0;
        let premul_absorb = compute_premul_absorb(intensity_target);

        // sRGB white (linear)
        let mut r = vec![1.0];
        let mut g = vec![1.0];
        let mut b = vec![1.0];

        linear_rgb_row_to_xyb(&mut r, &mut g, &mut b, &premul_absorb, 1);

        // Expected XYB for linear sRGB white (before scaling)
        // Calculated using the formulas (values might differ slightly from spec due to f32)
        let mixed = opsin_absorbance(1.0, 1.0, 1.0, array_ref!(premul_absorb, 0, 9), &opsin_consts::K_OPSIN_ABSORBANCE_BIAS);
        let bias_cbrt = array_ref!(premul_absorb, 9, 3);
        let cr0 = mixed[0].max(0.0).cbrt() + bias_cbrt[0];
        let cr1 = mixed[1].max(0.0).cbrt() + bias_cbrt[1];
        let cr2 = mixed[2].max(0.0).cbrt() + bias_cbrt[2];
        let expected_x = 0.5 * (cr0 - cr1);
        let expected_y = 0.5 * (cr0 + cr1);
        let expected_b = cr2;

        assert_approx_eq_slice(&r, &[expected_x], TOLERANCE);
        assert_approx_eq_slice(&g, &[expected_y], TOLERANCE);
        assert_approx_eq_slice(&b, &[expected_b], TOLERANCE);
    }

    #[test]
    fn test_xyb_scaling() {
        // Use the white point XYB values from previous test
        let intensity_target = 255.0;
        let premul_absorb = compute_premul_absorb(intensity_target);
        let mixed = opsin_absorbance(1.0, 1.0, 1.0, array_ref!(premul_absorb, 0, 9), &opsin_consts::K_OPSIN_ABSORBANCE_BIAS);
        let bias_cbrt = array_ref!(premul_absorb, 9, 3);
        let cr0 = mixed[0].max(0.0).cbrt() + bias_cbrt[0];
        let cr1 = mixed[1].max(0.0).cbrt() + bias_cbrt[1];
        let cr2 = mixed[2].max(0.0).cbrt() + bias_cbrt[2];
        let mut x = vec![0.5 * (cr0 - cr1)];
        let mut y = vec![0.5 * (cr0 + cr1)];
        let mut b = vec![cr2];

        let y_orig = y[0];
        let x_orig = x[0];
        let b_orig = b[0];

        scale_xyb_row(&mut x, &mut y, &mut b, 1);

        use opsin_consts::{K_SCALED_XYB_OFFSET, K_SCALED_XYB_SCALE};
        let expected_b_scaled = (b_orig - y_orig + K_SCALED_XYB_OFFSET[2]) * K_SCALED_XYB_SCALE[2];
        let expected_x_scaled = (x_orig + K_SCALED_XYB_OFFSET[0]) * K_SCALED_XYB_SCALE[0];
        let expected_y_scaled = (y_orig + K_SCALED_XYB_OFFSET[1]) * K_SCALED_XYB_SCALE[1];

        assert_approx_eq_slice(&b, &[expected_b_scaled], TOLERANCE);
        assert_approx_eq_slice(&x, &[expected_x_scaled], TOLERANCE);
        assert_approx_eq_slice(&y, &[expected_y_scaled], TOLERANCE);

    }
} 