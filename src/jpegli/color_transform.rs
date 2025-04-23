// Ported from lib/jpegli/color_transform.cc

use alloc::vec::Vec;
use crate::error::EncodingError;

// --- Constants for BT.601 Full Range YCbCr <-> RGB ---
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;

// RGB -> YCbCr constants
const AMP_R: f32 = 0.701; // 1.0 - KR
const AMP_B: f32 = 0.886; // 1.0 - KB
const DIFF_R: f32 = AMP_R + KR; // 1.0
const DIFF_B: f32 = AMP_B + KB; // 1.0
// Note: The C++ code has norm factors derived differently, leading to 1 / (1 + KG + KB) approx 0.5 * 1/KB?
// Let's re-derive based on standard equations: Cb = 0.5 * (B - Y) / (1 - KB), Cr = 0.5 * (R - Y) / (1 - KR)
// Cb = 0.5 * (B - (KR*R + KG*G + KB*B)) / (1 - KB)
// Cr = 0.5 * (R - (KR*R + KG*G + KB*B)) / (1 - KR)
// To map to [0, 255] (or equivalent float range centered at 128):
// Cb' = 128 + Cb * (2 * (1 - KB)) = 128 + B - Y
// Cr' = 128 + Cr * (2 * (1 - KR)) = 128 + R - Y
// Let's use these direct forms, matching typical implementations.

// YCbCr -> RGB constants
const CRCR: f32 = 1.402;          // 2 * (1 - KR)
const CGCB: f32 = -0.344136;     // -KB * (2 * (1 - KB)) / KG
const CGCR: f32 = -0.714136;     // -KR * (2 * (1 - KR)) / KG
const CBCB: f32 = 1.772;          // 2 * (1 - KB)


// --- Color Transform Functions (Operating on f32 planar data [0.0, 1.0]) --- //

/// Converts Linear RGB planes to YCbCr planes (in-place modification of first 3 planes).
pub fn linear_rgb_to_ycbcr(planes: &mut [Vec<f32>], num_pixels: usize) {
    if planes.len() < 3 { return; }
    let (r_plane, rest) = planes.split_at_mut(1);
    let (g_plane, b_plane) = rest.split_at_mut(1);
    let r_plane = &mut r_plane[0];
    let g_plane = &mut g_plane[0];
    let b_plane = &mut b_plane[0];

    for i in 0..num_pixels {
        let r = r_plane[i];
        let g = g_plane[i];
        let b = b_plane[i];

        let y  = KR * r + KG * g + KB * b;
        let cb = 0.5 + 0.5 * (b - y) / (1.0 - KB); // Scaled to [0, 1], center 0.5
        let cr = 0.5 + 0.5 * (r - y) / (1.0 - KR); // Scaled to [0, 1], center 0.5

        r_plane[i] = y;
        g_plane[i] = cb;
        b_plane[i] = cr;
    }
}

/// Converts YCbCr planes (in first 3 planes) to Linear RGB planes (in-place).
pub fn ycbcr_to_linear_rgb(planes: &mut [Vec<f32>], num_pixels: usize) {
     if planes.len() < 3 { return; }
    let (y_plane, rest) = planes.split_at_mut(1);
    let (cb_plane, cr_plane) = rest.split_at_mut(1);
    let y_plane = &mut y_plane[0];
    let cb_plane = &mut cb_plane[0];
    let cr_plane = &mut cr_plane[0];

    for i in 0..num_pixels {
        let y  = y_plane[i];
        let cb = cb_plane[i] - 0.5; // Center around 0
        let cr = cr_plane[i] - 0.5; // Center around 0

        let r = y            + cr * CRCR;
        let g = y + cb * CGCB + cr * CGCR;
        let b = y + cb * CBCB;

        // Clamp results to [0, 1] as YCbCr can represent out-of-gamut RGB
        y_plane[i] = r.clamp(0.0, 1.0);
        cb_plane[i] = g.clamp(0.0, 1.0);
        cr_plane[i] = b.clamp(0.0, 1.0);
    }
}

/// Converts CMYK planes to YCCK planes (in-place modification).
/// Assumes input CMYK is [0, 1] range (0=no ink, 1=max ink).
pub fn cmyk_to_ycck(planes: &mut [Vec<f32>], num_pixels: usize) {
    if planes.len() < 4 { return; }

    // 1. CMY -> Linear RGB (Invert: 1.0 - C, 1.0 - M, 1.0 - Y)
    for c in 0..3 {
        for i in 0..num_pixels {
            planes[c][i] = 1.0 - planes[c][i];
        }
    }

    // 2. Linear RGB -> YCbCr (modifies first 3 planes)
    linear_rgb_to_ycbcr(planes, num_pixels);

    // K channel (plane 3) remains unchanged.
}

/// Converts YCCK planes to CMYK planes (in-place modification).
/// Assumes output CMYK is [0, 1] range (0=no ink, 1=max ink).
pub fn ycck_to_cmyk(planes: &mut [Vec<f32>], num_pixels: usize) {
     if planes.len() < 4 { return; }

    // 1. YCbCr -> Linear RGB (modifies first 3 planes)
    ycbcr_to_linear_rgb(planes, num_pixels);

    // 2. Linear RGB -> CMY (Invert: 1.0 - R, 1.0 - G, 1.0 - B)
     for c in 0..3 {
        for i in 0..num_pixels {
            planes[c][i] = 1.0 - planes[c][i];
        }
    }
    // K channel (plane 3) remains unchanged.
}

/// Converts Grayscale plane (plane 0) to RGB planes (copies L to G and B).
pub fn grayscale_to_rgb(planes: &mut [Vec<f32>], num_pixels: usize) {
    if planes.len() < 3 { return; }
    if planes[0].len() < num_pixels || planes[1].len() < num_pixels || planes[2].len() < num_pixels {
        // Avoid panic if planes are too small (shouldn't happen in normal use)
        return;
    }
    // Use split_at_mut to avoid simultaneous borrows
    let (plane0_slice, rest) = planes.split_at_mut(1);
    let plane0 = &plane0_slice[0]; // Immutable borrow of plane 0 is fine now
    
    let (plane1_slice, plane2_slice) = rest.split_at_mut(1);
    let plane1 = &mut plane1_slice[0];
    let plane2 = &mut plane2_slice[0];

    plane1[..num_pixels].copy_from_slice(&plane0[..num_pixels]);
    plane2[..num_pixels].copy_from_slice(&plane0[..num_pixels]);
}

// --- Plane-based YCbCr conversion ---

#[inline(always)]
fn rgb_to_ycbcr_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    (y, cb, cr)
}

pub(crate) fn rgb_to_ycbcr_planes(
    r_plane: &mut Vec<f32>,
    g_plane: &mut Vec<f32>,
    b_plane: &mut Vec<f32>,
    num_pixels: usize,
) {
    assert!(r_plane.len() >= num_pixels);
    assert!(g_plane.len() >= num_pixels);
    assert!(b_plane.len() >= num_pixels);

    // Process pixel by pixel, modifying planes in place
    for i in 0..num_pixels {
        let r = r_plane[i];
        let g = g_plane[i];
        let b = b_plane[i];
        let (y, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
        r_plane[i] = y;
        g_plane[i] = cb;
        b_plane[i] = cr;
    }
    // Rename planes conceptually (r_plane now holds Y, etc.)
}

// --- Plane-based YCCK conversion ---

#[inline(always)]
fn cmyk_to_ycck_pixel(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32, f32) {
    // CMYK to RGB (approximate, assumes additive inverse)
    let r = (255.0 - c) * (255.0 - k) / 255.0;
    let g = (255.0 - m) * (255.0 - k) / 255.0;
    let b = (255.0 - y) * (255.0 - k) / 255.0;
    // RGB to YCbCr
    let (yc, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
    // Keep K channel
    (yc, cb, cr, k)
}

pub(crate) fn cmyk_to_ycck_planes(
    c_plane: &mut Vec<f32>,
    m_plane: &mut Vec<f32>,
    y_plane: &mut Vec<f32>,
    k_plane: &mut Vec<f32>,
    num_pixels: usize,
) {
    assert!(c_plane.len() >= num_pixels);
    assert!(m_plane.len() >= num_pixels);
    assert!(y_plane.len() >= num_pixels);
    assert!(k_plane.len() >= num_pixels);

    for i in 0..num_pixels {
        let c = c_plane[i];
        let m = m_plane[i];
        let y = y_plane[i];
        let k = k_plane[i];
        let (yc, cb, cr, k_out) = cmyk_to_ycck_pixel(c, m, y, k);
        c_plane[i] = yc; // Y
        m_plane[i] = cb; // Cb
        y_plane[i] = cr; // Cr
        k_plane[i] = k_out; // K (unmodified)
    }
    // Rename planes conceptually
}

// --- Tests --- //
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TOLERANCE_PRECISE: f32 = 1e-5; // Original tolerance
    const TOLERANCE_COLOR: f32 = 1e-3; // Slightly higher tolerance for color conversions

    fn assert_approx_eq_vec(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((va - vb).abs() < tolerance, "Mismatch at index {}: {} vs {}", i, va, vb);
        }
    }

    #[test]
    fn test_rgb_ycbcr_roundtrip() {
        let r_orig = vec![0.1, 0.5, 1.0, 0.0, 0.75];
        let g_orig = vec![0.2, 0.4, 0.9, 1.0, 0.25];
        let b_orig = vec![0.3, 0.6, 0.8, 0.5, 0.50];
        let num_pixels = r_orig.len();

        let mut planes = vec![r_orig.clone(), g_orig.clone(), b_orig.clone()];

        linear_rgb_to_ycbcr(&mut planes, num_pixels);
        ycbcr_to_linear_rgb(&mut planes, num_pixels);

        assert_approx_eq_vec(&planes[0], &r_orig, TOLERANCE_PRECISE);
        assert_approx_eq_vec(&planes[1], &g_orig, TOLERANCE_PRECISE);
        assert_approx_eq_vec(&planes[2], &b_orig, TOLERANCE_PRECISE);
    }

    #[test]
    fn test_grayscale_to_rgb() {
        let l_orig = vec![0.1, 0.5, 1.0, 0.0];
        let num_pixels = l_orig.len();
        let mut planes = vec![
            l_orig.clone(),
            vec![0.0; num_pixels],
            vec![0.0; num_pixels],
        ];

        grayscale_to_rgb(&mut planes, num_pixels);

        // Use direct equality check for exact copies
        assert_eq!(&planes[0], &l_orig);
        assert_eq!(&planes[1], &l_orig);
        assert_eq!(&planes[2], &l_orig);
    }

    #[test]
    fn test_cmyk_ycck_roundtrip() {
        let c_orig = vec![0.1, 0.9, 0.0, 0.5];
        let m_orig = vec![0.2, 0.8, 1.0, 0.4];
        let y_orig = vec![0.3, 0.7, 0.5, 0.6];
        let k_orig = vec![0.4, 0.1, 0.0, 1.0];
        let num_pixels = c_orig.len();

        let mut planes = vec![
            c_orig.clone(),
            m_orig.clone(),
            y_orig.clone(),
            k_orig.clone(),
        ];

        cmyk_to_ycck(&mut planes, num_pixels);
        ycck_to_cmyk(&mut planes, num_pixels);

        assert_approx_eq_vec(&planes[0], &c_orig, TOLERANCE_PRECISE);
        assert_approx_eq_vec(&planes[1], &m_orig, TOLERANCE_PRECISE);
        assert_approx_eq_vec(&planes[2], &y_orig, TOLERANCE_PRECISE);
        assert_approx_eq_vec(&planes[3], &k_orig, TOLERANCE_PRECISE);
    }

    #[test]
    fn test_rgb_to_ycbcr_planes() {
        let mut r = vec![255.0, 0.0, 0.0];
        let mut g = vec![0.0, 255.0, 0.0];
        let mut b = vec![0.0, 0.0, 255.0];
        let n = 3;

        // Calculate expected values using the *exact* pixel function
        let mut expected_y = Vec::with_capacity(n);
        let mut expected_cb = Vec::with_capacity(n);
        let mut expected_cr = Vec::with_capacity(n);
        for i in 0..n {
            let (y, cb, cr) = rgb_to_ycbcr_pixel(r[i], g[i], b[i]);
            expected_y.push(y);
            expected_cb.push(cb);
            expected_cr.push(cr);
        }

        rgb_to_ycbcr_planes(&mut r, &mut g, &mut b, n);

        // Use increased tolerance for color conversion results
        assert_approx_eq_vec(&r, &expected_y, TOLERANCE_COLOR);
        assert_approx_eq_vec(&g, &expected_cb, TOLERANCE_COLOR);
        assert_approx_eq_vec(&b, &expected_cr, TOLERANCE_COLOR);
    }

    #[test]
    fn test_cmyk_to_ycck_planes() {
        let mut c = vec![0.0, 255.0, 0.0, 255.0];
        let mut m = vec![255.0, 0.0, 0.0, 255.0];
        let mut y_p = vec![255.0, 0.0, 255.0, 255.0];
        let mut k = vec![0.0, 0.0, 0.0, 128.0]; // K=0 and K=128
        let n = 4;

        // Calculate expected values using the *exact* pixel function
        let mut expected_yc = Vec::with_capacity(n);
        let mut expected_cb = Vec::with_capacity(n);
        let mut expected_cr = Vec::with_capacity(n);
        let mut expected_k = Vec::with_capacity(n);
        for i in 0..n {
             let (yc, cb, cr, k_out) = cmyk_to_ycck_pixel(c[i], m[i], y_p[i], k[i]);
             expected_yc.push(yc);
             expected_cb.push(cb);
             expected_cr.push(cr);
             expected_k.push(k_out);
        }

        cmyk_to_ycck_planes(&mut c, &mut m, &mut y_p, &mut k, n);

        // Use increased tolerance for color conversion results
        assert_approx_eq_vec(&c, &expected_yc, TOLERANCE_COLOR);
        assert_approx_eq_vec(&m, &expected_cb, TOLERANCE_COLOR);
        assert_approx_eq_vec(&y_p, &expected_cr, TOLERANCE_COLOR);
        assert_approx_eq_vec(&k, &expected_k, TOLERANCE_COLOR);
    }
} 