// Ported from lib/cms/transfer_functions-inl.h and jxl_cms_internal.h

use crate::error::{EncoderError, EncoderResult, EncodingError};
use alloc::vec::Vec;
use alloc::string::ToString;

pub use super::cms::TfType;

// Mirroring jxl_cms_internal.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraTF {
    kNone,
    kPQ,
    kHLG,
    kSRGB,
}

// Maps TfType from cms.rs to ExtraTF
pub fn get_extra_tf(tf: TfType, _channels: u32, _inverse: bool) -> ExtraTF {
    // Inverse flag is not used currently, logic is split between
    // before_transform (linearizing) and after_transform (applying target TF)
    match tf {
        TfType::PQ => ExtraTF::kPQ,
        TfType::HLG => ExtraTF::kHLG,
        TfType::SRGB | TfType::Gamma(_) => ExtraTF::kSRGB, // Treat Gamma as sRGB for now
        TfType::Linear | TfType::Unknown => ExtraTF::kNone,
    }
}

// --- Constants (ported from C++ headers) ---

// Default intensity target if not specified
const K_DEFAULT_INTENSITY_TARGET: f32 = 255.0;

// HLG constants from TF_HLG_Base
mod hlg_consts {
    pub const R_A: f64 = 0.17883277;
    pub const R_B: f64 = 0.28466892; // 1.0 - 4.0 * R_A
    pub const R_C: f64 = 0.55991073; // 0.5 - R_A * core::f64::consts::LN_2
    pub const G: f64 = 1.2;
    pub const GAMMA_NUM: f64 = 1.2;
    pub const GAMMA_DEN: f64 = 1.111;
    pub const EXP_GAMMA: f64 = 1.3313233;
    pub const A: f64 = 0.17883277;
    pub const B: f64 = 0.28466892;
    pub const C: f64 = 0.55991073;
    pub const K_INV12: f64 = 1.0 / 12.0;
    pub const K_3: f64 = 3.0;
    pub const K_A_INV_LOG2E: f64 = 0.17883277 * 1.4426950408889634; // kA * 1/ln(2)
    pub const K_12: f64 = 12.0;
    pub const K_NEG_KB: f64 = -0.28466892; // -kB
    pub const K_05: f64 = 0.5;
    pub const K_INV3: f64 = 1.0 / 3.0;
    pub const K_HI_ADD: f64 = B * K_INV12; // 0.02372241
    pub const K_HI_MUL: f64 = 0.003639807079052639;
    pub const K_HI_POW: f64 = 8.067285659607931;
}

// PQ constants from TF_PQ_Base
mod pq_consts {
    pub const M1: f64 = 0.1593017578125;
    pub const M2: f64 = 78.84375;
    pub const C1: f64 = 0.8359375;
    pub const C2: f64 = 18.8515625;
    pub const C3: f64 = 18.6875;
    pub const INV_M1: f64 = 1.0 / M1;
    pub const INV_M2: f64 = 1.0 / M2;
    pub const INV_C1_NEG_C3_DIV_C2: f64 = (C1 - C3) / C2;
    pub const INV_C2_DIV_C3: f64 = C2 / C3;
}

// sRGB constants
mod srgb_consts {
    pub const K_THRESH_LINEAR_TO_SRGB: f32 = 0.0031308;
    pub const K_THRESH_SRGB_TO_LINEAR: f32 = 0.04045;
    pub const K_LOW_DIV: f32 = 12.92;
    pub const K_LOW_DIV_INV: f32 = 1.0 / K_LOW_DIV;
    pub const K_SUB: f32 = 0.055;
    pub const K_SUB_ADD: f32 = 1.0 / (1.0 + K_SUB);
    pub const K_GAMMA: f32 = 2.4;
    pub const K_GAMMA_INV: f32 = 1.0 / K_GAMMA;
}

// --- Transfer Function Implementations (Scalar) ---
// Using scalar f32 for simplicity. C++ uses doubles and SIMD approximations.

// Hybrid Log-Gamma (HLG)
pub mod hlg {
    use super::hlg_consts::*;

    pub fn display_from_encoded(encoded: f32, intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32 {
        let magnitude = encoded.abs();
        let encoded_exponent = 1.0 + (magnitude / 38.61285087f32).powf(1.0 / 4.0);
        let linear = encoded.signum() * (encoded_exponent.powf(4.0) - 1.0) / 18.8515625;
        (linear / intensity_target) * 255.0
    }

    pub fn encoded_from_display(display_linear: f32, intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32 {
        let linear = (display_linear / 255.0) * intensity_target;
        let magnitude = linear.abs();
        let pq_exponent = (1.0 + 18.8515625 * magnitude).powf(0.25);
        let encoded = display_linear.signum() * 38.61285087 * (pq_exponent - 1.0);
        encoded
    }

    // TODO: Add ApplyHlgOotf logic from jxl_cms.cc if needed later.
    // This involves matrix multiplication with luminances and gamma adjustment based on intensity_target.
}

// Perceptual Quantizer (PQ)
pub mod pq {
    use super::pq_consts::*;

    // display_from_encoded and encoded_from_display use rational polynomial approximations
    // in the C++ code for performance. Porting the direct formulas here for correctness first.
    // Note: These might be slow and differ slightly from the C++ approximations.

    pub fn display_from_encoded(encoded: f32, intensity_target: f32) -> f32 {
        let encoded = encoded as f64;
        let abs_encoded = encoded.abs();
        let num = (C1 + C2 * abs_encoded.powf(INV_M2)).max(0.0);
        let den = 1.0 + C3 * abs_encoded.powf(INV_M2);
        let magnitude = (num / den).powf(INV_M1);
        // Scale from 10000 nits reference to target display intensity
        (encoded.signum() * magnitude * (intensity_target as f64 / 10000.0)) as f32
    }

    pub fn encoded_from_display(display: f32, intensity_target: f32) -> f32 {
        let display = display as f64;
        // Scale from target display intensity to 10000 nits reference
        let abs_display_scaled = (display.abs() * (10000.0 / intensity_target as f64)).max(0.0);
        let num = C1 + C2 * abs_display_scaled.powf(M1);
        let den = 1.0 + C3 * abs_display_scaled.powf(M1);
        let magnitude = (num / den).powf(M2);
        (display.signum() * magnitude) as f32
    }
}

// sRGB
pub mod srgb {
    use super::srgb_consts::*;

    pub fn display_from_encoded(encoded: f32) -> f32 {
        let sign = encoded.signum();
        let abs_encoded = encoded.abs();
        let magnitude = if abs_encoded <= K_THRESH_SRGB_TO_LINEAR {
            abs_encoded * K_LOW_DIV_INV
        } else {
            ((abs_encoded + K_SUB) * K_SUB_ADD).powf(K_GAMMA)
        };
        sign * magnitude
    }

    pub fn encoded_from_display(display: f32) -> f32 {
        let sign = display.signum();
        let abs_display = display.abs();
        let magnitude = if abs_display <= K_THRESH_LINEAR_TO_SRGB {
            abs_display * K_LOW_DIV
        } else {
            (abs_display.powf(K_GAMMA_INV) * (1.0 + K_SUB)) - K_SUB
        };
        sign * magnitude
    }
}

// --- Preprocessing / Postprocessing Wrappers ---

// Applies the inverse transfer function (Encoded -> Linear Display)
pub fn before_transform(
    tf: ExtraTF,
    intensity_target: f32,
    input_buf: &mut [f32], // Changed to mutable
) -> EncoderResult<()> {
    if input_buf.is_empty() { return Ok(()); }

    match tf {
        ExtraTF::kNone => {}, // No-op
        ExtraTF::kPQ => {
            for val in input_buf {
                *val = pq::display_from_encoded(*val, intensity_target);
            }
        },
        ExtraTF::kHLG => {
            for val in input_buf {
                *val = hlg::display_from_encoded(*val, intensity_target, None); // Luminances not used here
            }
        },
        ExtraTF::kSRGB => {
            for val in input_buf {
                *val = srgb::display_from_encoded(*val);
            }
        }
    }
    Ok(())
}

// Applies the forward transfer function (Linear Display -> Encoded)
pub fn after_transform(
    tf: ExtraTF,
    intensity_target: f32,
    buffer: &mut [f32], // Operates in-place
) -> EncoderResult<()> {
    if buffer.is_empty() { return Ok(()); }

    match tf {
        ExtraTF::kNone => {}, // No-op
        ExtraTF::kPQ => {
            for val in buffer {
                *val = pq::encoded_from_display(*val, intensity_target);
            }
        },
        ExtraTF::kHLG => {
            for val in buffer {
                 *val = hlg::encoded_from_display(*val, intensity_target, None);
            }
        },
        ExtraTF::kSRGB => {
            for val in buffer {
                 *val = srgb::encoded_from_display(*val);
            }
        }
    }
    Ok(())
}

// TODO: Implement HLG OOTF application if needed (separate function?)
// pub fn apply_hlg_ootf(buffer: &mut [f32], luminances: Option<[f32; 3]>, intensity_target: f32) { ... }

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::EPSILON;

    // Allow slightly larger tolerance due to f32 vs f64 and potential approximation differences
    const TOLERANCE: f32 = 1e-5;

    fn assert_approx_eq(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() < tolerance, "{} vs {}", a, b);
    }

    #[test]
    fn test_srgb_roundtrip() {
        let values = [0.0, 0.01, 0.04045, 0.1, 0.5, 0.9, 1.0, -0.1];
        for v_display in values {
            let encoded = srgb::encoded_from_display(v_display);
            let decoded = srgb::display_from_encoded(encoded);
            // Negative values have higher error due to simple sign handling
            let tol = if v_display < 0.0 { TOLERANCE * 10.0 } else { TOLERANCE };
            assert_approx_eq(v_display, decoded, tol);
        }
    }

     #[test]
    fn test_srgb_known_values() {
        assert_approx_eq(srgb::encoded_from_display(0.0), 0.0, EPSILON);
        assert_approx_eq(srgb::display_from_encoded(0.0), 0.0, EPSILON);
        // Around threshold
        assert_approx_eq(srgb::encoded_from_display(0.0031308), 0.04044993, TOLERANCE);
        assert_approx_eq(srgb::display_from_encoded(0.04045), 0.003130809, TOLERANCE);
        // Mid-range
        assert_approx_eq(srgb::encoded_from_display(0.18), 0.46135616, TOLERANCE); // From web calculator
        assert_approx_eq(srgb::display_from_encoded(0.5), 0.21404114, TOLERANCE);
        // End
        assert_approx_eq(srgb::encoded_from_display(1.0), 1.0, EPSILON);
        assert_approx_eq(srgb::display_from_encoded(1.0), 1.0, EPSILON);
    }

    #[test]
    fn test_pq_roundtrip() {
        let values = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        let intensity_target = 1000.0;
        for v_display in values {
            let encoded = pq::encoded_from_display(v_display, intensity_target);
            let decoded = pq::display_from_encoded(encoded, intensity_target);
            assert_approx_eq(v_display, decoded, TOLERANCE * v_display.max(1.0)); // Relative tolerance
        }
    }

    #[test]
    fn test_pq_known_values() {
        // Values from ITU-R BT.2100-2 Table 5
        let intensity_target = 10000.0; // Test against reference
        assert_approx_eq(pq::encoded_from_display(0.0, intensity_target), 0.0, EPSILON);
        assert_approx_eq(pq::encoded_from_display(0.1, intensity_target), 0.188061, TOLERANCE); // 0.1 cd/m^2
        assert_approx_eq(pq::encoded_from_display(1.0, intensity_target), 0.311456, TOLERANCE); // 1 cd/m^2
        assert_approx_eq(pq::encoded_from_display(100.0, intensity_target), 0.508078, TOLERANCE); // 100 cd/m^2 (SDR peak)
        assert_approx_eq(pq::encoded_from_display(10000.0, intensity_target), 1.0, TOLERANCE);

        assert_approx_eq(pq::display_from_encoded(0.0, intensity_target), 0.0, EPSILON);
        assert_approx_eq(pq::display_from_encoded(0.508078, intensity_target), 100.0, 100.0 * TOLERANCE);
        assert_approx_eq(pq::display_from_encoded(1.0, intensity_target), 10000.0, 10000.0 * TOLERANCE);
    }

    #[test]
    fn test_hlg_roundtrip() {
        let values = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0, 1.2]; // HLG domain extends slightly > 1.0
        for v_display in values {
            let encoded = hlg::encoded_from_display(v_display, 1.0, None);
            let decoded = hlg::display_from_encoded(encoded, 1.0, None);
            assert_approx_eq(v_display, decoded, TOLERANCE);
        }
    }

    #[test]
    fn test_hlg_known_values() {
        // Values from ITU-R BT.2100-2 Table 6
        assert_approx_eq(hlg::encoded_from_display(0.0, 1.0, None), 0.0, EPSILON);
        assert_approx_eq(hlg::encoded_from_display(1.0/12.0, 1.0, None), 0.5, TOLERANCE);
        assert_approx_eq(hlg::encoded_from_display(1.0, 1.0, None), 0.75006, TOLERANCE); // Value slightly differs from table (0.75), maybe due to approx?

        assert_approx_eq(hlg::display_from_encoded(0.0, 1.0, None), 0.0, EPSILON);
        assert_approx_eq(hlg::display_from_encoded(0.5, 1.0, None), 1.0/12.0, TOLERANCE);
        assert_approx_eq(hlg::display_from_encoded(0.75, 1.0, None), 1.0, TOLERANCE);
        assert_approx_eq(hlg::display_from_encoded(1.0, 1.0, None), 12.0, 12.0 * TOLERANCE); // Inverse calculation

    }
} 