// Ported from lib/cms/transfer_functions-inl.h and jxl_cms_internal.h

#![allow(non_camel_case_types)]

use crate::error::EncodingError;
use crate::jpegli::cms::TfType; // Import TfType
use core::{f32, f64};
#[cfg(feature = "std")]
use std::println;

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
    pub(super) const R_A: f64 = 0.17883277;
    pub(super) const R_B: f64 = 0.28466892; // 1.0 - 4.0 * R_A
    pub(super) const R_C: f64 = 0.55991073; // 0.5 - R_A * core::f64::consts::LN_2
    pub(super) const G: f64 = 1.2;
    pub(super) const GAMMA_NUM: f64 = 1.2;
    pub(super) const GAMMA_DEN: f64 = 1.111;
    pub(super) const EXP_GAMMA: f64 = 1.3313233;
    pub(super) const A: f64 = 0.17883277;
    pub(super) const B: f64 = 0.28466892;
    pub(super) const C: f64 = 0.55991073;
    pub(super) const K_INV12: f64 = 1.0 / 12.0;
    pub(super) const K_3: f64 = 3.0;
    pub(super) const K_A_INV_LOG2E: f64 = 0.17883277 * 1.4426950408889634; // kA * 1/ln(2)
    pub(super) const K_12: f64 = 12.0;
    pub(super) const K_NEG_KB: f64 = -0.28466892; // -kB
    pub(super) const K_05: f64 = 0.5;
    pub(super) const K_INV3: f64 = 1.0 / 3.0;
    pub(super) const K_HI_ADD: f64 = B * K_INV12; // 0.02372241
    pub(super) const K_HI_MUL: f64 = 0.003639807079052639;
    pub(super) const K_HI_POW: f64 = 8.067285659607931;
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
    use core::f64::consts::LN_2; // Import natural log of 2 if needed for C constant

    const A_F64: f64 = A;
    const B_F64: f64 = B;
    const C_F64: f64 = C;
    const INV12_F64: f64 = K_INV12;

    /// HLG EOTF (Electro-Optical Transfer Function): Encoded -> Linear Display (0-1 range relative to peak)
    pub fn display_from_encoded(encoded: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32 {
        #[cfg(feature = "std")]
        println!("[HLG_DEBUG] display_from_encoded input: encoded={}", encoded);
        let e_prime = encoded as f64;
        let linear = if e_prime <= 0.5 {
            let result = e_prime * e_prime / K_3; // Equivalent to e_prime^2 / 3.0
            #[cfg(feature = "std")]
            println!("[HLG_DEBUG] display_from_encoded (<=0.5 branch): e'={}, result={}", e_prime, result);
            result
        } else {
            // exp((E' - c) / a)
            let exp_arg = (e_prime - C_F64) / A_F64;
            // ((exp(exp_arg) + b) / 12)
            let result = (exp_arg.exp() + B_F64) * INV12_F64; // Equivalent to / 12.0
            #[cfg(feature = "std")]
            println!("[HLG_DEBUG] display_from_encoded (>0.5 branch): e'={}, exp_arg={}, result={}", e_prime, exp_arg, result);
            result
        };
        linear as f32
    }

    /// HLG OETF (Opto-Electrical Transfer Function): Linear Display (0-1 range relative to peak) -> Encoded
    pub fn encoded_from_display(display_linear: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32 {
        #[cfg(feature = "std")]
        println!("[HLG_DEBUG] encoded_from_display input: display_linear={}", display_linear);
        let l = display_linear.max(0.0) as f64; // Ensure non-negative
        let encoded = if l <= INV12_F64 / K_3 { // Equivalent to L <= 1.0/12.0
             let result = (K_3 * l).sqrt(); // Equivalent to sqrt(3 * L)
             #[cfg(feature = "std")]
             println!("[HLG_DEBUG] encoded_from_display (<=1/12 branch): l={}, result={}", l, result);
             result
        } else {
             // a * ln(12*L - b) + c
             let log_arg = K_12 * l - B_F64;
             let result = A_F64 * log_arg.ln() + C_F64;
             #[cfg(feature = "std")]
             println!("[HLG_DEBUG] encoded_from_display (>1/12 branch): l={}, log_arg={}, result={}", l, log_arg, result);
             result
        };
        encoded as f32
    }

    // Structure to hold HLG OOTF parameters
    // Corresponds to HlgOOTF in C++
    #[derive(Debug, Clone, Copy)]
    pub struct HlgOotf {
        apply_ootf: bool,
        exponent: f32,
        // Precomputed luminance factors (assuming BT.709 primaries for OOTF calculation)
        // Note: C++ uses actual primaries, this is a simplification for now.
        // TODO: Use actual primaries from ColorEncodingInternal if available.
        red_y: f32,
        green_y: f32,
        blue_y: f32,
    }

    impl HlgOotf {
        // Corresponds to HlgOOTF::FromSceneLight
        pub fn from_scene_light(display_luminance: f32) -> Self {
            let display_luminance = display_luminance.max(1e-6); // Avoid log(0)
            // Gamma calculation matching C++
            let gamma = 1.2 * (1.111f32).powf((display_luminance / 1000.0).log2());
            // Exponent is gamma - 1.0
            let exponent = gamma - 1.0;
            
            // Use standard BT.709 luminance coefficients
            let red_y = 0.2126;
            let green_y = 0.7152;
            let blue_y = 0.0722;

            Self {
                apply_ootf: true, // Assume OOTF should be applied by default
                exponent,
                red_y,
                green_y,
                blue_y,
            }
        }

        // Corresponds to HlgOOTF::ToSceneLight (Inverse OOTF)
        pub fn to_scene_light(display_luminance: f32) -> Self {
            let display_luminance = display_luminance.max(1e-6);
            // Gamma calculation matching C++
            let gamma = (1.0 / 1.2) * (1.111f32).powf(-(display_luminance / 1000.0).log2());
            let exponent = gamma - 1.0;

            // Use standard BT.709 luminance coefficients
            let red_y = 0.2126;
            let green_y = 0.7152;
            let blue_y = 0.0722;

            Self {
                apply_ootf: true,
                exponent,
                red_y,
                green_y,
                blue_y,
            }
        }

        /// Applies the HLG OOTF (forward or inverse based on exponent) to RGB pixel data.
        /// Operates in-place on planar buffers.
        pub fn apply(&self, r: &mut [f32], g: &mut [f32], b: &mut [f32], num_pixels: usize) {
            if !self.apply_ootf { return; }
            assert!(r.len() >= num_pixels);
            assert!(g.len() >= num_pixels);
            assert!(b.len() >= num_pixels);

            for i in 0..num_pixels {
                let luminance = self.red_y * r[i] + self.green_y * g[i] + self.blue_y * b[i];
                // Use fast_pow2f for pow(luminance, exponent)
                // exponent = gamma - 1
                // pow(lum, exp) = pow(lum, gamma-1) = pow(lum, gamma) / lum
                // C++ uses FastPowf(df, luminance, Set(df, exponent_))
                // Let's match the direct exponent application using standard powf for now.
                // Avoid powf if exponent is 0.
                let ratio = if self.exponent == 0.0 {
                    1.0 
                } else {
                    luminance.powf(self.exponent).min(1e9) // Clamp to avoid overflow
                };

                // Handle potential division by zero or NaN if luminance is <= 0
                let safe_ratio = if luminance <= 1e-9 { 1.0 } else { ratio }; // Keep original color if luminance is near zero

                r[i] *= safe_ratio;
                g[i] *= safe_ratio;
                b[i] *= safe_ratio;
            }
        }
    }

    // TODO: Implement ApplyHlgOotf (Opto-Optical Transfer Function).
    // This function should take linear RGB display light (relative to peak display luminance,
    // typically after CMS conversion) and apply the HLG system gamma adjustment based on
    // peak luminance (intensity_target) and potentially scene/surround assumptions
    // (see ITU-R BT.2100). It produces the non-linear HLG signal *before* the OETF.
    // The inverse function (Inverse OOTF) would be needed in `before_transform`.
    // C++ reference implementation location unclear.
    pub fn apply_hlg_ootf(
        // r: &mut f32, g: &mut f32, b: &mut f32, // Or operate on buffer
        // intensity_target: f32, // Peak display luminance in nits (e.g., 1000)
        // scene_luminance: f32, // Assumed scene white luminance (e.g. 100?)
    ) -> () {
        unimplemented!("HLG OOTF is not implemented.");
    }
}

// Perceptual Quantizer (PQ)
pub mod pq {
    use super::pq_consts::*;

    /* 
     * Original C++ implementation from jpegli/lib/cms/transfer_functions-inl.h:
     *
     * For PQ transfer functions, reference: SMPTE ST 2084:2014
     *
     * class TF_PQ_Base {
     *  public:
     *   HIR HIR_INLINE DisplayFromEncoded(const HIR encoded,
     *                                     const HIR intensity_target) const {
     *     if (encoded == 0.0) {
     *       return 0.0;
     *     }
     *     const HIR abs_encoded = abs(encoded);
     *     const HIR pow_inv_m2 = pow(abs_encoded, HIR(kINV_M2));
     *     const HIR num = max(pow_inv_m2 - kC1, 0.0);
     *     const HIR den = kC2 - kC3 * pow_inv_m2;
     *     const HIR magnitude = pow(num / den, HIR(kINV_M1));
     *     return CopySign(magnitude * (HIR(10000.0) / intensity_target), encoded);
     *   }
     *
     *   HIR HIR_INLINE EncodedFromDisplay(const HIR display,
     *                                     const HIR intensity_target) const {
     *     if (display == 0.0) {
     *       return 0.0;
     *     }
     *     const HIR abs_display = abs(display);
     *     // Y = reference display light level, scaled to the PQ range of 0-10000 nits
     *     const HIR Y = abs_display * (intensity_target / HIR(10000.0));
     *     const HIR Y_pow_m1 = pow(Y, HIR(kM1));
     *     const HIR num = kC1 + kC2 * Y_pow_m1;
     *     const HIR den = 1.0 + kC3 * Y_pow_m1;
     *     if (abs(den) < 1e-15) {
     *       return CopySign(pow(kC2 / kC3, HIR(kM2)), display);
     *     }
     *     const HIR result = CopySign(pow(num / den, HIR(kM2)), display);
     *     return result;
     *   }
     * };
     */

    pub fn display_from_encoded(encoded: f32, intensity_target: f32) -> f32 {
        #[cfg(feature = "std")]
        println!("\n[PQ_DEBUG] display_from_encoded input: encoded={}, intensity_target={}", encoded, intensity_target);

        let encoded_f64 = encoded as f64;
        if encoded_f64 == 0.0 { 
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] Early return for encoded=0.0: result=0.0");
            return 0.0;
        }

        let abs_encoded = encoded_f64.abs();

        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] Constants: M1={}, M2={}, C1={}, C2={}, C3={}", M1, M2, C1, C2, C3);
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] Derived: INV_M1={}, INV_M2={}", INV_M1, INV_M2);

        // EOTF based on C++ TF_PQ_Base::DisplayFromEncoded
        // d = pow((max(e^INV_M2 - C1, 0)) / (C2 - C3 * e^INV_M2), INV_M1)
        let pow_inv_m2 = abs_encoded.powf(INV_M2);
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] abs_encoded={}, pow_inv_m2=abs_encoded^INV_M2={}", abs_encoded, pow_inv_m2);

        let num = (pow_inv_m2 - C1).max(0.0);
        let den = C2 - C3 * pow_inv_m2;
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] num=max(pow_inv_m2-C1,0)={}, den=C2-C3*pow_inv_m2={}", num, den);

        let magnitude_unscaled = if den.abs() < 1e-15 { // Avoid division by zero/very small numbers
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] Denominator near zero! encoded={}, pow_inv_m2={}", encoded_f64, pow_inv_m2);
            // This case implies pow_inv_m2 is approximately C2/C3
            // The C++ base returns pow(C2/C3, INV_M1), let's approximate that
            // result = pow(num / den, INV_M1)
            // if den -> 0, and num -> max( (C2/C3) - C1, 0)
            // If C2/C3 > C1, then num > 0.
            // Let's return a large value or handle as error? For now, use the base impl's fallback idea
            // (C2 / C3).powf(INV_M1) // This would be the limit? Let's use 1.0 for now before scaling.
            1.0 // Before scaling by intensity_target
        } else {
            let result = (num / den).powf(INV_M1);
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] magnitude_unscaled = (num/den)^INV_M1 = ({}/{})^({}) = {}", num, den, INV_M1, result);
            result
        };

        // Scale to the target display intensity (output is 0-1 relative to intensity_target)
        // C++ scaling: magnitude * (10000.0 / intensity_target)
        // Our output should be relative 0-1, so scaling is needed here.
        let scale_factor = 10000.0 / (intensity_target as f64);
        let magnitude_relative = magnitude_unscaled * scale_factor;
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] scale_factor = 10000/intensity = {}, magnitude_relative = mag_unscaled*scale = {}", scale_factor, magnitude_relative);

        // Apply original sign and convert back to f32
        let final_result = magnitude_relative.copysign(encoded_f64) as f32;
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] display_from_encoded final_result: {}", final_result);
        // TODO: Verify if the output range should be [0, 1] relative to intensity_target
        // or absolute [0, intensity_target]. Current scaling matches C++ code structure.
        final_result
    }

    pub fn encoded_from_display(display_relative: f32, intensity_target: f32) -> f32 {
        #[cfg(feature = "std")]
        println!("\n[PQ_DEBUG] encoded_from_display input: display_relative={}, intensity_target={}", display_relative, intensity_target);

        let display_relative_f64 = display_relative as f64;
        if display_relative_f64 == 0.0 {
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] Early return for display_relative=0.0: result=0.0");
            return 0.0;
        }

        let abs_display_relative = display_relative_f64.abs();

        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] Constants: M1={}, M2={}, C1={}, C2={}, C3={}", M1, M2, C1, C2, C3);

        // OETF based on C++ TF_PQ_Base::EncodedFromDisplay
        // Y = reference display light level, scaled to the PQ range of 0-10000 nits
        // Input 'display_relative' is already relative to intensity_target (0-1)
        // Y = abs_display_relative * intensity_target / 10000.0
        let intensity_target_f64 = intensity_target as f64;
        let y = abs_display_relative * (intensity_target_f64 / 10000.0);
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] abs_display_relative={}, Y = abs_disp_rel * (intensity/10k) = {}", abs_display_relative, y);

        // e = pow((C1 + C2 * Y^M1) / (1 + C3 * Y^M1), M2)
        let y_pow_m1 = y.powf(M1);
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] y_pow_m1 = Y^M1 = {}^{} = {}", y, M1, y_pow_m1);

        let num = C1 + C2 * y_pow_m1;
        let den = 1.0 + C3 * y_pow_m1;
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] num = C1+C2*y_pow_m1={}, den=1+C3*y_pow_m1={}", num, den);

        let magnitude = if den.abs() < 1e-15 { // Avoid division by zero (should not happen for valid Y >= 0)
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] Denominator near zero! Y={}, y_pow_m1={}", y, y_pow_m1);
            // Fallback from C++ base: pow(C2 / C3, M2)
            (C2 / C3).powf(M2)
        } else {
            let result = (num / den).powf(M2);
            #[cfg(feature = "std")]
            println!("[PQ_DEBUG] magnitude = (num/den)^M2 = ({}/{})^({}) = {}", num, den, M2, result);
            result
        };

        // Apply original sign and convert back to f32
        let final_result = magnitude.copysign(display_relative_f64) as f32;
        #[cfg(feature = "std")]
        println!("[PQ_DEBUG] encoded_from_display final_result: {}", final_result);
        final_result
    }

    /*
    TODO: Add ApplyPqOotf logic from jxl_cms.cc if needed later.
    It involves scaling based on RGB values and isn't just a simple per-channel TF.
    It might belong in a different module or function that handles color transformations.
    */
}

// sRGB
mod srgb {
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
) -> Result<(), EncodingError> {
    if input_buf.is_empty() { return Ok(()); }

    match tf {
        ExtraTF::kNone => {}, // No-op
        ExtraTF::kPQ => {
            for val in input_buf.iter_mut() { // Use iter_mut()
                *val = pq::display_from_encoded(*val, intensity_target);
            }
        },
        ExtraTF::kHLG => {
            for val in input_buf.iter_mut() { // Use iter_mut()
                // TODO: Apply HLG OOTF/inverse OOTF if needed (currently just EOTF)
                *val = hlg::display_from_encoded(*val, intensity_target, None); // Luminances not used here
            }
        },
        ExtraTF::kSRGB => {
            for val in input_buf.iter_mut() { // Use iter_mut()
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
) -> Result<(), EncodingError> {
    if buffer.is_empty() { return Ok(()); }

    match tf {
        ExtraTF::kNone => {}, // No-op
        ExtraTF::kPQ => {
            for val in buffer.iter_mut() { // Use iter_mut()
                *val = pq::encoded_from_display(*val, intensity_target);
            }
        },
        ExtraTF::kHLG => {
            for val in buffer.iter_mut() { // Use iter_mut()
                 // TODO: Apply HLG OOTF/inverse OOTF if needed (currently just OETF)
                 *val = hlg::encoded_from_display(*val, intensity_target, None);
            }
        },
        ExtraTF::kSRGB => {
            for val in buffer.iter_mut() { // Use iter_mut()
                 *val = srgb::encoded_from_display(*val);
            }
        }
    }
    Ok(())
}

// TODO: Implement HLG OOTF application if needed (separate function?)
// pub fn apply_hlg_ootf(buffer: &mut [f32], luminances: Option<[f32; 3]>, intensity_target: f32) { ... }
// CURRENTLY UNIMPLEMENTED