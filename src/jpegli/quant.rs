use alloc::vec;
use alloc::vec::Vec;
//use core::num::NonZeroU16; // Likely not needed directly here anymore
use std::f32;
//use std::fmt::Write; // Not needed for core quant logic
use lazy_static::lazy_static; // Keep for standard tables

// Import constants from the dedicated module
use crate::jpegli::quant_constants::*;
// Use path relative to src/lib.rs for ffi types if defined there or re-exported
// use crate::{MAX_COMPONENTS}; // Assuming MAX_COMPONENTS is at crate root
use crate::error::EncodingError;
use crate::JpegColorType; // Import JpegColorType from crate root
use crate::SamplingFactor;

use super::SimplifiedTransferCharacteristics;
use super::Subsampling;
// Remove unresolved imports
// use crate::{ffi, MAX_COMPONENTS};

// Constants
// use crate::jpeg::MAX_COMPONENTS; // Removed this line
// use crate::ffi::DCTSIZE2; // DCTSIZE2 is likely part of ffi module

// --- Locally Defined Constants (replacing FFI) ---
pub(crate) const MAX_COMPONENTS: usize = 4;
pub(crate) const DCTSIZE2: usize = 64;

// --- Rust Replacements for FFI Types ---

/// Represents JPEG color spaces relevant to quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpegliColorSpace {
    RGB,
    YCbCr,
    GRAYSCALE,
    Unknown,
    // Add CMYK, YCCK if needed by other logic later
}

/// Parameters for a single component relevant to quantization.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JpegliComponentParams {
    pub h_samp_factor: u8,
    pub v_samp_factor: u8,
    pub quant_tbl_no: u8,
}

// Helper to convert JpegColorType to JpegliColorSpace
pub(crate) fn jpeg_color_type_to_jpegli_space(color_type: JpegColorType) -> JpegliColorSpace {
    match color_type {
        JpegColorType::Luma => JpegliColorSpace::GRAYSCALE,
        JpegColorType::Ycbcr => JpegliColorSpace::YCbCr,
        // Revert Rgb mapping for now
        // JpegColorType::Rgb => JpegliColorSpace::YCbCr, 
        JpegColorType::Cmyk | JpegColorType::Ycck => JpegliColorSpace::Unknown,
        _ => JpegliColorSpace::Unknown, // Default or handle other cases like Rgb
    }
}

// --- End Rust Replacements ---

// Enum mirroring C++ QuantPass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantPass {
    NoSearch,
    SearchFirstPass,
    SearchSecondPass,
}

// --- End Standard Annex K Constants ---

/// Helper function ported from jpegli DistanceToScale, using f32.
pub(crate) fn distance_to_scale(distance: f32, k: usize) -> f32 {
    if distance < DIST0 {
        distance
    } else {
        let exp = EXPONENT[k];
        let mul = DIST0.powf(1.0 - exp);
        // C++ uses fmaxf, equivalent to Rust's .max()
        (mul * distance.powf(exp)).max(0.5 * distance)
    }
}

/// Helper function ported from jpegli ScaleToDistance, using f32.
pub(crate) fn scale_to_distance(scale: f32, k: usize) -> f32 {
    if scale < DIST0 {
        scale
    } else {
        let exp = 1.0 / EXPONENT[k];
        let mul = DIST0.powf(1.0 - exp);
        // C++ uses std::min<float>
        (mul * scale.powf(exp)).min(2.0 * scale)
    }
}


/// Maps a libjpeg quality factor (1..100) to a jpegli Butteraugli distance.
/// Ported from jpegli C++ implementation (appears unchanged).
pub fn quality_to_distance(quality: u8) -> f32 {
    let quality = quality as f32;
    if quality >= 100.0 {
        0.01
    } else if quality >= 30.0 {
        0.1 + (100.0 - quality) * 0.09
    } else {
        // Adjusted quadratic formula from C++:
        // 53.0 / 3000.0 * q*q - 23.0 / 20.0 * q + 25.0
        // approx 0.017666 * q*q - 1.15 * q + 25.0
        (53.0 / 3000.0) * quality.powi(2) - (23.0 / 20.0) * quality + 25.0
    }
}

/// Replicates C++ DistanceToLinearQuality logic (used for standard tables).
pub(crate) fn distance_to_linear_quality(distance: f32) -> f32 {
    // Clamping added for safety, although C++ might rely on caller constraints
    let distance = distance.max(0.0);
    if distance <= 0.1 {
        1.0
    } else if distance <= 4.6 {
        (200.0 / 9.0) * (distance - 0.1)
    } else if distance <= 6.4 {
        // This formula diverges near quality 100 (distance 0.1)
        // and needs careful handling if distance is very low.
        // The C++ code likely assumes distance > 4.6 here.
        let denominator = 100.0 - (distance - 0.1) / 0.09;
        if denominator <= 0.0 {
            f32::INFINITY // Or a large number? C++ behavior unclear.
        } else {
            5000.0 / denominator
        }
    } else if distance < 25.0 {
        // Ensure argument to sqrt is non-negative
        let sqrt_arg = (848.0 * distance - 5330.0) / 120.0;
        if sqrt_arg < 0.0 {
            // Handle edge case - C++ might have UB or different result.
            // Return a large value or based on expected outcome.
             f32::INFINITY
        } else {
             let denominator = 3450.0 - 300.0 * sqrt_arg.sqrt();
             if denominator <= 0.0 {
                  f32::INFINITY
             } else {
                 530000.0 / denominator
             }
        }
    } else {
        5000.0 // Corresponds to quality 0?
    }
}


/// Creates a raw (unshifted) quantization table based on Jpegli's distance-based scaling or standard table scaling.
/// This function now aligns closely with the loop inside C++ SetQuantMatrices.
pub(crate) fn compute_quant_table_values(
    distance: f32,
    global_scale: f32,
    base_quant_matrix: &[f32; 64], // Slice reference to the base matrix (YCbCr, XYB, or Std)
    non_linear_scaling: bool,      // True for jpegli YCbCr/XYB, false for standard tables
    is_chroma_420: bool,           // True if this table is for chroma AND we are in 420 mode
    quant_max: i32,                // 255 for baseline, 32767 otherwise
) -> [u16; 64] {
    let mut table_data = [1u16; 64];

    for k in 0..64 {
        let mut scale = global_scale; // Start with global scale

        if non_linear_scaling {
            // Apply jpegli non-linear distance scaling
            scale *= distance_to_scale(distance.max(0.0), k); // Use f32 distance_to_scale
            if is_chroma_420 {
                // Apply RESCALE_420 only for chroma in 420 mode (using imported constant)
                scale *= RESCALE_420[k];
            }
        } else {
            // Apply standard table scaling (linear quality based)
            scale *= distance_to_linear_quality(distance);
        }

        let base_val = base_quant_matrix[k];
        let qval_f = scale * base_val; // f32 calculation
        let qval = qval_f.round() as i32;

        // Clamp the result - NO AdjustQuantVal step here for DQT values
        table_data[k] = qval.clamp(1, quant_max) as u16;
    }
    table_data
}

/// Estimates Butteraugli distance from quantization table values.
/// Corresponds to C++ QuantValsToDistance.
pub(crate) fn quant_vals_to_distance(
    raw_quant_tables: &[Option<[u16; 64]>; 4],
    num_components: usize,
    comp_info: &[JpegliComponentParams], // Use Rust struct
    jpeg_color_space: JpegliColorSpace, // Use Rust enum
    cicp_transfer_function: u8,
    force_baseline: bool,
) -> f32 {
    // Replicate logic from C++ QuantValsToDistance
    // Needs access to GLOBAL_SCALE_YCBCR, BASE_QUANT_MATRIX_YCBCR, etc.
    // from quant_constants
    let mut global_scale = GLOBAL_SCALE_YCBCR;
    if cicp_transfer_function == TRANSFER_FUNCTION_PQ {
        global_scale *= 0.4;
    } else if cicp_transfer_function == TRANSFER_FUNCTION_HLG {
        global_scale *= 0.5;
    }
    let quant_max = if force_baseline { 255 } else { 32767 };
    const K_DIST_MAX: f32 = 10000.0;
    let mut dist_min = 0.0f32;
    let mut dist_max = K_DIST_MAX;

    for c in 0..num_components {
        let quant_idx = comp_info[c].quant_tbl_no as usize;
        if quant_idx >= 4 || raw_quant_tables[quant_idx].is_none() {
             // Error or default? C++ doesn't check here, assumes valid tables.
             // Returning a default or large distance might be safer.
             return 10.0; 
        }
        let quantval = raw_quant_tables[quant_idx].as_ref().unwrap();
        
        // Determine base matrix based on color space assumption (C++ is implicit)
        // Assuming YCbCr for now, as C++ code does.
        let base_qm: &[f32];
         if quant_idx == 0 {
             base_qm = &BASE_QUANT_MATRIX_YCBCR[0..64];
         } else if quant_idx == 1 {
             // C++ uses Cr table (index 2) for slot 1 if not add_two_chroma
             // Assuming that case here for simplicity for now.
              base_qm = &BASE_QUANT_MATRIX_YCBCR[128..192]; 
         } else if quant_idx == 2 {
              base_qm = &BASE_QUANT_MATRIX_YCBCR[128..192]; 
         } else { continue; } // Skip invalid table index

        for k in 0..DCTSIZE2 {
            let mut dmin = 0.0f32;
            let mut dmax = K_DIST_MAX;
            let invq = 1.0 / base_qm[k] / global_scale;
            let qval_u16 = quantval[k];
            if qval_u16 > 1 {
                let scale_min = (qval_u16 as f32 - 0.5) * invq;
                dmin = scale_to_distance(scale_min, k);
            }
            if (qval_u16 as i32) < quant_max {
                let scale_max = (qval_u16 as f32 + 0.5) * invq;
                dmax = scale_to_distance(scale_max, k);
            }
            if dmin <= dist_max {
                dist_min = dist_min.max(dmin);
            }
            if dmax >= dist_min {
                dist_max = dist_max.min(dmax);
            }
        }
    }

    let distance: f32;
    if dist_min == 0.0 {
        distance = dist_max;
    } else if dist_max == K_DIST_MAX {
        distance = dist_min;
    } else {
        distance = 0.5 * (dist_min + dist_max);
    }
    distance
}

/// Checks if the sampling factors correspond to YUV420.
/// Corresponds to C++ IsYUV420.
pub(crate) fn is_yuv420(num_components: usize, comp_info: &[JpegliComponentParams], jpeg_color_space: JpegliColorSpace) -> bool {
    if jpeg_color_space != JpegliColorSpace::YCbCr || num_components < 3 {
        return false;
    }
    comp_info[0].h_samp_factor == 2 && comp_info[0].v_samp_factor == 2 &&
    comp_info[1].h_samp_factor == 1 && comp_info[1].v_samp_factor == 1 &&
    comp_info[2].h_samp_factor == 1 && comp_info[2].v_samp_factor == 1
}


// --- Ported Core Functions ---

/// Computes quantization table values based on distance/settings.
/// Corresponds to C++ SetQuantMatrices.
/// Accepts validated JpegliQuantParams.
pub(crate) fn set_quant_matrices(
    // Accept the params struct (mutable because it might update quant_tbl_no)
    params: &mut JpegliQuantParams,
) -> Result<[Option<[u16; 64]>; 4], &'static str> {
    let mut computed_tables: [Option<[u16; 64]>; 4] = [None; 4];
    let is_yuv420 = is_yuv420(params.num_components, &params.comp_params, params.jpegli_color_space);

    let mut global_scale: f32;
    let mut non_linear_scaling = true;
    let mut base_quant_matrix_slices: [&[f32]; 4] = [&[]; 4];
    let mut num_base_tables: usize;

    if params.xyb_mode && params.jpegli_color_space == JpegliColorSpace::RGB {
        global_scale = GLOBAL_SCALE_XYB;
        num_base_tables = 3;
        base_quant_matrix_slices[0] = &BASE_QUANT_MATRIX_XYB[0..64];
        base_quant_matrix_slices[1] = &BASE_QUANT_MATRIX_XYB[64..128];
        base_quant_matrix_slices[2] = &BASE_QUANT_MATRIX_XYB[128..192];
    } else if (params.jpegli_color_space == JpegliColorSpace::YCbCr ||
               params.jpegli_color_space == JpegliColorSpace::GRAYSCALE) &&
              !params.use_std_tables {
        global_scale = GLOBAL_SCALE_YCBCR;
        // Adjust scale based on transfer function and 420 status (matches C++)
        if params.cicp_transfer_function == TRANSFER_FUNCTION_PQ {
            global_scale *= 0.4;
        } else if params.cicp_transfer_function == TRANSFER_FUNCTION_HLG {
            global_scale *= 0.5;
        }
        if is_yuv420 {
            global_scale *= GLOBAL_SCALE_420;
        }
        
        if params.add_two_chroma_tables && params.num_components >= 3 {
             if params.comp_params.len() < 3 { return Err("Not enough components for add_two_chroma_tables"); }
            params.comp_params[2].quant_tbl_no = 2; // Assign distinct table index
            num_base_tables = 3;
            base_quant_matrix_slices[0] = &BASE_QUANT_MATRIX_YCBCR[0..64];
            base_quant_matrix_slices[1] = &BASE_QUANT_MATRIX_YCBCR[64..128]; // Cb
            base_quant_matrix_slices[2] = &BASE_QUANT_MATRIX_YCBCR[128..192]; // Cr
        } else {
            num_base_tables = if params.num_components == 1 { 1 } else { 2 };
            base_quant_matrix_slices[0] = &BASE_QUANT_MATRIX_YCBCR[0..64]; // Y
            if num_base_tables > 1 {
                 // C++ uses Cr table (index 2) for chroma slot 1
                base_quant_matrix_slices[1] = &BASE_QUANT_MATRIX_YCBCR[128..192]; // Cr
            }
        }
    } else { // Standard tables
        global_scale = 0.01; // Matches C++ scale for std tables
        non_linear_scaling = false;
        num_base_tables = if params.num_components == 1 { 1 } else { 2 };
        base_quant_matrix_slices[0] = &BASE_QUANT_MATRIX_STD[0..64]; // Luma
        if num_base_tables > 1 {
            base_quant_matrix_slices[1] = &BASE_QUANT_MATRIX_STD[64..128]; // Chroma
        }
    }

    let quant_max = if params.force_baseline { 255 } else { 32767 };

    for quant_idx in 0..num_base_tables {
        let base_qm = base_quant_matrix_slices[quant_idx];
        if base_qm.len() != 64 {
             return Err("Base quant matrix slice has incorrect length");
        }
        let mut current_table = [0u16; 64];
        for k in 0..DCTSIZE2 {
            let mut scale = global_scale;
            if non_linear_scaling {
                scale *= distance_to_scale(params.distance, k);
                if is_yuv420 && quant_idx > 0 { // Apply 420 rescale only to chroma tables
                    scale *= RESCALE_420[k];
                }
            } else {
                scale *= distance_to_linear_quality(params.distance);
            }
            let qval_f = scale * base_qm[k];
            let qval = qval_f.round() as i32;
            current_table[k] = qval.clamp(1, quant_max) as u16;
        }
        computed_tables[quant_idx] = Some(current_table);
    }

    Ok(computed_tables)
}

/// Computes derived quantizer state (multipliers, zero-bias tables).
/// Corresponds to C++ InitQuantizer.
/// Accepts validated JpegliQuantParams.
pub(crate) fn init_quantizer(
    raw_quant_tables: &[Option<[u16; 64]>; 4],
    // Accept the params struct 
    params: &JpegliQuantParams,
    pass: QuantPass, 
) -> Result<([[f32; 64]; MAX_COMPONENTS], // quant_mul
             [[f32; 64]; MAX_COMPONENTS], // zero_bias_mul
             [[f32; 64]; MAX_COMPONENTS]), // zero_bias_offset
            &'static str>
{
    // Initialize output arrays
    let mut quant_mul = [[0.0f32; 64]; MAX_COMPONENTS];
    let mut zero_bias_mul = [[0.0f32; 64]; MAX_COMPONENTS];
    let mut zero_bias_offset = [[0.0f32; 64]; MAX_COMPONENTS];

    // --- Compute quant_mul --- 
    for c in 0..params.num_components {
        if c >= MAX_COMPONENTS { break; } // Prevent out-of-bounds access
        let quant_idx = params.comp_params[c].quant_tbl_no as usize;
        if quant_idx >= 4 || raw_quant_tables[quant_idx].is_none() {
            return Err("Missing or invalid quantization table index for component");
        }
        let table = raw_quant_tables[quant_idx].as_ref().unwrap();
        for k in 0..DCTSIZE2 {
            let val = table[k];
            if val == 0 {
                return Err("Invalid quant value 0 in table");
            }
            match pass {
                QuantPass::NoSearch => {
                    quant_mul[c][k] = 8.0 / val as f32;
                }
                 // TODO: Implement other passes if needed
                _ => return Err("Search passes not implemented yet"),
            }
        }
    }

    // --- Compute Zero Bias Tables --- 
    if params.use_adaptive_quantization {
        // Initialize defaults
        for c in 0..params.num_components {
            if c >= MAX_COMPONENTS { break; } 
            for k in 1..DCTSIZE2 { // Skip DC (k=0)
                 zero_bias_mul[c][k] = 0.5;
                 zero_bias_offset[c][k] = 0.5;
            }
             // DC component is always 0 offset/mul
             zero_bias_mul[c][0] = 0.0;
             zero_bias_offset[c][0] = 0.0;
        }

        if params.jpegli_color_space == JpegliColorSpace::YCbCr {
             let distance = quant_vals_to_distance(
                 raw_quant_tables,
                 params.num_components,
                 &params.comp_params,
                 params.jpegli_color_space,
                 params.cicp_transfer_function,
                 params.force_baseline,
             );

            // Constants from C++ for interpolation
const K_DIST_HQ: f32 = 1.0;
const K_DIST_LQ: f32 = 3.0;
            let mix0 = ((distance - K_DIST_HQ) / (K_DIST_LQ - K_DIST_HQ)).clamp(0.0, 1.0);
            let mix1 = 1.0 - mix0;

            for c in 0..params.num_components {
                if c >= 3 { break; } // YCbCr only has 3 components with specific tables
                if c >= MAX_COMPONENTS { break; }
                let zb_mul_lq = &ZERO_BIAS_MUL_YCBCR_LQ[c * 64 .. (c + 1) * 64];
                let zb_mul_hq = &ZERO_BIAS_MUL_YCBCR_HQ[c * 64 .. (c + 1) * 64];
                
                for k in 0..DCTSIZE2 {
                     zero_bias_mul[c][k] = mix0 * zb_mul_lq[k] + mix1 * zb_mul_hq[k];
                     zero_bias_offset[c][k] = if k == 0 {
                         ZERO_BIAS_OFFSET_YCBCR_DC[c]
                     } else {
                         ZERO_BIAS_OFFSET_YCBCR_AC[c]
                     };
                }
            }
        }
        // Note: No specific zero-bias logic for XYB or Grayscale in C++ InitQuantizer
        // when adaptive quant is enabled, it uses the 0.5 default.

    } else if params.jpegli_color_space == JpegliColorSpace::YCbCr {
        // Adaptive quant disabled, but YCbCr - set offsets only
         for c in 0..params.num_components {
              if c >= 3 { break; }
              if c >= MAX_COMPONENTS { break; }
               for k in 0..DCTSIZE2 {
                 zero_bias_offset[c][k] = if k == 0 {
                     ZERO_BIAS_OFFSET_YCBCR_DC[c]
                 } else {
                     ZERO_BIAS_OFFSET_YCBCR_AC[c]
                 };
                 // zero_bias_mul remains 0.0 (default initialization)
            }
         }
    }
    // Else (Grayscale, RGB, CMYK, etc. without adaptive quant) -> zero_bias tables remain 0.0

    Ok((quant_mul, zero_bias_mul, zero_bias_offset))
}

// --- Quantizer State Struct ---
#[derive(Debug, Clone)] // Add Clone if needed
pub(crate) struct JpegliQuantizerState {
    // Raw DQT tables (u16, 0-32767 or 0-255)
    pub raw_quant_tables: [Option<[u16; DCTSIZE2]>; 4],
    // Derived tables used during coefficient quantization
    pub zero_bias_offsets: [[f32; DCTSIZE2]; MAX_COMPONENTS],
    pub zero_bias_multipliers: [[f32; DCTSIZE2]; MAX_COMPONENTS],
    // Might add quant_mul later if needed externally
}

impl JpegliQuantizerState {
    /// Creates a new quantizer state using validated parameters.
    pub(crate) fn new(
        params: &mut JpegliQuantParams, 
        pass: QuantPass,
    ) -> Result<Self, &'static str> {
        
        let computed_raw_tables = set_quant_matrices(params)?;

        let (_quant_mul, zb_mul, zb_offset) = init_quantizer(
            &computed_raw_tables, 
            params, // Pass the validated params struct
            pass
        )?;

        Ok(Self {
            raw_quant_tables: computed_raw_tables,
            zero_bias_offsets: zb_offset,
            zero_bias_multipliers: zb_mul,
        })
    }
}

// --- End Quantizer State Struct ---

// --- Quantizer Input Parameters Struct ---
#[derive(Debug, Clone)]
pub(crate) struct JpegliQuantParams {
    pub distance: f32,
    pub xyb_mode: bool,
    pub use_std_tables: bool,
    pub num_components: usize,
    pub comp_params: Vec<JpegliComponentParams>,
    pub jpegli_color_space: JpegliColorSpace,
    pub cicp_transfer_function: u8,
    pub force_baseline: bool,
    pub add_two_chroma_tables: bool,
    pub use_adaptive_quantization: bool,
    pub sampling_factor: SamplingFactor,
}

impl JpegliQuantParams {
    /// Tries to create validated low-level params from high-level config options.
    pub(crate) fn from_config(config: &JpegliQuantConfigOptions) -> Result<Self, &'static str> {
        // Apply defaults from JpegliQuantConfigOptions::default() if Option is None
        let xyb_mode = config.xyb_mode.unwrap_or(false);
        let use_std_tables = config.use_std_tables.unwrap_or(false);
        let use_adaptive_quantization = config.use_adaptive_quantization.unwrap_or(true);
        let force_baseline = config.force_baseline.unwrap_or(true);
        let cicp_transfer_function = config.cicp_transfer_function.unwrap_or(SimplifiedTransferCharacteristics::Default);

        // 1. Determine Distance (handles default and precedence)
        let distance = match (config.quality, config.distance) {
            (Some(q), None) => quality_to_distance(q.clamp(1, 100)), // Clamp quality
            (None, Some(d)) => d,
            (None, None) => 1.0, // Default distance 1.0
            (Some(_), Some(_)) => return Err("Cannot specify both quality and distance"),
        }.clamp(0.1, 25.0); // Clamp final distance

        // 2. Validate and Determine Sampling Factor
        let sampling_factor = match config.chroma_subsampling {
            Some(sf) => {
                // Check if the provided sampling factor is one of the allowed ones
                match sf {
                     Subsampling::YCbCr444 => {
                        SamplingFactor::F_1_1
                     }
                     Subsampling::YCbCr440 => {
                        SamplingFactor::F_1_2
                     }
                     Subsampling::YCbCr422 => {
                        SamplingFactor::F_2_1
                     }
                     Subsampling::YCbCr420 => {
                        SamplingFactor::F_2_2
                     }
                     _ => return Err("Unsupported chroma subsampling factor provided"),
                }
            },
            None => {
                // Apply default heuristic based on distance
                if distance >= 1.0 && config.jpeg_color_type.get_num_components() > 1 {
                    SamplingFactor::F_2_2 // 4:2:0
                } else {
                    SamplingFactor::F_1_1 // 4:4:4 or grayscale
                }
            }
        };
        
        // 3. Determine Jpegli Internal Color Space
        let jpegli_color_space = if xyb_mode {
            JpegliColorSpace::RGB 
        } else {
            match config.jpeg_color_type {
                JpegColorType::Luma => JpegliColorSpace::GRAYSCALE,
                JpegColorType::Ycbcr => JpegliColorSpace::YCbCr,
                 // Default assumption for other inputs needing processing
                _ => JpegliColorSpace::YCbCr, 
            }
        };
         if xyb_mode && jpegli_color_space != JpegliColorSpace::RGB {
             return Err("XYB mode requires RGB color space indication"); // Should be caught earlier ideally
         }

         let num_components = config.jpeg_color_type.get_num_components();
        // 4. Generate Initial Component Params
        let (max_h_samp, max_v_samp) = sampling_factor.get_sampling_factors();
        let mut comp_params: Vec<JpegliComponentParams> = Vec::with_capacity(num_components);
        match config.jpeg_color_type {
            JpegColorType::Luma => {
                if num_components != 1 { return Err("Luma requires 1 component"); }
                comp_params.push(JpegliComponentParams { h_samp_factor: 1, v_samp_factor: 1, quant_tbl_no: 0 });
            }
            JpegColorType::Ycbcr => { // Handle Ycbcr explicitly
                if num_components != 3 { return Err("YCbCr requires 3 components"); }
                comp_params.push(JpegliComponentParams { h_samp_factor: max_h_samp, v_samp_factor: max_v_samp, quant_tbl_no: 0 }); // Y
                comp_params.push(JpegliComponentParams { h_samp_factor: 1, v_samp_factor: 1, quant_tbl_no: 1 }); // Cb
                comp_params.push(JpegliComponentParams { h_samp_factor: 1, v_samp_factor: 1, quant_tbl_no: 1 }); // Cr
            }
             JpegColorType::Cmyk | JpegColorType::Ycck => {
                 return Err("CMYK/YCCK quantization setup not implemented yet");
            }
             // Handle other cases (like Rgb input, treat as YCbCr components)
             _ => {
                 if num_components != 3 { return Err("Assumed YCbCr (from RGB?) requires 3 components"); }
                 comp_params.push(JpegliComponentParams { h_samp_factor: max_h_samp, v_samp_factor: max_v_samp, quant_tbl_no: 0 }); // Y
                 comp_params.push(JpegliComponentParams { h_samp_factor: 1, v_samp_factor: 1, quant_tbl_no: 1 }); // Cb
                 comp_params.push(JpegliComponentParams { h_samp_factor: 1, v_samp_factor: 1, quant_tbl_no: 1 }); // Cr
            }
        }

        // 5. Determine add_two_chroma_tables (internal decision)
        let add_two_chroma_tables = false; // Keep false as default

        // 6. Perform final validation via try_new
        Self::try_new(
            distance,
            xyb_mode,
            use_std_tables,
            num_components,
            &comp_params,
            jpegli_color_space,
            cicp_transfer_function.to_int(),
            force_baseline,
            add_two_chroma_tables,
            use_adaptive_quantization,
            sampling_factor,
        )
    }

    /// Internal validation function (called by from_config).
    pub(crate) fn try_new(
        distance: f32,
        xyb_mode: bool,
        use_std_tables: bool,
        num_components: usize,
        comp_params: &[JpegliComponentParams], 
        jpegli_color_space: JpegliColorSpace,
        cicp_transfer_function: u8,
        force_baseline: bool,
        add_two_chroma_tables: bool,
        use_adaptive_quantization: bool,
        sampling_factor: SamplingFactor,
    ) -> Result<Self, &'static str> {
        // Validations moved from from_config or kept here
        if distance < 0.0 {
            return Err("Distance must be non-negative");
        }
        if num_components == 0 || num_components > MAX_COMPONENTS {
            return Err("Invalid number of components");
        }
        if comp_params.len() != num_components {
            return Err("Component params length mismatch");
        }
        if use_std_tables && xyb_mode {
             return Err("Cannot use standard tables with XYB mode");
        }
         if add_two_chroma_tables && num_components < 3 {
             return Err("Cannot add two chroma tables with less than 3 components");
        }
        if add_two_chroma_tables && jpegli_color_space != JpegliColorSpace::YCbCr {
             return Err("Adding two chroma tables is only supported for YCbCr");
        }
        for comp in comp_params {
             if comp.quant_tbl_no >= 4 { 
                 return Err("Invalid quantization table index");
             }
        }
        // Add any other necessary low-level validations

        Ok(Self {
            distance,
            xyb_mode,
            use_std_tables,
            num_components,
            comp_params: comp_params.to_vec(),
            jpegli_color_space,
            cicp_transfer_function,
            force_baseline,
            add_two_chroma_tables,
            use_adaptive_quantization,
            sampling_factor,
        })
    }
}

// --- End Quantizer Input Parameters Struct ---

// --- High-level Configuration Options (Now mostly optional) ---
#[derive(Debug, Clone)]
pub(crate) struct JpegliQuantConfigOptions {
    // Quality/Distance (mutually exclusive)
    pub quality: Option<u8>,
    pub distance: Option<f32>,
    // Flags mirroring cjpegli
    pub xyb_mode: Option<bool>,
    pub use_std_tables: Option<bool>,
    pub use_adaptive_quantization: Option<bool>,
    pub force_baseline: Option<bool>, 
    pub chroma_subsampling: Option<Subsampling>,
    pub jpeg_color_type: JpegColorType,
    pub cicp_transfer_function: Option<SimplifiedTransferCharacteristics>,
    // add_two_chroma_tables is likely an internal decision, not a direct user config
}

// Implement Default to provide cjpegli-like defaults
impl Default for JpegliQuantConfigOptions {
    fn default() -> Self {
        Self {
            quality: None, // Default quality (90) handled via distance default
            distance: None, // Default distance (1.0) applied if both are None
            xyb_mode: None,
            use_std_tables: None,
            use_adaptive_quantization: None,
            force_baseline: None,
            chroma_subsampling: None, // Default determined by distance later
            jpeg_color_type: JpegColorType::Cmyk, // Placeholder, must be set
            cicp_transfer_function: None, // Default to unknown/none
        }
    }
}
impl JpegliQuantConfigOptions {
    pub fn new_distance(
        distance: Option<f32>,
        color_type: JpegColorType,
        cicp_transfer_function: Option<SimplifiedTransferCharacteristics>) -> Self {
            Self {
                distance: distance,
                jpeg_color_type: color_type,
                cicp_transfer_function: cicp_transfer_function,
                ..Default::default()
            }
        
    }
}


