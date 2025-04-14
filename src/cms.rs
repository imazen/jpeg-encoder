use lcms2::{Profile, Transform, Intent, PixelFormat, ColorSpaceSignature, ProfileClassSignature, TagSignature, ToneCurve, CIExyY, CIEXYZ, TagTypeSignature};
use std::ffi::c_void;
use std::sync::Mutex;
use alloc::vec::Vec;
use alloc::string::String;

use crate::error::{EncoderError, EncoderResult};
use crate::tf::{self, ExtraTF};

// TODO: Define equivalents for JxlColorEncoding if needed, or reuse lcms2 structures.

// Represents parsed information from an ICC profile relevant for our logic.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorEncodingInternal {
    pub color_space: ColorSpaceSignature, // e.g., SigRgbData, SigGrayData
    pub white_point: Option<CIExyY>,
    pub primaries: Option<CIExyYTRIPLE>, // RGB primaries
    pub transfer_function: TfType,
    pub is_cmyk: bool,
}

// Simplified representation of transfer function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfType {
    Linear,
    SRGB,
    PQ,
    HLG,
    Gamma(u32), // Store gamma as scaled integer, e.g., 2.2 -> 2200
    Unknown,
}

// Helper struct for RGB Primaries
#[derive(Debug, Clone, PartialEq)]
pub struct CIExyYTRIPLE {
    pub r: CIExyY,
    pub g: CIExyY,
    pub b: CIExyY,
}

// Equivalent of JxlColorProfile
#[derive(Debug)]
pub struct ColorProfile {
    pub icc: Vec<u8>,
    // Parsed internal representation
    pub internal: Option<ColorEncodingInternal>,
    pub num_channels: usize,
}

impl ColorProfile {
    pub fn new(icc_data: Vec<u8>) -> EncoderResult<Self> {
        let (internal, num_channels, _) = set_fields_from_icc(&icc_data)?;
        Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
            num_channels,
        })
    }
    // Create a standard sRGB profile placeholder
    pub fn srgb() -> Self {
        // We don't strictly *need* the ICC bytes for sRGB if we handle it specially,
        // but having a placeholder can be useful.
        // Using lcms2::Profile::new_srgb() might be better if we always use lcms.
        let internal = ColorEncodingInternal {
            color_space: ColorSpaceSignature::SigRgbData,
            white_point: Some(CIExyY{x: 0.3127, y: 0.3290, Y: 1.0}), // D65
            primaries: Some(CIExyYTRIPLE {
                 r: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                 g: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                 b: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
            }),
            transfer_function: TfType::SRGB,
            is_cmyk: false,
        };
        ColorProfile {
            icc: Vec::new(), // No actual ICC bytes needed for this path usually
            internal: Some(internal),
            num_channels: 3,
        }
    }
     // Create a standard linear sRGB profile placeholder
    pub fn linear_srgb() -> Self {
        let mut profile = Self::srgb();
        if let Some(ref mut internal) = profile.internal {
            internal.transfer_function = TfType::Linear;
        }
        profile
    }

     // Create a standard Gray profile placeholder (Gamma 2.2)
    pub fn gray_gamma22() -> Self {
        let internal = ColorEncodingInternal {
            color_space: ColorSpaceSignature::SigGrayData,
            white_point: Some(CIExyY{x: 0.3127, y: 0.3290, Y: 1.0}), // D65
            primaries: None,
            transfer_function: TfType::Gamma(2200), // Gamma 2.2
            is_cmyk: false,
        };
         ColorProfile {
            icc: Vec::new(),
            internal: Some(internal),
            num_channels: 1,
        }
    }
}

// Manages the actual LCMS transform data. Equivalent of C++ JxlCms struct.
pub struct JxlCms {
    transform: Option<Transform<f32, f32>>,
    apply_hlg_ootf: bool,
    hlg_ootf_luminances: Option<[f32; 3]>, // Y component of primaries for OOTF

    channels_src: usize,
    channels_dst: usize,

    // Buffers for preprocessing/postprocessing.
    // Using Mutex assuming `run_transform` might be called from multiple threads.
    src_storage: Mutex<Vec<Vec<f32>>>, // One buffer per thread
    dst_storage: Mutex<Vec<Vec<f32>>>, // One buffer per thread

    intensity_target: f32,
    skip_lcms: bool,
    preprocess: ExtraTF,
    postprocess: ExtraTF,
}

impl JxlCms {
    pub fn new(
        input_profile: &ColorProfile,
        output_profile: &ColorProfile, // Target profile *before* things like YCbCr conversion
        intensity_target: f32,
        num_threads: usize,
        pixels_per_thread: usize,
    ) -> EncoderResult<Self> {
        let internal_in = input_profile.internal.as_ref()
            .ok_or(EncoderError::CmsError("Input profile internal data missing".to_string()))?;
        let internal_out = output_profile.internal.as_ref()
            .ok_or(EncoderError::CmsError("Output profile internal data missing".to_string()))?;

        let channels_src = input_profile.num_channels;
        let channels_dst = output_profile.num_channels;

        // --- Determine Pre/Post processing steps (ExtraTF) ---
        let preprocess = match internal_in.transfer_function {
            TfType::PQ => ExtraTF::kPQ,
            TfType::HLG => ExtraTF::kHLG,
            TfType::SRGB => ExtraTF::kSRGB,
            _ => ExtraTF::kNone,
        };
        let postprocess = match internal_out.transfer_function {
            TfType::PQ => ExtraTF::kPQ,
            TfType::HLG => ExtraTF::kHLG,
            TfType::SRGB => ExtraTF::kSRGB,
            _ => ExtraTF::kNone,
        };

        // --- Determine if LCMS transform can be skipped ---
        // Basic check: if internal representations are identical and no pre/post processing needed.
        // C++ code has more complex checks (e.g., specific profile types).
        let skip_lcms = preprocess == ExtraTF::kNone &&
                        postprocess == ExtraTF::kNone &&
                        internal_in == internal_out &&
                        channels_src == channels_dst;

        // --- HLG OOTF Handling ---
        // Apply HLG OOTF/Inverse OOTF if converting between HLG and non-HLG with mismatched intents.
        // Simplified logic compared to C++ jxl_cms.cc DetermineExtraTF.
        let mut apply_hlg_ootf = false;
        let mut hlg_ootf_luminances = None;
        if preprocess == ExtraTF::kHLG && postprocess != ExtraTF::kHLG {
             apply_hlg_ootf = true; // Forward OOTF
        } else if preprocess != ExtraTF::kHLG && postprocess == ExtraTF::kHLG {
             apply_hlg_ootf = true; // Inverse OOTF (applied in after_transform)
        }
        if apply_hlg_ootf {
             if let Some(primaries) = &internal_in.primaries {
                 // Approximate Y from primaries (needs conversion from xyY to XYZ first)
                 // For simplicity, using placeholder. Real implementation needs xyY->XYZ matrix.
                 hlg_ootf_luminances = Some([primaries.r.Y, primaries.g.Y, primaries.b.Y]); // Placeholder!
             } else if channels_src == 1 { // Grayscale HLG
                 hlg_ootf_luminances = Some([1.0, 1.0, 1.0]);
             }
             // If luminances couldn't be determined, OOTF can't be applied.
             if hlg_ootf_luminances.is_none() { apply_hlg_ootf = false; }
        }

        // --- LCMS Transform Creation (if not skipped) ---
        let mut lcms_transform = None;
        if !skip_lcms {
            let profile_src = Profile::new_icc(&input_profile.icc)
                .map_err(|_| EncoderError::CmsError("Failed to create source profile".to_string()))?;
            let profile_dst = Profile::new_icc(&output_profile.icc)
                .map_err(|_| EncoderError::CmsError("Failed to create destination profile".to_string()))?;

            // Determine PixelFormat based on num_channels. Uses TYPE_FLOAT.
            // LCMS uses specific flags in the u32 format specifier.
            // T_FLOAT = 1 << 7 = 128
            // CHANNELS_SH(n) = n
            // e.g., TYPE_RGB_FLT = (CHANNELS_SH(3) | T_FLOAT) = 3 | 128 = 131
            fn make_flt_format(channels: usize) -> u32 {
                const T_FLOAT: u32 = 1 << 7;
                (channels as u32) | T_FLOAT
            }

            let input_format_u32 = make_flt_format(channels_src);
            let output_format_u32 = make_flt_format(channels_dst);

            // Flags: cmsFLAGS_COPY_ALPHA might be needed if alpha exists?
            // cmsFLAGS_NOOPTIMIZE is used in C++ sometimes.
            let flags = 0;
            let intent = Intent::Perceptual; // TODO: Get from profile/config if needed

            let transform = Transform::new_flags(&profile_src, input_format_u32, &profile_dst, output_format_u32, intent, flags)
                .map_err(|_| EncoderError::CmsError("Failed to create LCMS transform".to_string()))?;
            lcms_transform = Some(transform);
        }

        let src_buf_size = if preprocess != ExtraTF::kNone { pixels_per_thread * channels_src } else { 0 };
        let dst_buf_size = if postprocess != ExtraTF::kNone { pixels_per_thread * channels_dst } else { 0 };

        Ok(JxlCms {
            transform: lcms_transform,
            apply_hlg_ootf,
            hlg_ootf_luminances,
            channels_src,
            channels_dst,
            src_storage: Mutex::new(vec![vec![0.0; src_buf_size]; num_threads]),
            dst_storage: Mutex::new(vec![vec![0.0; dst_buf_size]; num_threads]), // Not used yet, but for after_transform
            intensity_target,
            skip_lcms,
            preprocess,
            postprocess,
        })
    }

    pub fn run_transform(
        &self,
        thread_id: usize,
        input_buffer: &[f32],
        output_buffer: &mut [f32],
        num_pixels: usize,
    ) -> EncoderResult<()> {
        let expected_input_len = num_pixels * self.channels_src;
        let expected_output_len = num_pixels * self.channels_dst;
        if input_buffer.len() < expected_input_len || output_buffer.len() < expected_output_len {
             return Err(EncoderError::CmsError(format!(
                 "Buffer size mismatch. Input: {}/{}, Output: {}/{}",
                 input_buffer.len(), expected_input_len,
                 output_buffer.len(), expected_output_len
             )));
        }

        // Slice to actual size being processed
        let input_slice = &input_buffer[..expected_input_len];
        let output_slice = &mut output_buffer[..expected_output_len];

        // --- Preprocessing --- 
        let mut preprocessed_input_guard;
        let lcms_input: &[f32] = if self.preprocess != ExtraTF::kNone {
            preprocessed_input_guard = self.src_storage.lock().unwrap(); // Lock mutex
            let temp_buf = &mut preprocessed_input_guard[thread_id];
            if temp_buf.len() < expected_input_len {
                 return Err(EncoderError::CmsError("Preprocessing buffer too small".to_string()));
            }
            let temp_slice = &mut temp_buf[..expected_input_len];
            tf::before_transform(self.preprocess, self.intensity_target, input_slice, temp_slice)?;
             // TODO: Apply forward HLG OOTF here if self.apply_hlg_ootf and preprocess is HLG
            temp_slice
        } else {
            input_slice
        };

        // --- CMYK Adjustment (if needed) --- 
        let mut cmyk_adjusted_input: Vec<f32>; // Needs allocation if used
        let final_lcms_input = if self.channels_src == 4 && !self.skip_lcms {
            // LCMS expects CMYK as 0=white, 100=max ink. Input is likely 0=black, 1=white (or 0..255).
            // Assuming input is [0,1] range for C, M, Y, K.
            // If input_buffer was [0,255], it should have been scaled to [0,1] earlier.
            cmyk_adjusted_input = lcms_input.iter().map(|&x| (1.0 - x) * 100.0).collect();
            &cmyk_adjusted_input
        } else {
            lcms_input
        };

        // --- LCMS Transform --- 
        if let Some(ref transform) = self.transform {
             transform.transform_pixels(final_lcms_input, output_slice, num_pixels);
        } else if self.skip_lcms {
            // Copy input to output if buffers differ and transform is skipped
            if final_lcms_input.as_ptr() != output_slice.as_ptr() {
                 output_slice.copy_from_slice(&final_lcms_input[..expected_output_len]); // Assumes channels_src == channels_dst
            }
        } else {
             return Err(EncoderError::CmsError("CMS transform not initialized and not skipped".to_string()));
        }

        // --- Postprocessing --- 
        if self.postprocess != ExtraTF::kNone {
            // TODO: Apply inverse HLG OOTF here if self.apply_hlg_ootf and postprocess is HLG
            tf::after_transform(self.postprocess, self.intensity_target, output_slice)?;
        }

        Ok(())
    }
}

// --- Public Interface Functions (Mirroring JxlCmsInterface but idiomatic Rust) ---

// Parses an ICC profile using LCMS2 to extract relevant info.
pub fn set_fields_from_icc(icc_data: &[u8]) -> EncoderResult<(ColorEncodingInternal, usize, bool)> {
    if icc_data.is_empty() {
        return Err(EncoderError::CmsError("ICC data is empty".to_string()));
    }
    let profile = Profile::new_icc(icc_data)
        .map_err(|e| EncoderError::CmsError(format!("Failed to parse ICC profile: {}", e)))?;

    let color_space_sig = profile.color_space();
    let pcs = profile.pcs();
    let class = profile.profile_class();

    let num_channels = lcms2::color_space_channels(color_space_sig);
    let is_cmyk = color_space_sig == ColorSpaceSignature::SigCmykData;

    let mut encoding = ColorEncodingInternal {
        color_space: color_space_sig,
        white_point: None,
        primaries: None,
        transfer_function: TfType::Unknown,
        is_cmyk,
    };

    // Try reading tags (ignore errors, keep defaults if tags missing/invalid)
    if class == ProfileClassSignature::SigDisplayClass || class == ProfileClassSignature::SigInputClass || class == ProfileClassSignature::SigOutputClass || class == ProfileClassSignature::SigColorSpaceClass {
        if let Some(wp_tag) = profile.read_tag(TagSignature::SigMediaWhitePointTag) {
            if let TagTypeSignature::SigXYZType = wp_tag.get_type() {
                 if let Some(xyz) = CIEXYZ::read(wp_tag.as_bytes()) {
                     encoding.white_point = Some(CIExyY::from(*xyz));
                 }
            }
        }
        if color_space_sig == ColorSpaceSignature::SigRgbData {
            let r_tag = profile.read_tag(TagSignature::SigRedColorantTag);
            let g_tag = profile.read_tag(TagSignature::SigGreenColorantTag);
            let b_tag = profile.read_tag(TagSignature::SigBlueColorantTag);
            if let (Some(r), Some(g), Some(b)) = (r_tag, g_tag, b_tag) {
                 if r.get_type() == TagTypeSignature::SigXYZType &&
                    g.get_type() == TagTypeSignature::SigXYZType &&
                    b.get_type() == TagTypeSignature::SigXYZType
                 {
                    if let (Some(rxyz), Some(gxyz), Some(bxyz)) = (
                        CIEXYZ::read(r.as_bytes()),
                        CIEXYZ::read(g.as_bytes()),
                        CIEXYZ::read(b.as_bytes())
                    ) {
                        encoding.primaries = Some(CIExyYTRIPLE {
                            r: CIExyY::from(*rxyz),
                            g: CIExyY::from(*gxyz),
                            b: CIExyY::from(*bxyz),
                        });
                    }
                 }
            }
            // Determine Transfer Function for RGB
            // Check Green TRC first as per jxl_cms.cc logic (or Gray for grayscale)
            let trc_tag_sig = if color_space_sig == ColorSpaceSignature::SigGrayData {
                 TagSignature::SigGrayTRCTag
            } else {
                 TagSignature::SigGreenTRCTag
            };
            if let Some(trc_tag) = profile.read_tag(trc_tag_sig) {
                 match trc_tag.get_type() {
                    TagTypeSignature::SigCurveType => {
                         if let Ok(curve) = ToneCurve::read(trc_tag.as_bytes()) {
                             // Check for known curves (simplified check)
                             if curve.estimated_gamma() > 2.1 && curve.estimated_gamma() < 2.3 {
                                 encoding.transfer_function = TfType::Gamma(2200);
                             } else if curve.estimated_gamma() > 0.99 && curve.estimated_gamma() < 1.01 {
                                 // Check if it's truly linear
                                 if curve.is_linear() { // Requires lcms2 5.0+
                                     encoding.transfer_function = TfType::Linear;
                                 } else {
                                     encoding.transfer_function = TfType::Gamma(1000);
                                 }
                             }
                             // Cannot reliably detect PQ/HLG/sRGB from SigCurveType alone
                             // Jxl uses cicp tag or specific curve parameters for that.
                         }
                    },
                    TagTypeSignature::SigParametricCurveType => {
                        // TODO: Parse parametric curve type and parameters to identify sRGB, PQ, HLG, Gamma
                        // See cmsReadParametricCurve() and types in lcms2.h
                         encoding.transfer_function = TfType::Unknown; // Placeholder
                    },
                    _ => { encoding.transfer_function = TfType::Unknown; }
                 }
            }
            // TODO: Check cicp tag for PQ/HLG/sRGB overrides (see MaybeCreateICCCICPTag)
        }
    }

    Ok((encoding, num_channels, is_cmyk))
}

/// Initializes a CMS transform state.
pub fn cms_init(
    input_profile: &ColorProfile,
    output_profile: &ColorProfile,
    intensity_target: f32,
    num_threads: usize,
    pixels_per_thread: usize,
) -> EncoderResult<Box<JxlCms>> {
    let cms_state = JxlCms::new(input_profile, output_profile, intensity_target, num_threads, pixels_per_thread)?;
    Ok(Box::new(cms_state))
}

/// Runs the color transform.
pub fn cms_run(
    cms_state: &JxlCms,
    thread_id: usize,
    input_buffer: &[f32],
    output_buffer: &mut [f32],
    num_pixels: usize,
) -> EncoderResult<()> {
    cms_state.run_transform(thread_id, input_buffer, output_buffer, num_pixels)
}

// --- Tests --- //
#[cfg(test)]
mod tests {
    use super::*;
    use lcms2::Profile;

    // Helper to create a dummy ColorProfile for testing
    fn dummy_profile(num_channels: usize, tf: TfType, cs: ColorSpaceSignature, is_cmyk: bool) -> ColorProfile {
         let internal = ColorEncodingInternal {
            color_space: cs,
            white_point: Some(CIExyY{x: 0.3127, y: 0.3290, Y: 1.0}), // D65
            primaries: if cs == ColorSpaceSignature::SigRgbData { Some(CIExyYTRIPLE {
                 r: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                 g: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                 b: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
            })} else { None },
            transfer_function: tf,
            is_cmyk,
        };
        // Create minimal valid ICC bytes using lcms for Profile::new_icc to succeed
        let icc_bytes = match cs {
            ColorSpaceSignature::SigRgbData => Profile::new_srgb().icc_bytes().unwrap(),
            ColorSpaceSignature::SigGrayData => Profile::new_gray(None, None).icc_bytes().unwrap(), // No easy way to set gamma in new_gray
            ColorSpaceSignature::SigCmykData => Profile::new_lab4(None).icc_bytes().unwrap(), // Hack: use Lab4 as placeholder for CMYK profile creation
            _ => vec![0u8; 128], // Invalid placeholder
        };
         ColorProfile {
            icc: icc_bytes,
            internal: Some(internal),
            num_channels,
        }
    }

    #[test]
    fn test_cms_init_skip() {
        let srgb = dummy_profile(3, TfType::SRGB, ColorSpaceSignature::SigRgbData, false);
        let cms_state = cms_init(&srgb, &srgb, 255.0, 1, 16).unwrap();
        assert!(cms_state.skip_lcms);
        assert_eq!(cms_state.preprocess, ExtraTF::kSRGB);
        assert_eq!(cms_state.postprocess, ExtraTF::kSRGB);
    }

    #[test]
    fn test_cms_init_no_skip_diff_tf() {
        let srgb = dummy_profile(3, TfType::SRGB, ColorSpaceSignature::SigRgbData, false);
        let linear = dummy_profile(3, TfType::Linear, ColorSpaceSignature::SigRgbData, false);
        let cms_state = cms_init(&srgb, &linear, 255.0, 1, 16).unwrap();
        assert!(!cms_state.skip_lcms);
        assert_eq!(cms_state.preprocess, ExtraTF::kSRGB);
        assert_eq!(cms_state.postprocess, ExtraTF::kNone);
    }

     #[test]
    fn test_cms_init_no_skip_diff_channels() {
        let gray = dummy_profile(1, TfType::Gamma(2200), ColorSpaceSignature::SigGrayData, false);
        let srgb = dummy_profile(3, TfType::SRGB, ColorSpaceSignature::SigRgbData, false);
        let cms_state = cms_init(&gray, &srgb, 255.0, 1, 16).unwrap();
        assert!(!cms_state.skip_lcms);
        assert_eq!(cms_state.preprocess, ExtraTF::kNone); // Gamma 2.2 isn't an ExtraTF
        assert_eq!(cms_state.postprocess, ExtraTF::kSRGB);
    }

    #[test]
    fn test_set_fields_basic_srgb() {
        let srgb_profile = Profile::new_srgb();
        let icc_data = srgb_profile.icc_bytes().unwrap();
        let (internal, channels, is_cmyk) = set_fields_from_icc(&icc_data).unwrap();

        assert_eq!(channels, 3);
        assert!(!is_cmyk);
        assert_eq!(internal.color_space, ColorSpaceSignature::SigRgbData);
        assert!(internal.white_point.is_some());
        assert!(internal.primaries.is_some());
        // Basic lcms sRGB profile might be detected as Gamma 2.2 curve, not specifically sRGB parametric
        assert!(internal.transfer_function == TfType::Gamma(2200) || internal.transfer_function == TfType::SRGB);
    }

     #[test]
     fn test_cms_run_identity() {
        let srgb = dummy_profile(3, TfType::SRGB, ColorSpaceSignature::SigRgbData, false);
        let cms_state = cms_init(&srgb, &srgb, 255.0, 1, 4).unwrap();
        assert!(cms_state.skip_lcms);

        let input = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0]; // 4 pixels
        let mut output = [0.0f32; 12];

        cms_run(&cms_state, 0, &input, &mut output, 4).unwrap();

        // Pre/Post should handle the SRGB conversion even if LCMS is skipped
        let mut expected = [0.0f32; 12];
        let mut temp_linear = [0.0f32; 12];
        tf::before_transform(ExtraTF::kSRGB, 255.0, &input, &mut temp_linear).unwrap();
        expected.copy_from_slice(&temp_linear);
        tf::after_transform(ExtraTF::kSRGB, 255.0, &mut expected).unwrap();

        for i in 0..12 {
             assert!((output[i] - expected[i]).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, output[i], expected[i]);
        }
     }
} 