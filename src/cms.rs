use lcms2::{ColorSpaceSignature, Intent, PixelFormat, Profile, TagSignature, Transform, CIEXYZ, CIExyY, CIExyYTRIPLE, ToneCurve, ProfileClassSignature, Flags, Tag, TagTypeSignature, Context};
use std::ffi::c_void;
use std::sync::Mutex;
use alloc::vec::Vec;
use alloc::string::{String, ToString};

use crate::error::{EncodingError, EncoderResult};
use crate::tf::{self, ExtraTF};
use crate::ColorProfile; // Import ColorProfile

// TODO: Define equivalents for JxlColorEncoding if needed, or reuse lcms2 structures.

// Represents parsed information from an ICC profile relevant for our logic.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorEncodingInternal {
    pub color_space: ColorSpaceSignature, // e.g., RgbData, GrayData
    pub white_point: Option<CIExyY>,
    pub primaries: Option<CIExyYTRIPLE>, // RGB primaries
    pub transfer_function: TfType,
    pub is_cmyk: bool,
    pub rendering_intent: Intent,
    pub spot_color_names: Vec<String>,
    pub channels: u32, // Added channels field
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
    pub Red: CIExyY,
    pub Green: CIExyY,
    pub Blue: CIExyY,
}

// Equivalent of JxlColorProfile
#[derive(Debug, Clone)] // Added Clone derive
pub struct ColorProfile {
    pub icc: Vec<u8>,
    pub internal: Option<ColorEncodingInternal>,
}

impl ColorProfile {
    pub fn new(icc_data: Vec<u8>) -> EncoderResult<Self> {
        let internal = set_fields_from_icc(&icc_data)?;
        Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }
    pub fn srgb() -> EncoderResult<Self> {
        let internal = ColorEncodingInternal {
            channels: 3,
            color_space: ColorSpaceSignature::RgbData,
            white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }), // Wrapped in Some
            primaries: Some(CIExyYTRIPLE {
                Red: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                Blue: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
            }),
            transfer_function: TfType::SRGB, // Use correct field name
            is_cmyk: false,
            rendering_intent: Intent::Perceptual,
            spot_color_names: Vec::new(),
        };
        let profile = Profile::new_srgb();
        let icc_data = profile.save_to_icc_bytes()
            .map_err(|e| EncodingError::CmsError(format!("Failed to save sRGB profile: {}", e)))?;
        Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }
    pub fn linear_srgb() -> EncoderResult<Self> {
        let mut profile = Self::srgb()?;
        if let Some(ref mut internal) = profile.internal {
            internal.transfer_function = TfType::Linear; // Use correct field name
        }
        let linear_profile = Profile::new_rgb_context(
                Context::new(),
                &CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 },
                &CIExyYTRIPLE {
                    Red: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                    Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                    Blue: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
                },
                &[&ToneCurve::new(1.0)]
            )
            .map_err(|e| EncodingError::CmsError(format!("Failed to create linear sRGB profile: {}", e)))?;

        profile.icc = linear_profile.save_to_icc_bytes()
            .map_err(|e| EncodingError::CmsError(format!("Failed to save linear sRGB profile: {}", e)))?;

        Ok(profile)
    }
    pub fn gray_gamma22() -> EncoderResult<Self> {
        let internal = ColorEncodingInternal {
            channels: 1,
            color_space: ColorSpaceSignature::GrayData,
            white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }), // Wrapped in Some
            primaries: None,
            transfer_function: TfType::Gamma(2200), // Use correct field name
            is_cmyk: false,
            rendering_intent: Intent::Perceptual,
            spot_color_names: Vec::new(),
        };
        let gray_profile = Profile::new_gray_context(
                Context::new(),
                &CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 },
                &ToneCurve::new(2.2)
            )
             .map_err(|e| EncodingError::CmsError(format!("Failed to create gray profile: {}", e)))?;

        let icc_data = gray_profile.save_to_icc_bytes()
             .map_err(|e| EncodingError::CmsError(format!("Failed to save gray profile: {}", e)))?;

         Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }

    pub fn internal(&mut self) -> EncoderResult<&ColorEncodingInternal> {
        if self.internal.is_none() {
            self.internal = Some(set_fields_from_icc(&self.icc)?);
        }
        Ok(self.internal.as_ref().unwrap())
    }

    pub fn icc(&self) -> &[u8] {
        &self.icc
    }
}

// Manages the actual LCMS transform data. Equivalent of C++ JxlCms struct.
pub struct JxlCms {
    transform: Option<Mutex<Transform<f32, f32>>>,
    apply_hlg_ootf: bool,
    hlg_ootf_luminances: Option<[f32; 3]>,

    pub channels_src: usize, // Made public
    pub channels_dst: usize, // Made public

    src_storage: Mutex<Vec<f32>>,
    dst_storage: Mutex<Vec<f32>>,

    pub intensity_target: f32, // Made public
    pub skip_lcms: bool, // Made public
    pub preprocess: ExtraTF, // Made public
    pub postprocess: ExtraTF, // Made public
}

fn get_pixel_format(channels: u32, is_float: bool) -> Option<PixelFormat> {
    match (channels, is_float) {
        (1, true) => Some(PixelFormat::GRAY_FLT),
        (1, false) => Some(PixelFormat::GRAY_8),
        (3, true) => Some(PixelFormat::RGB_FLT),
        (3, false) => Some(PixelFormat::RGB_8),
        (4, true) => Some(PixelFormat::CMYK_FLT),
        (4, false) => Some(PixelFormat::CMYK_8),
        _ => None,
    }
}

impl JxlCms {
    pub fn new(
        input_profile: &ColorProfile,
        output_profile: &ColorProfile,
        intensity_target: f32,
    ) -> EncoderResult<Self> {
        let internal_in = input_profile.internal.as_ref()
            .ok_or_else(|| EncodingError::CmsError("Input profile internal data missing".to_string()))?;
        let internal_out = output_profile.internal.as_ref()
            .ok_or_else(|| EncodingError::CmsError("Output profile internal data missing".to_string()))?;

        let channels_src = internal_in.channels as usize;
        let channels_dst = internal_out.channels as usize;

        let preprocess = tf::get_extra_tf(internal_in.transfer_function, channels_src as u32, false);
        let postprocess = tf::get_extra_tf(internal_out.transfer_function, channels_dst as u32, true);

        let skip_lcms = input_profile.icc == output_profile.icc ||
                        (internal_in.transfer_function == internal_out.transfer_function &&
                         internal_in.color_space == internal_out.color_space &&
                         channels_src == channels_dst &&
                         preprocess == ExtraTF::kNone &&
                         postprocess == ExtraTF::kNone);

        let mut apply_hlg_ootf = false;
        let mut hlg_ootf_luminances = None;
        if internal_in.transfer_function == TfType::HLG && internal_out.transfer_function != TfType::HLG {
            apply_hlg_ootf = true;
            log::warn!("HLG OOTF needed but not fully implemented.");
        } else if internal_in.transfer_function != TfType::HLG && internal_out.transfer_function == TfType::HLG {
            apply_hlg_ootf = true;
             log::warn!("Inverse HLG OOTF needed but not fully implemented.");
        }

        let lcms_transform = if !skip_lcms {
            let profile_src = Profile::new_icc(&input_profile.icc)
                .map_err(|e| EncodingError::CmsError(format!("Failed to create source profile: {}", e.to_string())))?;
            let profile_dst = Profile::new_icc(&output_profile.icc)
                .map_err(|e| EncodingError::CmsError(format!("Failed to create destination profile: {}", e.to_string())))?;

            let input_format = get_pixel_format(channels_src as u32, true)
                .ok_or_else(|| EncodingError::CmsError(format!("Unsupported source channel count: {}", channels_src)))?;
            let output_format = get_pixel_format(channels_dst as u32, true)
                .ok_or_else(|| EncodingError::CmsError(format!("Unsupported destination channel count: {}", channels_dst)))?;

            let intent = internal_in.rendering_intent;
            let flags = Flags::default();

            let transform = Transform::new_flags(&profile_src, input_format, &profile_dst, output_format, intent, flags)
                .map_err(|e| EncodingError::CmsError(format!("Failed to create LCMS transform: {}", e.to_string())))?;
            Some(Mutex::new(transform))
        } else {
            None
        };

        let src_storage = Mutex::new(Vec::new());
        let dst_storage = Mutex::new(Vec::new());

        Ok(JxlCms {
            transform: lcms_transform,
            apply_hlg_ootf,
            hlg_ootf_luminances,
            channels_src,
            channels_dst,
            src_storage,
            dst_storage,
            intensity_target,
            skip_lcms,
            preprocess,
            postprocess,
        })
    }

    pub fn run_transform(&self, input_slice: &[f32], output_slice: &mut [f32], num_pixels: usize) -> EncoderResult<()> {
        let expected_input_len = num_pixels * self.channels_src;
        let expected_output_len = num_pixels * self.channels_dst;

        if input_slice.len() < expected_input_len || output_slice.len() < expected_output_len {
            return Err(EncodingError::CmsError(format!(
                "Buffer size mismatch: input {} vs {}, output {} vs {}",
                input_slice.len(), expected_input_len,
                output_slice.len(), expected_output_len
            ).to_string()));
        }

        let mut src_guard = self.src_storage.lock().unwrap();
        let mut dst_guard = self.dst_storage.lock().unwrap();

        // Ensure storage is large enough
        if src_guard.len() < expected_input_len {
            src_guard.resize(expected_input_len, 0.0);
        }
        if dst_guard.len() < expected_output_len {
            dst_guard.resize(expected_output_len, 0.0);
        }

        let src_buffer = &mut src_guard[..expected_input_len];
        src_buffer.copy_from_slice(&input_slice[..expected_input_len]);

        // Apply pre-processing TF
        tf::before_transform(self.preprocess, self.intensity_target, src_buffer)?;

        // Run LCMS transform if needed
        if let Some(transform_mutex) = &self.transform {
            let mut transform = transform_mutex.lock().unwrap();
            let dst_buffer = &mut dst_guard[..expected_output_len];
            transform.transform_pixels(src_buffer, dst_buffer);
        } else if self.skip_lcms {
            // If skipping LCMS but channels match, copy src to dst buffer for post-processing
            if self.channels_src == self.channels_dst {
                 dst_guard[..expected_output_len].copy_from_slice(src_buffer);
            } else {
                // Should not happen if skip_lcms is true unless TF handles channel change?
                return Err(EncodingError::CmsError("Channel mismatch when skipping LCMS".to_string()));
            }
        } else {
             return Err(EncodingError::CmsError("CMS transform not available".to_string()));
        }

        // Apply post-processing TF (operates in-place on dst_buffer)
        let dst_buffer = &mut dst_guard[..expected_output_len];
        tf::after_transform(self.postprocess, self.intensity_target, dst_buffer)?;

        // Apply HLG OOTF if needed (operates in-place)
        if self.apply_hlg_ootf {
            // TODO: Implement HLG OOTF logic
            log::warn!("HLG OOTF apply step not implemented.");
            // tf::apply_hlg_ootf(dst_buffer, self.hlg_ootf_luminances, self.intensity_target);
        }

        // Copy result to output slice
        output_slice[..expected_output_len].copy_from_slice(dst_buffer);

        Ok(())
    }
}

// Tries to parse ICC profile to extract relevant fields.
pub fn set_fields_from_icc(icc_data: &[u8]) -> EncoderResult<ColorEncodingInternal> {
    let profile = Profile::new_icc(icc_data)
        .map_err(|e| EncodingError::CmsError(format!("Failed to parse ICC profile: {}", e.to_string())))?;

    let class = profile.profile_class_signature();
    let color_space_sig = profile.color_space();
    let pcs = profile.pcs(); // Profile Connection Space

    let channels = match color_space_sig {
        ColorSpaceSignature::GrayData => 1,
        ColorSpaceSignature::RgbData => 3,
        ColorSpaceSignature::CmykData => 4,
        _ => return Err(EncodingError::CmsError(format!("Unsupported ICC color space: {:?}", color_space_sig)))
    };

    let mut encoding = ColorEncodingInternal {
        channels,
        color_space: color_space_sig,
        white_point: None,
        primaries: None,
        transfer_function: TfType::Unknown,
        is_cmyk: color_space_sig == ColorSpaceSignature::CmykData,
        rendering_intent: profile.rendering_intent(),
        spot_color_names: Vec::new(), // TODO: Extract spot colors if needed
    };

    // Only extract detailed fields for display/input/output/color space classes
    if class == ProfileClassSignature::DisplayInputDevice || class == ProfileClassSignature::InputDevice || class == ProfileClassSignature::OutputDevice || class == ProfileClassSignature::ColorSpaceConversion {
        if let Some(wp_tag_data) = profile.tag_data(TagSignature::MediaWhitePointTag).ok() {
            if let Ok(xyz) = Tag::read_xyz(&wp_tag_data) {
                 if let Some(xyy) = CIExyY::from_xyz(&xyz) {
                     encoding.white_point = Some(xyy);
                 } else {
                      log::warn!("ICC WP XYZ to xyY conversion failed, using D50 (PCS default)");
                     // Default to D50 if conversion fails (PCS white point)
                     encoding.white_point = Some(CIExyY { x: 0.3457, y: 0.3585, Y: 1.0 });
                 }
            } else {
                 log::warn!("Failed to read XYZ from MediaWhitePointTag, using D50");
                 encoding.white_point = Some(CIExyY { x: 0.3457, y: 0.3585, Y: 1.0 });
            }
        } else {
            log::warn!("MediaWhitePointTag not found, using D50");
             encoding.white_point = Some(CIExyY { x: 0.3457, y: 0.3585, Y: 1.0 });
        }

        if color_space_sig == ColorSpaceSignature::RgbData {
            let r_tag_data = profile.tag_data(TagSignature::RedColorantTag).ok();
            let g_tag_data = profile.tag_data(TagSignature::GreenColorantTag).ok();
            let b_tag_data = profile.tag_data(TagSignature::BlueColorantTag).ok();

            if let (Some(r_data), Some(g_data), Some(b_data)) = (r_tag_data, g_tag_data, b_tag_data) {
                 if let (Ok(r_xyz), Ok(g_xyz), Ok(b_xyz)) = (Tag::read_xyz(&r_data), Tag::read_xyz(&g_data), Tag::read_xyz(&b_data)) {
                    if let (Some(r_xyy), Some(g_xyy), Some(b_xyy)) = (CIExyY::from_xyz(&r_xyz), CIExyY::from_xyz(&g_xyz), CIExyY::from_xyz(&b_xyz)) {
                         encoding.primaries = Some(CIExyYTRIPLE {
                             Red: r_xyy,
                             Green: g_xyy,
                             Blue: b_xyy,
                         });
                    } else {
                        log::warn!("Failed to convert primary XYZ to xyY");
                    }
                 } else {
                      log::warn!("Failed to read XYZ from primary colorant tags");
                 }
            }
        }
    }

    // Determine Transfer Function (TRC)
    let trc_tag_sig = match color_space_sig {
         ColorSpaceSignature::GrayData => TagSignature::GrayTRCTag,
         ColorSpaceSignature::RgbData => TagSignature::GreenTRCTag, // Green is often representative for RGB
         _ => TagSignature::Unknown, // No TRC for CMYK etc.
    };

    if trc_tag_sig != TagSignature::Unknown {
        if let Some(trc_tag_data) = profile.tag_data(trc_tag_sig).ok() {
             if let Ok(curve) = Tag::read_curve(&trc_tag_data) {
                 if curve.is_linear() {
                     encoding.transfer_function = TfType::Linear;
                 } else {
                      if curve.type_signature() == TagTypeSignature::ParametricCurve {
                         let params = curve.parametric_params();
                         if curve.parametric_type() == 4 { // sRGB
                             encoding.transfer_function = TfType::SRGB;
                         } else if curve.parametric_type() == 5 { // PQ approx
                             encoding.transfer_function = TfType::PQ;
                              log::warn!("Detected parametric curve type 5, assuming PQ.");
                         } else if curve.parametric_type() == 6 { // HLG approx
                             encoding.transfer_function = TfType::HLG;
                             log::warn!("Detected parametric curve type 6, assuming HLG.");
                         } else if curve.parametric_type() >= 1 && curve.parametric_type() <= 3 && !params.is_empty() {
                              let gamma = params[0];
                              if gamma > 0.0 {
                                  encoding.transfer_function = TfType::Gamma((gamma * 1000.0).round() as u32);
                              }
                         } else {
                              log::warn!("Unknown parametric curve type: {}", curve.parametric_type());
                             encoding.transfer_function = TfType::Unknown;
                         }
                      } else {
                         // Estimate from table if not parametric
                         let gamma = curve.estimated_gamma();
                         if (gamma - 1.8).abs() < 0.01 {
                              log::warn!("Gamma near 1.8 found from table, treating as sRGB (could be BT.709). Parametric check recommended.");
                             encoding.transfer_function = TfType::SRGB;
                         } else if gamma > 0.0 {
                             encoding.transfer_function = TfType::Gamma((gamma * 1000.0).round() as u32);
                         }
                      }
                 }
             } else {
                  log::warn!("Failed to read curve from TRC tag ({:?})", trc_tag_sig);
             }
         } else {
              log::warn!("TRC tag ({:?}) not found", trc_tag_sig);
             // Default based on colorspace if TRC missing
             if color_space_sig == ColorSpaceSignature::RgbData { encoding.transfer_function = TfType::SRGB; }
             else if color_space_sig == ColorSpaceSignature::GrayData { encoding.transfer_function = TfType::Gamma(2200); }
         }
    } else if color_space_sig == ColorSpaceSignature::CmykData {
        // Assume linear for CMYK if no specific handling
        encoding.transfer_function = TfType::Linear;
    }

    Ok(encoding)
}

pub fn cms_init(
    input_profile: &ColorProfile,
    output_profile: &ColorProfile,
    intensity_target: f32,
) -> EncoderResult<Box<JxlCms>> {
    let cms_state = JxlCms::new(input_profile, output_profile, intensity_target)?;
    Ok(Box::new(cms_state))
}

pub fn cms_run(
    cms_state: &JxlCms,
    input_buffer: &[f32],
    output_buffer: &mut [f32],
    num_pixels: usize,
) -> EncoderResult<()> {
    cms_state.run_transform(input_buffer, output_buffer, num_pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcms2::Profile;
    use approx::assert_relative_eq;

    fn dummy_internal(channels: u32, tf: TfType, cs: ColorSpaceSignature, is_cmyk: bool) -> ColorEncodingInternal {
        ColorEncodingInternal {
            channels,
            color_space: cs,
            white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }),
            primaries: if cs == ColorSpaceSignature::RgbData { Some(CIExyYTRIPLE {
                Red: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                Blue: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
            }) } else { None },
            transfer_function: tf,
            is_cmyk,
            rendering_intent: Intent::Perceptual,
            spot_color_names: Vec::new(),
        }
    }

    fn dummy_profile(profile_type: &str) -> EncoderResult<ColorProfile> {
        match profile_type {
            "srgb" => ColorProfile::srgb(),
            "linear_srgb" => ColorProfile::linear_srgb(),
            "gray_gamma22" => ColorProfile::gray_gamma22(),
            _ => Err(EncodingError::CmsError("Unknown dummy profile type".to_string()))
        }
    }

    #[test]
    fn test_srgb_profile_creation() {
        let profile = ColorProfile::srgb().unwrap();
        assert!(!profile.icc.is_empty());
        let internal = profile.internal.as_ref().unwrap();
        assert_eq!(internal.channels, 3);
        assert_eq!(internal.transfer_function, TfType::SRGB);
        assert_eq!(internal.color_space, ColorSpaceSignature::RgbData);
        assert_relative_eq!(internal.white_point.x, 0.3127, epsilon = 1e-4);
        assert_relative_eq!(internal.white_point.y, 0.3290, epsilon = 1e-4);
    }

     #[test]
    fn test_linear_srgb_profile_creation() {
        let profile = ColorProfile::linear_srgb().unwrap();
        assert!(!profile.icc.is_empty());
        let internal = profile.internal.as_ref().unwrap();
        assert_eq!(internal.channels, 3);
        assert_eq!(internal.transfer_function, TfType::Linear);
        assert_eq!(internal.color_space, ColorSpaceSignature::RgbData);
    }

    #[test]
    fn test_gray_gamma22_profile_creation() {
         let profile = ColorProfile::gray_gamma22().unwrap();
         assert!(!profile.icc.is_empty());
         let internal = profile.internal.as_ref().unwrap();
         assert_eq!(internal.channels, 1);
         assert_eq!(internal.transfer_function, TfType::Gamma(2200));
         assert_eq!(internal.color_space, ColorSpaceSignature::GrayData);
     }

    #[test]
    fn test_set_fields_from_known_icc() {
        let srgb_profile = Profile::new_srgb();
        let icc_data = srgb_profile.save_to_icc_bytes().unwrap();
        let internal = set_fields_from_icc(&icc_data).unwrap();

        assert_eq!(internal.channels, 3);
        assert!(matches!(internal.transfer_function, TfType::Gamma(_) | TfType::SRGB));
        assert_eq!(internal.color_space, ColorSpaceSignature::RgbData);
        assert_relative_eq!(internal.white_point.x, 0.3127, epsilon = 1e-4);
        assert_relative_eq!(internal.white_point.y, 0.3290, epsilon = 1e-4);
        assert!(internal.primaries.is_some());
        assert_eq!(internal.rendering_intent, Intent::Perceptual);
    }

    #[test]
    fn test_cms_init_basic() {
        let srgb = dummy_profile("srgb").unwrap();
        let linear = dummy_profile("linear_srgb").unwrap();
        let cms = cms_init(&srgb, &linear, 255.0).unwrap();

        assert!(!cms.skip_lcms);
        assert!(cms.transform.is_some());
        assert_eq!(cms.preprocess, ExtraTF::kSRGB);
        assert_eq!(cms.postprocess, ExtraTF::kNone);
        assert_eq!(cms.channels_src, 3);
        assert_eq!(cms.channels_dst, 3);
    }

     #[test]
    fn test_cms_init_skip() {
        let srgb1 = dummy_profile("srgb").unwrap();
        let srgb2 = dummy_profile("srgb").unwrap();
        let cms = cms_init(&srgb1, &srgb2, 255.0).unwrap();

        let linear1 = dummy_profile("linear_srgb").unwrap();
        let linear2 = dummy_profile("linear_srgb").unwrap();
        let cms_linear_skip = cms_init(&linear1, &linear2, 255.0).unwrap();
        assert!(cms_linear_skip.skip_lcms);
        assert!(cms_linear_skip.transform.is_none());
        assert_eq!(cms_linear_skip.preprocess, ExtraTF::kNone);
        assert_eq!(cms_linear_skip.postprocess, ExtraTF::kNone);

        let cms_srgb_noskip = cms_init(&srgb1, &srgb2, 255.0).unwrap();
         assert!(!cms_srgb_noskip.skip_lcms);
         assert_eq!(cms_srgb_noskip.preprocess, ExtraTF::kSRGB);
         assert_eq!(cms_srgb_noskip.postprocess, ExtraTF::kSRGB);

         let linear_target = dummy_profile("linear_srgb").unwrap();
         let cms_srgb_linear = cms_init(&srgb1, &linear_target, 255.0).unwrap();
         assert!(!cms_srgb_linear.skip_lcms);
         assert_eq!(cms_srgb_linear.preprocess, ExtraTF::kSRGB);
         assert_eq!(cms_srgb_linear.postprocess, ExtraTF::kNone);
    }

    #[test]
    fn test_cms_run_transform_rgb_to_linear() {
        let srgb = dummy_profile("srgb").unwrap();
        let linear = dummy_profile("linear_srgb").unwrap();
        let cms = cms_init(&srgb, &linear, 255.0).unwrap();

        let input: [f32; 3] = [0.73535696, 0.53775376, 0.8813725 ];
        let mut output: [f32; 3] = [0.0; 3];
        let num_pixels = 1;

        cms_run(&cms, &input, &mut output, num_pixels).unwrap();

        assert_relative_eq!(output[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(output[1], 0.25, epsilon = 1e-3);
        assert_relative_eq!(output[2], 0.75, epsilon = 1e-3);
    }

     #[test]
    fn test_cms_run_transform_skip() {
        let linear1 = dummy_profile("linear_srgb").unwrap();
        let linear2 = dummy_profile("linear_srgb").unwrap();
        let cms = cms_init(&linear1, &linear2, 255.0).unwrap();
        assert!(cms.skip_lcms);

        let input: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut output: [f32; 6] = [0.0; 6];
        let num_pixels = 2;

        cms_run(&cms, &input, &mut output, num_pixels).unwrap();

        assert_eq!(input, output);
    }
} 