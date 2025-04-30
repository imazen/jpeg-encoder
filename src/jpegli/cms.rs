use lcms2::{ColorSpaceSignature, Intent, PixelFormat, Profile, TagSignature, Transform, CIEXYZ, CIExyY, CIExyYTRIPLE, ToneCurve, ProfileClassSignature, Flags, Tag};
use std::eprintln;
use std::ffi::c_void;
use std::sync::{Mutex, Arc};
use alloc::vec::Vec;
use crate::error::EncodingError;
use crate::jpegli::tf;
use core::convert::TryInto;

// Represents parsed information from an ICC profile relevant for our logic.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorEncodingInternal {
    pub color_space: ColorSpaceSignature, // e.g., RgbData, GrayData
    pub white_point: Option<CIExyY>,
    pub primaries: Option<lcms2::CIExyYTRIPLE>,
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

// Equivalent of JxlColorProfile
#[derive(Debug, Clone)] // Added Clone derive
pub struct ColorProfile {
    pub icc: Vec<u8>,
    pub internal: Option<ColorEncodingInternal>,
}

impl ColorProfile {
    pub fn new(icc_data: Vec<u8>) -> Result<Self, EncodingError> {
        let internal = set_fields_from_icc(&icc_data)?;
        Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }
    pub fn srgb() -> Result<Self, EncodingError> {
        let mut profile = Profile::new_srgb();
        profile.set_device_class(ProfileClassSignature::DisplayClass);
        let icc_data = profile.icc()
            .map_err(|e| EncodingError::CmsError(format!("Failed to get sRGB profile ICC: {}", e)))?.to_vec();
        let internal = ColorEncodingInternal {
            channels: 3,
            color_space: ColorSpaceSignature::RgbData,
            white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }),
            primaries: Some(lcms2::CIExyYTRIPLE {
                Red: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
                Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
                Blue: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
            }),
            transfer_function: TfType::SRGB,
            is_cmyk: false,
            rendering_intent: Intent::Perceptual,
            spot_color_names: Vec::new(),
        };
        Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }
  // Create a linear sRGB profile manually, similar to C++ cmsCreateRGBProfile approach.
pub fn linear_srgb() -> Result<Self, EncodingError> {
    let white_point = CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 };
    let primaries = CIExyYTRIPLE {
        Red:   CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
        Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
        Blue:  CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
    };
    // Create a linear tone curve (gamma 1.0)
    let linear_gamma = ToneCurve::new(1.0);
    let linear_curves = [&linear_gamma, &linear_gamma, &linear_gamma];

    // Create the profile using new_rgb
    let mut profile = Profile::new_rgb(&white_point, &primaries, &linear_curves)
        .map_err(|e| EncodingError::CmsError(format!("Failed to create linear sRGB profile using new_rgb: {}", e)))?;
    
    profile.set_device_class(ProfileClassSignature::DisplayClass);

    let icc_data = profile.icc()
        .map_err(|e| EncodingError::CmsError(format!("Failed to get ICC data for manually created linear sRGB: {}", e)))?;

    let internal = ColorEncodingInternal {
        channels: 3,
        color_space: ColorSpaceSignature::RgbData,
        white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }),
        primaries: Some(lcms2::CIExyYTRIPLE {
            Red: CIExyY { x: 0.64, y: 0.33, Y: 1.0 },
            Green: CIExyY { x: 0.30, y: 0.60, Y: 1.0 },
            Blue: CIExyY { x: 0.15, y: 0.06, Y: 1.0 },
        }),
        transfer_function: TfType::Linear, 
        is_cmyk: false,
        rendering_intent: Intent::Perceptual,
        spot_color_names: Vec::new(),
    };
    Ok(ColorProfile {
        icc: icc_data.to_vec(),
        internal: Some(internal),
    })
    }
    pub fn gray_gamma22() -> Result<Self, EncodingError> {
        let internal = ColorEncodingInternal {
            channels: 1,
            color_space: ColorSpaceSignature::GrayData,
            white_point: Some(CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 }),
            primaries: None,
            transfer_function: TfType::Gamma(2200),
            is_cmyk: false,
            rendering_intent: Intent::Perceptual,
            spot_color_names: Vec::new(),
        };
        let gray_profile = Profile::new_gray(
                &CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 },
                &ToneCurve::new(2.2)
            )
             .map_err(|e| EncodingError::CmsError(format!("Failed to create gray profile: {}", e)))?;

        let icc_data = gray_profile.icc()
             .map_err(|e| EncodingError::CmsError(format!("Failed to save gray profile: {}", e)))?.to_vec();

         Ok(ColorProfile {
            icc: icc_data,
            internal: Some(internal),
        })
    }

    pub fn internal(&mut self) -> Result<&ColorEncodingInternal, EncodingError> {
        if self.internal.is_none() {
            self.internal = Some(set_fields_from_icc(&self.icc)?);
        }
        self.internal.as_ref().ok_or_else(|| EncodingError::CmsError("Failed to get internal profile data".to_string()))
    }

    pub fn icc(&self) -> &[u8] {
        &self.icc
    }
}

// Manages the actual LCMS transform data. Equivalent of C++ JxlCms struct.
pub struct JxlCms {
    input_profile: Option<Arc<ColorProfile>>,
    output_profile: Option<Arc<ColorProfile>>,
    intensity_target: f32,
    transform: Option<Mutex<Transform<f32, f32>>>,
    channels_src: usize,
    channels_dst: usize,
    skip_lcms: bool,
    preprocess: tf::ExtraTF,
    postprocess: tf::ExtraTF,
    apply_hlg_ootf: bool,
    hlg_ootf_luminances: Option<[f32; 3]>, 
    src_storage: Mutex<Vec<f32>>,
    dst_storage: Mutex<Vec<f32>>,
    hlg_ootf_instance: Option<tf::hlg::HlgOotf>,
}

impl JxlCms {
    pub fn new(
        input_profile: &ColorProfile,
        output_profile: &ColorProfile,
        intensity_target: f32,
    ) -> Result<Self, EncodingError> {
        let internal_in = input_profile.internal.as_ref()
            .ok_or_else(|| EncodingError::CmsError("Input profile internal data missing".to_string()))?;
        let internal_out = output_profile.internal.as_ref()
            .ok_or_else(|| EncodingError::CmsError("Output profile internal data missing".to_string()))?;

        let channels_src = internal_in.channels as usize;
        let channels_dst = internal_out.channels as usize;

        let preprocess = tf::get_extra_tf(internal_in.transfer_function, channels_src as u32, false);
        let postprocess = tf::get_extra_tf(internal_out.transfer_function, channels_dst as u32, true);

        let skip_lcms = (input_profile.icc == output_profile.icc) && 
                        (preprocess == tf::ExtraTF::kNone) && 
                        (postprocess == tf::ExtraTF::kNone);

        let mut apply_hlg_ootf = false;
        let mut hlg_ootf_luminances = None;
        let mut hlg_ootf_instance: Option<tf::hlg::HlgOotf> = None;

        if internal_in.transfer_function == TfType::HLG && internal_out.transfer_function != TfType::HLG {
            apply_hlg_ootf = true;
            hlg_ootf_instance = Some(tf::hlg::HlgOotf::from_scene_light(intensity_target));
            eprintln!("HLG OOTF needed (Forward).");
        } else if internal_in.transfer_function != TfType::HLG && internal_out.transfer_function == TfType::HLG {
            apply_hlg_ootf = true;
            hlg_ootf_instance = Some(tf::hlg::HlgOotf::to_scene_light(intensity_target));
            eprintln!("HLG OOTF needed (Inverse).");
        }

        let lcms_transform = if !skip_lcms {
            let profile_src = Profile::new_icc(&input_profile.icc)
                .map_err(|e| EncodingError::CmsError(format!("Failed to create source profile: {}", e.to_string())))?;
            let profile_dst = Profile::new_icc(&output_profile.icc)
                .map_err(|e| EncodingError::CmsError(format!("Failed to create destination profile: {}", e.to_string())))?;

            let input_format = match channels_src {
                1 => PixelFormat::GRAY_FLT,
                3 => PixelFormat::RGB_FLT,
                4 => PixelFormat::CMYK_FLT, 
                _ => return Err(EncodingError::CmsError(format!("Unsupported source channel count: {}", channels_src))),
            };
            let output_format = match channels_dst {
                1 => PixelFormat::GRAY_FLT,
                3 => PixelFormat::RGB_FLT,
                4 => PixelFormat::CMYK_FLT, 
                _ => return Err(EncodingError::CmsError(format!("Unsupported destination channel count: {}", channels_dst))),
            };

            let intent = Intent::RelativeColorimetric; 
            let flags = Flags::NO_OPTIMIZE | Flags::BLACKPOINT_COMPENSATION; 

            let transform = Transform::new_flags(
                &profile_src,
                input_format,
                &profile_dst,
                output_format,
                intent,
                flags,
            )
            .map_err(|e| EncodingError::CmsError(format!("Failed to create LCMS transform: {}", e.to_string())))?;
            Some(Mutex::new(transform))
        } else {
            None
        };

        let src_storage = Mutex::new(Vec::new());
        let dst_storage = Mutex::new(Vec::new());

        Ok(JxlCms {
            input_profile: None,
            output_profile: None,
            intensity_target,
            transform: lcms_transform,
            channels_src,
            channels_dst,
            skip_lcms,
            preprocess,
            postprocess,
            apply_hlg_ootf,
            hlg_ootf_luminances,
            src_storage,
            dst_storage,
            hlg_ootf_instance,
        })
    }

    pub fn run_transform(&self, input_slice: &[f32], output_slice: &mut [f32], num_pixels: usize) -> Result<(), EncodingError> {
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

        if src_guard.len() < expected_input_len {
            src_guard.resize(expected_input_len, 0.0);
        }
        if dst_guard.len() < expected_output_len {
            dst_guard.resize(expected_output_len, 0.0);
        }

        let src_buffer = &mut src_guard[..expected_input_len];
        src_buffer.copy_from_slice(&input_slice[..expected_input_len]);

        tf::before_transform(self.preprocess, self.intensity_target, src_buffer)?;

        if let Some(transform_mutex) = &self.transform {
            let mut transform = transform_mutex.lock().unwrap();
            let dst_buffer = &mut dst_guard[..expected_output_len];
            transform.transform_pixels(src_buffer, dst_buffer);
        } else if self.skip_lcms {
            if self.channels_src == self.channels_dst {
                 dst_guard[..expected_output_len].copy_from_slice(src_buffer);
            } else {
                return Err(EncodingError::CmsError("Channel mismatch when skipping LCMS".to_string()));
            }
        } else {
             return Err(EncodingError::CmsError("CMS transform not available".to_string()));
        }

        if self.apply_hlg_ootf && self.postprocess == tf::ExtraTF::kHLG {
            if let Some(ootf) = &self.hlg_ootf_instance {
                if self.channels_src == 3 {
                     let (r, rest) = src_buffer.split_at_mut(num_pixels);
                     let (g, b) = rest.split_at_mut(num_pixels);
                     ootf.apply(r, g, b, num_pixels);
                } else {
                    eprintln!("Warning: Cannot apply inverse HLG OOTF to non-RGB data ({} channels)", self.channels_src);
                }
            }
        }

        if self.apply_hlg_ootf && self.preprocess == tf::ExtraTF::kHLG {
             if let Some(ootf) = &self.hlg_ootf_instance {
                 if self.channels_dst == 3 {
                    let dst_buffer = &mut dst_guard[..expected_output_len];
                    let (r, rest) = dst_buffer.split_at_mut(num_pixels);
                    let (g, b) = rest.split_at_mut(num_pixels);
                    ootf.apply(r, g, b, num_pixels);
                 } else {
                    eprintln!("Warning: Cannot apply forward HLG OOTF to non-RGB data ({} channels)", self.channels_dst);
                 }
             }
        }

        tf::after_transform(self.postprocess, self.intensity_target, &mut dst_guard[..expected_output_len])?;

        output_slice[..expected_output_len].copy_from_slice(&dst_guard[..expected_output_len]);

        Ok(())
    }
}

pub fn set_fields_from_icc(icc_data: &[u8]) -> Result<ColorEncodingInternal, EncodingError> {
    let profile = Profile::new_icc(icc_data)
        .map_err(|e| EncodingError::CmsError(format!("Failed to parse ICC profile: {}", e.to_string())))?;

    let class = profile.device_class();
    let color_space_sig = profile.color_space();
    let _pcs = profile.pcs(); 

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
        rendering_intent: profile.header_rendering_intent(),
        spot_color_names: Vec::new(), 
    };

    if class == ProfileClassSignature::DisplayClass
    || class == ProfileClassSignature::InputClass
    || class == ProfileClassSignature::OutputClass
    || class == ProfileClassSignature::ColorSpaceClass
    {
        let wp_tag = profile.read_tag(TagSignature::MediaWhitePointTag);
        if let Tag::CIEXYZ(xyz_ref) = wp_tag {
            let xyz = *xyz_ref;
             if let Ok(xyy) = CIExyY::try_from(xyz) {
                 encoding.white_point = Some(xyy);
             } else {
                  eprintln!("ICC WP XYZ to xyY conversion failed, using D50 (PCS default)");
                 encoding.white_point = Some(CIExyY { x: 0.3457, y: 0.3585, Y: 1.0 });
             }
        } else {
             eprintln!("MediaWhitePointTag not found or not a CIEXYZ tag, using D50");
             encoding.white_point = Some(CIExyY { x: 0.3457, y: 0.3585, Y: 1.0 });
        }
        if color_space_sig == ColorSpaceSignature::RgbData {
            let r_tag = profile.read_tag(TagSignature::RedColorantTag);
            let g_tag = profile.read_tag(TagSignature::GreenColorantTag);
            let b_tag = profile.read_tag(TagSignature::BlueColorantTag);

            if let (Tag::CIEXYZ(r_xyz_ref), Tag::CIEXYZ(g_xyz_ref), Tag::CIEXYZ(b_xyz_ref)) = (r_tag, g_tag, b_tag) {
                let r_xyz = *r_xyz_ref;
                let g_xyz = *g_xyz_ref;
                let b_xyz = *b_xyz_ref;
                 if let (Ok(r_xyy), Ok(g_xyy), Ok(b_xyy)) = (CIExyY::try_from(r_xyz), CIExyY::try_from(g_xyz), CIExyY::try_from(b_xyz)) {
                     encoding.primaries = Some(lcms2::CIExyYTRIPLE {
                         Red: r_xyy,
                         Green: g_xyy,
                         Blue: b_xyy,
                     });
                 } else {
                    eprintln!("Failed to convert primary XYZ to xyY");
                 }
            } else {
                 eprintln!("One or more primary colorant tags not found or not CIEXYZ tags.");
            }
        }
    }

    let trc_tag_sig = match color_space_sig {
         ColorSpaceSignature::GrayData => Some(TagSignature::GrayTRCTag),
         ColorSpaceSignature::RgbData => Some(TagSignature::GreenTRCTag), 
         _ => None,
    };

    if let Some(sig) = trc_tag_sig {
        let trc_tag = profile.read_tag(sig);
        if let Tag::ToneCurve(curve_ref) = trc_tag {
             if curve_ref.is_linear() {
                 encoding.transfer_function = TfType::Linear;
             } else {
                 let parametric_type = curve_ref.parametric_type();
                 if parametric_type == 4 {
                     encoding.transfer_function = TfType::SRGB;
                 } else if parametric_type == 5 {
                     encoding.transfer_function = TfType::PQ;
                     eprintln!("Detected parametric curve type 5, assuming PQ.");
                 } else if parametric_type == 6 {
                     encoding.transfer_function = TfType::HLG;
                     eprintln!("Detected parametric curve type 6, assuming HLG.");
                 } else if let Some(gamma) = curve_ref.estimated_gamma(0.001) {
                     if gamma > 0.0 {
                         encoding.transfer_function = TfType::Gamma((gamma * 1000.0).round() as u32);
                         if parametric_type >= 1 && parametric_type <= 3 {
                              // It was likely a parametric gamma curve
                         } else {
                              eprintln!("Estimated gamma {} from non-parametric curve", gamma);
                         }
                     } else {
                          eprintln!("Estimated gamma is not positive: {}", gamma);
                          encoding.transfer_function = TfType::Unknown;
                     }
                 } else {
                      eprintln!("Unknown parametric curve type: {} and gamma estimation failed", parametric_type);
                     encoding.transfer_function = TfType::Unknown;
                 }
             }
        } else {
            eprintln!("TRC tag ({:?}) not found or not a ToneCurve variant.", sig);
            if color_space_sig == ColorSpaceSignature::RgbData { encoding.transfer_function = TfType::SRGB; }
            else if color_space_sig == ColorSpaceSignature::GrayData { encoding.transfer_function = TfType::Gamma(2200); }
            else { encoding.transfer_function = TfType::Unknown; }
        }
    } else if color_space_sig == ColorSpaceSignature::CmykData {
        encoding.transfer_function = TfType::Linear;
    }

    Ok(encoding)
}

pub fn cms_init(
    input_profile: &ColorProfile,
    output_profile: &ColorProfile,
    intensity_target: f32,
) -> Result<Box<JxlCms>, EncodingError> {
    let cms_state = JxlCms::new(input_profile, output_profile, intensity_target)?;
    Ok(Box::new(cms_state))
}

pub fn cms_run(
    cms_state: &JxlCms,
    input_buffer: &[f32],
    output_buffer: &mut [f32],
    num_pixels: usize,
) -> Result<(), EncodingError> {
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
            primaries: if cs == ColorSpaceSignature::RgbData { Some(lcms2::CIExyYTRIPLE {
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

    fn dummy_profile(profile_type: &str) -> Result<ColorProfile, EncodingError> {
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
        assert!(internal.white_point.is_some());
        assert_relative_eq!(internal.white_point.as_ref().unwrap().x, 0.3127, epsilon = 1e-4);
        assert_relative_eq!(internal.white_point.as_ref().unwrap().y, 0.3290, epsilon = 1e-4);
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
        let icc_data = srgb_profile.icc().unwrap().to_vec();
        let internal = set_fields_from_icc(&icc_data).unwrap();

        assert_eq!(internal.channels, 3);
        assert!(matches!(internal.transfer_function, TfType::Gamma(_) | TfType::SRGB));
        assert_eq!(internal.color_space, ColorSpaceSignature::RgbData);
        assert!(internal.white_point.is_some());
        assert_relative_eq!(internal.white_point.as_ref().unwrap().x, 0.3457, epsilon = 1e-4);
        assert_relative_eq!(internal.white_point.as_ref().unwrap().y, 0.3585, epsilon = 1e-4);
        assert!(internal.primaries.is_some());
        assert_eq!(internal.rendering_intent, Intent::Perceptual);
    }

    #[test]
    fn test_cms_init_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Load sRGB profile using include_bytes!
        const SRGB_ICC_DATA: &[u8] = include_bytes!("tinysrgb.icc");
        let srgb_profile_from_bytes = ColorProfile::new(SRGB_ICC_DATA.to_vec()).expect("Failed to create sRGB profile from bytes");

        // Use the embedded-bytes profile for both input and output
        let cms = JxlCms::new(&srgb_profile_from_bytes, &srgb_profile_from_bytes, 100.0)?;
        assert!(cms.skip_lcms, "CMS transform should be skipped for identical file profiles");
        Ok(())
    }

     #[test]
    fn test_cms_init_skip() -> Result<(), Box<dyn std::error::Error>> {
        let srgb1 = ColorProfile::srgb()?; // Keep using generated for this specific skip test
        let srgb2 = ColorProfile::srgb()?;
        println!("ICC data identical for srgb1 and srgb2? {}", srgb1.icc() == srgb2.icc());
        let mut cms = JxlCms::new(&srgb1, &srgb2, 100.0)?;
        assert!(cms.skip_lcms, "CMS transform should be skipped for identical generated sRGB profiles");

        // Test with linear sRGB (should also skip if identical)
        let linear1 = ColorProfile::linear_srgb()?; // Use actual linear sRGB
        let linear2 = ColorProfile::linear_srgb()?;
        println!("ICC data identical for linear1 and linear2? {}", linear1.icc() == linear2.icc());
        let mut cms_linear = JxlCms::new(&linear1, &linear2, 100.0)?;
        assert!(cms_linear.skip_lcms, "CMS transform should be skipped for identical generated linear sRGB profiles");

        // Test with different profiles (should not skip)
        let mut cms_diff = JxlCms::new(&srgb1, &linear1, 100.0)?;
        assert!(!cms_diff.skip_lcms, "CMS transform should NOT be skipped for different profiles");
        Ok(())
    }

    #[test]
    fn test_cms_run_transform_rgb_to_linear() -> Result<(), Box<dyn std::error::Error>> {
        // Load sRGB profile using include_bytes!
        const SRGB_ICC_DATA: &[u8] = include_bytes!("tinysrgb.icc");
        let srgb_profile_from_bytes = ColorProfile::new(SRGB_ICC_DATA.to_vec()).expect("Failed to create sRGB profile from bytes");

        // Use generated linear sRGB for output
        let linear_srgb_profile = ColorProfile::linear_srgb()?;

        // Create CMS instance with embedded sRGB -> generated linear sRGB
        let cms = JxlCms::new(&srgb_profile_from_bytes, &linear_srgb_profile, 100.0)?;
        assert!(!cms.skip_lcms, "Transform should not be skipped");

        // Prepare dummy input/output data (e.g., 1 pixel)
        let input_rgb: Vec<f32> = vec![0.5, 0.5, 0.5]; // Example RGB values
        let mut output_linear: Vec<f32> = vec![0.0; 3]; // Output buffer

        // Run the transform using the internal lcms2::Transform
        // Check if transform exists before unwrapping and locking
        if let Some(transform_mutex) = &cms.transform {
            let mut transform_guard = transform_mutex.lock().unwrap(); // Lock the mutex
            transform_guard.transform_pixels(&input_rgb, &mut output_linear);
        } else {
            return Err("Transform was unexpectedly skipped or None".into());
        }

        // Basic sanity check (replace with actual expected values if known)
        // For sRGB(0.5) -> Linear, output should be approx 0.214
        assert!(output_linear[0] > 0.2 && output_linear[0] < 0.23, "Unexpected R value: {}", output_linear[0]);
        assert!(output_linear[1] > 0.2 && output_linear[1] < 0.23, "Unexpected G value: {}", output_linear[1]);
        assert!(output_linear[2] > 0.2 && output_linear[2] < 0.23, "Unexpected B value: {}", output_linear[2]);

        println!("Transformed sRGB {:?} -> Linear {:?}", input_rgb, output_linear);

        Ok(())
    }

     #[test]
    fn test_cms_run_transform_skip() {
        let linear1 = dummy_profile("linear_srgb").unwrap();
        let linear2 = dummy_profile("linear_srgb").unwrap();
        let cms = cms_init(&linear1, &linear2, 255.0).unwrap();
        assert!(cms.skip_lcms);

        let srgb1 = dummy_profile("srgb").unwrap();
        let srgb2 = dummy_profile("srgb").unwrap();
        eprintln!("ICC data identical for linear1 and linear2? {}", linear1.icc == linear2.icc);
        let cms_linear_skip = cms_init(&linear1, &linear2, 255.0).unwrap();
        assert!(cms_linear_skip.skip_lcms);
        assert!(cms_linear_skip.transform.is_none());
        assert_eq!(cms_linear_skip.preprocess, tf::ExtraTF::kNone);
        assert_eq!(cms_linear_skip.postprocess, tf::ExtraTF::kNone);

        let srgb1 = dummy_profile("srgb").unwrap();
        let srgb2 = dummy_profile("srgb").unwrap();
        let cms_srgb_noskip = cms_init(&srgb1, &srgb2, 255.0).unwrap();
         assert!(!cms_srgb_noskip.skip_lcms);
         assert_eq!(cms_srgb_noskip.preprocess, tf::ExtraTF::kSRGB);
         assert_eq!(cms_srgb_noskip.postprocess, tf::ExtraTF::kSRGB);

         let linear_target = dummy_profile("linear_srgb").unwrap();
         let cms_srgb_linear = cms_init(&srgb1, &linear_target, 255.0).unwrap();
         assert!(!cms_srgb_linear.skip_lcms);
         assert_eq!(cms_srgb_linear.preprocess, tf::ExtraTF::kSRGB);
         assert_eq!(cms_srgb_linear.postprocess, tf::ExtraTF::kNone);
    }
}