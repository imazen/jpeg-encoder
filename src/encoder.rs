use alloc::{boxed::Box, vec, vec::Vec};
use alloc::string::ToString; // Import ToString for .to_string()

// Keep only imports from *other* modules
use crate::huffman::{CodingClass, HuffmanTable, HuffmanCode};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::adaptive_quantization::{compute_adaptive_quant_field, K_INPUT_SCALING};
use crate::quantization::{QuantizationTable, QuantizationTableType, quality_to_distance, compute_zero_bias_tables};
use crate::writer::{JfifWrite, JfifWriter, ZIGZAG};
use crate::error::{EncodingError, EncoderResult}; // Use EncoderResult alias
use crate::fdct::{fdct, forward_dct_float}; // Import both DCT functions
use crate::Density; // Add import for Density from lib.rs
use crate::cms::{self, ColorProfile, JxlCms, ColorSpaceSignature};
use crate::tf::{self, ExtraTF}; // Import ExtraTF
use crate::xyb; // Import xyb module
use crate::color_transform; // Import color_transform module

#[cfg(feature = "std")]
use std::{io::BufWriter, eprintln};

#[cfg(feature = "std")]
use std::fs::File;

#[cfg(feature = "std")]
use std::path::Path;

/// # Color types used in encoding
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JpegColorType {
    /// One component grayscale colorspace
    Luma,

    /// Three component YCbCr colorspace
    Ycbcr,

    /// 4 Component CMYK colorspace
    Cmyk,

    /// 4 Component YCbCrK colorspace
    Ycck,
}

impl JpegColorType {
    pub(crate) fn get_num_components(self) -> usize {
        use JpegColorType::*;

        match self {
            Luma => 1,
            Ycbcr => 3,
            Cmyk | Ycck => 4,
        }
    }
}

/// # Color types for input images
///
/// Available color input formats for [Encoder::encode]. Other types can be used
/// by implementing an [ImageBuffer](crate::ImageBuffer).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    /// Grayscale with 1 byte per pixel
    Luma,

    /// RGB with 3 bytes per pixel
    Rgb,

    /// Red, Green, Blue with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Rgba,

    /// RGB with 3 bytes per pixel
    Bgr,

    /// RGBA with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Bgra,

    /// YCbCr with 3 bytes per pixel.
    Ycbcr,

    /// CMYK with 4 bytes per pixel.
    Cmyk,

    /// CMYK with 4 bytes per pixel. Encoded as YCCK (YCbCrK)
    CmykAsYcck,

    /// YCCK (YCbCrK) with 4 bytes per pixel.
    Ycck,
}

impl ColorType {
    pub(crate) fn get_bytes_per_pixel(self) -> usize {
        use ColorType::*;

        match self {
            Luma => 1,
            Rgb | Bgr | Ycbcr => 3,
            Rgba | Bgra | Cmyk | CmykAsYcck | Ycck => 4,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// # Sampling factors for chroma subsampling
///
/// ## Warning
/// Sampling factor of 4 are not supported by all decoders or applications
#[allow(non_camel_case_types)]
pub enum SamplingFactor {
    F_1_1 = 1 << 4 | 1,
    F_2_1 = 2 << 4 | 1,
    F_1_2 = 1 << 4 | 2,
    F_2_2 = 2 << 4 | 2,
    F_4_1 = 4 << 4 | 1,
    F_4_2 = 4 << 4 | 2,
    F_1_4 = 1 << 4 | 4,
    F_2_4 = 2 << 4 | 4,

    /// Alias for F_1_1
    R_4_4_4 = 0x80 | 1 << 4 | 1,

    /// Alias for F_1_2
    R_4_4_0 = 0x80 | 1 << 4 | 2,

    /// Alias for F_1_4
    R_4_4_1 = 0x80 | 1 << 4 | 4,

    /// Alias for F_2_1
    R_4_2_2 = 0x80 | 2 << 4 | 1,

    /// Alias for F_2_2
    R_4_2_0 = 0x80 | 2 << 4 | 2,

    /// Alias for F_2_4
    R_4_2_1 = 0x80 | 2 << 4 | 4,

    /// Alias for F_4_1
    R_4_1_1 = 0x80 | 4 << 4 | 1,

    /// Alias for F_4_2
    R_4_1_0 = 0x80 | 4 << 4 | 2,
}

impl SamplingFactor {
    /// Get variant for supplied factors or None if not supported
    pub fn from_factors(horizontal: u8, vertical: u8) -> Option<SamplingFactor> {
        use SamplingFactor::*;

        match (horizontal, vertical) {
            (1, 1) => Some(F_1_1),
            (1, 2) => Some(F_1_2),
            (1, 4) => Some(F_1_4),
            (2, 1) => Some(F_2_1),
            (2, 2) => Some(F_2_2),
            (2, 4) => Some(F_2_4),
            (4, 1) => Some(F_4_1),
            (4, 2) => Some(F_4_2),
            _ => None,
        }
    }

    pub(crate) fn get_sampling_factors(self) -> (u8, u8) {
        let value = self as u8;
        ((value >> 4) & 0x07, value & 0xf)
    }

    pub(crate) fn supports_interleaved(self) -> bool {
        use SamplingFactor::*;

        // Interleaved mode is only supported with h/v sampling factors of 1 or 2.
        // Sampling factors of 4 needs sequential encoding
        matches!(
            self,
            F_1_1 | F_2_1 | F_1_2 | F_2_2 | R_4_4_4 | R_4_4_0 | R_4_2_2 | R_4_2_0
        )
    }
}

pub(crate) struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

macro_rules! add_component {
    ($components:expr, $id:expr, $dest:expr, $h_sample:expr, $v_sample:expr) => {
        $components.push(Component {
            id: $id,
            quantization_table: $dest,
            dc_huffman_table: $dest,
            ac_huffman_table: $dest,
            horizontal_sampling_factor: $h_sample,
            vertical_sampling_factor: $v_sample,
        });
    };
}

/// # The JPEG encoder
pub struct Encoder<W: JfifWrite> {
    writer: JfifWriter<W>,
    density: Density,
    quality: u8,
    jpegli_distance: Option<f32>,

    // CMS related fields
    input_profile: Option<ColorProfile>,
    internal_color_profile: ColorProfile, // Target profile for internal processing (e.g., Linear sRGB)
    cms_state: Option<Box<JxlCms>>, // Initialized later
    intensity_target: f32, // Needed for CMS
    xyb_mode: bool, // Jpegli XYB colorspace mode
    premul_absorb: Option<[f32; 12]>, // Precomputed matrix/constants for XYB

    components: Vec<Component>,
    quantization_tables: [QuantizationTableType; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

    sampling_factor: SamplingFactor,

    progressive_scans: Option<u8>,

    restart_interval: Option<u16>,

    optimize_huffman_table: bool,

    app_segments: Vec<(u8, Vec<u8>)>,

    /// Whether to enable Jpegli-style adaptive quantization.
    use_adaptive_quantization: bool,

    // Precomputed zero-bias tables (used for adaptive quantization thresholding)
    // These are computed based on distance/quality.
    zero_bias_offsets: Vec<[f32; 64]>,
    zero_bias_multipliers: Vec<[f32; 64]>,

    use_float_dct: bool,
}

impl<W: JfifWrite> Encoder<W> {
    /// Create a new encoder with the given quality
    ///
    /// The quality must be between 1 and 100 where 100 is the highest image quality.<br>
    /// By default, quality settings below 90 use a chroma subsampling (2x2 / 4:2:0) which can
    /// be changed with [set_sampling_factor](Encoder::set_sampling_factor).
    /// Default quantization uses standard Annex K tables.
    /// Assumes sRGB input profile by default.
    pub fn new(w: W, quality: u8) -> Encoder<W> {
        let huffman_tables = [
            (
                HuffmanTable::default_luma_dc(),
                HuffmanTable::default_luma_ac(),
            ),
            (
                HuffmanTable::default_chroma_dc(),
                HuffmanTable::default_chroma_ac(),
            ),
        ];

        // Default to Standard Annex K tables
        let quantization_tables = [
            QuantizationTableType::Default,
            QuantizationTableType::Default,
        ];

        let sampling_factor = if quality < 90 {
            SamplingFactor::R_4_2_0
        } else {
            SamplingFactor::R_4_4_4
        };

        let jpegli_distance = Some(quality_to_distance(quality));
        let (zero_bias_offsets, zero_bias_multipliers) = compute_zero_bias_tables(jpegli_distance.unwrap());

        Encoder {
            writer: JfifWriter::new(w),
            density: Density::default(),
            quality,
            jpegli_distance,

            input_profile: Some(ColorProfile::srgb()), // Default to sRGB input
            internal_color_profile: ColorProfile::linear_srgb(), // Default internal target is Linear sRGB
            cms_state: None,
            intensity_target: 255.0, // Default intensity target
            xyb_mode: false,
            premul_absorb: None,

            components: Vec::with_capacity(4),
            quantization_tables,
            huffman_tables,
            sampling_factor,
            progressive_scans: None,
            restart_interval: None,
            optimize_huffman_table: false,
            app_segments: Vec::new(),
            use_adaptive_quantization: false,
            zero_bias_offsets: zero_bias_offsets,
            zero_bias_multipliers: zero_bias_multipliers,
            use_float_dct: false,
        }
    }

    /// Sets the quality value for the encoder.
    /// Resets quantization tables to default scaled by quality, and resets Jpegli distance mode.
    pub fn set_quality(&mut self, quality: u8) {
        assert!(quality >= 1 && quality <= 100);
        self.quality = quality;
        self.jpegli_distance = None;
        self.quantization_tables = [
            QuantizationTableType::Default,
            QuantizationTableType::Default,
        ];
        // Recalculate zero bias tables for quality
        let distance = quality_to_distance(quality);
        let (zb_offsets, zb_multipliers) = compute_zero_bias_tables(distance);
        self.zero_bias_offsets = zb_offsets;
        self.zero_bias_multipliers = zb_multipliers;
    }

    /// Sets the target Butteraugli distance for Jpegli mode.
    /// Sets quantization tables to Jpegli defaults scaled by distance.
    pub fn set_jpegli_distance(&mut self, distance: f32) {
        assert!(distance > 0.0);
        self.jpegli_distance = Some(distance);
        self.quantization_tables = [
            QuantizationTableType::JpegliDefault,
            QuantizationTableType::JpegliDefault,
        ];
         // Recalculate zero bias tables for distance
        let (zb_offsets, zb_multipliers) = compute_zero_bias_tables(distance);
        self.zero_bias_offsets = zb_offsets;
        self.zero_bias_multipliers = zb_multipliers;
    }

    /// Sets the input color profile using raw ICC profile bytes.
    pub fn set_input_profile(&mut self, icc_data: Vec<u8>) -> EncoderResult<()> {
        let profile = ColorProfile::new(icc_data)?;
        self.input_profile = Some(profile);
        self.cms_state = None; // Invalidate previous CMS state
        Ok(())
    }

    /// Sets the target display intensity in cd/m^2 for color transforms (e.g., PQ/HLG).
    pub fn set_intensity_target(&mut self, nits: f32) {
        self.intensity_target = nits;
        self.cms_state = None; // Invalidate previous CMS state
        self.premul_absorb = None; // Invalidate XYB precomputation
    }

     /// Enables or disables the Jpegli XYB color space mode.
    /// When enabled, internal processing happens in XYB.
    pub fn set_xyb_mode(&mut self, enabled: bool) {
        self.xyb_mode = enabled;
        // Set internal target profile based on mode
        if enabled {
            // Need a way to represent XYB profile internally for CMS setup?
            // For now, we know conversion happens *after* CMS to linear sRGB.
             self.internal_color_profile = ColorProfile::linear_srgb();
             // Precompute XYB matrix if not done already
             if self.premul_absorb.is_none() {
                 self.premul_absorb = Some(xyb::compute_premul_absorb(self.intensity_target));
             }
        } else {
             self.internal_color_profile = ColorProfile::linear_srgb();
        }
        self.cms_state = None; // Invalidate previous CMS state
    }

    /// Set pixel density for the image
    ///
    /// By default, this value is None which is equal to "1 pixel per pixel".
    pub fn set_density(&mut self, density: Density) {
        self.density = density;
    }

    /// Return pixel density
    pub fn density(&self) -> Density {
        self.density
    }

    /// Set chroma subsampling factor
    pub fn set_sampling_factor(&mut self, sampling: SamplingFactor) {
        self.sampling_factor = sampling;
    }

    /// Get chroma subsampling factor
    pub fn sampling_factor(&self) -> SamplingFactor {
        self.sampling_factor
    }

    /// Set quantization tables for luma and chroma components
    pub fn set_quantization_tables(
        &mut self,
        luma: QuantizationTableType,
        chroma: QuantizationTableType,
    ) {
        self.quantization_tables = [luma, chroma];
    }

    /// Get configured quantization tables
    pub fn quantization_tables(&self) -> &[QuantizationTableType; 2] {
        &self.quantization_tables
    }

    /// Controls if progressive encoding is used.
    ///
    /// By default, progressive encoding uses 4 scans.<br>
    /// Use [set_progressive_scans](Encoder::set_progressive_scans) to use a different number of scans
    pub fn set_progressive(&mut self, progressive: bool) {
        self.progressive_scans = if progressive { Some(4) } else { None };
    }

    /// Set number of scans per component for progressive encoding
    ///
    /// Number of scans must be between 2 and 64.
    /// There is at least one scan for the DC coefficients and one for the remaining 63 AC coefficients.
    ///
    /// # Panics
    /// If number of scans is not within valid range
    pub fn set_progressive_scans(&mut self, scans: u8) {
        assert!(
            (2..=64).contains(&scans),
            "Invalid number of scans: {}",
            scans
        );
        self.progressive_scans = Some(scans);
    }

    /// Return number of progressive scans if progressive encoding is enabled
    pub fn progressive_scans(&self) -> Option<u8> {
        self.progressive_scans
    }

    /// Set restart interval
    ///
    /// Set numbers of MCUs between restart markers.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = if interval == 0 { None } else { Some(interval) };
    }

    /// Return the restart interval
    pub fn restart_interval(&self) -> Option<u16> {
        self.restart_interval
    }

    /// Set if optimized huffman table should be created
    ///
    /// Optimized tables result in slightly smaller file sizes but decrease encoding performance.
    pub fn set_optimized_huffman_tables(&mut self, optimize_huffman_table: bool) {
        self.optimize_huffman_table = optimize_huffman_table;
    }

    /// Returns if optimized huffman table should be generated
    pub fn optimized_huffman_tables(&self) -> bool {
        self.optimize_huffman_table
    }

    /// Appends a custom app segment to the JFIF file
    ///
    /// Segment numbers need to be in the range between 1 and 15<br>
    /// The maximum allowed data length is 2^16 - 2 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment number is invalid or data exceeds the allowed size
    pub fn add_app_segment(&mut self, segment_nr: u8, data: &[u8]) -> Result<(), EncodingError> {
        if segment_nr == 0 || segment_nr > 15 {
            Err(EncodingError::InvalidAppSegment(segment_nr))
        } else if data.len() > 65533 {
            Err(EncodingError::AppSegmentTooLarge(data.len()))
        } else {
            self.app_segments.push((segment_nr, data.to_vec()));
            Ok(())
        }
    }

    /// Add an ICC profile
    ///
    /// The maximum allowed data length is 16,707,345 bytes.
    ///
    /// # Errors
    ///
    /// Returns an Error if the data exceeds the maximum size for the ICC profile
    pub fn add_icc_profile(&mut self, data: &[u8]) -> Result<(), EncodingError> {
        // Based on https://www.color.org/ICC_Minor_Revision_for_Web.pdf
        // B.4  Embedding ICC profiles in JFIF files

        const MARKER: &[u8; 12] = b"ICC_PROFILE\0";
        const MAX_CHUNK_LENGTH: usize = 65535 - 2 - 12 - 2;

        let num_chunks = ceil_div(data.len(), MAX_CHUNK_LENGTH);

        // Sequence number is stored as a byte and starts with 1
        if num_chunks >= 255 {
            return Err(EncodingError::IccTooLarge(data.len()));
        }

        let mut chunk_data = Vec::with_capacity(MAX_CHUNK_LENGTH);

        for (i, data) in data.chunks(MAX_CHUNK_LENGTH).enumerate() {
            chunk_data.clear();
            chunk_data.extend_from_slice(MARKER);
            chunk_data.push(i as u8 + 1);
            chunk_data.push(num_chunks as u8);
            chunk_data.extend_from_slice(data);

            self.add_app_segment(2, &chunk_data)?;
        }

        Ok(())
    }

    /// Encode an image
    ///
    /// Data format and length must conform to specified width, height and color type.
    pub fn encode(
        self,
        data: &[u8],
        width: u16,
        height: u16,
        color_type: ColorType,
    ) -> Result<(), EncodingError> {
        let required_data_len = width as usize * height as usize * color_type.get_bytes_per_pixel();

        if data.len() < required_data_len {
            return Err(EncodingError::BadImageData {
                length: data.len(),
                required: required_data_len,
            });
        }

        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("avx2") {
                use crate::avx2::*;

                return match color_type {
                    ColorType::Luma => self
                        .encode_image_internal::<_, AVX2Operations>(GrayImage(data, width, height)),
                    ColorType::Rgb => self.encode_image_internal::<_, AVX2Operations>(
                        RgbImageAVX2(data, width, height),
                    ),
                    ColorType::Rgba => self.encode_image_internal::<_, AVX2Operations>(
                        RgbaImageAVX2(data, width, height),
                    ),
                    ColorType::Bgr => self.encode_image_internal::<_, AVX2Operations>(
                        BgrImageAVX2(data, width, height),
                    ),
                    ColorType::Bgra => self.encode_image_internal::<_, AVX2Operations>(
                        BgraImageAVX2(data, width, height),
                    ),
                    ColorType::Ycbcr => self.encode_image_internal::<_, AVX2Operations>(
                        YCbCrImage(data, width, height),
                    ),
                    ColorType::Cmyk => self
                        .encode_image_internal::<_, AVX2Operations>(CmykImage(data, width, height)),
                    ColorType::CmykAsYcck => self.encode_image_internal::<_, AVX2Operations>(
                        CmykAsYcckImage(data, width, height),
                    ),
                    ColorType::Ycck => self
                        .encode_image_internal::<_, AVX2Operations>(YcckImage(data, width, height)),
                };
            }
        }

        match color_type {
            ColorType::Luma => self.encode_image(GrayImage(data, width, height))?,
            ColorType::Rgb => self.encode_image(RgbImage(data, width, height))?,
            ColorType::Rgba => self.encode_image(RgbaImage(data, width, height))?,
            ColorType::Bgr => self.encode_image(BgrImage(data, width, height))?,
            ColorType::Bgra => self.encode_image(BgraImage(data, width, height))?,
            ColorType::Ycbcr => self.encode_image(YCbCrImage(data, width, height))?,
            ColorType::Cmyk => self.encode_image(CmykImage(data, width, height))?,
            ColorType::CmykAsYcck => self.encode_image(CmykAsYcckImage(data, width, height))?,
            ColorType::Ycck => self.encode_image(YcckImage(data, width, height))?,
        }

        Ok(())
    }

    /// Encode an image
    pub fn encode_image<I: ImageBuffer>(self, image: I) -> Result<(), EncodingError> {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("avx2") {
                use crate::avx2::*;
                return self.encode_image_internal::<_, AVX2Operations>(image);
            }
        }
        self.encode_image_internal::<_, DefaultOperations>(image)
    }

    fn encode_image_internal<I: ImageBuffer, OP: Operations>(
        mut self,
        image: I,
    ) -> EncoderResult<()> {
        let color = image.get_jpeg_color_type(); // Get JPEG color type
        self.init_components(color);

        // Determine target profile for internal processing
        let target_profile = match self.xyb_mode {
            true => ColorProfile::linear_srgb()?,
            false => self.internal_color_profile.clone(),
        };

        // Determine input profile (use target if none explicitly set)
        let input_profile_ref = self.input_profile.as_ref().unwrap_or(&target_profile);

        // Initialize CMS if input and target profiles differ
        let use_cms = input_profile_ref.icc != target_profile.icc;
        if use_cms {
            self.cms_state = Some(Box::new(cms::cms_init(input_profile_ref, &target_profile, self.intensity_target)?));
        }

        // Determine quantization tables (Jpegli distance or standard quality)
        let distance = self.jpegli_distance.unwrap_or_else(|| quality_to_distance(self.quality));
        let q_tables = [
            QuantizationTable::new(self.quantization_tables[0], distance, 0, false)?,
            QuantizationTable::new(self.quantization_tables[1], distance, 1, false)?,
        ];

        // --- Adaptive Quantization Field Calculation (if enabled) ---
        let mut adapt_quant_field: Option<Vec<f32>> = None;
        let mut temp_f32_planes: Vec<Vec<f32>> = Vec::new(); // Delay initialization

        if self.use_adaptive_quantization {
            // Needs the Y (luma) channel data as f32 scaled to [0, 1]
            // Run CMS/Color Conversion *before* AQ calculation
            temp_f32_planes = image_to_f32_planes(&image, use_cms, self.cms_state.as_deref())?;

            // Determine which plane to use for AQ based on the *target* color space
            let target_cs = target_profile.internal.as_ref().ok_or(
                EncodingError::CmsError("Missing target profile internal data for AQ".to_string())
            )?.color_space;

            let y_plane_idx = match target_cs {
                ColorSpaceSignature::GrayData | ColorSpaceSignature::RgbData => 0, // Use Gray or R for RGB
                // Add other cases if internal target can be different (e.g. YCbCr)
                _ => return Err(EncodingError::Other("AQ requires Gray or RGB target profile".into())),
            };

            if y_plane_idx >= temp_f32_planes.len() {
                return Err(EncodingError::Other("Could not get plane for AQ".into()));
            }

            // Input to AQ should be scaled [0, 1]
            // Assuming image_to_f32_planes outputs [0, 255.0] range floats
            let y_channel_div_255: Vec<f32> = temp_f32_planes[y_plane_idx].iter().map(|&p| p * K_INPUT_SCALING).collect();

            adapt_quant_field = Some(compute_adaptive_quant_field(
                image.width(),
                image.height(),
                &y_channel_div_255,
                distance,
                q_tables[0].get_raw(1) as i32, // Pass base quant for AC(0,1) (Luma table)
            ));
        }

        // --- Write Headers ---
        self.write_frame_header(&image, color, &q_tables)?;
        self.write_scan_header(0)?;

        let mut fdct_scratch = vec![0.0f32; 64]; // Scratch space for float DCT

        // --- Prepare F32 Planes for Block Processing --- 
        // If AQ wasn't used, compute the f32 planes now.
        if temp_f32_planes.is_empty() {
            temp_f32_planes = image_to_f32_planes(&image, use_cms, self.cms_state.as_deref())?;
        }

        // Apply post-CMS transforms (XYB, YCbCr) if needed
        if self.xyb_mode {
            if temp_f32_planes.len() < 3 {
                return Err(EncodingError::Other("XYB needs 3 planes".into()));
            }
            if let Some(premul) = self.premul_absorb {
                // Process row by row for XYB conversion
                let width = image.width() as usize;
                let height = image.height() as usize;
                let num_pixels_total = width * height;
                let mut row_r = vec![0.0; width];
                let mut row_g = vec![0.0; width];
                let mut row_b = vec![0.0; width];

                for y in 0..height {
                    let offset = y * width;
                    row_r.copy_from_slice(&temp_f32_planes[0][offset..offset + width]);
                    row_g.copy_from_slice(&temp_f32_planes[1][offset..offset + width]);
                    row_b.copy_from_slice(&temp_f32_planes[2][offset..offset + width]);

                    xyb::linear_rgb_row_to_xyb(&mut row_r, &mut row_g, &mut row_b, &premul, width);

                    temp_f32_planes[0][offset..offset + width].copy_from_slice(&row_r);
                    temp_f32_planes[1][offset..offset + width].copy_from_slice(&row_g);
                    temp_f32_planes[2][offset..offset + width].copy_from_slice(&row_b);
                }
                // Apply scaling *after* row processing
                xyb::scale_xyb_row(
                    &mut temp_f32_planes[0],
                    &mut temp_f32_planes[1],
                    &mut temp_f32_planes[2],
                    num_pixels_total
                );
            } else {
                 return Err(EncodingError::Other("XYB enabled but constants missing".into()));
            }
        } else if color == JpegColorType::Ycbcr || color == JpegColorType::Ycck {
             // Apply YCbCr conversion if needed (i.e., if input was RGB/Gray and target wasn't XYB)
             // Check if target_profile is YCbCr based? This logic is complex.
             // Assume for now if jpeg color type is YCbCr/YCCK, the planes *should* be YCbCr.
             // If the input was RGB and target is YCbCr, CMS should have handled it.
             // If input was RGB and target is RGB (no XYB), we need to convert here.
             let target_cs = target_profile.internal.as_ref().ok_or(
                EncodingError::CmsError("Missing target profile internal data".to_string())
            )?.color_space;
             if target_cs == ColorSpaceSignature::RgbData { // Convert RGB planes to YCbCr
                 if temp_f32_planes.len() < 3 {
                     return Err(EncodingError::Other("YCbCr conversion needs 3 planes".into()));
                 }
                 color_transform::rgb_to_ycbcr_planes(
                     &mut temp_f32_planes[0],
                     &mut temp_f32_planes[1],
                     &mut temp_f32_planes[2],
                     image.width() as usize * image.height() as usize,
                 );
             }
             // If YCCK, the K plane should already be present as the 4th plane
        }

        // --- Process MCUs ---
        let width = image.width() as usize;
        let height = image.height() as usize;
        let num_pixels = width * height;
        let num_components = image.get_num_components(); // Use correct method

        let mut blocks: [Vec<[i16; 64]>; 4] = Default::default();
        let mut block_buffers: [Vec<[f32; 64]>; 4] = Default::default();
        let mut dct_coeffs: [Vec<[f32; 64]>; 4] = Default::default();
        let mut scratch_space: [Vec<[f32; 64]>; 4] = Default::default();

        let h_samp = self.components[0].horizontal_sampling_factor as usize;
        let v_samp = self.components[0].vertical_sampling_factor as usize;
        let mcu_width = h_samp * 8;
        let mcu_height = v_samp * 8;
        let mcus_w = ceil_div(width, mcu_width);
        let mcus_h = ceil_div(height, mcu_height);
        let num_mcus = mcus_w * mcus_h;

        for c in 0..num_components {
            let comp = &self.components[c];
            let n_blocks_h = ceil_div(width * comp.horizontal_sampling_factor as usize, h_samp * 8);
            let n_blocks_v = ceil_div(height * comp.vertical_sampling_factor as usize, v_samp * 8);
            let n_blocks = n_blocks_h * n_blocks_v;
            blocks[c].resize(n_blocks, [0i16; 64]);
            if self.use_float_dct {
                block_buffers[c].resize(n_blocks, [0f32; 64]);
                dct_coeffs[c].resize(n_blocks, [0f32; 64]);
                scratch_space[c].resize(n_blocks, [0f32; 64]);
            }

            let plane = &temp_f32_planes[c];
            let component_width = n_blocks_h * 8;
            let component_height = n_blocks_v * 8;

            for y_b in 0..n_blocks_v {
                for x_b in 0..n_blocks_h {
                    let block_idx = y_b * n_blocks_h + x_b;
                    let start_x = x_b * 8;
                    let start_y = y_b * 8;

                    let block_f32: &mut [f32; 64] = if self.use_float_dct {
                        &mut block_buffers[c][block_idx]
                    } else {
                        // Need a temporary f32 buffer for int DCT path too
                        // This is inefficient - ideally int DCT takes i16 input.
                         // For now, use scratch space temporarily
                         &mut scratch_space[c][block_idx]
                    };

                    // Extract 8x8 block from f32 plane, scaling [0,1] -> [-128, 127]
                    for y in 0..8 {
                        let py = (start_y + y).min(component_height - 1);
                        let row_offset = py * component_width; // Use component width here
                        for x in 0..8 {
                            let px = (start_x + x).min(component_width - 1);
                            // Map [0.0, 1.0] to [-128.0, 127.0] for DCT input
                            block_f32[y * 8 + x] = plane[row_offset + px] * 255.0 - 128.0;
                        }
                    }

                    // Apply DCT
                    if self.use_float_dct {
                        let coeff_block = &mut dct_coeffs[c][block_idx];
                        let scratch = &mut scratch_space[c][block_idx];
                        OP::forward_dct_float(block_f32, coeff_block, scratch);
                        OP::quantize_float_block(
                            coeff_block,
                            &mut blocks[c][block_idx],
                            &q_tables[comp.quantization_table as usize],
                            adapt_quant_field.as_deref(),
                            block_idx, // TODO: Need correct block index for AQ field lookup
                            &self.zero_bias_offsets[c],
                            &self.zero_bias_multipliers[c],
                        );
                    } else {
                         // Integer DCT path - needs rework as fdct takes i16
                         // 1. Convert f32 block [-128, 127] to i16 block
                         let mut block_i16 = [0i16; 64];
                         for i in 0..64 {
                             block_i16[i] = block_f32[i].round() as i16;
                         }
                         // 2. Apply integer DCT
                         OP::fdct(&mut block_i16);
                         // 3. Quantize integer DCT output
                         OP::quantize_block(
                            &block_i16,
                            &mut blocks[c][block_idx],
                            &q_tables[comp.quantization_table as usize],
                            adapt_quant_field.as_deref(),
                            block_idx, // TODO: Need correct block index
                             &self.zero_bias_offsets[c],
                            &self.zero_bias_multipliers[c],
                        );
                    }
                }
            }
        }

        // --- Entropy Coding and Writing ---
        if self.optimize_huffman_table {
            // Optimize tables based on actual coefficient frequencies
            self.optimize_huffman_tables_internal(&blocks)?;
        }
        self.write_scan_header(0)?;

        self.writer.write_scan_data(&blocks, &self.components, &self.huffman_tables)?;

        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn init_components(&mut self, color: JpegColorType) {
        let (horizontal_sampling_factor, vertical_sampling_factor) =
            self.sampling_factor.get_sampling_factors();

        match color {
            JpegColorType::Luma => {
                add_component!(self.components, 0, 0, 1, 1);
            }
            JpegColorType::Ycbcr => {
                add_component!(
                    self.components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
            }
            JpegColorType::Cmyk => {
                add_component!(self.components, 0, 1, 1, 1);
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(
                    self.components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            }
            JpegColorType::Ycck => {
                add_component!(
                    self.components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(
                    self.components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            }
        }
    }

    fn get_max_sampling_size(&self) -> (usize, usize) {
        let max_h_sampling = self.components.iter().fold(1, |value, component| {
            value.max(component.horizontal_sampling_factor)
        });

        let max_v_sampling = self.components.iter().fold(1, |value, component| {
            value.max(component.vertical_sampling_factor)
        });

        (usize::from(max_h_sampling), usize::from(max_v_sampling))
    }

    fn write_frame_header<I: ImageBuffer>(
        &mut self,
        image: &I,
        color: JpegColorType,
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        self.writer.write_marker(Marker::SOI)?;
        self.writer.write_jfif_header(self.density)?;
        for (marker, data) in &self.app_segments {
            self.writer.write_app_segment(*marker, data)?;
        }
        self.writer.write_dqt(q_tables)?;
        self.writer.write_sof(
            image.width(),
            image.height(),
            &self.components.iter().collect::<Vec<_>>(),
            self.progressive_scans.is_some(),
        )?;
        if let Some(interval) = self.restart_interval {
            self.writer.write_dri(interval)?;
        }
        Ok(())
    }

    fn write_scan_header(&mut self, scan_idx: usize) -> EncoderResult<()> {
        // If optimizing, DHT is written *after* optimization, right before scan data.
        // This is handled in the main encode loop now.
        if !self.optimize_huffman_table {
             self.writer.write_dht(&self.huffman_tables)?;
        }

        if let Some(_scans) = self.progressive_scans { // Use _scans
            // TODO: Implement progressive scan header writing
            return Err(EncodingError::Other("Progressive scan headers not implemented".into()));
        } else {
            // Baseline scan
            self.writer.write_sos(&self.components)?;
        }
        Ok(())
    }

    /// Enable or disable Jpegli-style adaptive quantization.
    ///
    /// This analyzes the image to apply spatially varying quantization,
    /// potentially improving perceived quality for a given file size.
    /// It may increase encoding time.
    /// This is independent of the `set_jpegli_distance` setting,
    /// though typically used with it.
    pub fn set_adaptive_quantization(&mut self, enabled: bool) {
        self.use_adaptive_quantization = enabled;
    }

    pub fn set_float_dct(&mut self, enable: bool) {
        self.use_float_dct = enable;
    }
}

#[cfg(feature = "std")]
impl Encoder<BufWriter<File>> {
    /// Create a new decoder that writes into a file
    ///
    /// See [new](Encoder::new) for further information.
    ///
    /// # Errors
    ///
    /// Returns an `IoError(std::io::Error)` if the file can't be created
    pub fn new_file<P: AsRef<Path>>(
        path: P,
        quality: u8,
    ) -> Result<Encoder<BufWriter<File>>, EncodingError> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        Ok(Self::new(buf, quality))
    }
}

fn get_block(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> [i16; 64] {
    let mut block = [0i16; 64];

    for y in 0..8 {
        for x in 0..8 {
            let ix = start_x + (x * col_stride);
            let iy = start_y + (y * row_stride);

            block[y * 8 + x] = (data[iy * width + ix] as i16) - 128;
        }
    }

    block
}

fn ceil_div(value: usize, div: usize) -> usize {
    value / div + usize::from(value % div != 0)
}

fn get_num_bits(mut value: i16) -> u8 {
    if value < 0 {
        value = -value;
    }

    let mut num_bits = 0;

    while value > 0 {
        num_bits += 1;
        value >>= 1;
    }

    num_bits
}

pub(crate) trait Operations {
    #[inline(always)]
    fn fdct(data: &mut [i16; 64]) {
        fdct(data);
    }

    #[inline(always)]
    fn forward_dct_float(pixels: &[f32; 64], coefficients: &mut [f32; 64], scratch_space: &mut [f32; 64]) {
        forward_dct_float(pixels, coefficients, scratch_space);
    }

    #[inline(always)]
    fn quantize_block(
        block: &[i16; 64], // Integer DCT output
        q_block: &mut [i16; 64],
        table: &QuantizationTable,
        adapt_quant_field: Option<&[f32]>,
        block_idx: usize,
        zero_bias_offset: &[f32; 64],
        zero_bias_mul: &[f32; 64],
    ) {
        const SHIFT: i32 = 3;

        // Get the jpegli-style aq_strength value (offset-based) for this block
        let aq_strength = adapt_quant_field.map_or(0.0, |field| {
            field.get(block_idx).copied().unwrap_or(0.0)
        });

        for i in 0..64 {
            let z = ZIGZAG[i] as usize & 0x3f;
            let value = block[z] as i32; // Value from integer DCT
            let q_val = table.get_raw(z) as i32;
            let divisor = q_val << SHIFT;

            // Ensure divisor is not zero, though table values should be >= 1
            if divisor == 0 { continue; }

            // Standard quantization with rounding: round(value / divisor)
            let half_divisor_signed = (divisor as f32 / 2.0).copysign(value as f32) as i32;
            let mut q_coeff = (value + half_divisor_signed) / divisor;

            // Jpegli-style adaptive quantization thresholding (approximated)
            // Applied *after* initial integer quantization for this path.
            if i > 0 && q_coeff != 0 && adapt_quant_field.is_some() {
                let zb_offset = zero_bias_offset[i];
                let zb_mul = zero_bias_mul[i];
                let threshold = zb_offset + zb_mul * aq_strength;

                // Value to compare against threshold (absolute value before quantization)
                // Reconstruct approximate original coeff value (scaled by 8 * q_val)
                // This might be inaccurate. C++ compares `std::abs(scaled_val)` where
                // scaled_val = coeff * qmc[k]; and coeff is float dct, qmc = 1.0/quant
                // Let's use the absolute value of the *quantized* coefficient as a proxy.
                let abs_quant_coeff = q_coeff.abs() as f32;
                // Calculate threshold delta based on quantized value
                let quant_delta = abs_quant_coeff * zb_mul;

                // Jpegli check: if std::abs(scaled_val) < bias + quant_delta
                if abs_quant_coeff < threshold {
                    q_coeff = 0;
                }
            }

            q_block[i] = q_coeff as i16;
        }
    }

    #[inline(always)]
    fn quantize_float_block(
        coeffs: &[f32; 64], // Float DCT output
        q_block: &mut [i16; 64],
        table: &QuantizationTable,
        adapt_quant_field: Option<&[f32]>,
        block_idx: usize,
        zero_bias_offset: &[f32; 64],
        zero_bias_mul: &[f32; 64],
    ) {
        // Get the jpegli-style aq_strength value (additive bias offset)
        let aq_strength = adapt_quant_field.map_or(0.0, |field| {
            field.get(block_idx).copied().unwrap_or(0.0)
        });

        for i in 0..64 {
            let z = ZIGZAG[i] as usize; // Use direct zigzag index
            let value_f = coeffs[z]; // Value from float DCT
            let q_val_f = table.get_raw(z) as f32;

            // Ensure q_val_f is not zero
            if q_val_f == 0.0 { continue; }

            // Quantization: value_f / q_val_f (jpegli divides by quant value)
            // Jpegli internal `QuantizeBlock` multiplies by qmc (quant multiplier = 1.0/q_val_f)
            let scaled_val = value_f / q_val_f;

            // Rounding (round half away from zero)
            let q_coeff_f = scaled_val.round(); // .round() rounds half to even, use custom?
            // Let's use simple rounding first
            let mut q_coeff = q_coeff_f as i32;

            // Jpegli-style adaptive quantization thresholding (zeroing bias)
            // Applied *before* rounding for the float path, based on unrounded `scaled_val`.
            if i > 0 && q_coeff != 0 && adapt_quant_field.is_some() {
                let zb_offset = zero_bias_offset[i];
                let zb_mul = zero_bias_mul[i];
                let bias = zb_offset + aq_strength; // Base threshold + AQ offset
                let quant_delta = scaled_val.abs() * zb_mul; // Threshold delta based on unquantized value

                // Jpegli check: if std::abs(scaled_val) < bias + quant_delta
                if scaled_val.abs() < bias + quant_delta {
                    q_coeff = 0;
                }
            }

            q_block[i] = q_coeff as i16;
        }
    }
}

pub(crate) struct DefaultOperations;

impl Operations for DefaultOperations {}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::encoder::get_num_bits;
    use crate::writer::get_code;
    use crate::{ColorType, Encoder, QuantizationTableType, SamplingFactor};

    // Helper to create a small grayscale image (e.g., 16x16)
    fn create_test_image(width: usize, height: usize) -> (Vec<u8>, u16, u16, ColorType) {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                // Simple gradient
                data.push(((x + y) % 256) as u8);
            }
        }
        (data, width as u16, height as u16, ColorType::Luma)
    }

    #[test]
    fn test_get_num_bits() {
        let min_max = 2i16.pow(13);

        for value in -min_max..=min_max {
            let num_bits1 = get_num_bits(value);
            let (num_bits2, _) = get_code(value);

            assert_eq!(
                num_bits1, num_bits2,
                "Difference in num bits for value {}: {} vs {}",
                value, num_bits1, num_bits2
            );
        }
    }

    #[test]
    fn test_encode_default() {
        // Test default path (integer DCT, no AQ, quality based)
        let (data, width, height, color_type) = create_test_image(16, 16);
        let encoder = Encoder::new(Vec::new(), 90);
        assert!(encoder.encode(&data, width, height, color_type).is_ok());
    }

    #[test]
    fn test_encode_float_dct() {
        // Test float DCT path
        let (data, width, height, color_type) = create_test_image(16, 16);
        let mut encoder = Encoder::new(Vec::new(), 90);
        encoder.set_float_dct(true);
        assert!(encoder.encode(&data, width, height, color_type).is_ok());
    }

    #[test]
    fn test_encode_adaptive_quant() {
        // Test adaptive quantization path (with default int DCT)
        let (data, width, height, color_type) = create_test_image(16, 16);
        let mut encoder = Encoder::new(Vec::new(), 90);
        encoder.set_adaptive_quantization(true);
        // Encoding might fail if image is too small for AQ padding/analysis?
        // For now, just check if it runs.
        let result = encoder.encode(&data, width, height, color_type);
        // Allow Ok or specific errors related to AQ constraints if any
        assert!(result.is_ok(), "AQ encoding failed: {:?}", result.err());
    }

    #[test]
    fn test_encode_jpegli_distance() {
        // Test Jpegli distance mode (with default int DCT, no AQ)
        let (data, width, height, color_type) = create_test_image(16, 16);
        let mut encoder = Encoder::new(Vec::new(), 90); // Initial quality doesn't matter
        encoder.set_jpegli_distance(1.0);
        assert!(encoder.encode(&data, width, height, color_type).is_ok());
    }

    #[test]
    fn test_encode_float_dct_aq() {
        // Test float DCT + adaptive quantization
        let (data, width, height, color_type) = create_test_image(16, 16);
        let mut encoder = Encoder::new(Vec::new(), 90);
        encoder.set_float_dct(true);
        encoder.set_adaptive_quantization(true);
        let result = encoder.encode(&data, width, height, color_type);
        assert!(result.is_ok(), "Float DCT + AQ encoding failed: {:?}", result.err());
    }

    #[test]
    fn test_encode_jpegli_float_dct_aq() {
        // Test Jpegli distance + float DCT + adaptive quantization
        let (data, width, height, color_type) = create_test_image(16, 16);
        let mut encoder = Encoder::new(Vec::new(), 90);
        encoder.set_jpegli_distance(1.0);
        encoder.set_float_dct(true);
        encoder.set_adaptive_quantization(true);
        let result = encoder.encode(&data, width, height, color_type);
        assert!(result.is_ok(), "Jpegli + Float DCT + AQ encoding failed: {:?}", result.err());
    }
}

// Helper to convert ImageBuffer (u8) to Vec<Vec<f32>> (planar)
fn image_to_f32_planes<I: ImageBuffer>(image: &I) -> EncoderResult<Vec<Vec<f32>>> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let num_pixels = width * height;
    let num_components = image.color_type().get_num_components();
    let mut planes = vec![vec![0.0f32; num_pixels]; num_components];

    // TODO: This assumes get_pixel gives components in the expected order (R,G,B or L or C,M,Y,K)
    // Needs careful checking based on ImageBuffer implementations.
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = y * width + x;
            let pixel_data = image.get_pixel(x, y);
            for c in 0..num_components {
                // Scale u8 [0, 255] to f32 [0.0, 1.0]
                planes[c][pixel_idx] = (pixel_data[c] as f32) / 255.0;
            }
        }
    }
    Ok(planes)
}

/// Optimized Huffman table calculation based on coefficient statistics.
fn optimize_huffman_tables_internal(&mut self, blocks: &[Vec<[i16; 64]>; 4]) -> EncoderResult<()> {
    let mut counts = [[0u32; 256]; 4]; // DC counts [0..1], AC counts [0..3]
    let mut dc_bits = [[0u8; 16]; 2]; // Bit lengths for DC Huffman
    let mut ac_bits = [[0u8; 16]; 2]; // Bit lengths for AC Huffman

    let num_components = self.components.len();

    for c in 0..num_components {
        let comp = &self.components[c];
        let h_samp = comp.horizontal_sampling_factor as usize;
        let v_samp = comp.vertical_sampling_factor as usize;
        let num_comp_blocks = blocks[c].len();

        // Count frequencies
        let mut last_dc = 0;
        for i in 0..num_comp_blocks {
            let block = &blocks[c][i];
            // DC coefficient
            let dc_diff = block[0] - last_dc;
            last_dc = block[0];
            let nbits = get_num_bits(dc_diff);
            if nbits >= 12 { return Err(EncodingError::Other("DC coeff out of range".into())); }
            counts[comp.dc_huffman_table as usize][nbits as usize] += 1;

            // AC coefficients
            let mut zero_run = 0;
            for k in 1..64 {
                let ac_val = block[k];
                if ac_val == 0 {
                    zero_run += 1;
                } else {
                    while zero_run > 15 {
                        counts[2 + comp.ac_huffman_table as usize][0xf0] += 1; // ZRL code
                        zero_run -= 16;
                    }
                    let nbits = get_num_bits(ac_val);
                     if nbits >= 11 { return Err(EncodingError::Other("AC coeff out of range".into())); }
                    let symbol = (zero_run << 4) | nbits;
                    counts[2 + comp.ac_huffman_table as usize][symbol as usize] += 1;
                    zero_run = 0;
                }
            }
            if zero_run > 0 {
                counts[2 + comp.ac_huffman_table as usize][0x00] += 1; // EOB code
            }
        }
    }

    // Generate Huffman tables
    for i in 0..2 { // DC tables
        let bits = HuffmanTable::build_huffman_bits(&counts[i], &mut dc_bits[i]);
        self.huffman_tables[i].0 = HuffmanTable::from_counts(&counts[i], bits)?;
    }
    for i in 0..2 { // AC tables
        let bits = HuffmanTable::build_huffman_bits(&counts[2 + i], &mut ac_bits[i]);
        self.huffman_tables[i].1 = HuffmanTable::from_counts(&counts[2+i], bits)?;
    }

    Ok(())
}

/// Extracts f32 planes from the image, handling color conversion and CMS.
fn image_to_f32_planes<I: ImageBuffer>(
    image: &I,
    use_cms: bool,
    cms_state: Option<&JxlCms>,
) -> EncoderResult<Vec<Vec<f32>>> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let num_pixels = width * height;
    let num_components = image.get_num_components(); // Use correct method

    let mut initial_planes: Vec<Vec<f32>> = vec![vec![0.0; num_pixels]; num_components];

    // Extract initial planes (e.g., R, G, B or Luma)
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = y * width + x;
            let pixel_values = image.get_pixel_f32(x, y)?; // Use f32 method
            for c in 0..num_components {
                initial_planes[c][pixel_idx] = pixel_values.get(c).copied().unwrap_or(0.0);
            }
        }
    }

    if use_cms {
        let cms = cms_state.ok_or_else(|| EncodingError::CmsError("CMS state missing".to_string()))?;
        let num_cms_in = cms.channels_src;
        let num_cms_out = cms.channels_dst;

        if initial_planes.len() != num_cms_in {
             return Err(EncodingError::CmsError("Input plane count mismatch for CMS".to_string()));
        }

        // Interleave input planes for CMS
        let mut interleaved_in = vec![0.0f32; num_pixels * num_cms_in];
        for p in 0..num_pixels {
            for c in 0..num_cms_in {
                interleaved_in[p * num_cms_in + c] = initial_planes[c][p];
            }
        }

        let mut interleaved_out = vec![0.0f32; num_pixels * num_cms_out];
        cms.run_transform(&interleaved_in, &mut interleaved_out, num_pixels)?;

        // Deinterleave output planes
        let mut output_planes: Vec<Vec<f32>> = vec![vec![0.0; num_pixels]; num_cms_out];
        for p in 0..num_pixels {
            for c in 0..num_cms_out {
                output_planes[c][p] = interleaved_out[p * num_cms_out + c];
            }
        }
        Ok(output_planes)
    } else {
        Ok(initial_planes)
    }
}

/// Extracts an 8x8 f32 block from a plane.
fn get_block_f32(plane: &[f32], start_x: usize, start_y: usize, stride: usize, block: &mut [f32; 64]) {
    let plane_height = plane.len() / stride;
    for r in 0..8 {
        let y = (start_y + r).min(plane_height - 1);
        let row_start = y * stride;
        for c in 0..8 {
            let x = (start_x + c).min(stride - 1);
            block[r * 8 + c] = plane[row_start + x];
        }
    }
}
