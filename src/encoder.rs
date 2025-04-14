use crate::fdct::fdct;
use crate::huffman::{CodingClass, HuffmanTable};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::quantization::{QuantizationTable, QuantizationTableType, quality_to_distance};
use crate::writer::{JfifWrite, JfifWriter, ZIGZAG};
use crate::{Density, EncodingError};

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::io::BufWriter;

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
}

impl<W: JfifWrite> Encoder<W> {
    /// Create a new encoder with the given quality
    ///
    /// The quality must be between 1 and 100 where 100 is the highest image quality.<br>
    /// By default, quality settings below 90 use a chroma subsampling (2x2 / 4:2:0) which can
    /// be changed with [set_sampling_factor](Encoder::set_sampling_factor).
    /// Default quantization uses standard Annex K tables.
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
            QuantizationTableType::Default, // Keep original default
            QuantizationTableType::Default,
        ];

        let sampling_factor = if quality < 90 {
            SamplingFactor::F_2_2
        } else {
            SamplingFactor::F_1_1
        };

        Encoder {
            writer: JfifWriter::new(w),
            density: Density::None,
            quality: quality.clamp(1, 100),
            jpegli_distance: None, // Jpegli mode not active by default
            components: vec![],
            quantization_tables,
            huffman_tables,
            sampling_factor,
            progressive_scans: None,
            restart_interval: None,
            optimize_huffman_table: false,
            app_segments: Vec::new(),
            use_adaptive_quantization: false, // Default to off
            zero_bias_offsets: Vec::new(), // Initialized later
            zero_bias_multipliers: Vec::new(), // Initialized later
        }
    }

    /// Set the target quality (1-100).
    ///
    /// This uses the standard Annex K quantization tables scaled by quality.
    /// Calling this method disables Jpegli distance mode if it was previously set.
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality.clamp(1, 100);
        self.jpegli_distance = None; // Disable Jpegli distance mode
        // Adjust default sampling factor based on quality if needed (optional)
        self.sampling_factor = if self.quality < 90 {
            // Ensure zero-bias tables are cleared or recomputed if needed when switching modes
            // For simplicity, we'll recompute them in encode_image_internal
            self.zero_bias_offsets.clear();
            self.zero_bias_multipliers.clear();
            SamplingFactor::F_2_2
        } else {
            SamplingFactor::F_1_1
        };
        // Ensure default quant table type is selected if user switches back to quality
        self.quantization_tables = [
            QuantizationTableType::Default,
            QuantizationTableType::Default,
        ];
    }

    /// Set the target Butteraugli distance and enable Jpegli quantization mode.
    ///
    /// This overrides the quality setting and uses Jpegli's psychovisual quantization tables scaled by distance.
    /// Lower distance means higher quality.
    /// Recommended range: 0.1 (visually lossless) to 25+ (low quality).
    pub fn set_jpegli_distance(&mut self, distance: f32) {
        self.jpegli_distance = Some(distance.max(0.0));
        // We don't need to set self.quality or self.quantization_tables here,
        // as the encoding logic will use the distance directly.
        // Adaptive quant should ideally be enabled automatically with jpegli distance,
        // but we'll keep it explicit for now.
        // self.use_adaptive_quantization = true; // Consider this default
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
    ) -> Result<(), EncodingError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(EncodingError::ZeroImageDimensions {
                width: image.width(),
                height: image.height(),
            });
        }

        // Determine if YUV420 based on sampling factor
        let (h_samp, v_samp) = self.sampling_factor.get_sampling_factors();
        let is_yuv420 = h_samp == 2 && v_samp == 2;
        // Default force_baseline to true for now, matching jpegli_set_quality behaviour
        let force_baseline = true;

        // Determine effective distance for quantization and zero-bias calculation
        let distance = self.jpegli_distance.unwrap_or_else(|| quality_to_distance(self.quality));

        let q_tables = if let Some(d) = self.jpegli_distance {
            // Jpegli distance mode is active
            [
                QuantizationTable::new_with_jpegli_distance(
                    d, // Use the stored distance
                    true, // is_luma
                    is_yuv420,
                    force_baseline,
                ),
                QuantizationTable::new_with_jpegli_distance(
                    d,
                    false, // is_luma
                    is_yuv420,
                    force_baseline,
                ),
            ]
        } else {
            // Standard quality mode is active
            // Calculate distance from quality for zero-bias tables
            let zb_distance = quality_to_distance(self.quality);
            [
                QuantizationTable::new_with_quality(
                    &self.quantization_tables[0],
                    self.quality,
                    true, // is_luma
                    is_yuv420,
                    force_baseline,
                ),
                QuantizationTable::new_with_quality(
                    &self.quantization_tables[1],
                    self.quality,
                    false, // is_luma
                    is_yuv420,
                    force_baseline,
                ),
            ]
        };

        let jpeg_color_type = image.get_jpeg_color_type();
        self.init_components(jpeg_color_type);

        // --- Compute Zero Bias Tables --- 
        let num_components = self.components.len();
        (self.zero_bias_offsets, self.zero_bias_multipliers) = 
            crate::quantization::compute_zero_bias_tables(distance, num_components);
        // Ensure tables have the correct size
        assert_eq!(self.zero_bias_offsets.len(), num_components);
        assert_eq!(self.zero_bias_multipliers.len(), num_components);
        // -------------------------------

        // --- Adaptive Quantization Field Calculation ---
        let adapt_quant_field: Option<Vec<f32>> = if self.use_adaptive_quantization {
            // Attempt to get the primary channel data (e.g., Luma)
            if let Some(channel_data_u8) = image.get_adaptive_quant_channel() {
                // Convert u8 data to f32 for adaptive quant function
                let channel_data_f32: Vec<f32> = channel_data_u8.iter().map(|&p| p as f32).collect();

                // Scale to [0, 1] range as expected by jpegli's internal functions
                let mut channel_data_scaled: Vec<f32> = channel_data_f32.iter().map(|&p| p / 255.0).collect();

                // Get the effective distance (either from jpegli mode or quality mapping)
                let distance = self.jpegli_distance.unwrap_or_else(|| quality_to_distance(self.quality));

                // Get y_quant_01 (quant value of first AC coeff in Luma table)
                // Note: This requires the quant tables to be computed *before* adaptive field.
                let y_quant_01 = q_tables[0].get_raw(1) as i32; // Index 1 is first AC coeff

                // Call the adaptive quant function
                Some(compute_adaptive_quant_field(
                    image.width(),
                    image.height(),
                    &mut channel_data_scaled, // Pass mutable for potential in-place padding
                    distance,
                    y_quant_01,
                ))
            } else {
                // Warn or error? For now, just disable AQ if channel data is unavailable.
                eprintln!("Warning: Could not get channel data for adaptive quantization. Disabling.");
                None
            }
        } else {
            None
        };
        // -----------------------------------------------

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        if jpeg_color_type == JpegColorType::Cmyk {
            //Set ColorTransform info to "Unknown"
            let app_14 = b"Adobe\0\0\0\0\0\0\0";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        } else if jpeg_color_type == JpegColorType::Ycck {
            //Set ColorTransform info to YCCK
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        }

        for (nr, data) in &self.app_segments {
            self.writer.write_segment(Marker::APP(*nr), data)?;
        }

        if let Some(scans) = self.progressive_scans {
            self.encode_image_progressive::<_, OP>(image, scans, &q_tables, adapt_quant_field.as_deref())?;
        } else if self.optimize_huffman_table || !self.sampling_factor.supports_interleaved() {
            self.encode_image_sequential::<_, OP>(image, &q_tables, adapt_quant_field.as_deref())?;
        } else {
            self.encode_image_interleaved::<_, OP>(image, &q_tables, adapt_quant_field.as_deref())?;
        }

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
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        self.writer.write_frame_header(
            image.width(),
            image.height(),
            &self.components,
            self.progressive_scans.is_some(),
        )?;

        self.writer.write_quantization_segment(0, &q_tables[0])?;
        self.writer.write_quantization_segment(1, &q_tables[1])?;

        self.writer
            .write_huffman_segment(CodingClass::Dc, 0, &self.huffman_tables[0].0)?;

        self.writer
            .write_huffman_segment(CodingClass::Ac, 0, &self.huffman_tables[0].1)?;

        if image.get_jpeg_color_type().get_num_components() >= 3 {
            self.writer
                .write_huffman_segment(CodingClass::Dc, 1, &self.huffman_tables[1].0)?;

            self.writer
                .write_huffman_segment(CodingClass::Ac, 1, &self.huffman_tables[1].1)?;
        }

        if let Some(restart_interval) = self.restart_interval {
            self.writer.write_dri(restart_interval)?;
        }

        Ok(())
    }

    fn init_rows(&mut self, buffer_size: usize) -> [Vec<u8>; 4] {
        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new(),
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
    }

    /// Encode all components with one scan
    ///
    /// This is only valid for sampling factors of 1 and 2
    fn encode_image_interleaved<I: ImageBuffer, OP: Operations>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 2],
        adapt_quant_field: Option<&[f32]>,
    ) -> Result<(), EncodingError> {
        self.write_frame_header(&image, q_tables)?;
        self.writer
            .write_scan_header(&self.components.iter().collect::<Vec<_>>(), None)?;

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let width = image.width();
        let height = image.height();

        let num_cols = ceil_div(usize::from(width), 8 * max_h_sampling);
        let num_rows = ceil_div(usize::from(height), 8 * max_v_sampling);

        let buffer_width = num_cols * 8 * max_h_sampling;
        let buffer_size = buffer_width * 8 * max_v_sampling;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        let mut prev_dc = [0i16; 4];

        let restart_interval = self.restart_interval.unwrap_or(0);
        let mut restarts = 0;
        let mut restarts_to_go = restart_interval;

        for block_y in 0..num_rows {
            for r in &mut row {
                r.clear();
            }

            for y in 0..(8 * max_v_sampling) {
                let y = y + block_y * 8 * max_v_sampling;
                let y = (y.min(height as usize - 1)) as u16;

                image.fill_buffers(y, &mut row);

                for _ in usize::from(width)..buffer_width {
                    for channel in &mut row {
                        if !channel.is_empty() {
                            channel.push(channel[channel.len() - 1]);
                        }
                    }
                }
            }

            for block_x in 0..num_cols {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc[0] = 0;
                    prev_dc[1] = 0;
                    prev_dc[2] = 0;
                    prev_dc[3] = 0;
                }

                for (i, component) in self.components.iter().enumerate() {
                    for v_offset in 0..component.vertical_sampling_factor as usize {
                        for h_offset in 0..component.horizontal_sampling_factor as usize {
                            let mut block = get_block(
                                &row[i],
                                block_x * 8 * max_h_sampling + (h_offset * 8),
                                v_offset * 8,
                                max_h_sampling
                                    / component.horizontal_sampling_factor as usize,
                                max_v_sampling
                                    / component.vertical_sampling_factor as usize,
                                buffer_width,
                            );

                            OP::fdct(&mut block);

                            let mut q_block = [0i16; 64];

                            let comp_h_blocks = ceil_div(usize::from(width), 8 * (max_h_sampling / component.horizontal_sampling_factor as usize));
                            let block_idx_in_comp = (block_y * component.vertical_sampling_factor as usize + v_offset) * comp_h_blocks +
                                                    (block_x * component.horizontal_sampling_factor as usize + h_offset);

                            OP::quantize_block(
                                &block,
                                &mut q_block,
                                &q_tables[component.quantization_table as usize],
                                adapt_quant_field,
                                block_idx_in_comp,
                                &self.zero_bias_offsets[i],
                                &self.zero_bias_multipliers[i],
                            );

                            self.writer.write_block(
                                &q_block,
                                prev_dc[i],
                                &self.huffman_tables[component.dc_huffman_table as usize].0,
                                &self.huffman_tables[component.ac_huffman_table as usize].1,
                            )?;

                            prev_dc[i] = q_block[0];
                        }
                    }
                }

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }
        }

        self.writer.finalize_bit_buffer()?;

        Ok(())
    }

    /// Encode components with one scan per component
    fn encode_image_sequential<I: ImageBuffer, OP: Operations>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 2],
        adapt_quant_field: Option<&[f32]>,
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks::<_, OP>(&image, q_tables, adapt_quant_field);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image, q_tables)?;

        for (i, component) in self.components.iter().enumerate() {
            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;

            self.writer.write_scan_header(&[component], None)?;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc = 0;
                }

                self.writer.write_block(
                    block,
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                    &self.huffman_tables[component.ac_huffman_table as usize].1,
                )?;

                prev_dc = block[0];

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }

            self.writer.finalize_bit_buffer()?;
        }

        Ok(())
    }

    /// Encode image in progressive mode
    ///
    /// This only support spectral selection for now
    fn encode_image_progressive<I: ImageBuffer, OP: Operations>(
        &mut self,
        image: I,
        scans: u8,
        q_tables: &[QuantizationTable; 2],
        adapt_quant_field: Option<&[f32]>,
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks::<_, OP>(&image, q_tables, adapt_quant_field);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image, q_tables)?;

        // Phase 1: DC Scan
        //          Only the DC coefficients can be transfer in the first component scans
        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component], Some((0, 0)))?;

            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc = 0;
                }

                self.writer.write_dc(
                    block[0],
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                )?;

                prev_dc = block[0];

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }

            self.writer.finalize_bit_buffer()?;
        }

        // Phase 2: AC scans
        let scans = scans as usize - 1;

        let values_per_scan = 64 / scans;

        for scan in 0..scans {
            let start = (scan * values_per_scan).max(1);
            let end = if scan == scans - 1 {
                // ensure last scan is always transfers the remaining coefficients
                64
            } else {
                (scan + 1) * values_per_scan
            };

            for (i, component) in self.components.iter().enumerate() {
                let restart_interval = self.restart_interval.unwrap_or(0);
                let mut restarts = 0;
                let mut restarts_to_go = restart_interval;

                self.writer
                    .write_scan_header(&[component], Some((start as u8, end as u8 - 1)))?;

                for block in &blocks[i] {
                    if restart_interval > 0 && restarts_to_go == 0 {
                        self.writer.finalize_bit_buffer()?;
                        self.writer
                            .write_marker(Marker::RST((restarts % 8) as u8))?;
                    }

                    self.writer.write_ac_block(
                        block,
                        start,
                        end,
                        &self.huffman_tables[component.ac_huffman_table as usize].1,
                    )?;

                    if restart_interval > 0 {
                        if restarts_to_go == 0 {
                            restarts_to_go = restart_interval;
                            restarts += 1;
                            restarts &= 7;
                        }
                        restarts_to_go -= 1;
                    }
                }

                self.writer.finalize_bit_buffer()?;
            }
        }

        Ok(())
    }

    fn encode_blocks<I: ImageBuffer, OP: Operations>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 2],
        adapt_quant_field: Option<&[f32]>,
    ) -> [Vec<[i16; 64]>; 4] {
        let width = image.width();
        let height = image.height();

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let num_cols = ceil_div(usize::from(width), 8 * max_h_sampling) * max_h_sampling;
        let num_rows = ceil_div(usize::from(height), 8 * max_v_sampling) * max_v_sampling;

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        let buffer_width = num_cols * 8;
        let buffer_size = num_cols * num_rows * 64;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        for y in 0..num_rows * 8 {
            let y = (y.min(usize::from(height) - 1)) as u16;

            image.fill_buffers(y, &mut row);

            for _ in usize::from(width)..num_cols * 8 {
                for channel in &mut row {
                    if !channel.is_empty() {
                        channel.push(channel[channel.len() - 1]);
                    }
                }
            }
        }

        let num_cols = ceil_div(usize::from(width), 8);
        let num_rows = ceil_div(usize::from(height), 8);

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        let mut blocks: [Vec<_>; 4] = self.init_block_buffers(buffer_size / 64);

        for (i, component) in self.components.iter().enumerate() {
            let h_scale = max_h_sampling / component.horizontal_sampling_factor as usize;
            let v_scale = max_v_sampling / component.vertical_sampling_factor as usize;

            let cols = ceil_div(num_cols, h_scale);
            let rows = ceil_div(num_rows, v_scale);

            debug_assert!(cols > 0);
            debug_assert!(rows > 0);

            for block_y in 0..rows {
                for block_x in 0..cols {
                    let mut block = get_block(
                        &row[i],
                        block_x * 8 * h_scale,
                        block_y * 8 * v_scale,
                        h_scale,
                        v_scale,
                        buffer_width,
                    );

                    OP::fdct(&mut block);

                    let mut q_block = [0i16; 64];

                    let comp_h_blocks = ceil_div(usize::from(width), 8 * (max_h_sampling / component.horizontal_sampling_factor as usize));
                    let block_idx_in_comp = (block_y * component.vertical_sampling_factor as usize + v_scale) * comp_h_blocks +
                                            (block_x * component.horizontal_sampling_factor as usize + h_scale);

                    OP::quantize_block(
                        &block,
                        &mut q_block,
                        &q_tables[component.quantization_table as usize],
                        adapt_quant_field,
                        block_idx_in_comp,
                        &self.zero_bias_offsets[i],
                        &self.zero_bias_multipliers[i],
                    );

                    blocks[i].push(q_block);
                }
            }
        }
        blocks
    }

    fn init_block_buffers(&mut self, buffer_size: usize) -> [Vec<[i16; 64]>; 4] {
        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new(),
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
    }

    // Create new huffman tables optimized for this image
    fn optimize_huffman_table(&mut self, blocks: &[Vec<[i16; 64]>; 4]) {
        // TODO: Find out if it's possible to reuse some code from the writer

        let max_tables = self.components.len().min(2) as u8;

        for table in 0..max_tables {
            let mut dc_freq = [0u32; 257];
            dc_freq[256] = 1;
            let mut ac_freq = [0u32; 257];
            ac_freq[256] = 1;

            let mut had_ac = false;
            let mut had_dc = false;

            for (i, component) in self.components.iter().enumerate() {
                if component.dc_huffman_table == table {
                    had_dc = true;

                    let mut prev_dc = 0;

                    debug_assert!(!blocks[i].is_empty());

                    for block in &blocks[i] {
                        let value = block[0];
                        let diff = value - prev_dc;
                        let num_bits = get_num_bits(diff);

                        dc_freq[num_bits as usize] += 1;

                        prev_dc = value;
                    }
                }

                if component.ac_huffman_table == table {
                    had_ac = true;

                    if let Some(scans) = self.progressive_scans {
                        let scans = scans as usize - 1;

                        let values_per_scan = 64 / scans;

                        for scan in 0..scans {
                            let start = (scan * values_per_scan).max(1);
                            let end = if scan == scans - 1 {
                                // Due to rounding we might need to transfer more than values_per_scan values in the last scan
                                64
                            } else {
                                (scan + 1) * values_per_scan
                            };

                            debug_assert!(!blocks[i].is_empty());

                            for block in &blocks[i] {
                                let mut zero_run = 0;

                                for &value in &block[start..end] {
                                    if value == 0 {
                                        zero_run += 1;
                                    } else {
                                        while zero_run > 15 {
                                            ac_freq[0xF0] += 1;
                                            zero_run -= 16;
                                        }
                                        let num_bits = get_num_bits(value);
                                        let symbol = (zero_run << 4) | num_bits;

                                        ac_freq[symbol as usize] += 1;

                                        zero_run = 0;
                                    }
                                }

                                if zero_run > 0 {
                                    ac_freq[0] += 1;
                                }
                            }
                        }
                    } else {
                        for block in &blocks[i] {
                            let mut zero_run = 0;

                            for &value in &block[1..] {
                                if value == 0 {
                                    zero_run += 1;
                                } else {
                                    while zero_run > 15 {
                                        ac_freq[0xF0] += 1;
                                        zero_run -= 16;
                                    }
                                    let num_bits = get_num_bits(value);
                                    let symbol = (zero_run << 4) | num_bits;

                                    ac_freq[symbol as usize] += 1;

                                    zero_run = 0;
                                }
                            }

                            if zero_run > 0 {
                                ac_freq[0] += 1;
                            }
                        }
                    }
                }
            }

            assert!(had_dc, "Missing DC data for table {}", table);
            assert!(had_ac, "Missing AC data for table {}", table);

            self.huffman_tables[table as usize] = (
                HuffmanTable::new_optimized(dc_freq),
                HuffmanTable::new_optimized(ac_freq),
            );
        }
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
    fn quantize_block(
        block: &[i16; 64],
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
            let value = block[z] as i32;
            let q_val = table.get_raw(z) as i32;
            let divisor = q_val << SHIFT;

            // Ensure divisor is not zero, though table values should be >= 1
            if divisor == 0 { continue; }

            // Standard quantization with rounding
            let mut q_coeff = (value + divisor.copysign(value) / 2) / divisor;

            // Jpegli-style adaptive quantization thresholding (approximated)
            // Apply only to AC coefficients (i > 0)
            if i > 0 && q_coeff != 0 && adapt_quant_field.is_some() {
                // Use the precomputed tables passed as arguments.
                let zb_offset = zero_bias_offset[i]; // Index `i` corresponds to coefficient k
                let zb_mul = zero_bias_mul[i];
                
                // Calculate threshold: threshold = zb_offset + zb_mul * aq_strength
                let threshold = zb_offset + zb_mul * aq_strength;

                // Value to compare: Abs(original_coeff * 8.0 / q_val)
                // Use block[z] which is the original coefficient before FDCT scaling adjustment.
                let abs_unquant_scaled = (block[z] as f32 * (8.0 / q_val as f32)).abs();

                // Zero out coefficient if its scaled value is below the adaptive threshold
                if abs_unquant_scaled < threshold {
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

    use crate::encoder::get_num_bits;
    use crate::writer::get_code;
    use crate::{Encoder, SamplingFactor};

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
    fn sampling_factors() {
        assert_eq!(SamplingFactor::F_1_1.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::F_2_1.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::F_1_2.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::F_2_2.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::F_4_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::F_4_2.get_sampling_factors(), (4, 2));
        assert_eq!(SamplingFactor::F_1_4.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::F_2_4.get_sampling_factors(), (2, 4));

        assert_eq!(SamplingFactor::R_4_4_4.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::R_4_4_0.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::R_4_4_1.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::R_4_2_2.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::R_4_2_0.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::R_4_2_1.get_sampling_factors(), (2, 4));
        assert_eq!(SamplingFactor::R_4_1_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::R_4_1_0.get_sampling_factors(), (4, 2));
    }

    #[test]
    fn test_set_progressive() {
        let mut encoder = Encoder::new(vec![], 100);
        encoder.set_progressive(true);
        assert_eq!(encoder.progressive_scans(), Some(4));

        encoder.set_progressive(false);
        assert_eq!(encoder.progressive_scans(), None);
    }
}
