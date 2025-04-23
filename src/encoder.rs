use crate::{Density};
use crate::huffman::*;
use crate::quantization::QuantizationTable;

use crate::error::*;

use crate::huffman::{CodingClass};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::quantization::{QuantizationTableType};
use crate::writer::{JfifWrite, JfifWriter, ZIGZAG};
use crate::{ EncodingError};
use alloc::format;
#[cfg(feature = "jpegli")]
use crate::jpegli::fdct_jpegli::forward_dct_float;
#[cfg(feature = "jpegli")]
use crate::jpegli::adaptive_quantization::compute_adaptive_quant_field;

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::io::{BufWriter, Write};

#[cfg(feature = "std")]
use std::fs::File;

#[cfg(feature = "std")]
use std::path::Path;

#[cfg(feature = "jpegli")]
use crate::jpegli::JpegliConfig;

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

// Define the core operations trait (object safety not required anymore)
// We can simplify this trait - the enum impl will handle Jpegli logic.
pub(crate) trait Operations {
    fn fdct(&self, data: &mut [i16; 64]);
}

// Default (Scalar) Implementation
#[derive(Clone)]
pub(crate) struct DefaultOperations;
impl Operations for DefaultOperations {
    #[inline(always)]
    fn fdct(&self, data: &mut [i16; 64]) {
        crate::fdct::fdct(data);
    }
}

// AVX2 Implementation Struct (Definition likely in src/avx2.rs)
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub(crate) use crate::avx2::AVX2Operations;

// Enum for Dispatch
#[derive(Clone)] // If Default/AVX2Operations are Clone
pub(crate) enum OperationsImpl {
    Default(DefaultOperations),
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
    Avx2(AVX2Operations),
    // Add Neon variant later if needed
}

// Implement the core logic dispatch via the enum
impl OperationsImpl {
    // FDCT dispatch
    fn fdct(&self, data: &mut [i16; 64]) {
        match self {
            OperationsImpl::Default(op) => op.fdct(data),
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
            OperationsImpl::Avx2(op) => op.fdct(data),
        }
    }
}

/// # The JPEG encoder
pub struct Encoder<W: JfifWrite> {
    writer: JfifWriter<W>,
    density: Density,
    quality: u8,

    components: Vec<Component>,
    quantization_tables: [QuantizationTableType; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

    sampling_factor: SamplingFactor,

    progressive_scans: Option<u8>,

    restart_interval: Option<u16>,

    optimize_huffman_table: bool,

    app_segments: Vec<(u8, Vec<u8>)>,

    // Jpegli specific config
    #[cfg(feature = "jpegli")]
    pub(crate) jpegli_config: Option<JpegliConfig>,

    // Store the operations enum directly
    operations: OperationsImpl,
}

impl<W: JfifWrite> Encoder<W> {
    /// Create a new encoder with the given quality
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

        let quantization_tables = [
            QuantizationTableType::Default,
            QuantizationTableType::Default,
        ];

        let sampling_factor = if quality < 90 {
            SamplingFactor::F_2_2
        } else {
            SamplingFactor::F_1_1
        };

        // Select operations implementation at runtime
        let operations = match () {
                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                    _ if std::is_x86_feature_detected!("avx2")=> {
                        OperationsImpl::Avx2(AVX2Operations) // Assuming AVX2Operations is constructible
                    },
                
                _ => OperationsImpl::Default(DefaultOperations)

        };

        Encoder {
            writer: JfifWriter::new(w),
            density: Density::None,
            quality,
            components: vec![],
            quantization_tables,
            huffman_tables,
            sampling_factor,
            progressive_scans: None,
            restart_interval: None,
            optimize_huffman_table: false,
            app_segments: Vec::new(),
            #[cfg(feature = "jpegli")]
            jpegli_config: None,
            operations, // Store the selected operations enum variant
        }
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

    /// Configures the encoder to use Jpegli algorithms with the specified distance.
    ///
    /// This enables Jpegli-specific quantization and potentially other features like
    /// float DCT and adaptive quantization (controlled separately).
    /// Calling this replaces any previous Jpegli configuration.
    #[cfg(feature = "jpegli")]
    pub fn configure_jpegli(&mut self, distance: f32, use_float_dct: Option<bool>, use_adaptive_quantization: Option<bool>) {
        let distance = distance.max(0.0);
        // We create the config here, but it might be recreated in encode_internal if num_components differs.
        // This requires JpegliConfig::new to be robust or encode_internal to handle it.
        let num_components_guess = if self.components.is_empty() { 3 } else { self.components.len() };
        let mut config = JpegliConfig::new(distance, self.sampling_factor, num_components_guess);

        if let Some(enable) = use_float_dct {
            config.set_float_dct(enable);
        }
        if let Some(enable) = use_adaptive_quantization {
            config.set_adaptive_quantization(enable);
        }
        
        self.jpegli_config = Some(config);
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
        mut self,
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

        // Runtime selection of ImageBuffer based on ColorType and potential SIMD
        // We now pass the selected ImageBuffer to encode_image_internal directly.
        // Remove the old encode_image method and the SIMD dispatch block here.

        match color_type {
            ColorType::Luma => self.encode_image_internal(GrayImage(data, width, height)),
            ColorType::Rgb => {
                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                {
                    if std::is_x86_feature_detected!("avx2") {
                       return self.encode_image_internal(crate::avx2::RgbImageAVX2(data, width, height));
                    }
                }
                self.encode_image_internal(RgbImage(data, width, height))
            }
            ColorType::Rgba => {
                 #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                {
                    if std::is_x86_feature_detected!("avx2") {
                       return self.encode_image_internal(crate::avx2::RgbaImageAVX2(data, width, height));
                    }
                }
                self.encode_image_internal(RgbaImage(data, width, height))
            }
             ColorType::Bgr => {
                 #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                {
                    if std::is_x86_feature_detected!("avx2") {
                       return self.encode_image_internal(crate::avx2::BgrImageAVX2(data, width, height));
                    }
                }
                self.encode_image_internal(BgrImage(data, width, height))
            }
             ColorType::Bgra => {
                 #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                {
                    if std::is_x86_feature_detected!("avx2") {
                       return self.encode_image_internal(crate::avx2::BgraImageAVX2(data, width, height));
                    }
                }
                self.encode_image_internal(BgraImage(data, width, height))
            }
            ColorType::Ycbcr => self.encode_image_internal(YCbCrImage(data, width, height)),
            ColorType::Cmyk => self.encode_image_internal(CmykImage(data, width, height)),
            ColorType::CmykAsYcck => self.encode_image_internal(CmykAsYcckImage(data, width, height)),
            ColorType::Ycck => self.encode_image_internal(YcckImage(data, width, height)),
        }
    }

    // Update encode_image_internal to remove the OP generic parameter
    fn encode_image_internal<I: ImageBuffer>(
        mut self,
        image: I,
    ) -> Result<(), EncodingError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(EncodingError::ZeroImageDimensions {
                width: image.width(),
                height: image.height(),
            });
        }

        let jpeg_color_type = image.get_jpeg_color_type();
        self.init_components(jpeg_color_type);
        let num_components = jpeg_color_type.get_num_components();

        // --- Jpegli Configuration Finalization --- 
        #[cfg(feature = "jpegli")]
        {
            if let Some(config) = &mut self.jpegli_config {
                if config.zero_bias_offsets.len() != num_components {
                    *config = JpegliConfig::new(config.distance, self.sampling_factor, num_components);
                }
                 if config.use_adaptive_quantization {
                     // TODO: Implement AQ field calculation using image.get_adaptive_quant_channel()
                     // Need to add get_adaptive_quant_channel() to ImageBuffer trait if not present
                     // and implement it for relevant image types.
                     // let aq_field = compute_adaptive_quant_field(&luma_data, image.width(), image.height(), config.distance);
                     // config.adaptive_quant_field = Some(aq_field);
                 }
            }
        }

        // Generate standard Quantization tables (used as base or fallback)
        let standard_q_tables = [
            QuantizationTable::new_with_quality(&self.quantization_tables[0], self.quality, true),
            QuantizationTable::new_with_quality(&self.quantization_tables[1], self.quality, false),
        ];

        self.writer.write_marker(Marker::SOI)?;
        self.writer.write_header(&self.density)?;

        // ... (Write APP14, custom APP segments) ...
         if jpeg_color_type == JpegColorType::Cmyk {
            let app_14 = b"Adobe\0\0\0\0\0\0\0";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        } else if jpeg_color_type == JpegColorType::Ycck {
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        }

        for (nr, data) in &self.app_segments {
            self.writer.write_segment(Marker::APP(*nr), data)?;
        }

        // Dispatch to encoding loop based on progressive/interleaved/sequential
        if let Some(scans) = self.progressive_scans {
            self.encode_image_progressive(image, scans, &standard_q_tables)?;
        } else if self.sampling_factor.supports_interleaved() {
             self.encode_image_interleaved(image, &standard_q_tables)?;
        } else {
            self.encode_image_sequential(image, &standard_q_tables)?;
        }

        self.writer.write_marker(Marker::EOI)?;
        Ok(())
    }

    // ... (init_components, get_max_sampling_size, write_frame_header, init_rows) ...
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

        // Always write the standard tables derived from quality/custom setting
        // Jpegli raw tables are not written in DQT but used directly in quantization step.
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

    /// Encode all components with one scan (Interleaved)
    fn encode_image_interleaved<I: ImageBuffer>(
        &mut self,
        image: I,
        standard_q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        self.write_frame_header(&image, standard_q_tables)?;
        self.writer.write_scan_header(&self.components.iter().collect::<Vec<_>>(), None)?;

        let (max_h, max_v) = self.get_max_sampling_size();
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let num_mcu_cols = ceil_div(image.width() as usize, mcu_width);
        let num_mcu_rows = ceil_div(image.height() as usize, mcu_height);

        let mut rows = self.init_rows(mcu_width * mcu_height); // Adjust row buffer size?

        let mut prev_dc = [0i16; 4];
        let mut restart_markers_to_go = self.restart_interval.unwrap_or(0);
        let mut restarts = 0u32; // Add a counter for restart markers

        for mcu_y in 0..num_mcu_rows {
            // Fill buffer for one MCU row
            // This needs adjustment - fill buffer needs to provide enough rows for one MCU height
            for y_offset in 0..mcu_height {
                 let y = (mcu_y * mcu_height + y_offset) as u16;
                 if y < image.height() {
                      image.fill_buffers(y, &mut rows);
                      // Padding logic might be needed here or within fill_buffers/get_block
                 } else {
                      // Handle bottom padding
                 }
            }

            for mcu_x in 0..num_mcu_cols {
                if self.restart_interval.is_some() && restart_markers_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    // Use simple counter for restart index
                    self.writer.write_marker(Marker::RST((restarts % 8) as u8))?;
                    prev_dc = [0i16; 4];
                    restart_markers_to_go = self.restart_interval.unwrap(); // Reset countdown
                    restarts += 1; // Increment counter
                }

                for (comp_idx, component) in self.components.iter().enumerate() {
                    let comp_h = component.horizontal_sampling_factor as usize;
                    let comp_v = component.vertical_sampling_factor as usize;
                    for v_block in 0..comp_v {
                        for h_block in 0..comp_h {
                            let block_x_img = mcu_x * mcu_width + h_block * 8 * (max_h / comp_h);
                            let block_y_img = mcu_y * mcu_height + v_block * 8 * (max_v / comp_v);
                            // Need a robust get_block that handles image boundaries and uses MCU row buffer
                            let mut block = get_block_from_mcu_buffer(
                                &rows[comp_idx],
                                h_block * 8, // x within component's part of MCU buffer
                                v_block * 8, // y within component's part of MCU buffer
                                max_h / comp_h, // stride within component data
                                max_v / comp_v, // stride within component data
                                mcu_width * (comp_h as usize/ max_h), // width of component's part of MCU buffer
                                image.width(), image.height(), // Original image dimensions
                                block_x_img, block_y_img // Global block coords for padding
                            );

                            let use_float_dct = cfg!(feature = "jpegli") && self.jpegli_config.as_ref().map_or(false, |c| c.use_float_dct);

                            if use_float_dct {
                                let mut block_f32 = [0.0f32; 64];
                                let mut coeffs_f32 = [0.0f32; 64];
                                let mut scratch_f32 = [0.0f32; 64];
                                for i in 0..64 { block_f32[i] = block[i] as f32; }
                                crate::jpegli::fdct_jpegli::forward_dct_float(&block_f32, &mut coeffs_f32, &mut scratch_f32);
                                for i in 0..64 { block[i] = coeffs_f32[i].round() as i16; }
                            } else {
                                self.operations.fdct(&mut block);
                            }

                            let mut q_block = [0i16; 64];
                            // --- Quantization Dispatch --- 
                            if cfg!(feature = "jpegli") && self.jpegli_config.is_some() {
                                self.jpegli_quantize_block(
                                    &block,
                                    &mut q_block,
                                    component.quantization_table,
                                    mcu_x, // Pass MCU coords for potential AQ lookup
                                    mcu_y,
                                )?;
                            } else {
                                Self::standard_quantize_block(
                                    &block,
                                    &mut q_block,
                                    &standard_q_tables[component.quantization_table as usize],
                                    mcu_x,
                                    mcu_y,
                                );
                            }
                            // --- End Quantization --- 

                            self.writer.write_block(
                                &q_block,
                                prev_dc[comp_idx],
                                &self.huffman_tables[component.dc_huffman_table as usize].0,
                                &self.huffman_tables[component.ac_huffman_table as usize].1,
                            )?;
                            prev_dc[comp_idx] = q_block[0];
                        }
                    }
                }
                // Decrement countdown after processing the MCU
                if self.restart_interval.is_some() { 
                    restart_markers_to_go -= 1;
                }
            }
        }
        self.writer.finalize_bit_buffer()?;
        Ok(())
    }

    /// Encode components with one scan per component (Sequential)
    fn encode_image_sequential<I: ImageBuffer>(
        &mut self,
        image: I,
        standard_q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks(&image, standard_q_tables)?;

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }
        self.write_frame_header(&image, standard_q_tables)?;

        for (i, component) in self.components.iter().enumerate() {
             let restart_interval = self.restart_interval.unwrap_or(0);
             let mut restarts = 0;
             let mut restarts_to_go = restart_interval;
             self.writer.write_scan_header(&[component], None)?;
             let mut prev_dc = 0;

             for block in &blocks[i] {
                 if restart_interval > 0 && restarts_to_go == 0 {
                     self.writer.finalize_bit_buffer()?;
                     self.writer.write_marker(Marker::RST((restarts % 8) as u8))?;
                     prev_dc = 0;
                     restarts_to_go = restart_interval;
                     restarts += 1;
                 }
                 self.writer.write_block(
                     block,
                     prev_dc,
                     &self.huffman_tables[component.dc_huffman_table as usize].0,
                     &self.huffman_tables[component.ac_huffman_table as usize].1,
                 )?;
                 prev_dc = block[0];
                 if restart_interval > 0 { restarts_to_go -= 1; }
             }
             self.writer.finalize_bit_buffer()?;
        }
        Ok(())
    }

    /// Encode image in progressive mode
    fn encode_image_progressive<I: ImageBuffer>(
        &mut self,
        image: I,
        scans: u8,
        standard_q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks(&image, standard_q_tables)?;

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }
        self.write_frame_header(&image, standard_q_tables)?;

        // Phase 1: DC Scan
        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component], Some((0, 0)))?;
            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;
            let mut prev_dc = 0;

            for block in &blocks[i] {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer.write_marker(Marker::RST((restarts % 8) as u8))?;
                    prev_dc = 0;
                    restarts_to_go = restart_interval;
                    restarts += 1;
                }
                self.writer.write_dc(
                    block[0],
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                )?;
                prev_dc = block[0];
                if restart_interval > 0 { restarts_to_go -= 1; }
            }
            self.writer.finalize_bit_buffer()?;
        }

        // Phase 2: AC scans
        let scans = scans as usize - 1;
        let values_per_scan = 64 / scans;

        for scan in 0..scans {
            let start = (scan * values_per_scan).max(1);
            let end = if scan == scans - 1 { 64 } else { (scan + 1) * values_per_scan };

            for (i, component) in self.components.iter().enumerate() {
                let restart_interval = self.restart_interval.unwrap_or(0);
                let mut restarts = 0;
                let mut restarts_to_go = restart_interval;
                self.writer.write_scan_header(&[component], Some((start as u8, end as u8 - 1)))?;

                for block in &blocks[i] {
                    if restart_interval > 0 && restarts_to_go == 0 {
                        self.writer.finalize_bit_buffer()?;
                        self.writer.write_marker(Marker::RST((restarts % 8) as u8))?;
                        restarts_to_go = restart_interval;
                        restarts += 1;
                    }
                    self.writer.write_ac_block(
                        block,
                        start,
                        end,
                        &self.huffman_tables[component.ac_huffman_table as usize].1,
                    )?;
                    if restart_interval > 0 { restarts_to_go -= 1; }
                }
                self.writer.finalize_bit_buffer()?;
            }
        }
        Ok(())
    }

    // Update encode_blocks to remove OP generic and use self.operations
    fn encode_blocks<I: ImageBuffer>(
        &mut self,
        image: &I,
        standard_q_tables: &[QuantizationTable; 2],
    ) -> Result<[Vec<[i16; 64]>; 4], EncodingError> {
        let (max_h, max_v) = self.get_max_sampling_size();
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let num_mcu_cols = ceil_div(image.width() as usize, mcu_width);
        let num_mcu_rows = ceil_div(image.height() as usize, mcu_height);
        let total_mcus = num_mcu_cols * num_mcu_rows;

        let mut blocks = self.init_block_buffers(total_mcus); // Size hint based on MCUs

        let mut rows = self.init_rows(mcu_width * mcu_height); // Buffer for one MCU row

        for mcu_y in 0..num_mcu_rows {
             // Fill buffer for one MCU row
             for y_offset in 0..mcu_height {
                 let y = (mcu_y * mcu_height + y_offset) as u16;
                 if y < image.height() {
                      image.fill_buffers(y, &mut rows);
                      // TODO: Padding logic needs refinement
                 } else {
                      // Handle bottom padding
                 }
             }

            for mcu_x in 0..num_mcu_cols {
                for (comp_idx, component) in self.components.iter().enumerate() {
                    let comp_h = component.horizontal_sampling_factor as usize;
                    let comp_v = component.vertical_sampling_factor as usize;
                     for v_block in 0..comp_v {
                         for h_block in 0..comp_h {
                             let block_x_img = mcu_x * mcu_width + h_block * 8 * (max_h / comp_h);
                             let block_y_img = mcu_y * mcu_height + v_block * 8 * (max_v / comp_v);
                             let mut block = get_block_from_mcu_buffer(
                                 &rows[comp_idx],
                                 h_block * 8,
                                 v_block * 8,
                                 max_h / comp_h,
                                 max_v / comp_v,
                                 mcu_width * (comp_h as usize / max_h),
                                 image.width(), image.height(),
                                 block_x_img, block_y_img
                             );

                            let use_float_dct = cfg!(feature = "jpegli") && self.jpegli_config.as_ref().map_or(false, |c| c.use_float_dct);

                            if use_float_dct {
                                let mut block_f32 = [0.0f32; 64];
                                let mut coeffs_f32 = [0.0f32; 64];
                                let mut scratch_f32 = [0.0f32; 64];
                                for i in 0..64 { block_f32[i] = block[i] as f32; }
                                crate::jpegli::fdct_jpegli::forward_dct_float(&block_f32, &mut coeffs_f32, &mut scratch_f32);
                                for i in 0..64 { block[i] = coeffs_f32[i].round() as i16; }
                            } else {
                                self.operations.fdct(&mut block);
                            }

                            let mut q_block = [0i16; 64];
                            // --- Quantization Dispatch --- 
                            if cfg!(feature = "jpegli") && self.jpegli_config.is_some() {
                                self.jpegli_quantize_block(
                                    &block,
                                    &mut q_block,
                                    component.quantization_table,
                                    mcu_x, // Pass MCU coords for potential AQ lookup
                                    mcu_y,
                                )?;
                            } else {
                                Self::standard_quantize_block(
                                    &block,
                                    &mut q_block,
                                    &standard_q_tables[component.quantization_table as usize],
                                    mcu_x,
                                    mcu_y,
                                );
                            }
                            // --- End Quantization --- 

                            blocks[comp_idx].push(q_block);
                        }
                    }
                }
            }
        }
        Ok(blocks)
    }

    // ... (init_block_buffers, optimize_huffman_table) ...
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
                Vec::with_capacity(buffer_size / (self.sampling_factor.get_sampling_factors().0 as usize * self.sampling_factor.get_sampling_factors().1 as usize)),
                Vec::with_capacity(buffer_size / (self.sampling_factor.get_sampling_factors().0 as usize * self.sampling_factor.get_sampling_factors().1 as usize)),
                Vec::new(),
            ],
            4 => [
                // Estimate based on typical YCCK/CMYK sampling
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size / (self.sampling_factor.get_sampling_factors().0 as usize * self.sampling_factor.get_sampling_factors().1 as usize)),
                Vec::with_capacity(buffer_size / (self.sampling_factor.get_sampling_factors().0 as usize * self.sampling_factor.get_sampling_factors().1 as usize)),
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

                    let mut prev_dc: i32 = 0;

                    debug_assert!(!blocks[i].is_empty());

                    for block in &blocks[i] {
                        let value = block[0] as i32;
                        let diff_i32 = value - prev_dc;

                        let diff_i16 = diff_i32.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

                        let num_bits = get_num_bits(diff_i16);

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
                                let mut zero_run: i32 = 0;

                                for &value in &block[start..end] {
                                    if value == 0 {
                                        zero_run += 1;
                                    } else {
                                        while zero_run > 15 {
                                            ac_freq[0xF0] += 1;
                                            zero_run -= 16;
                                        }
                                        let num_bits = get_num_bits(value);
                                        // Cast zero_run (i32) to u8 before bitwise OR
                                        let symbol = ((zero_run as u8) << 4) | num_bits;

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
                            let mut zero_run: i32 = 0;

                            for &value in &block[1..] {
                                if value == 0 {
                                    zero_run += 1;
                                } else {
                                    while zero_run > 15 {
                                        ac_freq[0xF0] += 1;
                                        zero_run -= 16;
                                    }
                                    let num_bits = get_num_bits(value);
                                    // Cast zero_run (i32) to u8 before bitwise OR
                                    let symbol = ((zero_run as u8) << 4) | num_bits;

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

    // --- Standard Quantization Helper --- 
    #[inline(always)]
    fn standard_quantize_block(
        block: &[i16; 64],
        q_block: &mut [i16; 64],
        table: &QuantizationTable,
        _block_x: usize, // Keep for potential future use (e.g., AVX2 standard quant)
        _block_y: usize,
    ) {
        // TODO: Consider AVX2 version of standard quantization here if needed.
        for i in 0..64 {
            let z = ZIGZAG[i] as usize & 0x3f;
            q_block[i] = table.quantize(block[z], z);
        }
    }

    // --- Jpegli Quantization Helper --- 
    #[cfg(feature = "jpegli")]
    #[inline(always)]
    fn jpegli_quantize_block(
        &self,
        block: &[i16; 64],
        q_block: &mut [i16; 64],
        component_q_table_index: u8,
        block_x: usize, // Retained for AQ field lookup
        block_y: usize,
    ) -> Result<(), EncodingError> { // Return Result for safety
        // This assumes self.jpegli_config is Some, checked at call site
        let config = self.jpegli_config.as_ref().unwrap();
        
        let raw_q_table = if component_q_table_index == 0 {
            &config.luma_table_raw
        } else {
            &config.chroma_table_raw
        };
        let bias_offsets = &config.zero_bias_offsets;
        let bias_multipliers = &config.zero_bias_multipliers;
        let aq_field = config.adaptive_quant_field.as_deref();

        let bias_component_index = match self.components[component_q_table_index as usize].id {
            0 => 0, 1 => 1, 2 => 2,
            3 => if self.components.len() == 4 { 0 } else { 1 }, // Basic YCCK/CMYK assumption
            _ => component_q_table_index as usize,
        };

        if bias_component_index >= bias_offsets.len() || bias_component_index >= bias_multipliers.len() {
            // This should ideally not happen if config is created correctly, but handle defensively.
             return Err(EncodingError::JpegliError(
                 format!("Jpegli bias table index out of bounds: index {}, len {}", 
                         bias_component_index, bias_offsets.len())
             ));
        }

        let aq_mult = if config.use_adaptive_quantization {
            if let Some(_field) = aq_field {
                // TODO: Proper AQ field lookup using block_x, block_y and MCU grid info
                1.0
            } else { 1.0 }
        } else { 1.0 };

        let component_offsets = &bias_offsets[bias_component_index];
        let component_multipliers = &bias_multipliers[bias_component_index];

        // Perform the actual Jpegli quantization (scalar for now)
        // TODO: Add SIMD dispatch for Jpegli quantization here if needed later
        for i in 0..64 {
            let z = ZIGZAG[i] as usize & 0x3f;
            let dct_coeff = block[z] as f32;
            if raw_q_table[z] == 0 { q_block[i] = 0; continue; }
            let inv_quant_step = 1.0 / (raw_q_table[z] as f32);
            let zero_bias = component_offsets[z] * component_multipliers[z];
            let quantized_f32 = (dct_coeff * inv_quant_step * aq_mult) + zero_bias;
            q_block[i] = quantized_f32.round() as i16;
        }
        Ok(() as ())
    }
}

// ... (Encoder<BufWriter<File>> impl) ...
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

// get_block needs significant update to work with MCU row buffers and padding
fn get_block_from_mcu_buffer(
    component_mcu_buffer: &[u8],
    block_x_in_comp_mcu: usize, // X coord of 8x8 block within this component's part of MCU buffer
    block_y_in_comp_mcu: usize, // Y coord of 8x8 block within this component's part of MCU buffer
    _h_stride: usize, // Stride between pixels if subsampled (often 1 after extraction)
    _v_stride: usize, // Stride between pixels if subsampled (often 1 after extraction)
    comp_mcu_buffer_width: usize, // Width of this component's data within the MCU buffer
    img_width: u16, img_height: u16, // Original image dimensions
    global_block_x: usize, global_block_y: usize // Top-left coords of this block in original image
) -> [i16; 64] {
    let mut block = [0i16; 64];
    let img_width = img_width as usize;
    let img_height = img_height as usize;

    for y in 0..8 {
        for x in 0..8 {
            let src_x = global_block_x + x; // Use global coords for boundary check
            let src_y = global_block_y + y;

            let val = if src_x >= img_width || src_y >= img_height {
                // Handle padding: Repeat edge pixels by clamping coordinates
                let clamped_global_x = src_x.clamp(0, img_width - 1);
                let clamped_global_y = src_y.clamp(0, img_height - 1);

                // Calculate the local offset (0-7) within the block for the clamped edge pixel.
                // Use saturating_sub to prevent panic if global_block_x > clamped_global_x (shouldn't happen?)
                let local_edge_x = clamped_global_x.saturating_sub(global_block_x);
                let local_edge_y = clamped_global_y.saturating_sub(global_block_y);

                // Calculate index using the block's base offset + the local edge offset.
                 component_mcu_buffer[(block_y_in_comp_mcu + local_edge_y) * comp_mcu_buffer_width + (block_x_in_comp_mcu + local_edge_x)]
            } else {
                // Calculate index in the component_mcu_buffer based on local coords
                 component_mcu_buffer[(block_y_in_comp_mcu + y) * comp_mcu_buffer_width + (block_x_in_comp_mcu + x)]
            };
            block[y * 8 + x] = (val as i16) - 128;
        }
    }
    block
}

// ... (ceil_div, get_num_bits) ...
fn ceil_div(value: usize, div: usize) -> usize {
    value / div + usize::from(value % div != 0)
}

fn get_num_bits(value: i16) -> u8 {
    // Use unsigned_abs() to safely handle i16::MIN
    let abs_value = value.unsigned_abs();

    let mut num_bits = 0;
    let mut temp_val = abs_value; // Work with a temporary variable

    // Calculate bits needed for the absolute value (u16)
    while temp_val > 0 {
        num_bits += 1;
        temp_val >>= 1;
    }

    num_bits
}

#[cfg(test)]
mod tests {
    // ... tests ...
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
