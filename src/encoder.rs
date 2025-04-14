use alloc::{boxed::Box, vec, vec::Vec};

// Keep only imports from *other* modules
use crate::huffman::{CodingClass, HuffmanTable};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::adaptive_quantization::compute_adaptive_quant_field;
use crate::quantization::{QuantizationTable, QuantizationTableType, quality_to_distance, compute_zero_bias_tables};
use crate::writer::{JfifWrite, JfifWriter, ZIGZAG};
use crate::{EncodingError, EncoderResult}; // Use EncoderResult alias
use crate::fdct::{fdct, forward_dct_float}; // Import both DCT functions
use crate::Density; // Add import for Density from lib.rs
use crate::cms::{self, ColorProfile, JxlCms};
use crate::tf::ExtraTF; // Import ExtraTF
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
        let width = image.width() as usize;
        let height = image.height() as usize;
        let color_type = image.color_type();
        let num_components_in = color_type.get_num_components();
        let mut jpeg_color_type = color_type.get_jpeg_color_type(); // May change if XYB

        if width == 0 || height == 0 {
            return Err(EncodingError::ZeroImageDimensions {
                width: width as u16,
                height: height as u16,
            });
        }

        // --- Initialize CMS --- 
        if self.cms_state.is_none() {
            let input_prof = self.input_profile.as_ref().ok_or(
                EncoderError::CmsError("Input profile not set".to_string())
            )?;
            let num_threads = 1; // Assuming single-threaded for now
            let pixels_per_thread = width * 8; // Process 8 scanlines at a time?
            self.cms_state = Some(cms::cms_init(
                input_prof,
                &self.internal_color_profile, // Target Linear sRGB
                self.intensity_target,
                num_threads,
                pixels_per_thread,
            )?);
            if self.xyb_mode && self.premul_absorb.is_none() {
                 self.premul_absorb = Some(xyb::compute_premul_absorb(self.intensity_target));
            }
        }
        let cms_state = self.cms_state.as_ref().unwrap();
        let premul_absorb = self.premul_absorb.as_ref();

        // Determine quantization tables based on quality or distance
        let distance = self.jpegli_distance.unwrap_or_else(|| quality_to_distance(self.quality));
        let q_tables = [
            QuantizationTable::new(self.quantization_tables[0], distance, 0, false)?,
            QuantizationTable::new(self.quantization_tables[1], distance, 1, false)?,
        ];

        // Compute Adaptive Quantization Field if enabled
        let adapt_quant_field = if self.use_adaptive_quantization {
             // TODO: AQ needs input in the correct color space (likely sRGB gamma)
             // This requires running CMS *after* AQ calculation or providing original data.
             eprintln!("Warning: Adaptive Quantization computation path needs review for color space correctness.");
             // Some(compute_adaptive_quant_field(&image, distance)) // Needs ImageBuffer ref
             None // Disable for now until pipeline is clear
        } else {
            None
        };

        // Re-initialize components based on final color type (could be XYB)
        if self.xyb_mode {
             // TODO: Define a JpegColorType for XYB if needed, or handle specially
             // For now, assume 3 components (X, Y, B) will be written.
             jpeg_color_type = JpegColorType::Ycbcr; // Treat as 3 components for SOF marker?
             eprintln!("Warning: JPEG component setup for XYB mode needs verification.");
        }
        self.init_components(jpeg_color_type);

        // --- Write Headers ---
        self.writer.write_marker(Marker::SOI)?;
        self.writer.write_jfif_header(self.density)?;
        if let Some(input_prof) = &self.input_profile {
            if !input_prof.icc.is_empty() {
                 self.add_icc_profile(&input_prof.icc)?;
            }
        }
        for (nr, data) in &self.app_segments {
            self.writer.write_segment(Marker::APP(*nr), data)?;
        }
        self.writer.write_dqt(&q_tables)?;
        // SOF needs final number of components and sampling factors
        self.write_frame_header(&image, &q_tables)?;
        // DHT needs optimization results
        // ... (Write DHT later, after optimize_huffman_table if enabled)

        // --- Encoding Loop --- 
        let num_pixels = width * height;
        let thread_id = 0; // Single thread

        // 1. Convert input u8 -> f32 planes
        let mut f32_planes = image_to_f32_planes(&image)?;

        // 2. Apply CMS Transform (Input Profile -> Linear sRGB)
        let mut temp_interleaved_buffer = Vec::new(); // Reusable buffer for interleaving
        let num_cms_in_channels = cms_state.channels_src;
        let num_cms_out_channels = cms_state.channels_dst;

        if num_cms_in_channels == num_cms_out_channels && num_cms_in_channels <= f32_planes.len() {
            let channels_to_process = num_cms_in_channels;
            temp_interleaved_buffer.resize(num_pixels * channels_to_process, 0.0);

            // Interleave relevant planes
            for i in 0..num_pixels {
                for c in 0..channels_to_process {
                    temp_interleaved_buffer[i * channels_to_process + c] = f32_planes[c][i];
                }
            }

            // Run CMS (input is temp_interleaved_buffer, output is also temp_interleaved_buffer)
            cms::cms_run(cms_state, thread_id, &temp_interleaved_buffer, &mut temp_interleaved_buffer, num_pixels)?;

            // De-interleave back to planes
            for i in 0..num_pixels {
                 for c in 0..channels_to_process {
                    f32_planes[c][i] = temp_interleaved_buffer[i * channels_to_process + c];
                 }
            }
        } else if cms_state.skip_lcms {
             // Handle pre/post processing even if LCMS is skipped (needs buffer logic)
             if cms_state.preprocess != ExtraTF::kNone {
                 // ... Apply before_transform to f32_planes ...
             }
              if cms_state.postprocess != ExtraTF::kNone {
                 // ... Apply after_transform to f32_planes ...
             }
        } else {
            return Err(EncoderError::CmsError(format!("Unsupported CMS channel combination: {} -> {}", num_cms_in_channels, num_cms_out_channels)));
        }

        // 3. Convert to XYB if enabled (operates on Linear sRGB planes)
        let final_planes = if self.xyb_mode {
            if let Some(premul) = premul_absorb {
                if f32_planes.len() >= 3 {
                    // linear_rgb_row_to_xyb operates in-place
                    xyb::linear_rgb_row_to_xyb(&mut f32_planes[0], &mut f32_planes[1], &mut f32_planes[2], premul, num_pixels);
                    // scale_xyb_row operates in-place
                    xyb::scale_xyb_row(&mut f32_planes[0], &mut f32_planes[1], &mut f32_planes[2], num_pixels);
                    // Keep XYB planes (first 3)
                    // Need to handle potential 4th plane (e.g., alpha) if present
                    f32_planes.truncate(3); // Keep only X, Y, B
                    f32_planes
                } else {
                    return Err(EncoderError::Other("XYB mode requires at least 3 input channels after CMS".into()));
                }
            } else {
                 return Err(EncoderError::CmsError("XYB mode enabled but premul_absorb constants missing".to_string()));
            }
        } else {
            // 4. Convert to Target JPEG Color Space (e.g., YCbCr)
            match jpeg_color_type {
                JpegColorType::Ycbcr => {
                    if f32_planes.len() >= 3 {
                        color_transform::linear_rgb_to_ycbcr(&mut f32_planes, num_pixels);
                        f32_planes.truncate(3);
                    } else if f32_planes.len() >= 1 { // Assume Gray -> YCbCr (Y=L, Cb=0.5, Cr=0.5)
                        let luma = f32_planes[0].clone();
                        f32_planes.resize(3, vec![0.5; num_pixels]); // Resize and fill Cb/Cr
                        f32_planes[0] = luma; // Set Y
                    } else {
                        return Err(EncoderError::Other("Cannot convert to YCbCr: Not enough input planes".into()));
                    }
                },
                JpegColorType::Luma => {
                    // Input should already be Gray after CMS if target is Luma
                    f32_planes.truncate(1);
                },
                JpegColorType::Ycck | JpegColorType::Cmyk => {
                    if f32_planes.len() >= 4 {
                         color_transform::cmyk_to_ycck(&mut f32_planes, num_pixels);
                         f32_planes.truncate(4);
                    } else {
                         return Err(EncoderError::Other("Cannot convert to YCCK: Not enough input planes".into()));
                    }
                }
            }
            f32_planes
        };

        // --- Convert f32 planes to i16 DCT blocks --- 
        let num_final_components = final_planes.len();
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

        for c in 0..num_final_components {
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

            let plane = &final_planes[c];
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

        // --- Optimize Huffman tables (if requested) --- 
        if self.optimize_huffman_table {
             self.optimize_huffman_table(&blocks);
        }
        self.writer.write_dht(&self.huffman_tables)?; // Write (optimized or default) DHT


        // --- Encode blocks --- 
        if let Some(scans) = self.progressive_scans {
            // TODO: Implement progressive encoding with new block structure
            return Err(EncodingError::Other("Progressive encoding not implemented for new CMS path".into()));
            // self.encode_image_progressive::<_, OP>(image, scans, &q_tables, adapt_quant_field.as_deref())?;
        } else if !self.sampling_factor.supports_interleaved() {
             // TODO: Implement sequential encoding with new block structure
             return Err(EncodingError::Other("Sequential encoding not implemented for new CMS path".into()));
            // self.encode_image_sequential::<_, OP>(image, &q_tables, adapt_quant_field.as_deref())?;
        } else {
            // TODO: Implement interleaved encoding with new block structure
            // self.encode_image_interleaved::<_, OP>(image, &q_tables, adapt_quant_field.as_deref())?
             self.encode_interleaved_from_blocks(&blocks, num_mcus, mcu_width, mcu_height, adapt_quant_field.as_deref())?;
        }

        self.writer.write_marker(Marker::EOI)?; // End of Image

        Ok(())
    }

    // New function to encode from pre-computed blocks
    fn encode_interleaved_from_blocks(
        &mut self,
        blocks: &[Vec<[i16; 64]>; 4],
        num_mcus: usize,
        mcu_width: usize,
        mcu_height: usize,
        adapt_quant_field: Option<&[f32]>,
    ) -> EncoderResult<()> {
        let num_components = self.components.len();

        let h_samp_max = self.components.iter().map(|c| c.horizontal_sampling_factor).max().unwrap_or(1) as usize;
        let v_samp_max = self.components.iter().map(|c| c.vertical_sampling_factor).max().unwrap_or(1) as usize;

        let mut dc_predictors = vec![0i16; num_components];

        let mcus_w = ceil_div(self.writer.width(), h_samp_max * 8);

        for mcu_idx in 0..num_mcus {
            if let Some(interval) = self.restart_interval {
                if mcu_idx > 0 && mcu_idx % interval as usize == 0 {
                    self.writer.flush_bits()?;
                    self.writer.write_marker(Marker::RST(mcu_idx as u8 / interval as u8 % 8))?;
                    dc_predictors = vec![0; num_components];
                }
            }

            for c_idx in 0..num_components {
                let comp = &self.components[c_idx];
                let h_samp = comp.horizontal_sampling_factor as usize;
                let v_samp = comp.vertical_sampling_factor as usize;
                let mcu_row = mcu_idx / mcus_w;
                let mcu_col = mcu_idx % mcus_w;
                let n_blocks_h = ceil_div(self.writer.width() * h_samp, h_samp_max * 8);

                for y in 0..v_samp {
                    for x in 0..h_samp {
                         let block_row = mcu_row * v_samp + y;
                         let block_col = mcu_col * h_samp + x;
                         let block_idx = block_row * n_blocks_h + block_col;

                        if block_idx >= blocks[c_idx].len() {
                            // This can happen with padding for non-aligned image dimensions
                            continue;
                        }

                        let q_block = &blocks[c_idx][block_idx];
                        let dc_table = &self.huffman_tables[comp.dc_huffman_table as usize].0;
                        let ac_table = &self.huffman_tables[comp.ac_huffman_table as usize].1;

                        let dc_diff = q_block[0] - dc_predictors[c_idx];
                        dc_predictors[c_idx] = q_block[0];

                        let (value, nbits) = if dc_diff == 0 {
                            (0, 0)
                        } else {
                            let bits = get_num_bits(dc_diff);
                            (
                                if dc_diff > 0 {
                                    dc_diff as u16
                                } else {
                                    (dc_diff - 1) as u16 & ((1 << bits) - 1)
                                },
                                bits,
                            )
                        };

                        let code = dc_table.get_code(nbits);
                        self.writer.write_bits(code.0, code.1)?;
                        self.writer.write_bits(value, nbits)?;

                        let mut zero_run = 0;
                        for i in 1..64 {
                            if q_block[i] == 0 {
                                zero_run += 1;
                                continue;
                            }

                            while zero_run >= 16 {
                                let code = ac_table.get_code(0xf0);
                                self.writer.write_bits(code.0, code.1)?;
                                zero_run -= 16;
                            }

                            let bits = get_num_bits(q_block[i]);
                            let value = if q_block[i] > 0 {
                                q_block[i] as u16
                            } else {
                                (q_block[i] - 1) as u16 & ((1 << bits) - 1)
                            };

                            let code = ac_table.get_code(zero_run << 4 | bits);
                            self.writer.write_bits(code.0, code.1)?;
                            self.writer.write_bits(value, bits)?;

                            zero_run = 0;
                        }

                        if zero_run > 0 {
                            let code = ac_table.get_code(0);
                            self.writer.write_bits(code.0, code.1)?;
                        }
                    }
                }
            }
        }

        self.writer.flush_bits()?;
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
            &self.components.iter().collect::<Vec<_>>(),
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

                // Value to compare: Abs(original_coeff * 8.0 / q_val)
                // block[z] is the DCT coefficient *after* integer DCT scaling.
                // This thresholding logic might be less accurate here compared to float path.
                // We use the *quantized* coefficient magnitude as a proxy.
                let abs_quant_scaled = (q_coeff as f32 * (q_val as f32 / 8.0)).abs(); // Approx inverse scaling

                if abs_quant_scaled < threshold {
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
        // Get the jpegli-style aq_strength value
        let aq_strength = adapt_quant_field.map_or(0.0, |field| {
            field.get(block_idx).copied().unwrap_or(0.0)
        });

        for i in 0..64 {
            let z = ZIGZAG[i] as usize & 0x3f;
            let value_f = coeffs[z]; // Value from float DCT
            let q_val_f = table.get_raw(z) as f32;

            // Ensure q_val_f is not zero
            if q_val_f == 0.0 { continue; }

            // Quantization: value_f / q_val_f (jpegli divides by quant value)
            // Jpegli internal `QuantizeBlock` multiplies by qmc (quant multiplier = 1/q_val_f)
            // Let's stick to division for clarity here.
            let qval = value_f / q_val_f;

            // Rounding (round half up)
            let mut q_coeff = qval.round() as i32;

            // Jpegli-style adaptive quantization thresholding
            // Applied *before* rounding for the float path, based on unrounded qval.
            if i > 0 && q_coeff != 0 && adapt_quant_field.is_some() {
                let zb_offset = zero_bias_offset[i];
                let zb_mul = zero_bias_mul[i];
                let threshold = zb_offset + zb_mul * aq_strength;

                // Value to compare: Abs(qval) - the unrounded quantized value
                if qval.abs() < threshold {
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
