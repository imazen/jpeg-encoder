use alloc::vec::Vec;

pub(crate) mod adaptive_quantization;

pub(crate) mod quant_constants;
pub mod cms;
pub mod color_transform;
pub mod fdct_jpegli;
pub mod quant;
pub mod tf;
pub mod xyb;

pub mod jpegli_encoder;
pub use jpegli_encoder::JpegliEncoder;

use serde::{Deserialize, Serialize};
use serde_repr::*;

mod reference_test_data;

#[cfg(test)]
mod reference_tests;

#[cfg(test)]
mod tests;

// Define the configuration and state for Jpegli encoding
#[derive(Debug, Clone)] // Added Clone for convenience, might need review
pub struct JpegliConfig {
    pub distance: f32,
    pub use_float_dct: bool,
    pub use_adaptive_quantization: bool,

    // Computed Jpegli data
    pub(crate) luma_table_raw: [u16; 64], // Assuming these are always computed if Jpegli is used
    pub(crate) chroma_table_raw: [u16; 64],
    pub(crate) zero_bias_offsets: Vec<[f32; 64]>,
    pub(crate) zero_bias_multipliers: Vec<[f32; 64]>,
    pub(crate) adaptive_quant_field: Option<Vec<f32>>, // Keep as Option for now
}

impl JpegliConfig {
    /// Creates a basic JpegliConfig, computing initial tables.
    /// More complex setup (like AQ) might happen later.
    pub(crate) fn new(distance: f32, sampling_factor: crate::SamplingFactor, num_components: usize) -> Self {
        // Initial computation based on distance and basic params
        let force_baseline = false; // Assuming standard jpegli behavior
        let is_yuv420 = sampling_factor == crate::SamplingFactor::F_2_2 || sampling_factor == crate::SamplingFactor::R_4_2_0;

        // Determine quant_max based on force_baseline
        let quant_max = if force_baseline { 255 } else { 32767 };

        // Always use Jpegli computation path
        // Note: Using constants directly from `quant` module for locality - NOW FROM quant_constants
        let luma_table_raw = crate::jpegli::quant::compute_quant_table_values(
            distance,
            crate::jpegli::quant_constants::GLOBAL_SCALE_YCBCR, // Use quant_constants
            // Slice the first 64 elements (Luma) from the YCbCr base matrix
            crate::jpegli::quant_constants::BASE_QUANT_MATRIX_YCBCR[0..64]
                .try_into()
                .expect("Slice with incorrect length for Luma quant table"),
            true, // non_linear_scaling = true for Jpegli
            false, // is_chroma_420 = false for Luma
            quant_max,
        );
        let chroma_table_raw = crate::jpegli::quant::compute_quant_table_values(
            distance,
            crate::jpegli::quant_constants::GLOBAL_SCALE_YCBCR, // Use quant_constants
            // Slice the next 64 elements (Cb) from the YCbCr base matrix
            crate::jpegli::quant_constants::BASE_QUANT_MATRIX_YCBCR[64..128]
                .try_into()
                .expect("Slice with incorrect length for Chroma quant table"),
            true, // non_linear_scaling = true for Jpegli
            is_yuv420, // is_chroma_420 depends on sampling factor
            quant_max,
        );

        // Removed call to compute_zero_bias_tables - logic needs integration elsewhere
        // Zero bias tables will be initialized later, likely within the encoder state
        let zero_bias_offsets: Vec<[f32; 64]> = Vec::with_capacity(num_components);
        let zero_bias_multipliers: Vec<[f32; 64]> = Vec::with_capacity(num_components);

        Self {
            distance,
            use_float_dct: true, // Default Jpegli behavior often uses float DCT
            use_adaptive_quantization: true, // Default Jpegli behavior often uses AQ
            luma_table_raw,
            chroma_table_raw,
            zero_bias_offsets,
            zero_bias_multipliers,
            adaptive_quant_field: None, // Computed later if needed
        }
    }

    // Add methods to update use_float_dct and use_adaptive_quantization if needed
    pub fn set_float_dct(&mut self, enable: bool) {
        self.use_float_dct = enable;
    }

    pub fn set_adaptive_quantization(&mut self, enable: bool) {
        self.use_adaptive_quantization = enable;
        if !enable {
            self.adaptive_quant_field = None; // Clear AQ field if disabled
        }
    }
}



// ITU-T H.273 / ISO 23091-2 Table 3 — TransferCharacteristics code points


// Decimal	Identifier (canonical name)	Typical shorthand / common use-case
// 0	reserved	―
// 1	ITU-R BT.709	“bt709”, Rec. 709 HDTV SDR gamma ≈ 2.4 (also reused for BT.601/2020 SDR) 
// GitHub
// 2	unspecified	encoder didn't signal – decoder must assume container defaults 
// matroska.org
// 3	reserved	―
// 4	BT.470 System M	CRT gamma 2.2 (NTSC-M SDTV)
// 5	BT.470 System BG	CRT gamma 2.8 (PAL/SECAM SDTV)
// 6	SMPTE 170M	U.S. SDTV (identical OETF to 1)
// 7	SMPTE 240M	early HDTV cameras (unused today)
// 8	Linear	linear-light RGB (no OETF)
// 9	Log 100	log transfer, 100:1 dynamic range
// 10	Log Sqrt 100*√10	log transfer, 100 √10 : 1 range
// 11	IEC 61966-2-4	xvYCC
// 12	ITU-R BT.1361	“Extended Gamut” CRT system
// 13	IEC 61966-2-1	sRGB / sYCC (“srgb”)
// 14	ITU-R BT.2020-10	Rec. 2020 SDR 10-bit (same curve as 1)
// 15	ITU-R BT.2020-12	Rec. 2020 SDR 12-bit (same curve as 1)
// 16	SMPTE ST 2084	Perceptual Quantisation (PQ, HDR10)
// 17	SMPTE ST 428-1	Cinema D-CI X′ = E^(1/2.6)
// 18	ARIB STD-B67	Hybrid-Log-Gamma (HLG)
// All integers > 18 are currently undefined/reserved; do not use them in bit-streams. 
// matroska.org
// GitHub

// Practical notes & decoder behaviour
// Values 1 / 6 / 14 / 15 are mathematically identical; many tool-chains treat them as synonyms. 
// W3C

// HEVC/AV1, WebCodecs, GStreamer, FFmpeg/Libav and AVIF/JPEG-XL libraries fully parse the table above. Libjpeg/libjpeg-turbo ignore CICP because "legacy" JPEG has no carriage for it; color is assumed sRGB.

// When a full ICC profile is present (e.g. in HEIF/AVIF), the CICP triplet is advisory and may be overridden.

// For SDR JPEG workflows, signalling 1/13/6 (BT.709 primaries, sRGB TRC, BT.601 coeffs) mirrors the implicit assumptions of most JPEG decoders and avoids surprise gamut shifts.

// Use only the enumerated values; anything else risks being rejected or silently mapped to 'unspecified'.

/// Unspecified and reserved valus are not permitted in this enum
#[derive(Debug, Clone, Copy)]
pub(crate) enum TransferCharacteristics{
    /// ITU-R BT.709 (default SDR), Rec. 709 HDTV SDR gamma ≈ 2.4 (also reused for BT.601/2020 SDR)
    Bt709 = 1,
    /// BT.470 System M (CRT gamma 2.2, NTSC-M SDTV)
    Bt470SystemM = 4,
    /// BT.470 System BG (CRT gamma 2.8, PAL/SECAM SDTV)
    Bt470SystemBG = 5,
    /// SMPTE 170M (U.S. SDTV, identical OETF to Bt709(1))
    Smpte170M = 6,
    /// SMPTE 240M (early HDTV cameras, unused today)
    Smpte240M = 7,
    /// Linear RGB (no OETF)
    LinearRGB = 8,
    /// Log 100 (100:1 dynamic range)
    Log100 = 9,
    /// Log Sqrt 100*√10 (100 √10 : 1 range)
    LogSqrt100 = 10,
    /// xvYCC
    XvYcc = 11,
    /// ITU-R BT.1361 (Extended Gamut) CRT system
    Bt1361 = 12,
    /// sRGB / sYCC (“srgb”)
    Srgb = 13,
    /// Rec. 2020 SDR 10-bit (same curve as Bt709(1))
    Bt2020_10 = 14,
    /// Rec. 2020 SDR 12-bit (same curve as Bt709(1))
    Bt2020_12 = 15,
    /// SMPTE ST 2084 (Perceptual Quantisation, PQ, HDR10)
    SmpteSt2084 = 16,
    /// Cinema D-CI X′ = E^(1/2.6)
    CinemaDciX = 17,
    /// Arib Std-B67 (Hybrid-Log-Gamma, HLG)
    AribStdB67 = 18,
}

/// 444|422|420|440
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subsampling{
    /// 4:4:4
    YCbCr444 = 1,
    /// 4:2:2
    YCbCr422 = 2,
    /// 4:2:0
    YCbCr420 = 3,   
    /// 4:4:0
    YCbCr440 = 4,
}

// void jpegli_set_colorspace(j_compress_ptr cinfo, J_COLOR_SPACE colorspace) {
//     CheckState(cinfo, jpegli::kEncStart);
//     cinfo->jpeg_color_space = colorspace;
//     switch (colorspace) {
//       case JCS_GRAYSCALE:
//         cinfo->num_components = 1;
//         break;
//       case JCS_RGB:
//       case JCS_YCbCr:
//         cinfo->num_components = 3;
//         break;
//       case JCS_CMYK:
//       case JCS_YCCK:
//         cinfo->num_components = 4;
//         break;
//       case JCS_UNKNOWN:
//         cinfo->num_components =
//             std::min<int>(jpegli::kMaxComponents, cinfo->input_components);
//         break;
//       default:
//         JPEGLI_ERROR("Unsupported jpeg colorspace %d", colorspace);
//     }
//     // Adobe marker is only needed to distinguish CMYK and YCCK JPEGs.
//     cinfo->write_Adobe_marker = TO_JXL_BOOL(cinfo->jpeg_color_space == JCS_YCCK);
//     if (cinfo->comp_info == nullptr) {
//       cinfo->comp_info =
//           jpegli::Allocate<jpeg_component_info>(cinfo, MAX_COMPONENTS);
//     }
//     memset(cinfo->comp_info, 0,
//            jpegli::kMaxComponents * sizeof(jpeg_component_info));
//     for (int c = 0; c < cinfo->num_components; ++c) {
//       jpeg_component_info* comp = &cinfo->comp_info[c];
//       comp->component_index = c;
//       comp->component_id = c + 1;
//       comp->h_samp_factor = 1;
//       comp->v_samp_factor = 1;
//       comp->quant_tbl_no = 0;
//       comp->dc_tbl_no = 0;
//       comp->ac_tbl_no = 0;
//     }
//     if (colorspace == JCS_RGB) {
//       cinfo->comp_info[0].component_id = 'R';
//       cinfo->comp_info[1].component_id = 'G';
//       cinfo->comp_info[2].component_id = 'B';
//       if (cinfo->master->xyb_mode) {
//         // Subsample blue channel.
//         cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
//         cinfo->comp_info[1].h_samp_factor = cinfo->comp_info[1].v_samp_factor = 2;
//         cinfo->comp_info[2].h_samp_factor = cinfo->comp_info[2].v_samp_factor = 1;
//         // Use separate quantization tables for each component
//         cinfo->comp_info[1].quant_tbl_no = 1;
//         cinfo->comp_info[2].quant_tbl_no = 2;
//       }
//     } else if (colorspace == JCS_CMYK) {
//       cinfo->comp_info[0].component_id = 'C';
//       cinfo->comp_info[1].component_id = 'M';
//       cinfo->comp_info[2].component_id = 'Y';
//       cinfo->comp_info[3].component_id = 'K';
//     } else if (colorspace == JCS_YCbCr || colorspace == JCS_YCCK) {
//       // Use separate quantization and Huffman tables for luma and chroma
//       cinfo->comp_info[1].quant_tbl_no = 1;
//       cinfo->comp_info[2].quant_tbl_no = 1;
//       cinfo->comp_info[1].dc_tbl_no = cinfo->comp_info[1].ac_tbl_no = 1;
//       cinfo->comp_info[2].dc_tbl_no = cinfo->comp_info[2].ac_tbl_no = 1;
//       // Use chroma subsampling by default
//       cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
//       if (colorspace == JCS_YCCK) {
//         cinfo->comp_info[3].h_samp_factor = cinfo->comp_info[3].v_samp_factor = 2;
//       }
//     }
//   }

// if (!jpeg_settings.chroma_subsampling.empty()) {
//     if (jpeg_settings.chroma_subsampling == "444") {
//       cinfo.comp_info[0].h_samp_factor = 1;
//       cinfo.comp_info[0].v_samp_factor = 1;
//     } else if (jpeg_settings.chroma_subsampling == "440") {
//       cinfo.comp_info[0].h_samp_factor = 1;
//       cinfo.comp_info[0].v_samp_factor = 2;
//     } else if (jpeg_settings.chroma_subsampling == "422") {
//       cinfo.comp_info[0].h_samp_factor = 2;
//       cinfo.comp_info[0].v_samp_factor = 1;
//     } else if (jpeg_settings.chroma_subsampling == "420") {
//       cinfo.comp_info[0].h_samp_factor = 2;
//       cinfo.comp_info[0].v_samp_factor = 2;
//     } else {
//       return false;
//     }
//     for (int i = 1; i < cinfo.num_components; ++i) {
//       cinfo.comp_info[i].h_samp_factor = 1;
//       cinfo.comp_info[i].v_samp_factor = 1;
//     }
//   } else if (!jpeg_settings.xyb) {
//     // Default is no chroma subsampling.
//     cinfo.comp_info[0].h_samp_factor = 1;
//     cinfo.comp_info[0].v_samp_factor = 1;
//   }
impl Subsampling{
    pub fn from_str(value: &str) -> Option<Self>{
        match value{
            "444" => Some(Self::YCbCr444),
            "440" => Some(Self::YCbCr440),
            "422" => Some(Self::YCbCr422),
            "420" => Some(Self::YCbCr420),
            _ => None,
        }
    }
    pub fn to_str(&self) -> &'static str{
        match self{
            Self::YCbCr444 => "444",
            Self::YCbCr440 => "440",
            Self::YCbCr422 => "422",
            Self::YCbCr420 => "420",
        }
    }
    pub fn to_h_v_samp_factor(&self) -> (u8, u8){
        match self{
            Self::YCbCr444 => (1, 1),
            Self::YCbCr440 => (1, 2),
            Self::YCbCr422 => (2, 1),
            Self::YCbCr420 => (2, 2),
        }
    }

    pub fn is_yuv420(&self) -> bool{
        matches!(self, Self::YCbCr420)  
    }

    pub fn to_sampling_factor(&self) -> crate::SamplingFactor{
        match self{
            Self::YCbCr444 => crate::SamplingFactor::F_1_1,
            Self::YCbCr440 => crate::SamplingFactor::F_1_2,
            Self::YCbCr422 => crate::SamplingFactor::F_2_1,
            Self::YCbCr420 => crate::SamplingFactor::F_2_2,
        }
    }

    pub fn from_sampling_factor(value: crate::SamplingFactor) -> Option<Self>{
        match value{
            crate::SamplingFactor::F_1_1 => Some(Self::YCbCr444),
            crate::SamplingFactor::F_1_2 => Some(Self::YCbCr440),
            crate::SamplingFactor::F_2_1 => Some(Self::YCbCr422),
            crate::SamplingFactor::F_2_2 => Some(Self::YCbCr420),
            _ => None,
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimplifiedTransferCharacteristics{
    /// SDR (default)
    Default = 1,
    /// PQ (HDR10)
    Pq = 16,
    /// HLG (HLG)   
    Hlg = 18,
}


impl SimplifiedTransferCharacteristics{

    pub fn to_int(&self) -> u8{
        match self{
            Self::Default => 1,
        Self::Pq => 16,
            Self::Hlg => 18,
        }
    }
    pub fn from_int(value: i32) -> Option<Self>{
        match value{
            1 | 13 | 6 => Some(Self::Default),
            16 => Some(Self::Pq),
            18 => Some(Self::Hlg),
            _ => None,
        }
    }
}

#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(i32)]
pub enum JpegColorSpace {
    /// error/unspecified
    Unknown = 0,            /* error/unspecified */
    /// monochrome
    Grayscale = 1,          /* monochrome */
    /// red/green/blue as specified by the RGB_RED,
    /// RGB_GREEN, RGB_BLUE, and RGB_PIXELSIZE macros */
    Rgb = 2,                /* red/green/blue as specified by the RGB_RED,
                               RGB_GREEN, RGB_BLUE, and RGB_PIXELSIZE macros */
    /// Y/Cb/Cr (also known as YUV)
    YCbCr = 3,              /* Y/Cb/Cr (also known as YUV) */
    /// C/M/Y/K
    Cmyk = 4,               /* C/M/Y/K */
    /// Y/Cb/Cr/K
    Ycck = 5,               /* Y/Cb/Cr/K */
    /// red/green/blue
    ExtRgb = 6,            /* red/green/blue */
    /// red/green/blue/x
    ExtRgbx = 7,           /* red/green/blue/x */
    /// blue/green/red
    ExtBgr = 8,            /* blue/green/red */
    /// blue/green/red/x
    ExtBgrx = 9,           /* blue/green/red/x */
    /// x/blue/green/red
    ExtXbgr = 10,           /* x/blue/green/red */
    /// x/red/green/blue
    ExtXrgb = 11,           /* x/red/green/blue */
    /// red/green/blue/alpha
    ExtRgba = 12,           /* red/green/blue/alpha */
    /// blue/green/red/alpha
    ExtBgra = 13,           /* blue/green/red/alpha */
    /// alpha/blue/green/red
    ExtAbgr = 14,           /* alpha/blue/green/red */
    /// alpha/red/green/blue
    ExtArgb = 15,           /* alpha/red/green/blue */
    /// 5-bit red/6-bit green/5-bit blue
    Rgb565 = 16,           /* 5-bit red/6-bit green/5-bit blue */
}

impl TryFrom<i32> for JpegColorSpace {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if JpegColorSpace::from_i32(value).is_some() {
            Ok(JpegColorSpace::from_i32(value).unwrap())
        } else {
            Err("Invalid value for JpegColorSpace".to_string())
        }
    }
}
impl From<crate::encoder::JpegColorType> for JpegColorSpace {
    fn from(value: crate::encoder::JpegColorType) -> Self {
        match value{
            crate::encoder::JpegColorType::Luma => Self::Grayscale,
            crate::encoder::JpegColorType::Ycbcr => Self::YCbCr,
            crate::encoder::JpegColorType::Cmyk => Self::Cmyk,
            crate::encoder::JpegColorType::Ycck => Self::Ycck,
            _ => panic!("Invalid color type"),
        }
    }
}
impl JpegColorSpace {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Unknown),
            1 => Some(Self::Grayscale),
            2 => Some(Self::Rgb),
            3 => Some(Self::YCbCr),
            4 => Some(Self::Cmyk),
            5 => Some(Self::Ycck),
            6 => Some(Self::ExtRgb),
            7 => Some(Self::ExtRgbx),
            8 => Some(Self::ExtBgr),
            9 => Some(Self::ExtBgrx),
            10 => Some(Self::ExtXbgr),
            11 => Some(Self::ExtXrgb),
            12 => Some(Self::ExtRgba),
            13 => Some(Self::ExtBgra),
            14 => Some(Self::ExtAbgr),
            15 => Some(Self::ExtArgb),
            16 => Some(Self::Rgb565),
            _ => None,
        }
    }
    /// What the output format is for the given imput format.
    pub fn to_output_color_type(&self) -> crate::encoder::JpegColorType {
        match self {
            Self::Grayscale => crate::encoder::JpegColorType::Luma,
            Self::Cmyk => crate::encoder::JpegColorType::Cmyk,
            Self::Ycck => crate::encoder::JpegColorType::Ycck,
            _ => crate::encoder::JpegColorType::Ycbcr,
        }
    }

    pub fn get_num_components(&self) -> usize{
        match self{
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::YCbCr => 3,
            Self::Cmyk => 4,
            Self::Ycck => 4,
            Self::ExtRgb => 3,
            Self::ExtBgr => 3,
            _ => panic!("Invalid color space"),
        }
    }
}
