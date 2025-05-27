use serde_repr::Serialize_repr;
use serde_repr::Deserialize_repr;
use target_features::SimdType;

use super::simd_width::SimdWidth;


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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimplifiedTransferCharacteristics{
    /// SDR (default)
    Default = 1,
    /// PQ (HDR10)
    Pq = 16,
    /// HLG (HLG)   
    Hlg = 18,
}


/// Represents component information needed for Jpegli encoding.
#[derive(Clone, Debug)]
pub struct JpegliComponentInfo {
    pub id: u8,
    pub quantization_table_index: u8,
    pub dc_huffman_table_index: u8,
    pub ac_huffman_table_index: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
    // Dimensions in blocks, useful for processing
    pub width_in_blocks: usize,
    pub height_in_blocks: usize,
    // Component dimensions in pixels
    pub width: usize,
    pub height: usize,
}


#[derive(Clone, Debug, Copy)]
pub struct JpegliComponentSettings {
    pub index: u8,
    pub letter: char,
    pub quantization_table_index: u8,
    pub dc_huffman_table_index: u8,
    pub ac_huffman_table_index: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}
impl JpegliComponentSettings {

    pub const EMPTY: Self = JpegliComponentSettings { index: 0, letter: '\0', quantization_table_index: 0, dc_huffman_table_index: 0, ac_huffman_table_index: 0, horizontal_sampling_factor: 0, vertical_sampling_factor: 0 };

    pub fn new(index: u8, letter: char, quantization_table_index: u8, dc_huffman_table_index: u8, ac_huffman_table_index: u8, horizontal_sampling_factor: u8, vertical_sampling_factor: u8) -> Self {
        Self { index, letter, quantization_table_index, dc_huffman_table_index, ac_huffman_table_index, horizontal_sampling_factor, vertical_sampling_factor }
    }

    pub fn default(index: u8) -> Self {
        Self {
            index,
            letter: (index + 1) as char,
            quantization_table_index: 0,
            dc_huffman_table_index: 0,
            ac_huffman_table_index: 0,
            horizontal_sampling_factor: 1,
            vertical_sampling_factor: 1,
        }
    }

    pub fn with_letter(self, letter: char) -> Self {
        Self { letter, ..self }
    }
    pub fn with_quant_ix(self, quant_ix: u8) -> Self {
        Self { quantization_table_index: quant_ix, ..self }
    }
    /// Set both the DC and AC Huffman tables to the same index
    pub fn with_huff_ix(self, huff_ix: u8) -> Self {
        Self { dc_huffman_table_index: huff_ix, ac_huffman_table_index: huff_ix, ..self }
    }
    pub fn with_h_v_sampling(self, h_sampling_factor: u8, v_sampling_factor: u8) -> Self {
        Self { horizontal_sampling_factor: h_sampling_factor, vertical_sampling_factor: v_sampling_factor, ..self }
    }
    
    
}


// ITU-T H.273 / ISO 23091-2 Table 3 — TransferCharacteristics code points
// Decimal	Identifier (canonical name)	Typical shorthand / common use-case
// 0	reserved	―
// 1	ITU-R BT.709	"bt709", Rec. 709 HDTV SDR gamma ≈ 2.4 (also reused for BT.601/2020 SDR) 
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
// 12	ITU-R BT.1361	"Extended Gamut" CRT system
// 13	IEC 61966-2-1	sRGB / sYCC ("srgb")
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
    /// sRGB / sYCC ("srgb")
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
    pub fn to_luma_h_v_samp_factor(&self) -> (u8, u8){
        match self{
            Self::YCbCr444 => (1, 1),
            Self::YCbCr440 => (1, 2),
            Self::YCbCr422 => (2, 1),
            Self::YCbCr420 => (2, 2),
            // Rare: 4:1:1 -> (4, 1)
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
impl From<crate::old_encoder::OutputJpegColorType> for JpegColorSpace {
    fn from(value: crate::old_encoder::OutputJpegColorType) -> Self {
        match value{
            crate::old_encoder::OutputJpegColorType::Luma => Self::Grayscale,
            crate::old_encoder::OutputJpegColorType::Ycbcr => Self::YCbCr,
            crate::old_encoder::OutputJpegColorType::Cmyk => Self::Cmyk,
            crate::old_encoder::OutputJpegColorType::Ycck => Self::Ycck,
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
    pub fn to_output_color_type(&self) -> crate::old_encoder::OutputJpegColorType {
        match self {
            Self::Grayscale => crate::old_encoder::OutputJpegColorType::Luma,
            Self::Cmyk => crate::old_encoder::OutputJpegColorType::Cmyk,
            Self::Ycck => crate::old_encoder::OutputJpegColorType::Ycck,
            _ => crate::old_encoder::OutputJpegColorType::Ycbcr,
        }
    }

    pub fn get_num_components(&self) -> usize{
        match self{
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::ExtBgr => 3,
            Self::YCbCr => 3,
            Self::Cmyk => 4,
            Self::Ycck => 4,
            Self::ExtRgb => 3,
            Self::ExtXbgr => 4,
            Self::ExtXrgb => 4,
            Self::ExtRgba => 4,
            Self::ExtBgra => 4,
            Self::ExtAbgr => 4,
            Self::ExtArgb => 4,
            Self::Rgb565 => 3,
            Self::ExtBgrx => 4,
            Self::ExtRgbx => 4,
            Self::Unknown => panic!("Invalid color space JpegColorSpace::Unknown used with get_num_components"),
            
        }
    }
}

pub trait RowBuffer<T: Copy + SimdWidth + Sized> {
    fn info(&self) -> &RowBufferInfo;
    /// Should return the buffer, starting at 0,0 of the padded area.
    fn get_buffer(&self) -> &[T];
    fn get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T]);
    fn get_buffer_mut(&mut self) -> &mut [T]{
        let (_, buffer) = self.get_info_and_buffer_mut();
        buffer
    }
    fn get_info_and_buffer(&self) -> (&RowBufferInfo, &[T]){
        (self.info(), self.get_buffer())
    }
    fn is_aligned(&self) -> bool{
        T::simd_is_minimally_aligned(self.get_buffer()) &&
        self.info().stride % T::simd_min_alignment_in_elements() == 0
    }
    fn is_optimally_aligned(&self) -> bool{
        T::simd_is_optimally_aligned(self.get_buffer()) &&
        self.info().stride % T::simd_optimal_alignment_in_elements() == 0
    }
    /// Returns a new buffer that is a region of the current buffer, relative to the full buffer area including padding.
    /// Allows for cropping, or for redefining what is padding and what is not.
    /// Result will likely not be aligned except to a multiple of sizeof<T>
    fn region_mut_core(&mut self, x: usize, y: usize, window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<(RowBufferInfo, &mut [T])>{
        let (info, buffer) = self.get_info_and_buffer_mut();

        if padding_left > x || padding_top > y ||
            padding_bottom > (info.full_height - y - window_height) ||
            padding_right > (info.full_width - x - window_width)
            || window_width > info.full_width - x - padding_right
            || window_height > info.full_height - y - padding_bottom{
            return None;
        }
        let new_full_width = window_width + padding_left + padding_right;
        let new_full_height = window_height + padding_top + padding_bottom;
        let new_info = RowBufferInfo{
            full_width: new_full_width,
            full_height: new_full_height,
            window_width,
            window_height,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            stride: info.stride,
        };

        let x_offset = x - padding_left;
        let y_offset = y - padding_top;
        let new_buffer = &mut buffer[y_offset * info.stride + x_offset..(y_offset + new_full_height) * info.stride + x_offset];
        Some((new_info, new_buffer))
    }

    fn region_mut_shrink_padding_core(&mut self, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<(RowBufferInfo, &mut [T])>{
        let (left, top, width, height) = (self.info().padding_left, self.info().padding_top, self.info().window_width, self.info().window_height);
        self.region_mut_core(left, top, width, height, padding_left, padding_right, padding_top, padding_bottom)
    }
    fn split_at_mut_core(&mut self, row: usize) -> (RowBufferInfo, &mut [T], RowBufferInfo, &mut [T]){
        if row >= self.info().full_height || row == 0{
            panic!("RowBuffer::split_at_mut: cannot create an empty buffer or exceed the buffer height");
        }
        let (info, buffer) = self.get_info_and_buffer_mut();
        let (top, bottom) = buffer.split_at_mut(row * info.stride);
        let top_info = RowBufferInfo{
            full_width: info.full_width,
            full_height: row,
            window_width: info.window_width,
            padding_left: info.padding_left,
            padding_right: info.padding_right,
            stride: info.stride,
            padding_top: row.min(info.padding_top),
            window_height: (info.window_height as isize).min(row as isize - info.padding_top as isize).max(0) as usize,
            padding_bottom: (info.padding_bottom as isize).min(info.full_height as isize - row as isize).max(0) as usize,
        };
        let bottom_info = RowBufferInfo{
            full_width: info.full_width,
            full_height: info.full_height - row,
            window_width: info.window_width,
            padding_left: info.padding_left,
            padding_right: info.padding_right,
            stride: info.stride,
            padding_top: (info.padding_top as isize - row as isize).max(0) as usize,
            window_height: (((info.window_height as isize + info.padding_top as isize) - row as isize) - info.padding_top as isize).max(0) as usize,
            padding_bottom: (info.padding_bottom as isize - (info.full_height - row) as isize).max(0) as usize,
        };
        if !top_info.is_valid() || !bottom_info.is_valid(){
            panic!("RowBuffer::split_at_mut: invalid row buffer info");
        }
        
        (top_info, top, bottom_info, bottom)   
    }

    /// Get a reference to a row of the windowed area.
    fn get_window_row(&self, y: usize) -> Option<&[T]>{
        let info = self.info();
        if info.is_window_empty(){ return None;}
        let buffer = self.get_buffer();
        let row_start = (y + info.padding_top) * info.stride + info.padding_left;
        buffer.get(row_start..row_start + info.window_width)
    }
    /// Get a mutable reference to a row of the windowed area.
    fn get_window_row_mut(&mut self, y: usize) -> Option<&mut [T]>{
        let (info, buffer) = self.get_info_and_buffer_mut();
        let row_start = (y + info.padding_top) * info.stride + info.padding_left;
        buffer.get_mut(row_start..row_start + info.window_width)
    }
    fn get_padded_row(&self, y: usize) -> Option<&[T]>{
        let (info, buffer) = self.get_info_and_buffer();
        let row_start = y * info.stride;
        buffer.get(row_start..row_start + info.full_width)
    }
    fn get_padded_row_mut(&mut self, y: usize) -> Option<&mut [T]>{ 
        let (info, buffer) = self.get_info_and_buffer_mut();
        let row_start = y * info.stride;
        buffer.get_mut(row_start..row_start + info.full_width)
    }

    fn get_padded_row_window_relative(&self, windowed_y: isize) -> Option<&[T]>{
        let y_abs = windowed_y + self.info().padding_top as isize;
        if y_abs < 0 || y_abs >= self.info().full_height as isize{
            return None;
        }
        self.get_padded_row(y_abs as usize)
    }
    fn get_padded_row_window_relative_mut(&mut self, windowed_y: isize) -> Option<&mut [T]>{
        let y_abs = windowed_y + self.info().padding_top as isize;
        if y_abs < 0 || y_abs >= self.info().full_height as isize{
            return None;
        }
        self.get_padded_row_mut(y_abs as usize)
    }

    #[deprecated(since = "0.1.0", note = "Use get_padded_row_mut instead")]
    fn row_mut_old(&mut self, windowed_y: isize) -> Option<&mut [T]>{
        self.get_padded_row_window_relative_mut(windowed_y)
    }

    #[deprecated(since = "0.1.0", note = "Use get_padded_row instead")]
    fn row_old(&self, windowed_y: isize) -> Option<&[T]>{
        self.get_padded_row_window_relative(windowed_y)
    }
    
    fn fill_full(&mut self, value: T, x: usize, y: usize, width: usize, height: usize){
        if x + width > self.info().full_width || y + height > self.info().full_height{
            panic!("fill_full: out of bounds");
        }
        if width == 0 || height == 0{
            return;
        }
        for row in y..y+height{
            self.get_padded_row_mut(row).unwrap()[x..x+width].fill(value);
        }
    }
    fn fill_window(&mut self, value: T, x: usize, y: usize, width: usize, height: usize){
        if x + width > self.info().window_width || y + height > self.info().window_height{
            panic!("fill_window: out of bounds");
        }
        if width == 0 || height == 0{
            return;
        }
        for row in y..y+height{
            self.get_window_row_mut(row).unwrap()[x..x+width].fill(value);
        }
    }
    fn get_window_pixel(&self, x: usize, y: usize) -> Option<&T>{
        if let Some(row) = self.get_window_row(y){
            row.get(x)
        } else {
            None
        }
    }
    fn get_window_pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut T>{   
        if let Some(row) = self.get_window_row_mut(y){
            row.get_mut(x)
        } else {
            None
        }
    }
    fn get_padded_pixel(&self, x: usize, y: usize) -> Option<&T>{
        if let Some(row) = self.get_padded_row(y){
            row.get(x)
        } else {
            None
        }
    }
    fn get_padded_pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut T>{
        if let Some(row) = self.get_padded_row_mut(y){
            row.get_mut(x)
        } else {
            None
        }
    }
    
    fn copy_row_section(&mut self, from: usize, to: usize, x: usize, width: usize){
        if from >= self.info().full_height || to >= self.info().full_height{
            panic!("copy_full_row: out of bounds");
        }
        if x + width > self.info().full_width{
            panic!("copy_row_section: out of bounds");
        }
        if from == to{
            return;
        }
        let (info, buffer) = self.get_info_and_buffer_mut();
        if from < to{
            let (_, rest) = buffer.split_at_mut(from * info.stride);
            let (from_row, rest) = rest.split_at_mut(info.stride);
            let (_, to_and_rest) = rest.split_at_mut((to - from - 1) * info.stride);
            to_and_rest[x..x+width].copy_from_slice(&from_row[x..x+width]);
        } else {
            let (_, rest) = buffer.split_at_mut(to * info.stride);
            let (to_row, rest) = rest.split_at_mut(info.stride);
            let (_, from_and_rest) = rest.split_at_mut((from - to - 1) * info.stride);
            from_and_rest[x..x+width].copy_from_slice(&to_row[x..x+width]);
        }
    }
    fn copy_window_row(&mut self, from: usize, to: usize){
        self.copy_row_section(from, to, self.info().padding_left, self.info().window_width);
    }
    fn copy_full_row(&mut self, from: usize, to: usize){
        self.copy_row_section(from, to, 0, self.info().full_width);
    }

    fn has_padding(&self) -> bool{
        self.info().has_padding()
    }
    fn pad_using_edges(&mut self){
        if !self.has_padding() || self.info().is_window_empty(){
            return;
        }
        let (info, buffer) = self.get_info_and_buffer_mut();
        if info.padding_left > 0 || info.padding_right > 0{
            let left_val = buffer[info.padding_left];
            let right_val = buffer[info.window_width - 1 + info.padding_left];
            for row in 0..info.window_height{
                let row_start = row * info.stride;
                buffer[row_start..row_start + info.padding_left].fill(left_val);
                buffer[row_start + info.window_width + info.padding_left..info.full_width].fill(right_val);
            }
        }
        if info.padding_top > 0 || info.padding_bottom > 0{
            // copy nonoverlapping rows, using split_at_mut
            let (padding_top, rest) = buffer.split_at_mut(info.padding_top * info.stride);
            let (top_source_row, rest) = rest.split_at_mut(info.stride);
            let (_, rest) = rest.split_at_mut(info.stride * (info.window_height - 1));
            let (bottom_source_row, padding_bottom) = rest.split_at_mut(info.stride);
            let copy_width = info.full_width;
            for to_row in padding_top.chunks_mut(info.stride){
                to_row[..copy_width].copy_from_slice(&top_source_row[..copy_width]);
            }
            for to_row in padding_bottom.chunks_mut(info.stride){
                to_row[..copy_width].copy_from_slice(&bottom_source_row[..copy_width]);
            }
        }
    }

    fn pad_row_full_width(&mut self, row: usize){
        if !self.has_padding() || self.info().is_window_empty(){
            return;
        }
        let (info, buffer) = self.get_info_and_buffer_mut();
        if info.padding_left > 0 || info.padding_right > 0{
            let left_val = buffer[info.padding_left];
            let right_val = buffer[info.window_width - 1 + info.padding_left];
            let row_start = row * info.stride;
            buffer[row_start..row_start + info.padding_left].fill(left_val);
            buffer[row_start + info.window_width + info.padding_left..info.full_width].fill(right_val);
        }
    }
    fn pad_row_width_old(&mut self, windowed_y: usize, copy_to_right: usize, border: usize){
        //TODO: These constraints are not enforced by C++, but we def want to know if they are violated
        if copy_to_right != self.info().window_width + self.info().padding_left - 1{
            panic!("pad_row_width_old: copy_to_right is not equal to window_width + padding_left - 1");
        }
        if windowed_y >= self.info().window_height{
            panic!("pad_row_width_old: windowed_y is too large");
        }
        if border != self.info().padding_left && border != self.info().padding_right{
            panic!("pad_row_width_old: border is not equal to padding_left or padding_right");
        }
        self.pad_row_full_width(windowed_y);
    }

    fn copy_row_window_relative(&mut self, from: isize, to: isize){
        let from_abs = from + self.info().padding_top as isize;
        let to_abs = to + self.info().padding_top as isize;
        if from_abs < 0 || to_abs >= self.info().full_height as isize{
            panic!("copy_row_window_relative: out of bounds");
        }
        self.copy_full_row(from_abs as usize, to_abs as usize);
    }

    #[deprecated(since = "0.1.0", note = "Use info().window_width instead")]
    fn xsize(&self) -> usize{
        self.info().window_width
    }
    #[deprecated(since = "0.1.0", note = "Use info().window_height instead")]
    fn ysize(&self) -> usize{
        self.info().window_height
    }


    fn get_padded_row_and_two_neighbors_mut(&mut self, windowed_y_center: isize) -> Option<(&mut [T], &mut [T], &mut [T])>{
        let y_first = windowed_y_center + self.info().padding_top as isize - 1 as isize;
        let y_last = windowed_y_center + self.info().padding_top as isize + 1 as isize;
        
        if y_first < 0 || y_last >= self.info().full_height as isize{
            return None;
        }
        let stride = self.info().stride;
        let rest = &mut self.get_buffer_mut()[y_first as usize * stride..];
        let (a,rest) = rest.split_at_mut(stride);
        let (b,rest) = rest.split_at_mut(stride);
        let (c,_) = rest.split_at_mut(stride);
        Some((a,b,c))
    }

    fn get_padded_row_and_two_neighbors(&self, windowed_y_center: isize) -> Option<(&[T], &[T], &[T])>{
        let y_first = windowed_y_center + self.info().padding_top as isize - 1 as isize;
        let y_last = windowed_y_center + self.info().padding_top as isize + 1 as isize;
        
        if y_first < 0 || y_last >= self.info().full_height as isize{
            return None;
        }
        let stride = self.info().stride;
        let rest = &self.get_buffer()[y_first as usize * stride..];
        let (a,rest) = rest.split_at(stride);
        let (b,rest) = rest.split_at(stride);
        let (c,_) = rest.split_at(stride);
        Some((a,b,c))
    }
    fn get_padded_row_and_neighbors_mut(&mut self, windowed_y_center: isize, vertical_padding: usize) -> Option<Vec<&mut [T]>>{
        let y_first = windowed_y_center + self.info().padding_top as isize - vertical_padding as isize;
        let y_last = windowed_y_center + self.info().padding_top as isize + vertical_padding as isize;
        
        if y_first < 0 || y_last >= self.info().full_height as isize{
            return None;
        }
        let (y_first, y_last) = (y_first as usize, y_last as usize);

        let mut rows = Vec::with_capacity(vertical_padding * 2 + 1);
        let stride = self.info().stride;
        let mut rest = &mut self.get_buffer_mut()[y_first * stride..];
        let mut row;
        for y in y_first..=y_last{
            (row, rest) = rest.split_at_mut(stride);
            rows.push(row);
        }
        Some(rows)
    }
    fn get_padded_row_and_neighbors(&self, windowed_y_center: isize, vertical_padding: usize) -> Option<Vec<&[T]>>{
        let y_first = windowed_y_center + self.info().padding_top as isize - vertical_padding as isize;
        let y_last = windowed_y_center + self.info().padding_top as isize + vertical_padding as isize;
        
        if y_first < 0 || y_last >= self.info().full_height as isize{
            return None;
        }
        let (y_first, y_last) = (y_first as usize, y_last as usize);

        let mut rows = Vec::with_capacity(vertical_padding * 2 + 1);
        let stride = self.info().stride;
        let mut rest = &self.get_buffer()[y_first * stride..];
        let mut row;
        for y in y_first..=y_last{
            (row, rest) = rest.split_at(stride);
            rows.push(row);
        }
        Some(rows)
    }
}

#[derive(Debug,Clone,Copy)]
pub struct RowBufferInfo {
    pub full_width: usize,
    pub full_height: usize,
    pub window_width: usize,
    pub window_height: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padding_top: usize,
    pub padding_bottom: usize,
    pub stride: usize,
}

impl RowBufferInfo {
    pub fn is_valid(&self) -> bool {
        let width_ok = self.full_width == self.window_width + self.padding_left + self.padding_right;
        let height_ok = self.full_height == self.window_height + self.padding_top + self.padding_bottom;
        let stride_ok = self.stride >= self.full_width;
        width_ok && height_ok && stride_ok
    }
    pub fn is_window_empty(&self) -> bool{
        self.window_width == 0 || self.window_height == 0
    }
    pub fn is_empty(&self) -> bool{
        self.full_width == 0 || self.full_height == 0
    }
    pub fn has_padding(&self) -> bool{
        self.padding_left > 0 || self.padding_right > 0 || self.padding_top > 0 || self.padding_bottom > 0
    }
    pub fn padding_is(&self, value: usize) -> bool{
        self.padding_left == value && self.padding_right == value && self.padding_top == value && self.padding_bottom == value
    }
    pub fn min_buffer_size(&self) -> usize{
        self.stride * self.full_height
    }
}

#[derive(Debug)]
pub struct RowBufferRef<'a,T: Copy + 'a + SimdWidth>  {
    info: RowBufferInfo,
    data: &'a mut [T],
}

impl<'a,T: Copy + 'a + SimdWidth> RowBuffer<T> for RowBufferRef<'a,T> {
    fn info(&self) -> &RowBufferInfo{
        &self.info
    }
    fn get_buffer(&self) -> &[T]{
        self.data
    }
    fn get_buffer_mut(&mut self) -> &mut [T]{
        self.data
    }
    fn get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T]){
        (&self.info, &mut self.data)
    }
   
}

impl<'a,T: Copy + 'a + SimdWidth> RowBufferRef<'a,T> {
    pub fn new(info: RowBufferInfo, data: &'a mut [T]) -> Self{
        if data.len() < info.min_buffer_size(){
            panic!("RowBufferRef::new: data is too small");
        }
        Self{info, data}
    }

    fn region_mut(&mut self, x: usize, y: usize, window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<RowBufferRef<T>>
     where T:Copy + 'a {
        if let Some((new_info, new_buffer)) = self.region_mut_core(x, y, window_width, window_height, padding_left, padding_right, padding_top, padding_bottom){
            Some(RowBufferRef::new(new_info, new_buffer))
        } else {
            None
        }
    }
    fn region_mut_shrink_padding(&mut self, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<RowBufferRef<T>>{
        if let Some((new_info, new_buffer)) = self.region_mut_shrink_padding_core(padding_left, padding_right, padding_top, padding_bottom){
            Some(RowBufferRef::new(new_info, new_buffer))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedRowBuffer<T: Copy + Default> where T:SimdWidth {
    info: RowBufferInfo,
    offset: usize,
    data: Vec<T>,
}

impl<T: Copy + Default> RowBuffer<T> for OwnedRowBuffer<T> where T:SimdWidth {
    fn info(&self) -> &RowBufferInfo{
        &self.info
    }
    fn get_buffer(&self) -> &[T]{
        &self.data[self.offset..]
    }
    fn get_buffer_mut(&mut self) -> &mut [T]{
        &mut self.data[self.offset..]
    }
    fn get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T]){
        (&self.info, &mut self.data[self.offset..])
    }
}
impl<T: Copy + Default> OwnedRowBuffer<T> where T:SimdWidth {
    pub fn new(window_width: usize, window_height: usize, padding: usize, fill: T) -> Self{
        Self::new_with_padding(window_width, window_height, padding, padding, padding, padding, fill)
    }
    pub fn new_with_padding(window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize, fill: T) -> Self{
        let alignment = T::simd_optimal_alignment_in_elements();
        let full_width = window_width + padding_left + padding_right;
        let full_height = window_height + padding_top + padding_bottom;
        let stride = T::simd_stride_in_elements(full_width);

        let size = stride * full_height + alignment;

        let mut data = Vec::with_capacity(size);
        data.resize(size, fill);
        let offset = &data.as_ptr().addr() % alignment;

        let info = RowBufferInfo{
            full_width,
            full_height,
            window_width,
            window_height,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            stride,
        };
        Self{info, offset, data}    
    }

    pub fn as_ref(&mut self) -> RowBufferRef<'_,T>{
        RowBufferRef::new(self.info, &mut self.data[self.offset..])
    }
}


