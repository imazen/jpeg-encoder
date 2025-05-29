/// MAX_COMPONENTS is the maximum number of components in a JPEG image
pub const MAX_COMPONENTS: usize = 4;
/// DCTSIZE is the size of the DCT block
pub const DCTSIZE: usize = 8;
/// DCTSIZE_U8 is the size of the DCT block as a u8
pub const DCTSIZE_U8: u8 = 8;
/// DCTSIZE2 is the size of the DCT block squared
pub const DCTSIZE2: usize = 64;
/// DCTSIZE2_U8 is the size of the DCT block squared as a u8
pub const DCTSIZE2_U8: u8 = 64;
/// MAX_SAMP_FACTOR is the maximum sampling factor (like in 4:2:0)
pub const MAX_SAMP_FACTOR: usize = 4;
/// MAX_SAMP_FACTOR_U8 is the maximum sampling factor as a u8
pub const MAX_SAMP_FACTOR_U8: u8 = 4;
/// JPEG_MAX_DIMENSION is the maximum dimension of a JPEG image
pub const JPEG_MAX_DIMENSION: usize = 65500;
/// Padding for pre-erosion stage
pub const K_PRE_EROSION_BORDER: usize = 1;

/// MAX_COMPS_IN_SCAN is the maximum number of components in a scan
pub const MAX_COMPS_IN_SCAN: usize = 4;
/// C_MAX_BLOCKS_IN_MCU is the maximum number of blocks in a MCU
pub const C_MAX_BLOCKS_IN_MCU: usize = 10;

/// NUM_QUANT_TBLS is the number of quantization tables
pub const NUM_QUANT_TBLS: usize = 4;
/// NUM_HUFF_TBLS is the number of Huffman tables
pub const NUM_HUFF_TBLS: usize = 4;
/// MAX_REFINEMENT_BIT is the maximum refinement bit
pub const MAX_REFINEMENT_BIT: i32 = 10;