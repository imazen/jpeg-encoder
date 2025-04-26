use serde::Deserialize;

// Common types used across tests (assuming these exist or will be created in Rust)
// TODO: Define these properly based on Rust implementation
pub type JpegColorSpace = i32; // Represents J_COLOR_SPACE
pub type QuantPass = i32;      // Represents jpegli::QuantPass enum
pub type JpegliDataType = i32; // Represents JpegliDataType enum
pub type JpegliEndianness = i32; // Represents JpegliEndianness enum

// Minimal component info needed for various setup steps
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct ComponentInfoMinimal {
    pub component_index: usize,
    pub h_samp_factor: i32,
    pub v_samp_factor: i32,
    pub quant_tbl_no: i32,
    // Add other fields if needed for specific tests (e.g., dc/ac_tbl_no)
    pub width_in_blocks: usize,
    pub height_in_blocks: usize,
}

// Represents a slice of a planar float buffer, common input/output format
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct RustRowBufferSliceF32 {
    pub component_index: usize, // Which component this slice belongs to
    pub start_row: isize, // Starting row index relative to original buffer
    pub num_rows: usize, // Number of rows included in the data
    pub start_col: isize, // Starting column index (often 0 or -border)
    pub num_cols: usize, // Number of columns included in the data (width + borders)
    pub stride: usize, // Stride of the original buffer (elements between rows)
    // Data stored row-major. Each inner Vec represents one row.
    pub data: Vec<Vec<f32>>,
}

// Represents a full block (8x8 = 64 elements)
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct BlockF32 {
   pub data: Vec<f32>, // Should always have 64 elements
}
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct BlockI32 {
   pub data: Vec<i32>, // Should always have 64 elements
}

// Represents the actual Token struct in Rust (assuming its definition)
// TODO: Ensure this matches the actual Token definition in Rust
#[derive(Debug, PartialEq, Copy, Clone, Deserialize)]
pub struct Token {
    pub context: i32,
    pub symbol: i32,
    pub bits: u32,
}


// --- Test Data Structures ---

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct SetQuantMatricesTest {
    // Inputs to configure the Rust JpegCompressStruct state before the call
    pub config_force_baseline: bool,
    pub config_use_std_tables: bool,
    pub config_xyb_mode: bool,
    pub config_jpeg_color_space: JpegColorSpace,
    pub config_cicp_transfer_function: i32,
    pub config_components: Vec<ComponentInfoMinimal>,

    // Direct function argument (or derived from quality/linear_quality)
    pub input_distances: Vec<f32>, // Size NUM_QUANT_TBLS (typically 4)

    // Expected state *after* the function call (quant tables in cinfo)
    // Using Vec<u16> directly assumes table indices 0..N are used contiguously.
    // Option<Vec<u16>> allows for potentially skipped/null tables.
    pub expected_quant_tables: Vec<Option<Vec<u16>>>, // Size NUM_QUANT_TBLS (4)
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct InitQuantizerTest {
     // Inputs to configure the Rust JpegCompressStruct state before the call
    pub config_use_adaptive_quantization: bool,
    pub config_jpeg_color_space: JpegColorSpace,
    pub config_quant_tables: Vec<Option<Vec<u16>>>, // Initial state of tables
    pub config_components: Vec<ComponentInfoMinimal>, // Needed for QuantValsToDistance if adaptive
    pub config_force_baseline: bool, // Needed for QuantValsToDistance
    pub config_cicp_transfer_function: i32, // Needed for QuantValsToDistance

    // Direct input parameter
    pub input_pass: QuantPass,

    // Expected state *after* the function call (fields within master state)
    // Dimensions: [component_index][k] where k=0..63
    pub expected_quant_mul: Vec<Vec<f32>>,
    pub expected_zero_bias_mul: Vec<Vec<f32>>,
    pub expected_zero_bias_offset: Vec<Vec<f32>>,
}


#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct PadInputBufferTest {
    // Input Config/Parameters
    pub config_image_width: usize, // Original component width
    pub config_image_height: usize, // Original component height
    pub config_buffer_xsize: usize, // Padded width (width_in_blocks * 8)
    pub config_buffer_ysize: usize, // Padded height (height_in_blocks * 8)

    // Parameters describing the specific operation instance tested
    pub input_component_index: usize,
    pub input_row_index: isize, // Row being padded (-1 to image_height-1)
    pub input_border: usize, // e.g., 1

    // Input State (The row *before* padding is applied, including border area)
    // Length = config_buffer_xsize + 2 * input_border
    pub input_row_slice_before: Vec<f32>,

    // Expected Output State
    // Length = config_buffer_xsize + 2 * input_border
    pub expected_row_slice_after: Vec<f32>,
    // Include the rows copied during vertical padding if input_row_index == image_height-1
    // Vec<Vec<f32>> where each inner Vec is a full padded row slice.
    pub expected_vertically_padded_rows: Option<Vec<Vec<f32>>>,
}


#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct ComputePreErosionTest {
    // Input State (Slice of Y input buffer with required context/borders)
    pub input_buffer_y_slice: RustRowBufferSliceF32,

    // Config/Parameters (can verify against slice dimensions)
    pub config_xsize: usize,
    pub config_y0: usize, // Start row in original buffer coordinates
    pub config_ylen: usize, // Number of rows processed
    pub config_border: i32,

    // Expected Output State (Slice of the pre_erosion buffer)
    // Dimensions should be roughly xsize/4 x ylen/4
    pub expected_pre_erosion_slice: RustRowBufferSliceF32,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct FuzzyErosionTest {
    // Input State (Slice of pre-erosion buffer with required context/borders)
    pub input_pre_erosion_slice: RustRowBufferSliceF32,

    // Config/Parameters (block coordinates)
    pub config_yb0: usize, // Start block row
    pub config_yblen: usize, // Number of block rows processed

    // Expected Output State (Slice of the quant_field buffer *after* erosion)
    // Dimensions should be roughly pre_erosion_xsize/2 x yblen
    pub expected_quant_field_slice: RustRowBufferSliceF32,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct PerBlockModulationsTest {
    // Config/Parameters
    pub config_y_quant_01: f32, // Base Y quant level (AC01)
    pub config_yb0: usize, // Start block row
    pub config_yblen: usize, // Number of block rows processed

    // Input State
    // Slice of Y input buffer covering blocks yb0..yb0+yblen
    pub input_buffer_y_slice: RustRowBufferSliceF32,
    // Slice of quant_field *before* modulation (output of FuzzyErosion)
    pub input_quant_field_slice_before: RustRowBufferSliceF32,

    // Expected Output State
    // Slice of quant_field *after* modulation (before final adjustment step)
    pub expected_quant_field_slice_after: RustRowBufferSliceF32,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct ComputeAdaptiveQuantFieldTest {
     // ----- Inputs -----
     // Config/Parameters from cinfo/master
    pub config_use_adaptive_quantization: bool, // Should be true
    pub config_y_channel_index: usize,
    pub config_jpeg_color_space: JpegColorSpace, // Needed? Maybe not directly
    pub config_y_quant_01: f32,
    pub config_next_iMCU_row: usize,
    pub config_total_iMCU_rows: usize,
    pub config_max_v_samp_factor: i32,
    pub config_y_comp_width_in_blocks: usize,
    pub config_y_comp_height_in_blocks: usize,

    // Input State: Slice of Y input buffer including context rows needed for filters
    pub input_buffer_y_slice: RustRowBufferSliceF32,

    // ----- Outputs -----
    // Expected Output State: Final relevant slice of quant_field after all steps + final adjustment
    pub expected_quant_field_slice: RustRowBufferSliceF32,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct RgbToYCbCrTest {
    // Inputs (One row of planar float RGB)
    pub input_r_row: Vec<f32>,
    pub input_g_row: Vec<f32>,
    pub input_b_row: Vec<f32>,
    pub width: usize,

    // Outputs (One row of planar float YCbCr)
    pub expected_y_row: Vec<f32>,
    pub expected_cb_row: Vec<f32>,
    pub expected_cr_row: Vec<f32>,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct DownsampleInputBufferTest {
    // ----- Inputs -----
    // Config
    pub config_components: Vec<ComponentInfoMinimal>, // Samp factors needed

    // State: Slices of m->input_buffer (already padded) for each component
    pub input_buffer_slices: Vec<RustRowBufferSliceF32>,

    // ----- Outputs -----
    // State: Expected slices of m->raw_data for each component after downsampling
    // Contains only valid downsampled data (e.g. 4x4 for Cb/Cr) padded to block size (8x8)
    pub expected_raw_data_slices: Vec<RustRowBufferSliceF32>,
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct ComputeCoefficientBlockTest {
     // Input Config/Parameters for the block
     pub config_component_index: usize,
     pub input_last_dc_coeff: i16, // State before processing this block

     // Input Data for the block
     pub input_pixels_block: BlockF32, // 8x8 (64 elements) from raw_data
     pub input_qmc_block: BlockF32, // 64 multipliers (m->quant_mul[c])
     pub input_aq_strength: f32, // From m->quant_field for this block
     pub input_zero_bias_offset_block: BlockF32, // 64 offsets (m->zero_bias_offset[c])
     pub input_zero_bias_mul_block: BlockF32, // 64 multipliers (m->zero_bias_mul[c])

     // Expected Output State & Results for the block
     pub expected_coeffs_block: BlockI32, // 64 quantized coefficients (int32)
     pub expected_new_last_dc_coeff: i16, // State after processing this block
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct ComputeTokensForBlockTest {
     // Input Config/Parameters
     pub input_dc_context: i32, // Component index c
     pub input_ac_context: i32, // Component index c + 4

     // Input Data
     pub input_coeffs_block: BlockI32, // 64 quantized coefficients (DC diff-coded)

     // Expected Output
     pub expected_tokens: Vec<Token>, // Sequence of tokens generated for this block
 } 