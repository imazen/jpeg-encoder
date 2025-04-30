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

#[cfg(test)]
mod reference_test_data;

#[cfg(test)]
mod reference_tests;

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