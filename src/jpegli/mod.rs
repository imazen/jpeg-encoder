use alloc::vec::Vec;

pub(crate) mod adaptive_quantization;
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

        let luma_table_raw = crate::jpegli::quant::compute_jpegli_quant_table(distance, true, is_yuv420, force_baseline, None);
        let chroma_table_raw = crate::jpegli::quant::compute_jpegli_quant_table(distance, false, is_yuv420, force_baseline, None);

        let (zero_bias_offsets, zero_bias_multipliers) = crate::jpegli::quant::compute_zero_bias_tables(distance, num_components);

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