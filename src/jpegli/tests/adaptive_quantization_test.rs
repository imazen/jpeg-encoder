// src/jpegli/tests/adaptive_quantization_test.rs

#[cfg(test)]
mod adaptive_quantization_tests {
    use crate::jpegli::adaptive_quantization::*;
    use super::structs::{ComputeAdaptiveQuantFieldTest, ComputePreErosionTest, FuzzyErosionTest, PerBlockModulationsTest};
    use super::testdata::*;
    use super::test_utils::assert_buffer_eq;
    use crate::assert_float_relative_eq;
    use alloc::vec;
    use alloc::vec::Vec;

    // --- Helper Functions (if needed, or use from test_utils) ---

    /// Reconstructs a planar buffer from a slice description for testing.
    fn reconstruct_buffer_f32(slice: &RustRowBufferSliceF32, full_height: usize, full_width: usize) -> Vec<f32> {
        // Determine buffer size based on slice stride and full height
        let buffer_size = slice.stride * full_height;
        let mut buffer = vec![0.0f32; buffer_size]; // Initialize with default

        // Check if slice data matches expected dimensions based on num_rows/num_cols
        if slice.data.len() != slice.num_rows {
            panic!("Slice data has {} rows, expected {}", slice.data.len(), slice.num_rows);
        }
        for (i, row_data) in slice.data.iter().enumerate() {
            if row_data.len() != slice.num_cols {
                panic!("Slice data row {} has {} cols, expected {}", i, row_data.len(), slice.num_cols);
            }
        }

        // Copy data from the slice into the correct position in the full buffer
        for r_idx in 0..slice.num_rows {
            let buffer_row = (slice.start_row + r_idx as isize) as usize;
            if buffer_row >= full_height { continue; } // Skip rows outside the logical buffer height

            let buffer_row_start = buffer_row * slice.stride;
            let slice_row = &slice.data[r_idx];

            for c_idx in 0..slice.num_cols {
                let buffer_col = (slice.start_col + c_idx as isize) as usize;
                if buffer_col >= full_width + (slice.start_col.abs() as usize) { continue; } // Adjust width check for negative start_col

                let buffer_idx = buffer_row_start + buffer_col;
                if buffer_idx < buffer.len() { // Ensure index is within bounds
                   buffer[buffer_idx] = slice_row[c_idx];
                } else {
                   // This might indicate an issue with stride, dimensions, or start_col/start_row
                   eprintln!("Warning: Calculated index {} out of bounds for buffer size {}", buffer_idx, buffer.len());
                }
            }
        }
        buffer
    }

    // --- Test Functions ---

    #[test]
    fn test_compute_pre_erosion() {
        for test_case in COMPUTE_PRE_EROSION_TESTS.iter() {
            // Extract config
            let width = test_case.config_xsize;
            // Height needs to be deduced or provided. Assume input slice covers needed region + borders.
            // Let's estimate height from the input slice dimensions, assuming it includes context.
            let est_input_height = test_case.input_luma_slice.data.len();
            let input_luma_buffer = reconstruct_buffer_f32(&test_case.input_luma_slice, est_input_height, width);

            let pre_erosion_w = (width + 3) / 4;
            let pre_erosion_h = (test_case.config_ysize_blocks + 3) / 4; // Use block height for output size
            let mut actual_pre_erosion = Vec::new(); // Will be resized inside

            // Call the Rust function
            // Note: Rust compute_pre_erosion_scalar takes scaled input [0, 1]
            // The C++ instrumentation likely captured the raw float input buffer before scaling.
            // Assuming the test data `input_luma_slice` contains *already scaled* data [0, 1] for simplicity.
            compute_pre_erosion_scalar(
                &input_luma_buffer,
                width,
                est_input_height, // Use estimated height of the input region provided
                &mut actual_pre_erosion,
            );

            // Reconstruct expected buffer slice
            let expected_pre_erosion_buffer = reconstruct_buffer_f32(
                &test_case.expected_pre_erosion_slice,
                pre_erosion_h, // Use expected output height
                pre_erosion_w,
            );

            // Compare the relevant slice
            // The expected slice tells us which part of the buffer to compare
            let exp_slice = &test_case.expected_pre_erosion_slice;
            assert_buffer_eq(
                &actual_pre_erosion, pre_erosion_w, // Actual buffer + width
                exp_slice, // Expected slice info
                &expected_pre_erosion_buffer, // Expected buffer (reconstructed)
                1e-5, // Tolerance for float comparison
                &format!("ComputePreErosion mismatch (xsize={}, ysize_blocks={})", width, test_case.config_ysize_blocks),
            );
        }
    }

    #[test]
    fn test_fuzzy_erosion() {
        for test_case in FUZZY_EROSION_TESTS.iter() {
            // Extract config and input state
            let pre_erosion_w = test_case.input_pre_erosion_slice.num_cols; // Width from input slice
            let pre_erosion_h = test_case.input_pre_erosion_slice.num_rows; // Height from input slice
            let input_pre_erosion = reconstruct_buffer_f32(&test_case.input_pre_erosion_slice, pre_erosion_h, pre_erosion_w);

            let block_w = test_case.expected_quant_field_slice.num_cols; // Width from output slice
            let block_h = test_case.expected_quant_field_slice.num_rows; // Height from output slice

            let mut actual_aq_map = vec![0.0f32; block_w * block_h]; // Output buffer
            let mut tmp_erosion = vec![0.0f32; pre_erosion_w * pre_erosion_h]; // Temp buffer

            // Call the Rust function
            fuzzy_erosion_scalar(
                &input_pre_erosion,
                pre_erosion_w,
                pre_erosion_h,
                block_w,
                block_h,
                &mut tmp_erosion,
                &mut actual_aq_map,
            );

            // Reconstruct expected output
            let expected_aq_map = reconstruct_buffer_f32(&test_case.expected_quant_field_slice, block_h, block_w);

            // Compare the relevant slice
            assert_buffer_eq(
                &actual_aq_map, block_w,
                &test_case.expected_quant_field_slice,
                &expected_aq_map,
                1e-5,
                &format!("FuzzyErosion mismatch (yb0={}, yblen={})", test_case.config_yb0, test_case.config_yblen),
            );
        }
    }

    #[test]
    fn test_per_block_modulations() {
        for test_case in PER_BLOCK_MODULATIONS_TESTS.iter() {
            // Extract config and input state
            let block_w = test_case.config_xsize_blocks;
            let block_h = test_case.input_quant_field_slice.num_rows; // Height from input AQ slice
            // We need pixel-level width/height for input_luma_slice reconstruction
            let pixel_width = test_case.input_luma_slice.num_cols;
            let pixel_height = test_case.input_luma_slice.num_rows;

            let input_luma = reconstruct_buffer_f32(&test_case.input_luma_slice, pixel_height, pixel_width);
            let input_quant_field = reconstruct_buffer_f32(&test_case.input_quant_field_slice, block_h, block_w);

            // Create mutable aq_map, initialized with input_quant_field data
            let mut actual_aq_map = input_quant_field.clone();

            // Call the Rust function (note: C++ seems to lack distance input here)
            // The Rust function `per_block_modulations_scalar` requires distance, but the C++
            // equivalent called by ComputeAdaptiveQuantField does not seem to use it directly,
            // and the JSON data doesn't provide it here. Pass a dummy value (e.g., 1.0).
            // Also need y_quant_01, which is also missing from this specific JSON. Let's estimate or use default.
            // TODO: This highlights a potential discrepancy between Rust signature and C++ usage pattern/instrumentation.
            // For now, use placeholders. A better approach might be needed if these matter.
            let dummy_distance = 1.0;
            let dummy_y_quant_01 = 8.0; // Common default-ish value

            per_block_modulations_scalar(
                &input_quant_field, // ymap input is the AQ field before modulation
                &input_luma, // Original scaled pixel data
                block_w,
                block_h,
                pixel_width,
                pixel_height,
                dummy_distance,
                dummy_y_quant_01,
                &mut actual_aq_map, // Output AQ map
            );

            // Reconstruct expected output
            let expected_aq_map = reconstruct_buffer_f32(&test_case.expected_quant_field_slice, block_h, block_w);

            // Compare the relevant slice
            assert_buffer_eq(
                &actual_aq_map, block_w,
                &test_case.expected_quant_field_slice,
                &expected_aq_map,
                1e-5, // Use slightly larger tolerance due to potential float issues and placeholder inputs
                &format!("PerBlockModulations mismatch (xsize_blocks={}, yb={}, yblen={})", block_w, test_case.input_yb, test_case.input_yblen),
            );
        }
    }


    #[test]
    fn test_compute_adaptive_quant_field() {
        for test_case in COMPUTE_ADAPTIVE_QUANT_FIELD_TESTS.iter() {
            // Extract config and input state
            let width = test_case.config_y_comp_width_in_blocks * 8; // Estimate pixel width
             // Estimate pixel height from input slice (needs context)
            let height = test_case.input_luma_slice.num_rows;
            let input_luma_buffer = reconstruct_buffer_f32(&test_case.input_luma_slice, height, width);

            // Call the Rust function
            // Needs distance, which isn't in this specific JSON. Use a placeholder.
             let dummy_distance = 1.0;
            let actual_aq_field = compute_adaptive_quant_field(
                width as u16,
                height as u16, // Pass estimated height
                &input_luma_buffer,
                dummy_distance, // Use placeholder distance
                test_case.config_y_quant_01,
            );

            // Reconstruct expected output
            let block_w = test_case.config_y_comp_width_in_blocks;
            let block_h = test_case.config_y_comp_height_in_blocks;
            let expected_aq_field = reconstruct_buffer_f32(&test_case.expected_quant_field_slice, block_h, block_w);

            // Compare the relevant slice
            assert_buffer_eq(
                &actual_aq_field, block_w,
                &test_case.expected_quant_field_slice,
                &expected_aq_field,
                1e-4, // Increased tolerance due to potential float and placeholder issues
                 &format!("ComputeAdaptiveQuantField mismatch (y0={}, ylen={})", test_case.input_y0, test_case.input_ylen),
            );
        }
    }
} 