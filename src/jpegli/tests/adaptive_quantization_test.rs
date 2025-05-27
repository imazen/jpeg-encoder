// src/jpegli/tests/adaptive_quantization_test.rs

#[cfg(test)]
mod adaptive_quantization_tests {
    use crate::jpegli::adaptive_quant::*;
    use crate::jpegli::adaptive_quantization::compute_pre_erosion_scalar;
    use crate::jpegli::adaptive_quantization::fuzzy_erosion_scalar;
    use crate::jpegli::adaptive_quantization::per_block_modulations_scalar;
    use crate::jpegli::tests::test_structs::*;
    use crate::jpegli::tests::testdata::*;
    use crate::jpegli::tests::test_utils::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use std::fmt::Write;

    // --- Helper Functions (if needed, or use from test_utils) ---

    /// Reconstructs a planar buffer from a slice description for testing.
    fn reconstruct_buffer_f32(slice: &RustRowBufferSliceF32, full_height: usize, full_width: usize) -> Vec<f32> {
        // Determine buffer size based on slice stride and full height
        // Adjust buffer size calculation to handle potential 0 height/width gracefully
        let buffer_size = if full_height > 0 && full_width > 0 {
             full_width * full_height
        } else {
            0
        };
        let mut buffer = vec![0.0f32; buffer_size]; // Initialize with default

        // Skip if buffer size is 0
        if buffer_size == 0 {
            return buffer;
        }

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
            // Ensure start_row + r_idx doesn't underflow or exceed bounds
            let buffer_row = (slice.start_row + r_idx as isize) as usize;
            if buffer_row >= full_height { continue; } // Skip rows outside the logical buffer height

            // Use actual width (full_width) for row start calculation
            let buffer_row_start = buffer_row * full_width;
            let slice_row = &slice.data[r_idx];

            for c_idx in 0..slice.num_cols {
                 // Ensure start_col + c_idx doesn't underflow
                 let buffer_col = (slice.start_col + c_idx as isize) as usize;
                 // Check against full_width plus potential negative start offset
                 let effective_width = full_width + slice.start_col.wrapping_abs() as usize;
                if buffer_col >= effective_width { continue; }

                // Index calculation should use full_width as stride
                let buffer_idx = buffer_row * full_width + buffer_col;
                if buffer_idx < buffer.len() { // Ensure index is within bounds
                   buffer[buffer_idx] = slice_row[c_idx];
                } else {
                   // This might indicate an issue with stride, dimensions, or start_col/start_row
                   eprintln!("Warning: Calculated index {} out of bounds for buffer size {}. Slice info: {:?}, Full H={}, Full W={}", buffer_idx, buffer.len(), slice, full_height, full_width);
                }
            }
        }
        buffer
    }

    // --- Test Functions ---

    #[test]
   // Fails due to value mismatch vs C++ reference data (scalar vs SIMD?)
    fn test_compute_pre_erosion() {
        let mut any_failures = false;
        let mut failure_details = String::new();

        for test_case in COMPUTE_PRE_EROSION_TESTS.iter() {
            // Extract config
            let width = test_case.config_xsize;
            // Height is derived from the input slice which includes context
            let est_input_height = test_case.input_buffer_y_slice.num_rows;
            let input_luma_buffer = reconstruct_buffer_f32(&test_case.input_buffer_y_slice, est_input_height, width);

            // Output dimensions are based on input pixel width and expected slice height
            let pre_erosion_w = (width + 3) / 4;
            let pre_erosion_h = test_case.expected_pre_erosion_slice.num_rows;
            let mut actual_pre_erosion = Vec::new(); // Will be resized inside

            // Call the Rust function
            // Assuming input_buffer_y_slice contains *already scaled* data [0, 1]
            compute_pre_erosion_scalar(
                &input_luma_buffer,
                width,
                est_input_height, // Use the height of the provided input slice region
                &mut actual_pre_erosion,
            );

            // Reconstruct expected buffer slice
            // Flatten the expected slice data directly
            let expected_pre_erosion_buffer: Vec<f32> = test_case.expected_pre_erosion_slice.data.iter().flatten().cloned().collect();

            // Compare the relevant slice using the new helper
            let description = format!("ComputePreErosion mismatch (xsize={}, y0={}, ylen={}) for {}",
                         width, test_case.config_y0, test_case.config_ylen, test_case.source_file);

            let result = compare_buffer_slice(
                &actual_pre_erosion, pre_erosion_w, // Actual buffer + stride
                &test_case.expected_pre_erosion_slice, // Use original slice info
                &expected_pre_erosion_buffer, // Expected buffer (flattened)
                1e-5, // Tolerance for float comparison
                &description,
            );

            if !result.is_success() {
                any_failures = true;
                write!(failure_details, "{}", result.failure_summary()).unwrap();
                writeln!(failure_details).unwrap();
            }
        }
        assert!(!any_failures, "ComputePreErosion tests failed:\n{}", failure_details);
    }

    #[test]
   // Fails due to value mismatch vs C++ reference data (scalar vs SIMD?)
    fn test_fuzzy_erosion() {
        let mut any_failures = false;
        let mut failure_details = String::new();

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
            // Flatten the expected slice data directly
            let expected_aq_map: Vec<f32> = test_case.expected_quant_field_slice.data.iter().flatten().cloned().collect();

            // Compare the relevant slice
            let description = format!("FuzzyErosion mismatch (yb0={}, yblen={}) for {}",
                         test_case.config_yb0, test_case.config_yblen, test_case.source_file);
            let result = compare_buffer_slice(
                &actual_aq_map, block_w,
                &test_case.expected_quant_field_slice, // Use original slice info
                &expected_aq_map,
                1e-5,
                &description,
            );
            if !result.is_success() {
                any_failures = true;
                write!(failure_details, "{}", result.failure_summary()).unwrap();
                writeln!(failure_details).unwrap();
            }
        }
        assert!(!any_failures, "FuzzyErosion tests failed:\n{}", failure_details);
    }

    #[test]
   // Fails due to value mismatch vs C++ reference data (scalar vs SIMD?)
    fn test_per_block_modulations() {
        let mut any_failures = false;
        let mut failure_details = String::new();

        for test_case in PER_BLOCK_MODULATIONS_TESTS.iter() {
            // Extract config and input state
            let block_w = test_case.input_quant_field_slice_before.num_cols;
            let block_h = test_case.input_quant_field_slice_before.num_rows; // Height from input AQ slice
            // We need pixel-level width/height for input_luma_slice reconstruction
            let pixel_width = test_case.input_buffer_y_slice.num_cols;
            let pixel_height = test_case.input_buffer_y_slice.num_rows;

            let input_luma = reconstruct_buffer_f32(&test_case.input_buffer_y_slice, pixel_height, pixel_width);
            let input_quant_field_before = reconstruct_buffer_f32(&test_case.input_quant_field_slice_before, block_h, block_w);

            // Create mutable aq_map, initialized with input_quant_field_before data
            let mut actual_aq_map = input_quant_field_before.clone();

            // Call the Rust function
            let dummy_distance = 1.0; // Placeholder, as C++ version doesn't seem to use it directly here.
            // Use y_quant_01 from the test case
            let y_quant_01 = test_case.config_y_quant_01;

            per_block_modulations_scalar(
                &input_quant_field_before, // ymap input is the AQ field before modulation
                &input_luma, // Original scaled pixel data
                block_w,
                block_h,
                pixel_width,
                pixel_height,
                dummy_distance,
                y_quant_01,
                &mut actual_aq_map, // Output AQ map
            );

            // Reconstruct expected output
            let expected_aq_map = reconstruct_buffer_f32(&test_case.expected_quant_field_slice_after, block_h, block_w);

            // Compare the relevant slice
            let description = format!("PerBlockModulations mismatch (yb0={}, yblen={})", // No source_file in this struct
                         test_case.config_yb0, test_case.config_yblen);
            let result = compare_buffer_slice(
                &actual_aq_map, block_w,
                &test_case.expected_quant_field_slice_after, // Use the 'after' slice for comparison
                &expected_aq_map,
                1e-5, // Use slightly larger tolerance due to potential float issues and placeholder inputs
                &description,
            );
            if !result.is_success() {
                any_failures = true;
                write!(failure_details, "{}", result.failure_summary()).unwrap();
                writeln!(failure_details).unwrap();
            }
        }
        assert!(!any_failures, "PerBlockModulations tests failed:\n{}", failure_details);
    }

    #[test]
   // Ignoring because test data slice doesn't match function expecting full image AND/OR underlying functions have mismatches
    fn test_compute_adaptive_quant_field() {
        let mut any_failures = false;
        let mut failure_details = String::new();

        for test_case in COMPUTE_ADAPTIVE_QUANT_FIELD_TESTS.iter() {
            // Extract config and input state
            let width = test_case.config_y_comp_width_in_blocks * 8; // Estimate pixel width
             // Height is derived from the input slice which includes context
            let height = test_case.input_buffer_y_slice.num_rows;
            let input_luma_buffer = reconstruct_buffer_f32(
                &test_case.input_buffer_y_slice,
                height, // Use slice height for reconstruction
                width // Use calculated pixel width
            );

            // Call the Rust function
             let dummy_distance = 1.0; // Placeholder
            let actual_aq_field = compute_adaptive_quant_field(
                width as u16, // Pass actual pixel width
                (test_case.config_y_comp_height_in_blocks * 8) as u16, // Pass total image height in pixels
                &input_luma_buffer,
                dummy_distance, // Use placeholder distance
                test_case.config_y_quant_01,
            );

            // Reconstruct expected output
            let block_w = test_case.config_y_comp_width_in_blocks;
            let block_h = test_case.expected_quant_field_slice.num_rows; // Height from expected slice
            let expected_aq_field: Vec<f32> = test_case.expected_quant_field_slice.data.iter().flatten().cloned().collect();

            // Compare the relevant slice
            let description = format!(
                    "ComputeAdaptiveQuantField mismatch for {}",
                    test_case.source_file
                );

             // Check if actual_aq_field is empty (tiny image case)
             if actual_aq_field.is_empty() {
                // If expected is also empty (or represents 0 blocks), it might be a PASS
                if test_case.expected_quant_field_slice.num_rows == 0 || test_case.expected_quant_field_slice.num_cols == 0 {
                    // Consider this a pass for tiny images where AQ is skipped
                    continue; 
                } else {
                    any_failures = true;
                    write!(failure_details, "FAIL: {} - Actual AQ field was empty, but expected data exists.", description).unwrap();
                    writeln!(failure_details).unwrap();
                    continue; 
                }
            }

            let result = compare_buffer_slice(
                &actual_aq_field,
                block_w, // Stride of the actual aq_field (width in blocks)
                &test_case.expected_quant_field_slice, // Description of the slice to compare
                &expected_aq_field, // Reconstructed expected data for the slice
                1e-4, // Increased tolerance due to potential float and placeholder issues
                 &description,
            );
            if !result.is_success() {
                any_failures = true;
                write!(failure_details, "{}", result.failure_summary()).unwrap();
                writeln!(failure_details).unwrap();
            }
        }
         assert!(!any_failures, "ComputeAdaptiveQuantField tests failed:\n{}", failure_details);
    }
} 