// Helper functions for tests

use serde::{Deserialize, Deserializer};
use crate::jpegli::tests::test_structs::RustRowBufferSliceF32; // Import the slice struct
use alloc::string::{String, ToString}; // Add imports for String formatting
use alloc::vec::Vec;                   // Keep Vec import
use core::fmt::Write;                  // Add Write import

// Example deserialization helper for BlockF32/BlockI32 data field
// Assumes JSON data is a flat list of 64 numbers

pub fn deserialize_block_f32<'de, D>(deserializer: D) -> Result<[f32; 64], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<f32> = Vec::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<f32>| {
        serde::de::Error::custom(format!(
            "Expected a list of 64 f32 elements, got {}",
            v.len()
        ))
    })
}

pub fn deserialize_block_i32<'de, D>(deserializer: D) -> Result<[i32; 64], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<i32> = Vec::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<i32>| {
        serde::de::Error::custom(format!(
            "Expected a list of 64 i32 elements, got {}",
            v.len()
        ))
    })
}

// Add other test utilities here, e.g., float comparison with tolerance
pub fn assert_float_eq(a: f32, b: f32, tolerance: f32, message: &str) {
    if (a - b).abs() > tolerance {
        panic!("{}: {} != {} (tolerance: {})", message, a, b, tolerance);
    }
}

pub fn assert_block_f32_eq(a: &[f32; 64], b: &[f32; 64], tolerance: f32, message: &str) {
    for i in 0..64 {
        if (a[i] - b[i]).abs() > tolerance {
            panic!("{}: Block mismatch at index {}: {} != {} (tolerance: {})",
                   message, i, a[i], b[i], tolerance);
        }
    }
}

pub fn assert_block_i32_eq(a: &[i32; 64], b: &[i32; 64], message: &str) {
    for i in 0..64 {
        if a[i] != b[i] {
            panic!("{}: Block mismatch at index {}: {} != {}",
                   message, i, a[i], b[i]);
        }
    }
}

// --- Struct to hold comparison results ---
#[derive(Debug, Clone)]
pub(crate) struct BufferCompareResult {
    pub description: String, // e.g., "ComputePreErosion mismatch for image.png"
    pub diff_count: usize,
    pub max_diff: f32,
    pub sum_abs_diff: f64, // Use f64 for precision
    pub diff_details: Option<String>, // List of differences
}

impl BufferCompareResult {
    fn success(description: &str) -> Self {
        Self {
            description: description.to_string(),
            diff_count: 0,
            max_diff: 0.0,
            sum_abs_diff: 0.0,
            diff_details: None,
        }
    }

    fn failure(
        description: &str,
        diff_count: usize,
        max_diff: f32,
        sum_abs_diff: f64,
        diff_details: String,
    ) -> Self {
        Self {
            description: description.to_string(),
            diff_count,
            max_diff,
            sum_abs_diff,
            diff_details: Some(diff_details),
        }
    }

    pub fn is_success(&self) -> bool {
        self.diff_count == 0
    }

    // Helper to format the failure details nicely
    pub fn failure_summary(&self) -> String {
        if self.is_success() {
            return format!("PASS: {}", self.description);
        }
        let avg_error = if self.diff_count > 0 { self.sum_abs_diff / self.diff_count as f64 } else { 0.0 };
        let mut summary = format!(
            "FAIL: {} (Diffs: {}, Max: {:.6}, Avg: {:.6}, Sum: {:.6})",
            self.description, self.diff_count, self.max_diff, avg_error, self.sum_abs_diff
        );
        if let Some(details) = &self.diff_details {
            write!(summary, "\n  Details (up to 10):\n{}", details).unwrap();
        }
        summary
    }
}
// --- End Struct Definition ---


/// Compares an actual planar buffer against the data provided in an expected buffer slice.
/// Assumes both buffers represent the *same slice* starting at index 0.
pub(crate) fn compare_buffer_slice(
    actual_buffer: &[f32],
    actual_stride: usize, // Stride (width) of the actual_buffer
    expected_slice_info: &RustRowBufferSliceF32, // Used only for expected dimensions
    expected_buffer: &[f32], // This buffer *is* the slice data, flat
    tolerance: f32,
    description: &str,
) -> BufferCompareResult {
    let expected_slice_rows = expected_slice_info.num_rows;
    let expected_slice_cols = expected_slice_info.num_cols;
    let expected_len = expected_slice_rows * expected_slice_cols;

    // Basic size check for the expected buffer
    if expected_buffer.len() != expected_len {
        return BufferCompareResult::failure(
            description, 1, 0.0, 0.0,
            format!("  FATAL: Expected buffer size mismatch. Slice requires {}x{}={} elements, but expected_buffer has {}. Check test data generation.",
                    expected_slice_rows, expected_slice_cols, expected_len, expected_buffer.len())
        );
    }

     // Check if actual buffer has at least the expected length
     // Functions might produce slightly different sizes due to rounding (e.g., width+3)/4
     // Allow actual to be slightly larger, but must contain the expected slice.
    if actual_buffer.len() < expected_len {
         return BufferCompareResult::failure(
            description, 1, 0.0, 0.0,
            format!("  FATAL: Actual buffer too small. Expected at least {} elements ({}x{}), but actual_buffer has only {} elements.",
                    expected_len, expected_slice_rows, expected_slice_cols, actual_buffer.len())
        );
    }
     // Check if actual_stride is valid
    if actual_stride == 0 && expected_slice_cols > 0 {
         return BufferCompareResult::failure(
            description, 1, 0.0, 0.0,
            "  FATAL: Actual buffer stride cannot be 0.".to_string()
        );
    }

    let mut diff_count = 0;
    let mut max_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut diff_output = String::new();
    let mut diff_details_count = 0;
    const MAX_DETAILS_TO_PRINT: usize = 10;

    // Iterate through the expected slice dimensions
    for r_slice_idx in 0..expected_slice_rows {
        for c_slice_idx in 0..expected_slice_cols {
            // Index in the flat expected buffer
            let expected_idx = r_slice_idx * expected_slice_cols + c_slice_idx;
            let expected_val = expected_buffer[expected_idx]; // Bounds checked earlier

            // Index in the actual buffer (relative to slice start, 0-based)
            let actual_idx = r_slice_idx * actual_stride + c_slice_idx;

            // Check bounds for actual_buffer access
            if actual_idx >= actual_buffer.len() {
                return BufferCompareResult::failure(
                    description, diff_count + 1, max_diff, sum_abs_diff,
                    format!("{}\n  FATAL: Actual buffer index out of bounds. Tried index {} (row {}, col {}) for slice element, but buffer length is {}. Check function output dimensions/stride.",
                           diff_output, actual_idx, r_slice_idx, c_slice_idx, actual_buffer.len())
                );
            }
            let actual_val = actual_buffer[actual_idx];

            // Perform comparison
            let diff = (actual_val - expected_val).abs();
             let is_mismatch = diff > tolerance; // Assuming non-NaN for simplicity now

            if is_mismatch {
                 diff_count += 1;
                 max_diff = max_diff.max(diff);
                 sum_abs_diff += diff as f64;

                if diff_details_count < MAX_DETAILS_TO_PRINT {
                    // Use slice row/col for reporting
                    writeln!(diff_output, "  Slice[r={},c={}] (Actual Idx {}): Expected={:.6}, Actual={:.6}, Diff={:.6}",
                            r_slice_idx, c_slice_idx, actual_idx,
                             expected_val, actual_val, diff).unwrap();
                    diff_details_count += 1;
                } else if diff_details_count == MAX_DETAILS_TO_PRINT {
                     writeln!(diff_output, "  ... (further differences omitted)").unwrap();
                     diff_details_count += 1;
                }
            }
        }
    }

    if diff_count > 0 {
        BufferCompareResult::failure(
            description,
            diff_count,
            max_diff,
            sum_abs_diff,
            diff_output,
        )
    } else {
        BufferCompareResult::success(description)
    }
}

// Remove the old assert_buffer_eq or comment it out
/*
pub fn assert_buffer_eq(
    actual_buffer: &[f32],
    actual_stride: usize,
    expected_slice: &RustRowBufferSliceF32,
    expected_buffer: &[f32], // This buffer *is* the slice data, flat
    tolerance: f32,
    message: &str,
) {
    // Check dimensions of the expected buffer match the slice description
    if expected_buffer.len() != expected_slice.num_rows * expected_slice.num_cols {
        panic!(
            "{}: Expected buffer size mismatch. Slice description requires {}x{}={} elements, but expected_buffer has {} elements.",
            message,
            expected_slice.num_rows, expected_slice.num_cols,
            expected_slice.num_rows * expected_slice.num_cols,
            expected_buffer.len()
        );
    }

    let expected_slice_width = expected_slice.num_cols;

    for r_slice_idx in 0..expected_slice.num_rows {
        let actual_row = (expected_slice.start_row + r_slice_idx as isize) as usize;

        for c_slice_idx in 0..expected_slice.num_cols {
            let actual_col = (expected_slice.start_col + c_slice_idx as isize) as usize;

            // Calculate index in the actual flat buffer
            let actual_idx = actual_row * actual_stride + actual_col;

            // Calculate index in the expected flat buffer (which only contains the slice)
            let expected_idx = r_slice_idx * expected_slice_width + c_slice_idx;

            if actual_idx >= actual_buffer.len() {
                 panic!(
                    "{}: Actual buffer index out of bounds. Tried to access index {} (row {}, col {}) for slice element ({}, {}), but buffer length is {}. Check strides and dimensions.",
                    message, actual_idx, actual_row, actual_col, r_slice_idx, c_slice_idx, actual_buffer.len()
                 );
            }
             if expected_idx >= expected_buffer.len() {
                 // This should theoretically be caught by the size check earlier, but double-check.
                 panic!(
                    "{}: Expected buffer index out of bounds. Tried to access index {} for slice element ({}, {}), but expected buffer length is {}. Check slice description.",
                    message, expected_idx, r_slice_idx, c_slice_idx, expected_buffer.len()
                 );
             }

            let actual_val = actual_buffer[actual_idx];
            let expected_val = expected_buffer[expected_idx];

            if (actual_val - expected_val).abs() > tolerance {
                panic!(
                    "{}: Mismatch at slice coords (row={}, col={}) [Actual buffer index {} (row {}, col {})]. Expected: {}, Actual: {}, Tolerance: {}",
                    message,
                    r_slice_idx, c_slice_idx,
                    actual_idx, actual_row, actual_col,
                    expected_val, actual_val, tolerance
                );
            }
        }
    }
}
*/ 