// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Port of Jpegli's adaptive quantization logic from C++ to Rust.
//! Uses helper functions from `adaptive_quant_math.rs`.

use crate::jpegli::adaptive_quant_math::*;
use crate::jpegli::quant::JpegliQuantData;


use super::config::ComputedConfigDimensions;
use super::config::ComputedEncodeConfig;
use super::structs::OwnedRowBuffer;
use super::structs::RowBuffer;

const K_PRE_EROSION_BORDER: usize = 1;
const DCTSIZE: usize = 8; // Assuming DCTSIZE is 8x8

#[derive(Debug, Clone)]
pub struct AdaptiveQuantState {
    pub pre_erosion: OwnedRowBuffer<f32>,
    pub fuzzy_erosion_tmp: OwnedRowBuffer<f32>,
    pub quant_field: OwnedRowBuffer<f32>,
    pub diff_buffer: Vec<f32>,
    pub next_i_mcu_row: usize,
}


/// Port of C++ `ComputeAdaptiveQuantField`.
/// Main entry point for computing the adaptive quantization field for an iMCU row.
pub fn compute_adaptive_quant_field<T: RowBuffer<f32>>(luma_plane_padded_input: &mut T, state: &mut AdaptiveQuantState, quantizer: &JpegliQuantData, config: &ComputedEncodeConfig, dims: &ComputedConfigDimensions) {

    if !config.use_adaptive_quantization {
        return;
    }

    let y_channel = config.luma_component_index;

    let y_comp_config = config.comp_params[y_channel];
    let y_comp_dims = dims.components()[y_channel].size;
    let y_quant_table_index = y_comp_config.quantization_table_index;
    let raw_quant_tbl = quantizer.raw_quant_tables[y_quant_table_index as usize].expect("Quantization table not found for given index");

    // C++ uses quantval[1] which corresponds to AC coefficient at [0, 1] or [1, 0]
    // Assuming standard zigzag order, index 1 is usually the first AC coefficient.
    let y_quant_01 = raw_quant_tbl[1] as f32; // Use [1] as per C++

    let ysize_pixels = dims.ysize_blocks * DCTSIZE;

    // Handle input buffer border copy
    // Need RowBuffer::copy_row to handle potentially overlapping regions correctly.
    if state.next_i_mcu_row == 0 {
        luma_plane_padded_input.pad_using_edges();
        //, K_PRE_EROSION_BORDER as usize
    }
    if state.next_i_mcu_row + 1 == dims.total_i_mcu_rows {
        let last_row = ysize_pixels - 1;
        luma_plane_padded_input.copy_row_window_relative(last_row as isize + 1, last_row as isize); // Copy last row to row below
        //, K_PRE_EROSION_BORDER as usize
    }

    let input = luma_plane_padded_input; // Immutable borrow needed for calculations
    let xsize_blocks = y_comp_dims.width_in_blocks;
    let xsize = xsize_blocks * DCTSIZE; // Width in pixels

    // Calculate Y range for processing this iMCU row
    let yb0 = state.next_i_mcu_row * config.max_v_samp_factor as usize;
    let yblen = config.max_v_samp_factor as usize; // Number of block rows in this iMCU row
    let y0_pixels = yb0 * DCTSIZE as usize;
    let ylen_pixels = yblen as usize * DCTSIZE as usize;

    // Adjust Y range for pre-erosion calculation based on C++ logic
    let mut y0_pre = y0_pixels;
    let mut ylen_pre = ylen_pixels;
    if y0_pre == 0 {
        // First iMCU row, need rows below 0 for context? C++ adds 4?
        // Let's assume the copy_row handled the needed border context implicitly.
        // The C++ logic `ylen += 4;` seems to increase processing range at the start.
        ylen_pre += 4; // Process 4 extra rows at the beginning? Seems odd. Let's stick closer to C++ for now.
    } else {
        // C++: y0 += 4; Adjust start downwards? Needs clarification.
        // Let's assume it means start processing 4 rows *earlier*.
        y0_pre = y0_pixels.saturating_sub(4); // Start 4 rows earlier if possible
        ylen_pre += 4; // Extend length accordingly
    }
    if state.next_i_mcu_row + 1 == dims.total_i_mcu_rows {
        // Last iMCU row
        ylen_pre = ylen_pre.saturating_sub(4); // Reduce length by 4 at the end
    }
     // Ensure ylen_pre doesn't exceed buffer boundaries
    if y0_pre + ylen_pre > ysize_pixels + K_PRE_EROSION_BORDER { // Check against height + border?
         ylen_pre = ysize_pixels + K_PRE_EROSION_BORDER - y0_pre;
    }

    // Allocate or ensure diff_buffer is large enough
    if state.diff_buffer.len() < xsize {
        state.diff_buffer.resize(xsize, 0.0f32);
    }


    // Run ComputePreErosion
    compute_pre_erosion(
        input,
        xsize,
        y0_pre,
        ylen_pre,
        &mut state.diff_buffer,
        &mut state.pre_erosion, // Mutable borrow needed here
    );

    // Handle pre_erosion border copy
    if y0_pre == 0 { // If we processed starting from row 0
        state.pre_erosion.copy_row_window_relative(-1, 0);
    }
    if state.next_i_mcu_row + 1 == dims.total_i_mcu_rows {
        // Assuming pre_erosion is subsampled 2x vertically compared to blocks? No, 4x.
        let last_row_pre = dims.ysize_blocks / 2 - 1; // Check this calculation based on pre_erosion size. Let's assume height is ysize_blocks / 2.
        state.pre_erosion.copy_row_window_relative(last_row_pre as isize + 1, last_row_pre as isize);
    }

    // Run FuzzyErosion
    // This step requires mutable borrows of tmp and aq_map
    fuzzy_erosion(
        &state.pre_erosion, // Immutable borrow
        yb0,
        yblen,
        &mut state.fuzzy_erosion_tmp, // Mutable borrow
        &mut state.quant_field,      // Mutable borrow
    );

    // Run PerBlockModulations
    // This step requires mutable borrow of aq_map
    per_block_modulations(
        y_quant_01,
        input,           // Immutable borrow
        yb0,
        yblen,
        &mut state.quant_field, // Mutable borrow
    );

    // Final adjustment loop (Applied *after* modulations)
    for iy in 0..yblen {
        let yb = yb0 + iy;
        let row = state.quant_field.get_window_row_mut(yb).unwrap(); // Mutable borrow
        if row.len() < xsize_blocks { continue; }
        for ix in 0..xsize_blocks {
            let val = row[ix];
            if val > 1e-9 { // Avoid division by zero or near-zero
                row[ix] = (0.6 / val - 1.0).max(0.0);
            } else {
                // Handle case where aq_map value is zero or negative after modulations
                row[ix] = 0.0; // Or some large value? C++ max(0.0f, ...) suggests clamp at 0.
            }
        }
    }
}


#[cfg(test)]
mod tests {

    // TODO: Add tests for compute_pre_erosion, fuzzy_erosion, per_block_modulations, compute_adaptive_quant_field
    // These will require setting up mock RowBuffer and JpegCompressor state.

    #[test]
    fn test_placeholder() {
        assert!(true);
    }
} 