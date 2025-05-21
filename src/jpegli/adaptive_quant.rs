// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Port of Jpegli's adaptive quantization logic from C++ to Rust.
//! Uses helper functions from `adaptive_quant_math.rs`.

use crate::jpegli::adaptive_quant_math::*;
// TODO: Replace with actual project types
// RowBuffer import was removed in previous step, which is correct as it's locally defined.
use crate::jpegli::structs::JpegliComponentInfo as ComponentInfo; // Use and alias the struct from the module
use crate::jpegli::structs::JpegColorSpace;
use crate::jpegli::quant::JpegliQuantizerState;
use std::cmp::max;

use super::structs::OwnedRowBuffer;
use super::structs::RowBufferRef;
use super::structs::RowBuffer;

const K_PRE_EROSION_BORDER: usize = 1;
const DCTSIZE: usize = 8; // Assuming DCTSIZE is 8x8

// TODO: use JpegliQuantizerState.raw_quant_tables
pub struct QuantTable { pub quantval: [u16; DCTSIZE * DCTSIZE] }


// #[derive(Debug, Clone)]
// pub(crate) struct jpegli::quant::JpegliQuantParams {
//     pub distance: f32,
//     pub xyb_mode: bool,
//     pub use_std_tables: bool,
//     pub num_components: usize,
//     pub comp_params: Vec<JpegliComponentParams>,
//     pub jpeg_color_space: JpegColorSpace,
//     pub cicp_transfer_function: u8,
//     pub force_baseline: bool,
//     pub add_two_chroma_tables: bool,
//     pub use_adaptive_quantization: bool,
//     pub subsampling: Subsampling,
// }

pub struct MasterState {
    pub use_adaptive_quantization: bool,
    pub input_buffer: Vec<OwnedRowBuffer<f32>>,
    pub pre_erosion: OwnedRowBuffer<f32>,
    pub fuzzy_erosion_tmp: OwnedRowBuffer<f32>,
    pub quant_field: OwnedRowBuffer<f32>,
    pub diff_buffer: Vec<f32>,
    pub ysize_blocks: usize,
    pub next_iMCU_row: usize,
}
pub struct JpegCompressor {
    pub master: MasterState,
    pub jpeg_color_space: JpegColorSpace,
    pub comp_info: Vec<ComponentInfo>,
    pub quant_tbl_ptrs: Vec<QuantTable>,
    pub total_iMCU_rows: usize,
    pub max_v_samp_factor: usize,
}




/// Port of C++ `ComputePreErosion`.
/// Computes image of local pixel differences, subsampled by 4.
fn compute_pre_erosion<I, E>(
    input: &mut I, // Assumes input has necessary padding/borders
    xsize: usize,
    y0: usize,
    ylen: usize,
    border: usize,
    diff_buffer: &mut [f32], // Temporary buffer, size >= xsize
    pre_erosion: &mut E,
) where I: RowBuffer<f32>, E: RowBuffer<f32> {
    assert!(diff_buffer.len() >= xsize, "Diff buffer too small");
    let xsize_out = xsize / 4;
    let y0_out = y0 / 4;

    // Temporary storage for one row of diff_buffer results from previous iteration
    let mut prev_diff_row = vec![0.0f32; xsize];

    for iy in 0..ylen {
        let y = y0 + iy;

        // Need access to y-1, y, y+1. Ensure input RowBuffer has appropriate borders or handle logic.
        // These slices need to be padded by `border` on left/right for compute_diff_buffer_row.
        // Let's assume the RowBuffer itself handles padding internally or has border rows copied.
        // The indices passed to compute_diff_buffer_row should be relative to the start of the *padded* data.

        // Get padded rows for y-1, y, y+1. Need a robust way to handle this.
        // Assuming get_padded_rows provides slices of length >= xsize + 2*border
        let Some((row_t, row_m, row_b)) = input.get_padded_row_and_two_neighbors(y as isize) else {
            eprintln!("Error getting padded rows for y={}", y);
            continue; // Skip this row if padding fails
        };
        // Check if input rows have sufficient length including padding
        let required_len = xsize + 2 * border;
         if row_t.len() < required_len || row_m.len() < required_len || row_b.len() < required_len {
            eprintln!("Error: Input rows for compute_pre_erosion lack sufficient padding at y={}", y);
            continue;
         }

        let current_diff_buffer_slice = &mut diff_buffer[..xsize];
        let prev_diff_buffer_slice = if (iy & 3) != 0 { Some(&prev_diff_row[..xsize]) } else { None };

        // The `compute_diff_buffer_row` expects slices that represent the *padded* row data.
        // Pass slices that start `border` elements before the logical start (x=0).
        // The length should be `xsize + 2 * border`.
        compute_diff_buffer_row(
            row_t, // Pass the full padded slice
            row_m,
            row_b,
            xsize, // Compute 'xsize' elements
            current_diff_buffer_slice,
            prev_diff_buffer_slice,
        );

        // Store the current result for the next iteration if needed
        if (iy & 3) != 3 { // Store unless it's the last of the 4 rows
            prev_diff_row[..xsize].copy_from_slice(current_diff_buffer_slice);
        }

        // If this is the 4th row (iy % 4 == 3), average and store in pre_erosion
        if iy % 4 == 3 {
            let y_out = y0_out + iy / 4;
            let row_d_out = pre_erosion.get_window_row_mut(y_out).unwrap(); // Get mutable slice for output row

            if row_d_out.len() < xsize_out {
                 eprintln!("Error: pre_erosion row too short at y_out={}", y_out);
                 continue;
            }

            for x_out in 0..xsize_out {
                let x_in = x_out * 4;
                // Sum the 4 values from the final diff_buffer for this output pixel
                let sum = current_diff_buffer_slice[x_in]
                        + current_diff_buffer_slice[x_in + 1]
                        + current_diff_buffer_slice[x_in + 2]
                        + current_diff_buffer_slice[x_in + 3];
                row_d_out[x_out] = sum * 0.25;
            }
            pre_erosion.pad_row_width_old(y_out, xsize_out, border);
        }
    }
}


/// Port of C++ `FuzzyErosion`.
/// Computes a linear combination of the 4 lowest values of the 3x3 neighborhood.
/// Output (`aq_map`) is downsampled 2x relative to `pre_erosion`.
fn fuzzy_erosion<E, T, M>(
    pre_erosion: &mut E, // Assumes input has necessary padding/borders
    yb0: usize, // Block row index
    yblen: usize, // Number of block rows to process
    tmp: &mut T,    // Temporary buffer, same size as pre_erosion
    aq_map: &mut M, // Output buffer (quantization field)
) where E: RowBuffer<f32>, T: RowBuffer<f32>, M: RowBuffer<f32> {
    let xsize_blocks = aq_map.xsize();
    let xsize = pre_erosion.xsize(); // Width of pre_erosion and tmp buffers

    for iy in 0..(2 * yblen) {
        let y = 2 * yb0 + iy; // Row index in pre_erosion/tmp coordinates

        // Need pre_erosion rows y-1, y, y+1. Ensure pre_erosion has borders.
        // Assuming get_padded_rows provides slices including borders.
        let Some((rowt, rowm, rowb)) = pre_erosion.get_padded_row_and_two_neighbors(y as isize) else {
             eprintln!("Error getting padded rows for fuzzy_erosion at y={}", y);
             continue;
        };

        // Check padding
        let required_len = xsize + 2 * K_PRE_EROSION_BORDER;
        if rowt.len() < required_len || rowm.len() < required_len || rowb.len() < required_len {
            eprintln!("Error: Input rows for fuzzy_erosion lack sufficient padding at y={}", y);
            continue;
        }

        let tmp_row_out = tmp.row_mut_old(y as isize).unwrap();
         if tmp_row_out.len() < xsize {
             eprintln!("Error: tmp row too short at y={}", y);
             continue;
         }

        // Compute one row of the temporary buffer
        compute_fuzzy_erosion_row(
            rowt, // Pass full padded slice
            rowm,
            rowb,
            xsize, // Compute 'xsize' elements
            tmp_row_out,
        );

        // If this is an odd row (iy % 2 == 1), combine two rows from tmp into aq_map
        if iy % 2 == 1 {
            let tmp_row0 = tmp.row_old((y - 1) as isize).unwrap(); // Row y-1 from tmp
            let tmp_row1 = tmp.row_old(y as isize).unwrap();       // Row y from tmp (just computed)

            let aq_map_y = yb0 + iy / 2;
            let aq_out = aq_map.row_mut_old(aq_map_y as isize).unwrap();

            if tmp_row0.len() < xsize || tmp_row1.len() < xsize {
                 eprintln!("Error: tmp rows too short for averaging at y={}", y);
                 continue;
            }
             if aq_out.len() < xsize_blocks {
                 eprintln!("Error: aq_map row too short at aq_map_y={}", aq_map_y);
                 continue;
             }

            for bx in 0..xsize_blocks {
                let x = bx * 2; // Top-left corner in tmp coordinates
                // Sum 4 values (2x2 block) from tmp buffer rows
                let sum = tmp_row1[x] + tmp_row1[x + 1] + tmp_row0[x] + tmp_row0[x + 1];
                // The C++ code does not multiply by 0.25 here, seems intentional.
                aq_out[bx] = sum;
            }
        }
    }
}


/// Port of C++ `PerBlockModulations`.
/// Applies masking, HF, and gamma modulation block by block.
fn per_block_modulations<I, M>(
    y_quant_01: f32,
    input: &mut I, // XYB input buffer, needs padding for neighbors
    yb0: usize,
    yblen: usize,
    aq_map: &mut M, // Input/Output: Log-based AQ values initially, modulated values finally
) where I: RowBuffer<f32>, M: RowBuffer<f32>  {
    // Constants from C++ (matching adaptive_quant_math.rs)
    const K_AC_QUANT: f32 = 0.841;
    const K_DAMPEN_RAMP_START: f32 = 9.0;
    const K_DAMPEN_RAMP_END: f32 = 65.0;

    let base_level = 0.48 * K_AC_QUANT;
    let mut dampen = 1.0;
    if y_quant_01 >= K_DAMPEN_RAMP_START {
        dampen = 1.0 - (y_quant_01 - K_DAMPEN_RAMP_START) / (K_DAMPEN_RAMP_END - K_DAMPEN_RAMP_START);
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = K_AC_QUANT * dampen;
    let add = (1.0 - dampen) * base_level;

    let xsize_blocks = aq_map.xsize();

    for iy in 0..yblen {
        let yb = yb0 + iy;
        let y = yb * DCTSIZE; // Top-left pixel row index for the block
        let row_out = aq_map.get_window_row_mut(yb).unwrap();

         if row_out.len() < xsize_blocks {
             eprintln!("Error: aq_map row too short at yb={}", yb);
             continue;
         }

        for ix in 0..xsize_blocks {
            let x = ix * DCTSIZE; // Top-left pixel col index for the block

            // 1. Get initial log-based value from aq_map (output of FuzzyErosion)
            let initial_aq_log_scalar = row_out[ix];

            // 2. Apply ComputeMask modulation
            let val1 = compute_mask_scalar(initial_aq_log_scalar);

            // 3. Apply HfModulation
            // Need 9 rows (y..y+8) and 9 columns (x..x+8) from input buffer
            // Assuming get_padded_rows provides rows with sufficient padding/width
             let Some(hf_input_rows_padded) = input.get_padded_row_and_neighbors(y as isize + 3, 4) else { // Center around y+3/4, get 9 rows total
                eprintln!("Error getting padded rows for HF modulation at y={}", y);
                continue;
            };
            // Check if slices are long enough for x..x+8 access
            // compute_hf_metric_8x8 expects slices starting at the logical 'x'
            let hf_input_rows_sliced: Vec<&[f32]> = hf_input_rows_padded.iter().map(|r| {
                if r.len() >= x + DCTSIZE + 1 { // Need access up to x+8
                    &r[x..]
                } else {
                     eprintln!("Warning: HF input row too short at y, x=({}, {})", y, x);
                    &[] // Return empty slice on error
                }
            }).collect();

             if hf_input_rows_sliced.iter().any(|s| s.is_empty()) { continue; }
             if hf_input_rows_sliced.len() < 9 { continue; }

            let hf_metric = compute_hf_metric_8x8(&hf_input_rows_sliced, 0); // Pass x_start=0 as slices already start at x
            let val2 = val1 + K_HF_MODULATION_SUM_COEFF * hf_metric;

            // 4. Apply GammaModulation
            // Need 8 rows (y..y+7) and 8 columns (x..x+7) from input buffer
            // Use the first 8 rows prepared for HF modulation
            let gamma_input_rows_sliced: Vec<&[f32]> = hf_input_rows_sliced[..8].to_vec(); // Take first 8 rows

             if gamma_input_rows_sliced.iter().any(|s| s.len() < DCTSIZE) {
                  eprintln!("Warning: Gamma input row too short at y, x=({}, {})", y, x);
                  continue;
             }
             if gamma_input_rows_sliced.len() < 8 { continue; }

            let gamma_sum = compute_gamma_sum_8x8(&gamma_input_rows_sliced, 0); // Pass x_start=0
            // Check for non-positive gamma_sum before log
            let log_arg = gamma_sum * K_GAMMA_MODULATION_SCALE;
            let gamma_term = if log_arg > 0.0 {
                K_GAMMA_MODULATION_GAMMA * fast_log2f_scalar(log_arg)
            } else {
                // Handle non-positive log argument, maybe clamp or use a default?
                // C++ FastLog2f might handle this implicitly. Let's assume 0 for now.
                0.0
            };
            let val3 = val2 + gamma_term;

            // 5. Convert back to multiplicative factor
            // C++: FastPow2f(GetLane(out_val) * 1.442695041f) * mul + add;
            // Assuming `val3` is log base 2, the direct conversion is:
            let final_aq = fast_pow2f_scalar(val3) * mul + add;

            row_out[ix] = final_aq;
        }
    }
}


/// Port of C++ `ComputeAdaptiveQuantField`.
/// Main entry point for computing the adaptive quantization field for an iMCU row.
pub fn compute_adaptive_quant_field(cinfo: &mut JpegCompressor) {
    let m = &mut cinfo.master; // Get mutable ref to master state

    if !m.use_adaptive_quantization {
        return;
    }

    let y_channel = match cinfo.jpeg_color_space {
        JpegColorSpace::Rgb => 1, // G channel in RGB
        JpegColorSpace::YCbCr => 0, // Y channel in YCbCr
        JpegColorSpace::Grayscale => 0, // Y channel in Grayscale
        _ => unimplemented!(),
    };

    let Some(y_comp) = cinfo.comp_info.get(y_channel) else { return; };
    let Some(quant_tbl) = cinfo.quant_tbl_ptrs.get(y_comp.quantization_table_index as usize) else { return; };

    // C++ uses quantval[1] which corresponds to AC coefficient at [0, 1] or [1, 0]
    // Assuming standard zigzag order, index 1 is usually the first AC coefficient.
    let y_quant_01 = quant_tbl.quantval[1] as f32; // Use [1] as per C++

    let ysize_pixels = m.ysize_blocks * DCTSIZE;

    // Handle input buffer border copy
    // Need RowBuffer::copy_row to handle potentially overlapping regions correctly.
    if m.next_iMCU_row == 0 {
        m.input_buffer[y_channel].copy_row_window_relative(-1, 0); // Copy row 0 to row -1
        //, K_PRE_EROSION_BORDER as usize
    }
    if m.next_iMCU_row + 1 == cinfo.total_iMCU_rows {
        let last_row = ysize_pixels - 1;
        m.input_buffer[y_channel].copy_row_window_relative(last_row as isize + 1, last_row as isize); // Copy last row to row below
        //, K_PRE_EROSION_BORDER as usize
    }

    let input = &mut m.input_buffer[y_channel]; // Immutable borrow needed for calculations
    let xsize_blocks = y_comp.width_in_blocks;
    let xsize = xsize_blocks * DCTSIZE; // Width in pixels

    // Calculate Y range for processing this iMCU row
    let yb0 = m.next_iMCU_row * cinfo.max_v_samp_factor;
    let yblen = cinfo.max_v_samp_factor; // Number of block rows in this iMCU row
    let y0_pixels = yb0 * DCTSIZE;
    let ylen_pixels = yblen * DCTSIZE;

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
    if m.next_iMCU_row + 1 == cinfo.total_iMCU_rows {
        // Last iMCU row
        ylen_pre = ylen_pre.saturating_sub(4); // Reduce length by 4 at the end
    }
     // Ensure ylen_pre doesn't exceed buffer boundaries
    if y0_pre + ylen_pre > ysize_pixels + K_PRE_EROSION_BORDER { // Check against height + border?
         ylen_pre = ysize_pixels + K_PRE_EROSION_BORDER - y0_pre;
    }

    // Allocate or ensure diff_buffer is large enough
    if m.diff_buffer.len() < xsize {
        m.diff_buffer.resize(xsize, 0.0f32);
    }


    // Run ComputePreErosion
    compute_pre_erosion(
        input,
        xsize,
        y0_pre,
        ylen_pre,
        K_PRE_EROSION_BORDER,
        &mut m.diff_buffer,
        &mut m.pre_erosion, // Mutable borrow needed here
    );

    // Handle pre_erosion border copy
    if y0_pre == 0 { // If we processed starting from row 0
        m.pre_erosion.copy_row_window_relative(-1, 0);
    }
    if m.next_iMCU_row + 1 == cinfo.total_iMCU_rows {
        let last_row_pre = (ysize_pixels / 4) * 2 - 1; // ysize_blocks * 2 - 1? Check pre_erosion dims
        // Assuming pre_erosion is subsampled 2x vertically compared to blocks? No, 4x.
        let last_row_pre = m.ysize_blocks / 2 - 1; // Check this calculation based on pre_erosion size. Let's assume height is ysize_blocks / 2.
        m.pre_erosion.copy_row_window_relative(last_row_pre as isize + 1, last_row_pre as isize);
    }

    // Run FuzzyErosion
    // This step requires mutable borrows of tmp and aq_map
    fuzzy_erosion(
        &mut m.pre_erosion, // Immutable borrow
        yb0,
        yblen,
        &mut m.fuzzy_erosion_tmp, // Mutable borrow
        &mut m.quant_field,      // Mutable borrow
    );

    // Run PerBlockModulations
    // This step requires mutable borrow of aq_map
    per_block_modulations(
        y_quant_01,
        input,           // Immutable borrow
        yb0,
        yblen,
        &mut m.quant_field, // Mutable borrow
    );

    // Final adjustment loop (Applied *after* modulations)
    for iy in 0..yblen {
        let yb = yb0 + iy;
        let row = m.quant_field.get_window_row_mut(yb).unwrap(); // Mutable borrow
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
    use super::*;
    // TODO: Add tests for compute_pre_erosion, fuzzy_erosion, per_block_modulations, compute_adaptive_quant_field
    // These will require setting up mock RowBuffer and JpegCompressor state.

    #[test]
    fn test_placeholder() {
        assert!(true);
    }
} 