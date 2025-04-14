//! Adaptive quantization logic ported from Jpegli.

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

// Constants ported from adaptive_quantization.cc
const K_MASK_MULTIPLIER: f32 = 0.855;
const K_EDGE_MULTIPLIER: f32 = 0.6;
const K_BORDER_MULTIPLIER: f32 = 0.125;

// Gamma-related constant from ComputePreErosion
const MATCH_GAMMA_OFFSET: f32 = 0.019; // Note: Jpegli divides by kInputScaling (255.0), applied at usage.
const LIMIT: f32 = 0.2;

// Constants from MaskingSqrt
const K_LOG_OFFSET_SQRT: f32 = 28.0;
const K_MUL_SQRT: f32 = 211.50759899638012 * 1e8;

// Constants for PerBlockModulations
const K_AC_QUANT: f32 = 0.841;
// Note: kInputScaling (255.0) is applied where needed
const K_GAMMA_MOD_BIAS: f32 = 0.16; // 0.16f / kInputScaling
const K_GAMMA_MOD_SCALE: f32 = 1.0 / 64.0; // kInputScaling / 64.0f / kInputScaling
const K_GAMMA_MOD_GAMMA: f32 = -0.15526878023684174 * 0.6931471805599453; // Includes kInvLog2e
const K_HF_MOD_COEFF: f32 = -2.0052193233688884 / 112.0; // * kInputScaling / kInputScaling

// Constants for ComputeMask
const K_MASK_BASE: f32 = -0.74174993;
const K_MASK_MUL4: f32 = 3.2353257320940401;
const K_MASK_MUL2: f32 = 12.906028311180409;
const K_MASK_OFFSET2: f32 = 305.04035728311436;
const K_MASK_MUL3: f32 = 5.0220313103171232;
const K_MASK_OFFSET3: f32 = 2.1925739705298404;
const K_MASK_OFFSET4: f32 = 0.25 * K_MASK_OFFSET3;
const K_MASK_MUL0: f32 = 0.74760422233706747;

// --- Helper Functions --- 

/// Simple 1D Gaussian kernel generation.
fn gaussian_kernel(sigma: f32, radius: usize) -> Vec<f32> {
    let mut kernel = vec![0.0; 2 * radius + 1];
    let sigma_sq = sigma * sigma;
    let norm_factor = 1.0 / (2.0 * PI * sigma_sq).sqrt();
    let mut sum = 0.0;

    for i in 0..=radius {
        let dist_sq = (i * i) as f32;
        let val = norm_factor * (-dist_sq / (2.0 * sigma_sq)).exp();
        kernel[radius + i] = val;
        kernel[radius - i] = val;
        sum += if i == 0 { val } else { 2.0 * val };
    }

    // Normalize kernel
    if sum > 1e-6 {
        for val in &mut kernel {
            *val /= sum;
        }
    }

    kernel
}

/// Apply horizontal 1D convolution.
fn convolve_horizontal(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) {
    let radius = kernel.len() / 2;
    for y in 0..height {
        let row_in_start = y * width;
        let row_out_start = y * width;
        for x in 0..width {
            let mut sum = 0.0;
            for k_idx in 0..kernel.len() {
                let offset = k_idx as i32 - radius as i32;
                let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as usize;
                sum += input[row_in_start + sample_x] * kernel[k_idx];
            }
            output[row_out_start + x] = sum;
        }
    }
}

/// Apply vertical 1D convolution.
fn convolve_vertical(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) {
    let radius = kernel.len() / 2;
    for y in 0..height {
        let row_out_start = y * width;
        for x in 0..width {
            let mut sum = 0.0;
            for k_idx in 0..kernel.len() {
                let offset = k_idx as i32 - radius as i32;
                let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as usize;
                sum += input[sample_y * width + x] * kernel[k_idx];
            }
            output[row_out_start + x] = sum;
        }
    }
}

/// Scalar implementation of a 2D Gaussian blur (approximating XYLinear).
fn gaussian_blur_scalar(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    sigma: f32,
) {
    // Determine radius based on sigma (common heuristic: 3*sigma)
    let radius = (sigma * 3.0).ceil().max(1.0) as usize;
    let kernel = gaussian_kernel(sigma, radius);

    // Temporary buffer for horizontal pass result
    let mut temp = vec![0.0f32; width * height];

    convolve_horizontal(input, &mut temp, width, height, &kernel);
    convolve_vertical(&temp, output, width, height, &kernel);
}

/// Downsamples a pixel-level map to a block-level map by averaging.
fn downsample_to_blocks(
    pixel_map: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    block_map: &mut [f32],
) {
    assert_eq!(block_map.len(), block_w * block_h);
    for by in 0..block_h {
        for bx in 0..block_w {
            let mut sum = 0.0;
            let mut count = 0;
            let start_y = by * 8;
            let start_x = bx * 8;
            for iy in 0..8 {
                let y = start_y + iy;
                if y >= height { continue; }
                for ix in 0..8 {
                    let x = start_x + ix;
                    if x >= width { continue; }
                    sum += pixel_map[y * width + x];
                    count += 1;
                }
            }
            block_map[by * block_w + bx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }
}

/// Scalar implementation approximating jpegli's Masking function.
fn masking_scalar(
    y_channel_f32: &[f32],
    width: usize,
    height: usize,
    distance: f32,
    block_w: usize,
    block_h: usize,
    masking_map: &mut [f32], // Output block-level map
) {
    let num_pixels = width * height;
    let mut blurred1 = vec![0.0f32; num_pixels];
    let mut blurred2 = vec![0.0f32; num_pixels];
    let mut diff_map_pixels = vec![0.0f32; num_pixels];

    // Sigmas used in jpegli Masking (via XYLinear)
    // Note: Jpegli might use slightly different sigma values or filter types.
    let sigma1 = 0.4 * distance + 0.4;
    let sigma2 = 0.8 * distance + 0.8;

    gaussian_blur_scalar(y_channel_f32, &mut blurred1, width, height, sigma1);
    gaussian_blur_scalar(&blurred1, &mut blurred2, width, height, sigma2);

    // Calculate difference and apply non-linearity (approximating jpegli logic)
    for i in 0..num_pixels {
        let diff = blurred1[i] - blurred2[i];
        // Non-linear mapping based on jpegli's approach (using difference)
        // diff_map_pixels[i] = Scale(diff * diff, 40.0 / distance) * 0.8 + 0.2;
        // Simplified approximation for scalar:
        let diff_scaled = diff.abs() / distance.max(0.1); // Normalize by distance
        diff_map_pixels[i] = (1.0 + diff_scaled).ln(); // Log-like response
    }

    // Downsample the pixel-level difference map to block level
    downsample_to_blocks(&diff_map_pixels, width, height, block_w, block_h, masking_map);

    // Apply multiplier (done later in compute_adaptive_quant_field)
    // for val in masking_map.iter_mut() {
    //     *val *= K_MASK_MULTIPLIER;
    // }
}

/// Scalar implementation approximating jpegli's EdgeDetector function.
fn edge_detector_scalar(
    y_channel_f32: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    edge_map: &mut [f32], // Output block-level map
) {
    let num_pixels = width * height;
    let mut grad_x = vec![0.0f32; num_pixels];
    let mut grad_y = vec![0.0f32; num_pixels];
    let mut edge_map_pixels = vec![0.0f32; num_pixels];

    // Simple Sobel-like gradient calculation
    for y in 1..(height - 1) {
        let row_prev = (y - 1) * width;
        let row_curr = y * width;
        let row_next = (y + 1) * width;
        for x in 1..(width - 1) {
            // Gx
            let gx = (y_channel_f32[row_prev + x + 1] + 2.0 * y_channel_f32[row_curr + x + 1] + y_channel_f32[row_next + x + 1])
                   - (y_channel_f32[row_prev + x - 1] + 2.0 * y_channel_f32[row_curr + x - 1] + y_channel_f32[row_next + x - 1]);
            // Gy
            let gy = (y_channel_f32[row_next + x - 1] + 2.0 * y_channel_f32[row_next + x] + y_channel_f32[row_next + x + 1])
                   - (y_channel_f32[row_prev + x - 1] + 2.0 * y_channel_f32[row_prev + x] + y_channel_f32[row_prev + x + 1]);

            grad_x[row_curr + x] = gx;
            grad_y[row_curr + x] = gy;
        }
    }

    // Calculate gradient magnitude (squared) and potentially smooth it
    for i in 0..num_pixels {
        edge_map_pixels[i] = grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i];
    }

    // Note: Jpegli's EdgeDetector might involve more sophisticated filtering or scaling.
    // This is a basic approximation.

    // Downsample to block level
    downsample_to_blocks(&edge_map_pixels, width, height, block_w, block_h, edge_map);

    // Apply multiplier (done later in compute_adaptive_quant_field)
    // for val in edge_map.iter_mut() {
    //     *val *= K_EDGE_MULTIPLIER;
    // }
}

/// Approximates RatioOfDerivativesOfCubicRootToSimpleGamma from Jpegli
/// Computes ratio of derivatives: (d/dx x^(1/3)) / (d/dx x) = 1/3 * x^(-2/3)
/// Jpegli uses a slightly different gamma (~2.6) and input scaling.
/// `invert = true` corresponds to the version used in GammaModulation.
#[inline]
fn ratio_of_derivatives(val: f32, invert: bool) -> f32 {
    // Simple approximation, ignoring the exact gamma/scaling differences for now.
    // Jpegli: ratio = pow(iny * (1.0 / 0.118), -0.73) * 0.15;
    // Here we use the 1/3 * x^(-2/3) formula directly for simplicity.
    let term = (1.0/3.0) * val.powf(-2.0/3.0);
    if invert {
        1.0 / term.max(1e-6) // Avoid division by zero
    } else {
        term
    }
}

/// Approximates MaskingSqrt from Jpegli.
/// Jpegli: return 0.25f * Sqrt(MulAdd(v, Sqrt(mul_v), offset_v))
#[inline]
fn masking_sqrt(v: f32) -> f32 {
    0.25 * (v * K_MUL_SQRT.sqrt() + K_LOG_OFFSET_SQRT).sqrt()
}

/// Simple scalar pad function (copies border pixels).
fn pad_buffer(buffer: &mut [f32], width: usize, height: usize, border: usize) {
    // Pad top/bottom
    for y_off in 0..border {
        let src_y = y_off;
        let dst_y = height + y_off;
        let src_idx = src_y * width;
        let dst_idx = dst_y * width;
        buffer[dst_idx..(dst_idx+width)].copy_from_slice(&buffer[src_idx..(src_idx+width)]);

        let src_y_b = height - 1 - y_off;
        let dst_y_b = height + border + y_off;
        let src_idx_b = src_y_b * width;
        let dst_idx_b = dst_y_b * width;
        buffer[dst_idx_b..(dst_idx_b+width)].copy_from_slice(&buffer[src_idx_b..(src_idx_b+width)]);
    }
    // Pad left/right (row by row)
    let total_h = height + 2 * border;
    for y in 0..total_h {
        let row_start = y * width;
        for x_off in 0..border {
            buffer[row_start + width + x_off] = buffer[row_start + width - 1 - x_off]; // right
            buffer[row_start - border + x_off] = buffer[row_start + x_off]; // left (assuming buffer starts at -border offset)
        }
    }
     // TODO: Need to handle buffer layout properly if it doesn't naturally include negative indices.
     // This padding assumes the buffer has space allocated *before* index 0 for the left border.
}

/// Scalar implementation of ComputePreErosion.
fn compute_pre_erosion_scalar(
    input_scaled: &mut [f32], // Input scaled to [0, 1], mutable for potential padding
    width: usize,
    height: usize,
    border: usize,
    pre_erosion: &mut Vec<f32>, // Output, downsampled by 4x
) {
    let xsize_out = width / 4;
    let ysize_out = height / 4;
    pre_erosion.resize(xsize_out * ysize_out, 0.0);

    // TODO: Handle padding properly. Jpegli pads the input buffer.
    // For simplicity here, we might skip exact padding and rely on clamping inside loops,
    // or assume the caller handles padding.
    // pad_buffer(input_scaled, width, height, border); // Requires adjusting buffer access

    let mut diff_buffer = vec![0.0f32; width]; // Temporary buffer for one row of accumulated diffs

    for y_out in 0..ysize_out {
        // Process 4 input rows at a time
        for iy4 in 0..4 {
            let y = y_out * 4 + iy4;
            if y >= height { continue; } // Skip if out of bounds

            // Zero out diff buffer for the first row of the 4x4 block
            if iy4 == 0 {
                diff_buffer.fill(0.0);
            }

            let y_prev = (y as i32 - 1).clamp(0, height as i32 - 1) as usize;
            let y_next = (y + 1).clamp(0, height - 1);

            let row_in = y * width;
            let row_in_prev = y_prev * width;
            let row_in_next = y_next * width;

            for x in 0..width {
                let x_prev = (x as i32 - 1).clamp(0, width as i32 - 1) as usize;
                let x_next = (x + 1).clamp(0, width - 1);

                let in_val = input_scaled[row_in + x];
                let in_r = input_scaled[row_in + x_next];
                let in_l = input_scaled[row_in + x_prev];
                let in_t = input_scaled[row_in_prev + x];
                let in_b = input_scaled[row_in_next + x];

                let base = 0.25 * (in_r + in_l + in_t + in_b);
                // Apply input scaling division here for MATCH_GAMMA_OFFSET
                let gamma_in = in_val + MATCH_GAMMA_OFFSET / 255.0;
                let gammacv = ratio_of_derivatives(gamma_in, false);

                let mut diff = gammacv * (in_val - base);
                diff = diff * diff;
                diff = diff.min(LIMIT);
                diff = masking_sqrt(diff);

                // Accumulate diffs vertically into diff_buffer
                diff_buffer[x] += diff;
            }

            // After processing the 4th row, average horizontally and store in pre_erosion
            if iy4 == 3 {
                let pre_erosion_row_start = y_out * xsize_out;
                for x_out in 0..xsize_out {
                    let x_start = x_out * 4;
                    let sum = diff_buffer[x_start] + diff_buffer[x_start + 1] +
                              diff_buffer[x_start + 2] + diff_buffer[x_start + 3];
                    pre_erosion[pre_erosion_row_start + x_out] = sum * 0.25;
                }
            }
        }
    }
     // TODO: Jpegli pads the output pre_erosion buffer as well.
     // pre_erosion->PadRow(y_out, xsize_out, border);
}

/// Scalar implementation of Sort4.
#[inline]
fn sort4(v: &mut [f32; 4]) {
    // Simple bubble sort for 4 elements
    if v[0] > v[1] { v.swap(0, 1); }
    if v[2] > v[3] { v.swap(2, 3); }
    if v[0] > v[2] { v.swap(0, 2); }
    if v[1] > v[3] { v.swap(1, 3); }
    if v[1] > v[2] { v.swap(1, 2); }
}

/// Scalar implementation of UpdateMin4.
#[inline]
fn update_min4(val: f32, mins: &mut [f32; 4]) {
    if val < mins[3] {
        if val < mins[2] {
            mins[3] = mins[2];
            if val < mins[1] {
                mins[2] = mins[1];
                if val < mins[0] {
                    mins[1] = mins[0];
                    mins[0] = val;
                } else {
                    mins[1] = val;
                }
            } else {
                mins[2] = val;
            }
        } else {
            mins[3] = val;
        }
    }
}

/// Scalar implementation of FuzzyErosion.
fn fuzzy_erosion_scalar(
    pre_erosion: &[f32],
    pre_erosion_w: usize,
    pre_erosion_h: usize,
    block_w: usize,
    block_h: usize,
    tmp: &mut [f32], // Temporary buffer, size (pre_erosion_w * pre_erosion_h)
    aq_map: &mut [f32], // Output, size (block_w * block_h)
) {
    // Weights for the 4 minimum neighbours
    const MUL0: f32 = 0.125;
    const MUL1: f32 = 0.075;
    const MUL2: f32 = 0.06;
    const MUL3: f32 = 0.05;

    // First pass: compute weighted minimums into tmp buffer (size of pre_erosion)
    for y in 1..(pre_erosion_h - 1) { // Iterate excluding borders
        let row_t = (y - 1) * pre_erosion_w;
        let row_m = y * pre_erosion_w;
        let row_b = (y + 1) * pre_erosion_w;
        let tmp_row_start = y * pre_erosion_w;

        for x in 1..(pre_erosion_w - 1) {
            // Gather 3x3 neighborhood
            let neighbors = [
                pre_erosion[row_t + x - 1], pre_erosion[row_t + x], pre_erosion[row_t + x + 1],
                pre_erosion[row_m + x - 1], pre_erosion[row_m + x], pre_erosion[row_m + x + 1],
                pre_erosion[row_b + x - 1], pre_erosion[row_b + x], pre_erosion[row_b + x + 1],
            ];

            // Find 4 minimums (inefficient scalar sort)
            let mut mins = [neighbors[0], neighbors[1], neighbors[2], neighbors[3]];
            sort4(&mut mins);
            update_min4(neighbors[4], &mut mins);
            update_min4(neighbors[5], &mut mins);
            update_min4(neighbors[6], &mut mins);
            update_min4(neighbors[7], &mut mins);
            update_min4(neighbors[8], &mut mins);

            // Calculate weighted sum
            tmp[tmp_row_start + x] = MUL0 * mins[0] + MUL1 * mins[1] + MUL2 * mins[2] + MUL3 * mins[3];
        }
        // TODO: Handle borders of tmp buffer if necessary (jpegli might pad or handle differently)
    }

    // Second pass: downsample tmp by 2x into aq_map (block level)
    for by in 0..block_h {
        let y_tmp = by * 2;
        if y_tmp + 1 >= pre_erosion_h { continue; } // Ensure we have two rows in tmp
        let tmp_row0 = y_tmp * pre_erosion_w;
        let tmp_row1 = (y_tmp + 1) * pre_erosion_w;
        let aq_row_start = by * block_w;

        for bx in 0..block_w {
            let x_tmp = bx * 2;
            if x_tmp + 1 >= pre_erosion_w { continue; } // Ensure we have two columns in tmp
            aq_map[aq_row_start + bx] = tmp[tmp_row0 + x_tmp] + tmp[tmp_row0 + x_tmp + 1] +
                                      tmp[tmp_row1 + x_tmp] + tmp[tmp_row1 + x_tmp + 1];
        }
    }
}

/// Scalar implementation of ComputeMask.
#[inline]
fn compute_mask_scalar(out_val: f32) -> f32 {
    let v1 = (out_val * K_MASK_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + K_MASK_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + K_MASK_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + K_MASK_OFFSET4);
    K_MASK_BASE + K_MASK_MUL4 * v4 + K_MASK_MUL2 * v2 + K_MASK_MUL3 * v3
}

/// Scalar implementation of HfModulation.
#[inline]
fn hf_modulation_scalar(
    x: usize, y: usize, 
    input_scaled: &[f32], width: usize, height: usize,
    current_val: f32
) -> f32 {
    let mut sum_abs_diff = 0.0;
    let start_idx = y * width + x;

    for dy in 0..8 {
        let current_row_idx = (y + dy).clamp(0, height - 1) * width;
        let next_row_idx = (y + dy + 1).clamp(0, height - 1) * width;
        for dx in 0..8 {
            let current_idx = current_row_idx + (x + dx).clamp(0, width - 1);
            let right_idx = current_row_idx + (x + dx + 1).clamp(0, width - 1);
            let down_idx = next_row_idx + (x + dx).clamp(0, width - 1);
            
            let p = input_scaled[current_idx];
            // Don't compute difference for rightmost pixel in 8x8 block
            if dx < 7 {
                 let pr = input_scaled[right_idx];
                 sum_abs_diff += (p - pr).abs();
            }
            // Don't compute difference for bottommost pixel in 8x8 block
            if dy < 7 {
                let pd = input_scaled[down_idx];
                sum_abs_diff += (p - pd).abs();
            }
        }
    }
    current_val + sum_abs_diff * K_HF_MOD_COEFF
}

/// Scalar implementation of GammaModulation.
#[inline]
fn gamma_modulation_scalar(
    x: usize, y: usize,
    input_scaled: &[f32], width: usize, height: usize,
    current_val: f32
) -> f32 {
    let mut overall_ratio = 0.0;
    let bias = K_GAMMA_MOD_BIAS / 255.0; // Apply scaling here

    for dy in 0..8 {
         let row_idx_base = (y + dy).clamp(0, height - 1) * width;
         for dx in 0..8 {
            let idx = row_idx_base + (x + dx).clamp(0, width - 1);
            let iny = input_scaled[idx] + bias;
            overall_ratio += ratio_of_derivatives(iny, true);
         }
    }
    overall_ratio *= K_GAMMA_MOD_SCALE;
    current_val + K_GAMMA_MOD_GAMMA * overall_ratio.max(1e-6).log2()
}

/// Scalar implementation of PerBlockModulations.
fn per_block_modulations_scalar(
    y_quant_01: i32, // Quant value of first AC coeff
    input_scaled: &[f32],
    width: usize, height: usize,
    block_w: usize, block_h: usize,
    aq_map: &mut [f32], // Input is fuzzy erosion, output is modulated map
) {
    let base_level = 0.48 * K_AC_QUANT;
    let dampen_ramp_start = 9.0;
    let dampen_ramp_end = 65.0;
    let mut dampen = 1.0;

    if y_quant_01 as f32 >= dampen_ramp_start {
        dampen = 1.0 - ((y_quant_01 as f32 - dampen_ramp_start) / (dampen_ramp_end - dampen_ramp_start));
        dampen = dampen.max(0.0);
    }

    let mul = K_AC_QUANT * dampen;
    let add = (1.0 - dampen) * base_level;

    for by in 0..block_h {
        let aq_row_start = by * block_w;
        for bx in 0..block_w {
            let aq_idx = aq_row_start + bx;
            let current_val = aq_map[aq_idx];
            let x_pix = bx * 8;
            let y_pix = by * 8;

            let mask_val = compute_mask_scalar(current_val);
            let hf_val = hf_modulation_scalar(x_pix, y_pix, input_scaled, width, height, mask_val);
            let gamma_val = gamma_modulation_scalar(x_pix, y_pix, input_scaled, width, height, hf_val);

            // We want multiplicative quantization field, so everything
            // until this point has been modulating the exponent.
            // Jpegli: row_out[ix] = FastPow2f(GetLane(out_val) * 1.442695041f) * mul + add;
            let final_modulated_val = (gamma_val * 1.442695041).exp2() * mul + add;
            aq_map[aq_idx] = final_modulated_val;
        }
    }
}

// --- Main Function --- 

pub(crate) fn compute_adaptive_quant_field(
    width: u16,
    height: u16,
    y_channel_scaled: &mut [f32], // Input scaled to [0, 1], mutable for padding
    distance: f32,
    y_quant_01: i32, // <-- Added parameter
) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    let num_pixels = width * height;

    if y_channel_scaled.len() != num_pixels {
        // Basic validation
        eprintln!("Error: y_channel_scaled size mismatch in adaptive quant");
        // Return default non-adaptive field on error
        let block_w = (width + 7) / 8;
        let block_h = (height + 7) / 8;
        return vec![1.0f32; block_w * block_h];
    }

    let block_w = (width + 7) / 8;
    let block_h = (height + 7) / 8;
    let num_blocks = block_w * block_h;

    // Allocate buffers
    let pre_erosion_width = width / 4;
    let pre_erosion_height = height / 4;
    let mut pre_erosion_map = vec![0.0f32; pre_erosion_width * pre_erosion_height];
    let mut fuzzy_erosion_map = vec![0.0f32; num_blocks]; // Output of FuzzyErosion
    // Buffers needed by FuzzyErosion (size based on C++ impl)
    let fuzzy_tmp_width = width / 2;
    let fuzzy_tmp_height = height / 2;
    let mut fuzzy_erosion_tmp = vec![0.0f32; fuzzy_tmp_width * fuzzy_tmp_height];

    let mut final_field = vec![1.0f32; num_blocks];   // The output field

    // 1. Compute Pre-Erosion Map
    compute_pre_erosion_scalar(
        y_channel_scaled,
        width,
        height,
        1, // kPreErosionBorder = 1
        &mut pre_erosion_map
    );

    // 2. Fuzzy Erosion
    fuzzy_erosion_scalar(
        &pre_erosion_map,
        pre_erosion_width, pre_erosion_height,
        block_w, block_h,
        &mut fuzzy_erosion_tmp,
        &mut fuzzy_erosion_map // Output goes here
    );

    // 3. Per-Block Modulations (input is fuzzy_erosion_map)
    per_block_modulations_scalar(
        y_quant_01,
        y_channel_scaled, // Original scaled pixel data needed here
        width, height,
        block_w, block_h,
        &mut fuzzy_erosion_map // Modifies this map in-place
    );

    // 4. Final Adjustment (input is fuzzy_erosion_map after modulations)
    for i in 0..num_blocks {
        // Placeholder: use fuzzy_erosion_map[i] eventually
        let modulated_val = fuzzy_erosion_map[i]; // Use output of FuzzyErosion now
        final_field[i] = (0.6 / (modulated_val + 1e-6) - 1.0).max(0.0);
        // Note: Jpegli's quantizer seems to expect this offset format.
        // The actual quantization logic needs to be updated to use this.
    }

    // --- Placeholder --- 
    // Returning default field until FuzzyErosion and PerBlockModulations are done.
    // Return the final computed field now.
    // vec![1.0f32; num_blocks]
    // --- End Placeholder ---

    final_field // Return this eventually
}

/// Helper for Gaussian blur on block-level maps (not pixel level).
fn gaussian_blur_scalar_on_blocks(
    input: &[f32],
    output: &mut [f32],
    block_w: usize,
    block_h: usize,
    sigma: f32,
) {
    // This reuses the pixel-level convolution functions, just treating blocks as pixels.
    let radius = (sigma * 3.0).ceil().max(1.0) as usize;
    let kernel = gaussian_kernel(sigma, radius);
    let mut temp = vec![0.0f32; block_w * block_h];

    convolve_horizontal(input, &mut temp, block_w, block_h, &kernel);
    convolve_vertical(&temp, output, block_w, block_h, &kernel);
}

#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module

    // Helper to create a simple gradient image for testing
    fn create_test_image(width: usize, height: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                img[y * width + x] = (x + y) as f32 * (255.0 / (width + height) as f32);
            }
        }
        img
    }

    #[test]
    fn test_downsample_to_blocks_simple() {
        let width = 16;
        let height = 8;
        let block_w = 2;
        let block_h = 1;
        let pixel_map: Vec<f32> = (0..(width * height)).map(|i| i as f32).collect();
        let mut block_map = vec![0.0f32; block_w * block_h];

        downsample_to_blocks(&pixel_map, width, height, block_w, block_h, &mut block_map);

        // Block 0 (0,0) average of pixels 0..7 (x) and 0..7 (y)
        let avg0 = (0..(8*8)).map(|i| i as f32).sum::<f32>() / 64.0;
        // Block 1 (1,0) average of pixels 8..15 (x) and 0..7 (y)
        let avg1 = ((8*8)..(16*8)).map(|i| i as f32).sum::<f32>() / 64.0;

        assert!((block_map[0] - avg0).abs() < 1e-5);
        assert!((block_map[1] - avg1).abs() < 1e-5);
    }

    #[test]
    fn test_compute_adaptive_quant_field_runs() {
        // Basic test to ensure the function runs without panicking and returns a correctly sized vector.
        let width: u16 = 32;
        let height: u16 = 24;
        let distance = 1.0;
        let test_image = create_test_image(width as usize, height as usize);

        let field = compute_adaptive_quant_field(width, height, &mut test_image.clone(), distance, 0);

        let block_w = (width + 7) / 8;
        let block_h = (height + 7) / 8;
        assert_eq!(field.len(), (block_w * block_h) as usize);
        // Check if values are somewhat reasonable (e.g., not NaN or infinite)
        for &val in &field {
            assert!(val.is_finite());
            assert!(val > 0.0);
        }
    }

    // TODO: Add more detailed tests comparing scalar output to known values or properties.
    // This would likely require generating reference data from jpegli itself.
} 