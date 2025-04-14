//! Adaptive quantization logic ported from Jpegli.

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

// Constants ported from adaptive_quantization.cc
const K_MASK_MULTIPLIER: f32 = 0.855;
const K_EDGE_MULTIPLIER: f32 = 0.6;
const K_BORDER_MULTIPLIER: f32 = 0.125;

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

// --- Main Function --- 

pub(crate) fn compute_adaptive_quant_field(
    width: u16,
    height: u16,
    y_channel_f32: &[f32], // Expect f32 input now
    distance: f32,
) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    let num_pixels = width * height;

    if y_channel_f32.len() != num_pixels {
        // Basic validation
        eprintln!("Error: y_channel_f32 size mismatch in adaptive quant");
        // Return default non-adaptive field on error
        let block_w = (width + 7) / 8;
        let block_h = (height + 7) / 8;
        return vec![1.0f32; block_w * block_h];
    }

    let block_w = (width + 7) / 8;
    let block_h = (height + 7) / 8;
    let num_blocks = block_w * block_h;

    // Allocate buffers for block-level maps
    let mut masking_map_blocks = vec![0.0f32; num_blocks];
    let mut edge_map_blocks = vec![0.0f32; num_blocks];
    let mut smoothed_aq_map = vec![0.0f32; num_blocks]; // Buffer for final smoothing
    let mut final_field = vec![1.0f32; num_blocks];   // The output field

    // 1. Compute Masking (scalar version)
    masking_scalar(
        y_channel_f32,
        width,
        height,
        distance,
        block_w,
        block_h,
        &mut masking_map_blocks
    );

    // 2. Detect Edges (scalar version)
    edge_detector_scalar(y_channel_f32, width, height, block_w, block_h, &mut edge_map_blocks);

    // 3. Combine masking and edges into `smoothed_aq_map` buffer (before smoothing)
    for i in 0..num_blocks {
        // Note: Jpegli likely combines these differently. This is a simple weighted sum approximation.
        // It might use min, max, or a more complex formula based on Butteraugli masking.
        smoothed_aq_map[i] = masking_map_blocks[i] * K_MASK_MULTIPLIER 
                           + edge_map_blocks[i] * K_EDGE_MULTIPLIER;
    }

    // 4. Smooth the combined aq_map (in-place into smoothed_aq_map)
    // Jpegli uses sigma = 0.8 * distance + 0.5 for the final smoothing
    // Note: This Gaussian blur operates on the block-level map, not pixels.
    let final_sigma = 0.8 * distance + 0.5;
    // Need a temporary buffer for the block-level blur
    let mut temp_block_map = smoothed_aq_map.clone();
    gaussian_blur_scalar_on_blocks(
        &temp_block_map, // Input is the combined map
        &mut smoothed_aq_map, // Output is the same buffer
        block_w,
        block_h,
        final_sigma
    );

    // 5. Final adjustment and conversion to multiplier
    for i in 0..num_blocks {
        // Invert the smoothed value to get the quantization multiplier.
        // Add a small epsilon to prevent division by zero.
        let inverse_val = 1.0 / (smoothed_aq_map[i] + 1e-6);

        // Apply clamping similar to Jpegli's QuantMultiplier
        // Lower bound prevents excessive quantization (value from jpegli).
        // Upper bound might also be needed depending on exact formula.
        final_field[i] = inverse_val.clamp(0.1, 10.0); // Clamp between 0.1x and 10x original Q
    }

    final_field
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

        let field = compute_adaptive_quant_field(width, height, &test_image, distance);

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