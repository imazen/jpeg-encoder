use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU16;
use std::f32;
use std::fmt::Write;
use lazy_static::lazy_static;

// --- Start Jpegli Constants ---

// Base quantization tables ported from jpegli quant.cc (kBaseQuantMatrixYCbCr)
// Luma table (first 64 values)
pub(crate) const JPEGLI_DEFAULT_LUMA_QTABLE_F32: [f32; 64] = [
    1.2397409, 1.7227115, 2.9212167, 2.8127374, 3.3398197, 3.4636038, 3.8409152, 3.8695600,
    1.7227115, 2.0928894, 2.8456761, 2.7045068, 3.4407674, 3.1662324, 4.0252087, 4.0353245,
    2.9212167, 2.8456761, 2.9587404, 3.3862949, 3.6195238, 3.9046280, 3.7578358, 4.0496073,
    2.8127374, 2.7045068, 3.3862949, 3.1295824, 3.7035120, 4.3547106, 4.2037473, 3.9457080,
    3.3398197, 3.4407674, 3.6195238, 3.7035120, 4.0587358, 4.8218517, 4.8176765, 4.1348114,
    3.4636038, 3.1662324, 3.9046280, 4.3547106, 4.8218517, 5.3049545, 5.0859237, 4.6540699,
    3.8409152, 4.0252087, 3.7578358, 4.2037473, 4.8176765, 5.0859237, 5.2007284, 5.1318064,
    3.8695600, 4.0353245, 4.0496073, 3.9457080, 4.1348114, 4.6540699, 5.1318064, 5.3104744,
];
// Chroma table (second 64 values from kBaseQuantMatrixYCbCr - assuming Cb=Cr)
pub(crate) const JPEGLI_DEFAULT_CHROMA_QTABLE_F32: [f32; 64] = [
    1.4173750, 3.4363859, 3.7492752, 4.2684789, 4.8839750, 5.1342621, 5.3053384, 5.2941780,
    3.4363859, 3.3934350, 3.7151461, 4.4069610, 5.0667987, 5.0575762, 5.3007593, 5.2948112,
    3.7492752, 3.7151461, 4.0639019, 4.7990928, 5.0091391, 5.1409049, 5.2947245, 5.2915106,
    4.2684789, 4.4069610, 4.7990928, 4.7969780, 5.1343479, 5.1429081, 5.3214135, 5.4269948,
    4.8839750, 5.0667987, 5.0091391, 5.1343479, 5.2924175, 5.2911520, 5.4630551, 5.5700078,
    5.1342621, 5.0575762, 5.1409049, 5.1429081, 5.2911520, 5.3632350, 5.5484371, 5.5723948,
    5.3053384, 5.3007593, 5.2947245, 5.3214135, 5.4630551, 5.5484371, 5.5720239, 5.5726733,
    5.2941780, 5.2948112, 5.2915106, 5.4269948, 5.5700078, 5.5723948, 5.5726733, 5.5728049,
];

// Other constants ported from jpegli quant.cc
pub(crate) const K_GLOBAL_SCALE_YCBCR: f32 = 1.73966010;
pub(crate) const K_420_GLOBAL_SCALE: f32 = 1.22; // Applied when YUV420
pub(crate) const K_420_RESCALE: [f32; 64] = [
    0.4093, 0.3209, 0.3477, 0.3333, 0.3144, 0.2823, 0.3214, 0.3354,
    0.3209, 0.3111, 0.3489, 0.2801, 0.3059, 0.3119, 0.4135, 0.3445,
    0.3477, 0.3489, 0.3586, 0.3257, 0.2727, 0.3754, 0.3369, 0.3484,
    0.3333, 0.2801, 0.3257, 0.3020, 0.3515, 0.3410, 0.3971, 0.3839,
    0.3144, 0.3059, 0.2727, 0.3515, 0.3105, 0.3397, 0.2716, 0.3836,
    0.2823, 0.3119, 0.3754, 0.3410, 0.3397, 0.3212, 0.3203, 0.0726,
    0.3214, 0.4135, 0.3369, 0.3971, 0.2716, 0.3203, 0.0798, 0.0553,
    0.3354, 0.3445, 0.3484, 0.3839, 0.3836, 0.0726, 0.0553, 0.3368,
];
pub(crate) const K_EXPONENT: [f32; 64] = [
    1.00, 0.51, 0.67, 0.74, 1.00, 1.00, 1.00, 1.00,
    0.51, 0.66, 0.69, 0.87, 1.00, 1.00, 1.00, 1.00,
    0.67, 0.69, 0.84, 0.83, 0.96, 1.00, 1.00, 1.00,
    0.74, 0.87, 0.83, 1.00, 1.00, 0.91, 0.91, 1.00,
    1.00, 1.00, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
];
pub(crate) const K_DIST0: f32 = 1.5; // Distance where non-linearity kicks in.

// Constants for transfer functions (from C++ quant.cc)
pub(crate) const K_TRANSFER_FUNCTION_PQ: u8 = 16;
pub(crate) const K_TRANSFER_FUNCTION_HLG: u8 = 18;

// --- End Jpegli Constants ---

// --- Start Standard Annex K Constants ---
// From libjpeg-turbo's jquant.c (scaled by default quality 50 factor)
// Or directly from Annex K.1
const STANDARD_LUMA_QTABLE_U8: [u8; 64] = [
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
];

const STANDARD_CHROMA_QTABLE_U8: [u8; 64] = [
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
];

// Convert standard tables to f32 for consistency in calculation
fn standard_table_to_f32(table_u8: &[u8; 64]) -> [f32; 64] {
    let mut table_f32 = [0.0f32; 64];
    for i in 0..64 {
        table_f32[i] = table_u8[i] as f32;
    }
    table_f32
}

lazy_static! {
    static ref STANDARD_LUMA_QTABLE_F32: [f32; 64] = standard_table_to_f32(&STANDARD_LUMA_QTABLE_U8);
    static ref STANDARD_CHROMA_QTABLE_F32: [f32; 64] = standard_table_to_f32(&STANDARD_CHROMA_QTABLE_U8);
}

// --- End Standard Annex K Constants ---

// Helper function ported from jpegli DistanceToScale
pub(crate) fn distance_to_scale(distance: f32, k: usize) -> f32 {
    if distance < K_DIST0 {
        distance
    } else {
        let exp = K_EXPONENT[k];
        let mul = K_DIST0.powf(1.0 - exp);
        (mul * distance.powf(exp)).max(0.5 * distance) // Max ensures scale doesn't decrease too much
    }
}

/// Maps a libjpeg quality factor (1..100) to a jpegli Butteraugli distance.
/// Ported from jpegli C++ implementation.
pub fn quality_to_distance(quality: u8) -> f32 {
    let quality = quality as f32;
    if quality >= 100.0 {
        0.01
    } else if quality >= 30.0 {
        0.1 + (100.0 - quality) * 0.09
    } else {
        // Adjusted quadratic formula from C++:
        // 53.0 / 3000.0 * q*q - 23.0 / 20.0 * q + 25.0
        // approx 0.017666 * q*q - 1.15 * q + 25.0
        (53.0 / 3000.0) * quality.powi(2) - (23.0 / 20.0) * quality + 25.0
    }
}

/// Creates a raw (unshifted) quantization table using Jpegli's distance-based scaling.
pub(crate) fn compute_jpegli_quant_table(
    distance: f32,
    is_luma: bool,
    is_yuv420: bool,
    force_baseline: bool,
    cicp_transfer_function: Option<u8>,
) -> [u16; 64] {
    // Always use Jpegli base tables
    let base_table_f32: &[f32; 64] = if is_luma {
            &JPEGLI_DEFAULT_LUMA_QTABLE_F32
        } else {
            &JPEGLI_DEFAULT_CHROMA_QTABLE_F32
        };

    // Always use Jpegli global scale
    let mut global_scale = K_GLOBAL_SCALE_YCBCR; // 1.73966010

    if is_yuv420 { // Jpegli 420 scaling
        global_scale *= K_420_GLOBAL_SCALE; // 1.22
    }

    // Apply scaling based on transfer function
    if let Some(tf_code) = cicp_transfer_function {
        if tf_code == K_TRANSFER_FUNCTION_PQ {
            global_scale *= 0.4;
        } else if tf_code == K_TRANSFER_FUNCTION_HLG {
            global_scale *= 0.5;
        }
    }

    let quant_max = if force_baseline { 255 } else { 32767 };
    let mut table_data = [1u16; 64];

    for k in 0..64 {
        // Jpegli scaling path
        let mut scale = global_scale; // Start with the Jpegli overall global scale
        scale *= distance_to_scale(distance.max(0.0), k); // Apply distance scaling per coefficient
        if is_yuv420 && !is_luma { // Apply k420Rescale only for chroma in 420 mode
            scale *= K_420_RESCALE[k];
        }

        let qval_f = scale * base_table_f32[k]; // Multiply final scale by Jpegli base table value
        let qval = qval_f.round() as i32; // Jpegli path uses round
        let qval_clamped = qval.clamp(1, quant_max) as u16; // Clamp to [1, max]
        table_data[k] = qval_clamped;
    }
    table_data
}

// --- Zero Bias Constants (ported from jpegli quant.cc) ---
// Note: These are large tables.
const K_ZERO_BIAS_MUL_YCBCR_LQ: [[f32; 64]; 3] = [
    // c = 0 (Y)
    [ 0.0000, 0.0568, 0.3880, 0.6190, 0.6190, 0.4490, 0.4490, 0.6187,
      0.0568, 0.5829, 0.6189, 0.6190, 0.6190, 0.7190, 0.6190, 0.6189,
      0.3880, 0.6189, 0.6190, 0.6190, 0.6190, 0.6190, 0.6187, 0.6100,
      0.6190, 0.6190, 0.6190, 0.6190, 0.5890, 0.3839, 0.7160, 0.6190,
      0.6190, 0.6190, 0.6190, 0.5890, 0.6190, 0.3880, 0.5860, 0.4790,
      0.4490, 0.7190, 0.6190, 0.3839, 0.3880, 0.6190, 0.6190, 0.6190,
      0.4490, 0.6190, 0.6187, 0.7160, 0.5860, 0.6190, 0.6204, 0.6190,
      0.6187, 0.6189, 0.6100, 0.6190, 0.4790, 0.6190, 0.6190, 0.3480 ],
    // c = 1 (Cb) - Explicitly listing all 64 elements
    [ 0.0000, 1.1640, 0.9373, 1.1319, 0.8016, 0.9136, 1.1530, 0.9430, 1.1640, 0.9188, 0.9160, 1.1980, 1.1830, 0.9758, 0.9430, 0.9430, 0.9373, 0.9160, 0.8430, 1.1720, 0.7083, 0.9430, 0.9430, 0.9430, 1.1319, 1.1980, 1.1720, 1.1490, 0.8547, 0.9430, 0.9430, 0.9430, 0.8016, 1.1830, 0.7083, 0.8547, 0.9430, 0.9430, 0.9430, 0.9430, 0.9136, 0.9758, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 1.1530, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9480, 0.9430, 0.9430, 0.9430, 0.9430, 0.9430, 0.9480, 0.9430, 0.9430, 0.9430 ],
    // c = 2 (Cr)
    [ 0.0000, 1.3190, 0.4308, 0.4460, 0.0661, 0.0660, 0.2660, 0.2960,
      1.3190, 0.3280, 0.3093, 0.0750, 0.0505, 0.1594, 0.3060, 0.2113,
      0.4308, 0.3093, 0.3060, 0.1182, 0.0500, 0.3060, 0.3915, 0.2426,
      0.4460, 0.0750, 0.1182, 0.0512, 0.0500, 0.2130, 0.3930, 0.1590,
      0.0661, 0.0505, 0.0500, 0.0500, 0.3055, 0.3360, 0.5148, 0.5403,
      0.0660, 0.1594, 0.3060, 0.2130, 0.3360, 0.5060, 0.5874, 0.3060,
      0.2660, 0.3060, 0.3915, 0.3930, 0.5148, 0.5874, 0.3060, 0.3060,
      0.2960, 0.2113, 0.2426, 0.1590, 0.5403, 0.3060, 0.3060, 0.3060 ],
];

const K_ZERO_BIAS_MUL_YCBCR_HQ: [[f32; 64]; 3] = [
    // c = 0 (Y)
    [ 0.0000, 0.0044, 0.2521, 0.6547, 0.8161, 0.6130, 0.8841, 0.8155,
      0.0044, 0.6831, 0.6553, 0.6295, 0.7848, 0.7843, 0.8474, 0.7836,
      0.2521, 0.6553, 0.7834, 0.7829, 0.8161, 0.8072, 0.7743, 0.9242,
      0.6547, 0.6295, 0.7829, 0.8654, 0.7829, 0.6986, 0.7818, 0.7726,
      0.8161, 0.7848, 0.8161, 0.7829, 0.7471, 0.7827, 0.7843, 0.7653,
      0.6130, 0.7843, 0.8072, 0.6986, 0.7827, 0.7848, 0.9508, 0.7653,
      0.8841, 0.8474, 0.7743, 0.7818, 0.7843, 0.9508, 0.7839, 0.8437,
      0.8155, 0.7836, 0.9242, 0.7726, 0.7653, 0.7653, 0.8437, 0.7819 ],
    // c = 1 (Cb)
    [ 0.0000, 1.0816, 1.0556, 1.2876, 1.1554, 1.1567, 1.8851, 0.5488,
      1.0816, 1.1537, 1.1850, 1.0712, 1.1671, 2.0719, 1.0544, 1.4764,
      1.0556, 1.1850, 1.2870, 1.1981, 1.8181, 1.2618, 1.0564, 1.1191,
      1.2876, 1.0712, 1.1981, 1.4753, 2.0609, 1.0564, 1.2645, 1.0564,
      1.1554, 1.1671, 1.8181, 2.0609, 0.7324, 1.1163, 0.8464, 1.0564,
      1.1567, 2.0719, 1.2618, 1.0564, 1.1163, 1.0040, 1.0564, 1.0564,
      1.8851, 1.0544, 1.0564, 1.2645, 0.8464, 1.0564, 1.0564, 1.0564,
      0.5488, 1.4764, 1.1191, 1.0564, 1.0564, 1.0564, 1.0564, 1.0564 ],
    // c = 2 (Cr)
    [ 0.0000, 0.5392, 0.6659, 0.8968, 0.6829, 0.6328, 0.5802, 0.4836,
      0.5392, 0.6746, 0.6760, 0.6102, 0.6015, 0.6958, 0.7327, 0.4897,
      0.6659, 0.6760, 0.6957, 0.6543, 0.4396, 0.6330, 0.7081, 0.2583,
      0.8968, 0.6102, 0.6543, 0.5913, 0.6457, 0.5828, 0.5139, 0.3565,
      0.6829, 0.6015, 0.4396, 0.6457, 0.5633, 0.4263, 0.6371, 0.5949,
      0.6328, 0.6958, 0.6330, 0.5828, 0.4263, 0.2847, 0.2909, 0.6629,
      0.5802, 0.7327, 0.7081, 0.5139, 0.6371, 0.2909, 0.6644, 0.6644,
      0.4836, 0.4897, 0.2583, 0.3565, 0.5949, 0.6629, 0.6644, 0.6644 ],
];

const K_ZERO_BIAS_OFFSET_YCBCR_DC: [f32; 3] = [0.0, 0.0, 0.0];
const K_ZERO_BIAS_OFFSET_YCBCR_AC: [f32; 3] = [ 0.59082, 0.58146, 0.57988 ];
// --- End Jpegli Constants ---

/// Computes the zero-bias offset and multiplier tables based on distance.
/// Ported from jpegli InitQuantizer logic for YCbCr colorspace.
pub(crate) fn compute_zero_bias_tables(
    distance: f32,
    num_components: usize // Should be 1 for Luma, 3 for YCbCr
) -> (Vec<[f32; 64]>, Vec<[f32; 64]>) { // (offsets, multipliers)
    let mut zero_bias_offsets = vec![[0.0f32; 64]; num_components];
    let mut zero_bias_multipliers = vec![[0.0f32; 64]; num_components];

    // Default initialization (from InitQuantizer)
    for c in 0..num_components {
        for k in 0..64 {
            zero_bias_multipliers[c][k] = if k == 0 { 0.0 } else { 0.5 };
            zero_bias_offsets[c][k] = if k == 0 { 0.0 } else { 0.5 };
        }
    }

    // Apply jpegli zero bias logic if distance is appropriate
    if distance >= 0.1 {
        let log_dist_ac = (distance / 0.3).log2().clamp(-1.0, 1.0);
        let mix0 = ((log_dist_ac + 1.0) * 0.5).max(0.0);
        let mix1 = 1.0 - mix0;

        for c in 0..num_components {
            for k in 0..64 {
                let mul0 = K_ZERO_BIAS_MUL_YCBCR_LQ[c][k];
                let mul1 = K_ZERO_BIAS_MUL_YCBCR_HQ[c][k];
                zero_bias_multipliers[c][k] = mix0 * mul0 + mix1 * mul1;
                zero_bias_offsets[c][k] =
                    if k == 0 {
                         K_ZERO_BIAS_OFFSET_YCBCR_DC[c]
                    } else {
                         K_ZERO_BIAS_OFFSET_YCBCR_AC[c]
                    };
            }
        }
    }
    // Note: Jpegli has further adjustments based on CICP transfer functions etc.,
    // which are not ported here.

    (zero_bias_offsets, zero_bias_multipliers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegli::reference_test_data::REFERENCE_QUANT_TEST_DATA; // Import reference data

    // Helper to recalculate expected table values for comparison
    fn calculate_expected_jpegli_table(
        distance: f32,
        is_luma: bool,
        is_yuv420: bool,
        force_baseline: bool,
        cicp_transfer_function: Option<u8>,
    ) -> [u16; 64] {
        let base_table_f32 = if is_luma {
            &JPEGLI_DEFAULT_LUMA_QTABLE_F32
        } else {
            &JPEGLI_DEFAULT_CHROMA_QTABLE_F32
        };

        let mut global_scale = K_GLOBAL_SCALE_YCBCR;
        if is_yuv420 {
            global_scale *= K_420_GLOBAL_SCALE;
        }

        // Apply scaling based on transfer function in helper
        if let Some(tf_code) = cicp_transfer_function {
            if tf_code == K_TRANSFER_FUNCTION_PQ {
                global_scale *= 0.4;
            } else if tf_code == K_TRANSFER_FUNCTION_HLG {
                global_scale *= 0.5;
            }
        }

        let quant_max = if force_baseline { 255 } else { 32767 };
        let mut expected_table = [0u16; 64];

        for k in 0..64 {
            let mut scale = global_scale;
            scale *= distance_to_scale(distance.max(0.0), k);
            if is_yuv420 && !is_luma {
                scale *= K_420_RESCALE[k];
            }

            let qval_f = scale * base_table_f32[k];
            let qval = qval_f.round() as i32;
            let qval_clamped = qval.clamp(1, quant_max) as u16;
            expected_table[k] = qval_clamped;
        }
        expected_table
    }

    #[test]
    fn test_compute_jpegli_quant_table_distance_1_0() {
        let distance = 1.0;
        let is_yuv420 = false; // Assuming not 4:2:0 for simplicity
        let force_baseline = true;

        // Calculate expected values internally
        let expected_luma_table = calculate_expected_jpegli_table(distance, true, is_yuv420, force_baseline, None);
        let expected_chroma_table = calculate_expected_jpegli_table(distance, false, is_yuv420, force_baseline, None);

        // Call the function under test
        let luma_table_u16 = compute_jpegli_quant_table(
            distance,
            true, // is_luma
            is_yuv420,
            force_baseline,
            None, // No TF specified for this test
        );
        let chroma_table_u16 = compute_jpegli_quant_table(
            distance,
            false, // is_luma
            is_yuv420,
            force_baseline,
            None, // No TF specified for this test
        );

        // Compare!
        assert_eq!(luma_table_u16, expected_luma_table, "Luma table mismatch for distance 1.0");
        assert_eq!(chroma_table_u16, expected_chroma_table, "Chroma table mismatch for distance 1.0");
    }

    #[test]
    fn test_compute_jpegli_quant_table_matches_reference_d1_0() {
        let distance = 1.0;
        let is_yuv420 = false;
        let force_baseline = true;

        // Find the reference data for a distance 1.0 test case
        let ref_data = REFERENCE_QUANT_TEST_DATA
            .iter()
            .find(|d| d.input_filename == "colorful_chessboards.png" && (d.cjpegli_distance - distance).abs() < 1e-6)
            .expect("Reference data for colorful_chessboards.png at distance 1.0 not found");

        let expected_luma_table = ref_data.expected_luma_dqt;
        let expected_chroma_table = ref_data.expected_chroma_dqt;

        // Call the function under test
        let computed_luma_table = compute_jpegli_quant_table(
            distance,
            true, // is_luma
            is_yuv420,
            force_baseline,
            None, // No TF specified for this test
        );
        let computed_chroma_table = compute_jpegli_quant_table(
            distance,
            false, // is_luma
            is_yuv420,
            force_baseline,
            None, // No TF specified for this test
        );

        // Compare with a helper that prints diffs
        compare_quant_tables_quant_test("Luma (d=1.0)", &computed_luma_table, &expected_luma_table, 0);
        compare_quant_tables_quant_test("Chroma (d=1.0)", &computed_chroma_table, &expected_chroma_table, 0);
    }

    #[test]
    fn test_quality_to_distance() {
        // Test values derived from running the C++ code or known reference points.
        assert!((quality_to_distance(100) - 0.01).abs() < 1e-6);
        assert!((quality_to_distance(90) - (0.1 + 10.0 * 0.09)).abs() < 1e-6); // 1.0
        assert!((quality_to_distance(75) - (0.1 + 25.0 * 0.09)).abs() < 1e-6); // 2.35
        assert!((quality_to_distance(50) - (0.1 + 50.0 * 0.09)).abs() < 1e-6); // 4.6
        assert!((quality_to_distance(30) - (0.1 + 70.0 * 0.09)).abs() < 1e-6); // 6.4

        // Lower range - using the quadratic formula part
        let q20 = (53.0 / 3000.0) * 20.0f32.powi(2) - (23.0 / 20.0) * 20.0 + 25.0;
        assert!((quality_to_distance(20) - q20).abs() < 1e-6); // approx 9.0666
        let q10 = (53.0 / 3000.0) * 10.0f32.powi(2) - (23.0 / 20.0) * 10.0 + 25.0;
        assert!((quality_to_distance(10) - q10).abs() < 1e-6); // approx 15.2666
        let q1 = (53.0 / 3000.0) * 1.0f32.powi(2) - (23.0 / 20.0) * 1.0 + 25.0;
        assert!((quality_to_distance(1) - q1).abs() < 1e-6); // approx 23.8676
    }
    // TODO: Add similar tests for other distances (e.g., 0.5, 10.0) and maybe is_yuv420=true

    // Helper function local to quant tests for detailed comparison
    fn compare_quant_tables_quant_test(
        label: &str,
        generated: &[u16; 64],
        expected: &[u16; 64],
        tolerance: u16,
    ) {
        let mut diff_count = 0;
        let mut diff_output = String::new();
        writeln!(
            diff_output,
            "Comparing {} Quantization Table:",
            label
        ).unwrap();
        writeln!(diff_output, "------------------------------------------------------------------------").unwrap();

        for y in 0..8 {
            write!(diff_output, "Row {}: ", y).unwrap();
            for x in 0..8 {
                let index = y * 8 + x;
                let gen_val = generated[index];
                let exp_val = expected[index];
                let diff = gen_val.abs_diff(exp_val);

                if diff > tolerance {
                    diff_count += 1;
                    // Show generated(expected)
                    write!(diff_output, "{:>4}({:>4}) ", gen_val, exp_val).unwrap();
                } else {
                    // Show just the value if within tolerance
                    write!(diff_output, "{:>4}       ", gen_val).unwrap();
                }
            }
            writeln!(diff_output).unwrap(); // Newline after each row
        }
        writeln!(diff_output, "------------------------------------------------------------------------").unwrap();

        if diff_count > 0 {
            // Use std::println! for test output
            std::println!("{}", diff_output);
            panic!("{} quantization table mismatch. Found {} differences with tolerance {}.", label, diff_count, tolerance);
        }
    }
} 