//! Entropy coding module: histogram-based Huffman optimization and bit-level encoding.
//!
//! This module provides functions to: build symbol histograms from quantized
//! coefficients, optimize Huffman tables using those histograms, and apply
//! the optimized tables before bit-level encoding. It mirrors key routines
//! from C++ `entropy_coding.cc`, adapted for Rust and `HuffmanTable`.

use crate::huffman::HuffmanTable;
use crate::error::EncodingError;


use super::structs::JpegliComponentSettings;

/// Size of JPEG Huffman alphabet plus one sentinel.
pub const HUFFMAN_ALPHABET_SIZE: usize = 257;

#[inline]
fn bit_width(x: u32) -> usize {
    if x == 0 { 0 } else { (32 - x.leading_zeros()) as usize }
}

/// Builds DC and AC histograms for each Huffman table index.
///
/// - `coefficients`: quantized DCT blocks per component.
/// - `components`: per-component info (including `dc_huffman_table_index` and `ac_huffman_table_index`).
///
/// Returns two vectors:
/// - `dc_histograms[idx]` holds counts for symbols 0..255 and sentinel at 256 for DC table `idx`.
/// - `ac_histograms[idx]` holds counts for AC run-length symbols.
pub fn build_histograms(
    coefficients: &[Vec<[i16; 64]>],
    components: &[JpegliComponentSettings],
) -> (
    Vec<[u32; HUFFMAN_ALPHABET_SIZE]>,
    Vec<[u32; HUFFMAN_ALPHABET_SIZE]>,
) {
    if components.len() != coefficients.len() {
        panic!("components and coefficients must have the same length");
    }

    let num_dc = components.iter().map(|c| c.dc_huffman_table_index as usize).max().unwrap_or(0) + 1;
    let num_ac = components.iter().map(|c| c.ac_huffman_table_index as usize).max().unwrap_or(0) + 1;
    let mut dc_hist = vec![[0u32; HUFFMAN_ALPHABET_SIZE]; num_dc];
    let mut ac_hist = vec![[0u32; HUFFMAN_ALPHABET_SIZE]; num_ac];
    // Ensure sentinel counts
    for hist in &mut dc_hist { hist[256] = 1; }
    for hist in &mut ac_hist { hist[256] = 1; }
    // Populate histograms
    for (comp, blocks) in components.iter().zip(coefficients) {
        let dc_idx = comp.dc_huffman_table_index as usize;
        let ac_idx = comp.ac_huffman_table_index as usize;
        let mut last_dc = 0i32;
        for block in blocks {
            // DC difference category
            let dc = block[0] as i32;
            let diff = dc - last_dc;
            last_dc = dc;
            let abs_diff = diff.abs() as u32;
            let cat = bit_width(abs_diff);
            dc_hist[dc_idx][cat] += 1;
            // AC run-length and size symbols
            let mut run = 0usize;
            for &coef in &block[1..] {
                if coef == 0 {
                    run += 1;
                } else {
                    let mut r = run;
                    while r > 15 {
                        ac_hist[ac_idx][0xF0] += 1; // ZRL symbol
                        r -= 16;
                    }
                    let abs_coef = coef.abs() as u32;
                    let bits = bit_width(abs_coef);
                    let sym = (r << 4) + bits;
                    ac_hist[ac_idx][sym] += 1;
                    run = 0;
                }
            }
            // End-of-block symbol
            ac_hist[ac_idx][0x00] += 1;
        }
    }
    (dc_hist, ac_hist)
}

/// Creates optimized Huffman tables from DC and AC histograms.
///
/// - `dc_histograms`: histogram per DC table index.
/// - `ac_histograms`: histogram per AC table index.
///
/// Returns a vector of `(dc_table, ac_table)` pairs for each index.
pub fn optimize_huffman_tables(
    dc_histograms: &[[u32; HUFFMAN_ALPHABET_SIZE]],
    ac_histograms: &[[u32; HUFFMAN_ALPHABET_SIZE]],
) -> Vec<(HuffmanTable, HuffmanTable)> {
    dc_histograms.iter().zip(ac_histograms.iter()).map(|(dc, ac)| {
        let mut dc_freq = *dc;
        let mut ac_freq = *ac;
        dc_freq[256] = dc_freq[256].max(1);
        ac_freq[256] = ac_freq[256].max(1);
        let dc_table = HuffmanTable::new_optimized(dc_freq);
        let ac_table = HuffmanTable::new_optimized(ac_freq);
        (dc_table, ac_table)
    }).collect()
}



/// Single entry to perform full two-pass Huffman optimization:
/// 1. Build histograms
/// 2. Optimize tables
/// There should be an equal number of components and coefficient vectors.
/// Each block in a vector is a quantized 8x8 DCT coefficient block.
/// Returns a vector of (dc_table, ac_table) pairs for the number of tables actually in use (fewer than num_components, usually)
pub fn optimize_entropy(
    components: &[JpegliComponentSettings],
    coefficients: &[Vec<[i16; 64]>],
) -> Result<Vec<(HuffmanTable, HuffmanTable)>, EncodingError> {
    let (dc_hist, ac_hist) = build_histograms(coefficients, components);
    let tables = optimize_huffman_tables(&dc_hist, &ac_hist);
    Ok(tables)
} 