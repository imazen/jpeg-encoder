//! Ported data structures and configuration for JPEG encoding stages.
//!
//! This module defines user-facing options, resolved configurations, and
//! per-stage buffers that minimize mutability and encode the state machine
//! transitions clearly.

/// Buffers for the input and downsample stages.
/// Populated by `AllocateBuffers()` and mutated by `ReadInputRow` /
/// `PadInputBuffer` / `ApplyInputSmoothing` / `DownsampleInputBuffer`.
pub struct InputBuffers {
    /// Raw image rows per component.
    pub input_buffer: [RowBuffer<f32>; 4],

    /// Smoothed rows when smoothing_factor > 0.
    pub smooth_input: [RowBuffer<f32>; 4],

    /// Downsampled rows when subsampling >1.
    pub raw_data: [RowBuffer<f32>; 4],
}

/// Buffers for adaptive-quantization:
/// `diff_buffer`: mutated by `ComputePreErosion()`;
/// `pre_erosion`, `fuzzy_erosion_tmp`, and `quant_field` mutated by
/// `ComputePreErosion()`, `FuzzyErosion()`, `PerBlockModulations()`,
/// and final exponent adjustment in `ComputeAdaptiveQuantField()`.
pub struct AdaptiveQuantBuffers {
    pub diff_buffer: Vec<f32>,
    pub pre_erosion: RowBuffer<f32>,
    pub fuzzy_erosion_tmp: RowBuffer<f32>,
    pub quant_field: RowBuffer<f32>,
}



/// Huffman and entropy coding data, set up in `CopyHuffmanTables` and
/// optimized in `OptimizeHuffmanCodes`, packed by `InitEntropyCoder`.
pub struct EntropyCodingBuffers {
    /// Raw JPEG Huffman tables for DHT emission.
    pub huffman_tables: Vec<JpegHuffTable>,

    /// Slot IDs for each table index.
    pub slot_id_map: Vec<u8>,

    /// Context-to-table mapping for DC/AC contexts.
    pub context_map: Vec<u8>,

    /// Precomputed code/depth tables used by `WriteiMCURow()`.
    pub coding_tables: Vec<HuffmanCodeTable>,
}

/// Tokens emitted during progressive or optimized encoding.
pub struct TokenBuffers {
    /// Per-row or per-scan token arrays populated in `TokenizeJpeg()`.
    pub token_arrays: Vec<TokenArray>,

    /// Refinement tokens for progressive scans.
    pub refinement_tokens: Vec<RefToken>,

    /// Bits counts corresponding to refinement tokens.
    pub refinement_bits: Vec<u8>,
}

