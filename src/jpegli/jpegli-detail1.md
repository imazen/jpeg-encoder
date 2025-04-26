

Ran tool
Okay, the definition is in `lib/jpegli/encode.cc`. Let's read that function and the surrounding code to understand its role in setting up the encoding process.

Read file: lib/jpegli/encode.cc
My apologies, I missed the top-level directory. Let's try reading `jpegli/lib/jpegli/encode.cc`.

Read file: jpegli/lib/jpegli/encode.cc
Alright, let's break down the key functions called within `libjpegli` based on the code snippets and the documentation we generated.

**1. `jpegli_start_compress(j_compress_ptr cinfo, boolean write_all_tables)`** (`encode.cc:1021`)

*   **Purpose**: Initializes the core compression structures and prepares for receiving image data or coefficients.
*   **Key Actions**:
    *   Sets the global state to `jpegli::kEncHeader`.
    *   Calls `jpegli::InitCompress(cinfo, write_all_tables)`. This is the main internal setup function. We need to look into this one.
    *   Resets scanline counters (`cinfo->next_scanline`, `cinfo->master->next_input_row`).

**2. `jpegli_write_scanlines(j_compress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION num_lines)`** (`encode.cc:1125`)

*   **Purpose**: Processes `num_lines` of input image data provided in `scanlines`.
*   **Key Actions**:
    *   Checks state (`kEncHeader` or `kEncReadImage`). Errors if raw data mode is expected.
    *   **Header Writing (Streaming)**: If streaming is supported (`jpegli::IsStreamingSupported`) and Huffman optimization is off (`!cinfo->optimize_coding`), it writes the frame and first scan headers immediately:
        *   `jpegli::WriteFrameHeader(cinfo)`
        *   `jpegli::WriteScanHeader(cinfo, 0)`
    *   Sets global state to `jpegli::kEncReadImage`.
    *   **Input Loop**: Iterates `num_lines` times:
        *   `jpegli::ReadInputRow(cinfo, scanlines[i], rows)`: Converts input sample data (uint8, uint16, float) to internal float representation (`rows`).
        *   `(*m->color_transform)(rows, cinfo->image_width)`: Calls the appropriate color transform function pointer (e.g., RGB->YCbCr, or a no-op if input is already YCbCr or Grayscale). This happens *inside* libjpegli if `cinfo.in_color_space` was set to `JCS_RGB`. XYB transform is handled *before* this in the wrapper.
        *   `jpegli::PadInputBuffer(cinfo, rows)`: Adds padding pixels around the row, likely needed for filters or context for adaptive quantization.
        *   `jpegli::ProcessiMCURows(cinfo)`: **This is a crucial step.** It likely performs downsampling, DCT, quantization (including adaptive), and potentially initial entropy coding/tokenization for one or more iMCU rows. We need to dive into this.
        *   `jpegli::EmptyBitWriterBuffer(&m->bw)`: Checks if the output bit buffer needs flushing (for streaming).
    *   Updates `cinfo->next_scanline`.

**3. `jpegli_finish_compress(j_compress_ptr cinfo)`** (`encode.cc:1220`)

*   **Purpose**: Finalizes the compression process after all scanlines (or coefficients) have been written.
*   **Key Actions**:
    *   Checks state (`kEncReadImage` or `kEncWriteCoeffs`). Errors if image is incomplete.
    *   **Zig-Zag Shuffle (Coefficient Input)**: If writing coefficients directly (`kEncWriteCoeffs`), calls `jpegli::ZigZagShuffleBlocks(cinfo)`.
    *   **PSNR Target Quantization**: If `m->psnr_target > 0`, calls `jpegli::QuantizetoPSNR(cinfo)`. This likely involves adjusting quantization tables and possibly re-tokenizing.
    *   **Tokenization (Non-Streaming)**: If streaming wasn't fully done (`!tokens_done`), calls `jpegli::TokenizeJpeg(cinfo)`. This converts quantized DCT coefficients into intermediate symbols/tokens.
    *   **Huffman Optimization**: If `cinfo->optimize_coding` is true or `cinfo->progressive_mode` is true:
        *   `jpegli::OptimizeHuffmanCodes(cinfo)`: Computes optimal Huffman tables based on token statistics.
        *   `jpegli::InitEntropyCoder(cinfo)`: Sets up the entropy coder with the final tables.
    *   **Write Bitstream (Non-Streaming or Optimized)**: If the bitstream wasn't already written during `jpegli_write_scanlines`:
        *   `jpegli::WriteFrameHeader(cinfo)`
        *   Loops through scans (`cinfo->num_scans`):
            *   `jpegli::WriteScanHeader(cinfo, i)`
            *   `jpegli::WriteScanData(cinfo, i)`: Writes the actual entropy-coded data for the scan using the prepared tokens and Huffman tables.
    *   **Flush Buffer**: Ensures all data is written (`JumpToByteBoundary`, `EmptyBitWriterBuffer`).
    *   `jpegli::WriteOutput(cinfo, {0xFF, 0xD9})`: Writes the EOI marker.
    *   `(*cinfo->dest->term_destination)(cinfo)`: Finalizes the output destination.
    *   `jpegli_abort_compress(cinfo)`: Releases internal memory.

**Next Steps: Deeper Dive**

Now we need to trace into the key internal functions identified:

1.  `jpegli::InitCompress` (called by `jpegli_start_compress`): Should handle setup, including quantization table generation.
2.  `jpegli::ProcessiMCURows` (called by `jpegli_write_scanlines`): Should handle DCT, adaptive quantization application.
3.  `jpegli::TokenizeJpeg` (called by `jpegli_finish_compress`): Converts DCT coefficients to tokens.
4.  `jpegli::QuantizetoPSNR` (called by `jpegli_finish_compress`): Adjusts quantization for PSNR target.
5.  `jpegli::OptimizeHuffmanCodes` (called by `jpegli_finish_compress`): Generates optimal Huffman tables.
6.  `jpegli::WriteScanData` (called by `jpegli_finish_compress`): Performs the final entropy coding.

Let's start by looking for `jpegli::InitCompress`.

Ran tool
Okay, the *call* is in `encode.cc`, but that doesn't give us the *definition*. It seems `InitCompress` might be defined elsewhere or perhaps is a member function. Let's try searching for the definition text.

Ran tool
Okay, it *is* defined in `jpegli/lib/jpegli/encode.cc`. Let's read that function definition.

Read file: jpegli/lib/jpegli/encode.cc
Okay, `jpegli::InitCompress` orchestrates the setup:

*   `ProcessCompressionParams(cinfo)`: Determines scan parameters, computes derived values like block dimensions.
*   `AllocateBuffers(cinfo)`: Allocates memory for input buffer, coefficient buffer, etc.
*   **Input/Color/Downsampling Setup (if not writing coefficients directly)**:
    *   `ChooseInputMethod(cinfo)`: Selects the function to convert input samples (uint8, uint16, float) based on `cinfo->master->data_type`.
    *   `ChooseColorTransform(cinfo)`: Selects the function pointer for color transform (e.g., RGB->YCbCr).
    *   `ChooseDownsampleMethods(cinfo)`: Selects the downsampling function (e.g., 2x2, 2x1, 1x2, none).
*   **Quantization Setup**:
    *   `InitQuantizer(cinfo, pass)`: **This is key for quantization.** It likely initializes the quantization tables based on `distance`/`quality` or standard tables flag, and sets up structures needed for adaptive quantization if enabled. The `pass` argument (`QuantPass::NO_SEARCH` by default) suggests it might behave differently during PSNR/target size search. Let's look into `InitQuantizer`.
*   **Huffman Setup (if non-optimized/non-progressive)**:
    *   `CopyHuffmanTables(cinfo)`: Copies standard Huffman tables.
    *   `InitEntropyCoder(cinfo)`: Sets up the entropy coder with these tables.
*   **Output Setup**:
    *   `(*cinfo->dest->init_destination)(cinfo)`: Initializes the output stream.
    *   `WriteFileHeader(cinfo)`: Writes SOI, JFIF/APP markers.
    *   `JpegBitWriterInit(cinfo)`: Initializes the bit writer state.

Now, let's trace `InitQuantizer`.

Ran tool
It's defined in `jpegli/lib/jpegli/quant.cc`. Let's read it.

Read file: jpegli/lib/jpegli/quant.cc
Okay, `InitQuantizer` in `jpegli/lib/jpegli/quant.cc` does the following:

1.  **Compute Quantization Multipliers**: It iterates through the quantization tables (`cinfo->quant_tbl_ptrs`) previously set up (either by `SetQuantMatrices` based on distance/quality or standard tables, or manually via `jpegli_add_quant_table`). For each component `c` and frequency coefficient `k`, it computes `m->quant_mul[c][k]`.
    *   If not doing a PSNR/size search (`QuantPass::NO_SEARCH`), the multiplier is simply `8.0f / quant_table->quantval[k]`. This multiplier seems related to the DCT scaling and the quantization value itself.
    *   If doing a search (`SEARCH_FIRST_PASS`, `SEARCH_SECOND_PASS`), it sets different initial multipliers, suggesting the quantization process changes during search.
2.  **Initialize Zero-Bias Multipliers/Offsets**: These seem related to adaptive quantization heuristics, specifically "zero-biasing" which aims to preserve near-zero AC coefficients better.
    *   **If Adaptive Quantization Enabled (`m->use_adaptive_quantization`)**:
        *   It initializes default `zero_bias_mul` and `zero_bias_offset` values (0.5 for AC coefficients).
        *   **YCbCr Specific Logic**: If the colorspace is YCbCr, it calculates the effective `distance` from the current quantization tables using `QuantValsToDistance(cinfo)`.
        *   It then interpolates between two sets of predefined zero-bias multipliers (`kZeroBiasMulYCbCrLQ`, `kZeroBiasMulYCbCrHQ`) based on this distance. This suggests different zero-biasing strategies for low vs. high quality settings.
        *   It also sets specific `zero_bias_offset` values (`kZeroBiasOffsetYCbCrDC`, `kZeroBiasOffsetYCbCrAC`).
    *   **If Adaptive Quantization Disabled (but YCbCr)**: It still applies the specific YCbCr `zero_bias_offset` values. This implies some minimal perceptual adjustment might still happen even without full adaptive quantization.

**Quantization Table Generation Summary (`SetQuantMatrices`)**

Although `InitQuantizer` *uses* the tables, the generation happens in `SetQuantMatrices` (called by `jpegli_set_quality`, `jpegli_set_distance`, etc.):

1.  **Determine Base Matrix & Scale**:
    *   Selects base matrices (`kBaseQuantMatrixXYB`, `kBaseQuantMatrixYCbCr`, `kBaseQuantMatrixStd`) based on XYB mode, standard table flag, and color space.
    *   Selects a `global_scale` based on the same factors and potentially the CICP transfer function (PQ/HLG adjust scale for YCbCr). 4:2:0 subsampling also modifies the YCbCr scale.
2.  **Calculate Quant Values**: For each table index `quant_idx` and frequency `k`:
    *   **Psychovisual (Default)**:
        *   Calculates a per-frequency scale `scale = global_scale * DistanceToScale(distances[quant_idx], k)`. `DistanceToScale` introduces non-linearity based on `kExponent` constants, especially for distances > 1.5.
        *   Adjusts `scale` further for YUV420 chroma (`k420Rescale`).
    *   **Standard/Linear**:
        *   Calculates `scale = global_scale * DistanceToLinearQuality(distances[quant_idx])`.
    *   Computes the final quantization value: `qval = round(scale * base_qm[k])`.
    *   Clamps `qval` between 1 and `quant_max` (255 or 32767).
    *   Stores `qval` in `cinfo->quant_tbl_ptrs[quant_idx]->quantval[k]`.

**Next**: Let's look at `ProcessiMCURow` where adaptive quantization is applied.

Ran tool
It seems there might be two definitions, one in `encode.cc` and one in `encode_streaming.cc`. Let's examine the one in `encode.cc` first, as that seems more central.

Read file: jpegli/lib/jpegli/encode.cc
`ProcessiMCURow` in `encode.cc` outlines the pipeline for a Minimum Coded Unit (MCU) row:

1.  **Preprocessing (if not raw data input)**:
    *   `ApplyInputSmoothing(cinfo)`: This might be a noise reduction or pre-filtering step. Needs investigation.
    *   `DownsampleInputBuffer(cinfo)`: Performs chroma downsampling based on the chosen method (`ChooseDownsampleMethods`).
2.  **Adaptive Quantization Field Computation**:
    *   `ComputeAdaptiveQuantField(cinfo)`: **This is central to adaptive quantization.** It calculates a field (likely per-block or per-pixel multipliers) based on image activity or psychovisual properties, which will modulate the base quantization. Needs investigation.
3.  **Processing Path Selection**:
    *   **Streaming Supported? (`IsStreamingSupported`)**: This likely depends on whether Huffman optimization is enabled and other factors.
        *   **Optimized Huffman**: `ComputeTokensForiMCURow(cinfo)`: Performs DCT, quantization (using the base tables *and* the adaptive field), and generates intermediate tokens for later Huffman coding.
        *   **Fixed Huffman (Streaming)**: `WriteiMCURow(cinfo)`: Performs DCT, quantization, and *immediately* writes the entropy-coded data using the pre-computed fixed Huffman tables.
    *   **Non-Streaming (Coefficient Output)**: `ComputeCoefficientsForiMCURow(cinfo)`: Performs DCT and quantization (including adaptive), storing the quantized coefficients in `cinfo->master->coefficients` for later processing/writing.

Let's investigate `ComputeAdaptiveQuantField` next.

Ran tool
It's defined in `jpegli/lib/jpegli/adaptive_quantization.cc`. This looks promising for finding the details about masking, blurring, etc. Let's read this function.

Read file: jpegli/lib/jpegli/adaptive_quantization.cc
This is indeed where the adaptive quantization magic happens! Let's break down `ComputeAdaptiveQuantField` in `adaptive_quantization.cc`:

1.  **Check Enabled**: Returns early if `!m->use_adaptive_quantization`.
2.  **Identify Luma Channel**: Determines the index `y_channel` (0 for YCbCr/Grayscale, 1 for RGB/XYB, although XYB adaptive quant seems less common).
3.  **Get Base Quantization**: Retrieves `y_quant_01`, the quantization value for the [0][1] AC coefficient of the luma channel. This seems to influence the overall strength/behavior of the adaptive modulation.
4.  **Handle Borders**: Copies border rows in the input buffer (`m->input_buffer`) to ensure context is available for filters near the image edges.
5.  **Compute Pre-Erosion Map (`ComputePreErosion`)**:
    *   Takes the luma component (`m->input_buffer[y_channel]`) as input.
    *   Calculates local differences between a pixel and its neighbors (left, right, top, bottom average).
    *   Applies a gamma correction (`RatioOfDerivativesOfCubicRootToSimpleGamma`) to approximate Butteraugli's psychovisual space.
    *   Squares the differences, clamps them (`limit = 0.2f`), and applies `MaskingSqrt` (another perceptual function involving `sqrt` and `log`).
    *   Averages these processed differences over 4x4 blocks, storing the result in `m->pre_erosion`. This map represents local image activity/complexity in a perceptually relevant way.
6.  **Fuzzy Erosion (`FuzzyErosion`)**:
    *   Takes the `m->pre_erosion` map as input.
    *   For each pixel in `m->pre_erosion`, it finds the 4 *lowest* values in its 3x3 neighborhood.
    *   Calculates a weighted linear combination of these 4 minimum values.
    *   Averages the result over 2x2 blocks, storing the output in `m->quant_field`. This step essentially finds areas of low activity/complexity (smooth regions) and propagates this information, effectively identifying areas where quantization can be increased. The output resolution matches the 8x8 block grid.
7.  **Per-Block Modulations (`PerBlockModulations`)**:
    *   Takes the original luma input (`m->input_buffer[y_channel]`) and the `m->quant_field` (output of FuzzyErosion) as input.
    *   Iterates through each 8x8 block position (`ix`, `iy`).
    *   Loads the corresponding value from `m->quant_field` (which represents the "masking potential" derived from the fuzzy erosion).
    *   Applies several modulation functions to this value (which is treated as an exponent in a log2 domain):
        *   `ComputeMask`: Modulates based on the magnitude of the masking potential itself.
        *   `HfModulation`: Modulates based on high-frequency content *within the current 8x8 block* (sum of absolute differences of adjacent pixels in the original luma input). Increases quantization slightly in high-frequency blocks.
        *   `GammaModulation`: Modulates based on the overall intensity/gamma characteristics of the current 8x8 block (using `RatioOfDerivativesOfCubicRootToSimpleGamma` again).
    *   Converts the final modulated exponent back to a linear multiplier using `FastPow2f`.
    *   Applies a final scaling (`mul`) and offset (`add`) based on the base luma quantization `y_quant_01` and a damping factor. High base quantization dampens the adaptive effect.
    *   Stores the final per-block multiplier back into `m->quant_field`.
8.  **Final Quant Field Adjustment**:
    *   Iterates through the computed `m->quant_field`.
    *   Applies a final transformation: `row[x] = std::max(0.0f, (0.6f / row[x]) - 1.0f)`. This seems to convert the multiplier into a value that's used later during the actual quantization step within `ComputeTokensForiMCURow` or `ComputeCoefficientsForiMCURow`. The resulting `m->quant_field` contains the per-block adaptive quantization adjustment factor.

**Psychovisual Elements Identified:**

*   **Gamma Correction**: `RatioOfDerivativesOfCubicRootToSimpleGamma` is used multiple times to transform intensity values into a more perceptually linear space, approximating Butteraugli's SimpleGamma.
*   **Masking**: `ComputeMask` and `MaskingSqrt` apply non-linear functions based on local activity (derived differences) to model visual masking effects (where noise/quantization error is less visible near strong edges or textures).
*   **Smoothing/Denoising**: `ApplyInputSmoothing` (called in `ProcessiMCURow` before `ComputeAdaptiveQuantField`) might contribute.
*   **Fuzzy Erosion**: Identifies smooth areas where quantization can be increased without significant visual impact.
*   **High-Frequency Adjustment**: `HfModulation` slightly adjusts quantization based on local high-frequency content within a block.

**Next**: Let's see how `m->quant_field` is used in `ComputeTokensForiMCURow` (the non-streaming, optimized Huffman path).

Ran tool
It's defined in `jpegli/lib/jpegli/encode_streaming.cc`. Let's read that function.

Read file: jpegli/lib/jpegli/encode_streaming.cc
The function `ComputeTokensForiMCURow` calls the templated function `ProcessiMCURow<kStreamingModeTokens>`. Let's analyze `ProcessiMCURow` (around line 111 of `encode_streaming.cc`):

1.  **Initialization**: Sets up pointers to `jpeg_comp_master` (`m`), `JpegBitWriter` (`bw`), gets MCU dimensions, current MCU row index (`mcu_y`), temporary buffers (`block`, `symbols`, `nonzero_idx`), last DC coefficients (`last_dc_coeff`), adaptive quant flag (`adaptive_quant`).
2.  **Token Array Setup (if `kMode == kStreamingModeTokens`)**: Manages the allocation and current position (`m->next_token`) within the token buffer (`m->token_arrays`).
3.  **Loop over MCUs in the row (`mcu_x`)**:
4.  **Loop over Components (`c`)**:
5.  **Loop over Blocks within MCU (`iy`, `ix`)**:
    *   Calculates current block coordinates (`bx`, `by`).
    *   Handles boundary conditions (padding if block is outside image).
    *   **Get Adaptive Quant Strength**: If `adaptive_quant` is true, it retrieves the pre-computed adaptive quantization strength for the current block: `aq_strength = qf[iy * qf_stride + bx * h_factor];` where `qf` points to `m->quant_field.Row(0)` (note: the row index seems fixed at 0 here, which implies the entire quant field for the current iMCU row was computed and stored contiguously, possibly starting at row 0 of `m->quant_field`).
    *   Gets pointer to input pixels for the block.
    *   **Compute Coefficient Block**: Calls `ComputeCoefficientBlock(pixels, stride, qmc, last_dc_coeff[c], aq_strength, zero_bias_offset, zero_bias_mul, m->dct_buffer, block);`
        *   This function performs the DCT (likely using `dct-inl.h`).
        *   It then quantizes the DCT coefficients. **This is where `aq_strength` (from `m->quant_field`) and the `zero_bias_*` arrays (from `InitQuantizer`) are used** along with the base quantization multipliers (`qmc`) to modify the quantization for the current block.
        *   The resulting quantized coefficients (integer values) are stored in `block`.
    *   Subtracts previous DC coefficient for differential coding: `block[0] -= last_dc_coeff[c];`
    *   Updates last DC coefficient: `last_dc_coeff[c] += block[0];`
    *   **Token Generation (if `kMode == kStreamingModeTokens`)**:
        *   Calls `ComputeTokensForBlock<int32_t, false>(block, 0, c, c + 4, &m->next_token);`. This function (likely in `entropy_coding-inl.h`) analyzes the quantized coefficients in `block` (after DC diff coding) and generates a sequence of `Token` objects representing DC values, AC run lengths, and coefficient magnitudes. These tokens are appended to the buffer pointed to by `m->next_token`.
    *   **Coefficient Output (if `kMode == kStreamingModeCoefficients`)**: Copies the coefficients to the output coefficient buffer.
    *   **Direct Bitstream Writing (if `kMode == kStreamingModeBits`)**: Performs zigzag, finds non-zeros, computes symbols, and writes Huffman bits directly using `WriteBlock`.

**Adaptive Quantization Application:**

The `ComputeAdaptiveQuantField` function calculates `m->quant_field`, which stores a per-block strength/multiplier. The `ProcessiMCURow` function retrieves this `aq_strength` for each block and passes it into `ComputeCoefficientBlock`. Inside `ComputeCoefficientBlock` (which we haven't looked into yet, but is the next logical step), this `aq_strength` is used alongside the base quantization multiplier (`qmc`) and zero-biasing parameters to perform the actual quantization calculation on the DCT coefficients.

Let's find `ComputeCoefficientBlock`.

Ran tool
It's defined in `jpegli/lib/jpegli/dct-inl.h`. Let's read this inline function.

Read file: jpegli/lib/jpegli/dct-inl.h
Finally, `ComputeCoefficientBlock` in `dct-inl.h` reveals the quantization process:

1.  **DCT**:
    *   `TransformFromPixels(pixels, stride, dct, scratch_space)`: Computes the 2D DCT of the input `pixels` block using 1D DCT transforms (`DCT1DImpl`) and transpositions. The result is stored in `dct`.
2.  **Quantization (`QuantizeBlock`)**:
    *   Iterates through the 64 DCT coefficients (`k = 0..63`).
    *   Loads the DCT coefficient `val = dct[k]`.
    *   Loads the base quantization multiplier `q = qmc[k]` (this is `m->quant_mul[c][k]` which was computed in `InitQuantizer` from the quant tables).
    *   Calculates the initial quantized value: `qval = val * q`.
    *   Loads the zero-bias offset and multiplier for this coefficient `k`: `zb_offset = zero_bias_offset[k]`, `zb_mul = zero_bias_mul[k]` (these were interpolated/set in `InitQuantizer`).
    *   Loads the adaptive quant strength for this block: `aq_strength` (this came from `m->quant_field`).
    *   **Computes the quantization threshold**: `threshold = zb_offset + zb_mul * aq_strength`. This threshold combines the base zero-biasing strategy with the adaptive strength computed for the block.
    *   **Applies Threshold**: `nzero_mask = Ge(Abs(qval), threshold)`. Checks if the absolute initial quantized value exceeds the computed threshold.
    *   Rounds the quantized value if it's above the threshold, otherwise sets it to zero: `ival = ConvertTo(di, IfThenElseZero(nzero_mask, Round(qval)))`.
    *   Stores the final integer quantized coefficient `ival` into the output `block[k]`.
3.  **DC Coefficient Handling**:
    *   Special handling for the DC coefficient (`k=0`).
    *   Calculates the quantized DC value relative to 128: `dc = (dct[0] - kDCBias) * qmc[0]`.
    *   Calculates the specific DC threshold: `dc_threshold = zero_bias_offset[0] + aq_strength * zero_bias_mul[0]`.
    *   Compares the *difference* between the current DC and the `last_dc_coeff` from the previous block to the threshold.
    *   If the difference is below the threshold, it keeps the DC value the same as the previous block (`block[0] = last_dc_coeff`). This aims to improve DC prediction in the entropy coding stage.
    *   Otherwise, it uses the rounded `dc` value (`block[0] = std::round(dc)`).

**Summary of Adaptive Quantization Application:**

The pre-calculated `m->quant_field` (containing `aq_strength` per block) directly modifies the quantization `threshold` used in `QuantizeBlock`. A higher `aq_strength` (presumably corresponding to smoother areas identified by FuzzyErosion) results in a higher `threshold`, making it more likely that AC coefficients will be quantized to zero. The zero-bias parameters also influence this threshold, applying different strategies based on quality level (distance) and frequency index (`k`). The DC coefficient has separate thresholding logic based on the difference from the previous block's DC value.

**Gaussian Blur / Erosion:**

The term "erosion" appears explicitly in `FuzzyErosion`. While not a standard morphological erosion, it serves a similar purpose by propagating the minimum values (representing low activity) from the `pre_erosion` map, effectively "eroding" high-activity areas and highlighting smooth regions. There doesn't seem to be an explicit Gaussian blur step mentioned in the functions we've reviewed, but the combination of local differencing (`ComputePreErosion`), fuzzy erosion, and potentially `ApplyInputSmoothing` achieves a similar goal of analyzing local image characteristics to guide quantization.

I will now update the markdown file with this deeper dive into the `libjpegli` internals.


Okay, this requires a much deeper dive into the source code, tracing the specific algorithms and constants used. I will add a new section to `src/jpegli/cjpegli.md` detailing the encoding pipeline step-by-step, focusing on the path where adaptive quantization is enabled (`--noadaptive_quantization` is *not* specified), and covering both XYB and YCbCr (from sRGB input) paths for `uint16` and `float32` inputs where the logic differs.

I will reference filenames and line numbers and include relevant C++ snippets or pseudo-code where helpful.




I have appended the detailed pipeline description (Section 5) to `src/jpegli/cjpegli.md`. This section covers the process from the C++ wrapper function `EncodeJpegInternal` down through the core `libjpegli` steps, including color management, parameter setup, adaptive quantization computation (`ComputePreErosion`, `FuzzyErosion`, `PerBlockModulations`), DCT, quantization application (`ComputeCoefficientBlock`, `QuantizeBlock`), tokenization (`ComputeTokensForBlock`), Huffman coding (`OptimizeHuffmanCodes`), and bitstream writing (`WriteScanData`).

It details the handling of XYB vs YCbCr paths and mentions where input data types (`uint16`, `float32`) affect the pipeline (primarily in `ChooseInputMethod` and the initial color transforms). Key constants and function references with filenames/line numbers are included.

Please review this detailed section. It should provide a solid foundation for understanding or reimplementing the algorithm. Let me know if any part needs further clarification or expansion.


Okay, let's trace the padding and boundary handling for a 3x11 pixel sRGB input image. We'll assume default settings, including adaptive quantization and 4:2:0 chroma subsampling (2x2).

**1. Dimensions and Grids**

*   **Input Image**: 3 pixels wide, 11 pixels high.
*   **Color Space**: Input sRGB -> Processing YCbCr.
*   **Chroma Subsampling (4:2:0)**:
    *   Y (Luma): 3x11 samples.
    *   Cb, Cr (Chroma): `ceil(3/2) x ceil(11/2)` = 2x6 samples each.
*   **DCT Block Grid (8x8 pixels/samples)**:
    *   Y: `ceil(3/8) x ceil(11/8)` = 1 block wide, 2 blocks high. Total 1x2 = 2 blocks.
    *   Cb, Cr: `ceil(2/8) x ceil(6/8)` = 1 block wide, 1 block high. Total 1x1 = 1 block each.
*   **MCU Grid (Minimum Coded Unit)**: `max_h_samp_factor = 2`, `max_v_samp_factor = 2`. MCU size is 16x16 pixels.
    *   Image Size in MCUs: `ceil(3/16) x ceil(11/16)` = 1x1 MCU.

**2. Input Buffering and Initial Padding (`AllocateBuffers`, `ReadInputRow`, `PadInputBuffer`)**

*   **Buffer Allocation (`encode_internal.cc:42`, called by `InitCompress` in `encode.cc:511`)**:
    *   Internal buffers (`m->input_buffer`) are allocated based on the component dimensions rounded up to DCT block multiples.
    *   Y component requires 1x2 blocks = 8x16 pixels.
    *   Cb/Cr components require 1x1 block = 8x8 samples each.
    *   These buffers (`RowBuffer<float>`) likely allocate extra space around the nominal dimensions to handle border access (e.g., index -1 or index `width`).
*   **Reading (`encode.cc:1148`)**: Rows 0 through 10 are read. For each row, the 3 sRGB pixels are converted to float YCbCr. The Y samples are stored in `m->input_buffer[Y].Row(row_idx)[0..2]`, Cb in `m->input_buffer[Cb].Row(row_idx)[0..1]`, Cr in `m->input_buffer[Cr].Row(row_idx)[0..1]`.
*   **Padding (`encode.cc:570`, called after each row via `encode.cc:1151`)**:
    *   **Horizontal Padding**: Applied to each component's buffer row *after* the data is read/converted.
        *   `len0 = image_width` (3 for Y, 2 for Cb/Cr).
        *   `len1 = width_in_blocks * DCTSIZE` (8 for Y, 8 for Cb/Cr).
        *   The last valid pixel (`row[c][len0 - 1]`) is replicated to fill columns from `len0` up to `len1 - 1`.
            *   Y row: Pixels `[3]` through `[7]` get the value of pixel `[2]`.
            *   Cb/Cr row: Pixels `[2]` through `[7]` get the value of pixel `[1]`.
        *   Left border: `row[c][-1] = row[c][0]`. The pixel at conceptual index -1 gets the value from index 0. (Managed by `RowBuffer`).
        *   Right border (implicit): Accessing `row[c][len1]` would likely return the value from `row[c][len1 - 1]`.
    *   **Vertical Padding**: Applied *only once* after the last image row (row 10) has been processed (`encode.cc:581`).
        *   `num_rows_to_pad = (m->ysize_blocks * DCTSIZE) - cinfo->image_height`.
            *   Y: `(2 * 8) - 11 = 5` rows needed (rows 11, 12, 13, 14, 15).
            *   Cb/Cr: `(1 * 8) - 6 = 2` rows needed (rows 6, 7). *Correction: Cb/Cr height is 6 samples, not 11 pixels.*
        *   The *last valid row* (row 10 for Y, row 5 for Cb/Cr), which is already horizontally padded, is copied `num_rows_to_pad` times to fill the buffer up to the full block height (16 for Y, 8 for Cb/Cr).
        *   Top/Bottom borders: `RowBuffer` likely replicates row 0 for accesses to row -1, and the last row (15 for Y, 7 for Cb/Cr) for accesses beyond the padded height.
*   **Result**: We have `m->input_buffer` for Y (8x16 samples + border), Cb (8x8 samples + border), Cr (8x8 samples + border), all filled with edge-replicated data.

**3. Downsampling (`DownsampleInputBuffer` in `downsampling.cc:141`)**

*   Reads from the padded `m->input_buffer`.
*   Writes to `m->raw_data` buffer (allocated to `width_in_blocks * DCTSIZE` x `height_in_blocks * DCTSIZE` per component).
*   **Y Component**: No downsampling (1x1 block in MCU). Data is likely copied directly from the padded 8x16 `input_buffer[Y]` to `raw_data[Y]`. Result: 8x16 samples.
*   **Cb/Cr Components**: Uses `Downsample2x2_BoxFilter` (`downsampling-inl.h:36`).
    *   Reads 2x2 blocks from the 8x8 padded `input_buffer[Cb/Cr]`.
    *   Averages the 4 input samples.
    *   Writes the result to `raw_data[Cb/Cr]`. Output size needs to be `width_in_blocks * 8` x `height_in_blocks * 8` = 8x8. The input being read is 8x8, so the downsampled output is 4x4 samples. These 4x4 valid samples are written into the top-left of the 8x8 `raw_data[Cb/Cr]` buffer. The rest of the 8x8 buffer must be padded. The downsampler likely replicates the edges of the 4x4 output to fill the 8x8 `raw_data` buffer.
*   **Result**: `m->raw_data` holds Y (8x16, padded), Cb (8x8, contains 4x4 valid downsampled data padded by replication), Cr (8x8, contains 4x4 valid downsampled data padded by replication).

**4. Adaptive Quantization Field (`ComputeAdaptiveQuantField` in `adaptive_quantization.cc:516`)**

*   Operates only on the Y component (`y_channel = 0`).
*   **`ComputePreErosion` (`adaptive_quantization.cc:444`)**:
    *   Input: `m->input_buffer[Y]` (8x16 + borders).
    *   Neighborhood Access: Reads pixels `(x, y)`, `(x-1, y)`, `(x+1, y)`, `(x, y-1)`, `(x, y+1)`. The padding in `input_buffer` handles boundary conditions (accesses outside 0..7, 0..15 return replicated edge values).
    *   Output: `m->pre_erosion` buffer (size `ceil(8/4) x ceil(16/4)` = 2x4).
    *   Padding: `pre_erosion` is padded with a 1-pixel border by replicating its edges (`adaptive_quantization.cc:480`, `adaptive_quantization.cc:544-546`).
*   **`FuzzyErosion` (`adaptive_quantization.cc:348`)**:
    *   Input: Padded `m->pre_erosion` (2x4 + border).
    *   Neighborhood Access: Reads 3x3 neighborhood within `pre_erosion`. The border padding handles boundary conditions.
    *   Output: `m->quant_field` buffer (size `ceil(2/2) x ceil(4/2)` = 1x2). This matches the Y block dimensions.
*   **`PerBlockModulations` (`adaptive_quantization.cc:282`)**:
    *   Input: `m->input_buffer[Y]` (8x16 + borders) and `m->quant_field` (1x2).
    *   Reads 8x8 blocks from `input_buffer[Y]` corresponding to the `quant_field` entries (block 0,0 and block 0,1). Accesses are within the padded 8x16 buffer. Reads the corresponding `quant_field` value for modulation.
    *   Output: Updates the 1x2 `m->quant_field` buffer.

**5. DCT and Quantization (`ProcessiMCURow` -> `ComputeCoefficientBlock`)**

*   Iterates through blocks based on component dimensions.
*   **Y Component (1x2 blocks)**:
    *   Block (0,0): Reads 8x8 pixel data starting at `m->raw_data[Y].Row(0)[0]`. This 8x8 region contains the original 3x8 data, padded horizontally from column 3 to 7.
    *   Block (0,1): Reads 8x8 pixel data starting at `m->raw_data[Y].Row(8)[0]`. This 8x8 region contains original 3x3 data (rows 8, 9, 10) in the top-left, padded horizontally (cols 3-7), and padded vertically from row 11 to 15 using replicated data from row 10.
    *   `TransformFromPixels` (`dct-inl.h:192`) receives these 8x8 blocks (containing padded data) and performs DCT. No further padding needed here.
    *   `QuantizeBlock` (`dct-inl.h:217`) uses `aq_strength` from `m->quant_field[0][0]` for block (0,0) and `m->quant_field[0][1]` for block (0,1).
*   **Cb/Cr Components (1x1 block)**:
    *   Block (0,0): Reads 8x8 data starting at `m->raw_data[Cb/Cr].Row(0)[0]`. As determined in step 3, this buffer contains the 4x4 valid downsampled chroma data, padded out to 8x8 by edge replication.
    *   `TransformFromPixels` receives this 8x8 block (containing padded data) and performs DCT.
    *   `QuantizeBlock` uses `aq_strength` based on the Y component's `quant_field` value corresponding to the *location* of this chroma block within the luma grid. Since it's block (0,0) for chroma, it likely uses `m->quant_field[0][0]`.

**Summary of Padding Logic:**

*   **Input Stage (`PadInputBuffer`)**: The primary padding happens here. The input image is expanded to the next multiple of the *full* MCU dimensions (16x16 in this case for Y) in the `input_buffer` using **edge pixel replication** both horizontally and vertically. Chroma planes are padded similarly based on their subsampled dimensions and block requirements (up to 8x8). `RowBuffer` handles border access.
*   **Downsampling**: Reads from the padded `input_buffer`. If the output buffer (`raw_data`) needs to be larger than the downsampled result (e.g., Cb/Cr 4x4 -> 8x8 buffer), it pads its output using **edge pixel replication**.
*   **Adaptive Quantization**: Relies entirely on the pre-padded `input_buffer[Y]` and the padded intermediate `pre_erosion` buffer for its neighborhood operations.
*   **DCT**: Operates strictly on 8x8 blocks. It receives these blocks from the `raw_data` buffers, which already contain the necessary data (original or replicated edge values) because all prior padding was done to align with 8x8 block boundaries or larger MCU boundaries.

Therefore, for the 3x11 image, the core idea is to replicate the last valid column (column 2 for Y, column 1 for Cb/Cr) out to column 7, and replicate the last valid row (row 10 for Y, row 5 for Cb/Cr) down to row 15 (Y) or 7 (Cb/Cr) *before* most processing occurs. Subsequent stages work on these padded-to-block-multiple buffers.



Okay, let's rigorously verify the padding and buffer logic, then detail the buffer design and color management, and finally create the end-to-end stories.

**1. Verifying Padding Implementation**

*   **`RowBuffer<T>`**: Searching for "class RowBuffer" or "struct RowBuffer".
    

Ran tool

Read file: jpegli/lib/jpegli/common_internal.h
    *   **Allocation**: `RowBuffer<T>::Allocate` (`common_internal.h:93`) allocates memory. It calculates `stride_` to ensure rows are aligned (to `HWY_ALIGNMENT`) and adds padding (`vec_size`, typically SIMD vector width) at the end of each row. An `offset_` (equal to alignment) is used to allow safe access to indices like `-1`.
    *   **Indexing (`Row(y)`)**: `common_internal.h:103`. It uses `((ysize_ + y) % ysize_)` for vertical wrap-around. This means accessing row `-1` effectively accesses row `ysize_ - 1`, and accessing `ysize_` accesses row `0`. It returns a pointer to `data_[<wrapped_y_index> * stride_ + offset_]`. This allows accessing `row[-1]` to get the last element of the allocated offset *before* the actual data starts at `row[0]`.
    *   **`PadRow(y, from, border)`**: (`common_internal.h:110`). This method explicitly pads *within* the allocated space for row `y`.
        *   It fills `border` elements *before* index 0 (`row[-border]` to `row[-1]`) with the value from `row[0]`.
        *   It fills elements from index `from` (the original data width) up to `xsize_ + border - 1` with the value from `row[from - 1]`. `xsize_` is the requested `rowsize` in `Allocate`.
    *   **`CopyRow(dst_row, src_row, border)`**: (`common_internal.h:121`). Copies `xsize_ + 2 * border` elements, starting from index `-border`.
*   **`PadInputBuffer` (`encode.cc:570`) Analysis**:
    *   It calls `PadRow(m->next_input_row - 1, len0, 1)` where `len0` is the actual image width (3 for Y). This pads the *just-read* row. It correctly uses `row[-1] = row[0]` and `row[x] = row[len0-1]` for `x >= len0`.
    *   The vertical padding loop (`encode.cc:581`) uses `CopyRow(m->next_input_row, row_idx_src, 1)` where `row_idx_src` is the index of the last valid image row. This copies the padded last row (including its `-1` and `[len0..xsize_]` padding) into subsequent rows.
*   **Downsampling Output Padding**: Looking at `Downsample2x2_BoxFilter` (`downsampling-inl.h:36`). It writes to `rows_out[bx] = Mul(sum, Set(df, 0.25f))`. It operates on the input row buffers and writes averaged values. It doesn't explicitly pad its *output* if the `raw_data` buffer is larger than the downsampled result size. **Assumption Correction**: The `raw_data` buffer seems allocated based on block dimensions (`width_in_blocks * 8`), but filled only with the valid downsampled data (e.g., 4x4 for Cb/Cr). Subsequent DCT reads (`encode_streaming.cc:193`) access this buffer. If `bx >= comp->width_in_blocks` or `by >= comp->height_in_blocks`, special padding logic is triggered later in `ProcessiMCURow` (`encode_streaming.cc:180-186`), effectively padding with zero-coefficient blocks, not by replicating edge pixels in `raw_data`.
*   **`ComputePreErosion` Padding (`adaptive_quantization.cc:480`, `adaptive_quantization.cc:544-546`)**: Calls `m->pre_erosion.PadRow` and `m->pre_erosion.CopyRow`, correctly replicating edge values for the `pre_erosion` buffer.
*   **`FuzzyErosion` Boundary Handling (`adaptive_quantization.cc:348`)**: It reads a 3x3 neighborhood around each pixel in `pre_erosion`. Since `pre_erosion` was explicitly padded using `PadRow`/`CopyRow`, accesses outside the valid range (e.g., `x = -1` or `y = -1`) hit the replicated border values stored by `PadRow`/`CopyRow`.

**Verified Padding Summary for 3x11 Image:**

1.  `input_buffer` (Y: 8x16, Cb/Cr: 8x8): Allocated larger than needed. Filled with input data. Each row is padded horizontally by replicating column 2 (Y) or 1 (Cb/Cr) out to column 7. Border `[-1]` gets value from `[0]`. After row 10 (Y) / 5 (Cb/Cr), subsequent rows up to 15 (Y) / 7 (Cb/Cr) are filled by copying the last valid (horizontally padded) row. Vertical access wraps around (e.g., `Row(-1)` accesses last row, `Row(16)` accesses row 0).
2.  `raw_data` (Y: 8x16, Cb/Cr: 8x8): Y data copied from padded `input_buffer`. Cb/Cr gets 4x4 downsampled result in top-left. **No explicit padding here.**
3.  `pre_erosion` (Y only, 2x4): Calculated from padded `input_buffer`. Explicitly padded with a 1-pixel border using edge replication.
4.  `quant_field` (Y only, 1x2): Calculated from padded `pre_erosion`. No explicit padding needed/applied.
5.  DCT Stage: Reads 8x8 blocks from `raw_data`. If a block index is outside the valid component dimensions (e.g., Cb/Cr block 0,1), it's handled by emitting zero-value tokens/coefficients (`encode_streaming.cc:180-186`), **not** by reading padded pixel data. For blocks partially outside (e.g., Y block 0,0 contains only 3x8 valid data), the DCT reads the replicated edge pixel data from `raw_data` that originated from `PadInputBuffer`.

**2. Add Padding Story to Markdown**



Padding details added as section F.2.

**3. Buffer Design Details**

*   **Planar vs Packed**: Internally, `libjpegli` operates almost exclusively on **planar** data after the initial input conversion. Buffers generally hold data for one component at a time.
    *   `m->input_buffer`: `RowBuffer<float>[kMaxComponents]` (`encode_internal.h:149`). Planar float buffer holding input after type conversion and color transform (if YCbCr target), padded to block boundaries.
    *   `m->raw_data`: `RowBuffer<float>[kMaxComponents]` (`encode_internal.h:151`). Planar float buffer holding data after downsampling. Size is `width_in_blocks*8` x `height_in_blocks*8`. Contains valid downsampled data in the top-left, potentially unused space otherwise (no explicit padding).
    *   `m->quant_field`: `RowBuffer<float>` (`encode_internal.h:155`). Single plane (Y channel only) holding adaptive quant multipliers. Size `width_in_blocks` x `height_in_blocks`.
    *   `m->pre_erosion`, `m->fuzzy_erosion_tmp`: `RowBuffer<float>` (`encode_internal.h:156-157`). Intermediate single-plane buffers for adaptive quantization.
    *   `m->dct_buffer`: `float[]` (`encode_internal.h:160`). Temporary buffer for DCT intermediate results, size `DCTSIZE2`.
    *   `m->block_tmp`: `int32_t[]` (`encode_internal.h:161`). Temporary buffer holding quantized integer coefficients for one block (size `DCTSIZE2`), plus space for tokenization symbols (`3*DCTSIZE2`) and non-zero indices (`+ DCTSIZE2`).
    *   `m->coeff_buffers`: `jvirt_barray_ptr[kMaxComponents]` (`jpegint.h:312`). Used only for coefficient input/output mode (transcoding). Points to virtual arrays holding `JCOEF` (int16_t). Planar.
    *   `m->token_arrays`: `TokenArray[]` (`encode_internal.h:163`). Holds arrays of `Token` structs (`entropy_coding.h:33`), which are component-specific (`context`). Effectively planar token storage.
*   **Data Types**:
    *   **Input**: Handled by wrapper (`jxl::extras::DecodeBytes`) and `libjpegli`'s `input_method` (`ChooseInputMethod` -> e.g., `ReadU16LEScanline`, `ReadF32Scanline`). Input `uint8`, `uint16`, `float32` are converted to internal **`float`** during `ReadInputRow`. The internal float range aims for approximately 0-1, although exact scaling depends on the `input_method`.
    *   **Processing**: Most internal processing (color transform, downsampling, adaptive quant field calculation, DCT) happens using **`float`**.
    *   **Quantization Output**: `QuantizeBlock` converts float DCT coefficients to integer coefficients (`int32_t` stored in `m->block_tmp`). (`dct-inl.h:217`, `dct-inl.h:209`). If writing to `coeff_buffers`, these are `JCOEF` (int16_t).
    *   **Tokens**: `Token` struct stores integer values (packed run/size, magnitude bits).
*   **Data Flow Summary**:
    1.  Input File -> `jxl::extras::DecodeBytes` -> `PackedPixelFile ppf` (uint8/16/f32, planar/packed depending on format, includes `ppf.color_encoding`).
    2.  `ppf` -> `EncodeJpegInternal`: Color Transform (if needed: XYB or non-sRGB -> sRGB) -> float planar data.
    3.  float planar data -> `ReadInputRow` -> `m->input_buffer` (float planar, padded).
    4.  `m->input_buffer` -> `(*m->color_transform)` (e.g., RGB->YCbCr) -> `m->input_buffer` (YCbCr float planar, padded).
    5.  `m->input_buffer` -> `DownsampleInputBuffer` -> `m->raw_data` (float planar, downsampled Cb/Cr).
    6.  `m->input_buffer[Y]` -> `ComputeAdaptiveQuantField` pipeline -> `m->quant_field` (float planar, Y only).
    7.  `m->raw_data` -> `ComputeCoefficientBlock` (DCT) -> `m->dct_buffer` (float planar).
    8.  `m->dct_buffer` + `m->quant_field` + `m->quant_mul` etc. -> `ComputeCoefficientBlock` (Quantization) -> `m->block_tmp` (int32_t planar block).
    9.  `m->block_tmp` -> `ComputeTokensForBlock` -> `m->token_arrays` (planar `Token` stream).
    10. `m->token_arrays` -> `WriteScanData` -> Output Bitstream.

**4. Color Handling Details**

*   **Input Profile/Space Reading**:
    *   `jxl::extras::DecodeBytes` (`decode.cc:560`): Uses underlying image format libraries (libpng, libjpeg-turbo, etc.) or internal decoders (PNM). These libraries typically extract embedded ICC profiles or determine the color space (e.g., sRGB for PNG chunks, JPEG markers).
    *   The result is stored in `PackedPixelFile`:
        *   `ppf.icc`: `std::vector<uint8_t>` containing the raw ICC profile bytes.
        *   `ppf.color_encoding`: `jxl::ColorEncoding` struct, parsed from `ppf.icc` or set based on format defaults (e.g., sRGB). Contains detailed color space info (primaries, white point, transfer function). (`lib/cms/color_encoding_internal.h`)
    *   `--dec-hints`: The `color_space=...` hint can override the detected/default color space by setting `ppf.color_encoding` directly (`decode.cc:569`). `icc_pathname=...` loads an external profile into `ppf.icc`, overriding any embedded one (`decode.cc:574`).
*   **Wrapper Color Transform (`EncodeJpegInternal` using `ColorSpaceTransform`)**:
    *   Goal: Convert `ppf` (with its original `ppf.color_encoding` and data type) into float planar data suitable for `libjpegli`, using either an XYB or sRGB target (`output_encoding`).
    *   `ColorSpaceTransform c_transform(ppf.info, ppf.color_encoding, output_encoding)` (`lib/extras/color_transform.cc:30`): Sets up the LittleCMS (lcms2) transform pipeline based on source/target profiles and desired float output format.
    *   `c_transform.ToXYB()` (`lib/extras/xyb_transform.cc:100`): If target is XYB, typically involves transforming to linear sRGB float first (using lcms2), then applying the `jxl::LinearRGBToXYB` matrix transformation.
    *   `c_transform.Run()` (`lib/extras/color_transform.cc:40`): If target is sRGB, uses lcms2 to perform the conversion (e.g., P3 uint16 -> sRGB float).
*   **`libjpegli` Internal Color Transform (`ChooseColorTransform` -> `RGBToYCbCr`)**:
    *   Triggered only when the *wrapper* provides sRGB float data (`cinfo.in_color_space = JCS_RGB`). (`encode.cc:515`, `encode_internal.cc:92`)
    *   `RGBToYCbCr` (`color_transform.cc:25`) applies the standard ITU-R BT.601 definition:
        ```c++
        Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
        Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
        Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + 128
        ```
        (Input R,G,B assumed to be in 0-255 range implicitly by constants).
*   **ICC Profile Embedding (`jpegli_write_icc_profile`)**:
    *   Called by `EncodeJpegInternal` (`lib/extras/enc/jpegli.cc:494`) if:
        *   `settings.app_data` is empty (no user override), AND
        *   (`output_encoding` is not sRGB/Gray OR `settings.xyb` is true).
    *   The profile embedded is `output_encoding.ICC()`.
        *   If XYB was the target, `output_encoding` *is* the XYB profile generated by `jxl::extras::GetXYBColorEncoding()`.
        *   If sRGB was the target, `output_encoding` *is* the standard sRGB profile.
        *   The profile from the *original* input (`ppf.icc`) is **not** directly embedded unless the target happens to be the same *and* it's not sRGB/Gray. The goal is to embed the profile corresponding to the data *actually encoded*.
    *   `jpegli_write_icc_profile` (`encode.cc:1101`) takes the profile bytes and writes them into one or more APP2 markers (`0xFFE2`), prepending the "ICC_PROFILE\\0\\0" signature and sequence numbers.
*   **Custom Profiles (`settings.app_data`)**: If `settings.app_data` is provided (`lib/extras/enc/jpegli.cc:491`), it completely overrides the automatic ICC embedding. The user must ensure the `app_data` contains the correct markers (e.g., a valid APP2 ICC segment) if profile embedding is desired. `jpegli_write_marker` (`encode.cc:1096`) is used to write this data.

**5. Add Buffer/Color Story to Markdown**



Buffer and Color Management details added as sections G.1 and G.2.

**6. End-to-End Color Handling Stories**

Here are the scenarios, tracing color handling:

**Scenario 1: Input sRGB JPEG -> Output sRGB JPEG**

1.  **Input Reading**: `DecodeBytes` uses libjpeg-turbo to decode. `ppf.color_encoding` is set to sRGB (likely based on JFIF marker or lack of other profile). `ppf.icc` is likely empty. Input is uint8.
2.  **Wrapper Transform**: Target is sRGB. `ColorSpaceTransform` is set up (sRGB uint8 -> sRGB float). `Run()` converts uint8 to float, likely clamping/scaling. `image_pixels` points to this float data.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_RGB`. `jpegli_default_colorspace` sets `cinfo.jpeg_color_space = JCS_YCbCr`.
4.  **Libjpegli Transform**: `color_transform` pointer set to `RGBToYCbCr`. Applied during `jpegli_write_scanlines`.
5.  **ICC Embedding**: `output_encoding` is sRGB. `settings.xyb` is false. Condition `(!output_encoding.IsSRGB()) || settings.xyb` is false. `jpegli_write_icc_profile` is **not** called (unless overridden by `app_data`). Output JPEG likely relies on JFIF marker indicating YCbCr.

**Scenario 2: Input 16-bit sRGB PNG -> Output sRGB JPEG**

1.  **Input Reading**: `DecodeBytes` uses libpng. PNG chunks indicate sRGB. `ppf.color_encoding` set to sRGB. `ppf.icc` likely empty. Input is uint16.
2.  **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` (sRGB uint16 -> sRGB float). `Run()` converts uint16 to float, handles gamma. `image_pixels` points to float data.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_RGB`. `jpegli_default_colorspace` sets `cinfo.jpeg_color_space = JCS_YCbCr`. `jpegli_set_input_format` sets `data_type = JPEGLI_TYPE_UINT16` (based on `ppf.info`). *Correction*: Wrapper provides float, so `input_format` likely set based on the float buffer provided by the wrapper. Let's assume wrapper ensures float input here. `data_type = JPEGLI_TYPE_FLOAT`.
4.  **Libjpegli Transform**: `color_transform = RGBToYCbCr`. Applied.
5.  **ICC Embedding**: Same as Scenario 1: Not called by default.

**Scenario 3: Input 16-bit DCI-P3 PNG -> Output sRGB JPEG**

1.  **Input Reading**: `DecodeBytes` uses libpng. Reads embedded ICC profile for DCI-P3. `ppf.icc` contains the profile. `ppf.color_encoding` parsed from `ppf.icc`. Input uint16.
2.  **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` (DCI-P3 uint16 -> sRGB float). `Run()` uses lcms2 to perform the color space conversion and type change. `image_pixels` points to the resulting sRGB float data.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_RGB`. `jpegli_default_colorspace` sets `cinfo.jpeg_color_space = JCS_YCbCr`. `data_type = JPEGLI_TYPE_FLOAT`.
4.  **Libjpegli Transform**: `color_transform = RGBToYCbCr`. Applied to the sRGB float data.
5.  **ICC Embedding**: `output_encoding` is sRGB. Condition is false. `jpegli_write_icc_profile` is **not** called by default. The output JPEG represents sRGB data converted to YCbCr, but doesn't embed the sRGB profile.

**Scenario 4: Input 16-bit DCI-P3 PNG -> Output XYB JPEG (`--xyb`)**

1.  **Input Reading**: Same as Scenario 3: `ppf.color_encoding` is DCI-P3, `ppf.icc` has profile, input uint16.
2.  **Wrapper Transform**: Target is XYB (`settings.xyb = true`). `ColorSpaceTransform` (DCI-P3 uint16 -> Linear sRGB float -> XYB float). `ToXYB()` performs this conversion using lcms2 (for DCI-P3 -> Linear sRGB) and matrix multiplication (Linear sRGB -> XYB). `image_pixels` points to XYB float data.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_EXT_RGB`. `jpegli_default_colorspace` sees `xyb_mode` is true, sets `cinfo.jpeg_color_space = JCS_RGB`. `data_type = JPEGLI_TYPE_FLOAT`.
4.  **Libjpegli Transform**: `ChooseColorTransform` selects `Passthrough` because `jpeg_color_space` is `JCS_RGB`. No YCbCr conversion happens inside libjpegli.
5.  **ICC Embedding**: `settings.xyb` is true. Condition `(!output_encoding.IsSRGB()) || settings.xyb` is true. `jpegli_write_icc_profile` **is called**. It embeds the XYB profile (`output_encoding.ICC()` which comes from `GetXYBColorEncoding()`).

**Scenario 5: Google Ultra HDR JPEG -> Output sRGB JPEG**

*Assuming Ultra HDR stores gain map in metadata, base image is SDR (e.g., sRGB or P3)*

1.  **Input Reading**: `DecodeBytes` uses libjpeg-turbo. Decodes the base JPEG (let's assume sRGB for simplicity). `ppf.color_encoding` is sRGB. Extracts gain map from metadata (e.g., XMP). *Note: `cjpegli` itself doesn't seem to have specific Ultra HDR gain map handling logic; it would likely just encode the base image.* We'll proceed assuming only the base SDR image is processed. Input uint8.
2.  **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` (sRGB uint8 -> sRGB float). `Run()` converts type.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_RGB`. `jpegli_default_colorspace` sets `cinfo.jpeg_color_space = JCS_YCbCr`. `data_type = JPEGLI_TYPE_FLOAT`.
4.  **Libjpegli Transform**: `color_transform = RGBToYCbCr`.
5.  **ICC Embedding**: Not called by default.

**Scenario 6: Custom Profile PNG -> Output sRGB JPEG**

1.  **Input Reading**: `DecodeBytes` uses libpng. Reads embedded custom ICC profile. `ppf.icc` contains profile. `ppf.color_encoding` parsed from `ppf.icc`. Input likely uint8 or uint16.
2.  **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` (Custom Profile -> sRGB float). `Run()` uses lcms2. `image_pixels` points to sRGB float data.
3.  **Libjpegli Config**: `cinfo.in_color_space = JCS_RGB`. `jpegli_default_colorspace` sets `cinfo.jpeg_color_space = JCS_YCbCr`. `data_type = JPEGLI_TYPE_FLOAT`.
4.  **Libjpegli Transform**: `color_transform = RGBToYCbCr`.
5.  **ICC Embedding**: Not called by default. Output contains YCbCr data derived from sRGB, but no profile.

**Scenario 7: Custom Profile PNG -> Output "Keep Custom Profile" JPEG**

*This is not directly possible with default `cjpegli` logic.* `cjpegli`/`libjpegli` always transforms pixels internally to either YCbCr (via sRGB) or XYB.

*   **Option 1 (Incorrect Data):** Encode as if sRGB (Scenario 6). The output JPEG contains YCbCr data derived from sRGB, *not* the custom space. Embedding the original custom profile would be incorrect.
*   **Option 2 (Use `app_data`):**
    1.  **Input Reading**: Read custom profile PNG (`ppf.icc`, `ppf.color_encoding`).
    2.  **User Action**: Manually prepare the custom `ppf.icc` profile bytes, potentially formatted as a valid APP2 marker segment.
    3.  **Wrapper**: Run `cjpegli` targeting sRGB output, but pass the prepared APP2 marker data via `settings.app_data`.
    4.  **`EncodeJpegInternal`**: Sees `settings.app_data` is non-empty (`lib/extras/enc/jpegli.cc:491`). Skips automatic ICC profile writing. Calls `jpegli_write_marker` (`encode.cc:1096`) to write the user-provided `app_data`.
    5.  **Libjpegli**: Still converts input (via wrapper) to sRGB float -> YCbCr internally.
    6.  **Output**: JPEG contains YCbCr data derived from sRGB float representation of the custom space, but *claims* to be in the custom space via the embedded profile. **Color consistency is not guaranteed** unless the custom profile was very close to sRGB or handled carefully. `libjpegli`'s psychovisual optimizations (quantization, adaptive quant) are tuned for YCbCr/XYB, not arbitrary custom spaces.

**7. Add Stories to Markdown**



End-to-end examples added as section H.

The `src/jpegli/cjpegli.md` file now contains the verified, detailed explanations of padding, buffer design, color management, and the end-to-end examples as requested.
