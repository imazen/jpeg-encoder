The assistant should ALWAYS update this file to reflect new files discoverd in /external/ and /jpegli/ as well as every file that exists in the /src/ and /src/jpegli/ directories. Any time refactoring changes method signatures or the organization of the project, the assistant should update this file.


# C++ Jpegli structure and inventory

The main encoder implementation in `jpegli/lib/jpegli/encode.cc` depends on the following headers (relative to the jpegli subdir)

*   **C API:** `jpegli/lib/jpegli/encode.h`
*   **Standard Libraries:** `<algorithm>`, `<cstddef>`, `<cstdint>`, `<cstring>`, `<vector>`
*   **Jpegli Base:** `jpegli/lib/base/types.h`
*   **Jpegli Common:** `jpegli/lib/jpegli/common.h`, `jpegli/lib/jpegli/common_internal.h`, `jpegli/lib/jpegli/types.h`, `jpegli/lib/jpegli/error.h`, `jpegli/lib/jpegli/memory_manager.h`, `jpegli/lib/jpegli/simd.h`
*   **Jpegli Encoding Stages:**
    *   `jpegli/lib/jpegli/input.h`
    *   `jpegli/lib/jpegli/color_transform.h`
    *   `jpegli/lib/jpegli/downsample.h`
    *   `jpegli/lib/jpegli/adaptive_quantization.h`
    *   `jpegli/lib/jpegli/quant.h`
    *   `jpegli/lib/jpegli/entropy_coding.h`
    *   `jpegli/lib/jpegli/huffman.h`
    *   `jpegli/lib/jpegli/bitstream.h`, `jpegli/lib/jpegli/bit_writer.h`
    *   `jpegli/lib/jpegli/encode_streaming.h`, `jpegli/lib/jpegli/encode_finish.h`
*   **Internal Helpers:** `jpegli/lib/jpegli/encode_internal.h`

**Relevant Files from jpegli/lib/jpegli/ C++:**

| Feature                     | Jpegli Source (`jpegli/lib/jpegli/`)        |
| :-------------------------- | :----------------------------------- |
| Quantization (Base Tables)  | `quant.cc`                           |
| Quantization (Distance)   | `quant.cc`                           |
| Quantization (Zero Bias)  | `quant.cc`                           |
| Adaptive Quantization     | `adaptive_quantization.cc/.h`        |
| Float DCT                   | `dct-inl.h`                          |
| Encoder API/Logic         | `encode.cc/.h`                       |
| Encoding Loop Integration | `encode_streaming.cc`, `encode.cc` |
| Color Transform (YCbCr)   | `color_transform.cc/.h`              |
| Color Transform (XYB)     | `color_transform.cc/.h`              |
| Input Handling              | `input.cc/.h`                        |
| Color Transform (XYB)     | `color_transform.cc/.h`              |
| Input Handling              | `input.cc/.h`                        |
| Rendering/Output            | `render.cc/.h`                       |
| Transfer Functions          | `lib/cms/transfer_functions.h`, `lib/cms/transfer_functions-inl.h` |
| Quantization (using TF)     | `quant.cc` |
*   `streaming_test.cc`
*   `transcode_api_test.cc`
*   `lib/cms/transfer_functions_test.cc`
*   Note: `test_utils.cc/.h`, `test_utils-inl.h`, `libjpeg_test_util.cc/.h`, and `test_params.h` contain common testing infrastructure and helpers.

## Jpegli Algorithmic Differences vs. Standard libjpeg-turbo

Based on analysis of the `jpegli` encoder source code and its API (`encode.h`, `encode.cc`), here are some key algorithmic differences compared to a standard `libjpeg-turbo` implementation:

1.  **Adaptive Quantization:** Jpegli implements and enables *adaptive quantization* by default (`jpegli_enable_adaptive_quantization`, `ComputeAdaptiveQuantField`). This means it analyzes local image features (like edges and textures) and adjusts the quantization strength accordingly, aiming to preserve detail where it's visually important and save bits where it's not. Standard libjpeg uses non-adaptive quantization unless specific extensions (like Trellis quantization, often slower) are enabled.
2.  **Psychovisually Tuned Quantization Tables & Distance Metric:** Jpegli uses different default quantization tables than the standard Annex K tables used by libjpeg-turbo. These tables are likely derived from psychovisual modeling (related to the Butteraugli metric). Instead of just a `quality` factor (0-100), jpegli allows setting a target *Butteraugli distance* (`jpegli_set_distance`) which provides a more perceptually uniform measure of image quality/difference. While `jpegli_set_quality` exists for compatibility, it maps to an underlying distance. Libjpeg-turbo's quality setting directly scales the standard tables. Jpegli *can* use the standard tables if requested (`jpegli_use_standard_quant_tables`).
3.  **XYB Color Space Option:** Jpegli offers the option to use the perceptually optimized XYB color space (`jpegli_set_xyb_mode`) derived from JPEG XL. This can lead to better compression efficiency compared to the traditional YCbCr color space used by default in libjpeg-turbo for color images.
4.  **Default DCT Method:** Jpegli seems to default to a floating-point DCT (`cinfo->dct_method = JDCT_FLOAT;` in `InitializeCompressParams`). While libjpeg-turbo *supports* float DCT, its default and often faster options are integer DCTs (`JDCT_ISLOW`, `JDCT_IFAST`).
5.  **Different Defaults:** Jpegli uses different defaults for parameters like progressive scan scripts (`jpegli_set_progressive_level`, `SetDefaultScanScript`) aiming for potentially better compression or perceived quality out-of-the-box.

**In summary:** While `jpegli` maintains API compatibility with `libjpeg-turbo`, its internal algorithms are significantly enhanced with techniques focused on psychovisual optimization. It leverages adaptive quantization, perceptually derived quantization tables (Butteraugli-based), and the XYB color space to achieve higher quality compression for a given file size compared to standard JPEG encoders like `libjpeg-turbo`.


## Current `src/` Structure Overview

This describes the current state of the source files based on recent analysis, without asserting completion status.

*   **`src/huffman.rs`:**
    *   Defines `HuffmanTable` struct holding lookup tables and raw table data.
    *   Provides constructors for default Annex K tables (`default_luma_dc`, etc.).
    *   Contains `new_optimized` function to generate optimized tables from frequency counts.
    *   Includes static arrays for default Annex K table definitions.

*   **`src/writer.rs`:**
    *   Defines `JfifWrite` trait for output byte writing (works in `no_std`).
    *   Defines `JfifWriter` struct for managing bitstream output, handling bit buffering and byte stuffing.
    *   Provides methods for writing standard JPEG markers (APP0, DQT, DHT, SOF, SOS, DRI).
    *   Includes methods for Huffman encoding DC and AC coefficients (`write_dc`, `write_ac_block`, `write_block`).
    *   Defines the `ZIGZAG` coefficient ordering array.

*   **`src/marker.rs`:**
    *   Defines the `Marker` enum representing JPEG markers.
    *   Contains structs for segment headers (e.g., `SOF`, `SOSHeader`).

*   **`src/error.rs`:**
    *   Defines the `EncodingError` and `JpegError` enums for error handling.

*   **`src/jpegli/mod.rs`:**
    *   Root of the Jpegli-specific module (conditional on `jpegli` feature).
    *   Declares submodules: `adaptive_quant_math`, `color_transform`, `fdct_jpegli`, `quant`, `tf`, `xyb`, `cms`, etc.

*   **`src/jpegli/adaptive_quantization.rs`:** (DELETED - Logic moved to `adaptive_quant.rs`)
*   **`src/jpegli/adaptive_quant_math.rs`:**
    *   Contains SIMD (using `wide`) and scalar mathematical helper functions ported from C++ `adaptive_quantization.cc`.
    *   Intended to be used by the main adaptive quantization implementation (e.g., `adaptive_quant.rs`).
    *   Key public (`pub(crate)`) functions and macros:
        *   `masking_sqrt(v: f32x8) -> f32x8`: SIMD masking sqrt calculation.
        *   `eval_rational_polynomial!(x, p, q)`: Macro to evaluate rational polynomials (SIMD).
        *   `fast_log2f(x: f32x8) -> f32x8`: SIMD `log2(x)` approximation.
        *   `fast_log2f_scalar(x: f32) -> f32`: Scalar `log2(x)` approximation.
        *   `fast_pow2f_scalar(x: f32) -> f32`: Scalar `2^x` approximation.
        *   `fast_pow2f(x: f32x8) -> f32x8`: SIMD `2^x` approximation.
        *   `ratio_of_derivatives_of_cubic_root_to_simple_gamma<const INVERT: bool>(v: f32x8) -> f32x8`: SIMD ratio calculation.
        *   `compute_mask(out_val: f32x8) -> f32x8`: SIMD compute mask calculation.
        *   `sort4(min0: &mut f32x8, ...)`: SIMD helper to sort 4 vectors.
        *   `update_min4(v: f32x8, min0: &mut f32x8, ...)`: SIMD helper to update 4 minimum vectors.
        *   `fast_reciprocal_nr(x: f32x8) -> f32x8`: SIMD fast reciprocal (Newton-Raphson).
        *   `compute_hf_metric_8x8(rows: &[&[f32]], x_start: usize) -> f32`: Computes HF metric sum for an 8x8 block using SIMD.
        *   `compute_gamma_sum_8x8(rows: &[&[f32]], x_start: usize) -> f32`: Computes gamma-related sum for an 8x8 block using SIMD.
        *   `compute_diff_buffer_row(...)`: Computes one row of the pre-erosion difference buffer using SIMD.
        *   `compute_fuzzy_erosion_row(...)`: Computes one row of the fuzzy erosion temporary buffer using SIMD.
        *   `scalar_ratio_of_derivatives<const INVERT: bool>(v_scalar: f32) -> f32`: Scalar version of ratio calculation.
        *   `compute_mask_scalar(out_val: f32) -> f32`: Scalar version of mask computation.
        *   `scalar_masking_sqrt(v: f32) -> f32`: Scalar version of masking sqrt.
*   **`src/jpegli/config.rs`:**
    *   **Structs:**
        *   `ComputedEncodeConfig`
        *   `ComponentDimensions`
        *   `ComputedConfigDimensions`
        *   `AppSegment`
        *   `EncodeOptions`
        *   `InputImageInfo`
    *   **Functions:**
        *   `validate(&self) -> Result<(), EncodingError>` (impl for `ComputedEncodeConfig`)
        *   `new(segment_nr: u8, data: Vec<u8>) -> Result<Self, EncodingError>` (impl for `AppSegment`)
        *   `default() -> Self` (impl for `EncodeOptions`)
        *   `compute(&self) -> Result<ComputedEncodeConfig, EncodingError>` (impl for `EncodeOptions`)
        *   `ceil_div(value: usize, div: usize) -> usize`
        *   `new(width: usize, height: usize, factor: (u8, u8), max_factor: (u8, u8)) -> ComponentDimensions` (impl for `ComponentDimensions`)
        *   `from_component_settings(width: usize, height: usize, components_settings: &[JpegliComponentSettings]) -> Vec<ComponentDimensions>` (impl for `ComponentDimensions`)
        *   `new(cinfo: &ComputedEncodeConfig, image_width: usize, image_height: usize) -> Result<Self, EncodingError>` (impl for `ComputedConfigDimensions`)

*   **`src/jpegli/structs.rs`:**
    *   **Enums:**
        *   `Subsampling`
        *   `JpegColorSpace`
        *   `SimplifiedTransferCharacteristics`
        *   `TransferCharacteristics`
    *   **Structs:**
        *   `JpegliComponentInfo`
        *   `JpegliComponentSettings`
        *   `RowBufferInfo`
        *   `RowBufferRef`
        *   `OwnedRowBuffer`
    *   **Traits:**
        *   `RowBuffer<T: Copy + SimdWidth + Sized>`
    *   **Functions:**
        *   `from_str(value: &str) -> Option<Self>` (impl for `Subsampling`)
        *   `to_str(&self) -> &'static str` (impl for `Subsampling`)
        *   `to_luma_h_v_samp_factor(&self) -> (u8, u8)` (impl for `Subsampling`)
        *   `is_yuv420(&self) -> bool` (impl for `Subsampling`)
        *   `to_sampling_factor(&self) -> crate::SamplingFactor` (impl for `Subsampling`)
        *   `from_sampling_factor(value: crate::SamplingFactor) -> Option<Self>` (impl for `Subsampling`)
        *   `to_int(&self) -> u8` (impl for `SimplifiedTransferCharacteristics`)
        *   `from_int(value: i32) -> Option<Self>` (impl for `SimplifiedTransferCharacteristics`)
        *   `try_from(value: i32) -> Result<Self, Self::Error>` (impl for `TryFrom<i32>` for `JpegColorSpace`)
        *   `from(value: crate::old_encoder::OutputJpegColorType) -> Self` (impl for `From<crate::old_encoder::OutputJpegColorType>` for `JpegColorSpace`)
        *   `from_i32(value: i32) -> Option<Self>` (impl for `JpegColorSpace`)
        *   `to_output_color_type(&self) -> crate::old_encoder::OutputJpegColorType` (impl for `JpegColorSpace`)
        *   `get_num_components(&self) -> usize` (impl for `JpegColorSpace`)
        *   `info(&self) -> &RowBufferInfo` (impl for `RowBuffer`)
        *   `get_buffer(&self) -> &[T]` (impl for `RowBuffer`)
        *   `get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T])` (impl for `RowBuffer`)
        *   `get_buffer_mut(&mut self) -> &mut [T]` (impl for `RowBuffer`)
        *   `get_info_and_buffer(&self) -> (&RowBufferInfo, &[T])` (impl for `RowBuffer`)
        *   `is_aligned(&self) -> bool` (impl for `RowBuffer`)
        *   `is_optimally_aligned(&self) -> bool` (impl for `RowBuffer`)
        *   `region_mut_core(&mut self, x: usize, y: usize, window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<(RowBufferInfo, &mut [T])>` (impl for `RowBuffer`)
        *   `region_mut_shrink_padding_core(&mut self, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<(RowBufferInfo, &mut [T])>` (impl for `RowBuffer`)
        *   `split_at_mut_core(&mut self, row: usize) -> (RowBufferInfo, &mut [T], RowBufferInfo, &mut [T])` (impl for `RowBuffer`)
        *   `get_window_row(&self, y: usize) -> Option<&[T]>` (impl for `RowBuffer`)
        *   `get_window_row_mut(&mut self, y: usize) -> Option<&mut [T]>` (impl for `RowBuffer`)
        *   `get_padded_row(&self, y: usize) -> Option<&[T]>` (impl for `RowBuffer`)
        *   `get_padded_row_mut(&mut self, y: usize) -> Option<&mut [T]>` (impl for `RowBuffer`)
        *   `get_padded_row_window_relative(&self, windowed_y: isize) -> Option<&[T]>` (impl for `RowBuffer`)
        *   `get_padded_row_window_relative_mut(&mut self, windowed_y: isize) -> Option<&mut [T]>` (impl for `RowBuffer`)
        *   `row_mut_old(&mut self, windowed_y: isize) -> Option<&mut [T]>` (impl for `RowBuffer`) (Deprecated)
        *   `row_old(&self, windowed_y: isize) -> Option<&[T]>` (impl for `RowBuffer`) (Deprecated)
        *   `fill_full(&mut self, value: T, x: usize, y: usize, width: usize, height: usize)` (impl for `RowBuffer`)
        *   `fill_window(&mut self, value: T, x: usize, y: usize, width: usize, height: usize)` (impl for `RowBuffer`)
        *   `get_window_pixel(&self, x: usize, y: usize) -> Option<&T>` (impl for `RowBuffer`)
        *   `get_window_pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut T>` (impl for `RowBuffer`)
        *   `get_padded_pixel(&self, x: usize, y: usize) -> Option<&T>` (impl for `RowBuffer`)
        *   `get_padded_pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut T>` (impl for `RowBuffer`)
        *   `copy_row_section(&mut self, from: usize, to: usize, x: usize, width: usize)` (impl for `RowBuffer`)
        *   `copy_window_row(&mut self, from: usize, to: usize)` (impl for `RowBuffer`)
        *   `copy_full_row(&mut self, from: usize, to: usize)` (impl for `RowBuffer`)
        *   `has_padding(&self) -> bool` (impl for `RowBuffer`)
        *   `pad_using_edges(&mut self)` (impl for `RowBuffer`)
        *   `pad_row_full_width(&mut self, row: usize)` (impl for `RowBuffer`)
        *   `pad_row_width_old(&mut self, windowed_y: usize, copy_to_right: usize, border: usize)` (impl for `RowBuffer`)
        *   `copy_row_window_relative(&mut self, from: isize, to: isize)` (impl for `RowBuffer`)
        *   `xsize(&self) -> usize` (impl for `RowBuffer`) (Deprecated)
        *   `ysize(&self) -> usize` (impl for `RowBuffer`) (Deprecated)
        *   `get_padded_row_and_two_neighbors_mut(&mut self, windowed_y_center: isize) -> Option<(&mut [T], &mut [T], &mut [T])>` (impl for `RowBuffer`)
        *   `get_padded_row_and_two_neighbors(&self, windowed_y_center: isize) -> Option<(&[T], &[T], &[T])>` (impl for `RowBuffer`)
        *   `get_padded_row_and_neighbors_mut(&mut self, windowed_y_center: isize, vertical_padding: usize) -> Option<Vec<&mut [T]>>` (impl for `RowBuffer`)
        *   `get_padded_row_and_neighbors(&self, windowed_y_center: isize, vertical_padding: usize) -> Option<Vec<&[T]>>` (impl for `RowBuffer`)
        *   `is_valid(&self) -> bool` (impl for `RowBufferInfo`)
        *   `is_window_empty(&self) -> bool` (impl for `RowBufferInfo`)
        *   `is_empty(&self) -> bool` (impl for `RowBufferInfo`)
        *   `has_padding(&self) -> bool` (impl for `RowBufferInfo`)
        *   `padding_is(&self, value: usize) -> bool` (impl for `RowBufferInfo`)
        *   `min_buffer_size(&self) -> usize` (impl for `RowBufferInfo`)
        *   `info(&self) -> &RowBufferInfo` (impl for `RowBufferRef`)
        *   `get_buffer(&self) -> &[T]` (impl for `RowBufferRef`)
        *   `get_buffer_mut(&mut self) -> &mut [T]` (impl for `RowBufferRef`)
        *   `get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T])` (impl for `RowBufferRef`)
        *   `new(info: RowBufferInfo, data: &'a mut [T]) -> Self` (impl for `RowBufferRef`)
        *   `region_mut(&mut self, x: usize, y: usize, window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<RowBufferRef<T>>` (impl for `RowBufferRef`)
        *   `region_mut_shrink_padding(&mut self, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize) -> Option<RowBufferRef<T>>` (impl for `RowBufferRef`)
        *   `info(&self) -> &RowBufferInfo` (impl for `OwnedRowBuffer`)
        *   `get_buffer(&self) -> &[T]` (impl for `OwnedRowBuffer`)
        *   `get_buffer_mut(&mut self) -> &mut [T]` (impl for `OwnedRowBuffer`)
        *   `get_info_and_buffer_mut(&mut self) -> (&RowBufferInfo, &mut [T])` (impl for `OwnedRowBuffer`)
        *   `new(window_width: usize, window_height: usize, padding: usize, fill: T) -> Self` (impl for `OwnedRowBuffer`)
        *   `new_with_padding(window_width: usize, window_height: usize, padding_left: usize, padding_right: usize, padding_top: usize, padding_bottom: usize, fill: T) -> Self` (impl for `OwnedRowBuffer`)
        *   `as_ref(&mut self) -> RowBufferRef<'_, T>` (impl for `OwnedRowBuffer`)

*   **`src/jpegli/progressive_scan.rs`:**
    *   **Structs:**
        *   `RefToken`
        *   `ScanTokenInfo`
        *   `ScanConfiguration`
        *   `ProgressiveScan`
        *   `JpegScanInfo`
    *   **Functions:**
        *   `new(symbol: u8, refbits: u8) -> Self` (impl for `RefToken`)
        *   `new() -> Self` (impl for `ScanTokenInfo`)
        *   `new() -> Self` (impl for `ScanConfiguration`)
        *   `new(ss: i32, se: i32, ah: i32, al: i32, interleaved: bool) -> Self` (impl for `ProgressiveScan`)
        *   `new() -> Self` (impl for `JpegScanInfo`)
        *   `set_default_scan_script(config: &ComputedEncodeConfig) -> Result<Vec<JpegScanInfo>, String>`
        *   `validate_scan_script(scan_script: &[JpegScanInfo], config: &ComputedEncodeConfig, progressive_mode: bool) -> Result<(), String>`
        *   `div_ceil(a: usize, b: usize) -> usize`
        *   `is_progressive_mode(scan_script: &[JpegScanInfo]) -> bool`
        *   `process_compression_params_scans(config: &ComputedEncodeConfig, image_width: usize, image_height: usize, components: &[super::structs::JpegliComponentInfo]) -> Result<ScanConfiguration, String>`
        *   `setup_scan_token_info(scan_config: &mut ScanConfiguration, config: &ComputedEncodeConfig, image_width: usize, image_height: usize, components: &[super::structs::JpegliComponentInfo]) -> Result<(), String>`

*   **`src/jpegli/entropy_coding.rs`:**
    *   **Functions:**
        *   `bit_width(x: u32) -> usize`
        *   `build_histograms(coefficients: &[Vec<[i16; 64]>], components: &[JpegliComponentSettings]) -> (Vec<[u32; HUFFMAN_ALPHABET_SIZE]>, Vec<[u32; HUFFMAN_ALPHABET_SIZE]>`
        *   `optimize_huffman_tables(dc_histograms: &[[u32; HUFFMAN_ALPHABET_SIZE]], ac_histograms: &[[u32; HUFFMAN_ALPHABET_SIZE]]) -> Vec<(HuffmanTable, HuffmanTable)>`
        *   `optimize_entropy(components: &[JpegliComponentSettings], coefficients: &[Vec<[i16; 64]>]) -> Result<Vec<(HuffmanTable, HuffmanTable)>, EncodingError>`

*   **`src/jpegli/adaptive_quant.rs`:**
    *   **Structs:**
        *   `AdaptiveQuantState`
    *   **Functions:**
        *   `compute_adaptive_quant_field<T: RowBuffer<f32>>(luma_plane_padded_input: &mut T, state: &mut AdaptiveQuantState, quantizer: &JpegliQuantizerState, config: &ComputedEncodeConfig, dims: &ComputedConfigDimensions)`

*   **`src/jpegli/adaptive_quant_math.rs`:**
    *   **Functions:**
        *   `masking_sqrt(v: f32x8) -> f32x8`
        *   `fast_log2f(x: f32x8) -> f32x8`
        *   `fast_log2f_scalar(x: f32) -> f32`
        *   `fast_pow2f_scalar(x: f32) -> f32`
        *   `fast_pow2f(x: f32x8) -> f32x8`
        *   `ratio_of_derivatives_of_cubic_root_to_simple_gamma<const INVERT: bool>(v: f32x8) -> f32x8`
        *   `compute_mask(out_val: f32x8) -> f32x8`
        *   `sort4(min0: &mut f32x8, min1: &mut f32x8, min2: &mut f32x8, min3: &mut f32x8)`
        *   `update_min4(v: f32x8, min0: &mut f32x8, min1: &mut f32x8, min2: &mut f32x8, min3: &mut f32x8)`
        *   `fast_reciprocal_nr(x: f32x8) -> f32x8`
        *   `compute_hf_metric_8x8(rows: &[&[f32]], x_start: usize) -> f32`
        *   `compute_gamma_sum_8x8(rows: &[&[f32]], x_start: usize) -> f32`
        *   `compute_diff_buffer_row(row_t: &[f32], row_m: &[f32], row_b: &[f32], xsize: usize, diff_out: &mut [f32], prev_diff_out: Option<&[f32]>`
        *   `scalar_ratio_of_derivatives<const INVERT: bool>(v_scalar: f32) -> f32`
        *   `compute_mask_scalar(out_val: f32) -> f32`
        *   `scalar_masking_sqrt(v: f32) -> f32`
        *   `compute_fuzzy_erosion_row(row_t: &[f32], row_m: &[f32], row_b: &[f32], unpadded_elements: usize, out_unpadded: &mut [f32])`
        *   `compute_pre_erosion<I, E>(input: &I, xsize: usize, y0: usize, ylen: usize, diff_buffer: &mut [f32], pre_erosion: &mut E)`
        *   `fuzzy_erosion<E, T, M>(pre_erosion: &E, yb0: usize, yblen: usize, tmp: &mut T, aq_map: &mut M)`
        *   `per_block_modulations<I, M>(y_quant_01: f32, input: &mut I, yb0: usize, yblen: usize, aq_map: &mut M)`

*   **`src/jpegli/cms.rs`:**
    *   **Structs:**
        *   `ColorEncodingInternal`
        *   `ColorProfile`
        *   `JxlCms`
        *   `HlgOotf` (within `hlg` module)
    *   **Enums:**
        *   `TfType`
    *   **Functions:**
        *   `new(icc_data: Vec<u8>) -> Result<Self, EncodingError>` (impl for `ColorProfile`)
        *   `srgb() -> Result<Self, EncodingError>` (impl for `ColorProfile`)
        *   `linear_srgb() -> Result<Self, EncodingError>` (impl for `ColorProfile`)
        *   `gray_gamma22() -> Result<Self, EncodingError>` (impl for `ColorProfile`)
        *   `internal(&mut self) -> Result<&ColorEncodingInternal, EncodingError>` (impl for `ColorProfile`)
        *   `icc(&self) -> &[u8]` (impl for `ColorProfile`)
        *   `new(input_profile: &ColorProfile, output_profile: &ColorProfile, intensity_target: f32) -> Result<Self, EncodingError>` (impl for `JxlCms`)
        *   `run_transform(&self, input_slice: &[f32], output_slice: &mut [f32], num_pixels: usize) -> Result<(), EncodingError>` (impl for `JxlCms`)
        *   `set_fields_from_icc(icc_data: &[u8]) -> Result<ColorEncodingInternal, EncodingError>`
        *   `cms_init(input_profile: &ColorProfile, output_profile: &ColorProfile, intensity_target: f32) -> Result<Box<JxlCms>, EncodingError>`
        *   `cms_run(cms_state: &JxlCms, input_buffer: &[f32], output_buffer: &mut [f32], num_pixels: usize) -> Result<(), EncodingError>`
        *   `from_scene_light(display_luminance: f32) -> Self` (impl for `HlgOotf`)
        *   `to_scene_light(display_luminance: f32) -> Self` (impl for `HlgOotf`)
        *   `apply(&self, r: &mut [f32], g: &mut [f32], b: &mut [f32], num_pixels: usize)` (impl for `HlgOotf`)
        *   `display_from_encoded(encoded: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32` (within `hlg` module)
        *   `encoded_from_display(display_linear: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32` (within `hlg` module)
        *   `apply_hlg_ootf(...) -> ()` (within `hlg` module) (Unimplemented)
        *   `display_from_encoded(encoded: f32, intensity_target: f32) -> f32` (within `pq` module)
        *   `encoded_from_display(display_relative: f32, intensity_target: f32) -> f32` (within `pq` module)
        *   `display_from_encoded(encoded: f32) -> f32` (within `srgb` module)
        *   `encoded_from_display(display: f32) -> f32` (within `srgb` module)

*   **`src/jpegli/color_transform.rs`:**
    *   **Functions:**
        *   `linear_rgb_to_ycbcr(planes: &mut [Vec<f32>], num_pixels: usize)`
        *   `ycbcr_to_linear_rgb(planes: &mut [Vec<f32>], num_pixels: usize)`
        *   `cmyk_to_ycck(planes: &mut [Vec<f32>], num_pixels: usize)`
        *   `ycck_to_cmyk(planes: &mut [Vec<f32>], num_pixels: usize)`
        *   `grayscale_to_rgb(planes: &mut [Vec<f32>], num_pixels: usize)`
        *   `rgb_to_ycbcr_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32)`
        *   `rgb_to_ycbcr_planes(r_plane: &mut Vec<f32>, g_plane: &mut Vec<f32>, b_plane: &mut Vec<f32>, num_pixels: usize)`
        *   `cmyk_to_ycck_pixel(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32, f32)`
        *   `cmyk_to_ycck_planes(c_plane: &mut Vec<f32>, m_plane: &mut Vec<f32>, y_plane: &mut Vec<f32>, k_plane: &mut Vec<f32>, num_pixels: usize)`

*   **`src/jpegli/fdct_jpegli.rs`:**
    *   **Structs:**
        *   `DCT1DImpl<const N: usize>`
    *   **Traits:**
        *   `DCT1DImplTrait`
    *   **Functions:**
        *   `compute(mem: &mut [f32])` (impl for `DCT1DImpl<1>`, `DCT1DImpl<2>`, `DCT1DImpl<8>`, `DCT1DImpl<4>`)
        *   `transpose_8x8_block(input: &[f32; 64], output: &mut [f32; 64])`
        *   `add_reverse<const N_HALF: usize>(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32])`
        *   `sub_reverse<const N_HALF: usize>(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32])`
        *   `multiply<const N_HALF: usize>(coeff_second_half: &mut [f32], multipliers: &[f32])`
        *   `b<const N: usize>(coeff: &mut [f32])`
        *   `inverse_even_odd<const N: usize>(a_in: &[f32], a_out: &mut [f32])`
        *   `dct_1d(pixels: &[f32], output: &mut [f32])`
        *   `forward_dct_float(pixels: &[f32; 64], coefficients: &mut [f32; 64], scratch_space: &mut [f32; 64])`

*   **`src/jpegli/quant.rs`:**
    *   **Structs:**
        *   `JpegliQuantizerState`
    *   **Enums:**
        *   `QuantPass`
    *   **Functions:**
        *   `distance_to_scale(distance: f32, k: usize) -> f32`
        *   `scale_to_distance(scale: f32, k: usize) -> f32`
        *   `quality_to_distance(quality: u8) -> f32`
        *   `distance_to_linear_quality(distance: f32) -> f32`
        *   `quant_vals_to_distance(raw_quant_tables: &[Option<[u16; 64]>; 4], num_components: usize, comp_info: &[JpegliComponentSettings], cicp_transfer_function: SimplifiedTransferCharacteristics, force_8bit_dct_baseline: bool) -> f32`
        *   `set_quant_matrices(params: &ComputedEncodeConfig) -> Result<[Option<[u16; 64]>; 4], &'static str>`
        *   `init_quantizer(raw_quant_tables: &[Option<[u16; 64]>; 4], params: &ComputedEncodeConfig, pass: QuantPass) -> Result<([[f32; 64]; MAX_COMPONENTS], [[f32; 64]; MAX_COMPONENTS], [[f32; 64]; MAX_COMPONENTS]), &'static str>`
        *   `new(params: &ComputedEncodeConfig) -> Result<Self, &'static str>` (impl for `JpegliQuantizerState`)

*   **`src/jpegli/tf.rs`:**
    *   **Enums:**
        *   `ExtraTF`
    *   **Functions:**
        *   `get_extra_tf(tf: TfType, _channels: u32, _inverse: bool) -> ExtraTF`
        *   `display_from_encoded(encoded: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32` (within `hlg` module)
        *   `encoded_from_display(display_linear: f32, _intensity_target: f32, _luminances: Option<[f32; 3]>) -> f32` (within `hlg` module)
        *   `from_scene_light(display_luminance: f32) -> Self` (within `hlg` module impl for `HlgOotf`)
        *   `to_scene_light(display_luminance: f32) -> Self` (within `hlg` module impl for `HlgOotf`)
        *   `apply(&self, r: &mut [f32], g: &mut [f32], b: &mut [f32], num_pixels: usize)` (within `hlg` module impl for `HlgOotf`)
        *   `apply_hlg_ootf(...) -> ()` (within `hlg` module) (Unimplemented)
        *   `display_from_encoded(encoded: f32, intensity_target: f32) -> f32` (within `pq` module)
        *   `encoded_from_display(display_relative: f32, intensity_target: f32) -> f32` (within `pq` module)
        *   `display_from_encoded(encoded: f32) -> f32` (within `srgb` module)
        *   `encoded_from_display(display: f32) -> f32` (within `srgb` module)
        *   `before_transform(tf: ExtraTF, intensity_target: f32, input_buf: &mut [f32]) -> Result<(), EncodingError>`
        *   `after_transform(tf: ExtraTF, intensity_target: f32, buffer: &mut [f32]) -> Result<(), EncodingError>`

*   **`src/jpegli/xyb_transform.rs`:**
    *   **Functions:**
        *   `opsin_absorbance_simd(r: f32x8, g: f32x8, b: f32x8, premul_absorb: &[f32x8; 9], bias: f32x8) -> [f32x8; 3]`
        *   `store_xyb_simd(r: f32x8, g: f32x8, b: f32x8, valx: &mut f32x8, valy: &mut f32x8, valz: &mut f32x8)`
        *   `linear_rgb_to_xyb_simd(r: f32x8, g: f32x8, b: f32x8, premul_absorb: &[f32x8; 12], valx: &mut f32x8, valy: &mut f32x8, valz: &mut f32x8)`
        *   `linear_rgb_row_to_xyb(row0: &mut [f32], row1: &mut [f32], row2: &mut [f32], premul_absorb: &[f32x8; 12], xsize: usize, intensity_target: f32)`
        *   `compute_premul_absorb(intensity_target: f32, premul_absorb: &mut [f32x8; 12])`
        *   `create_premul_absorb(intensity_target: f32) -> [f32x8; 12]`
        *   `scale_xyb_row(row0: &mut [f32], row1: &mut [f32], row2: &mut [f32], xsize: usize)`
