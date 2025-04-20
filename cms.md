
## Current Implementation Pipeline (`encoder.rs::encode_image_internal`) (ALL OF THIS MAY BE OUT OF DATE)

The core color processing pipeline within `encode_image_internal` is as follows:

1.  **Input Conversion:** The input image data (`u8`) is converted into planar `f32` buffers (one per channel) in the range [0.0, 1.0] using `image_to_f32_planes`.
2.  **CMS Initialization:** If not already done, `cms::cms_init` is called to create a `JxlCms` state. This parses the input ICC profile (defaulting to sRGB if none provided) and the target internal profile (`linear_srgb` by default), determines necessary pre/post-processing steps (`ExtraTF`), and sets up the LCMS transform if needed (`skip_lcms` is false). For XYB mode, `xyb::compute_premul_absorb` is also called.
3.  **CMS Transform:**
    *   Input `f32` planes are interleaved into a single buffer.
    *   `cms::cms_run` is called, which performs:
        *   `tf::before_transform`: Applies inverse transfer function (e.g., sRGB gamma -> linear) if `preprocess` is set.
        *   CMYK adjustment (if `channels_src == 4`).
        *   `lcms2::Transform::transform_pixels`: Executes the main LCMS transform (if `!skip_lcms`) to convert to the internal profile space (e.g., linear sRGB).
        *   `tf::after_transform`: Applies forward transfer function if `postprocess` is set (currently less common as the internal target is linear).
    *   The resulting interleaved buffer is de-interleaved back into `f32` planes, now representing the image in the internal color space (e.g., linear sRGB).
4.  **XYB Conversion (Optional):** If `xyb_mode` is enabled:
    *   `xyb::linear_rgb_row_to_xyb` converts the first three (Linear R, G, B) planes into intermediate XYB.
    *   `xyb::scale_xyb_row` applies affine scaling to the XYB planes.
    *   The number of active planes is typically reduced to 3 (X, Y, B).
5.  **Standard Color Space Conversion (Optional):** If `xyb_mode` is *not* enabled:
    *   `color_transform::linear_rgb_to_ycbcr`: Converts Linear RGB planes to YCbCr for standard JPEG output.
    *   `color_transform::cmyk_to_ycck`: Converts CMYK planes (after CMS to linear RGB internally) to YCCK.
    *   Grayscale is handled (planes truncated to 1).
6.  **DCT Input Preparation:**
    *   Planes are processed into 8x8 blocks.
    *   Values are scaled from [0.0, 1.0] to [-128.0, 127.0].
7.  **DCT:**
    *   If `use_float_dct` is true, `fdct::forward_dct_float` is used.
    *   Otherwise, the integer DCT `fdct::fdct` is used (after converting the f32 block to i16).
8.  **Quantization:**
    *   If `use_float_dct` is true, `quantization::quantize_float_block` is used.
    *   Otherwise, `quantization::quantize_block` is used.
    *   Both paths incorporate Jpegli-style adaptive quantization thresholding using precomputed zero-bias tables if `use_adaptive_quantization` is enabled and `adapt_quant_field` is available.
9.  **Entropy Coding:**
    *   Huffman tables are potentially optimized if `optimize_huffman_table` is true.
    *   `encode_interleaved_from_blocks` handles the actual writing of DCT coefficients using Huffman coding, supporting restart markers.

## Deprecated Transformation Paths Section

*(The previous "Transformation Paths" section described a more hypothetical flow based on the C++ code. The section above, "Current Implementation Pipeline", reflects the actual Rust implementation.)*

## Implementation Notes (`jpeg-encoder/src/cms.rs`)

*   The `ColorProfile` struct stores the raw ICC bytes and optionally the parsed `ColorEncodingInternal`. Factory functions like `srgb()`, `linear_srgb()`, `gray_gamma22()` provide standard profiles.
*   `set_fields_from_icc` uses `lcms2` functions (`Profile::new_icc`, `profile.color_space()`, `profile.read_tag`, etc.) to populate `ColorEncodingInternal`. Transfer function detection currently relies on `ToneCurve::estimated_gamma()` and `is_linear()`, with limited support for parametric curves (PQ/HLG detection from tags is TODO).
*   The `JxlCms` struct holds the optional `lcms2::Transform`, parameters (`intensity_target`, `skip_lcms`, `preprocess`, `postprocess`), channel counts, and per-thread temporary buffers (`src_storage`, `dst_storage`) managed by `Mutex`.
*   `JxlCms::new` sets up the transform based on comparing input and output `ColorEncodingInternal` data. HLG OOTF logic is present but calculation of luminances and application points are marked as TODO.
*   `JxlCms::run_transform` uses the pre-allocated thread-local buffer from `src_storage` for preprocessing, calls the LCMS transform, and then performs postprocessing in-place on the output buffer. It handles the necessary interleaving/de-interleaving around the core LCMS call.
*   XYB conversion (`xyb.rs`) and standard YCbCr/YCCK conversion (`color_transform.rs`) are handled *outside* of `cms.rs`, operating on the `f32` planes *after* the `cms::cms_run` step in `encoder.rs`.

## Current Limitations / TODOs

*   **HLG OOTF:** Logic exists in `JxlCms::new` to detect when OOTF *should* be applied, but the calculation of necessary luminances from profile primaries and the actual application points within `cms_run` (or `tf.rs`) are not fully implemented.
*   **Parametric Curves:** Detection of sRGB/PQ/HLG based on `SigParametricCurveType` tags in `set_fields_from_icc` is not yet implemented.
*   **Adaptive Quantization:** While the thresholding logic exists in quantization, the `compute_adaptive_quant_field` function itself needs review to ensure it operates on the correct input color space (likely requires running *before* CMS or having access to original gamma-corrected data). The block index calculation also needs verification in the new flow.
*   **Progressive/Sequential:** The encoding functions `encode_image_progressive` and `encode_image_sequential` need to be updated to work with the new block generation pipeline based on `f32` planes.
*   **Error Handling:** More specific error types and checks for channel mismatches or unsupported profile features could be added.
'''
))