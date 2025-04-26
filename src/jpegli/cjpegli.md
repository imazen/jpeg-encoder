# cjpegli Execution Flow and Settings Documentation

This document details the execution flow of the `cjpegli` command-line tool, focusing on how command-line flags translate into settings and which algorithms are engaged during the JPEG encoding process powered by the `libjpegli` library.

## 1. Entry Point (`main` in `tools/cjpegli.cc`)

- The `main` function simply calls `jpegxl::tools::CJpegliMain`.

## 2. Main Function (`CJpegliMain` in `tools/cjpegli.cc`)

1.  **Argument Parsing**:
    *   An `Args` struct holds parsed arguments and `jxl::extras::JpegSettings`.
    *   `CommandLineParser` is used to define and parse flags. Key flags and their corresponding `Args` / `JpegSettings` members:
        *   `INPUT` (positional): `args.file_in`
        *   `OUTPUT` (positional): `args.file_out`
        *   `--disable_output`: `args.disable_output` (bool)
        *   `--dec-hints key=value`: Populates `args.color_hints_proxy` (used during input decoding).
        *   `-d`, `--distance <float>`: `args.settings.distance` (Default: 1.0, corresponds roughly to Q90). Mutually exclusive with quality/target_size.
        *   `-q`, `--quality <int>`: `args.quality` (Default: 90). Mapped to `args.settings.quality` if used. Mutually exclusive with distance/target_size.
        *   `--chroma_subsampling <string>`: `args.settings.chroma_subsampling` ("444", "440", "422", "420").
        *   `-p`, `--progressive_level <int>`: `args.settings.progressive_level` (0-2, Default: 2).
        *   `--xyb`: `args.settings.xyb` (bool, Default: false).
        *   `--std_quant`: `args.settings.use_std_quant_tables` (bool, Default: false).
        *   `--noadaptive_quantization`: `args.settings.use_adaptive_quantization` (bool, set to `false`, Default: true).
        *   `--fixed_code`: `args.settings.optimize_coding` (bool, set to `false`, Default: true). Requires `-p 0`.
        *   `--target_size <uint>`: `args.settings.target_size` (Default: 0). Mutually exclusive with distance/quality.
        *   `--num_reps <uint>`: `args.num_reps` (Default: 1).
        *   `--quiet`: `args.quiet` (bool).
        *   `-v`, `--verbose`: `args.verbose` (bool).
    *   `ValidateArgs`: Checks for invalid flag values or combinations.
    *   `SetDistance`: Ensures only one of distance, quality, or target_size is effectively set. If quality was passed, `settings.quality` is populated. Otherwise, `settings.distance` is used implicitly or explicitly.

2.  **Input Loading**:
    *   `ReadFile`: Reads the input file specified by `args.file_in`.
    *   `jxl::extras::DecodeBytes`: Decodes the input byte stream into a `jxl::extras::PackedPixelFile ppf`.
        *   Uses format detection (PNG, JPEG, GIF, PNM, etc.).
        *   Applies decoding hints from `args.color_hints_proxy`.
        *   The `ppf` contains image dimensions (`xsize`, `ysize`), pixel data (`ppf.frames[0].color`), bit depth (`ppf.info.bits_per_sample`), color encoding (`ppf.info.color_encoding`), and metadata.

3.  **Encoding Loop**:
    *   Iterates `args.num_reps` times.
    *   Calls `jxl::extras::EncodeJpeg(ppf, args.settings, nullptr, &jpeg_bytes)`. This is the primary call into the encoding library wrapper.

4.  **Output**:
    *   `WriteFile`: Writes the resulting `jpeg_bytes` to `args.file_out` unless `--disable_output` was specified.

## Summary Table: Flags -> Settings -> libjpegli Functions

| cjpegli Flag                | `JpegSettings` Member        | Key `libjpegli` Function(s) Affected        | Default Behavior                                   |
| :-------------------------- | :--------------------------- | :------------------------------------------ | :------------------------------------------------- |
| `-q <N>` / `-d <N>`         | `quality` / `distance`       | `jpegli_set_quality`/`jpegli_set_distance` | `distance = 1.0` (implicit Q90)                    |
| `--chroma_subsampling <S>`  | `chroma_subsampling`         | `jpegli_set_chroma_subsampling`             | Auto-selected based on distance/quality          |
| `-p <N>`                    | `progressive_level`        | `jpegli_set_progressive_level`            | `2` (multi-scan progressive)                       |
| `--xyb`                     | `xyb`                        | `jpegli_set_xyb_mode`, Color Transform      | `false` (YCbCr color space used internally)        |
| `--std_quant`               | `use_std_quant_tables`     | `jpegli_use_standard_quant_tables`        | `false` (psychovisually derived tables)            |
| `--noadaptive_quantization` | `use_adaptive_quantization`| `jpegli_set_adaptive_quantization`        | `true` (adaptive quantization enabled)             |
| `--fixed_code`              | `optimize_coding`          | `jpegli_set_optimize_coding`              | `true` (Huffman tables optimized)                  |
| (Input Bit Depth)         | (n/a - from `ppf.info`)    | `jpegli_set_input_format`                 | Input converted to 8-bit internally by libjpegli |
| (Input Color Profile)     | (n/a - from `ppf.info`)    | `ColorSpaceTransform`, `jpegli_write_icc_profile` | sRGB assumed if no profile; Profile embedded if needed |



## 3. Encoding Wrapper (`jxl::extras::EncodeJpeg` in `lib/extras/enc/jpegli.cc`)

This function acts as a bridge between the high-level `PackedPixelFile` and the C-style `libjpegli` API.

1.  **Input Validation (`VerifyInput`)**: Checks `ppf` compatibility (1 frame, supported types/channels).
2.  **Target Size / Quality Mode Handling**:
    *   **Target Size (`settings.target_size > 0`)**: Calls `EncodeJpegToTargetSize`. This performs a binary search over the `distance` setting, repeatedly calling `EncodeJpegInternal` until the output size is close to the target. *This is significantly slower.*
    *   **Libjpeg Quality Match (`settings.libjpeg_quality > 0`)**: Encodes once with `libjpeg-turbo` (via `Encoder::FromExtension(".jpg")`) to get a target size, then calls `EncodeJpegToTargetSize` with that size.
    *   **Default (Distance/Quality based)**: Calls `EncodeJpegInternal` directly.

3.  **Internal Encoding Call (`EncodeJpegInternal`)**: This function orchestrates the core `libjpegli` interaction.

    *   **Setup**:
        *   Initializes `jpeg_compress_struct cinfo` and `jpeg_error_mgr jerr`.
        *   Sets up `setjmp`/`longjmp` error handling via `cinfo.client_data` and `MyErrorExit`.
        *   Configures memory destination using `jpegli_mem_dest`.

    *   **Color Management & Transform**:
        *   Determines the `output_encoding` (target for `libjpegli`):
            *   If `settings.xyb`: `JXL_COLOR_SPACE_XYB`.
            *   Otherwise: `JXL_COLOR_SPACE_RGB` (sRGB primaries, D65 white point, sRGB transfer function).
        *   Sets up `ColorSpaceTransform c_transform` if the input `ppf.color_encoding` needs conversion to the `output_encoding`.
        *   **XYB Path (`settings.xyb == true`)**:
            *   Input data (potentially after conversion to linear RGB) is transformed to XYB float data using `c_transform.ToXYB()`.
            *   `cinfo.in_color_space` is set to `JCS_EXT_RGB` (special value indicating XYB input to `libjpegli`).
            *   `cinfo.input_components` is 3.
            *   `jpegli_set_input_format` is set to `JPEGLI_TYPE_FLOAT`.
            *   An XYB ICC profile is generated and embedded later via `jpegli_write_icc_profile`.
        *   **YCbCr Path (`settings.xyb == false`)**:
            *   Input data is transformed to the sRGB `output_encoding` (float) if necessary, using `c_transform.Run()`.
            *   `cinfo.in_color_space` is set to `JCS_RGB` or `JCS_GRAYSCALE`. `libjpegli` will perform the RGB->YCbCr conversion internally for `JCS_RGB`.
            *   `cinfo.input_components` is set from `ppf.info.num_color_channels`.
            *   `jpegli_set_input_format` is set based on the original `ppf.info.bits_per_sample` (e.g., `JPEGLI_TYPE_UINT8`, `JPEGLI_TYPE_UINT16`).
            *   If the original input color space was not sRGB/Grayscale, its ICC profile is transformed to sRGB by `c_transform` and embedded later via `jpegli_write_icc_profile`.

    *   **Set `libjpegli` Parameters**:
        *   `jpegli_set_defaults(&cinfo)`: Applies initial defaults.
        *   `jpegli_set_progressive_level(&cinfo, settings.progressive_level)`
        *   `jpegli_set_xyb_mode(&cinfo)` (if `settings.xyb`)
        *   `jpegli_use_standard_quant_tables(&cinfo)` (if `settings.use_std_quant_tables`)
        *   `jpegli_set_adaptive_quantization(&cinfo, settings.use_adaptive_quantization)`
        *   `jpegli_set_optimize_coding(&cinfo, settings.optimize_coding)`
        *   `jpegli_set_chroma_subsampling(&cinfo, settings.chroma_subsampling)` (if specified, otherwise `libjpegli` chooses based on quality/distance).
        *   Quality/Distance:
            *   If `settings.quality > 0`: `jpegli_set_quality(&cinfo, settings.quality, TRUE)`
            *   Else: `jpegli_set_distance(&cinfo, settings.distance)`

    *   **Start Compression**: `jpegli_start_compress(&cinfo, TRUE)`: Writes SOI, prepares for scan data.

    *   **Write Metadata**:
        *   If `settings.app_data` is non-empty: Writes custom APP markers using `jpegli_write_marker`.
        *   Else (and if needed): Writes the determined ICC profile (XYB or transformed sRGB) using `jpegli_write_icc_profile`.

    *   **Write Pixel Data**:
        *   Loops `ppf.info.ysize` times.
        *   Prepares one row of pixel data in the format specified by `jpegli_set_input_format` (float for XYB, uint8/uint16 otherwise). Handles potential channel mismatches (e.g., dropping alpha).
        *   Calls `jpegli_write_scanlines(&cinfo, row_ptr, 1)` to feed the data to `libjpegli`.

    *   **Finish Compression**: `jpegli_finish_compress(&cinfo)`: Writes EOI, finalizes entropy coding, flushes output.

    *   **Cleanup**: `jpegli_destroy_compress(&cinfo)`: Frees resources.

    *   **Return**: Copies the compressed data from the memory buffer to the output `compressed` vector.

## 4. Core Algorithms (within `libjpegli`)

Based on the settings passed via the `jpegli_set_*` functions, `libjpegli` performs the following:

*   **Color Transform (RGB -> YCbCr)**: If `cinfo.in_color_space` was `JCS_RGB`, `libjpegli` performs its internal conversion to YCbCr. XYB data bypasses this.
*   **Downsampling**: Applies chroma subsampling based on `jpegli_set_chroma_subsampling` or defaults.
*   **DCT**: Performs Discrete Cosine Transform on 8x8 blocks.
*   **Quantization**:
    *   Uses tables derived from `distance`/`quality` or standard Annex K tables (`--std_quant`).
    *   Applies adaptive quantization (`--noadaptive_quantization` to disable) adjusting quantization per-block based on psychovisual models (masking, etc.).
*   **Entropy Coding (Huffman)**:
    *   Zig-zags quantized DCT coefficients.
    *   Performs DC/AC prediction.
    *   Generates Huffman codes. Optimizes tables unless `--fixed_code` is used.
    *   Writes coefficients according to scan script determined by `progressive_level`.

## 5. Detailed Encoding Pipeline (`jxl::extras::EncodeJpegInternal` and `libjpegli`)

This section provides a detailed walkthrough of the encoding process, assuming adaptive quantization is enabled and focusing on the default settings where applicable. It traces the flow from the C++ wrapper into the core `libjpegli` functions.

**Input**: `PackedPixelFile ppf` (containing pixels, color encoding `ppf.color_encoding`, bit depth `ppf.info.bits_per_sample`, dimensions, etc.) and `JpegSettings settings`.

**A. Initial Setup (`EncodeJpegInternal` in `lib/extras/enc/jpegli.cc:373`)**

1.  **Error Handling**: Sets up `jpeg_compress_struct cinfo`, `jpeg_error_mgr jerr`, and uses `setjmp`/`longjmp` for C-style error handling (`lib/extras/enc/jpegli.cc:53-58`, `lib/extras/enc/jpegli.cc:380-386`).
2.  **Memory Destination**: Configures `libjpegli` to write to a memory buffer (`jpegli_mem_dest` - `lib/jpegli/dest.cc:73`).
3.  **Image Dimensions**: Sets `cinfo.image_width = ppf.info.xsize` and `cinfo.image_height = ppf.info.ysize` (`lib/extras/enc/jpegli.cc:389-390`).

**B. Color Management and Input Preparation (`EncodeJpegInternal` in `lib/extras/enc/jpegli.cc:392-464`)**

1.  **Determine Target Encoding (`output_encoding`)**: 
    *   If `settings.xyb` is true: `output_encoding` is set to XYB (`jxl::ColorEncoding::LinearSRGB(false)` followed by transformation, effectively `JXL_COLOR_SPACE_XYB`). (`lib/extras/enc/jpegli.cc:401, 417`)
    *   Otherwise: `output_encoding` is set to sRGB (`jxl::ColorEncoding::SRGB(false)`). `libjpegli` will handle the YCbCr conversion internally. (`lib/extras/enc/jpegli.cc:403, 418`)
2.  **Setup Color Transform (`ColorSpaceTransform c_transform`)**: 
    *   Initialized using `ppf.color_encoding` as source and `output_encoding` as target (`lib/extras/enc/jpegli.cc:419`).
    *   The transform handles converting the input pixel data format (`ppf.info.bits_per_sample`, `ppf.info.num_color_channels`) and color space to the target format needed (float sRGB or float XYB).
3.  **Perform Color Transform / Prepare Pixel Pointer (`image_pixels`)**: 
    *   **XYB Path (`settings.xyb == true`)**: 
        *   Input pixels are converted to `float` linear sRGB by `c_transform` if necessary (e.g., from gamma-encoded sRGB uint16).
        *   `c_transform.ToXYB()` converts the linear sRGB float data to XYB float data, storing it in an internal buffer (`xyb`). (`lib/extras/enc/jpegli.cc:424`)
        *   `image_pixels` pointer is set to the `xyb` buffer. 
    *   **YCbCr Path (`settings.xyb == false`)**: 
        *   If the input is not already sRGB (`!ppf.IsSRGB()`) or not Gray, `c_transform.Run()` converts the input pixels to sRGB float (gamma encoded), storing the result in `rgb_storage`. (`lib/extras/enc/jpegli.cc:432`). The specific conversion depends on `ppf.color_encoding`.
        *   `image_pixels` pointer is set to `rgb_storage` if conversion happened, otherwise it points to the original `ppf.pixels()` buffer. 
4.  **Set `libjpegli` Input Color Space**: 
    *   XYB Path: `cinfo.in_color_space = JCS_EXT_RGB`, `cinfo.input_components = 3`. (`lib/extras/enc/jpegli.cc:442-443`). `JCS_EXT_RGB` signals XYB input to `libjpegli`.
    *   YCbCr Path: `cinfo.in_color_space` set based on `ppf.info.num_color_channels` (e.g., `JCS_RGB` or `JCS_GRAYSCALE`). `cinfo.input_components` set accordingly. (`lib/extras/enc/jpegli.cc:447-452`). `JCS_RGB` signals that `libjpegli` should perform the RGB->YCbCr transform.

**C. Configure `libjpegli` Parameters (`EncodeJpegInternal` & `libjpegli/encode.cc`)**

1.  **Set Defaults (`jpegli_set_defaults`)**: (`lib/extras/enc/jpegli.cc:454`, calls `encode.cc:698`)
    *   Calls `jpegli::InitializeCompressParams(cinfo)` (`encode_internal.cc:115`): Resets many internal parameters.
    *   Calls `jpegli_default_colorspace(cinfo)` (`encode.cc:710`): Sets `cinfo.jpeg_color_space` (e.g., to `JCS_YCbCr` if `in_color_space` was `JCS_RGB`, or `JCS_RGB` if `xyb_mode` is true) and sets up `cinfo.num_components` and default component parameters (sampling factors, table indices) based on the *output* JPEG colorspace. Sets chroma subsampling to 2x2 by default for YCbCr.
    *   Calls `jpegli_set_quality(cinfo, 90, TRUE)` (`encode.cc:701`, calls `encode.cc:873`): Sets default distance based on Q90 using `jpegli_quality_to_distance` (`encode.cc:855`), then calls `jpegli::SetQuantMatrices`. See **Section C.8** for `SetQuantMatrices` detail.
    *   Calls `jpegli_set_progressive_level(cinfo, jpegli::kDefaultProgressiveLevel)` (`encode.cc:702`, calls `encode.cc:947`): Sets `cinfo.master->progressive_level` (default 2).
    *   Calls `jpegli::AddStandardHuffmanTables` (`encode.cc:703-706`, defined in `huffman.cc:131`): Populates `cinfo->dc_huff_tbl_ptrs` and `cinfo->ac_huff_tbl_ptrs` with standard JPEG Huffman tables.
2.  **Apply Settings from `JpegSettings`**: (`lib/extras/enc/jpegli.cc:456-474`)
    *   `jpegli_set_progressive_level(&cinfo, settings.progressive_level)` (`encode.cc:947`)
    *   `jpegli_set_xyb_mode(&cinfo)` (if `settings.xyb`) (`encode.cc:693`): Sets `cinfo.master->xyb_mode = true`.
    *   `jpegli_use_standard_quant_tables(&cinfo)` (if `settings.use_std_quant_tables`) (`encode.cc:914`): Sets `cinfo.master->use_std_tables = true`.
    *   `jpegli_set_adaptive_quantization(&cinfo, settings.use_adaptive_quantization)` (`encode.cc:937`): Sets `cinfo.master->use_adaptive_quantization` (true by default).
    *   `jpegli_set_optimize_coding(&cinfo, settings.optimize_coding)` (`encode_internal.cc:131`): Sets `cinfo->optimize_coding` (true by default).
    *   `jpegli_set_chroma_subsampling(&cinfo, settings.chroma_subsampling)` (if specified) (`encode_internal.cc:137`): Parses string ("444", "420", etc.) and sets `h_samp_factor`/`v_samp_factor` for chroma components (`cinfo->comp_info[1]`, `cinfo->comp_info[2]`).
    *   **Quality/Distance**: (`lib/extras/enc/jpegli.cc:470-474`)
        *   If `settings.quality > 0`: `jpegli_set_quality(&cinfo, settings.quality, TRUE)` (`encode.cc:873`) is called. It converts quality to distance using `jpegli_quality_to_distance` (`encode.cc:855`) and calls `jpegli::SetQuantMatrices`. 
        *   Else: `jpegli_set_distance(&cinfo, settings.distance)` (`encode.cc:850`) is called. It directly calls `jpegli::SetQuantMatrices` with the given distance.
3.  **Set Input Format (`jpegli_set_input_format`)**: (`lib/extras/enc/jpegli.cc:483-487`, calls `encode.cc:953`)
    *   If XYB Path: `data_type = JPEGLI_TYPE_FLOAT`, `endianness = JPEGLI_NATIVE_ENDIAN`.
    *   Else: `data_type` mapped from `ppf.info.data_type` (e.g., `JXL_TYPE_UINT16` -> `JPEGLI_TYPE_UINT16`), `endianness` mapped (`ConvertDataType`, `ConvertEndianness` in `lib/extras/enc/jpegli.cc:74-109`). Sets `cinfo.master->data_type` and `cinfo.master->endianness`.
4.  **Start Compression (`jpegli_start_compress`)**: (`lib/extras/enc/jpegli.cc:488`, calls `encode.cc:1021`)
    *   Sets `cinfo->global_state = jpegli::kEncHeader`.
    *   Calls `jpegli::InitCompress(cinfo, TRUE)`. **See Section D**. 
    *   Resets `cinfo->next_scanline = 0`.
5.  **Write Metadata**: (`lib/extras/enc/jpegli.cc:489-498`)
    *   If `settings.app_data` provided: `jpegli_write_marker` (`encode.cc:1096`) is used.
    *   Else if ICC profile needed (XYB or non-sRGB input): `jpegli_write_icc_profile` (`encode.cc:1101`) writes the profile from `output_encoding.ICC()`, chunking it into multiple APP2 markers if necessary.

**D. Core Compression Initialization (`jpegli::InitCompress` in `encode.cc:506`)**

Called by `jpegli_start_compress`.

1.  **Process Parameters (`ProcessCompressionParams`)**: (`encode.cc:509`, calls `encode_internal.cc:146`)
    *   Validates component sampling factors.
    *   Calculates `cinfo->total_iMCU_rows`.
    *   Calculates `max_h_samp_factor`, `max_v_samp_factor`.
    *   Calculates component dimensions in blocks (`width_in_blocks`, `height_in_blocks`).
    *   Sets up `cinfo->comps_in_scan`, `cinfo->cur_comp_info` based on progressive level and color space.
    *   Calls `jpegli::SelectScans` (`scan_scripts.cc:58`) to populate `cinfo->scan_info` with scan script details (component indices, spectral range `Ss`/`Se`, bit approximation `Ah`/`Al`) based on `cinfo.master->progressive_level`. Default level 2 uses `kBQSProgressiveScans` or `kXybProgressiveScans`.
2.  **Allocate Buffers (`AllocateBuffers`)**: (`encode.cc:511`, calls `encode_internal.cc:42`)
    *   Allocates `cinfo.master->input_buffer` (per-component `RowBuffer<float>`) large enough to hold padded input rows for one iMCU row + context.
    *   Allocates `cinfo.master->quant_field` (`RowBuffer<float>`) for adaptive quantization multipliers (size = image dimensions in blocks).
    *   Allocates `cinfo.master->dct_buffer`, `cinfo.master->block_tmp`, etc. for intermediate calculations.
    *   Allocates `cinfo.master->coeff_buffers` if needed (not for streaming input).
    *   Allocates `cinfo.master->token_arrays` if `optimize_coding` is true.
3.  **Choose Input Method (`ChooseInputMethod`)**: (`encode.cc:513`, calls `encode_internal.cc:72`)
    *   Sets `cinfo.master->input_method` function pointer based on `cinfo.master->data_type` and `cinfo.master->endianness`.
    *   Example: For `JPEGLI_TYPE_UINT16` + `JPEGLI_LITTLE_ENDIAN`, selects `ReadU16LEScanline`. This function reads pairs of bytes, combines them into `uint16_t`, converts to `float` (typically `val / 65535.0f * 255.0f` or similar scaling might occur internally to map to ~0-255 range expected by later stages, although jpegli aims for 0-1 internal range).
    *   Example: For `JPEGLI_TYPE_FLOAT`, selects `ReadF32Scanline`, likely involving clamping or scaling to the internal 0-1 range.
4.  **Choose Color Transform (`ChooseColorTransform`)**: (`encode.cc:515`, calls `encode_internal.cc:92`)
    *   Sets `cinfo.master->color_transform` function pointer.
    *   If `cinfo.jpeg_color_space == JCS_YCbCr`, selects `RGBToYCbCr`. This function performs the standard RGB to YCbCr conversion math (`color_transform.cc:25`).
    *   Otherwise, selects `Passthrough`. (XYB transform happens earlier in the wrapper).
5.  **Choose Downsampler (`ChooseDownsampleMethods`)**: (`encode.cc:516`, calls `downsampling.cc:116`)
    *   Sets `cinfo.master->downsample_method[c]` for each component based on `h_samp_factor` and `v_samp_factor`.
    *   Selects appropriate downsampling functions (e.g., `Downsample2x2_BoxFilter`, `Downsample2x1_BoxFilter`, `NoDownsampling`). Box filter averages pixels in the 2x2 or 2x1 region.
6.  **Initialize Quantizer (`InitQuantizer`)**: (`encode.cc:520`, calls `quant.cc:706`) - **See Section D.1**. 
7.  **Setup Huffman Tables/Entropy Coder (if Fixed)**: (`encode.cc:525-528`)
    *   If `!cinfo->optimize_coding && !cinfo->progressive_mode`:
        *   `CopyHuffmanTables`: Ensures standard tables are ready.
        *   `InitEntropyCoder`: Prepares the entropy coder state with these fixed tables.
8.  **Initialize Output Destination**: (`encode.cc:530`, `cinfo->dest->init_destination`)
9.  **Write File Header (`WriteFileHeader`)**: (`encode.cc:531`, calls `marker.cc:37`)
    *   Writes SOI marker (`0xFFD8`).
    *   Writes JFIF marker (APP0) if `cinfo->write_JFIF_header` is true. (`marker.cc:46`)
    *   Writes Adobe APP14 marker if `cinfo->write_Adobe_marker` is true (for YCCK). (`marker.cc:58`)
10. **Initialize Bit Writer (`JpegBitWriterInit`)**: (`encode.cc:532`, calls `bit_writer.h:37`)
    *   Initializes `cinfo.master->bw` state.

**D.1. Quantizer Initialization (`InitQuantizer` in `quant.cc:706`)**

1.  **Compute Quantization Multipliers (`m->quant_mul`)**: (`quant.cc:711-727`)
    *   For each component `c`, gets the quantization table `quant_table = cinfo->quant_tbl_ptrs[comp->quant_tbl_no]`.
    *   For each frequency `k` (0-63):
        *   Gets `val = quant_table->quantval[k]` (These values were computed earlier by `SetQuantMatrices` - see **Section D.1.1**).
        *   Sets `m->quant_mul[c][k] = 8.0f / val`. (Assumes `pass == QuantPass::NO_SEARCH`).
2.  **Initialize Zero-Bias Multipliers/Offsets (`m->zero_bias_mul`, `m->zero_bias_offset`)**: (`quant.cc:728-767`)
    *   **Adaptive Quant Enabled (`m->use_adaptive_quantization`)**: 
        *   Initializes `zero_bias_mul[c][k] = 0.5f`, `zero_bias_offset[c][k] = 0.5f` (for k > 0).
        *   **YCbCr Specific**: (`quant.cc:734`)
            *   Calculates effective `distance = QuantValsToDistance(cinfo)` (`quant.cc:621`). This function estimates the perceptual distance corresponding to the current quantizer values.
            *   Calculates interpolation factor `mix0` between low-quality (LQ) and high-quality (HQ) presets based on `distance` (`quant.cc:740`).
            *   Interpolates `zero_bias_mul[c][k]` between `kZeroBiasMulYCbCrLQ` and `kZeroBiasMulYCbCrHQ` constants using `mix0` (`quant.cc:745`).
            *   Sets `zero_bias_offset[c][k]` from constants `kZeroBiasOffsetYCbCrDC` and `kZeroBiasOffsetYCbCrAC` (`quant.cc:747`).
    *   **Adaptive Quant Disabled (but YCbCr)**: (`quant.cc:750-756`)
        *   Only sets `zero_bias_offset` from `kZeroBiasOffsetYCbCrDC`/`AC` constants.

**D.1.1. Quantization Table Calculation (`SetQuantMatrices` in `quant.cc:657`)**

This function calculates the actual `quantval[k]` values stored in the tables, based on distance.

1.  **Select Base Matrix/Scale/Mode**: (`quant.cc:660-694`)
    *   Determines `xyb` mode, `is_yuv420`.
    *   Sets `global_scale`, `non_linear_scaling` flag, `base_quant_matrix` pointers (`kBaseQuantMatrixXYB`, `kBaseQuantMatrixYCbCr`, `kBaseQuantMatrixStd`), and `num_base_tables` based on XYB mode, YCbCr/Gray, `use_std_tables` flag.
    *   Adjusts `global_scale` for PQ/HLG transfer functions and YUV420 if applicable.
2.  **Compute `quantval`**: (`quant.cc:697-704`)
    *   Loops `quant_idx` from 0 to `num_base_tables - 1`.
    *   Loops `k` from 0 to `DCTSIZE2 - 1`.
    *   Calculates per-frequency scale:
        *   If `non_linear_scaling`: `s = DistanceToScale(distances[quant_idx], k)` (`quant.cc:593`), adjusts for YUV420 chroma (`k420Rescale`), `scale = global_scale * s`.
        *   Else (linear/standard): `s = DistanceToLinearQuality(distances[quant_idx])` (`quant.cc:582`), `scale = global_scale * s`.
    *   `qval = round(scale * base_quant_matrix[quant_idx][k])`.
    *   `(*qtable)->quantval[k] = clamp(qval, 1, quant_max)`. 

**E. Per-Scanline Processing (`jpegli_write_scanlines` in `encode.cc:1125`)**

Loops `num_lines` times.

1.  **Read Input Row (`ReadInputRow`)**: (`encode.cc:1148`, calls chosen `input_method`)
    *   Uses the selected `input_method` (e.g., `ReadU16LEScanline`) to read the input `scanlines[i]` and convert it to float, storing in `m->input_buffer[c].Row(m->next_input_row)` for each component `c`.
2.  **Color Transform**: (`encode.cc:1150`)
    *   Calls `(*m->color_transform)(rows, cinfo->image_width)`. If YCbCr path, this performs RGB->YCbCr conversion using `jpegli::RGBToYCbCr`. If XYB/Gray, it's `Passthrough`.
3.  **Pad Input Buffer (`PadInputBuffer`)**: (`encode.cc:1151`, defined `encode.cc:570`)
    *   Adds 1 pixel border (repeating edge pixel) to left/right. Pads rows vertically if at the last row.
4.  **Process iMCU Rows (`ProcessiMCURows`)**: (`encode.cc:1152`, defined `encode.cc:610`)
    *   Calls `ProcessiMCURow` (`encode.cc:594`) when enough input lines for an iMCU row (+ context) are available. **See Section F**.
5.  **Check Output Buffer (`EmptyBitWriterBuffer`)**: (`encode.cc:1153`)
    *   Flushes bits to the output destination if the internal buffer is full (relevant for streaming with fixed Huffman tables).

**F. Per-iMCU Row Processing (`ProcessiMCURow` in `encode.cc:594`)**

Processes one row of MCUs.

1.  **Input Smoothing (`ApplyInputSmoothing`)**: (`encode.cc:597`, calls `smoothing.cc:83`)
    *   If enabled (seems default), applies a smoothing filter, possibly a bilateral filter or similar, to the input buffer (`m->input_buffer`). 
2.  **Downsampling (`DownsampleInputBuffer`)**: (`encode.cc:598`, calls `downsampling.cc:141`)
    *   Calls the chosen `downsample_method` (e.g., `Downsample2x2_BoxFilter`) for chroma components, writing the result to `m->raw_data` buffer.
3.  **Compute Adaptive Quant Field (`ComputeAdaptiveQuantField`)**: (`encode.cc:600`, calls `adaptive_quantization.cc:516`) - **See Section F.1**. 
4.  **Branch based on Mode**: 
    *   If streaming supported and `optimize_coding`: `ComputeTokensForiMCURow(cinfo)` (`encode_streaming.cc:255`) - **See Section G**. 
    *   If streaming supported and `!optimize_coding`: `WriteiMCURow(cinfo)` (`encode_streaming.cc:259`).
    *   Else (coefficient output): `ComputeCoefficientsForiMCURow(cinfo)` (`encode_streaming.cc:251`).

**F.1. Adaptive Quant Field Computation (`ComputeAdaptiveQuantField` in `adaptive_quantization.cc:516`)**

Calculates `m->quant_field` for the current iMCU row.

1.  **Get Luma Channel/Quant**: (`adaptive_quantization.cc:522-524`)
2.  **Pad Input Borders**: (`adaptive_quantization.cc:525-530`)
3.  **Compute Pre-Erosion Map (`ComputePreErosion`)**: (`adaptive_quantization.cc:542`, calls HWY_DYNAMIC_DISPATCH version, defined `adaptive_quantization.cc:444`)
    *   Loops over 4x4 blocks (`y = y0..y0+ylen-1`, `x = 0..xsize-1`).
    *   Calculates local differences (`diff`) vs neighbors average.
    *   Applies gamma correction: `gammacv = RatioOfDerivativesOfCubicRootToSimpleGamma(in + offset)` (`adaptive_quantization.cc:464`).
    *   `diff = gammacv * (in - base_avg)`.
    *   `diff = diff * diff`.
    *   `diff = min(diff, 0.2f)`.
    *   `diff = MaskingSqrt(diff)` (`adaptive_quantization.cc:470`, defined `adaptive_quantization.cc:312`).
    *   Averages `diff` over 4x4 block into `m->pre_erosion`. (`adaptive_quantization.cc:477`).
    *   Pads `m->pre_erosion` borders.
4.  **Fuzzy Erosion (`FuzzyErosion`)**: (`adaptive_quantization.cc:547`, calls HWY_DYNAMIC_DISPATCH version, defined `adaptive_quantization.cc:348`)
    *   Loops over 2x2 blocks in `m->pre_erosion` (`y = 2*yb0..2*(yb0+yblen)-1`, `x = 0..xsize-1`).
    *   For each pixel `(x,y)` in `m->pre_erosion`:
        *   Finds the 4 minimum values (`min0..min3`) in the 3x3 neighborhood using `Sort4`/`UpdateMin4`. (`adaptive_quantization.cc:358-368`)
        *   Computes weighted sum: `v = 0.125*min0 + 0.075*min1 + 0.06*min2 + 0.05*min3`. (`adaptive_quantization.cc:370`)
        *   Stores `v` in temporary buffer `m->fuzzy_erosion_tmp`.
    *   Averages `m->fuzzy_erosion_tmp` over 2x2 blocks into `m->quant_field` (`aq_out[bx] = (row_out[x] + ...)*0.25f`) (`adaptive_quantization.cc:376`).
5.  **Per-Block Modulations (`PerBlockModulations`)**: (`adaptive_quantization.cc:549`, calls HWY_DYNAMIC_DISPATCH version, defined `adaptive_quantization.cc:282`)
    *   Loops over 8x8 blocks (`iy = 0..yblen-1`, `ix = 0..xsize_blocks-1`).
    *   Gets `out_val` from `m->quant_field[yb+iy][ix]` (output of erosion).
    *   Applies masking modulation: `out_val = ComputeMask(out_val)` (`adaptive_quantization.cc:292`, defined `adaptive_quantization.cc:146`). Uses rational polynomial approximation.
    *   Applies HF modulation: `out_val = HfModulation(x, y, input, out_val)` (`adaptive_quantization.cc:293`, defined `adaptive_quantization.cc:251`). Sums absolute differences in the 8x8 input block, multiplies by `kSumCoeff`.
    *   Applies Gamma modulation: `out_val = GammaModulation(x, y, input, out_val)` (`adaptive_quantization.cc:294`, defined `adaptive_quantization.cc:216`). Averages `RatioOfDerivatives...` over the 8x8 input block, takes log, multiplies by `kGamma`, adds to `out_val`.
    *   Converts modulated exponent `out_val` back to linear scale: `linear_qf = FastPow2f(out_val * 1.442f)` (`adaptive_quantization.cc:297`, `FastPow2f` defined `adaptive_quantization.cc:133`).
    *   Applies final damping based on `y_quant_01`: `mul = kAcQuant * dampen`, `add = (1.0 - dampen) * base_level`. (`adaptive_quantization.cc:289`)
    *   Stores `linear_qf * mul + add` back into `m->quant_field`. (`adaptive_quantization.cc:297`)
6.  **Final Adjustment**: (`adaptive_quantization.cc:552-556`)
    *   `row[x] = max(0.0f, (0.6f / row[x]) - 1.0f)`. Stores the final per-block factor used during quantization.

**G. Compute Tokens (`ComputeTokensForiMCURow` -> `ProcessiMCURow<kStreamingModeTokens>` in `encode_streaming.cc:111`)**

Processes one iMCU row, performing DCT, Quantization, and Tokenization.

1.  **Loops**: Over MCUs (`mcu_x`), components (`c`), blocks within MCU (`iy`, `ix`).
2.  **Get Adaptive Quant Strength**: `aq_strength = qf[iy * qf_stride + bx * h_factor];` (`encode_streaming.cc:190`)
3.  **Compute Coefficient Block (`ComputeCoefficientBlock`)**: (`encode_streaming.cc:193`, calls function in `dct-inl.h:241`)
    *   **DCT**: `TransformFromPixels(pixels, stride, dct, scratch_space)` (`dct-inl.h:247`). Performs 2D DCT using 1D Cooley-Tukey style algorithm (`DCT1DImpl`), stores result in `dct` buffer.
    *   **Quantization (`QuantizeBlock`)**: (`dct-inl.h:248`, defined `dct-inl.h:217`)
        *   Loops `k` from 0 to 63.
        *   `val = dct[k]`
        *   `q = qmc[k]` (base multiplier from `InitQuantizer`)
        *   `qval = val * q`
        *   `zb_offset = zero_bias_offset[k]`, `zb_mul = zero_bias_mul[k]`
        *   `threshold = zb_offset + zb_mul * aq_strength` **<-- Adaptive Quant Application**
        *   `nzero_mask = abs(qval) >= threshold`
        *   `ival = nzero_mask ? round(qval) : 0`
        *   `StoreQuantizedValue(ival, block + k)` (stores integer coefficient).
    *   **DC Handling**: (`dct-inl.h:250-258`)
        *   `dc = (dct[0] - 128.0f) * qmc[0]`
        *   `dc_threshold = zero_bias_offset[0] + aq_strength * zero_bias_mul[0]`
        *   `if (abs(dc - last_dc_coeff) < dc_threshold) block[0] = last_dc_coeff; else block[0] = round(dc);`
4.  **DC Difference Coding**: `block[0] -= last_dc_coeff[c]; last_dc_coeff[c] += block[0];` (`encode_streaming.cc:200-201`).
5.  **Token Generation (`ComputeTokensForBlock`)**: (`encode_streaming.cc:203`, calls function in `entropy_coding-inl.h:151`)
    *   Iterates through the zig-zag ordered `block` of quantized coefficients.
    *   Emits DC token: `Token(c, kDCSymbol, PackDC(block[0]))`.
    *   Finds runs of zeros and subsequent non-zero AC coefficients.
    *   Emits AC tokens: `Token(c + 4, PackAC(run_length, num_bits), coefficient_bits)`. `PackAC` combines run length (0-15) and coefficient size (1-10 bits). Special ZRL (Zero Run Length) token for runs > 15. EOB (End Of Block) token if rest are zeros.
    *   Appends tokens to `m->next_token`.

**H. Finalization (`jpegli_finish_compress` in `encode.cc:1220`)**

1.  **Check Completion**: Ensures `cinfo->next_scanline == cinfo->image_height`.
2.  **Optimize Huffman Codes (if `cinfo->optimize_coding` or progressive)**: (`encode.cc:1239`, calls `huffman.cc:256`)
    *   `ComputeTokenHistograms`: Builds histograms of DC/AC symbols from all `m->token_arrays`.
    *   Loops through histograms:
        *   `BuildHuffmanTable`: Creates canonical Huffman codes based on symbol frequencies.
        *   Stores tables in `m->coding_tables` and also copies them to `cinfo->dc_huff_tbl_ptrs`/`ac_huff_tbl_ptrs`.
3.  **Initialize Entropy Coder (`InitEntropyCoder`)**: (`encode.cc:1240`, calls `entropy_coding.cc:33`)
    *   Sets up internal lookup tables (`m->context_map`, etc.) based on component Huffman table assignments (`dc_tbl_no`, `ac_tbl_no`).
4.  **Write Headers (Frame/Scan)**: (`encode.cc:1243-1246`)
    *   `WriteFrameHeader`: Writes DQT (from `cinfo->quant_tbl_ptrs`), DHT (from `cinfo->dc/ac_huff_tbl_ptrs`), SOS markers based on `cinfo->scan_info` generated by `SelectScans`.
    *   `WriteScanHeader`: Writes SOS marker for subsequent scans if progressive.
5.  **Write Scan Data (`WriteScanData`)**: (`encode.cc:1247`, calls `encode_finish.cc:38`)
    *   Iterates through tokens for the current scan (`m->token_arrays`).
    *   For each token, looks up the Huffman code in `m->coding_tables`.
    *   Calls `WriteBits(&m->bw, code.depth, code.code)` to write the Huffman code.
    *   Writes extra bits for coefficient magnitude if needed.
    *   Handles restart markers (`RSTm`) based on `cinfo->restart_interval`.
6.  **Write EOI**: (`encode.cc:1251`, calls `WriteOutput`) Writes `0xFFD9`.
7.  **Terminate Destination**: (`encode.cc:1252`, `cinfo->dest->term_destination`)
8.  **Cleanup**: (`encode.cc:1255`, `jpegli_abort_compress`) Releases memory.

This provides a highly detailed trace. Reimplementing this *exactly* requires careful attention to the specific constants, lookup tables (e.g., `kZeroBias*`, `kBaseQuantMatrix*`), rounding modes, and bit-packing details within the called functions (especially in `-inl.h` files). The use of Highway SIMD intrinsics also adds complexity to a direct C++ reimplementation, although the logic can be replicated with scalar operations.


**F.2. Padding Details**

Padding is crucial for handling image dimensions that are not multiples of block/MCU sizes and for providing context for filters.

1.  **Input Buffer (`m->input_buffer`) Padding (`PadInputBuffer` in `encode.cc:570`)**:
    *   This is the primary padding stage, occurring after each row is read and color-transformed.
    *   **Allocation**: `RowBuffer` (`common_internal.h:91`) allocates rows with alignment/padding to allow safe access to `[-1]` and `[width]`. Vertical access wraps around (e.g., `Row(-1)` gets the last row).
    *   **Horizontal**: For each component row, the last valid pixel (e.g., pixel `width-1`) is replicated to fill columns up to the next multiple of 8 (`width_in_blocks * 8`). The `[-1]` index is filled with the value from index `[0]`. (`PadRow` in `common_internal.h:110`).
    *   **Vertical**: After the *last* image row is processed, the remaining rows in `input_buffer` needed to reach the full block height (`height_in_blocks * 8`) are filled by copying the last valid, horizontally padded row using `CopyRow` (`common_internal.h:121`).
2.  **Downsampling Output (`m->raw_data`)**: The downsampler (e.g., `Downsample2x2_BoxFilter` in `downsampling-inl.h:36`) reads from the padded `input_buffer`. It writes only the valid downsampled result (e.g., 4x4 for Cb/Cr in the 3x11 example) into the top-left of the `raw_data` buffer. This buffer itself is *not* explicitly edge-padded by the downsampler.
3.  **Adaptive Quant Intermediate Buffers**: 
    *   `m->pre_erosion` (`adaptive_quantization.cc:444`): Calculated from padded `input_buffer[Y]`. Padded with a 1-pixel border using `PadRow`/`CopyRow` for `FuzzyErosion` context. (`adaptive_quantization.cc:480, 544-546`).
    *   `m->quant_field`: No explicit padding.
4.  **DCT/Quantization Stage Boundary Handling**: (`encode_streaming.cc:177-186`)
    *   Reads 8x8 blocks from `m->raw_data`.
    *   If a block's coordinates (`bx`, `by`) are completely outside the component's `width_in_blocks` or `height_in_blocks`, it skips DCT/Quant for that block and emits zero-value tokens/coefficients.
    *   If a block is partially outside (e.g., Y block 0,0 in the 3x11 example), the DCT reads the valid data plus the replicated edge padding that was originally added in `PadInputBuffer` and propagated through downsampling (for Y) or exists in the unused parts of the chroma `raw_data` buffer.


**G.1. Buffer Design and Data Types**

*   **Planar Processing**: After initial input conversion, `libjpegli` operates primarily on **planar** data, meaning each color component (Y, Cb, Cr, or X, Y, B) is stored in separate buffers.
*   **Key Buffers** (`jpeg_comp_master* m` state in `jpegint.h`):
    *   `m->input_buffer`: `RowBuffer<float>[kMaxComponents]` (`common_internal.h:91`). Stores input pixels after type conversion (to float) and potentially color transform (e.g., RGB->YCbCr). Padded using edge replication to DCT block/MCU boundaries.
    *   `m->raw_data`: `RowBuffer<float>[kMaxComponents]`. Stores planar float data after downsampling (if any). Size matches component dimensions rounded to 8x8 blocks. Contains valid downsampled data; boundary blocks outside the component are handled later, not by padding this buffer explicitly.
    *   `m->quant_field`: `RowBuffer<float>`. Single plane (Y component) holding per-block adaptive quantization multipliers. Size `width_in_blocks` x `height_in_blocks`.
    *   `m->dct_buffer`, `m->block_tmp`: Temporary `float[]` and `int32_t[]` buffers used during DCT and quantization of a single block.
    *   `m->token_arrays`: `TokenArray[]`. Stores sequences of `Token` structs (representing quantized DCT coefficients) if Huffman optimization is enabled.
*   **Data Types**: 
    *   Input (`uint8`, `uint16`, `float32`) is converted to **`float`** by the `input_method` (`ReadInputRow` in `encode.cc:555`). The internal float range is conceptually 0-1, though precise scaling might vary.
    *   Most internal processing (color transform, downsampling, adaptive quant, DCT) uses **`float`**.
    *   Quantization (`QuantizeBlock` in `dct-inl.h:217`) converts float DCT coefficients to **`int32_t`** (stored temporarily in `m->block_tmp`). Note: If using coefficient writing mode, the output type is `JCOEF` (int16_t).
    *   Entropy coding operates on integer tokens or coefficients.

**G.2. Color Management Flow**

1.  **Input Reading (`jxl::extras::DecodeBytes`)**: Reads the image file. Extracts embedded ICC profile (`ppf.icc`) or determines color space from format metadata. Parses/sets `ppf.color_encoding`. `--dec-hints` can override this.
2.  **Wrapper Transform (`EncodeJpegInternal` using `ColorSpaceTransform`)**: 
    *   Determines the target space for `libjpegli`: XYB (`--xyb`) or sRGB (default).
    *   Uses `ColorSpaceTransform` (lcms2) to convert `ppf` pixel data (any input type/space) to the target space as `float` planar data.
    *   For XYB target: `ppf` -> Linear sRGB float -> XYB float (using `ToXYB`).
    *   For sRGB target: `ppf` -> sRGB float (gamma encoded) (using `Run`).
3.  **`libjpegli` Input Configuration**: 
    *   `cinfo.in_color_space` is set (`JCS_EXT_RGB` for XYB, `JCS_RGB`/`JCS_GRAYSCALE` otherwise).
    *   `jpegli_default_colorspace` sets `cinfo.jpeg_color_space` (e.g., `JCS_YCbCr` if input was `JCS_RGB`).
4.  **`libjpegli` Internal Transform (`ChooseColorTransform` -> `RGBToYCbCr`)**: 
    *   If `cinfo.in_color_space` was `JCS_RGB` and `cinfo.jpeg_color_space` is `JCS_YCbCr`, this transform (`color_transform.cc:25`) is applied internally to the float data in `m->input_buffer`.
5.  **ICC Profile Embedding (`jpegli_write_icc_profile`)**: 
    *   Called *unless* `settings.app_data` is set.
    *   Embeds the profile corresponding to the **target encoding** used for the wrapper transform (`output_encoding.ICC()`): either the specific XYB profile or a standard sRGB profile.
    *   The original input profile (`ppf.icc`) is generally *not* embedded directly, as the pixel data was transformed.
    *   Writes profile into APP2 markers (`encode.cc:1101`).

**H. End-to-End Color Handling Examples**

These examples trace the color management flow for different input types and output settings.

**1. Input sRGB JPEG -> Output sRGB JPEG**
   - **Decode**: `DecodeBytes` reads JPEG, sets `ppf.color_encoding=sRGB`.
   - **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` converts uint8 input to sRGB float.
   - **Libjpegli Config**: `in_color_space=JCS_RGB`, `jpeg_color_space=JCS_YCbCr`.
   - **Libjpegli Transform**: Applies internal `RGBToYCbCr`.
   - **ICC Output**: None embedded by default.

**2. Input 16-bit sRGB PNG -> Output sRGB JPEG**
   - **Decode**: `DecodeBytes` reads PNG, sets `ppf.color_encoding=sRGB`. Input uint16.
   - **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` converts uint16 input to sRGB float.
   - **Libjpegli Config**: `in_color_space=JCS_RGB`, `jpeg_color_space=JCS_YCbCr`. Input format likely set to float based on wrapper output.
   - **Libjpegli Transform**: Applies internal `RGBToYCbCr`.
   - **ICC Output**: None embedded by default.

**3. Input 16-bit DCI-P3 PNG -> Output sRGB JPEG**
   - **Decode**: `DecodeBytes` reads PNG, extracts embedded DCI-P3 ICC profile into `ppf.icc`, parses `ppf.color_encoding`. Input uint16.
   - **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` uses lcms2 to convert DCI-P3 uint16 -> sRGB float.
   - **Libjpegli Config**: `in_color_space=JCS_RGB`, `jpeg_color_space=JCS_YCbCr`. Input format float.
   - **Libjpegli Transform**: Applies internal `RGBToYCbCr` to the sRGB float data.
   - **ICC Output**: None embedded by default. (Output contains YCbCr derived from sRGB).

**4. Input 16-bit DCI-P3 PNG -> Output XYB JPEG (`--xyb`)**
   - **Decode**: Reads DCI-P3 PNG as above.
   - **Wrapper Transform**: Target XYB. `ColorSpaceTransform` uses lcms2 (DCI-P3 uint16 -> Linear sRGB float) then matrix multiply (`ToXYB`) -> XYB float.
   - **Libjpegli Config**: `in_color_space=JCS_EXT_RGB`, `jpeg_color_space=JCS_RGB`. Input format float.
   - **Libjpegli Transform**: `Passthrough` (no YCbCr conversion).
   - **ICC Output**: Embeds the standard XYB ICC profile via `jpegli_write_icc_profile`.

**5. Google Ultra HDR JPEG -> Output sRGB JPEG**
   - *Assumption*: Only the base SDR JPEG is processed by `cjpegli`.
   - **Decode**: Reads base JPEG (assume sRGB uint8). Gain map metadata ignored by core encoding.
   - **Wrapper Transform**: Target sRGB. Converts uint8 -> sRGB float.
   - **Libjpegli Config**: `in_color_space=JCS_RGB`, `jpeg_color_space=JCS_YCbCr`.
   - **Libjpegli Transform**: Applies internal `RGBToYCbCr`.
   - **ICC Output**: None embedded by default.

**6. Custom Profile PNG -> Output sRGB JPEG**
   - **Decode**: Reads PNG, extracts custom ICC (`ppf.icc`), parses `ppf.color_encoding`.
   - **Wrapper Transform**: Target sRGB. `ColorSpaceTransform` uses lcms2 to convert Custom Profile -> sRGB float.
   - **Libjpegli Config**: `in_color_space=JCS_RGB`, `jpeg_color_space=JCS_YCbCr`.
   - **Libjpegli Transform**: Applies internal `RGBToYCbCr`.
   - **ICC Output**: None embedded by default.

**7. Custom Profile PNG -> Output "Keep Custom Profile" JPEG**
   - **Not directly supported.** `libjpegli` transforms pixels internally.
   - **Workaround (`--app_data`)**: 
      - Decode PNG, get custom `ppf.icc`.
      - User manually prepares `ppf.icc` as APP2 marker data.
      - Run `cjpegli` targeting sRGB, passing marker data via `--app_data` (not a standard `cjpegli` flag, implies modifying `EncodeJpegInternal` or using a different tool setup).
      - Wrapper still converts Custom->sRGB float.
      - Libjpegli still converts sRGB->YCbCr.
      - `jpegli_write_marker` embeds the custom profile from `app_data`.
      - **Result**: Output contains YCbCr data (derived from sRGB representation), but *claims* to be in the custom space via the embedded profile. Color appearance is likely incorrect. Psychovisual optimizations are not tuned for the custom space. 