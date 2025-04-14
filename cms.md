# Color Management System (CMS) in `jpeg-encoder`

This document outlines the color management strategy used in the `jpeg-encoder` crate, primarily leveraging the `lcms2` crate for handling ICC profiles and color transformations. The goal is to replicate the color handling capabilities of the original C++ `jpegli` library.

## Core Concepts

*   **ICC Profiles:** Color spaces are primarily defined using embedded ICC profiles. The `lcms2` crate is used to parse these profiles and create color transformations.
*   **Input/Output Profiles:** The encoder needs both an input color profile (describing the source image data) and an output profile (the target for the JPEG encoding process). For standard JPEG, the output is typically YCbCr (derived from sRGB or similar). For `jpegli` features, the internal processing might happen in a different space like XYB.
*   **Little CMS 2 (`lcms2`):** This library provides the core functionality for:
    *   Opening and parsing ICC profiles (`cmsOpenProfileFromMem`).
    *   Creating color transformations between profiles (`cmsCreateTransform`).
    *   Executing transformations on pixel data (`cmsDoTransform`).
    *   Querying profile information (color space, transfer function, etc. - `cmsGetColorSpace`, `cmsReadTag`, `cmsIsToneCurveLinear` etc.).
    *   Handling different data formats (e.g., `TYPE_RGB_FLT`, `TYPE_GRAY_FLT`, `TYPE_CMYK_FLT`).
*   **Preprocessing/Postprocessing (`ExtraTF`):** Some transfer functions (PQ, HLG, sRGB) require special handling *outside* the main LCMS transform. This involves applying the inverse transfer function before `cmsDoTransform` and the forward function afterwards. This is managed via the `ExtraTF` enum and corresponding Rust functions in `tf.rs` (ported from C++ `transfer_functions-inl.h`).
*   **Intensity Target:** A luminance value (in cd/m²) used to correctly map between relative (gamma-encoded) and absolute (PQ/HLG) color spaces.
*   **XYB Color Space:** A perceptually optimized color space used internally by Jpegli/JPEG XL. Conversion to/from XYB involves specific matrix multiplications and non-linear adjustments (gamma correction, offsets) ported from `xyb_transform.cc`.

## Key `lcms2` Functions Used (Identified from `jxl_cms.cc`):

*   `cmsOpenProfileFromMem`: To load ICC profiles from byte buffers.
*   `cmsCloseProfile`: To release profile handles.
*   `cmsCreateTransform`: To create the transformation pipeline between input and output profiles, specifying input/output formats (`TYPE_*_FLT`), intent (`INTENT_*`), and flags (e.g., `cmsFLAGS_NOOPTIMIZE`, `cmsFLAGS_COPY_ALPHA`).
*   `cmsDeleteTransform`: To release the transform handle.
*   `cmsDoTransform`: To execute the color transformation on batches of pixels.
*   `cmsGetColorSpace`: To determine the profile's color space (e.g., `cmsSigRgbData`, `cmsSigGrayData`, `cmsSigCmykData`).
*   `cmsGetPCS`: To get the profile connection space (usually `cmsSigXYZData` or `cmsSigLabData`).
*   `cmsReadTag`: Used extensively to read specific ICC tags like:
    *   `cmsSigRedColorantTag`, `cmsSigGreenColorantTag`, `cmsSigBlueColorantTag`: To get primaries.
    *   `cmsSigMediaWhitePointTag`: To get the white point.
    *   `cmsSigRedTRCTag`, `cmsSigGreenTRCTag`, `cmsSigBlueTRCTag`, `cmsSigGrayTRCTag`: To check transfer characteristics.
*   `cmsIsToneCurveLinear`: To check if a transfer curve tag represents a linear function.
*   `(Potentially others for specific tag types like parametric curves)`

## Transformation Paths

1.  **Standard JPEG (e.g., RGB/Gray/CMYK input -> YCbCr/YCCK output):**
    *   **Input:** Image data (`u8`) + Input ICC Profile (or assumed sRGB/Gray).
    *   Convert `u8` data to `f32` ([0.0, 1.0] range).
    *   **Preprocessing:** If input profile uses PQ/HLG/sRGB non-linear TF, apply the inverse TF (e.g., `tf::pq_display_from_encoded`) to get linear data relative to the profile's gamut.
    *   **LCMS Transform:** Use `cmsCreateTransform` with input profile and a standard *sRGB output profile* (`cmsCreate_sRGBProfile`) to convert input pixels to *linear sRGB*. The format would be `TYPE_RGB_FLT` or `TYPE_GRAY_FLT`.
    *   **Postprocessing:** None needed if outputting linear sRGB.
    *   **RGB -> YCbCr:** Apply the standard RGB to YCbCr conversion matrix (from `color_transform.cc`, `image_buffer.rs`) to the linear sRGB data.
    *   **Quantization/DCT/Entropy Coding:** Proceed with standard JPEG steps.
    *   *(Note: For CMYK input, the path involves `cmsCreateTransform` to sRGB, then RGB->YCbCr, and handling the K channel separately, resulting in YCCK).* `jxl_cms.cc` also applies a `100.0 - 100.0 * x` transform to CMYK floats *before* `cmsDoTransform` because LCMS expects 0=white, 100=max ink.

2.  **Jpegli XYB Mode (e.g., RGB/Gray input -> XYB internal -> YCbCr output):**
    *   **Input:** Image data (`u8`) + Input ICC Profile (or assumed sRGB/Gray).
    *   Convert `u8` data to `f32` ([0.0, 1.0] range).
    *   **Preprocessing:** If input profile uses PQ/HLG/sRGB non-linear TF, apply the inverse TF to get linear data relative to the profile's gamut.
    *   **LCMS Transform:** Use `cmsCreateTransform` with input profile and a standard *linear sRGB output profile* to convert input pixels to *linear sRGB*. Format `TYPE_RGB_FLT` or `TYPE_GRAY_FLT`.
    *   **Linear sRGB -> XYB:**
        *   Apply the Opsin Absorbance matrix (`opsin_params.h`, `xyb_transform.cc`) optionally premultiplied by `intensity_target`.
        *   Apply cube root.
        *   Apply XYB conversion (`(R-G)/2`, `(R+G)/2`, `B`).
        *   *(This happens in `LinearRGBRowToXYB`)*
    *   **XYB Scaling:** Apply affine scaling (`ScaleXYBRow`) to bring XYB values into a predictable range (often needed before quantization).
    *   **Quantization/DCT/Entropy Coding:** Perform modified JPEG steps adapted for XYB data.
    *   **Internal -> Output:** If the final JPEG needs to be standard YCbCr, an inverse transform from internal XYB back to YCbCr would be needed before writing markers, but `jpegli` likely encodes XYB coefficients directly when XYB mode is active.

3.  **Identity Transform (Input Profile == Output Profile or Simple Cases):**
    *   If the input and output profiles are effectively the same (e.g., both sRGB, or both grayscale with same gamma), the LCMS transform (`cmsDoTransform`) can be skipped (`skip_lcms = true` in C++).
    *   Preprocessing/Postprocessing might still be needed if the *target internal representation* differs from the profile's linear space (e.g., if internally using XYB, the sRGB -> XYB step is still required even if input/output profiles are both sRGB).
    *   Simple conversions like Grayscale -> RGB are handled directly without LCMS (`GrayscaleToRGB` in `color_transform.cc`).

## Implementation Notes (`jpeg-encoder/src/cms.rs`)

*   The `JxlCms` struct holds the `lcms2::Transform` (if needed), input/output channel counts, and flags/parameters for `ExtraTF` handling.
*   `JxlCms::new` determines `preprocess`, `postprocess`, and `skip_lcms` based on comparing input/output profile characteristics (obtained via `lcms2` tag reading) and the target internal space (standard linear sRGB or XYB).
*   `JxlCms::run_transform` orchestrates the process:
    1.  Call `tf::before_transform` (if `preprocess != ExtraTF::kNone`).
    2.  Apply CMYK adjustment if `channels_src == 4`.
    3.  Call `transform.transform_pixels` (if `!skip_lcms`).
    4.  Call `tf::after_transform` (if `postprocess != ExtraTF::kNone`).
*   XYB conversion logic will reside in a separate module (`xyb.rs` or similar), called after the LCMS step when XYB output is desired. 