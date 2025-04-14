'''
# `lcms2` Crate API Usage in `jpeg-encoder`

This document outlines the key APIs from the `lcms2` Rust crate (a binding to Little CMS 2) that are relevant to the color management implementation within the `jpeg-encoder` project, primarily in `src/cms.rs`.

## Core Concepts

*   **Context (`lcms2::Context`):** Represents an LCMS session. Often implicitly handled by the crate, but explicit contexts (`Context::new()`) can be created for thread-safety or advanced features.
*   **Profiles (`lcms2::Profile`):** Encapsulate ICC color profile data.
*   **Transforms (`lcms2::Transform`):** Precomputed pipelines for converting between profiles.
*   **Tone Curves (`lcms2::ToneCurve`):** Define gamma or transfer characteristics.
*   **Tags (`lcms2::Tag`, `lcms2::TagSignature`):** Allow access to specific data blocks within an ICC profile.
*   **Signatures & Enums:** Standard ICC identifiers (e.g., `ColorSpaceSignature`, `Intent`, `PixelFormat`).
*   **Error Handling:** Functions typically return `lcms2::LCMSResult<T>` (aliased to `Result<T, lcms2::Error>`).

## `lcms2::Profile`

Represents an ICC profile.

*   **`Profile::new_icc(data: &[u8]) -> LCMSResult<Self>`**
    *   Creates a profile handle from raw ICC data bytes.
    *   *Usage (`cms.rs`):* Used in `JxlCms::new` and `set_fields_from_icc` to load input/output profiles.

*   **`Profile::new_srgb() -> Self`**
    *   Creates a profile handle for the standard sRGB color space.
    *   *Usage (`cms.rs`):* Used in `ColorProfile::srgb()` factory and tests.

*   **`Profile::new_gray_context(context: impl AsRef<Ctx>, white_point: &CIExyY, curve: &ToneCurve) -> LCMSResult<Self>`**
    *   Creates a grayscale profile given a white point and a tone curve.
    *   *Usage (`cms.rs`):* Used in `ColorProfile::gray_gamma22()` factory.

*   **`Profile::new_rgb_context(context: impl AsRef<Ctx>, white_point: &CIExyY, primaries: &CIExyYTRIPLE, transfer_functions: &[&ToneCurve]) -> LCMSResult<Self>`**
    *   Creates an RGB profile from primaries, white point, and transfer curves (one per channel, often duplicated).
    *   *Usage (`cms.rs`):* Used in `ColorProfile::linear_srgb()` factory.

*   **`profile.color_space() -> ColorSpaceSignature`**
    *   Returns the color space defined in the profile header (e.g., `RgbData`, `GrayData`).
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to determine channel count and processing paths.

*   **`profile.profile_class() -> ProfileClassSignature`**
    *   Returns the class of the profile (e.g., `Display`, `Input`, `Output`).
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to infer if certain tags (like primaries) are likely present.

*   **`profile.rendering_intent() -> Intent`**
    *   Returns the default rendering intent stored in the profile header.
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to populate `ColorEncodingInternal`.

*   **`profile.tag_data(sig: TagSignature) -> LCMSResult<Tag>`**
    *   Reads the raw data (`Tag`) associated with a specific tag signature.
    *   *Usage (`cms.rs`):* Used extensively in `set_fields_from_icc` to get white point, colorants, and TRC data.

*   **`profile.save_to_icc_bytes() -> LCMSResult<Vec<u8>>`**
    *   Serializes the profile handle back into ICC data bytes.
    *   *Usage (`cms.rs`):* Used in profile factory functions (`srgb`, `linear_srgb`, `gray_gamma22`) to store the generated ICC data.

## `lcms2::Transform`

Represents a color transformation pipeline.

*   **`Transform::new_flags(input: &Profile, input_format: PixelFormat, output: &Profile, output_format: PixelFormat, intent: Intent, flags: Flags) -> LCMSResult<Self>`**
    *   Creates a transform pipeline between input and output profiles for the specified data formats, rendering intent, and flags.
    *   *Usage (`cms.rs`):* Used in `JxlCms::new` to create the core LCMS transformation.

*   **`transform.transform_pixels(src: &[InputPixelFormat], dst: &mut [OutputPixelFormat])`**
    *   Executes the precomputed transform on batches of pixel data. The slice types (`InputPixelFormat`, `OutputPixelFormat`) must match the `PixelFormat`s used during creation (e.g., `f32` for `PixelFormat::RGB_FLT`).
    *   *Usage (`cms.rs`):* Used in `JxlCms::run_transform` to perform the actual color conversion.

## `lcms2::ToneCurve`

Represents transfer characteristics (gamma).

*   **`ToneCurve::new(gamma: f64) -> Self`**
    *   Creates a simple power-law gamma curve (e.g., `ToneCurve::new(2.2)`).
    *   *Usage (`cms.rs`):* Used in factories like `ColorProfile::gray_gamma22()` and potentially for creating linear curves (`ToneCurve::new(1.0)`).

*   **`ToneCurve::new_parametric(curve_type: u16, params: &[f64]) -> LCMSResult<Self>`**
    *   Creates a curve based on standard ICC parametric definitions (e.g., type 4 for sRGB).
    *   *Usage (`cms.rs`):* Not directly used yet, but could be used for creating profiles with specific standard curves.

*   **`curve.is_linear() -> bool`**
    *   Checks if the curve represents gamma = 1.0.
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to detect linear transfer functions.

*   **`curve.estimated_gamma() -> f64`**
    *   Provides a numerical estimate of the gamma value. Useful for table-based or non-standard curves, but can be inaccurate.
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` as a fallback to guess the gamma if parametric/linear checks fail.

*   **`curve.type_signature() -> TagTypeSignature`**
    *   Returns the type of the underlying tag data (e.g., `SigCurveType`, `SigParametricCurveType`).
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to differentiate between simple table curves and parametric curves.

*   **`curve.parametric_type() -> u16`**
    *   For parametric curves, returns the standard ICC type identifier (e.g., 1-3 for gamma, 4 for sRGB IEC 61966-2.1, potentially 5 for PQ Rec. 2100, 6 for HLG Rec. 2100).
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to specifically detect sRGB, PQ, HLG, or gamma-based parametric curves.

*   **`curve.parametric_params() -> Vec<f64>`**
    *   Returns the parameters associated with a parametric curve (e.g., gamma value for types 1-3).
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to get the gamma value from parametric types 1-3.

## `lcms2::Tag`

Represents raw data read from a profile tag.

*   **`Tag::read_xyz(tag: &Tag) -> LCMSResult<CIEXYZ>`**
    *   Interprets the tag data as a `CIEXYZ` value.
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to read white point (`MediaWhitePoint`) and RGB colorant (`RedColorant`, etc.) tags.

*   **`Tag::read_curve(tag: &Tag) -> LCMSResult<ToneCurve>`**
    *   Interprets the tag data as a `ToneCurve`.
    *   *Usage (`cms.rs`):* Used in `set_fields_from_icc` to read TRC tags (`GrayTRC`, `GreenTRC`, etc.).

## Important Enums and Structs

*   **`ColorSpaceSignature`:** Identifiers like `RgbData`, `GrayData`, `CmykData`, `XyzData`, `LabData`.
*   **`ProfileClassSignature`:** Identifiers like `Input`, `Display`, `Output`, `ColorSpace`.
*   **`TagSignature`:** Identifiers for specific tags like `MediaWhitePoint`, `RedColorant`, `GreenColorant`, `BlueColorant`, `GrayTRC`, `GreenTRC`, `Cicp`.
*   **`TagTypeSignature`:** Identifiers for the *type* of data within a tag, like `SigCurveType`, `SigParametricCurveType`, `SigXYZType`.
*   **`Intent`:** Rendering intents like `Perceptual`, `RelativeColorimetric`, `Saturation`, `AbsoluteColorimetric`.
*   **`PixelFormat`:** Defines channel order, data type, and packing for `Transform::new_flags` and `transform_pixels`. Examples: `RGB_FLT`, `GRAY_FLT`, `CMYK_FLT`, `RGB_8`.
*   **`CIEXYZ`, `CIExyY`, `CIExyYTRIPLE`:** Structs for representing CIE color values and chromaticities.
*   **`Flags`:** Bitflags for modifying behavior (e.g., `NOOPTIMIZE`, `BLACKPOINTCOMPENSATION`, `COPY_ALPHA`).

## Error Handling

Most functions that interact with the LCMS C library return `LCMSResult<T>`, which is `Result<T, lcms2::Error>`. Errors should be handled appropriately, often by converting them into `EncoderError::CmsError`.
```'' 