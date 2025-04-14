# Jpegli Porting Status to jpeg-encoder

This document tracks the porting status of C++ components from the `jpegli` library (`lib/` and `tools/`) to the Rust `jpeg-encoder` crate.

**Jpegli Components Not Fully Ported to `jpeg-encoder`**

| Category          | Jpegli Component(s) (`lib/` or `tools/`)                      | Status / Notes                                                                 | Rust Target (`jpeg-encoder/src/`)    |
| :---------------- | :------------------------------------------------------------ | :----------------------------------------------------------------------------- | :----------------------------------- |
| **Core Encoding** | `downsample.cc/.h`                                            | **Not Ported:** Logic for chroma downsampling.                               | *(Needs implementation)*             |
|                   | `adaptive_quantization.cc/.h`                               | **In Progress:** Rust skeleton exists, but core calculations are placeholders. | `adaptive_quantization.rs`         |
|                   | `dct-inl.h` (Float DCT specifics)                             | **Blocked:** Rust scalar float DCT ported, but tests panic. SIMD needed.       | `fdct.rs`, `avx2/fdct.rs`            |
| **Color**         | `color_transform.cc/.h` (XYB part)                            | **Not Started:** XYB color space transform logic.                              | *(Needs implementation, likely modifying `image_buffer.rs`)* |
|                   | `lib/cms/` (Entire directory)                                 | **Not Ported:** Color Management System (transfer functions, tone mapping).    | *(Needs implementation)*             |
| **Decoding**      | `decode*.cc/.h`, `idct.cc/.h`, `render.cc/.h`, `upsample.cc/.h`, `source_manager.cc/.h`, `destination_manager.cc/.h`, `lib/extras/dec/` | **Not Ported:** The entire decoding pipeline is out of scope for `jpeg-encoder`. | *(Out of Scope)*                     |
| **Input/Output**  | `input.cc/.h` (HDR/float input handling)                      | **Not Started:** Handling non-`u8` input types.                                | *(Needs extension in `image_buffer.rs` or `lib.rs`)* |
|                   | `lib/extras/enc/` (Various image format encoders)             | **Not Ported:** Encoding helpers for PNM, PGX, JPG, EXR, etc.                   | *(Out of Scope)*                     |
|                   | `lib/extras/dec/` (Various image format decoders)             | **Not Ported:** Decoding helpers for various formats.                            | *(Out of Scope)*                     |
| **Utilities**     | `lib/base/` (Most components)                                 | **Not Ported:** Low-level utilities (memory, math, OS, parallelism).         | *(Relies on Rust std/crates)*      |
|                   | `lib/extras/` (Butteraugli, Image/Ops, Convolve, etc.)        | **Not Ported:** Various extras like Butteraugli metric, image ops, convolution helpers (needed for full AQ). | *(Needs implementation as needed)*   |
|                   | `simd.cc/.h`                                                  | **Partial:** Basic SIMD abstraction exists, but specific jpegli SIMD code needs porting (beyond float DCT). | `avx2.rs`, *(Needs more)*           |
| **Testing**       | `*_test.cc`, `test_*.h`, `fuzztest.h`, `libjpeg_wrapper.cc`     | **Not Ported:** C++ unit tests, fuzzers, and test utilities.                   | *(Rust uses its own tests)*          |
| **Tools**         | `tools/` (Entire directory except build files)                | **Not Ported:** Command-line tools (`cjpegli`, `djpegli`), benchmarks, optimizers, fuzzers, metrics (ssimulacra2), JNI wrappers. | *(Out of Scope for core crate)*    |

**Summary of Key Missing Encoder Features:**

1.  **Fully functional Adaptive Quantization:** Requires porting the convolution and analysis logic from `adaptive_quantization.cc` and potentially `lib/extras/convolve*.cc`.
2.  **XYB Color Transform:** Requires porting logic from `color_transform.cc` and `lib/extras/xyb_transform.cc`.
3.  **Working Float DCT (including SIMD):** Requires debugging the panic and porting SIMD versions from `dct-inl.h`.
4.  **Downsampling:** Porting `downsample.cc`.
5.  **HDR Input:** Extending input handling. 