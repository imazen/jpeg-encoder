# Test Data Generation and Integration Plan

This document outlines the plan for instrumenting the C++ `jpegli` library to generate detailed test data, processing that data, and integrating it into the Rust test suite for `src/jpegli/`.

## 1. Overview

The goal is to capture the exact inputs and outputs of key internal C++ functions within `libjpegli` during a standard execution of the `cjpegli` tool. This captured data will be formatted as JSON line-delimited records, processed by a Python script into final JSON arrays, and then loaded into Rust tests using the structures defined in `src/jpegli/tests/structs.rs`. This allows for high-fidelity, data-driven testing of the Rust reimplementation against the C++ reference.

## 2. C++ Instrumentation (`jpegli/lib/jpegli/`)

Modifications will be made *only* to `.cc` and `.h` files within `jpegli/lib/jpegli/` relative to the workspace root. `CMakeLists.txt` files will be updated to include the new source file.

**2.1. Target Functions:**

Instrument the *end* of the following C++ functions (and potentially helpers they call internally if state is complex):

-   `jpegli::SetQuantMatrices` (`quant.cc`) - *Done*
-   `jpegli::InitQuantizer` (`quant.cc`)
-   `jpegli::PadInputBuffer` (within `encode.cc`) - Capture relevant row state before/after padding.
-   `jpegli::ComputePreErosion` (`adaptive_quantization.cc`)
-   `jpegli::FuzzyErosion` (`adaptive_quantization.cc`)
-   `jpegli::PerBlockModulations` (`adaptive_quantization.cc`)
-   `jpegli::ComputeAdaptiveQuantField` (`adaptive_quantization.cc`) - Capture final output field slice.
-   `jpegli::RGBToYCbCr` (`color_transform.cc`) - Capture input/output rows.
-   `jpegli::DownsampleInputBuffer` (`downsampling.cc`) - Capture input/output buffer slices.
-   `jpegli::ComputeCoefficientBlock` (`dct-inl.h`) - Capture block data in/out.
-   `jpegli::ComputeTokensForBlock` (`entropy_coding-inl.h`) - Capture coefficients in, tokens out.
-   `jpegli::WriteIccProfile` (`encode.cc`) - Optional: Capture arguments, marker count/lengths.

**2.2. Control Mechanism:**

-   **Compile-Time Guard:** A C++ macro `ENABLE_RUST_TEST_INSTRUMENTATION` (defaulting to 1 via `#define` in `test_data_gen.h`) guards all instrumentation code blocks (`#if ENABLE_RUST_TEST_INSTRUMENTATION ... #endif`). This allows disabling instrumentation completely at compile time if needed (e.g., via `-DENABLE_RUST_TEST_INSTRUMENTATION=0` in build flags).
-   **Runtime Check:** An environment variable `GENERATE_RUST_TEST_DATA` (checked for value `"1"`) controls whether the instrumentation *actually writes data* during execution. This check is performed efficiently using a thread-safe, cached approach within the `WriteTestDataJsonLine` helper.

**2.3. Headers and Globals:**

-   Create `jpegli/lib/jpegli/test_data_gen.h`:
    -   Includes ALL implementations for the test data generation code.

**2.4. JSON Formatting Helpers (C++):**

-   Implement helper functions in `test_data_gen.h` to format C++ data types and structures into standard JSON strings (bool, float, string, vectors, `RowBuffer` slices, blocks, tokens, component info, etc.). Floats should use sufficient precision.

**2.5. Synchronized File Writing Helper (C++):**

-   Define `WriteTestDataJsonLine(const std::string& filename_base, const std::stringstream& ss)` in `test_data_gen.h`.
-   This function first checks `IsRustTestDataEnabled()`. If disabled, it returns immediately.
-   If enabled, it acquires a lock on `g_testdata_mutex`.
-   Constructs the output filename (e.g., `FuncName.testdata`) relative to the current working directory (which will be `/data` mapped to `reference_dct/`).
-   Opens the file in append mode (`std::ios::app`).
-   Writes the complete JSON object string from the `stringstream`, followed immediately by a **comma and a newline** (`<< ss.str() << "," << std::endl;`).
-   Includes basic error handling for file opening.

**2.6. Instrumentation Pattern:**

-   At the end of each target C++ function `Func`:
    ```c++
    #if ENABLE_RUST_TEST_INSTRUMENTATION
    // Potentially guard expensive state capture with the runtime check:
    // if (IsRustTestDataEnabled()) { ... capture state ... }

    // --- Original Function Body Executes --- 

    // Guard the main instrumentation logic (string building, writing)
    if (IsRustTestDataEnabled()) {
        // Capture output state
        std::stringstream ss;
        ss << "{";
        ss << "\"test_type\": \"FuncNameTest\", ";
        // ... serialize config_, input_, expected_ fields ...
        ss << "}";
        WriteTestDataJsonLine("FuncName", ss);
    }
    #endif // ENABLE_RUST_TEST_INSTRUMENTATION
    ```

**2.7. Build System Integration (CMake):**

-   None. Header only include, no compilation units.

## 3. Docker Build and Execution

-   Use the existing `jpegli/docker/Dockerfile`, potentially adding build dependencies like `libpng-dev` to the `builder` stage if needed for input formats. Never use --no-cache, it is NEVER the problem. NEVER just simulate the build or script - ALWAYS run them for real. C++ errors are not fixable via --no-cache - you probably need to inventory the codebase properly to find the actual issue or path problems. 
-   Build the image using the `jpegli` directory as the context:
    ```bash
    # Run from workspace root
    docker build -t jpegli-builder-image -f jpegli/docker/Dockerfile jpegli
    ```
    *   No special build args are needed by default (`ENABLE_RUST_TEST_INSTRUMENTATION` defaults to 1).
    *   To build *without* instrumentation code compiled in: `docker build --build-arg CXX_EXTRA_FLAGS="-DENABLE_RUST_TEST_INSTRUMENTATION=0" ...`
-   Run the container using the Python script.

## 4. Python Orchestration Script (`reference_dct/generate_test_data.py`)

-   **Location:** `reference_dct/generate_test_data.py`.
-   **Configuration:** Defines `TEST_PATTERNS` (list of tuples: `(group_name, input_glob_pattern, cjpegli_args_list)`) and `TARGET_FUNCTIONS` (list should include all instrumented functions: `["SetQuantMatrices", "InitQuantizer", ...]`).
-   **Responsibilities:**
    1.  Clean old `*.testdata` files from `reference_dct/` and old `*.json` files from `src/jpegli/tests/json/`.
    2.  Iterate through `TEST_PATTERNS`:
        *   Use `glob.glob` to find input files matching the pattern relative to `reference_dct/`.
        *   For each found input file:
            *   Construct output JPG path within a subdirectory named after `group_name` inside `reference_dct/`. Create this subdir.
            *   Determine relative input path and output *filename* (no subdir) for the container.
            *   Construct the `docker run` command:
                *   Use `--rm`.
                *   Set environment variable: `-e GENERATE_RUST_TEST_DATA=1`.
                *   Mount volumes:
                    *   `-v <abs_host_path_to_reference_dct>:/data`
                    *   `-v <abs_host_path_to_src>:/src_out`
                *   Set working directory: `-w /data`.
                *   Specify image: `jpegli-builder-image`.
                *   Arguments passed to entrypoint: `INPUT_PATH OUTPUT_FILENAME OPTIONS...` (e.g., `testdata/img.png out.jpg --distance 1.0`).
            *   Execute the `docker run` command.
    3.  After all Docker runs complete, iterate through `TARGET_FUNCTIONS`:
        *   Construct paths: `testdata_file = reference_dct/FuncName.testdata`, `json_outfile = src/jpegli/tests/json/FuncNameTest.json`.
        *   Read all lines from `testdata_file` if it exists.
        *   If content exists: Join lines, remove the *final* trailing comma/whitespace from the entire joined string, prepend `[\n`, append `\n]`, validate with `json.loads` (optional but recommended).
        *   If no content or file missing: Set output string to `[]`.
        *   Write the resulting JSON array string to `json_outfile`.

## 5. Rust Test Integration (`src/jpegli/tests/`)

-   **`structs.rs`:** Contains Rust struct definitions matching the generated JSON structure, with `#[derive(Deserialize)]`.
-   **`testdata.rs`:**
    *   Uses `include_str!("json/FuncNameTest.json")` for each test type (e.g., `include_str!("json/SetQuantMatricesTest.json")`).
    *   Uses `once_cell::sync::Lazy` and `serde_json::from_str` to parse JSON into static `Vec<FuncNameTest>` variables (e.g., `static SET_QUANT_MATRICES_TESTS: Lazy<Vec<SetQuantMatricesTest>> = ...`).
-   **Test Modules (e.g., `quant_test.rs`):**
    *   Import data from `crate::jpegli::tests::testdata` and structs from `crate::jpegli::tests::structs`.
    *   Use the static vectors (e.g., `testdata::SET_QUANT_MATRICES_TESTS.iter()`) to drive tests.
    *   Set up Rust state from `test_case.config_*`, call function with `test_case.input_*`, assert against `test_case.expected_*`.

## 6. File Structure Summary

```
jpeg-encoder/              <-- Workspace Root
├── jpegli/
│   ├── lib/
│   │   └── jpegli/      <-- C++ Instrumentation here (.cc, .h)
│   └── docker/
│       └── Dockerfile   <-- Used for build
├── src/
│   └── jpegli/
│       ├── tests/
│       │   ├── json/
│       │   │   ├── SetQuantMatricesTest.json  <-- Generated by Python
│       │   │   └── ...
│       │   ├── structs.rs
│       │   ├── testdata.rs
│       │   └── quant_test.rs
│       │   └── ...
│       └── ... (Rust source code)
├── reference_dct/          <-- Working dir for cjpegli run (mounted as /data)
│   ├── generate_test_data.py
│   ├── input.png
│   ├── FuncName.testdata  <-- Raw JSON Lines (comma-terminated) output from C++
│   └── ...
└── ...
```

## 7. Workflow Summary

1.  Modify/Add C++ instrumentation code in `jpegli/lib/jpegli/`, guarded by `#if ENABLE_RUST_TEST_INSTRUMENTATION`.
2.  Ensure `test_data_gen.cc` is added unconditionally to sources in `jpegli/lib/jxl_lists.cmake`.
3.  Build Docker image (no special build args needed by default): `docker build -t jpegli-builder-image -f jpegli/docker/Dockerfile jpegli`.
4.  Place/update `generate_test_data.py` and input images in `reference_dct/`.
5.  Run Python script from workspace root: `python reference_dct/generate_test_data.py`.
    *   Script runs `docker run -e GENERATE_RUST_TEST_DATA=1 ...`.
    *   `cjpegli` inside container checks env var, calls `WriteTestDataJsonLine` if enabled, appending comma-terminated JSON lines to `*.testdata` in `reference_dct/`.
    *   Script post-processes `*.testdata` files into valid JSON arrays in `src/jpegli/tests/json/`.
6.  Run Rust tests: `cargo test`.