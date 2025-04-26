# Test Data Generation and Integration Plan

This document outlines the plan for instrumenting the C++ `jpegli` library to generate detailed test data, processing that data, and integrating it into the Rust test suite for `src/jpegli/`.

## 1. Overview

The goal is to capture the exact inputs and outputs of key internal C++ functions within `libjpegli` during a standard execution of the `cjpegli` tool. This captured data will be formatted as JSON, processed by a Python script, and then loaded into Rust tests using the structures defined in `src/jpegli/tests/structs.rs`. This allows for high-fidelity, data-driven testing of the Rust reimplementation against the C++ reference.

## 2. C++ Instrumentation (`jpegli/lib/jpegli/`)

Modifications will be made *only* to `.cc` and `.h` files within `jpegli/lib/jpegli/` relative to the workspace root. No Makefiles or CMakeLists.txt will be touched.

**2.1. Target Functions:**

Instrument the *end* of the following C++ functions (and potentially helpers they call internally if state is complex):

-   `jpegli::SetQuantMatrices` (`quant.cc`)
-   `jpegli::InitQuantizer` (`quant.cc`)
-   `jpegli::PadInputBuffer` (or its internal call to `RowBuffer::PadRow`/`CopyRow` within `encode.cc`) - Capture row state before/after.
-   `jpegli::ComputePreErosion` (`adaptive_quantization.cc`)
-   `jpegli::FuzzyErosion` (`adaptive_quantization.cc`)
-   `jpegli::PerBlockModulations` (`adaptive_quantization.cc`)
-   `jpegli::ComputeAdaptiveQuantField` (`adaptive_quantization.cc`) - Capture final output field slice.
-   `jpegli::RGBToYCbCr` (within `color_transform.cc` or where it's called if more convenient)
-   `jpegli::DownsampleInputBuffer` (or its internal calls to specific downsamplers in `downsampling.cc`)
-   `jpegli::ComputeCoefficientBlock` (`dct-inl.h`)
-   `jpegli::ComputeTokensForBlock` (`entropy_coding-inl.h`)
-   Potentially: `jpegli::WriteIccProfile` (`encode.cc`) - Capture arguments, marker count/lengths.

**2.2. Headers and Globals:**

-   Include `<mutex>`, `<sstream>`, `<fstream>`, `<vector>`, `<iomanip>`, `<string>`, `<cmath>`, `<type_traits>`, `<cstdlib>` (`getenv`), `<atomic>` where needed.
-   Define a global `std::mutex g_testdata_mutex;` (e.g., in `common.cc` or a new `test_data_gen.cc`).
-   Define helper functions/variables (e.g., in a new `test_data_gen.h` included where needed):
    ```c++
    #include <cstdlib>
    #include <string>
    #include <atomic>

    // Default definition: enable instrumentation code compilation.
    // Can be overridden by build system later (e.g., -DENABLE_RUST_TEST_INSTRUMENTATION=0).
    #ifndef ENABLE_RUST_TEST_INSTRUMENTATION
    #define ENABLE_RUST_TEST_INSTRUMENTATION 1
    #endif

    // Cache the result of the environment variable check.
    inline bool IsRustTestDataEnabled() {
        // Use atomic flag for thread-safe initialization check.
        static std::atomic<int> checked = {-1}; // -1: unchecked, 0: false, 1: true
        int current_value = checked.load(std::memory_order_relaxed);

        if (current_value == -1) {
            const char* env_p = std::getenv("GENERATE_RUST_TEST_DATA");
            bool enabled = (env_p != nullptr && std::string(env_p) == "1");
            current_value = enabled ? 1 : 0;
            // Atomically try to set it. Doesn't matter if another thread races and sets it first.
            int expected = -1;
            checked.compare_exchange_strong(expected, current_value, std::memory_order_relaxed);
            // Return the value we determined, even if another thread set it concurrently.
            return enabled;
        }
        return current_value == 1;
    }
    ```
-   Instrumentation logic within functions should be guarded by `#if ENABLE_RUST_TEST_INSTRUMENTATION` and *also* use the runtime check `if (IsRustTestDataEnabled())` (or have it checked within the write helper).

**2.3. JSON Formatting Helpers (C++):**

-   Create helper functions (potentially in `test_data_gen.cc` or similar) to format data as JSON strings within a `std::stringstream`.
    -   `std::string format_json_string(const std::string& s)`: Escapes quotes, backslashes, etc.
    -   `std::string format_json_bool(bool b)`: Returns "true" or "false".
    -   `std::string format_json_float(float f)`: Uses standard float format for JSON (e.g., sufficient precision via `std::scientific`, `std::setprecision`).
    -   `template<typename T> std::string format_json_vec_1d(const std::vector<T>& v, std::function<std::string(T)> formatter)`: Formats a 1D vector as `[val1, val2, ...]$.
    -   `template<typename T> std::string format_json_vec_2d(...)`: Formats a 2D vector or slice.
    -   `std::string format_json_rowbufferslice(...)`: Takes `RowBuffer<float>`, start/count rows/cols, stride and formats the `RustRowBufferSliceF32` JSON object.
    -   `std::string format_json_blockf32(...)`, `format_json_blocki32(...)`: Formats 64-element block data.
    -   `std::string format_json_token(...)`, `format_json_vec_token(...)`: Formats `Token` structs.
    -   `std::string format_json_component_info(...)`: Formats `ComponentInfoMinimal`.
    -   `std::string format_json_option_vec_u16(...)`: Formats `Vec<Option<Vec<u16>>>`.

**2.4. Synchronized File Writing Helper (C++):**

Create a helper function (e.g., in `test_data_gen.cc` or header if inline) to handle checking, locking, and writing:

```c++
#include <mutex>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio> // For fprintf

// Assumes IsRustTestDataEnabled() and g_testdata_mutex are accessible
extern std::mutex g_testdata_mutex;

inline void WriteTestDataJsonLine(const std::string& filename_base, const std::stringstream& ss) {
    // Runtime check: Only proceed if generation is enabled via env var.
    if (!IsRustTestDataEnabled()) {
        return;
    }

    // Acquire lock and write
    std::lock_guard<std::mutex> lock(g_testdata_mutex);
    std::string filename = filename_base + ".testdata";
    // Assuming execution happens in reference_dct/ as planned
    std::ofstream outfile(filename, std::ios::app);
    if (outfile.is_open()) {
        // Append comma and newline for easier aggregation later by Python
        outfile << ss.str() << "," << std::endl;
    } else {
        // Use fprintf for stderr as it's simpler in this context.
        fprintf(stderr, "Error opening testdata file %s\n", filename.c_str());
    }
}
```

**2.5. Instrumentation Pattern:**

At the end of each target function `Func`:

```c++
#if ENABLE_RUST_TEST_INSTRUMENTATION
// Compile-time guard for the instrumentation block

// Runtime check happens inside WriteTestDataJsonLine, but guarding
// the potentially expensive state capture and stringstream build avoids
// unnecessary work if disabled at runtime.
if (IsRustTestDataEnabled()) {
    // 1. Capture Input State (if needed, before modification)
    //    Deep copy relevant input buffers/structs if they are modified in place.

    // --- Function Body Executes ---

    // 2. Capture Output State
    //    Copy output data (return values, modified buffers/structs).

    // 3. Build JSON String using stringstream and helpers
    std::stringstream ss;
    ss << "{";
    ss << "\"test_type\": \"FuncNameTest\", "; // Identify the struct type
    ss << "\"config_some_flag\": " << format_json_bool(cinfo->master->some_flag) << ", ";
    ss << "\"input_arg1\": " << input_arg1 << ", ";
    ss << "\"input_buffer_slice\": " << format_json_rowbufferslice(...) << ", ";
    // ... all other config_, input_, and expected_ fields ...
    ss << "\"expected_output_buffer\": " << format_json_blocki32(...) << " "; // No trailing comma
    ss << "}"; // No newline here, add it during file write

    // 4. Write JSON string to file using helper (which includes lock & check)
    WriteTestDataJsonLine("FuncName", ss);
}
#endif // ENABLE_RUST_TEST_INSTRUMENTATION
```

**2.6. Data Types & Buffers:**

-   Ensure C++ types map correctly to Rust types defined in `src/jpegli/tests/structs.rs`.
-   Capture buffer *slices* (`RustRowBufferSliceF32`) instead of entire large buffers where appropriate, providing `start_row`, `num_rows`, `start_col`, `num_cols`, `stride` for context.

## 3. Docker Build and Execution

-   Use the existing `jpegli/docker/Dockerfile` to build `cjpegli`. No special build flags needed for instrumentation (it uses runtime check).
-   The Python orchestration script will run `docker run ...` commands.
-   Mount the `reference_dct/` directory from the host (Windows) into the container at `/data`.
-   Mount the `src/` directory from the host to `/src_out` inside the container so the python script can write the final JSON files to the correct location (`/src_out/jpegli/tests/json/`).
-   The `cjpegli` command will be executed with its working directory set to the mounted `/data` (`reference_dct/` on host) inside the container.
-   Pass the environment variable to enable instrumentation when running the container: `-e GENERATE_RUST_TEST_DATA=1`.
-   Example run command within Python script:
    ```bash
    docker run --rm -e GENERATE_RUST_TEST_DATA=1 \
           -v /c/Users/user/path/to/jpeg-encoder/reference_dct:/data \
           -v /c/Users/user/path/to/jpeg-encoder/src:/src_out \
           jpegli-builder-image \
           bash -c "cd /data && ./cjpegli [args...] input.png output.jpg"
    ```
    (Or potentially calling the python script inside the container).

## 4. Python Orchestration Script (`reference_dct/generate_test_data.py`)

This script manages the data generation process.

**4.1. Location:** Resides in `reference_dct/` on the host.

**4.2. Responsibilities:**

-   **Define Test Cases:** Specify different `cjpegli` arguments and input files.
-   **Clean Previous Data:** Delete existing `*.testdata` files in `reference_dct/`.
-   **Run `cjpegli` via Docker:** For each test case, execute the `docker run ...` command ensuring `-e GENERATE_RUST_TEST_DATA=1` is set.
-   **Collect Raw Data:** Find all `FuncName.testdata` files in `reference_dct/`.
-   **Process Data:**
    -   Read each `*.testdata` file line by line. Each line *contains* a complete JSON object followed by a comma.
    -   Validate each line (potentially after removing the trailing comma if needed by validator, though `json.loads` might tolerate it if inside an array later).
    -   Group the loaded JSON objects by their `test_type` field.
-   **Aggregate and Write JSON:**
    -   For each test type:
        -   Read all lines from the corresponding `FuncName.testdata` file.
        -   If lines exist:
            -   Join all lines together into a single string.
            -   Remove the trailing comma and newline from the *last* line in the joined string.
            -   Prepend `[\n` and append `\n]` to the modified string to form a valid JSON array.
        -   Else (no lines): Write `[]`.
    -   Determine the correct output path relative to the workspace root (e.g., `src/jpegli/tests/json/FuncNameTest.json`).
    -   Write the final JSON array string to the corresponding output file.

## 5. Rust Test Integration (`src/jpegli/tests/`)

**5.1. `src/jpegli/tests/structs.rs`:** Contains Rust struct definitions with `#[derive(Deserialize)]`.

**5.2. `src/jpegli/tests/testdata.rs` (New Module):**

-   Uses `include_str!` to embed JSON file contents.
    ```rust
    const SET_QUANT_MATRICES_JSON: &str = include_str!("json/SetQuantMatricesTest.json");
    // ... etc
    ```
-   Provides lazy-loaded, parsed test data using `once_cell::sync::Lazy` and `serde_json::from_str`.
    ```rust
    use once_cell::sync::Lazy;
    use crate::jpegli::tests::structs::*;

    pub static SET_QUANT_MATRICES_TESTS: Lazy<Vec<SetQuantMatricesTest>> = Lazy::new(|| {
        serde_json::from_str(SET_QUANT_MATRICES_JSON).expect("...")
    });
    // ... etc
    ```

**5.3. Test Modules (e.g., `src/jpegli/tests/quant_test.rs`):**

-   Import test data and structs.
-   Iterate through static test data (`testdata::SET_QUANT_MATRICES_TESTS.iter()`).
-   For each `test_case`:
    -   Set up Rust state using `test_case.config_*`.
    -   Prepare inputs using `test_case.input_*`.
    -   Call the Rust function.
    -   Assert results against `test_case.expected_*`.

## 6. File Structure Summary

```
jpeg-encoder/              <-- Workspace Root
├── jpegli/
│   ├── lib/
│   │   └── jpegli/      <-- C++ Instrumentation here (.cc, .h)
│   └── docker/
│       └── Dockerfile   <-- Used for build (no changes)
├── src/
│   └── jpegli/
│       ├── tests/
│       │   ├── json/
│       │   │   ├── SetQuantMatricesTest.json  <-- Generated by Python script
│       │   │   └── ...
│       │   ├── structs.rs
│       │   ├── testdata.rs
│       │   └── quant_test.rs
│       │   └── ...
│       └── ... (Rust source code)
├── reference_dct/          <-- Working directory for cjpegli run (mounted as /data)
│   ├── generate_test_data.py
│   ├── input.png
│   ├── FuncName.testdata  <-- Raw JSON Lines output from C++ (temporary)
│   └── ...
└── ...
```

## 7. Workflow Summary

1.  Modify C++ code in `jpegli/lib/jpegli/` with instrumentation guarded by `#if ENABLE_RUST_TEST_INSTRUMENTATION`. The core logic inside the guard should call `WriteTestDataJsonLine`, which contains the runtime `if (IsRustTestDataEnabled())` check.
2.  Build the Docker image using `jpegli/docker/Dockerfile` (no special build args needed by default).
3.  Place `generate_test_data.py` and test input images in `reference_dct/`.
4.  Run `python reference_dct/generate_test_data.py` on the host.
    -   Script executes `docker run ... -e GENERATE_RUST_TEST_DATA=1 ... cjpegli ...`.
    -   `cjpegli` checks env var, calls `WriteTestDataJsonLine` which appends JSON lines (with trailing comma) to `*.testdata` files in `/data` (`reference_dct/`).
    -   Script aggregates lines (removing last comma, adding `[]`) into final JSON arrays in `src/jpegli/tests/json/`.
5.  Run `cargo test` from the `jpeg-encoder` workspace root.
6.  (Later) To disable instrumentation at compile time, modify Docker build: `docker build --build-arg CXX_EXTRA_FLAGS="-DENABLE_RUST_TEST_INSTRUMENTATION=0" ...`.

</rewritten_file> 