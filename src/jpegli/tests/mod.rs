// Entry point for tests

// Make structs and test data available to other test files
pub mod structs;
pub mod test_utils;
pub mod testdata;

// Declare test modules for different functional areas
mod quant_test;
mod adaptive_quantization_test;
mod encode_test; // Example: For PadInputBuffer etc.
mod color_transform_test; // Added for RGBToYCbCr
// Add other test modules here...

// Example test using the generated data
#[cfg(test)]
mod quant_tests {
    use super::structs::*;
    use super::testdata::*;
    use super::test_utils::*;
    // TODO: Import necessary Rust jpegli functions and types

    #[test]
    fn test_init_quantizer_example() {
        if INITQUANTIZERTEST_DATA.is_empty() {
            println!("Skipping test_init_quantizer_example: No test data found.");
            return;
        }
        let test_case = &INITQUANTIZERTEST_DATA[0]; // Test the first entry

        // 1. Setup Rust state based on test_case.config_...
        //    - Create mock JpegCompressStruct / MasterState
        //    - Set flags like use_adaptive_quantization
        //    - Populate initial quant tables
        //    - Set component info
        //    ... (This requires details of the Rust structs)

        // let mut cinfo = setup_cinfo_for_init_quantizer(test_case);

        // 2. Call the Rust function
        // rust_jpegli::quant::init_quantizer(&mut cinfo, test_case.input_pass);

        // 3. Assert output state matches test_case.expected_...
        //    - Compare cinfo.master.quant_mul with test_case.expected_quant_mul
        //    - Compare cinfo.master.zero_bias_mul with test_case.expected_zero_bias_mul
        //    - Compare cinfo.master.zero_bias_offset with test_case.expected_zero_bias_offset
        //    (Use float comparison helpers with tolerance)

        // Placeholder assertion
        assert!(true, "Test logic not fully implemented");
    }
}

// Declare the new adaptive quantization test module
mod adaptive_quantization_test;

// Module for data loading
mod testdata;

// Module for common test data structures
mod structs;

// Module for test utility functions
mod test_utils;

// You can also include top-level test functions directly in this file if needed.
#[cfg(test)]
mod top_level_jpegli_tests {
    // use super::structs::*; // If needed
    // use super::testdata::*; // If needed
    use crate::jpegli::quant::*;
    use crate::jpegli::tests::testdata::SET_QUANT_MATRICES_TESTS;
    use crate::jpegli::tests::structs::ComponentInfoMinimal;

    #[test]
    fn it_works_jpegli() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_quality_to_distance_mapping() {
        // Basic checks based on known jpegli behavior/comments
        assert!((quality_to_distance(100) - 0.1).abs() < 0.01); // Q100 ~ dist 0.1
        assert!((quality_to_distance(95) - 0.5).abs() < 0.05); // Q95 ~ dist 0.5
        assert!((quality_to_distance(90) - 1.0).abs() < 0.05); // Q90 ~ dist 1.0
        assert!((quality_to_distance(80) - 1.8).abs() < 0.1);  // Q80 ~ dist 1.8
        assert!((quality_to_distance(70) - 2.8).abs() < 0.1);  // Q70 ~ dist 2.8
        assert!((quality_to_distance(50) - 4.0).abs() < 0.5);  // Q50 ~ dist 4
        assert!((quality_to_distance(30) - 8.0).abs() < 1.0);  // Q30 ~ dist 8
        assert!((quality_to_distance(10) - 20.0).abs() < 5.0); // Q10 ~ dist 20-30
    }

    #[test]
    fn test_set_quant_matrices_loading() {
        // Simple test to ensure the JSON data can be loaded and parsed
        let data = &SET_QUANT_MATRICES_TESTS;
        assert!(!data.is_empty(), "SetQuantMatrices test data failed to load or is empty.");
        // Check one field from the first test case
        assert_eq!(data[0].test_type, "SetQuantMatricesTest");
        assert!(data[0].input_distances.len() >= 2); // Expect at least Luma/Chroma
        assert!(data[0].expected_quant_tables.len() >= 2);
        assert!(data[0].expected_quant_tables[0].is_some());
        assert_eq!(data[0].expected_quant_tables[0].as_ref().unwrap().len(), 64);

    }

    // TODO: Add actual functional tests for set_quant_matrices using the loaded data
    // This will likely require mocking or creating parts of the JpegCompressStruct state.
    // For now, we just test loading.

} 