use crate::jpegli::tests::structs::*;
use once_cell::sync::Lazy;
use serde_json::from_str;

// --- JSON Data Loading ---

// Helper macro to reduce boilerplate
macro_rules! load_test_data {
    ($json_path:literal, $struct_type:ty) => {
        Lazy::new(|| {
            let json_str = include_str!($json_path);
            from_str::<Vec<$struct_type>>(json_str)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {}\nContent: {}\n\nRun reference_dct/generate_test_data.py to regenerate.", $json_path, e, json_str))
        })
    };
}

// Load data for each instrumented function
pub static SET_QUANT_MATRICES_TESTS: Lazy<Vec<SetQuantMatricesTest>> = load_test_data!("json/SetQuantMatricesTest.json", SetQuantMatricesTest);
pub static INIT_QUANTIZER_TESTS: Lazy<Vec<InitQuantizerTest>> = load_test_data!("json/InitQuantizerTest.json", InitQuantizerTest);
// pub static PAD_INPUT_BUFFER_TESTS: Lazy<Vec<PadInputBufferTest>> = load_test_data!("json/PadInputBufferTest.json", PadInputBufferTest);
pub static COMPUTE_PRE_EROSION_TESTS: Lazy<Vec<ComputePreErosionTest>> = load_test_data!("json/ComputePreErosionTest.json", ComputePreErosionTest);
pub static FUZZY_EROSION_TESTS: Lazy<Vec<FuzzyErosionTest>> = load_test_data!("json/FuzzyErosionTest.json", FuzzyErosionTest);
pub static PER_BLOCK_MODULATIONS_TESTS: Lazy<Vec<PerBlockModulationsTest>> = load_test_data!("json/PerBlockModulationsTest.json", PerBlockModulationsTest);
pub static COMPUTE_ADAPTIVE_QUANT_FIELD_TESTS: Lazy<Vec<ComputeAdaptiveQuantFieldTest>> = load_test_data!("json/ComputeAdaptiveQuantFieldTest.json", ComputeAdaptiveQuantFieldTest);
// pub static RGB_TO_YCBCR_TESTS: Lazy<Vec<RgbToYCbCrTest>> = load_test_data!("json/RGBToYCbCrTest.json", RgbToYCbCrTest);

// TODO: Add static Lazy variables for other test data files as they are generated
// e.g.:
// pub static RGB_TO_YCBCR_TESTS: Lazy<Vec<RgbToYCbCrTest>> = load_test_data!("json/RgbToYCbCrTest.json", RgbToYCbCrTest);
// etc. 