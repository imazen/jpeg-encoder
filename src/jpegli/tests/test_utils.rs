// Helper functions for tests

use serde::{Deserialize, Deserializer};

// Example deserialization helper for BlockF32/BlockI32 data field
// Assumes JSON data is a flat list of 64 numbers

pub fn deserialize_block_f32<'de, D>(deserializer: D) -> Result<[f32; 64], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<f32> = Vec::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<f32>| {
        serde::de::Error::custom(format!(
            "Expected a list of 64 f32 elements, got {}",
            v.len()
        ))
    })
}

pub fn deserialize_block_i32<'de, D>(deserializer: D) -> Result<[i32; 64], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<i32> = Vec::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<i32>| {
        serde::de::Error::custom(format!(
            "Expected a list of 64 i32 elements, got {}",
            v.len()
        ))
    })
}

// Add other test utilities here, e.g., float comparison with tolerance
pub fn assert_float_eq(a: f32, b: f32, tolerance: f32, message: &str) {
    if (a - b).abs() > tolerance {
        panic!("{}: {} != {} (tolerance: {})", message, a, b, tolerance);
    }
}

pub fn assert_block_f32_eq(a: &[f32; 64], b: &[f32; 64], tolerance: f32, message: &str) {
    for i in 0..64 {
        if (a[i] - b[i]).abs() > tolerance {
            panic!("{}: Block mismatch at index {}: {} != {} (tolerance: {})",
                   message, i, a[i], b[i], tolerance);
        }
    }
}

pub fn assert_block_i32_eq(a: &[i32; 64], b: &[i32; 64], message: &str) {
    for i in 0..64 {
        if a[i] != b[i] {
            panic!("{}: Block mismatch at index {}: {} != {}",
                   message, i, a[i], b[i]);
        }
    }
}

// Add helpers to construct Rust RowBuffers from test slices if needed 