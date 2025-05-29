#![allow(dead_code)]


pub(crate) mod adaptive_quant;
pub(crate) mod adaptive_quant_math;
pub(crate) mod xyb_transform;
pub(crate) mod quant_constants;
pub(crate) mod simd_width;
pub mod cms;
pub mod color_transform;
pub mod fdct_jpegli;
pub mod quant;
pub mod tf;
pub mod xyb;
pub mod structs;
pub mod c_structs;
pub mod config;
pub mod entropy_coding;
pub mod progressive_scan;
pub mod encode;
pub mod consts;

// #[cfg(test)]
// mod reference_tests;

#[cfg(test)]
mod tests;
