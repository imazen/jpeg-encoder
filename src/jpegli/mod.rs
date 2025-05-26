#![allow(dead_code)]

use alloc::vec::Vec;

pub(crate) mod adaptive_quant;
pub(crate) mod adaptive_quant_math;
pub(crate) mod adaptive_quantization;
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

pub mod jpegli_encoder;
pub use jpegli_encoder::JpegliEncoder;

use serde::{Deserialize, Serialize};
use serde_repr::*;

#[cfg(test)]
mod reference_tests;

#[cfg(test)]
mod tests;
