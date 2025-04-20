#![cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
#![allow(clippy::excessive_precision)]

pub(crate) mod fdct;
pub(crate) mod ycbcr;

pub(crate) use self::fdct::fdct_avx2;
pub(crate) use self::ycbcr::{BgrImageAVX2, BgraImageAVX2, RgbImageAVX2, RgbaImageAVX2};

use crate::encoder::{DefaultOperations, Operations};

#[derive(Clone)]
pub(crate) struct AVX2Operations;

impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(&self, data: &mut [i16; 64]) {
        // Safety: AVX2 feature is checked by the cfg attribute on the module
        // and also explicitly checked during Encoder construction.
        unsafe { fdct_avx2(data) };
    }

    // No quantize_block needed here anymore
}
