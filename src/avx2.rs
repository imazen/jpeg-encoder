#![cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]

pub(crate) mod fdct;
pub(crate) mod ycbcr;

use crate::encoder::{Operations, DefaultOperations};
use crate::quantization::{QuantizationTable};
use crate::writer::ZIGZAG;

#[derive(Clone)]
pub(crate) struct AVX2Operations;

impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(&self, data: &mut [i16; 64]) {
        fdct::fdct_avx2(data);
    }

    #[inline(always)]
    fn quantize_block(
        &self,
        block: &[i16; 64],
        q_block: &mut [i16; 64],
        table: &QuantizationTable,
        block_x: usize,
        block_y: usize,
    ) {
        // Placeholder: No AVX2 quantization currently implemented here.
        // Delegate to the scalar version for now.
        // TODO: Implement AVX2 quantization if desired/available.
        DefaultOperations.quantize_block(block, q_block, table, block_x, block_y);
    }
}
