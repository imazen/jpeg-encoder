
pub trait SimdWidth where Self:Sized {
    /// Suggested lanes for the given type, useful for alignment
    fn simd_width() -> usize;

    fn simd_width_in_bytes() -> usize {
        Self::simd_width() * std::mem::size_of::<Self>()
    }

    fn simd_min_alignment_in_bytes() -> usize {
        let alignment = Self::simd_width_in_bytes().max(std::mem::align_of::<Self>());
        debug_assert!(alignment.is_power_of_two());
        alignment
    }
    fn simd_min_alignment_in_elements() -> usize {
        Self::simd_min_alignment_in_bytes() / std::mem::size_of::<Self>()
    }

    fn simd_optimal_alignment_in_bytes() -> usize {
        const HWY_ALIGNMENT: usize = 128; // Just for L1/L2 cache reasons. 
        Self::simd_width_in_bytes().max(HWY_ALIGNMENT).max(std::mem::align_of::<Self>())
    }
    // We do a minimum alignment of 128 bytes
    fn simd_optimal_alignment_in_elements() -> usize {
        Self::simd_optimal_alignment_in_bytes() / std::mem::size_of::<Self>()
    }

    fn simd_stride_in_bytes(row_count: usize) -> usize {
        let alignment = Self::simd_optimal_alignment_in_bytes();
        let min = alignment + row_count * std::mem::size_of::<Self>() + Self::simd_width_in_bytes();
        min.div_ceil(alignment) * alignment
    }
    fn simd_stride_in_elements(row_count: usize) -> usize {
        Self::simd_stride_in_bytes(row_count) / std::mem::size_of::<Self>()
    }
    fn simd_is_optimally_aligned(slice: &[Self]) -> bool {
        slice.as_ptr().addr() & (Self::simd_optimal_alignment_in_bytes() - 1) == 0
    }
    fn simd_is_minimally_aligned(slice: &[Self]) -> bool {
        slice.as_ptr().addr() & (Self::simd_min_alignment_in_bytes() - 1) == 0
    }


}
// Placeholder constants for SIMD/Alignment - Replace with actual values or imports
fn simd_width<T : target_features::SimdType>() -> usize {
    multiversion::target_features::CURRENT_TARGET.suggested_simd_width::<T>().unwrap_or(1)
}
impl SimdWidth for f32 {
    fn simd_width() -> usize {
        simd_width::<f32>()
    }
}
impl SimdWidth for f64 {
    fn simd_width() -> usize {
        simd_width::<f64>()
    }
}
impl SimdWidth for wide::f32x4 {
    fn simd_width() -> usize {
        simd_width::<f32>() / 4
    }
}
impl SimdWidth for wide::f64x2 {
    fn simd_width() -> usize {
        simd_width::<f64>() / 2
    }
}
impl SimdWidth for wide::f32x8 {
    fn simd_width() -> usize {
        simd_width::<f32>() / 8
    }
}
impl SimdWidth for wide::f64x4 {
    fn simd_width() -> usize {
        simd_width::<f64>() / 4
    }
}
impl SimdWidth for wide::u8x16 {
    fn simd_width() -> usize {
        simd_width::<u8>() / 16
    }
}
impl SimdWidth for wide::u16x8 {
    fn simd_width() -> usize {
        simd_width::<u16>() / 8
    }
}
impl SimdWidth for wide::u16x16  {
    fn simd_width() -> usize {
        simd_width::<u16>() / 16
    }
}

impl SimdWidth for wide::u32x4 {
    fn simd_width() -> usize {
        simd_width::<u32>() / 4
    }
}
impl SimdWidth for wide::u32x8 {
    fn simd_width() -> usize {
        simd_width::<u32>() / 8
    }
}
impl SimdWidth for wide::u64x2 {
    fn simd_width() -> usize {
        simd_width::<u64>() / 2
    }
}
impl SimdWidth for wide::u64x4 {
    fn simd_width() -> usize {
        simd_width::<u64>() / 16
    }
}
impl SimdWidth for wide::i8x16 {
    fn simd_width() -> usize {
        simd_width::<i8>() / 16
    }
}
impl SimdWidth for wide::i8x32 {
    fn simd_width() -> usize {
        simd_width::<i8>() / 32
    }
}
impl SimdWidth for wide::i16x8 {
    fn simd_width() -> usize {
        simd_width::<i16>() / 8
    }
}
impl SimdWidth for wide::i16x16 {
    fn simd_width() -> usize {
        simd_width::<i16>() / 16
    }
}
impl SimdWidth for wide::i32x4 {
    fn simd_width() -> usize {
        simd_width::<i32>() / 4
    }
}
impl SimdWidth for wide::i32x8 {
    fn simd_width() -> usize {
        simd_width::<i32>() / 8
    }
}
impl SimdWidth for wide::i64x2 {
    fn simd_width() -> usize {
        simd_width::<i64>() / 2
    }
}
impl SimdWidth for wide::i64x4 {
    fn simd_width() -> usize {
        simd_width::<i64>() / 4
    }
}


