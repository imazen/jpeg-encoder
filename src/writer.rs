use crate::encoder::Component;
use crate::huffman::{CodingClass, HuffmanTable};
use crate::marker::{Marker, SOFType};
use crate::quantization::QuantizationTable;
use crate::EncodingError;

/// Density settings
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Density {
    /// No pixel density is set, which means "1 pixel per pixel"
    None,

    /// Horizontal and vertical dots per inch (dpi)
    Inch { x: u16, y: u16 },

    /// Horizontal and vertical dots per centimeters
    Centimeter { x: u16, y: u16 },
}

/// Zig-zag sequence of quantized DCT coefficients
///
/// Figure A.6
pub static ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const BUFFER_SIZE: usize = core::mem::size_of::<usize>() * 8;

/// A no_std alternative for `std::io::Write`
///
/// An implementation of a subset of `std::io::Write` necessary to use the encoder without `std`.
/// This trait is implemented for `std::io::Write` if the `std` feature is enabled.
pub trait JfifWrite {
    /// Writes the whole buffer. The behavior must be identical to std::io::Write::write_all
    /// # Errors
    ///
    /// Return an error if the data can't be written
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError>;
}

#[cfg(not(feature = "std"))]
impl<W: JfifWrite + ?Sized> JfifWrite for &mut W {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        (**self).write_all(buf)
    }
}

#[cfg(not(feature = "std"))]
impl JfifWrite for alloc::vec::Vec<u8> {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.extend_from_slice(buf);
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<W: std::io::Write + ?Sized> JfifWrite for W {
    #[inline(always)]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.write_all(buf)?;
        Ok(())
    }
}

pub(crate) struct JfifWriter<W: JfifWrite> {
    w: W,
    bit_buffer: usize,
    free_bits: i8,
}

impl<W: JfifWrite> JfifWriter<W> {
    pub fn new(w: W) -> Self {
        JfifWriter {
            w,
            bit_buffer: 0,
            free_bits: BUFFER_SIZE as i8,
        }
    }

    #[inline(always)]
    pub fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.w.write_all(buf)
    }

    #[inline(always)]
    pub fn write_u8(&mut self, value: u8) -> Result<(), EncodingError> {
        self.w.write_all(&[value])
    }

    #[inline(always)]
    pub fn write_u16(&mut self, value: u16) -> Result<(), EncodingError> {
        self.w.write_all(&value.to_be_bytes())
    }

    pub fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        self.write_bits(0x7F, 7)?;
        self.flush_bit_buffer()?;
        self.bit_buffer = 0;
        self.free_bits = BUFFER_SIZE as i8;

        Ok(())
    }

    pub fn flush_bit_buffer(&mut self) -> Result<(), EncodingError> {
        while self.free_bits <= (BUFFER_SIZE as i8 - 8) {
            self.flush_byte_from_bit_buffer(self.free_bits)?;
            self.free_bits += 8;
        }

        Ok(())
    }

    #[inline(always)]
    fn flush_byte_from_bit_buffer(&mut self, free_bits: i8) -> Result<(), EncodingError> {
        let value = (self.bit_buffer >> (BUFFER_SIZE as i8 - 8 - free_bits)) & 0xFF;

        self.write_u8(value as u8)?;

        if value == 0xFF {
            self.write_u8(0x00)?;
        }

        Ok(())
    }

    #[inline(always)]
    #[allow(overflowing_literals)]
    fn write_bit_buffer(&mut self) -> Result<(), EncodingError> {
        if (self.bit_buffer
            & 0x8080808080808080
            & !(self.bit_buffer.wrapping_add(0x0101010101010101)))
            != 0
        {
            for i in 0..(BUFFER_SIZE / 8) {
                self.flush_byte_from_bit_buffer((i * 8) as i8)?;
            }
            Ok(())
        } else {
            self.w.write_all(&self.bit_buffer.to_be_bytes())
        }
    }

    pub fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        let size = size as i8;
        let value = value as usize;

        let free_bits = self.free_bits - size;

        if free_bits < 0 {
            self.bit_buffer = (self.bit_buffer << (size + free_bits)) | (value >> -free_bits);
            self.write_bit_buffer()?;
            self.bit_buffer = value;
            self.free_bits = free_bits + BUFFER_SIZE as i8;
        } else {
            self.free_bits = free_bits;
            self.bit_buffer = (self.bit_buffer << size) | value;
        }
        Ok(())
    }

    pub fn write_marker(&mut self, marker: Marker) -> Result<(), EncodingError> {
        self.write(&[0xFF, marker.into()])
    }

    pub fn write_segment(&mut self, marker: Marker, data: &[u8]) -> Result<(), EncodingError> {
        self.write_marker(marker)?;
        self.write_u16(data.len() as u16 + 2)?;
        self.write(data)?;

        Ok(())
    }

    pub fn write_header(&mut self, density: &Density) -> Result<(), EncodingError> {
        self.write_marker(Marker::APP(0))?;
        self.write_u16(16)?;

        self.write(b"JFIF\0")?;
        self.write(&[0x01, 0x02])?;

        match *density {
            Density::None => {
                self.write_u8(0x00)?;
                self.write_u16(1)?;
                self.write_u16(1)?;
            }
            Density::Inch { x, y } => {
                self.write_u8(0x01)?;
                self.write_u16(x)?;
                self.write_u16(y)?;
            }
            Density::Centimeter { x, y } => {
                self.write_u8(0x02)?;
                self.write_u16(x)?;
                self.write_u16(y)?;
            }
        }

        self.write(&[0x00, 0x00])
    }

    /// Append huffman table segment
    ///
    /// - `class`: 0 for DC or 1 for AC
    /// - `dest`: 0 for luma or 1 for chroma tables
    ///
    /// Layout:
    /// ```txt
    /// |--------|---------------|--------------------------|--------------------|--------|
    /// | 0xFFC4 | 16 bit length | 4 bit class / 4 bit dest |  16 byte num codes | values |
    /// |--------|---------------|--------------------------|--------------------|--------|
    /// ```
    ///
    pub fn write_huffman_segment(
        &mut self,
        class: CodingClass,
        destination: u8,
        table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        assert!(destination < 4, "Bad destination: {}", destination);

        self.write_marker(Marker::DHT)?;
        self.write_u16(2 + 1 + 16 + table.values().len() as u16)?;

        self.write_u8(((class as u8) << 4) | destination)?;
        self.write(table.length())?;
        self.write(table.values())?;

        Ok(())
    }

    /// Append a quantization table
    ///
    /// - `precision`: 0 which means 1 byte per value.
    /// - `dest`: 0 for luma or 1 for chroma tables
    ///
    /// Layout:
    /// ```txt
    /// |--------|---------------|------------------------------|--------|--------|-----|--------|
    /// | 0xFFDB | 16 bit length | 4 bit precision / 4 bit dest | V(0,0) | V(0,1) | ... | V(7,7) |
    /// |--------|---------------|------------------------------|--------|--------|-----|--------|
    /// ```
    ///
    pub fn write_quantization_segment(
        &mut self,
        destination: u8,
        table: &QuantizationTable,
    ) -> Result<(), EncodingError> {
        assert!(destination < 4, "Bad destination: {}", destination);

        self.write_marker(Marker::DQT)?;
        self.write_u16(2 + 1 + 64)?;

        self.write_u8(destination)?;

        for &v in ZIGZAG.iter() {
            self.write_u8(table.get_raw(v as usize) as u8)?;
        }

        Ok(())
    }

    pub fn write_dri(&mut self, restart_interval: u16) -> Result<(), EncodingError> {
        self.write_marker(Marker::DRI)?;
        self.write_u16(4)?;
        self.write_u16(restart_interval)
    }

    #[inline]
    pub fn huffman_encode(&mut self, val: u8, table: &HuffmanTable) -> Result<(), EncodingError> {
        let &(size, code) = table.get_for_value(val);
        self.write_bits(code as u32, size)
    }

    #[inline]
    pub fn huffman_encode_value(
        &mut self,
        size: u8,
        symbol: u8,
        value: u16,
        table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let &(num_bits, code) = table.get_for_value(symbol);

        let mut temp = value as u32;
        temp |= (code as u32) << size;
        let size = size + num_bits;

        self.write_bits(temp, size)
    }

    pub fn write_block(
        &mut self,
        block: &[i16; 64],
        prev_dc: i16,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        self.write_dc(block[0], prev_dc, dc_table)?;
        self.write_ac_block(block, 1, 64, ac_table)
    }

    pub fn write_dc(
        &mut self,
        value: i16,
        prev_dc: i16,
        dc_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let diff = value - prev_dc;
        let (size, value) = get_code(diff);

        self.huffman_encode_value(size, size, value, dc_table)?;

        Ok(())
    }

    pub fn write_ac_block(
        &mut self,
        block: &[i16; 64],
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let mut zero_run = 0;

        for &value in &block[start..end] {
            if value == 0 {
                zero_run += 1;
            } else {
                while zero_run > 15 {
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }

                let (size, value) = get_code(value);
                let symbol = (zero_run << 4) | size;

                self.huffman_encode_value(size, symbol, value, ac_table)?;

                zero_run = 0;
            }
        }

        if zero_run > 0 {
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }

    pub fn write_frame_header(
        &mut self,
        width: u16,
        height: u16,
        components: &[Component],
        progressive: bool,
    ) -> Result<(), EncodingError> {
        if progressive {
            self.write_marker(Marker::SOF(SOFType::ProgressiveDCT))?;
        } else {
            self.write_marker(Marker::SOF(SOFType::BaselineDCT))?;
        }

        self.write_u16(2 + 1 + 2 + 2 + 1 + (components.len() as u16) * 3)?;

        // Precision
        self.write_u8(8)?;

        self.write_u16(height)?;
        self.write_u16(width)?;

        self.write_u8(components.len() as u8)?;

        for component in components.iter() {
            self.write_u8(component.id)?;
            self.write_u8(
                (component.horizontal_sampling_factor << 4) | component.vertical_sampling_factor,
            )?;
            self.write_u8(component.quantization_table)?;
        }

        Ok(())
    }

    pub fn write_scan_header(
        &mut self,
        components: &[&Component],
        spectral: Option<(u8, u8)>,
    ) -> Result<(), EncodingError> {
        self.write_marker(Marker::SOS)?;

        self.write_u16(2 + 1 + (components.len() as u16) * 2 + 3)?;

        self.write_u8(components.len() as u8)?;

        for component in components.iter() {
            self.write_u8(component.id)?;
            self.write_u8((component.dc_huffman_table << 4) | component.ac_huffman_table)?;
        }

        let (spectral_start, spectral_end) = spectral.unwrap_or((0, 63));

        // Start of spectral or predictor selection
        self.write_u8(spectral_start)?;

        // End of spectral selection
        self.write_u8(spectral_end)?;

        // Successive approximation bit position high and low
        self.write_u8(0)?;

        Ok(())
    }
}

#[inline]
pub(crate) fn get_code(value: i16) -> (u8, u16) {
    let temp = value - (value.is_negative() as i16);
    let temp2 = value.abs();

    /*
     * Doing this instead of 16 - temp2.leading_zeros()
     * Gives the compiler the information that leadings_zeros
     * is always called on a non zero value, which removes a branch on x86
     */
    let num_bits = 15 - (temp2 << 1 | 1).leading_zeros() as u16;

    let coefficient = temp & ((1 << num_bits as usize) - 1);

    (num_bits as u8, coefficient as u16)
}
