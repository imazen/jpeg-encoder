// This file contains the actual tests that use the generated reference data.

use std::println;
use std::vec::Vec;
use std::fmt::Write;
use alloc::string::String;
use alloc::vec;

use crate::Encoder; // Assuming Encoder is in the crate root
use crate::ColorType; // Assuming ColorType is in the crate root
use crate::JpegColorType; // Assuming JpegColorType is in the crate root
// Potentially need image decoders like png, ppm etc.
// use image; 

// Import the generated data module
use super::reference_test_data::REFERENCE_QUANT_TEST_DATA;
use std::io::Cursor;

// Helper function to compare quantization tables with tolerance and print differences
fn compare_quant_tables(
    label: &str,
    generated: &[u16; 64],
    expected: &[u16; 64],
    tolerance: u16,
    filename: &str,
) {
    let mut diff_count = 0;
    let mut diff_output = String::new();
    writeln!(
        diff_output,
        "Comparing {} Quantization Table for {}:",
        label,
        filename
    ).unwrap();
    writeln!(diff_output, "------------------------------------------------------------------------").unwrap();

    for y in 0..8 {
        write!(diff_output, "Row {}: ", y).unwrap();
        for x in 0..8 {
            let index = y * 8 + x;
            let gen_val = generated[index];
            let exp_val = expected[index];
            let diff = gen_val.abs_diff(exp_val);

            if diff > tolerance {
                diff_count += 1;
                // Show generated(expected)
                write!(diff_output, "{:>4}({:>4}) ", gen_val, exp_val).unwrap();
            } else {
                // Show just the value if within tolerance
                write!(diff_output, "{:>4}       ", gen_val).unwrap();
            }
        }
        writeln!(diff_output).unwrap(); // Newline after each row
    }
    writeln!(diff_output, "------------------------------------------------------------------------").unwrap();

    if diff_count > 0 {
        println!("{}", diff_output); // Print the detailed diff
        panic!("{} quantization table mismatch for {}. Found {} differences with tolerance {}.", label, filename, diff_count, tolerance);
    }
}

#[test]
fn compare_quantization_with_reference() {
    for test_case in REFERENCE_QUANT_TEST_DATA {
        println!(
            "Testing reference: {} (Source: {}, Format: {}, Distance: {:.1})",
            test_case.input_filename,
            test_case.source_group,
            test_case.input_format,
            test_case.cjpegli_distance
        );

        // 1. Decode the input image data
        let (pixels, width, height, color_type) = match test_case.input_format {
            "PNG" => {
                let decoder = png::Decoder::new(Cursor::new(test_case.input_data));
                let mut reader = decoder.read_info().expect("Failed to read PNG info");
                let mut buf = vec![0; reader.output_buffer_size()];
                let info = reader.next_frame(&mut buf).expect("Failed to decode PNG frame");
                let bytes = &buf[..info.buffer_size()];

                let encoder_color_type = match info.color_type {
                    png::ColorType::Grayscale => ColorType::Luma,
                    png::ColorType::Rgb => ColorType::Rgb,
                    png::ColorType::Rgba => ColorType::Rgba,
                    // Indexed, GrayscaleAlpha might need conversion or specific handling
                    // For now, panic if we encounter unexpected types in test data
                    _ => panic!(
                        "Unsupported PNG color type {:?} in test data {}",
                        info.color_type,
                        test_case.input_filename
                    ),
                };
                (bytes.to_vec(), info.width as u16, info.height as u16, encoder_color_type)
            }
            _ => panic!(
                "Unsupported input format {} for test case {}",
                test_case.input_format,
                test_case.input_filename
            ),
        };

        // 2. Create a jpeg_encoder::Encoder instance.
        let mut encoded_output = Vec::new();
        // Use a dummy quality, as Jpegli config overrides it.
        let mut encoder = Encoder::new(&mut encoded_output, 75);
        
        // 3. Configure the encoder with the reference distance using Jpegli.
        // We specifically test Jpegli's table generation here.
        // encoder.configure_jpegli(test_case.cjpegli_distance, None, None);
        
        // 5. Extract the quantization tables from the encoder's jpegli_config *before* encoding consumes it.
        /* // Commented out until JpegliEncoder integration
        let jpegli_config = encoder.jpegli_config.as_ref()
            .expect("JpegliConfig should be present after configure_jpegli");
        let rust_luma_dqt = jpegli_config.luma_table_raw;
        let rust_chroma_dqt = jpegli_config.chroma_table_raw;
        */
        // Placeholder - these would need to come from JpegliEncoder
        let rust_luma_dqt: [u16; 64] = [16u16; 64]; // Dummy table
        let rust_chroma_dqt: [u16; 64] = [17u16; 64]; // Dummy table
        
        // 6. Compare the encoder's tables with expected values. Use tolerance 0 for exact match.
        compare_quant_tables(
            "Luma",
            &rust_luma_dqt,
            &test_case.expected_luma_dqt,
            0, // Use 0 tolerance for now
            test_case.input_filename,
        );
        compare_quant_tables(
            "Chroma",
            &rust_chroma_dqt,
            &test_case.expected_chroma_dqt,
            0, // Use 0 tolerance for now
            test_case.input_filename,
        );
        
        // 4. Call encoder.encode(...) with decoded image data.
        // This triggers internal setup including quantization table generation.
        // We test the tables *before* this consuming call, but run it to check for other errors.
        let encode_result = encoder.encode(&pixels, width, height, color_type);
        assert!(encode_result.is_ok(), "Encoding failed for {}: {:?}", test_case.input_filename, encode_result.err());
        
        // Test passed for this case if no panics occurred.
        println!(" -> Quantization tables match for {}.", test_case.input_filename);
    }
} 