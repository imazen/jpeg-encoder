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
use crate::jpegli::jpegli_encoder::JpegliEncoder; // Add import for JpegliEncoder

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

        // Map ColorType to JpegColorType for JpegliEncoder
        let jpeg_color_type = match color_type {
            ColorType::Luma => JpegColorType::Luma,
            ColorType::Rgb | ColorType::Rgba | ColorType::Bgr | ColorType::Bgra => JpegColorType::Ycbcr, // Assume conversion for test
            ColorType::Ycbcr => JpegColorType::Ycbcr,
            ColorType::Cmyk => JpegColorType::Cmyk,
            ColorType::CmykAsYcck | ColorType::Ycck => JpegColorType::Ycck,
            // ColorType::Unknown => panic!("Cannot handle Unknown ColorType in test"), // Removed this arm
        };

        // 2. Create a JpegliEncoder instance. Clamp distance to valid range.
        let distance_clamped = test_case.cjpegli_distance.clamp(0.01, 25.0); // Ensure valid range and float literal
        let mut jpegli_encoder = JpegliEncoder::new(vec![], distance_clamped);

        // 3. Manually trigger setup (as encode_image is not called in this test)
        // We need the color type to determine the number of components for quant setup.
        let num_components = jpeg_color_type.get_num_components();
        // Manually call the setup function (assuming it's accessible for testing or refactor needed)
        // NOTE: This might need adjustment depending on JpegliEncoder's final API
        jpegli_encoder.init_components(jpeg_color_type, width, height).expect("init_components failed");
        jpegli_encoder.setup_jpegli_quantization(jpeg_color_type).expect("setup_jpegli_quantization failed");

        // 4. Extract the quantization tables from the encoder.
        let rust_luma_dqt = jpegli_encoder.raw_quant_tables[0].expect("Luma quant table missing");
        let rust_chroma_dqt = if num_components > 1 {
            jpegli_encoder.raw_quant_tables[1].expect("Chroma quant table missing")
        } else {
             // For Luma-only, create a dummy Chroma table for comparison to pass,
             // though the comparison might be meaningless/skipped later.
             // Or, adjust the test logic to only compare chroma if num_components > 1.
             // Let's make it compare against expected Chroma if present.
             test_case.expected_chroma_dqt // Use expected if it exists, otherwise this comparison fails correctly.
        };

        // 5. Compare the encoder's tables with expected values. Use tolerance 0 for exact match.
        compare_quant_tables(
            "Luma",
            &rust_luma_dqt,
            &test_case.expected_luma_dqt,
            0, // Use 0 tolerance for now
            test_case.input_filename,
        );
        if num_components > 1 {
            compare_quant_tables(
                "Chroma",
                &rust_chroma_dqt,
                &test_case.expected_chroma_dqt,
                0, // Use 0 tolerance for now
                test_case.input_filename,
            );
        }

        // 6. Optionally, call encode to check for other errors (but tables are tested above).
        // This part requires ImageBuffer implementation or using the old encode method.
        // We skip the full encode for now as we focus on table generation.
        // let encode_result = encoder.encode(&pixels, width, height, color_type);
        // assert!(encode_result.is_ok(), "Encoding failed for {}: {:?}", test_case.input_filename, encode_result.err());

        // Test passed for this case if no panics occurred.
        println!(" -> Quantization tables match for {}.", test_case.input_filename);
    }
} 