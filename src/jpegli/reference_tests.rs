// This file contains the actual tests that use the generated reference data.

use std::println;
use std::vec::Vec;
use std::fmt::Write;
use alloc::string::String;
use alloc::vec;
use std::collections::{HashMap, HashSet};
use core::convert::TryInto; // Added for Vec to Array conversion

use crate::jpegli::{SimplifiedTransferCharacteristics, Subsampling}; // Removed JpegColorType from here
use crate::JpegColorType; // Added direct import

// Import the new data source and struct
// use super::reference_test_data::REFERENCE_QUANT_TEST_DATA; // Removed old import
use super::tests::testdata::SET_QUANT_MATRICES_TESTS;
use super::tests::structs::SetQuantMatricesTest;

// Removed unused JpegliEncoder import
use crate::jpegli::quant::{self, quality_to_distance, JpegliColorSpace, JpegliQuantizerState, QuantPass, DCTSIZE2, JpegliQuantParams, JpegliQuantConfigOptions};
// Removed unused: compute_quant_table_values, JpegliComponentParams, MAX_COMPONENTS

// Constants for table indices
const MAX_QUANT_TABLES: usize = 4;

// --- Struct to hold comparison results ---
#[derive(Debug, Clone)]
struct QuantCompareResult {
    // filename: String, // Use test_case_id instead
    test_case_id: String, // Changed from filename
    component_label: String,
    diff_count: usize,
    max_diff: u16,
    sum_abs_diff: u64, // Use u64 to avoid overflow
    diff_details: Option<String>, // Optionally store the detailed diff grid
}

impl QuantCompareResult {
    // Updated to use test_case_id
    fn success(test_case_id: &str, component_label: &str) -> Self {
        Self {
            test_case_id: test_case_id.to_string(),
            component_label: component_label.to_string(),
            diff_count: 0,
            max_diff: 0,
            sum_abs_diff: 0,
            diff_details: None,
        }
    }

    // Updated to use test_case_id
    fn failure(
        test_case_id: &str,
        component_label: &str,
        diff_count: usize,
        max_diff: u16,
        sum_abs_diff: u64,
        diff_details: String,
    ) -> Self {
        Self {
            test_case_id: test_case_id.to_string(),
            component_label: component_label.to_string(),
            diff_count,
            max_diff,
            sum_abs_diff,
            diff_details: Some(diff_details),
        }
    }

    fn is_success(&self) -> bool {
        self.diff_count == 0
    }
}
// --- End Struct Definition ---

// --- Helper Function to Print Failure Details (New) ---
// Updated signature to use test_case_id
fn print_failure_details(
    test_case_id: &str, // Changed from filename
    distance_str: &str,
    result: &QuantCompareResult,
    seen_signatures: &mut HashSet<(String, usize, u16, u64)>,
) -> Option<String> {
    if result.is_success() {
        return None;
    }
    let mut output = String::new();
    let signature = (
        result.component_label.clone(),
        result.diff_count,
        result.max_diff,
        result.sum_abs_diff,
    );

    // Calculate concise stats
    let correct_count = 64 - result.diff_count;
    let percent_correct = ((correct_count as f64 / 64.0) * 100.0).round() as u8;

    let avg_error = result.sum_abs_diff as f64 / 64.0;

    let concise_stats = format!(
        "{: >2.0}% ({}/64) correct; error avg={: >3.0}; max={: >3.0}; total={}",
        percent_correct,
        correct_count,
        avg_error,
        result.max_diff,
        result.sum_abs_diff
    );

    // Use simplified prefix like "Test Case #N dX.Y:"
    let context_prefix = format!("{} d{}:", test_case_id, distance_str);

    let padding = " ".repeat(7usize.saturating_sub(result.component_label.len())); // Adjusted padding slightly
    if seen_signatures.insert(signature) {
        // First time seeing this failure pattern
        writeln!(output,
            "{} New {} table: {}", // "New", component label, concise stats
            context_prefix,
            result.component_label,
            concise_stats
        ).unwrap();
        if let Some(details) = &result.diff_details {
            // Print the grid assuming 'details' only contains the 8 rows
            writeln!(output, "  ------------------------------------------------------------------------").unwrap();
            for line in details.lines() {
                 writeln!(output, "  {}", line).unwrap(); // Indent the grid row
            }
            writeln!(output, "  ------------------------------------------------------------------------").unwrap();
        }
    } else {
        // Repeated failure pattern
        writeln!(output,
            "{} {} table (repeat):{} {}", // component label, concise stats
            context_prefix,
            result.component_label,
            padding,
            concise_stats
        ).unwrap();
    }
    Some(output)
}
// --- End Helper Function ---

// Helper function to compare quantization tables with tolerance
// Returns QuantCompareResult instead of panicking
// Updated signature to use test_case_id
fn compare_quant_tables(
    label: &str,
    generated: &[u16; 64],
    expected: &[u16; 64],
    _tolerance: u16, // Marked unused as effective_tolerance is hardcoded
    test_case_id: &str, // Changed from filename
) -> QuantCompareResult {
    let effective_tolerance = 2; // Re-apply tolerance of 2
    let mut diff_count = 0;
    let mut max_diff = 0u16;
    let mut sum_abs_diff = 0u64;
    let mut diff_output = String::new();

    for y in 0..8 {
        write!(diff_output, "Row {}: ", y).unwrap();
        for x in 0..8 {
            let index = y * 8 + x;
            let gen_val = generated[index];
            let exp_val = expected[index];
            let diff = gen_val.abs_diff(exp_val);

            if diff > effective_tolerance {
                diff_count += 1;
                max_diff = max_diff.max(diff);
                sum_abs_diff += diff as u64;
                // Show generated(expected)
                write!(diff_output, "{:>4}({:>4}) ", gen_val, exp_val).unwrap();
            } else {
                // Show just the value if within tolerance
                write!(diff_output, "{:>4}       ", gen_val).unwrap();
            }
        }
        writeln!(diff_output).unwrap(); // Newline after each row
    }

    if diff_count > 0 {
        // Return failure result with stats and details (grid only)
        QuantCompareResult::failure( // Updated call
            test_case_id,
            label,
            diff_count,
            max_diff,
            sum_abs_diff,
            diff_output,
        )
    } else {
        // Return success result
        QuantCompareResult::success(test_case_id, label) // Updated call
    }
}

// Helper function to get a standard label for a quantization table index
fn get_table_label(index: usize) -> String {
    match index {
        0 => "Luma".to_string(),
        1 => "Chroma1".to_string(),
        2 => "Chroma2".to_string(),
        3 => "Table 3".to_string(), // Or handle as needed
        _ => format!("Table {}", index),
    }
}

// --- New Helper Function to Print Summary Table (for SetQuantMatricesTest) ---
// Renamed and adjusted for the new results structure
fn print_set_quant_summary_table(
    results: &HashMap<String, HashMap<String, [Option<u64>; MAX_QUANT_TABLES]>>, // Adjusted value type
    unique_distances: &HashSet<String>,
    row_key_label: &str, // e.g., "Test Case"
    skip_passing_results: bool,
) {
    if results.is_empty() {
        println!("\n--- No SetQuantMatrices results to summarize. ---");
        return;
    }

    // --- Sort Keys for Consistent Output ---
    let mut sorted_distances: Vec<String> = unique_distances.iter().cloned().collect();
    sorted_distances.sort_by(|a, b| a.parse::<f32>().unwrap_or(f32::NAN).partial_cmp(&b.parse::<f32>().unwrap_or(f32::NAN)).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_row_keys: Vec<String> = results.keys().cloned().collect();
    // Sort by the numeric part of "#N" if possible
    sorted_row_keys.sort_by_key(|id| id.trim_start_matches('#').split(':').next().unwrap_or("").trim().parse::<usize>().unwrap_or(usize::MAX));

    // --- RE-ADD Column Width Calculations (Strict Max) ---
    let max_row_key_width = sorted_row_keys.iter().map(|f| f.len()).max().unwrap_or(10).max(row_key_label.len());

    let mut dist_col_widths: HashMap<String, usize> = HashMap::new();
    for dist_str in &sorted_distances {
        let mut max_width = dist_str.len(); // Start with header width

        for row_key in &sorted_row_keys {
            if let Some(row_results) = results.get(row_key) {
                let passed = row_results.get(dist_str).map(|table_results| {
                    table_results.iter()
                        .filter_map(|result_opt| result_opt.as_ref()) // Keep only Some(value)
                        .all(|result| *result == 0) // Check if all results are 0
                }).unwrap_or(false);
                if passed && skip_passing_results {
                    continue;
                }
                // Adjust cell length calculation: only join existing results
                let table_results_str = row_results.get(dist_str).map(|table_results| {
                    table_results.iter()
                        .filter_map(|result_opt| result_opt.as_ref()) // Keep only Some(value)
                        .map(|result| { // result here is &u64
                            match result { // Match against references
                                &0 => "0".to_string(),
                                &u64::MAX => "ERR".to_string(),
                                diff => diff.to_string(),
                            }
                        })
                        .collect::<Vec<_>>().join("/")
                }).unwrap_or_else(|| "".to_string());

                max_width = max_width.max(table_results_str.len());
            }
        }
        // REMOVED: max_width = max_width.max(X); // NO minimum width enforcement
        dist_col_widths.insert(dist_str.clone(), max_width);
    }

    let mut test_output = String::new();
    // --- Generate Aligned Table Output (Strict Width) ---
    writeln!(test_output, "\n--- Quantization Table Comparison Summary (SetQuantMatrices Data) ---").unwrap();
    writeln!(test_output, "(Cells show SumAbsDiff T0/T1/T2..; 0=Match, >0=Mismatch, ERR=Error, Blank=Inactive/NA)").unwrap();

    // Print Header Line (Aligned with strict padding)
    // Don't put labels in the table, sep lines. write!(test_output, "{:<width$}", row_key_label, width = max_row_key_width).unwrap(); // Left align header
    for dist_str in &sorted_distances {
        let col_width = dist_col_widths.get(dist_str).cloned().unwrap_or(dist_str.len()); // Default to header len if no data
        write!(test_output, "|{:^width$}", dist_str, width = col_width).unwrap(); // REMOVED leading space before |
    }
    writeln!(test_output).unwrap();

    // Print Data Rows (Aligned with strict padding)
    for row_key in &sorted_row_keys {
        let mut passed = true;
        let mut row_string = String::new();
        write!(row_string, "{:<width$}\n", row_key, width = max_row_key_width).unwrap(); // Left align row key
        if let Some(row_results) = results.get(row_key) {

            for dist_str in &sorted_distances {
                let col_width = dist_col_widths.get(dist_str).cloned().unwrap_or(0); // Default to 0 if no data
                // Format cell content: join only existing results
                 let combined_cell = row_results.get(dist_str).map(|table_results| {
                     table_results.iter()
                         .filter_map(|result_opt| result_opt.as_ref())
                         .map(|result| { // result here is &u64
                             match result { // Match against references
                                 &0 => "0".to_string(),
                                 &u64::MAX => "ERR".to_string(),
                                 diff => diff.to_string(),
                             }
                         })
                         .collect::<Vec<_>>().join("/")
                 }).unwrap_or_else(|| "".to_string());

                let all_zero = row_results.get(dist_str).map(|table_results| {
                table_results.iter()
                    .filter_map(|result_opt| result_opt.as_ref()) // Keep only Some(value)
                    .all(|result| *result == 0)
                }).unwrap_or(false);

                if !combined_cell.is_empty() && !all_zero {  // Check if all results are 0
                    passed = false;
                }
                write!(row_string, "|{:^width$}", combined_cell, width = col_width).unwrap(); // REMOVED leading space before |
            }
        } else {
            // Print empty padded cells for missing data row
             for dist_str in &sorted_distances {
                 let col_width = dist_col_widths.get(dist_str).cloned().unwrap_or(0);
                write!(row_string, "|{:^width$}", "", width = col_width).unwrap(); // REMOVED leading space before |
             }
        }
        if !passed || !skip_passing_results {
            writeln!(test_output, "{}", row_string).unwrap(); // Newline after each row
        }
    }
    println!("{}", test_output);
}
// --- End Helper Function ---

#[test]
// Renamed test function
fn compare_set_quant_matrices_with_reference() {
    // Data structure: HashMap<test_case_id, HashMap<distance_str, [Option<u64>; MAX_QUANT_TABLES]>>
    let mut results: HashMap<String, HashMap<String, [Option<u64>; MAX_QUANT_TABLES]>> = HashMap::new();
    let mut unique_distances: HashSet<String> = HashSet::new();
    let mut any_failures = false; // Track if any comparison failed
    let mut seen_failure_signatures: HashSet<(String, usize, u16, u64)> = HashSet::new(); // Track unique failure patterns

    // Use the new data source
    let test_subset = &*SET_QUANT_MATRICES_TESTS;

    // No longer filtering by filename
    // let subset_filenames = vec![...];
    // let test_subset = REFERENCE_QUANT_TEST_DATA.iter()...

    let total_tests = test_subset.len(); // Keep total test count
    let mut tests_run = 0;

    println!("\n--- Running Quant Table Comparison (SetQuantMatrices Data, {} cases) ---", total_tests);

    // Iterate over SetQuantMatricesTest data
    for test_case_data in test_subset {
        let mut test_output = String::new();
        tests_run += 1;
        let mut case_failed_this_run = false; // Track failure for this specific case

        // Create a Test Case ID (e.g., using index and source file) - Changed prefix
        let mut test_case_id = format!("#{}: {}", tests_run, test_case_data.source_file);

        // Parse distance from command_params
        let mut distance_f32 = f32::NAN;
        if let Some(index) = test_case_data.command_params.iter().position(|s| s == "--distance") {
            if let Some(dist_str) = test_case_data.command_params.get(index + 1) {
                if let Ok(d) = dist_str.parse::<f32>() {
                    distance_f32 = d;
                }
            }
        }
        // Parse subsampling from command_params
        let mut subsampling = None;
        if let Some(index) = test_case_data.command_params.iter().position(|s| s == "--chroma_subsampling") {
            if let Some(sub_str) = test_case_data.command_params.get(index + 1) {
                subsampling = Subsampling::from_str(sub_str);
            }
        }
        let distance_str = format!("{:.1}", distance_f32); // Use parsed distance
        // Format active tables based on component order
        let component_table_indices: Vec<usize> = test_case_data.config_components.iter()
            .map(|comp| comp.quant_tbl_no as usize)
            .collect();
        let active_tables_str = format!("tables=[{}]", component_table_indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));

        // Updated Config Print Line
        let header_line = format!(
            "{} d{} (base={},std={},xyb={},add2chr={},adapt={},sub={},{}\n",
            test_case_id,
            distance_str,
            test_case_data.config_force_baseline,
            test_case_data.config_use_std_tables,
            test_case_data.config_xyb_mode,
            test_case_data.config_add_two_chroma_tables,
            test_case_data.config_use_adaptive_quantization,
            subsampling.map_or("444", |s|s.to_str()),   
            active_tables_str // Add formatted active tables
        );
        test_case_id = header_line.clone();
        test_output.push_str(&header_line);

            
        if distance_f32.is_nan() {
            writeln!(test_output, "Skipping {}: Could not parse --distance from command_params: {:?}", test_case_id, test_case_data.command_params).unwrap();
            results.entry(test_case_id).or_default().insert("N/A".to_string(), [Some(u64::MAX); MAX_QUANT_TABLES]); // Mark all tables as error for this case
            any_failures = true;
            case_failed_this_run = true; // Mark case as failed
            eprintln!("FAIL {}", test_output);
            continue;
        }

        // Now distance_str can be safely created
        unique_distances.insert(distance_str.clone());

 
        // Determine num_components based on the *config's* JpegColorSpace
        let num_components = test_case_data.config_jpeg_color_space.get_num_components();

        test_output.push_str(&format!("num_components: {}\n", num_components));
        // --- Configure Quant Params from Test Case ---
        let distance_clamped = distance_f32.clamp(0.0, 25.0); // Use parsed distance
        let config_options = JpegliQuantConfigOptions {
            distance: Some(distance_clamped),
            quality: None,
            xyb_mode: Some(test_case_data.config_xyb_mode),
            use_std_tables: Some(test_case_data.config_use_std_tables),
            use_adaptive_quantization: Some(test_case_data.config_use_adaptive_quantization),
            force_baseline: Some(test_case_data.config_force_baseline),
            // Assume 444 for simplicity, component info/subsampling isn't used directly for quant table gen here
            chroma_subsampling: subsampling,
            jpeg_color_space: test_case_data.config_jpeg_color_space,
            cicp_transfer_function: SimplifiedTransferCharacteristics::from_int(
                test_case_data.config_cicp_transfer_function
            ),
            add_two_chroma_tables: Some(test_case_data.config_add_two_chroma_tables),
        };
        if config_options.cicp_transfer_function.is_some() &&
            config_options.cicp_transfer_function.unwrap() != SimplifiedTransferCharacteristics::Default {
            writeln!(test_output, "DEBUG {}: config_cicp_transfer_function: {:?}", test_case_id, config_options.cicp_transfer_function).unwrap();
        }

        // --- Create Validated Params & State (as before) ---
        let quant_params_result = JpegliQuantParams::from_config(&config_options);
        if quant_params_result.is_err() {
             writeln!(test_output, "Skipping {}: JpegliQuantParams creation failed: {:?}", test_case_id, quant_params_result.err().unwrap()).unwrap();
             results.entry(test_case_id.clone()).or_default().insert(distance_str.clone(), [Some(u64::MAX); MAX_QUANT_TABLES]);
             any_failures = true;
             case_failed_this_run = true; // Mark case as failed
             eprintln!("FAIL {}", test_output);
             continue;
        }
        let mut quant_params = quant_params_result.unwrap();

        let quantizer_state_result =
         JpegliQuantizerState::new(&mut quant_params, QuantPass::NoSearch);
        if quantizer_state_result.is_err() {
             writeln!(test_output, "Skipping {}: JpegliQuantizerState creation failed: {:?}", test_case_id, quantizer_state_result.err().unwrap()).unwrap();
             results.entry(test_case_id.clone()).or_default().insert(distance_str.clone(), [Some(u64::MAX); MAX_QUANT_TABLES]);
             any_failures = true;
             case_failed_this_run = true; // Mark case as failed
             eprintln!("FAIL {}", test_output);
             continue;
        }
        let quantizer_state = quantizer_state_result.unwrap();
        let maybe_rust_tables = quantizer_state.raw_quant_tables;

        // --- Determine Active Tables --- 
        let active_table_indices: HashSet<usize> = test_case_data.config_components.iter()
            .map(|comp| comp.quant_tbl_no as usize)
            .collect();
        // println!("DEBUG {}: Active table indices (set): {:?}", test_case_id, active_table_indices);


        // Remove the sort step
        // let mut sorted_active_indices: Vec<usize> = active_table_indices.iter().cloned().collect();
        // sorted_active_indices.sort();
        
 
        // --- Refactored Comparison Logic --- 
        let mut current_results: [Option<u64>; MAX_QUANT_TABLES] = [None; MAX_QUANT_TABLES];

        for table_idx in 0..MAX_QUANT_TABLES { // Check all potential table slots

            let is_active = active_table_indices.contains(&table_idx);
            let table_label = get_table_label(table_idx);

            let generated_table_opt = maybe_rust_tables.get(table_idx).and_then(|t| t.as_ref());
            let expected_table_opt = test_case_data.expected_quant_tables.get(table_idx).and_then(|o| o.as_ref());

            let mut comparison_result: Option<u64> = None; // Default to inactive/not compared
            let mut error_occurred = false;

            match (generated_table_opt, expected_table_opt) {
                (Some(generated_vec), Some(expected_vec)) => {
                    // Both exist, compare if active
                    if is_active {
                        if let (Ok(generated_arr), Ok(expected_arr)) =
                           (<&[u16; DCTSIZE2]>::try_from(generated_vec.as_slice()), <&[u16; DCTSIZE2]>::try_from(expected_vec.as_slice()))
                        {
                           let result = compare_quant_tables(&table_label, generated_arr, expected_arr, 0, &test_case_id);
                           comparison_result = Some(result.sum_abs_diff); // 0 for match, >0 for mismatch
                           if !result.is_success() {
                               any_failures = true;
                               case_failed_this_run = true; // Mark case as failed
                               let failure_details = print_failure_details(&test_case_id, &distance_str, &result, &mut seen_failure_signatures);
                               if let Some(details) = failure_details {
                                   write!(test_output, "{}", details).unwrap();
                               }
                           }
                        } else {
                            writeln!(test_output, "Error {}: Size mismatch for active Table {} (Generated: {}, Expected: {})", test_case_id, table_idx, generated_vec.len(), expected_vec.len()).unwrap();
                            error_occurred = true;
                            case_failed_this_run = true; // Mark case as failed
                        }
                    } else {
                         // Both exist, but table is not active - potentially WARN if different?
                         // For now, treat as inactive (None result)
                    }
                }
                (Some(generated_vec), None) => {
                    // Generated exists, but expected is missing (or None entry)
                    if is_active {
                         writeln!(test_output, "Error {}: Generated Table {} but expected None or missing", test_case_id, table_idx).unwrap();
                         error_occurred = true;
                         case_failed_this_run = true; // Mark case as failed
                    } else {
                        // Generated exists, but inactive -> Not an error, just ignore
                        // println!("WARN {}: Generated inactive Table {} ({:?}...)", test_case_id, table_idx, &generated_vec[0..4.min(generated_vec.len())]);
                    }
                }
                (None, Some(expected_vec)) => {
                     // Generated is missing, but expected exists
                     if is_active {
                          writeln!(test_output, "Error {}: Missing generated Table {}, but expected", test_case_id, table_idx).unwrap();
                          error_occurred = true;
                          case_failed_this_run = true; // Mark case as failed
                     } else {
                          // Expected exists, but inactive -> Not an error, just ignore
                          // println!("WARN {}: Expected inactive Table {} ({:?}...)", test_case_id, table_idx, &expected_vec[0..4.min(expected_vec.len())]);
                     }
                }
                (None, None) => {
                     // Both missing, OK whether active or not.
                     // comparison_result remains None
                }
            }

            if error_occurred {
                 current_results[table_idx] = Some(u64::MAX); // Store Error marker
                 any_failures = true;
                 // case_failed_this_run is already set above when error_occurred is true
            } else {
                 current_results[table_idx] = comparison_result; // Store None, Some(0), or Some(diff)
            }
        }

        // --- End Refactored Comparison Logic ---

        // Store results using Test Case ID and distance string
        results.entry(test_case_id.clone()).or_default().insert(distance_str.clone(), current_results);

        // Print PASS/FAIL for this specific test case
        if case_failed_this_run {
            eprintln!("FAIL {}", test_output.trim_end());
        } else {
            eprintln!("PASS {}", test_output.trim_end());
        }

    }

    // --- Generate Summary Table using Helper ---
    print_set_quant_summary_table(&results, &unique_distances, "Test Case", true); // Use updated helper

    // --- Final Verdict ---
    if any_failures {
        // Adjust panic message
        panic!("Quantization table comparison failed for SetQuantMatrices tests. See table above for details (SumAbsDiff > 0 or ERR indicates failure).");
    } else {
        println!("All {} SetQuantMatrices test cases produced matching quantization tables (within tolerance)!", tests_run);
    }
}

// --- Tests moved from quant.rs ---

#[test]
fn test_quality_to_distance() {
    // Test values derived from running the C++ code or known reference points.
    assert!((quality_to_distance(100) - 0.01).abs() < 1e-6);
    assert!((quality_to_distance(90) - (0.1 + 10.0 * 0.09)).abs() < 1e-6); // 1.0
    assert!((quality_to_distance(75) - (0.1 + 25.0 * 0.09)).abs() < 1e-6); // 2.35
    assert!((quality_to_distance(50) - (0.1 + 50.0 * 0.09)).abs() < 1e-6); // 4.6
    assert!((quality_to_distance(30) - (0.1 + 70.0 * 0.09)).abs() < 1e-6); // 6.4

    // Lower range - using the quadratic formula part
    let q20 = (53.0 / 3000.0) * 20.0f32.powi(2) - (23.0 / 20.0) * 20.0 + 25.0;
    assert!((quality_to_distance(20) - q20).abs() < 1e-6); // approx 9.0666
    let q10 = (53.0 / 3000.0) * 10.0f32.powi(2) - (23.0 / 20.0) * 10.0 + 25.0;
    assert!((quality_to_distance(10) - q10).abs() < 1e-6); // approx 15.2666
    let q1 = (53.0 / 3000.0) * 1.0f32.powi(2) - (23.0 / 20.0) * 1.0 + 25.0;
    assert!((quality_to_distance(1) - q1).abs() < 1e-6); // approx 23.8676
}
