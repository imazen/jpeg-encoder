import os
print("--- Python script starting execution ---", flush=True)
import subprocess
import pathlib
import glob
import sys
import platform
import argparse
import shutil
import struct # Needed for unpacking binary data
from typing import List, Tuple, Optional

# --- Configuration ---

# Default list of input image source patterns. Paths are relative to SCRIPT_DIR
# Format: (group_name, pattern)
# DEFAULT_SOURCE_PATTERNS: List[Tuple[str, str]] = [
#     ("jxl-chessboard", "testdata/jxl/chessboard/*.png"),
#     ("jxl-blending", "testdata/jxl/blending/*.png"),
#     ("raw.pixls", "testdata/external/raw.pixls/*.png"),
#     ("wesaturate-64px", "testdata/external/wesaturate/64px/*.png"),
#     ("wesaturate-500px", "testdata/external/wesaturate/500px/*.png"),
#     ("wide-gamut-tests", "testdata/external/wide-gamut-tests/*.png"),
# ]

# Explicit list of test images. Paths are relative to SCRIPT_DIR/testdata
# Format: (group_name, relative_path_in_testdata)
EXPLICIT_TEST_FILES: List[Tuple[str, str]] = [
    ("wesaturate-500px", "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png"),
    ("wesaturate-64px", "external/wesaturate/64px/vgqcws_vin_709_g1.png"),
]

# Default directory to store cjpegli output, relative to SCRIPT_DIR
DEFAULT_CJPEGLI_OUTPUT_DIR = "cjpegli_results"
# Default Rust test file path relative to SCRIPT_DIR
DEFAULT_RUST_TEST_FILE = "../src/jpegli/reference_test_data.rs"
# Default distances to test
DEFAULT_DISTANCES = [0, 0.5, 1, 1.4, 2.3, 3.2, 4.3, 7.4, 13, 25]
# Default Docker image if cjpegli is not found locally
DEFAULT_DOCKER_IMAGE = "imazen/jpegli-tools:latest" # Use user-provided image name

# --- Helper Functions ---

def run_command(cmd_list, check=True, **kwargs):
    """Runs a command and prints output/errors."""
    print(f"Running command: {' '.join(map(str, cmd_list))}", flush=True)
    process = None # Initialize process variable
    try:
        process = subprocess.run(cmd_list, check=check, capture_output=True, text=True, **kwargs)
        # Always print exit code on success if check=True didn't raise
        print(f"  Exit code: {process.returncode}", flush=True)
        if process.stdout:
            print("  stdout:", process.stdout.strip(), flush=True)
        if process.stderr:
            print("  stderr:", process.stderr.strip(), flush=True)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr, flush=True)
        if e.stdout:
            print("Stdout:", e.stdout, file=sys.stderr, flush=True)
        if e.stderr:
            print("Stderr:", e.stderr, file=sys.stderr, flush=True)
        # Print captured process info if available (might be None if run failed early)
        # if process:
        #    print(f"  Process info on failure: {process}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Failed to run command: {e}", file=sys.stderr, flush=True)
        # Print captured process info if available
        # if process:
        #    print(f"  Process info on failure: {process}", file=sys.stderr)
        raise

def get_docker_path(host_path: pathlib.Path) -> str:
    """Converts a host pathlib.Path to a Docker volume path (Linux style)."""
    # Assumes standard Docker Desktop/Engine path conversion for Windows drives
    drive = host_path.drive.lower().replace(":", "")
    path_str = str(host_path.relative_to(host_path.anchor)).replace("\\", "/")
    if platform.system() == "Windows":
         return f"/{drive}/{path_str}"
    else: # Assume Linux/macOS path
        return str(host_path)

def find_cjpegli_command(script_dir: pathlib.Path, docker_image: str) -> Tuple[List[str], bool]:
    """
    Finds the cjpegli executable or prepares a Docker command.
    Returns (command_list, is_docker)
    """
    print("Searching for cjpegli...")
    # 1. Check PATH
    cjpegli_path = shutil.which("cjpegli")
    if cjpegli_path:
        print(f"Found cjpegli in PATH: {cjpegli_path}")
        return [cjpegli_path], False

    # 2. Check relative to script directory (../tools/cjpegli)
    relative_cjpegli_path = (script_dir.parent / "tools" / "cjpegli").resolve()
    if relative_cjpegli_path.is_file() and os.access(relative_cjpegli_path, os.X_OK):
        print(f"Found cjpegli relative to script: {relative_cjpegli_path}")
        return [str(relative_cjpegli_path)], False

    # 3. Check for Docker
    print("cjpegli not found locally. Checking for Docker...")
    if shutil.which("docker"):
        print(f"Found Docker. Will use image '{docker_image}'.")
        # Check if image exists locally
        try:
            run_command(["docker", "image", "inspect", docker_image], check=True)
        except subprocess.CalledProcessError:
            print(f"Docker image '{docker_image}' not found locally. Attempting to pull...")
            try:
                run_command(["docker", "pull", docker_image], check=True)
            except Exception as e:
                print(f"ERROR: Failed to pull Docker image '{docker_image}': {e}", file=sys.stderr)
                print("Please ensure Docker is running and you can pull images, or install cjpegli locally.", file=sys.stderr)
                print("Exiting due to Docker pull failure.", file=sys.stderr) # Added print
                sys.exit(1)

        # Prepare Docker command structure (placeholders for paths)
        # Mount script_dir as /work, workdir is /work
        uid = os.getuid() if hasattr(os, 'getuid') else 1000 # Default UID for Windows
        gid = os.getgid() if hasattr(os, 'getgid') else 1000 # Default GID for Windows
        docker_base_cmd = [
            "docker", "run", "--rm",
            "-u", f"{uid}:{gid}",
            "-v", f"{get_docker_path(script_dir)}:/work",
            "-w", "/work",
            docker_image, "cjpegli"
        ]
        return docker_base_cmd, True
    else:
        print("ERROR: cjpegli not found in PATH or relative to script, and Docker is not available.", file=sys.stderr)
        print("Exiting because cjpegli/Docker was not found.", file=sys.stderr) # Added print
        sys.exit(1)

def extract_and_format_dqt(jpeg_path: pathlib.Path, relative_input_include_path: str, distance: float, input_basename: str, input_extension: str, source_group: str, relative_cjpegli_include_path: str) -> Optional[str]:
    """
    Extracts DQT from a JPEG file using direct marker parsing and formats the Rust struct entry.
    Returns the formatted string or None on critical failure.
    """
    print(f"      Extracting DQT from {jpeg_path} using manual parser...")
    qtables = {} # Dictionary to store tables {id: [u16 values]}

    try:
        with open(jpeg_path, 'rb') as f:
            # Check for SOI marker
            soi = f.read(2)
            if soi != b'\xff\xd8':
                print(f"      WARNING: No SOI marker (FFD8) found at start of {jpeg_path}. File might be invalid or non-standard.", file=sys.stderr)
                # Continue processing cautiously, might fail later

            while True:
                # Find next marker (usually prefixed by 0xFF)
                marker_prefix = f.read(1)
                while marker_prefix and marker_prefix != b'\xff':
                    # Skip non-FF bytes, potentially skipping fill bytes (0xFF) after 0xFF00 escapes
                    marker_prefix = f.read(1)

                if not marker_prefix:
                    # This is often a normal EOF condition after EOI or last segment
                    # print(f"      INFO: Reached end of file while searching for marker prefix (0xFF) in {jpeg_path}. Assuming normal EOF.")
                    break # EOF

                marker_code = f.read(1)
                if not marker_code:
                     print(f"      ERROR: Reached end of file unexpectedly while reading marker code after 0xFF in {jpeg_path}.", file=sys.stderr)
                     break # EOF

                marker = marker_code[0]
                # print(f"Found marker 0xFF{marker:02X}") # Debug

                # Markers without length field (standalone)
                # SOI (FFD8) should only be at the start. EOI (FFD9) marks the end.
                # RSTn (FFD0-FFD7), TEM (FF01) are standalone.
                if marker == 0xD8: # SOI
                    # Already handled at the start, unusual to find it mid-stream
                    print(f"      INFO: Found SOI marker mid-stream.", file=sys.stderr)
                    continue
                elif marker == 0xD9: # EOI
                    print(f"      Found EOI marker (FFD9). Stopping scan.")
                    break
                elif 0xD0 <= marker <= 0xD7 or marker == 0x01: # RSTn, TEM
                    # print(f"      Found standalone marker 0xFF{marker:02X}")
                    continue
                # SOS (FFDA) has a length, but its data stream follows until EOI, special handling not needed for DQT extraction
                # APPn (FFE0-FFEF) and COM (FFFE) have length fields we need to read and skip
                elif marker == 0xDA or (0xE0 <= marker <= 0xEF) or marker == 0xFE:
                     pass # Handled by length reading below
                # DQT (FFDB), DHT (FFC4), DRI (FFDD), SOF0/SOF2 (FFC0/FFC2, common), others have length
                elif marker == 0xDB or marker == 0xC4 or marker == 0xDD or marker == 0xC0 or marker == 0xC2:
                     pass # Handled by length reading below
                else:
                     # Unknown marker or one without standard length field we care about here
                     # print(f"      INFO: Found unhandled marker 0xFF{marker:02X}. Reading length if possible.")
                     pass # Assume it has a length field and try to read it

                # Read segment length (Big Endian short)
                len_bytes = f.read(2)
                if not len_bytes or len(len_bytes) < 2:
                     print(f"      ERROR: Reached end of file unexpectedly while reading length for marker 0xFF{marker:02X} in {jpeg_path}.", file=sys.stderr)
                     break
                segment_len = struct.unpack('>H', len_bytes)[0]
                payload_len = segment_len - 2 # Length includes the 2 length bytes

                if payload_len < 0:
                    print(f"      ERROR: Invalid segment length ({segment_len}) for marker 0xFF{marker:02X} in {jpeg_path}. Attempting to skip.", file=sys.stderr)
                    # Cannot reliably skip, better to stop
                    break

                # If it's the DQT marker, process it
                if marker == 0xDB:
                    print(f"      Found DQT marker (FFDB) with segment length {segment_len}, payload {payload_len} bytes.")
                    processed_in_segment = 0
                    while processed_in_segment < payload_len:
                        # Check if enough bytes remain for the table info byte
                        if processed_in_segment + 1 > payload_len:
                             print(f"      ERROR: DQT segment ends prematurely before table info byte (processed {processed_in_segment}/{payload_len}) in {jpeg_path}.", file=sys.stderr)
                             break # Avoid reading past boundary
                        table_info_byte = f.read(1)
                        if not table_info_byte:
                             print(f"      ERROR: Unexpected EOF reading DQT table info byte in {jpeg_path}.", file=sys.stderr)
                             break
                        processed_in_segment += 1

                        precision = (table_info_byte[0] >> 4) & 0x0F # Pq: 0 = 8-bit, 1 = 16-bit
                        table_id = table_info_byte[0] & 0x0F       # Tq: 0..3

                        if table_id > 3:
                             print(f"      WARNING: Invalid DQT table ID {table_id} found in {jpeg_path}. Skipping table data if possible.", file=sys.stderr)
                             bytes_to_skip = 64 * (2 if precision == 1 else 1)
                             if processed_in_segment + bytes_to_skip > payload_len:
                                 print(f"      ERROR: Cannot skip invalid DQT table {table_id}, segment length too short in {jpeg_path}. Skipping rest of segment.", file=sys.stderr)
                                 f.seek(payload_len - processed_in_segment, os.SEEK_CUR) # Skip remaining bytes in segment
                                 processed_in_segment = payload_len # Mark segment as processed
                                 break # Exit inner loop for this segment
                             f.seek(bytes_to_skip, os.SEEK_CUR)
                             processed_in_segment += bytes_to_skip
                             continue # Next table within the segment

                        table_data_size = 64 * (2 if precision == 1 else 1)
                        # Check if enough bytes remain for the table data
                        if processed_in_segment + table_data_size > payload_len:
                             print(f"      ERROR: DQT segment length ({payload_len}) too short for table {table_id} (processed {processed_in_segment}, need {table_data_size}) in {jpeg_path}.", file=sys.stderr)
                             f.seek(payload_len - processed_in_segment, os.SEEK_CUR) # Skip remaining bytes
                             processed_in_segment = payload_len
                             break # Avoid reading past segment boundary

                        table_bytes = f.read(table_data_size)
                        if len(table_bytes) < table_data_size:
                            print(f"      ERROR: Unexpected EOF reading DQT table {table_id} data ({table_data_size} bytes read, expected {len(table_bytes)}) in {jpeg_path}.", file=sys.stderr)
                            processed_in_segment += len(table_bytes) # Track partial read
                            break # Exit inner DQT loop, outer loop might terminate due to EOF

                        processed_in_segment += table_data_size

                        table = []
                        if precision == 0: # 8-bit entries -> treat as u16
                            table = [int(b) for b in table_bytes]
                            print(f"        Read 8-bit DQT for ID {table_id} ({len(table)} entries)")
                        else: # 16-bit entries (Big Endian)
                            try:
                                table = list(struct.unpack(f'>{"H"*64}', table_bytes))
                                print(f"        Read 16-bit DQT for ID {table_id} ({len(table)} entries)")
                            except struct.error as e:
                                print(f"      ERROR: Failed to unpack 16-bit DQT table {table_id} data: {e}. Using zeros.", file=sys.stderr)
                                table = [0] * 64

                        if len(table) == 64:
                            qtables[table_id] = table
                        else:
                            # Should be caught by read/unpack checks, but as a fallback
                            print(f"      WARNING: DQT table {table_id} data read resulted in {len(table)} entries, expected 64. Using zeros.", file=sys.stderr)
                            qtables[table_id] = [0] * 64

                    # Check if we consumed the exact payload length after the inner loop
                    if processed_in_segment != payload_len:
                         print(f"      WARNING: DQT segment processing finished having read {processed_in_segment} bytes, but payload length was {payload_len}. File might be corrupt or segment structure unexpected.", file=sys.stderr)
                         # Attempt to seek to end of segment if we read less than expected
                         if processed_in_segment < payload_len:
                            try:
                                bytes_to_skip = payload_len - processed_in_segment
                                print(f"           Seeking forward {bytes_to_skip} bytes to reach expected segment end.")
                                f.seek(bytes_to_skip, os.SEEK_CUR)
                            except OSError as e:
                                 print(f"      ERROR: Failed to seek to end of DQT segment after partial read: {e}. Stopping scan.", file=sys.stderr)
                                 break # Cannot reliably continue

                else:
                    # Skip other segments by seeking past their payload
                    # print(f"      Skipping marker 0xFF{marker:02X} with payload length {payload_len}")
                    try:
                        f.seek(payload_len, os.SEEK_CUR)
                    except OSError as e:
                         print(f"      ERROR: Failed to seek past segment for marker 0xFF{marker:02X} (payload {payload_len} bytes): {e}. Stopping scan.", file=sys.stderr)
                         break # Cannot reliably continue

    except FileNotFoundError:
        print(f"      ERROR: Input JPEG file not found for DQT extraction: {jpeg_path}", file=sys.stderr)
        return None # Indicate critical failure
    except Exception as e:
        print(f"      ERROR: Unexpected exception during DQT parsing for {jpeg_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Fallback to zero tables if possible, but signal potential issue
        luma_dqt = [0] * 64
        chroma1_dqt = [0] * 64
        chroma2_dqt = None
        # Proceed to formatting, but the file is likely problematic

    # Assign tables based on ID (outside the main parsing loop)
    if not qtables:
        print(f"      WARNING: No quantization tables extracted from {jpeg_path}. Using zeros.", file=sys.stderr)
        luma_dqt = [0] * 64
        chroma1_dqt = [0] * 64
        chroma2_dqt = None
    else:
        luma_dqt = qtables.get(0)
        chroma1_dqt = qtables.get(1)
        chroma2_dqt = qtables.get(2)
        chroma3_dqt = qtables.get(3) # Check if present

        if luma_dqt is None:
            print(f"      WARNING: Luma DQT (ID 0) not found in extracted tables {sorted(qtables.keys())} from {jpeg_path}. Using first table found or zeros.", file=sys.stderr)
            luma_dqt = next(iter(qtables.values()), [0]*64) # Use first table found or zeros
        if chroma1_dqt is None:
             # If only Luma exists (e.g., grayscale), Chroma often uses the same table
             if len(qtables) == 1 and 0 in qtables:
                 print(f"      INFO: Chroma DQT (ID 1) not found, using Luma DQT (ID 0) as fallback for {jpeg_path}.")
                 chroma1_dqt = qtables[0]
             else:
                 print(f"      WARNING: Chroma DQT (ID 1) not found in extracted tables {sorted(qtables.keys())} from {jpeg_path}. Using Luma (ID 0) or zeros.", file=sys.stderr)
                 chroma1_dqt = qtables.get(0, [0]*64) # Fallback to luma or zeros

        # Report what was found
        found_ids = sorted(qtables.keys())
        print(f"      Found DQT IDs: {found_ids}. Assigned Luma=ID 0, Chroma1=ID 1, Chroma2=ID 2.")
        if chroma3_dqt:
             print(f"      INFO: Found DQT for ID 3, but it is currently unused in the Rust struct.")

        # Ensure they are 64 elements (should be guaranteed by parser if successful)
        # Make copies to avoid modifying the original dict values if padding is needed
        luma_dqt = (list(luma_dqt) + [0]*64)[:64] if luma_dqt else [0]*64
        chroma1_dqt = (list(chroma1_dqt) + [0]*64)[:64] if chroma1_dqt else [0]*64
        if chroma2_dqt:
            chroma2_dqt = (list(chroma2_dqt) + [0]*64)[:64]

    # Format the Rust struct entry
    luma_dqt_str = ", ".join(map(str, luma_dqt))
    # Chroma DQTs are optional in the struct, so only include if found (or defaulted)
    chroma1_entry = f"Some([{', '.join(map(str, chroma1_dqt))}])" if chroma1_dqt is not None else "None"
    chroma2_entry = f"Some([{', '.join(map(str, chroma2_dqt))}])" if chroma2_dqt is not None else "None"


    # Escape backslashes in relative paths for Rust string literals
    relative_input_include_path_rs = relative_input_include_path.replace("\\", "\\\\")
    relative_cjpegli_include_path_rs = relative_cjpegli_include_path.replace("\\", "\\\\")

    entry = f"""\
    ReferenceQuantTestData {{
        source_group: "{source_group}",
        input_filename: "{input_basename}",
        input_format: "{input_extension.lower().lstrip('.')}",
        input_data: include_bytes!("{relative_input_include_path_rs}"),
        cjpegli_distance: {distance:.1f},
        cjpegli_output_data: include_bytes!("{relative_cjpegli_include_path_rs}"),
        expected_luma_dqt: [{luma_dqt_str}], // Always present, even if zeros
        expected_chroma1_dqt: {chroma1_entry},
        expected_chroma2_dqt: {chroma2_entry},
    }}"""
    return entry


def write_rust_header(f, script_name: str):
    f.write(f"// Generated by {script_name}. DO NOT EDIT.\n\n")
    f.write("""\
// use crate::quantization::QuantizationTable; // Assuming not needed directly now

#[derive(Debug)]
pub struct ReferenceQuantTestData {
    pub source_group: &'static str,
    pub input_filename: &'static str,
    pub input_format: &'static str, // Format of the *original* input
    pub input_data: &'static [u8],  // Bytes of the *original* input file via include_bytes!
    pub cjpegli_distance: f32,
    pub cjpegli_output_data: &'static [u8], // Bytes of the cjpegli *output* JPEG file via include_bytes!
    pub expected_luma_dqt: [u16; 64],       // Extracted Luma DQT
    pub expected_chroma1_dqt: Option<[u16; 64]>, // Extracted Chroma DQT (usually ID 1)
    pub expected_chroma2_dqt: Option<[u16; 64]>, // Extracted Chroma DQT (usually ID 2, optional)
}

pub const REFERENCE_QUANT_TEST_DATA: &[ReferenceQuantTestData] = &[
""")

def write_rust_footer(f):
    f.write("""\
]; // Close the main data array

// Ensure the tests module wraps the test function
#[cfg(test)]
mod tests {
    use super::*;

    // Example test - implement actual comparison logic here
    #[test]
    fn test_reference_quantization_tables_exist() {
        assert!(!REFERENCE_QUANT_TEST_DATA.is_empty(), "No reference test data found.");
        for test_case in REFERENCE_QUANT_TEST_DATA {
            println!("Checking reference data for {} (source: {}, distance: {:.1})...",
                     test_case.input_filename, test_case.source_group, test_case.cjpegli_distance);
            assert!(test_case.input_data.len() > 0, "Input data is empty for {}", test_case.input_filename);
            assert!(test_case.cjpegli_output_data.len() > 0, "cjpegli output data is empty for {}", test_case.input_filename);
            assert_eq!(test_case.expected_luma_dqt.len(), 64);
            // Only check chroma1 if Some
            if let Some(chroma1_table) = test_case.expected_chroma1_dqt {
                 assert_eq!(chroma1_table.len(), 64, "Chroma1 table has incorrect size for {}", test_case.input_filename);
            } else {
                 println!("  INFO: Chroma1 table is None for {}", test_case.input_filename); // Optional info
            }
            // Check chroma2 only if it exists and is Some
            if let Some(chroma2_table) = test_case.expected_chroma2_dqt {
                assert_eq!(chroma2_table.len(), 64, "Chroma2 table has incorrect size for {}", test_case.input_filename);
            }
            // Basic check that tables contain non-zero values if expected (might need refinement)
            // assert!(test_case.expected_luma_dqt.iter().any(|&x| x > 0), "Luma table seems empty for {}", test_case.input_filename);
            // if let Some(chroma1_table) = test_case.expected_chroma1_dqt {
            //    assert!(chroma1_table.iter().any(|&x| x > 0), "Chroma1 table seems empty for {}", test_case.input_filename);
            // }
        }
    }
}
""")

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate reference JPEG files using cjpegli and extract DQTs for Rust tests.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_CJPEGLI_OUTPUT_DIR,
                        help=f"Directory to store cjpegli output JPEGs (relative to script dir). Default: {DEFAULT_CJPEGLI_OUTPUT_DIR}")
    parser.add_argument("--rust-file", type=pathlib.Path, default=DEFAULT_RUST_TEST_FILE,
                        help=f"Path to the output Rust test file (relative to script dir). Default: {DEFAULT_RUST_TEST_FILE}")
    parser.add_argument("--distances", type=float, nargs='+', default=DEFAULT_DISTANCES,
                        help=f"List of cjpegli --distance values to test. Default: {' '.join(map(str, DEFAULT_DISTANCES))}")
    parser.add_argument("--docker-image", type=str, default=DEFAULT_DOCKER_IMAGE,
                        help=f"Docker image to use if cjpegli is not found locally. Default: {DEFAULT_DOCKER_IMAGE}")
    # TODO: Add argument for source patterns if needed

    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    script_name = pathlib.Path(__file__).name
    print(f"--- {script_name} ---")
    print(f"Script Directory: {script_dir}")

    # Resolve absolute paths based on script_dir
    abs_cjpegli_output_dir = (script_dir / args.output_dir).resolve()
    abs_rust_test_file = (script_dir / args.rust_file).resolve()
    abs_rust_file_dir = abs_rust_test_file.parent

    print(f"cjpegli Output Dir (Absolute): {abs_cjpegli_output_dir}")
    print(f"Rust Test File (Absolute): {abs_rust_test_file}")
    print(f"Distances to test: {args.distances}")

    # Ensure cjpegli output directory exists
    abs_cjpegli_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured cjpegli output directory exists: {abs_cjpegli_output_dir}")

    # Find cjpegli command
    cjpegli_base_cmd, is_docker = find_cjpegli_command(script_dir, args.docker_image)
    print(f"Using cjpegli command ({'Docker' if is_docker else 'Local'}): {' '.join(cjpegli_base_cmd)}")

    # Write Rust file header (OVERWRITE)
    print(f"\n--- Setting up Rust output file: {abs_rust_test_file} ---")
    try:
        with open(abs_rust_test_file, 'w') as f_rust:
            write_rust_header(f_rust, script_name)
        print("Rust file header written.")
    except Exception as e:
        print(f"ERROR: Failed to write header to {abs_rust_test_file}: {e}", file=sys.stderr)
        print("Exiting due to Rust header write failure.", file=sys.stderr) # Added print
        sys.exit(1)

    print("\n--- Processing source images for each distance ---")
    first_entry = True

    # Outer loop: Distances
    for distance in args.distances:
        print(f"\n--- Processing for Distance: {distance} ---")
        # Inner loop: Source Patterns (now explicit files)
        # for group_name, pattern in DEFAULT_SOURCE_PATTERNS:
        for group_name, relative_file_path in EXPLICIT_TEST_FILES:
            # Construct full path relative to script_dir
            # full_pattern = script_dir / pattern
            abs_input_path = (script_dir / "testdata" / relative_file_path).resolve()
            print(f"  Processing group '{group_name}' with file '{abs_input_path}'...")

            # input_files = sorted(glob.glob(str(full_pattern)))
            # if not input_files:
            #     print(f"    Warning: No input files found for pattern: {full_pattern}. Skipping group.")
            #     continue

            # Innermost loop: Files per pattern (now just one file)
            # for input_path_str in input_files:
            # abs_input_path = pathlib.Path(input_path_str).resolve()
            input_basename = abs_input_path.name
            input_ext = abs_input_path.suffix
            input_name_no_ext = abs_input_path.stem

            print(f"    Processing input file: {input_basename}")
            if not abs_input_path.is_file():
                print(f"    ERROR: Path is not a file: {abs_input_path}. Skipping.", file=sys.stderr)
                continue

            # Define output JPEG path
            output_jpeg_subdir = abs_cjpegli_output_dir / group_name
            output_jpeg_subdir.mkdir(parents=True, exist_ok=True)
            output_jpeg_filename = f"{input_name_no_ext}_d{distance}.jpg"
            abs_output_jpeg_path = (output_jpeg_subdir / output_jpeg_filename).resolve()

            # Delete existing output file for this specific run
            if abs_output_jpeg_path.exists():
                print(f"      Removing existing output file: {abs_output_jpeg_path}")
                abs_output_jpeg_path.unlink()

            # Calculate relative paths for include_bytes! and Docker command
            try:
                relative_input_include_path = os.path.relpath(abs_input_path, abs_rust_file_dir)
                relative_cjpegli_include_path = os.path.relpath(abs_output_jpeg_path, abs_rust_file_dir)
                # For Docker command, paths need to be relative to the mapped volume (script_dir -> /work)
                docker_input_path = os.path.relpath(abs_input_path, script_dir).replace("\\", "/")
                docker_output_path = os.path.relpath(abs_output_jpeg_path, script_dir).replace("\\", "/")
            except ValueError as e:
                print(f"    ERROR: Could not calculate relative path for {abs_input_path} or {abs_output_jpeg_path} from {script_dir} or {abs_rust_file_dir}. Skipping. Error: {e}", file=sys.stderr)
                continue

            print(f"      Input include path: {relative_input_include_path}")
            print(f"      Output file: {abs_output_jpeg_path}")
            print(f"      Output include path: {relative_cjpegli_include_path}")

            # Construct and run cjpegli command
            cmd = list(cjpegli_base_cmd) # Copy base command
            cmd.extend(["--distance", str(distance)])

            if is_docker:
                # === Docker Specific File Check ===
                print(f"      Checking file existence in Docker: '{docker_input_path}'...")
                # check_cmd = list(cjpegli_base_cmd[:-1]) # Get docker run base, exclude cjpegli command itself
                # Use 'ls' as a simple check that works on most Linux images
                # check_cmd.extend(["ls", docker_input_path])
                # Construct check command explicitly, overriding entrypoint
                uid = os.getuid() if hasattr(os, 'getuid') else 1000
                gid = os.getgid() if hasattr(os, 'getgid') else 1000
                check_cmd = [
                    "docker", "run", "--rm",
                    "-u", f"{uid}:{gid}",
                    "-v", f"{get_docker_path(script_dir)}:/work",
                    "-w", "/work",
                    "--entrypoint", "", # Override entrypoint!
                    args.docker_image, # Use image from args
                    "ls", docker_input_path
                ]
                try:
                    # Run without check=True initially, as non-existence is a handled case
                    check_process = run_command(check_cmd, check=False)
                    if check_process.returncode != 0:
                        print(f"    ERROR: File '{docker_input_path}' not found or accessible inside Docker container (ls exit code: {check_process.returncode}). Skipping cjpegli.", file=sys.stderr)
                        if check_process.stderr:
                            print(f"      (Docker ls stderr: {check_process.stderr.strip()})", file=sys.stderr)
                        continue # Skip to next file
                    else:
                        print(f"      File check successful (ls exit code: {check_process.returncode}).")
                        # print(f"      (Docker ls stdout: {check_process.stdout.strip()})") # Optional debug
                except Exception as check_e:
                    print(f"    ERROR: Failed to execute file check command in Docker: {check_e}. Skipping cjpegli.", file=sys.stderr)
                    continue # Skip to next file
                # ===================================

                print(f"      Encoding (Docker) '{docker_input_path}' -> '{docker_output_path}'...")
                # Add the actual cjpegli command part now
                cmd = list(cjpegli_base_cmd)
                cmd.extend(["--distance", str(distance), docker_input_path, docker_output_path])
            else:
                print(f"      Encoding (Local) '{abs_input_path}' -> '{abs_output_jpeg_path}'...")
                cmd = list(cjpegli_base_cmd)
                cmd.extend(["--distance", str(distance), str(abs_input_path), str(abs_output_jpeg_path)])

            try:
                run_command(cmd, check=True)
                print(f"      Encoding complete: {abs_output_jpeg_path}")
            except Exception as e:
                print(f"    ERROR: cjpegli command failed for input '{input_basename}' at distance {distance}. Error: {e}. Skipping.", file=sys.stderr)
                # Clean up potentially partial file if cjpegli failed
                if abs_output_jpeg_path.exists():
                     try:
                         abs_output_jpeg_path.unlink()
                     except OSError as unlink_e:
                         print(f"      Warning: Failed to remove partial output file {abs_output_jpeg_path}: {unlink_e}", file=sys.stderr)
                continue # Skip to next file

            # Check if output JPEG exists before extracting DQT
            if not abs_output_jpeg_path.is_file():
                print(f"    ERROR: Output JPEG '{abs_output_jpeg_path}' not found after cjpegli supposedly succeeded. Skipping.", file=sys.stderr)
                continue

            # Extract DQT and generate Rust entry
            rust_entry = extract_and_format_dqt(
                abs_output_jpeg_path,
                relative_input_include_path,
                distance,
                input_basename,
                input_ext,
                group_name,
                relative_cjpegli_include_path
            )

            if rust_entry:
                # Append to Rust file
                try:
                    with open(abs_rust_test_file, 'a') as f_rust:
                        if not first_entry:
                            f_rust.write(",\n") # Add comma before the next entry
                        f_rust.write(rust_entry)
                    first_entry = False
                    print(f"      Appended data for {input_basename} (d={distance}) to Rust file.")
                except Exception as e:
                     print(f"    ERROR: Failed to append entry to {abs_rust_test_file}: {e}", file=sys.stderr)
                     # Decide if we should stop or continue
                     continue # Continue processing other files
            else:
                 print(f"    ERROR: Failed to generate Rust entry for {input_basename} (d={distance}), likely due to DQT extraction failure. Skipping append.", file=sys.stderr)
                 # Keep the generated JPEG file for inspection? Or delete it? Keep for now.

    # --- Finalize Rust File ---
    print("\n--- Finalizing Rust test file ---")
    try:
        with open(abs_rust_test_file, 'a') as f_rust:
            f_rust.write("\n") # Add newline before footer
            write_rust_footer(f_rust)
        print(f"Rust file finalized: {abs_rust_test_file}")
    except Exception as e:
        print(f"ERROR: Failed to write footer to {abs_rust_test_file}: {e}", file=sys.stderr)

    print(f"\n--- {script_name} finished ---")
    print(f"Check {abs_rust_test_file} and {abs_cjpegli_output_dir}")


if __name__ == "__main__":
    # Removed the check for jpeg-decoder

    try:
        main()
    except Exception as e:
        print(f"\nERROR: Script failed with unhandled exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("Exiting due to unhandled exception in main.", file=sys.stderr) # Added print
        sys.exit(1) 