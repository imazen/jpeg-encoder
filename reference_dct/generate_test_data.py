import os
import subprocess
import json
import pathlib
import glob
import sys
import platform
import random # Added for sampling
import math # Added for ceiling

# --- Configuration ---
DOCKER_IMAGE = "jpegli-builder-image" # Replace with your actual built image name
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent # Assumes script is in reference_dct/
REFERENCE_DCT_DIR = SCRIPT_DIR
TEST_JSON_OUT_DIR = WORKSPACE_ROOT / "src" / "jpegli" / "tests" / "json"
CJPEGLI_PATH_IN_CONTAINER = "/jpegli/build/tools/cjpegli" # Path where cjpegli is built inside the container
JSON_SIZE_CAP_KB = 100 # Cap JSON files at roughly this size

# Define test cases: (group_name, input_glob_pattern, cjpegli_args_list)
# Input pattern is relative to REFERENCE_DCT_DIR
TEST_PATTERNS = [
    (
        "wesaturate_64px_q90",
        "testdata/external/wesaturate/64px/*.png",
        ["--distance", "1.0"]
    ),
    (
        "wesaturate_64px_d4.3",
        "testdata/external/wesaturate/64px/*.png",
        ["--distance", "4.3"]
    ),
    # Add more patterns for other groups and settings
    # ("jxl_chessboard_q90", "testdata/jxl/chessboard/*.png", ["--distance", "1.0"]),
]

# Functions to generate data for (must match filename_base in C++)
TARGET_FUNCTIONS = [
    "SetQuantMatrices",
    "InitQuantizer",
    # "PadInputBuffer", # Removed as requested
    "ComputePreErosion",
    "FuzzyErosion",
    "PerBlockModulations",
    "ComputeAdaptiveQuantField",
    # "RGBToYCbCr", # Removed due to instrumentation issues
    # "DownsampleInputBuffer", # No .testdata file found yet
    # "ComputeCoefficientBlock", # No .testdata file found yet
    # "ComputeTokensForBlock", # No .testdata file found yet
]

# --- Helper Functions ---

def run_command(cmd_list, check=True, **kwargs):
    """Runs a command and prints output/errors."""
    print(f"Running command: {' '.join(cmd_list)}", flush=True)
    try:
        process = subprocess.run(cmd_list, check=check, capture_output=True, text=True, **kwargs)
        if process.stdout:
            print("Command stdout:", process.stdout, flush=True)
        if process.stderr:
            print("Command stderr:", process.stderr, flush=True)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr, flush=True)
        if e.stdout:
            print("Stdout:", e.stdout, file=sys.stderr, flush=True)
        if e.stderr:
            print("Stderr:", e.stderr, file=sys.stderr, flush=True)
        raise
    except Exception as e:
        print(f"Failed to run command: {e}", file=sys.stderr, flush=True)
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

# --- Main Logic ---

def main():
    print("--- Test Data Generation Script ---")
    print(f"Workspace Root: {WORKSPACE_ROOT}")
    print(f"Reference DCT Dir (Host): {REFERENCE_DCT_DIR}")
    print(f"JSON Output Dir (Host): {TEST_JSON_OUT_DIR}")
    print(f"Docker Image: {DOCKER_IMAGE}")

    # Ensure JSON output directory exists
    TEST_JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean old data
    print("\n--- Cleaning previous data ---")
    cleaned_testdata = 0
    all_possible_testdata_files = glob.glob(str(REFERENCE_DCT_DIR / "*.testdata"))
    for f_path_str in all_possible_testdata_files:
         f_path = pathlib.Path(f_path_str)
         print(f"Removing {f_path}")
         f_path.unlink()
         cleaned_testdata += 1
    # Clean specific targets that might not have been run previously
    # for func_name in TARGET_FUNCTIONS:
    #     testdata_file = REFERENCE_DCT_DIR / f"{func_name}.testdata"
    #     if testdata_file.exists():
    #         print(f"Removing {testdata_file}")
    #         testdata_file.unlink()
    #         cleaned_testdata += 1
    print(f"Removed {cleaned_testdata} *.testdata files from {REFERENCE_DCT_DIR}")

    cleaned_json = 0
    all_possible_json_files = glob.glob(str(TEST_JSON_OUT_DIR / "*.json"))
    for f_path_str in all_possible_json_files:
        f_path = pathlib.Path(f_path_str)
        # Specifically keep SetQuantMatrices if it exists and others don't, to avoid full regen if only adding one
        # if f_path.name == "SetQuantMatricesTest.json" and len(all_possible_json_files) > 1:
        #    continue
        print(f"Removing {f_path}")
        f_path.unlink()
        cleaned_json += 1
    # for json_file in TEST_JSON_OUT_DIR.glob("*.json"):
    #     print(f"Removing {json_file}")
    #     json_file.unlink()
    #     cleaned_json += 1
    print(f"Removed {cleaned_json} *.json files from {TEST_JSON_OUT_DIR}")

    # 2. Run cjpegli via Docker for each test pattern
    print("\n--- Running cjpegli via Docker ---")
    host_reference_dct_path_docker = get_docker_path(REFERENCE_DCT_DIR)
    host_src_path_docker = get_docker_path(WORKSPACE_ROOT / "src")

    for group_name, input_pattern, cjpegli_args in TEST_PATTERNS:
        print(f"\nProcessing pattern: {group_name} ({input_pattern})")
        # Find files matching the glob pattern relative to REFERENCE_DCT_DIR
        input_files = sorted(glob.glob(str(REFERENCE_DCT_DIR / input_pattern)))

        if not input_files:
            print(f"WARNING: No input files found for pattern: {input_pattern}", file=sys.stderr)
            continue

        for abs_input_path_str in input_files:
            abs_input_path = pathlib.Path(abs_input_path_str).resolve()
            input_filename = abs_input_path.name
            input_name_no_ext = abs_input_path.stem

            print(f"  Processing input file: {input_filename}")
            if not abs_input_path.is_file():
                print(f"ERROR: Found path is not a file: {abs_input_path}", file=sys.stderr)
                continue

            # Construct output path relative to REFERENCE_DCT_DIR
            # Include group name to avoid collisions if same filename exists in multiple groups
            output_subdir = REFERENCE_DCT_DIR / group_name
            output_subdir.mkdir(parents=True, exist_ok=True)
            # Output filename only, without the group prefix for cjpegli command
            output_filename = f"{input_name_no_ext}.jpg"
            output_relpath_in_group = pathlib.Path(group_name) / output_filename
            abs_output_path = (REFERENCE_DCT_DIR / output_relpath_in_group).resolve()

            # Get paths relative to REFERENCE_DCT_DIR for use inside container
            try:
                container_input_path = str(abs_input_path.relative_to(REFERENCE_DCT_DIR)).replace("\\", "/")
                # Pass only the filename to cjpegli, as its working dir is /data
                container_output_filename = output_filename
            except ValueError:
                print(f"ERROR: Could not make path relative: {abs_input_path} from {REFERENCE_DCT_DIR}")
                continue

            docker_cmd = [
                "docker", "run", "--rm",
                "-e", "GENERATE_RUST_TEST_DATA=1",
                "-v", f"{host_reference_dct_path_docker}:/data",
                "-v", f"{host_src_path_docker}:/src_out",
                "-w", "/data",
                DOCKER_IMAGE,
                # Remove explicit cjpegli path - entrypoint handles it
                # CJPEGLI_PATH_IN_CONTAINER,
                # Add INPUT and OUTPUT
                container_input_path,
                container_output_filename,
            ]
            # Add OPTIONS last
            docker_cmd.extend(cjpegli_args)

            try:
                run_command(docker_cmd)
                print(f"  Successfully processed {input_filename} -> {output_relpath_in_group}")
            except Exception as e:
                print(f"ERROR processing {input_filename}: {e}", file=sys.stderr)
                # For now, exit on first error during generation
                sys.exit(1)

    # 3. Process *.testdata files into *.json
    print("\n--- Aggregating test data into JSON files ---")
    size_cap_bytes = JSON_SIZE_CAP_KB * 1024

    for func_name in TARGET_FUNCTIONS:
        print(f"Processing data for {func_name}...")
        testdata_file = REFERENCE_DCT_DIR / f"{func_name}.testdata"
        json_outfile = TEST_JSON_OUT_DIR / f"{func_name}Test.json"

        unique_lines = set()
        if testdata_file.exists():
            try:
                # Read lines and add non-empty stripped lines to a set for deduplication
                with open(testdata_file, 'r') as f:
                    for line in f:
                        stripped_line = line.strip()
                        if stripped_line:
                            # Remove trailing comma before adding to set
                            if stripped_line.endswith(','):
                                stripped_line = stripped_line[:-1]
                            unique_lines.add(stripped_line)
            except Exception as e:
                print(f"ERROR reading {testdata_file}: {e}", file=sys.stderr)
                continue
        else:
            print(f"  WARNING: Test data file not found: {testdata_file}")

        processed_lines = list(unique_lines)
        num_unique = len(processed_lines)
        print(f"  Found {num_unique} unique test records.")

        # Apply size capping with random sampling if needed
        if num_unique > 0:
            estimated_total_size = sum(len(line) + 1 for line in processed_lines) # +1 for newline/comma
            if estimated_total_size > size_cap_bytes:
                avg_line_size = estimated_total_size / num_unique
                target_num_lines = math.ceil(size_cap_bytes / avg_line_size)
                target_num_lines = min(target_num_lines, num_unique) # Ensure we don't sample more than available
                print(f"  Estimated size ({estimated_total_size / 1024:.1f} KB) exceeds cap ({JSON_SIZE_CAP_KB} KB). Sampling {target_num_lines} records.")
                processed_lines = random.sample(processed_lines, target_num_lines)
            else:
                 print(f"  Estimated size ({estimated_total_size / 1024:.1f} KB) within cap.")

        # Format the final JSON output
        json_output = "[]"
        if processed_lines:
            # Join the selected lines with commas and newlines
            all_lines_str = ",\n".join(processed_lines)
            json_output = f"[\n{all_lines_str}\n]"
            # Optional: Validate JSON structure
            try:
                json.loads(json_output)
                print(f"  Generated valid JSON structure for {func_name} ({len(processed_lines)} records)")
            except json.JSONDecodeError as e:
                print(f"ERROR: Generated invalid JSON for {func_name}: {e}", file=sys.stderr)
                print("-- Problematic JSON content --")
                # Limit printing large invalid JSON
                print(json_output[:2000] + ("... (truncated)" if len(json_output) > 2000 else ""))
                print("----------------------------")
                # continue # Decide whether to write invalid JSON
        else:
            print(f"  No unique data found in {testdata_file}, writing empty JSON array.")

        try:
            json_outfile.write_text(json_output)
            print(f"  Successfully wrote {json_outfile}")
        except Exception as e:
            print(f"ERROR writing {json_outfile}: {e}", file=sys.stderr)

    print("\n--- Test data generation finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: Script failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1) 