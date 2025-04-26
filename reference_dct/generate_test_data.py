import os
import subprocess
import json
import pathlib
import glob
import sys
import platform

# --- Configuration ---
DOCKER_IMAGE = "jpegli-builder-image" # Replace with your actual built image name
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent # Assumes script is in reference_dct/
REFERENCE_DCT_DIR = SCRIPT_DIR
TEST_JSON_OUT_DIR = WORKSPACE_ROOT / "src" / "jpegli" / "tests" / "json"
CJPEGLI_PATH_IN_CONTAINER = "/jpegli/build/tools/cjpegli" # Path where cjpegli is built inside the container

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
    # Add other function names as they are instrumented
    # "InitQuantizer",
    # "PadInputBuffer", # Might need custom handling if split by row/comp
    # "ComputePreErosion",
    # "FuzzyErosion",
    # "PerBlockModulations",
    # "ComputeAdaptiveQuantField",
    # "RGBToYCbCr",
    # "DownsampleInputBuffer",
    # "ComputeCoefficientBlock",
    # "ComputeTokensForBlock",
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
    for func_name in TARGET_FUNCTIONS:
        testdata_file = REFERENCE_DCT_DIR / f"{func_name}.testdata"
        if testdata_file.exists():
            print(f"Removing {testdata_file}")
            testdata_file.unlink()
            cleaned_testdata += 1
    print(f"Removed {cleaned_testdata} *.testdata files from {REFERENCE_DCT_DIR}")

    cleaned_json = 0
    for json_file in TEST_JSON_OUT_DIR.glob("*.json"):
        print(f"Removing {json_file}")
        json_file.unlink()
        cleaned_json += 1
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
    for func_name in TARGET_FUNCTIONS:
        print(f"Processing data for {func_name}...")
        testdata_file = REFERENCE_DCT_DIR / f"{func_name}.testdata"
        json_outfile = TEST_JSON_OUT_DIR / f"{func_name}Test.json"

        all_lines = ""
        if testdata_file.exists():
            try:
                # Read all lines, strip trailing whitespace from each
                lines = [line.strip() for line in testdata_file.read_text().splitlines() if line.strip()]
                # Join with newline, ensuring the last comma is handled
                if lines:
                    # Remove trailing comma from the last actual JSON line
                    if lines[-1].endswith(','):
                       lines[-1] = lines[-1][:-1]
                    all_lines = "\n".join(lines)

            except Exception as e:
                print(f"ERROR reading or processing {testdata_file}: {e}", file=sys.stderr)
                continue

        json_output = "[]"
        if all_lines:
            json_output = f"[\n{all_lines}\n]"
            # Optional: Validate JSON structure here before writing
            try:
                json.loads(json_output)
                print(f"  Generated valid JSON structure for {func_name}")
            except json.JSONDecodeError as e:
                print(f"ERROR: Generated invalid JSON for {func_name}: {e}", file=sys.stderr)
                print("-- Problematic JSON content --")
                print(json_output)
                print("----------------------------")
                # continue # Decide whether to write invalid JSON
        else:
            print(f"  No data found in {testdata_file}, writing empty JSON array.")

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