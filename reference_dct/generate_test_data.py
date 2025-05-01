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


# EXPLICIT_TEST_FILES: List[Tuple[str, str]] = [
#     ("wesaturate-500px", "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png"),
#     ("wesaturate-64px", "external/wesaturate/64px/vgqcws_vin_709_g1.png"),
# ]
# DISTANCES=(0 0.5 1 1.4 2.3 3.2 4.3 7.4 13 25) # Add more distances as needed

DISTANCES = [0, 0.1, 0.4, 0.5, 0.9, 1, 1.1, 2.3, 4.3, 7.4, 25]
FLAGS = [["--chroma_subsampling", "420"], ["--chroma_subsampling", "422"], ["--chroma_subsampling", "440"], ["--chroma_subsampling", "444"],
                ["--progressive_level", "0"], 
                ["--xyb"], 
                ["--std_quant"], 
                ["--noadaptive_quantization"]]
EXPLICIT_TEST_FILES = [
    ("wesaturate-64px", "testdata/external/wesaturate/64px/vgqcws_vin_709_g1.png"),
    ("wesaturate-500px", "testdata/external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png"),
]
# Define test cases: (group_name, input_glob_pattern, cjpegli_args_list)
# Input pattern is relative to REFERENCE_DCT_DIR
TEST_PATTERNS = []

#     (
#         "wesaturate_64px_q90",
#         "testdata/external/wesaturate/64px/*.png",
#         ["--distance", "1.0"]
#     ),
#     (
#         "wesaturate_64px_d4.3",
#         "testdata/external/wesaturate/64px/*.png",
#         ["--distance", "4.3"]
#     ),
#     # Add more patterns for other groups and settings
#     # ("jxl_chessboard_q90", "testdata/jxl/chessboard/*.png", ["--distance", "1.0"]),
# ]

for distance in DISTANCES:
    for input_pattern in EXPLICIT_TEST_FILES:   
        for flags in FLAGS:
            TEST_PATTERNS.append(
                (
                    f"{input_pattern[0]}_d{distance}",
                    input_pattern[1],
                    ["--distance", str(distance)] + flags
                )
            )

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
    # --- Initialize Aggregated Data Store ---
    aggregated_data = {func_name: [] for func_name in TARGET_FUNCTIONS}
    processed_files_count = 0
    # ----------------------------------------

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

    # === Build Docker Image ===
    print("\n--- Building Docker image --- ")
    dockerfile_path = WORKSPACE_ROOT / "jpegli" / "docker" / "Dockerfile"
    build_context_path = WORKSPACE_ROOT / "jpegli"
    if not dockerfile_path.is_file():
        print(f"ERROR: Dockerfile not found at {dockerfile_path}", file=sys.stderr)
        sys.exit(1)
    if not build_context_path.is_dir():
        print(f"ERROR: Docker build context directory not found at {build_context_path}", file=sys.stderr)
        sys.exit(1)

    build_cmd = [
        "docker", "build",
        "-t", DOCKER_IMAGE,
        "-f", str(dockerfile_path),
        str(build_context_path)
    ]
    try:
        # Run the build command from the WORKSPACE_ROOT, as the paths are relative to it.
        run_command(build_cmd, check=True, cwd=WORKSPACE_ROOT)
        print(f"Successfully built Docker image: {DOCKER_IMAGE}")
    except Exception as e:
        print(f"ERROR: Failed to build Docker image '{DOCKER_IMAGE}'. Ensure Docker is running and the build context is correct.", file=sys.stderr)
        print(f"Build command was: {' '.join(map(str, build_cmd))}")
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)
    # === End Build Docker Image ===

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

            # === Process generated .testdata immediately ===
            print(f"    Processing .testdata files generated by {input_filename}...")
            for func_name in TARGET_FUNCTIONS:
                testdata_file = REFERENCE_DCT_DIR / f"{func_name}.testdata"
                if testdata_file.exists():
                    lines_processed_for_file = 0
                    try:
                        with open(testdata_file, 'r') as f_td:
                            for line in f_td:
                                stripped_line = line.strip()
                                if stripped_line:
                                    # Remove trailing comma before parsing
                                    if stripped_line.endswith(','):
                                        stripped_line = stripped_line[:-1]
                                    try:
                                        data_obj = json.loads(stripped_line)
                                        # Inject context
                                        data_obj["source_file"] = input_filename
                                        data_obj["command_params"] = cjpegli_args
                                        # Store the augmented object
                                        aggregated_data[func_name].append(data_obj)
                                        lines_processed_for_file += 1
                                    except json.JSONDecodeError as json_e:
                                        print(f"      WARN: Skipping invalid JSON line in {testdata_file}: {json_e}", file=sys.stderr)
                                        print(f"        Line: {stripped_line[:100]}...", file=sys.stderr)
                        if lines_processed_for_file > 0:
                             print(f"      Processed {lines_processed_for_file} lines from {testdata_file.name}")
                        # Delete the file after processing
                        testdata_file.unlink()
                    except Exception as read_e:
                        print(f"      ERROR reading or processing {testdata_file}: {read_e}", file=sys.stderr)
                        # Decide whether to continue or exit
                        # continue
            # ============================================
            processed_files_count += 1

    # 3. Process aggregated data into final *.json files
    print(f"\n--- Aggregating collected data ({processed_files_count} files processed) into JSON files ---")
    size_cap_bytes = JSON_SIZE_CAP_KB * 1024

    # Reworked loop to use aggregated_data
    # for func_name in TARGET_FUNCTIONS:
    for func_name, collected_objects in aggregated_data.items():
        print(f"Aggregating data for {func_name}...")
        json_outfile = TEST_JSON_OUT_DIR / f"{func_name}Test.json"

        # --- Deduplication based on original C++ data --- 
        deduplicated_objects = []
        seen_representations = set()
        num_original = len(collected_objects)

        for obj in collected_objects:
            try:
                # Create a temporary copy excluding the injected fields
                temp_obj = obj.copy()
                temp_obj.pop("source_file", None) 
                temp_obj.pop("command_params", None)
                # Create a stable, hashable representation (sorted JSON string)
                representation = json.dumps(temp_obj, sort_keys=True)
                
                if representation not in seen_representations:
                    seen_representations.add(representation)
                    deduplicated_objects.append(obj) # Keep the original object with all fields
            except TypeError as e:
                 # Handle cases where the object might not be serializable/comparable easily
                 print(f"  WARN: Could not create representation for deduplication for an object in {func_name}: {e}. Keeping object.", file=sys.stderr)
                 deduplicated_objects.append(obj) # Keep if representation fails
            except Exception as e:
                 print(f"  WARN: Unexpected error during deduplication for an object in {func_name}: {e}. Keeping object.", file=sys.stderr)
                 deduplicated_objects.append(obj) # Keep if representation fails


        processed_objects = deduplicated_objects # Use the deduplicated list
        num_records = len(processed_objects)
        num_removed = num_original - num_records
        print(f"  Found {num_original} records, {num_records} unique after deduplication (removed {num_removed})." )

        # Apply size capping with random sampling if needed
        if num_records > 0:
            # Estimate size based on serializing each object
            try:
                 estimated_total_size = sum(len(json.dumps(obj)) + 1 for obj in processed_objects) # +1 for newline/comma
            except Exception as dump_e:
                 print(f"  WARN: Could not estimate JSON size accurately due to serialization error: {dump_e}. Skipping size capping.", file=sys.stderr)
                 estimated_total_size = 0 # Avoid capping if estimation fails

            if estimated_total_size > size_cap_bytes and estimated_total_size > 0:
                avg_obj_size = estimated_total_size / num_records
                if avg_obj_size > 0:
                     target_num_lines = math.ceil(size_cap_bytes / avg_obj_size)
                     target_num_lines = min(target_num_lines, num_records) # Ensure we don't sample more than available
                     print(f"  Estimated size ({estimated_total_size / 1024:.1f} KB) exceeds cap ({JSON_SIZE_CAP_KB} KB). Sampling {target_num_lines} records.")
                     processed_objects = random.sample(processed_objects, target_num_lines)
                     num_records = len(processed_objects) # Update count after sampling
                else:
                     print(f"  WARN: Average object size is zero, cannot perform size capping.", file=sys.stderr)

            else:
                 print(f"  Estimated size ({estimated_total_size / 1024:.1f} KB) within cap.")

        # Format the final JSON output
        json_output = "[]" # Default to empty array
        if processed_objects:
            try:
                # Serialize the list of dictionaries with indentation for readability
                json_output = json.dumps(processed_objects, indent=2)
                print(f"  Generated valid JSON structure for {func_name} ({num_records} records)")
            except Exception as dump_e:
                print(f"ERROR: Failed to serialize final data for {func_name}: {dump_e}", file=sys.stderr)
                # Optionally write empty array or skip writing
                # continue
        else:
            print(f"  No data collected for {func_name}, writing empty JSON array.")

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