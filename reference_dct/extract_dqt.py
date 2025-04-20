import sys
import struct
import os

# --- DQT Extraction Logic (from previous version, slightly modified error handling) ---
def extract_dqt(jpeg_path):
    tables = {}
    try:
        with open(jpeg_path, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {jpeg_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    pos = 0
    while pos < len(data):
        if data[pos:pos+2] == b'\xff\xd8': # SOI
            pos += 2
            continue
        if pos >= len(data) or data[pos] != 0xff:
            # Allow skipping non-marker bytes
            pos += 1
            continue

        marker = data[pos:pos+2]
        pos += 2

        if marker in [b'\xff\xd9', b'\xff\xd0', b'\xff\xd1', b'\xff\xd2', b'\xff\xd3', b'\xff\xd4', b'\xff\xd5', b'\xff\xd6', b'\xff\xd7', b'\xff\x01']: # EOI or standalone
            continue

        if pos + 2 > len(data):
            print(f"Error: Truncated JPEG file, missing length after marker {marker.hex()} at pos {pos-2}", file=sys.stderr)
            break
        length = struct.unpack('>H', data[pos:pos+2])[0]
        if length < 2:
            print(f"Error: Invalid segment length {length} for marker {marker.hex()} at pos {pos-2}", file=sys.stderr)
            # Try to skip marker code if length is wrong
            pos += 0 # Stay put to reread marker hopefully
            continue

        if pos + length > len(data):
            print(f"Error: Segment length {length} for marker {marker.hex()} exceeds file size at pos {pos-2}", file=sys.stderr)
            break # Prevent reading past end of data
        payload_pos = pos + 2
        segment_end = payload_pos + length - 2 # Position after the payload

        if marker == b'\xff\xdb': # DQT Marker
            current_pos_in_payload = payload_pos
            while current_pos_in_payload < segment_end:
                if current_pos_in_payload + 1 > segment_end:
                    print(f"Error: Truncated DQT segment at pos {current_pos_in_payload}", file=sys.stderr)
                    break
                pq_tq = data[current_pos_in_payload]
                current_pos_in_payload += 1
                precision = (pq_tq >> 4) & 0x0F
                table_id = pq_tq & 0x0F

                if table_id > 3:
                    print(f"Warning: Invalid table ID {table_id} in DQT segment at pos {current_pos_in_payload-1}", file=sys.stderr)
                    break # Stop parsing this DQT segment

                num_quant_values = 64
                bytes_per_val = 2 if precision == 1 else 1
                num_bytes_needed = num_quant_values * bytes_per_val

                if current_pos_in_payload + num_bytes_needed > segment_end:
                    print(f"Error: DQT table data truncated in segment at pos {current_pos_in_payload}. Required {num_bytes_needed}, available {segment_end - current_pos_in_payload}", file=sys.stderr)
                    break

                table = []
                if precision == 1: # 16-bit
                    for _ in range(num_quant_values):
                        table.append(struct.unpack('>H', data[current_pos_in_payload:current_pos_in_payload+2])[0])
                        current_pos_in_payload += 2
                else: # 8-bit
                    for _ in range(num_quant_values):
                        table.append(data[current_pos_in_payload])
                        current_pos_in_payload += 1

                if len(table) == num_quant_values:
                    # Multiply by 8 as per original script's intent for Rust constants
                    tables[table_id] = [x * 8 for x in table]
                else:
                    print(f"Warning: Incomplete table data read for table {table_id} ({len(table)}/{num_quant_values} vals) in DQT segment", file=sys.stderr)
                    break # Stop parsing this DQT segment

            # Ensure parser moves past this segment regardless of inner loops
            pos = segment_end
        else:
            # Skip other segments
            pos = segment_end

    # Add default tables if missing
    if 0 not in tables:
        print(f"Warning: Luma table (Index 0) not found in {jpeg_path}! Using default empty.", file=sys.stderr)
        tables[0] = [0]*64
    if 1 not in tables:
        print(f"Warning: Chroma table (Index 1) not found in {jpeg_path}! Using default empty.", file=sys.stderr)
        tables[1] = [0]*64

    return tables
# --- End DQT Extraction Logic ---

def format_array_as_rust(arr):
    if not arr or len(arr) != 64:
        arr = [0]*64 # Default to empty on error
    rows = []
    for i in range(0, 64, 8):
        rows.append("        " + ", ".join(map(str, arr[i:i+8]))) # Indent lines within struct
    # Indent the whole array block
    return "    [\n" + ",\n".join(rows) + "\n    ]"

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 extract_dqt.py <jpeg_file> <relative_include_path> <distance> <orig_filename> <orig_extension> <source_group>", file=sys.stderr)
        sys.exit(1)

    jpeg_file = sys.argv[1]
    relative_include_path_arg = sys.argv[2]
    distance_arg = float(sys.argv[3])
    orig_filename_arg = sys.argv[4]
    orig_extension_arg = sys.argv[5].upper()
    source_group_arg = sys.argv[6]

    dqt_tables = extract_dqt(jpeg_file)

    luma_rust_str = format_array_as_rust(dqt_tables.get(0))
    chroma_rust_str = format_array_as_rust(dqt_tables.get(1))

    # Print just the Rust struct instance, indented for the array
    print(f"        ReferenceQuantTestData {{")
    print(f"            source_group: \"{source_group_arg}\",")
    print(f"            input_filename: \"{orig_filename_arg}\",")
    print(f"            input_format: \"{orig_extension_arg}\",")
    print(f"            input_data: include_bytes!(\"{relative_include_path_arg}\"),")
    print(f"            cjpegli_distance: {distance_arg:.1}, ")
    print(f"            expected_luma_dqt: {luma_rust_str},")
    print(f"            expected_chroma_dqt: {chroma_rust_str}")
    print(f"        }}") # No comma here, handled by bash script

