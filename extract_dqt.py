import sys
import struct

def extract_dqt(jpeg_path):
    tables = {}
    try:
        with open(jpeg_path, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {jpeg_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    pos = 0
    while pos < len(data):
        # Find SOI marker (should be at the beginning)
        if data[pos:pos+2] == b'\xff\xd8':
            pos += 2
            continue

        # Find any marker
        if data[pos] != 0xff:
            # Should not happen in a valid JPEG structure after SOI unless data segment
            # This basic parser might fail here on some JPEGs.
            # print(f"Warning: Expected marker byte (0xFF) at pos {pos}, found {data[pos]:02x}")
            pos += 1 # Try to resync
            continue

        marker = data[pos:pos+2]
        pos += 2

        # End of Image or Standalone markers
        if marker in [b'\xff\xd9', b'\xff\xd0', b'\xff\xd1', b'\xff\xd2', b'\xff\xd3', b'\xff\xd4', b'\xff\xd5', b'\xff\xd6', b'\xff\xd7', b'\xff\x01']:
             continue

        # Markers with length
        if pos + 2 > len(data):
            print(f"Error: Truncated JPEG file, missing length after marker {marker.hex()} at pos {pos-2}")
            break
        length = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2

        if marker == b'\xff\xdb': # DQT Marker
            segment_end = pos + length - 2
            if segment_end > len(data):
                 print(f"Error: DQT segment length {length} exceeds file size at pos {pos-4}")
                 break

            while pos < segment_end:
                if pos + 1 > len(data):
                    print(f"Error: Truncated DQT segment at pos {pos}")
                    break
                pq_tq = data[pos]
                pos += 1
                precision = (pq_tq >> 4) & 0x0F # Pq: 0 = 8-bit, 1 = 16-bit
                table_id = pq_tq & 0x0F        # Tq: Quantization table destination identifier

                if table_id > 3:
                    print(f"Warning: Invalid table ID {table_id} in DQT segment at pos {pos-1}")
                    # Attempt to skip the rest of this table definition within the segment
                    bytes_per_val = 2 if precision == 1 else 1
                    pos = segment_end # Skip rest of DQT segment if invalid ID found
                    continue

                num_bytes = 128 if precision == 1 else 64
                if pos + num_bytes > segment_end:
                     print(f"Error: DQT table data truncated in segment at pos {pos}. Required {num_bytes}, available {segment_end - pos}")
                     break # Exit inner loop

                table = []
                if precision == 1: # 16-bit
                    for _ in range(64):
                        if pos + 2 > segment_end: break
                        table.append(struct.unpack('>H', data[pos:pos+2])[0])
                        pos += 2
                else: # 8-bit
                    for _ in range(64):
                        if pos + 1 > segment_end: break
                        table.append(data[pos])
                        pos += 1

                if len(table) == 64:
                    tables[table_id] = table
                else:
                     print(f"Warning: Incomplete table data read for table {table_id} in DQT segment")
                     # Attempt to recover by moving pos to expected end of this table
                     pos = min(pos + num_bytes - (len(table) * (2 if precision==1 else 1)), segment_end)


            # Ensure pos is at the end of the segment even if parsing failed partially
            pos = segment_end

        else:
            # Skip other segments
            pos += length - 2 # Move to the start of the next marker

    return tables

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_dqt.py <jpeg_file>")
        sys.exit(1)

    jpeg_file = sys.argv[1]
    dqt_tables = extract_dqt(jpeg_file)

    if 0 not in dqt_tables:
        print("// Luma table (Index 0) not found!")
    else:
        # Multiply by 8 for Rust constants
        luma_str = ", ".join(map(lambda x: str(x * 8), dqt_tables[0]))
        print(f"const REF_LUMA_D1_0: [u16; 64] = [\n    {luma_str}\n];")

    if 1 not in dqt_tables:
        print("// Chroma table (Index 1) not found!")
    else:
         # Multiply by 8 for Rust constants
        chroma_str = ", ".join(map(lambda x: str(x * 8), dqt_tables[1]))
        print(f"const REF_CHROMA_D1_0: [u16; 64] = [\n    {chroma_str}\n];")

