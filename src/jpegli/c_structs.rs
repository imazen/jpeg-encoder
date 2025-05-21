use super::structs::JpegColorSpace;
// --- Start C Struct Translations ---
// These structs are translated from jpegli/third_party/libjpeg-turbo/jpeglib.h
// and jpegli/third_party/libjpeg-turbo/jpegint.h.
// Comments indicate the original C struct/field names.
// We use raw pointers (`*mut`, `*const`) and `unsafe extern "C"` function pointers
// initially to closely mirror the C API structure. Making this idiomatic and safe Rust
// will require further refactoring and understanding of ownership/lifetimes.

// Define basic types based on jmorecfg.h and standard C types
// Assuming standard sizes. Might need refinement based on jconfig.h/jmorecfg.h.
pub type JpegSample = u8; // JSAMPLE
pub type JpegCoef = i16;  // JCOEF
pub type JpegOctet = u8;  // JOCTET
pub type JpegDimension = usize; // JDIMENSION
pub type JpegBoolean = bool; // boolean (usually defined as int in C, but conceptually boolean)

// Pointers to image data rows/arrays/images
pub type JpegSampleRow = *mut JpegSample;     // JSAMPROW
pub type JpegSampleArray = *mut JpegSampleRow;  // JSAMPARRAY
pub type JpegSampleImage = *mut JpegSampleArray; // JSAMPIMAGE

// Pointers to DCT coefficient blocks/rows/arrays/images
pub const DCTSIZE: usize = 8;
pub const DCTSIZE2: usize = 64;
pub type JpegBlock = [JpegCoef; DCTSIZE2]; // JBLOCK
pub type JpegBlockRow = *mut JpegBlock;      // JBLOCKROW
pub type JpegBlockArray = *mut JpegBlockRow;   // JBLOCKARRAY
pub type JpegBlockImage = *mut JpegBlockArray; // JBLOCKIMAGE

pub type JpegCoefPtr = *mut JpegCoef;        // JCOEFPTR

// Constants from jpeglib.h
pub const NUM_QUANT_TBLS: usize = 4;
pub const NUM_HUFF_TBLS: usize = 4;
pub const NUM_ARITH_TBLS: usize = 16;
pub const MAX_COMPS_IN_SCAN: usize = 4;
pub const MAX_SAMP_FACTOR: usize = 4;
pub const C_MAX_BLOCKS_IN_MCU: usize = 10;
pub const D_MAX_BLOCKS_IN_MCU: usize = 10; // Should match jconfig.h if overridden

// Forward declarations using raw pointers (unsafe)
// Represents j_common_ptr, j_compress_ptr, j_decompress_ptr
pub type JCommonPtr = *mut JpegCommonStruct;
pub type JCompressPtr = *mut JpegCompressStruct;
pub type JDecompressPtr = *mut JpegDecompressStruct;

// Represents jpeg_marker_struct *
pub type JpegSavedMarkerPtr = *mut JpegMarkerStruct;

// Represents jvirt_sarray_ptr, jvirt_barray_ptr
pub type JpegVirtSarrayPtr = *mut JpegVirtSarrayControl;
pub type JpegVirtBarrayPtr = *mut JpegVirtBarrayControl;

// Placeholder types for opaque structs referenced by pointers
// Their actual definitions might be in jpegint.h or internal C files.
#[repr(C)] pub struct JpegVirtSarrayControl { _private: [u8; 0] }
#[repr(C)] pub struct JpegVirtBarrayControl { _private: [u8; 0] }
#[repr(C)] pub struct JpegCompMaster { _private: [u8; 0] }
#[repr(C)] pub struct JpegCMainController { _private: [u8; 0] }
#[repr(C)] pub struct JpegCPrepController { _private: [u8; 0] }
#[repr(C)] pub struct JpegCCoefController { _private: [u8; 0] }
#[repr(C)] pub struct JpegMarkerWriter { _private: [u8; 0] }
#[repr(C)] pub struct JpegColorConverter { _private: [u8; 0] }
#[repr(C)] pub struct JpegDownsampler { _private: [u8; 0] }
#[repr(C)] pub struct JpegForwardDct { _private: [u8; 0] }
#[repr(C)] pub struct JpegEntropyEncoder { _private: [u8; 0] }
#[repr(C)] pub struct JpegDecompMaster { _private: [u8; 0] }
#[repr(C)] pub struct JpegDMainController { _private: [u8; 0] }
#[repr(C)] pub struct JpegDCoefController { _private: [u8; 0] }
#[repr(C)] pub struct JpegDPostController { _private: [u8; 0] }
#[repr(C)] pub struct JpegInputController { _private: [u8; 0] }
#[repr(C)] pub struct JpegMarkerReader { _private: [u8; 0] }
#[repr(C)] pub struct JpegEntropyDecoder { _private: [u8; 0] }
#[repr(C)] pub struct JpegInverseDct { _private: [u8; 0] }
#[repr(C)] pub struct JpegUpsampler { _private: [u8; 0] }
#[repr(C)] pub struct JpegColorDeconverter { _private: [u8; 0] }
#[repr(C)] pub struct JpegColorQuantizer { _private: [u8; 0] }


// --- Structs from jpeglib.h ---

/// Original C struct: JQUANT_TBL
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JpegQuantTable {
    /// quantval[DCTSIZE2]: quantization step for each coefficient (natural order)
    pub quantval: [u16; DCTSIZE2],
    /// sent_table: TRUE when table has been output (used during compression)
    pub sent_table: JpegBoolean,
}

/// Original C struct: JHUFF_TBL
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JpegHuffTable {
    /// bits[17]: bits[k] = # of symbols with codes of length k bits; bits[0] is unused
    pub bits: [u8; 17],
    /// huffval[256]: The symbols, in order of incr code length
    pub huffval: [u8; 256],
    /// sent_table: TRUE when table has been output (used during compression)
    pub sent_table: JpegBoolean,
}

/// Original C struct: jpeg_component_info
/// Note: This struct exists above as `JpegliComponentInfo`. This is the direct C translation.
/// We might need to reconcile these later.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JpegComponentInfoC {
    // --- Fixed over the whole image ---
    /// component_id: identifier for this component (0..255)
    pub component_id: i32,
    /// component_index: its index in SOF or cinfo->comp_info[]
    pub component_index: i32,
    /// h_samp_factor: horizontal sampling factor (1..4)
    pub h_samp_factor: i32,
    /// v_samp_factor: vertical sampling factor (1..4)
    pub v_samp_factor: i32,
    /// quant_tbl_no: quantization table selector (0..3)
    pub quant_tbl_no: i32,

    // --- May vary between scans (decompression only for these two) ---
    /// dc_tbl_no: DC entropy table selector (0..3)
    pub dc_tbl_no: i32,
    /// ac_tbl_no: AC entropy table selector (0..3)
    pub ac_tbl_no: i32,

    // --- Computed during startup ---
    /// width_in_blocks: Component's size in DCT blocks (excluding dummy blocks)
    pub width_in_blocks: JpegDimension,
    /// height_in_blocks: Component's size in DCT blocks (excluding dummy blocks)
    pub height_in_blocks: JpegDimension,

    // DCT_scaled_size was split into DCT_h_scaled_size and DCT_v_scaled_size in libjpeg v7+
    /// DCT_h_scaled_size: Size of output from one DCT block horizontally (decompression)
    pub dct_h_scaled_size: i32,
    /// DCT_v_scaled_size: Size of output from one DCT block vertically (decompression)
    pub dct_v_scaled_size: i32,
    // Older version compatibility (pre-v7):
    // pub dct_scaled_size: i32,

    /// downsampled_width: actual width in samples (considering Hmax/Vmax and DCT scaling)
    pub downsampled_width: JpegDimension,
    /// downsampled_height: actual height in samples (considering Hmax/Vmax and DCT scaling)
    pub downsampled_height: JpegDimension,
    /// component_needed: do we need the value of this component? (decompression only)
    pub component_needed: JpegBoolean,

    // --- Computed before starting a scan ---
    /// MCU_width: number of blocks per MCU, horizontally
    pub mcu_width: i32,
    /// MCU_height: number of blocks per MCU, vertically
    pub mcu_height: i32,
    /// MCU_blocks: MCU_width * MCU_height
    pub mcu_blocks: i32,
    /// MCU_sample_width: MCU width in samples, MCU_width * DCT_h_scaled_size
    pub mcu_sample_width: i32,
    /// last_col_width: # of non-dummy blocks across in last MCU
    pub last_col_width: i32,
    /// last_row_height: # of non-dummy blocks down in last MCU
    pub last_row_height: i32,

    // --- Decompression only ---
    /// quant_table: Saved quantization table for component; NULL if none yet saved.
    pub quant_table: *mut JpegQuantTable, // Pointer to JQUANT_TBL

    // --- Private storage ---
    /// dct_table: Private per-component storage for DCT or IDCT subsystem.
    pub dct_table: *mut std::ffi::c_void, // void*
}

/// Original C struct: jpeg_scan_info
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JpegScanInfo {
    /// comps_in_scan: number of components encoded in this scan
    pub comps_in_scan: i32,
    /// component_index[MAX_COMPS_IN_SCAN]: their SOF/comp_info[] indexes
    pub component_index: [i32; MAX_COMPS_IN_SCAN],
    /// Ss, Se: progressive JPEG spectral selection parms
    pub ss: i32,
    pub se: i32,
    /// Ah, Al: progressive JPEG successive approx. parms
    pub ah: i32,
    pub al: i32,
}

/// Original C struct: jpeg_marker_struct
#[repr(C)]
#[derive(Debug)] // Clone might be tricky due to raw pointer `data`
pub struct JpegMarkerStruct {
    /// next: next in list, or NULL
    pub next: JpegSavedMarkerPtr, // Self-referential pointer (linked list)
    /// marker: marker code: JPEG_COM, or JPEG_APP0+n
    pub marker: u8,
    /// original_length: # bytes of data in the file
    pub original_length: u32,
    /// data_length: # bytes of data saved at data[]
    pub data_length: u32,
    /// data: the data contained in the marker
    pub data: *mut JpegOctet, // JOCTET *
}


// Enum translations (already present above in similar forms, providing C versions for clarity)

/// Original C enum: J_DCT_METHOD
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDctMethod {
    /// JDCT_ISLOW: accurate integer method
    Islow = 0,
    /// JDCT_IFAST: less accurate integer method [legacy feature]
    Ifast = 1,
    /// JDCT_FLOAT: floating-point method [legacy feature]
    Float = 2,
}

/// Original C enum: J_DITHER_MODE
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDitherMode {
    /// JDITHER_NONE: no dithering
    None = 0,
    /// JDITHER_ORDERED: simple ordered dither
    Ordered = 1,
    /// JDITHER_FS: Floyd-Steinberg error diffusion dither
    Fs = 2,
}

// --- Common Fields ---
// This corresponds to the `jpeg_common_fields` macro in C.
// We embed this struct within JpegCompressStruct and JpegDecompressStruct.
#[repr(C)]
#[derive(Debug)] // Clone needs care due to pointers
pub struct JpegCommonStruct {
    /// err: Error handler module
    pub err: *mut JpegErrorMgr,
    /// mem: Memory manager module
    pub mem: *mut JpegMemoryMgr,
    /// progress: Progress monitor, or NULL if none
    pub progress: *mut JpegProgressMgr,
    /// client_data: Available for use by application
    pub client_data: *mut std::ffi::c_void, // void*
    /// is_decompressor: So common code can tell which is which
    pub is_decompressor: JpegBoolean,
    /// global_state: For checking call sequence validity
    pub global_state: i32,
}

// --- Manager Structs (Public Interfaces) ---

/// Original C struct: jpeg_error_mgr
#[repr(C)]
#[derive(Debug)]
pub struct JpegErrorMgr {
    // --- Methods ---
    /// error_exit: Error exit handler: does not return to caller
    pub error_exit: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,
    /// emit_message: Conditionally emit a trace or warning message
    pub emit_message: Option<unsafe extern "C" fn(cinfo: JCommonPtr, msg_level: i32)>,
    /// output_message: Routine that actually outputs a trace or error message
    pub output_message: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,
    /// format_message: Format a message string for the most recent JPEG error or message
    pub format_message: Option<unsafe extern "C" fn(cinfo: JCommonPtr, buffer: *mut std::os::raw::c_char)>,
    /// reset_error_mgr: Reset error state variables at start of a new image
    pub reset_error_mgr: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,

    // --- Message state ---
    /// msg_code: The message ID code
    pub msg_code: i32,
    /// msg_parm: Parameters for the message (union in C)
    //pub msg_parm: JpegErrorMsgParm, // Union translated to a struct or enum

    // --- Standard state variables ---
    /// trace_level: max msg_level that will be displayed
    pub trace_level: i32,
    /// num_warnings: number of corrupt-data warnings
    pub num_warnings: i64, // long

    // --- Message tables ---
    /// jpeg_message_table: Library errors
    pub jpeg_message_table: *const *const std::os::raw::c_char,
    /// last_jpeg_message: Table contains strings 0..last_jpeg_message
    pub last_jpeg_message: i32,
    /// addon_message_table: Non-library errors
    pub addon_message_table: *const *const std::os::raw::c_char,
    /// first_addon_message: code for first string in addon table
    pub first_addon_message: i32,
    /// last_addon_message: code for last string in addon table
    pub last_addon_message: i32,
}

// C union 'msg_parm' translation (simplistic approach)
// pub const JMSG_STR_PARM_MAX: usize = 80;
// #[repr(C)]
// #[derive(Clone, Copy, Debug)]
// pub union JpegErrorMsgParm {
//     /// i[8]: integer parameters
//     pub i: [i32; 8],
//     /// s[JMSG_STR_PARM_MAX]: string parameter
//     pub s: [std::os::raw::c_char; JMSG_STR_PARM_MAX],
// }


/// Original C struct: jpeg_progress_mgr
#[repr(C)]
#[derive(Debug)]
pub struct JpegProgressMgr {
    /// progress_monitor: Called periodically during processing
    pub progress_monitor: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,
    /// pass_counter: work units completed in this pass
    pub pass_counter: i64, // long
    /// pass_limit: total number of work units in this pass
    pub pass_limit: i64, // long
    /// completed_passes: passes completed so far
    pub completed_passes: i32,
    /// total_passes: total number of passes expected
    pub total_passes: i32,
}

/// Original C struct: jpeg_destination_mgr
#[repr(C)]
#[derive(Debug)]
pub struct JpegDestinationMgr {
    /// next_output_byte: => next byte to write in buffer
    pub next_output_byte: *mut JpegOctet, // JOCTET *
    /// free_in_buffer: # of byte spaces remaining in buffer
    pub free_in_buffer: usize, // size_t

    // --- Methods ---
    /// init_destination: Initialize destination manager
    pub init_destination: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// empty_output_buffer: Called when buffer is full
    pub empty_output_buffer: Option<unsafe extern "C" fn(cinfo: JCompressPtr) -> JpegBoolean>,
    /// term_destination: Cleanup destination manager
    pub term_destination: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
}

/// Original C struct: jpeg_source_mgr
#[repr(C)]
#[derive(Debug)]
pub struct JpegSourceMgr {
    /// next_input_byte: => next byte to read from buffer
    pub next_input_byte: *const JpegOctet, // const JOCTET *
    /// bytes_in_buffer: # of bytes remaining in buffer
    pub bytes_in_buffer: usize, // size_t

    // --- Methods ---
    /// init_source: Initialize source manager
    pub init_source: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// fill_input_buffer: Called when buffer is empty
    pub fill_input_buffer: Option<unsafe extern "C" fn(cinfo: JDecompressPtr) -> JpegBoolean>,
    /// skip_input_data: Skip num_bytes worth of data
    pub skip_input_data: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, num_bytes: i64)>, // long num_bytes
    /// resync_to_restart: Attempt to find next RSTm marker
    pub resync_to_restart: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, desired: i32) -> JpegBoolean>,
    /// term_source: Cleanup source manager
    pub term_source: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
}

// JPOOL constants
pub const JPOOL_PERMANENT: i32 = 0;
pub const JPOOL_IMAGE: i32 = 1;
pub const JPOOL_NUMPOOLS: i32 = 2;


/// Original C struct: jpeg_memory_mgr
#[repr(C)]
#[derive(Debug)]
pub struct JpegMemoryMgr {
    // --- Methods ---
    /// alloc_small: Allocate small object (pool lifetime)
    pub alloc_small: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, sizeofobject: usize) -> *mut std::ffi::c_void>,
    /// alloc_large: Allocate large object (pool lifetime)
    pub alloc_large: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, sizeofobject: usize) -> *mut std::ffi::c_void>,
    /// alloc_sarray: Allocate a 2-D sample array
    pub alloc_sarray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, samplesperrow: JpegDimension, numrows: JpegDimension) -> JpegSampleArray>,
    /// alloc_barray: Allocate a 2-D coefficient block array
    pub alloc_barray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, blocksperrow: JpegDimension, numrows: JpegDimension) -> JpegBlockArray>,
    /// request_virt_sarray: Request virtual sample array
    pub request_virt_sarray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, pre_zero: JpegBoolean, samplesperrow: JpegDimension, numrows: JpegDimension, maxaccess: JpegDimension) -> JpegVirtSarrayPtr>,
    /// request_virt_barray: Request virtual block array
    pub request_virt_barray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32, pre_zero: JpegBoolean, blocksperrow: JpegDimension, numrows: JpegDimension, maxaccess: JpegDimension) -> JpegVirtBarrayPtr>,
    /// realize_virt_arrays: Allocate virtual arrays
    pub realize_virt_arrays: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,
    /// access_virt_sarray: Access rows of virtual sample array
    pub access_virt_sarray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, ptr: JpegVirtSarrayPtr, start_row: JpegDimension, num_rows: JpegDimension, writable: JpegBoolean) -> JpegSampleArray>,
    /// access_virt_barray: Access rows of virtual block array
    pub access_virt_barray: Option<unsafe extern "C" fn(cinfo: JCommonPtr, ptr: JpegVirtBarrayPtr, start_row: JpegDimension, num_rows: JpegDimension, writable: JpegBoolean) -> JpegBlockArray>,
    /// free_pool: Free all objects in specified pool
    pub free_pool: Option<unsafe extern "C" fn(cinfo: JCommonPtr, pool_id: i32)>,
    /// self_destruct: Called when master struct is destroyed
    pub self_destruct: Option<unsafe extern "C" fn(cinfo: JCommonPtr)>,

    // --- Limits ---
    /// max_memory_to_use: Limit on memory allocation (advisory)
    pub max_memory_to_use: i64, // long
    /// max_alloc_chunk: Maximum allocation request accepted by alloc_large
    pub max_alloc_chunk: i64, // long
}


// --- Main Compression/Decompression Structs ---

/// Original C struct: jpeg_compress_struct
#[repr(C)]
#[derive(Debug)]
pub struct JpegCompressStruct {
    // --- Common Fields --- (using composition)
    pub common: JpegCommonStruct, // jpeg_common_fields

    // --- Destination Manager ---
    /// dest: Destination for compressed data
    pub dest: *mut JpegDestinationMgr,

    // --- Source Image Description ---
    /// image_width: input image width
    pub image_width: JpegDimension,
    /// image_height: input image height
    pub image_height: JpegDimension,
    /// input_components: # of color components in input image
    pub input_components: i32,
    /// in_color_space: colorspace of input image
    pub in_color_space: JpegColorSpace, // Assuming this enum maps correctly
    /// input_gamma: image gamma of input image
    pub input_gamma: f64,

    // --- Compression Parameters ---
    // scale_num, scale_denom only available JPEG_LIB_VERSION >= 70
    /// scale_num, scale_denom: fraction by which to scale image
    pub scale_num: u32,
    pub scale_denom: u32,
    /// jpeg_width: scaled JPEG image width
    pub jpeg_width: JpegDimension,
    /// jpeg_height: scaled JPEG image height
    pub jpeg_height: JpegDimension,

    /// data_precision: bits of precision in image data
    pub data_precision: i32,
    /// num_components: # of color components in JPEG image
    pub num_components: i32,
    /// jpeg_color_space: colorspace of JPEG image
    pub jpeg_color_space: JpegColorSpace, // Assuming this enum maps correctly

    /// comp_info: Describes components in SOF order
    pub comp_info: *mut JpegComponentInfoC, // Pointer to array of jpeg_component_info

    /// quant_tbl_ptrs: Ptrs to coefficient quantization tables
    pub quant_tbl_ptrs: [*mut JpegQuantTable; NUM_QUANT_TBLS],
    // q_scale_factor only available JPEG_LIB_VERSION >= 70
    /// q_scale_factor: Scale factors for quantization tables (%)
    pub q_scale_factor: [i32; NUM_QUANT_TBLS],

    /// dc_huff_tbl_ptrs: Ptrs to DC Huffman coding tables
    pub dc_huff_tbl_ptrs: [*mut JpegHuffTable; NUM_HUFF_TBLS],
    /// ac_huff_tbl_ptrs: Ptrs to AC Huffman coding tables
    pub ac_huff_tbl_ptrs: [*mut JpegHuffTable; NUM_HUFF_TBLS],

    /// arith_dc_L: L values for DC arith-coding tables
    pub arith_dc_l: [u8; NUM_ARITH_TBLS], // UINT8 arith_dc_L
    /// arith_dc_U: U values for DC arith-coding tables
    pub arith_dc_u: [u8; NUM_ARITH_TBLS], // UINT8 arith_dc_U
    /// arith_ac_K: Kx values for AC arith-coding tables
    pub arith_ac_k: [u8; NUM_ARITH_TBLS], // UINT8 arith_ac_K

    /// num_scans: # of entries in scan_info array
    pub num_scans: i32,
    /// scan_info: script for multi-scan file, or NULL
    pub scan_info: *const JpegScanInfo, // const jpeg_scan_info *

    /// raw_data_in: TRUE=caller supplies downsampled data
    pub raw_data_in: JpegBoolean,
    /// arith_code: TRUE=arithmetic coding, FALSE=Huffman
    pub arith_code: JpegBoolean,
    /// optimize_coding: TRUE=optimize entropy encoding parms
    pub optimize_coding: JpegBoolean,
    /// CCIR601_sampling: TRUE=first samples are cosited
    pub ccir601_sampling: JpegBoolean,
    // do_fancy_downsampling only available JPEG_LIB_VERSION >= 70
    /// do_fancy_downsampling: TRUE=apply fancy downsampling
    pub do_fancy_downsampling: JpegBoolean,
    /// smoothing_factor: 1..100, or 0 for no input smoothing
    pub smoothing_factor: i32,
    /// dct_method: DCT algorithm selector
    pub dct_method: JpegDctMethod,

    /// restart_interval: MCUs per restart, or 0 for no restart
    pub restart_interval: u32,
    /// restart_in_rows: if > 0, MCU rows per restart interval
    pub restart_in_rows: i32,

    // --- Marker Emission Parameters ---
    /// write_JFIF_header: should a JFIF marker be written?
    pub write_jfif_header: JpegBoolean,
    /// JFIF_major_version: What to write for the JFIF version number
    pub jfif_major_version: u8, // UINT8
    /// JFIF_minor_version: What to write for the JFIF version number
    pub jfif_minor_version: u8, // UINT8
    /// density_unit: JFIF code for pixel size units (0=unknown, 1=dpi, 2=dpcm)
    pub density_unit: u8, // UINT8
    /// X_density: Horizontal pixel density
    pub x_density: u16, // UINT16
    /// Y_density: Vertical pixel density
    pub y_density: u16, // UINT16
    /// write_Adobe_marker: should an Adobe marker be written?
    pub write_adobe_marker: JpegBoolean,

    // --- State Variables ---
    /// next_scanline: index of next scanline to be written (0 .. image_height-1)
    pub next_scanline: JpegDimension,

    // --- Internal Compressor Fields ---
    /// progressive_mode: TRUE if scan script uses progressive mode
    pub progressive_mode: JpegBoolean,
    /// max_h_samp_factor: largest h_samp_factor
    pub max_h_samp_factor: i32,
    /// max_v_samp_factor: largest v_samp_factor
    pub max_v_samp_factor: i32,

    // min_DCT..._scaled_size only available JPEG_LIB_VERSION >= 70
    /// min_DCT_h_scaled_size: smallest DCT_h_scaled_size of any component
    pub min_dct_h_scaled_size: i32,
    /// min_DCT_v_scaled_size: smallest DCT_v_scaled_size of any component
    pub min_dct_v_scaled_size: i32,

    /// total_iMCU_rows: # of iMCU rows to be input to coef ctlr
    pub total_imcu_rows: JpegDimension,

    // --- Per-Scan Fields ---
    /// comps_in_scan: # of JPEG components in this scan
    pub comps_in_scan: i32,
    /// cur_comp_info: Ptrs to component info for current scan
    pub cur_comp_info: [*mut JpegComponentInfoC; MAX_COMPS_IN_SCAN],

    /// MCUs_per_row: # of MCUs across the image
    pub mcus_per_row: JpegDimension,
    /// MCU_rows_in_scan: # of MCU rows in the image
    pub mcu_rows_in_scan: JpegDimension,

    /// blocks_in_MCU: # of DCT blocks per MCU
    pub blocks_in_mcu: i32,
    /// MCU_membership: Index in cur_comp_info for each block in MCU
    pub mcu_membership: [i32; C_MAX_BLOCKS_IN_MCU],

    /// Ss, Se, Ah, Al: progressive JPEG parameters for scan
    pub ss: i32,
    pub se: i32,
    pub ah: i32,
    pub al: i32,

    // block_size, natural_order, lim_Se only available JPEG_LIB_VERSION >= 80
    /// block_size: the basic DCT block size: 1..16
    pub block_size: i32,
    /// natural_order: natural-order position array
    pub natural_order: *const i32, // const int *
    /// lim_Se: min( Se, DCTSIZE2-1 )
    pub lim_se: i32,

    // --- Links to Subobjects ---
    /// master: Master control module
    pub master: *mut JpegCompMaster,
    /// main: Main buffer control
    pub main: *mut JpegCMainController,
    /// prep: Preprocessing controller
    pub prep: *mut JpegCPrepController,
    /// coef: Coefficient controller
    pub coef: *mut JpegCCoefController,
    /// marker: Marker writing module
    pub marker: *mut JpegMarkerWriter,
    /// cconvert: Colorspace conversion module
    pub cconvert: *mut JpegColorConverter,
    /// downsample: Downsampling module
    pub downsample: *mut JpegDownsampler,
    /// fdct: Forward DCT module
    pub fdct: *mut JpegForwardDct,
    /// entropy: Entropy encoding module
    pub entropy: *mut JpegEntropyEncoder,
    /// script_space: workspace for jpeg_simple_progression
    pub script_space: *mut JpegScanInfo,
    /// script_space_size: size of script_space
    pub script_space_size: i32,
}

/// Original C struct: jpeg_decompress_struct
#[repr(C)]
#[derive(Debug)]
pub struct JpegDecompressStruct {
    // --- Common Fields --- (using composition)
    pub common: JpegCommonStruct, // jpeg_common_fields

    // --- Source Manager ---
    /// src: Source of compressed data
    pub src: *mut JpegSourceMgr,

    // --- Image Description (read from header) ---
    /// image_width: nominal image width (from SOF marker)
    pub image_width: JpegDimension,
    /// image_height: nominal image height
    pub image_height: JpegDimension,
    /// num_components: # of color components in JPEG image
    pub num_components: i32,
    /// jpeg_color_space: colorspace of JPEG image
    pub jpeg_color_space: JpegColorSpace, // Assuming this enum maps correctly

    // --- Decompression Parameters (set before start_decompress) ---
    /// out_color_space: colorspace for output
    pub out_color_space: JpegColorSpace, // Assuming this enum maps correctly
    /// scale_num, scale_denom: fraction by which to scale image
    pub scale_num: u32,
    pub scale_denom: u32,
    /// output_gamma: image gamma wanted in output
    pub output_gamma: f64,
    /// buffered_image: TRUE=multiple output passes
    pub buffered_image: JpegBoolean,
    /// raw_data_out: TRUE=downsampled data wanted
    pub raw_data_out: JpegBoolean,
    /// dct_method: IDCT algorithm selector
    pub dct_method: JpegDctMethod,
    /// do_fancy_upsampling: TRUE=apply fancy upsampling
    pub do_fancy_upsampling: JpegBoolean,
    /// do_block_smoothing: TRUE=apply interblock smoothing
    pub do_block_smoothing: JpegBoolean,
    /// quantize_colors: TRUE=colormapped output wanted
    pub quantize_colors: JpegBoolean,
    /// dither_mode: type of color dithering to use
    pub dither_mode: JpegDitherMode,
    /// two_pass_quantize: TRUE=use two-pass color quantization
    pub two_pass_quantize: JpegBoolean,
    /// desired_number_of_colors: max # colors to use in created colormap
    pub desired_number_of_colors: i32,
    /// enable_1pass_quant: enable future use of 1-pass quantizer (buffered)
    pub enable_1pass_quant: JpegBoolean,
    /// enable_external_quant: enable future use of external colormap (buffered)
    pub enable_external_quant: JpegBoolean,
    /// enable_2pass_quant: enable future use of 2-pass quantizer (buffered)
    pub enable_2pass_quant: JpegBoolean,

    // --- Output Image Description (computed by start_decompress) ---
    /// output_width: scaled image width
    pub output_width: JpegDimension,
    /// output_height: scaled image height
    pub output_height: JpegDimension,
    /// out_color_components: # of color components in out_color_space
    pub out_color_components: i32,
    /// output_components: # of color components returned (1 for quantized)
    pub output_components: i32,
    /// rec_outbuf_height: min recommended height of scanline buffer
    pub rec_outbuf_height: i32,

    // --- Colormap (when quantizing) ---
    /// actual_number_of_colors: number of entries in use
    pub actual_number_of_colors: i32,
    /// colormap: The color map as a 2-D pixel array (JSAMPARRAY)
    pub colormap: JpegSampleArray, // JSAMPARRAY

    // --- State Variables (read-only for application) ---
    /// output_scanline: Row index of next scanline to be read (0 .. output_height-1)
    pub output_scanline: JpegDimension,
    /// input_scan_number: Number of SOS markers seen so far
    pub input_scan_number: i32,
    /// input_iMCU_row: Number of iMCU rows completed
    pub input_imcu_row: JpegDimension,
    /// output_scan_number: Nominal scan number being displayed
    pub output_scan_number: i32,
    /// output_iMCU_row: Number of iMCU rows read
    pub output_imcu_row: JpegDimension,
    /// coef_bits: Precision of DCT coefficients (progressive)
    pub coef_bits: *mut [i32; DCTSIZE2], // int (*coef_bits)[DCTSIZE2]

    // --- Internal JPEG Parameters ---
    /// quant_tbl_ptrs: Ptrs to coefficient quantization tables
    pub quant_tbl_ptrs: [*mut JpegQuantTable; NUM_QUANT_TBLS],
    /// dc_huff_tbl_ptrs: Ptrs to DC Huffman coding tables
    pub dc_huff_tbl_ptrs: [*mut JpegHuffTable; NUM_HUFF_TBLS],
    /// ac_huff_tbl_ptrs: Ptrs to AC Huffman coding tables
    pub ac_huff_tbl_ptrs: [*mut JpegHuffTable; NUM_HUFF_TBLS],

    /// data_precision: bits of precision in image data
    pub data_precision: i32,
    /// comp_info: Describes components in SOF order
    pub comp_info: *mut JpegComponentInfoC, // Pointer to array of jpeg_component_info

    // is_baseline only available JPEG_LIB_VERSION >= 80
    /// is_baseline: TRUE if Baseline SOF0 encountered
    pub is_baseline: JpegBoolean,
    /// progressive_mode: TRUE if SOFn specifies progressive mode
    pub progressive_mode: JpegBoolean,
    /// arith_code: TRUE=arithmetic coding, FALSE=Huffman
    pub arith_code: JpegBoolean,

    /// arith_dc_L: L values for DC arith-coding tables
    pub arith_dc_l: [u8; NUM_ARITH_TBLS], // UINT8 arith_dc_L
    /// arith_dc_U: U values for DC arith-coding tables
    pub arith_dc_u: [u8; NUM_ARITH_TBLS], // UINT8 arith_dc_U
    /// arith_ac_K: Kx values for AC arith-coding tables
    pub arith_ac_k: [u8; NUM_ARITH_TBLS], // UINT8 arith_ac_K

    /// restart_interval: MCUs per restart interval, or 0 for no restart
    pub restart_interval: u32,

    // --- Optional Marker Data ---
    /// saw_JFIF_marker: TRUE iff a JFIF APP0 marker was found
    pub saw_jfif_marker: JpegBoolean,
    /// JFIF_major_version: JFIF version number
    pub jfif_major_version: u8, // UINT8
    /// JFIF_minor_version: JFIF version number
    pub jfif_minor_version: u8, // UINT8
    /// density_unit: JFIF code for pixel size units
    pub density_unit: u8, // UINT8
    /// X_density: Horizontal pixel density
    pub x_density: u16, // UINT16
    /// Y_density: Vertical pixel density
    pub y_density: u16, // UINT16
    /// saw_Adobe_marker: TRUE iff an Adobe APP14 marker was found
    pub saw_adobe_marker: JpegBoolean,
    /// Adobe_transform: Color transform code from Adobe marker
    pub adobe_transform: u8, // UINT8
    /// CCIR601_sampling: TRUE=first samples are cosited
    pub ccir601_sampling: JpegBoolean,
    /// marker_list: Head of list of saved markers
    pub marker_list: JpegSavedMarkerPtr, // jpeg_saved_marker_ptr

    // --- Internal Decompressor Fields (computed during startup) ---
    /// max_h_samp_factor: largest h_samp_factor
    pub max_h_samp_factor: i32,
    /// max_v_samp_factor: largest v_samp_factor
    pub max_v_samp_factor: i32,

    // min_DCT..._scaled_size only available JPEG_LIB_VERSION >= 70
    /// min_DCT_h_scaled_size: smallest DCT_h_scaled_size of any component
    pub min_dct_h_scaled_size: i32,
    /// min_DCT_v_scaled_size: smallest DCT_v_scaled_size of any component
    pub min_dct_v_scaled_size: i32,
    // Older version compatibility (pre-v7):
    // pub min_dct_scaled_size: i32,

    /// total_iMCU_rows: # of iMCU rows in image
    pub total_imcu_rows: JpegDimension,
    /// sample_range_limit: table for fast range-limiting
    pub sample_range_limit: *mut JpegSample, // JSAMPLE *

    // --- Per-Scan Fields (input side only) ---
    /// comps_in_scan: # of JPEG components in this scan
    pub comps_in_scan: i32,
    /// cur_comp_info: Ptrs to component info for current scan
    pub cur_comp_info: [*mut JpegComponentInfoC; MAX_COMPS_IN_SCAN],

    /// MCUs_per_row: # of MCUs across the image
    pub mcus_per_row: JpegDimension,
    /// MCU_rows_in_scan: # of MCU rows in the image
    pub mcu_rows_in_scan: JpegDimension,

    /// blocks_in_MCU: # of DCT blocks per MCU
    pub blocks_in_mcu: i32,
    /// MCU_membership: Index in cur_comp_info for each block in MCU
    pub mcu_membership: [i32; D_MAX_BLOCKS_IN_MCU],

    /// Ss, Se, Ah, Al: progressive JPEG parameters for scan
    pub ss: i32,
    pub se: i32,
    pub ah: i32,
    pub al: i32,

    // block_size, natural_order, lim_Se only available JPEG_LIB_VERSION >= 80
    /// block_size: the basic DCT block size: 1..16
    pub block_size: i32,
    /// natural_order: natural-order position array for entropy decode
    pub natural_order: *const i32, // const int *
    /// lim_Se: min( Se, DCTSIZE2-1 ) for entropy decode
    pub lim_se: i32,

    /// unread_marker: Code of unprocessed marker, or 0
    pub unread_marker: i32,

    // --- Links to Subobjects ---
    /// master: Master control module
    pub master: *mut JpegDecompMaster,
    /// main: Main buffer control
    pub main: *mut JpegDMainController,
    /// coef: Coefficient controller
    pub coef: *mut JpegDCoefController,
    /// post: Postprocessing controller
    pub post: *mut JpegDPostController,
    /// inputctl: Input controller
    pub inputctl: *mut JpegInputController,
    /// marker: Marker reading module
    pub marker: *mut JpegMarkerReader,
    /// entropy: Entropy decoding module
    pub entropy: *mut JpegEntropyDecoder,
    /// idct: Inverse DCT module
    pub idct: *mut JpegInverseDct,
    /// upsample: Upsampling module
    pub upsample: *mut JpegUpsampler,
    /// cconvert: Colorspace conversion module
    pub cconvert: *mut JpegColorDeconverter,
    /// cquantize: Color quantization module
    pub cquantize: *mut JpegColorQuantizer,
}

// --- Structs from jpegint.h (Internal Modules) ---

/// Original C enum: J_BUF_MODE (from jpegint.h)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegBufMode {
    /// JBUF_PASS_THRU: Plain stripwise operation
    PassThru = 0,
    /// JBUF_SAVE_SOURCE: Run source subobject only, save output
    SaveSource = 1,
    /// JBUF_CRANK_DEST: Run dest subobject only, using saved data
    CrankDest = 2,
    /// JBUF_SAVE_AND_PASS: Run both subobjects, save output
    SaveAndPass = 3,
}

// Constants for global_state (from jpegint.h)
pub const CSTATE_START: i32 = 100;
pub const CSTATE_SCANNING: i32 = 101;
pub const CSTATE_RAW_OK: i32 = 102;
pub const CSTATE_WRCOEFS: i32 = 103;
pub const DSTATE_START: i32 = 200;
pub const DSTATE_INHEADER: i32 = 201;
pub const DSTATE_READY: i32 = 202;
pub const DSTATE_PRELOAD: i32 = 203;
pub const DSTATE_PRESCAN: i32 = 204;
pub const DSTATE_SCANNING: i32 = 205;
pub const DSTATE_RAW_OK: i32 = 206;
pub const DSTATE_BUFIMAGE: i32 = 207;
pub const DSTATE_BUFPOST: i32 = 208;
pub const DSTATE_RDCOEFS: i32 = 209;
pub const DSTATE_STOPPING: i32 = 210;

// Type aliases from jpegint.h
pub type JpegLong = i64; // JLONG (must hold signed 32-bit, typically long)
pub type JpegUintPtr = usize; // JUINTPTR (typically size_t)

// --- Internal Module Struct Definitions ---
// These mirror the structs in jpegint.h, defining the interfaces
// between internal JPEG library modules. They primarily consist of
// function pointers.

/// Original C struct: jpeg_comp_master (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegCompMasterInternal {
    /// prepare_for_pass
    pub prepare_for_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// pass_startup
    pub pass_startup: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// finish_pass
    pub finish_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    // State variables
    /// call_pass_startup: True if pass_startup must be called
    pub call_pass_startup: JpegBoolean,
    /// is_last_pass: True during last pass
    pub is_last_pass: JpegBoolean,
}

/// Original C struct: jpeg_c_main_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegCMainControllerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr, pass_mode: JpegBufMode)>,
    /// process_data
    pub process_data: Option<unsafe extern "C" fn(cinfo: JCompressPtr, input_buf: JpegSampleArray, in_row_ctr: *mut JpegDimension, in_rows_avail: JpegDimension)>,
}

/// Original C struct: jpeg_c_prep_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegCPrepControllerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr, pass_mode: JpegBufMode)>,
    /// pre_process_data
    pub pre_process_data: Option<unsafe extern "C" fn(
        cinfo: JCompressPtr,
        input_buf: JpegSampleArray,
        in_row_ctr: *mut JpegDimension,
        in_rows_avail: JpegDimension,
        output_buf: JpegSampleImage,
        out_row_group_ctr: *mut JpegDimension,
        out_row_groups_avail: JpegDimension,
    )>,
}

/// Original C struct: jpeg_c_coef_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegCCoefControllerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr, pass_mode: JpegBufMode)>,
    /// compress_data
    pub compress_data: Option<unsafe extern "C" fn(cinfo: JCompressPtr, input_buf: JpegSampleImage) -> JpegBoolean>,
}

/// Original C struct: jpeg_color_converter (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegColorConverterInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// color_convert
    pub color_convert: Option<unsafe extern "C" fn(
        cinfo: JCompressPtr,
        input_buf: JpegSampleArray,
        output_buf: JpegSampleImage,
        output_row: JpegDimension,
        num_rows: i32,
    )>,
}

/// Original C struct: jpeg_downsampler (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegDownsamplerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// downsample
    pub downsample: Option<unsafe extern "C" fn(
        cinfo: JCompressPtr,
        input_buf: JpegSampleImage,
        in_row_index: JpegDimension,
        output_buf: JpegSampleImage,
        out_row_group_index: JpegDimension,
    )>,
    /// need_context_rows: TRUE if need rows above & below
    pub need_context_rows: JpegBoolean,
}

/// Original C struct: jpeg_forward_dct (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegForwardDctInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// forward_DCT
    pub forward_dct: Option<unsafe extern "C" fn(
        cinfo: JCompressPtr,
        compptr: *mut JpegComponentInfoC,
        sample_data: JpegSampleArray,
        coef_blocks: JpegBlockRow,
        start_row: JpegDimension,
        start_col: JpegDimension,
        num_blocks: JpegDimension,
    )>,
}

/// Original C struct: jpeg_entropy_encoder (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegEntropyEncoderInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr, gather_statistics: JpegBoolean)>,
    /// encode_mcu
    pub encode_mcu: Option<unsafe extern "C" fn(cinfo: JCompressPtr, mcu_data: *mut JpegBlockRow) -> JpegBoolean>, // JBLOCKROW* MCU_data
    /// finish_pass
    pub finish_pass: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
}

/// Original C struct: jpeg_marker_writer (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegMarkerWriterInternal {
    /// write_file_header
    pub write_file_header: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// write_frame_header
    pub write_frame_header: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// write_scan_header
    pub write_scan_header: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// write_file_trailer
    pub write_file_trailer: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    /// write_tables_only
    pub write_tables_only: Option<unsafe extern "C" fn(cinfo: JCompressPtr)>,
    // Exported routines for extra markers
    /// write_marker_header
    pub write_marker_header: Option<unsafe extern "C" fn(cinfo: JCompressPtr, marker: i32, datalen: u32)>,
    /// write_marker_byte
    pub write_marker_byte: Option<unsafe extern "C" fn(cinfo: JCompressPtr, val: i32)>,
}


/// Original C struct: jpeg_decomp_master (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegDecompMasterInternal {
    /// prepare_for_output_pass
    pub prepare_for_output_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// finish_output_pass
    pub finish_output_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    // State variables
    /// is_dummy_pass: True during 1st pass for 2-pass quant
    pub is_dummy_pass: JpegBoolean,
    // Partial decompression variables
    /// first_iMCU_col
    pub first_imcu_col: JpegDimension,
    /// last_iMCU_col
    pub last_imcu_col: JpegDimension,
    // MAX_COMPONENTS is not defined here, assuming a reasonable upper bound or requires definition
    // For now, using a placeholder size.
    // pub const MAX_COMPONENTS_INTERNAL: usize = 4; // Placeholder - needs actual definition
    /// first_MCU_col[MAX_COMPONENTS] - Size depends on MAX_COMPONENTS definition
    // pub first_mcu_col: [JpegDimension; MAX_COMPONENTS_INTERNAL],
    /// last_MCU_col[MAX_COMPONENTS] - Size depends on MAX_COMPONENTS definition
    // pub last_mcu_col: [JpegDimension; MAX_COMPONENTS_INTERNAL],
    /// jinit_upsampler_no_alloc
    pub jinit_upsampler_no_alloc: JpegBoolean,
    /// last_good_iMCU_row: Last iMCU row successfully decoded
    pub last_good_imcu_row: JpegDimension,

    // Added fields to avoid placeholder array issues, assuming MAX_COMPONENTS is 4
    pub first_mcu_col_placeholder: [JpegDimension; 4],
    pub last_mcu_col_placeholder: [JpegDimension; 4],

}

/// Original C struct: jpeg_input_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegInputControllerInternal {
    /// consume_input
    pub consume_input: Option<unsafe extern "C" fn(cinfo: JDecompressPtr) -> i32>,
    /// reset_input_controller
    pub reset_input_controller: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// start_input_pass
    pub start_input_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// finish_input_pass
    pub finish_input_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    // State variables
    /// has_multiple_scans: True if file has multiple scans
    pub has_multiple_scans: JpegBoolean,
    /// eoi_reached: True when EOI has been consumed
    pub eoi_reached: JpegBoolean,
}

/// Original C struct: jpeg_d_main_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegDMainControllerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, pass_mode: JpegBufMode)>,
    /// process_data
    pub process_data: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, output_buf: JpegSampleArray, out_row_ctr: *mut JpegDimension, out_rows_avail: JpegDimension)>,
}

/// Original C struct: jpeg_d_coef_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegDCoefControllerInternal {
    /// start_input_pass
    pub start_input_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// consume_data
    pub consume_data: Option<unsafe extern "C" fn(cinfo: JDecompressPtr) -> i32>,
    /// start_output_pass
    pub start_output_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// decompress_data
    pub decompress_data: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, output_buf: JpegSampleImage) -> i32>,
    /// coef_arrays: Pointer to array of coefficient virtual arrays, or NULL
    pub coef_arrays: *mut JpegVirtBarrayPtr, // jvirt_barray_ptr*
}

/// Original C struct: jpeg_d_post_controller (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegDPostControllerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, pass_mode: JpegBufMode)>,
    /// post_process_data
    pub post_process_data: Option<unsafe extern "C" fn(
        cinfo: JDecompressPtr,
        input_buf: JpegSampleImage,
        in_row_group_ctr: *mut JpegDimension,
        in_row_groups_avail: JpegDimension,
        output_buf: JpegSampleArray,
        out_row_ctr: *mut JpegDimension,
        out_rows_avail: JpegDimension,
    )>,
}

// Function pointer type for marker parsers
pub type JpegMarkerParserMethod = Option<unsafe extern "C" fn(cinfo: JDecompressPtr) -> JpegBoolean>;

/// Original C struct: jpeg_marker_reader (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegMarkerReaderInternal {
    /// reset_marker_reader
    pub reset_marker_reader: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// read_markers: Read markers until SOS or EOI
    pub read_markers: Option<unsafe extern "C" fn(cinfo: JDecompressPtr) -> i32>,
    /// read_restart_marker: Read a restart marker (for entropy decoder)
    pub read_restart_marker: JpegMarkerParserMethod, // jpeg_marker_parser_method
    // State variables
    /// saw_SOI: found SOI?
    pub saw_soi: JpegBoolean,
    /// saw_SOF: found SOF?
    pub saw_sof: JpegBoolean,
    /// next_restart_num: next restart number expected (0-7)
    pub next_restart_num: i32,
    /// discarded_bytes: # of bytes skipped looking for a marker
    pub discarded_bytes: u32,
}

/// Original C struct: jpeg_entropy_decoder (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegEntropyDecoderInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// decode_mcu
    pub decode_mcu: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, mcu_data: *mut JpegBlockRow) -> JpegBoolean>, // JBLOCKROW* MCU_data
    // Shared state
    /// insufficient_data: set TRUE after emitting warning
    pub insufficient_data: JpegBoolean,
}

// Function pointer type for inverse DCT methods
pub type InverseDctMethodPtr = Option<unsafe extern "C" fn(
    cinfo: JDecompressPtr,
    compptr: *mut JpegComponentInfoC,
    coef_block: JpegCoefPtr, // JCOEFPTR
    output_buf: JpegSampleArray,
    output_col: JpegDimension,
)>;

/// Original C struct: jpeg_inverse_dct (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegInverseDctInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// inverse_DCT: Array of IDCT method pointers per component
    // Size depends on MAX_COMPONENTS definition - using placeholder
    // pub inverse_dct: [InverseDctMethodPtr; MAX_COMPONENTS_INTERNAL],
    pub inverse_dct_placeholder: [InverseDctMethodPtr; 4],
}

/// Original C struct: jpeg_upsampler (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegUpsamplerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// upsample
    pub upsample: Option<unsafe extern "C" fn(
        cinfo: JDecompressPtr,
        input_buf: JpegSampleImage,
        in_row_group_ctr: *mut JpegDimension,
        in_row_groups_avail: JpegDimension,
        output_buf: JpegSampleArray,
        out_row_ctr: *mut JpegDimension,
        out_rows_avail: JpegDimension,
    )>,
    /// need_context_rows: TRUE if need rows above & below
    pub need_context_rows: JpegBoolean,
}

/// Original C struct: jpeg_color_deconverter (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegColorDeconverterInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// color_convert
    pub color_convert: Option<unsafe extern "C" fn(
        cinfo: JDecompressPtr,
        input_buf: JpegSampleImage,
        input_row: JpegDimension,
        output_buf: JpegSampleArray,
        num_rows: i32,
    )>,
}

/// Original C struct: jpeg_color_quantizer (from jpegint.h)
#[repr(C)]
#[derive(Debug)]
pub struct JpegColorQuantizerInternal {
    /// start_pass
    pub start_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr, is_pre_scan: JpegBoolean)>,
    /// color_quantize
    pub color_quantize: Option<unsafe extern "C" fn(
        cinfo: JDecompressPtr,
        input_buf: JpegSampleArray,
        output_buf: JpegSampleArray,
        num_rows: i32,
    )>,
    /// finish_pass
    pub finish_pass: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
    /// new_color_map
    pub new_color_map: Option<unsafe extern "C" fn(cinfo: JDecompressPtr)>,
}

// --- End C Struct Translations ---


// --- Start common_internal.h Translations ---
// Definitions translated from jpegli/lib/jpegli/common_internal.h

/// Original C++ enum: jpegli::State
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegliState {
    DecNull,
    DecStart,
    DecInHeader,
    DecHeaderDone,
    DecProcessMarkers,
    DecProcessScan,
    EncNull,
    EncStart,
    EncHeader,
    EncReadImage,
    EncWriteCoeffs,
}

// Utility functions (equivalent to DivCeil and RoundUpTo)
// Note: Rust's standard library provides `div_ceil` on integers since 1.73.
// We provide generic versions here for broader compatibility or specific types.
#[inline]
pub fn div_ceil<T>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + From<u8> + Copy,
{
    (a + b - T::from(1u8)) / b
}

#[inline]
pub fn round_up_to<T>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + From<u8>
        + Copy,
{
    div_ceil(a, b) * b
}

// Constants from common_internal.h
pub const K_DCT_BLOCK_SIZE: usize = 64; // kDCTBlockSize (Note: DCTSIZE2 already defined above)
// kMaxComponents = 4 is consistent with MAX_COMPS_IN_SCAN = 4 defined earlier.
pub const K_MAX_COMPONENTS: usize = 4;
pub const K_MAX_QUANT_TABLES: usize = 4; // kMaxQuantTables (consistent with NUM_QUANT_TBLS)
pub const K_JPEG_PRECISION: usize = 8;
pub const K_MAX_HUFFMAN_TABLES: usize = 4; // kMaxHuffmanTables (consistent with NUM_HUFF_TBLS)
pub const K_JPEG_HUFFMAN_MAX_BIT_LENGTH: usize = 16;
pub const K_JPEG_HUFFMAN_ALPHABET_SIZE: usize = 256;
pub const K_JPEG_DC_ALPHABET_SIZE: usize = 12;
pub const K_MAX_DHT_MARKERS: usize = 512;
pub const K_MAX_DIM_PIXELS: usize = 65535;
pub const K_APP1: u8 = 0xE1;
pub const K_APP2: u8 = 0xE2;

// Tag constants
pub const K_ICC_PROFILE_TAG: &[u8; 12] = b"ICC_PROFILE\0"; // Null-terminated? Check usage. Assumed yes.
pub const K_EXIF_TAG: &[u8; 6] = b"Exif\0\0"; // Original C had \\0, assuming one null. Added second for size match.
pub const K_XMP_TAG: &[u8; 28] = b"http://ns.adobe.com/xap/1.0/";

// Coefficient order arrays
/// Original C++ array: jpegli::kJPEGNaturalOrder
pub const JPEG_NATURAL_ORDER: [u32; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40,
    48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
    22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63,
    // extra entries for safety in decoder
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];

/// Original C++ array: jpegli::kJPEGZigZagOrder
pub const JPEG_ZIGZAG_ORDER: [u32; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30,
    41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,
    33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57,
    58, 62, 63,
];



// Represents JPOOL_IMAGE_ALIGNED - Needs mapping to actual JpegMemoryMgr pool IDs/flags
const JPOOL_IMAGE_ALIGNED: i32 = JPOOL_IMAGE; // Placeholder, assume same as JPOOL_IMAGE


// --- End common_internal.h Translations ---

