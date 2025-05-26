use super::structs::{JpegColorSpace, SimplifiedTransferCharacteristics, Subsampling};


// Define the configuration and state for Jpegli encoding
#[derive(Debug, Clone)]
pub(crate) struct ComputedConfig {
    pub distance: f32,
    pub xyb_mode: bool,
    pub use_std_tables: bool,
    pub num_components: usize,
    pub comp_params: Vec<JpegliComponentInfo>,
    pub jpeg_color_space: JpegColorSpace,
    pub cicp_transfer_function: u8,
    pub force_baseline: bool,
    pub add_two_chroma_tables: bool,
    pub use_adaptive_quantization: bool,
    pub subsampling: Subsampling,
}

#[derive(Debug, Clone)] 
pub struct EncodeOptions{
       // Quality/Distance (mutually exclusive)
       pub quality: Option<u8>,
       pub distance: Option<f32>,
       // Flags mirroring cjpegli
       pub xyb_mode: Option<bool>,
       pub use_std_tables: Option<bool>,
       pub use_adaptive_quantization: Option<bool>,
       pub force_dct_bits_baseline: Option<bool>, 
       pub chroma_subsampling: Option<Subsampling>,
       pub jpeg_color_space: JpegColorSpace,
       pub cicp_transfer_function: Option<SimplifiedTransferCharacteristics>,
       pub add_two_chroma_tables: Option<bool>,
}

impl JpegliConfig {
    /// Creates a basic JpegliConfig, computing initial tables.
    /// More complex setup (like AQ) might happen later.
    pub(crate) fn new(distance: f32, sampling_factor: crate::SamplingFactor, num_components: usize) -> Self {
        // Initial computation based on distance and basic params
        let force_baseline = false; // Assuming standard jpegli behavior
        let is_yuv420 = sampling_factor == crate::SamplingFactor::F_2_2 || sampling_factor == crate::SamplingFactor::R_4_2_0;

        // Determine quant_max based on force_baseline
        let quant_max = if force_baseline { 255 } else { 32767 };

        // Always use Jpegli computation path
        // Note: Using constants directly from `quant` module for locality - NOW FROM quant_constants
        let luma_table_raw = crate::jpegli::quant::compute_quant_table_values(
            distance,
            crate::jpegli::quant_constants::GLOBAL_SCALE_YCBCR, // Use quant_constants
            // Slice the first 64 elements (Luma) from the YCbCr base matrix
            crate::jpegli::quant_constants::BASE_QUANT_MATRIX_YCBCR[0..64]
                .try_into()
                .expect("Slice with incorrect length for Luma quant table"),
            true, // non_linear_scaling = true for Jpegli
            false, // is_chroma_420 = false for Luma
            quant_max,
        );
        let chroma_table_raw = crate::jpegli::quant::compute_quant_table_values(
            distance,
            crate::jpegli::quant_constants::GLOBAL_SCALE_YCBCR, // Use quant_constants
            // Slice the next 64 elements (Cb) from the YCbCr base matrix
            crate::jpegli::quant_constants::BASE_QUANT_MATRIX_YCBCR[64..128]
                .try_into()
                .expect("Slice with incorrect length for Chroma quant table"),
            true, // non_linear_scaling = true for Jpegli
            is_yuv420, // is_chroma_420 depends on sampling factor
            quant_max,
        );

        // Removed call to compute_zero_bias_tables - logic needs integration elsewhere
        // Zero bias tables will be initialized later, likely within the encoder state
        let zero_bias_offsets: Vec<[f32; 64]> = Vec::with_capacity(num_components);
        let zero_bias_multipliers: Vec<[f32; 64]> = Vec::with_capacity(num_components);

        Self {
            distance,
            use_float_dct: true, // Default Jpegli behavior often uses float DCT
            use_adaptive_quantization: true, // Default Jpegli behavior often uses AQ
            luma_table_raw,
            chroma_table_raw,
            zero_bias_offsets,
            zero_bias_multipliers,
            adaptive_quant_field: None, // Computed later if needed
        }
    }

    // Add methods to update use_float_dct and use_adaptive_quantization if needed
    pub fn set_float_dct(&mut self, enable: bool) {
        self.use_float_dct = enable;
    }

    pub fn set_adaptive_quantization(&mut self, enable: bool) {
        self.use_adaptive_quantization = enable;
        if !enable {
            self.adaptive_quant_field = None; // Clear AQ field if disabled
        }
    }
}


// void jpegli_set_colorspace(j_compress_ptr cinfo, J_COLOR_SPACE colorspace) {
//     CheckState(cinfo, jpegli::kEncStart);
//     cinfo->jpeg_color_space = colorspace;
//     switch (colorspace) {
//       case JCS_GRAYSCALE:
//         cinfo->num_components = 1;
//         break;
//       case JCS_RGB:
//       case JCS_YCbCr:
//         cinfo->num_components = 3;
//         break;
//       case JCS_CMYK:
//       case JCS_YCCK:
//         cinfo->num_components = 4;
//         break;
//       case JCS_UNKNOWN:
//         cinfo->num_components =
//             std::min<int>(jpegli::kMaxComponents, cinfo->input_components);
//         break;
//       default:
//         JPEGLI_ERROR("Unsupported jpeg colorspace %d", colorspace);
//     }
//     // Adobe marker is only needed to distinguish CMYK and YCCK JPEGs.
//     cinfo->write_Adobe_marker = TO_JXL_BOOL(cinfo->jpeg_color_space == JCS_YCCK);
//     if (cinfo->comp_info == nullptr) {
//       cinfo->comp_info =
//           jpegli::Allocate<jpeg_component_info>(cinfo, MAX_COMPONENTS);
//     }
//     memset(cinfo->comp_info, 0,
//            jpegli::kMaxComponents * sizeof(jpeg_component_info));
//     for (int c = 0; c < cinfo->num_components; ++c) {
//       jpeg_component_info* comp = &cinfo->comp_info[c];
//       comp->component_index = c;
//       comp->component_id = c + 1;
//       comp->h_samp_factor = 1;
//       comp->v_samp_factor = 1;
//       comp->quant_tbl_no = 0;
//       comp->dc_tbl_no = 0;
//       comp->ac_tbl_no = 0;
//     }
//     if (colorspace == JCS_RGB) {
//       cinfo->comp_info[0].component_id = 'R';
//       cinfo->comp_info[1].component_id = 'G';
//       cinfo->comp_info[2].component_id = 'B';
//       if (cinfo->master->xyb_mode) {
//         // Subsample blue channel.
//         cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
//         cinfo->comp_info[1].h_samp_factor = cinfo->comp_info[1].v_samp_factor = 2;
//         cinfo->comp_info[2].h_samp_factor = cinfo->comp_info[2].v_samp_factor = 1;
//         // Use separate quantization tables for each component
//         cinfo->comp_info[1].quant_tbl_no = 1;
//         cinfo->comp_info[2].quant_tbl_no = 2;
//       }
//     } else if (colorspace == JCS_CMYK) {
//       cinfo->comp_info[0].component_id = 'C';
//       cinfo->comp_info[1].component_id = 'M';
//       cinfo->comp_info[2].component_id = 'Y';
//       cinfo->comp_info[3].component_id = 'K';
//     } else if (colorspace == JCS_YCbCr || colorspace == JCS_YCCK) {
//       // Use separate quantization and Huffman tables for luma and chroma
//       cinfo->comp_info[1].quant_tbl_no = 1;
//       cinfo->comp_info[2].quant_tbl_no = 1;
//       cinfo->comp_info[1].dc_tbl_no = cinfo->comp_info[1].ac_tbl_no = 1;
//       cinfo->comp_info[2].dc_tbl_no = cinfo->comp_info[2].ac_tbl_no = 1;
//       // Use chroma subsampling by default
//       cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
//       if (colorspace == JCS_YCCK) {
//         cinfo->comp_info[3].h_samp_factor = cinfo->comp_info[3].v_samp_factor = 2;
//       }
//     }
//   }

// if (!jpeg_settings.chroma_subsampling.empty()) {
//     if (jpeg_settings.chroma_subsampling == "444") {
//       cinfo.comp_info[0].h_samp_factor = 1;
//       cinfo.comp_info[0].v_samp_factor = 1;
//     } else if (jpeg_settings.chroma_subsampling == "440") {
//       cinfo.comp_info[0].h_samp_factor = 1;
//       cinfo.comp_info[0].v_samp_factor = 2;
//     } else if (jpeg_settings.chroma_subsampling == "422") {
//       cinfo.comp_info[0].h_samp_factor = 2;
//       cinfo.comp_info[0].v_samp_factor = 1;
//     } else if (jpeg_settings.chroma_subsampling == "420") {
//       cinfo.comp_info[0].h_samp_factor = 2;
//       cinfo.comp_info[0].v_samp_factor = 2;
//     } else {
//       return false;
//     }
//     for (int i = 1; i < cinfo.num_components; ++i) {
//       cinfo.comp_info[i].h_samp_factor = 1;
//       cinfo.comp_info[i].v_samp_factor = 1;
//     }
//   } else if (!jpeg_settings.xyb) {
//     // Default is no chroma subsampling.
//     cinfo.comp_info[0].h_samp_factor = 1;
//     cinfo.comp_info[0].v_samp_factor = 1;
//   }
