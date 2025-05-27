use crate::{Density, EncodingError};

use super::{quant::{quality_to_distance, MAX_COMPONENTS}, structs::{JpegColorSpace, JpegliComponentSettings, SimplifiedTransferCharacteristics, Subsampling}};

const MAX_SAMP_FACTOR: u8 = 4;
const JPEG_MAX_DIMENSION: usize = 65500;
const DCTSIZE: u8 = 8;


// Define the configuration and state for Jpegli encoding
#[derive(Debug, Clone)]
pub(crate) struct ComputedEncodeConfig {
    pub distance: f32,
    pub quality: Option<u8>,
    pub xyb_mode: bool,
    pub use_std_tables: bool,
    pub num_components: usize,
    pub luma_component_index: usize,
    pub comp_params: Vec<JpegliComponentSettings>,
    pub max_h_samp_factor: u8,
    pub max_v_samp_factor: u8,
    pub jpeg_color_space: JpegColorSpace,
    pub optimize_coding: bool,
    pub cicp_transfer_function: SimplifiedTransferCharacteristics,
    // pub force_dct_bits_baseline: bool, We always have 8 bit dct. No point in implementing 12/16 bit when nobody can read it outside of medical.

    pub add_two_chroma_tables: bool,
    pub use_adaptive_quantization: bool,
    pub subsampling: Subsampling,
    pub write_jfif_density: Option<Density>,
    /// Adobe marker is only needed to distinguish CMYK and YCCK JPEGs.
    pub write_adobe_ycck: bool,
    pub icc_profile: Option<Vec<u8>>,
    pub app_segments: Option<Vec<AppSegment>>,
    pub smoothing: u8,
    pub progressive_level: u8,
    /// Ignored if restart_interval_in_rows is set.
    pub restart_interval: u16,
    /// Preferred over restart_interval.
    pub restart_interval_in_rows: u16,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentDimensions{
    pub downsampled_width: usize,
    pub downsampled_height: usize,
    pub width_in_blocks: usize,
    pub height_in_blocks: usize,
    pub h_factor: f32,
    pub v_factor: f32,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentInfo{
    pub size: ComponentDimensions,
    pub config: JpegliComponentSettings,

}
impl ComponentInfo{
    pub const EMPTY: Self = ComponentInfo{
        size: ComponentDimensions::EMPTY,
        config: JpegliComponentSettings::EMPTY,
    };
}

pub(crate) struct ComputedConfigDimensions{
    pub image_width: usize,
    pub image_height: usize,
    pub total_i_mcu_rows: usize,
    pub total_i_mcu_cols: usize,
    pub num_components: usize,
    pub components_fixed: [ComponentInfo; MAX_COMPONENTS],
    pub xsize_blocks: usize,
    pub ysize_blocks: usize,
    pub blocks_per_i_mcu_row: usize,
    pub progressive_mode: bool,
}

impl ComputedConfigDimensions{
    pub fn components(&self) -> &[ComponentInfo]{
        &self.components_fixed[..self.num_components]
    }
}

#[derive(Debug, Clone)]
pub struct AppSegment{
    marker: u8,
    data: Vec<u8>,
}
impl AppSegment{
    /// Appends a custom APPn segment to the JFIF file.
    pub fn new(segment_nr: u8, data: Vec<u8>) -> Result<Self, EncodingError> {
        if !(1..=15).contains(&segment_nr) { // APP0 is reserved for JFIF
            Err(EncodingError::InvalidAppSegment(segment_nr))
        } else if data.len() > 65533 {
            Err(EncodingError::AppSegmentTooLarge(data.len()))
        } else {
            Ok(Self { marker: segment_nr, data })
        }
    }
}

/// The user-friendly configuration for encoding. .compute() will validate and convert to the internal config.
#[derive(Debug, Clone)] 
pub struct EncodeOptions{
    /// JPEG quality [1..100]. Mapped to the jpegli butteraugli distance. Default is None
    pub quality: Option<u8>,
    /// Distance. If set, quality is ignored. Default is 1.0.
    pub distance: Option<f32>,
    /// Progressive encoding level: 0=sequential, >0 progressive. Default is 2, the maximum.
    /// Passed to `jpegli_set_progressive_level()`.
    pub progressive_level: Option<u8>,
    /// Smoothing factor 0..100. Used by `ApplyInputSmoothing()`.
    pub smoothing: Option<u8>,

    /// Enable XYB (butteraugli) mode. Uses a custom ICC profile, can add 1020% improvement. 
    pub xyb_mode: Option<bool>,
    /// Use the standard huffman tables, instead of the optimized ones. (default is false)
    pub use_standard_tables: Option<bool>,
    /// Use jpegli adaptive quantization (default is true).
    pub use_adaptive_quantization: Option<bool>,

    /// Default is true.
    pub optimize_coding: Option<bool>,
    /// Chroma subsampling. 420 is the default.
    pub chroma_subsampling: Option<Subsampling>,
    /// JPEG output color space. Default is YCbCr.
    pub jpeg_color_space: Option<JpegColorSpace>,
    /// CICP transfer function.
    pub cicp_transfer_function: Option<SimplifiedTransferCharacteristics>,
    /// Add two chroma tables (default is true).
    pub add_two_chroma_tables: Option<bool>,
    /// Write JFIF APP0 marker to communicate pixels per inch/centimeter information. 
    pub jfif_density: Option<Density>,

    /// Embed the given ICC profile. Not compatible with XYB mode.
    pub icc_profile: Option<Vec<u8>>,

    /// Embed the given APP segments.
    pub app_segments: Option<Vec<AppSegment>>,
    /// Restart interval in rows. Default is 0. Ignored if restart_interval is set.
    pub restart_interval_in_rows: Option<u16>,
}

impl ComputedEncodeConfig { 
    fn validate(&self) -> Result<(), EncodingError>{
        if self.distance < 0.0 {
            return Err(EncodingError::JpegliError("Distance must be non-negative".into()));
        }
        if self.num_components == 0 || self.num_components > MAX_COMPONENTS {
            return Err(EncodingError::JpegliError("Invalid number of components".into()));
        }
        if self.comp_params.len() != self.num_components {
            return Err(EncodingError::JpegliError("Component params length mismatch".into()));
        }
        if self.use_std_tables && self.xyb_mode {
             return Err(EncodingError::JpegliError("Cannot use standard tables with XYB mode".into()));
        }
         if self.add_two_chroma_tables && self.num_components < 3 {
             return Err(EncodingError::JpegliError("Cannot add two chroma tables with less than 3 components".into()));
        }
        if self.add_two_chroma_tables && self.jpeg_color_space == JpegColorSpace::Grayscale {
             return Err(EncodingError::JpegliError("Adding two chroma tables is not supported for grayscale".into()));
        }
        for comp in &self.comp_params {
             if comp.quantization_table_index >= 4 { 
                 return Err(EncodingError::JpegliError("Invalid quantization table index".into()));
             }
        }
        if self.xyb_mode && self.icc_profile.is_some() {
            return Err(EncodingError::JpegliError("XYB mode does not support ICC profile".into()));
        }
        if self.xyb_mode && self.jpeg_color_space != JpegColorSpace::Rgb {
            return Err(EncodingError::JpegliError("XYB mode requires RGB color space".into()));
        }
        if self.xyb_mode && self.subsampling != Subsampling::YCbCr444 {
            return Err(EncodingError::JpegliError("XYB mode requires YCbCr444 chroma subsampling".into()));
        }
        if self.restart_interval as usize > 65535 {
            return Err(EncodingError::JpegliError("Restart interval too big".into()));
        }
        if self.smoothing > 100 {
            return Err(EncodingError::JpegliError("Smoothing factor too big".into()));
        }

        for (ix, comp) in self.comp_params.iter().enumerate() {
            if comp.horizontal_sampling_factor == 0 || comp.vertical_sampling_factor == 0 {
                return Err(EncodingError::JpegliError("Sampling factor must be non-zero".into()));
            }
            if comp.horizontal_sampling_factor > MAX_SAMP_FACTOR || comp.vertical_sampling_factor > MAX_SAMP_FACTOR {
                return Err(EncodingError::JpegliError("Sampling factor too big".into()));
            }
            if comp.index != ix as u8 {
                return Err(EncodingError::JpegliError("Component index mismatch".into()));
            }
            if comp.horizontal_sampling_factor % self.max_h_samp_factor != 0 || comp.vertical_sampling_factor % self.max_v_samp_factor != 0 {
                return Err(EncodingError::JpegliError("Non-integral sampling ratios are not supported".into()));
            }
            if comp.horizontal_sampling_factor > self.max_h_samp_factor || comp.vertical_sampling_factor > self.max_v_samp_factor {
                return Err(EncodingError::JpegliError("Sampling factor larger than max sampling factor".into()));
            }
        }
        // if self.comp_params.unique_by(|a, b| a.component_id == b.component_id).count() != self.num_components {
        //     return Err(EncodingError::JpegliError("Duplicate component id".into()));
        // }
        if self.num_components == 1 {
            if self.comp_params[0].horizontal_sampling_factor != 1 || self.comp_params[0].vertical_sampling_factor != 1 {
                return Err(EncodingError::JpegliError("Single component must have 1x1 sampling factor".into()));
            }
        }
        if self.restart_interval > 0 {
            if self.restart_interval < 10 || self.restart_interval as usize > 65535 {
                return Err(EncodingError::JpegliError("Restart interval must be between 10 and 65535".into()));
            }
        }
        if self.use_adaptive_quantization {
            match self.jpeg_color_space {
                JpegColorSpace::YCbCr => {
                    if self.comp_params[0].horizontal_sampling_factor != self.max_h_samp_factor ||
                    self.comp_params[0].vertical_sampling_factor != self.max_v_samp_factor {
                        return Err(EncodingError::JpegliError("With adaptive quantization, Luma (Y) component must have 1x1 sampling factor".into()));
                    }
                },
                JpegColorSpace::Rgb => {
                    if self.comp_params[1].horizontal_sampling_factor != self.max_h_samp_factor ||
                    self.comp_params[1].vertical_sampling_factor != self.max_v_samp_factor {
                        return Err(EncodingError::JpegliError("With adaptive quantization, Green(G) in RGB must have 1x1 sampling factor".into()));
                    }
                }
                _ => {}
            }
        }
        // progressive_level
        if self.progressive_level > 2 {
            return Err(EncodingError::JpegliError("Progressive level must be 0, 1, or 2".into()));
        }
        if self.progressive_level > 0 && !self.optimize_coding{
            return Err(EncodingError::JpegliError("Optimize coding must be true for progressive encoding".into()));
        }
        Ok(())
    }
}



impl ComputedConfigDimensions{
    pub(crate) fn new(cinfo: &ComputedEncodeConfig, image_width: usize, image_height: usize) -> Result<Self, EncodingError>{
        cinfo.validate()?;
        
        if image_width < 1 || image_height < 1 {
            return Err(EncodingError::JpegliError("Empty input image".into()));
        }
        if image_width > JPEG_MAX_DIMENSION || image_height > JPEG_MAX_DIMENSION{
            return Err(EncodingError::JpegliError("Input image too big".into()));
        }
        let imcu_width = DCTSIZE * cinfo.max_h_samp_factor;
        let imcu_height = DCTSIZE * cinfo.max_v_samp_factor;
        let total_i_mcu_cols = ceil_div(image_width, imcu_width as usize);
        let total_i_mcu_rows = ceil_div(image_height, imcu_height as usize);
        let xsize_blocks = total_i_mcu_cols * cinfo.max_h_samp_factor as usize;
        let ysize_blocks = total_i_mcu_rows * cinfo.max_v_samp_factor as usize;

        let component_sizes = ComponentDimensions::from_component_settings(image_width, image_height, &cinfo.comp_params);
        
        let mut components = [ComponentInfo::EMPTY; MAX_COMPONENTS];
       
        
        let mut blocks_per_i_mcu = 0;
        for (ix, comp) in cinfo.comp_params.iter().enumerate() {
            components[ix].size = component_sizes[ix];
            components[ix].config = *comp;
            
            blocks_per_i_mcu += comp.horizontal_sampling_factor * comp.vertical_sampling_factor;
        }
        let blocks_per_i_mcu_row = total_i_mcu_cols * blocks_per_i_mcu as usize;

        let component_dimension_vec = ComponentDimensions::from_component_settings
            (image_width, image_height, &cinfo.comp_params);



        let components_fixed = [
            ComponentInfo{
                size: component_dimension_vec[0],
                config: cinfo.comp_params[0],
            },
            ComponentInfo{
                size: *component_dimension_vec.get(1).unwrap_or(&ComponentDimensions::EMPTY),
                config: *cinfo.comp_params.get(1).unwrap_or(&JpegliComponentSettings::EMPTY),
            },
            ComponentInfo{
                size: *component_dimension_vec.get(2).unwrap_or(&ComponentDimensions::EMPTY),
                config: *cinfo.comp_params.get(2).unwrap_or(&JpegliComponentSettings::EMPTY),
            },
            ComponentInfo{
                size: *component_dimension_vec.get(3).unwrap_or(&ComponentDimensions::EMPTY),
                config: *cinfo.comp_params.get(3).unwrap_or(&JpegliComponentSettings::EMPTY),
            },
        ];
        
        
        Ok(ComputedConfigDimensions {
            image_width,
            image_height,
            total_i_mcu_rows,
            total_i_mcu_cols,
            num_components: cinfo.num_components,
            components_fixed,
            xsize_blocks,
            ysize_blocks,
            blocks_per_i_mcu_row,
            progressive_mode: cinfo.progressive_level > 0,
        })
    }
}



#[derive(Debug, Clone)] 
pub struct InputImageInfo{
    pub width: usize,
    pub height: usize,
    pub color_space: JpegColorSpace,
}

impl Default for EncodeOptions{
    
    fn default() -> Self{

        Self {
            quality: None,
            distance: None,
            progressive_level: None,
            smoothing: None,
            xyb_mode: None,
            use_standard_tables: None,
            use_adaptive_quantization: None,
            chroma_subsampling: None, // Appears to be ignored - always 4:2:0?
            jpeg_color_space: None,
            cicp_transfer_function: None,
            add_two_chroma_tables: None,
            jfif_density: None,
            icc_profile: None,
            optimize_coding: None,
            restart_interval_in_rows: None,
            app_segments: None,
        }
    }
}
impl EncodeOptions{

    pub fn compute(&self) -> Result<ComputedEncodeConfig, EncodingError>{
        let xyb_mode = self.xyb_mode.unwrap_or(false);
        // 3. Determine Jpegli Output Color Space
        let jpeg_color_space = if xyb_mode {
            JpegColorSpace::Rgb 
        } else {
            self.jpeg_color_space.unwrap_or(JpegColorSpace::YCbCr)
        };
        let num_components = jpeg_color_space.get_num_components();
        if num_components == 0 {
            return Err(EncodingError::JpegliError("Cannot setup quantization for 0 components".into()));
        }
    
       
        let use_std_tables = self.use_standard_tables.unwrap_or(false);
        let use_adaptive_quantization = self.use_adaptive_quantization.unwrap_or(true);

    
        // 1. Determine Distance (handles default and precedence)
        let distance = match (self.quality, self.distance) {
            (Some(q), None) => quality_to_distance(q.clamp(1, 100)), // Clamp quality
            (None, Some(d)) => d,
            (None, None) => 1.0, // Default distance 1.0
            (Some(_), Some(_)) => return Err(EncodingError::JpegliError("Cannot specify both quality and distance".into())),
        }.clamp(0.0, 25.0); // Clamp final distance

        let add_two_chroma_tables = self.add_two_chroma_tables.unwrap_or(self.distance.is_some())
                && jpeg_color_space != JpegColorSpace::Grayscale;

        // 2. Validate and Determine Sampling Factor
         // TODO: fix this to adapt based on distance/quality?
        let subsampling= self.chroma_subsampling.unwrap_or(Subsampling::YCbCr444);
        
        
        let num_components = jpeg_color_space.get_num_components();
        // 4. Generate Initial Component Params
        let (max_h_samp_factor, max_v_samp_factor) = subsampling.to_luma_h_v_samp_factor();
        let mut comp_params: Vec<JpegliComponentSettings> = Vec::with_capacity(num_components);
        let luma_component_index;
        match jpeg_color_space {
            JpegColorSpace::Rgb  => {
                comp_params.push(JpegliComponentSettings::default(0).with_letter('R'));
                comp_params.push(JpegliComponentSettings::default(1).with_letter('G'));
                comp_params.push(JpegliComponentSettings::default(2).with_letter('B'));
                if xyb_mode {
                    comp_params[0] = comp_params[0].with_h_v_sampling(2, 2).with_quant_ix(0);
                    comp_params[1] = comp_params[1].with_h_v_sampling(2, 2).with_quant_ix(1);
                    comp_params[2] = comp_params[2].with_h_v_sampling(1, 1).with_quant_ix(2);
                }
                luma_component_index = 1;
            }
            JpegColorSpace::Grayscale => {
                if num_components != 1 { return Err(EncodingError::JpegliError("Grayscale requires 1 component".into())); }
                comp_params.push(JpegliComponentSettings::default(0));
                luma_component_index = 0;
            }
            JpegColorSpace::Cmyk => {
                // TODO: shouldn't we have different quant tables, at least for K?
                comp_params.push(JpegliComponentSettings::default(0).with_letter('C'));
                comp_params.push(JpegliComponentSettings::default(1).with_letter('M'));
                comp_params.push(JpegliComponentSettings::default(2).with_letter('Y'));
                comp_params.push(JpegliComponentSettings::default(3).with_letter('K'));
                luma_component_index = 3;
            }
            JpegColorSpace::Ycck => {
                // CC are lower res, Y/K are full. 
                comp_params.push(JpegliComponentSettings::default(0).with_h_v_sampling(2, 2));
                comp_params.push(JpegliComponentSettings::default(1).with_quant_ix(1).with_huff_ix(1));
                comp_params.push(JpegliComponentSettings::default(2).with_quant_ix(1).with_huff_ix(1));
                comp_params.push(JpegliComponentSettings::default(3).with_h_v_sampling(2, 2));
                luma_component_index = 0;
            }
            
            JpegColorSpace::YCbCr => { // Handle Ycbcr explicitly
                // Default is 4:2:0, where luma is full res, and chroma is half res both horizontally and vertically.
                let (luma_h, luma_v) = self.chroma_subsampling.unwrap_or(Subsampling::YCbCr420).to_luma_h_v_samp_factor();
                comp_params.push(JpegliComponentSettings::default(0).with_h_v_sampling(luma_h, luma_v));
                comp_params.push(JpegliComponentSettings::default(1).with_quant_ix(1).with_huff_ix(1));
                if add_two_chroma_tables {
                    comp_params.push(JpegliComponentSettings::default(2).with_quant_ix(2).with_huff_ix(1));
                }else{
                    comp_params.push(JpegliComponentSettings::default(2).with_quant_ix(1).with_huff_ix(1));
                }
                luma_component_index = 0;
            }
            // Handle other cases (like Rgb input, treat as YCbCr components)
            _ => panic!("Unsupported output color space - not RGB, CMYK, YCCK, or YCbCr"),
        }

        

        let r = ComputedEncodeConfig {
            distance,
            quality: self.quality,
            xyb_mode,
            use_std_tables,
            use_adaptive_quantization,
            num_components,
            comp_params,
            subsampling,
            optimize_coding: self.optimize_coding.unwrap_or(true),
            smoothing: self.smoothing.unwrap_or(0),
            progressive_level: self.progressive_level.unwrap_or(2),
            write_jfif_density: self.jfif_density.clone(),
            write_adobe_ycck: jpeg_color_space == JpegColorSpace::Ycck,
            jpeg_color_space,
            cicp_transfer_function: self.cicp_transfer_function.unwrap_or(SimplifiedTransferCharacteristics::Default),
            add_two_chroma_tables,
            max_h_samp_factor,
            max_v_samp_factor,
            restart_interval: 0,
            restart_interval_in_rows: self.restart_interval_in_rows.unwrap_or(0),
            luma_component_index,
            icc_profile: self.icc_profile.clone(),
            app_segments: self.app_segments.clone(),
        };
        r.validate()?;
        Ok(r)
    }
}



fn ceil_div(value: usize, div: usize) -> usize {
    value / div + usize::from(value % div != 0)
}

impl ComponentDimensions{ 

    pub const EMPTY: Self = ComponentDimensions { downsampled_width: 0, downsampled_height: 0, width_in_blocks: 0, height_in_blocks: 0, h_factor: 0.0, v_factor: 0.0 };
    fn new(width: usize, height: usize, factor: (u8, u8), max_factor: (u8, u8)) -> ComponentDimensions{
        let downsampled_width = ceil_div(width * factor.0 as usize, max_factor.0 as usize);
        let downsampled_height = ceil_div(height * factor.1 as usize, max_factor.1 as usize);
        let width_in_blocks = ceil_div(downsampled_width, DCTSIZE as usize);
        let height_in_blocks = ceil_div(downsampled_height, DCTSIZE as usize);
        let h_factor = max_factor.0 as f32 / factor.0 as f32;
        let v_factor = max_factor.1 as f32 / factor.1 as f32;
        ComponentDimensions { downsampled_width, downsampled_height, width_in_blocks, height_in_blocks, h_factor, v_factor }
    }

    pub(crate) fn from_component_settings(width: usize, height: usize, components_settings: &[JpegliComponentSettings]) -> Vec<ComponentDimensions>{
        let max_h = components_settings.iter().max_by_key(|s|s.horizontal_sampling_factor).unwrap().horizontal_sampling_factor;
        let max_v = components_settings.iter().max_by_key(|s|s.vertical_sampling_factor).unwrap().vertical_sampling_factor;
        
        let width_usize = width as usize;
        let height_usize = height as usize;

        let mut components = Vec::new();
        for comp_settings in components_settings.iter() {
            let factor = (comp_settings.horizontal_sampling_factor, comp_settings.vertical_sampling_factor);
            let max_factor = (max_h, max_v);
            components.push(ComponentDimensions::new(width_usize, height_usize, factor, max_factor));
        }
        components
    }
}

