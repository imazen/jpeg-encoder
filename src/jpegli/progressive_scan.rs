use super::config::{ ComponentInfo, ComputedEncodeConfig};

/// Constants
const DCTSIZE: usize = 8;
const DCTSIZE2: usize = 64;
const MAX_COMPONENTS: usize = 4;
const MAX_REFINEMENT_BIT: i32 = 10;

pub const MAX_COMPS_IN_SCAN: usize = 4;
pub const C_MAX_BLOCKS_IN_MCU: usize = 10;

/// Refinement token for progressive AC scans
#[derive(Debug, Clone)]
pub struct RefToken {
    pub symbol: u8,
    pub refbits: u8,
}

impl RefToken {
    pub fn new(symbol: u8, refbits: u8) -> Self {
        Self { symbol, refbits }
    }
}

/// Scan token information for encoding
#[derive(Debug, Clone)]
pub struct ScanTokenInfo {
    pub tokens: Vec<RefToken>,
    pub num_tokens: usize,
    pub refbits: Vec<u8>,
    pub eobruns: Vec<u16>,
    pub restarts: Vec<usize>,
    pub num_restarts: usize,
    pub num_nonzeros: usize,
    pub num_future_nonzeros: usize,
    pub token_offset: usize,
    pub restart_interval: usize,
    pub mcus_per_row: usize,
    pub mcu_rows_in_scan: usize,
    pub blocks_in_mcu: usize,
    pub num_blocks: usize,
}

impl ScanTokenInfo {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            num_tokens: 0,
            refbits: Vec::new(),
            eobruns: Vec::new(),
            restarts: Vec::new(),
            num_restarts: 0,
            num_nonzeros: 0,
            num_future_nonzeros: 0,
            token_offset: 0,
            restart_interval: 0,
            mcus_per_row: 0,
            mcu_rows_in_scan: 0,
            blocks_in_mcu: 0,
            num_blocks: 0,
        }
    }
}

/// Computed scan configuration after processing compression parameters
#[derive(Debug, Clone)]
pub struct ScanConfiguration {
    pub scan_script: Vec<JpegScanInfo>,
    pub scan_token_info: Vec<ScanTokenInfo>,
    pub ac_ctx_offset: Vec<u8>,
    pub num_contexts: usize,
    pub progressive_mode: bool,
}

impl ScanConfiguration {
    pub fn new() -> Self {
        Self {
            scan_script: Vec::new(),
            scan_token_info: Vec::new(),
            ac_ctx_offset: Vec::new(),
            num_contexts: 0,
            progressive_mode: false,
        }
    }
}

/// Progressive scan configuration
#[derive(Debug, Clone, Copy)]
pub struct ProgressiveScan {
    pub ss: i32,  // Spectral start
    pub se: i32,  // Spectral end  
    pub ah: i32,  // Successive approximation high
    pub al: i32,  // Successive approximation low
    pub interleaved: bool,
}

impl ProgressiveScan {
    pub fn new(ss: i32, se: i32, ah: i32, al: i32, interleaved: bool) -> Self {
        Self { ss, se, ah, al, interleaved }
    }
}

/// JPEG scan information
/// TODO: Can we use u8 instead of usize, etc?
#[derive(Debug, Clone)]
pub struct JpegScanInfo {
    pub comps_in_scan: usize,
    pub component_index: [usize; MAX_COMPS_IN_SCAN],
    pub ss: i32,
    pub se: i32,
    pub ah: i32,
    pub al: i32,
    pub interleaved: bool,
}

impl JpegScanInfo {
    pub fn new() -> Self {
        Self {
            comps_in_scan: 0,
            component_index: [0; MAX_COMPS_IN_SCAN],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            interleaved: false,
        }
    }
}

/// Generate default scan script based on progressive level
pub fn set_default_scan_script(config: &ComputedEncodeConfig) -> Result<Vec<JpegScanInfo>, String> {
    let level = config.progressive_level;
    let mut progressive_mode = Vec::new();
    
    let interleave_dc = config.max_h_samp_factor == 1 && config.max_v_samp_factor == 1;
    
    if level == 0 {
        progressive_mode.push(ProgressiveScan::new(0, 63, 0, 0, true));
    } else if level == 1 {
        progressive_mode.push(ProgressiveScan::new(0, 0, 0, 0, interleave_dc));
        progressive_mode.push(ProgressiveScan::new(1, 63, 0, 1, false));
        progressive_mode.push(ProgressiveScan::new(1, 63, 1, 0, false));
    } else {
        progressive_mode.push(ProgressiveScan::new(0, 0, 0, 0, interleave_dc));
        progressive_mode.push(ProgressiveScan::new(1, 2, 0, 0, false));
        progressive_mode.push(ProgressiveScan::new(3, 63, 0, 2, false));
        progressive_mode.push(ProgressiveScan::new(3, 63, 2, 1, false));
        progressive_mode.push(ProgressiveScan::new(3, 63, 1, 0, false));
    }

    // Calculate total script size needed
    let mut script_space_size = 0;
    for scan in &progressive_mode {
        let comps = if scan.interleaved { MAX_COMPS_IN_SCAN } else { 1 };
        script_space_size += div_ceil(config.num_components, comps);
    }

    // Generate scan script
    let mut scan_script = Vec::with_capacity(script_space_size);
    
    for scan in &progressive_mode {
        let comps = if scan.interleaved { MAX_COMPS_IN_SCAN } else { 1 };
        let mut c = 0;
        while c < config.num_components {
            let mut scan_info = JpegScanInfo::new();
            scan_info.ss = scan.ss;
            scan_info.se = scan.se;
            scan_info.ah = scan.ah;
            scan_info.al = scan.al;
            scan_info.comps_in_scan = std::cmp::min(comps, config.num_components - c);
            
            for j in 0..scan_info.comps_in_scan {
                scan_info.component_index[j] = c + j;
            }
            
            scan_info.interleaved = scan.interleaved;
            
            scan_script.push(scan_info);
            c += comps;
        }
    }

    Ok(scan_script)
}

/// Validate scan script follows JPEG progressive rules
pub fn validate_scan_script(
    scan_script: &[JpegScanInfo], 
    config: &ComputedEncodeConfig,
    progressive_mode: bool
) -> Result<(), String> {
    // Mask of coefficient bits defined by the scan script, for each component and coefficient index
    let mut comp_mask = [[0u16; DCTSIZE2]; MAX_COMPONENTS];

    for (i, scan_info) in scan_script.iter().enumerate() {
        // Validate number of components in scan
        if scan_info.comps_in_scan < 1 || scan_info.comps_in_scan > MAX_COMPS_IN_SCAN {
            return Err(format!("Invalid number of components in scan {}: {}", 
                             i, scan_info.comps_in_scan));
        }

        // Validate component indices are in order and within bounds
        let mut last_ci = -1i32;
        for j in 0..scan_info.comps_in_scan {
            let ci = scan_info.component_index[j] as i32;
            if ci < 0 || ci >= config.num_components as i32 {
                return Err(format!("Invalid component index {} in scan {}", ci, i));
            } else if ci == last_ci {
                return Err(format!("Duplicate component index {} in scan {}", ci, i));
            } else if ci < last_ci {
                return Err(format!("Out of order component index {} in scan {}", ci, i));
            }
            last_ci = ci;
        }

        // Validate spectral range
        if scan_info.ss < 0 || scan_info.se < scan_info.ss || scan_info.se >= DCTSIZE2 as i32 {
            return Err(format!("Invalid spectral range {} .. {} in scan {}", 
                             scan_info.ss, scan_info.se, i));
        }

        // Validate refinement bits
        if scan_info.ah < 0 || scan_info.al < 0 || scan_info.al > MAX_REFINEMENT_BIT {
            return Err(format!("Invalid refinement bits {}/{} in scan {}", 
                             scan_info.ah, scan_info.al, i));
        }

        // Validate progressive vs sequential mode constraints
        if !progressive_mode {
            if scan_info.ss != 0 || scan_info.se != (DCTSIZE2 as i32 - 1) || 
               scan_info.ah != 0 || scan_info.al != 0 {
                return Err(format!("Invalid scan for sequential mode at scan {}", i));
            }
        } else {
            if scan_info.ss == 0 && scan_info.se != 0 {
                return Err(format!("DC and AC together in progressive scan {}", i));
            }
        }

        // AC scans must be non-interleaved
        if scan_info.ss != 0 && scan_info.comps_in_scan != 1 {
            return Err(format!("Interleaved AC only scan at scan {}", i));
        }

        // Validate progressive bit progression
        for j in 0..scan_info.comps_in_scan {
            let ci = scan_info.component_index[j];
            
            // AC before DC check
            if scan_info.ss != 0 && comp_mask[ci][0] == 0 {
                return Err(format!("AC before DC in component {} of scan {}", ci, i));
            }

            // Update coefficient masks
            for k in scan_info.ss as usize..=scan_info.se as usize {
                if comp_mask[ci][k] == 0 {
                    if scan_info.ah != 0 {
                        return Err(format!("Invalid first scan refinement bit in scan {}", i));
                    }
                    comp_mask[ci][k] = (0xffffu16 << scan_info.al) & 0xffffu16;
                } else {
                    if comp_mask[ci][k] != ((0xffffu16 << scan_info.ah) & 0xffffu16) ||
                       scan_info.al != scan_info.ah - 1 {
                        return Err(format!("Invalid refinement bit progression in scan {}", i));
                    }
                    comp_mask[ci][k] |= 1u16 << scan_info.al;
                }
            }
        }

        // Validate MCU size for interleaved scans
        if scan_info.comps_in_scan > 1 {
            let mut mcu_size = 0;
            for j in 0..scan_info.comps_in_scan {
                let ci = scan_info.component_index[j];
                let comp = &config.comp_params[ci];
                mcu_size += comp.horizontal_sampling_factor as usize * comp.vertical_sampling_factor as usize;
            }
            if mcu_size > C_MAX_BLOCKS_IN_MCU {
                return Err(format!("MCU size too big in scan {}: {}", i, mcu_size));
            }
        }
    }

    // Verify all coefficients are covered for all components
    for c in 0..config.num_components {
        for k in 0..DCTSIZE2 {
            if comp_mask[c][k] != 0xffffu16 {
                return Err(format!("Incomplete scan of component {} and frequency {}", c, k));
            }
        }
    }

    Ok(())
}

/// Helper function for ceiling division
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Check if scan script represents progressive mode
pub fn is_progressive_mode(scan_script: &[JpegScanInfo]) -> bool {
    if scan_script.is_empty() {
        return false;
    }
    let first_scan = &scan_script[0];
    first_scan.ss != 0 || first_scan.se != (DCTSIZE2 as i32 - 1)
}

/// Process compression parameters to set up scan configuration
/// This corresponds to the scan processing part of ProcessCompressionParams in the C++ code
pub fn create_scan_configuration(
    config: &ComputedEncodeConfig,
    image_width: usize,
    image_height: usize,
    components: &[ComponentInfo],
) -> Result<ScanConfiguration, String> {
    let mut scan_config = ScanConfiguration::new();
    
    // If no scan info exists, generate default scan script
    if scan_config.scan_script.is_empty() {
        scan_config.scan_script = set_default_scan_script(config)?;
    }
    
    // Determine progressive mode
    scan_config.progressive_mode = is_progressive_mode(&scan_config.scan_script);
    
    // Validate scan script
    validate_scan_script(&scan_config.scan_script, config, scan_config.progressive_mode)?;
    
    // Set up scan token info
    setup_scan_token_info(&mut scan_config, config, image_width, image_height, components)?;
    
    Ok(scan_config)
}

impl ScanConfiguration {
    pub fn create(config: &ComputedEncodeConfig, image_width: usize, image_height: usize, components: &[ComponentInfo]) -> Result<Self, String> {
        create_scan_configuration(config, image_width, image_height, components)
    }
}
/// Set up scan token information for each scan
fn setup_scan_token_info(
    scan_config: &mut ScanConfiguration,
    config: &ComputedEncodeConfig,
    image_width: usize,
    image_height: usize,
    components: &[ComponentInfo],
) -> Result<(), String> {
    let num_scans = scan_config.scan_script.len();
    
    // Initialize scan token info array
    scan_config.scan_token_info = vec![ScanTokenInfo::new(); num_scans];
    
    // Initialize AC context offset array
    scan_config.ac_ctx_offset = vec![0u8; num_scans];
    
    let mut num_ac_contexts = 0u8;
    
    for (i, scan_info) in scan_config.scan_script.iter().enumerate() {
        scan_config.ac_ctx_offset[i] = 4 + num_ac_contexts;
        
        if scan_info.se > 0 {
            num_ac_contexts += scan_info.comps_in_scan as u8;
        }
        
        if num_ac_contexts > 252 {
            return Err("Too many AC scans in image".to_string());
        }
        
        let sti = &mut scan_config.scan_token_info[i];
        
        if scan_info.comps_in_scan == 1 {
            // Single component scan
            let comp_idx = scan_info.component_index[0];
            if comp_idx >= components.len() {
                return Err(format!("Invalid component index {}", comp_idx));
            }
            let comp = &components[comp_idx].size;
            sti.mcus_per_row = comp.width_in_blocks;
            sti.mcu_rows_in_scan = comp.height_in_blocks;
            sti.blocks_in_mcu = 1;
        } else {
            // Multi-component scan
            sti.mcus_per_row = div_ceil(image_width, DCTSIZE * config.max_h_samp_factor as usize);
            sti.mcu_rows_in_scan = div_ceil(image_height, DCTSIZE * config.max_v_samp_factor as usize);
            sti.blocks_in_mcu = 0;
            
            for j in 0..scan_info.comps_in_scan {
                let comp_idx = scan_info.component_index[j];
                if comp_idx >= components.len() {
                    return Err(format!("Invalid component index {}", comp_idx));
                }
                let comp = &components[comp_idx].config;
                sti.blocks_in_mcu += comp.horizontal_sampling_factor as usize * comp.vertical_sampling_factor as usize;
            }
        }
        
        let num_mcus = sti.mcu_rows_in_scan * sti.mcus_per_row;
        sti.num_blocks = num_mcus * sti.blocks_in_mcu;
        
        if config.restart_interval_in_rows <= 0 {
            sti.restart_interval = config.restart_interval as usize;
        } else {
            sti.restart_interval = std::cmp::min(
                sti.mcus_per_row * config.restart_interval_in_rows as usize,
                65535
            );
        }
        
        sti.num_restarts = if sti.restart_interval > 0 {
            div_ceil(num_mcus, sti.restart_interval)
        } else {
            1
        };
        
        // Allocate restarts array
        sti.restarts = vec![0; sti.num_restarts];
    }
    
    scan_config.num_contexts = 4 + num_ac_contexts as usize;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::jpegli::{config::{ComponentDimensions, EncodeOptions}, structs::JpegliComponentSettings};

    use super::*;
    
    
    fn create_test_config(progressive_level: u8) -> ComputedEncodeConfig {
        EncodeOptions{
            progressive_level: Some(progressive_level),
            ..Default::default()
        }.compute().unwrap()
    }
    
    fn create_test_components() -> Vec<ComponentInfo> {
        let width = 320;
        let height = 240;

        let configs = vec![
            JpegliComponentSettings::default(0).with_h_v_sampling(2, 2),
            JpegliComponentSettings::default(1).with_h_v_sampling(1, 1).with_quant_ix(1).with_huff_ix(1),
            JpegliComponentSettings::default(2).with_h_v_sampling(1, 1).with_quant_ix(1).with_huff_ix(1),
        ];

        ComponentDimensions::from_component_settings(width, height, &configs)
        .into_iter()
        .zip(configs)
        .map(|(size, config)| ComponentInfo { size, config })
        .collect()
    }

    #[test]
    fn test_set_default_scan_script_level_0() {
        let mut config = create_test_config(0);
        config.progressive_level = 0;
        
        let result = set_default_scan_script(&config);
        assert!(result.is_ok());
        
        let scans = result.unwrap();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 63);
    }

    #[test]
    fn test_set_default_scan_script_level_1() {
        let mut config = create_test_config(1);
        config.progressive_level = 1;
        
        let result = set_default_scan_script(&config);
        assert!(result.is_ok());
        
        let scans = result.unwrap();
        assert!(scans.len() > 1);
    }

    #[test]
    fn test_validate_scan_script_valid() {
        let config = create_test_config(1);
        let scans = set_default_scan_script(&config).unwrap();
        let progressive = is_progressive_mode(&scans);
        
        let result = validate_scan_script(&scans, &config, progressive);
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_progressive_mode() {
        let config = create_test_config(1);
        
        // Test level 0 (sequential)
        let mut config_seq = config.clone();
        config_seq.progressive_level = 0;
        let scans_seq = set_default_scan_script(&config_seq).unwrap();
        assert!(!is_progressive_mode(&scans_seq));
        
        // Test level 2 (progressive)
        let scans_prog = set_default_scan_script(&config).unwrap();
        assert!(is_progressive_mode(&scans_prog));
    }

    #[test]
    fn test_progressive_mode_detection() {
        // Sequential mode
        let sequential_scan = vec![JpegScanInfo {
            component_index: [0, 1, 2, 0],
            comps_in_scan: 3,
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
            interleaved: true,
        }];
        assert_eq!(is_progressive_mode(&sequential_scan), false);
        
        // Progressive mode (SS != 0)
        let progressive_scan = vec![JpegScanInfo {
            component_index: [0, 0, 0, 0],
            comps_in_scan: 1,
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
            interleaved: false,
        }];
        assert!(is_progressive_mode(&progressive_scan));
        
        // Progressive mode (SE != 63)
        let progressive_scan2 = vec![JpegScanInfo {
            component_index: [0, 0, 0, 0],
            comps_in_scan: 1,
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            interleaved: false,
        }];
        assert!(is_progressive_mode(&progressive_scan2));
    }

    #[test]
    fn test_scan_token_info_creation() {
        let sti = ScanTokenInfo::new();
        assert_eq!(sti.num_tokens, 0);
        assert_eq!(sti.num_restarts, 0);
        assert_eq!(sti.restart_interval, 0);
        assert_eq!(sti.mcus_per_row, 0);
        assert_eq!(sti.mcu_rows_in_scan, 0);
        assert_eq!(sti.blocks_in_mcu, 0);
        assert_eq!(sti.num_blocks, 0);
    }

    #[test]
    fn test_scan_configuration_creation() {
        let config = ScanConfiguration::new();
        assert!(config.scan_script.is_empty());
        assert!(config.scan_token_info.is_empty());
        assert!(config.ac_ctx_offset.is_empty());
        assert_eq!(config.num_contexts, 0);
        assert_eq!(config.progressive_mode, false);
    }

    #[test]
    fn test_process_compression_params_sequential() {
        let config = create_test_config(0);
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            components.as_slice(),
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        assert_eq!(scan_config.progressive_mode, false);
        assert_eq!(scan_config.scan_script.len(), 1);
        
        // Check the sequential scan
        let scan = &scan_config.scan_script[0];
        assert_eq!(scan.comps_in_scan, 3);
        assert_eq!(scan.ss, 0);
        assert_eq!(scan.se, 63);
        assert_eq!(scan.ah, 0);
        assert_eq!(scan.al, 0);
        assert_eq!(scan.interleaved, true);
    }

    #[test]
    fn test_process_compression_params_progressive() {
        let config = create_test_config(1);
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        assert_eq!(scan_config.progressive_mode, true);
        assert!(scan_config.scan_script.len() > 1);
        
        // Check that we have DC and AC scans
        let has_dc_scan = scan_config.scan_script.iter().any(|s| s.ss == 0 && s.se == 0);
        let has_ac_scan = scan_config.scan_script.iter().any(|s| s.ss > 0);
        assert!(has_dc_scan);
        assert!(has_ac_scan);
    }

    #[test]
    fn test_scan_token_info_single_component() {
        let config = create_test_config(1);
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        
        // Find a single-component scan
        for (i, scan) in scan_config.scan_script.iter().enumerate() {
            if scan.comps_in_scan == 1 {
                let sti = &scan_config.scan_token_info[i];
                assert_eq!(sti.blocks_in_mcu, 1);
                assert_eq!(sti.mcus_per_row, components[scan.component_index[0]].size.width_in_blocks);
                assert_eq!(sti.mcu_rows_in_scan, components[scan.component_index[0]].size.height_in_blocks);
                assert_eq!(sti.num_blocks, sti.mcus_per_row * sti.mcu_rows_in_scan);
                break;
            }
        }
    }

    #[test]
    fn test_scan_token_info_multi_component() {
        let config = create_test_config(0); // Sequential has multi-component scan
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        
        // Check the multi-component scan
        let scan = &scan_config.scan_script[0];
        let sti = &scan_config.scan_token_info[0];
        
        assert_eq!(scan.comps_in_scan, 3);
        assert_eq!(sti.mcus_per_row, div_ceil(320, DCTSIZE * 2)); // max_h_samp_factor = 2
        assert_eq!(sti.mcu_rows_in_scan, div_ceil(240, DCTSIZE * 2)); // max_v_samp_factor = 2
        
        // blocks_in_mcu should be sum of h_samp * v_samp for all components
        let expected_blocks = 2*2 + 1*1 + 1*1; // Y + Cb + Cr
        assert_eq!(sti.blocks_in_mcu, expected_blocks);
    }

    #[test]
    fn test_ac_context_offset() {
        let config = create_test_config(1);
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        
        // Check AC context offsets
        let mut expected_offset = 4u8;
        for (i, scan) in scan_config.scan_script.iter().enumerate() {
            assert_eq!(scan_config.ac_ctx_offset[i], expected_offset);
            if scan.se > 0 {
                expected_offset += scan.comps_in_scan as u8;
            }
        }
        
        assert_eq!(scan_config.num_contexts, expected_offset as usize);
    }

    #[test]
    fn test_restart_interval_calculation() {
        let mut config = create_test_config(0);
        config.restart_interval = 100;
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        let sti = &scan_config.scan_token_info[0];
        
        assert_eq!(sti.restart_interval, 100);
        assert_eq!(sti.num_restarts, div_ceil(sti.mcus_per_row * sti.mcu_rows_in_scan, 100));
    }

    #[test]
    fn test_restart_interval_in_rows() {
        let mut config = create_test_config(0);
        config.restart_interval_in_rows = 5;
        let components = create_test_components();
        
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        
        assert!(result.is_ok());
        let scan_config = result.unwrap();
        let sti = &scan_config.scan_token_info[0];
        
        let expected_restart_interval = std::cmp::min(sti.mcus_per_row * 5, 65535);
        assert_eq!(sti.restart_interval, expected_restart_interval);
    }

    #[test]
    fn test_too_many_ac_contexts() {
        // This would require a very complex scan script to trigger,
        // but we can test the boundary condition
        let config = create_test_config(1);
        let components = create_test_components();
        
        // Normal case should work
        let result = ScanConfiguration::create(
            &config,
            320,
            240,
            &components,
        );
        assert!(result.is_ok());
        
        // The actual limit checking is done in the setup function
        // and would require a pathological scan script to trigger
    }
} 