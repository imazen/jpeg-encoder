  // ... existing code ...
    if class == ProfileClassSignature::Display || class == ProfileClassSignature::Input || class == ProfileClassSignature::Output || class == ProfileClassSignature::ColorSpace {
        // Use profile.tag()
        if let Some(wp_tag) = profile.tag(TagSignature::MediaWhitePointTag) {
            // Use tag_instance.read_xyz()
            if let Ok(xyz) = wp_tag.read_xyz() {
                 // Use CIExyY::from() and match Ok(xyy)
                 if let Ok(xyy) = CIExyY::from(&xyz) {
                     encoding.white_point = Some(xyy);
                 } else {
                     log::warn!("ICC WP XYZ to xyY conversion failed, using D50 (PCS default)");
                 }
            } else {
                // Handle error
            }
        }
    }
    if let (Some(r_data), Some(g_data), Some(b_data)) = (r_tag, g_tag, b_tag) {
         // Use tag_instance.read_xyz()
         if let (Ok(r_xyz), Ok(g_xyz), Ok(b_xyz)) = (r_data.read_xyz(), g_data.read_xyz(), b_data.read_xyz()) {
            // Use CIExyY::from() and match Ok(xyy)
            if let (Ok(r_xyy), Ok(g_xyy), Ok(b_xyy)) = (CIExyY::from(&r_xyz), CIExyY::from(&g_xyz), CIExyY::from(&b_xyz)) {
                 encoding.primaries = Some(CIExyYTRIPLE {
                     Red: r_xyy,
                     Green: g_xyy,
                     Blue: b_xyy,
                 });
            } else {
                log::warn!("Failed to convert primary XYZ to xyY");
            }
         } else {
             // Handle error
         }
    }
    // ... existing code ...