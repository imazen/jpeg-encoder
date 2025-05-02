# Get all Rust files recursively from the src directory
$rustFiles = Get-ChildItem -Path src -Recurse -Filter '*.rs'

# Define the output file path
$outputFile = "concatenated_rust.rs"

# Clear the output file if it already exists
Clear-Content -Path $outputFile -ErrorAction SilentlyContinue

# Loop through each Rust file and append its content to the output file
foreach ($file in $rustFiles) {
    Write-Host "Adding $($file.FullName)..."
    # Add a comment indicating the start of the file content
    Add-Content -Path $outputFile -Value "`n// --- Start of file: $($file.FullName) ---`n"
    # Get the content and append it
    Get-Content -Path $file.FullName -Raw | Add-Content -Path $outputFile
    # Add a comment indicating the end of the file content
    Add-Content -Path $outputFile -Value "`n// --- End of file: $($file.FullName) ---`n"
}

Write-Host "Concatenation complete. Output saved to $outputFile" 