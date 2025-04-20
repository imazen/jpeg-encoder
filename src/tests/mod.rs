// This file links the test modules.

// Include the generated reference data module
#[cfg(test)]
mod reference_data;

// Include the actual tests that use the reference data
#[cfg(test)]
mod reference_tests; 

#[cfg(test)]
mod general; 