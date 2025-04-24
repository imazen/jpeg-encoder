# Rules for porting the C++ jpegli encoder algorithms to the Rust jpeg-encoder crate (workspace root)

Always make sure you have loaded @inventory.md and @porting.md into your context. Don't load or manually edit src/jpegli/reference_test_data.rs, instead, edit and rerun @extract_dqt.py and @dct.sh to update it.

# Tool use tips

* rg "pq|transfer" --type cpp > relevant_cpp_pq.txt and reading relevant_cpp_pq.txt is the best way to search for something in the C++ codebase.  
* cargo test SEARCH_STRING, where SEARCH_STRING is the unqualified name of the module, NOT a file path.
* If an edit doesn't apply, reload files from disk. 
* Keep changes inside src/jpegli/
* Don't use ancestor module or path names with cargo test, just call 'cargo test' or 'cargo test

# Rules

Be ABSOLUTELY MINIMAL in making changes outside of the src/jpegli/ directory, we want to preserve the jpeg-encoder API and make as few changes as needed to support the jpegli algorithm. Do NOT make tests pass by skipping or commenting them out. Be CORRECT. NEVER INFER A FILE's CONTENTS, READ IT.

We are making our own separate jpegli/jpegli_encoder.rs interface, so that we do not touch encoder.rs in the root.

And if changes fail to apply, reload files from disk since they must have been applied early.

Work to make code idiomatic in a way that the rust compiler can optimize.
chunks_exact_mut, eliminating branching inside loops, etc. 


1. The jpegli encoder source is in @jpegli/lib/jpegli jpegli/lib/jpegli
2. External dependencies like `external/rust-lcms2` (@lcms.md), `external/image-png`, `external/image`, and `external/moxcms` (@moxcms_apis.txt) provide functionality for color management and image loading.
3. We target stable Rust, and keep any unsafe code (like SIMD abstractions) simple; study jpeg-encoder and follow those patterns. Auto-vectorization is the goal.
4. Before porting a C++ component, we examine all the headers it references and build a list of all the functions it actually depends on, and add that info as comments in the C++ header.
5. We work methodically, search for a replacement in jpeg-encoder or create one, and try to create idomatic but performant and correct solutions.
6. We add new rules when we glean insight about jpegli, its structure, organization
7. We add rules whenever we establish a mapping from a C++ component to a rust component, including function signatures.
8. We always port tests and run them regularly.
9. Create new functions, and don't delete existing quanitzation tables or methods. We want to be able to compare and benchmark new and old, side by side.
10. We put testdata images in reference_dct/testdata/, see referenced_dct/dct.sh for ones we found most useful.AS


## The assistant should update this file when it learns new things or is given guidance. 


