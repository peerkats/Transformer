


fn main() {
    // Add CUDA library search paths
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
    
    // Link CUDA runtime library
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    
    // Link your custom library (without .so extension)
    println!("cargo:rustc-link-search=native=.");
    // println!("cargo:rustc-link-lib=vector_add");
    // println!("cargo:rustc-link-lib=tensor_dot");
    println!("cargo:rustc-link-lib=tensordot");
    
    // println!("cargo:rerun-if-changed=vector_add.cu");
}