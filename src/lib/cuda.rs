use std::ffi::c_void;
use std::ptr;
use std::os::raw::{c_int};
use crate::lib::math::*;

extern "C" {
   pub fn launch_tensor_dot(a: *mut f32, b: *mut f32, c: *mut f32, m: c_int, k: c_int, n: c_int);
   pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
   pub fn cudaFree(ptr: *mut c_void) -> i32;
   pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
   pub fn cudaDeviceSynchronize() -> i32;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;


pub struct Cuda;

impl Cuda {
    pub fn dot2d(tensor: Tensor<f32>, tensor2: Tensor<f32>) -> Tensor<f32>{
        assert_eq!(tensor.dimensions.len(), 2, "tensor must be 2-dimensional");
        assert_eq!(tensor2.dimensions.len(), 2, "tensor2 must be 2-dimensional");
        let m = tensor.dimensions[0];
        let k = tensor.dimensions[1];
        let n = tensor2.dimensions[1];
        assert_eq!(
            tensor.dimensions[1], tensor2.dimensions[0],
            "Inner dimensions must match: got {} (tensor.cols) vs {} (tensor2.rows)",
            tensor.dimensions[1], tensor2.dimensions[0]
        );
        let mut output = Tensor::new(vec![m, n], vec![0.0f32; m * n]);
        
        let mem_size = vec![tensor.data.len() * std::mem::size_of::<f32>(), tensor2.data.len() * std::mem::size_of::<f32>(), output.data.len() * std::mem::size_of::<f32>()];
        
        unsafe{
            let mut ptr_a: *mut c_void = ptr::null_mut();
            let mut ptr_b: *mut c_void = ptr::null_mut();
            let mut ptr_c: *mut c_void = ptr::null_mut();

            cudaMalloc(&mut ptr_a, mem_size[0]);
            cudaMalloc(&mut ptr_b, mem_size[1]);
            cudaMalloc(&mut ptr_c, mem_size[2]);

            cudaMemcpy(ptr_a, tensor.data.as_ptr() as *const _, mem_size[0], CUDA_MEMCPY_HOST_TO_DEVICE);
            cudaMemcpy(ptr_b, tensor2.data.as_ptr() as *const _, mem_size[1], CUDA_MEMCPY_HOST_TO_DEVICE);

            launch_tensor_dot(ptr_a as *mut f32, ptr_b as *mut f32, ptr_c as *mut f32, m as c_int, k as c_int, n as c_int);
            cudaDeviceSynchronize();

            cudaMemcpy(output.data.as_mut_ptr() as *mut _, ptr_c, mem_size[2], CUDA_MEMCPY_DEVICE_TO_HOST);
            
            cudaFree(ptr_a);
            cudaFree(ptr_b);
            cudaFree(ptr_c);

        }

        output
        


        
    }

}

