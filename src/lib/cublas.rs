use std::ffi::c_void;
use std::ptr;
use std::os::raw::{c_int};
use crate::lib::math::*;


extern "C" {
    fn launch_cublas_dot(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;

}

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;


pub struct Cublas;

impl Cublas{
    pub fn cublas_dot(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>{
        let m = a.dimensions[0];
        let k = a.dimensions[1];
        let n = b.dimensions[1];

        let mut output = Tensor::new(vec![m, n], vec!(0.0f32; m * n));

        let size_a = a.data.len() * std::mem::size_of::<f32>();
        let size_b = b.data.len() * std::mem::size_of::<f32>();
        let size_c = output.data.len() * std::mem::size_of::<f32>();

        unsafe {
            let mut ptr_a: *mut c_void = ptr::null_mut();
            let mut ptr_b: *mut c_void = ptr::null_mut();
            let mut ptr_c: *mut c_void = ptr::null_mut();

            cudaMalloc(&mut ptr_a, size_a);
            cudaMalloc(&mut ptr_b, size_b);
            cudaMalloc(&mut ptr_c, size_c);

            cudaMemcpy(ptr_a, a.data.as_ptr() as *const _, size_a, CUDA_MEMCPY_HOST_TO_DEVICE);
            cudaMemcpy(ptr_b, b.data.as_ptr() as *const _, size_b, CUDA_MEMCPY_HOST_TO_DEVICE);

            launch_cublas_dot(ptr_a as *mut f32, ptr_b as *mut f32, ptr_c as *mut f32, m as i32, k as i32, n as i32);

            cudaMemcpy(output.data.as_mut_ptr() as *mut _, ptr_c, size_c, CUDA_MEMCPY_DEVICE_TO_HOST);

            cudaFree(ptr_a);
            cudaFree(ptr_b);
            cudaFree(ptr_c);
        }

        output
    } 

}



