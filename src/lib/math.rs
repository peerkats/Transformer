use rand::*;
use rayon::prelude::*;
use std::ops::Add;
use std::ops::Mul;


#[derive(Debug)]
pub struct Tensor<T> {
    pub dimensions: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Clone + Send + Sync> Tensor<T> {
    pub fn new(dimensions: Vec<usize>, data: Vec<T>) -> Tensor<T> {
        let total_len: usize = dimensions.iter().product();
        if data.len() != total_len {
            panic!(
                "data length ({}) must equal product of dimensions ({})",
                data.len(),
                total_len
            );
        }
        Tensor { dimensions, data }
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.dimensions.len());
        let mut index = 0;
        let mut stride = 1;
        for (&i, &dim) in indices.iter().rev().zip(self.dimensions.iter().rev()) {
            assert!(i < dim, "Index out of bounds");
            index += i * stride;
            stride *= dim;
        }
        index
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.flat_index(indices);
        &self.data[idx]
    }

    pub fn get_row(&self, row: usize) -> Vec<&T> {
        assert!(row < self.dimensions[0], "Row index out of bounds");
        let mut result = Vec::new();
        for i in 0..self.dimensions[1] {
            result.push(self.get(&[row, i]));
        }
        result
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.flat_index(indices);
        &mut self.data[idx]
    }
    
    pub fn add_tensor(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Add<Output = T> + Clone,
    {
        assert_eq!(self.dimensions, other.dimensions);
        let data: Vec<T> = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Tensor {
            dimensions: self.dimensions.clone(),
            data,
        }
    }

    pub fn dot_tensor(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + std::iter::Sum + Clone + Send + Sync + Default,
    {
        let self_dims = &self.dimensions;
        let other_dims = &other.dimensions;

        if self_dims.is_empty() || other_dims.is_empty() {
            panic!("Dot product is not supported for 0-dimensional tensors.");
        }

        let k = *self_dims.last().unwrap();
        let other_k_dim_index = if other_dims.len() > 1 { other_dims.len() - 2 } else { 0 };
        let other_k = other_dims[other_k_dim_index];

        assert_eq!(
            k, other_k,
            "The last dimension of the first tensor must match the second-to-last dimension of the second tensor."
        );

        let mut result_dims = Vec::new();
        result_dims.extend_from_slice(&self_dims[..self_dims.len() - 1]);
        if other_dims.len() > 1 {
            result_dims.extend_from_slice(&other_dims[..other_k_dim_index]);
            result_dims.push(*other_dims.last().unwrap());
        }

        if result_dims.is_empty() {
            // This handles the vector-vector dot product case, resulting in a scalar.
            let sum = self.data.par_iter().zip(other.data.par_iter()).map(|(a, b)| a.clone() * b.clone()).sum();
            return Tensor {
                dimensions: vec![1],
                data: vec![sum],
            };
        }

        let result_size: usize = result_dims.iter().product();
        let mut result_data = vec![T::default(); result_size];

        let self_prefix_dims = &self_dims[..self_dims.len() - 1];
        let self_prefix_size: usize = self_prefix_dims.iter().product();

        let other_n = if other_dims.len() > 1 { *other_dims.last().unwrap() } else { 1 };

        result_data.par_iter_mut().enumerate().for_each(|(i, val)| {
            let mut current_indices = Vec::with_capacity(result_dims.len());
            let mut temp_i = i;
            for &dim in result_dims.iter().rev() {
                current_indices.insert(0, temp_i % dim);
                temp_i /= dim;
            }

            let self_prefix_indices = &current_indices[..self_prefix_dims.len()];
            let mut self_prefix_flat_index = 0;
            let mut stride = 1;
            for (&idx, &dim) in self_prefix_indices.iter().rev().zip(self_prefix_dims.iter().rev()) {
                self_prefix_flat_index += idx * stride;
                stride *= dim;
            }

            let other_suffix_indices = &current_indices[self_prefix_dims.len()..];
            let mut other_prefix_flat_index = 0;
            stride = 1;
            if other_dims.len() > 2 {
                let other_prefix_dims = &other_dims[..other_dims.len() - 2];
                for (&idx, &dim) in other_suffix_indices[..other_suffix_indices.len()-1].iter().rev().zip(other_prefix_dims.iter().rev()) {
                    other_prefix_flat_index += idx * stride;
                    stride *= dim;
                }
            }
            let other_col = if !other_suffix_indices.is_empty() { *other_suffix_indices.last().unwrap() } else { 0 };


            let sum = (0..k)
                .map(|j| {
                    let self_idx = self_prefix_flat_index * k + j;
                    let other_idx = (other_prefix_flat_index * other_k + j) * other_n + other_col;
                    self.data[self_idx].clone() * other.data[other_idx].clone()
                })
                .sum();
            *val = sum;
        });

        Tensor {
            dimensions: result_dims,
            data: result_data,
        }
    }
}

impl<T: Default> Default for Tensor<T> {
    fn default() -> Self {
        Tensor {
            dimensions: vec![],
            data: vec![],
        }
    }
}

impl Add for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Tensor<f32> {
        self.add_tensor(&other)
    }
}

impl Mul for Tensor<f32> {

    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Tensor<f32> {
        self.dot_tensor(&other)
    }


}

impl Tensor<f32> {
    pub fn new_rand(dimensions: Vec<usize>) -> Tensor<f32> {
        let total_len: usize = dimensions.iter().product();
        let data: Vec<f32> = (0..total_len)
            .into_par_iter()
            .map(|_| rand::thread_rng().gen_range(0.0..1.0))
            .collect();
        Tensor { dimensions, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensors() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let tensor2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let result = tensor1.add_tensor(&tensor2);
        assert_eq!(result.dimensions, vec![2, 2]);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic]
    fn test_add_tensors_mismatched_dimensions() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let tensor2 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        tensor1.add_tensor(&tensor2);
    }

    #[test]
    fn test_tensor_add_operator() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let tensor2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let result = tensor1 + tensor2;
        assert_eq!(result.dimensions, vec![2, 2]);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_dot_product_for_vectors() {
        let tensor1 = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let tensor2 = Tensor::new(vec![4], vec![5.0, 6.0, 7.0, 8.0]);
        let result = tensor1 * tensor2;
        assert_eq!(result.data, vec![70.0]);
    }

    #[test]
    // fn test_elementwise_mul_2d() {
    //     let tensor1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    //     let tensor2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    //     let result = tensor1.dot_tensor(&tensor2);
    //     assert_eq!(result.dimensions, vec![2, 2]);
    //     assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
    // }

    #[test]
    fn test_matrix_multiplication() {
        // (2x3) * (3x2) = (2x2)
        let tensor1 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor2 = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = tensor1.dot_tensor(&tensor2);
        assert_eq!(result.dimensions, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(*tensor.get(&[0, 1]), 2.0);
        *tensor.get_mut(&[0, 1]) = 10.0;
        assert_eq!(*tensor.get(&[0, 1]), 10.0);
    }

    #[test]
    fn test_get_row() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = tensor.get_row(1);
        let expected: Vec<f32> = vec![4.0, 5.0, 6.0];
        for (a, b) in row.iter().zip(expected.iter()) {
            assert_eq!(**a, *b);
        }
    }

    #[test]
    fn test_new_rand_tensor() {
        let tensor = Tensor::new_rand(vec![2, 2]);
        assert_eq!(tensor.dimensions, vec![2, 2]);
        assert_eq!(tensor.data.len(), 4);
        for v in tensor.data.iter() {
            assert!(*v >= 0.0 && *v < 1.0);
        }
    }

    #[test]
    fn test_high_dimensional_tensor_elementwise_mul() {
        let tensor1 = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tensor2 = Tensor::new(vec![2, 2, 2], vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let result = tensor1.dot_tensor(&tensor2);
        assert_eq!(result.dimensions, vec![2, 2, 2]);
        assert_eq!(result.data, vec![8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]);
    }
}