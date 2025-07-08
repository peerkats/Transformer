use rand::*;
use rayon::prelude::*;

pub struct Tensor<T> {
    pub dimensions: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Clone> Tensor<T> {
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