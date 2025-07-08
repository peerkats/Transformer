// File: src/lib/transformer.rs
use crate::lib::vocab::Vocab;
use crate::lib::math::Tensor;

struct embedding {
    pub vocab: Vocab,
    pub d_model: usize,
    pub data: Tensor<f32>,
}

