// File: src/lib/transformer.rs
use crate::lib::vocab::Vocab;
use crate::lib::math::Tensor;

pub struct Embedding {
    pub vocab: Vocab,
    pub d_model: usize,
    pub data: Tensor<f32>,
}


impl Embedding {

    pub fn new(vocab: Vocab, d_model : usize) -> Self {
        let data = Tensor::new_rand(vec![vocab.index, d_model]);
        Embedding {
            vocab,
            d_model,
            data,
        }
    }

    pub fn embed(&self, sentence: &str) -> Option<Tensor<f32>> {
        
        let seq: Vec<_> = sentence.split_whitespace().map(|word| word.trim_matches(|c: char| !c.is_alphanumeric())).filter(|word| !word.is_empty()).enumerate().map(|(i, word)| (i, word.to_string())).collect();
        let mut pos_arg: Vec<_> = Vec::new();
        for i in 0..seq.len() {
            let word_idx = self.vocab.get(&seq[i].1)?;
            let embedding_row = self.data.get_row(word_idx.1 as usize);
            pos_arg.push((embedding_row, seq[i].0 as u32));
        }

    //     for i in 0..seq

        

        return Some(Tensor::new_rand(vec![seq.len(), self.d_model])) } // Placeholder for actual embedding logic 


    }





