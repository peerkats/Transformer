// file: src/lib/vocab.rs
use crate::lib::file::read_file_lines;
use std::collections::HashMap;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Vocab{
    pub index: usize,
    pub data: Vec<(String, u32)>,
    lookup: HashMap<String, u32>,
}

impl Vocab{
    pub fn new(data: Vec<String>) -> Self{

        let data_out: Vec<(String, u32)> = data
            .par_iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        

        let lookup: HashMap<String, u32> = data_out
            .par_iter()
            .map(|(s, idx)| (s.clone(), *idx))
            .collect();
        
        Vocab{
            index: data.len(),
            data: data_out,
            lookup,
        }
    }
    
    pub fn get(&self, string: &str) -> Option<(String, u32)> {
        self.lookup.get(string).map(|&idx| (string.to_string(), idx))
    }
    
    pub fn get_dimensions(&self) -> usize {
        self.index
    }
}

impl From<&str> for Vocab {
    fn from(path: &str) -> Self {
        let data = read_file_lines(path);
        Vocab::new(data.expect("Failed to read file"))
    }
}