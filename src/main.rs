mod lib;
use lib::vocab::Vocab;
use lib::file::read_file_lines;
use lib::math::Tensor;
use lib::transformer::Embedding; // Assuming Embedding is defined in transformer.rs
use std::io;

const PATH: &str = "../transformer/20k.txt";
fn main(){  
    
    // let vocab = Vocab::from(PATH);  // Use From trait instead of new
    // let d_model = 256;

    
    // // get an input from the user
    // println!("Enter a word to get its index:");
    // let mut input = String::new();
    // io::stdin().read_line(&mut input).expect("Failed to read line");
    // println!("You entered: {}", input.trim());
    // println!("Index: {:?}", vocab.get(input.trim()));

    // let tensort = Tensor::new_rand(vec![vocab.index, d_model]);
    // let data = tensort.get_row(vocab.get(input.trim()).unwrap().1 as usize);

    // let test = Tensor::new_rand(vec![1, 2, 3]);
    // let test2 = Tensor::new_rand(vec![1, 2, 3]);
    // let mut test3 = test + test2;

    let tensor1 = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = Tensor::new(vec![4], vec![5.0, 6.0, 7.0, 8.0]);
    let mut result = tensor1 *  tensor2;
    println!("{:?}", result);
    // let embedding = Embedding::new(vocab, d_model);
    // embedding.embed(input.trim()).expect("Failed to embed sentence");
}