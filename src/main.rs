mod lib;
use lib::vocab::Vocab;
use lib::file::read_file_lines;
use lib::math::Tensor;
use std::io;

const PATH: &str = "../transformer/20k.txt";
fn main(){  
    
    let vocab = Vocab::from(PATH);  // Use From trait instead of new
    let d_model = 256;
    // get an input from the user
    println!("Enter a word to get its index:");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    println!("You entered: {}", input.trim());
    println!("Index: {:?}", vocab.get(input.trim()));

    let tensort = Tensor::new_rand(vec![vocab.index, d_model]);
    let data = tensort.get_row(vocab.get(input.trim()).unwrap().1 as usize);

    println!("Tensor data for the word '{}': {:?}", input.trim(), data);
    println!("Tensor dimensions: {:?}", tensort.dimensions);
    println!("Tensor data length: {}", tensort.data.len());
}