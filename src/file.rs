use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;


/// Read the entire content of a file as a string
pub fn read_file_to_string<P: AsRef<Path>>(path: P) -> io::Result<String> {
    fs::read_to_string(path)
}

/// Read a file line by line and return a vector of lines
pub fn read_file_lines<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Result<Vec<String>, io::Error> = reader.lines().collect();
    lines
}

pub fn read_file_lines_iter<P: AsRef<Path>>(path: P) -> io::Result<impl Iterator<Item = io::Result<String>>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines())
}

pub fn file_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}


