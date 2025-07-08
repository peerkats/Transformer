
#[derive(Debug)]
pub struct Vocab{
    pub index: usize,
    pub data: Vec<(String, u32)>

}

impl Vocab{
    pub fn new(data: Vec<String>) -> Self{
        let mut data_out: Vec<(String, u32)> = Vec::new();
        for i in 0..data.len(){
            data_out.push((data[i].to_string(), i.try_into().unwrap()));
        }
        Vocab{
            index: data.len(),
            data: data_out,
        }
    }
    pub fn get(&self, string: &str) -> Option<(String, u32)> {
        for (s, idx) in &self.data {
            if s == string {
                return Some((s.clone(), *idx));
            }
        }
        None
    }
    pub fn get_dimensions(&self) -> usize {
        self.index
    }

    
}
