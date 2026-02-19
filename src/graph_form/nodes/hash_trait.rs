

pub trait FromHashMap: Sized {
    fn from_hashmap(attrs: &HashMap<String,>) -> Result<Self>;
}