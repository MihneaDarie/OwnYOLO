use anyhow::Result;

pub trait Node {
    fn pass(&self) {}
    fn self_count(&self, count: usize) -> usize;
    fn insert(&mut self, next: Box<dyn Node>) -> Result<()>;
}
