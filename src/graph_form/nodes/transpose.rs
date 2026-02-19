use crate::graph_form::nodes::node::Node;
use anyhow::Result;

#[derive(Default)]
pub struct TransposeNode {
    perm: [i64; 4],

    next_node: Option<Box<dyn Node>>,
}
impl TransposeNode {
    pub fn new( perm: [i64; 4],) -> Self {
        Self { perm, next_node: None }
    }
}

impl Node for TransposeNode {
    fn pass(&self) {
        todo!()
    }
    fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.next_node {
            next.self_count(count + 1)
        } else {
            count
        }
    }
    fn insert(&mut self, next: Box<dyn Node>) -> Result<()> {
        if let Some(next_node) = &mut self.next_node {
            next_node.insert(next)?;
            return Ok(());
        } else {
            self.next_node = Some(next)
        }
        Ok(())
    }
}
