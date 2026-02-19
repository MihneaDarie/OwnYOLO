use crate::graph_form::nodes::node::Node;
use anyhow::Result;

#[derive(Default)]
pub struct ReshapeNode {
    allow_zero: bool,
    next_node: Option<Box<dyn Node>>,
}

impl ReshapeNode {
    pub fn new(allow_zero: bool) -> Self {
        Self {
            allow_zero,
            next_node: None,
        }
    }
}

impl Node for ReshapeNode {
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
