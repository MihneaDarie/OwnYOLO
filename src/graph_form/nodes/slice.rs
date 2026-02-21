use crate::graph_form::nodes::node::Node;
use anyhow::Result;

#[derive(Default)]
pub struct SliceNode {

    data: String,
    starts: String,
    ends: String,
    axes: String,

    o: String,

    next_node: Option<Box<dyn Node>>,
}

impl SliceNode {
    pub fn add_input_strings(
        &mut self,
        data: String,
        starts: String,
        ends: String,
        axes: String,
    ) {
        self.data = data;
        self.starts = starts;
        self.ends = ends;
        self.axes = axes;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl Node for SliceNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("slice-{{}}");
        if let Some(next) = &self.next_node {
            next.print();
        }
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
