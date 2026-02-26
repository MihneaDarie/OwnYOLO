use std::collections::HashMap;

use crate::graph_form::typed_array::TypedArray;

pub struct TensorMap {
    inner: HashMap<String, TypedArray>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: String, val: TypedArray) {
        self.inner.insert(key, val.ensure_contiguous());
    }

    #[inline]
    pub fn get_disjoint_mut<const N: usize>(
        &mut self,
        key: [&str; N],
    ) -> [Option<&mut TypedArray>; N] {
        self.inner.get_disjoint_mut(key)
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&TypedArray> {
        self.inner.get(key)
    }

    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut TypedArray> {
        self.inner.get_mut(key)
    }

    #[inline]
    pub fn remove(&mut self, key: &str) -> Option<TypedArray> {
        self.inner.remove(key)
    }

    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TypedArray)> {
        self.inner.iter()
    }
}
