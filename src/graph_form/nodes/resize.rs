use std::default;

use crate::graph_form::nodes::node::Node;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    #[default]
    Nearest,
    Linear,
    Cubic,
}

impl Mode {
    pub fn from_str(str: &str) -> Self {
        match str {
            "linear" => Self::Linear,
            "cubic" => Self::Cubic,
            _ => Self::Nearest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateTransformationMode {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    #[default]
    Asymmetric,
    TfCropAndResize,
}

impl CoordinateTransformationMode {
    pub fn from_str(str: &str) -> Self {
        match str {
            "half_pixel" => Self::HalfPixel,
            "half_pixel_symmetric" => Self::HalfPixelSymmetric,
            "pytorch_half_pixel" => Self::PytorchHalfPixel,
            "align_corners" => Self::AlignCorners,
            "tf_crop_and_resize" => Self::TfCropAndResize,
            _ => Self::Asymmetric
        }
    }   
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepAspectRatioPolicy {
    #[default]
    NotLarger,
    NotSmaller,
}

impl  KeepAspectRatioPolicy {
    pub fn from_str(str: &str) -> Self {
        match str {
            "not_smaller" => Self::NotSmaller,
            _ => Self::NotLarger
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NearestMode {
    #[default]
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

impl NearestMode {
    pub fn from_str(str: &str) -> Self {
        match str {
            "round_prefer_ceil" => Self::RoundPreferCeil,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            _ => Self::RoundPreferFloor
        }
    }
}

#[derive(Default)]
pub struct ResizeNode {
    antialias: i64,
    mode: Mode,
    cubic_coeff_a: f32,
    exclude_outside: bool,
    extrapolation_value: f32,
    keep_aspect_ratio_policy: KeepAspectRatioPolicy,
    neares_mode: NearestMode,
    coordinate_transformation_mode: CoordinateTransformationMode,

    next_node: Option<Box<dyn Node>>,
}

impl ResizeNode {
    pub fn new(
        antialias: i64,
        mode: &str,
        cubic_coeff_a: f32,
        exclude_outside: bool,
        extrapolation_value: f32,
        keep_aspect_ratio_policy: &str,
        coordinate_transformation_mode: &str,
        neares_mode: &str,
    ) -> Self { 
        Self {
            antialias,
            mode: Mode::from_str(mode),
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy: KeepAspectRatioPolicy::from_str(keep_aspect_ratio_policy),
            neares_mode: NearestMode::from_str(neares_mode),
            coordinate_transformation_mode: CoordinateTransformationMode::from_str(coordinate_transformation_mode),
            next_node: None,
        }
    }
}

impl Node for ResizeNode {
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
