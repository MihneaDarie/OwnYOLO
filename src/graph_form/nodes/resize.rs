use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

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
            _ => Self::Asymmetric,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepAspectRatioPolicy {
    #[default]
    NotLarger,
    NotSmaller,
}

impl KeepAspectRatioPolicy {
    pub fn from_str(str: &str) -> Self {
        match str {
            "not_smaller" => Self::NotSmaller,
            _ => Self::NotLarger,
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
            _ => Self::RoundPreferFloor,
        }
    }
}

#[derive(Default)]
pub struct ResizeNode {
    antialias: i64,
    axes: Vec<i64>,
    mode: Mode,
    cubic_coeff_a: f32,
    exclude_outside: bool,
    extrapolation_value: f32,
    keep_aspect_ratio_policy: KeepAspectRatioPolicy,
    neares_mode: NearestMode,
    coordinate_transformation_mode: CoordinateTransformationMode,

    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for ResizeNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            antialias: match attrs.get("antialias") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            axes: {
                match attrs.get("axes") {
                    Some(av) => av.as_ints().unwrap().to_vec(),
                    None => vec![],
                }
            },
            mode: match attrs.get("mode") {
                Some(av) => Mode::from_str(&av.as_string().unwrap()),
                None => Mode::default(),
            },
            cubic_coeff_a: match attrs.get("cubic_coeff_a") {
                Some(av) => av.as_float().unwrap(),
                None => 0.0f32,
            },
            exclude_outside: match attrs.get("exclude_outside") {
                Some(av) => av.as_int().unwrap() != 0,
                None => false,
            },
            extrapolation_value: match attrs.get("extrapolation_value") {
                Some(av) => av.as_float().unwrap(),
                None => 0.0f32,
            },
            keep_aspect_ratio_policy: match attrs.get("keep_aspect_ratio_policy") {
                Some(av) => KeepAspectRatioPolicy::from_str(av.as_string().unwrap()),
                None => KeepAspectRatioPolicy::default(),
            },
            neares_mode: match attrs.get("nearest_mode") {
                Some(av) => NearestMode::from_str(av.as_string().unwrap()),
                None => NearestMode::default(),
            },
            coordinate_transformation_mode: match attrs.get("coordinate_transformation_mode") {
                Some(av) => CoordinateTransformationMode::from_str(av.as_string().unwrap()),
                None => CoordinateTransformationMode::default(),
            },
            next_node: None,
        })
    }
}

impl ResizeNode {
    pub fn new(
        antialias: i64,
        axes: Vec<i64>,
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
            axes,
            mode: Mode::from_str(mode),
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy: KeepAspectRatioPolicy::from_str(keep_aspect_ratio_policy),
            neares_mode: NearestMode::from_str(neares_mode),
            coordinate_transformation_mode: CoordinateTransformationMode::from_str(
                coordinate_transformation_mode,
            ),
            next_node: None,
        }
    }
}

impl Node for ResizeNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!(
            "resize-{},{:?},{:?},{},{},{},{:?},{:?},{:?}",
            self.antialias,
            self.axes,
            self.coordinate_transformation_mode,
            self.cubic_coeff_a,
            self.exclude_outside,
            self.extrapolation_value,
            self.keep_aspect_ratio_policy,
            self.mode,
            self.neares_mode
        );
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
