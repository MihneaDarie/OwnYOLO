mod graph_form;
mod yolo;
use std::time::Instant;

use anyhow::{Ok, Result};
use ndarray::{Array, Array4, Ix3};
use opencv::{core::*, highgui, imgproc, prelude::*, videoio};

use crate::{
    graph_form::{graph::GraphForm, tensor_map::TensorMap, typed_array::TypedArray},
    yolo::{
        buffers::Buffers, gemms::appcontext::get_global_context, yolov8::{COCO_CLASSES, Detection, YoloV8}
    },
};

const COLORS: [VecN<f64, 4>; 8] = [
    Scalar::new(255.0, 0.0, 0.0, 0.0),
    Scalar::new(0.0, 255.0, 0.0, 0.0),
    Scalar::new(0.0, 0.0, 255.0, 0.0),
    Scalar::new(255.0, 255.0, 0.0, 0.0),
    Scalar::new(255.0, 0.0, 255.0, 0.0),
    Scalar::new(0.0, 255.0, 255.0, 0.0),
    Scalar::new(128.0, 0.0, 255.0, 0.0),
    Scalar::new(255.0, 128.0, 0.0, 0.0),
];

fn preprocess_image(frame: &Mat) -> Result<(Array4<f32>, f32, f32, f32)> {
    let in_w = frame.cols() as f32;
    let in_h = frame.rows() as f32;

    let new_size = 640.0f32;
    let scale = (new_size / in_w).min(new_size / in_h);

    let resized_w = (in_w * scale) as i32;
    let resized_h = (in_h * scale) as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(resized_w, resized_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let pad_w = (640 - resized_w).max(0);
    let pad_h = (640 - resized_h).max(0);

    let pad_left = pad_w / 2;
    let pad_right = pad_w - pad_left;
    let pad_top = pad_h / 2;
    let pad_bottom = pad_h - pad_top;

    let mut padded = Mat::default();
    copy_make_border(
        &resized,
        &mut padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        BORDER_CONSTANT,
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;

    let mut rgb = Mat::default();

    imgproc::cvt_color(
        &padded,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_ACCURATE,
    )?;

    let mut normalized = Mat::default();
    rgb.convert_to(&mut normalized, CV_32F, 1.0 / 255.0, 0.0)?;

    let rows = normalized.rows() as usize;
    let cols = normalized.cols() as usize;
    let channels = normalized.channels() as usize;

    let data = normalized.data_bytes()?;
    let float_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut chw = vec![0.0f32; channels * rows * cols];
    for h in 0..rows {
        for w in 0..cols {
            let base = (h * cols + w) * channels;
            for c in 0..channels {
                chw[c * rows * cols + h * cols + w] = float_data[base + c];
            }
        }
    }

    let input = Array::from_shape_vec((1, channels, rows, cols), chw)?;

    Ok((input, scale, pad_left as f32, pad_top as f32))
}

fn draw_detections(
    frame: &mut Mat,
    detections: &[Detection],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Result<()> {
    let fw = frame.cols() as f32;
    let fh = frame.rows() as f32;

    for det in detections {
        let color = COLORS[det.class_id % COLORS.len()];

        let x1 = ((det.bbox[0] - pad_x) / scale).clamp(0.0, fw - 1.0) as i32;
        let y1 = ((det.bbox[1] - pad_y) / scale).clamp(0.0, fh - 1.0) as i32;
        let x2 = ((det.bbox[2] - pad_x) / scale).clamp(0.0, fw - 1.0) as i32;
        let y2 = ((det.bbox[3] - pad_y) / scale).clamp(0.0, fh - 1.0) as i32;

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        imgproc::rectangle(
            frame,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            color,
            2,
            imgproc::LINE_8,
            0,
        )?;

        let label = format!("{}: {:.2}", COCO_CLASSES[det.class_id], det.confidence);
        imgproc::put_text(
            frame,
            &label,
            Point::new(x1, (y1 - 5).max(15)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let (mut graph, mut omap) = GraphForm::<f32>::from_onnx_file("models/yolov8n.onnx")?;

    graph.optimize();
    graph.print();

    // return Ok(());
    let c = get_global_context();

    println!("{c:?}");

    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    //windows
    // let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    // camera.set(videoio::CAP_PROP_FRAME_WIDTH, 720.0)?;
    // camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 1280.0)?;

    // linux
    let mut camera = videoio::VideoCapture::from_file("/dev/video0", videoio::CAP_V4L2)?;
    camera.set(videoio::CAP_PROP_FRAME_WIDTH, 720.0)?;
    camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 1280.0)?;

    if !videoio::VideoCapture::is_opened(&camera)? {
        anyhow::bail!("Failed to open camera!");
    }

    let mut frame = Mat::default();
    let window_name = "YOLOv8 Object Detection (Press 'q' to exit)";
    highgui::named_window(window_name, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(window_name, 500, 700)?;

    let conf_threshold = 0.25;
    let iou_threshold = 0.45;

    let mut frame_count = 0;
    let mut total_inference_time = 0.0;

    println!("Starting detection. Press 'q' to quit.");

    loop {
        frame_count += 1;

        let read_success = camera.read(&mut frame)?;
        if !read_success || frame.empty() {
            break;
        }

        let (input, scale, pad_x, pad_y) = preprocess_image(&frame)?;

        let start_inference = Instant::now();
        graph.pass(&mut omap, &input.into_dyn());
        let inference_time = start_inference.elapsed();
        total_inference_time += inference_time.as_secs_f32();

        let output = match omap.get("output0").unwrap() {
            TypedArray::F32(a) => a.view().into_dimensionality::<Ix3>().unwrap(),
            _ => panic!("output0 is not F32"),
        };

        let detections = postprocess_onnx(&output, conf_threshold, iou_threshold);

        draw_detections(&mut frame, &detections, scale, pad_x, pad_y)?;

        let avg_inference_time = total_inference_time / frame_count as f32;
        let fps = 1.0 / avg_inference_time;
        let info = format!(
            "FPS: {:.1} | Inference: {:.0}ms | Detections: {}",
            fps,
            inference_time.as_millis(),
            detections.len()
        );

        imgproc::put_text(
            &mut frame,
            &info,
            Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        highgui::imshow(window_name, &frame)?;

        if highgui::wait_key(1)? == 113 {
            break;
        }
        // break;
    }

    highgui::destroy_window(window_name)?;
    Ok(())
}

fn nms(detections: &mut [Detection], iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i]);

        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }

            if detections[i].class_id == detections[j].class_id {
                let iou = compute_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let a_area = (a[2] - a[0]) * (a[3] - a[1]);
    let b_area = (b[2] - b[0]) * (b[3] - b[1]);

    let union_area = a_area + b_area - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

fn postprocess_onnx(
    output: &ndarray::ArrayView3<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Vec<Detection> {
    let num_boxes = output.shape()[2];
    let mut detections = Vec::new();

    for i in 0..num_boxes {
        let mut best_class = 0;
        let mut best_score = 0.0f32;
        for c in 0..80 {
            let score = output[[0, 4 + c, i]];
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        let cx = output[[0, 0, i]];
        let cy = output[[0, 1, i]];
        let w = output[[0, 2, i]];
        let h = output[[0, 3, i]];

        detections.push(Detection {
            bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            confidence: best_score,
            class_id: best_class,
        });
    }

    nms(&mut detections, iou_threshold)
}
