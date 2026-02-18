mod yolo;
use std::time::Instant;

use anyhow::{Ok, Result};
use ndarray::{Array, Array4};
use opencv::{core::*, highgui, imgproc, prelude::*, videoio};

use crate::yolo::{
    buffers::Buffers,
    context::appcontext::get_global_context,
    yolov8::{COCO_CLASSES, Detection, YoloV8},
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

    let resized_w = (in_w * scale).round() as i32;
    let resized_h = (in_h * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(resized_w, resized_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let pad_w = 640 - resized_w;
    let pad_h = 640 - resized_h;

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
    // imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

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
    let c = get_global_context();

    println!("{c:?}");

    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();
    // let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    let mut camera = videoio::VideoCapture::from_file("/dev/video0", videoio::CAP_V4L2)?;

    camera.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;

    if !videoio::VideoCapture::is_opened(&camera)? {
        anyhow::bail!("Failed to open camera!");
    }

    let mut frame = Mat::default();

    let window_name = "YOLOv8 Object Detection (Press 'q' to exit)";
    highgui::named_window(window_name, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(window_name, 1280, 720)?;

    println!("Loading YOLOv8 model...");
    let model = YoloV8::new("data/yolo_weights_fused.npz")?;
    let mut buffers = Buffers::new();
    println!("Model loaded!");

    let conf_threshold = 0.25;
    let iou_threshold = 0.35;

    let mut frame_count = 0;
    let mut total_inference_time = 0.0;

    println!("Starting detection. Press 'q' to quit.");

    loop {
        frame_count += 1;

        let read_success = camera.read(&mut frame)?;
        if !read_success || frame.empty() {
            println!("Could not read frame from camera. Exiting.");
            break;
        }

        let (input, scale, pad_x, pad_y) = preprocess_image(&frame)?;

        let start_inference = Instant::now();
        model.forward(&input, &mut buffers)?;
        let inference_time = start_inference.elapsed();
        total_inference_time += inference_time.as_secs_f32();

        let detections = model.postprocess(
            &buffers.model_22_buffer.final_output,
            conf_threshold,
            iou_threshold,
        );

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

        let key = highgui::wait_key(1)?;
        if key == 113 || key == 27 {
            break;
        }
        // break;
    }

    highgui::destroy_window(window_name)?;
    Ok(())
}
