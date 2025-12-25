mod yolo;
use std::time::Instant;

use anyhow::Result;
use ndarray::{Array, Array4};
use opencv::{core::*, highgui, imgproc, prelude::*, videoio};

use crate::yolo::{
    buffers::Buffers,
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

fn preprocess_image(frame: &Mat) -> Result<Array4<f32>> {
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        frame,
        &mut resized,
        Size::new(640, 640),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let mut normalized = Mat::default();
    rgb.convert_to(&mut normalized, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

    let rows = normalized.rows();
    let cols = normalized.cols();
    let channels = normalized.channels();

    let data = normalized.data_bytes()?;
    let float_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut chw_data = vec![0.0f32; channels as usize * rows as usize * cols as usize];
    for h in 0..rows as usize {
        for w in 0..cols as usize {
            for c in 0..channels as usize {
                let hwc_idx = (h * cols as usize + w) * channels as usize + c;
                let chw_idx = c * (rows as usize * cols as usize) + h * cols as usize + w;
                chw_data[chw_idx] = float_data[hwc_idx];
            }
        }
    }

    let array = Array::from_shape_vec(
        (1, channels as usize, rows as usize, cols as usize),
        chw_data,
    )?;
    Ok(array)
}

fn draw_detections(
    frame: &mut Mat,
    detections: &[Detection],
    scale_x: f32,
    scale_y: f32,
) -> Result<()> {
    for det in detections {
        let color = COLORS[det.class_id % COLORS.len()];

        let x1 = (det.bbox[0] * scale_x) as i32;
        let y1 = (det.bbox[1] * scale_y) as i32;
        let x2 = (det.bbox[2] * scale_x) as i32;
        let y2 = (det.bbox[3] * scale_y) as i32;

        imgproc::rectangle(
            frame,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            color,
            2,
            imgproc::LINE_8,
            0,
        )?;

        let label = format!("{}: {:.2}", COCO_CLASSES[det.class_id], det.confidence);
        let font_scale = 0.5;
        let thickness = 1;
        let mut baseline = 0;
        let text_size = imgproc::get_text_size(
            &label,
            imgproc::FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
            &mut baseline,
        )?;

        let label_y = (y1 - 5).max(text_size.height + 5);

        imgproc::rectangle(
            frame,
            Rect::new(
                x1,
                label_y - text_size.height - 5,
                text_size.width + 10,
                text_size.height + 10,
            ),
            color,
            -1,
            imgproc::LINE_8,
            0,
        )?;

        imgproc::put_text(
            frame,
            &label,
            Point::new(x1 + 5, label_y),
            imgproc::FONT_HERSHEY_SIMPLEX,
            font_scale,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            thickness,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let camera_index = 0;
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();
    let mut camera = videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;

    camera.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;

    if !videoio::VideoCapture::is_opened(&camera)? {
        anyhow::bail!("Failed to open camera!");
    }

    let mut frame = Mat::default();

    let window_name = "YOLOv8 Object Detection (Press 'q' to exit)";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    println!("Loading YOLOv8 model...");
    let model = YoloV8::new("data/yolo_weights.npz")?;
    let mut buffers = Buffers::new();
    println!("Model loaded!");

    let conf_threshold = 0.25;
    let iou_threshold = 0.45;

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

        let frame_width = frame.cols() as f32;
        let frame_height = frame.rows() as f32;
        let scale_x = frame_width / 640.0;
        let scale_y = frame_height / 640.0;

        let input = preprocess_image(&frame)?;

        let start_inference = Instant::now();
        model.forward(&input, &mut buffers)?;
        let inference_time = start_inference.elapsed();
        total_inference_time += inference_time.as_secs_f32();

        let detections = model.postprocess(
            &buffers.model_22_buffer.final_output,
            conf_threshold,
            iou_threshold,
        );

        draw_detections(&mut frame, &detections, scale_x, scale_y)?;

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
