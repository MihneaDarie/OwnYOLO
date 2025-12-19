mod yolo;
use anyhow::Result;
use ndarray::{Array, Array4};
use opencv::{core::*, highgui, imgproc, prelude::*, videoio};

use crate::yolo::{buffers::Buffers, yolov8::YoloV8};

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
    imgproc::cvt_color(
        &resized,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_ACCURATE,
    )?;

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

fn main() -> Result<()> {
    let camera_index = 0;

    let mut camera = videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;

    camera.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 640.0)?;

    if !videoio::VideoCapture::is_opened(&camera)? {
        anyhow::bail!("Failed to open camera !");
    }

    let mut frame = Mat::default();

    let window_name = "Webcam Feed (Press 'q' to exit)";

    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    println!("Webcam feed open. Press 'q' to close the window.");

    let model = YoloV8::new("data/yolo_weights.npz")?;
    let mut buffers = Buffers::new();

    loop {
        let read_success = camera.read(&mut frame)?;

        let input = preprocess_image(&frame)?;

        model.forward(&input, &mut buffers)?;

        if !read_success {
            println!("Could not read frame from camera. Exiting.");
            break;
        }



        if frame.size()?.width > 0 {
            highgui::imshow(window_name, &frame)?;
        }

        let key = highgui::wait_key(1)?;

        if key == 113 || key == 27 {
            break;
        }
    }

    highgui::destroy_window(window_name)?;

    Ok(())
}
