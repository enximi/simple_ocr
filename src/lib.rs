//! # Simple OCR
//! 一个简单的OCR封装。
//! ## 功能
//! 输入图片输出文字以及概率。
//! ## 注意事项
//! - 只支持裁剪好的单行文本图片的识别。图片高度缩放到32像素时，需要识别的字符与图片边缘的距离为4-8像素。
//! - 只支持简体中文和英文识别。
//! - 识别规整的文字效果好，例如截屏或者扫描的文本。
//! - 模型是[CnOCR](https://github.com/breezedeus/cnocr)的cnocr-v2.3-doc-densenet_lite_136-gru-epoch=004-ft-model.onnx。
//! ## 使用方法
//! ```rust
//! use simple_ocr::{ocr, ocr_with_image_path};
//!
//! let image_path = "assets/images/for_test/教训.png";
//! let (text, _prob) = ocr_with_image_path(image_path);
//! assert_eq!(text, "教训");
//!
//! let image = image::open(image_path).unwrap();
//! let (text, _prob) = ocr(image);
//! assert_eq!(text, "教训");
//! ```

use std::path::Path;

use image::{DynamicImage, GenericImage, GrayImage};
use lazy_static::lazy_static;
use ndarray::{array, Array, ArrayBase, Axis, Dim, Dimension, Ix1, Ix3, Ix4, IxDyn, OwnedRepr};
use ort::{inputs, Session};

lazy_static! {
    static ref MODEL: Session = Session::builder()
        .unwrap()
        .commit_from_memory(include_bytes!(
            "../assets/models/cnocr-v2.3-doc-densenet_lite_136-gru-epoch=004-ft-model.onnx"
        ))
        .unwrap();
    static ref TEXT_LABELS: Vec<&'static str> = include_str!("../assets/labels/label_cn.txt")
        .lines()
        .filter(|text| !text.is_empty())
        .map(|text| match text {
            "<space>" => " ",
            text => text,
        })
        .collect();
}

pub fn ocr_with_image_path<P: AsRef<Path>>(image_path: P) -> (String, f32) {
    let image = image::open(image_path).unwrap();
    ocr(image)
}

pub fn ocr(image: DynamicImage) -> (String, f32) {
    let (input_images, input_lengths) = image_to_onnx_input(image);
    let logits = infer(input_images, input_lengths);
    post_process(logits)
}

fn image_to_onnx_input(image: DynamicImage) -> (Array<f32, Ix4>, Array<i64, Ix1>) {
    fn to_light_background_gray(image: DynamicImage) -> GrayImage {
        let (width, height) = (image.width(), image.height());
        let image = image.into_luma8();
        let image_data = image.into_vec();
        let avg = image_data.iter().map(|&x| x as f32).sum::<f32>() / image_data.len() as f32;
        let image_data = if avg > 255.0 / 2.0 {
            image_data
        } else {
            image_data.into_iter().map(|x| 255 - x).collect::<Vec<u8>>()
        };
        GrayImage::from_vec(width, height, image_data).unwrap()
    }

    const HEIGHT: u32 = 32;

    fn resize_to_32_height_and_at_least_8_width(image: GrayImage) -> GrayImage {
        let image = {
            let new_width =
                (HEIGHT as f32 / image.height() as f32 * image.width() as f32).round() as u32;
            DynamicImage::ImageLuma8(image).resize_exact(
                new_width,
                HEIGHT,
                image::imageops::FilterType::Lanczos3,
            )
        };
        const MIN_WIDTH: u32 = 8;
        if image.width() < MIN_WIDTH {
            let new_image_data = vec![255; MIN_WIDTH as usize * image.height() as usize];
            let new_image = GrayImage::from_vec(MIN_WIDTH, HEIGHT, new_image_data).unwrap();
            let mut new_image = DynamicImage::ImageLuma8(new_image);
            new_image.copy_from(&image, 0, 0).unwrap();
            new_image.into_luma8()
        } else {
            image.into_luma8()
        }
    }

    /// 标准图片转为输入数组形式。标准图片：灰度图，高度32，宽度至少为8，浅色背景，深色文字，字符距离图片边界4-8像素。
    /// 输入数组形状：通道数*高度*宽度，通道数为1，高度为32，宽度为图片宽度。
    fn stand_image_to_input_array(stand_image: GrayImage) -> Array<f32, Ix3> {
        let image_width = stand_image.width();
        let image_data = stand_image
            .into_vec()
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        Array::from_shape_vec((HEIGHT as usize, image_width as usize, 1), image_data)
            .unwrap()
            .permuted_axes([2, 0, 1])
            / 255.0
    }

    let image = to_light_background_gray(image);
    let image = resize_to_32_height_and_at_least_8_width(image);
    let width = image.width();
    let input_images = stand_image_to_input_array(image).insert_axis(Axis(0));
    let input_lengths = array![width as i64];
    (input_images, input_lengths)
}

fn infer(input_images: Array<f32, Ix4>, input_lengths: Array<i64, Ix1>) -> Array<f32, IxDyn> {
    let output = MODEL
        .run(
            inputs![
                "x" => input_images,
                "input_lengths" => input_lengths,
            ]
            .unwrap(),
        )
        .unwrap();
    output["logits"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .into_owned()
}

fn post_process(logits: ArrayBase<OwnedRepr<f32>, IxDyn>) -> (String, f32) {
    // batch size is 1 so logits_shape[0] is 1, remove batch axis
    let logits = logits.remove_axis(Axis(0));

    // softmax
    let probs_shape = logits.shape();
    let probs = logits.map_axis(Axis(1), |x| {
        let max = x.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
        let sum = x.iter().map(|&v| (v - max).exp()).sum::<f32>();
        x.iter()
            .map(|&v| (v - max).exp() / sum)
            .collect::<Vec<f32>>()
    });
    let probs_data = probs
        .into_raw_vec()
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<f32>>();
    let probs = Array::from_shape_vec(probs_shape, probs_data)
        .unwrap()
        .permuted_axes(Dim([1, 0]).into_dyn());

    let mut best_path = probs
        .map_axis(Axis(0), |x| {
            x.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(index, max_value), (i, &v)| {
                    if v > max_value {
                        (i, v)
                    } else {
                        (index, max_value)
                    }
                })
                .0 as i64
        })
        .into_raw_vec();
    best_path.dedup();
    let text = best_path
        .into_iter()
        .filter_map(|x| TEXT_LABELS.get(x as usize))
        .copied()
        .collect::<String>();
    let prob = probs
        .map_axis(Axis(0), |x| {
            x.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v))
        })
        .into_raw_vec()
        .into_iter()
        .fold(f32::INFINITY, |acc, v| acc.min(v));

    (text, prob)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ocr() {
        let image_path = "assets/images/for_test/教训.png";
        let (text, prob) = ocr_with_image_path(image_path);
        assert_eq!(text, "教训");
        assert!(prob > 0.9);
    }
}
