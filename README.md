# Simple OCR

一个简单的OCR封装。

## 功能

输入图片输出文字以及概率。

## 注意事项

- 只支持裁剪好的单行文本图片的识别。图片高度缩放到32像素时，需要识别的字符与图片边缘的距离为4-8像素。
- 只支持简体中文和英文识别。
- 识别规整的文字效果好，例如截屏或者扫描的文本。
- 模型是[CnOCR](https://github.com/breezedeus/cnocr)的cnocr-v2.3-doc-densenet_lite_136-gru-epoch=004-ft-model.onnx。

## 图片示例

![图片示例1](doc/image_1.png)

## 使用方法

```rust
use simple_ocr::{ocr, ocr_with_image_path};

let image_path = "assets/images/for_test/教训.png";
let (text, prob) = ocr_with_image_path(image_path);
assert_eq!(text, "教训");
assert!(prob > 0.9);

let image = image::open(image_path).unwrap();
let (text, prob) = ocr(image);
assert_eq!(text, "教训");
assert!(prob > 0.9);
```
