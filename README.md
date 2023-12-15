# 二维码识别和解析工具

这个项目提供了一个用于识别和解析图片中二维码的工具。它结合了YOLOv8模型和OpenCV库，能够处理大量的图像文件，并且从中检测和解析二维码。

## 特点

- 使用YOLOv8进行二维码检测。
- 结合了OpenCV和WeChat QR解析器来增强解析能力。
- 支持多种图像格式，包括`.png`、`.jpg`、`.jpeg`和`.bmp`。
- 可以处理单个文件或扫描整个目录。
- 通过多进程提高处理效率。

## 安装

本项目依赖于Python 3.x以及一些外部库。你可以通过以下步骤安装所需的库：

```bash
pip install opencv-python
pip install ultralytics
```

注意：YOLOv8模型文件（`best.pt`）需要单独下载或训练。代码中的模型文件是已经训练过，可自行再次训练。

## 使用方法

1. 将你的二维码图片放置在一个目录中。
2. 指定目录路径和输出文件路径。
3. 运行程序。

### 示例

```python
# 设置要扫描的目录路径和输出文件路径
directory_path = "path_to_your_directory"
output_file_path = "path_to_your_output_file"

# 执行扫描操作
scan_directory_for_images(directory_path, output_file_path)
```

## 注意事项

- 确保YOLOv8模型文件`best.pt`位于正确的路径。
- WeChat QR解析器需要额外的模型文件。

## 贡献

如果您想为这个项目贡献代码或想法，请随时提交pull requests或开issue。

## 许可证

[MIT License](LICENSE)
