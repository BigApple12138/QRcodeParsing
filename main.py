import os
import time
from multiprocessing import Pool
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode

# 加载预训练的 YOLOv8 模型
model = YOLO('best.pt')  # 指定你的模型文件路径


def detect_qrcode_from_image(image):
    # 读取图片
    result = detect_with_wechat_qrcode(image)
    if result:
        return result
    else:
        return detect_with_pyzbar(image)

def detect_with_pyzbar(image):
    """使用 pyzbar 库检测图像中的二维码，并返回所有检测到的二维码的解析结果"""
    def adjust_contrast_brightness(image, alpha=1.0, beta=0):
        """调整图像的对比度和亮度"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    scale_factor = 1.5
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    # 图片放大
    enlarged = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # 调整亮度和对比度
    enhanced = adjust_contrast_brightness(enlarged, alpha=3, beta=0)

    decoded_objects = decode(enhanced)
    qr_codes = [obj for obj in decoded_objects if obj.type == 'QRCODE']
    if qr_codes:
        return qr_codes[0].data.decode('utf-8')
    return None


def detect_with_wechat_qrcode(image):
    scale_factor = 3
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    # 图片放大
    enlarged = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    wechat_qr = cv2.wechat_qrcode_WeChatQRCode(
        "models/detect.prototxt",
        "models/detect.caffemodel",
        "models/sr.prototxt",
        "models/sr.caffemodel"
    )
    decoded_info, points = wechat_qr.detectAndDecode(enlarged)
    return decoded_info, points


def save_detected_qrcodes_to_file(detected_qrcodes, save_file):
    """将检测到的二维码数据保存到指定文件"""
    with open(save_file, 'w') as file:
        for qrcode in detected_qrcodes:
            file.write(qrcode + '\n')


def detect_qrcode_yolov8(image):
    """使用 YOLOv8 模型检测图像中的二维码，并返回所有检测到的二维码的边界框信息"""
    results = model.predict(image)
    qrcode_boxes = []  # 存储检测到的所有二维码的边界框

    for bbox in results[0].boxes.data:
        x1, y1, x2, y2, score, label = bbox
        qrcode_boxes.append((x1, y1, x2, y2))

    return qrcode_boxes


def process_image(file_path):
    """处理单个图像文件并返回检测到的所有二维码的解析结果"""
    decoded_infos = []  # 用于存储解析出的二维码信息

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image = cv2.imread(file_path)
        if image is None:
            return False, []  # 如果图像未正确加载

        # 使用 YOLOv8 检测二维码
        qrcode_boxes = detect_qrcode_yolov8(image)

        if qrcode_boxes:
            for qr_bbox in qrcode_boxes:
                # 裁剪二维码区域
                x1, y1, x2, y2 = [int(coord) for coord in qr_bbox]
                qr_roi = image[y1:y2, x1:x2]

                # # 保存裁剪的二维码图片
                # qr_image_path = os.path.join(r"/TEST",
                #                              f"qr_{os.path.basename(file_path)}")
                # cv2.imwrite(qr_image_path, qr_roi)

                # 使用 WeChat QR 解析器解析二维码
                decoded_info, points = detect_qrcode_from_image(qr_roi)

                if decoded_info:
                    print(f"在图片 {file_path} 中检测到并成功解析二维码：{decoded_info[0]}")
                    decoded_infos.append(decoded_info[0])

            if decoded_infos:
                return True, decoded_infos  # 二维码检测到且至少有一个解析成功
            else:
                print(f"在图片 {file_path} 中检测到二维码，但未能成功解析任何一个")
                return True, []  # 二维码检测到但全部解析失败
        else:
            print(f"图片 {file_path} 中未检测到二维码。")
            # os.remove(file_path)
            return False, []  # 二维码未检测到

    return False, []  # 文件格式不符合


def scan_directory_for_images(directory, output_file, processes=10):
    start_time = time.time()

    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    with Pool(processes) as pool:
        results = pool.map(process_image, files)

    # 统计检测到二维码的图片数量和成功解析的二维码数量
    qr_detected_count = sum(1 for detected, _ in results if detected)
    successfully_decoded_count = sum(len(result) for _, result in results if result)

    # 保存检测到的所有二维码数据
    detected_qrcodes = [qr for _, qrcodes in results for qr in qrcodes if qrcodes]
    save_detected_qrcodes_to_file(detected_qrcodes, output_file)

    # 打印统计信息
    total_images = len(files)
    recognition_rate = (successfully_decoded_count / qr_detected_count) * 100 if qr_detected_count > 0 else 0
    print(f"\n处理的图片总数: {total_images}")
    print(f"检测到二维码的图片数: {qr_detected_count}")
    print(f"成功解析的二维码数: {successfully_decoded_count}")
    print(f"识别率: {recognition_rate:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"处理时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    directory_path = r""
    output_file_path = r""
    scan_directory_for_images(directory_path, output_file_path)
