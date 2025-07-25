import json
from flask import Flask, request, send_file, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from flask_cors import CORS
import io
# from test_object_recognition_cpu import main
from test_object_recognition import main
from tqdm import tqdm

app = Flask(__name__)
CORS(app, expose_headers=['Label'])  # 允许自定义 Label 响应头

# 存储上一次检测的结果
last_detection_result = None

@app.route('/detect', methods=['POST'])
def detect():
    global last_detection_result  # 声明使用全局变量

    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # 获取颜色参数
    color = request.form.get('color', 'red')  # 默认颜色为 'red'
    print(color)

    # 获取 bbox 边框粗细和字体大小
    bbox_thickness = int(request.form.get('bboxThickness', 5))  # 默认边框粗细为 2
    font_size = int(request.form.get('fontSize', 20))  # 默认字体大小为 12

    # 获取信号值
    signal = request.form.get('signal', 'detect')  # 默认信号值为 'detect'
    print(f"Signal: {signal}")

    # 获取前端传来的数据集名称
    dataset = request.form.get('dataset', 'FAIR1M')  # 默认数据集为 'FAIR1M'
    print(f"Dataset: {dataset}")  # 打印传来的数据集名称

    try:
        print(file)
        image = Image.open(file.stream)
        # print(image)
        # print("文件名:", file.filename)
        # print("图片形状:", image.size)

        if signal == 'detect':
            # 进行目标检测
            bboxes = main(file.filename, image, dataset)  # 将 dataset 传递给 main 函数
            last_detection_result = bboxes  # 存储检测结果
        elif signal == 'redraw' and last_detection_result is not None:
            # 在之前目标检测的基础上重新绘制
            bboxes = last_detection_result
        else:
            return "没有检测结果可供重新绘制", 400

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font='DejaVuSans.ttf', size=font_size)  
        label = {}
        for i, bbox in tqdm(enumerate(bboxes)):
            name = bbox[1]     
            confidence = float(bbox[2])  # 获取置信度
            x1, y1 = float(bbox[3]), float(bbox[4])
            x2, y2 = float(bbox[5]), float(bbox[6])
            x3, y3 = float(bbox[7]), float(bbox[8])
            x4, y4 = float(bbox[9]), float(bbox[10])
            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline=color, width=bbox_thickness)
            draw.text((x4, y4), str(i+1), fill=color, font=font)
            # label[str(i+1)] = {'name': name, 'confidence': confidence}
            label[str(i+1)] = name
            # print(label)

        # 将处理后的图片转换为字节流
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # 将 label 字典转换为 JSON 格式的字符串
        label_json = json.dumps(label)
        # print(label_json)

        # 返回字节流和 label 字典
        response = send_file(img_byte_arr, mimetype='image/png')
        response.headers['Label'] = label_json
        return response

    except Exception as e:
        print("错误:", str(e))
        return "图片处理失败", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
