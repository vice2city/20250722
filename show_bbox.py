
import xml.etree.ElementTree as ET

from PIL import Image,ImageDraw,ImageFont
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd

import glob,os
# import cv2
import numpy as np

def get_file_names(folder_path):
    excel_files = glob.glob(os.path.join(folder_path, '*.png'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in excel_files if os.path.isfile(file)]
    return file_names

def get_xml_file_names(folder_path):
    file_names = glob.glob(os.path.join(folder_path, '*.xml'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_names if os.path.isfile(file)]
    return file_names

def show_rotate(file_list,csv_folder_path,img_folder_path,output_save_folder):
    dataset_name = "DOTA"
    for file in file_list:
        print(file)
        csv_path = f"{csv_folder_path}/{file}_{dataset_name}_output_bboxes.xlsx"
        img_path = f"{img_folder_path}/{file}.png"
        image = Image.open(img_path)
        bbox_thickness = int(5)  # 默认边框粗细为 2
        color = "red"
        df = pd.read_excel(csv_path)
        bboxes = df.values.tolist()
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font='DejaVuSans.ttf', size=80)  
        for bbox in bboxes:
            index = bbox[0]
            name = bbox[1]
            confidence = float(bbox[2])  # 获取置信度
            x1, y1 = float(bbox[3]), float(bbox[4])
            x2, y2 = float(bbox[5]), float(bbox[6])
            x3, y3 = float(bbox[7]), float(bbox[8])
            x4, y4 = float(bbox[9]), float(bbox[10])
            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline=color, width=bbox_thickness)
            draw.text((x4, y4), str(index), fill=color, font=font)
        image.save(f"{output_save_folder}/{file}.png")
        print(f"Visualization {output_save_folder}/{file}.png finished!")
        # break

img_folder_path = "/data5/laiping/tianzhibei/pretrain_datasets/17suo"
csv_folder_path = "/data5/laiping/tianzhibei/pretrain_datasets/目标坐标"
img_files_path = get_file_names(img_folder_path)
output_folder = "/data5/laiping/tianzhibei/pretrain_datasets/17suo_vis"
# 检查文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"文件夹 {output_folder} 创建成功")
else:
    print(f"文件夹 {output_folder} 已经存在")
# print()
show_rotate(img_files_path,csv_folder_path,img_folder_path,output_folder)
