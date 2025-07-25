import xml.etree.ElementTree as ET

from PIL import Image,ImageDraw

import glob,os
import cv2
import numpy as np
import json
from tqdm import tqdm

def get_file_names(folder_path):
    file_names = glob.glob(os.path.join(folder_path, '*.xml'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_names if os.path.isfile(file)]
    return file_names


def rotate_crop(file_list,xml_file,image_path,output_folder):
    print("crop image")
    new_data = []
    for file in tqdm(file_list):
        # print(file)
        i = file.split(".")[0]
        tree = ET.parse(f'{xml_file}/{i}.xml')
        jpg_path = f'{image_path}/{i}.jpg'
        tif_path = f'{image_path}/{i}.tif'

        if os.path.exists(jpg_path):
            img_path = jpg_path
            # print(f"使用 JPG 文件路径: {img_path}")
        elif os.path.exists(tif_path):
            img_path = tif_path
            # print(f"使用 TIF 文件路径: {img_path}")
        else:
            print("未找到 JPG 或 TIF 文件")
        # img_path = f'{image_files}/{i}.jpg'
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        root = tree.getroot()

    # 查找所有的 <points> 标签
        object_list = root.findall('./object')
        if len(object_list) == 0:
            object_list = root.findall('./objects/object')
        classes = root.findall(".//origin")[0].text
        # print(len(object_list))

        for idx,object in enumerate(object_list):
            data = {
                "classes":"",
                "filename":"",
            }
            
            name = object.findall('.//name')[0].text
            name = "_".join(name.split())
            # print(name)
            # 获取 <points> 标签的文本内容
            points_list = object.findall('.//point')
            points_data_list = []
            for points in points_list:
                points_data = points.text.split(",")
                float_list = [float(element) for element in points_data]
                points_data_list.append(float_list)

            x1, y1 = points_data_list[0]
            x2, y2 = points_data_list[1]
            x3, y3 = points_data_list[2]
            x4, y4 = points_data_list[3]
            points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            img = cv2.imread(img_path)
            points_list = np.array(points_list,dtype=np.float32)
            # 定义旋转框的四个顶点坐标
            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
            # 计算目标矩形的宽度和高度
            width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
            height = int(max(np.linalg.norm(points[1] - points[2]), np.linalg.norm(points[3] - points[0])))

            target_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(points, target_points)

            # 进行透视变换以裁剪出旋转框
            cropped_image = cv2.warpPerspective(img, matrix, (width, height))
            
            # 构造新文件名
            output_name = f"{os.path.splitext(file)[0]}_cropped_{idx}_{name}.png"
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, cropped_image)
            # print(f"Saved cropped image: {output_path}")
            data["classes"] = name
            data['filename'] = output_name
            new_data.append(data)
                # break
        #     break
        # break   
    return new_data

def rotate_crop_1(file_list,xml_file,image_path,output_folder):
    print("crop image")
    new_data = []
    for file in tqdm(file_list):
        # print(file)
        i = file.split(".")[0]
        tree = ET.parse(f'{xml_file}/{i}.xml')
        jpg_path = f'{image_path}/{i}.jpg'
        tif_path = f'{image_path}/{i}.tif'

        if os.path.exists(jpg_path):
            img_path = jpg_path
            # print(f"使用 JPG 文件路径: {img_path}")
        elif os.path.exists(tif_path):
            img_path = tif_path
            # print(f"使用 TIF 文件路径: {img_path}")
        else:
            print("未找到 JPG 或 TIF 文件")
        # img_path = f'{image_files}/{i}.jpg'
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        root = tree.getroot()

    # 查找所有的 <points> 标签
        object_list = root.findall('./object')
        if len(object_list) == 0:
            object_list = root.findall('./objects/object')
        classes = root.findall(".//origin")[0].text
        # print(len(object_list))

        for idx,object in enumerate(object_list):
            data = {
                "classes":"",
                "filename":"",
            }
            
            name = object.findall('.//name')[0].text
            name = "_".join(name.split())
            # # print(name)
            # # 获取 <points> 标签的文本内容
            # points_list = object.findall('.//point')
            # points_data_list = []
            # for points in points_list:
            #     points_data = points.text.split(",")
            #     float_list = [float(element) for element in points_data]
            #     points_data_list.append(float_list)

            # x1, y1 = points_data_list[0]
            # x2, y2 = points_data_list[1]
            # x3, y3 = points_data_list[2]
            # x4, y4 = points_data_list[3]
            # points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            # img = cv2.imread(img_path)
            # points_list = np.array(points_list,dtype=np.float32)
            # # 定义旋转框的四个顶点坐标
            # points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
            # # 计算目标矩形的宽度和高度
            # width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
            # height = int(max(np.linalg.norm(points[1] - points[2]), np.linalg.norm(points[3] - points[0])))

            # target_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

            # # 计算透视变换矩阵
            # matrix = cv2.getPerspectiveTransform(points, target_points)

            # # 进行透视变换以裁剪出旋转框
            # cropped_image = cv2.warpPerspective(img, matrix, (width, height))
            
            # # 构造新文件名
            output_name = f"{os.path.splitext(file)[0]}_cropped_{idx}_{name}.png"
            # output_path = os.path.join(output_folder, output_name)
            # cv2.imwrite(output_path, cropped_image)
            # print(f"Saved cropped image: {output_path}")
            data["classes"] = name
            data['filename'] = output_name
            new_data.append(data)
                # break
        #     break
        # break   
    return new_data


def std_crop(file_list,xml_file,image_path,output_folder):
    new_data = []
    for file in file_list:
        # print(file)
        i = file.split(".")[0]
        tree = ET.parse(f'{xml_file}/{i}.xml')
        jpg_path = f'{image_path}/{i}.jpg'
        tif_path = f'{image_path}/{i}.tif'

        if os.path.exists(jpg_path):
            img_path = jpg_path
            # print(f"使用 JPG 文件路径: {img_path}")
        elif os.path.exists(tif_path):
            img_path = tif_path
            # print(f"使用 TIF 文件路径: {img_path}")
        else:
            print("未找到 JPG 或 TIF 文件")
        # img_path = f'{image_files}/{i}.jpg'
        image = Image.open(img_path)
        root = tree.getroot()

        # 查找所有的 <points> 标签
        object_list = root.findall('./object')
        # print(len(object_list))
        # 遍历每个 <points> 标签
        with Image.open(img_path) as img:
            for idx,object in enumerate(object_list):
                data = {
                    "classes":"",
                    "filename":"",
                }
                name = object.findall('.//name')[0].text
                name = "_".join(name.split())
                
                # 获取 <points> 标签的文本内容
                xmin = float(object.findall('.//xmin')[0].text)
                ymin = float(object.findall('.//ymin')[0].text)
                xmax = float(object.findall('.//xmax')[0].text)
                ymax = float(object.findall('.//ymax')[0].text)

                x1, y1 = xmin,ymin
                x2, y2 = xmin,ymax
                x3, y3 = xmax,ymax
                x4, y4 = xmax,ymin
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                # 构造新文件名
                output_name = f"{os.path.splitext(file)[0]}_cropped_{idx}_{name}.png"
                output_path = os.path.join(output_folder, output_name)
                cropped_img.save(output_path)
                # print(f"Saved cropped image: {output_path}")
                data["classes"] = name
                data['filename'] = output_name
                new_data.append(data)
    return new_data

# # # query_crop
# xml_folder_path = "/data5/laiping/tianzhibei/demo/query/labelxml_ratote"
# output_folder_rotate = "/data5/laiping/tianzhibei/demo/query_crop"
# # 检查文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder_rotate):
#     os.makedirs(output_folder_rotate)
#     print(f"文件夹 {output_folder_rotate} 创建成功")
# else:
#     print(f"文件夹 {output_folder_rotate} 已经存在")

# image_folder_path = "/data5/laiping/tianzhibei/demo/query/images"
# file_list_rotate = get_file_names(xml_folder_path)
# new_dat = rotate_crop(file_list_rotate,xml_folder_path,image_folder_path,output_folder_rotate)
# with open(f"{output_folder_rotate}_rotate.json",'w') as f:
#     json.dump(new_dat,f)



# xml_folder_path_std = "/data5/laiping/tianzhibei/demo/query/labelxml_std"
# output_folder_std = "/data5/laiping/tianzhibei/demo/query_crop"
# image_folder_path = "/data5/laiping/tianzhibei/demo/query/images"

# file_list_std = get_file_names(xml_folder_path_std)
# # 检查文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder_std):
#     os.makedirs(output_folder_std)
#     print(f"文件夹 {output_folder_std} 创建成功")
# else:
#     print(f"文件夹 {output_folder_std} 已经存在")
# new_data = std_crop(file_list_std,xml_folder_path_std,image_folder_path,output_folder_std)
# with open(f"{output_folder_std}_std.json",'w') as f:
#     json.dump(new_data,f)

# xml_folder_path = "/data5/laiping/tianzhibei/demo/output_path/demo-test"
# output_folder_rotate = "/data5/laiping/tianzhibei/demo/show_data/gallery_crop"
# # 检查文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder_rotate):
#     os.makedirs(output_folder_rotate)
#     print(f"文件夹 {output_folder_rotate} 创建成功")
# else:
#     print(f"文件夹 {output_folder_rotate} 已经存在")

# image_folder_path = "/data5/laiping/tianzhibei/demo/gallery/images"
# file_list_rotate = get_file_names(xml_folder_path)
# new_data =rotate_crop(file_list_rotate,xml_folder_path,image_folder_path,output_folder_rotate)
# with open(f"{output_folder_rotate}.json",'w') as f:
#     json.dump(new_data,f)


# # gallery_gt_crop
# xml_folder_path = "/data5/laiping/tianzhibei/demo/gallery/gt_rotate"
# output_folder_rotate = "/data5/laiping/tianzhibei/demo/show_data/gallery_gt_crop"
# # 检查文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder_rotate):
#     os.makedirs(output_folder_rotate)
#     print(f"文件夹 {output_folder_rotate} 创建成功")
# else:
#     print(f"文件夹 {output_folder_rotate} 已经存在")

# image_folder_path = "/data5/laiping/tianzhibei/demo/gallery/images"
# file_list_rotate = get_file_names(xml_folder_path)
# new_data = rotate_crop(file_list_rotate,xml_folder_path,image_folder_path,output_folder_rotate)
# with open(f"{output_folder_rotate}_label_filename.json",'w') as f:
#     json.dump(new_data,f)


# xml_folder_path_std = "/data5/laiping/tianzhibei/demo/gallery/gt_std"
# output_folder_std = "/data5/laiping/tianzhibei/demo/show_data/gallery_gt_crop"
# image_folder_path = "/data5/laiping/tianzhibei/demo/gallery/images"

# file_list_std = get_file_names(xml_folder_path_std)
# # 检查文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder_std):
#     os.makedirs(output_folder_std)
#     print(f"文件夹 {output_folder_std} 创建成功")
# else:
#     print(f"文件夹 {output_folder_std} 已经存在")
# # std_crop(file_list_std,xml_folder_path_std,image_folder_path,output_folder_std)

# new_data = std_crop(file_list_std,xml_folder_path_std,image_folder_path,output_folder_std)
# with open(f"{output_folder_std}_std_label_filename.json",'w') as f:
#     json.dump(new_data,f)