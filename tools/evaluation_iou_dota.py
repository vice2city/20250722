import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from shapely.geometry import Polygon
from prettytable import PrettyTable




class DetectionPoint:
    def __init__(self, name, points):
        self.class_name = name
        self.points = points  # [(x1, y1), ... , (x4, y4)]
        self.min_x = min([p[0] for p in points])
        self.min_y = min([p[1] for p in points])
        self.max_x = max([p[0] for p in points])
        self.max_y = max([p[1] for p in points])


class ClassScore:
    def __init__(self, name, is_gt):
        self.class_name = name
        self.is_gt = is_gt
        self.obj_num = 0
        self.det_num = 0

    def det_precision(self):
        obj_num = self.obj_num
        det_num = self.det_num

        return det_num / obj_num


def get_points(file_path, is_gt,class_score_list):
    """从 XML 文件中获取目标的相关信息"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    obj_list = []
    object_list = root.findall('./object')
    if len(object_list) > 0:
        for obj in root.findall('./object'):
            name = "_".join(obj.findall('.//name')[0].text.split())
            # print(name)
            points_list = obj.findall('.//point')
            pos_list = []
            for points in points_list:
                points_data = points.text.split(",")
                pos_list.append((float(points_data[0]), float(points_data[1])))
            assert len(pos_list) >= 4, f'points lose: {file_path}'
            pos_list = pos_list[:4]
            obj_list.append(DetectionPoint(label, pos_list))
            if label not in class_score_list.keys():
                class_score_list[label] = ClassScore(label, is_gt)
            else:
                class_score_list[label].is_gt = class_score_list[label].is_gt or is_gt
            if is_gt:
                class_score_list[label].obj_num += 1
        if is_gt:
            return obj_list,len(root.findall('./object'))
        else:
            return obj_list
    else:
        for obj in root.findall('./objects/object'):
            name = "_".join(obj.findall('.//name')[0].text.split())
            # print(name)
            points_list = obj.findall('.//point')
            pos_list = []
            for points in points_list:
                points_data = points.text.split(",")
                pos_list.append((float(points_data[0]), float(points_data[1])))
            assert len(pos_list) >= 4, f'points lose: {file_path}'
            pos_list = pos_list[:4]
            label = name
            obj_list.append(DetectionPoint(label, pos_list))
            if label not in class_score_list.keys():
                class_score_list[label] = ClassScore(label, is_gt)
            else:
                class_score_list[label].is_gt = class_score_list[label].is_gt or is_gt
            if is_gt:
                class_score_list[label].obj_num += 1
        if is_gt:
            return obj_list,len(root.findall('./objects/object'))
        else:
            return obj_list


def print_class_scores(class_score_list):
    table = PrettyTable()
    table.field_names = ["Class", "GT Num", "Det Num", "Det Precision (%)"]

    i = 0
    sum_precision = 0
    for class_score in class_score_list.values():
        if class_score.is_gt:
            precision = class_score.det_precision() * 100 # 转换为百分比
            table.add_row([
                class_score.class_name,
                class_score.obj_num,
                class_score.det_num,
                f"{precision:.2f}%"  # 格式化为百分比形式
            ])
            i += 1
            sum_precision += precision

    print(table)
    print(f"Average Det Precision Score: {sum_precision / i:.2f}%" if i > 0 else "No GT classes found")

def calculate_iou_rotated(obj1, obj2):
    """计算两个旋转矩形框的 IoU"""
    poly1 = Polygon(obj1.points)
    poly2 = Polygon(obj2.points)

    if not poly1.intersects(poly2):
        return 0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def evaluate_image(result_path, gt_path,re_count,gt_count,class_score_list):
    # print(re_count,gt_count)
    """对比同一图像的预测结果和真实标签"""
    result_objects = get_points(result_path, False,class_score_list)
    gt_objects,gt_num = get_points(gt_path, True,class_score_list)
    gt_count += gt_num
    # 搜索所有匹配的检测框
    for r_obj in result_objects:
        for gt_obj in gt_objects:
            # 两个框的最小外接矩形不重叠，直接判定它们不相交
            if (r_obj.max_x <= gt_obj.min_x or r_obj.min_x >= gt_obj.max_x
                    or r_obj.max_y <= gt_obj.min_y or r_obj.min_y >= gt_obj.max_y):
                continue
            # 计算 IoU
            iou = calculate_iou_rotated(r_obj, gt_obj)
            # 该预测框与真实框匹配
            if iou >= 0.2:
                re_count += 1
                if r_obj.class_name == gt_obj.class_name:
                    assert gt_obj.class_name in class_score_list, f'lost class: {gt_obj.class_name}'
                    class_score_list[gt_obj.class_name].det_num += 1.0
                break

    return re_count,gt_count

def main(results_folder_path,gt_folder_path):
    gt_counts=0
    re_counts=0
    class_score_list = {}
    file_pattern = os.path.join(results_folder_path, '**', '*')
    file_list = glob.glob(file_pattern, recursive=True)
    for result_file_path in tqdm(file_list):
        img_name = os.path.splitext(os.path.basename(result_file_path))[0]
        gt_file_path = os.path.join(gt_folder_path, f"{img_name}.xml")
        assert os.path.isfile(gt_file_path), f"gt file not found: {gt_file_path}"
        re_count,gt_count = evaluate_image(result_file_path, gt_file_path,re_counts,gt_counts,class_score_list)
        re_counts = re_count
        gt_counts = gt_count
    print("re_counts:",re_counts)
    print("gt_counts:",gt_counts)
    print("precsion:",re_counts/gt_counts)
    print_class_scores(class_score_list)



results_folder_path = '/data5/laiping/tianzhibei/test-recogntion-12-19-all-dota-part-cpu/output_path'
gt_folder_path = '/data5/laiping/tianzhibei/pretrain_datasets/DOTA/val/xml'
main(results_folder_path,gt_folder_path)
