import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from tqdm import tqdm
def filter_bboxes(outputs,thr=0.5):
    new_outputs = []
    for data in tqdm(outputs):
        for i in range(len(data)):
            data = filter_low_score(data,i,thr)
        new_data = filter_zero(data)
        new_outputs.append(new_data)
    return new_outputs

def filter_zero(data):
    new_data = []
    for data1 in data:
        if len(data1) > 0:
            nonzero_indices = np.where(~np.all(data1 == 0, axis=1))[0]
            data1 = data1[nonzero_indices]
        new_data.append(data1)
    return new_data

def filter_low_score(data,start_number,thr=0.5):
    boxes = []
    for box in data:
        box = torch.tensor(box)
        boxes.append(box)
    # 生成与原始框列表相同形状的遍历结果
    result_boxes = [torch.tensor([])] * len(boxes)

    # 遍历每一对框
    for i in range(start_number,len(boxes)):
        current_boxes = boxes[i]
        if len(current_boxes) == 0:
            continue
        current_scores = current_boxes[:, 5]  # 获取当前框的得分

        # 遍历比当前框后面的框
        for j in range(i + 1, len(boxes)):
            compare_boxes = boxes[j]
            if len(compare_boxes) == 0:
                continue
            compare_scores = compare_boxes[:, 5]  # 获取比较框的得分

            if len(current_boxes) == 0 or len(compare_boxes) == 0:
                continue

            # 计算IoU
            iou = box_iou_rotated(current_boxes[:, :5], compare_boxes[:, :5],aligned=False,mode="iou")

            # 找到IoU大于0.5的索引
            iou_indices_1 = torch.nonzero(iou > thr, as_tuple=False)
            if len(iou_indices_1) > 0:
                # 找到IoU大于0.5的框对应的得分
                selected_current_scores = current_scores[iou_indices_1[:, 0]]
                selected_compare_scores = compare_scores[iou_indices_1[:, 1]]
                # 将得分低的框置为空
                labels = selected_current_scores>=selected_compare_scores
                for number,label in enumerate(labels):
                    if label == True:
                        index = [j,iou_indices_1[number][1].tolist()]
                        try:
                            data[index[0]][index[1]] = np.zeros((6,), dtype=np.float32)
                        except:
                            import ipdb;ipdb.set_trace()
                    else:
                        index = [i,iou_indices_1[number][0].tolist()]
                        try:
                            data[index[0]][index[1]] = np.zeros((6,), dtype=np.float32)
                        except:
                            import ipdb;ipdb.set_trace()
            # break
    return data
