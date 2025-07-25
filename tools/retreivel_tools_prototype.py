import torch
import torch.nn as nn

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import json
import glob
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, util
from transformers import ViTImageProcessor, ViTModel
import requests

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from prettytable import PrettyTable

def get_file_names(folder_path,suffix):
    file_names = glob.glob(os.path.join(folder_path, f'*.{suffix}'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_names if os.path.isfile(file)]
    return file_names

def trans_emb(images_list,model):
    img_embs = []
    for img_file in tqdm(images_list):
        img_emb = model.encode(Image.open(img_file))
        img_embs.append(img_emb)
    
    img_embs = np.array(img_embs)

    return img_embs

def trans_feats(images_list,processor,model,batch_size=32):
    image_feats = []
    imgs = []
    for image_file in tqdm(images_list):
        try:
            image = Image.open(image_file)
            inputs = processor(images=image,return_tensors="pt")
            imgs.append(inputs['pixel_values'])
        except:
            continue
    # 将图像张量列表堆叠成一个张量
    imgs = torch.cat(imgs, dim=0).to("cuda")

    # 创建 DataLoader 以便按 batch 处理
    dataset = TensorDataset(imgs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch in tqdm(dataloader) :
        with torch.no_grad():
            outputs = model(batch[0])
        last_hidden_states = outputs.last_hidden_state.to(torch.float16)
        features = last_hidden_states[:,0,:]
        image_feats.append(features)
    image_feats = torch.cat(image_feats, 0)
    return image_feats



# 假设你已经有了 imgs 和 features，imgs 是图像张量列表，features 是从 ViT 中提取的特征
def conduct_prototype(features,labels):
    # 计算原型向量
    prototype_vectors = {}
    for i, feature in enumerate(features):
        label = labels[i]  # 假设你有相应的标签来标识每个样本的类别
        if label not in prototype_vectors:
            prototype_vectors[label] = feature
        else:
            prototype_vectors[label] += feature
    for label in prototype_vectors:
        prototype_vectors[label] /= len(prototype_vectors[label])
    return prototype_vectors

# 在测试阶段进行类别匹配和分类 距离的方式
def classify_with_prototypes(test_features, prototype_vectors,gallery_labels_gt,threshod):
    predictions = []
    for test_feature,gt_label in zip(test_features,gallery_labels_gt):
        min_distance = float('inf')
        predicted_label = None
        for label, prototype_vector in prototype_vectors.items():
            distance = torch.norm(test_feature - prototype_vector)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        # if gt_label != int(11):
        #     print(min_distance)
        if min_distance > threshod:
            predicted_label = 11
        predictions.append(predicted_label)
    return predictions

# # 在测试阶段进行类别匹配和分类 相似度的方式
# def classify_with_prototypes(test_features, prototype_vectors,gallery_labels_gt):
#     predictions = []

#     prototype_vector_list = []
#     labels = []
#     for label,prototype_vector in prototype_vectors.items():
#         prototype_vector_list.append(prototype_vector)
#         labels.append(label)
    
#     prototype_vector_list = torch.stack(prototype_vector_list)
#     labels = torch.tensor(labels)
#     similarities = test_features @ prototype_vector_list.t()
#     for sim in similarities:
#         max_indice = torch.argmax(sim)
#         max_sim = sim[max_indice]
#         if max_sim <= 1.9:
#             predicted_label = int(11)
#         else:
#             predicted_label = labels[max_indice].item()
#         predictions.append(predicted_label)

#     return predictions



def get_data(load_data,folder_path):
    file_paths = []
    labels = []
    for data in load_data:
        file_path = os.path.join(folder_path,data['filename'])
        file_paths.append(file_path)
        labels.append(data['classes'])
    return file_paths,labels

def compute(query_file_paths,gallery_file_paths,query_label,gallery_label):
    device = "cuda"
    processor = ViTImageProcessor.from_pretrained('/data5/laiping/checkpoint/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('/data5/laiping/checkpoint/vit-base-patch16-224-in21k')
    model.to("cuda")
    query_label = np.array(query_label)
    gallery_label = np.array(gallery_label)

    query_feats = trans_feats(query_file_paths,processor,model)
    torch.cuda.empty_cache()
    gallery_feats = trans_feats(gallery_file_paths,processor,model)
    torch.cuda.empty_cache()

    prototype_vectors = conduct_prototype(query_feats,query_label)
    i = 4.0
    while i < 7.1:
        predictions = classify_with_prototypes(gallery_feats, prototype_vectors,gallery_label,i)
        pred_labels = np.array(predictions)
        matches = (pred_labels == gallery_label)
        acc = np.sum(matches) / len(matches) # cumulative sum

        acc = acc.astype(float) * 100
        # print(acc)
        print(f"距离阈值为{i}, 正确率为{acc}")
        i += 0.1

    
    return acc

if __name__ == "__main__":
    query_file = "/data5/laiping/tianzhibei/datasets/FAIR1M_v2/query.json"
    gallery_file = "/data5/laiping/tianzhibei/datasets/FAIR1M_v2/gallery.json"
    with open(query_file,'r') as f:
        new_data_query = json.load(f)
    with open(gallery_file,'r') as f:
        new_data_gallery = json.load(f)
    
    query_file_paths = []
    gallery_file_paths = []
    query_label = []
    gallery_label = []
    query_imgs = "/data5/laiping/tianzhibei/datasets/FAIR1M_v2/query_crop"
    gallery_imgs = "/data5/laiping/tianzhibei/datasets/FAIR1M_v2/gallery_crop"
    for data in new_data_query:
        image_file = os.path.join(query_imgs,data['filename'])
        query_file_paths.append(image_file)
        query_label.append(data['classes'])

    # for data in new_data_gallery:
    #     image_file = os.path.join(gallery_imgs,data['filename'])
    #     gallery_file_paths.append(image_file)
    #     gallery_label.append(data['classes'])

    
    for data in new_data_gallery:
        image_file = os.path.join(gallery_imgs,data['filename'])
        gallery_file_paths.append(image_file)
        if data['classes'] > 10:
            gallery_label.append(int(11))
        else:
            gallery_label.append(data['classes'])
    compute(query_file_paths,gallery_file_paths,query_label,gallery_label)
    
