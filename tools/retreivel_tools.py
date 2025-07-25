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

def trans_feats(images_list,processor,model,batch_size=128):
    image_feats = []
    imgs = []
    # for image_file in tqdm(images_list):
    #     try:
    #         image = Image.open(image_file)
    #         inputs = processor(images=image,return_tensors="pt")
    #         imgs.append(inputs['pixel_values'])
    #     except:
    #         continue
    # # 将图像张量列表堆叠成一个张量
    # imgs = torch.cat(imgs, dim=0).to("cuda")

    # # 创建 DataLoader 以便按 batch 处理
    # dataset = TensorDataset(imgs)
    # dataloader = DataLoader(dataset, batch_size=batch_size)

    # for batch in tqdm(dataloader) :
    #     with torch.no_grad():
    #         outputs = model(batch[0])
    #         last_hidden_states = outputs.last_hidden_state.to(torch.float16)
    #         features = last_hidden_states[:,0,:]
    #         image_feats.append(features)
    # image_feats = torch.cat(image_feats, 0)


    for image_file in tqdm(images_list):
        image = Image.open(image_file)
        inputs = processor(images=image,return_tensors="pt")
        imgs.append(inputs['pixel_values'])
        imgs = torch.cat(imgs, dim=0).to("cuda")
        with torch.no_grad():
            outputs = model(imgs)
        last_hidden_states = outputs.last_hidden_state.to(torch.float16)
        features = last_hidden_states[:,0,:]


    return image_feats

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
    similarities = query_feats @ gallery_feats.t()
    torch.cuda.empty_cache()

    similarities = similarities.cpu().detach().numpy()
    indices = np.argsort(similarities, axis=1)[:, ::-1]
    pred_labels = np.take_along_axis(gallery_label[None, :], indices, axis=1)
    matches = (pred_labels == query_label[:, None])
    all_cmc = matches[:, :100].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.astype(float).mean(0) * 100
    print(all_cmc)
    table = PrettyTable(["task", "R1", "R5", "R10","R15","R20"])
    table.add_row(['t2i', all_cmc[0], all_cmc[4], all_cmc[9],all_cmc[14],all_cmc[19]])
    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R15"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R20"] = lambda f, v: f"{v:.3f}"
    print(str(table))
    return all_cmc

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
    
    for data in new_data_gallery:
        image_file = os.path.join(gallery_imgs,data['filename'])
        gallery_file_paths.append(image_file)
        gallery_label.append(data['classes'])
    compute(query_file_paths,gallery_file_paths,query_label,gallery_label)
    
