import torch
import torch.nn as nn

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector 
from mmrotate.utils import compat_cfg, setup_multi_processes
from mmrotate.apis import inference_detector_by_patches
from tools import add_xml
# from tools.data.dota.split import img_split_new
from tools.data.fair import img_split_new_auto
import glob
from tools import filter

class ObjectRetrievalSystem:
    def __init__(self, model, retrieval_database):
        self.model = model  # 目标检测模型
        self.retrieval_database = retrieval_database  # 检索数据库，包含图像

    def detect_objects(self, image):
        # 在图像中检测对象
        # 返回检测到的对象的特征和位置信息
        detected_objects = self.model(image)  # 假设返回的是特征和位置
        return detected_objects

    def measure_distance(self, target_feature, detected_objects):
        # 计算目标特征与检测到的对象特征之间的距离
        distances = []
        for obj in detected_objects:
            obj_feature, obj_position = obj  # 假设返回对象特征和位置
            distance = self.calculate_distance(target_feature, obj_feature)
            distances.append((distance, obj_position))
        return distances

    def calculate_distance(self, feature1, feature2):
        # 计算两个特征之间的距离，这里使用欧氏距离
        return torch.norm(feature1 - feature2).item()

    def rank_objects(self, distances, top_k=5):
        # 按距离对对象进行排序并返回排名前K的对象位置
        sorted_distances = sorted(distances, key=lambda x: x[0])
        top_objects = sorted_distances[:top_k]
        return [obj[1] for obj in top_objects]  # 返回位置

    def retrieve_objects(self, target_image, top_k=5):
        # 给定检索目标图像，返回排名前K的对象位置
        target_feature = self.model(target_image)  # 获取目标特征
        all_positions = []

        for image in self.retrieval_database:
            detected_objects = self.detect_objects(image)
            distances = self.measure_distance(target_feature, detected_objects)
            top_positions = self.rank_objects(distances, top_k)
            all_positions.extend(top_positions)

        return all_positions  # 返回所有检索到的对象位置

def get_file_names(folder_path,suffix):
    file_names = glob.glob(os.path.join(folder_path, f'*.{suffix}'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_names if os.path.isfile(file)]
    return file_names


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--checkpoint', help='checkpoint file')
    
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def fairv2_test(args):
    args.config = "/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90.py"
    args.checkpoint  = "/data5/laiping/tianzhibei/checkpoint/lsknet_s_fair_epoch12.pth"
    args.format_only = True
    # args.eval_options = {"submission_dir":"/data5/laiping/tianzhibei/work_dirs/fair1m_v2"}
    save_dirs = "/data5/laiping/tianzhibei/data/DIOR/test/"

    # img_split_new_auto.main(args.input_path,save_dirs)
    print(args)


    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    cfg.data.test['img_prefix'] = "/data5/laiping/tianzhibei/show_data/dior_crop"
    cfg.data.test['ann_file'] = "/data5/laiping/tianzhibei/show_data/dior_crop"
    if args.format_only and cfg.mp_start_method != 'spawn':
        warnings.warn(
            '`mp_start_method` in `cfg` is set to `spawn` to use CUDA '
            'with multiprocessing when formatting output result.')
        cfg.mp_start_method = 'spawn'

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    # gallery = []
    # for i, data in enumerate(data_loader):
    #     img_metas = data['img_metas'][0].data[0]  # 提取 img_metas
    #     images = data['img'][0].data[0]  # 提取 img
    #     # 移动到设备
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     images = images.to(device)  # 移动图像到设备
    #     with torch.no_grad():  # 不需要计算梯度
    #         # features = model(images, img_metas)
    #         gallery.append(images)

    # gallery = torch.cat(gallery, 0)
    retrieval_database = get_file_names("/data5/laiping/tianzhibei/show_data/dior_crop","png")
    retrieval_database_list = []

    for img_file in retrieval_database:
        img_file_path = os.path.join("/data5/laiping/tianzhibei/show_data/dior_crop",img_file,'.png')
        retrieval_database_list.append(img_file_path)
    
    # retrieval_system = ObjectRetrievalSystem(model, retrieval_database_list)
    target_image = "/data5/laiping/tianzhibei/show_data/dior_true_crop/00008_cropped_0.png"
    img = mmcv.imread(target_image)
    result = inference_detector_by_patches(model,cfg,img,sizes=[14,14],
                                  steps=[7,7],
                                  ratios=1.0,
                                  merge_iou_thr=0.5)

    # top_object_positions = retrieval_system.retrieve_objects(target_image, top_k=5)


def main():
    args = parse_args()
    fairv2_test(args)


if __name__ == '__main__':
    # # 运行你的代码
    main()