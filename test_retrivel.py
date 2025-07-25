import torch
import torch.nn as nn
import torch.nn.functional as F

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np

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
from mmcv.image import tensor2imgs

from tools import add_xml
# from tools.data.dota.split import img_split_new
from tools.data.fair import img_split_new_auto
import glob
from tools import filter
from mmrotate.models import ROTATED_DETECTORS, build_backbone, build_head, build_neck

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

def extract_feat(model,data_loader):
    model.eval()
    feats_list = []
    file_paths = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _,x = model(return_loss=False, rescale=True, **data)
        # print(x[0].size())
        # feat = F.adaptive_avg_pool2d(x[0][0], (1, 1)).view(-1)
  
        # feat = x[0][0].flatten()
        feat = x[0][0]
        feats_list.append(feat)
        file_path = data['img_metas'][0].data[0][0]['filename']
        file_paths.append(file_path)
    
    feats  = torch.stack(feats_list)
    return feats,file_paths


def fairv2_test(args):
    args.config = "/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_fair_le90_new_retrieval.py"
    args.checkpoint  = "/data5/laiping/tianzhibei/exp/LSKNet-fair1m_v2_new/epoch_7.pth"
    args.format_only = True
    # args.eval_options = {"submission_dir":"/data5/laiping/tianzhibei/work_dirs/fair1m_v2"}
    save_dirs = "/data5/laiping/tianzhibei/data/LSKNet-fair1m_v2_new/test/"

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
    # print(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    cfg = compat_cfg(cfg)
    print(cfg)
    # import ipdb;ipdb.set_trace()
    cfg.data_root = save_dirs
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

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
    if isinstance(cfg.data.query, dict):
        cfg.data.query.test_mode = True
        if 'samples_per_gpu' in cfg.data.query:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.query.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.query.pipeline = replace_ImageToTensor(
                cfg.data.query.pipeline)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    query_dataset = build_dataset(cfg.data.query)
    # import ipdb;ipdb.set_trace()
    query_data_loader = build_dataloader(query_dataset, **test_loader_cfg)


    if isinstance(cfg.data.gallery, dict):
        cfg.data.gallery.test_mode = True
        if 'samples_per_gpu' in cfg.data.gallery:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.gallery.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.gallery.pipeline = replace_ImageToTensor(
                cfg.data.gallery.pipeline)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    gallery_dataset = build_dataset(cfg.data.gallery)
    # import ipdb;ipdb.set_trace()
    gallery_data_loader = build_dataloader(gallery_dataset, **test_loader_cfg)


    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    
    q_feats,q_file = extract_feat(model,query_data_loader)
    g_feats,g_file = extract_feat(model,gallery_data_loader)
    cosine_similarity = torch.mm(q_feats, g_feats.T)
    indices = torch.argsort(cosine_similarity, dim=1, descending=True)
    for indice,similarity,query  in zip(indices,cosine_similarity,q_file):
        for i,idx in enumerate(indice[:10]):
            print(f"{query} 与图像 {g_file[idx]} 的相似度为{similarity[idx]} , 排名第{i + 1}")
        print("done!")



def main():
    args = parse_args()
    fairv2_test(args)


if __name__ == '__main__':
    # # 运行你的代码
    main()


# # backbone = build_backbone(cfg['model']['backbone'])
    # # neck = build_neck(cfg['model']['neck'])
    # # backbone = backbone.cuda()
    # # neck = neck.cuda()
    
    # import torch.nn.functional as F
    # # 假设所有特征图经过池化后均为 [256]
    # feat1 = feats_list[44]
    # file1 = file_paths[44]
    # # feat2 = feats_list[1]