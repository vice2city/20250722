# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmengine import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector 
from mmrotate.utils import compat_cfg, setup_multi_processes
from tools import add_xml
# from tools.data.dota.split import img_split_new
from tools.data import img_split_auto
import glob
from tools import filter
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
import pandas as pd



def get_file_names(folder_path):
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    tif_files = glob.glob(os.path.join(folder_path, '*.tif'))

    # 合并后缀为jpg或者tif的文件列表
    all_files = jpg_files + tif_files + png_files
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in all_files if os.path.isfile(file)]
    return file_names


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument("--input_path",type=str,default="")
    parser.add_argument('--output_path', type=str, default="")
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

def fairv2_test(args,filename,image,dataset_name):

    if dataset_name == "FAIR1M":        
        args.config = "/data5/laiping/tianzhibei/code/object_detection/configs/lsknet/lsk_s_fpn_1x_fair_le90.py"
        args.checkpoint  = "/data5/laiping/tianzhibei/checkpoint/lsknet_s_fair_epoch12.pth"
    elif dataset_name == "DOTA":
        args.config = "/data5/laiping/tianzhibei/code/object_detection/configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py"
        args.checkpoint  = "/data5/laiping/tianzhibei/checkpoint/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth"
    elif dataset_name == "SAR":
        args.config = "/data5/laiping/tianzhibei/code/object_detection/configs/lsknet-tianhzibei/lsk_s_ema_fpn_1x_plane_sar_le90.py"
        args.checkpoint  = "/data5/laiping/tianzhibei/checkpoint/plane_sar.pth"
    args.format_only = True
    start_time = time.time()
    img_split_auto.main(filename,image)

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
    # print(cfg)
    temp_file = "temp_file"
    
    
    cfg.data_root = temp_file
    cfg.data.test['ann_file'] = temp_file
    cfg.data.test['img_prefix'] = temp_file

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
        cfg.gpu_ids = [0]

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
    # print(dataset)
    # import ipdb;ipdb.set_trace()
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 统计参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Info] Total model parameters: {total_params / 1e6:.2f} M\n")
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
    else:
        model.CLASSES = dataset.CLASSES
    # import ipdb;ipdb.set_trace()
    if not distributed:
        model = MMDataParallel(model.cpu(), device_ids=cfg.gpu_ids)
        # 推理时间与显存统计（仅在 single_gpu 模式下可测）
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
        num_samples = len(dataset)
        avg_time = total_time / num_samples * 1000  # ms
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        print(f"\n[Info] Avg inference time per image: {avg_time:.2f} ms")
        print(f"[Info] Max GPU memory used: {max_memory:.2f} GB\n")
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    # import ipdb;ipdb.set_trace()
    outputs = filter.filter_bboxes(outputs)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            bboxes = dataset.format_results(outputs,bboxes_flag=True, **kwargs)
    params = {
        "params_M": total_params,
        "avg_infer_ms": avg_time,
        "max_mem_GB": max_memory,
        "total_time" : total_time,
    }
    # print(f"[Summary] Inference Report for {dataset_name}")
    # print(f"  - Parameters: {total_params / 1e6:.2f} M")
    # print(f"  - Total Inference Time: {total_time:.2f} s")
    # print(f"  - Avg Inference Time: {avg_time:.2f} ms")
    # print(f"  - Max GPU Memory: {max_memory:.2f} GB")

    return bboxes,params
        


def main(filename,image,dataset_name,filename_without_ext=None):
    args = parse_args()

    bboxes,params = fairv2_test(args,filename,image,dataset_name)
    # results = [[i] + box[1:] for i, box in enumerate(bboxes)]
    # results = [[i] + box[1:] for i, box in enumerate(bboxes) if 'ship' or "boat" in box[1] ]
    # results = [[i] + box[1:] for i, box in enumerate(bboxes) if box[1] == 'ship']
    # # 假设 bboxes 是类似你提供的那种列表结构，转换为 DataFrame
    # columns = ['Index', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    # bbox_df = pd.DataFrame(results, columns=columns)
    
    # # 保存为 Excel 文件
    # bbox_df.to_excel(f"/data5/laiping/tianzhibei/pretrain_datasets/目标坐标/{filename_without_ext}_{dataset_name}_confidence1e6_output_bboxes.xlsx", index=False)

    # print("Data has been saved to 'output_bboxes.xlsx'")
    avg_times = params['total_time'] / 1 * 1000  # ms
    print(f"[Summary] Inference Report for {dataset_name}")
    print(f"  - Parameters: {params['params_M'] / 1e6:.2f} M")
    print(f"  - Total Inference Time: {params['total_time']:.2f} s")
    print(f"  - Avg Inference Time: {avg_times:.2f} ms")
    print(f"  - Max GPU Memory: {params['max_mem_GB']:.2f} GB")
    return bboxes
    



if __name__ == '__main__':
    # # 运行你的代码
    color = "red"
    
    # image = Image.open('/data5/laiping/tianzhibei/pretrain_datasets/FAIR1M/test/images/0.tif')
    # main("0.png",image,"FAIR1M")
    filepath = '/data5/laiping/tianzhibei/pretrain_datasets/DOTA/test/images/P0006.png'
    
    image = Image.open(filepath)
    
    # 获取带扩展名的文件名
    filename_with_ext = os.path.basename(filepath)  # 'P0006.png'
    filename_without_ext = os.path.splitext(filename_with_ext)[0]  # 'P0006'
    
    main("0.png",image,"DOTA",filename_without_ext)

    # folderpath = "/data5/laiping/tianzhibei/pretrain_datasets/17suo"
    # file_names = get_file_names(folderpath)
    # for file in file_names:
    #     filepath = f"{folderpath}/{file}.png"
        
    #     image = Image.open(filepath)
    #     # main("0.png",image,"FAIR1M")
    #     main(f"{file}.png",image,"DOTA",file)
        # main(f"{file}.png",image,"FAIR1M",file)
        # break

