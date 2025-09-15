# Copyright (c) OpenMMLab. All rights reserved.
# Written by jbwang1997
# Reference: https://github.com/jbwang1997/BboxToolkit

import argparse
import codecs
import datetime
import itertools
import json
import logging
import os
import os.path as osp
import time
from functools import partial, reduce
from math import ceil
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None


def add_parser(parser):
    """Add arguments."""
    parser.add_argument(
        '--base-json',
        type=str,
        default="",
        help='json config file for split images')
    parser.add_argument(
        '--nproc', type=int, default=10, help='the procession number')

    # argument for loading data
    parser.add_argument(
        '--img-dirs',
        nargs='+',
        type=str,
        default=None,
        help='images dirs, must give a value')
    parser.add_argument(
        '--ann-dirs',
        nargs='+',
        type=str,
        default=None,
        help='annotations dirs, optional')

    # argument for splitting image
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[1024],
        help='the sizes of sliding windows')
    parser.add_argument(
        '--gaps',
        nargs='+',
        type=int,
        default=[512],
        help='the steps of sliding widnows')
    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        default=[1.],
        help='same as DOTA devkit rate, but only change windows size')
    parser.add_argument(
        '--img-rate-thr',
        type=float,
        default=0.6,
        help='the minimal rate of image in window and window')
    parser.add_argument(
        '--iof-thr',
        type=float,
        default=0.7,
        help='the minimal iof between a object and a window')
    parser.add_argument(
        '--no-padding',
        action='store_true',
        help='not padding patches in regular size')
    parser.add_argument(
        '--padding-value',
        nargs='+',
        type=int,
        default=[0],
        help='padding value, 1 or channel number')

    # argument for saving
    parser.add_argument(
        '--save-dir',
        type=str,
        default='.',
        help='to save pkl and split images')
    parser.add_argument(
        '--save-ext',
        type=str,
        default='.png',
        help='the extension of saving images')


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting images')
    # print(parser)
    add_parser(parser)
    # print(parser)
    parser.base_json = "/data5/laiping/tianzhibei/code/object_detection/tools/data/fair/ss_test.json"
    args = parser.parse_args()
    # print(args)

    if args.base_json is not None:
        with open(args.base_json, 'r') as f:
            prior_config = json.load(f)

        for action in parser._actions:
            if action.dest not in prior_config or \
                    not hasattr(action, 'default'):
                continue
            action.default = prior_config[action.dest]
        args = parser.parse_args()

    
    # assert arguments
    assert args.img_dirs is not None, "argument img_dirs can't be None"
    assert args.ann_dirs is None or len(args.ann_dirs) == len(args.img_dirs)
    assert len(args.sizes) == len(args.gaps)
    assert len(args.sizes) == 1 or len(args.rates) == 1
    assert args.save_ext in ['.png', '.jpg', 'bmp', '.tif']
    assert args.iof_thr >= 0 and args.iof_thr < 1
    assert args.iof_thr >= 0 and args.iof_thr <= 1
    assert not osp.exists(args.save_dir), \
        f'{osp.join(args.save_dir)} already exists'
    return args


def get_sliding_window(info, sizes, gaps, img_rate_thr):
    """Get sliding windows.

    Args:
        info (dict): Dict of image's width and height.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        list[np.array]: Information of valid windows.
    """
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for size, gap in zip(sizes, gaps):
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size

        start = np.array(
            list(itertools.product(x_start, y_start)), dtype=np.int64)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]


def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


def bbox_overlaps_iof(bboxes1, bboxes2, eps=1e-6):
    """Compute bbox overlaps (iof).

    Args:
        bboxes1 (np.array): Horizontal bboxes1.
        bboxes2 (np.array): Horizontal bboxes2.
        eps (float, optional): Defaults to 1e-6.

    Returns:
        np.array: Overlaps.
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = poly2hbb(bboxes1)
    hbboxes2 = bboxes2
    hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    l, t, r, b = [bboxes2[..., i] for i in range(4)]
    polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
    if shgeo is None:
        raise ImportError('Please run "pip install shapely" '
                          'to install shapely first.')
    sg_polys1 = [shgeo.Polygon(p) for p in bboxes1.reshape(rows, -1, 2)]
    sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def get_window_obj(info, windows, iof_thr):
    """

    Args:
        info (dict): Dict of bbox annotations.
        windows (np.array): information of sliding windows.
        iof_thr (float): Threshold of overlaps between bbox and window.

    Returns:
        list[dict]: List of bbox annotations of every window.
    """
    bboxes = info['ann']['bboxes']
    iofs = bbox_overlaps_iof(bboxes, windows)

    window_anns = []
    for i in range(windows.shape[0]):
        win_iofs = iofs[:, i]
        pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

        win_ann = dict()
        for k, v in info['ann'].items():
            try:
                win_ann[k] = v[pos_inds]
            except TypeError:
                win_ann[k] = [v[i] for i in pos_inds]
        win_ann['trunc'] = win_iofs[pos_inds] < 1
        window_anns.append(win_ann)
    return window_anns


# def crop_and_save_img(info, windows, window_anns, img, no_padding,
#                       padding_value, img_ext):
#     """
#     Args:
#         info (dict): Image's information.
#         windows (np.array): Information of sliding windows.
#         window_anns (list[dict]): List of bbox annotations of every window.
#         img (np.array): The image data in numpy array format.
#         no_padding (bool): If True, no padding.
#         padding_value (tuple[int|float]): Padding value.
#         img_ext (str): Picture suffix.

#     Returns:
#         list[dict]: Information of paths.
#     """
#     patch_infos = []
#     # 指定文件夹路径
#     folder_path = "/data5/laiping/tianzhibei/data/webtest-test"

#     # 检查文件夹是否存在
#     if os.path.exists(folder_path):
#         # 遍历文件夹中的所有文件
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             # 检查是否是文件
#             if os.path.isfile(file_path):
#                 # 删除文件
#                 os.remove(file_path)
#                 print(f"Deleted file: {file_path}")
#             else:
#                 print(f"Skipped non-file item: {file_path}")
#     else:
#         print(f"Folder does not exist: {folder_path}")

#     for i in range(windows.shape[0]):
#         patch_info = {k: v for k, v in info.items() if k not in ['id', 'filename', 'width', 'height', 'ann']}
#         window = windows[i]
#         x_start, y_start, x_stop, y_stop = window.tolist()

#         patch_info['x_start'] = x_start
#         patch_info['y_start'] = y_start
#         patch_info['id'] = f"0__{x_stop - x_start}___{x_start}___{y_start}"
#         # patch_info['ori_id'] = info['id']

#         ann = window_anns[i]
#         ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
#         patch_info['ann'] = ann

#         patch = img[y_start:y_stop, x_start:x_stop]
#         if not no_padding:
#             height, width = y_stop - y_start, x_stop - x_start
#             if height > patch.shape[0] or width > patch.shape[1]:
#                 padding_patch = np.empty((height, width, patch.shape[-1]), dtype=np.uint8)
#                 padding_patch[...] = padding_value
#                 padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
#                 patch = padding_patch
#         patch.save(osp.join(folder_path,patch_info['id']+img_ext))
#         patch_info['height'], patch_info['width'] = patch.shape[:2]
#         patch_info['patch'] = patch  # Store the cropped patch in the info dictionary
#         patch_infos.append(patch_info)

#     return patch_infos

def crop_and_save_img(info, windows, window_anns, img, no_padding,
                      padding_value, img_ext):
    """
    Args:
        info (dict): Image's information.
        windows (np.array): Information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img (PIL.Image.Image): The image data in PIL Image format.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.
        img_ext (str): Picture suffix.

    Returns:
        list[dict]: Information of paths.
    """
    patch_infos = []
    # 指定文件夹路径
    folder_path = "temp_file"

    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 检查是否是文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
    else:
        print(f"Folder does not exist: {folder_path}")
        os.makedirs(folder_path)  # 创建文件夹
        print(f"Folder created: {folder_path}")

    for i in range(windows.shape[0]):
        patch_info = {k: v for k, v in info.items() if k not in ['id', 'filename', 'width', 'height', 'ann']}
        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()

        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        patch_info['id'] = "0" + '__' + str(x_stop - x_start) + '__' + str(x_start) + '___' + str(y_start)
        # info['id'] + '__' + str(x_stop - x_start) + '__' + str(x_start) + '___' + str(y_start)
        # patch_info['ori_id'] = info['id']

        ann = window_anns[i]
        ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann

        # Crop the image using PIL
        patch = img.crop((x_start, y_start, x_stop, y_stop))

        if not no_padding:
            height, width = y_stop - y_start, x_stop - x_start
            if height > patch.size[1] or width > patch.size[0]:
                # Convert patch to numpy array for padding
                patch_np = np.array(patch)
                padding_patch = np.empty((height, width, patch_np.shape[-1]), dtype=np.uint8)
                padding_patch[...] = padding_value
                padding_patch[:patch_np.shape[0], :patch_np.shape[1], ...] = patch_np
                patch = Image.fromarray(padding_patch)

        # Save the cropped image
        patch.save(osp.join(folder_path, patch_info['id'] + img_ext))
        patch_info['height'], patch_info['width'] = patch.size
        patch_info['patch'] = patch  # Store the cropped patch in the info dictionary
        patch_infos.append(patch_info)

    return patch_infos



def single_split(info, sizes, gaps, img_rate_thr, iof_thr, no_padding,padding_value,img_ext,image):
    """

    Args:
        arguments (object): Parameters.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.
        iof_thr (float): Threshold of overlaps between bbox and window.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.
        lock (object): Lock of Manager.
        prog (object): Progress of Manager.
        total (object): Length of infos.
        logger (object): Logger.

    Returns:
        list[dict]: Information of paths.
    """
    # info, img_dir = arguments

    # image = np.array(image)
    windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
    window_anns = get_window_obj(info, windows, iof_thr)
    patch_infos = crop_and_save_img(info, windows, window_anns,image, no_padding, padding_value, img_ext)
    assert patch_infos

    

    return patch_infos


def setup_logger(log_path):
    """Setup logger.

    Args:
        log_path (str): Path of log.

    Returns:
        object: Logger.
    """
    logger = logging.getLogger('img split')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = osp.join(log_path, now + '.log')
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, 'w')]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def translate(bboxes, x, y):
    """Map bboxes from window coordinate back to original coordinate.

    Args:
        bboxes (np.array): bboxes with window coordinate.
        x (float): Deviation value of x-axis.
        y (float): Deviation value of y-axis

    Returns:
        np.array: bboxes with original coordinate.
    """
    dim = bboxes.shape[-1]
    translated = bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)
    return translated


def load_dota(filename,image,  nproc=10):
    """Load DOTA dataset.

    Args:
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.
        nproc (int): number of processes.

    Returns:
        list: Dataset's contents.
    """
    
    _load_func = _load_dota_single(filename,image)
    contents = []
    contents.append(_load_func)
    contents = [c for c in contents if c is not None]
   
    return contents


def _load_dota_single(imgfile, image):
    """Load DOTA's single image.

    Args:
        imgfile (str): Filename of single image.
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.

    Returns:
        dict: Content of single image.
    """
    
    size = image.size
    txtfile = None
    content = _load_dota_txt(txtfile)
  
    content.update(
        dict(width=size[0], height=size[1], filename=imgfile))
    return content


def _load_dota_txt(txtfile):
    """Load DOTA's txt annotation.

    Args:
        txtfile (str): Filename of single txt annotation.

    Returns:
        dict: Annotation of single image.
    """
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(items[8])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 8), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
        np.zeros((0,), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)

def parse_json_file(json_file):
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 读取 JSON 文件并解析为字典
    with open(json_file) as file:
        data = json.load(file)

    args = argparse.Namespace(**data)
    return args


def main(filename,image):
    # image = np.array(image)[:,:,:3]

    """Main function of image split."""
    print("Main function of image split.")
    # 解析 JSON 文件并存储为 args 变量
    json_file = '/data5/laiping/tianzhibei/code/object_detection/tools/data/fair/ss_test.json'
    args = parse_json_file(json_file)
    padding_value = args.padding_value[0] \
        if len(args.padding_value) == 1 else args.padding_value
    sizes, gaps = [], []
    for rate in args.rates:
        sizes += [int(size / rate) for size in args.sizes]
        gaps += [int(gap / rate) for gap in args.gaps]

    print('Loading original data!!!')
    _infos = load_dota(filename,image)

    print('Start splitting images!!!')
    start = time.time()
    manager = Manager()
    worker = single_split(
        info=_infos[0],
        sizes=sizes,
        gaps=gaps,
        img_rate_thr=args.img_rate_thr,
        iof_thr=args.iof_thr,
        no_padding=args.no_padding,
        padding_value=padding_value,
        img_ext=args.save_ext,
        image = image,
        )
    
    patch_infos = []
    patch_infos.append(worker)
    patch_infos = reduce(lambda x, y: x + y, patch_infos)
    stop = time.time()
    print(f'Finish splitting images in {int(stop - start)} second!!!')
    print(f'Total images number: {len(patch_infos)}')
    # return args.save_dir

if __name__ == '__main__':
    # image = cv2.imread("/data5/laiping/tianzhibei/data/webtest/0.tif")
    img = Image.open("/data5/laiping/tianzhibei/data/webtest/0.tif")  
    # img = Image.open("/data5/laiping/tianzhibei/pretrain_datasets/DOTA/train/images/P0000.png") 
    main("0.tif",img)
