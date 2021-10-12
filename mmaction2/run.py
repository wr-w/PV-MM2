# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import decord
import numpy as np
import torch
from mmcv import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from modelcl import BaseModelInspector

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size per GPU/CPU for test')
    parser.add_argument(
        '--batch_num', type=int, default=10, help='Batch number per GPU/CPU for test')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=device)

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    benchmark=BaseModelInspector(args.video,args.batch_num,args.batch_size,args.label,model)
    benchmark.run_model()

if __name__ == '__main__':
    main()
