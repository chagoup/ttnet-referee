"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.18
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: utils functions that use for model

部署说明（边缘推理包）
----------------------
本文件是从原项目裁剪而来，**只保留推理所需**的函数。
已移除 create_model() 中对 Unbalance_Loss_Model / Multi_Task_Learning_Model 的依赖，
因为这两个训练包装类会把 losses、sklearn（经 data_process）等仅训练用的重依赖
一并拉进来，在边缘板上既无用又增加部署体积。

推理时请直接构造 models.TTNet.TTNet 并调用其 run_demo()，见 src/infer.py。
"""

import sys
import os

import torch

sys.path.append('../')

from models.TTNet import TTNet


def build_ttnet(meta, device='cpu'):
    """按权重文件里附带的元信息构造 TTNet（不经过训练包装类）。

    :param meta: torch.load 得到的 dict，需含 tasks / input_size /
                 num_frames_sequence / dropout_p / thresh_ball_pos_mask
    :param device: 'cpu' / 'cuda:0' / 'npu:0' …，跟随实际硬件
    :return: 已加载权重并切到 eval 的 TTNet
    """
    model = TTNet(dropout_p=meta.get('dropout_p', 0.5),
                  tasks=meta.get('tasks', ['global', 'local', 'event', 'seg']),
                  input_size=tuple(meta.get('input_size', (320, 128))),
                  thresh_ball_pos_mask=meta.get('thresh_ball_pos_mask', 0.05),
                  num_frames_sequence=meta.get('num_frames_sequence', 9))
    model.load_state_dict(meta['state_dict'])
    model = model.to(torch.device(device))
    model.eval()
    return model


def get_num_parameters(model):
    """统计可训练参数量（部署自检时用来核对权重是否完整）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
