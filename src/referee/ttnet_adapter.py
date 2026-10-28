# -*- coding: utf-8 -*-
"""
TTNet 适配器：把 TTNet 的多任务原始输出转换成统一的逐帧 Observation。

TTNet 专有信息（相对 YOLO / TrackNet 的增量）：
1. 自带事件分数 bounce / net_hit —— 无需纯靠轨迹推断；
2. 自带语义分割 —— 可自动标定球台 ROI 与记分牌，用于几何校验和侧别判定；
3. 两级球检测 global / local —— 精度更高，但 local 强依赖 global（级联脆弱）。

本适配器处理的几个关键工程问题
--------------------------------
A. 解码占位 bug
   原 `get_prediction_ball_pos_right_test` 在「热图全部低于阈值」时，argmax 仍返回
   索引 0，于是丢失的球被解码成 (0, 0) —— 映射回原图就是左上角。这正是「球乱飞
   到屏幕左上角」的直接来源。此处先检查峰值概率，低于阈值一律判定 visible=False。

B. 输出对应哪一帧
   TTNet 输入 9 帧（[t-8, t]），预测的是**最后一帧** t，不是中间帧。
   训练时 `last_f_idx = smooth_idx + num_frames_from_event` 可证。
   因此时间戳必须绑定窗口末帧，否则所有事件会系统性提前约 4 帧（120fps 下约 33ms）。

C. 坐标还原
   local 阶段是在以 global 预测点为中心、320x128 的裁剪窗口上做精定位，因此
   原始坐标 = global_x * w_ratio - crop_w/2 + local_x。两分支都可用时取 local 精修值。

D. 置信度语义
   TTNet 的球位置输出是 x、y 两条**独立的一维概率分布**（sigmoid），不是 2D 热图。
   峰值 1.0 表示「球在该列/该行的概率」。故置信度取两个峰值中的较小者（保守），
   与 YOLO 的检测框置信度、TrackNet 的热图分数**含义不同，不能直接跨模型比较**。
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.unified_types import (
    BallObservation, Observation, Side,
)

# TTNet 固定规格
W_ORG, H_ORG = 1920, 1080
W_IN, H_IN = 320, 128
NUM_FRAMES = 9
W_RATIO = W_ORG / W_IN      # 6.0
H_RATIO = H_ORG / H_IN      # 8.4375

# 分割通道语义。
#
# 注意：这里容易踩坑。标注文件（cv2.imread 直接读）的通道是
#     ch0=记分牌, ch1=运动员, ch2=球台
# 但训练时 `load_raw_img` 做了 cv2.cvtColor(BGR2RGB)，等于把通道顺序反转，
# 因此**模型输出**的通道顺序是
#     ch0=球台, ch1=运动员, ch2=记分牌
# 已对模型实际输出逐一可视化 + bbox 几何核验确认：
#     ch0 bbox x[60,247] y[60,78]  → 宽扁横条，球台
#     ch1 bbox x[0,290]  y[32,106] → 覆盖两侧，运动员
#     ch2 bbox x[160,168] y[46,50] → 极小且居中，记分牌
SEG_CHANNELS = {0: 'table', 1: 'player', 2: 'scoreboard'}
SEG_CHANNELS_TABLE = 0
SEG_CHANNELS_PLAYER = 1


class TTNetAdapter:
    """把 TTNet 包装成「逐帧 Observation 生成器」。

    用法::

        adapter = TTNetAdapter(configs, '../checkpoints/.../ttnet_3rd_phase_epoch_30.pth')
        for obs in adapter.observations_from_video('test_2.mp4'):
            ...
    """

    def __init__(self, pretrained_path, device='cpu',
                 ball_thresh=0.05, seg_thresh=0.5,
                 decode='argmax', compute_seg=True,
                 min_confidence=0.25, stuck_patience=3):
        """
        :param pretrained_path: 权重路径（本项目导出的 ttnet_model.pth，
            内含 tasks / input_size / num_frames_sequence 等结构元信息，
            因此不再需要外部传 configs —— 边缘板上无需构造 argparse 配置）
        :param device: 'cpu'（边缘板）/ 'cuda:0'（开发机）/ 'npu:0' 等
        :param ball_thresh: 球位置概率阈值，低于此值视为不可见
        :param seg_thresh: 分割阈值
        :param decode: 'argmax'（与原实现一致）或 'centroid'（亚像素，更平滑）
        :param compute_seg: 是否保留分割结果（判罚需要球台 ROI 时必须为 True）
        :param min_confidence: 置信度下限。实测（test_2，415 个有标注帧）：
            误差 >25px 的帧置信度全部 <=0.204，而准确帧 P5=0.254。
            取 0.25 可过滤掉全部 5 个离群帧，仅牺牲 4.68% 的准确帧。
            对裁判场景而言，「宁可不报」远优于「报错位置」。
        :param stuck_patience: 连续多少帧预测出完全相同的坐标就判为失效。
            global 检测失败时，local 会退化为「裁剪图像中心」，表现为坐标冻结
            在 (300.0, ...)（原图 x≈300）。这类帧必须丢弃。
        """
        from models.model_utils import build_ttnet

        self.pretrained_path = pretrained_path
        self.device = torch.device(device)
        self.ball_thresh = ball_thresh
        self.seg_thresh = seg_thresh
        self.decode = decode
        self.compute_seg = compute_seg
        self.min_confidence = min_confidence
        self.stuck_patience = stuck_patience
        self.source = 'ttnet'

        # 失效检测状态
        self._last_center = None
        self._stuck_count = 0
        self.frozen_frames_dropped = 0
        self.low_conf_dropped = 0

        meta = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        self.meta = meta
        self.model = build_ttnet(meta, device=str(device))

    # ------------------------------------------------------------ 解码
    def _decode_1d(self, prob):
        """把一条一维概率分布解码成 (坐标, 峰值概率)。

        注意：峰值为 0 时返回 None，而不是 0 —— 这是修复占位 (0,0) 的关键。
        """
        prob = prob.detach().float().cpu().numpy().ravel()
        peak = float(prob.max())
        if peak < self.ball_thresh:
            return None, peak

        if self.decode == 'centroid':
            # 只取峰值附近 ±3 个 bin，避免背景小概率把重心拉偏
            idx = int(prob.argmax())
            lo, hi = max(0, idx - 3), min(len(prob), idx + 4)
            window = prob[lo:hi]
            if window.sum() <= 0:
                return float(idx), peak
            coords = np.arange(lo, hi, dtype=np.float32)
            # 阈值截断后加权，抑制长尾
            weights = np.clip(window - self.ball_thresh, 0, None)
            if weights.sum() <= 0:
                return float(idx), peak
            return float((coords * weights).sum() / weights.sum()), peak

        return float(int(prob.argmax())), peak

    def _decode_ball_stage(self, pred_ball):
        """解码一个球检测阶段（global 或 local）的输出。

        :param pred_ball: ((320,), (128,)) 的 x/y 概率分布
        :return: (x, y, confidence) ，未检出的分量为 None
        """
        prob_x, prob_y = pred_ball[0].squeeze(0), pred_ball[1].squeeze(0)
        x, px = self._decode_1d(prob_x)
        y, py = self._decode_1d(prob_y)
        if x is None or y is None:
            return None, None, None
        # 两条分布独立，取较小峰值作为保守置信度
        return x, y, min(px, py)

    # ------------------------------------------------------------ 主推理
    def infer(self, seq_chw):
        """对单个 9 帧窗口推理，返回原始输出。

        :param seq_chw: (27, 128, 320) numpy
        """
        tensor = torch.from_numpy(seq_chw).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            pred_ball_global, pred_ball_local, pred_events, pred_seg = self.model.run_demo(tensor)
        return pred_ball_global, pred_ball_local, pred_events, pred_seg

    def build_observation(self, frame_id, fps, pred_ball_global, pred_ball_local,
                          pred_events, pred_seg):
        """把一帧的原始输出组装成统一 Observation。"""
        gx, gy, g_conf = self._decode_ball_stage(pred_ball_global)
        lx, ly, l_conf = self._decode_ball_stage(pred_ball_local)

        # ---- 坐标还原到原始分辨率
        center = None
        confidence = None
        if gx is not None:
            # global 单独可用时：直接按比例放大
            ox = gx * W_RATIO
            oy = gy * H_RATIO
            confidence = g_conf
            if lx is not None:
                # local 是在以 global 为中心的裁剪窗口上精定位，需减去半窗偏移
                ox = gx * W_RATIO + lx - W_IN / 2.0
                oy = gy * H_RATIO + ly - H_IN / 2.0
                # 两级都可信时取更保守的
                confidence = min(g_conf, l_conf)
            center = [float(np.clip(ox, 0, W_ORG - 1)), float(np.clip(oy, 0, H_ORG - 1))]
            center, confidence = self._guard_invalid(center, confidence)

        visible = center is not None
        ball = BallObservation(visible=visible, center_px=center,
                               bbox_xyxy=None, confidence=confidence)

        # ---- 事件分数（bounce, net_hit），顺序由 configs.events_dict 决定
        ev = pred_events.detach().float().cpu().numpy().ravel()
        event_scores = {'bounce': float(ev[0]), 'net_hit': float(ev[1])}

        # ---- 分割扩展信息：球台几何用于侧别判定与落点校验
        extras = {}
        if self.compute_seg and pred_seg is not None:
            seg = pred_seg.detach().float().cpu().numpy().squeeze(0)  # (3,128,320)
            extras['table_box'] = self._table_box(seg[SEG_CHANNELS_TABLE])
            extras['table_center_x'] = extras['table_box'][0] + extras['table_box'][2] / 2.0 \
                if extras['table_box'] else None

        return Observation(
            frame_id=frame_id,
            timestamp_s=frame_id / float(fps),
            source=self.source,
            image_size=[W_ORG, H_ORG],
            ball=ball,
            event_scores=event_scores,
            extras=extras,
        )

    def _guard_invalid(self, center, confidence):
        """剔除「低置信度」与「坐标冻结」这两类已知失效模式。

        失效机理：global 阶段定位失败时，`__crop_original_batch_right__` 会退化为
        裁剪图像中心，于是 local 阶段得到的相对坐标恒定，最终输出坐标被冻结在
        一个固定值上（实测原图 x≈300）。这种帧会直接导致轨迹「乱飞」，
        必须在适配器层拦掉，而不是交给下游猜测。
        """
        if confidence is None or confidence < self.min_confidence:
            self.low_conf_dropped += 1
            self._reset_stuck()
            return None, None

        if self._last_center is not None and \
                abs(center[0] - self._last_center[0]) < 0.5 and \
                abs(center[1] - self._last_center[1]) < 0.5:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._last_center = center

        if self._stuck_count >= self.stuck_patience:
            self.frozen_frames_dropped += 1
            return None, None
        return center, confidence

    def _reset_stuck(self):
        self._last_center = None
        self._stuck_count = 0

    def _table_box(self, table_prob):
        """从球台分割概率图取外接框（原始分辨率坐标）。

        球台掩码是判罚的关键几何依据：用于判断落点是否在台内、以及球在台面哪一侧。
        """
        mask = (table_prob > self.seg_thresh).astype(np.uint8)
        if mask.sum() < 50:
            return None
        ys, xs = np.where(mask > 0)
        x0 = float(xs.min()) * W_RATIO
        x1 = float(xs.max()) * W_RATIO
        y0 = float(ys.min()) * H_RATIO
        y1 = float(ys.max()) * H_RATIO
        return [x0, y0, x1 - x0, y1 - y0]

    # ------------------------------------------------------------ 视频流
    def observations_from_video(self, video_path, max_frames=None, progress_every=200,
                                infer_stride=1):
        """流式读取视频，逐帧产出 Observation。

        窗口为 [t-8, t]，输出对应末帧 t，因此前 8 帧不产出观测。

        :param infer_stride: 推理步长，默认 1（每帧都推理）。
            边缘板上把它调到 4，表示只对每第 4 帧推理一次，算力降到 1/4。

            **注意这里刻意不用「跳帧抽帧」**：跳帧会让输入窗口里的 9 帧
            相隔 stride 倍的时间，等于把 120fps 的球速变成 480fps 的位移，
            与训练分布不符，会明显损伤 bounce/net 的判据。
            本实现始终用**连续的 9 帧**做窗口，只是降低「多久推一次」的频率，
            因此在被推理的那些帧上，模型输入与原版逐比特一致。
            代价是事件时间分辨率变粗（stride=4 即 33ms），
            相对 ±100ms 的评估容差可以接受。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('无法打开视频: {}'.format(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(infer_stride))

        frames = []
        frame_id = -1
        produced = 0
        inferred = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if max_frames is not None and frame_id >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (W_IN, H_IN)))
            if len(frames) > NUM_FRAMES:
                frames.pop(0)
            if len(frames) < NUM_FRAMES:
                continue

            # 只降低推理频率，窗口本身仍是连续 9 帧
            if stride > 1 and (frame_id % stride) != 0:
                continue

            seq = np.dstack(frames).transpose(2, 0, 1)  # (27,128,320)
            preds = self.infer(seq)
            obs = self.build_observation(frame_id, fps, *preds)
            produced += 1
            inferred += 1
            if progress_every and produced % progress_every == 0:
                print('  [ttnet] 已推理 {} 帧 (frame_id={}, stride={})'.format(
                    produced, frame_id, stride))
            yield obs

        cap.release()
