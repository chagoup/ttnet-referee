# -*- coding: utf-8 -*-
"""比分 UI 渲染 + 带 UI 的视频输出。

设计要点
--------
1. **中文必须用 Pillow 渲染**：OpenCV 的 putText 只支持 ASCII，写中文会变成 "???"。
   因此把「中文文字 + 半透明面板」先用 Pillow 画到 RGBA 图层，再贴回 OpenCV 帧。
2. **比分区左上角常驻**：裁判场景下比分是最重要信息，位置固定、字号够大。
3. **事件提示条**：落台/擦网发生时在比分区下方短暂显示，便于人工复核。
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 中文字体候选（按优先级）
FONT_CANDIDATES = [
    'C:/Windows/Fonts/msyhbd.ttc',   # 微软雅黑 Bold
    'C:/Windows/Fonts/msyh.ttc',     # 微软雅黑
    'C:/Windows/Fonts/simhei.ttf',   # 黑体
]


def load_font(size):
    """加载中文字体，失败时回退到 Pillow 默认字体（英文仍可显示）。"""
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


class ScoreboardUI:
    """在帧左上角绘制比分面板。

    面板布局（宽 560 x 高 190，坐标基于 1920x1080；其他分辨率自动缩放）：

        ┌──────────────────────────────┐
        │  第 1 局        发球 ●       │
        │                              │
        │   A          11 : 8        B │
        │                              │
        │  A=陈     B=王               │
        └──────────────────────────────┘
    """

    def __init__(self, ref_w=1920, ref_h=1080, margin=28, max_events=2):
        self.ref_w = ref_w
        self.ref_h = ref_h
        self.margin = margin
        self.max_events = max_events

        # 面板尺寸（基准 1920x1080）
        self.panel_w = 600
        self.panel_h = 230

        # 字体（基准分辨率下）
        self.f_title = load_font(30)
        self.f_score = load_font(76)
        self.f_name = load_font(26)
        self.f_event = load_font(26)

        # ---- 面板缓存
        # 比分在一次视频里只变十几次，但帧有数万。每帧重绘 Pillow 图层是最大瓶颈
        # （实测 27.5ms/帧，占比远超推理）。这里按「比分状态 + 事件文案 + 淡出档位」
        # 做缓存，绝大多数帧直接命中，渲染成本降到一次 alpha 混合。
        self._panel_cache = {}
        self._alpha_cache = {}

    def _scale(self, frame_w, frame_h):
        """返回按实际分辨率换算的缩放系数。"""
        return min(frame_w / self.ref_w, frame_h / self.ref_h)

    def draw(self, frame, score_a, score_b, games_a, games_b, server,
             event_text=None, event_age=None, names=('A 方', 'B 方'),
             cumulative=False):
        """把比分面板画到 frame 上（原地修改并返回）。

        :param score_a: A 方本局得分
        :param server: 'A' / 'B'，当前发球方
        :param event_text: 最近事件文案，如 '落台 · B 侧'
        :param event_age: 事件已过去的时间（秒），用于淡出
        :param cumulative: True 时标题显示「累计得分」而非「第 N 局」
            （短片演示用累计模式，局分始终 0:0，若显示「第 1 局」会自相矛盾）
        """
        h, w = frame.shape[:2]
        k = self._scale(w, h)
        mx = int(self.margin * k)
        my = int(self.margin * k)

        show_event = bool(event_text and event_age is not None and event_age < 1.6)
        # 事件提示条的淡出每帧都在变，因此把 alpha 量化成 16 档，让缓存能命中
        alpha_step = None
        if show_event:
            alpha_step = int(round(255 * min(1.0, max(0.0, (1.6 - event_age) / 0.5)) / 16.0)) * 16
            alpha_step = max(0, min(240, alpha_step))

        # 缓存键：比分 + 局分 + 发球方 + 事件文案 + 淡出档位
        key = (score_a, score_b, games_a, games_b, server,
               event_text if show_event else None, alpha_step,
               int(600 * k), cumulative)

        cached = self._panel_cache.get(key)
        if cached is None:
            pw_key = int(600 * k)
            layer = self._build_layer(pw_key, int(self.panel_h * k), k,
                                      score_a, score_b, games_a, games_b,
                                      server, event_text, alpha_step, names,
                                      cumulative)
            canvas = np.array(layer)                 # RGBA uint8
            alpha = canvas[:, :, 3]
            bgr = np.ascontiguousarray(canvas[:, :, :3][:, :, ::-1])
            cached = (alpha, bgr)
            self._panel_cache[key] = cached

        alpha, bgr = cached
        layer_h, layer_w = alpha.shape[:2]
        x0, y0 = mx, my
        x1 = min(w, x0 + layer_w)
        y1 = min(h, y0 + layer_h)
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            return frame

        # 只在面板覆盖的 ROI 上混合，并用整型运算。
        # 手写 float32 逐像素运算在面板区域约 10ms，uint16 整型乘加可降到 1~2ms。
        roi = frame[y0:y1, x0:x1]
        a3 = alpha[:ch, :cw, None]
        inv = 255 - a3
        out = (roi.astype(np.uint16) * inv + bgr[:ch, :cw].astype(np.uint16) * a3) // 255
        frame[y0:y1, x0:x1] = out.astype(np.uint8)
        return frame

    def _build_layer(self, pw, ph, k, score_a, score_b, games_a, games_b,
                     server, event_text, alpha_step, names, cumulative=False):
        """构建 RGBA 图层（较慢，但会被缓存复用）。

        :param alpha_step: 事件条淡出档位（0/16/.../240），None 表示无事件条
        """
        show_event = alpha_step is not None
        # 图层高度额外留出事件提示条的空间，否则提示条会被裁掉
        extra_h = int(62 * k) if show_event else 0
        layer = Image.new('RGBA', (pw, ph + extra_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        # 半透明深色圆角底
        radius = int(16 * k)
        d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=radius,
                            fill=(18, 20, 28, 205),
                            outline=(120, 200, 255, 220), width=max(1, int(2 * k)))

        # 标题行：局数（累计模式显示「累计得分」）+ 发球指示
        f_title = load_font(max(12, int(30 * k)))
        title = '累计得分' if cumulative else '第 {} 局'.format(games_a + games_b + 1)
        d.text((int(20 * k), int(12 * k)), title,
               font=f_title, fill=(170, 200, 225, 255))

        # 发球指示点：在「发球」文字右侧，A/B 用左右两个位置区分
        # 注意 dot_y 要与文字垂直居中（文字基线在 14k、字高约 24k），否则圆点会压住文字
        dot_r = int(9 * k)
        f_srv = load_font(max(11, int(24 * k)))
        srv_txt = '发球'
        sb = d.textbbox((0, 0), srv_txt, font=f_srv)
        srv_w = sb[2] - sb[0]
        txt_y = int(14 * k)
        txt_h = sb[3] - sb[1]
        # 文字整体右对齐到「圆点左侧 10k」
        dot_cx = pw - int(22 * k)
        d.ellipse([dot_cx - dot_r, txt_y + txt_h // 2 - dot_r,
                   dot_cx + dot_r, txt_y + txt_h // 2 + dot_r],
                  fill=(255, 196, 0, 255))
        d.text((dot_cx - dot_r - int(10 * k) - srv_w, txt_y), srv_txt,
               font=f_srv, fill=(255, 196, 0, 255))
        # A/B 用圆点颜色旁边的短标记区分发球方
        side_txt = 'A' if server != 'B' else 'B'
        f_side = load_font(max(10, int(20 * k)))
        sbb = d.textbbox((0, 0), side_txt, font=f_side)
        d.text((dot_cx - dot_r - int(10 * k) - srv_w - int(22 * k), txt_y + int(2 * k)),
               side_txt, font=f_side, fill=(255, 220, 120, 255))

        # ---- 2) 比分主体
        f_score = load_font(max(20, int(70 * k)))
        f_name = load_font(max(11, int(24 * k)))
        cy = int(64 * k)

        # A 方标签
        d.text((int(26 * k), cy), 'A', font=f_score,
               fill=(120, 210, 255, 255))
        # 比分 "11 : 8"
        score_txt = '{} : {}'.format(score_a, score_b)
        bbox = d.textbbox((0, 0), score_txt, font=f_score)
        tw = bbox[2] - bbox[0]
        d.text(((pw - tw) / 2, cy), score_txt, font=f_score,
               fill=(255, 255, 255, 255))
        # B 方标签
        bbox_b = d.textbbox((0, 0), 'B', font=f_score)
        d.text((pw - int(26 * k) - (bbox_b[2] - bbox_b[0]), cy), 'B',
               font=f_score, fill=(255, 170, 170, 255))

        # 底部信息行：队名（左/右） + 局分（居中，仅非累计模式显示）
        info_y = int(176 * k)
        d.text((int(26 * k), info_y), names[0], font=f_name,
               fill=(160, 190, 210, 255))
        nb = d.textbbox((0, 0), names[1], font=f_name)
        d.text((pw - int(26 * k) - (nb[2] - nb[0]), info_y), names[1],
               font=f_name, fill=(200, 175, 175, 255))

        if not cumulative:
            # 累计模式下局分恒为 0:0，显示出来只会造成困惑，故隐藏
            f_games = load_font(max(10, int(21 * k)))
            games_txt = '局分 {} : {}'.format(games_a, games_b)
            gb = d.textbbox((0, 0), games_txt, font=f_games)
            d.text(((pw - (gb[2] - gb[0])) / 2, info_y + int(3 * k)), games_txt,
                   font=f_games, fill=(135, 160, 180, 255))

        # ---- 3) 事件提示条
        if show_event:
            alpha = alpha_step
            f_ev = load_font(max(12, int(28 * k)))
            eb = d.textbbox((0, 0), event_text, font=f_ev)
            ew = (eb[2] - eb[0]) + int(34 * k)
            eh = int(48 * k)
            ey = ph + int(8 * k)
            d.rounded_rectangle([0, ey, ew, ey + eh], radius=int(10 * k),
                                fill=(210, 80, 60, min(alpha, 220)))
            d.text((int(16 * k), ey + int(9 * k)), event_text, font=f_ev,
                   fill=(255, 255, 255, alpha))

        return layer


def render_video_with_ui(frame_iter, out_path, fps, ui, on_frame=None):
    """把「带比分信息的帧生成器」写成 mp4。

    :param frame_iter: 迭代产出 (frame_bgr, info_dict) 的生成器
    :param on_frame: 每帧回调，用于进度打印
    """
    writer = None
    n = 0
    for frame, info in frame_iter:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (w, h))
        ui.draw(frame,
                score_a=info['score_a'], score_b=info['score_b'],
                games_a=info['games_a'], games_b=info['games_b'],
                server=info['server'],
                event_text=info.get('event_text'),
                event_age=info.get('event_age'))
        writer.write(frame)
        n += 1
        if on_frame is not None:
            on_frame(n)
    if writer is not None:
        writer.release()
    return n
