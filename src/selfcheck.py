# -*- coding: utf-8 -*-
"""部署自检：在边缘板上首次运行前先跑这个，30 秒内确认环境是否就绪。

检查项
------
1. Python / torch / numpy / cv2 / PIL 版本
2. 推理设备（CPU 还是 GPU）与线程数
3. 权重文件完整性（键数量、参数量、能否加载）
4. 单帧前向能否跑通，并测出实际推理速度
5. 视频文件规格（分辨率必须是 16:9，否则坐标会偏）

用法::

    python src/selfcheck.py
    python src/selfcheck.py --device cpu
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OK = '[ OK ]'
WARN = '[警告]'
FAIL = '[失败]'


def check_env():
    print('--- 1. 运行环境 ---')
    print('  Python      : {}'.format(sys.version.split()[0]))
    problems = 0
    for mod, name in (('torch', 'torch'), ('numpy', 'numpy'),
                      ('cv2', 'opencv'), ('PIL', 'Pillow')):
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', '?')
            print('  {:<12s}: {}'.format(name, ver))
        except ImportError:
            print('  {:<12s}: {} 未安装'.format(name, FAIL))
            problems += 1
    return problems


def check_device(requested):
    import torch
    print('\n--- 2. 推理设备 ---')
    print('  CUDA 可用   : {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('  GPU         : {}'.format(torch.cuda.get_device_name(0)))
    print('  CPU 线程数  : {}'.format(torch.get_num_threads()))
    if requested == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested
    print('  本次使用    : {}'.format(device))
    if device == 'cpu':
        print('  {} 使用 CPU 推理。板上预计 3~10 fps，'
              '建议按 README 第七节做降采样或转 RKNN。'.format(WARN))
    return device


def check_weights(path):
    import torch
    print('\n--- 3. 权重文件 ---')
    if not os.path.isfile(path):
        print('  {} 不存在: {}'.format(FAIL, path))
        return 1, None
    size_mb = os.path.getsize(path) / 1048576
    meta = torch.load(path, map_location='cpu', weights_only=False)
    need = {'state_dict', 'tasks', 'input_size', 'num_frames_sequence'}
    miss = need - set(meta)
    if miss:
        print('  {} 缺少字段: {}'.format(FAIL, miss))
        return 1, None
    sd = meta['state_dict']
    n_param = sum(v.numel() for v in sd.values())
    print('  路径        : {}'.format(path))
    print('  体积        : {:.1f} MB'.format(size_mb))
    print('  张量数      : {}'.format(len(sd)))
    print('  参数量      : {:.2f} M'.format(n_param / 1e6))
    print('  tasks       : {}'.format(meta['tasks']))
    print('  input_size  : {}'.format(meta['input_size']))
    print('  帧序列长度  : {}'.format(meta['num_frames_sequence']))
    return 0, meta


def check_forward(path, device):
    import torch
    print('\n--- 4. 单帧前向 ---')
    from models.model_utils import build_ttnet
    meta = torch.load(path, map_location='cpu', weights_only=False)
    t0 = time.time()
    model = build_ttnet(meta, device=device)
    print('  模型加载    : {:.2f}s'.format(time.time() - t0))
    nfs = meta['num_frames_sequence']
    x = torch.rand(1, nfs * 3, meta['input_size'][1], meta['input_size'][0])
    x = x.to(device)          # 输入必须与模型同设备，否则会报 dtype/device 不匹配

    # 先预热：首次前向包含 CUDA kernel 编译/内存分配，直接计时会严重高估耗时
    with torch.no_grad():
        model.run_demo(x)

    runs = 3
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            pg, pl, pe, ps = model.run_demo(x)
    dt = (time.time() - t0) / runs
    print('  单帧前向    : {:.3f}s  ({:.1f} fps)  [{} 次均值，已预热]'.format(
        dt, 1 / dt if dt > 0 else 0, runs))
    print('  输出形状    : global={} events={} seg={}'.format(
        tuple(pg[0].shape), tuple(pe.shape), tuple(ps.shape)))
    # 120fps 源视频无论怎么跑，每一帧都要推理一次；这里算「喂满 120fps
    # 输入」所需的实时倍率，>1 即意味着跟不上原始帧率。
    need_fps = 120.0
    ratio = need_fps * dt
    print('  估算吞吐    : 若按原始 120fps 逐帧推理，需 {:.1f}x 实时，当前 {:.2f}x'.format(
        ratio, 1 / ratio if ratio > 0 else 0))
    if ratio > 1:
        print('  {} 跟不上 120fps 原始帧率。解法：按 1/4 抽帧（30fps）'
              '只需 {:.2f}x，或转 RKNN/ONNX 上 NPU。'.format(WARN, 30.0 * dt))
        return 1
    return 0


def check_videos(vdir):
    import cv2
    print('\n--- 5. 视频文件 ---')
    if not os.path.isdir(vdir):
        os.makedirs(vdir, exist_ok=True)
        print('  {} videos/ 目录不存在，已自动创建'.format(WARN))
        print('       请先获取测试视频：python scripts/download_videos.py')
        return 0
    files = sorted(f for f in os.listdir(vdir) if f.lower().endswith('.mp4'))
    if not files:
        print('  {} videos/ 下没有 mp4，尚未获取测试视频'.format(WARN))
        print('       请运行：python scripts/download_videos.py')
        return 0
    problems = 0
    for f in files:
        p = os.path.join(vdir, f)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print('  {:<16s} {} 无法打开'.format(f, FAIL))
            problems += 1
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ratio_ok = abs(w / h - 16 / 9) < 0.01
        flag = OK if ratio_ok else '{} 非 16:9'.format(WARN)
        print('  {:<16s} {}x{} @{:.0f}fps  {}帧  {}'.format(
            f, w, h, fps or 0, n, flag))
        if not ratio_ok:
            problems += 1
            print('      模型按 1920x1080 固定缩放，非 16:9 会导致落点坐标偏移')
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description='TTNet 部署自检')
    ap.add_argument('--weights', default=os.path.join(_ROOT, 'checkpoints', 'ttnet_model.pth'))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--videos', default=os.path.join(_ROOT, 'videos'))
    args = ap.parse_args(argv)

    print('=' * 60)
    print('TTNet 边缘部署自检')
    print('  包根目录: {}'.format(_ROOT))
    print('=' * 60)

    problems = 0
    problems += check_env()
    if problems:
        print('\n{} 缺少依赖，请先 pip install -r requirements.txt'.format(FAIL))
        return 1

    device = check_device(args.device)
    p, meta = check_weights(args.weights)
    problems += p
    if meta is None:
        return 1

    # 性能不足算「警告」而非「问题」：功能仍然可用，只是慢
    perf_warn = check_forward(args.weights, device)
    problems += check_videos(args.videos)

    print('\n' + '=' * 60)
    if problems:
        print('自检完成：{} 个阻塞问题，见上方 [失败]'.format(problems))
        return 1
    if perf_warn:
        print('功能自检通过，但存在性能警告（见上方 [警告]）。')
        print('可直接跑通，只是达不到实时；建议按 README 第六节做降采样或转 RKNN。')
        print('  python src/infer.py --video videos/test_6.mp4 --device {}'.format(device))
        return 0
    print('自检全部通过，可以运行：')
    print('  python src/infer.py --video videos/test_6.mp4 --device {}'.format(device))
    return 0


if __name__ == '__main__':
    sys.exit(main())
