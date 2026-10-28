# -*- coding: utf-8 -*-
"""TTNet 乒乓球裁判 · 边缘板推理入口。

设计约束（针对硬件边缘板）
--------------------------
1. **只用推理**：不 import 任何训练模块（losses / dataloader / sklearn /
   训练包装类），减少依赖与内存占用。
2. **不依赖命令行参数**：不调用 parse_configs()。模型结构信息全部从权重文件
   自身携带的元信息读取，因此在板上直接 `python src/infer.py --video ...` 即可，
   不会因为缺少某个 argparse 参数而崩。
3. **设备自适应**：--device 默认 auto，有 CUDA 用 cuda，否则回落到 cpu。
   RK3588 这类板子若装了厂商的 torch_npu / rknn 适配层，可显式传对应 device。

用法
----
    # 单个视频
    python src/infer.py --video videos/test_1.mp4

    # 两个视频依次跑
    python src/infer.py --video videos/test_1.mp4 videos/test_6.mp4

    # 指定设备与输出目录
    python src/infer.py --video videos/test_6.mp4 --device cpu --out results

输出
----
每个视频在 --out 下产生：
    <name>_score.json    判分明细（逐分时刻、得分方、判据、比分）
    <name>_report.txt    人类可读摘要
批量还会写一份 summary.json 汇总。
"""

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import torch  # noqa: E402

from referee.ttnet_adapter import TTNetAdapter                     # noqa: E402
from referee.event_processing import TrajectoryProcessor, EventExtractor  # noqa: E402
from referee.score_state_machine import TableTennisStateMachine    # noqa: E402
from referee.unified_types import EventType, Side                  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(_ROOT, 'checkpoints', 'ttnet_model.pth')

# 判分原因 -> 中文说明（报告里用）
REASON_ZH = {
    'two_bounces_same_side': '同侧台面连续两次弹跳（未接到）',
    'serve_first_bounce_wrong_side': '发球首跳落在接发球方半台',
}


def pick_device(requested='auto'):
    """选择推理设备。边缘板上通常没有 CUDA，自动回落到 CPU。"""
    if requested and requested != 'auto':
        return requested
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run_one(video_path, weights, out_dir, device='cpu',
            bounce_thresh=0.65, net_thresh=0.5,
            ball_thresh=0.05, min_confidence=0.25,
            max_frames=None, render_video=False, stride=1, quiet=False):
    """对单个视频跑「推理 → 事件 → 判分」，并落盘结果。"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(video_path):
        raise FileNotFoundError('视频不存在: {}'.format(video_path))
    if not os.path.isfile(weights):
        raise FileNotFoundError('权重不存在: {}'.format(weights))

    t0 = time.time()
    if not quiet:
        print('=' * 68)
        print('TTNet 裁判推理 · {}'.format(name))
        print('  视频   : {}'.format(video_path))
        print('  权重   : {}'.format(weights))
        print('  设备   : {}'.format(device))
        print('=' * 68)

    adapter = TTNetAdapter(weights, device=device,
                           ball_thresh=ball_thresh,
                           min_confidence=min_confidence)
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh)
    machine = TableTennisStateMachine(verbose=False, cumulative_mode=True)

    n_frames = n_visible = 0
    last_ts_seen = None
    events_all = []
    for obs in adapter.observations_from_video(
            video_path, max_frames=max_frames,
            progress_every=None if quiet else 500,
            infer_stride=stride):
        n_frames += 1
        last_ts_seen = obs.timestamp_s
        if obs.ball.visible:
            n_visible += 1
        traj = tracker.update(obs)
        for ev in extractor.process(obs, traj):
            events_all.append(ev)
            machine.consume(ev)
        # 必须用 tick_deep 而不是 tick：
        # tick()      只处理「回合超时休眠」（不判分）
        # tick_deep() 先跑 _check_rally_break()，才包含「落台序列中断 -> 判分」
        # 用错会导致整场只靠「同侧连弹」判分，漏掉约 2/3 的分。
        machine.tick_deep(obs.timestamp_s)

    for ev in extractor.flush():
        events_all.append(ev)
        machine.consume(ev)

    # 视频末尾收尾：最后一个回合没有后续时间窗口，必须显式结算，
    # 否则末尾的落台永远无法判分（原项目实测 test_2 末 8 个 bounce 全丢）。
    if last_ts_seen is not None:
        machine.finalize(last_ts_seen)

    sb = machine.scoreboard
    elapsed = time.time() - t0

    result = {
        'video': os.path.basename(video_path),
        'frames_processed': n_frames,
        'ball_visible_frames': n_visible,
        'ball_visible_rate': round(n_visible / n_frames, 4) if n_frames else 0.0,
        'score': {'A': sb['score'][Side.A], 'B': sb['score'][Side.B]},
        'total_points': len(machine.points),
        'points': machine.points,
        'needs_review': machine.reviews,
        'suppressed_by_guard': machine.suppressed_count,
        'event_counts': _count(events_all),
        'elapsed_s': round(elapsed, 1),
        'device': device,
        'weights': os.path.basename(weights),
        'infer_stride': stride,
        'params': {
            'bounce_thresh': bounce_thresh,
            'net_thresh': net_thresh,
            'ball_thresh': ball_thresh,
            'min_confidence': min_confidence,
        },
    }

    score_json = os.path.join(out_dir, '{}_score.json'.format(name))
    with open(score_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    report_txt = os.path.join(out_dir, '{}_report.txt'.format(name))
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(_format_report(result))

    if not quiet:
        print(_format_report(result))
        print('结果已写入: {}'.format(score_json))

    if render_video:
        try:
            out_mp4 = os.path.join(out_dir, '{}_score.mp4'.format(name))
            _render(video_path, out_mp4, machine, events_all, adapter)
            result['rendered_video'] = os.path.basename(out_mp4)
            if not quiet:
                print('比分视频: {}'.format(out_mp4))
        except Exception as exc:      # 渲染失败不应影响判分结果
            print('[警告] 渲染失败，判分结果已保留: {}'.format(exc))

    result['_paths'] = {'score_json': score_json, 'report': report_txt}
    return result


def _count(events):
    counts = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    return counts


def _format_report(r):
    lines = []
    lines.append('')
    lines.append('-' * 68)
    lines.append('视频           : {}'.format(r['video']))
    lines.append('处理帧数       : {}'.format(r['frames_processed']))
    lines.append('球可见帧数     : {} ({:.1%})'.format(
        r['ball_visible_frames'], r['ball_visible_rate']))
    lines.append('事件统计       : {}'.format(
        ', '.join('{}={}'.format(k, v) for k, v in r['event_counts'].items()) or '无'))
    lines.append('-' * 68)
    lines.append('最终比分       : A {} : {} B   （共判定 {} 分）'.format(
        r['score']['A'], r['score']['B'], r['total_points']))
    lines.append('-' * 68)
    if r['points']:
        lines.append('判分明细：')
        lines.append('  {:>9s}  {:^6s}  {:>6s}  {}'.format('时刻', '得分方', '比分', '判据'))
        for p in r['points']:
            reason = REASON_ZH.get(p['reason'], p['reason'])
            lines.append('  {:>8.3f}s  {:^8s}  {:>6s}  {}'.format(
                p['ts'], p['winner'], p['score'], reason))
    else:
        lines.append('判分明细：本视频未判定任何得分')
    if r['needs_review']:
        lines.append('')
        lines.append('待人工确认   : {} 处（证据不足，未计分）'.format(len(r['needs_review'])))
    if r['suppressed_by_guard']:
        lines.append('护栏拦截     : {} 次（两次判分间隔过短，判为误判）'.format(
            r['suppressed_by_guard']))
    lines.append('')
    lines.append('耗时 {:.1f}s | 设备 {} | 权重 {}'.format(
        r['elapsed_s'], r['device'], r['weights']))
    lines.append('-' * 68)
    return '\n'.join(lines) + '\n'


def _render(video_path, out_path, machine, events_all, adapter):
    """可选：把比分叠加回视频。边缘板上若嫌慢可不开 --render。"""
    from referee.scoreboard_ui import ScoreboardUI

    # 复用判分轨迹：按事件时刻重建「帧号 -> 比分」
    evs = sorted([e for e in events_all if e.type != EventType.BALL_LOST],
                 key=lambda e: (e.timestamp_s, e.type))
    from referee.unified_types import Event as OldEvent
    m2 = TableTennisStateMachine(verbose=False, cumulative_mode=True)
    timeline = []
    idx = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    snapshots = {}
    last_ts = evs[-1].timestamp_s if evs else 0.0
    t = 0.0
    fid = 0
    while t <= last_ts:
        while idx < len(evs) and evs[idx].timestamp_s <= t:
            e = evs[idx]
            m2.consume(e)
            if e.type in (EventType.BOUNCE, EventType.NET_HIT):
                side = ' · {} 侧'.format(e.side) if e.side in (Side.A, Side.B) else ''
                timeline.append((e.timestamp_s,
                                 ('落台' if e.type == EventType.BOUNCE else '擦网') + side))
            idx += 1
        m2.tick_deep(t)
        sb = m2.scoreboard
        snapshots[fid] = (sb['score'][Side.A], sb['score'][Side.B], sb['server'])
        fid += 1
        t = fid / fps

    ui = ScoreboardUI()
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cur = (0, 0, 'A')
    ei = 0
    last_ev_ts = None
    last_ev_text = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n in snapshots:
            cur = snapshots[n]
        ts = n / float(fps)
        while ei < len(timeline) and timeline[ei][0] <= ts:
            last_ev_ts, last_ev_text = timeline[ei]
            ei += 1
        age = (ts - last_ev_ts) if last_ev_ts is not None else None
        text = last_ev_text if (age is not None and age < 1.6) else None
        ui.draw(frame, score_a=cur[0], score_b=cur[1], games_a=0, games_b=0,
                server=cur[2], event_text=text, event_age=age, cumulative=True)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description='TTNet 乒乓球裁判 · 边缘推理（只推理，不训练）')
    ap.add_argument('--video', nargs='+', required=True,
                    help='一个或多个视频路径')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS, help='权重路径')
    ap.add_argument('--out', default=os.path.join(_ROOT, 'results'),
                    help='结果输出目录')
    ap.add_argument('--device', default='auto',
                    help="推理设备：auto / cpu / cuda:0 等")
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.5)
    ap.add_argument('--ball-thresh', type=float, default=0.05)
    ap.add_argument('--min-confidence', type=float, default=0.25)
    ap.add_argument('--max-frames', type=int, default=None,
                    help='只跑前 N 帧（调试用）')
    ap.add_argument('--stride', type=int, default=1,
                    help='推理步长。边缘板建议 4（120fps->30fps 输出），'
                         '算力降到 1/4，窗口仍是连续 9 帧不影响精度')
    ap.add_argument('--render', action='store_true',
                    help='额外输出叠加比分的 mp4（较慢）')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    results = []
    failed = []
    for v in args.video:
        try:
            r = run_one(v, args.weights, args.out, device=device,
                        bounce_thresh=args.bounce_thresh,
                        net_thresh=args.net_thresh,
                        ball_thresh=args.ball_thresh,
                        min_confidence=args.min_confidence,
                        max_frames=args.max_frames,
                        render_video=args.render,
                        stride=args.stride,
                        quiet=args.quiet)
            results.append(r)
        except Exception as exc:
            traceback.print_exc()
            failed.append({'video': v, 'error': str(exc)})

    if len(args.video) > 1:
        summary = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device,
            'weights': os.path.basename(args.weights),
            'results': [{k: v for k, v in r.items() if k != '_paths'}
                        for r in results],
            'failed': failed,
        }
        p = os.path.join(args.out, 'summary.json')
        os.makedirs(args.out, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print('\n' + '=' * 68)
            print('汇总（设备 {}）'.format(device))
            print('=' * 68)
            for r in results:
                print('  {:<14s} A {:>2d} : {:<2d} B   共 {:>2d} 分   球可见 {:.1%}   {:.1f}s'.format(
                    r['video'], r['score']['A'], r['score']['B'],
                    r['total_points'], r['ball_visible_rate'], r['elapsed_s']))
            for f_ in failed:
                print('  {:<14s} 失败: {}'.format(f_['video'], f_['error']))
            print('汇总已写入: {}'.format(p))
            print('=' * 68)

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
