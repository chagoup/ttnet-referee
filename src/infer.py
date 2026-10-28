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
            max_frames=None, render_video=False, render_scale=1.0,
            stride=1, quiet=False):
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
    # 渲染用的逐帧快照，直接在主推理循环里顺手记录，
    # 而不是渲染时再重建一遍状态机 —— 重建会有两个隐患：
    #   1) 两份结果可能不一致（视频比分 ≠ JSON 比分）；
    #   2) 重建路径容易漏调 finalize()，末尾那一分就丢了。
    snapshots = []          # 每项: (timestamp_s, score_a, score_b, server)
    timeline = []           # 每项: (timestamp_s, 事件文案)

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
            if ev.type in (EventType.BOUNCE, EventType.NET_HIT):
                suffix = ' · {} 侧'.format(ev.side) if ev.side in (Side.A, Side.B) else ''
                timeline.append((ev.timestamp_s,
                                 ('落台' if ev.type == EventType.BOUNCE else '擦网') + suffix))
        # 必须用 tick_deep 而不是 tick：
        # tick()      只处理「回合超时休眠」（不判分）
        # tick_deep() 先跑 _check_rally_break()，才包含「落台序列中断 -> 判分」
        # 用错会导致整场只靠「同侧连弹」判分，漏掉约 2/3 的分。
        machine.tick_deep(obs.timestamp_s)

        if render_video:
            s = machine.scoreboard
            snapshots.append((obs.timestamp_s, s['score'][Side.A],
                              s['score'][Side.B], s['server']))

    for ev in extractor.flush():
        events_all.append(ev)
        machine.consume(ev)

    # 视频末尾收尾：最后一个回合没有后续时间窗口，必须显式结算，
    # 否则末尾的落台永远无法判分（原项目实测 test_2 末 8 个 bounce 全丢）。
    if last_ts_seen is not None:
        machine.finalize(last_ts_seen)
        # 终局比分只补到最后一帧，绝不向前倒灌（倒灌会制造时序错位）
        if render_video and snapshots:
            s = machine.scoreboard
            snapshots[-1] = (last_ts_seen, s['score'][Side.A],
                             s['score'][Side.B], s['server'])

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
            print('[渲染] 叠加比分 UI…')
            n_rendered, codec = _render(video_path, out_mp4, snapshots, timeline,
                                        max_frames=max_frames, scale=render_scale)
            result['rendered_video'] = os.path.basename(out_mp4)
            result['rendered_frames'] = n_rendered
            result['render_codec'] = codec
            if not quiet:
                size_mb = os.path.getsize(out_mp4) / 1048576 if os.path.isfile(out_mp4) else 0
                print('比分视频: {}  ({} 帧, {} 编码, {:.1f} MB)'.format(
                    out_mp4, n_rendered, codec, size_mb))
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


def _open_writer(out_path, fps, w, h):
    """选一个体积合理的编码器创建 VideoWriter。

    实测（本机 OpenCV 5.0 / Windows，1080p 真实球赛帧，60 帧样本）：

        avc1  0.250 MB/帧   ← 缺 openh264 库，退回近无损，体积失控
        mp4v  0.027 MB/帧   ← 默认选它
        MJPG  0.128 MB/帧

    所以这里**优先 mp4v**，不是常说的 H.264。avc1 只在显式确认可用时才该用，
    否则 16800 帧的 test_1 会写出 4GB+ 的文件。
    """
    attempts = [('mp4v', 'MPEG-4'), ('avc1', 'H.264'), ('MJPG', 'Motion-JPEG')]
    for fourcc, label in attempts:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise RuntimeError('没有可用的视频编码器（试过 mp4v / avc1 / MJPG）')


def _render(video_path, out_path, snapshots, timeline, fade_s=1.6,
            progress_every=1000, max_frames=None, scale=1.0):
    """把比分 UI 叠加回原视频。

    使用的 snapshots / timeline 由主推理循环产出，因此视频里看到的比分
    与 JSON 报告**必然一致**，不存在两套逻辑各自算一遍的风险。

    :param snapshots: [(timestamp_s, score_a, score_b, server), ...] 时间升序
    :param timeline:  [(timestamp_s, 事件文案), ...] 时间升序
    :param scale: 输出缩放比例。0.5 即输出 960x540，体积约为 1/4。
        边缘板上如果只是看比分，没必要保留 1080p。
    """
    from referee.scoreboard_ui import ScoreboardUI

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频: {}'.format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = max(2, int(src_w * scale) // 2 * 2)   # 保持偶数，编码器要求
    h = max(2, int(src_h * scale) // 2 * 2)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer, used_codec = _open_writer(out_path, fps, w, h)

    ui = ScoreboardUI()
    cur = (0, 0, 'A')       # 起始比分；首帧还没观测时用这个
    si = 0                  # snapshots 指针
    ti = 0                  # timeline 指针
    last_ev_ts = None
    last_ev_text = None
    n = 0
    while True:
        # --max-frames 调试时只渲染这些帧，不去读整段视频
        if max_frames is not None and n >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        ts = n / float(fps)

        # 推进比分快照：取所有 ts <= 当前时刻的最后一个
        while si < len(snapshots) and snapshots[si][0] <= ts:
            cur = snapshots[si][1:]
            si += 1
        # 推进事件提示
        while ti < len(timeline) and timeline[ti][0] <= ts:
            last_ev_ts, last_ev_text = timeline[ti]
            ti += 1

        age = (ts - last_ev_ts) if last_ev_ts is not None else None
        text = last_ev_text if (age is not None and age < fade_s) else None

        # 先缩放再画 UI：比分面板按帧尺寸自适应，缩放后字号依然清晰，
        # 且比「画完再缩」省一半时间。
        if (w, h) != (src_w, src_h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        ui.draw(frame, score_a=cur[0], score_b=cur[1], games_a=0, games_b=0,
                server=cur[2], event_text=text, event_age=age, cumulative=True)
        writer.write(frame)

        n += 1
        if progress_every and n % progress_every == 0:
            pct = ' ({:.0%})'.format(n / total) if total else ''
            print('  渲染 {}/{} 帧{}'.format(n, total or '?', pct))

    cap.release()
    writer.release()
    return n, used_codec


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
    ap.add_argument('--render-scale', type=float, default=1.0,
                    help='渲染输出缩放比例，默认 1.0（原分辨率）。'
                         '传 0.5 输出 960x540，体积约为 1/4，看比分足够')
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
                        render_scale=args.render_scale,
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
