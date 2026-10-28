# -*- coding: utf-8 -*-
"""获取测试视频 test_1.mp4 / test_6.mp4。

测试视频总 1.7GB，超出 GitHub 单文件 100MB 限制，因此不入库。
本脚本提供三条获取路径，按顺序自动尝试：

1. **本地原项目**：如果本机还有 TTnet1.0 项目，直接从它的 dataset 目录拷贝。
   这是最快的方式（同盘拷贝，几秒钟）。
2. **环境变量/参数指定源目录**：`--from D:/path/to/TTnet1.0/dataset/test/videos`
3. **提示手动获取**：OpenTTGames 数据集需自行下载后放到 videos/ 目录。

用法::

    # 自动查找本机原项目并拷贝
    python scripts/download_videos.py

    # 指定源目录
    python scripts/download_videos.py --from "D:/TTnet1.0/dataset/test/videos"

    # 只检查现状，不拷贝
    python scripts/download_videos.py --check
"""

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
VIDEO_DIR = os.path.join(_ROOT, 'videos')

WANTED = ['test_1.mp4', 'test_6.mp4']

# 常见的原项目相对位置（同盘/同级）
CANDIDATE_DIRS = [
    os.path.join(os.path.dirname(_ROOT), 'TTnet1.0', 'dataset', 'test', 'videos'),
    os.path.join(os.path.dirname(_ROOT), 'dataset', 'test', 'videos'),
    os.path.join(_ROOT, '..', 'TTnet1.0', 'dataset', 'test', 'videos'),
]


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return '%.1f %s' % (n, unit)
        n /= 1024.0
    return '%.1f TB' % n


def check():
    """检查 videos/ 下已有文件，返回缺失列表。"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    missing = []
    print('视频目录: %s' % VIDEO_DIR)
    for name in WANTED:
        p = os.path.join(VIDEO_DIR, name)
        if os.path.isfile(p):
            print('  [OK]   %-14s %s' % (name, human(os.path.getsize(p))))
        else:
            print('  [缺失] %-14s' % name)
            missing.append(name)
    return missing


def find_sources():
    """在候选目录里找能提供所需视频的源目录。"""
    found = []
    for d in CANDIDATE_DIRS:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        has = [n for n in WANTED if os.path.isfile(os.path.join(d, n))]
        if has:
            found.append((d, has))
    return found


def copy_from(src_dir, missing):
    print('\n从 %s 拷贝...' % src_dir)
    ok = []
    for name in missing:
        src = os.path.join(src_dir, name)
        dst = os.path.join(VIDEO_DIR, name)
        if not os.path.isfile(src):
            continue
        size = os.path.getsize(src)
        print('  %-14s %s ...' % (name, human(size)), end=' ', flush=True)
        try:
            shutil.copy2(src, dst)
            print('完成')
            ok.append(name)
        except OSError as exc:
            print('失败: %s' % exc)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description='获取 TTNet 测试视频')
    ap.add_argument('--from', dest='src', default=None,
                    help='源视频目录（含 test_1.mp4 / test_6.mp4）')
    ap.add_argument('--check', action='store_true', help='只检查，不拷贝')
    args = ap.parse_args(argv)

    print('=' * 62)
    print('TTNet 测试视频获取')
    print('=' * 62)

    missing = check()
    if not missing:
        print('\n全部视频已就位，可以直接运行：')
        print('  python src/infer.py --video videos/test_1.mp4 videos/test_6.mp4')
        return 0
    if args.check:
        return 1

    # 路径 1：显式指定
    if args.src:
        if not os.path.isdir(args.src):
            print('\n[错误] 源目录不存在: %s' % args.src)
            return 1
        got = copy_from(args.src, missing)
        missing = [n for n in missing if n not in got]
        if not missing:
            print('\n完成。可以运行 src/infer.py 了。')
            return 0

    # 路径 2：自动查找
    sources = find_sources()
    if sources:
        src_dir, has = sources[0]
        print('\n自动找到源目录: %s' % src_dir)
        got = copy_from(src_dir, missing)
        missing = [n for n in missing if n not in got]
        if not missing:
            print('\n完成。可以运行 src/infer.py 了。')
            return 0

    # 路径 3：手动
    print('\n' + '-' * 62)
    print('未能自动获取以下视频：%s' % ', '.join(missing))
    print('-' * 62)
    print('请手动获取后放到: %s' % VIDEO_DIR)
    print()
    print('来源说明：')
    print('  test_1.mp4 / test_6.mp4 来自 OpenTTGames 乒乓球数据集。')
    print('  项目地址: https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch')
    print('  数据集说明见该项目 prepare_dataset/README.md。')
    print()
    print('或者，如果你有原项目 TTnet1.0，可显式指定：')
    print('  python scripts/download_videos.py --from "D:/TTnet1.0/dataset/test/videos"')
    print()
    print('注意：视频规格必须是 1920x1080 / 16:9，否则落点坐标会偏移。')
    print('用 python src/selfcheck.py 可校验规格。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
