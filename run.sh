#!/usr/bin/env bash
# TTNet 边缘板一键运行脚本
#
# 用法：
#   bash run.sh              # 用默认设置跑 videos/ 下两个视频
#   bash run.sh cpu          # 强制 CPU（边缘板）
#   bash run.sh cpu 4        # CPU + 抽帧步长 4（板上推荐）
#
# 说明：脚本会自动先跑自检，自检不通过（缺依赖/权重损坏）就停下，
# 避免跑到一半才报错。

set -e

cd "$(dirname "$0")"

DEVICE="${1:-auto}"
STRIDE="${2:-1}"

# 优先用环境里的 python，其次尝试常见路径
PY="${PYTHON:-python}"

echo "=============================================="
echo "TTNet 裁判 · 边缘推理"
echo "  设备: $DEVICE   抽帧步长: $STRIDE"
echo "=============================================="

# 1) 自检
echo
echo "[1/2] 部署自检..."
if ! "$PY" src/selfcheck.py --device "$DEVICE"; then
    echo
    echo "自检未通过，已中止。请按上面提示处理后再运行。"
    exit 1
fi

# 2) 推理
echo
echo "[2/2] 开始推理..."
VIDEOS=()
for f in videos/*.mp4; do
    [ -e "$f" ] || continue
    VIDEOS+=("$f")
done

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "videos/ 目录下没有 mp4 文件"
    exit 1
fi

"$PY" src/infer.py \
    --video "${VIDEOS[@]}" \
    --out results \
    --device "$DEVICE" \
    --stride "$STRIDE"

echo
echo "完成。结果在 results/ 目录："
ls -la results/ 2>/dev/null || true
