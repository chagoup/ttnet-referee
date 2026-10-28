# TTNet 乒乓球裁判 · 边缘板部署包

从 `TTnet1.0` 裁剪而来，**只保留推理与判分**，剔除全部训练代码与数据集。  
目标：可直接拷贝到硬件边缘板（RK3588 等）上运行，输入视频，输出比分。

---

## 零、快速开始（新机器部署）

```bash
# 1) 克隆（仓库约 61MB）
git clone https://github.com/chagoup/ttnet-referee.git
cd ttnet-referee

# 2) 装依赖
pip install -r requirements.txt

# 3) 获取测试视频（1.7GB，未入库，需单独准备）
python scripts/download_videos.py
#   会自动在本机找原项目的 dataset 目录；
#   找不到时按提示手动放置，或显式指定：
#   python scripts/download_videos.py --from "D:/TTnet1.0/dataset/test/videos"

# 4) 部署自检（30 秒，确认环境/权重/视频规格/推理速度）
python src/selfcheck.py

# 5) 跑
python src/infer.py --video videos/test_1.mp4 videos/test_6.mp4
#   边缘板推荐：加 --device cpu --stride 4
```

> **视频为什么不在仓库里**：test_1.mp4 有 1.05GB、test_6.mp4 有 675MB，  
> 远超 GitHub 单文件 100MB 上限，Git LFS 免费额度（1GB）也装不下 1.7GB。  
> 代码与权重（60MB）都在仓库内，clone 完即可运行，只差视频。

---

## 一、目录结构

```text
ttnet-referee/
├── src/
│   ├── infer.py               ← 唯一入口：推理 + 事件 + 判分 + 出结果
│   ├── selfcheck.py           部署自检（依赖/权重/设备/视频规格/速度）
│   ├── models/
│   │   ├── TTNet.py           模型定义（已修复 CPU 设备问题）
│   │   └── model_utils.py     仅保留 build_ttnet()，剔除训练包装类
│   └── referee/
│       ├── ttnet_adapter.py   TTNet 输出 -> 统一 Observation（支持 --stride）
│       ├── event_processing.py 轨迹净化 + 迟滞驼峰 bounce/net 提取
│       ├── score_state_machine.py 11 分制计分状态机（含防误判护栏）
│       ├── unified_types.py   Observation / Event 数据结构
│       └── scoreboard_ui.py   中文比分面板渲染（仅在 --render 时用到）
├── scripts/
│   └── download_videos.py     测试视频获取脚本
├── checkpoints/
│   └── ttnet_model.pth        纯推理权重（60MB，已剥离 optimizer）
├── videos/                    测试视频放这里（未入库，需自行获取）
├── results/                   运行后生成的结果
├── requirements.txt
├── run.sh / run.bat           一键运行（先自检再推理）
└── README.md
```

全包 15 个 .py 文件，代码总量不到 2700 行，无训练/数据集依赖。

---

## 二、与原项目的差异（三处，都必要）

| 改动                        | 原状                                                   | 现状                           | 为什么必须改                                                                                  |
| ------------------------- | ---------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------- |
| **`TTNet.__normalize__`** | 硬编码 `self.mean.cuda()`                               | 跟随输入张量设备                     | **边缘板无 CUDA，原代码必然崩**<br />`RuntimeError: Expected all tensors to be on the same device` |
| **权重文件**                  | 180.6MB，含 optimizer/lr_scheduler                     | 62MB，纯 state_dict + 结构元信息    | 减少 2/3 体积；且元信息内嵌，板上无需 argparse 配置                                                       |
| **模型加载**                  | `create_model()` 走训练包装类<br />→ 间接引入 losses / sklearn | `build_ttnet()` 直接构造 `TTNet` | 砍掉 sklearn、losses、dataloader 等训练依赖，边缘板无需安装                                              |

**精度已交叉验证**：新旧两套代码对同一视频前 400 帧，逐帧球坐标最大偏差  
**0.000000 px**，392 帧完全一致。这是等价裁剪，不是重写。

---

## 三、依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 只含推理必需的：

```text
torch>=1.5          # 板上建议用厂商提供的版本（RK3588 可用 torch + rknn-toolkit2）
numpy>=1.20
opencv-python>=4.5
Pillow>=8.0         # 仅 --render 需要（中文比分面板）
```

**不需要**：scikit-learn、tensorboardX、tqdm、easydict。  
已从依赖链中移除，因为它们是训练侧的。

---

## 四、用法

```bash
# 单个视频，自动选设备（有 GPU 用 GPU，否则 CPU）
python src/infer.py --video videos/test_1.mp4

# 两个视频批量跑，结果写进 results/
python src/infer.py --video videos/test_1.mp4 videos/test_6.mp4 --out results

# 边缘板上显式指定 CPU
python src/infer.py --video videos/test_6.mp4 --device cpu

# 边缘板推荐：CPU + 抽帧推理（算力降到 1/4）
python src/infer.py --video videos/test_6.mp4 --device cpu --stride 4

# 额外输出叠加比分的 mp4（较慢，板上一般不开）
python src/infer.py --video videos/test_6.mp4 --render

# 只跑前 1000 帧快速验证（--render 时也只渲染这些帧）
python src/infer.py --video videos/test_6.mp4 --max-frames 1000 --render
```

### 主要参数

| 参数                 | 默认                            | 说明                                            |
| ------------------ | ----------------------------- | --------------------------------------------- |
| `--video`          | 必填                            | 一个或多个视频路径                                     |
| `--weights`        | `checkpoints/ttnet_model.pth` | 权重路径                                          |
| `--out`            | `results`                     | 结果输出目录                                        |
| `--device`         | `auto`                        | `auto` / `cpu` / `cuda:0`                     |
| `--bounce-thresh`  | 0.65                          | 落台驼峰上阈值（test_2 网格搜索确定，test_3 验证泛化）            |
| `--net-thresh`     | 0.5                           | 擦网驼峰上阈值                                       |
| `--ball-thresh`    | 0.05                          | 球位置概率阈值                                       |
| `--min-confidence` | 0.25                          | 置信度下限（过滤坐标冻结/离群帧）                             |
| `--render`         | 关                             | 是否输出比分视频                                      |
| `--render-scale`   | 1.0                            | 渲染缩放。0.5 输出 960x540，体积约 1/4                |
| `--max-frames`     | 全部                            | 调试用，只跑前 N 帧                                   |
| `--stride`         | 1                             | 推理步长。板上建议 4（120fps→30fps），算力降到 1/4，窗口仍是连续 9 帧 |

---

## 五、输出

每个视频生成两份：

**`<name>_score.json`** — 机器可读，供上位机集成

```json
{
  "video": "test_6.mp4",
  "frames_processed": 10792,
  "ball_visible_frames": 3602,
  "ball_visible_rate": 0.334,
  "score": {"A": 3, "B": 11},
  "total_points": 14,
  "points": [
    {"ts": 7.675, "winner": "A", "reason": "...", "confidence": 0.82, "score": "1 : 0"}
  ],
  "needs_review": [],
  "suppressed_by_guard": 1,
  "event_counts": {"BALL_LOST": 276, "NET_HIT": 55, "HIT": 66, "BOUNCE": 59},
  "elapsed_s": 316.9,
  "device": "cuda:0",
  "infer_stride": 1
}
```

**`<name>_report.txt`** — 人类可读摘要，含逐分时刻与判据。

**`<name>_score.mp4`** — 仅加 `--render` 时生成，画面左上角叠加比分面板
（累计得分、发球方、落台/擦网事件提示）。视频里的比分与 JSON 报告
来自同一份状态机快照，**不会出现两个结果**。

批量跑时额外写 `summary.json` 汇总所有视频。

### 本包实测结果（已与原项目逐项核对）

| 视频         | 帧数    | 球可见率  | 判分 | 最终比分         | 护栏拦截 | 耗时(GPU) |
| ---------- | ----- | ----- | -- | ------------ | ---- | ------- |
| test_1.mp4 | 16792 | 31.6% | 14 | **A6 : B8**  | 2    | 493 s   |
| test_6.mp4 | 10792 | 33.4% | 14 | **A3 : B11** | 1    | 317 s   |

与 `../TTnet1.0/TEST1_REPORT.md` 记录的历史结果**完全一致**  
（14 分 / A6:B8 / 护栏 2；14 分 / A3:B11 / 护栏 1），  
事件数也逐项相同（BOUNCE 111/59、NET_HIT 111/55、HIT 116/66、BALL_LOST 396/276）。  
**这证明本包是原项目的等价裁剪，不是重写。**

---

## 六、性能

本机实测（RTX 2080 Ti / i7 / 4~10 线程），`src/selfcheck.py` 可直接复现：

| 设备                   | 单帧前向    | 等效 fps |
| -------------------- | ------- | ------ |
| **GPU（RTX 2080 Ti）** | 0.073 s | 13.8   |
| **CPU（10 线程）**       | 0.672 s | 1.5    |

推理之外：

| 环节 | 耗时 | 输出体积（1080p / 120fps） |
|---|---|---|
| 判分（纯 Python） | 毫秒级，与推理完全解耦 | — |
| 渲染 + 中文比分 UI | 约 20~40 ms/帧（含编码） | 约 0.027 MB/帧 → 16800 帧约 460 MB |

> **编码器选择有坑**：本项目默认用 `mp4v`，不是常见的 H.264。
> 实测本机 OpenCV 5.0 缺 `openh264` 库，`avc1` 会静默退回近无损模式，
> 每帧 0.250 MB（是 mp4v 的 9 倍），16800 帧会写出 4GB+。
> 若你的板子有真正的 H.264 编码器，可在 `_open_writer()` 里调整优先级。
> 嫌体积大就用 `--render-scale 0.5`，960x540 看比分完全够，体积降到 1/4。

### 边缘板上的现实与对策

源视频是 **120fps**。逐帧推理意味着每秒要推 120 次：

| 场景                  | 所需算力   | 说明                  |
| ------------------- | ------ | ------------------- |
| CPU 逐帧跑 120fps      | 80x 实时 | 不可能                 |
| CPU 按 1/4 抽帧到 30fps | 20x 实时 | 仍然很慢（90s 视频约 30 分钟） |
| GPU/NPU 按 1/4 抽帧    | 约 1.2x | 接近可用                |

> 结论直说：**这套模型在纯 CPU 边缘板上做不到实时**。  
> 90 秒的 test_6 用 CPU + stride 4 约需 30 分钟。  
> 若板上只能跑 CPU，请务必做 RKNN/ONNX 转换上 NPU，或接受离线批处理的定位  
> （录完再算，而不是边录边判）。

**三条加速路径，按性价比排序**：

1. **抽帧（最简单，先做这个，已内置）**

   判分逻辑全部按秒设计（0.15s 拟合窗、0.3s 去重、1.2s 判分最小间隔），  
   30fps 完全够用。用 `--stride 4` 把推理频率降到 1/4：
   ```bash
   python src/infer.py --video videos/test_6.mp4 --device cpu --stride 4
   ```
   本机实测（test_6 前 500 帧，纯 CPU）：
   | 模式           | 推理帧数 | 耗时      | 提速        |
   | ------------ | ---- | ------- | --------- |
   | `--stride 1` | 492  | 304.6 s | 1.0x      |
   | `--stride 4` | 123  | 80.8 s  | **3.77x** |
   **注意本包的实现方式**：它不是「跳过 4 帧再取 9 帧」那种跳帧抽帧，  
   而是**窗口始终取连续 9 帧，只降低推理频率**。  
   原因：跳帧会让输入窗口里相邻帧相隔 4 倍时间，等于把 120fps 的球速  
   变成 480fps 的位移，与训练分布不符，会明显损伤 bounce/net 判据。  
   当前做法在被推理的帧上，模型输入与原版**逐比特一致**，  
   代价只是事件时间分辨率变粗（stride=4 即 33ms），  
   相对 ±100ms 的评估容差可以接受。
2. **转 RKNN / ONNX（收益最大）**  
   TTNet 是纯 CNN（无动态控制流），适合量化部署。  
   `rknn-toolkit2` 通常能把这类模型跑到 20~30fps。注意：
   - 输入固定 `1×27×128×320`，静态形状，转换友好；
   - 分割头输出 `1×3×128×320`，若只判分可裁掉 seg 分支进一步提速；
   - 量化校准集用本包 `videos/` 里的帧即可。
3. **裁剪冗余分支**  
   判分实际只用到 `bounce/net` 分数 + 球坐标 + 球台掩码。  
   `seg` 的记分牌通道（ch2）和运动员通道（ch1）当前未使用，  
   可只保留 ch0 球台，减少分割头计算量。

---

## 七、迁移到硬件板：注意事项

1. **设备字符串**：`--device` 传 `cpu` 即可；若厂商 torch 支持 `npu:0`，直接传。  
   `TTNet.__normalize__` 已改为跟随输入设备，不再假设 CUDA。
2. **输入分辨率固定**：模型输入是 1920×1080 → 320×128 的固定缩放，  
   视频**必须是 16:9**，否则坐标会偏。
3. **帧率不影响判分逻辑**：所有阈值用秒而非帧数（`0.15s` 窗口、`0.3s` 去重），  
   30/60/120fps 通用。
4. **内存**：模型 15.77M 参数，CPU 推理常驻约 300MB；若开 `--render`  
   额外需要视频帧缓冲。
5. **单线程瓶颈**：判分层纯 Python，实测 13000 观测/秒，远快于推理，不是瓶颈。

### 移植时最容易踩的坑（本包已处理，改造时勿回退）

| 坑                                   | 现象                                                       | 本包的处理                            |
| ----------------------------------- | -------------------------------------------------------- | -------------------------------- |
| `__normalize__` 硬编码 `.cuda()`       | CPU 上直接报 `Expected all tensors to be on the same device` | 改为跟随输入张量设备                       |
| 用 `machine.tick()` 而非 `tick_deep()` | 判分只有个位数（test_6 只判 5 分而非 14 分）                            | 用 `tick_deep()`，它才含「落台序列中断 → 判分」 |
| 漏调 `machine.finalize()`             | 末尾最后一回合永远判不出来                                            | 视频结束显式调 `finalize(last_ts)`      |
| 权重键名带 `model.` 前缀                   | `load_state_dict` 大量 missing keys                        | 导出时剥离前缀                          |
| 跳帧抽帧                                | 球速分布失真，bounce/net 判据明显变差                                 | `--stride` 只降推理频率，窗口仍是连续 9 帧     |
| 用训练包装类加载                            | 被迫安装 sklearn / losses 等训练依赖                              | 用 `build_ttnet()` 直接构造 `TTNet`   |

---

## 八、已知局限（部署前必须知道）

1. **球可见率只有 25%~35%**：一半以上的帧看不到球（拖影、遮挡、飞出画面）。  
   这决定了单靠 TTNet 无法覆盖全部回合。
2. **判分准确率无真值**：OpenTTGames 只标注 bounce/net 位置，不标注谁得分。  
   当前输出「逻辑自洽」，不等于「判对」。
3. **擦网判据偏多**：`serve_net_short` 类判分占比偏高，因为 TTNet 的 net  
   精确率仅 54%~~72%，一次误报 + 一次侧别误判就会凭空判一分。  
   生产环境建议把 `--net-thresh` 调高（0.7~~0.8）或转人工确认。
4. **片段制视频会累计分数**：状态机默认 `cumulative_mode=True`，  
   到 11 分不换局（测试片不存在打完一局）。正式比赛需改为 `False`。

---

## 九、复现本包的裁剪过程

如需重新生成，参见原项目 `TTnet1.0`：

- 源码来自 `TTnet1.0/src`
- 权重来自 `TTnet1.0/checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth`  
  （键名带 `model.` 前缀，需剥离后重存）
- 视频来自 `TTnet1.0/dataset/test/videos`

---

## 十、来源与授权（重要）

本仓库是**衍生作品**，模型代码与权重源自：

- [maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch)
- 论文：TTNet, CVPR 2020（Roman Voeikov, Nikolay Falaleev, Ruslan Baikulov）
- 测试视频来自 OpenTTGames 数据集

**上游项目未附带 LICENSE 文件，因此本仓库不主张任何开源许可  
（No license granted）。**

仓库当前为 Public 仅为便于跨设备部署与交流，**公开可见不等于授权使用**。  
正式对外发布前请先取得上游授权，详见 [LICENSE](LICENSE)。

本项目独立编写、与上游无关的部分：`src/referee/` 全部代码、  
`src/infer.py`、`src/selfcheck.py`、`scripts/download_videos.py`、  
以及 `TTNet.__normalize__` 的设备自适应修复。

引用请优先使用上游论文：

```bibtex
@article{TTNet,
  author = {Roman Voeikov, Nikolay Falaleev, Ruslan Baikulov},
  title = {TTNet: Real-time temporal and spatial video analysis of table tennis},
  year = {2020},
  conference = {CVPR 2020},
}
```
