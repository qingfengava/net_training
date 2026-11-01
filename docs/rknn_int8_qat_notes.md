# 装甲检测模型在 RK3576 上的 INT8 量化：实测与 QAT 建议

来源：空中系统仓库 `~/rm_micro_uav_swarm` 在 RK3576（RKNN NPU）上做的转换实验。本文是这些实验的整理，给 net_training 规划 QAT INT8 时参考。
- 第 1–3 节是在 RK3576 上测得的结果；
- 第 4 节是对训练和导出的建议；
- 第 5 节列出哪些结论也适用于 TensorRT；
- 第 8 节是社区三家开源旧装甲模型的量化、训练和评估经验，以及和 net_training 实测的对照。

实验都用 rknn-toolkit2 2.3.2 的模拟器，只说明数值一致性，延迟没有在板上测。本仓库的文件没有改动；脚本和结果都在 `~/rm_micro_uav_swarm`（`tools/rknn_check*.py`、`log/p07/`）。

## 0. 结论

1. **现有的 YOLO26n 导出模型（`exports/y26n_simm7_e100`）做 INT8 PTQ，关键点几乎不受影响**：与 FP32 相比，误差 p95 为 1.7 网络像素。**主要损失在置信度上**：FP32 下置信度 ≥0.5 的检测，INT8 后有 13% 掉到阈值以下。QAT 最该补的是置信度和类别分支，不是关键点。
2. **YOLO26 pose 头的结构对 INT8 是友好的**：框、类别、关键点各自是独立的卷积输出，关键点解码只乘 stride。导出给 RKNN 时要保持这一点，**在 9 个检测头卷积处切图**，后面的解码和 top-k 放到 CPU 上。
3. **rknn-toolkit2 能加载 QDQ 格式的 ONNX**（QAT 常用的导出格式），构建时沿用模型里的量化参数。YOLO26n 要设 `optimization_level=2`，否则构建时段错误。
4. **评估必须按检测结果对比**：漏检、阈值下的召回、置信度偏移、关键点误差、编号一致率。不能用输出张量的余弦相似度，第 1 节有反例：相似度 0.99，但一个检测都没有。
5. **在 RK3576 上，INT8 之外能用的只有 FP16。** w16a16i 在模拟器上能跑、精度也好，但 RK3576 不支持这种量化类型。
6. **社区经验印证了上面几条**（第 8 节）。河北科大 AT 把 YOLO26n-pose 部署到 Axera650 NPU 时：
   - 即使激活用 16 位（w8a16），也必须先做 QAT；
   - 把 C2PSA 换成了对量化友好的 CoordAtt；
   - 把锚框偏移相关的解码放到 CPU 上用 FP32 算，和第 4.1 节的切图做法一致。

## 1. 旧模型的教训：0526.onnx（jlu_vision_26 用的深大权重）

这是 YOLOv5 结构加四角点的旧装甲模型，输入 640×640，输出 [1, 25200, 22]。每行依次是 8 个角点坐标、1 个置信度 logit、4 个颜色分数、9 个类别分数。测试用 demo.avi 的 50 帧，共 47 个检测。

它就是深大 RobotPilots 2024 赛季开源的模型（第 8 节的 RP24）：与 `broalantaps/RobotDetectionModel` 仓库里的 `Model/0526.onnx` 字节完全相同，git blob 都是 c6fc0dcd。按它的 README，主干是 MobileNetV3，原本在 NUC 上用 OpenVINO 跑 FP16，作者没有为 INT8 做过设计。

| 方案 | recall | 平均框 IoU | 角点误差 px | 说明 |
|---|---|---|---|---|
| INT8，整图 | 0 | — | — | 输出余弦相似度仍有 0.99 |
| INT8，切在检测头 | 0.979 | 0.833 | 6.2 | |
| INT8，mmse 校准 | 0.979 | 0.822 | 6.6 | 构建 12 分钟，没有改善 |
| INT8，检测头留 FP16 | 0.979 | 0.871 | 4.1 | 混合量化 |
| FP16 | 1.000 | 0.999 | 0.03 | |

失效的机制有三层：

- **整图 INT8 把置信度量化掉了。** 一个张量只有一个量化尺度。输出里角点坐标的范围是 −2409 到 1339，步长 14.7，置信度 logit（约 −52 到 2.9）因此只剩 0、±14.7 这几档。最高的 2.91 被量化成 0，分数变成 0.5，全部低于 0.7 的阈值。
- **余弦相似度看不出问题。** 角点坐标占了输出平方和的 99.93%，其余 14 列即使全错，相似度也有 0.9996。
- **切图以后角点仍差 4–6 px，原因有两个：**
  - 每个检测头仍把角点、置信度、类别放在同一个卷积输出里共用尺度，范围被 logit 撑到 ±40–57；
  - 角点解码是"原始输出 × 锚框尺寸 + 网格偏移"，锚框尺寸 10–373 px，原始输出的小误差被放大几十倍。
  
  检测头留成 FP16 后还剩约 4 px，这部分来自主干网络各层 INT8 特征的累积误差。

**教训：** 输出设计决定了 INT8 能不能用。不同物理量（像素坐标、logit）不要挤进同一个张量；解码乘数要小。

## 2. 现有模型：YOLO26n（y26n_simm7_e100）的 INT8 PTQ

方法（`~/rm_micro_uav_swarm/tools/rknn_check_y26.py`）：
- 数据：`baked/simm7_c2/images/val` 按文件名排序，偶数号 100 张做校准、奇数号 200 张做测试；
- NPU 只算到 9 个检测头卷积，后面的子图（解码和 top-30）用 ONNX Runtime 在 CPU 上算；
- 参照：ONNX Runtime 跑原模型（FP32）；
- 配对：FP32 输出中置信度 ≥阈值的检测，与 RKNN 输出按 4 个关键点的中心配对，距离 ≤8 px。

### 2.1 检测头的取值范围（200 张测试图）

| 分支 | stride 8 | stride 16 | stride 32 | INT8 步长 | 换算成像素的步长 |
|---|---|---|---|---|---|
| 框（4 通道） | −0.6 到 8.2 | 0.05 到 13.8 | 0.7 到 6.6 | 0.023–0.054 | 0.28–0.86 px |
| 关键点（12 通道） | −19 到 11 | −25 到 9 | −10 到 7 | 0.065–0.135 | 0.95–2.16 px |
| 类别 logit（13 通道） | −101 到 4.0 | −104 到 4.9 | −37 到 2.1 | 0.15–0.43 | — |

- 关键点解码是 `(原始输出 + 网格坐标) × stride`，乘数 8–32；框是 `(网格 ± 原始输出) × stride`。三类量分开量化，量化步长换算成像素只有 1–2 px，这是比 0526 好得多的根本原因。
- 类别分支的 logit 下限到 −100 左右，可能是一对一分配把背景位置压得很低造成的。类别分支的步长因此最粗（约 0.4 logit）。

### 2.2 结果（网络输入 768×576 像素）

| 阈值 | FP32 检测数 | 有对应框 | 仍高于阈值 | 编号一致 | 颜色一致 | 关键点误差 p50 / p95 |
|---|---|---|---|---|---|---|
| 0.5 | 402 | 100% | 87.1% | 0.993 | 0.998 | 0.60 / 1.71 px |
| 0.2（nt.eval 用的阈值） | 638 | 100% | 84.0% | 0.939 | 0.998 | 0.53 / 1.68 px |

阈值 0.2 时置信度变化（INT8 减 FP32）的 p5、p50、p95 是 −0.46、−0.03、+0.06。大部分检测的置信度几乎不变，少数检测掉得很多。

### 2.3 怎么理解这些数字

- **关键点：** INT8 引入的误差 p95 为 1.7 网络像素。按 AGENT_HANDOFF 第 7 节 TensorRT FP16 一行的比例（0.21 网络像素对应原图 0.36 px）换算，约合原图 2.9 px。模型本身相对真值的角点误差 p95 是原图 6.2 px，INT8 带来的误差不可忽略，但量级比 0526 小得多。
- **置信度：** 这是 PTQ 最明显的损失，阈值附近的检测会掉出去。编号在低阈值时有 6% 改变，这些主要是本来就不确定的检测。
- **类别 logit 截断试验：** 在类别头卷积后插入 Clip(±16) 再量化（sigmoid(±16) 已经饱和，检测结果不变），步长从约 0.41 缩到 0.125。结果没有稳定改善：阈值 0.2 时"仍高于阈值"从 84.0% 变成 87.1%，阈值 0.5 时反而从 87.1% 降到 85.1%。说明置信度误差主要来自上游的 INT8 特征，不是类别头输出的量化步长。**只调输出范围不够，要靠 QAT 调权重。**

## 3. QAT 模型能否被 rknn-toolkit2 加载

**官方说明**（RKNN2 SDK 开发指南 V2.5.0，2.2.5 节）：
- PyTorch 方面支持 FX 接口（`prepare_qat_fx`、`convert_fx`）产生的量化模型，用 `load_pytorch` 加 `build(do_quantization=False)` 转换，官方示例是 `examples/pytorch/resnet18_qat`。
- 推荐的 qconfig 与 RKNPU 的量化方式一致：
  - 激活：`FakeQuantize` 配 `MovingAverageMinMaxObserver`，`quant_min=0, quant_max=255, reduce_range=False`，即按张量的 uint8 非对称量化；
  - 权重：`MovingAveragePerChannelMinMaxObserver`，`qint8`，`quant_min=-128, quant_max=127`，`qscheme=torch.per_channel_affine`，`reduce_range=False`。
- 指南建议先确认 PTQ 和混合量化都不满足要求，再做 QAT。
- 不能量化的算子保持浮点；前后都是量化算子时，工具会自动把它转成量化算子；否则插入量化和反量化，带来额外耗时。
- **版本限制：** 指南推荐用 PyTorch 1.10–1.13.1 导出已量化的 TorchScript。net_training 用的是 torch 2.11，这条路要另建环境。

**实测 ONNX QDQ 路线**（`~/rm_micro_uav_swarm/tools/rknn_qdq_probe.py`）：没有真的做 QAT，而是用 ONNX Runtime 的 `quantize_static` 生成同格式的 QDQ 模型代替（QuantizeLinear/DequantizeLinear，激活 uint8，权重 int8 逐通道）。
- **小模型**（两层卷积）：`load_onnx` 加 `build(do_quantization=False)` 成功，RKNN 模拟器的输出与 ONNX Runtime 跑同一个 QDQ 模型**逐元素一致**。说明 RKNN 用的是模型里的量化参数，没有重新校准。
- **YOLO26n**（只截到 9 个检测头，含 991 个 Q/DQ 节点）：
  - 默认的 `optimization_level=3` 在构建阶段**段错误**；
  - 加载时工具也提示："QAT model is loaded, but the config.optimization_level = 3 … will affect the accuracy … try 2"；
  - 设为 2 后能构建和推理。RKNN 与 ONNX Runtime 的 QDQ 推理之间，检测头输出的相对平均误差为 0.036；作为对照，ONNX Runtime 的 QDQ 与 FP32 之间为 0.074。大模型上两者不是逐元素一致，可能是算子融合或部分算子的量化处理不同，原因没有查。

## 4. 对 QAT 的建议

### 4.1 导出

- 给 RKNN 的模型**在 9 个检测头卷积处切图**（`one2one_cv2.*`、`one2one_cv3.*`、`one2one_cv4_kpts.*` 的最后一层 Conv）。
  - 后面的解码、sigmoid、top-30 放在 CPU 上，运算量很小。
  - 不切图的话，端到端图中有 21 个算子会落到 CPU 上（ReduceMax、TopK×2、GatherElements×3 等），NPU 和 CPU 要来回切换。
  - 最终输出 [1,30,18] 里像素坐标和分数又拼在一起，整图量化会重演第 1 节的问题。
- 这与 sleepy 的契约 v2（端到端 [1,30,18]）不同。可以考虑给 `nt.export` 加一个"只到检测头"的导出变体，并在 `contract.json` 里写明检测头张量的名字、顺序和解码方式。
- 保持框、类别、关键点三个分支分开，不要在 NPU 部分把它们拼成一个张量。

### 4.2 训练

- **重点是置信度和类别分支。** PTQ 下关键点已经基本可用，置信度掉得最多（第 2.2 节）。
- **评估要用部署阈值下的检测指标：** 有对应框的比例、仍高于阈值的比例、置信度偏移的分位数、编号和颜色一致率、关键点误差。也要和 FP32 比 `nt.eval` 的 AP50 和各分组指标。
- **INT8 模型要重做阈值扫描。** 置信度整体略降、少数掉得很多，FP32 上选的阈值不一定还是最优（`nt.eval` 已有阈值扫描）。
- **qconfig 按 RKNN 的量化方式配**（第 3 节）。只部署 RKNN 时，激活用非对称 uint8，权重用逐通道 affine。
  - 要和 TensorRT 共用一个 QAT 模型时：TensorRT 的 INT8 要求对称量化（zero point 为 0）。对称参数是非对称的特例，RKNN 能加载；本次试探用的 QDQ 模型就是对称权重，构建成功。
  - 但用对称参数训出的模型在 RKNN 上是否最优，没有验证。
- **注意力模块**（C2PSA 里的 MatMul、Softmax）和 SiLU 的 sigmoid：
  - 看 QAT 导出后是否带量化参数。没有的话，RKNN 会插入量化和反量化（第 3 节）；
  - 构建时开 `verbose=True`，看每个算子落在 NPU 还是 CPU、精度是什么。

### 4.3 转换和验收流程

1. FP16：`build(do_quantization=False)` 在模拟器上跑，对比 ONNX，结果应几乎一致。这一步用来排除前处理和归一化的配置错误。
2. INT8 PTQ 基线：本文第 2 节的数字。
3. QAT：导出 QDQ ONNX，`optimization_level=2`，`build(do_quantization=False)`。分别与 QAT 模型自身的伪量化推理、FP32 做第 4.2 节的检测级对比。QAT 后的模型应明显好于第 2 步。
4. 上板：同一批图片在 RK3576 上推理，对比模拟器的结果，并测延迟。

## 5. 平台支持和其他坑

- **模拟器不检查量化类型的平台支持。** 0526 用 w16a16i 在 rk3576 模拟器上精度接近 FP16，但构建日志有 `E RKNN: Conv Not Support u8->i16 For Now`。开发指南 2.2.3.1 节写明：
  - w16a16i 仅 RV1103/RV1106 支持；
  - w8a16 仅 RK3562 支持（在 rk3576 上 `config` 直接报错）；
  - w4a16 在 RK3576 上只量化卷积权重（4 位），0526 上检测数为 0。
  
  RK3576 上可用的是 w8a8（INT8）和 FP16。**构建日志里的 E 行要逐条看。**
- **rknn-toolkit2 2.3.2 的依赖：** 要 `onnx==1.16.1`（新版 onnx 删了 `onnx.mapping`）和 `setuptools<70`（要 `pkg_resources`）。这与 net_training 的 onnx 1.23.2 冲突，所以要放在单独的环境里。
- **mmse 校准**在 0526 上用了 738 s、3 GB 内存，精度反而略差；KL 校准把 recall 降到 0.936。对这类检测模型，校准算法不是主要矛盾。
- **混合量化**（`hybrid_quantization_step1` 的 `custom_hybrid` 指定子图保持 FP16）可以用，但对 0526 只把 IoU 从 0.833 提到 0.871。

## 6. 适用于 TensorRT 的部分

本文没有做任何 TensorRT INT8 实验。下面的判断来自量化原理，没有实测：
- **通用的结论：**
  - 激活按张量量化，不同物理量不要共用一个张量；
  - 解码乘数越小越好；
  - 必须用检测级指标评估，不能看张量相似度；
  - 置信度分支对量化最敏感。
- **只适用于 RKNN 的结论：** 量化类型的平台支持、`optimization_level`、切图的位置和 CPU 回退算子、rknn-toolkit2 的依赖版本。
- TensorRT 的显式量化（Q/DQ）只要求对称 INT8，没有放 Q/DQ 的层会以较高精度运行。解码部分在 TensorRT 里的处理方式与 RKNN 不同，需要单独验证。

## 7. 复现

在 `~/rm_micro_uav_swarm` 下，用 conda 环境 rknn（rknn-toolkit2 2.3.2）：

`baked/simm7_c2` 已于 2026-11-01 删除以腾磁盘。复现前先在 net_training 里重新烘焙（约 3 分钟，像素和原来相同）：`scripts/run_capped.sh 6 -- python -m nt.bake --src datasets/sim/simm7 --out baked/simm7_c2 --copies 2 --seed 0 --workers 6`。生成后可以用 `runs/baked_manifests/simm7_c2.meta.sha256` 核对。

```bash
py=~/miniforge3/envs/rknn/bin/python
# YOLO26n INT8 PTQ（约 2.5 分钟，峰值内存 1.9 GB）
~/net_training/scripts/run_capped.sh 5 -- $py -I tools/rknn_check_y26.py \
  --onnx ~/net_training/exports/y26n_simm7_e100/model.onnx --images ~/net_training/baked/simm7_c2/images/val --conf 0.2
#   --clip-cls 16 为类别 logit 截断试验；结果在 log/p07/y26n/
# QDQ 加载试探
~/net_training/scripts/run_capped.sh 6 -- $py -I tools/rknn_qdq_probe.py \
  --onnx ~/net_training/exports/y26n_simm7_e100/model.onnx --images ~/net_training/baked/simm7_c2/images/val --opt-level 2 --out log/p07/qdq_opt2
# 0526 的各方案对比
$py -I tools/rknn_check.py armor --onnx ~/jlu_vision_26/assets/0526.onnx --video ~/jlu_vision_26/assets/demo.avi [--no-cut-heads|--hybrid-heads|--no-quant]
```

0526 各方案的完整对比见 `~/rm_micro_uav_swarm/docs/perception_inventory.md` 第 4 节。

## 8. 社区三家开源旧装甲模型的经验

资料来自三家在 RoboMaster 论坛的开源帖（文章 1886180、54091、1942761）和对应 GitHub 仓库的 README，2026-10-31 读取。三家都**没有公开训练代码和数据集**，下面的内容全部来自 README 的文字和表格，不是我们复现的结果。凡是和 net_training 实测对照的地方，都注明了我们的数字。

### 8.1 概况

| | 河北科大 Actor&Thinker（AT） | 深大 RobotPilots 2024（RP24） | 深大 SZU 2026 部署库 |
|---|---|---|---|
| 仓库 | `PraySky1337/AT_NN_Detector`（2026-06） | `broalantaps/RobotDetectionModel`（2024–25） | `SZURPVision/26_NNDeployment_Lib_and_Detection_Models`（2026-08） |
| 结构 | YOLO26n-Pose，P5 用 C2PSA 或 CoordAtt | YOLOv5 改造，主干 MobileNetV3 | v5n-20250725（YOLOv5n）和 v8n-20260726（YOLOv8n-Pose，自定义头） |
| 输入 | 640×640 或 576×768 | 640×640 | v5n 640×640；v8n 480×640 |
| 输出 | [1,30,18]：框、置信度、12 类（大小×编号）、4 色、4 角点 | [1,25200,22]：8 个角点值、置信度 logit、4 色、9 类 | v5n 同 RP24；v8n 是 [1,21,6300] |
| 部署 | Orin NX Super 上 TensorRT FP16；Axera650 NPU 上 w8a16 | NUC12 上 OpenVINO GPU FP16 | NUC13 iGPU 上 OpenVINO FP16 |
| 数据 | 2.3 万张清洗过的图，2×RTX 3090 训练 | 约 1.5 万张，AutoDL 4×4090 训练 | 未说明数量 |
| 和我们的关系 | sleepy 现在用的 praysky 模型就来自这里 | 第 1 节的 `0526.onnx` 就是它 | — |

### 8.2 量化：只有 AT 做过 NPU INT8，经验和本文的实测一致

- **FP16 不需要特别处理；上 NPU 必须 QAT。** AT 部署到 Axera650（18 TOPS）时：
  - 先做 QAT，再用厂商工具链 Pulsar2 做 PTQ 和仿真，按量化报表逐层微调；
  - 最终用 w8a16，即激活保留 16 位，整条链路 170 FPS，纯推理 700 FPS。
  - **RK3576 不支持 w8a16**（第 5 节），只能用 w8a8。要求比 AT 当时更严，所以 QAT 应该从一开始就在计划里，不能指望 PTQ。
- **C2PSA 对量化不友好。** AT 的解释：PSA 里一个大卷积后面紧跟 Softmax，必然带来很大的离群值；CoordAtt 只有 XY 池化、1×1 卷积、BN 和 Hard-Swish，都是对量化友好的操作。
  - 他们上场的方案是把 C2PSA 换成 CoordAtt，代价是 mAP50-95 从 0.89 降到 0.87。
  - 我们的仿真 A/B（AGENT_HANDOFF 第 7 节）结果相近：CoordAtt 的 AP50 是 0.809，C2PSA 是 0.826。但 CoordAtt 两个种子之间就差 0.024，两者在仿真上分不出高下。机器人编号准确率 0.780 对 0.776，持平。
  - TensorRT FP16 延迟（RTX 5060）：CoordAtt 0.798 ms，C2PSA 0.838 ms。去掉注意力（AP50 0.780）损失明显，不能这么做。
- **锚框偏移放到 CPU 上用 FP32 算。** AT 手动把模型裁到检测头，锚框偏移相关的部分全部移到 CPU，代价是后处理负担变重。这和本文第 4.1 节"在 9 个检测头卷积处切图"是同一个思路。
- **AT 设想过但没试的方案：** 改写 forward 后用 LoRA 微调，这样能保留 C2PSA 和端到端导出，精度损失较小。
- **量化会放大脏数据的影响。** AT 的模型在 w8a16 下偶尔把前哨站识别异常，他们初步判断是数据集里有脏数据。做 QAT 前，校准集和训练集的标注质量要先清理。
- **RP24 和 SZU 都没做过 INT8**，都部署在 OpenVINO FP16 上。RP24 就是第 1 节的 0526，INT8 PTQ 后整图失效。这说明为 FP16 设计的输出格式，不经过结构上的处理，过不了 INT8。

### 8.3 训练经验

| 经验 | 出处 | 我们的对照 |
|---|---|---|
| 一定要训到收敛，YOLO 收敛慢 | RP24 | 40 轮 AP50 0.646，100 轮 0.827，80 轮后趋平 |
| 数据集决定上限；坏例多数来自脏数据；交换来的数据先清洗再并入；每类约 2000 张效果就不错 | AT | 我们 13 类，烘焙后只有 5182 张，按这个标准约需 2.6 万张。数据量实验还没做 |
| 负样本和难例迭代很重要，好模型是迭代出来的 | RP24、SZU | `nt.mine` 加 `nt.annotate` 的坏例闭环已具备；SZU 的 v5 训练时没有兑换站，会误识别己方兑换站 |
| MuSGD 比 SGD 稳定得多；也用了 STAL | AT | Ultralytics 8.4.30 自带 STAL；`optimizer=auto` 只有总迭代数超过 1 万才选 MuSGD。我们固定为 AdamW、lr0 0.000588，避免数据量变化时悄悄换优化器 |
| Albumentations 带来明显提点，但显存和速度代价大 | AT | 仿真 A/B：AP50 0.815 对 0.826，熄灭类召回 0.715 对 0.745，训练多花 15% 时间。仿真验证集没有压缩痕迹，看不出它对真实相机的作用，暂未启用。Ultralytics 默认的增强列表里有 ToGray，和颜色标签冲突，必须用白名单 |
| 不要追参数量和 GFLOPs：DWConv 让 GFLOPs 下降，在 Orin NX 上反而更慢，因为 NVIDIA 对标准卷积优化得更好 | AT | 对 RK3576 NPU 是否同样成立没测过 |
| batch 不宜太大 | RP24 | 我们用 batch 16，Ultralytics 会累积梯度到等效 batch 64 |
| 关键点不稳时，依次查数据集、损失函数、多任务损失权重，最后才调参数 | RP24 | — |
| 按综合指标选权重（框 mAP50 + 颜色准确率 + 关键点精度），不用 last.pt；他们训 250 轮，最好的在第 127 轮，末期验证损失回升 | SZU（AdamW，batch 16） | 我们训 100 轮时验证损失还在降，暂时用 last.pt；有真实验证集后改用 `nt.eval` 的综合指标从快照里挑 |

### 8.4 评估经验

- **不用标注的真实评估（SZU 的"六视频一致性测试"）。**
  - 做法：录只出现工程装甲的视频，置信度阈值 0.5。任何非工程的输出都算误识别，再加上 PnP 解算结果（yaw、pitch、距离）的帧间残差。
  - v8n：19,225 个检出，7 次类别错误（0.036%），yaw 0.694°、pitch 0.104°、距离 63.97 mm。
  - v5n：18,281 个检出，0 次错误，0.725°、0.112°、69.90 mm。
  - **用在 INT8 验收上：** 第 4.3 节第 4 步上板时，可以录"所有装甲都显示同一个编号"的视频，FP16 和 INT8 各跑一遍，比较误识别率和 `nt.replay` 的 PnP 抖动，不需要标注。
- **误检通常只闪一帧（RP24）。** 自瞄要求连续 3 帧识别才锁定，就能挡住大部分误检；哨兵要把置信度阈值调高。INT8 后置信度会整体下降，少数检测掉得很多（第 2.2 节），所以这个连续帧数和阈值要在 INT8 模型上重新确定。
- **不能用张量相似度评估**（本文第 0 节第 4 条）。三家都没提过这一点，是我们自己的教训。

### 8.5 输出和预处理约定对照

| | 角点顺序 | 预处理 | 后处理 |
|---|---|---|---|
| AT | README 写"顺时针"，但我们在 sleepy 解码器上实测，两个 praysky 模型在 2888 + 1295 个检测里全部输出 LT、LB、RB、RT | RGB、除以 255、letterbox 补边 114 | 端到端 TopK，或图内 NMS |
| RP24 | 从左上开始逆时针，在图像坐标里就是 LT、LB、RB、RT | — | 置信度是 logit，部署端要做 sigmoid |
| SZU 26 | LT、LB、RB、RT | letterbox 补边 124；还原后超出原图的候选直接丢弃 | 按关键点外接矩形做 NMS，IoU 0.2；按输出张量形状自动选后处理，形状不认识就报错 |
| net_training | LT、LB、RB、RT，"边缘"像素约定 | RGB、除以 255；768×576 输入，4:3 画面直接缩放不补边 | 契约 v2 [1,30,18]，`contract.json` 嵌在模型里，C++ 对拍 |

颜色顺序有出入：RP24 的 README 写的是"红蓝灰紫"，SZU 26 文档对同系列模型写的是"蓝、红、白、紫"。如果要拿 0526 做对照，颜色通道的顺序应该实测确认，不能照抄文档。

### 8.6 对 net_training 做 RKNN INT8 的具体启示

1. **先用现有的 CoordAtt 导出做一次 PTQ 对比。** 现成的模型有 `exports/y26n_simm7_coordatt_s0`、`exports/y26n_simm7_coordatt_s1`，用第 7 节的 `rknn_check_y26.py` 跑一遍，和第 2.2 节的 C2PSA 结果比较。重点看阈值 0.5 下"仍高于阈值"的比例，C2PSA 是 87.1%。如果 CoordAtt 明显更好，RKNN 用 CoordAtt，TensorRT FP16 继续用 C2PSA。这一步不需要训练，几分钟就能跑完。
2. **QAT 从一开始就计划在内，按 w8a8 做。** RK3576 不支持 AT 用的 w8a16。
3. **继续用"切到检测头、解码放 CPU"的导出方式。** AT 的做法和本文第 4.1 节一致，`nt.export` 需要加这个导出变体。
4. **做 QAT 前先清洗数据。** 量化会放大脏数据的影响（AT 前哨站的例子）。仿真数据的标注在几何上是精确的，但"太小、太暗看不清编号却仍带着编号标签"的样本，在量化后可能更容易出问题。
5. **用 INT8 输出选权重和阈值。** QAT 的快照要用 INT8 推理结果、按部署阈值的检测指标来挑，不能用 FP32 的 last.pt。
6. **上板验收加上 SZU 式的单编号视频测试**（第 8.4 节）。

详细的原始摘录、帖子正文的获取方法和许可证见 net_training 的记忆文件 `community-armor-training-lessons.md`、`community-armor-model-sources.md`。许可证：AT 权重 AGPL-3.0，论坛帖子 CC BY-NC-SA 4.0，SZU 仓库 MIT。
