# 样式 A→B 迁移实验

本轮按用户授权执行三组单种子实验。A 权重为
`runs/train/y26n_simm7_e100/weights/last.pt`，不重新训练 A。

## 固定实验条件

- 模型：YOLO26n-pose-armor，保留 C2PSA，13 类、四角点。
- B：七段数字 1/3/4、64×48 方点、字形占屏高 0.65～0.80。
- 屏幕尺寸、颜色、距离和成像增强沿用 A；视场角使用历史 A 数据的 35°～55°。
- 训练场景：种子 101，2,700 张，入库用途为 train，每张烘焙两份。
- 验证场景：种子 202，600 张，入库用途为 val，每张只烘焙一份。
- 检查场景：种子 303，40 张，不进入正式训练或验证。
- 统一烘焙种子 0、768×576 输入、batch 16、AdamW、lr0=0.000588、全模型训练。
- 本轮保留 A 历史屏幕刷新条纹分布用于控制变量，不把它视作真实全局快门相机的模型。

| 路线 | 初始化 | 轮数 | 种子 |
|---|---|---:|---:|
| b_scratch_e100_s0 | 随机 | 100 | 0 |
| b_from_a_e30_s0 | A 权重，重新建立优化器 | 30 | 0 |
| b_from_a_e100_s0 | 同一份 A 权重，重新建立优化器 | 100 | 0 |

两组微调相互独立。30 轮使用独立学习率计划，不能用 100 轮组的中间快照替代。
配置中的 `pretrained: false` 禁用的是自动取得通用预训练权重；显式传入
`--model .../last.pt` 仍会加载 A 权重。单轮预检已确认完整迁移 879/879 项参数，
并成功保存 `milestone_001.pt`；该预检不计入三组对比结果。
两条 100 轮路线的参数一致，用于比较初始化的作用。所有路线使用同一份
`baked/style_b_v1`，最终检查点作为主结果，不按验证结果反复修改 B。

## 数据准备

```bash
.venv/bin/python scripts/make_style_b_patterns.py
scripts/run_sim_dataset.sh configs/sim/style_b_smoke.toml 600
scripts/run_sim_dataset.sh configs/sim/style_b_train.toml 3600
scripts/run_sim_dataset.sh configs/sim/style_b_val.toml 1200
scripts/run_capped.sh 3 -- python -m nt.ingest --src runs/sim_style_b_train --format canonical-jsonl --source sim --name style_b_train --dst datasets/sim/style_b_train --role train --link
scripts/run_capped.sh 3 -- python -m nt.ingest --src runs/sim_style_b_val --format canonical-jsonl --source sim --name style_b_val --dst datasets/sim/style_b_val --role val --link
scripts/run_capped.sh 6 -- python -m nt.bake --src datasets/sim/style_b_train --src datasets/sim/style_b_val --out baked/style_b_v1 --copies 2 --seed 0 --workers 4
```

首次检查暴露了仿真器的固定点阵纹理更新问题：Bevy 0.19 对同尺寸 GPU
纹理的复用只上传 mip 0，远处使用的其他 mip 层仍显示旧数字。增加
settle_frames 也无法修复。已在仿真器 `mask_image` 中禁用该复用途径，
确保上传完整 mip 链；不修改类别标签来掩盖渲染错误。

修复前的检查数据保留在 `runs/sim_style_b_smoke`、`runs/sim_style_b_settle16`
及对应入库目录，仅用于复现错误，严禁用于训练。修复后的检查输出为
`runs/sim_style_b_smoke_fixed`。

本轮正式数据审计结果如下，入库拒收的样本不参与烘焙：

| 分区 | 原始图像 | 入库接受 | 烘焙图像 | 场景种子 |
|---|---:|---:|---:|---:|
| 训练 | 2700 | 2612 | 5224 | 101 |
| 验证 | 600 | 579 | 579 | 202 |

训练和验证来源无交叉，两者均覆盖全部 13 类。验证集烘焙后的标注共含
1610 个装甲，其中红蓝 1/3/4 号共 848 个。完整计数、仿真器二进制哈希及
修复差异分别保存在实验目录的 `data_audit.json` 与 `simulator.patch`。

## 训练与统一评估

```bash
.venv/bin/python scripts/run_style_transfer_ab.py --dry-run
.venv/bin/python scripts/run_style_transfer_ab.py --memory-gb 7
```

执行器先评估 A 在 B 上的直接表现，再串行训练三组；每个重任务通过
`run_capped.sh` 限制内存。实验输入哈希写入 `inputs.json`，完成结果只有
在输入一致时才允许复用；中断目录保留，不自动删除或覆盖。

`nt.train --milestones 10 30 60` 按实际完成轮数保存额外检查点，
避免把 Ultralytics 从零编号的 epoch10.pt 错当成训练了十轮。
两条长训练额外评估 10、30、60 轮；短训练额外评估 10 轮。

主结果固定使用端到端输出、置信度 0.2、最终 last.pt。输出包括：

- `runs/experiments/style_b_v1_s0/status.json`：当前步骤、进度和退出状态。
- 同目录各路线 JSON：AP50、机器人类别准确率、角点误差、误检和小目标分组。
- `comparison.json`、`comparison.md`：最终指标与实测训练分钟数。
- `learning_curves.csv`：各路线逐轮训练时间及 Ultralytics 验证指标。
- `transfer_curves.png`、`transfer_curves.svg`：按实际训练分钟数绘制的独立评估曲线。

报告还单独统计更换了字形的红蓝 1/3/4 号，以及宽度小于 16 网络输入像素
的小装甲，避免整体均值掩盖这些类别的问题。所有结果完成后，可单独重新
生成报告及曲线，不触发训练或推理：

```bash
.venv/bin/python scripts/run_style_transfer_ab.py --report-only
```

预先约定的初步判断：30 轮微调与 100 轮从头训练相比，AP50 损失不超过
0.01、机器人类别准确率损失不超过 1 个百分点，并检查角点 P95 和误检是否恶化。
该阈值仅用于工程筛选；单种子、单个假设 B 不能证明正式样式或真实数据的收益。

其中 `summary.robot_numbers.class_acc` 要求颜色和编号同时正确，范围为哨兵
及 1/3/4 号，分母仅含已匹配到的装甲；它不是纯数字准确率，也不是包含漏检
的端到端识别成功率。召回率和误检数采用不区分类别的 IoU 匹配，角点误差
采用已匹配装甲的可见角点，并换算到原图像素。独立评估 AP50 与训练日志
中的 Ultralytics mAP50 使用不同实现，不能跨口径比较。

## 本轮实测结果

2026-11-05，在 RTX 5060 Laptop 8 GB 上完成三组训练和 11 份独立评估。
三组训练阶段合计 228.64 分钟，约 3 小时 49 分钟；计时包含训练中的常规
验证和保存，不包含 A 的历史预训练、数据准备及独立评估。

| 路线 | B 训练分钟 | AP50 | 机器人类别准确率 | 角点 P95 / 原图 px | 误检数 |
|---|---:|---:|---:|---:|---:|
| A 直接评估 B | 0 | 0.8337 | 80.15% | 5.332 | 475 |
| B 从头训练 100 轮 | 99.51 | 0.8793 | 88.61% | 5.271 | 402 |
| A→B 独立微调 30 轮 | 29.89 | 0.8966 | 87.65% | 4.905 | 400 |
| A→B 独立微调 100 轮 | 99.24 | 0.9119 | 89.64% | 4.735 | 284 |

本轮结果支持复用 A 权重来缩短适配时间。30 轮微调节省 69.96% 的 B
训练时间，AP50 提高 0.0173、角点 P95 降低 0.366 像素，机器人类别准确率
下降 0.96 个百分点，刚好达到预设的 1 个百分点容差。只统计变更字形的
1/3/4 号时，AP50 为 0.9067，对比基线 0.9073 基本持平，但类别准确率
下降 1.50 个百分点。因此它适合作为快速适配候选，不能称为各项指标都胜过基线。

100 轮微调相对基线，AP50 提高 0.0326、机器人类别准确率提高 1.03 个
百分点、角点 P95 降低 0.536 像素，误检从 402 降至 284，减少约 29.4%。
更换字形的数字 AP50 也从 0.9073 提高至 0.9327。本次没有观察到整体指标
上的负迁移，可以将该组最终权重作为当前 B 样式的完整训练候选。

小目标仍是弱项：宽度小于 8 网络输入像素时，100 轮微调的召回率从
77.36% 提高至 82.70%，但匹配后类别准确率为 76.43%，与基线 76.83%
相比没有改善。8–16 像素的类别准确率则从 88.39% 提高至 91.16%。

长程组的第 60 轮检查点已达到 AP50 0.9085、机器人类别准确率 89.74%、
角点 P95 4.806 像素；再训练至 100 轮，AP50 增加 0.0034，类别准确率
下降 0.10 个百分点。它提示后半程的收益已经较小，但本轮仍按预先约定
使用第 100 轮作为主结果，没有事后更换比较终点。

结论限于单个种子、一个假设 B 和同一仿真域。B 更换了数字和点阵布局，
哨兵、前哨站、基地图案及角点语义仍保持原设置；不能据此保证官方完整
样式或真实相机上的迁移收益。后续优先用官方样式和真实采集数据复核。

完整产物：

- [对比报告](../runs/experiments/style_b_v1_s0/comparison.md)
- [收敛曲线](../runs/experiments/style_b_v1_s0/transfer_curves.png)
- [输入、轮数与评估检查](../runs/experiments/style_b_v1_s0/verification.json)
- [30 轮快速适配权重](../runs/train/b_from_a_e30_s0/weights/last.pt)
- [100 轮完整微调权重](../runs/train/b_from_a_e100_s0/weights/last.pt)

实现验证包括训练配置 7 项测试、烘焙 12 项测试、单轮真实 GPU 预检、
固定点阵重新渲染检查及三组正式实验。最终核对了输入哈希、实际训练
参数、完整轮数及全部评估对应的权重哈希；结果保存在 `verification.json`。
