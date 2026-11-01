# 官方装甲样式公布后的操作流程

从"官方公布点阵屏样式"到"真实数据训练出的模型上车"，按顺序执行。每一步都写了命令、用时和过关标准，没过关就不要进入下一步。

训练在租用的 GPU 上做（第 5、6 步），其余在本机。用时按本机（RTX 5060 8 GB）实测推算，标"估计"的没有实测过。本机的重任务都用 `scripts/run_capped.sh` 包一层。每天开始和收尾各跑一次 `python -m nt.doctor`，确认没有 FAIL。

| 阶段 | 内容 | 用时 |
|---|---|---|
| 1 | 读懂官方样式，做三个决定 | 0.5–2 小时 |
| 2 | 写屏幕规格、改标签规范 | 1–3 小时（估计） |
| 3 | 仿真冒烟：40 张，人工看图 | 15 分钟 |
| 4 | 生成正式仿真数据 | 3000 张约 10 分钟，1 万张约 35 分钟 |
| 5 | 打包，在云端烘焙、训练、评估、导出 | 打包几分钟；训练在本机是 78 分钟（3000 张），云端更快（估计） |
| 6 | 取回结果、C++ 对拍、测延迟 | 约 15 分钟 |
| 7 | 代理屏真实验证（不用等实物） | 半天 |
| 8 | 接入 sleepy、回放、上车 | sleepy 改造进度决定 |
| 9 | 实物到手：录制、标注、第二版训练 | 数天，按轮迭代 |

## 1. 读懂官方样式

把官方给的信息逐项记下来：

- 点阵有效区的尺寸，点阵行列数，点的形状和占空比。
- 发光颜色和亮度，熄灭时是什么样子。
- 每个编号显示的字形；哨兵、前哨站、基地的图案。
- 有没有动态内容：受击闪烁、血量、倒计时、底色变化。
- 模块的外形、厚度、边框，以及装在车上的位置和角度。
- 官方有没有给尺寸图纸、角点或识别区域的定义。

然后做三个决定：

1. **角点定义。** 选在图像里边界清楚、三维尺寸确定的四个点。通常是点阵有效区外框，或者屏幕玻璃外框。这个定义会写进标签规范和 PnP 尺寸，定下后不要再改，改了就要重新生成和标注全部数据。
2. **走哪条路。**
   - A 路：屏幕是一块平面点阵，装在原来板面的位置，内容按编号静态显示，角点就是点阵外框。只需要填规格文件，直接进入第 2 步。
   - B 路：外形或尺寸和现在的板面差别很大、有动态内容，或者角点定义不是点阵外框。需要先改 Daedalus：外形要改 glTF 模型（估计 1–3 天），动态内容要改 `src/robomaster/armor/screen.rs`，角点定义要改导出器取四边形的方式。改完跑 `cargo test` 再进入第 2 步。
3. **类别是否变化。** 现在是哨兵、1、3、4、前哨站、基地，加一个熄灭类，共 13 类。如果官方阵容或显示内容有变，在第 2 步一起改。

## 2. 写屏幕规格、改标签规范

```bash
cp configs/screens/rm27_hypothesis_a.toml configs/screens/rm27_official_v1.toml
```

在 `rm27_official_v1.toml` 里：

- 把 `name` 改成 `rm27_official_v1`，`corner_definition` 写第 1 步定下的定义。
- 每个 `[下限, 上限]` 填官方值。确定的值上下限设成一样；有公差的保留小范围；亮度和熄灭、闪烁的比例保留随机范围，因为相机曝光和屏幕实际亮度都会变。
- 官方字形和图案做成 PNG，亮的地方表示点亮，文件名必须是 `sentry.png`、`one.png`、`three.png`、`four.png`、`outpost.png`、`base.png`。放进一个新文件夹（例如 `configs/screens/patterns_official/`），把 `content.bitmap_dir` 指过去。哪个编号有 PNG 就用 PNG，没有的才用内置的 5×7 字库。

改标签规范：

- [configs/label_spec.yaml](../configs/label_spec.yaml)：`corner_definition` 改成新定义。类别有变时，同时改 `labels`；然后改四个模型 yaml 里的 `nc`（`configs/models/*.yaml`）；再按新类别数重新确定 `pose_default.yaml` 里的 `lr0`，类别数一变 `tests/test_train_config.py` 就会失败，提醒你这一步。
- [configs/geometry.yaml](../configs/geometry.yaml)：`corner_definition` 和 `size`（角点四边形的物理宽高，单位米）改成和新定义一致。PnP 回放评估和 sleepy 解算都用它。

注意：
- `label_spec.yaml` 一改，哈希就变了，之前入库的所有数据集都会被烘焙拒收，必须重新入库。这是有意的。
- 跑一遍测试，全部通过才继续：
  ```bash
  python -m unittest discover -s tests -p 'test_*.py'
  ```

## 3. 仿真冒烟

```bash
cp configs/sim/m7_smoke_screen.toml configs/sim/official_smoke.toml
# 改两处：out_dir = ".../runs/sim_official_smoke"，[armor] screen_spec = "../screens/rm27_official_v1.toml"
scripts/run_sim_dataset.sh configs/sim/official_smoke.toml 600
python -m nt.ingest --src runs/sim_official_smoke --format canonical-jsonl --source sim \
    --name offsmoke --dst datasets/sim/offsmoke --link
python -m nt.viz datasets/sim/offsmoke --n 30 --out runs/viz/offsmoke.png
```

把 `runs/viz/offsmoke.png` 和官方图片放在一起看。过关标准：

- 0 号角点在左上，四边形贴着第 1 步定义的边界。
- 每块装甲显示的字形或图案和标签一致，和官方图片一致。
- 有熄灭的屏，标成灰色。
- 前哨站、基地显示各自的图案。
- 入库报告里拒收很少：以前这类拒收约 3%，基本都是接近侧视的帧。

## 4. 生成正式仿真数据

```bash
cp configs/sim/m7_train_screen.toml configs/sim/official_train.toml
# 改：out_dir = ".../runs/sim_official"，screen_spec 指向官方规格；samples 按下面选
scripts/run_sim_dataset.sh configs/sim/official_train.toml 3600
python -m nt.ingest --src runs/sim_official --format canonical-jsonl --source sim \
    --name simoff --dst datasets/sim/simoff --link
# 可选：在本机先烘焙一小部分看看效果，正式烘焙在云端做
scripts/run_capped.sh 6 -- python -m nt.bake --src datasets/sim/simoff --out baked/simoff_check --limit 100 --workers 6
python -m nt.viz baked/simoff_check --baked --n 12 --cols 4 --out runs/viz/simoff_check.png
```

- 样本数：3000 张是实测过的规模，100 轮训练后接近平台，编号准确率约 78%。社区经验是每类约 2000 张图，按此 1 万张更合适。数据量实验还没做，第一版可以先用 3000 张出结果，同时渲染 1 万张。
- 上传的是入库后的数据集，3000 张约 2.9 GB。烘焙结果（约 4.2 GB）在云端生成，不用上传。
- 镜头畸变：`configs/bake/lens.yaml` 给 80% 的仿真帧加上车上自瞄相机（sleepy 的标定）的畸变，图像和角点一起变换。真实相机的画面本来就有畸变，所以第一版建议打开：打包时加 `--bake-config configs/bake/lens.yaml`。它在仿真验证集上看不出好坏（验证集也按同样方式烘焙），要等第 7 步的真实验证集来判断。
- 看一眼烘焙图：ROI 裁剪、干扰物和相机效果都正常。

## 5. 在云端训练第一版

训练都放在租用的 GPU 上。本机只负责打包，烘焙也在云端做。

```bash
scripts/pack_for_cloud.sh --name simoff_v1 --data datasets/sim/simoff
# 要做结构 A/B 时：--models yolo26n-pose-armor,yolo26n-pose-armor-coordatt --seeds 0,1
# 要从假设样式的模型微调时：--init runs/train/y26n_simm7_e100/weights/last.pt（类别表没变才行）
```

它会生成 `cloud/simoff_v1/`，里面有代码、数据集、`job.json`、`cloud_run.sh`、`README.md` 和 `SHA256SUMS`。打包前请先提交代码：`job.json` 记录提交号，有未提交的文件时会逐个列出来。

租机器的要求：
- NVIDIA 驱动 570.26 或更新（CUDA 12.8 的要求）；
- Python 3.10；
- 能访问 PyPI 和 download.pytorch.org，国内机器可以用 `NT_PIP_INDEX` 指定镜像，所有文件都按哈希校验，换镜像不影响安全；
- 内存 32 GB 以上。内存不够时，数据加载进程会自动减少。

把整个目录传上去，在机器上执行：

```bash
cd simoff_v1
nohup bash cloud_run.sh > cloud_run.log 2>&1 &
tail -f cloud_run.log
```

`cloud_run.sh` 依次执行：
1. 校验文件，解包；
2. 按带哈希的 `requirements.lock` 装环境；
3. 记录 GPU 和驱动；
4. 跑单元测试和 `nt.smoke_m0`（这一步不过就不训练）；
5. 检查数据集和标签规范是否一致；
6. 烘焙；
7. 对每个模型和种子：训练、评估、导出、契约检查；
8. 打包结果到 `results/simoff_v1_results.tgz`。

断线或中断后，再执行同一条命令会从断点继续，已完成的步骤会跳过。某一次训练失败不会挡住其他训练。

- 默认用 `configs/train/pose_cloud.yaml`：100 轮，AdamW 已固定，数据加载 workers 为 8，其余和本机一致。
- 用时（估计）：3000 张、100 轮，在本机 RTX 5060 上是 78 分钟，云端更快的 GPU 应该更短。第一次在某种 GPU 上跑时，记下实际用时。

## 6. 取回结果、C++ 对拍、测延迟

下载 `results/simoff_v1_results.tgz` 和同名的 `.sha256`，在本机执行：

```bash
scripts/receive_cloud_results.sh simoff_v1_results.tgz
python -m nt.bench_trt exports/simoff_v1_yolo26n-pose-armor_s0/model.onnx --out runs/bench/simoff_v1.json
```

部署用的是 TensorRT FP16。用同一批验证图对比 FP16 引擎和 ONNX 的 float32 输出：

```bash
python -m nt.eval --weights exports/simoff_v1_yolo26n-pose-armor_s0/model.onnx --backend trt-fp16 \
    --data baked/<烘焙目录> --out runs/eval/simoff_v1_trt_fp16.json
```

`vs_onnx` 一项给出引擎让角点和置信度移动了多少。假设样式模型的参考值：角点差 p95 0.21 网络像素（原图 0.36 px），置信度差 p95 0.004，对指标的影响在 0.01 以内。明显大于这个量级，就要查是哪一层在 FP16 下出了问题。云端结果包里没有烘焙图，所以这一步要在本机先烘焙一份验证集（`nt.bake --src ... --limit` 即可）。

取回脚本会：
- 校验文件；
- 解包到 `runs/train/`、`runs/eval/`、`runs/cloud/`、`exports/`，不会覆盖已有内容，也会拒绝路径不安全的文件；
- 对每个导出的模型跑 sleepy 的 C++ 解码器对拍。云端没有 sleepy，所以对拍只能在本机做。

过关标准：

- 每个模型都要通过 C++ 对拍。不过就不能交给 sleepy。
- 仿真验证集上的指标不应该比假设样式的模型差。假设样式模型的参考值（置信度 0.2）：AP50 0.827，类别准确率 84%，机器人编号准确率 78%，熄灭类召回 75%，角点 p95 6.2 px（原图像素）。官方规格是确定值，变化范围比假设规格窄，正常情况应该更好。
- 结果偏低时，先查第 3 步的标注图和第 4 步的烘焙图，再考虑调训练参数。云端每一步的日志都在 `runs/cloud/<任务名>/logs/`。
- 选权重：仿真阶段用 `last.pt`；有真实验证集以后，按真实验证集的综合指标从每 5 轮存一次的快照里挑。快照都在结果包里。
- 本机想复现某一步，用同样的命令加 `scripts/run_capped.sh 8 --` 前缀，workers 用 2（`pose_default.yaml` 的默认值）。

## 7. 代理屏真实验证（不用等实物）

用显示器显示官方样式的点阵，用车上相机录下来，靠 ArUco 码自动标注，得到第一个真实相机下的验证集。

```bash
python -m nt.proxy render --out proxies/off_3_red --label three --color red --text 3 \
    --cols <官方列数> --rows <官方行数>
# 全屏显示 proxies/off_3_red/proxy.png，sleepy 开 record.enable 录一段，换角度、距离、曝光
python -m nt.session create --src ~/sleepy/record/auto_aim/<时间>.avi --name proxy_off_3_red \
    --role val --camera ~/sleepy/config/<车>.yaml
python -m nt.proxy label --session sessions/proxy_off_3_red --proxy proxies/off_3_red/proxy.json \
    --dst datasets/real/proxy_off_3_red
scripts/run_capped.sh 6 -- python -m nt.bake --src datasets/real/proxy_off_3_red --out baked/proxy_off_3_red
python -m nt.eval --weights $W --data baked/proxy_off_3_red --out runs/eval/v1_on_proxy_off_3_red.json
python -m nt.replay sessions/proxy_off_3_red --model exports/y26n_simoff_v1/model.onnx --out runs/replay/v1_proxy
```

- 每个编号、红蓝两色都录一段。`--role val` 的会话永远不会进训练。
- 代理屏用的是内置 5×7 字库，和官方字形可能不同，也没有真实模块的外形。它只用来衡量"仿真训练出的模型在真实相机下掉多少"，不能代替实物。
- `--camera` 指向的配置，其内参分辨率必须和录像一致。hero.yaml 目前就对不上（相机 1440×1080，内参按 1280×1024 标定）。

## 8. 接入 sleepy、回放、上车

模型交给 sleepy 之前，这些要先改好（详见记忆里的 sleepy 审计结果）：

- 关掉灯条数字分类器和 R−B 颜色判断，它们会覆盖网络输出。
- PnP 改用第 2 步的新角点尺寸，1 号不再按大装甲解。
- 熄灭类（编号 UNKNOWN、颜色 NONE）交给跟踪器，不要丢掉。
- 解码器按 `contract.json` 读取；预处理和训练一致：RGB、除以 255、双线性缩放、4:3 画面不补边。
- 修 CUDA letterbox 的半像素偏移和补边值；推理并发数不要超过上下文数。
- 车上电脑用 `nt.bench_trt` 测 FP16 延迟；FP16 下的精度还没有验证过，要在这一步一起确认。

上车前，先用 sleepy 的回放模式（`camera.type: replay`）把代理屏录像接回整条链路跑一遍，确认识别、跟踪、瞄准都正常。

## 9. 实物到手：真实数据和第二版

录制：

- sleepy 开 `record.enable: true`。曝光、增益设置和比赛一致（低曝光、高增益）。能无损录制就无损录制。
- 按用途分会话录：
  - `train`：可以进训练；
  - `val`：真实验证集，换一天、换场地或换车单独录，永远不进训练；
  - `hard`：模糊、暗、遮挡；
  - `neg`：场地 LED、显示器、灯条模块、兑换站，画面里没有装甲。
- 录一些"所有装甲都显示同一个编号"的视频：任何别的编号输出都是误识别，不用标注就能统计。

建会话、预标注、人工审核、入库：

```bash
python -m nt.session create --src <录像>.avi --name s1201_infantry_a --role train --camera ~/sleepy/config/<车>.yaml --every 3
python -m nt.annotate export --session sessions/s1201_infantry_a --out annot/s1201_infantry_a \
    --model exports/y26n_simoff_v1/model.onnx --batch-size 200
# labelme 里逐帧修正并勾选 reviewed，规范见 docs/labeling_guide.md
python -m nt.annotate status annot/s1201_infantry_a
python -m nt.annotate check  annot/s1201_infantry_a
python -m nt.annotate import annot/s1201_infantry_a --dst datasets/real/s1201_infantry_a
python -m nt.viz datasets/real/s1201_infantry_a --n 30 --out runs/viz/s1201_infantry_a.png
```

训练第二版：仿真和真实一起烘焙，真实数据加权，从第一版接着微调，按真实验证集选模型。

```bash
scripts/pack_for_cloud.sh --name mix_v2 --data datasets/sim/simoff --data datasets/real/s1201_infantry_a:3 \
    --data datasets/real/s1203_val_b --init runs/train/simoff_v1_yolo26n-pose-armor_s0/weights/last.pt
# 上传、cloud_run.sh、receive_cloud_results.sh，同第 5、6 步
```

- 微调的轮数和学习率要专门试，上面的 100 轮只是起点。
- YOLO26n 还是 YOLOv8n、C2PSA 还是 CoordAtt，都在这一步按真实验证集来定。A/B 在云端做：打包时 `--models` 列出几种结构，`--seeds 0,1`。
- 之后每一轮迭代：
  1. `nt.mine` 找坏例；
  2. `nt.annotate export --frames` 复审；
  3. 入库成新版本数据集，入库后的数据集不再修改；
  4. 重新训练。

## 每一版模型都要留档

- 训练前先提交代码，在提交说明里写上模型名。`nt_meta.json` 只记录配置和数据的哈希，不记录代码版本。
- 留下这些：`runs/train/<名字>/nt_meta.json`、`runs/eval/<名字>.json`、`exports/<名字>/contract.json`，以及交给 sleepy 的那份 ONNX 的哈希。
- 车上出问题时，凭这些找到是哪份数据、哪版代码、哪个模型。
