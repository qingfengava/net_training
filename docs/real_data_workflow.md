# 真实数据流程（M6）

从录像到可训练数据，再到找坏例和回放评估。每一步的命令行参数见对应模块开头的说明，例如 `python -m nt.annotate --help`。

## 1. 采集会话

一段录像，或者一个装静态图的文件夹，就是一个会话（`nt.session`）。一个会话只有一种用途：

| role | 用途 |
|---|---|
| `train` | 可以进训练 |
| `val` | real_val，用来做模型决策，永远不进训练 |
| `hard` | real_hard，糊、暗、遮挡，单独统计 |
| `neg` | 必须一块装甲都没有的场景（场地 LED 屏、显示器、灯条模块），检出就是误检 |

同一段录像里的帧高度相关，所以训练集和验证集按会话切，不按帧切。烘焙时 `val`、`hard`、`neg` 只会进 val 分割，进不了 train（`nt/bake.py` 的 `split_of`）。

    python -m nt.session create --src 录像.avi --name s20261201_hero_a --role train --camera ~/sleepy/config/hero.yaml --every 3

- 带上 `--camera`，并指向录像所用相机的 sleepy 配置（读其中的 `camera_info`）。有了内参，回放评估才能算 PnP。
- 录像尽量无损。sleepy 现在用 MJPG 录像，H.264 和 MJPG 都会把点阵屏的细节压糊。会话会把有损编码记成警告。
- sleepy 内录（`record.enable: true`）会在 `<时间>.avi` 旁边写一个同名的 `<时间>.txt`，第 i 行是第 i 帧的 `t qw qx qy qz`。会话默认读取它：每帧的 `t_s` 用录像器记的真实时间，`q_wxyz` 是该帧的云台姿态（gimbal_odom 系），`--start`/`--end` 也按这个时间算。录像器写盘忙时会丢帧，所以不能用"帧号 ÷ 帧率"推时间。txt 里有坏行会直接报错（只有被 `kill -9` 截断的最后一行例外），行数和帧数对不上会记警告。txt 不在旁边时用 `--poses` 指定，`--no-poses` 则忽略它。
- 录给标注用的会话可以加大 `--every`；录给回放评估用的会话用 `--every 1`。

## 2. 预标注和分批

    python -m nt.annotate export --session sessions/S --out annot/P --model exports/最新/model.onnx --batch-size 200

- 置信度达到 `--conf`（默认 0.5）的检测，写成普通标签。
- 置信度在 `--review-conf` 和 `--conf` 之间的、几何可疑的、被图片边缘切断的，写成 `?` 标签，人必须处理。
- 没有模型时只导出空帧。现在可以用 sleepy 的 praysky 模型在旧录像上试跑流程。

每批是一个独立的文件夹，可以整个交给一个人。标注规则在 [labeling_guide.md](labeling_guide.md)。

    python -m nt.annotate status annot/P     # 每批的进度
    python -m nt.annotate check  annot/P     # 入库前的自检

## 3. 入库

    python -m nt.annotate import annot/P --dst datasets/real/P_v1

- 只有勾了 `reviewed` 的帧会入库，并且照常经过入库闸门（顺序、凸性、重复等检查）。
- 还有没做完的帧时会拒绝导入，除非加 `--allow-partial`。
- 数据集不可变。修正后要用新名字（`_v2`）重新导入。

## 4. 坏例挖掘

没有标注的录像：找模型拿不准、前后帧不一致（忽有忽无、类别跳变、角点跳动）、几何不合法的帧，导出成新批次去标。

    python -m nt.mine session sessions/S --model M.onnx --out runs/mine/S.txt --top 200
    python -m nt.annotate export --session sessions/S --frames runs/mine/S.txt --model M.onnx --out annot/S_mined

已标注的数据：找模型和标注差距大的帧（漏检、多检、类别不一致、角点差距大）。差距大的地方往往是标错了，导出复核批次，改好后导入为新版本。

    python -m nt.mine dataset datasets/real/D --model M.onnx --out runs/mine/D.txt
    python -m nt.annotate export --dataset datasets/real/D --frames runs/mine/D.txt --model M.onnx --out annot/D_review

## 5. 回放评估

不需要标注。逐帧跑模型，按 sleepy 的方式解 PnP，统计以下几项：
- 跟踪连续性：断帧、类别跳变。
- 距离、yaw、pitch 的帧间抖动。
- PnP 的 yaw 突跳。
- `neg` 会话里每分钟的误检数。

比较两个模型时，要用同一个会话。

    python -m nt.replay sessions/S --model M.onnx --out runs/replay/S_M --plot

PnP 用的物理尺寸在 `configs/geometry.yaml`。官方样式出来后，要和 `corner_definition` 一起更新。

## 6. 代理点阵屏

官方装甲到手之前，用显示器模拟一块点阵屏，得到真实镜头下带精确角点的图：

    python -m nt.proxy render --out proxies/red3 --label three --color red

1. 把 `proxy.png` 在显示器上 100% 缩放全屏显示，用机器人相机绕着拍，建成一个会话。
2. 运行下面的命令。它用四角的 ArUco 标记解出单应矩阵，自动算出点阵外框的四个角，不用人标。

    python -m nt.proxy label --session sessions/S --proxy proxies/red3/proxy.json --dst datasets/real/proxy_red3_v1

在合成图上实测，平均角点误差约 0.17 px，带镜头畸变时最大 0.66 px。

这只是代理。显示器的像素结构、亮度和视角衰减都和 LED 点阵不一样，所以这类数据集单独命名，默认不进训练。它的用途是：
- 看真实相机把点阵拍成什么样（摩尔纹、模糊、光晕）；
- 验证角点定义；
- 测试仿真训练出来的模型在真实光学下的表现。
