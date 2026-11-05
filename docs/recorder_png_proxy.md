# Recorder PNG 导入与 A/B 代理屏

2026-11-05：已打通 Recorder PNG → 采集会话 → 预标注/代理屏标注 → 入库。
以下命令在 net_training 项目根目录执行；显示器代理屏仍需使用真实相机采集，不能当作官方 LED 装甲板。

## 导入无损会话

sleepy 使用 `record.mode: png_sequence`，输出结构为：

```text
录制会话/
  frames/frame_000000000.png
  frames/frame_000000001.png
  frames.jsonl
  poses.txt
```

导入整段连续会话：

```bash
.venv/bin/python -m nt.session create \
  --src /实际路径/录制会话 --name proxy_b_red3_s01 --role val \
  --camera /实际路径/机器人配置.yaml --every 1 --dedupe-hamming -1
.venv/bin/python -m nt.session show sessions/proxy_b_red3_s01
```

也可以传入该会话的 `frames/` 子目录或 `frames.jsonl`，都会自动定位根目录。

- JSONL 为图像、时间与姿态的唯一配对依据。原始 `time_offset_s` 完整保存在 `t_s`，不回退到名义帧率、不截断小数；四元数归一化后保存在 `q_wxyz`。
- 保留 `write_index`、`camera_frame_id`、`source_image`。`index` 使用写入序号，经过 `--start/--end/--every/--max` 筛选后不会重新编号。
- `source_pixel_format` 表示采集端图像格式；`pixel_format` 表示 OpenCV 读取 PNG 后的 BGR 或 GRAY。Recorder 已完成 RGB→BGR，导入不能再交换通道。文件直接复制，PNG 内容不重新编码。
- PNG 默认保留所有连续帧。标注用途可显式设置 `--every 3` 或 `--dedupe-hamming 2`；回放用途保持 `--every 1 --dedupe-hamming -1`。
- `--no-poses` 只忽略输出姿态，仍保留 JSONL 时间和帧号；不支持用 `--poses` 替换 PNG 会话的 JSONL 姿态。`poses.txt` 留给旧工具使用。
- 缺图、重复图片引用、越界路径、重复/倒序序号或时间、已知相机帧号倒序、无效姿态和图像头不匹配都会报错。选中的图像还会检查是否能完整解码。
- 仅允许忽略最后一条没有换行且 JSON 语法未完成的尾记录，并明确告警；中间损坏或完整坏行会报错。清单外 PNG 不自动导入。
- 相机帧号间隔只表示编号不连续，可能来自主动限频、写盘或相机上游；不能单独当作 Recorder 丢帧率。统计位于 `source.camera_frame_ids`。
- `nt.annotate` 和 `nt.proxy label` 会把采集帧号、时间、姿态和像素格式保留到样本 `scene` 中，方便回溯。

现有 MJPG/FFV1 视频和普通图片目录入口仍可使用。视频继续默认读取同名 TXT 姿态；普通图片目录没有真实时间时保持 `t_s: null`。

## 生成与 A/B 一致的代理图案

```bash
.venv/bin/python -m nt.proxy render --out proxies/a_red3 \
  --screen-spec configs/screens/rm27_hypothesis_a.toml --label three --color red
.venv/bin/python -m nt.proxy render --out proxies/b_red3 \
  --screen-spec configs/screens/rm27_hypothesis_b.toml --label three --color red
```

本机已生成 [A 红 3](../proxies/a_red3/proxy.png) 和 [B 红 3](../proxies/b_red3/proxy.png)，均为 1920×1080；可直接使用。重复生成时换一个 `--out` 名称，已有目录不会覆盖。

`--screen-spec` 读取同一份 TOML 及相对于它的 `content.bitmap_dir`：

- 连续范围取中点、整数范围取向下取整的中点、点形取第一项，形成可复现的静态代表样式。A 默认为 34×28 点，B 为 64×48 方点。
- 存在 `<label>.png` 时使用该 PNG；否则使用规格中的文字和仿真器一致的整点缩放规则。B 的 1/3/4 自动读取七段字形，哨兵/前哨站/基地自动读取各自图案。
- 位图保持物理宽高比，最多占面板宽度 92%，每个点格使用 4×4 子采样，覆盖一半才点亮。面板比例与点阵行列数独立。
- 可以用 `--cols`、`--rows`、`--dot`、`--fill`、`--glyph-height`、`--panel-aspect` 显式覆盖。`--bitmap /path/pattern.png` 也可独立使用。`--text` 与位图来源冲突时会报错。
- `proxy.json` 记录规格与图案 SHA256、实际参数、点亮掩码和角点坐标；旧的不带规格的文本入口仍可用。

以 100% 缩放全屏显示生成的 `proxy.png`，保留四周 ArUco 标记，用比赛相机和实际曝光/增益录制。显示器只能复用静态图案与几何，不模拟绝对亮度、PWM、闪烁、每点亮度扰动或真实 LED 光学。

规格中 `nominal_size_mm` 只用于确定形状，不能当作显示器上实际毫米数。需要 PnP 时，实测显示区域宽高，并保证回放使用相同几何尺寸及当前分辨率下的相机内参。

## 自动标注并入库

```bash
.venv/bin/python -m nt.proxy label --session sessions/proxy_b_red3_s01 \
  --proxy proxies/b_red3/proxy.json --dst datasets/real/proxy_b_red3_s01
.venv/bin/python -m nt.viz datasets/real/proxy_b_red3_s01 \
  --n 20 --out runs/viz/proxy_b_red3_s01.png
```

先人工抽查角点及类别，再用于评估。角色为 `val` 的会话入库后仍是 `val`，不会混入训练；来自同一段采集的相邻帧不要拆到训练和验证两侧。

## 验证记录

- 42 项相关测试通过：PNG 字节与解码像素一致、时间/姿态/帧号配对、坏记录拒收、截断尾行恢复、标注入库保留元数据、A/B 图案与比例、透视变换后的 ArUco 角点及旧视频/图片/烘焙流程。
- 收紧尾行恢复规则后，8 项 Recorder 测试再次通过，含没有换行的损坏 JSON 拒收以及字符串、数字、嵌套对象被截断的恢复。
- 另用仿真器原始 Rust `bitmap_mask` 函数对照 A 的 3 个 PNG 图案及 B 的 6 个 PNG 图案；默认规格下，9 组点亮掩码逐格完全一致。
- 实际执行 CLI 的导入、查看、自动标注入库均通过；合成输入的 PNG 文件字节、时间与帧号完整保留。[验证目录](../runs/verification/recorder_png_proxy_20261105/) 保存对照结果、Rust 参考函数、CLI 日志及合成数据。
- 这些是合成输入和代码流程验证，尚未完成真实相机采集、录像性能测量或代理屏上的模型精度评估。
- Python 3.10 所需 `tomli==2.2.1` 已纳入 `requirements.in` 和哈希锁文件；本机环境已安装，其他环境按原有 `scripts/setup_env.sh` 更新。
