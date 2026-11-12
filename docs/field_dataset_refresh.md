# 原赛场数据重生成与负样本

本版本只使用当前 Daedalus 的原赛场，不增加地面、墙面或围挡材质。B 样式沿用当前七段字形和 64×48 方点规格。

## 仿真修改

步兵的 vehicle.glb 在 GIMBAL 下带有两个悬空的“激光检测总装.001 / .002”节点，位置分别约为云台坐标的 0.46 m 和 0.65 m 高处。它们不是装甲板。setup_vehicle 现在从机器人实例中移除这两个节点，交互仿真和数据导出均生效；HERO.glb 没有这两个节点。原场地的灯光、建筑和实际装甲保留。

scene.background_only = true 时，相机改为对准原赛场内的点，背景目标点 X/Z 范围由 scene.background_half_m 配置。机器人、前哨站和基地仍正常存在。每个装甲根节点用半径 0.25 m 的球保守包围整个小装甲模块；只接受所有球均完全处于相机视锥之外的画面。不按颜色、朝向、遮挡程度或投影尺寸放行，避免把熄灭和极小装甲当成背景。

每张负样本记录全部 24 个装甲中心、包围球半径、相机位置、四元数和视场角，审计脚本独立复算视锥排除条件，并检查空标签。背景视图会包括地面、标线、坡道和建筑局部，但不是把装甲从场地中抹掉得到的空场景。

## 数据版本和分区

| 配置 / 入库名称 | 角色 | 渲染张数 | 入库张数 | 场景种子 |
|---|---|---:|---:|---:|
| style_b_field_v1_train | train | 2700 | 2608 | 1101 |
| style_b_field_v1_val | val | 600 | 586 | 1202 |
| field_negative_v1_train | train | 300 | 300 | 2101 |
| field_negative_v1_val | val | 60 | 60 | 2202 |

原始导出在 runs/sim_名称，经过现有严格标签检查后入库到 datasets/sim/名称。训练 92 张、验证 14 张被现有角点顺序门禁拒收，未放宽检查规则；场地负样本没有拒收。四次渲染均没有捕获丢帧。新版本使用独立目录和种子，历史 A/B 数据及实验不覆盖。

负样本训练分区使用 train 角色：当前 nt.bake 会把 role=neg 全部分配到验证，不能用它表示训练负样本。训练、验证各有独立渲染种子，最终还按原图 SHA256 检查跨分区重复。

烘焙输出为 baked/style_b_field_v1，768×576 PNG，训练图每个原图生成两份，验证每个原图一份。配置为 configs/bake/field_clean.yaml：保留默认曝光、白平衡、泛光、中心对齐的运动模糊、散焦与噪声，关闭离线绘制的虚构发光干扰物。ROI 扰动、摩尔纹、拜耳、频闪仍按默认关闭。本轮不改变已有模型权重。

执行入口：

~~~bash
# 先构建仿真器；在本机权限环境下，需要能够访问真实 GPU。
scripts/run_capped.sh 5 -- cargo build --offline --release \
  --manifest-path /home/robo/bevy_robomaster_simulator/Cargo.toml --bin daedalus
.venv/bin/python scripts/regenerate_field_dataset.py --workers 3 --copies 2
~~~

脚本拒绝覆盖已有输出，渲染后检查数量、丢失计数和移除节点日志，再依次入库、烘焙和审计。大任务使用现有运行脚本；没有 systemd 用户会话时仍会退化为 nice 执行，不等于建立了硬内存上限。若只重复审计：

~~~bash
scripts/run_capped.sh 3 -- python scripts/audit_field_dataset.py \
  --baked baked/style_b_field_v1 --out runs/verification/field_refresh_20261106
~~~

结果、分区统计、渲染预览、耗时和仿真源码哈希在 runs/verification/field_refresh_20261106/。本轮完成训练 5816 张、验证 646 张；新增场地负样本烘焙后为 600 / 60 张，总空标注为 638 / 66 张，占 10.97% / 10.22%。全部新增负样本通过独立几何复核，训练与验证没有相同原图。渲染、入库、烘焙和审计合计约 970 秒，仿真器 9 项数据集测试通过。详细数量见 audit.json。

## 清理已完成实验的派生数据

scripts/archive_completed_bakes.py 仅针对明确列出的七份旧派生烘焙：style_b_v3 的 base / roi / moire / both / stress_val，以及 style_b_v4_bayer / bayer_val。它们对应的增强实验已经完成，历史样式 B 原图、标签、模型权重和报告均保留。

~~~bash
# 默认只查看名单与预计占用。
.venv/bin/python scripts/archive_completed_bakes.py
# 只有新数据 audit.json 已通过才执行；先归档全部重建材料，校验后清理。
scripts/run_capped.sh 3 -- python scripts/archive_completed_bakes.py --apply
~~~

归档目录为 runs/baked_manifests/cleanup_field_refresh_20261106/，每份包含完整烘焙配置、源清单指纹、原 bake_manifest.json、压缩的逐图 meta.jsonl、校验和以及 rebuild_argv。重建应读取归档内的配置；不要直接用之后可能变化的默认配置。清理的是图片、标签及其派生目录，完整的逐图重建记录仍然保留。

本轮已经完成归档和清理，回收约 20.8 GiB；归档本身约 27 MiB，清理后磁盘可用约 45.3 GiB。新 baked/style_b_field_v1 约 4.7 GiB。原始 A/B 数据、baked/simm7_c2、baked/style_b_v1、全部权重和实验报告保留。

旧 A/B 迁移与增强实验成绩仍属于历史数据。新原赛场版本还没有训练，不能声称移除组件或加入负样本已经带来精度提升。
