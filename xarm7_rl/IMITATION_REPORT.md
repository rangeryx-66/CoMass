# 视频姿态模仿：实现与试训记录

2026-10-30。已完成参考提取、重定向、软奖励和 BC/PPO 试验。**当前能提高举臂倾向，但还没有同时保住命中率与期望姿态，因此保留为可选实验模式，不替换原来的准确策略。**

## 参考从哪里来

实际提取使用 [MIT UAN 的公开抛球视频](https://uan.csail.mit.edu/)，文件为 [throw_annotated.m4v](https://uan.csail.mit.edu/rsc/throw_annotated.m4v)。另外查看了 [Ishikawa 的高速投球项目](https://ishikawa-vision.org/fusion/HighspeedThrowing/index-e.html)。动作模仿与任务目标联合训练的思路可参见 [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/)；本实现是简化的显式姿态先验，没有复刻该论文算法。

对 MIT 视频人工查看并标注了 6 个关键帧的肩、肘、腕、掌部二维点，保存帧号、视频呈现时间、原图坐标、来源 SHA256 和定性置信度。不是自动识别出的真实三维关节角。原视频有停帧、慢放和机身移动，因而没有将播放速度当作物理速度；只使用肩部相对的连杆方向和帧序。原视频未随代码分发。

源机器人与 xArm7 的结构不同。先取朝目标方向和竖直方向的二维连杆夹角，再用 PCHIP 插值为 64 个参考姿态，使用原生 xArm7 FK / IK 拟合七轴角度。深度不可观测，弱平面与连续性正则选出一组解；这些正则只用于离线参考拟合，不限制策略动作。投影拟合误差约 0.0017°，这个数只表示拟合残差，不代表人工标注或三维恢复具有此精度。参考 JSON 在 src/xarm7_rl/assets/motions/mit_overhand.json。

[参考姿态视频](verification/imitation/reference/reference.mp4) 是无物体的运动学预览，不是成功投掷；播放相位任意，不能当作源视频真实速度。

## 奖励和学习流程

原命中与落点奖励保留，成功半径仍是 10 cm。软模仿比较上臂、前臂、腕到掌部三段在目标竖直平面内的投影方向，选择相似度最高的参考帧，不强制出手时间或固定动作。相似度为 s=exp(−e²/(2σ²))，其中 e 是三段方向角差的加权 RMS，默认权重 [1,1,0.5]。

额外奖励为：λd·Δt·(s−1)，并可在回合结束添加 λm·(全程平均 s−1) 和 λr·(最终释放时 s−1)。负偏差形式不支付静止停留奖励，Δt 使用实际物理时间。出手指标在任何再接触后更新，使用最终真正脱离时的姿态。实验中持续时间代价仍可能促使更快结束，因此同时测试了回合均值和出手项。

直接续训的小权重 / 大权重组没有明显改变姿态。又从参考姿态生成 100 条真实动力学示教，包含 9,039 个控制步：控制器经普通 ThrowEnv.step 执行八维动作，初始持物、接触与释放均由原生模型产生，不注入物体速度或外力。示教准备时长、挥臂时长和出手相位随机化；示教仅在采集时使用，部署动作由网络独立输出。示教本身只命中 10/100，没有按命中筛选。

BC 使用 2,000 次 minibatch 更新，初始化 actor，保留原归一化器与 critic，并清空旧 PPO Adam 动量。纯 BC 在 7000–7039 的验证集上，两种步长均为 0/40 命中，但平均投影姿态偏差降至约 22.7°。随后加入任务与模仿奖励继续 PPO。另测试了只约束手臂七维动作的 BC 辅助损失，夹爪由任务学习，系数为 5；该组也没有解决取舍。

观测仍为 149 维、动作仍为八维；没有增加关节角、姿态、速度、加速度门槛、轨迹拒绝或额外终止条件。没有推理时动作混合。默认配置不启用这些模块。

## 新测试集的实际结果

五类几何、60–150 g、CoM 各轴 ±3 mm、目标 x∈[0.8,1.4] / y∈[−0.25,0.25] m，沿用此前扩展分布。验证用 7000–7079，正式测试用未参与选择的 9000–9099。每个模型 / 步长 100 个样本；同一参数与种子配对。所有正式模型取预先指定的 last checkpoint，不按正式测试成绩选取。测试时统一使用零模仿权重，仅记录相同的姿态指标，任务判定不变。

| 方案 | 2 ms 命中 | 1 ms 命中 | 2 / 1 ms 平均落点误差 | 2 / 1 ms 全程投影姿态 RMS |
|---|---:|---:|---:|---:|
| 上一轮条件策略 | 72/100 | 71/100 | 7.84 / 7.98 cm | 65.91 / 65.97° |
| 普通续训对照 | 69/100 | 71/100 | 7.92 / 8.11 cm | 66.37 / 66.43° |
| BC 预热＋RL | 53/100 | 57/100 | 10.57 / 10.62 cm | 54.47 / 54.32° |
| BC 预热＋较强姿态奖励 RL | 16/100 | 17/100 | 38.45 / 39.28 cm | 48.32 / 48.34° |

姿态 RMS 越低，越接近这份二维参考；不是三维关节角误差，也不是对动作美观的客观评分。普通续训对照与 BC＋RL 的 PPO 预算均为 262,272 个新增环境步；强姿态组为 131,328 步，BC 还使用额外示教和监督更新。这是探索性方案对比，不是严格单因素消融，也不能据此断言风格与准确率不可兼得。正式原始表中未释放回合的出手角误差按 180° 计；release_geometry.json 另给出有效释放的统计。

1 ms 下，最终出手时腕部高于肘部的比例为：原策略 5/100、普通续训 8/100、BC＋RL 33/100、强姿态组 88/92；强姿态组另外 8 次没有形成有效释放，均计为未命中。掌部出手高度的有效释放均值分别约 0.450、0.446、0.498、0.728 m。强姿态组的有效出手投影误差均值约 40.0°。所以“抬臂”确有变化，但距离稳定的投篮 / 棒球式动作仍有差距。

## 实际视频

[四方案同场景对比](verification/imitation/comparison/comparison.mp4)：固定 100 g 盒体、CoM=0、[1,0,0] m 目标、1 ms 物理步长。该场景预先固定，保留未命中。先原速，再标明 0.25× 重放；某个方案先落地后暂停，并明确标注。

左上为原策略，右上为纯视频模仿，左下为 BC＋RL，右下为强姿态奖励 RL。这一例原策略误差 3.25 cm、纯模仿 35.17 cm、BC＋RL 3.42 cm、强姿态 68.13 cm。视频说明了当前困难：单独模仿会举臂但投不准，RL 恢复准度时又倾向返回低位出手。单条视频不代表上表整体命中率。

## 如何复现

```bash
# 先看参考姿态；这不是实际投掷测试
python -m xarm7_rl.retarget --config xarm7_rl/configs/default.yaml \
  --output runs/reference --video

# 完整的物理示教和模仿预热
python -m xarm7_rl.imitate collect --config xarm7_rl/configs/imitation_appearance.yaml \
  --episodes 100 --output runs/my_demos
python -m xarm7_rl.imitate clone --config xarm7_rl/configs/imitation_bc_rl.yaml \
  --checkpoint xarm7_rl/verification/followup/checkpoints/conditioned.pt \
  --dataset runs/my_demos/demonstrations.npz --updates 2000 --output runs/my_bc.pt
python -m xarm7_rl.train --config xarm7_rl/configs/imitation_bc_rl.yaml \
  --resume runs/my_bc.pt --reset-best

# 直接复评本次的三个新模型和原策略（每个步长 100 次）
python -m xarm7_rl.benchmark --config xarm7_rl/configs/imitation_0.yaml \
  --checkpoint original:xarm7_rl/verification/followup/checkpoints/conditioned.pt \
  --checkpoint control:xarm7_rl/verification/imitation/checkpoints/control.pt \
  --checkpoint bc_rl:xarm7_rl/verification/imitation/checkpoints/bc_rl.pt \
  --checkpoint overhand:xarm7_rl/verification/imitation/checkpoints/overhand.pt \
  --episodes 100 --seed 9000 --output runs/imitation_retest
```

## 当前判断与验证

这个方向合理，但“从一个视频取几个姿态＋叠加奖励”尚不足以达到期望效果。单目深度、相位和速度不可靠；当前参考只有 6 个姿态，逐帧最近匹配也没有表达完整动作片段。更直接的下一步是得到同时满足所需风格和目标落点的 xArm7 示教，再用动作片段 / 相位的软模仿学习优化。继续单纯增大姿态权重并没有解决问题。

现有 19 项回归通过，包含模仿开 / 关时同动作的关节轨迹与观测逐元素一致、零权重奖励完全相同、目标方向 / 尺寸不变性、按物理时间缩放奖励，以及原有原生夹爪 / 七轴 / 物体属性 / GAE 测试。wheel 已验证包含动作参考 JSON 和 16 个原生 STL。检查模型、日志、配置、正式样本、压缩轨迹、视频与来源元数据可见 verification/imitation。未覆盖之前的 checkpoint；本轮所拥有训练进程均已退出。
