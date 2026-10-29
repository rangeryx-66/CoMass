# xArm7 投掷训练与验证

## 2026-10-30：条件策略续训与远投

本轮在固定基座 xArm7＋UFACTORY 原生夹爪上继续 PPO 训练。成功条件仍是首次物体—地面接触点距目标不超过 **10 cm**；没有增加动作、姿态、速度或加速度限制，没有修改原生执行器、接触参数和物体摩擦来凑成绩。

条件策略扩展为 box、cylinder、sphere、ellipsoid、capsule 五类，尺寸缩放 0.9–1.1，质量 60–150 g，CoM 各轴 ±3 mm；目标 x∈[0.8,1.4]、y∈[−0.25,0.25]、z=0 m。几何点、质量、CoM、惯量和相对目标进入同一个策略。训练逐回合混合 1 / 2 ms，始终保持 50 Hz 策略频率。

### 独立测试

验证种子为 3000–3079；正式测试种子为 5000–5199，未参与模型选择。每项为 200 个不同物体/目标组合，同一测试池内 baseline、本轮策略以及两种步长使用相同参数。首版 checkpoint 和本轮 checkpoint 都在修正后的释放状态实现中测试，原有首版记录另行保留。

| 测试分布 | 步长 | 首版命中 | 本轮命中 | 首版平均误差 | 本轮平均误差 |
|---|---:|---:|---:|---:|---:|
| 原范围 | 2 ms | 94/200（47%） | 169/200（84.5%） | 10.70 cm | 6.40 cm |
| 原范围 | 1 ms | 94/200（47%） | 173/200（86.5%） | 10.56 cm | 6.24 cm |
| 扩展范围 | 2 ms | 21/200（10.5%） | 150/200（75%） | 44.15 cm | 8.36 cm |
| 扩展范围 | 1 ms | 16/200（8%） | 141/200（70.5%） | 46.32 cm | 8.18 cm |

原范围为盒体/圆柱、尺寸 0.95–1.05、80–120 g、CoM 各轴 ±2 mm、目标 x∈[0.8,1.1] / y∈[−0.15,0.15] m。扩展范围见上。平均误差只对发生落地的回合计算；扩展范围的首版在 2 / 1 ms 下分别有 23 / 13 次超时，均计为未命中，本轮没有超时。新策略在两池、两步长中初始双侧接触率和最终释放率均为 100%。

### 扩展测试的分组结果

| 几何 | 样本数 | 2 ms 命中 | 1 ms 命中 |
|---|---:|---:|---:|
| box | 46 | 32（69.6%） | 29（63.0%） |
| capsule | 32 | 27（84.4%） | 25（78.1%） |
| cylinder | 40 | 32（80.0%） | 27（67.5%） |
| ellipsoid | 44 | 29（65.9%） | 30（68.2%） |
| sphere | 38 | 30（78.9%） | 30（78.9%） |

| 联合随机化中的分组 | 样本数 | 2 ms 命中 | 1 ms 命中 |
|---|---:|---:|---:|
| 质量 60–105 g | 104 | 75/104（72.1%） | 72/104（69.2%） |
| 质量 105–150 g | 96 | 75/96（78.1%） | 69/96（71.9%） |
| CoM 偏移模长 ≤3 mm | 123 | 97/123（78.9%） | 90/123（73.2%） |
| CoM 偏移模长 >3 mm | 77 | 53/77（68.8%） | 51/77（66.2%） |
| 目标 x∈[0.8,1.1) m | 104 | 92/104（88.5%） | 85/104（81.7%） |
| 目标 x∈[1.1,1.4] m | 96 | 58/96（60.4%） | 56/96（58.3%） |

这些是联合随机化测试的分组统计，不是单变量消融。较远目标和较大质心偏移仍需改善；未验证任意凸网格、分布外大物体或真机精度。当前属性来自配置，delta 属性估计尚未接入。

### 1 ms 与 2 ms 不一致的原因

使用首版固定物体策略的原始动作进行反事实重放：先冻结动作，再将不同步长的初始化替换为同一个 0.25 ms 沉降后状态。控制周期均为 20 ms。共保存 4 个步长 × 闭环/固定动作/共同初态 3 种情况的逐物理步轨迹。

| 共同初态＋相同原始动作 | 2 ms | 1 ms | 0.5 ms | 0.25 ms |
|---|---:|---:|---:|---:|
| 落点误差 | 8.89 cm | 11.80 cm | 11.63 cm | 10.50 cm |
| 最终脱离时间 | 0.572 s | 0.544 s | 0.541 s | 0.5615 s |
| 最终自由飞行的最大弹道残差 | 1.609 mm | 0.912 mm | 0.446 mm | 0.201 mm |

2 ms 与 1 ms 的二维落点相差 **16.13 cm**。共同初态和各自原生沉降初态的落点差只有约 0.013 / 0.004 mm，初始化不能解释主要差异；冻结动作后差异仍在，也排除了策略频率变化作为主要原因。

主要差异发生在第一次短暂脱离后的再接触：2 ms 出现左右两侧指垫接触，1 ms 同一阶段主要为左侧。0.504–0.580 s 内，CoM 速度增量扣除重力所得横向接触冲量 Jy 分别约 −0.08447 / −0.02451 N·s，对 100 g 物体相当于约 0.60 m/s 的横向速度差。最终脱离后只有毫米级自由飞行残差；最大求解迭代约 4–6 次，未触及默认 100 次上限。因此本例的主要来源是释放阶段接触离散化及再碰撞冲量，不是自由飞行积分或求解迭代打满。

原生指垫使用 solref=[0.004,1]，其时间常数在 2 ms 下只有两个物理步；这支持接触分辨率敏感的解释，但没有通过修改接触软硬来宣称解决。solref 时间常数的含义见 [MuJoCo 官方建模文档](https://mujoco.readthedocs.io/en/3.8.0/modeling.html#solver-parameters)。0.25 ms 的结果仍有变化，不能宣称已经证明严格数值收敛。

另修复首版释放记录：任何机械臂/夹爪再接触后清除旧释放状态，在最终脱离后重新记录，释放奖励只发一次。对原始固定策略的动作重放，落点仍是 8.8883 cm，但正确的最终释放时间是 0.572 s、CoM 自由飞行水平位移 **0.6134 m**、飞行时间 0.166 s；先前 0.852 m 的记录包含了接触运动，已修订。

混合步长训练提高了两个步长下的命中率，仍未消除所有差异。本轮扩展分布中，同一物体/目标的跨步长落点平均差 **2.87 cm**，95 分位差 **8.06 cm**；原范围的平均差为 3.58 cm。详见 verification/followup/paired_steps.json。

### 实际远投与视频

远投使用另一个针对固定 100 g、44×40×50 mm、CoM=0 盒体和 [2,0,0] m 目标精修的 PPO checkpoint，不将它算作五类几何条件策略的泛化成绩。训练同样混合 1 / 2 ms；将奖励中的距离塑形宽度从 0.5 改为 1.0 m，命中半径仍为 0.10 m。

| 物理步长 | 首次落点误差 | 最终脱离后 CoM 水平位移 | 自由飞行时间 |
|---|---:|---:|---:|
| 2 ms | 6.76 cm | **1.573 m** | 0.330 s |
| 1 ms | 7.25 cm | **1.502 m** | 0.319 s |

两种步长均命中，分别进行了 5 次相同条件确定性重放，不能将这些重复视为随机物体鲁棒率。目标距世界原点 2 m 与实际飞行距离分开报告。

- [2 ms 远投视频](verification/followup/videos/far_2ms.mp4)：同一连续真实仿真，包含 1× 原速、0.25× 重放和首次落地暂停；标签明确播放速度。
- [1 ms 远投视频](verification/followup/videos/far_1ms.mp4)。
- [条件策略视频](verification/followup/videos/conditioned.mp4)：按每类前两个测试样本录制，不按是否命中选择，共 10 次，含 5 次未命中。完整样本成绩以 200 次测试表为准。录像所用轨迹与对应正式评估 NPZ 逐元素一致。

### 模型、训练和复现

本轮条件训练共新增 1,109,304 个环境步（含从较早 best 分支继续的实验），最终训练计数 1,179,904；正式交付 checkpoint 的计数为 **1,071,616**。从验证过的 checkpoint 中按两步长平均验证命中率选择该模型（各 81.25%），不使用正式测试结果选模型。远投新增 689,276 步，最终计数 786,432，交付 checkpoint 699,648，按验证中两步长的较大误差最小选择。逐阶段配置、metrics 和计数保存在 verification/followup/training。

条件训练先扩展分布，再将学习率上限降至 1e−4 继续精修（前段上限 5e−4，实际学习率按 KL 自适应）。恢复训练现在保留 Adam 动量并采用当前配置学习率，支持 --reset-best；续训 FPS 仅计本次新增步数。中断保存 last.pt，关闭等待中的子进程有超时退出。前两个条件训练阶段与首段远投的旧 metrics 中 FPS 含恢复前计数，不用于吞吐结论；精修阶段已采用修正实现。

```bash
python -m xarm7_rl.benchmark --config xarm7_rl/configs/conditioned_refine.yaml \
  --checkpoint conditioned:xarm7_rl/verification/followup/checkpoints/conditioned.pt \
  --episodes 200 --seed 5000 --output runs/reproduce_expanded

python -m xarm7_rl.diagnose_steps --config xarm7_rl/configs/default.yaml \
  --checkpoint xarm7_rl/verification/default/best.pt \
  --actions xarm7_rl/verification/followup/diagnosis/legacy_actions.npz \
  --output runs/reproduce_diagnosis

python -m xarm7_rl.showcase --config xarm7_rl/configs/far_refine.yaml \
  --checkpoint xarm7_rl/verification/followup/checkpoints/far.pt \
  --timestep .002 --seed 5000 --slow-replay --output runs/reproduce_far_video
```

训练和主评估使用本地 macOS arm64 / Python 3.13.5 / MuJoCo 3.17.0 / PyTorch 2.15.0，版本见 requirements-followup.txt。服务器 Linux / PyTorch 2.8.0 独立复核了 10 个扩展样本，命中标志全部一致，最大落点位置差约 2.64 mm；同一远投模型也已重放。服务器环境版本见 requirements-tested.txt。完整原始 summary、逐回合日志、压缩轨迹、模型 SHA256 和诊断轨迹均随 verification/followup 保存。macOS 和 Linux 各 **15 项测试通过**。另在服务器 PyTorch 2.8.0 下将本次 checkpoint 从 1,071,616 恢复并实际训练至 1,071,632，完成一轮 PPO 更新，日志见 verification/followup/resume-linux.log。本轮所拥有训练进程均已退出。

---

## 首版历史记录（已修正自由飞行距离）


2026-10-30，服务器 labserver_inschool。新增独立 xarm7_rl 模块，旧 PiPER 工程与实验未修改。

## 已实现

固定基座 xArm7＋UFACTORY 原生夹爪，从真实接触持物状态开始，PPO 学习七轴关节目标与夹爪开度。没有移动底盘、来物、接物预训练、固定挥摆模板、固定出手角度、强制腕部联动、物体 weld 或释放速度注入。

物体几何、质量、三维质心、惯量和相对目标进入策略；默认 32 个表面点时观测 149 维，动作 8 维。支持五类基本几何及单个凸网格碰撞；固定配置重训和逐回合物体/目标随机化都可运行。物体属性当前来自配置，尚未实现 delta 识别。

使用 Menagerie 固定版本原生关节、执行器和夹爪参数，不另加速度/加速度淘汰门槛或姿态约束。动作尺度、回合时长、初始位姿、沉降时间和命中半径均在 YAML 中公开配置。默认模型的执行器参数尚未通过本项目真机辨识。

## 实际训练与独立评估

| 项目 | 固定物体和目标 | 联合随机化条件策略 |
|---|---|---|
| 训练步数 | 131,072 | 总计 131,072，含 4,096 步初始短训练 |
| 物体 | 100 g，44×40×50 mm 盒体，CoM=0 | 盒体/圆柱，尺寸 0.95–1.05 倍，80–120 g，CoM 三轴各 ±2 mm |
| 世界坐标目标 | [1,0,0] m | x∈[0.8,1.1]、y∈[−0.15,0.15]、z=0 m |
| 最后 200 个训练回合命中率 | 53.5% | 38.0% |
| 最后训练窗口平均落点误差 | 10.42 cm | 13.23 cm |
| 独立评估 | 20 次固定条件确定性重放 | 50 个新种子的随机物体/目标配置 |
| 命中，半径 10 cm | 20/20 | 28/50（56%） |
| 独立评估平均落点误差 | 8.89 cm | 9.34 cm |
| 初始双侧夹持／成功释放 | 20/20 | 50/50 |

固定配置的 20 次重放轨迹相同，不是随机鲁棒性证据。随机化评估中盒体命中 13/27、圆柱 15/23；不代表分布外物体或任意目标都能命中。当前策略仍可继续训练，未完成高精度收敛。

后续诊断修订：首版将第一次短暂脱离当成最终释放，没有在夹爪再接触后更新，因而曾报告 0.504 s 释放和 0.852 m 投距。对首版原始动作逐步重放，最终脱离实际发生在 0.572 s，CoM 线速度约 3.794 m/s，最终脱离至首次地面接触的 CoM 水平位移约 **0.613 m**、时间约 **0.166 s**。原始落点误差 8.89 cm 可复现。后续报告使用修正后的最终脱离记录，不能把目标距世界原点 1 m 写成实际投距 1 m。

将物理步长从 2 ms 减到 1 ms、保持策略 50 Hz 不变后，同一固定策略的两次确定性重放误差为 **12.72 cm**，超出 10 cm 命中半径。这一数值敏感性仍未解决，不能将 2 ms 的成绩解释为真机精度。原始结果保存在 verification/default/summary_1ms.json。

## 可复现命令

在 CoMass 根目录及已安装环境中：

```bash
# 固定物体／目标从头训练
xarm7-train --config xarm7_rl/configs/default.yaml \
  --set ppo.total_steps=131072 --set output=runs/xarm7_default

# 联合随机化，本次记录使用以下短训练与恢复流程
xarm7-train --config xarm7_rl/configs/conditioned.yaml \
  --set num_envs=4 --set device=cpu --set ppo.horizon=64 \
  --set ppo.epochs=2 --set ppo.total_steps=4096 \
  --set output=runs/xarm7_conditioned_smoke
xarm7-train --config xarm7_rl/configs/conditioned.yaml \
  --resume runs/xarm7_conditioned_smoke/last.pt \
  --set ppo.total_steps=131072 --set output=runs/xarm7_conditioned

# 直接评估本次交付的两个 checkpoint
MUJOCO_GL=egl xarm7-eval \
  --checkpoint xarm7_rl/verification/default/best.pt \
  --episodes 20 --output runs/replay_default --video
MUJOCO_GL=egl xarm7-eval \
  --checkpoint xarm7_rl/verification/conditioned/best.pt \
  --episodes 50 --seed 2000 --output runs/replay_conditioned --video
```

两份训练配置、逐更新 metrics.jsonl、checkpoint 和完整评估 summary.json 随交付保留。服务器 runs/xarm7_* 还保存全部逐回合轨迹；仓库保留默认示例的原始 trajectory.npz 与两个连续视频。轨迹列依次为 time、qpos(nq)、qvel(nv)、action(8)、ctrl(nu)，NPZ 保存 nq/nv/nu。

## 软件验证

13 项真实实现回归通过，覆盖原生双侧夹持与接触释放、七轴动作、五类几何、自定义网格、质量/CoM 写入实际模型及观测、目标条件、种子重复、超时判定、GAE 和策略梯度。日志与 XML 在 verification/tests.*。

完成 4096 步固定配置短训练、4096 步联合随机化短训练及从 4096 恢复到 4352 步的训练检查。checkpoint 保存模型、归一化、Adam、训练计数和 RNG；仿真回合在恢复时重新初始化。

已构建 wheel 并确认包含 16 个原始 STL 和来源清单。实际验证版本在 requirements-tested.txt；仿真和训练不依赖外部服务，训练不渲染，评估与实验结束后关闭所拥有进程。
