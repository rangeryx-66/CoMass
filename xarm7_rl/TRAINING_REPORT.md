# xArm7 投掷 RL 首版验证

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

固定策略在回合开始后约 0.504 s 实际脱离，CoM 线速度约 3.111 m/s，自由飞行水平距离约 0.852 m、飞行时间约 0.234 s。不能把目标距世界原点 1 m 写成实际投距 1 m。

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
