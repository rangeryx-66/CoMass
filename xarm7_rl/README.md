# xArm7 原生夹爪投掷 PPO

固定基座 xArm7 从持物状态开始，仅学习投掷。物体几何、质量、三维质心、惯量及目标落点进入同一个策略；可针对固定配置重新训练，也可逐回合随机化训练条件策略。当前属性来自配置，后续 delta 模块可替换相同属性输入。

## 安装和训练

在 CoMass 仓库根目录执行，Python 3.10 或更新版本：

```bash
python -m venv .venv-xarm7-rl
source .venv-xarm7-rl/bin/activate
pip install -e './xarm7_rl[test]'

xarm7-train --config xarm7_rl/configs/default.yaml
```

模型资产已随包提供，无需下载 Catch It 的移动底盘和 LEAP Hand。
CPU 运行可加 `--set device=cpu`。每个环境为独立 MuJoCo CPU 进程，网络使用指定的 PyTorch 设备；`num_envs` 可调整。

修改物体和目标后训练：

```bash
xarm7-train --config xarm7_rl/configs/default.yaml \
  --set 'env.object.type=box' \
  --set 'env.object.size=[0.025,0.020,0.030]' \
  --set env.object.mass=0.15 \
  --set 'env.object.com=[0.005,0,0]' \
  --set 'env.target.position=[1.2,0.2,0]' \
  --set output=runs/my_object
```

尺寸沿用 MuJoCo 约定：box/ellipsoid 为三个半轴，sphere 为半径，cylinder/capsule 为 `[半径,半长]`。自定义网格设 `type: mesh`、`mesh: path/to/object.stl`、`mesh_scale: [1,1,1]`；路径相对配置文件解析。网格文件也支持 MuJoCo 接受的 OBJ。MuJoCo 网格碰撞为凸包；复杂凹物体应预先拆分，当前单几何体接口不声称还原凹腔接触。

目标为世界坐标 `[x,y,z]`，`z` 定义水平落地平面的高度；不是空中穿越点或桶口。默认地面为 z=0。命中按首次物体—地面接触点的水平误差判定，`success_radius` 是可配置评价半径。

## 随机化与条件策略

同一训练过程支持通过 YAML 配置：

```yaml
# 放入 env 下，替换相应字段
target:
  position: [1.0, 0.0, 0.0]
  low: [0.7, -0.3, 0.0]
  high: [1.2, 0.3, 0.0]
  success_radius: 0.10
randomization:
  objects:
    - {type: box, size: [0.022, 0.020, 0.025]}
    - {type: cylinder, size: [0.020, 0.025]}
  mass_range: [0.06, 0.15]
  com_low: [-0.002, -0.002, -0.002]
  com_high: [0.002, 0.002, 0.002]
  size_scale_range: [0.9, 1.1]
```

可直接使用 `xarm7_rl/configs/conditioned.yaml` 启动联合随机化训练。

每回合更新真实质量、质心和惯量。未提供 `inertia` 时使用凸体均匀质量加内部点配重模型实现指定 CoM；提供时为物体系下关于 CoM 的 `[Ixx,Iyy,Izz,Ixy,Ixz,Iyz]`。物理参数必须能描述合法刚体。采样范围由用户指定，代码不把目标限制在预设工作区，也不按目标距离提前拒绝回合。

## 动作和观测

动作 8 维：七个关节位置目标增量、一个原生夹爪开度。所有关节可同时运动，不规定挥摆路径、固定出手角度、腕部联动、起抛方向或理想飞行时间。关节和执行器范围沿用上游模型；`action_rate` 只是从归一化动作到位置目标变化的可调尺度，没有额外速度/加速度超限淘汰条件。

默认 32 个物体系表面采样点描述几何；观测还包含七轴 q/dq、夹爪状态、物体世界位姿/CoM 线速度/角速度、相对目标、质量、物体系 CoM、完整惯量上三角、回合时间、释放/接触标志和上一动作。

初始位姿、抓持中心、物体姿态、接触沉降时间均可配置。reset 只在初始化时放置物体，并让真实原生夹爪闭合沉降；没有物体 weld、自由物体速度注入或持物阶段外力。不能被当前抓具夹持的物体会在初始化/回合指标中暴露，不会暗中改尺寸、改摩擦或替换物体。

奖励由持物时预计落点接近程度的增量、一次性释放奖励、真实落点误差和命中奖励组成；默认没有动作代价。预计落点仅用于塑形，终端成绩来自真实积分后的首次地面接触，不把时间耗尽记为命中。

## 评估与恢复

```bash
MUJOCO_GL=egl xarm7-eval --checkpoint runs/xarm7_throw/best.pt \
  --episodes 20 --output runs/eval --video

xarm7-train --config xarm7_rl/configs/default.yaml \
  --resume runs/xarm7_throw/last.pt --set ppo.total_steps=4000000

python -m pytest xarm7_rl/tests -q
```

评估默认使用 checkpoint 中的配置。指定 `--config` 和 `--set` 可评估新目标或物体，但分布外命中能力需实际测量。输出逐回合成功、落点误差、释放状态、实际自由飞行距离/时间、轨迹 NPZ，以及可选首回合连续视频。

checkpoint 保存模型、归一化统计、Adam、步数和 RNG 状态。恢复训练会重新初始化仿真回合，不声称跨进程逐物理步续接。训练不调用渲染；评估回合数有明确上限，退出后关闭环境进程与渲染资源。

算法与资产来源见 THIRD_PARTY.md。本模块仅仿真，不含硬件控制接口。训练预算、目标范围和物体分布是配置项；是否达到所需精度由独立评估结果判断。
