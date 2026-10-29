# Basketball-Style Long-Arc Throwing 实验报告

2026-10-28，纯 MuJoCo 仿真。旧 REPORT.md、14 项测试及已完成实验保留；没有使用 RL、神经网络训练、真实机械臂或 CAN。

## 结论与边界

状态：`no_feasible_arc_under_robot_limits`。在本次模型、控制器、动作族与有限预算下，最大合规高弧线距离：未找到 / 不适用 m；经扰动接触释放验证的最大稳定距离：未找到 / 不适用 m。
没有把失败候选、只满足上升高度的动作或单次理想命中报告为稳定高抛。负结果不能证明真实 PiPER 在所有控制器下均不可行。尤其 5 rad/s²、力矩预算与执行器响应尚未真机标定。

## 规划、场景与约束

PiPER 基座固定在 0.75 m 桌面，完整桌板、四条桌腿、原机械臂碰撞体、120×80×50 mm/100 g 隐藏配重盒、桶底和48段桶壁均参与物理接触。桶口高度0.25 m。固定基座与桌板安装面作为刚性装配排除相互碰撞；活动连杆及完整盒体保留碰撞。初始为已知的桌上方夹持姿态，随后准备、蓄力、抛出和制动；没有实现从桌面拾取。

Ballistic planner 由三维释放点、三维桶口和期望升高求 v_z=√(2gh)、下降过桶口时间与水平速度。MuJoCo 六轴 Jacobian 加有界线性最小二乘解释放关节速度，SciPy differential_evolution/Nelder-Mead 联合搜索释放关节姿态与加速时间；J2/J3/J4/J5可协同，未局限J2。全程检查准备、蓄力、制动的关节范围及碰撞。

新增平滑加速度平台轨迹族与原五次多项式准备段复用；保留平滑step速度轨迹对照。期望速度只用于求关节指令；自由刚体速度没有被手动设置。理想释放仅解除weld，接触释放无weld并真实开爪。

六轴指令及实际速度均限制3 rad/s，SDK关节位置与上游MJCF取交集；实际/指令加速度预算5 rad/s²，无旧实验5%/10%容差。检查实际跟踪误差、完整制动、滑动和再次碰臂。力矩限制[12,12,12,3,3,3] N·m为未标定的保守实验假设；PD增益、名义负载补偿与开爪时延亦为假设。完美被动重力补偿关闭，补偿通过相同的受限执行器施加。
[参数来源与假设](piper_throw/HARDWARE_LIMITS.md)。官方来源：[AgileX SDK固定提交](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/CHANGELOG.MD)。

## 坐标与指标定义

世界系W：x向前、y向左、z向上；基座B位于(0,0,0.75)m，方向同W。几何系O原点在盒体几何中心，T_WO=T_WB·T_Bhand(q)·T_handO；偏心抓取平移为hand系(0.025,0,0.105)m，盒体相对hand绕x转90°。p_Wcom=p_WO+R_WO·c_O。桶系原点为桶底中心，方向同W，可配置任意三维平移。
release_speed/linear_velocity为CoM速度，release.position为几何参考点；horizontal_range专指实际释放几何点到目标桶口中心的水平距离，并非未命中时物体实际飞行的水平位移。release_angle以rad保存，同时提供deg。apex为CoM实测自由飞行最高点，另存几何点最高高度；flight_time为几何参考点从释放到首次下降穿过桶口高度的时间。ballistic_target_error是以实际CoM释放状态解析预测的桶口平面误差，几何/CoM的实测穿越误差分别保存。所有长度m、质量kg、时间s、角度rad。

## 距离扫描与真实自由飞行

搜索先使用运动学释放位置构造候选目标。最终审计额外运行：依据该动作实测释放几何参考点确定桶位置，再从相同初态重新执行，确保下表的距离按实际释放点到桶口中心的水平距离计算。桶在每次正式积分开始前固定，不在飞行途中移动。所有标称距离均以实测距离核对，不用基座到桶距离代替。
主扫描真实试投 196 次，另有 14 次距离审计、88 次固定动作CoM试投、12 次步长复核。数值规划评估次数见各目录 planner.json；预试验与其他轨迹族不混入主扫描。

| 目标距离(m) | 模型 | 实测水平距离(m) | 释放速度(m/s) | 仰角(°) | v_z(m/s) | 实测升高(m) | 最终成功 | 滑动 | 主要失败 |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 | ideal | 0.800 | 2.289 | 57.4 | 1.928 | 0.188 | False | False | actual_acceleration_limit, insufficient_apex_rise, release_not_upward, tracking_error_limit |
| 0.8 | contact | 0.800 | 2.293 | 59.7 | 1.981 | 0.199 | False | True | actual_acceleration_limit, insufficient_apex_rise |
| 1.2 | ideal | 1.200 | 2.230 | 53.1 | 1.784 | 0.161 | False | False | actual_acceleration_limit, insufficient_apex_rise, release_not_upward, tracking_error_limit |
| 1.2 | contact | 1.200 | 2.177 | 51.2 | 1.696 | 0.146 | False | True | actual_acceleration_limit, insufficient_apex_rise, release_not_upward |
| 1.5 | ideal | 1.500 | 2.276 | 53.6 | 1.832 | 0.170 | False | False | actual_acceleration_limit, insufficient_apex_rise, release_not_upward, tracking_error_limit |
| 1.5 | contact | 1.500 | 2.254 | 55.3 | 1.852 | 0.174 | False | True | actual_acceleration_limit, insufficient_apex_rise, release_not_upward |
| 2.0 | ideal | 2.000 | 0.968 | 62.4 | 0.858 | 0.037 | False | False | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, no_entry_after_apex, release_not_upward, tracking_error_limit |
| 2.0 | contact | 2.000 | 0.857 | 68.2 | 0.796 | 0.032 | False | True | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, no_entry_after_apex, release_not_upward, tracking_error_limit |
| 3.0 | ideal | 3.000 | 1.508 | 48.7 | 1.133 | 0.065 | False | False | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, no_entry_after_apex, release_not_upward, tracking_error_limit |
| 3.0 | contact | 3.000 | 1.587 | 53.1 | 1.269 | 0.081 | False | True | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, no_entry_after_apex, release_not_upward, tracking_error_limit |
| 4.0 | ideal | 4.000 | 1.652 | 42.3 | 1.112 | 0.063 | False | False | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, no_entry_after_apex, release_not_upward, tracking_error_limit |
| 4.0 | contact | 4.000 | 1.769 | 45.2 | 1.255 | 0.080 | False | True | actual_acceleration_limit, actual_velocity_limit, contact_before_rim_crossing, insufficient_apex_rise, release_not_upward, tracking_error_limit |
| 5.0 | ideal | 5.000 | 2.066 | 29.9 | 1.031 | 0.054 | False | False | actual_acceleration_limit, actual_velocity_limit, insufficient_apex_rise, release_not_upward, tracking_error_limit |
| 5.0 | contact | 5.000 | 2.404 | 29.7 | 1.191 | 0.072 | False | True | actual_acceleration_limit, actual_velocity_limit, insufficient_apex_rise, release_not_upward, tracking_error_limit |

[逐距离CSV](results/basketball_throw/published/distance_scan.csv) · [实测速度/升高/加速度图](results/basketball_throw/published/distance_scan.png)

升高与最高点从已积分的CoM轨迹测量，并在第一次环境接触或桶口穿越处截断；同时保存几何参考点与CoM轨迹。球心/几何中心穿过桶口不等于完整入桶。成功仍要求完整外形持续低速留桶、无安全违规、无提前滑脱且真实先升后降。

## 固定动作的CoM复核

所选动作是否仅供失败诊断：**True**。基准实测释放速度 2.496 m/s，v_z=2.105 m/s；升高 0.225 m，最高点世界高度 0.963 m。动作、桶位置、物体几何与质量保持不变，四种物体沿用旧物理一致壳体+内部配重惯量。
理想与无扰动接触各配置只运行一次，属于确定性比较；扰动接触每配置20次，四种物体使用配对相同种子。释放延迟σ=4ms、增益σ=2.5%、摩擦对数σ=5%；没有为不同配重人为添加速度或落点偏置。

| 条件 | CoM间均值最大分离(mm) | object_0 命中 | object_1 命中 | object_2 命中 | object_3 命中 | 合规高抛结论 |
|---|---|---|---|---|---|---|
| contact | 115.399 | 合规0/1；留桶1/1 | 合规0/1；留桶1/1 | 合规0/1；留桶1/1 | 合规0/1；留桶1/1 | False |
| ideal | 66.686 | 合规0/1；留桶0/1 | 合规0/1；留桶0/1 | 合规0/1；留桶0/1 | 合规0/1；留桶0/1 | False |
| perturbed_contact | 114.219 | 合规0/20；留桶20/20 | 合规0/20；留桶20/20 | 合规0/20；留桶20/20 | 合规0/20；留桶20/20 | False |

[逐次CoM CSV](results/basketball_throw/published/com_repeats.csv) · [CoM偏移与误差图](results/basketball_throw/published/com_sensitivity.png) · [释放/飞行/姿态差异分解](results/basketball_throw/published/release_flight_decomposition.csv)

这些均值分离仅比较有桶口下降穿越记录的试投；缺失穿越不填零，缺失与失败仍计入分母。若动作违反安全或夹持条件，其分离只作失败诊断，不能用来宣称更远且稳定的高抛放大了CoM影响。旧0.8m实验的62.55mm属于另一动作与释放协议，不能直接据两个最大值作显著性判断。四种配重同时改变多个轴，也不能把散点图解释为单轴因果效应。

释放位置/速度、空中角速度、桶口姿态以及几何/CoM过平面位置均独立保存。无空气阻力时自由飞行CoM obeys gravity；差异可来自载荷跟踪、接触滑动、释放状态和绕不同CoM旋转的几何参考点。存在滑动的候选不能把全部差异归因于纯自由飞行。

## 可复现命令与视频

```bash
bash piper_throw/tools/bootstrap.sh
source .venv-piper/bin/activate
python -m pytest piper_throw/tests piper_throw/upstream/openarm-control/tests/test_throwing.py -q
python -m piper_throw.cli basketball --config configs/basketball_throw.yaml --output results/basketball_throw/new_run --budget 28 --repeats 20 --video
```

运行新实验时使用新输出目录，不覆盖本次记录。单个固定三维桶目标也可通过 plan_candidates(..., target_world=[x,y,z_rim]) 规划，再 run_throw 执行。

全景/侧面视频由同一份实际qpos回放生成，20/30fps不改变仿真时间，字幕显示实测速度、距离、升高与success。三维轨迹视频只绘制测得的几何/CoM样本。若无合规动作，演示标为失败诊断，不提供伪装成功的视频。

## 限制与后续

本次不能声称已完成稳定远距离投篮或真实Sim-to-Real。5m目标的弹道速度需求、Jacobian有界求解残差、实际速度/加速度、跟踪与夹持失败都已分别记录。有限预算和当前动作族/伺服近似也可能限制结果，不能仅凭失败把全部责任归为硬件最大速度。下一步应先做离线执行器/夹持模型辨识和更强的约束轨迹优化；本阶段仍不连接真机。

## 重复散布与数值复核

| 条件 | 物体 | 同CoM二维RMS(mm) | 滑动次数 |
|---|---|---|---|
| contact | object_0 | 0.000 | 1/1 |
| contact | object_1 | 0.000 | 1/1 |
| contact | object_2 | 0.000 | 1/1 |
| contact | object_3 | 0.000 | 1/1 |
| ideal | object_0 | 0.000 | 0/1 |
| ideal | object_1 | 0.000 | 0/1 |
| ideal | object_2 | 0.000 | 0/1 |
| ideal | object_3 | 0.000 | 0/1 |
| perturbed_contact | object_0 | 9.355 | 20/20 |
| perturbed_contact | object_1 | 14.726 | 20/20 |
| perturbed_contact | object_2 | 4.279 | 20/20 |
| perturbed_contact | object_3 | 4.256 | 20/20 |

留桶只是几何和静止判据，不能抵消滑动、安全或高弧线失败。确定性单次比较的RMS=0不代表统计稳定。
[步长复核CSV](results/basketball_throw/published/timestep_convergence.csv) · [自由飞行CoM弹道一致性](results/basketball_throw/published/freeflight_numerics.csv)

## 本次失败来源的定量拆分

| 目标距离(m) | 数值规划评估 | 释放速度目标残差(m/s) | 实际加速度超限 | 跟踪超限 | 速度超限 | 滑动 |
|---|---|---|---|---|---|---|
| 0.8 | 720 | 0.015 | 28/28 | 26/28 | 2/28 | 18/28 |
| 1.2 | 720 | 0.412 | 28/28 | 27/28 | 2/28 | 18/28 |
| 1.5 | 720 | 0.807 | 28/28 | 26/28 | 2/28 | 18/28 |
| 2.0 | 720 | 0.850 | 28/28 | 28/28 | 28/28 | 18/28 |
| 3.0 | 720 | 1.652 | 28/28 | 28/28 | 28/28 | 18/28 |
| 4.0 | 720 | 2.629 | 28/28 | 28/28 | 28/28 | 18/28 |
| 5.0 | 720 | 3.573 | 28/28 | 28/28 | 28/28 | 18/28 |

0.8m的运动学速度残差已很小，主要失败在受限执行器的动态跟踪、实际加速度和夹持保持，不能归为单纯运动学不可达。5m的有界Jacobian解残差明显增大，且实际运动继续触发速度/跟踪等限制。有限优化与模型/力矩预算假设同样限制结论，不能据此证明PiPER硬件绝对无法投到5m。
步长1/0.5/0.25ms复核的最大桶口位置差为 **10.636mm**；无再次碰臂片段的最大自由飞行CoM弹道位置残差为 **5.438mm**。这些误差并非零，不能把毫米级差异当作已验证的物理效应。

额外短距离诊断未改变速度/加速度预算：
- 0.3m：12次试投，合规命中0次，最大实测上升0.184m。
- 0.5m：12次试投，合规命中0次，最大实测上升0.182m。

## 视频、测试与复现当前诊断动作

[全景视频](results/basketball_throw/published/diagnostic_arc/panorama.mp4) · [侧面视频](results/basketball_throw/published/diagnostic_arc/side.mp4) · [三维实测轨迹视频](results/basketball_throw/published/diagnostic_arc/trajectory_3d.mp4) · [轨迹图](results/basketball_throw/published/diagnostic_arc/trajectory_3d.png)

旧14项测试保留，新增7项，共21项通过。最后核心回归16/16，原始OpenArm测试5/5通过。
[21项测试XML](results/basketball_throw/published/tests.xml) · [最后核心回归](results/basketball_throw/published/core_tests.xml)

静止夹持对照没有高抛目标：contact的最大实际关节加速度约4.912rad/s²，ideal约6.263rad/s²（理想解除/初始化瞬态已可触发保守5rad/s²门限）。因此当前伺服/接触瞬态模型与加速度预算本身也需要辨识，不能把本次负结果说成仅由几何可达性决定。

诊断动作初态在桌上方，蓄力时下降到桌沿前方；实际释放CoM高度0.738m，最高点0.963m。它不是从桌面拾取后直接上举的策略。

```bash
python -m piper_throw.cli throw --config results/basketball_throw/published/diagnostic_action.yaml --output results/basketball_throw/replay_diagnostic --video
```

完整原始轨迹、预试验和逐次优化记录保留本地。按需求GitHub仅发布代码、报告、配置、视频和精简汇总；重跑CLI会生成完整CSV/XML/NPZ。演示是明确标记的失败诊断动作，没有保存或宣称合规的最佳投篮动作。


同一扰动种子下跨CoM的最大单次分离为 125.157mm；把全部80个落点放在一起的最大样本分离为 140.653mm。这些是失败诊断样本的最大值，不是连续CoM空间的全局最大值。

最终扰动重复使用固定名义前馈增益，控制器不读取扰动后的真实执行器增益。早期已知增益的重复记录保留本地作方法诊断，未混入最终80次统计。

远距离CoM验证的前提尚未满足：当前只有0.8m不合规高抛动作的诊断数据，不能回答合规的更远距离动作是否放大CoM影响。需要先解决加速度/执行器跟踪与夹持保持，才能补做该部分正式结论。
