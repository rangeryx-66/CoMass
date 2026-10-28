# PiPER MuJoCo 投掷与 Hidden-Mass 实验报告

日期：2026-10-28（Asia/Shanghai）。全部为仿真；没有连接真实 PiPER、测量装置或 CAN 总线，也没有训练神经网络。

## 已验证的里程碑

| 里程碑 | 实测结果 |
|---|---|
| M1 OpenArm 原始示例 | 上游未修改投掷测试 5/5 通过；最终落点误差 5.778 mm。 |
| M2 PiPER 场景与单次投掷 | 六轴/原STL/夹爪可控；实测向前上抛v_z=0.366m/s，桶口误差 2.579mm，完整外形留桶。 |
| M3 自动搜索 | 保存100次约束上抛优化及反馈尝试；动作在盒体上的20次扰动验证 19/20，平均误差 21.910mm。 |
| M4 CoM扫描 | 上抛盒体 45 个可实现配置、80 个跳过；最大落点分离 54.072mm。另有胶囊一般摆臂扫描。 |
| M5 孪生交叉验证 | 一般摆臂模型中四种孪生独立优化，再冻结动作放入准确参考仿真；每种20次扰动试投。 |

核心自动化测试9/9、OpenArm原始测试5/5。锁定 Python 3.12.4、MuJoCo 3.8.1；所有依赖与提交见 [版本审计](../piper_throw/UPSTREAM_AUDIT.md) 与 [依赖锁](../piper_throw/requirements.lock)。

安装、CLI、接口、坐标、假设与真机适配边界见 [README](../piper_throw/README.md)。

## 明确向前上抛的验收与局限

最终配置加入实际释放速度约束v_x≥0.1m/s、v_z≥0.05m/s，释放后用五次多项式平滑减速。优化器通过真实试投筛除不向上、超限或再次碰臂的动作，不人为设置飞行速度。
[上抛执行配置](../configs/upward_box.yaml) · [上抛参数](../results/upward_final_optimization/best_params.json) · [上抛连续视频](../results/upward_box_best/throw.mp4) · [上抛扫描CSV](../results/upward_com_scan/scan.csv) · [上抛CoM图](../results/upward_com_scan/com_sensitivity.png)

盒体复用已搜索到的上抛动作，20次扰动 19/20 留桶；全部试投最小v_z=0.323m/s。冻结同一动作后 45/45 个可实现CoM配置标称留桶，最大几何落点分离 54.072mm。这45次是不同确定性配置，不等于每个配置均已做20次扰动重复。
不能把所有上抛候选称作稳定：一组胶囊上抛候选仅9/20留桶；固定早释放接触动作跨四种配重仅32/80满足全部约束，失败包括再次碰臂和实际v_z转为负值。原始失败记录保留在upward_validation与upward_hidden_contact。尚不能宣称已完成所有配重的稳定上抛接触释放。

以下一般摆臂搜索未限制正竖直释放速度，部分动作越过顶点后释放；其敏感排名和孪生结果不能直接当作上抛专属结论。复现这些数据使用configs/exploratory.yaml与configs/exploratory_com_scan.yaml。

## 一般摆臂最佳动作与完整数据

[最佳参数](../results/optimization_final/best_params.json) · [搜索历史](../results/optimization_final/history.csv) · [收敛曲线](../results/optimization_final/convergence.png) · [最佳连续视频](../results/best_throw/throw.mp4) · [接触释放盒体视频](../results/hidden_best_contact/throw.mp4)

默认桶中心 (0.8,0,0)m，桶高0.22m，名义内半径0.14m，由48个侧壁碰撞体和底部组成。控制预算是仿真假设：六轴指令限速 [4,4,4,5,5,5]rad/s，实际速度检查含5%容差、加速度检查含10%容差；不是已经标定的硬件能力。
每个试投目录保留 inputs.json、scene.xml、result.json、逐步 trajectory.csv 和完整 qpos 回放 replay.npz；任何试投可用 video 命令输出连续视频。完整批量数据与独立标记的 results/pilot/ 预试验保留在本地，不混入下述统计。GitHub仅保留报告引用的汇总与示例，不包含全部逐步轨迹。
异常或20m远目标在有限搜索中明确报告 no_valid_hit_within_search_budget；这不是不可达性的形式化证明。

## 实验A：冻结动作的三维CoM扫描

胶囊140×50×50mm、100g；背景质量15%、内部球配重85%，半径10.5mm。质量、外形、抓持几何系、桶与所有控制指令保持一致，惯量通过并行轴定理随配重一起改变。请求的±20mm横向CoM部分不可制造，因此自动跳过并加入可实现的±10mm扫描。
[扫描CSV](../results/com_scan/scan.csv) · [XYZ关系曲线](../results/com_scan/axis_curves.png) · [释放/飞行贡献分解](../results/com_scan/decomposition.csv)

| 仅改变的物体CoM方向 | 相对基准的几何桶口落点变化 |
|---|---|
| X | -40mm → Δx=+7.275mm, Δy=-0.036mm；-20mm → Δx=+3.498mm, Δy=-0.014mm；+20mm → Δx=-3.168mm, Δy=+0.013mm；+40mm → Δx=-6.144mm, Δy=+0.033mm |
| Y | -10mm → Δx=+0.011mm, Δy=+0.009mm；+10mm → Δx=+0.041mm, Δy=-0.008mm |
| Z | -10mm → Δx=-7.519mm, Δy=+0.005mm；+10mm → Δx=+7.584mm, Δy=-0.005mm |

本次有效网格 11/25 最终留桶。这是确定性配置网格的结果，不能解释为同一配置的随机命中概率；25个配置的最大分离也不是连续CoM空间的全局最大值。

自由飞行无空气阻力，均匀重力下CoM不因姿态旋转产生额外气动力。几何参考点满足 p_geometry=p_com−R c；实际释放的 v_com=v_geometry+ω×Rc。
本动作主要绕近似Y轴旋转，因此沿旋转轴的CoM移动效应很小。X偏移的继承旋转速度与几何旋转项可以互相抵消；Z偏移则可明显改变前后落点。释放位置/载荷伺服变化、继承速度、飞行时间与绕CoM旋转已分别记录在贡献分解CSV中；邻近帧姿态使最大分解残差约0.16mm。小于扰动散布的变化不宣称统计显著。

## Hidden-Mass：相同盒体的内部配重

盒体120×80×50mm、2mm壳板25g、内部18mm立方配重75g。所有盒体的几何、表面摩擦、总质量和初始抓取位姿相同。以下真值只供评测器与研究报告使用，Agent接口没有这些字段。

| 评测ID | 配重描述 | 物体局部CoM (mm) |
|---|---|---|
| object_0 | 中心 | (0,0,0) |
| object_1 | 左上对置 | (−30,+15,0) |
| object_2 | 右下对置 | (+30,−15,0) |
| object_3 | 三维偏心 | (+24,+12,+9) |

每种释放模型分别先为中心配重搜索，再在各自模型内冻结同一动作，跨四种物体测试：理想释放最大分离17.458mm，接触释放17.082mm，两组标称均4/4留桶；两个模型的基准动作不同。这是每配置一次确定性试投，不是随机成功率证据。后文另有完全相同腕部动作跨释放模型的验证。接触复核动作在2/1/0.5ms三个步长下，四种物体共12次均留桶；2ms与0.5ms的最大落点差约0.42mm。
[隐藏配重全数据](../results/hidden_mass/hidden_mass.csv) · [固定动作比较图](../results/hidden_mass/hidden_fixed_action.png) · [接触步长复核](../results/contact_timestep/convergence.csv)

| 释放模型 | 物体 | 各自优化后参考环境命中 | 平均桶口误差(mm) |
|---|---|---|---|
| ideal | object_0 | 20/20 | 3.675 |
| ideal | object_1 | 20/20 | 4.556 |
| ideal | object_2 | 1/20 | 213.365 |
| ideal | object_3 | 20/20 | 4.021 |
| contact | object_0 | 20/20 | 3.726 |
| contact | object_1 | 20/20 | 4.388 |
| contact | object_2 | 20/20 | 3.912 |
| contact | object_3 | 20/20 | 4.440 |

理想释放 object_2 的首条标称命中动作非常脆弱（1/20），没有删除这条失败。通过公开历史中的候选及前后/左右误差反馈，进行5种子稳定性筛选后，另一组20个未参与筛选的种子为 20/20，平均 6.241mm。筛选器没有读取真实CoM或配重位置。
[稳定动作与筛选记录](../results/hidden_mass/ideal/object_2/robust_refinement/summary.json) · [稳定动作参数](../results/hidden_mass/ideal/object_2/robust_refinement/best_params.json)

## 实验B：敏感而重复稳定的动作搜索

一般动作族（允许越顶后释放）：四种动作族×0.5/0.8/1.0/1.2m，每候选40次优化预算；先检查关节/速度/加速度与碰撞，再按不同CoM均值最大分离 / 同CoM二维散布RMS排序。分母设1mm下限。隐藏盒体另加8个局部候选，淘汰比值<1.5或RMS>20mm的弱分离动作；初始弱候选记录保留。以下排名来自每CoM20个不同扰动种子（释放延迟σ=4ms、执行器增益σ=2.5%、摩擦对数σ=5%），不是零随机性的重复。

### 胶囊

| 排名 | 动作/目标 | CoM均值最大分离(mm) | 同CoM RMS(mm) | 比值 | 留桶/试投 |
|---|---|---|---|---|---|
| 1 | eccentric_0.8m | 13.591 | 1.352 | 10.049 | 53/80 |
| 2 | planar_0.8m | 17.794 | 2.428 | 7.328 | 52/80 |
| 3 | wrist_1.0m | 30.376 | 10.317 | 2.944 | 25/80 |
| 4 | planar_1.0m | 10.615 | 3.620 | 2.932 | 46/80 |
| 5 | wrist_0.5m | 15.394 | 10.256 | 1.501 | 46/80 |

[完整动作参数](../results/action_search/top_actions.json) · [排名CSV](../results/action_search/ranking.csv) · [排名图](../results/action_search/ranking.png)

### 隐藏盒体

| 排名 | 动作/目标 | CoM均值最大分离(mm) | 同CoM RMS(mm) | 比值 | 留桶/试投 |
|---|---|---|---|---|---|
| 1 | wrist_0.8m | 57.752 | 6.001 | 9.624 | 79/80 |
| 2 | wrist_0.8m_local_07 | 60.656 | 6.316 | 9.604 | 79/80 |
| 3 | wrist_0.8m_local_05 | 55.279 | 6.811 | 8.116 | 79/80 |
| 4 | planar_0.8m | 20.097 | 2.760 | 7.283 | 80/80 |
| 5 | eccentric_0.8m | 19.161 | 2.740 | 6.994 | 80/80 |

[完整动作参数](../results/hidden_action_search/top_actions.json) · [排名CSV](../results/hidden_action_search/ranking.csv) · [排名图](../results/hidden_action_search/ranking.png)

隐藏盒体高敏感腕部动作保持参数不变，进一步改用物理夹爪接触与开爪，并加入同样扰动：
四种配重各20次，共 80/80 留桶；CoM间均值最大分离 62.550mm，同CoM落点RMS 5.926mm，比值 10.556。没有gross slip、再次碰臂或安全违规。这是接触模型下的仿真结果，不是已标定真机结论。
[接触模式20次重复记录](../results/hidden_sensitive_contact_transfer/repeats.csv) · [理想释放视频](../results/hidden_sensitive_best/throw.mp4) · [同动作接触释放视频](../results/hidden_sensitive_contact_transfer/throw.mp4)

近似平移族的角速度小，CoM效应也小；未制造差异以获得更高分。某些远距离候选未通过当前预算的命中筛选，详见screening.json。排名只覆盖本次候选与扰动分布，不代表机器人全动作空间的最敏感全局解。

## 实验C：错误质量分布孪生的交叉验证

四种孪生各有150次独立优化预算，动作冻结后统一在CoM准确参考环境中进行20次扰动试投。物理一致的CoM变化同时改变惯量，因此此处是质量分布模型误差，不是严格固定惯量的单变量实验。

| 孪生 | 孪生内标称误差(mm) | 准确参考命中 | 准确参考平均误差(mm) |
|---|---|---|---|
| accurate | 0.081 | 20/20 | 1.326 |
| plus20 | 56.891 | 17/20 | 78.894 |
| minus20 | 0.484 | 20/20 | 4.993 |
| plus40 | 0.103 | 20/20 | 5.155 |

各孪生独立搜索得到的动作和标称拟合残差不同。特别是+20mm孪生可能停在较大的标称残差：不能把其参考环境的全部误差归因于CoM偏差。有限预算优化、碰撞捕获与最终留桶判据使目标函数非光滑；误差不必随CoM偏差单调增加。
[逐次交叉验证CSV](../results/twins/twins.csv) · [训练残差与参考误差对比图](../results/twins/twins_comparison.png)

## 模型边界与可复现性

理想释放用weld携带载荷、释放时仅关闭约束，忽略物体与机器人碰撞；接触释放无weld，使用原夹爪pad和有限速度开爪。接触中超过20mm相对抓持漂移才判为gross slip；小于该阈值不表示绝对没有微滑动。释放需持续20ms无接触，之后再次碰臂判无效。
理想/扰动理想释放使用RK4，接触使用implicitfast。小效应要结合步长和接触标定误差解释。扰动幅度尚未来自真机标定，机器人惯量/驱动器、摩擦和开爪模型也未校准。物体质量与CoM在本阶段由物理配重设计生成，接口支持之后输入真实测量值及CAD/测量惯量。
Hidden-Mass使用严格观测白名单，但同文件系统权限的恶意客户端仍能读取评测源码；严格盲测应将评测器放在独立账号或服务中。当前自动优化器实际只使用公开观测。
全部结论为仿真交叉验证，不能称为已完成真实Sim-to-Real。RealPiper接口默认且始终禁用；未发送真机指令。没有启动Isaac演员；仿真批次退出释放模型，视频Renderer通过finally关闭。
