# PiPER MuJoCo throwing — simulation only

不连接 CAN，不导入或安装真机 SDK，不发送运动指令。不使用 RL 或神经网络训练。
上游固定版本和复用范围见 [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)。全部计算使用 SI 单位。

## 安装与运行

在本仓库根目录运行（已创建的环境可直接 `source .venv-piper/bin/activate`）：

```bash
bash piper_throw/tools/bootstrap.sh
source .venv-piper/bin/activate
python -m pytest piper_throw/tests -q
python -m pytest piper_throw/upstream/openarm-control/tests/test_throwing.py -q
python piper_throw/tools/run_openarm.py
python -m piper_throw.cli smoke
python -m piper_throw.cli throw --config configs/default.yaml
python -m piper_throw.cli optimize --config configs/default.yaml
python -m piper_throw.cli throw --config configs/upward_box.yaml --video
python -m piper_throw.cli sensitivity --config configs/com_scan.yaml
python -m piper_throw.cli compare-twins --config configs/com_scan.yaml
python -m piper_throw.cli hidden-mass --config configs/hidden_mass.yaml --budget 60 --repeats 20
python -m piper_throw.cli search-sensitive --config configs/exploratory.yaml --budget 40 --repeats 20
python -m piper_throw.cli search-sensitive --config configs/hidden_mass.yaml --budget 40 --repeats 20
python -m piper_throw.cli plot --results results/
MUJOCO_GL=egl python -m piper_throw.cli video --trial results/smoke
```

基础配置的 throw 是搜索前的较弱动作，未必命中；最佳动作使用 `configs/com_scan.yaml` 中的 params_file。
`default.yaml` 和 `upward.yaml` 限制实际释放v_z≥0.05m/s、v_x≥0.1m/s，释放后平滑制动；推荐上抛演示用 `upward_box.yaml`。
`upward_box.yaml` 的盒体复用已在胶囊中搜索到的动作并独立验证：标称命中、20次扰动19次留桶，45个可制造CoM配置标称全部留桶。
`exploratory.yaml` / `exploratory_com_scan.yaml` 复现未限制释放竖直速度的一般动作实验；敏感动作排名及孪生比较包含越顶后释放，不能当作上抛专属结论。
上抛的胶囊候选与早开爪接触候选仍有失败，详见 `REPORT.md`；不存在对所有配置和释放方式都保证命中的承诺。
敏感性配置默认使用交付的已验证参数；重新优化后，可将params_file改成新输出文件，并保持scene、初始构型与brake_after_release一致。
CLI 未指定 `--output` 时创建独立时间戳目录。固定 output 应使用新目录，避免混合历史。
`requirements.lock` 固定全部传递依赖；验证环境为 Python 3.12.4 / Linux x86_64 / MuJoCo 3.8.1。
机器人保留上游惯量、关节几何及 STL；position actuator 增益、完美机器人重力补偿与限速预算是仿真假设，未做真机标定。

## 动作与结果

Python API：`run_throw(params, object_config, scene_config, output_dir=...)`。
每次创建新 MjModel/MjData、显式 reset；不继承时间、控制器状态、激活、warmstart、接触或模型修改。
六轴五次多项式参数为 speed_scale、release_phase、swing_amplitude、direction_offset、wrist_pitch_offset。
translation 是近似保持末端方向的关节耦合动作族，不保证严格零角速度；报告保留实际角速度。
planar 使用中心抓取；eccentric 使用偏心抓取；wrist 额外耦合关节5旋转。

每个 trial 输出：`inputs.json`、`scene.xml`、`result.json`、逐步 `trajectory.csv`、`replay.npz`。
CSV 包含世界坐标几何参考点、CoM、姿态、实际关节位置/速度和所有关节控制指令；NPZ 保存完整 qpos，可输出连续轨迹视频。
释放状态包含物体 CoM 线速度与世界角速度，几何参考点速度通过 `v_geometry = v_com - omega × (p_com-p_geometry)` 得到。
几何参考点和 CoM 首次向下穿过桶口高度分别记录，使用线性插值；姿态取邻近积分帧（误差不超过一个步长）。
最终 success 要求有效释放、出现下降穿越、无安全违规/提前滑落，且最后0.3秒完整外形位于桶内并低速稳定。
撞沿后留桶可以成功，但 rim_contact 单独标记；CoM 穿越中心本身不会判成功。
未命中、未穿越、超限、滑落及再次碰臂均保留原因。有限预算搜索失败不构成不可达性的数学证明。

优化器为固定种子 SciPy differential evolution + 几何误差方向反馈 + Nelder-Mead。
约束上抛模式还使用一组实测试投得到的可行参数作为暖启动；它仍由真实MuJoCo验证，不预设落点。
只优化参数，无模型训练；每次调用真实 MuJoCo，优先成功留桶再减小桶口误差。
所有尝试保留输入、实际轨迹、公开观测与代价；`history.csv/jsonl`、`best_params.json`、`summary.json` 可审计。
标称重复是完全确定性的，不当作统计显著性证据。扰动验证使用不同种子；跨 CoM 使用相同种子以进行配对比较。

## 坐标与刚体质量

世界 W：z向上、x为默认正前方、y为左；机器人基座 B 默认位于 W 的 (0,0,0.35)，方向与W一致。
原始 MJCF 连杆变换定义 T_B_hand(q)。物体几何系 O 的原点是胶囊/盒体几何中心。
`T_W_O = T_W_B * T_B_hand(q) * T_hand_O`；默认偏心抓持 O 平移为 hand系 (0.025,0,0.105)m。
胶囊 O 相对 hand 无旋转，盒体相对 hand 绕x转90度，使50mm厚度方向位于夹爪开合方向，适配70mm开口。
桶 K 原点是桶底在地面的中心，坐标轴同W；桶口 z_W = bucket_position.z + bucket_height。
物体几何、初始抓取位姿与指令在每组固定动作扫描中一致；`p_W_com = p_W_O + R_W_O c_O`。

胶囊外形140×50×50mm：15%均匀背景质量 + 85%可移动球形配重（半径10.5mm），用并行轴定理构造物理惯量。
100g时配重密度约17,500kg/m³，可由钨合金实现；背景密度为轻质封装的等效近似。
盒体120×80×50mm：2mm厚六片不重叠壳板合计25g + 18mm立方配重75g（约12,860kg/m³）。
改变的是刚体 inertial.pos/fullinertia，碰撞体和几何系保持不变。
测量 m 与 c 可直接写入配置；惯量由相应配重族合理估计，真实物体若不属于此配重族应提供 CAD/测量惯量。
不是所有给定CoM都能制造：检查配重整体是否装得下，不能容纳的配置会跳过并记录原因。
胶囊扫描保留请求的±40/20mm x、±20mm y/z，并加入±10mm y/z；由于球体尺寸，部分±20mm横向配置不可实现。
`inertia_mode: fixed_diagnostic` 仅为固定惯量数值诊断，不能当作主要物理实验。

## 释放模型与数值精度

* ideal：摆臂阶段用物理 weld 传递载荷，释放时只停用约束，不设置飞行速度。理想模式忽略物体与机器人接触；地面/桶碰撞保留。这是理想夹爪假设。
* contact：从已知夹持姿态开始，无 weld，真实手指pad接触+有限速度开爪。预释放相对滑动超过20mm失败；接触断开持续20ms才确认释放，之后再次碰臂拒绝。记录实际释放延迟。
* perturbed：理想释放加零均值4ms释放延迟、2.5%执行器增益、5%对数摩擦扰动。contact 通过 `perturb_contact: true` 启用同样扰动。扰动幅度是待真机标定的假设。

ideal/perturbed 使用 RK4 抑制偏心刚体自由飞行数值漂移；contact 使用 implicitfast。默认步长2ms。
单元测试检查自由飞行重力、角动量、速度参考点转换、重置确定性及固定指令一致性。
报告中的小差异必须结合步长复核结果解释；没有空气阻力，所以均匀重力中CoM自身不会因旋转获得额外气动力。
几何参考点可因绕不同CoM旋转而产生不同轨迹；载荷、夹爪滑动与有限伺服响应也会改变释放状态。

## Hidden-Mass 与 Agent 工具

`python -m piper_throw.cli tool --config configs/hidden_mass.yaml` 是JSONL请求/响应工具：

```json
{"object_id":"object_0","params":{"speed_scale":1.2,"release_phase":0.75,"swing_amplitude":1.1}}
```

schema 见 `tools_api.schema.json`。响应只包含成功、几何误差、几何释放位置、碰撞/滑落状态，不含真实CoM、配重位置、惯量或私有轨迹。
隐藏基准的优化器实际使用这个观测白名单；ground truth 只写入 evaluator 记录。
这是实验协议隔离，不能阻止具有同一文件系统权限的恶意客户端读评测源码；严格盲测应将服务部署到独立进程/账号并只开放JSONL通道。
四种盒体分别为中心、左上/右下对置配重、三维偏心。固定动作跨物体测试和各自优化→匹配参考环境测试分别保存。
参考环境也在仿真中；本实现准确孪生采用同一质量构造，不能称为已完成真实 Sim-to-Real。

敏感动作搜索覆盖四种动作族×0.5/0.8/1.0/1.2m，每个候选先检查指令与实际安全约束。
以不同CoM几何落点均值的最大距离 / 同CoM二维波动RMS排名，分母设1mm下限。
最终候选每个CoM至少20次扰动验证。无足够有效候选时报告实际数量，不凑出前五。
隐藏盒体初筛中两组动作的分离比重复散布还小，因此继续做8个局部候选搜索；最终要求比值≥1.5且RMS≤20mm，保留原弱候选记录。
隐藏物体若标称优化后的扰动命中率低于80%，通过公开观测候选和误差反馈做5种子筛选，再用20个未参与筛选的种子验证。object_2的理想释放由1/20改进至20/20。

## 真机接口与资源释放

`real_piper.py` 定义SI单位轨迹接收接口，构造RealPiper始终报禁用，不会触发SDK导入。
未来需把轨迹映射到已核对SDK的 move_js 或 move_mit，验证固件模式、总线周期、负载、关节限位、速度、急停及开爪延迟。
SDK的编码范围不等于机械臂安全速度。当前实验不做硬件可执行性承诺。
MuJoCo批次退出即释放CPU模型/数据；视频函数finally关闭Renderer。没有启动Isaac或持续占用GPU的actor。

## GitHub 交付范围

按需求仅发布代码、报告和必要的图表/视频/汇总/最佳参数/测试记录。全文所述完整 results/ 是本地实验产物，逐步原始数据未上传。复现实验时 CLI 会在新目录重新生成输入、结果和逐步轨迹。
