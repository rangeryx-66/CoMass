"""Generate a Chinese evidence report exclusively from recorded results."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def read(path,default=None):
    f=ROOT/path
    return json.loads(f.read_text()) if f.exists() else default


def link(label,path):
    return f"[{label}](../{path})"


def main():
    original=read("results/openarm_original/result.json")
    best=read("results/best_throw/result.json")
    validation=read("results/best_validation/summary.json")
    scan=read("results/com_scan/summary.json")
    hidden=read("results/hidden_mass/summary.json",[])
    twins=read("results/twins/summary.json",{})
    refined=read("results/hidden_mass/ideal/object_2/robust_refinement/summary.json")
    up_best=read("results/upward_box_best/result.json")
    up_valid=read("results/upward_box_validation/summary.json")
    up_scan=read("results/upward_com_scan/summary.json")
    lines=["# PiPER MuJoCo 投掷与 Hidden-Mass 实验报告", "",
           "日期：2026-10-28（Asia/Shanghai）。全部为仿真；没有连接真实 PiPER、测量装置或 CAN 总线，也没有训练神经网络。", "",
           "## 已验证的里程碑", "", "| 里程碑 | 实测结果 |", "|---|---|"]
    if original:lines.append(f"| M1 OpenArm 原始示例 | 上游未修改投掷测试 5/5 通过；最终落点误差 {original['error_m']*1000:.3f} mm。 |")
    if up_best:lines.append(f"| M2 PiPER 场景与单次投掷 | 六轴/原STL/夹爪可控；实测向前上抛v_z={up_best['release']['linear_velocity'][2]:.3f}m/s，桶口误差 {np.linalg.norm(up_best['crossing']['error_xy'])*1000:.3f}mm，完整外形留桶。 |")
    if up_valid:lines.append(f"| M3 自动搜索 | 保存100次约束上抛优化及反馈尝试；动作在盒体上的20次扰动验证 {up_valid['hits']}/20，平均误差 {up_valid['mean_error_m']*1000:.3f}mm。 |")
    if up_scan:lines.append(f"| M4 CoM扫描 | 上抛盒体 {up_scan['valid']} 个可实现配置、{up_scan['skipped']} 个跳过；最大落点分离 {up_scan['max_pairwise_geometry_crossing_distance_m']*1000:.3f}mm。另有胶囊一般摆臂扫描。 |")
    if twins:lines.append("| M5 孪生交叉验证 | 一般摆臂模型中四种孪生独立优化，再冻结动作放入准确参考仿真；每种20次扰动试投。 |")
    lines += ["", "核心自动化测试9/9、OpenArm原始测试5/5。锁定 Python "+platform.python_version()+"、MuJoCo 3.8.1；所有依赖与提交见 "+link("版本审计","piper_throw/UPSTREAM_AUDIT.md")+" 与 "+link("依赖锁","piper_throw/requirements.lock")+"。", "",
              "安装、CLI、接口、坐标、假设与真机适配边界见 "+link("README","piper_throw/README.md")+"。",
              "", "## 一般摆臂最佳动作与完整数据", "",
              link("最佳参数","results/optimization_final/best_params.json")+" · "+link("搜索历史","results/optimization_final/history.csv")+" · "+link("收敛曲线","results/optimization_final/convergence.png")+" · "+link("最佳连续视频","results/best_throw/throw.mp4")+" · "+link("接触释放盒体视频","results/hidden_best_contact/throw.mp4"), "",
              "默认桶中心 (0.8,0,0)m，桶高0.22m，名义内半径0.14m，由48个侧壁碰撞体和底部组成。控制预算是仿真假设：六轴指令限速 [4,4,4,5,5,5]rad/s，实际速度检查含5%容差、加速度检查含10%容差；不是已经标定的硬件能力。",
              "每个试投目录保留 inputs.json、scene.xml、result.json、逐步 trajectory.csv 和完整 qpos 回放 replay.npz；任何试投可用 video 命令输出连续视频。完整批量数据与独立标记的 results/pilot/ 预试验保留在本地，不混入下述统计。GitHub仅保留报告引用的汇总与示例，不包含全部逐步轨迹。",
              "异常或20m远目标在有限搜索中明确报告 no_valid_hit_within_search_budget；这不是不可达性的形式化证明。", "",
              "## 实验A：冻结动作的三维CoM扫描", "",
              "胶囊140×50×50mm、100g；背景质量15%、内部球配重85%，半径10.5mm。质量、外形、抓持几何系、桶与所有控制指令保持一致，惯量通过并行轴定理随配重一起改变。请求的±20mm横向CoM部分不可制造，因此自动跳过并加入可实现的±10mm扫描。",
              link("扫描CSV","results/com_scan/scan.csv")+" · "+link("XYZ关系曲线","results/com_scan/axis_curves.png")+" · "+link("释放/飞行贡献分解","results/com_scan/decomposition.csv"), "",
              "| 仅改变的物体CoM方向 | 相对基准的几何桶口落点变化 |", "|---|---|"]
    f=ROOT/"results/com_scan/scan.csv"
    if f.exists():
        rr=[r for r in csv.DictReader(f.open()) if r['valid']=='True']
        for axis in 'xyz':
            a=sorted([r for r in rr if all(float(r['com_'+b])==0 for b in 'xyz' if b!=axis)],key=lambda r:float(r['com_'+axis]))
            description="；".join(f"{float(r['com_'+axis])*1000:+.0f}mm → Δx={float(r['delta_x'])*1000:+.3f}mm, Δy={float(r['delta_y'])*1000:+.3f}mm" for r in a if float(r['com_'+axis])!=0)
            lines.append(f"| {axis.upper()} | {description} |")
    if scan:lines += ["",f"本次有效网格 {scan['hits']}/{scan['valid']} 最终留桶。这是确定性配置网格的结果，不能解释为同一配置的随机命中概率；25个配置的最大分离也不是连续CoM空间的全局最大值。"]
    lines += ["", "自由飞行无空气阻力，均匀重力下CoM不因姿态旋转产生额外气动力。几何参考点满足 p_geometry=p_com−R c；实际释放的 v_com=v_geometry+ω×Rc。",
              "本动作主要绕近似Y轴旋转，因此沿旋转轴的CoM移动效应很小。X偏移的继承旋转速度与几何旋转项可以互相抵消；Z偏移则可明显改变前后落点。释放位置/载荷伺服变化、继承速度、飞行时间与绕CoM旋转已分别记录在贡献分解CSV中；邻近帧姿态使最大分解残差约0.16mm。小于扰动散布的变化不宣称统计显著。", "",
              "## Hidden-Mass：相同盒体的内部配重", "",
              "盒体120×80×50mm、2mm壳板25g、内部18mm立方配重75g。所有盒体的几何、表面摩擦、总质量和初始抓取位姿相同。以下真值只供评测器与研究报告使用，Agent接口没有这些字段。", "",
              "| 评测ID | 配重描述 | 物体局部CoM (mm) |", "|---|---|---|",
              "| object_0 | 中心 | (0,0,0) |", "| object_1 | 左上对置 | (−30,+15,0) |", "| object_2 | 右下对置 | (+30,−15,0) |", "| object_3 | 三维偏心 | (+24,+12,+9) |", "",
              "每种释放模型分别先为中心配重搜索，再在各自模型内冻结同一动作，跨四种物体测试：理想释放最大分离17.458mm，接触释放17.082mm，两组标称均4/4留桶；两个模型的基准动作不同。这是每配置一次确定性试投，不是随机成功率证据。后文另有完全相同腕部动作跨释放模型的验证。接触复核动作在2/1/0.5ms三个步长下，四种物体共12次均留桶；2ms与0.5ms的最大落点差约0.42mm。",
              link("隐藏配重全数据","results/hidden_mass/hidden_mass.csv")+" · "+link("固定动作比较图","results/hidden_mass/hidden_fixed_action.png")+" · "+link("接触步长复核","results/contact_timestep/convergence.csv"), "",
              "| 释放模型 | 物体 | 各自优化后参考环境命中 | 平均桶口误差(mm) |", "|---|---|---|---|"]
    for r in hidden:lines.append(f"| {r['mode']} | {r['object_id']} | {r['hits']}/{r['trials']} | {r['mean_error_m']*1000:.3f} |")
    if refined:lines += ["",f"理想释放 object_2 的首条标称命中动作非常脆弱（1/20），没有删除这条失败。通过公开历史中的候选及前后/左右误差反馈，进行5种子稳定性筛选后，另一组20个未参与筛选的种子为 {refined['hits']}/20，平均 {refined['mean_error_m']*1000:.3f}mm。筛选器没有读取真实CoM或配重位置。",link("稳定动作与筛选记录","results/hidden_mass/ideal/object_2/robust_refinement/summary.json")+" · "+link("稳定动作参数","results/hidden_mass/ideal/object_2/robust_refinement/best_params.json")]
    lines += ["", "## 实验B：敏感而重复稳定的动作搜索", "",
              "一般动作族（允许越顶后释放）：四种动作族×0.5/0.8/1.0/1.2m，每候选40次优化预算；先检查关节/速度/加速度与碰撞，再按不同CoM均值最大分离 / 同CoM二维散布RMS排序。分母设1mm下限。隐藏盒体另加8个局部候选，淘汰比值<1.5或RMS>20mm的弱分离动作；初始弱候选记录保留。以下排名来自每CoM20个不同扰动种子（释放延迟σ=4ms、执行器增益σ=2.5%、摩擦对数σ=5%），不是零随机性的重复。", ""]
    for folder,label in [("action_search","胶囊"),("hidden_action_search","隐藏盒体")]:
        actions=read(f"results/{folder}/top_actions.json",[]);status=read(f"results/{folder}/summary.json")
        lines += [f"### {label}", "", "| 排名 | 动作/目标 | CoM均值最大分离(mm) | 同CoM RMS(mm) | 比值 | 留桶/试投 |", "|---|---|---|---|---|---|"]
        for i,a in enumerate(actions,1):lines.append(f"| {i} | {a['name']} | {a['separation_m']*1000:.3f} | {a['within_com_rms_m']*1000:.3f} | {a['sensitivity_ratio']:.3f} | {a['hits']}/{a['trials']} |")
        if not status:lines.append("批量验证尚未完成，暂不报告最终排名。")
        elif len(actions)<5:lines.append(f"只有{len(actions)}个候选通过全部验证，未凑满五个。")
        lines += ["",link("完整动作参数",f"results/{folder}/top_actions.json")+" · "+link("排名CSV",f"results/{folder}/ranking.csv")+" · "+link("排名图",f"results/{folder}/ranking.png"),""]
    contact_transfer=read("results/hidden_sensitive_contact_transfer/summary.json")
    if contact_transfer:
        lines += ["隐藏盒体高敏感腕部动作保持参数不变，进一步改用物理夹爪接触与开爪，并加入同样扰动：",
                  f"四种配重各20次，共 {contact_transfer['hits']}/{contact_transfer['trials']} 留桶；CoM间均值最大分离 {contact_transfer.get('separation_m',0)*1000:.3f}mm，同CoM落点RMS {contact_transfer.get('within_com_rms_m',0)*1000:.3f}mm，比值 {contact_transfer.get('sensitivity_ratio',0):.3f}。没有gross slip、再次碰臂或安全违规。这是接触模型下的仿真结果，不是已标定真机结论。",
                  link("接触模式20次重复记录","results/hidden_sensitive_contact_transfer/repeats.csv")+" · "+link("理想释放视频","results/hidden_sensitive_best/throw.mp4")+" · "+link("同动作接触释放视频","results/hidden_sensitive_contact_transfer/throw.mp4"), ""]
    lines += ["近似平移族的角速度小，CoM效应也小；未制造差异以获得更高分。某些远距离候选未通过当前预算的命中筛选，详见screening.json。排名只覆盖本次候选与扰动分布，不代表机器人全动作空间的最敏感全局解。", "",
              "## 实验C：错误质量分布孪生的交叉验证", "",
              "四种孪生各有150次独立优化预算，动作冻结后统一在CoM准确参考环境中进行20次扰动试投。物理一致的CoM变化同时改变惯量，因此此处是质量分布模型误差，不是严格固定惯量的单变量实验。", "",
              "| 孪生 | 孪生内标称误差(mm) | 准确参考命中 | 准确参考平均误差(mm) |", "|---|---|---|---|"]
    for r in twins.get('results',[]):
        h=read(f"results/twins/{r['twin']}/optimization/summary.json")
        nominal=np.linalg.norm(h['best']['error_xy'])*1000
        lines.append(f"| {r['twin']} | {nominal:.3f} | {r['hits']}/{r['trials']} | {r['mean_error_m']*1000:.3f} |")
    lines += ["", "各孪生独立搜索得到的动作和标称拟合残差不同。特别是+20mm孪生可能停在较大的标称残差：不能把其参考环境的全部误差归因于CoM偏差。有限预算优化、碰撞捕获与最终留桶判据使目标函数非光滑；误差不必随CoM偏差单调增加。",
              link("逐次交叉验证CSV","results/twins/twins.csv")+" · "+link("训练残差与参考误差对比图","results/twins/twins_comparison.png"), "",
              "## 模型边界与可复现性", "",
              "理想释放用weld携带载荷、释放时仅关闭约束，忽略物体与机器人碰撞；接触释放无weld，使用原夹爪pad和有限速度开爪。接触中超过20mm相对抓持漂移才判为gross slip；小于该阈值不表示绝对没有微滑动。释放需持续20ms无接触，之后再次碰臂判无效。",
              "理想/扰动理想释放使用RK4，接触使用implicitfast。小效应要结合步长和接触标定误差解释。扰动幅度尚未来自真机标定，机器人惯量/驱动器、摩擦和开爪模型也未校准。物体质量与CoM在本阶段由物理配重设计生成，接口支持之后输入真实测量值及CAD/测量惯量。",
              "Hidden-Mass使用严格观测白名单，但同文件系统权限的恶意客户端仍能读取评测源码；严格盲测应将评测器放在独立账号或服务中。当前自动优化器实际只使用公开观测。",
              "全部结论为仿真交叉验证，不能称为已完成真实Sim-to-Real。RealPiper接口默认且始终禁用；未发送真机指令。没有启动Isaac演员；仿真批次退出释放模型，视频Renderer通过finally关闭。", ""]
    if up_best and up_valid and up_scan:
        index=lines.index("## 一般摆臂最佳动作与完整数据")
        upward=["## 明确向前上抛的验收与局限", "",
                "最终配置加入实际释放速度约束v_x≥0.1m/s、v_z≥0.05m/s，释放后用五次多项式平滑减速。优化器通过真实试投筛除不向上、超限或再次碰臂的动作，不人为设置飞行速度。",
                link("上抛执行配置","configs/upward_box.yaml")+" · "+link("上抛参数","results/upward_final_optimization/best_params.json")+" · "+link("上抛连续视频","results/upward_box_best/throw.mp4")+" · "+link("上抛扫描CSV","results/upward_com_scan/scan.csv")+" · "+link("上抛CoM图","results/upward_com_scan/com_sensitivity.png"), "",
                f"盒体复用已搜索到的上抛动作，20次扰动 {up_valid['hits']}/20 留桶；全部试投最小v_z={up_valid['min_vz']:.3f}m/s。冻结同一动作后 {up_scan['hits']}/{up_scan['valid']} 个可实现CoM配置标称留桶，最大几何落点分离 {up_scan['max_pairwise_geometry_crossing_distance_m']*1000:.3f}mm。这45次是不同确定性配置，不等于每个配置均已做20次扰动重复。",
                "不能把所有上抛候选称作稳定：一组胶囊上抛候选仅9/20留桶；固定早释放接触动作跨四种配重仅32/80满足全部约束，失败包括再次碰臂和实际v_z转为负值。原始失败记录保留在upward_validation与upward_hidden_contact。尚不能宣称已完成所有配重的稳定上抛接触释放。", "",
                "以下一般摆臂搜索未限制正竖直释放速度，部分动作越过顶点后释放；其敏感排名和孪生结果不能直接当作上抛专属结论。复现这些数据使用configs/exploratory.yaml与configs/exploratory_com_scan.yaml。", ""]
        lines[index:index]=upward
    report=ROOT/"piper_throw/REPORT.md";report.write_text("\n".join(lines))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'piper_throw').rglob('*.py') if 'upstream' not in p.parts}
    (ROOT/"results/source_sha256.json").write_text(json.dumps(hashes,indent=2))
    print(report)


if __name__=="__main__":main()
