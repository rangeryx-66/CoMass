"""Audit actual-release distances, freeze one action, plot and report measured data."""
from pathlib import Path
import csv,json
import numpy as np
from .basketball import write,serial,score,make_tables
from .basketball_controller import make_arc_commands
from .simulator import run_throw
from .hidden_mass import evaluator_objects


def audit_experiment(results,*,repeats=20):
    root=Path(results);out=root/'audit'
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    out.mkdir(exist_ok=True);scan=json.loads((root/'scan.json').read_text());audited=[];candidates=[]
    for entry in scan:
        distance=entry['requested_distance'];folder=root/f'distance_{distance:.1f}m'
        trials=json.loads((folder/'trials.json').read_text())
        for mode in ['ideal','contact']:
            choices=[r for r in trials if r['release_mode']==mode and r['release']]
            if not choices:continue
            best=min(choices,key=score);inputs=json.loads((Path(best['directory'])/'inputs.json').read_text())
            scene=inputs['scene'];p=inputs['params'];old=np.array(scene['bucket_position'])
            direction=np.asarray(scene.get('throw_direction',[1.,0.]));direction/=np.linalg.norm(direction)
            old[:2]=np.asarray(best['release']['position'][:2])+distance*direction
            scene={**scene,'bucket_position':old.tolist()}
            dest=out/f'distance_{distance:.1f}m_{mode}'
            r=run_throw(p,inputs['object'],scene,output_dir=dest)
            row=dict(requested_distance=distance,params=p,scene=scene,trial=str(dest),**serial(r))
            audited.append(row)
        # Keep genuine high-arc diagnostics for a fixed-action comparison, while preserving failure gates.
        for r in trials:
            if r.get('apex_height_above_release',0)>=.2 and r.get('descending_entry_after_apex'):
                candidates.append(r)
    write(out/'range_audit.json',audited)
    feasible=[r for r in audited if r['success']]
    if feasible:
        contact=[r for r in feasible if r['release_mode']=='contact']
        selected=max(contact or feasible,key=lambda r:r['horizontal_range'])
        inputs={'params':selected['params'],'scene':selected['scene'],'object':{'shape':'box','mass':.1,'com':[0,0,0]}}
        selected_trial=selected['trial'];diagnostic=False
    elif candidates:
        contact=[r for r in candidates if r['release_mode']=='contact']
        selected=min(contact or candidates,key=score);inputs=json.loads((Path(selected['directory'])/'inputs.json').read_text())
        distance=selected['requested_distance'];b=np.array(inputs['scene']['bucket_position']);direction=np.asarray(inputs['scene'].get('throw_direction',[1.,0.]));direction/=np.linalg.norm(direction)
        b[:2]=np.asarray(selected['release']['position'][:2])+distance*direction
        inputs['scene']={**inputs['scene'],'bucket_position':b.tolist()}
        selected_trial=str(out/'diagnostic_arc')
        r=run_throw(inputs['params'],inputs['object'],inputs['scene'],output_dir=selected_trial)
        selected=dict(trial=selected_trial,params=inputs['params'],scene=inputs['scene'],**serial(r));diagnostic=True
    else:
        selected=min(audited,key=score);selected_trial=selected['trial'];inputs={'params':selected['params'],'scene':selected['scene'],'object':{'shape':'box','mass':.1,'com':[0,0,0]}};diagnostic=True
    write(out/'fixed_action.json',dict(params=inputs['params'],scene=inputs['scene'],selected_trial=selected_trial,diagnostic=diagnostic,result=serial(selected)))
    cmd=make_arc_commands(inputs['params'],inputs['scene']);rows=[]
    for condition in ['ideal','contact','perturbed_contact']:
        for oid,obj in evaluator_objects().items():
            for k in range(repeats if condition=='perturbed_contact' else 1):
                scene={**inputs['scene'],'release_mode':'contact' if condition!='ideal' else 'ideal','perturb_contact':condition=='perturbed_contact'}
                r=run_throw(inputs['params'],obj,scene,seed=12000+k,frozen_commands=cmd,output_dir=out/'com'/condition/oid/f'repeat_{k:03d}')
                rows.append(dict(condition=condition,object_id=oid,**serial(r)))
        write(out/'com_results.json',rows)
    make_tables(out,[],rows)
    with (out/'distance_scan.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['requested_distance_m','release_mode','actual_distance_m','success','stayed_in_bucket','slipped','release_speed_m_s','release_angle_deg','release_vz_m_s','rise_m','apex_z_m','flight_time_s','ballistic_target_error_m','max_actual_velocity','max_actual_acceleration','max_tracking_error','violations'])
        for r in audited:w.writerow([r['requested_distance'],r['release_mode'],r.get('horizontal_range'),r['success'],r['stayed_in_bucket'],r['slipped'],r.get('release_speed'),r.get('release_angle_deg'),r.get('release_vertical_velocity'),r.get('apex_height_above_release'),r.get('apex_height_world'),r.get('flight_time'),r.get('ballistic_target_error'),max(r['tracking']['peak_velocity']),max(r['tracking']['peak_acceleration']),max(r['tracking']['peak_position_error']),';'.join(r['violations'])])
    # Check time-step dependence with the same fixed continuous control law and common object configurations.
    convergence=[]
    for dt in [.001,.0005,.00025]:
        for oid,obj in evaluator_objects().items():
            scene={**inputs['scene'],'release_mode':'contact','perturb_contact':False,'timestep':dt}
            r=run_throw(inputs['params'],obj,scene,output_dir=out/'timestep'/f'{dt:.5f}'/oid)
            convergence.append(dict(timestep=dt,object_id=oid,**serial(r)))
    write(out/'timestep_convergence.json',convergence)
    summary=dict(status='success' if feasible else 'no_feasible_arc_under_robot_limits',
                 maximum_nominal_feasible_range_m=max((r['horizontal_range'] for r in feasible),default=None),
                 stable_contact_maximum_range_m=None,diagnostic_action=diagnostic,
                 selected_trial=selected_trial,range_audited_trials=len(audited),
                 optimization_trials=sum(x['trials'] for x in scan),com_trials=len(rows),timestep_trials=len(convergence),
                 command_sha256=cmd['command_sha256'],fixed_action_repeats=repeats)
    # A stable range is only reported for an actually feasible selected action with >=19/20 perturbed reference hits.
    base=[r for r in rows if r['condition']=='perturbed_contact' and r['object_id']=='object_0']
    if not diagnostic and selected['release_mode']=='contact' and sum(r['success'] for r in base)>=.95*len(base):summary['stable_contact_maximum_range_m']=selected['horizontal_range']
    write(out/'summary.json',summary);return summary


def make_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(results);out=root/'audit';rows=json.loads((out/'range_audit.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for mode in ['ideal','contact']:
        rr=[r for r in rows if r['release_mode']==mode];x=[r['horizontal_range'] for r in rr]
        axs[0].plot(x,[r['release_speed'] for r in rr],'o-',label=mode)
        axs[1].plot(x,[r['apex_height_above_release'] for r in rr],'o-',label=mode)
        axs[2].plot(x,[max(r['tracking']['peak_acceleration']) for r in rr],'o-',label=mode)
    axs[1].axhline(.2,color='k',linestyle='--',label='required rise')
    axs[2].axhline(5,color='k',linestyle='--',label='assumed acceleration budget')
    for ax,y in zip(axs,['Measured release speed (m/s)','Measured free-flight rise (m)','Peak actual acceleration (rad/s²)']):ax.set(xlabel='Actual release-to-rim horizontal distance (m)',ylabel=y);ax.legend()
    fig.tight_layout();fig.savefig(out/'distance_scan.png',dpi=150);plt.close(fig)
    rr=json.loads((out/'com_results.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,4))
    for oid in evaluator_objects():
        points=np.array([r['crossing']['error_xy'] for r in rr if r['condition']=='perturbed_contact' and r['object_id']==oid and r['crossing']])
        com=evaluator_objects()[oid]['com']
        if not len(points):continue
        errors=np.linalg.norm(points,axis=1)*1000
        for k,ax in enumerate(axs):ax.errorbar(com[k]*1000,errors.mean(),yerr=errors.std(),fmt='o',label=oid,capsize=4)
    for ax,a in zip(axs,'xyz'):ax.set(xlabel=f'Object CoM {a} (mm)',ylabel='Measured rim-plane error (mm)');ax.legend()
    fig.suptitle('Fixed-action diagnostic; 20 perturbed contact trials per object; failed safety gates remain failures')
    fig.tight_layout();fig.savefig(out/'com_sensitivity.png',dpi=150);plt.close(fig)
    # Separate geometric-reference and COM crossings; attitude separation is computed from paired quaternions.
    decomposition=[]
    for condition in ['ideal','contact','perturbed_contact']:
        base={r['seed']:r for r in rr if r['condition']==condition and r['object_id']=='object_0'}
        for r in rr:
            if r['condition']!=condition or r['seed'] not in base:continue
            b=base[r['seed']];a=r.get('crossing');ab=b.get('crossing');rel=r.get('release');br=b.get('release')
            if not a or not ab or not rel or not br:continue
            dq=np.clip(abs(np.dot(a['quaternion'],ab['quaternion'])),0,1)
            decomposition.append(dict(condition=condition,object_id=r['object_id'],seed=r['seed'],
                  release_position_delta_m=float(np.linalg.norm(np.array(rel['com_position'])-br['com_position'])),
                  release_velocity_delta_m_s=float(np.linalg.norm(np.array(rel['linear_velocity'])-br['linear_velocity'])),
                  geometry_crossing_delta_m=float(np.linalg.norm(np.array(a['position'][:2])-ab['position'][:2])),
                  com_crossing_delta_m=(float(np.linalg.norm(np.array(r['com_crossing']['position'][:2])-b['com_crossing']['position'][:2])) if r.get('com_crossing') and b.get('com_crossing') else None),
                  attitude_difference_deg=float(np.degrees(2*np.arccos(dq))),slipped=r['slipped'],violations=';'.join(r['violations'])))
    if decomposition:
        with (out/'release_flight_decomposition.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(decomposition[0]));w.writeheader();w.writerows(decomposition)
    return str(out/'distance_scan.png')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--repeats',type=int,default=20);p.add_argument('--video',action='store_true');args=p.parse_args()
    result=audit_experiment(args.results,repeats=args.repeats);make_plots(args.results);numerical_checks(args.results);write_report(args.results)
    if args.video:
        from .visualize import render_video,plot_arc_trajectory,render_trajectory_video
        trial=result['selected_trial'];plot_arc_trajectory(trial)
        for view in ['panorama','side']:render_video(trial,Path(trial)/f'{view}.mp4',view=view,annotate=True)
        render_trajectory_video(trial)
    print(json.dumps(result,indent=2))




def write_report(results,report_path=None):
    """Evidence-only report. No interpolation of missing successful trials."""
    import os
    root=Path(results);out=root/'audit';report=Path(report_path or out/'BASKETBALL_REPORT.md')
    summary=json.loads((out/'summary.json').read_text());ranges=json.loads((out/'range_audit.json').read_text());com=json.loads((out/'com_summary.json').read_text())
    fixed=json.loads((out/'fixed_action.json').read_text());r=fixed['result']
    def link(label,path):return f"[{label}]({os.path.relpath(path,report.parent)})"
    def fmt(x,n=3):return '未找到 / 不适用' if x is None else f'{x:.{n}f}'
    lines=['# Basketball-Style Long-Arc Throwing 实验报告','',
           '2026-10-28，纯 MuJoCo 仿真。旧 REPORT.md、14 项测试及已完成实验保留；没有使用 RL、神经网络训练、真实机械臂或 CAN。','',
           '## 结论与边界','',
           f"状态：`{summary['status']}`。在本次模型、控制器、动作族与有限预算下，最大合规高弧线距离：{fmt(summary['maximum_nominal_feasible_range_m'])} m；经扰动接触释放验证的最大稳定距离：{fmt(summary['stable_contact_maximum_range_m'])} m。",
           '没有把失败候选、只满足上升高度的动作或单次理想命中报告为稳定高抛。负结果不能证明真实 PiPER 在所有控制器下均不可行。尤其 5 rad/s²、力矩预算与执行器响应尚未真机标定。','',
           '## 规划、场景与约束','',
           'PiPER 基座固定在 0.75 m 桌面，完整桌板、四条桌腿、原机械臂碰撞体、120×80×50 mm/100 g 隐藏配重盒、桶底和48段桶壁均参与物理接触。桶口高度0.25 m。固定基座与桌板安装面作为刚性装配排除相互碰撞；活动连杆及完整盒体保留碰撞。初始为已知的桌上方夹持姿态，随后准备、蓄力、抛出和制动；没有实现从桌面拾取。','',
           'Ballistic planner 由三维释放点、三维桶口和期望升高求 v_z=√(2gh)、下降过桶口时间与水平速度。MuJoCo 六轴 Jacobian 加有界线性最小二乘解释放关节速度，SciPy differential_evolution/Nelder-Mead 联合搜索释放关节姿态与加速时间；J2/J3/J4/J5可协同，未局限J2。全程检查准备、蓄力、制动的关节范围及碰撞。','',
           '新增平滑加速度平台轨迹族与原五次多项式准备段复用；保留平滑step速度轨迹对照。期望速度只用于求关节指令；自由刚体速度没有被手动设置。理想释放仅解除weld，接触释放无weld并真实开爪。','',
           '六轴指令及实际速度均限制3 rad/s，SDK关节位置与上游MJCF取交集；实际/指令加速度预算5 rad/s²，无旧实验5%/10%容差。检查实际跟踪误差、完整制动、滑动和再次碰臂。力矩限制[12,12,12,3,3,3] N·m为未标定的保守实验假设；PD增益、名义负载补偿与开爪时延亦为假设。完美被动重力补偿关闭，补偿通过相同的受限执行器施加。',
           link('参数来源与假设','piper_throw/HARDWARE_LIMITS.md')+'。官方来源：[AgileX SDK固定提交](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/CHANGELOG.MD)。','',
           '## 坐标与指标定义','',
           '世界系W：x向前、y向左、z向上；基座B位于(0,0,0.75)m，方向同W。几何系O原点在盒体几何中心，T_WO=T_WB·T_Bhand(q)·T_handO；偏心抓取平移为hand系(0.025,0,0.105)m，盒体相对hand绕x转90°。p_Wcom=p_WO+R_WO·c_O。桶系原点为桶底中心，方向同W，可配置任意三维平移。',
           'release_speed/linear_velocity为CoM速度，release.position为几何参考点；horizontal_range专指实际释放几何点到目标桶口中心的水平距离，并非未命中时物体实际飞行的水平位移。release_angle以rad保存，同时提供deg。apex为CoM实测自由飞行最高点，另存几何点最高高度；flight_time为几何参考点从释放到首次下降穿过桶口高度的时间。ballistic_target_error是以实际CoM释放状态解析预测的桶口平面误差，几何/CoM的实测穿越误差分别保存。所有长度m、质量kg、时间s、角度rad。','',
           '## 距离扫描与真实自由飞行','',
           '搜索先使用运动学释放位置构造候选目标。最终审计额外运行：依据该动作实测释放几何参考点确定桶位置，再从相同初态重新执行，确保下表的距离按实际释放点到桶口中心的水平距离计算。桶在每次正式积分开始前固定，不在飞行途中移动。所有标称距离均以实测距离核对，不用基座到桶距离代替。',
           f"主扫描真实试投 {summary['optimization_trials']} 次，另有 {summary['range_audited_trials']} 次距离审计、{summary['com_trials']} 次固定动作CoM试投、{summary['timestep_trials']} 次步长复核。数值规划评估次数见各目录 planner.json；预试验与其他轨迹族不混入主扫描。",'',
           '| 目标距离(m) | 模型 | 实测水平距离(m) | 释放速度(m/s) | 仰角(°) | v_z(m/s) | 实测升高(m) | 最终成功 | 滑动 | 主要失败 |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for x in ranges:lines.append(f"| {x['requested_distance']:.1f} | {x['release_mode']} | {fmt(x.get('horizontal_range'))} | {fmt(x.get('release_speed'))} | {fmt(x.get('release_angle_deg'),1)} | {fmt(x.get('release_vertical_velocity'))} | {fmt(x.get('apex_height_above_release'))} | {x['success']} | {x['slipped']} | {', '.join(x['violations'])} |")
    lines+=['',link('逐距离CSV',out/'distance_scan.csv')+' · '+link('实测速度/升高/加速度图',out/'distance_scan.png'),'',
            '升高与最高点从已积分的CoM轨迹测量，并在第一次环境接触或桶口穿越处截断；同时保存几何参考点与CoM轨迹。球心/几何中心穿过桶口不等于完整入桶。成功仍要求完整外形持续低速留桶、无安全违规、无提前滑脱且真实先升后降。', '',
            '## 固定动作的CoM复核','',
            f"所选动作是否仅供失败诊断：**{summary['diagnostic_action']}**。基准实测释放速度 {fmt(r.get('release_speed'))} m/s，v_z={fmt(r.get('release_vertical_velocity'))} m/s；升高 {fmt(r.get('apex_height_above_release'))} m，最高点世界高度 {fmt(r.get('apex_height_world'))} m。动作、桶位置、物体几何与质量保持不变，四种物体沿用旧物理一致壳体+内部配重惯量。",
            '理想与无扰动接触各配置只运行一次，属于确定性比较；扰动接触每配置20次，四种物体使用配对相同种子。释放延迟σ=4ms、增益σ=2.5%、摩擦对数σ=5%；没有为不同配重人为添加速度或落点偏置。','',
            '| 条件 | CoM间均值最大分离(mm) | object_0 命中 | object_1 命中 | object_2 命中 | object_3 命中 | 合规高抛结论 |',
            '|---|---|---|---|---|---|---|']
    for x in com:
        counts=[f"合规{o['hits']}/{o['trials']}；留桶{o.get('retained',0)}/{o['trials']}" for o in x['objects']]
        lines.append(f"| {x['condition']} | {x['max_mean_separation_m']*1000:.3f} | "+' | '.join(counts)+f" | {x['claim_is_feasible_high_arc']} |")
    lines+=['',link('逐次CoM CSV',out/'com_repeats.csv')+' · '+link('CoM偏移与误差图',out/'com_sensitivity.png')+' · '+link('释放/飞行/姿态差异分解',out/'release_flight_decomposition.csv'),'',
            '这些均值分离仅比较有桶口下降穿越记录的试投；缺失穿越不填零，缺失与失败仍计入分母。若动作违反安全或夹持条件，其分离只作失败诊断，不能用来宣称更远且稳定的高抛放大了CoM影响。旧0.8m实验的62.55mm属于另一动作与释放协议，不能直接据两个最大值作显著性判断。四种配重同时改变多个轴，也不能把散点图解释为单轴因果效应。', '',
            '释放位置/速度、空中角速度、桶口姿态以及几何/CoM过平面位置均独立保存。无空气阻力时自由飞行CoM obeys gravity；差异可来自载荷跟踪、接触滑动、释放状态和绕不同CoM旋转的几何参考点。存在滑动的候选不能把全部差异归因于纯自由飞行。','',
            '## 可复现命令与视频','',
            '```bash\nbash piper_throw/tools/bootstrap.sh\nsource .venv-piper/bin/activate\npython -m pytest piper_throw/tests piper_throw/upstream/openarm-control/tests/test_throwing.py -q\npython -m piper_throw.cli basketball --config configs/basketball_throw.yaml --output results/basketball_throw/new_run --budget 28 --repeats 20 --video\n```','',
            '运行新实验时使用新输出目录，不覆盖本次记录。单个固定三维桶目标也可通过 plan_candidates(..., target_world=[x,y,z_rim]) 规划，再 run_throw 执行。', '',
            '全景/侧面视频由同一份实际qpos回放生成，20/30fps不改变仿真时间，字幕显示实测速度、距离、升高与success。三维轨迹视频只绘制测得的几何/CoM样本。若无合规动作，演示标为失败诊断，不提供伪装成功的视频。','',
            '## 限制与后续','',
            '本次不能声称已完成稳定远距离投篮或真实Sim-to-Real。5m目标的弹道速度需求、Jacobian有界求解残差、实际速度/加速度、跟踪与夹持失败都已分别记录。有限预算和当前动作族/伺服近似也可能限制结果，不能仅凭失败把全部责任归为硬件最大速度。下一步应先做离线执行器/夹持模型辨识和更强的约束轨迹优化；本阶段仍不连接真机。','']
    lines+=['## 重复散布与数值复核','', '| 条件 | 物体 | 同CoM二维RMS(mm) | 滑动次数 |', '|---|---|---|---|']
    for group in com:
        for o in group['objects']:lines.append(f"| {group['condition']} | {o['object_id']} | {fmt(None if o['rms'] is None else o['rms']*1000)} | {o.get('slip_count',0)}/{o['trials']} |")
    lines+=['', '留桶只是几何和静止判据，不能抵消滑动、安全或高弧线失败。确定性单次比较的RMS=0不代表统计稳定。',link('步长复核CSV',out/'timestep_convergence.csv')+' · '+link('自由飞行CoM弹道一致性',out/'freeflight_numerics.csv'), '']
    report.write_text('\n'.join(lines));return str(report)



def numerical_checks(results):
    root=Path(results);out=root/'audit'
    convergence=json.loads((out/'timestep_convergence.json').read_text());finest={r['object_id']:r for r in convergence if r['timestep']==.00025}
    rows=[]
    for r in convergence:
        b=finest[r['object_id']];a=r.get('crossing');bb=b.get('crossing')
        shift=float(np.linalg.norm(np.asarray(a['position'][:2])-bb['position'][:2])) if a and bb else None
        rows.append(dict(object_id=r['object_id'],timestep_s=r['timestep'],success=r['success'],retained=r['stayed_in_bucket'],slipped=r['slipped'],rise_m=r.get('apex_height_above_release'),crossing_shift_vs_finest_m=shift,max_actual_acceleration=max(r['tracking']['peak_acceleration']),violations=';'.join(r['violations'])))
    with (out/'timestep_convergence.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    checks=[]
    for r in json.loads((out/'com_results.json').read_text()):
        if not r['release'] or r['post_release_robot_contact']:continue
        index=r['seed']-12000
        path=(Path(r['trial_directory'])/'trajectory.csv' if r.get('trial_directory') else out/'com'/r['condition']/r['object_id']/f'repeat_{index:03d}'/'trajectory.csv')
        if not path.exists():continue
        d=np.genfromtxt(path,delimiter=',',names=True);rel=r['release'];end=r['com_crossing']['time'] if r['com_crossing'] else d['time'][-1]
        if r.get('airborne_environment_contacts'):end=min(end,min(c['time'] for c in r['airborne_environment_contacts']))
        flight=d[(d['time']>rel['time']+.005)&(d['time']<end-.005)]
        if not len(flight):continue
        t=flight['time']-rel['time'];p=np.asarray(rel['com_position'])+t[:,None]*np.asarray(rel['linear_velocity'])
        p[:,2]-=9.81*t*t/2
        actual=np.c_[flight['com_x'],flight['com_y'],flight['com_z']]
        error=float(np.max(np.linalg.norm(actual-p,axis=1)))
        checks.append(dict(condition=r['condition'],object_id=r['object_id'],seed=r['seed'],max_freeflight_com_ballistic_error_m=error))
    if checks:
        with (out/'freeflight_numerics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    return dict(max_timestep_crossing_shift_m=max((r['crossing_shift_vs_finest_m'] for r in rows if r['crossing_shift_vs_finest_m'] is not None),default=None),max_freeflight_com_ballistic_error_m=max((r['max_freeflight_com_ballistic_error_m'] for r in checks),default=None))

if __name__=='__main__':main()
