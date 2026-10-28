"""Reproducible bounded long-arc search, simulation validation and fixed-action CoM tests."""
from pathlib import Path
import csv,json
import numpy as np
import mujoco
from scipy.optimize import differential_evolution, minimize
from .scene.build import build_model
from .basketball_controller import ballistic_plan,kinematic_release,solve_release_velocity,make_arc_commands
from .simulator import run_throw
from .hidden_mass import evaluator_objects


def serial(result):return {k:v for k,v in result.items() if k!='trajectory'}
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def plan_candidates(obj,scene,distance,*,seed=0,budget=1000,count=12,target_world=None):
    """Numerical IK/trajectory optimization; finite budget is not an impossibility proof."""
    model,_=build_model(obj,scene);data=mujoco.MjData(model)
    ids=[model.joint(f'joint{i}').id for i in range(1,7)]
    qa=model.jnt_qposadr[ids];limits=model.jnt_range[ids]
    arm={model.body(n).id for n in ['base_link']+[f'link{i}' for i in range(1,9)]}
    tested=[];seen=set();evaluations=0
    direction=np.asarray(scene.get('throw_direction',[1.,0.]));direction/=np.linalg.norm(direction)
    def evaluate(x):
        nonlocal evaluations
        evaluations+=1
        q=np.array([np.arctan2(direction[1],direction[0]),x[0],x[1],x[2],x[3],0])
        T=x[4];rise=x[5]
        p,J,W=kinematic_release(model,data,q,scene)
        target=(np.asarray(target_world,dtype=float) if target_world is not None else np.r_[p[:2]+distance*direction,scene.get('bucket_position',[0,0,0])[2]+scene.get('bucket_height',.25)])
        plan=ballistic_plan(p,target,rise=rise)
        margin=np.minimum(q-limits[:,0],limits[:,1]-q)-.04
        factor=1/(1-.12) if scene.get("arc_profile")=="smooth_acceleration" else 1.5
        vmax=np.minimum.reduce([np.asarray(scene['velocity_limits'])*.96,np.maximum(2*margin/T,.001),np.asarray(scene['acceleration_limits'])*T/factor*.9])
        v,residual=solve_release_velocity(J,plan['velocity'],vmax,W,.008)
        params=dict(family='basketball',release_joints=q.tolist(),release_joint_velocity=v.tolist(),ramp_time=float(T),brake_time=float(T),open_lead=0.,profile=scene.get('arc_profile','quartic'))
        q0=q-.5*v*T;qstop=q+.5*v*T
        ready=np.asarray(scene.get('ready_joints',[0,.8,-1.8,0,.5,0]))
        # These paths are monotone scalar interpolations in joint space. Sample
        # prepare, acceleration and braking separately without generating 7000 time samples.
        poses=np.vstack([a+(b-a)*f for a,b in [(ready,q0),(q0,q),(q,qstop)] for f in np.linspace(0,1,17)])
        # Sample the complete prepare/swing/braking command, rather than checking release alone.
        bad=0.;min_clearance=0.
        oq=model.joint('object_free').qposadr[0];hand=model.body('link6').id;object_id=model.body('object').id
        for pose in poses:
            data.qpos[qa]=pose;mujoco.mj_forward(model,data)
            point=data.xpos[hand]+data.xmat[hand].reshape(3,3)@np.asarray(scene.get('grasp_offset',[.025,0,.105]))
            data.qpos[oq:oq+3]=point
            mujoco.mju_mulQuat(data.qpos[oq+3:oq+7],data.xquat[hand],np.array([2**-.5,2**-.5,0,0]))
            mujoco.mj_forward(model,data)
            for contact in data.contact:
                a,b=(int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2])
                if (a in arm and b in arm) or ((a in arm or b in arm) and model.body('object').id not in {a,b}):
                    bad+=max(0.,-float(contact.dist))
                elif object_id in {a,b} and not (a in arm or b in arm):
                    bad+=max(0.,-float(contact.dist))
            point=data.xpos[model.body('link6').id]+data.xmat[model.body('link6').id].reshape(3,3)@np.asarray(scene.get('grasp_offset',[.025,0,.105]))
            # Full box outside ground; exact object/table/arm contacts are verified by dynamics.
            bad+=max(0.,.085-point[2])
        endpoints=np.array([q0,qstop,ready])
        joint_excess=np.maximum(endpoints-limits[:,1],0).sum()+np.maximum(limits[:,0]-endpoints,0).sum()
        cost=residual+100*bad+10*joint_excess+3*max(.8-p[2],0)+max(.15-p[0],0)
        row=dict(cost=float(cost),velocity_residual=float(residual),collision_penalty=float(bad),
                 params=params,planned_release=plan,bucket_position=np.r_[target[:2],target[2]-scene.get('bucket_height',.25)].tolist())
        key=tuple(np.round(x,5))
        if key not in seen:
            tested.append(row);seen.add(key)
        return cost
    bounds=[(.45,2.5),(-2.55,-.15),(-.45,.45),(-.95,.95),(.55,1.5),(.215,.36)]
    popsize=5;iters=max(1,budget//(popsize*len(bounds))-1)
    de=differential_evolution(evaluate,bounds,seed=seed,popsize=popsize,maxiter=iters,polish=False,workers=1,tol=0,atol=0)
    minimize(evaluate,de.x,method='Nelder-Mead',bounds=bounds,options={'maxfev':max(40,budget//5),'xatol':1e-4,'fatol':1e-5})
    tested.sort(key=lambda r:r['cost'])
    candidates=[]
    for row in tested:
        if not candidates or all(np.linalg.norm(np.asarray(row['params']['release_joints'])-np.asarray(z['params']['release_joints']))>.06 for z in candidates):
            candidates.append(row)
        if len(candidates)>=count:break
    # Always retain the best even if every candidate is kinematically infeasible.
    return dict(evaluations=evaluations,best=tested[0],candidates=candidates)


def score(result):
    # Priority: valid safety/release, high arc, retention, then landing and robustness.
    violations=result['violations'];safety=[v for v in violations if v not in {'insufficient_apex_rise','release_not_upward','apex_not_between_release_and_bucket','no_entry_after_apex','contact_before_rim_crossing'}]
    error=np.linalg.norm(result['crossing']['error_xy']) if result['crossing'] else 10
    return (1000*len(safety)+500*bool(result['slipped'])+200*bool(result['post_release_robot_contact'])+
            100*(result['release'] is None)+100*max(0,.20-result.get('apex_height_above_release',0))+
            30*int(not result.get('descending_entry_after_apex',False))+float(error)-20*int(result['success']))


def run_benchmark(config,output,*,planner_budget=None,simulation_budget=None,repeats=20):
    out=Path(output)
    if out.exists() and (out/'scan.json').exists():raise FileExistsError('choose a new basketball output directory')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'config.json',config)
    write(out/'runtime.json',dict(planner_budget=planner_budget or config['search'].get('planner_budget',1000),simulation_budget=simulation_budget or config['search'].get('simulation_budget',40),repeats=repeats))
    obj=config['object'];scene=config['scene'];search=config['search'];records=[];all_candidates=[]
    for j,distance in enumerate(search['distances']):
        target_dir=out/f'distance_{distance:.1f}m';target_dir.mkdir(exist_ok=True)
        plan=plan_candidates(obj,scene,distance,seed=search.get('seed',0)+j,budget=planner_budget or search.get('planner_budget',1000),count=10)
        write(target_dir/'planner.json',plan)
        best=None
        budget=simulation_budget or search.get('simulation_budget',40)
        candidates=plan['candidates']
        trials=[]
        def evaluate(params,bucket,mode,label,seed=0,perturb=False):
            sc={**scene,'bucket_position':bucket,'release_mode':mode,'perturb_contact':perturb}
            r=run_throw(params,obj,sc,seed=seed,output_dir=target_dir/label)
            row=dict(directory=str(target_dir/label),requested_distance=distance,objective=score(r),**serial(r))
            trials.append(row)
            return r,sc
        # All distances get genuine physical trials, even if the planner reports a residual.
        for k,candidate in enumerate(candidates[:max(1,min(len(candidates),budget//2))]):
            p=candidate['params'];bucket=candidate['bucket_position']
            r,sc=evaluate(p,bucket,'ideal',f'ideal_{k:03d}')
            rc,scc=evaluate({**p,'open_lead':.025},bucket,'contact',f'contact_{k:03d}')
            for result,scn,par,trial in [(r,sc,p,f'ideal_{k:03d}'),(rc,scc,{**p,'open_lead':.025},f'contact_{k:03d}')]:
                item=dict(objective=score(result),params=par,scene=scn,result=serial(result),trial=str(target_dir/trial))
                if best is None or item['objective']<best['objective']:best=item
                all_candidates.append(item)
            if len(trials)>=budget:break
        # Physical result feedback: adjust bounded joint speed/time and gripper lead, never object velocity.
        remaining=budget-len(trials)
        reference=best
        for k in range(remaining):
            scale=[.85,.95,1.05,1.12][k%4];lead=[.0,.015,.04,.065][(k//4)%4]
            p=dict(reference['params']);p['release_joint_velocity']=(np.array(p['release_joint_velocity'])*scale).tolist()
            # Scaling the ramp time preserves acceleration; command and actual limits still checked.
            p['ramp_time']*=max(1.,scale);p['brake_time']*=max(1.,scale);p['open_lead']=lead
            r,sc=evaluate(p,reference['scene']['bucket_position'],'contact',f'feedback_{k:03d}')
            item=dict(objective=score(r),params=p,scene=sc,result=serial(r),trial=str(target_dir/f'feedback_{k:03d}'))
            all_candidates.append(item)
            if item['objective']<best['objective']:best=item
        write(target_dir/'trials.json',trials);write(target_dir/'best.json',best)
        rec=dict(requested_distance=distance,planner_velocity_residual=plan['best']['velocity_residual'],
                 trials=len(trials),successes=sum(t['success'] for t in trials),best=best,
                 status='feasible_nominal' if any(t['success'] for t in trials) else 'no_feasible_arc_under_robot_limits',
                 caveat='finite search budget; no global infeasibility proof')
        records.append(rec);write(out/'scan.json',records)
        print(json.dumps(dict(distance=distance,status=rec['status'],trials=rec['trials'],best_rise=best['result'].get('apex_height_above_release'),violations=best['result']['violations'])),flush=True)
    # Prefer contact success; otherwise choose the best measured diagnostic, never relabel it feasible.
    ordered=sorted(all_candidates,key=lambda x:(not(x['result']['success'] and x['scene']['release_mode']=='contact'),not x['result']['success'],x['objective']))
    selected=ordered[0];write(out/'selected.json',selected)
    observed_arcs=[x for x in all_candidates if x['result'].get('apex_height_above_release',0)>=.20 and x['result'].get('descending_entry_after_apex',False)]
    diagnostic=max(observed_arcs or all_candidates,key=lambda x:x['result'].get('apex_height_above_release',0))
    write(out/'largest_observed_arc.json',diagnostic)
    if not selected['result']['success'] and observed_arcs:
        eligible=[x for x in observed_arcs if x['scene']['release_mode']=='contact'] or observed_arcs
        selected=min(eligible,key=lambda x:x['objective'])
    write(out/'com_action.json',selected)
    # Fixed-action validation belongs to the actual-distance audit stage.
    # Avoid validating the same action twice with different bucket anchors.
    make_tables(out,records,[])
    return dict(selected_trial=selected["trial"],maximum_nominal_feasible_distance_m=max((r["requested_distance"] for r in records if r["successes"]),default=None),status="success" if any(r["successes"] for r in records) else "no_feasible_arc_under_robot_limits",distance_count=len(records))


def make_tables(out,records,repeats):
    if records:
        with (out/'distance_scan.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['requested_distance_m','actual_best_range_m','status','physical_trials','hits','release_speed_m_s','release_vz_m_s','rise_m','angle_deg','planner_velocity_residual_m_s','violations'])
            for x in records:
                r=x['best']['result'];w.writerow([x['requested_distance'],r.get('horizontal_range'),x['status'],x['trials'],x['successes'],r.get('release_speed'),r.get('release_vertical_velocity'),r.get('apex_height_above_release'),r.get('release_angle_deg'),x['planner_velocity_residual'],';'.join(r['violations'])])
    if not repeats:return
    with (out/'com_repeats.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','object_id','seed','com_x','com_y','com_z','success','stayed_in_bucket','slipped','post_release_recontact','range_m','rise_m','release_speed','release_vz','flight_time','cross_x','cross_y','quaternion','release_position','release_velocity','angular_velocity','violations','command_sha256'])
        for r in repeats:
            rel=r['release'] or {};cross=r['crossing'] or {};w.writerow([r['condition'],r['object_id'],r['seed'],*r['com_local'],r['success'],r['stayed_in_bucket'],r['slipped'],r['post_release_robot_contact'],r.get('horizontal_range'),r.get('apex_height_above_release'),r.get('release_speed'),r.get('release_vertical_velocity'),r.get('flight_time'),*(cross.get('position',[None]*3)[:2]),json.dumps(cross.get('quaternion')),json.dumps(rel.get('com_position')),json.dumps(rel.get('linear_velocity')),json.dumps(rel.get('angular_velocity')),';'.join(r['violations']),r['command_sha256']])
    summary=[]
    for condition in sorted({r['condition'] for r in repeats}):
        means=[];rms=[];objects=[]
        for oid in evaluator_objects():
            rr=[r for r in repeats if r['condition']==condition and r['object_id']==oid]
            points=np.array([r['crossing']['position'][:2] for r in rr if r['crossing']])
            mean=points.mean(axis=0) if len(points) else None
            if mean is not None:means.append(mean);rms.append(float(np.sqrt(np.mean(np.sum((points-mean)**2,axis=1)))))
            objects.append(dict(object_id=oid,hits=sum(r['success'] for r in rr),retained=sum(r['stayed_in_bucket'] for r in rr),slip_count=sum(r['slipped'] for r in rr),trials=len(rr),valid_releases=sum(r['release'] is not None and not r['violations'] and not r['slipped'] for r in rr),mean_crossing=mean.tolist() if mean is not None else None,rms=rms[-1] if mean is not None else None))
        separation=max((np.linalg.norm(a-b) for a in means for b in means),default=0)
        summary.append(dict(condition=condition,objects=objects,max_mean_separation_m=float(separation),deterministic=condition!='perturbed_contact',claim_is_feasible_high_arc=all(r['success'] for r in repeats if r['condition']==condition)))
    write(out/'com_summary.json',summary)
