"""Contact-driven range experiments with one six-axis piecewise quintic action."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import yaml
from .quintic import quintic


def make_range_commands(params, scene):
    ready=np.asarray(scene['ready_joints'],float)
    wind, end, velocity, stop=(np.asarray(params[k],float) for k in ('windup','swing_end','swing_velocity','brake_end'))
    prep,swing,brake=(params[k] for k in ('prepare_s','swing_s','brake_s'))
    start=scene.get('settle_time',.6)
    swing_start=start+prep; swing_end=swing_start+swing; brake_end=swing_end+brake
    dt=scene['timestep']
    times=np.arange(int(np.ceil((brake_end+scene.get('flight_settle_time',6))/dt))+1)*dt
    q=[];v=[];a=[]
    for t in times:
        if t<swing_start:
            state=quintic(t-start,prep,ready,np.zeros(6),wind,np.zeros(6))
        elif t<swing_end:
            state=quintic(t-swing_start,swing,wind,np.zeros(6),end,velocity)
        else:
            state=quintic(t-swing_end,brake,end,velocity,stop,np.zeros(6))
        q.append(state[0]);v.append(state[1]);a.append(state[2])
    q=np.asarray(q);release=swing_start+params['release_phase']*swing
    return dict(time=times,q=q,qd=np.asarray(v),qdd=np.asarray(a),release_time=release,
                swing_start_time=swing_start,swing_end_time=swing_end,brake_end_time=brake_end,
                command_sha256=hashlib.sha256(q.tobytes()+np.array([release]).tobytes()).hexdigest(),parameters=params)


def decode(x, scene, restrictions):
    limits=np.asarray(scene['model_position_limits']);margin=.04 if 'margin' in restrictions else 0.
    lo=limits[:,0]+margin;hi=limits[:,1]-margin
    ready=np.asarray(scene['ready_joints']);vmax=np.asarray(scene['velocity_limits'])
    wind=lo+(x[:6]+1)/2*(hi-lo);end=lo+(x[6:12]+1)/2*(hi-lo)
    velocity=x[12:18]*vmax;stop=lo+(x[18:24]+1)/2*(hi-lo)
    prep,swing,brake=np.exp(x[24:27]*.9+np.log([1.8,1.1,1.1]))
    if 'symmetric' in restrictions:
        wind=end-.5*velocity*swing;stop=end+.5*velocity*swing;brake=swing
    for joint in ([0] if 'j1' in restrictions else [])+([3,5] if 'wrist' in restrictions else []):
        wind[joint]=end[joint]=stop[joint]=ready[joint];velocity[joint]=0
    if 'pitch' in restrictions:
        total=ready[[1,2,4]].sum()
        for pose in (wind,end,stop):pose[4]=total-pose[1]-pose[2]
        velocity[4]=-velocity[1]-velocity[2]
    # Time dilation enforces the chosen command budgets, without changing poses.
    # End velocity scales with time, preserving the same spatial quintic path.
    knots=[(ready,np.zeros(6),wind,np.zeros(6),prep),(wind,np.zeros(6),end,velocity,swing),(end,velocity,stop,np.zeros(6),brake)]
    reserve=.6 if 'reserve' in restrictions else 1.
    scale=1.
    for a,v0,b,v1,duration in knots:
        delta=b-a
        c3=(20*delta-(8*v1+12*v0)*duration)/(2*duration**3)
        c4=(-30*delta+(14*v1+16*v0)*duration)/(2*duration**4)
        c5=(12*delta-6*(v1+v0)*duration)/(2*duration**5)
        for j in range(6):
            velocity_times=[0.,duration]
            acceleration_times=[0.,duration]
            for roots,points in [(np.roots([20*c5[j],12*c4[j],6*c3[j]]),velocity_times),
                                 (np.roots([60*c5[j],24*c4[j],6*c3[j]]),acceleration_times)]:
                points.extend(float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<duration)
            peak_v=max(abs(quintic(t,duration,a,v0,b,v1)[1][j]) for t in velocity_times)
            peak_a=max(abs(quintic(t,duration,a,v0,b,v1)[2][j]) for t in acceleration_times)
            scale=max(scale,peak_v/vmax[j],np.sqrt(peak_a/(scene['acceleration_limits'][j]*reserve)))
    scale=np.nextafter(scale*np.exp(.7*(x[28]+1)),np.inf)
    return dict(family='trebuchet',windup=wind.tolist(),swing_end=end.tolist(),swing_velocity=(velocity/scale).tolist(),
                brake_end=stop.tolist(),prepare_s=float(prep*scale),swing_s=float(swing*scale),brake_s=float(brake*scale),
                release_phase=float(.12+.83*(x[27]+1)/2))


def evaluate(job):
    x,config,restrictions,seed=job
    from .simulator import run_throw
    params=decode(np.asarray(x),config['scene'],restrictions)
    r=run_throw(params,config['object'],config['scene'],seed=seed)
    r.pop('trajectory')
    tr=r['tracking'];sc=config['scene']
    excess=sum(max(0,np.max(np.asarray(tr[k])/np.asarray(sc[limit]))-1) for k,limit in [('peak_velocity','velocity_limits'),('peak_acceleration','acceleration_limits')])
    penalty=8*excess+5*len(set(r['violations'])-{'actual_acceleration_limit','actual_velocity_limit'})
    rel=r['release']
    if rel:
        penalty+=max(0,rel['end_effector_velocity'][2])*2
        penalty+=3*(not rel['during_swing'])+3*rel['before_open_command']
        penalty+=2*(not rel['downward_swing'])
    else:penalty+=10
    if r['first_flight_obstacle']:penalty+=3
    objective=(r['range_m'] or 0)-penalty
    return dict(x=list(x),params=params,result=r,objective=float(objective))


def search(config, output, budget=128, seed=0, restrictions=(), workers=8, warm=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    from .scene.build import build_model
    m,_=build_model(config['object'],config['scene'])
    config['scene']['model_position_limits']=[m.joint(f'joint{i}').range.tolist() for i in range(1,7)]
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    rng=np.random.default_rng(seed);population=[];history=[];best=None
    # All six joint endpoints, endpoint velocity, timings and release phase are searched.
    center=np.zeros(29);center[28]=-1.
    limits=np.array(config['scene']['model_position_limits']);ready=np.array(config['scene']['ready_joints'])
    for offset in (0,6,18):center[offset:offset+6]=2*(ready-limits[:,0])/(limits[:,1]-limits[:,0])-1
    center[27]=.35
    starts=list(warm or [])
    batch_size=workers*2
    with ProcessPoolExecutor(max_workers=workers) as pool, (out/'trials.jsonl').open('w') as records:
        while len(history)<budget:
            jobs=[]
            for k in range(min(batch_size,budget-len(history))):
                n=len(history)+k
                if n<len(starts):
                    x=np.asarray(starts[n]['x']);x=np.r_[x,-1.] if len(x)==28 else x
                elif n==0 and not population:x=center.copy()
                elif n%5==0 or not population:
                    x=np.clip(center+rng.normal(0,.55,29),-1,1)
                else:
                    parent=population[rng.integers(min(len(population),10))]['x']
                    parent=list(parent)+([-1.] if len(parent)==28 else [])
                    sigma=[.30,.15,.07,.035][min(3,len(history)//max(batch_size,budget//4))]
                    if n%4==0:
                        x=np.array(parent).copy();ratio=np.exp(rng.normal(0,.16))
                        x[28]+=np.log(ratio)/.7
                        x=np.clip(x,-1,1)
                    else:
                        x=np.clip(np.array(parent)+rng.normal(0,sigma,29),-1,1)
                jobs.append((x.tolist(),config,list(restrictions),seed))
            for item in pool.map(evaluate,jobs):
                history.append(item);population.append(item);population.sort(key=lambda a:a['objective'],reverse=True);population=population[:24]
                r=item['result']
                if r['success'] and (best is None or r['range_m']>best['result']['range_m']):best=item
                item['trial']=len(history)-1
                records.write(json.dumps(item,allow_nan=False)+'\n');records.flush()
            print(json.dumps(dict(evaluations=len(history),best_range_m=best['result']['range_m'] if best else None,
                                  best_objective=population[0]['objective'],failures=population[0]['result']['failure_reason'])),flush=True)
    chosen=best or population[0]
    (out/'best.json').write_text(json.dumps(chosen,indent=2))
    (out/'population.json').write_text(json.dumps(population))
    from .simulator import run_throw
    run_throw(chosen['params'],config['object'],config['scene'],output_dir=out/'best',seed=seed)
    return chosen


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/range_throw.yaml');p.add_argument('--output',required=True)
    p.add_argument('--budget',type=int,default=128);p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--restrictions',default='');p.add_argument('--warm');p.add_argument('--replay')
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    if args.replay:
        from .simulator import run_throw
        params=json.loads(Path(args.replay).read_text())['params']
        Path(args.output).mkdir(parents=True,exist_ok=False)
        r=run_throw(params,config['object'],config['scene'],output_dir=args.output,seed=args.seed);r.pop('trajectory');print(json.dumps(r,indent=2))
    else:
        warm=json.loads(Path(args.warm).read_text()) if args.warm else None
        search(config,args.output,args.budget,args.seed,args.restrictions.split(',') if args.restrictions else [],args.workers,warm)

if __name__=='__main__':main()
