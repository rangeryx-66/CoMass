"""Hidden mass evaluator. Ground truth never enters the observation allowlist."""
from pathlib import Path
import json
import numpy as np
from .com_sensitivity import write_csv
from .optimizer import optimize, public_observation
from .scene.build import mass_properties
from .simulator import run_throw
from .throw_controller import make_commands


def evaluator_objects():
    # Private evaluator truth. Public tool requests use opaque object IDs.
    return {
        "object_0":dict(shape="box",mass=.1,com=[0,0,0]),
        "object_1":dict(shape="box",mass=.1,com=[-.030,.015,0]),
        "object_2":dict(shape="box",mass=.1,com=[.030,-.015,0]),
        "object_3":dict(shape="box",mass=.1,com=[.024,.012,.009]),
    }


class HiddenThrowService:
    """Behavioral API boundary, not an OS sandbox against a malicious client."""
    def __init__(self,scene,output_dir,objects=None):
        self._scene=dict(scene);self._objects=objects or evaluator_objects()
        self._out=Path(output_dir);self._counter=0

    def run_throw(self,object_id,params,*,seed=0):
        if object_id not in self._objects:raise ValueError("unknown opaque object ID")
        r=run_throw(params,self._objects[object_id],self._scene,
                    output_dir=self._out/f"trial_{self._counter:05d}",seed=seed)
        self._counter+=1
        return public_observation(r)


def benchmark(scene,output_dir,*,budget=60,repeats=20):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    objects=evaluator_objects();truth=[]
    for name,obj in objects.items():
        c,I=mass_properties(obj)
        truth.append(dict(object_id=name,com=c.tolist(),inertia=I.tolist(),mass=.1))
    (out/"evaluator_truth.json").write_text(json.dumps(truth,indent=2))
    rows=[]
    for mode in ["ideal","contact"]:
        s={**scene,"release_mode":mode,"search_bounds":scene.get("search_bounds",[[1,1.8],[.5,.88],[.7,1.4],[-.1,.1],[-.2,.2]])}
        # Coding-agent-equivalent optimizer receives observations only.
        service=HiddenThrowService(s,out/mode/"private_trials")
        trained={}
        for name in objects:
            def observe(params,unused_dir,oid=name):return service.run_throw(oid,params)
            summary=optimize({},s,out/mode/name/"optimization",budget=budget,observer=observe)
            trained[name]=summary["best"]["params"]
        fixed=trained["object_0"];commands=make_commands(fixed,s)
        baseline=None
        for name,obj in objects.items():
            r=run_throw(fixed,obj,s,output_dir=out/mode/"fixed"/name,frozen_commands=commands)
            if name=="object_0":baseline=r
            row=dict(release_mode=mode,experiment="fixed_action",object_id=name,trial=0,success=r["success"],
                     command_sha256=r["command_sha256"],slipped=r["slipped"],failure_reason=r["failure_reason"],
                     com_x=obj["com"][0],com_y=obj["com"][1],com_z=obj["com"][2],valid=True,delta_x=None,delta_y=None)
            if r["crossing"] and baseline["crossing"]:
                delta=np.array(r["crossing"]["position"])[:2]-np.array(baseline["crossing"]["position"])[:2]
                row.update(delta_x=float(delta[0]),delta_y=float(delta[1]))
                row.update(error_x=r["crossing"]["error_xy"][0],error_y=r["crossing"]["error_xy"][1])
            if r["release"] and baseline["release"]:
                for field in ["position","linear_velocity","angular_velocity"]:
                    delta=np.array(r["release"][field])-np.array(baseline["release"][field])
                    for a,v in zip("xyz",delta):row[f"release_delta_{field}_{a}"]=v
            rows.append(row)
            # Each optimized twin transfers unchanged to the matching reference.
            # Exact measured CoM + analytic physical inertia => exact twin; no
            # pretense of an independently measured real world.
            for i in range(repeats):
                ref_scene={**s,"release_delay_std":.004,"release_mode":"perturbed" if mode=="ideal" else "contact", "perturb_contact":mode=="contact"}
                r=run_throw(trained[name],obj,ref_scene,output_dir=out/mode/name/f"reference_{i:03d}",seed=2000+i)
                rows.append(dict(release_mode=mode,experiment="matching_reference",object_id=name,trial=i,success=r["success"],
                                 error_m=float(np.linalg.norm(r["crossing"]["error_xy"])) if r["crossing"] else None,
                                 failure_reason=r["failure_reason"],slipped=r["slipped"]))
    write_csv(out/"hidden_mass.csv",rows)
    write_csv(out/"scan.csv",[r for r in rows if r["experiment"]=="fixed_action"])
    summary=[]
    for mode in ["ideal","contact"]:
        for oid in objects:
            rr=[r for r in rows if r["release_mode"]==mode and r["object_id"]==oid and r["experiment"]=="matching_reference"]
            errors=[r["error_m"] for r in rr if r["error_m"] is not None]
            summary.append(dict(mode=mode,object_id=oid,trials=len(rr),hits=sum(r["success"] for r in rr),mean_error_m=float(np.mean(errors)) if errors else None))
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    refinements=[]
    for item in summary:
        if item["hits"] < .8*item["trials"]:
            refinements.append(refine_hidden_action(scene,out,item["object_id"],item["mode"],repeats=repeats))
    (out/"robust_summary.json").write_text(json.dumps(refinements,indent=2))
    return dict(nominal_optimization_validation=summary,robust_refinements=refinements)


def refine_hidden_action(scene,root,object_id,mode="ideal",repeats=20):
    """Public-observation-only robust selection after a fragile nominal hit.

    Nominal optimizer's public history supplies candidates; neither the selection
    loop nor geometric correction reads a hidden CoM, inertia or internal pose.
    """
    root=Path(root);out=root/mode/object_id/"robust_refinement";out.mkdir(parents=True,exist_ok=True)
    history=[json.loads(line) for line in (root/mode/object_id/"optimization/history.jsonl").read_text().splitlines()]
    good=sorted([h for h in history if h["success"]],key=lambda h:h["objective"])
    candidates=[];seen=set()
    distance=max(.1,float(np.linalg.norm(np.asarray(scene.get("bucket_position",[.8,0,0]))[:2])))
    for h in good:
        p=h["params"];key=json.dumps(p,sort_keys=True)
        if key in seen:continue
        seen.add(key);candidates.append(p)
        if len(seen)>=8:break
    # Feedback correction uses observed short/long and lateral errors only.
    for p in list(candidates):
        h=next(h for h in good if h["params"]==p);e=h["error_xy"]
        q={**p,"speed_scale":float(np.clip(p["speed_scale"]-e[0]/distance,1,1.8)),
           "direction_offset":float(np.clip(p["direction_offset"]-np.arctan2(e[1],distance),-.1,.1))}
        candidates.append(q)
    s={**scene,"release_mode":"perturbed" if mode=="ideal" else "contact","perturb_contact":mode=="contact"}
    service=HiddenThrowService(s,out/"private_trials");screen=[]
    for ci,p in enumerate(candidates):
        obs=[service.run_throw(object_id,p,seed=5000+i) for i in range(5)]
        errors=[np.linalg.norm(o["error_xy"]) for o in obs if o["error_xy"] is not None]
        screen.append(dict(candidate=ci,params=p,hits=sum(o["success"] for o in obs),trials=5,
                           mean_error_m=float(np.mean(errors)) if errors else 100.,observations=obs))
    if not screen:raise ValueError("no nominal successful candidates for robust refinement")
    best=min(screen,key=lambda h:(-h["hits"],h["mean_error_m"]))
    (out/"screening.json").write_text(json.dumps(screen,indent=2));(out/"best_params.json").write_text(json.dumps(best["params"],indent=2))
    # Held-out seeds differ from both nominal reference and robust screening.
    obs=[service.run_throw(object_id,best["params"],seed=6000+i) for i in range(repeats)]
    errors=[np.linalg.norm(o["error_xy"]) for o in obs if o["error_xy"] is not None]
    summary=dict(object_id=object_id,mode=mode,trials=repeats,hits=sum(o["success"] for o in obs),
                 mean_error_m=float(np.mean(errors)) if errors else None,best_params=best["params"],
                 screening_seeds=[5000,5004],validation_seeds=[6000,6000+repeats-1],truth_visible_to_selector=False)
    (out/"validation.json").write_text(json.dumps(obs,indent=2));(out/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary
