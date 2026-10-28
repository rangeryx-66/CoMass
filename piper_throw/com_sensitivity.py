"""Privileged evaluator: fixed-command CoM scans and twin cross-validation."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from .scene.build import mass_properties
from .simulator import run_throw
from .throw_controller import make_commands
from .optimizer import optimize


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def scan(params,object_config,scene,output_dir,offsets=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    commands=make_commands(params,scene)
    base=run_throw(params,object_config,scene,output_dir=out/"baseline",frozen_commands=commands)
    offsets=offsets or itertools.product([-.04,-.02,0,.02,.04],[-.02,-.01,0,.01,.02],[-.02,-.01,0,.01,.02])
    rows=[]
    for i,com in enumerate(offsets):
        obj={**object_config,"com":list(com)}
        row=dict(com_x=com[0],com_y=com[1],com_z=com[2],valid=False,skip_reason="",success=False,
                 delta_x=None,delta_y=None)
        try:_,inertia=mass_properties(obj)
        except ValueError as e:
            row["skip_reason"]=str(e);rows.append(row);continue
        r=run_throw(params,obj,scene,output_dir=out/f"com_{i:03d}",frozen_commands=commands)
        row.update(valid=True,success=r["success"],command_sha256=r["command_sha256"],release_mode=r["release_mode"],
                   flight_time=r["flight_time"],slipped=r["slipped"],failure_reason=r["failure_reason"],
                   rim_contact=r["rim_contact"],inertia_json=json.dumps(inertia.tolist()))
        if r["release"]:
            for key in ["position","com_position","linear_velocity","angular_velocity"]:
                for ax,v in zip("xyz",r["release"][key]):row[f"release_{key}_{ax}"]=v
        if r["crossing"]:
            for ax,v in zip("xyz",r["crossing"]["position"]):row[f"geometry_crossing_{ax}"]=v
            for ax,v in zip("xy",r["crossing"]["error_xy"]):row[f"error_{ax}"]=v
            if base["crossing"]:
                delta=np.array(r["crossing"]["position"])[:2]-np.array(base["crossing"]["position"])[:2]
                row.update(delta_x=float(delta[0]),delta_y=float(delta[1]))
        if r["com_crossing"]:
            for ax,v in zip("xyz",r["com_crossing"]["position"]):row[f"com_crossing_{ax}"]=v
        rows.append(row)
    write_csv(out/"scan.csv",rows)
    valid=[r for r in rows if r["valid"]]
    pts=np.array([[r["geometry_crossing_x"],r["geometry_crossing_y"]] for r in valid if "geometry_crossing_x" in r])
    maxdist=float(np.max(np.linalg.norm(pts[:,None]-pts[None,:],axis=-1))) if len(pts) else None
    summary=dict(valid=len(valid),skipped=len(rows)-len(valid),hits=sum(r["success"] for r in valid),
                 max_pairwise_geometry_crossing_distance_m=maxdist,release_mode=scene.get("release_mode","ideal"),
                 randomization=False,statistical_note="Deterministic grid, no independent statistical repetitions.",
                 command_sha256=commands["command_sha256"])
    (out/"summary.json").write_text(json.dumps(summary,indent=2));return summary


def compare_twins(reference,scene,output_dir,*,budget=60,repeats=20):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    for name,dx in [("accurate",0),("plus20",.02),("minus20",-.02),("plus40",.04)]:
        com=np.array(reference.get("com",[0,0,0]))+[dx,0,0]
        obj={**reference,"com":com.tolist()}
        try:mass_properties(obj)
        except ValueError as e:
            rows.append(dict(twin=name,status="infeasible_twin",reason=str(e)));continue
        opt=optimize(obj,scene,out/name/"optimization",budget=budget)
        params=opt["best"]["params"]
        # Freeze all joint and gripper commands when transferring to reference.
        commands=make_commands(params,scene)
        for i in range(repeats):
            ref_scene={**scene,"release_mode":"perturbed"}
            r=run_throw(params,reference,ref_scene,output_dir=out/name/f"reference_{i:03d}",seed=1000+i,frozen_commands=commands)
            rows.append(dict(twin=name,status="evaluated",trial=i,twin_optimized_success=opt["target_reached"],
                             success=r["success"],error_m=float(np.linalg.norm(r["crossing"]["error_xy"])) if r["crossing"] else None,
                             command_sha256=r["command_sha256"],failure_reason=r["failure_reason"]))
    write_csv(out/"twins.csv",rows)
    summary=[]
    for name in dict.fromkeys(r["twin"] for r in rows):
        rr=[r for r in rows if r["twin"]==name and r["status"]=="evaluated"]
        if rr:
            errors=[r["error_m"] for r in rr if r["error_m"] is not None]
            summary.append(dict(twin=name,trials=len(rr),hits=sum(r["success"] for r in rr),
                                mean_error_m=float(np.mean(errors)) if errors else None))
    (out/"summary.json").write_text(json.dumps(dict(simulation_cross_validation_only=True,results=summary),indent=2))
    return summary
