"""Action-family/distance search for repeatable sensitivity (simulations only)."""
import json
from pathlib import Path
import numpy as np
from .com_sensitivity import write_csv
from .optimizer import optimize
from .simulator import run_throw
from .throw_controller import make_commands


def _evaluate(params,objects,scene,out,repeats):
    commands=make_commands(params,scene);rows=[];groups=[];valid=True
    for oi,obj in enumerate(objects):
        pts=[]
        for i in range(repeats):
            perturbed_scene={**scene,"release_mode":"contact" if scene.get("release_mode")=="contact" else "perturbed",
                             "perturb_contact":scene.get("release_mode")=="contact"}
            r=run_throw(params,obj,perturbed_scene,
                        output_dir=out/f"object_{oi}"/f"repeat_{i:03d}",seed=3000+i,frozen_commands=commands)
            is_valid=not r["violations"] and not r["slipped"] and r["crossing"] is not None and r["release"] is not None
            row=dict(object_index=oi,repeat=i,valid=is_valid,success=r["success"],rim_contact=r["rim_contact"],
                     seed=3000+i,command_sha256=r["command_sha256"],failure_reason=r["failure_reason"])
            if is_valid:
                xy=np.array(r["crossing"]["position"][:2]);pts.append(xy)
                row.update(landing_x=float(xy[0]),landing_y=float(xy[1]))
            else:valid=False
            rows.append(row)
        groups.append(np.asarray(pts))
    write_csv(out/"repeats.csv",rows)
    if not valid or any(len(g)!=repeats for g in groups):
        return dict(valid=False,reason="one_or_more_invalid_releases_or_safety_failures",hits=sum(r["success"] for r in rows))
    means=np.array([g.mean(axis=0) for g in groups])
    separation=float(np.max(np.linalg.norm(means[:,None]-means[None,:],axis=-1)))
    jitter=float(np.sqrt(np.mean([np.sum(np.var(g,axis=0,ddof=1)) for g in groups])))
    # A 1mm floor prevents division by near-zero numerical scatter. No claim
    # that deterministic repetition constitutes independent evidence.
    ratio=separation/max(.001,jitter)
    return dict(valid=True,separation_m=separation,within_com_rms_m=jitter,
                sensitivity_ratio=ratio,hits=sum(r["success"] for r in rows),trials=len(rows),
                repeat_count_per_com=repeats,randomization="4ms release delay, 2.5% gain, 5% log friction",seed_start=3000)


def search_sensitive(object_config,scene,output_dir,*,budget=30,repeats=20,top=5):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if object_config.get("shape")=="box":
        from .hidden_mass import evaluator_objects
        objects=list(evaluator_objects().values())
    else:
        objects=[{**object_config,"com":c} for c in [[0,0,0],[-.04,0,0],[.04,0,0],[.02,.008,.008]]]
    candidates=[]
    for family in ["translation","planar","eccentric","wrist"]:
        for distance in [.5,.8,1.,1.2]:
            name=f"{family}_{distance:.1f}m";s={**scene,"bucket_position":[distance,0,0]}
            opt=optimize(object_config,s,out/name/"optimization",budget=budget,family=family)
            params=opt["best"]["params"]
            assessment=_evaluate(params,objects,s,out/name/"screening",repeats=3)
            candidates.append(dict(name=name,family=family,distance_m=distance,params=params,
                                   nominal_target_reached=opt["target_reached"],**assessment))
            (out/"screening.json").write_text(json.dumps(candidates,indent=2))
    ranked=sorted([c for c in candidates if c["valid"] and c["nominal_target_reached"]],key=lambda c:c["sensitivity_ratio"],reverse=True)
    finalists=[]
    # Verify at least 20 perturbed trials per CoM before assigning final rank.
    for c in ranked[:max(top,8)]:
        s={**scene,"bucket_position":[c["distance_m"],0,0]}
        assessment=_evaluate(c["params"],objects,s,out/c["name"]/"validation",repeats=max(20,repeats))
        finalists.append({**c,**assessment})
    finalists=sorted([c for c in finalists if c["valid"]],key=lambda c:c["sensitivity_ratio"],reverse=True)[:top]
    (out/"top_actions.json").write_text(json.dumps(finalists,indent=2))
    write_csv(out/"ranking.csv",[{k:v for k,v in c.items() if k!="params"} for c in finalists])
    summary=dict(candidate_count=len(candidates),valid_screening=sum(c["valid"] for c in candidates),
                 verified_top_count=len(finalists),requested_top_count=top,
                 status="complete" if len(finalists)>=top else "fewer_than_five_candidates_satisfied_all_constraints",
                 release_mode="perturbed_ideal",sim_to_real=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    if any(c["sensitivity_ratio"]<1.5 or c["within_com_rms_m"]>.02 for c in finalists) or len(finalists)<top:
        return refine_sensitive(object_config,scene,out,top=top)
    return summary


def refine_sensitive(object_config,scene,output_dir,*,top=5):
    """Local action search when broad-family finalists are less separated than noise."""
    out=Path(output_dir)
    original=json.loads((out/"top_actions.json").read_text())
    if not (out/"top_actions_initial.json").exists():
        (out/"top_actions_initial.json").write_text(json.dumps(original,indent=2))
    strong=[c for c in original if c["valid"] and c["sensitivity_ratio"]>=1.5 and c["within_com_rms_m"]<=.02]
    if not strong:raise ValueError("no stable seed action for local sensitivity refinement")
    if object_config.get("shape")=="box":
        from .hidden_mass import evaluator_objects
        objects=list(evaluator_objects().values())
    else:objects=[{**object_config,"com":c} for c in [[0,0,0],[-.04,0,0],[.04,0,0],[.02,.008,.008]]]
    seed=strong[0];candidates=[]
    variants=[("wrist_pitch_offset",d) for d in [-.04,-.02,.02,.04]]+[("release_phase",d) for d in [-.015,.015]]+[("speed_scale",d) for d in [-.02,.02]]
    for i,(key,delta) in enumerate(variants):
        p={**seed["params"],key:seed["params"][key]+delta};name=f"{seed['name']}_local_{i:02d}"
        s={**scene,"bucket_position":[seed["distance_m"],0,0]}
        nominal=run_throw(p,objects[0],s,output_dir=out/name/"nominal")
        a=_evaluate(p,objects,s,out/name/"screening",3)
        candidates.append(dict(name=name,family=p["family"],distance_m=seed["distance_m"],params=p,nominal_target_reached=nominal["success"],**a))
    (out/"local_screening.json").write_text(json.dumps(candidates,indent=2))
    for c in sorted([c for c in candidates if c["valid"] and c["nominal_target_reached"]],key=lambda c:c["sensitivity_ratio"],reverse=True):
        s={**scene,"bucket_position":[c["distance_m"],0,0]}
        a=_evaluate(c["params"],objects,s,out/c["name"]/"validation",20)
        c={**c,**a}
        if c["valid"] and c["sensitivity_ratio"]>=1.5 and c["within_com_rms_m"]<=.02:strong.append(c)
        if len(strong)>=top:break
    strong=sorted(strong,key=lambda c:c["sensitivity_ratio"],reverse=True)[:top]
    (out/"top_actions.json").write_text(json.dumps(strong,indent=2))
    write_csv(out/"ranking.csv",[{k:v for k,v in c.items() if k!="params"} for c in strong])
    summary=dict(candidate_count=16+len(candidates),verified_top_count=len(strong),requested_top_count=top,
                 min_sensitivity_ratio=1.5,max_within_com_rms_m=.02,
                 status="complete" if len(strong)>=top else "fewer_than_five_stable_sensitive_candidates",
                 release_mode="perturbed_ideal",sim_to_real=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2));return summary
