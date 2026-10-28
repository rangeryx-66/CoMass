"""Seeded SciPy differential evolution and simulation feedback, no training."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .simulator import run_throw

NAMES = ["speed_scale","release_phase","swing_amplitude","direction_offset","wrist_pitch_offset"]
BOUNDS = [(1,2),(.32,.72),(.7,1.5),(-.15,.15),(-.2,.2)]


def public_observation(result):
    """Strict allowlist: no true mass distribution, CoM trajectory or inertia."""
    cross=result.get("crossing")
    rel=result.get("release")
    return dict(success=result["success"], error_xy=cross["error_xy"] if cross else None,
                release_geometry_position=rel["position"] if rel else None,
                rim_contact=result["rim_contact"], stayed_in_bucket=result["stayed_in_bucket"],
                failure_reason=result["failure_reason"], violations=result["violations"], slipped=result["slipped"])


def objective(obs):
    if obs["violations"]: return 100 + len(obs["violations"])
    if obs["slipped"]: return 50.
    if obs["error_xy"] is None: return 30.
    error=float(np.linalg.norm(obs["error_xy"]))
    return (-10 if obs["success"] else 1) + error + (.005 if obs["rim_contact"] else 0)


def optimize(object_config, scene_config, output_dir, *, seed=0, budget=100, family="eccentric", observer=None):
    """Optimizer sees only public observations; evaluator privately owns object.

    budget is the exact maximum evaluation count. All trials retain full raw
    telemetry and inputs in evaluator storage, outside the Agent tool response.
    """
    if budget<2:raise ValueError("optimization budget must be at least two trials")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/"history.jsonl").exists():
        raise FileExistsError("output contains optimization history; choose a fresh directory")
    bounds=scene_config.get("search_bounds", BOUNDS)
    history=[]; best=None; best_result=None
    def evaluate(x):
        nonlocal best,best_result
        if len(history)>=budget: return 1e4
        params=dict(zip(NAMES,map(float,x)));params["family"]=family
        trial_dir=out/f"trial_{len(history):04d}"
        r=(observer(params,trial_dir) if observer else run_throw(params,object_config,scene_config,output_dir=trial_dir,seed=seed))
        obs=public_observation(r) if "crossing" in r else r
        cost=objective(obs)
        row=dict(trial=len(history),params=params,objective=cost,**obs)
        history.append(row)
        with (out/"history.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        if best is None or cost<best["objective"]:
            best=row;best_result=r
        return cost
    # Start with a reproducible weak baseline and a modest speed improvement.
    evaluate([1,.48,1.1,0,0]); evaluate([1.45,.48,1.1,0,0])
    if scene_config.get("min_release_vertical_velocity",-1)>0 and len(history)<budget:
        evaluate([1.5,.52,1.1,0,0])  # measured feasible upward-throw warm start
    de_budget=max(1,int(budget*.65)-2)
    popsize=5
    nit=max(0,de_budget//(popsize*5)-1)
    differential_evolution(evaluate,bounds,seed=seed,popsize=popsize,maxiter=nit,
                           polish=False,workers=1,updating="immediate",tol=0,atol=0)
    if len(history)<budget and best["error_xy"]:
        x=np.array([best["params"][n] for n in NAMES])
        distance=np.linalg.norm(np.asarray(scene_config.get("bucket_position",[.8,0,0]))[:2])
        x[3]-=np.arctan2(best["error_xy"][1],max(distance,.1))
        evaluate(np.clip(x,np.array(bounds)[:,0],np.array(bounds)[:,1]))
    if len(history)<budget:
        x=np.array([best["params"][n] for n in NAMES])
        minimize(evaluate,x,method="Nelder-Mead",bounds=bounds,
                 options=dict(maxfev=budget-len(history),xatol=1e-4,fatol=1e-5,
                              initial_simplex=np.vstack([x]+[np.clip(x+np.eye(5)[i]*[.06,.02,.05,.025,.04],np.array(bounds)[:,0],np.array(bounds)[:,1]) for i in range(5)])))
    (out/"best_params.json").write_text(json.dumps(best["params"],indent=2))
    summary=dict(best=best, trials=len(history), target_reached=bool(best["success"]),
                 status="success" if best["success"] else "no_valid_hit_within_search_budget",
                 deterministic=scene_config.get("release_mode","ideal")!="perturbed")
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    with (out/"history.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["trial","objective","success","error_x","error_y"]+NAMES)
        for h in history:w.writerow([h["trial"],h["objective"],h["success"],*(h["error_xy"] or [None,None]),*[h["params"][n] for n in NAMES]])
    return summary
