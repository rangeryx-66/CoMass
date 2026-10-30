"""Cross-test fixed-parameter specialists without revealing changed inertials."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .config import load_config
from .evaluate import evaluate


def cross_evaluate(policies, cases, output, episodes=200, seed=40000,
                   timesteps=(.001,), diagonal_only=False):
    root = Path(output)
    root.mkdir(parents=True,exist_ok=True)
    frozen = {}
    for name, source in policies.items():
        target = root / f"{name}.pt"
        shutil.copyfile(source,target)
        state = torch.load(target,map_location="cpu",weights_only=False)
        training = state["config"]["env"]
        frozen[name] = (target,state)
    rows = []
    initial_targets = {}
    for policy,(checkpoint,state) in frozen.items():
        training = state["config"]["env"]
        estimate = training.get("inertial_estimate") or {key:training["object"].get(key) for key in ("mass","com","inertia")}
        for world,physical_config in cases.items():
            if diagonal_only and world != policy:
                continue
            actual = physical_config["env"]
            for key in ("type","size","mesh","mesh_scale","friction"):
                if training["object"].get(key) != actual["object"].get(key):
                    raise ValueError(f"Geometry/friction differs for {policy}, {world}: {key}")
            for dt in timesteps:
                config = deepcopy(physical_config)
                config["device"] = "cpu"
                config["ppo"]["hidden"] = state["config"]["ppo"]["hidden"]
                period = actual["timestep"]*actual["substeps"]
                config["env"].update(inertial_estimate=deepcopy(estimate),timestep=dt,substeps=round(period/dt))
                config["env"]["randomization"].pop("timesteps",None)
                destination = root / f"{policy}__on__{world}__dt_{dt:g}"
                destination.mkdir(exist_ok=True)
                (destination/"config.json").write_text(json.dumps(config,indent=2)+"\n")
                with (destination/"episodes.jsonl").open("w") as log, redirect_stdout(log):
                    summary = evaluate(str(checkpoint),config,episodes,str(destination),seed=seed)
                results = summary["results"]
                targets = [r["target"] for r in results]
                if "targets" not in initial_targets:
                    initial_targets["targets"] = targets
                assert targets == initial_targets["targets"], "Evaluation target sets must be paired"
                starts = [r["initial_arm_target"] for r in results]
                if "arm_targets" not in initial_targets:
                    initial_targets["arm_targets"] = starts
                assert starts == initial_targets["arm_targets"], "Initial joint setpoints must be paired"
                row = {"policy":policy,"physics":world,"timestep":dt,"episodes":episodes,"seed":seed,
                       "checkpoint_steps":state["steps"],"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "hits":sum(r["success"] for r in results),"success_radius":actual["target"]["success_radius"],
                       "physical_object":actual["object"],"policy_inertial_estimate":estimate,
                       "timeouts":sum(r["outcome"]=="timeout" for r in results),
                       "mean_free_flight_m":float(np.mean([r["flight_com_horizontal_distance"] for r in results if "flight_com_horizontal_distance" in r])) if any("flight_com_horizontal_distance" in r for r in results) else None,
                       **{k:v for k,v in summary.items() if k!="results"}}
                rows.append(row)
                (root/"table.json").write_text(json.dumps(rows,indent=2)+"\n")
                print(json.dumps({k:v for k,v in row.items() if k not in ("physical_object","policy_inertial_estimate")}),flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",action="append",required=True,help="name:checkpoint")
    parser.add_argument("--case",action="append",required=True,help="name:configuration.yaml")
    parser.add_argument("--output",required=True)
    parser.add_argument("--episodes",type=int,default=200)
    parser.add_argument("--seed",type=int,default=40000)
    parser.add_argument("--timesteps",type=float,nargs="+",default=[.001])
    parser.add_argument("--diagonal-only",action="store_true")
    args = parser.parse_args()
    policies = dict(item.split(":",1) for item in args.policy)
    cases = {name:load_config(path) for name,path in (item.split(":",1) for item in args.case)}
    cross_evaluate(policies,cases,args.output,args.episodes,args.seed,args.timesteps,args.diagonal_only)


if __name__=="__main__":
    main()
