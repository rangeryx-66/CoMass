"""python -m piper_throw.cli --help"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import yaml


def load_config(path):
    with open(path) as f:return yaml.safe_load(f)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Simulation-only PiPER throwing benchmark (SI units)")
    parser.add_argument("command",choices=["smoke","throw","optimize","sensitivity","compare-twins","plot","video","tool","hidden-mass","search-sensitive"])
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--results",default="results")
    parser.add_argument("--output")
    parser.add_argument("--trial")
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--budget",type=int)
    parser.add_argument("--repeats",type=int,default=20)
    args=parser.parse_args(argv)
    if args.command=="plot":
        from .visualize import plot_results
        from .analysis import plot_evidence,plot_axis_scan
        extra=[plot_axis_scan(p.parent) for p in Path(args.results).rglob("scan.csv") if "pilot" not in p.parts and p.parent.name!="hidden_mass"]
        print(json.dumps(plot_results(args.results)+plot_evidence(args.results)+extra,indent=2));return
    if args.command=="video":
        if not args.trial:parser.error("video requires --trial")
        from .visualize import render_video
        print(render_video(args.trial,args.output));return
    c=load_config(args.config);obj=c.get("object",{});scene=c.get("scene",{})
    params=c.get("params",{})
    if c.get("params_file") and args.command in {"throw","sensitivity"}:
        params=json.loads(Path(c["params_file"]).read_text())
    out=Path(args.output or Path(args.results)/f"{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    budget=args.budget or c.get("search",{}).get("budget",100)
    if args.command=="tool":
        from .hidden_mass import HiddenThrowService
        service=HiddenThrowService(scene,out)
        for line in sys.stdin:
            try:
                req=json.loads(line)
                if set(req)-{"object_id","params"}:raise ValueError("unsupported request keys")
                print(json.dumps(service.run_throw(req["object_id"],req["params"])),flush=True)
            except (ValueError,KeyError,TypeError) as e:print(json.dumps({"error":str(e)}),flush=True)
        return
    if args.command in {"smoke","throw"}:
        from .simulator import run_throw
        if args.command=="smoke":
            upward=scene.get("min_release_vertical_velocity",-1)>0
            params=dict(speed_scale=1.5 if upward else 1.45,release_phase=.52 if upward else .48,swing_amplitude=1.1)
        result=run_throw(params,obj,scene,output_dir=out)
        result.pop("trajectory",None)
        if args.video:
            from .visualize import render_video
            result["video"]=render_video(out)
    elif args.command=="optimize":
        from .optimizer import optimize
        result=optimize(obj,scene,out,budget=budget,seed=c.get("search",{}).get("seed",0))
        if args.video:
            from .simulator import run_throw
            from .visualize import render_video
            run_throw(result["best"]["params"],obj,scene,output_dir=out/"best_execution")
            result["video"]=render_video(out/"best_execution")
    elif args.command=="sensitivity":
        from .com_sensitivity import scan
        result=scan(params,obj,scene,out)
    elif args.command=="compare-twins":
        from .com_sensitivity import compare_twins
        result=compare_twins(obj,scene,out,budget=budget,repeats=args.repeats)
    elif args.command=="hidden-mass":
        from .hidden_mass import benchmark
        result=benchmark(scene,out,budget=budget,repeats=args.repeats)
    elif args.command=="search-sensitive":
        from .experiment import search_sensitive
        result=search_sensitive(obj,scene,out,budget=budget,repeats=args.repeats)
    print(json.dumps(dict(output_dir=str(out.resolve()),result=result),indent=2))


if __name__=="__main__":main()
