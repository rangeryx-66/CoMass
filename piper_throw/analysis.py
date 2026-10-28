"""Evidence-only summary plots and free-flight geometry decomposition."""
import csv
import json
from pathlib import Path
import numpy as np
import mujoco
from .com_sensitivity import write_csv


def rotation(q):
    R=np.zeros(9);mujoco.mju_quat2Mat(R,np.asarray(q));return R.reshape(3,3)


def decompose_scan(directory):
    root=Path(directory)
    b=json.loads((root/"baseline/result.json").read_text())
    bc=np.array(b["com_local"]);br=b["release"];bt=b["flight_time"]
    if not br or bt is None or not b["crossing"]:return []
    def terms(r):
        c=np.asarray(r["com_local"]);rel=r["release"];t=r["flight_time"]
        R=rotation(rel["quaternion"]);Rc=rotation(r["crossing"]["quaternion"])
        v=np.array(rel["linear_velocity"]);omega=np.array(rel["angular_velocity"])
        spin=np.cross(omega,R@c);vg=v-spin
        return np.array(rel["position"]),v,vg,spin,R@c-Rc@c,t
    p0,v0,vg0,spin0,off0,t0=terms(b);rows=[]
    for trial in root.glob("com_*"):
        if not trial.is_dir():continue
        r=json.loads((trial/"result.json").read_text())
        if not r["crossing"] or not r["release"]:continue
        p,v,vg,spin,off,t=terms(r)
        components=dict(release_geometry_position=p-p0,carrier_velocity=(vg-vg0)*t0,
                        inherited_spin_velocity=(spin-spin0)*t0,flight_time=v*(t-t0),
                        rotation_about_com=off-off0)
        total=np.array(r["crossing"]["position"])-np.array(b["crossing"]["position"])
        residual=total-sum(components.values())
        row=dict(trial=trial.name,com_x=r["com_local"][0],com_y=r["com_local"][1],com_z=r["com_local"][2],rim_contact=r["rim_contact"])
        for name,val in {**components,"actual_shift":total,"decomposition_residual":residual}.items():
            for axis,x in zip("xy",val[:2]):row[f"{name}_{axis}_m"]=float(x)
        rows.append(row)
    write_csv(root/"decomposition.csv",rows);return rows


def plot_evidence(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(results);written=[]
    scan=root/"com_scan/scan.csv"
    if scan.exists():
        rows=[r for r in csv.DictReader(scan.open()) if r["valid"]=="True"]
        fig,axs=plt.subplots(1,3,figsize=(12,3.5))
        for ax,axis in zip(axs,"xyz"):
            rr=sorted([r for r in rows if all(float(r["com_"+b])==0 for b in "xyz" if b!=axis)],key=lambda r:float(r["com_"+axis]))
            for j,coord in enumerate("xy"):
                ax.plot([float(r["com_"+axis])*1000 for r in rr],[float(r["delta_"+coord])*1000 for r in rr],"o-",label=f"world {coord}")
            ax.set(xlabel=f"object CoM {axis} (mm)",ylabel="geometry crossing shift (mm)");ax.grid(alpha=.3);ax.legend()
        fig.suptitle("Frozen commands; physical inertia; ideal release (RK4)");fig.tight_layout()
        p=root/"com_scan/axis_curves.png";fig.savefig(p,dpi=160);plt.close(fig);written.append(str(p))
    for folder in ["action_search","hidden_action_search"]:
        f=root/folder/"top_actions.json"
        if not f.exists():continue
        rr=json.loads(f.read_text())
        if not rr:continue
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(rr))
        ax.bar(x,[r["separation_m"]*1000 for r in rr],label="between-CoM max mean separation")
        ax.errorbar(x,[r["separation_m"]*1000 for r in rr],yerr=[r["within_com_rms_m"]*1000 for r in rr],fmt="none",color="black",capsize=5,label="within-CoM RMS (not CI)")
        ax.set(xticks=x,xticklabels=[r["name"] for r in rr],ylabel="mm",title="20 perturbed throws / CoM; rank by separation / scatter")
        ax.tick_params(axis="x",rotation=20);ax.legend();fig.tight_layout();p=f.with_name("ranking.png");fig.savefig(p,dpi=150);plt.close(fig);written.append(str(p))
    f=root/"twins/twins.csv"
    if f.exists():
        rows=[r for r in csv.DictReader(f.open()) if r["status"]=="evaluated"]
        names=list(dict.fromkeys(r["twin"] for r in rows));means=[];std=[];train=[];hits=[]
        for name in names:
            rr=[r for r in rows if r["twin"]==name];e=[float(r["error_m"])*1000 for r in rr if r["error_m"]]
            means.append(np.mean(e));std.append(np.std(e,ddof=1));hits.append(sum(r["success"]=="True" for r in rr))
            h=json.loads((root/"twins"/name/"optimization/summary.json").read_text())
            train.append(np.linalg.norm(h["best"]["error_xy"])*1000)
        fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(names));ax.bar(x-.18,train,.36,label="optimized twin (nominal)")
        ax.bar(x+.18,means,.36,yerr=std,capsize=4,label="accurate reference (20 perturbed)")
        for i,n in enumerate(hits):ax.annotate(f"{n}/20",(i+.18,means[i]+std[i]),xytext=(0,4),textcoords="offset points",ha="center")
        ax.set(xticks=x,xticklabels=names,ylabel="geometry rim-plane error (mm)",title="Simulation cross-validation; error bars = sample SD");ax.legend();fig.tight_layout()
        p=f.with_name("twins_comparison.png");fig.savefig(p,dpi=160);plt.close(fig);written.append(str(p))
    f=root/"hidden_mass/hidden_mass.csv"
    if f.exists():
        rows=list(csv.DictReader(f.open()));fixed=[r for r in rows if r["experiment"]=="fixed_action"]
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,mode in zip(axs,["ideal","contact"]):
            rr=[r for r in fixed if r["release_mode"]==mode and r["delta_x"]]
            for r in rr:
                x,y=float(r["delta_x"])*1000,float(r["delta_y"])*1000
                ax.scatter(x,y);ax.annotate(r["object_id"],(x,y),xytext=(3,4),textcoords="offset points")
            ax.axhline(0,color="gray",lw=.5);ax.axvline(0,color="gray",lw=.5);ax.set(xlabel="delta world x (mm)",ylabel="delta world y (mm)",title=mode+" release; frozen commands");ax.grid(alpha=.3)
        fig.tight_layout();p=f.with_name("hidden_fixed_action.png");fig.savefig(p,dpi=160);plt.close(fig);written.append(str(p))
    return written


def plot_axis_scan(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(directory)
    rows=[r for r in csv.DictReader((root/"scan.csv").open()) if r["valid"]=="True"]
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,axis in zip(axs,"xyz"):
        rr=sorted([r for r in rows if all(float(r["com_"+b])==0 for b in "xyz" if b!=axis)],key=lambda r:float(r["com_"+axis]))
        for coord in "xy":
            ax.plot([float(r["com_"+axis])*1000 for r in rr],[float(r["delta_"+coord])*1000 for r in rr],"o-",label=f"world {coord}")
        ax.set(xlabel=f"object CoM {axis} (mm)",ylabel="geometry crossing shift (mm)");ax.grid(alpha=.3);ax.legend()
    fig.suptitle("Frozen action; physical mass distributions; SI data");fig.tight_layout()
    p=root/"axis_curves.png";fig.savefig(p,dpi=160);plt.close(fig);return str(p)
