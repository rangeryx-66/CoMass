"""Plots from saved physical telemetry, and video by replaying saved qpos."""
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def render_video(trial_dir, output=None, fps=30):
    os.environ.setdefault("MUJOCO_GL","egl")
    import mujoco
    import imageio.v2 as imageio
    trial=Path(trial_dir);output=Path(output or trial/"throw.mp4")
    states=np.load(trial/"replay.npz")
    if (trial/"scene.xml").exists():
        from .scene.build import MODEL_DIR
        xml=ET.parse(trial/"scene.xml").getroot()
        xml.find("compiler").set("meshdir",str(MODEL_DIR/"assets"))
        model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding="unicode"))
    else:
        scene=json.loads((trial/"result.json").read_text())["scene"]
        if not Path(scene).exists():
            scene=str(Path(__file__).resolve().parent/"upstream/openarm-control/v2/openarm_mujoco_v2/throw_scene.xml")
        model=mujoco.MjModel.from_xml_path(scene)
    data=mujoco.MjData(model)
    model.vis.headlight.ambient[:]=[.45,.45,.45]
    model.vis.headlight.diffuse[:]=[.8,.8,.8]
    times=states["time"]
    camera=mujoco.MjvCamera();camera.lookat[:]=[.4,0,.6];camera.distance=2.7;camera.azimuth=115;camera.elevation=-22
    renderer=mujoco.Renderer(model,height=480,width=640)
    options=mujoco.MjvOption();options.geomgroup[3]=1
    try:
        with imageio.get_writer(str(output),fps=fps,codec="libx264",quality=7,macro_block_size=16) as writer:
            for t in np.arange(times[0],times[-1],1/fps):
                i=min(np.searchsorted(times,t),len(times)-1)
                data.qpos[:]=states["qpos"][i];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=options);writer.append_data(renderer.render())
    finally:
        renderer.close()
    return str(output)


def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(results);written=[]
    for f in root.rglob("history.csv"):
        if "pilot" in f.parts:continue
        rows=list(csv.DictReader(f.open()))
        if not rows:continue
        y=np.array([float(r["objective"]) for r in rows])
        fig,ax=plt.subplots();ax.plot(np.minimum.accumulate(y));ax.set(xlabel="simulation evaluation",ylabel="best objective (lower is better)")
        fig.tight_layout();p=f.with_name("convergence.png");fig.savefig(p,dpi=140);plt.close(fig);written.append(str(p))
    for f in root.rglob("scan.csv"):
        if "pilot" in f.parts:continue
        rows=[r for r in csv.DictReader(f.open()) if r.get("valid")=="True" and r.get("delta_x") not in {"",None}]
        if not rows:continue
        fig,axs=plt.subplots(1,3,figsize=(12,3.6))
        for j,axis in enumerate("xyz"):
            x=np.array([float(r[f"com_{axis}"])*1000 for r in rows]);y=np.array([np.hypot(float(r["delta_x"]),float(r["delta_y"]))*1000 for r in rows])
            axs[j].scatter(x,y,c=[int(r["success"]=="True") for r in rows],cmap="coolwarm");axs[j].set(xlabel=f"CoM {axis} (mm)",ylabel="geometry crossing shift (mm)")
        fig.tight_layout();p=f.with_name("com_sensitivity.png");fig.savefig(p,dpi=140);plt.close(fig);written.append(str(p))
    for f in root.rglob("trajectory.csv"):
        if "pilot" in f.parts or any(re.match(r"^(trial|repeat|reference|com)_\d",p) for p in f.parts):continue
        d=np.genfromtxt(f,delimiter=",",names=True)
        if not len(d):continue
        fig,ax=plt.subplots();ax.plot(d["geometry_x"],d["geometry_z"],label="geometry reference");ax.plot(d["com_x"],d["com_z"],label="CoM")
        ax.set(xlabel="world x (m)",ylabel="world z (m)");ax.legend();fig.tight_layout();p=f.with_name("trajectory.png");fig.savefig(p,dpi=140);plt.close(fig);written.append(str(p))
    return written
