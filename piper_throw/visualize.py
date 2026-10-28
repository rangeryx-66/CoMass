"""Plots from saved physical telemetry, and video by replaying saved qpos."""
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def render_video(trial_dir, output=None, fps=30, *, view=None, annotate=False):
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
    result=json.loads((trial/"result.json").read_text())
    if view:
        trajectory=np.genfromtxt(trial/"trajectory.csv",delimiter=",",names=True)
        until=result["crossing"]["time"] if result.get("crossing") else times[-1]
        flight=trajectory[trajectory['time']<=until]
        xyz=np.c_[flight['geometry_x'],flight['geometry_y'],flight['geometry_z']]
        bucket=data.xpos[model.body('bucket').id] if model.body('bucket').id else np.zeros(3)
        mujoco.mj_forward(model,data);bucket=data.xpos[model.body('bucket').id].copy()
        points=np.vstack([xyz,bucket,[0,0,0],[0,0,1.5],[-1,-.5,0],[0,.5,0]])
        lo=points.min(axis=0);hi=points.max(axis=0)
        camera.lookat[:]=(lo+hi)/2;camera.distance=max(3.,float(np.linalg.norm(hi-lo))*1.6)
        camera.azimuth=90 if view=='side' else 125;camera.elevation=-8 if view=='side' else -23
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,848 if view else 640)
    model.vis.global_.offheight=max(model.vis.global_.offheight,480)
    renderer=mujoco.Renderer(model,height=480,width=848 if view else 640)
    options=mujoco.MjvOption();options.geomgroup[3]=1
    try:
        with imageio.get_writer(str(output),fps=fps,codec="libx264",quality=7,macro_block_size=16) as writer:
            for t in np.arange(times[0],times[-1],1/fps):
                i=min(np.searchsorted(times,t),len(times)-1)
                data.qpos[:]=states["qpos"][i];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=options)
                frame=renderer.render()
                if annotate:
                    from PIL import Image,ImageDraw
                    pic=Image.fromarray(frame);draw=ImageDraw.Draw(pic)
                    text=(f"SIMULATION | {view} | t={t:.2f}s | success={result['success']} retained={result['stayed_in_bucket']}\n"
                          f"actual v={result.get('release_speed',0):.3f} m/s | vz={result.get('release_vertical_velocity',0):.3f} m/s\n"
                          f"release-to-rim horizontal distance={result.get('horizontal_range',0):.3f} m\n"
                          f"measured rise={result.get('apex_height_above_release',0):.3f} m | apex z={result.get('apex_height_world',0):.3f} m")
                    draw.rectangle((5,5,750,76),fill=(0,0,0));draw.multiline_text((10,10),text,fill=(255,255,255),spacing=2)
                    if result.get('release') and t>=result['release']['time']:
                        phase='ASCENDING' if t<result.get('apex_time',0) else 'AFTER APEX'
                        draw.text((10,82),phase,fill=(255,180,30))
                    if result.get("violations") or result.get("slipped"):
                        draw.text((10,98),"DIAGNOSTIC: SAFETY / GRASP GATES FAILED",fill=(255,100,100))
                    frame=np.asarray(pic)
                writer.append_data(frame)
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


def plot_arc_trajectory(trial_dir):
    """Plot measured world geometry and COM without synthesizing a flight path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trial=Path(trial_dir);r=json.loads((trial/'result.json').read_text())
    rows=np.genfromtxt(trial/'trajectory.csv',delimiter=',',names=True)
    rel=r.get('release')
    if not rel:return None
    end=r['crossing']['time'] if r.get('crossing') else rows['time'][-1]
    d=rows[(rows['time']>=rel['time'])&(rows['time']<=end)]
    fig=plt.figure(figsize=(11,4.5));ax=fig.add_subplot(121,projection='3d');side=fig.add_subplot(122)
    for prefix,label in [('geometry','Geometry reference'),('com','CoM')]:
        x,y,z=(d[prefix+'_'+a] for a in 'xyz')
        ax.plot(x,y,z,label=label);side.plot(x,z,label=label)
    a=r['apex_position'];side.scatter([a[0]],[a[2]],marker='x',label='Measured apex')
    ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)');ax.legend()
    side.set(xlabel='World x (m)',ylabel='World z (m)',title=f"Measured rise {r['apex_height_above_release']:.3f} m; success={r['success']}");side.legend()
    fig.tight_layout();path=trial/'trajectory_3d.png';fig.savefig(path,dpi=160);plt.close(fig);return str(path)


def render_trajectory_video(trial_dir,fps=20):
    """Animate measured geometry/COM samples; no ballistic interpolation or invented states."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    trial=Path(trial_dir);r=json.loads((trial/'result.json').read_text());rel=r.get('release')
    if not rel:return None
    rows=np.genfromtxt(trial/'trajectory.csv',delimiter=',',names=True)
    end=r['crossing']['time'] if r.get('crossing') else min(rows['time'][-1],rel['time']+1.5)
    d=rows[(rows['time']>=rel['time'])&(rows['time']<=end)]
    if len(d)<2:return None
    geom=np.c_[d['geometry_x'],d['geometry_y'],d['geometry_z']];com=np.c_[d['com_x'],d['com_y'],d['com_z']]
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d')
    both=np.vstack([geom,com]);lo=both.min(axis=0)-.06;hi=both.max(axis=0)+.06
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)')
    lg,=ax.plot([],[],[],label='Geometry reference');lc,=ax.plot([],[],[],label='CoM');ax.legend()
    title=ax.set_title('');path=trial/'trajectory_3d.mp4'
    try:
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=7,macro_block_size=16) as writer:
            for t in np.arange(d['time'][0],d['time'][-1]+1/fps,1/fps):
                k=min(np.searchsorted(d['time'],t)+1,len(d))
                for line,points in [(lg,geom),(lc,com)]:line.set_data_3d(points[:k,0],points[:k,1],points[:k,2])
                title.set_text(f"Measured flight t={t-rel['time']:.2f}s | v={r.get('release_speed',0):.2f} m/s\nrise={r.get('apex_height_above_release',0):.3f} m | success={r['success']}")
                fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];writer.append_data(frame)
    finally:plt.close(fig)
    return str(path)
