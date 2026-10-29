from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "upstream/agilex_piper_mujoco/models/agilex_piper"


def vec(x):
    return " ".join(map(str, np.asarray(x).ravel()))


def mass_properties(config):
    """Uniform cuboid shell + movable spherical ballast, parallel-axis theorem.

    Collision envelope is a 140 x 50 x 50 mm rounded capsule. The shell is
    approximated by a uniform capsule; tungsten-like ballast radius is 10.5 mm.
    Ballast must fit wholly inside the capsule, not just its bounding box.
    """
    if "inertia" in config:
        c=np.asarray(config.get("com",[0,0,0]),dtype=float)
        I=np.asarray(config["inertia"],dtype=float)
        if float(config.get("mass",.1))<=0 or c.shape!=(3,) or not np.isfinite(c).all() or I.shape!=(3,3) or not np.isfinite(I).all() or not np.allclose(I,I.T):
            raise ValueError("invalid measured mass/CoM/inertia")
        eigen=np.linalg.eigvalsh(I)
        if eigen[0]<=0 or eigen[-1]>sum(eigen[:2])+1e-12:
            raise ValueError("inertia violates positive definiteness or triangle inequality")
        return c,I
    if config.get("shape") == "box":
        return box_mass_properties(config)
    m = float(config.get("mass", .1))
    c = np.asarray(config.get("com", [0, 0, 0]), dtype=float)
    if m <= 0 or c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("invalid mass/CoM")
    frac = .85
    p = c / frac
    nearest = p.copy(); nearest[0] -= np.clip(p[0], -.045, .045)
    r = .0105
    if np.linalg.norm(nearest) + r > .025:
        raise ValueError("ballast sphere cannot fit inside capsule at requested CoM")
    ms, mb = m * (1-frac), m * frac
    R, h = .025, .045
    vc, vs = np.pi*R*R*2*h, 4/3*np.pi*R**3
    mc, mh = ms*vc/(vc+vs), ms*vs/(vc+vs)
    axial = .5*mc*R**2 + .4*mh*R**2
    transverse = mc*(3*R**2+4*h**2)/12 + mh*(.4*R**2+h*h+.75*h*R)
    I = np.diag([axial, transverse, transverse])
    I += np.eye(3) * (2/5 * mb * r*r)
    for mass, pos in [(ms, np.zeros(3)), (mb, p)]:
        d = pos - c
        I += mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    if config.get("inertia_mode", "physical") == "fixed_diagnostic":
        I = mass_properties({"mass": m, "com": [0, 0, 0]})[1]
    return c, I


def box_mass_properties(config):
    """120x80x50mm hollow shell + 18mm cubic insert. Same collision geometry.

    Six non-overlapping plates, 2mm thick; 25% shell mass, 75% insert mass.
    At 100g the insert density is 12,860 kg/m³ (physically feasible alloy).
    Measured CoM determines insert position in this documented mass family.
    """
    m = float(config.get("mass", .1))
    c = np.asarray(config.get("com", [0,0,0]),dtype=float)
    dims = np.array([.12,.08,.05]); t=.002; bsize=np.full(3,.018)
    p = c/.75
    if m<=0 or c.shape!=(3,) or not np.isfinite(c).all() or np.any(np.abs(p)+bsize/2 > dims/2-t):
        raise ValueError("internal cubic ballast does not fit inside hollow box")
    parts=[]
    for axis in range(3):
        d=dims.copy(); d[:axis]-=2*t; d[axis]=t
        for sign in [-1,1]:
            pos=np.zeros(3);pos[axis]=sign*(dims[axis]-t)/2
            parts.append([float(np.prod(d)),pos,d])
    volume=sum(x[0] for x in parts)
    parts=[(.25*m*v/volume,p0,d) for v,p0,d in parts]+[(.75*m,p,bsize)]
    I=np.zeros((3,3))
    for mass,p0,d in parts:
        off=p0-c
        I+=np.diag(mass/12*(np.sum(d*d)-d*d))+mass*(np.dot(off,off)*np.eye(3)-np.outer(off,off))
    if config.get("inertia_mode") == "fixed_diagnostic":
        I=box_mass_properties({"mass":m,"com":[0,0,0]})[1]
    return c,I


def build_model(object_config, scene_config):
    root = ET.parse(MODEL_DIR / "piper.xml").getroot()
    root.find("compiler").set("meshdir", str(MODEL_DIR / "assets"))
    root.find("option").set("timestep", str(scene_config.get("timestep", .002)))
    root.find("option").set("gravity", "0 0 -9.81")
    root.find("option").set("integrator", scene_config.get("integrator", "RK4" if scene_config.get("release_mode","ideal")!="contact" else "implicitfast"))
    root.remove(root.find("keyframe"))
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("file").endswith(".obj") or mesh.get("name") == "link2_gray":
            asset.remove(mesh)
    # Use each original collision STL also as the visible surface. Original
    # transforms, inertial properties, joint axes, limits, finger pads remain.
    for body in root.findall(".//worldbody//body"):
        if scene_config.get("bounded_bias_compensation",False):body.set("gravcomp","0")
        for geom in list(body.findall("geom")):
            if geom.get("class") == "visual":
                body.remove(geom)
            elif geom.get("mesh"):
                geom.set("type", "mesh")
                geom.set("rgba", ".65 .68 .72 1")
        for j in body.findall("joint"):
            if j.get("class") != "finger":
                j.set("damping", ".4")
    for i,limits in enumerate(scene_config.get("position_limits",[]),1):
        joint=root.find(f".//joint[@name='joint{i}']")
        old=np.fromstring(joint.get("range"),sep=" ")
        joint.set("range",vec([max(old[0],limits[0]),min(old[1],limits[1])]))
    root.find("worldbody/body").set("pos", vec(scene_config.get("base_position", [0, 0, .35])))
    root.find("worldbody/body").set("quat", vec(scene_config.get("base_quaternion", [1,0,0,0])))
    for i, a in enumerate(root.find("actuator")):
        if i < 6:
            a.set("kp", "250" if i < 3 else "80")
            a.set("kv", "18" if i < 3 else "7")
            if "torque_limits" in scene_config:
                limit=scene_config["torque_limits"][i]
                a.set("forcerange",f"{-limit} {limit}")
        else:
            a.set("kp", str(scene_config.get("gripper_kp", 1000)))
            a.set("kv", "10")
            a.set("forcerange", "-20 20")
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="4 4 .1", rgba=".22 .27 .3 1")
    if scene_config.get("table"):
        cfg=scene_config["table"];h=cfg.get("height",.75);cx=cfg.get("center_x",-.30)
        half=np.asarray(cfg.get("half_size",[.55,.45,.025]))
        table=ET.SubElement(wb,"body",name="table")
        ET.SubElement(table,"geom",name="table_top",type="box",size=vec(half),pos=vec([cx,0,h-half[2]]),rgba=".43 .29 .17 1")
        for i,(x,y) in enumerate([(cx+sx*(half[0]-.06),sy*(half[1]-.06)) for sx in [-1,1] for sy in [-1,1]]):
            ET.SubElement(table,"geom",name=f"table_leg_{i}",type="box",size=vec([.035,.035,(h-2*half[2])/2]),pos=vec([x,y,(h-2*half[2])/2]),rgba=".3 .3 .3 1")
        ET.SubElement(root.find("contact"),"exclude",body1="base_link",body2="table") # fixed mounting interface
    else:
        ET.SubElement(wb, "geom", name="pedestal", type="cylinder", size=".085 .175", pos="0 0 .175", rgba=".3 .3 .3 1")
    if scene_config.get("include_bucket", True):
        bucket = scene_config.get("bucket_position", [.8, 0, .0])
        radius = scene_config.get("bucket_radius", .14)
        height = scene_config.get("bucket_height", .22)
        b = ET.SubElement(wb, "body", name="bucket", pos=vec(bucket))
        ET.SubElement(b, "geom", name="bucket_bottom", type="cylinder", size=f"{radius+.012} .01", pos="0 0 .01", rgba=".2 .5 .65 1")
        for i, th in enumerate(np.linspace(0, 2*np.pi, 48, endpoint=False)):
            ET.SubElement(b, "geom", name=f"bucket_wall_{i}", type="box",
                          pos=vec([(radius+.008)*np.cos(th), (radius+.008)*np.sin(th), height/2]),
                          size=vec([.01, (radius+.008)*np.tan(np.pi/48)*1.05, height/2]),
                          euler=f"0 0 {th}", rgba=".2 .55 .7 .65")
    obj = ET.SubElement(wb, "body", name="object", pos="0 0 1")
    ET.SubElement(obj, "freejoint", name="object_free")
    c, I = mass_properties(object_config)
    ET.SubElement(obj, "inertial", pos=vec(c), mass=str(object_config.get("mass", .1)),
                  fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
    attrs = dict(type="box",size=".06 .04 .025") if object_config.get("shape")=="box" else dict(type="capsule",fromto="-.045 0 0 .045 0 0",size=".025")
    ET.SubElement(obj, "geom", name="object_geom", **attrs,
                  friction=".8 .01 .001", rgba="1 .45 .08 1")
    eq = root.find("equality")
    ET.SubElement(eq, "weld", name="grasp_weld", body1="link6", body2="object", active="false", solref=".004 1")
    # Adjacent STL hulls overlap by construction; exclusions are explicit.
    contact = root.find("contact")
    for i in range(1,6):
        ET.SubElement(contact, "exclude", body1=f"link{i}", body2=f"link{i+1}")
    for name in ["link7", "link8"]:
        ET.SubElement(contact, "exclude", body1="link6", body2=name)
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml
