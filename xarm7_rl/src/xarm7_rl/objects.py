"""Geometry and physically consistent inertial parameters in object coordinates."""
import numpy as np
import trimesh

SHAPES = ("box", "sphere", "cylinder", "capsule", "ellipsoid", "mesh")


def object_geometry(spec, point_count):
    shape = spec["type"]
    size = np.asarray(spec.get("size", [1, 1, 1]), dtype=float)
    if shape == "box":
        mesh = trimesh.creation.box(extents=2 * size)
    elif shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif shape == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale(size)
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    elif shape == "capsule":
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[16, 16])
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    elif shape == "mesh":
        mesh = trimesh.load(spec["mesh"], force="mesh", process=True)
        mesh.apply_scale(spec.get("mesh_scale", [1, 1, 1]))
    else:
        raise ValueError(f"Unknown object type: {shape}; expected {SHAPES}")
    # MuJoCo's mesh collision is convex; use the same convex body for mass properties.
    hull = mesh.convex_hull
    center = np.asarray(hull.center_mass)
    com = np.asarray(spec.get("com", center), dtype=float)
    mass = float(spec["mass"])
    if mass <= 0 or not np.all(np.isfinite(com)):
        raise ValueError("Object mass must be positive and CoM finite")
    if spec.get("inertia") is not None:
        xx, yy, zz, xy, xz, yz = spec["inertia"]
        inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    else:
        unit_inertia = np.asarray(hull.moment_inertia) / hull.mass
        delta = com - center
        # Uniform convex body plus an internal point ballast, with the requested CoM.
        # Each face satisfies n.x + b <= 0. The ballast is on the center-to-CoM ray.
        from scipy.spatial import ConvexHull
        planes = ConvexHull(hull.vertices).equations
        margin = -(planes[:, :3] @ center + planes[:, 3])
        ratio = float(np.max((planes[:, :3] @ delta) / margin))
        if ratio >= 1:
            raise ValueError("CoM must be inside the object's convex collision geometry")
        alpha = np.sqrt(max(0.0, ratio))
        inertia = mass * (1 - alpha) * unit_inertia
        if alpha > 0:
            inertia += mass * (1 - alpha) / alpha * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    eig = np.linalg.eigvalsh(inertia)
    if eig[0] <= 0 or eig[-1] > eig[:2].sum() + 1e-12:
        raise ValueError("Inertia must be positive definite and satisfy the triangle inequality")
    # Fixed surface samples retain geometry beyond a primitive ID or bounding box.
    rng = np.random.default_rng(0)
    triangles = mesh.triangles[rng.choice(len(mesh.faces), point_count, p=mesh.area_faces / mesh.area)]
    uv = rng.random((point_count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    return mesh, mass, com, inertia, points.astype(np.float32)
