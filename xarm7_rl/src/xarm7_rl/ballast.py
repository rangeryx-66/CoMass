"""Exact rigid-body parameters of a hollow box with a fixed cuboid ballast."""
import numpy as np


def unit_box_inertia(half_sizes):
    x, y, z = np.asarray(half_sizes, dtype=float)
    return np.diag([y*y+z*z, x*x+z*z, x*x+y*y]) / 3


def parallel_axis(position):
    p = np.asarray(position, dtype=float)
    return (p @ p) * np.eye(3) - np.outer(p, p)


def ballasted_box(half_sizes, wall, shell_mass, ballast_half_sizes, ballast_mass, ballast_position):
    half = np.asarray(half_sizes, dtype=float)
    inner = half-wall
    block = np.asarray(ballast_half_sizes, dtype=float)
    position = np.asarray(ballast_position, dtype=float)
    if wall <= 0 or np.any(inner <= 0) or min(shell_mass, ballast_mass) <= 0:
        raise ValueError("Box wall and component masses must be positive")
    if np.any(np.abs(position)+block > inner+1e-12):
        raise ValueError("The fixed ballast must fit inside the hollow box")
    outer_volume, inner_volume = np.prod(2*half), np.prod(2*inner)
    density = shell_mass/(outer_volume-inner_volume)
    shell_inertia = density*(outer_volume*unit_box_inertia(half)-inner_volume*unit_box_inertia(inner))
    total_mass = shell_mass+ballast_mass
    com = ballast_mass/total_mass*position
    inertia = (shell_inertia + ballast_mass*unit_box_inertia(block)
               + shell_mass*parallel_axis(com) + ballast_mass*parallel_axis(position-com))
    return {"type":"box", "size":half.tolist(), "mass":float(total_mass), "com":com.tolist(),
            "inertia":[float(inertia[i,j]) for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],
            "ballast":{"wall_thickness":float(wall),"shell_mass":float(shell_mass),
                       "mass":float(ballast_mass),"half_sizes":block.tolist(),"position":position.tolist()}}
