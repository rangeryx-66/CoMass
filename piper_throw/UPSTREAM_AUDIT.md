# Upstream inspection and reuse

Inspection date: 2026-10-28. No hardware SDK is installed or executed.

| Repository | Fixed commit | Inspected implementation | Decision |
|---|---|---|---|
| [openarm-control](https://github.com/Manas-arumalla/openarm-control) | `14ea59faad6706b9add5d7f8d386e43239a6360f` | `openarm_control/throwing.py`, `catching.py`, `tests/test_throwing.py`, `benchmarks/throwing_benchmark.py`, `pyproject.toml` | Run original tests before porting; reuse quintic and the actual-simulation release search pattern. Apache-2.0. |
| [agilex_piper_mujoco](https://github.com/renesas-rdk/agilex_piper_mujoco) | `cf0d35c06e19002dc8fea059118b2dceea5a50e9` | `models/agilex_piper/piper.xml`, `piper_body.xml`, assets, launch/config | Native MJCF exists despite ROS2 package wrapper. Use `piper.xml`, without ROS plugins. Model has its own MIT license. |
| [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) | `d90e234ba8e408cfa964b38aa9120b3131fc9f85` | `pyproject.toml`, factory, Piper driver | Design only. `AgxArmFactory.create_arm(config)`, `connect`, `move_j`, `move_js`, `move_mit`; joint position rad, velocity rad/s. LGPL-3.0. |

OpenArm requires Python >=3.10 and explicitly pins **MuJoCo 3.8.1**. RL and neural-network extras are unnecessary and will not be installed. PiPER's ROS2 launch layer is unnecessary for native Python; mesh files are LFS pointers until separately downloaded. `tools/fetch_assets.py` checks each STL against its upstream SHA256.

## Interfaces that must change

* `ThrowController(model,data,arm,ball)` hardcodes seven-joint vectors through `GraspSolver/OpenArmKinematics`; replace with six named PiPER joints, MuJoCo Jacobians and SciPy IK where needed. Never reuse OpenArm QD_MAX as PiPER limits.
* `plan_release/_search_release/_true_landing`: retain simulation-in-loop semantics, but score a capsule's descending rim-plane crossing, collision history and full-body settled containment. Upstream sphere final-xy distance is insufficient for a 14 cm rod.
* `catching.quintic`: reuse its dimension-independent polynomial. Upstream `_move_to/_swing_q` hardcode `zeros(7)` and require six-dimensional replacement.
* `_save/_restore`: upstream saves qpos/qvel/eq_active/forces/ctrl/time only. New trials use new MjData and reset full integration state, activation, warmstart and controller flags; do not carry stale contact solver state.
* Weld attach: retain physical equality constraints for the idealized held phase; release only deactivates the constraint and preserves simulated velocities. Do not inject target-dependent object velocities. Contact release needs separate validation.
* Geometry adapter uses original STL hulls for display and collision, original axes/transforms/inertias/limits and finger coupling. Decoration OBJ files are omitted. Adjacent overlapping hulls require explicit exclusions. Controller gains and simulation budgets are configuration assumptions, not validated hardware capabilities.
* Original `piper_body.xml` uses damping 2000 and finger damping 1000 for its ROS demo; use the standalone `piper.xml` instead, with documented simulation actuator tuning. Neither is a calibrated real robot twin.

## Missing upstream functionality

No configurable rod mass distribution, circular collision bucket, reset-complete experiment runner, full containment test, CoM scan, perturbed release statistics, twin cross-validation or safe disabled hardware trajectory contract is supplied by the original throwing module. These require new implementation and tests.

Future hardware adapter accepts time_s, six joint positions_rad, optional velocities_rad_s and physical gripper opening_m. Firmware mode, timing, safety stop, measured delays, torque limits and unit conversion must be validated before enabling. SDK MIT field encoding ranges must never be treated as actual safe robot speeds. Current RealPiper constructor always raises and performs no imports/connections.
