# PiPER basketball experiment limits

Simulation only. No SDK import, CAN device, network robot connection or hardware motion is used.

Official primary source: [AgileX piper_sdk CHANGELOG](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/CHANGELOG.MD), inspected on 2026-10-28. Its parameter-limit table specifies **0–3 rad/s for each of six joints**; firmware-specific encoding is 0.001 rad/s before V1.5-8 and 0.01 rad/s thereafter. These are standard SDK software caps, not a claim that every PiPER hardware variant has identical physical maximum speed. Use the stricter 3 rad/s cap on both commands and measured trajectories, with zero speed tolerance in this experiment. Do not inherit the old experimental 4/5 rad/s budgets.

SDK position ranges are intersected with the pinned MuJoCo model, never expanded:

| Joint | Model/software intersection (rad) |
|---|---|
| J1 | [-2.6179, 2.6179] |
| J2 | [0, 3.14] |
| J3 | [-2.697, 0] (the model is narrower than SDK -2.967) |
| J4 | [-1.745, 1.745] |
| J5 | [-1.22, 1.22] |
| J6 | [-2.09439, 2.09439] |

The same changelog's v0.1.16 entry describes an acceleration-setting range up to 5 rad/s². We use **5 rad/s²** as a conservative, **uncalibrated experiment budget**, not a measured physical acceleration limit or guaranteed actuator response. Actual finite-difference velocity acceleration is checked at the integration timestep, including preparation, opening, unloading and braking. It is not silently exempted during transients.

Additional **uncalibrated simulation assumptions**:

- Position-actuator gains inherited from this project's simulation model: kp [250,250,250,80,80,80], kv [18,18,18,7,7,7]. Bounded bias/desired-velocity feedforward using fixed nominal gains (no access to randomized plant gains); optional MuJoCo mass-matrix acceleration feedforward and nominal 100 g point-payload feedforward. Hidden ballast positions and true CoM are not read by this controller.
- Torque caps [12,12,12,3,3,3] N·m are conservative test assumptions, **not verified official continuous/peak ratings**. The old inherited ±100 N·m limit is not used. Passive perfect `gravcomp` is disabled; compensation goes through the same capped actuators.
- Gripper kp 1000, kv 10, force cap 20 N; original two finger pads per jaw and collision geometry. Friction coefficient 0.8. Finite jaw opening, zero weld in contact mode.
- Position tracking-error limit 0.15 rad; final brake-speed limit 0.1 rad/s; no artificial reset after release. Friction/gain/delay perturbations retain the original 5% / 2.5% / 4 ms assumptions, pending hardware identification.
- Tabletop 0.75 m, 1.10 × 0.90 m with four solid legs. Base/table mounting interface excluded as a fixed fixture; moving links and the complete box retain table, floor and bucket collision.

Therefore `no_feasible_arc_under_robot_limits` means **no valid candidate found for this model, these assumptions, controller/action families and finite search budget**. It is not a certified global impossibility result for the physical PiPER. Failure of actual acceleration/tracking or grasp retention must not be rewritten as a pure joint-speed limitation.
