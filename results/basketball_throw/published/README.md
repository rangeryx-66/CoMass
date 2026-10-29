# Basketball evidence

See the root BASKETBALL_REPORT.md. Status: no_feasible_arc_under_robot_limits. All 80 perturbed contact trials retained the box in the bucket, but all failed grasp/safety gates; do not count them as valid successes.

The published config and runtime files capture the executed main search (600 requested numerical-planner budget, 28 physical parameter proposals per distance, 20 paired perturbations per object). Planner actual counts include local refinement.

```python
import yaml
from piper_throw.basketball import run_benchmark
c = yaml.safe_load(open("configs/basketball_throw.yaml"))
run_benchmark(c, "results/basketball_throw/new_exact_run", planner_budget=600, simulation_budget=28, repeats=20)
```

Then run `python -m piper_throw.basketball_analysis --results results/basketball_throw/new_exact_run --repeats 20 --video`. The config adds a stricter zero-tolerance actual-position check in the final implementation; strict_position_audit.csv also independently checks the recorded trajectories with zero tolerance. No successful candidate was reclassified by this change.

Full per-step/raw files remain local and are not included here. Metadata trial paths identify original local provenance. Run the standalone diagnostic config to reconstruct a fresh full trajectory.

The public range-audit JSON aggregates repeated contact samples by geometry into first/last time and sample count. All numerical metrics and failure flags are unchanged. Original per-step events remain in the local raw records.

Final perturbed results use fixed nominal feedforward gains, with no controller access to the randomized plant gain. Earlier gain-aware diagnostic repeats are retained locally and excluded. The final pipeline performs fixed-action validation only after actual-distance auditing, avoiding the earlier duplicate pre-audit validation.
