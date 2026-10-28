#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; configure your network proxy if required.
python -m venv .venv-piper
.venv-piper/bin/pip install -r piper_throw/requirements.lock
mkdir -p piper_throw/upstream
fetch_git() {
  local repo="$1" destination="$2" revision="$3"
  if [ -d "$destination" ] && [ ! -d "$destination/.git" ]; then
    test -f "$destination/.upstream-commit" && test "$(cat "$destination/.upstream-commit")" = "$revision"
    return
  fi
  if [ ! -d "$destination/.git" ]; then
    git -c filter.lfs.required=false -c filter.lfs.smudge= -c filter.lfs.process= clone --no-checkout "$repo" "$destination"
  fi
  git -C "$destination" -c filter.lfs.required=false -c filter.lfs.smudge= -c filter.lfs.process= checkout "$revision"
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}
fetch_git https://github.com/Manas-arumalla/openarm-control piper_throw/upstream/openarm-control 14ea59faad6706b9add5d7f8d386e43239a6360f
fetch_git https://github.com/renesas-rdk/agilex_piper_mujoco piper_throw/upstream/agilex_piper_mujoco cf0d35c06e19002dc8fea059118b2dceea5a50e9
fetch_git https://github.com/agilexrobotics/pyAgxArm piper_throw/upstream/pyAgxArm d90e234ba8e408cfa964b38aa9120b3131fc9f85
.venv-piper/bin/python piper_throw/tools/fetch_assets.py
.venv-piper/bin/python -m piper_throw.cli smoke
