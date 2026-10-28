# Third-party code and assets

* `quintic.py` extracts the dimension-independent `quintic` function without algorithm changes from `Manas-arumalla/openarm-control`, commit `14ea59faad6706b9add5d7f8d386e43239a6360f`. Apache-2.0; full license in `LICENSE_OPENARM.txt`. The upstream repository is retained under `upstream/openarm-control` after bootstrap.
* PiPER MJCF transforms, joint axes, limits, inertias, finger geometry and collision STL assets are loaded from `renesas-rdk/agilex_piper_mujoco`, commit `cf0d35c06e19002dc8fea059118b2dceea5a50e9`. Its model directory identifies the MIT license, copyright 2024 RosenYin; full model license is retained in `LICENSE_PIPER_MODEL.txt`. Runtime modifications are documented in `UPSTREAM_AUDIT.md` and `scene/build.py`.
* `agilexrobotics/pyAgxArm`, commit `d90e234ba8e408cfa964b38aa9120b3131fc9f85`, is inspected as an LGPL-3.0 SDK reference. The simulator does not import, install or execute it. Its original license remains with the downloaded source.
* MuJoCo, SciPy, NumPy and plotting/video dependencies retain their own upstream licenses. Exact installed versions are recorded in `requirements.lock`.
