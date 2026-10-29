from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path, overrides=()):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        parent = config
        parts = key.split(".")
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
        parent[parts[-1]] = yaml.safe_load(value)
    # Mesh paths are relative to the config, not the launch directory.
    for spec in [config["env"]["object"], *config["env"].get("randomization", {}).get("objects", [])]:
        if spec.get("mesh"):
            spec["mesh"] = str((path.parent / spec["mesh"]).resolve())
    return config


def merge_object(base, override):
    result = deepcopy(base)
    result.update(override)
    return result
