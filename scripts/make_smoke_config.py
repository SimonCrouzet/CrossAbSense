#!/usr/bin/env python3
"""Generate a smoke-test config from oracle_efficient_config with max_epochs=2."""
import yaml
from pathlib import Path

root = Path(__file__).parent.parent
src = root / "src" / "config" / "oracle_efficient_config.yaml"
dst = root / "src" / "config" / "smoke_config.yaml"

with open(src) as f:
    config = yaml.safe_load(f)

config["training"]["finetune"]["max_epochs"] = 2
config["training"]["finetune"]["early_stopping_patience"] = 999

for prop_cfg in config.get("property_specific", {}).values():
    ft = prop_cfg.get("training", {}).get("finetune", {})
    if "max_epochs" in ft:
        ft["max_epochs"] = 2
    if "early_stopping_patience" in ft:
        ft["early_stopping_patience"] = 999

with open(dst, "w") as f:
    yaml.dump(config, f)

print(f"Smoke config written to {dst}")
