"""Keep only full-head demotions validated by real-activation probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_validated(step: int, layer: int) -> bool:
    if layer == 28:
        return True
    if layer == 27:
        return step >= 25
    if layer == 26:
        return step >= 20
    if layer == 29:
        return step < 30
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--energy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    energy = json.loads(args.energy.read_text(encoding="utf-8"))
    retained = []
    for entry in energy["energy_aware_full_demotion"]["entries"]:
        step, layer, head = (int(entry[key]) for key in ("step", "layer", "head"))
        if not is_validated(step, layer):
            continue
        if base["masks"][step][layer][head] != "full":
            raise ValueError(f"expected full base head at {step=}, {layer=}, {head=}")
        base["masks"][step][layer][head] = [5, 3, 5]
        retained.append(entry)

    base["guarded_energy_demotion"] = {
        "source": str(args.energy),
        "count": len(retained),
        "rules": {
            "layer_28": "all energy-safe entries",
            "layer_27": "step >= 25",
            "layer_26": "step >= 20",
            "layer_29": "step < 30",
        },
        "entries": retained,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: retained {len(retained)} validated demotions")


if __name__ == "__main__":
    main()
