from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocir_reproduction.protocols import MAIN_PROTOCOL_V1, validate_main_protocol_v1


def main() -> int:
    errors = validate_main_protocol_v1(MAIN_PROTOCOL_V1)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    protocol_path = ROOT / "ocir_reproduction" / "protocols" / "main_protocol_v1.yaml"
    if not protocol_path.exists():
        print(f"FAIL: missing {protocol_path}")
        return 1

    print("main_protocol_v1 OK")
    print(f"version: {MAIN_PROTOCOL_V1['version']}")
    print(f"datasets: {', '.join(MAIN_PROTOCOL_V1['datasets'])}")
    print(f"window_length: {MAIN_PROTOCOL_V1['windowing']['window_length']}")
    print(f"rul_cap: {MAIN_PROTOCOL_V1['labels']['rul_cap']}")
    print(f"split_seed: {MAIN_PROTOCOL_V1['splits']['split_seed']}")
    print(f"model_seeds: {MAIN_PROTOCOL_V1['training']['model_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

