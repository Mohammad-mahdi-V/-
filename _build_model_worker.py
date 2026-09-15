
"""
_build_model_worker.py

Invoked by services/model_service.py as a subprocess.
Now runs the Unified Global Learning Cycle (3-day + weekly + monthly)
with independent per-horizon algorithm/params and independent promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_learning_cycle import run_global_learning_cycle  # noqa: E402


def emit_result(payload: dict) -> None:
    print("__RESULT_JSON__:" + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    # Legacy single-algorithm flags still accepted (applied to all horizons
    # unless --horizon-configs-json is provided).
    parser.add_argument("--algorithm", default="ridge",
                        choices=["ridge", "linear", "random_forest"])
    parser.add_argument("--params-json", default="{}")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    # Full per-horizon configuration from the Control Panel.
    parser.add_argument("--horizon-configs-json", default=None)
    args = parser.parse_args()

    try:
        params = json.loads(args.params_json)
    except json.JSONDecodeError as exc:
        emit_result({"ok": False, "error": f"Invalid params JSON: {exc}"})
        return 1

    horizon_configs = {
        "3day": {"algorithm": args.algorithm, "params": dict(params)},
        "weekly": {"algorithm": args.algorithm, "params": dict(params)},
        "monthly": {"algorithm": args.algorithm, "params": dict(params)},
    }
    if args.horizon_configs_json:
        try:
            user = json.loads(args.horizon_configs_json)
            for h, cfg in user.items():
                if h in horizon_configs:
                    horizon_configs[h] = {
                        "algorithm": str(cfg.get("algorithm", args.algorithm)).lower(),
                        "params": dict(cfg.get("params") or cfg.get("model_params") or {}),
                    }
        except json.JSONDecodeError as exc:
            emit_result({"ok": False, "error": f"Invalid horizon-configs JSON: {exc}"})
            return 1

    try:
        result = run_global_learning_cycle(
            promote=args.promote,
            horizon_configs=horizon_configs,
            rebuild_datasets=not args.skip_rebuild,
        )
        any_failed = any(
            m.get("status") == "failed" for m in result.get("models", {}).values()
        )
        emit_result({"ok": not any_failed, "report": result})
        return 1 if any_failed else 0
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        emit_result({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
