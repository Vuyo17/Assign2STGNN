"""Machine-readable pipeline status, used to auto-generate OUTSTANDING.md.

``results/status.json`` holds one record per named stage:

    {
      "stages": {
        "<stage_name>": {
          "state": "pending" | "running" | "done" | "failed",
          "started": "<iso timestamp>" | null,
          "finished": "<iso timestamp>" | null,
          "notes": "free text, e.g. best_val_mae=2.9, 41 epochs, 38m12s"
        },
        ...
      }
    }

Scripts call ``set_stage(...)`` at the start and end of each unit of work. The
file is the single source of truth that ``update_outstanding.py`` reads (together
with what actually exists on disk) to regenerate the human-readable checklist.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATUS_PATH = _PROJECT_ROOT / "results" / "status.json"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_status() -> dict:
    if _STATUS_PATH.exists():
        with open(_STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"stages": {}}


def save_status(status: dict) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def set_stage(name: str, state: str, notes: Optional[str] = None) -> None:
    """Update (or create) a stage record and persist it immediately.

    ``state`` should be one of "pending", "running", "done", "failed".
    Call with state="running" when a stage starts and state="done"/"failed"
    when it ends; ``notes`` on the terminal call is what shows up next to the
    checklist item in OUTSTANDING.md.
    """
    status = load_status()
    stages = status.setdefault("stages", {})
    record = stages.setdefault(
        name, {"state": "pending", "started": None, "finished": None, "notes": None}
    )
    record["state"] = state
    if state == "running" and record["started"] is None:
        record["started"] = _timestamp()
    if state in ("done", "failed"):
        record["finished"] = _timestamp()
    if notes is not None:
        record["notes"] = notes
    save_status(status)


def get_stage(name: str) -> dict:
    return load_status().get("stages", {}).get(
        name, {"state": "pending", "started": None, "finished": None, "notes": None}
    )
