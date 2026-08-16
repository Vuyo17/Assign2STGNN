"""Shared progress-logging helper.

Every long-running script (data prep, timing pilots, training, evaluation) should
create one ``ProgressLogger`` and use it instead of bare ``print``. Each call writes:

1. A verbose line to ``logs/<run_name>.log`` (full detail: batch timings, raw
   metrics, tracebacks) -- this is the "raw" log for a single run.
2. A compact milestone line appended to the shared ``PROGRESS_LOG.md`` at the
   project root -- this is the single file a human can tail to see the heartbeat
   of *everything* currently running or previously run, across all experiments.

Only call ``milestone()`` for events worth surfacing project-wide (run started,
epoch checkpoint, run finished, run failed); use ``detail()`` for finer-grained
lines that only belong in the per-run log.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR = _PROJECT_ROOT / "logs"
_PROGRESS_MD = _PROJECT_ROOT / "PROGRESS_LOG.md"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProgressLogger:
    def __init__(self, run_name: str):
        self.run_name = run_name
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = _LOGS_DIR / f"{run_name}.log"
        if not _PROGRESS_MD.exists():
            _PROGRESS_MD.write_text(
                "# Progress Log\n\n"
                "Auto-updated running log of every experiment/script executed for "
                "this assignment. Newest entries at the bottom. Full verbose logs "
                "live under `logs/<run_name>.log`.\n\n",
                encoding="utf-8",
            )

    def _write_raw(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_milestone(self, line: str) -> None:
        with open(_PROGRESS_MD, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def detail(self, message: str) -> None:
        """Fine-grained line: written only to the per-run log file."""
        self._write_raw(f"[{_timestamp()}] {message}")

    def milestone(self, message: str) -> None:
        """Notable event: written to both the per-run log and PROGRESS_LOG.md."""
        line = f"[{_timestamp()}] [{self.run_name}] {message}"
        self._write_raw(line)
        self._write_milestone(f"- {line}")

    def error(self, message: str) -> None:
        line = f"[{_timestamp()}] [{self.run_name}] ERROR: {message}"
        self._write_raw(line)
        self._write_milestone(f"- {line}")
