"""Generates ready-to-paste Markdown tables (overall performance, per-station,
training time, top-15 influential nodes) from whatever results currently
exist, written to report/tables/*.md. Idempotent / safe to re-run as more
experiments finish -- mirrors build_all_figures.py's "build what's available,
skip and log what isn't" approach.

Run: .venv/Scripts/python.exe -m code.experiments.build_tables
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from code.utils.progress_logger import ProgressLogger

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results"
_TABLES_DIR = _ROOT / "report" / "tables"
_TABLES_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "tts": "TTS",
    "gwn_predefined": "GWN (predefined)",
    "gwn_adaptive": "GWN (predefined+adaptive)",
    "agcrn": "AGCRN",
}
HORIZONS = ["15min", "30min", "60min"]
HORIZON_LABELS = {"15min": "15 min", "30min": "30 min", "60min": "60 min"}


def _df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def build_overall_table() -> str | None:
    rows = []
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "metrics.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for h in HORIZONS:
            m = data["overall"][h]
            rows.append({
                "Model": display, "Horizon": HORIZON_LABELS[h],
                "MSE (mph²)": round(m["mse"], 3), "MAE (mph)": round(m["mae"], 3),
                "MAPE (%)": round(m["mape"] * 100, 2),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return _df_to_md(df)


def build_per_station_table(node_indices=(0, 1, 2)) -> str | None:
    rows = []
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "metrics_per_node.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for node_id in node_indices:
            row = df[df["node_id"] == node_id]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            for h in HORIZONS:
                rows.append({
                    "Model": display, "Sensor": f"Sensor {node_id + 1}",
                    "Horizon": HORIZON_LABELS[h],
                    "MSE (mph²)": round(row[f"mse_{h}"], 3),
                    "MAE (mph)": round(row[f"mae_{h}"], 3),
                    "MAPE (%)": round(row[f"mape_{h}"] * 100, 2),
                })
    if not rows:
        return None
    return _df_to_md(pd.DataFrame(rows))


def build_training_time_table() -> str | None:
    rows = []
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "training_summary.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        total_min = round(s["total_training_seconds"] / 60, 1) if s.get("total_training_seconds") else "N/A*"
        avg_s = round(s["seconds_per_epoch_avg"], 1) if s.get("seconds_per_epoch_avg") else "N/A*"
        rows.append({
            "Model": display,
            "Epochs run": s["epochs_run"],
            "Early stopped": s.get("early_stopped", "N/A"),
            "Total time (min)": total_min,
            "Avg s/epoch": avg_s,
            "Best val MAE": round(s["best_val_mae"], 4) if s.get("best_val_mae") else None,
        })
    if not rows:
        return None
    return _df_to_md(pd.DataFrame(rows))


def build_top15_table() -> str | None:
    p = _RESULTS / "gwn_adaptive" / "top15_influential_nodes.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.rename(columns={
        "rank": "Rank", "node_id": "Source Node", "influence": "Influence Score",
        "influence_normalised": "Influence (normalised)",
        "most_influenced_nodes": "Most Influenced Node(s)",
    })
    df = df[["Rank", "Source Node", "Influence Score", "Influence (normalised)", "Most Influenced Node(s)"]]
    df["Influence Score"] = df["Influence Score"].round(3)
    df["Influence (normalised)"] = df["Influence (normalised)"].round(3)
    return _df_to_md(df)


def main():
    log = ProgressLogger("build_tables")
    builders = {
        "overall_performance.md": build_overall_table,
        "per_station.md": build_per_station_table,
        "training_time.md": build_training_time_table,
        "top15_influential_nodes.md": build_top15_table,
    }
    for filename, builder in builders.items():
        md = builder()
        if md is None:
            log.milestone(f"SKIPPED {filename} -- dependency not ready yet")
            continue
        (_TABLES_DIR / filename).write_text(md, encoding="utf-8")
        log.milestone(f"Wrote report/tables/{filename}")


if __name__ == "__main__":
    main()
