"""Regenerate OUTSTANDING.md from real state (results/status.json + files on disk).

This is never hand-edited -- run it (``python -m code.utils.update_outstanding``)
after any stage finishes, and every training/evaluation/plotting script calls it
automatically at the end of its ``main()``. It cross-checks the JSON status
records against the actual presence of expected output files, so a stage marked
"done" in status.json but missing its artifact is flagged rather than silently
trusted.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from code.utils.status import load_status

_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = _ROOT / "OUTSTANDING.md"

# (stage_key, section, human description, artifact path relative to project root
#  that should exist once the stage is genuinely complete -- used as a
#  cross-check against status.json, not as the sole source of truth)
STAGES = [
    # --- Setup ---
    ("env_setup", "Setup", "Python venv + torch/tsl/PyG installed, versions recorded",
     "environment.json"),
    ("data_pipeline_verified", "Setup", "METR-LA loaded, splits/scaler verified (no leakage)",
     "code/API_NOTES.md"),
    ("timing_pilot", "Setup", "CPU timing pilot run for all 4 architectures",
     "results/timing_pilot.json"),

    # --- Q1: TTS ---
    ("tts_train", "Q1 TimeThenSpace", "TTS trained to convergence (early stopping)",
     "results/tts/metrics.json"),
    ("tts_eval", "Q1 TimeThenSpace", "TTS evaluated: overall + per-horizon + per-node metrics",
     "results/tts/metrics_per_node.csv"),
    ("fig_adjacency_heatmap", "Q1 TimeThenSpace", "Predefined adjacency matrix heatmap",
     "figures/fig01_adjacency_heatmap.png"),
    ("fig_tts_overall", "Q1 TimeThenSpace", "TTS overall performance table + horizon-trend chart",
     "figures/fig02_tts_overall_performance.png"),
    ("fig_tts_per_station", "Q1 TimeThenSpace", "TTS actual-vs-predicted, sensors 1-3",
     "figures/fig03_tts_per_station.png"),

    # --- Q2: GraphWaveNet ---
    ("gwn_predefined_train", "Q2 GraphWaveNet", "GWN (predefined graph only) trained",
     "results/gwn_predefined/metrics.json"),
    ("gwn_adaptive_train", "Q2 GraphWaveNet", "GWN (predefined + adaptive) trained",
     "results/gwn_adaptive/metrics.json"),
    ("gwn_eval", "Q2 GraphWaveNet", "Both GWN configs evaluated: overall + per-horizon + per-node",
     "results/gwn_adaptive/metrics_per_node.csv"),
    ("fig_gwn_vs_tts", "Q2 GraphWaveNet", "TTS vs GWN-predefined vs GWN-adaptive comparison",
     "figures/fig04_gwn_tts_comparison.png"),
    ("fig_convergence", "Q2 GraphWaveNet", "Training/validation convergence curves (TTS+GWN)",
     "figures/fig05_convergence_curves.png"),
    ("fig_gwn_per_station", "Q2 GraphWaveNet", "Per-station comparison, nodes 1-3, all 3 models",
     "figures/fig06_per_station_comparison.png"),
    ("fig_learned_adjacency", "Q2 GraphWaveNet", "Learned adaptive adjacency heatmap (first 50 nodes)",
     "figures/fig07_learned_adjacency_heatmap.png"),
    ("top15_influential_nodes", "Q2 GraphWaveNet", "Top-15 influential nodes table (defined influence score)",
     "results/gwn_adaptive/top15_influential_nodes.csv"),
    ("fig_predefined_vs_learned", "Q2 GraphWaveNet", "Predefined vs learned adjacency comparison",
     "figures/fig08_predefined_vs_learned.png"),
    ("gwn_paper_comparison", "Q2 GraphWaveNet", "Written comparison with Wu et al. 2019 GWN paper",
     None),

    # --- Q3: AGCRN ---
    ("agcrn_epoch_selection", "Q3 AGCRN", "Epoch-selection experiment + justification",
     "results/agcrn/epoch_selection.json"),
    ("agcrn_train", "Q3 AGCRN", "AGCRN final training run",
     "results/agcrn/metrics.json"),
    ("agcrn_eval", "Q3 AGCRN", "AGCRN evaluated: overall + per-horizon + per-node",
     "results/agcrn/metrics_per_node.csv"),
    ("fig_agcrn_vs_gwn", "Q3 AGCRN", "AGCRN vs GWN performance/training-time comparison",
     "figures/fig09_agcrn_gwn_comparison.png"),
    ("fig_agcrn_per_station", "Q3 AGCRN", "AGCRN vs best GWN, per-station (nodes 1-3)",
     "figures/fig10_agcrn_per_station.png"),
    ("weather_paper_comparison", "Q3 AGCRN", "Written reflection vs Gaibie et al. 2024 weather paper",
     None),

    # --- Synthesis & Report ---
    ("final_synthesis", "Synthesis", "Cross-model final comparison + overall discussion drafted",
     None),
    ("report_written", "Report", "report/report.md written (all required sections)",
     "report/report.md"),
    ("report_pdf", "Report", "report.md rendered to PDF",
     "report/report.pdf"),
    ("final_zip", "Report", "Final [id][surname][initials].zip assembled (PDF at root)",
     None),
]

_ICONS = {"done": "✅", "running": "\U0001F504", "failed": "❌", "pending": "⬜"}


def regenerate() -> Path:
    status = load_status().get("stages", {})
    lines = [
        "# Outstanding / Status Board",
        "",
        "**Auto-generated** by `code/utils/update_outstanding.py` -- do not hand-edit; "
        "it is overwritten after every pipeline stage. Reflects `results/status.json` "
        "cross-checked against files actually present on disk.",
        "",
        f"Last regenerated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    sections: dict[str, list[tuple]] = {}
    for key, section, desc, artifact in STAGES:
        sections.setdefault(section, []).append((key, desc, artifact))

    done_count = 0
    total_count = len(STAGES)

    for section, items in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        for key, desc, artifact in items:
            rec = status.get(key, {"state": "pending", "notes": None})
            state = rec.get("state", "pending")
            artifact_exists = (_ROOT / artifact).exists() if artifact else None

            # Cross-check: if status says done but artifact missing, downgrade the
            # displayed state so the board never over-claims completion.
            if state == "done" and artifact_exists is False:
                state = "failed"
                note_suffix = " (marked done but expected artifact is missing!)"
            else:
                note_suffix = ""

            icon = _ICONS.get(state, "⬜")
            notes = rec.get("notes")
            note_str = f" -- {notes}" if notes else ""
            lines.append(f"- {icon} **{key}**: {desc}{note_str}{note_suffix}")
            if state == "done":
                done_count += 1
        lines.append("")

    lines.insert(4, f"Progress: **{done_count}/{total_count}** stages complete.")
    lines.insert(5, "")

    _OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return _OUT_PATH


if __name__ == "__main__":
    path = regenerate()
    print(f"Wrote {path}")
