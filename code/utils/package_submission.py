"""Assembles the final submission zip: [id][surname][initials].zip containing
the PDF at the zip ROOT and the code/ folder alongside it (per the assignment's
required structure -- PDF must NOT be inside code/).

Run: .venv/Scripts/python.exe -m code.utils.package_submission --id STUDENTID --surname SURNAME --initials XX
(defaults to a clearly-labelled placeholder name if not given, so the pipeline
is testable before the real student details are provided)
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# Directories/files that should NOT be included in the submitted code/ folder:
# large regenerable artefacts (dataset cache, venv, checkpoints, raw TensorBoard
# event files) -- keeping the zip a reasonable size and free of machine-specific
# binaries, while every script that PRODUCES those artefacts is still included.
_EXCLUDE_DIR_NAMES = {".venv", "__pycache__", "data", "checkpoints", ".git"}
_EXCLUDE_SUFFIXES = {".pyc"}
_EXCLUDE_PATTERNS_IN_PATH = {"version_", ".tfevents"}


def _should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in _EXCLUDE_DIR_NAMES for p in parts):
        return True
    if path.suffix in _EXCLUDE_SUFFIXES:
        return True
    if any(pat in str(path) for pat in _EXCLUDE_PATTERNS_IN_PATH):
        return True
    return False


def build_zip(student_id: str, surname: str, initials: str) -> Path:
    name = f"{student_id}{surname}{initials}"
    zip_path = _ROOT / f"{name}.zip"

    pdf_path = _ROOT / "report" / "report.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} does not exist -- build it first with "
            f"'python -m code.utils.build_report_pdf'"
        )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # PDF at the ZIP ROOT (not inside code/), named per the required convention.
        zf.write(pdf_path, arcname=f"{name}.pdf")

        # Everything under code/, plus top-level reproducibility artefacts
        # (environment.json, requirements.txt, PROGRESS_LOG.md, OUTSTANDING.md,
        # figures/, results/ minus checkpoints, report/report.md) under code/.
        include_roots = ["code", "figures", "results", "report",
                          "environment.json", "requirements.txt",
                          "PROGRESS_LOG.md", "OUTSTANDING.md"]
        for root_name in include_roots:
            src = _ROOT / root_name
            if not src.exists():
                continue
            if src.is_file():
                zf.write(src, arcname=f"code/{root_name}")
                continue
            for path in src.rglob("*"):
                if path.is_dir() or _should_skip(path.relative_to(_ROOT)):
                    continue
                if path == pdf_path:
                    continue  # already added at root, don't duplicate inside code/
                arcname = f"code/{path.relative_to(_ROOT).as_posix()}"
                zf.write(path, arcname=arcname)

    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="STUDENTID", dest="student_id")
    parser.add_argument("--surname", default="SURNAME")
    parser.add_argument("--initials", default="XX")
    args = parser.parse_args()

    zip_path = build_zip(args.student_id, args.surname, args.initials)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pdf_at_root = [n for n in names if n.endswith(".pdf") and "/" not in n]
        print(f"PDF(s) at zip root: {pdf_at_root}")
        print(f"Total entries: {len(names)}")


if __name__ == "__main__":
    main()
