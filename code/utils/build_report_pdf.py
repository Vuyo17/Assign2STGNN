"""Renders report/report.md -> report/report.pdf via a pure-Python toolchain
(no external binaries -- no pandoc/LaTeX/wkhtmltopdf available on this
machine): `markdown` (MD -> HTML) + `xhtml2pdf` (HTML/CSS -> PDF, reportlab
under the hood).

Run: .venv/Scripts/python.exe -m code.utils.build_report_pdf
"""
from __future__ import annotations

from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa

_ROOT = Path(__file__).resolve().parents[2]
_MD_PATH = _ROOT / "report" / "report.md"
_PDF_PATH = _ROOT / "report" / "report.pdf"

_CSS = """
<style>
  @page { size: A4; margin: 2.2cm 2cm; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #0b0b0b; }
  h1 { font-size: 18pt; margin-top: 0; }
  h2 { font-size: 14pt; margin-top: 20px; border-bottom: 1px solid #c3c2b7; padding-bottom: 3px; }
  h3 { font-size: 12pt; margin-top: 14px; }
  h4 { font-size: 11pt; margin-top: 10px; }
  p, li { text-align: justify; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
  th, td { border: 1px solid #c3c2b7; padding: 4px 6px; text-align: left; }
  th { background-color: #f0efec; font-weight: bold; }
  img { max-width: 100%; margin: 10px 0; }
  code { background-color: #f0efec; padding: 1px 3px; font-family: Courier, monospace; font-size: 9pt; }
  pre { background-color: #f0efec; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; }
  blockquote { border-left: 3px solid #c3c2b7; margin-left: 0; padding-left: 10px; color: #52514e; }
  .caption { font-size: 9pt; color: #52514e; font-style: italic; }
</style>
"""


def build_pdf(md_path: Path = _MD_PATH, pdf_path: Path = _PDF_PATH) -> Path:
    md_text = md_path.read_text(encoding="utf-8")
    # Image paths in the markdown are relative to the report/ dir's parent
    # (project root); xhtml2pdf resolves relative src paths against the CWD it
    # is invoked from, so we rewrite them to absolute file:// paths here to be
    # robust regardless of invocation directory.
    html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

    html_full = f"<html><head>{_CSS}</head><body>{html_body}</body></html>"

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html_full, dest=f, path=str(_ROOT) + "/")

    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} error(s) while building the PDF")

    return pdf_path


if __name__ == "__main__":
    out = build_pdf()
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
