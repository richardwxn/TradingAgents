"""Re-render existing analysis-only JSON reports as Markdown.

Usage:
    python render_report.py path/to/report.json [more.json ...]
    python render_report.py --glob 'reports/analysis_mvp/*.json'
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from tradingagents.analysis_only import render_markdown_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render analysis-only JSON report(s) as Markdown."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSON report paths to render",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Glob pattern to expand (e.g. 'reports/analysis_mvp/*.json')",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Defaults to writing each .md next to "
            "its source .json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[str] = list(args.paths or [])
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    if not paths:
        raise SystemExit("No input paths provided (pass paths or --glob).")
    out_dir = Path(args.output_dir) if args.output_dir else None
    for raw in paths:
        src = Path(raw)
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        target = (out_dir / (src.stem + ".md")) if out_dir else None
        written = render_markdown_file(src, output_path=target)
        print(f"Rendered: {written}")


if __name__ == "__main__":
    main()
