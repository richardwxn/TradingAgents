"""Phase 6 + Phase 7: backfill the LLM critic across the existing corpus.

For each report in `--reports-glob`, build the frozen critic prompt,
call the LLM at temperature=0 for each model in `--model` (1 or more),
validate the response against `CriticOutput`, and persist the result.

  - Phase 6 (single model): pass exactly one `--model`. The full
    block (status, provider, model, prompt_version, prompt_hash,
    output) is stored at the top-level key `llm_critic`.
  - Phase 7 (multi-model): pass 2+ `--model` values; each gets its
    own block stored under `llm_critic_multi.runs[*]`, plus a
    `llm_critic_multi.disagreement` summary (stdev of confidence
    adjustment, mean pairwise Jaccard distance on blindspots, etc.).
    The first-listed model's run is mirrored to top-level `llm_critic`
    so single-model downstream consumers keep working unchanged.

Reports that already have all requested models covered at the current
`CRITIC_PROMPT_VERSION` are skipped unless `--force` is passed. Failed
responses (parse, schema, init, call) are written back too so the
failure mode is debuggable; a future re-run retries them.

Rate-limit with `--pace-seconds` to stay under provider RPM caps.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradingagents.analysis_only.llm_critic import (  # noqa: E402
    CRITIC_PROMPT_VERSION,
    compute_llm_disagreement,
    critic_output_or_none,
    run_critic,
)


def _model_already_covered(
    payload: dict, *, provider: str, model: str,
) -> bool:
    """True if this exact (provider, model, prompt_version) was already run ok."""
    # Multi-model storage (Phase 7) takes precedence.
    multi = payload.get("llm_critic_multi") or {}
    runs = multi.get("runs") or []
    for r in runs:
        if (
            r.get("provider") == provider
            and r.get("model") == model
            and r.get("prompt_version") == CRITIC_PROMPT_VERSION
            and r.get("status") == "ok"
        ):
            return True
    # Single-model legacy fallback (Phase 6).
    block = payload.get("llm_critic") or {}
    return (
        block.get("provider") == provider
        and block.get("model") == model
        and block.get("prompt_version") == CRITIC_PROMPT_VERSION
        and block.get("status") == "ok"
    )


def _needs_run_for_any_model(
    payload: dict, *, provider: str, models: list[str], force: bool,
) -> list[str]:
    """Return the subset of models that still need to be run for this report."""
    if force:
        return list(models)
    return [m for m in models if not _model_already_covered(
        payload, provider=provider, model=m,
    )]


def _store_run(
    payload: dict, *, run_block: dict, primary_model: str,
) -> None:
    """Persist a single run into payload, replacing any prior entry for the
    same (provider, model) tuple. Also mirror the primary-model run to the
    legacy top-level `llm_critic` field for backward compatibility.
    """
    multi = payload.setdefault("llm_critic_multi", {"runs": []})
    runs: list = multi.setdefault("runs", [])
    runs = [
        r for r in runs
        if not (
            r.get("provider") == run_block.get("provider")
            and r.get("model") == run_block.get("model")
        )
    ]
    runs.append(run_block)
    multi["runs"] = runs
    if run_block.get("model") == primary_model:
        payload["llm_critic"] = run_block


def _refresh_disagreement(payload: dict) -> None:
    """Recompute `llm_critic_multi.disagreement` from the validated runs."""
    multi = payload.get("llm_critic_multi") or {}
    runs = multi.get("runs") or []
    validated_outputs = [
        critic_output_or_none(r) for r in runs
    ]
    validated_outputs = [o for o in validated_outputs if o is not None]
    if len(validated_outputs) < 2:
        # Drop the stale block so downstream code doesn't treat a single
        # model's run as a disagreement signal.
        multi.pop("disagreement", None)
        return
    multi["disagreement"] = compute_llm_disagreement(validated_outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    parser.add_argument("--provider", required=True)
    parser.add_argument(
        "--model", required=True, action="append",
        help=(
            "Model id to run. Pass once for Phase 6 (single critic) or "
            "2+ times for Phase 7 (multi-model disagreement). The "
            "first-listed model becomes the 'primary' mirrored to the "
            "legacy top-level `llm_critic` block."
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N reports (0 = unlimited).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a valid block exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling the LLM.")
    args = parser.parse_args()

    models: list[str] = list(dict.fromkeys(args.model))  # dedupe, keep order
    primary_model = models[0]

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2

    todo: list[tuple[str, dict, list[str]]] = []
    skipped = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text())
        except Exception as exc:
            print(f"[skip] cannot read {path}: {exc}")
            continue
        pending = _needs_run_for_any_model(
            payload, provider=args.provider, models=models, force=args.force,
        )
        if not pending:
            skipped += 1
            continue
        todo.append((path, payload, pending))

    total_calls = sum(len(p) for _, _, p in todo)
    print(f"Models requested: {models}  (primary={primary_model})")
    print(f"Total reports: {len(paths)}")
    print(
        f"Already fully covered (status=ok, v={CRITIC_PROMPT_VERSION}): "
        f"{skipped}"
    )
    print(f"Reports needing work: {len(todo)}  "
          f"(total LLM calls planned: {total_calls})")

    if args.limit:
        todo = todo[: args.limit]
        total_calls = sum(len(p) for _, _, p in todo)
        print(f"Limited to first {len(todo)} "
              f"({total_calls} calls per --limit)")

    if args.dry_run:
        return 0

    ok = 0
    failed = 0
    call_no = 0
    for i, (path, payload, pending) in enumerate(todo, 1):
        for model in pending:
            call_no += 1
            block = run_critic(
                payload,
                provider=args.provider,
                model=model,
                base_url=args.base_url,
            )
            _store_run(payload, run_block=block, primary_model=primary_model)
            status_tag = block.get("status") or "unknown"
            print(
                f"[{call_no}/{total_calls}] "
                f"{payload.get('symbol')} {payload.get('as_of_date')} "
                f"{model}: {status_tag}"
            )
            if block.get("status") == "ok":
                ok += 1
            else:
                failed += 1
            if args.pace_seconds > 0 and call_no < total_calls:
                time.sleep(args.pace_seconds)
        _refresh_disagreement(payload)
        try:
            Path(path).write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[fail] write {path}: {exc}")
            failed += 1
            continue
    print()
    print(f"Done. ok_calls={ok} failed_calls={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
