#!/usr/bin/env bash
#
# Morning trading report automation.
#
# On a trading-day morning (gated by scripts/is_trading_day.py) this:
#   1. refreshes per-ticker composites via analysis_mvp.py (LLM layers on),
#   2. builds the daily signals report (daily_signals.py),
#   3. builds broker-gated trade tickets + Codex packet (trade_workflow.py),
#   4. uses the local `claude` CLI to write a readable morning briefing,
#   5. posts a macOS notification and opens the briefing.
#
# Nothing here places live orders. The generated tickets remain the source of
# truth for external (manual / Codex / Robinhood-MCP) execution.
#
# Designed to be invoked headless by launchd. It resolves its own repo root, so
# the launchd working directory does not matter.
#
# Usage:
#   scripts/morning_report.sh                 # full run, trading-day gated
#   scripts/morning_report.sh --force         # ignore the trading-day gate
#   scripts/morning_report.sh --no-llm        # faster: skip analysis LLM flags
#   scripts/morning_report.sh --skip-refresh  # reuse existing composites
#   scripts/morning_report.sh --no-briefing   # skip the claude briefing step
#   scripts/morning_report.sh --no-notify     # skip macOS notification / open
#   scripts/morning_report.sh --date 2026-06-18

set -euo pipefail

# --- Resolve repo root (works regardless of caller cwd / launchd) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- Keep the Mac awake for the whole run ----------------------------------
# launchd can fire this during a scheduled "dark wake" where macOS would
# otherwise doze back to sleep mid-pipeline (the LLM refresh can take minutes).
# Re-exec self once under caffeinate so idle/system sleep is held off until the
# script exits. -i prevents idle sleep; -s prevents system sleep on AC power.
if [ -z "${MORNING_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export MORNING_CAFFEINATED=1
  exec caffeinate -i -s "${BASH:-/bin/bash}" "$0" "$@"
fi

# --- Defaults (override via flags or environment) --------------------------
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
LLM_PROVIDER="${MORNING_LLM_PROVIDER:-openai}"
LLM_MODEL="${MORNING_LLM_MODEL:-gpt-5.4-mini}"

FORCE=0
USE_LLM=1
DO_REFRESH=1
DO_BRIEFING=1
DO_NOTIFY=1
RUN_DATE=""

while [ $# -gt 0 ]; do
  case "$1" in
    -f|--force) FORCE=1 ;;
    --no-llm) USE_LLM=0 ;;
    --skip-refresh) DO_REFRESH=0 ;;
    --no-briefing) DO_BRIEFING=0 ;;
    --no-notify) DO_NOTIFY=0 ;;
    --date) shift; RUN_DATE="${1:-}" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- Environment ------------------------------------------------------------
# Keep matplotlib from complaining about an unwritable cache under launchd.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/state/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

# Load secrets from .env (KEY=VALUE; preserves spaces; ignores comments).
load_env() {
  local f="$1" line key val
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"   # strip leading whitespace
    case "$line" in ''|'#'*) continue ;; esac
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key//[[:space:]]/}"
    [ -z "$key" ] && continue
    export "$key=$val"
  done < "$f"
}
load_env "$ROOT/.env"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: python interpreter not found/executable at: $PYTHON" >&2
  echo "Create it with: uv venv --python 3.13 .venv && uv pip install -e ." >&2
  exit 1
fi

# --- Run date (US/Eastern) + logging ---------------------------------------
if [ -z "$RUN_DATE" ]; then
  RUN_DATE="$("$PYTHON" -c "import datetime,zoneinfo;print(datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).date())")"
fi

MORNING_DIR="$ROOT/reports/morning"
mkdir -p "$MORNING_DIR"
LOG="$MORNING_DIR/${RUN_DATE}.log"
# Tee everything (stdout+stderr) to the per-day log and the console.
exec > >(tee -a "$LOG") 2>&1

log() { printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "==== morning_report start (run_date=$RUN_DATE) ===="
log "root=$ROOT python=$PYTHON llm=$USE_LLM provider=$LLM_PROVIDER model=$LLM_MODEL"

# --- Trading-day gate -------------------------------------------------------
if [ "$FORCE" -eq 1 ]; then
  log "trading-day gate bypassed (--force)"
else
  if "$PYTHON" scripts/is_trading_day.py --date "$RUN_DATE"; then
    log "trading-day gate passed"
  else
    log "not a trading day; exiting cleanly"
    log "==== morning_report end (skipped) ===="
    exit 0
  fi
fi

# --- 1. Composite refresh (analysis_mvp.py per ticker, LLM on) -------------
read_universe() {
  "$PYTHON" - <<'PY' 2>/dev/null || true
import yaml
try:
    cfg = yaml.safe_load(open("configs/sizing.yaml")) or {}
    print("\n".join(str(t) for t in (cfg.get("universe") or [])))
except Exception:
    pass
PY
}

FAILED_TICKERS=()
if [ "$DO_REFRESH" -eq 1 ]; then
  mapfile -t UNIVERSE < <(read_universe)
  if [ "${#UNIVERSE[@]}" -eq 0 ]; then
    log "WARN: could not read universe from configs/sizing.yaml; skipping refresh"
  else
    LLM_FLAGS=()
    if [ "$USE_LLM" -eq 1 ]; then
      LLM_FLAGS=(--enable-llm-insights --enable-narrative --enable-llm-critic \
                 --llm-provider "$LLM_PROVIDER" --llm-model "$LLM_MODEL")
    fi
    log "refreshing ${#UNIVERSE[@]} composites: ${UNIVERSE[*]}"
    for T in "${UNIVERSE[@]}"; do
      [ -z "$T" ] && continue
      log "  analysis_mvp $T ..."
      # One bad ticker must not abort the whole morning run.
      if ! "$PYTHON" analysis_mvp.py --ticker "$T" --date "$RUN_DATE" \
            --no-json-stdout "${LLM_FLAGS[@]}"; then
        log "  WARN: analysis_mvp failed for $T"
        FAILED_TICKERS+=("$T")
      fi
    done
    if [ "${#FAILED_TICKERS[@]}" -gt 0 ]; then
      log "composite refresh completed with failures: ${FAILED_TICKERS[*]}"
    else
      log "composite refresh completed"
    fi
  fi
else
  log "composite refresh skipped (--skip-refresh)"
fi

# --- 2. Daily signals -------------------------------------------------------
log "building daily signals ..."
"$PYTHON" daily_signals.py \
  --reports-glob "reports/analysis_mvp/*.json" \
  --positions portfolio/positions.json \
  --sizing-config configs/sizing.yaml \
  --output-dir reports/daily_signals \
  --as-of "$RUN_DATE"
SIGNALS_MD="$ROOT/reports/daily_signals/${RUN_DATE}.md"
log "daily signals: $SIGNALS_MD"

# --- 3. Trade tickets + Codex workflow packet ------------------------------
log "building trade tickets + workflow packet ..."
"$PYTHON" trade_workflow.py \
  --reports-glob "reports/analysis_mvp/*.json" \
  --positions portfolio/positions.json \
  --sizing-config configs/sizing.yaml \
  --execution-config configs/execution.yaml \
  --tickets-dir reports/trade_tickets \
  --workflow-dir reports/trade_workflow \
  --as-of "$RUN_DATE"
TICKETS_MD="$ROOT/reports/trade_tickets/${RUN_DATE}.md"
REVIEW_MD="$ROOT/reports/trade_workflow/${RUN_DATE}_codex_review.md"
log "tickets: $TICKETS_MD"
log "codex packet: $REVIEW_MD"

# --- 4. LLM morning briefing (local claude CLI) ----------------------------
BRIEF_MD="$MORNING_DIR/${RUN_DATE}_briefing.md"
if [ "$DO_BRIEFING" -eq 1 ]; then
  if command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
    log "writing morning briefing via claude ..."
    {
      echo "===== DAILY SIGNALS ($RUN_DATE) ====="
      [ -f "$SIGNALS_MD" ] && cat "$SIGNALS_MD" || echo "(daily signals report missing)"
      echo
      echo "===== TRADE WORKFLOW / CODEX PACKET ($RUN_DATE) ====="
      [ -f "$REVIEW_MD" ] && cat "$REVIEW_MD" || echo "(codex packet missing)"
    } | "$CLAUDE_BIN" -p "$(cat "$ROOT/prompts/morning_briefing.md")" > "$BRIEF_MD" \
        && log "briefing: $BRIEF_MD" \
        || log "WARN: claude briefing step failed"
  else
    log "WARN: '$CLAUDE_BIN' not on PATH; skipping briefing"
    DO_BRIEFING=0
  fi
else
  log "briefing skipped (--no-briefing)"
fi

# --- 5. Notify / open -------------------------------------------------------
if [ "$DO_NOTIFY" -eq 1 ] && command -v osascript >/dev/null 2>&1; then
  MSG="Daily signals + tickets ready for $RUN_DATE"
  [ "${#FAILED_TICKERS[@]}" -gt 0 ] && MSG="$MSG (refresh issues: ${FAILED_TICKERS[*]})"
  osascript -e "display notification \"$MSG\" with title \"Morning Trading Report\"" || true
  TO_OPEN="$BRIEF_MD"
  [ -f "$TO_OPEN" ] || TO_OPEN="$SIGNALS_MD"
  [ -f "$TO_OPEN" ] && command -v open >/dev/null 2>&1 && open "$TO_OPEN" || true
fi

log "==== morning_report end (ok) ===="
