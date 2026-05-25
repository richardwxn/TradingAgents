"""Daily-signals + portfolio simulator package.

`portfolio.sizing` and `portfolio.signals` are pure functions used by the
`daily_signals.py` CLI to turn weekly analysis reports + the user's
position ledger into a markdown action list.

`portfolio.simulator` (Section 17) is the backtest harness that scores
sizing policies on the historical corpus to recommend a default for the
daily signals.
"""
