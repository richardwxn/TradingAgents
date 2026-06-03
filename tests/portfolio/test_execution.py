from __future__ import annotations

from portfolio.execution import (
    ExecutionConfig,
    build_execution_batch,
    execution_config_from_dict,
    option_intent_tickets_from_report,
    ticket_from_action,
)
from portfolio.signals import Action


def _action(**overrides):
    base = {
        "symbol": "NVDA",
        "action": "BUY",
        "direction": "bullish",
        "composite": 0.4,
        "confidence": 0.7,
        "target_weight": 0.10,
        "current_weight": 0.0,
        "delta_pp": 0.10,
        "target_shares": 6,
        "current_shares": 0,
        "delta_shares": 6,
        "limit_price": 145.0,
        "stop_loss": 137.5,
        "last_close": 150.0,
        "sma20": 145.0,
        "atr14": 5.0,
        "signal_age_days": 2,
        "notes": [],
    }
    base.update(overrides)
    return Action(**base)


def test_executable_actions_convert_to_equity_limit_tickets():
    cfg = ExecutionConfig()
    cases = [
        ("BUY", 6, "buy"),
        ("ADD", 4, "buy"),
        ("TRIM", -3, "sell"),
        ("EXIT", -10, "sell"),
    ]
    for source_action, delta_shares, side in cases:
        ticket = ticket_from_action(
            _action(action=source_action, delta_shares=delta_shares),
            as_of="2026-05-31",
            config=cfg,
        )
        assert ticket is not None
        assert ticket.asset_type == "equity"
        assert ticket.side == side
        assert ticket.order_type == "limit"
        assert ticket.quantity == str(abs(delta_shares))
        assert ticket.limit_price == 145.0
        assert ticket.status == "ready"


def test_execution_config_from_dict_includes_review_gate_flag():
    cfg = execution_config_from_dict({
        "tradingagents_review_apply_to_tickets": False,
        "unknown": True,
    })
    assert cfg.tradingagents_review_apply_to_tickets is False


def test_hold_skip_review_do_not_become_tickets():
    cfg = ExecutionConfig()
    for source_action in ("HOLD", "SKIP", "REVIEW"):
        ticket = ticket_from_action(
            _action(action=source_action, delta_shares=0),
            as_of="2026-05-31",
            config=cfg,
        )
        assert ticket is None


def test_stale_missing_price_and_zero_delta_are_blocked():
    cfg = ExecutionConfig(max_signal_age_days=7)
    ticket = ticket_from_action(
        _action(
            delta_shares=0,
            limit_price=None,
            last_close=None,
            signal_age_days=12,
        ),
        as_of="2026-05-31",
        config=cfg,
    )
    assert ticket is not None
    assert ticket.status == "blocked"
    assert "absolute delta shares" in ticket.blocked_reason
    assert "missing limit price" in ticket.blocked_reason
    assert "missing price context" in ticket.blocked_reason
    assert "stale signal" in ticket.blocked_reason


def test_ticket_id_is_stable_for_same_source_action():
    cfg = ExecutionConfig()
    first = ticket_from_action(_action(), as_of="2026-05-31", config=cfg)
    second = ticket_from_action(_action(), as_of="2026-05-31", config=cfg)
    assert first.ticket_id == second.ticket_id


def test_build_batch_splits_ready_and_blocked_tickets():
    cfg = ExecutionConfig()
    batch = build_execution_batch(
        actions=[
            _action(symbol="NVDA", delta_shares=6),
            _action(symbol="AMD", delta_shares=0),
            _action(symbol="AVGO", action="HOLD", delta_shares=0),
        ],
        as_of="2026-05-31",
        config=cfg,
    )
    assert len(batch.tickets) == 1
    assert len(batch.blocked_tickets) == 1
    assert batch.summary["ready_count"] == 1
    assert batch.summary["blocked_count"] == 1


def test_review_gate_blocks_ready_buy_ticket_with_metadata():
    cfg = ExecutionConfig()
    ticket = ticket_from_action(
        _action(
            review_gate_status="manual_review",
            review_gate_reason="Needs human confirmation.",
            review_execution_caveats=["Risk critique."],
        ),
        as_of="2026-05-31",
        config=cfg,
    )
    assert ticket is not None
    assert ticket.status == "blocked"
    assert "TradingAgents manual-review gate" in ticket.blocked_reason
    assert ticket.review_gate_status == "manual_review"
    assert ticket.review_execution_caveats == ["Risk critique."]


def test_review_gate_does_not_block_exit_ticket():
    cfg = ExecutionConfig()
    ticket = ticket_from_action(
        _action(
            action="EXIT",
            delta_shares=-6,
            review_gate_status="block_buy_add",
            review_gate_reason="Risk process is bearish.",
        ),
        as_of="2026-05-31",
        config=cfg,
    )
    assert ticket is not None
    assert ticket.status == "ready"
    assert ticket.side == "sell"


def test_option_strategy_reports_emit_blocked_intent_tickets():
    cfg = ExecutionConfig()
    report = {
        "symbol": "AMD",
        "key_features": {
            "option_strategies": {
                "strategies": [
                    {
                        "type": "sell_put",
                        "verdict": "consider",
                        "reason": "Pays premium to enter lower.",
                        "expiry": "2026-06-19",
                        "strike": 90.0,
                        "premium": 1.25,
                    },
                    {
                        "type": "sell_call",
                        "verdict": "unavailable",
                        "reason": "Needs 100 shares.",
                    },
                ]
            }
        },
    }
    tickets = option_intent_tickets_from_report(report, as_of="2026-05-31", config=cfg)
    assert len(tickets) == 1
    assert tickets[0].asset_type == "option_intent"
    assert tickets[0].status == "blocked"
    assert "options placement is not available" in tickets[0].blocked_reason
