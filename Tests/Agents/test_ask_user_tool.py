"""PRD Feature A: the ask_user LocalToolSpec, its gate, and its exemption."""

from __future__ import annotations

import json

import tldw_chatbook.Agents.local_tool_provider as provider_module
from Tests.Agents.test_local_tool_provider import ASK, make_provider
from tldw_chatbook.Agents.ask_user_questions import AskUserBusyRefusal
from tldw_chatbook.Agents.builtin_tool_gate import _gate_key_pairs, all_tool_gates
from tldw_chatbook.Agents.local_tool_provider import (
    LocalApprovalEffect,
    LocalToolExposure,
    LocalToolSpec,
)


def _exempt_spec(name: str = "ping") -> LocalToolSpec:
    return LocalToolSpec(
        name=name,
        description="pong",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: "pong",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(),
        gate_exempt=True,
    )


def _gated_spec(name: str = "gated") -> LocalToolSpec:
    return LocalToolSpec(
        name=name,
        description="gated",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: "ran",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(LocalApprovalEffect.MUTATES_LOCAL,),
    )


# --- Task 2: the permission-layer exemption ---------------------------------


def test_gate_exempt_defaults_false():
    assert _gated_spec().gate_exempt is False


def test_exempt_spec_runs_under_ask_with_no_approval_callback(tmp_path):
    provider = make_provider(
        state=ASK, root=tmp_path, specs=[_exempt_spec(), _gated_spec()]
    )
    result = provider.invoke("local:ping", {})
    assert result.ok is True and result.content == "pong"
    refused = provider.invoke("local:gated", {})
    assert refused.ok is False, "a non-exempt sibling still needs approval"


def test_exempt_spec_never_reaches_batch_review(tmp_path):
    provider = make_provider(state=ASK, root=tmp_path, specs=[_exempt_spec()])
    gate, resolve_failed = provider._resolve_pending_gate(
        "ping", {}, provider.hub_tool_for("ping")
    )
    assert gate is None and resolve_failed is False


# --- Task 3: the ask_user spec and its gate row -----------------------------


def _names(provider) -> set[str]:
    return {
        spec.name for spec in provider.specs_for_exposure(LocalToolExposure.CONSOLE_ONLY)
    }


def _raw(options=None):
    return {
        "questions": [
            {
                "question": "Which?",
                "header": "Pick",
                "options": options or [{"label": "a"}, {"label": "b"}],
            }
        ]
    }


def test_ask_user_registered_only_when_a_callback_is_supplied(tmp_path):
    assert "ask_user" not in _names(make_provider(root=tmp_path))
    provider = make_provider(
        root=tmp_path,
        ask_user=lambda questions: {"answered": False, "reason": "cancelled"},
    )
    assert "ask_user" in _names(provider)
    spec = next(
        s
        for s in provider.specs_for_exposure(LocalToolExposure.CONSOLE_ONLY)
        if s.name == "ask_user"
    )
    assert spec.gate_exempt is True
    assert spec.approval_effects == () and spec.tags == ()


def test_ask_user_absent_when_the_gate_is_off(tmp_path, monkeypatch):
    def fake_setting(section, key, default=None):
        if (section, key) == ("tools", "ask_user_enabled"):
            return False
        return default

    monkeypatch.setattr(provider_module, "get_cli_setting", fake_setting)
    provider = make_provider(root=tmp_path, ask_user=lambda questions: {})
    assert "ask_user" not in _names(provider)


def test_handler_validates_then_hands_cleaned_questions_to_the_callback(tmp_path):
    seen = []

    def _ask(questions):
        seen.append(questions)
        return {"answered": True, "answers": []}

    provider = make_provider(state=ASK, root=tmp_path, ask_user=_ask)
    result = provider.invoke("local:ask_user", _raw())
    assert result.ok is True, result.error
    assert json.loads(result.content) == {"answered": True, "answers": []}
    assert seen == [
        [
            {
                "question": "Which?",
                "header": "Pick",
                "multiSelect": False,
                "options": [
                    {"label": "a", "description": ""},
                    {"label": "b", "description": ""},
                ],
            }
        ]
    ]


def test_handler_rejects_bad_calls_with_an_actionable_error_and_never_calls_back(
    tmp_path,
):
    calls = []
    provider = make_provider(
        state=ASK, root=tmp_path, ask_user=lambda q: calls.append(q) or {}
    )
    result = provider.invoke("local:ask_user", _raw(options=[{"label": "one"}]))
    assert result.ok is False and "at least 2 items" in (result.error or "")
    assert calls == []


def test_busy_refusal_from_the_callback_is_a_tool_error(tmp_path):
    def _ask(questions):
        raise AskUserBusyRefusal("ask_user refused: busy twice")

    provider = make_provider(state=ASK, root=tmp_path, ask_user=_ask)
    result = provider.invoke("local:ask_user", _raw())
    assert result.ok is False and "busy twice" in (result.error or "")


def test_gate_row_is_enumerated_and_defaults_on():
    gate = next(g for g in all_tool_gates() if g.tool_name == "ask_user")
    assert (gate.section, gate.key, gate.group) == ("tools", "ask_user_enabled", "local")
    assert gate.enabled is True
    assert [(g.section, g.key) for g in all_tool_gates()] == _gate_key_pairs()


# --- task-31420: the description is a registry prompt ---------------------------


def _spec(provider):
    return next(
        s
        for s in provider.specs_for_exposure(LocalToolExposure.CONSOLE_ONLY)
        if s.name == "ask_user"
    )


def test_description_defaults_to_the_catalog_text(tmp_path):
    from tldw_chatbook.Agents.ask_user_questions import ASK_USER_DESCRIPTION

    provider = make_provider(root=tmp_path, ask_user=lambda questions: {})
    assert _spec(provider).description == ASK_USER_DESCRIPTION


def test_a_registry_override_reaches_the_built_spec(tmp_path, monkeypatch):
    import tldw_chatbook.config as config_module

    def fake_setting(section, key=None, default=None):
        if (section, key) == ("internal_prompts.agents", "ask_user_tool_description"):
            return "Ask only about deployment targets."
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_setting)
    provider = make_provider(root=tmp_path, ask_user=lambda questions: {})
    assert _spec(provider).description == "Ask only about deployment targets."
