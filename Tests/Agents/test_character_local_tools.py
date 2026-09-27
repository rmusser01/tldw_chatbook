"""TASK-32954 Task 4: character_* tools registration, gate, and timeout."""

from __future__ import annotations

from tldw_chatbook.Agents import local_tool_provider as ltp
from tldw_chatbook.Agents.local_tool_provider import (
    LocalToolExposure,
    LocalToolProvider,
)
from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy
from tldw_chatbook.Tools.character_tool_service import (
    CharacterReadGuard,
    CharacterToolService,
)


def _service():
    return CharacterToolService(service_loader=lambda: None,
                                 runtime_source_loader=lambda: "local",
                                 read_guard=CharacterReadGuard())


def _provider(tmp_path, **kw):
    return LocalToolProvider(workspace_root=tmp_path, **kw)


def test_registered_when_service_supplied_and_gate_default_on(tmp_path, monkeypatch):
    monkeypatch.setattr(ltp, "get_cli_setting", lambda s, k, d=None: d, raising=False)
    names = {e.name for e in _provider(tmp_path, character_service=_service()).list_catalog()}
    assert {"character_search", "character_get", "character_save"} <= names


def test_absent_without_service_or_when_gate_off(tmp_path, monkeypatch):
    # Patched throughout (not just for the gate-off assertion below): an
    # unpatched `get_cli_setting` reaches the real config loader, which on
    # this dev machine trips a pre-existing, unrelated `RecoveryRequired`
    # (ADR-126) -- reproduced identically against a clean origin/dev
    # checkout, see the Task 4 report. Every other test in this module
    # already avoids the real loader the same way.
    monkeypatch.setattr(ltp, "get_cli_setting", lambda s, k, d=None: d, raising=False)
    assert "character_save" not in {e.name for e in _provider(tmp_path).list_catalog()}
    monkeypatch.setattr(
        ltp, "get_cli_setting",
        lambda s, k, d=None: False if k == ltp.CHARACTER_TOOLS_GATE_KEY else d, raising=False)
    names = {e.name for e in _provider(tmp_path, character_service=_service()).list_catalog()}
    assert "character_save" not in names


def test_spec_properties(tmp_path, monkeypatch):
    monkeypatch.setattr(ltp, "get_cli_setting", lambda s, k, d=None: d, raising=False)
    p = _provider(tmp_path, character_service=_service())
    specs = {n: p._specs[n] for n in ("character_search", "character_get", "character_save")}
    assert all(s.exposure is LocalToolExposure.CONSOLE_ONLY for s in specs.values())
    assert specs["character_save"].tags == ("mutates",)
    assert specs["character_save"].execution_policy is ToolExecutionPolicy.DEFINITIVE_AFTER_START
    assert specs["character_get"].tags == () and specs["character_save"].approval_arguments
    assert p.hub_tool_for("character_save").tags == ("mutates",)


def test_save_timeout_override(tmp_path, monkeypatch):
    monkeypatch.setattr(ltp, "get_cli_setting", lambda s, k, d=None: d, raising=False)
    p = _provider(tmp_path, character_service=_service())
    assert p.timeout_for("local:character_save") >= 300
    assert p.timeout_for("local:character_get") is None


def test_hub_lists_character_gate_default_on(monkeypatch):
    from tldw_chatbook.Agents import builtin_tool_gate as btg
    monkeypatch.setattr(btg, "_gate_config_snapshot", lambda: ({}, {}))
    gate = next(g for g in btg.all_tool_gates() if g.key == ltp.CHARACTER_TOOLS_GATE_KEY)
    assert gate.enabled is True and gate.group == "local"
    assert ("tools", ltp.CHARACTER_TOOLS_GATE_KEY) in btg._gate_key_pairs()


def test_save_approval_uses_the_services_bound_summary(tmp_path, monkeypatch):
    # Final review I2: the card names the character via the service's
    # session name map, so the provider must register the bound method.
    monkeypatch.setattr(ltp, "get_cli_setting", lambda s, k, d=None: d, raising=False)
    service = _service()
    p = _provider(tmp_path, character_service=service)
    assert p._specs["character_save"].approval_arguments == service.approval_summary


def test_approval_summary_failure_still_builds_a_gate_without_raw_args(tmp_path):
    # Final review M1: a raising approval_arguments must not break the
    # approval build, and must never fall back to the raw (full-text) args.
    from Tests.Agents.test_local_tool_provider import ASK, make_provider
    from tldw_chatbook.Agents.local_tool_provider import (
        LocalApprovalEffect,
        LocalToolSpec,
    )

    def _boom(_args):
        raise RuntimeError("summary boom")

    spec = LocalToolSpec(
        name="summarised",
        description="d",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        handler=lambda args: "ran",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(LocalApprovalEffect.MUTATES_LOCAL,),
        approval_arguments=_boom,
    )
    provider = make_provider(state=ASK, root=tmp_path, specs=[spec])
    gate, resolve_failed = provider._resolve_pending_gate(
        "summarised", {"secret": "FULL TEXT"}, provider.hub_tool_for("summarised")
    )
    assert gate is not None and resolve_failed is False
    assert "FULL TEXT" not in str(gate.arguments)
