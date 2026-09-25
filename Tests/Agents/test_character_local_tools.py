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
