"""Authoring API: grouping, baseline hashing, override state, save/reset.
Real config round-trip via the scratch_config fixture (Tests/Internal_Prompts/
conftest.py) — no mocks."""

import pytest

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Internal_Prompts.catalog import CATALOG


def test_iter_groups_cover_all_specs_stable_order():
    groups = authoring.iter_specs_by_subsystem()
    subsystems = [name for name, _ in groups]
    # every subsystem appears once, first-seen (registration) order
    assert subsystems == list(dict.fromkeys(s.subsystem for s in CATALOG.values()))
    flat = [spec.id for _, specs in groups for spec in specs]
    assert sorted(flat) == sorted(CATALOG.keys())
    # specs within a group sorted by title
    for _, specs in groups:
        assert [s.title for s in specs] == sorted(s.title for s in specs)


def test_baseline_hash_stable_and_short():
    h = authoring.baseline_hash("hello world")
    assert h == authoring.baseline_hash("hello world")
    assert h != authoring.baseline_hash("hello worlD")
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


def test_state_default_when_no_override(scratch_config):
    scratch_config("")
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is False
    assert st.default_changed is False
    assert st.has_override_table is False
    assert st.active_text == CATALOG["agents.subagent_system"].default


def test_save_makes_customized_with_fresh_baseline(scratch_config):
    scratch_config("")
    assert authoring.save_override("agents.subagent_system", "CUSTOM") is True
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is True
    assert st.has_override_table is True
    assert st.default_changed is False  # baseline written == current default hash
    assert st.active_text == "CUSTOM"


def test_default_changed_when_baseline_stale(scratch_config):
    scratch_config(
        "[internal_prompts.agents]\n"
        'subagent_system = { text = "CUSTOM", baseline = "deadbeef0000" }\n'
    )
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is True
    assert st.default_changed is True  # stored baseline != hash(current default)


def test_reset_removes_override_returns_to_default(scratch_config):
    scratch_config("")
    authoring.save_override("agents.subagent_system", "CUSTOM")
    assert authoring.reset_override("agents.subagent_system") is True
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is False
    assert st.active_text == CATALOG["agents.subagent_system"].default


def test_reset_deletes_customized_legacy_key(scratch_config):
    # rolling_summarize_system has legacy_config_path chunking_config.summarize_system_prompt
    scratch_config(
        "[chunking_config]\n"
        'summarize_system_prompt = "MY CUSTOM ROLLING PROMPT {none}"\n'
    )
    pid = "summarization.rolling_summarize_system"
    assert authoring.override_state(pid).customized is True
    assert authoring.reset_override(pid) is True
    from tldw_chatbook.Internal_Prompts import get_internal_prompt
    assert get_internal_prompt(pid) == CATALOG[pid].default


def test_reset_leaves_uncustomized_shipped_legacy_key(scratch_config, monkeypatch):
    # A doc-gen user prompt whose legacy [prompts.document_generation.*].prompt
    # equals the shipped default must NOT have that key deleted on reset.
    from tldw_chatbook import config as config_mod
    pid = "document_generation.timeline_user"
    shipped = config_mod.DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"]["timeline"]["prompt"]
    scratch_config(
        "[prompts.document_generation.timeline]\n"
        f'prompt = {shipped!r}\ntemperature = 0.3\n'
    )
    assert authoring.override_state(pid).customized is False  # equals shipped -> not customized
    assert authoring.reset_override(pid) is True
    # the legacy key survives (temperature sibling proves the table is intact)
    tbl = config_mod.get_cli_setting("prompts.document_generation", "timeline", None)
    assert tbl is not None and tbl.get("temperature") == 0.3
    assert tbl.get("prompt") == shipped


def test_customized_count(scratch_config):
    scratch_config("")
    assert authoring.customized_count() == 0
    authoring.save_override("agents.subagent_system", "A")
    authoring.save_override("agents.console_agent_operating", "B")
    assert authoring.customized_count() == 2
