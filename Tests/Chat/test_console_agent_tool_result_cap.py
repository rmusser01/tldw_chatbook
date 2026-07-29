# Tests/Chat/test_console_agent_tool_result_cap.py
"""TASK-870: the Console's configurable agent tool-result display cap.

Covers:
  * the cap resolves env var -> config.toml -> default (mirrors
    ``run_log._setting``'s tier order), read fresh on every call;
  * the live step summary, the transcript TOOL marker, and a
    resumed/persisted step's summary all apply the SAME cap (AC#4);
  * a resumed step truncates on a word boundary with the ``(+N chars)``
    affordance, never a bare mid-word clip (AC#5, the exact defect
    task-350 fixed for the live path but not this one);
  * changing the setting affects the very next rendered step, live or
    resumed, without needing to reload/reimport anything (AC#3);
  * the "read the full result from the run log" affordance is available
    exactly when a run log exists for the run, and absent otherwise
    (AC#6/#7).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    AgentStep,
    STEP_TOOL_RESULT,
)
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    _console_tool_result_display_cap,
    format_agent_step_marker,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


LONG_RESULT = (
    "The traditional rollback procedure requires draining every in-flight "
    "request before the schema migration begins, otherwise a half-applied "
    "column default can leave orphaned rows that the backfill job never "
    "revisits, which is exactly the failure mode this runbook exists to "
    "prevent for anyone paging through it at 3am."
)


# -- resolution order -------------------------------------------------------


def test_default_cap_is_160_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda *a, **k: 160,
    )
    assert _console_tool_result_display_cap() == 160


def test_config_toml_value_is_honoured_when_no_env_var(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key, default: 500
        if (section, key) == ("console", "tool_result_display_chars")
        else default,
    )
    assert _console_tool_result_display_cap() == 500


def test_env_var_takes_precedence_over_config_toml(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", "300")
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda *a, **k: 500,  # would win if the env tier were skipped
    )
    assert _console_tool_result_display_cap() == 300


def test_out_of_range_config_value_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda *a, **k: 999_999,  # above MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS
    )
    assert _console_tool_result_display_cap() == 160


def test_unparsable_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", "not-a-number")
    assert _console_tool_result_display_cap() == 160


# -- AC#3: no restart needed -------------------------------------------------


def test_changing_the_env_var_affects_the_very_next_call(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 160)
    assert _console_tool_result_display_cap() == 160

    monkeypatch.setenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", "40")
    assert _console_tool_result_display_cap() == 40

    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    assert _console_tool_result_display_cap() == 160


def test_changing_the_setting_changes_a_resumed_steps_rendered_text(monkeypatch):
    """AC#3, exercised through the actual render path rather than the bare
    resolver -- a Settings save must be visible on the very next step
    rendered, with nothing cached anywhere in between."""
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    step = {"kind": STEP_TOOL_RESULT, "result": LONG_RESULT}

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 40)
    short = ConsoleAgentBridge._summarize_persisted_step(step)

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 200)
    longer = ConsoleAgentBridge._summarize_persisted_step(step)

    assert len(short) < len(longer)
    assert short != longer


# -- AC#4: all three render paths share one cap ------------------------------


def test_live_marker_and_resumed_paths_all_apply_the_same_cap(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 60)

    live_step = AgentStep(index=0, kind=STEP_TOOL_RESULT, result=LONG_RESULT)
    live_text = ConsoleAgentBridge._summarize(live_step)

    marker_text = format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="run_migration", result=LONG_RESULT
    )
    # Strip the "⚙ run_migration → " prefix format_agent_step_marker adds.
    marker_preview = marker_text.split("→ ", 1)[1]

    persisted_text = ConsoleAgentBridge._summarize_persisted_step(
        {"kind": STEP_TOOL_RESULT, "result": LONG_RESULT}
    )

    assert live_text == marker_preview == persisted_text
    assert live_text.endswith("chars)")


def test_raising_the_cap_shows_more_of_the_result_on_every_path(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 40)
    small_live = ConsoleAgentBridge._summarize(
        AgentStep(index=0, kind=STEP_TOOL_RESULT, result=LONG_RESULT)
    )
    small_persisted = ConsoleAgentBridge._summarize_persisted_step(
        {"kind": STEP_TOOL_RESULT, "result": LONG_RESULT}
    )

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 400)
    big_live = ConsoleAgentBridge._summarize(
        AgentStep(index=0, kind=STEP_TOOL_RESULT, result=LONG_RESULT)
    )
    big_persisted = ConsoleAgentBridge._summarize_persisted_step(
        {"kind": STEP_TOOL_RESULT, "result": LONG_RESULT}
    )

    assert len(small_live) < len(big_live)
    assert len(small_persisted) < len(big_persisted)
    # The 400-char cap covers the whole (much shorter) LONG_RESULT string,
    # so both long-form renders should recover the full text verbatim.
    assert big_live == LONG_RESULT
    assert big_persisted == LONG_RESULT


# -- AC#5: resumed steps never show a silent mid-word clip -------------------


def test_resumed_step_truncates_on_a_word_boundary_with_affordance(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    # 20 lands mid-word inside "rollback" for LONG_RESULT ("The traditional
    # roll..."). A bare str(raw)[:20] slice (this task's exact pre-fix
    # defect for the RESUMED path -- see _summarize_persisted_step's old
    # `return str(raw)[:200]`) would silently keep that partial word.
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 20)

    result = ConsoleAgentBridge._summarize_persisted_step(
        {"kind": STEP_TOOL_RESULT, "result": LONG_RESULT}
    )

    assert result != LONG_RESULT[:20]  # not a bare, unmarked slice
    assert "(+" in result and result.endswith("chars)")
    visible = result.split("…")[0]
    assert LONG_RESULT.startswith(visible)
    # The visible prefix ends exactly on a word boundary: either it is
    # empty, or the very next character in the source text is whitespace
    # (a mid-word cut like "rollba" would instead be followed by a letter).
    assert visible == "" or LONG_RESULT[len(visible)] == " "
    assert not visible.endswith("rollba")


def test_resumed_step_short_enough_result_is_returned_verbatim():
    step = {"kind": STEP_TOOL_RESULT, "result": "ok"}
    assert ConsoleAgentBridge._summarize_persisted_step(step) == "ok"


# -- AC#6/#7: the full-log affordance is present iff a log exists -----------


@pytest.fixture
def bridge(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    return ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None), db


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def test_run_log_available_is_false_when_no_log_was_ever_written(bridge, log_root):
    console_bridge, _db = bridge
    assert console_bridge.run_log_available("run-never-logged") is False


def test_run_log_available_is_true_once_a_record_is_written(bridge, log_root):
    console_bridge, _db = bridge
    writer = RunLogWriter()
    writer.bind("run-logged")
    writer.append(run_id="run-logged", kind="primary", type="model", content="hi")

    assert console_bridge.run_log_available("run-logged") is True


def test_load_run_log_text_is_empty_when_no_log_exists(bridge, log_root):
    console_bridge, _db = bridge
    assert console_bridge.load_run_log_text("run-absent") == ""


def test_load_run_log_text_returns_the_full_untruncated_result(bridge, log_root):
    console_bridge, _db = bridge
    writer = RunLogWriter()
    writer.bind("run-full")
    writer.append(
        run_id="run-full",
        kind="primary",
        type="tool_result",
        tool="grep_files",
        status="ok",
        content=LONG_RESULT,
    )

    text = console_bridge.load_run_log_text("run-full")

    assert LONG_RESULT in text
    assert "grep_files" in text


def test_latest_primary_run_id_resolves_the_newest_primary_run(bridge):
    console_bridge, db = bridge
    run_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)

    assert console_bridge.latest_primary_run_id("conv-1") == run_id


def test_latest_primary_run_id_is_none_for_an_unknown_conversation(bridge):
    console_bridge, _db = bridge
    assert console_bridge.latest_primary_run_id("conv-never-ran") is None


# -- Review finding B: a sub-agent's records live in its PRIMARY's log
# directory (only the primary binds a RunLogWriter), tagged with the
# sub-agent's OWN run id per record -- the affordance must resolve through
# the primary and filter to just that sub-agent's records. --


def test_run_log_available_is_true_for_a_primary_run(bridge, log_root):
    """Unchanged baseline: a primary run's own directory is still checked directly."""
    console_bridge, db = bridge
    primary_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    writer = RunLogWriter()
    writer.bind(primary_id)
    writer.append(run_id=primary_id, kind="primary", type="model", content="hi")

    assert console_bridge.run_log_available(primary_id) is True


def test_run_log_available_is_true_for_a_drilled_in_subagent_run(bridge, log_root):
    """Pre-fix: this always returned False -- a sub-agent never binds its own
    writer/directory, so checking the sub-agent's own run id directly could
    never find anything, and the "View full log" affordance could never
    appear once drilled into a sub-agent."""
    console_bridge, db = bridge
    primary_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    subagent_id = db.create_run(
        conversation_id="conv-1",
        agent_kind=AGENT_KIND_SUBAGENT,
        parent_run_id=primary_id,
        task="fetch docs",
    )
    writer = RunLogWriter()
    writer.bind(primary_id)  # only the PRIMARY id binds a writer/directory
    writer.append(run_id=primary_id, kind="primary", type="model", content="primary turn")
    writer.append(
        run_id=subagent_id,
        kind="subagent",
        type="tool_result",
        tool="fetch",
        content="sub-agent's own result",
    )

    assert console_bridge.run_log_available(subagent_id) is True


def test_run_log_available_is_false_for_a_subagent_that_never_logged_a_record(
    bridge, log_root
):
    """The primary's directory existing is not enough -- this sub-agent must
    have at least one record of its OWN in it, or the button would dangle."""
    console_bridge, db = bridge
    primary_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    subagent_id = db.create_run(
        conversation_id="conv-1",
        agent_kind=AGENT_KIND_SUBAGENT,
        parent_run_id=primary_id,
        task="fetch docs",
    )
    writer = RunLogWriter()
    writer.bind(primary_id)
    writer.append(run_id=primary_id, kind="primary", type="model", content="primary only")

    assert console_bridge.run_log_available(subagent_id) is False


def test_load_run_log_text_for_a_subagent_only_shows_that_subagents_records(
    bridge, log_root
):
    console_bridge, db = bridge
    primary_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    subagent_id = db.create_run(
        conversation_id="conv-1",
        agent_kind=AGENT_KIND_SUBAGENT,
        parent_run_id=primary_id,
        task="fetch docs",
    )
    other_subagent_id = db.create_run(
        conversation_id="conv-1",
        agent_kind=AGENT_KIND_SUBAGENT,
        parent_run_id=primary_id,
        task="summarize",
    )
    writer = RunLogWriter()
    writer.bind(primary_id)
    writer.append(
        run_id=primary_id, kind="primary", type="model", content="primary secret plan"
    )
    writer.append(
        run_id=subagent_id,
        kind="subagent",
        type="tool_result",
        tool="fetch",
        content="the target sub-agent's own result",
    )
    writer.append(
        run_id=other_subagent_id,
        kind="subagent",
        type="tool_result",
        tool="summarize",
        content="a DIFFERENT sub-agent's result",
    )

    text = console_bridge.load_run_log_text(subagent_id)

    assert "the target sub-agent's own result" in text
    assert "primary secret plan" not in text
    assert "a DIFFERENT sub-agent's result" not in text


def test_load_run_log_text_for_a_primary_run_is_unaffected_by_subagents(
    bridge, log_root
):
    console_bridge, db = bridge
    primary_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    subagent_id = db.create_run(
        conversation_id="conv-1",
        agent_kind=AGENT_KIND_SUBAGENT,
        parent_run_id=primary_id,
        task="fetch docs",
    )
    writer = RunLogWriter()
    writer.bind(primary_id)
    writer.append(run_id=primary_id, kind="primary", type="model", content="primary turn")
    writer.append(
        run_id=subagent_id, kind="subagent", type="tool_result", tool="fetch", content="sub"
    )

    text = console_bridge.load_run_log_text(primary_id)

    assert "primary turn" in text
    assert "sub" in text  # the primary's own view still shows everything


# -- Review finding E: the viewer's render window must cover whatever the
# CURRENT `run_log_max_record_bytes` config allows the writer to store --
# a fixed, smaller window could leave a real record behind an unreachable
# "Use offset=N to continue" marker. --


def test_load_run_log_text_window_grows_with_a_raised_max_record_bytes_config(
    bridge, log_root, monkeypatch
):
    console_bridge, db = bridge
    run_id = db.create_run(conversation_id="conv-1", agent_kind=AGENT_KIND_PRIMARY)
    big_content = "x" * 2_500_000  # bigger than the old fixed 2,000,000 window

    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key, default: 3_000_000
        if (section, key) == ("agents", "run_log_max_record_bytes")
        else default,
    )
    writer = RunLogWriter()
    writer.bind(run_id)
    writer.append(run_id=run_id, kind="primary", type="tool_result", tool="t", content=big_content)

    text = console_bridge.load_run_log_text(run_id)

    assert big_content in text
    assert "Use offset=" not in text
