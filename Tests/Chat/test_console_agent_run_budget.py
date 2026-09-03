# Tests/Chat/test_console_agent_run_budget.py
"""TASK-18600: the Console agent's user-configurable run budget.

Covers:
  * the five `[console] agent_max_*` keys resolve config -> default, read
    fresh on every call so a Settings save applies to the next run (AC#2);
  * the shipped defaults are the ones the owner specified (AC#3);
  * floors and fallbacks: a below-floor, above-safe-ceiling, unparsable, or
    hostile value falls back to the shipped default instead of breaking a run;
  * the engine's own RunBudget defaults are untouched (AC#5);
  * the bridge's mirrored literals cannot drift from config's (the
    duplication that exists only because this module may not import config
    at the top level);
  * the derived step floor still admits a full model-turn run at the new
    numbers -- the invariant `test_console_budget_step_cap_admits_a_full_
    model_turn_run` pins for the old ones.
"""

from __future__ import annotations

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Agents.agent_models import (
    AGENT_LIFECYCLE_INDEX_BASE,
    CONTROL_CAPTURE_INDEX_BASE,
    MAX_RUN_CONTROL_STEPS,
    TRACE_CAPTURE_INDEX_BASE,
    TRACE_STEP_INDEX_BASE,
    RunBudget,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    DEFAULT_CONSOLE_MAX_MODEL_TURNS,
    DEFAULT_CONSOLE_MAX_STEPS,
    DEFAULT_CONSOLE_MAX_TOOL_CALL_SECONDS,
    DEFAULT_CONSOLE_MAX_TOTAL_TOKENS,
    DEFAULT_CONSOLE_MAX_WALL_SECONDS,
    DEFAULT_CONSOLE_RUN_BUDGET,
    console_run_budget,
)


def _pin_console(monkeypatch, values: dict):
    """Make `get_cli_setting("console", key, default)` read from `values`."""

    def fake(section, key, default=None, *a, **k):
        if section != "console":
            return default
        return values.get(key, default)

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", fake)


# -- shipped defaults -------------------------------------------------------


def test_shipped_defaults_are_the_specified_numbers():
    """AC#3. Pinned as literals: these are an owner decision, not a
    derivation, so a change to them must be a deliberate edit here too."""
    assert DEFAULT_CONSOLE_MAX_MODEL_TURNS == 2000
    assert DEFAULT_CONSOLE_MAX_STEPS == 25000
    assert DEFAULT_CONSOLE_MAX_WALL_SECONDS == 86400.0  # 24 hours
    assert DEFAULT_CONSOLE_MAX_TOTAL_TOKENS == 25_000_000
    assert DEFAULT_CONSOLE_MAX_TOOL_CALL_SECONDS == 3600.0  # 1 hour


def test_bridge_default_budget_matches_config_defaults():
    """The bridge mirrors config's DEFAULT_CONSOLE_AGENT_MAX_* as literals
    because it may not import config at module level (every config read in
    it is function-local). This is the pin that makes that duplication
    safe rather than a drift waiting to happen."""
    assert DEFAULT_CONSOLE_MAX_MODEL_TURNS == (
        config_module.DEFAULT_CONSOLE_AGENT_MAX_MODEL_TURNS
    )
    assert DEFAULT_CONSOLE_MAX_STEPS == config_module.DEFAULT_CONSOLE_AGENT_MAX_STEPS
    assert DEFAULT_CONSOLE_MAX_WALL_SECONDS == (
        config_module.DEFAULT_CONSOLE_AGENT_MAX_WALL_SECONDS
    )
    assert DEFAULT_CONSOLE_MAX_TOTAL_TOKENS == (
        config_module.DEFAULT_CONSOLE_AGENT_MAX_TOTAL_TOKENS
    )
    assert DEFAULT_CONSOLE_MAX_TOOL_CALL_SECONDS == (
        config_module.DEFAULT_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS
    )


def test_engine_run_budget_defaults_are_untouched():
    """AC#5. The engine's own floor stays conservative for non-Console
    callers -- raising the Console budget must not raise theirs."""
    engine = RunBudget()
    assert engine.max_steps == 8
    assert engine.max_wall_seconds == 240.0
    assert engine.max_model_turns == 30
    assert engine.max_total_tokens == 0
    assert engine.max_tool_call_seconds == 300.0


def test_step_backstop_admits_a_full_model_turn_run():
    """A fence tool round costs 3 steps and the wrap-up reply costs 1, so
    N turns need 3*(N-1)+1 steps. The step cap must clear that or it, not
    the turn cap, silently becomes the limiter."""
    turns = DEFAULT_CONSOLE_RUN_BUDGET.max_model_turns
    assert DEFAULT_CONSOLE_RUN_BUDGET.max_steps >= 3 * (turns - 1) + 1


# -- config resolution ------------------------------------------------------


def test_defaults_apply_when_nothing_is_configured(monkeypatch):
    _pin_console(monkeypatch, {})
    assert console_run_budget() == DEFAULT_CONSOLE_RUN_BUDGET


def test_every_key_is_configurable(monkeypatch):
    """AC#1/#2: all five limits, not just some, come from config."""
    _pin_console(
        monkeypatch,
        {
            "agent_max_model_turns": 7,
            "agent_max_steps": 42,
            "agent_max_wall_seconds": 90.5,
            "agent_max_total_tokens": 1234,
            "agent_max_tool_call_seconds": 11.0,
            "agent_max_model_retries": 4,
            "agent_budget_warning_fraction": 0.5,
        },
    )
    budget = console_run_budget()
    assert budget.max_model_turns == 7
    assert budget.max_steps == 42
    assert budget.max_wall_seconds == 90.5
    assert budget.max_total_tokens == 1234
    assert budget.max_tool_call_seconds == 11.0
    assert budget.max_model_retries == 4
    assert budget.budget_warning_fraction == 0.5


def test_budget_warning_fraction_is_clamped_to_one(monkeypatch):
    """Review A-5: an inert or typo'd key name would be invisible without a
    real-config test -- the exact defect class (C3a) that reopened 25902.
    A fraction above 1.0 clamps rather than disabling exhaustion handling."""
    _pin_console(monkeypatch, {"agent_budget_warning_fraction": 3.5})

    assert console_run_budget().budget_warning_fraction == 1.0


def test_budget_warning_fraction_rejects_garbage(monkeypatch):
    _pin_console(monkeypatch, {"agent_budget_warning_fraction": "soon"})

    assert console_run_budget().budget_warning_fraction == 0.8


def test_step_limit_is_capped_below_the_trace_storage_band(monkeypatch):
    assert config_module.MAX_CONSOLE_AGENT_MAX_STEPS == MAX_RUN_CONTROL_STEPS
    assert (
        TRACE_STEP_INDEX_BASE
        < TRACE_CAPTURE_INDEX_BASE
        < CONTROL_CAPTURE_INDEX_BASE
        < AGENT_LIFECYCLE_INDEX_BASE
    )
    assert TRACE_STEP_INDEX_BASE + (5 * MAX_RUN_CONTROL_STEPS) + 2 < (
        TRACE_CAPTURE_INDEX_BASE
    )
    _pin_console(monkeypatch, {"agent_max_steps": MAX_RUN_CONTROL_STEPS + 1})
    assert console_run_budget().max_steps == DEFAULT_CONSOLE_MAX_STEPS


def test_config_is_re_read_on_every_call(monkeypatch):
    """AC#2: a Settings save must reach the NEXT run with no restart, so
    nothing here may cache. Same guarantee `_console_tool_result_display_
    cap` already makes."""
    values = {"agent_max_model_turns": 5}
    _pin_console(monkeypatch, values)
    assert console_run_budget().max_model_turns == 5
    values["agent_max_model_turns"] = 9
    assert console_run_budget().max_model_turns == 9


def test_unconfigured_fields_keep_their_engine_shape(monkeypatch):
    """Only the five length/cost limits are user-facing. The fields that
    bound a run's SHAPE stay at engine defaults deliberately."""
    _pin_console(monkeypatch, {"agent_max_model_turns": 11})
    budget = console_run_budget()
    engine = RunBudget()
    assert budget.max_subagents == engine.max_subagents
    assert budget.max_tool_result_chars == engine.max_tool_result_chars
    assert budget.max_subagent_result_chars == engine.max_subagent_result_chars


# -- floors, fallbacks, and the absence of ceilings -------------------------


@pytest.mark.parametrize(
    "key,bad",
    [
        ("agent_max_model_turns", 0),
        ("agent_max_model_turns", -1),
        ("agent_max_steps", 0),
        ("agent_max_wall_seconds", 0.0),
        ("agent_max_total_tokens", -1),
        ("agent_max_tool_call_seconds", -5.0),
    ],
)
def test_below_floor_values_fall_back_to_the_default(monkeypatch, key, bad):
    """AC#4: a below-floor value is not silently clamped to the floor --
    it falls back to the shipped default, so a user never ends up running
    at a number nobody chose (a 0-turn budget would make every run stuck
    before its first provider call)."""
    _pin_console(monkeypatch, {key: bad})
    assert console_run_budget() == DEFAULT_CONSOLE_RUN_BUDGET


@pytest.mark.parametrize(
    "bad", ["not-a-number", None, "", [], {}, True, float("nan"), float("inf")]
)
def test_unparsable_values_fall_back_to_the_default(monkeypatch, bad):
    """A malformed config must never make the Console unable to run an
    agent. `inf` is included deliberately: an infinite wall budget would
    make the run's wall-clock check unfireable, not merely generous."""
    _pin_console(
        monkeypatch,
        {"agent_max_wall_seconds": bad, "agent_max_model_turns": bad},
    )
    budget = console_run_budget()
    assert budget.max_wall_seconds == DEFAULT_CONSOLE_MAX_WALL_SECONDS
    assert budget.max_model_turns == DEFAULT_CONSOLE_MAX_MODEL_TURNS


def test_zero_is_accepted_where_it_means_unlimited(monkeypatch):
    """0 is a real, documented value for the two ceilings that support it
    -- and must not be confused with the below-floor rejection below.

    The two keys differ in how 0 reaches the engine: tokens pass through
    literally (the engine's 0 already means "no token ceiling"), while
    tool-call seconds are translated to `UNLIMITED_TOOL_CALL_DEADLINE_
    SECONDS` -- the engine's literal 0 would bypass the timeout wrapper,
    which is also the only thing polling Stop while a tool runs."""
    from tldw_chatbook.Chat.console_agent_bridge import (
        UNLIMITED_TOOL_CALL_DEADLINE_SECONDS,
    )

    _pin_console(
        monkeypatch,
        {"agent_max_total_tokens": 0, "agent_max_tool_call_seconds": 0.0},
    )
    budget = console_run_budget()
    assert budget.max_total_tokens == 0
    assert budget.max_tool_call_seconds == UNLIMITED_TOOL_CALL_DEADLINE_SECONDS


def test_unlimited_tool_call_seconds_still_wrap_the_cancellation_poll(monkeypatch):
    """A configured `agent_max_tool_call_seconds = 0` must NOT reach the
    engine as its literal 0.

    The engine's 0 means "bypass `_call_with_timeout` entirely" (pinned by
    `test_make_invoke_tool_bypasses_wrapper_when_unlimited`), but that
    wrapper is the run's ONLY Stop poller while a tool call is in flight --
    `run_agent_loop` checks `should_cancel()` at step boundaries, which a
    hung tool never returns to. Passing 0 through would make "unlimited"
    silently mean "Stop does nothing until the tool returns by itself,"
    contradicting the documented "Stop works throughout". The resolver
    maps 0 to a finite-but-unfireable deadline instead, so the wrapper --
    and with it the 0.5s cancellation poll -- stays alive.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        UNLIMITED_TOOL_CALL_DEADLINE_SECONDS,
    )
    from tldw_chatbook.Agents.agent_models import RunBudget

    _pin_console(monkeypatch, {"agent_max_tool_call_seconds": 0.0})
    budget = console_run_budget()
    assert budget.max_tool_call_seconds == UNLIMITED_TOOL_CALL_DEADLINE_SECONDS
    # The wrapper gate in agent_service is `if timeout and timeout > 0` --
    # the translated value must clear it, i.e. be a live deadline the poll
    # loop runs under, not the engine's bypass sentinel.
    assert budget.max_tool_call_seconds > 0
    # And it must be absurdly beyond any wall budget a user can set, so it
    # can never pre-empt the run's own wall-clock ceiling.
    assert budget.max_tool_call_seconds > RunBudget().max_wall_seconds * 100


def test_there_is_no_upper_ceiling(monkeypatch):
    """AC#4: owner decision -- these are user-owned trade-offs, same call
    as `max_parallel_runs`. A ceiling invented here would just get in the
    way of the long expensive sessions this task exists to allow."""
    _pin_console(
        monkeypatch,
        {
            "agent_max_model_turns": 10_000_000,
            "agent_max_total_tokens": 10**12,
            "agent_max_wall_seconds": 31_536_000.0,
        },
    )
    budget = console_run_budget()
    assert budget.max_model_turns == 10_000_000
    assert budget.max_total_tokens == 10**12
    assert budget.max_wall_seconds == 31_536_000.0


def test_a_broken_config_read_still_yields_a_usable_budget(monkeypatch):
    """The whole point of the try/except around the resolver: an agent run
    must survive a config layer that raises."""

    def boom(*a, **k):
        raise RuntimeError("config exploded")

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", boom)
    assert console_run_budget() == DEFAULT_CONSOLE_RUN_BUDGET
