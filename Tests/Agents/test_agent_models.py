"""Pure model tests: values, defaults, and the child-budget clamp."""

import dataclasses

from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD,
    INSTALL_SKILL_TOOL_NAME,
    LOOP_DETECTION_N,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RUN_RUNNING,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    RUN_STUCK,
    RUN_SUPERSEDED,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    TERMINAL_RUN_STATUSES,
    AgentConfig,
    AgentStep,
    ModelTurn,
    RunBudget,
    RunOutcome,
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
    clamp_child_budget,
)


def test_run_status_values_and_terminal_set():
    assert (
        RUN_RUNNING,
        RUN_DONE,
        RUN_ERROR,
        RUN_STUCK,
        RUN_CANCELLED,
        RUN_SUPERSEDED,
    ) == ("running", "done", "error", "stuck", "cancelled", "superseded")
    assert RUN_RUNNING not in TERMINAL_RUN_STATUSES
    assert TERMINAL_RUN_STATUSES == {
        "done",
        "error",
        "stuck",
        "cancelled",
        "superseded",
    }


def test_runtime_tool_names():
    assert SPAWN_TOOL_NAME == "spawn_subagent"
    assert RUNTIME_TOOL_NAMES == {
        "spawn_subagent",
        "find_tools",
        "load_tools",
        "skill_file",
        INSTALL_SKILL_TOOL_NAME,
        RUN_SKILL_SCRIPT_TOOL_NAME,
    }
    assert DIRECT_DISCLOSE_THRESHOLD == 16 and LOOP_DETECTION_N == 3


def test_budget_defaults():
    b = RunBudget()
    assert (
        b.max_steps,
        b.max_wall_seconds,
        b.max_subagents,
        b.max_active_tools,
        b.max_subagent_result_chars,
    ) == (8, 240.0, 2, 24, 4000)


def test_run_budget_default_model_turns_unreachable_and_child_clamp_carries():
    """Pin the engine-default unreachability invariant and child passthrough.

    At ``RunBudget()`` the model-turn cap must be at least ``max_steps`` (so
    the model-turn check can never fire before the step check), and
    ``clamp_child_budget`` must carry ``max_model_turns`` through unclamped.
    """
    b = RunBudget()
    # Unreachability invariant: each model turn appends >=1 step, so with
    # max_model_turns >= max_steps the step check always fires first (or
    # ties) at defaults -> engine-default behavior unchanged. The cap was
    # raised to 30 for callers that raise max_steps to match (the Console
    # bridge); it must never drop below max_steps here.
    assert b.max_model_turns >= b.max_steps
    assert (b.max_model_turns, b.max_steps) == (30, 8)
    child = clamp_child_budget(
        RunBudget(max_model_turns=3), parent_remaining_seconds=30.0
    )
    assert child.max_model_turns == 3


def test_clamp_child_budget_clamps_wall_clock_and_zeroes_spawn():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=30.0)
    assert child.max_wall_seconds == 30.0  # min(240, 30)
    assert child.max_subagents == 0  # depth 1: children never spawn
    assert child.max_steps == 8  # steps are per-run, not clamped


def test_clamp_child_budget_floors_at_one_second():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=-5.0)
    assert child.max_wall_seconds == 1.0


def test_models_construct_and_are_frozen_where_stated():
    entry = ToolCatalogEntry(
        id="builtin:x", name="x", one_line_description="d", source="builtin"
    )
    schema = ToolSchema(
        id="builtin:x", name="x", description="d", parameters={"type": "object"}
    )
    call = ToolCall(name="x", args={"a": 1})
    result = ToolResult(ok=True, content="42")
    turn = ModelTurn(text="hi", tool_calls=(call,))
    cfg = AgentConfig(model="m", system_prompt="s", allowed_tools=("x",))
    step = AgentStep(index=0, kind="model", summary="hi")
    outcome = RunOutcome(status=RUN_DONE, steps=[step], final_text="hi")
    assert turn.tool_calls[0].args == {"a": 1}
    assert cfg.budget.max_steps == 8 and outcome.subagents_spawned == 0
    for frozen in (entry, schema, call, result, turn, cfg):
        assert dataclasses.fields(frozen)  # constructed fine
        try:
            object.__setattr__  # noqa: B018 — presence check only
            frozen.__class__.__dataclass_params__.frozen
        except AttributeError:  # pragma: no cover
            pass
        assert frozen.__dataclass_params__.frozen is True


def test_modelturn_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import ModelTurn
    assert ModelTurn(text="hi").tokens == 0
    assert ModelTurn(text="hi", tokens=42).tokens == 42


def test_runbudget_max_total_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import RunBudget
    assert RunBudget().max_total_tokens == 0
    assert RunBudget(max_total_tokens=5000).max_total_tokens == 5000


def test_runoutcome_total_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import RunOutcome, RUN_DONE
    assert RunOutcome(RUN_DONE, []).total_tokens == 0
    assert RunOutcome(RUN_DONE, [], total_tokens=123).total_tokens == 123


def test_clamp_child_budget_preserves_max_total_tokens():
    from tldw_chatbook.Agents.agent_models import RunBudget, clamp_child_budget
    child = RunBudget(max_total_tokens=7000)
    assert clamp_child_budget(child, 10.0).max_total_tokens == 7000


def test_clamp_child_budget_propagates_tool_call_seconds():
    parent = RunBudget(max_tool_call_seconds=45.0)
    child = clamp_child_budget(parent, 30.0)
    assert child.max_tool_call_seconds == 45.0   # taken from the child arg (== parent here)
    assert child.max_subagents == 0              # existing invariant still holds


def test_pure_module_has_no_forbidden_imports():
    import tldw_chatbook.Agents.agent_models as mod

    src = open(mod.__file__, encoding="utf-8").read()
    for forbidden in (
        "textual",
        "sqlite3",
        "tldw_chatbook.DB",
        "tldw_chatbook.app",
        "httpx",
        "requests",
    ):
        assert forbidden not in src


def test_direct_disclose_threshold_admits_a_three_pack_catalog():
    """A files+corpus+authoring set is 14 tools; it must disclose directly.

    Below the threshold `initial_disclosure` skips find_tools/load_tools
    entirely, which is the point: those two round trips are pure overhead
    repeated on every user message.
    """
    from tldw_chatbook.Agents.agent_models import DIRECT_DISCLOSE_THRESHOLD

    assert DIRECT_DISCLOSE_THRESHOLD >= 14


def test_max_active_tools_clears_the_disclosure_threshold():
    """Everything directly disclosed must fit in the active set.

    `initial_disclosure` truncates to `max_active_tools`, so a ceiling below
    the threshold would silently drop tools it just decided to disclose.
    """
    from tldw_chatbook.Agents.agent_models import (
        DIRECT_DISCLOSE_THRESHOLD,
        RunBudget,
    )

    assert RunBudget().max_active_tools >= DIRECT_DISCLOSE_THRESHOLD
