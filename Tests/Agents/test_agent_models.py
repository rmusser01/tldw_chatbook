"""Pure model tests: values, defaults, and the child-budget clamp."""
import dataclasses

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, DIRECT_DISCLOSE_THRESHOLD,
    FIND_TOOLS_NAME, LOAD_TOOLS_NAME, LOOP_DETECTION_N, RUN_CANCELLED,
    RUN_DONE, RUN_ERROR, RUN_RUNNING, RUN_STUCK, RUN_SUPERSEDED,
    RUNTIME_TOOL_NAMES, SPAWN_TOOL_NAME, TERMINAL_RUN_STATUSES, AgentConfig,
    AgentStep, ModelTurn, RunBudget, RunOutcome, ToolCall, ToolCatalogEntry,
    ToolResult, ToolSchema, clamp_child_budget,
)


def test_run_status_values_and_terminal_set():
    assert (RUN_RUNNING, RUN_DONE, RUN_ERROR, RUN_STUCK, RUN_CANCELLED,
            RUN_SUPERSEDED) == ("running", "done", "error", "stuck",
                                "cancelled", "superseded")
    assert RUN_RUNNING not in TERMINAL_RUN_STATUSES
    assert TERMINAL_RUN_STATUSES == {
        "done", "error", "stuck", "cancelled", "superseded"}


def test_runtime_tool_names():
    assert SPAWN_TOOL_NAME == "spawn_subagent"
    assert RUNTIME_TOOL_NAMES == {"spawn_subagent", "find_tools", "load_tools"}
    assert DIRECT_DISCLOSE_THRESHOLD == 8 and LOOP_DETECTION_N == 3


def test_budget_defaults():
    b = RunBudget()
    assert (b.max_steps, b.max_wall_seconds, b.max_subagents,
            b.max_active_tools, b.max_subagent_result_chars) == (
                8, 240.0, 2, 8, 4000)


def test_run_budget_default_model_turns_equal_steps_and_child_clamp_carries():
    b = RunBudget()
    # Unreachability invariant: each model turn appends >=1 step, so with
    # max_model_turns == max_steps the step check always fires first (or
    # ties) at defaults -> engine-default behavior byte-identical.
    assert b.max_model_turns == b.max_steps == 8
    child = clamp_child_budget(RunBudget(max_model_turns=3),
                               parent_remaining_seconds=30.0)
    assert child.max_model_turns == 3


def test_clamp_child_budget_clamps_wall_clock_and_zeroes_spawn():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=30.0)
    assert child.max_wall_seconds == 30.0      # min(240, 30)
    assert child.max_subagents == 0            # depth 1: children never spawn
    assert child.max_steps == 8                # steps are per-run, not clamped


def test_clamp_child_budget_floors_at_one_second():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=-5.0)
    assert child.max_wall_seconds == 1.0


def test_models_construct_and_are_frozen_where_stated():
    entry = ToolCatalogEntry(id="builtin:x", name="x",
                             one_line_description="d", source="builtin")
    schema = ToolSchema(id="builtin:x", name="x", description="d",
                        parameters={"type": "object"})
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


def test_pure_module_has_no_forbidden_imports():
    import tldw_chatbook.Agents.agent_models as mod
    src = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("textual", "sqlite3", "tldw_chatbook.DB",
                      "tldw_chatbook.app", "httpx", "requests"):
        assert forbidden not in src
