from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.agent_runtime import render_tool_protocol
from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools
from tldw_chatbook.Chat.console_prepared_request import (
    ConsoleRequestTokenAccounting,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
)

from tldw_chatbook.Chat.console_skill_cost_report import (
    SKILL_CHILD_OBSERVATION,
    format_skill_cost_report,
    skill_report_accounting,
    observe_skill_use,
    prepared_agent_skill_observer,
    record_skill_cost,
    skill_report_scope,
)


def session(workspace="w"):
    return SimpleNamespace(
        incarnation_id="session",
        workspace_id=workspace,
        conversation_binding_revision=0,
    )


def test_observations_are_scoped_and_retry_safe():
    chat = session()
    scope = skill_report_scope(chat)
    observe_skill_use(chat, scope, ("review", "review"), turn_id="turn", now=100)
    observe_skill_use(chat, scope, ("review",), turn_id="turn", now=101)
    observe_skill_use(chat, scope, ("review",), body_tokens=12, now=102)
    assert "2 observed" in format_skill_cost_report(chat, {}, now=103)
    assert "12 tokens" in format_skill_cost_report(chat, {}, now=103)

    chat.workspace_id = "new"
    observe_skill_use(chat, scope, ("review",), body_tokens=12, now=104)
    assert "review" not in format_skill_cost_report(chat, {}, now=105)
    assert "usage unknown" in format_skill_cost_report(chat, {}, now=105)


def test_cost_estimates_partition_the_original_buckets():
    chat = session()
    accounting = SimpleNamespace(
        total_input_tokens=100, tool_schema_tokens=15, system_tokens=30
    )
    record_skill_cost(
        chat,
        skill_report_scope(chat),
        accounting,
        native=True,
        schema_weights=(("one", 3), ("two", 1), (None, 2)),
        now=100,
    )
    report = format_skill_cost_report(
        chat,
        {"available_skills": [{"name": "one"}, {"name": "two"}]},
        now=104,
    )
    assert "100 tokens" in report
    assert "10 tokens" in report
    assert "5 tokens" in report
    assert "4s ago" in report
    assert "one" in report
    assert "usage unknown" in report


def test_ambiguous_attribution_and_blocked_names_are_explicit():
    chat = session()
    accounting = SimpleNamespace(
        total_input_tokens=100, tool_schema_tokens=15, system_tokens=30
    )
    record_skill_cost(chat, skill_report_scope(chat), accounting, native=True, now=100)
    report = format_skill_cost_report(
        chat,
        {
            "available_skills": [{"name": "review"}, {"name": "REVIEW"}],
            "blocked_skills": [{"name": "blocked", "description": "SECRET"}],
        },
        now=101,
    )
    assert "unavailable" in report
    assert "Conflict" in report
    assert "blocked" in report
    assert "Library ▸ Skills" in report
    assert "SECRET" not in report


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [False, True], ids=["fenced", "native"])
def test_prepared_sample_uses_only_its_admitted_schema_rows(native):
    chat = session()
    scope = skill_report_scope(chat)
    schemas = [
        ToolSchema("skill:review", "review", "PRIVATE SKILL DESCRIPTION", {}),
        ToolSchema("builtin:search", "search", "PRIVATE TOOL DESCRIPTION", {}),
    ]
    system = "PRIVATE SYSTEM INSTRUCTIONS"
    protocol = ""
    if not native:
        protocol = render_tool_protocol(schemas)
        system += "\n\n" + protocol
    semantic = build_console_request(
        [{"role": "system", "content": system}, {"role": "user", "content": "help"}],
        tools=schemas_to_openai_tools(schemas) if native else (),
    )
    request = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="test-model",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=lambda rows, _: sum(len(row.get("content", "")) + 1 for row in rows),
    )
    prepared_agent_skill_observer(
        chat,
        scope,
        schemas,
        {"skill:review": "review"},
        native=native,
        protocol=protocol,
    )(request)
    catalog = {"available_skills": [{"name": "new-skill"}]}
    report = format_skill_cost_report(chat, catalog)
    assert "review: ~" in report
    assert "0 observed" in report
    assert "new-skill: no advertised schema in sample" in report
    assert "absent from the current catalog" in report
    assert "PRIVATE" not in report
    cost_rows = (
        "Advertised skills:",
        "Other tool schemas and framing:",
        "Remaining system prompt:",
        "Other request categories:",
    )
    estimates = [
        int(line.split(": ", 1)[1].split()[0].lstrip("~").replace(",", ""))
        for line in report.splitlines()
        if line.startswith(cost_rows)
    ]
    assert len(estimates) == len(cost_rows)
    assert sum(estimates) == request.accounting.total_input_tokens
    assert skill_report_accounting(chat) is request.accounting

    # A different selection with the same visible name is insufficient evidence.
    changed = [
        ToolSchema("skill:review", "review", "changed definition", {}),
        schemas[1],
    ]
    prepared_agent_skill_observer(
        chat,
        scope,
        changed,
        {"skill:review": "review"},
        native=native,
        protocol="" if native else render_tool_protocol(changed),
    )(request)
    assert "skill attribution unavailable" in format_skill_cost_report(chat, catalog)

    # Reusing a conversation slot and observing a new session both start empty.
    chat.conversation_binding_revision += 1
    assert "Prepared request costs unavailable" in format_skill_cost_report(
        chat, catalog
    )
    assert "usage unknown" in format_skill_cost_report(session(), catalog)


def test_agent_runner_observes_accepted_child_that_fails_but_excludes_refusal():
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Chat.console_agent_bridge import _BridgeSkillRunner

    class Skills:
        async def execute_skill(self, name, *, mode, args):
            return {"rendered_prompt": "PRIVATE BODY", "allowed_tools": []}

    chat = session()
    runner = _BridgeSkillRunner(
        skills_service=Skills(),
        skill_names=frozenset({"review"}),
        builtin_names=(),
        skill_report_owner=(chat, skill_report_scope(chat)),
    )
    refused = runner.run(
        "review", "PRIVATE ARGS", lambda *a, **kw: ToolResult(ok=False)
    )
    assert not refused.ok
    assert "review:" not in format_skill_cost_report(chat, {})

    def child_fails_after_preparing(task, **_kw):
        observation = SKILL_CHILD_OBSERVATION.get()
        observation.observe(
            SimpleNamespace(
                semantic=build_console_request(
                    [
                        {"role": "user", "content": task},
                    ]
                )
            )
        )
        assert "review: 1 observed" in format_skill_cost_report(chat, {})
        return ToolResult(ok=False, dispatch_state="settled")

    used = runner.run("review", "PRIVATE ARGS", child_fails_after_preparing)
    assert not used.ok
    report = format_skill_cost_report(chat, {})
    assert "review: 1 observed" in report
    assert "~3 tokens of loaded child skill text (separate)" in report
    assert "PRIVATE" not in report


def test_context_forecast_preserves_the_preflight_system_base():
    chat = session()
    scope = skill_report_scope(chat)
    preflight = ConsoleRequestTokenAccounting(
        100, 75, 0, 0, 0, 25, system_prompt_delta_tokens=30
    )
    final = ConsoleRequestTokenAccounting(120, 90, 0, 0, 0, 30)
    record_skill_cost(chat, scope, preflight, native=False)
    record_skill_cost(chat, scope, final, native=False, prepared=True)

    shared = skill_report_accounting(chat)
    assert shared.total_input_tokens == 120
    assert shared.system_prompt_delta_tokens == 45
    assert "120 tokens" in format_skill_cost_report(chat, {})


def test_plugin_status_distinguishes_disabled_and_unavailable_components():
    report = format_skill_cost_report(
        session(),
        {"available_skills": []},
        plugin_components=[
            {"name": "bundle:review", "plugin_available": True, "plugin_kind": "skill"},
            {
                "name": "bundle:fetch",
                "plugin_available": False,
                "plugin_selected": True,
            },
            {
                "name": "bundle:format",
                "plugin_available": False,
                "plugin_selected": False,
            },
        ],
    )
    assert "bundle:review: usage unknown" in report
    assert "Plugin components with unmet requirements: bundle:fetch" in report
    assert "Disabled plugin components: bundle:format" in report
    assert "Plugin inventory unavailable" in format_skill_cost_report(
        session(), {}, plugin_components=None
    )
