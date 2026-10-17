"""Real bounded child execution and producer semantics for vendor hook subsets."""

import json
import shlex

import pytest

from Tests.Agents.test_hooks_v2_execution import event
from Tests.hooks_v2_process_support import child_argv
from Tests.Plugins import platform_support
from tldw_chatbook.Agents.hooks_v2 import parse_handlers
from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
from tldw_chatbook.Agents.hooks_v2.engine import HookEngine


def vendor_command(
    tmp_path, code, *, dialect="cursor/1", source="preToolUse", **fields
):
    from tldw_chatbook.Plugins.adapters.vendor_hooks import normalize_vendor_hook

    argv = child_argv(code)
    script = tmp_path / "controlled.py"
    script.write_text(argv[-1])
    return normalize_vendor_hook(
        {
            "id": "vendor",
            "event": source,
            "command": shlex.join([argv[0], "-I", str(script)]),
            **fields,
        },
        dialect,
    )


def tool_event(*, name="PreToolUse", provider="mcp", status=None):
    base = {
        key: value for key, value in event().model_dump().items() if value is not None
    }
    base["event"] = name
    base["data"] = {
        "tool_id": "mcp:fixture:lookup",
        "tool_name": "lookup",
        "provider": provider,
        "operation": "mcp.call" if provider in {"mcp", "plugin_mcp"} else "file.read",
        "tool_args": {"query": "original"},
        "candidate_arguments": {"query": "candidate"},
    }
    if name != "PreToolUse":
        base["data"].pop("candidate_arguments")
    if name == "PostToolUseFailure":
        base["data"]["reason"] = "tool_error"
    if status is not None:
        base["data"].update(
            status=status, is_error=status != "success", result={"text": "result"}
        )
    from tldw_chatbook.Agents.hooks_v2 import parse_event

    return parse_event(base)


@platform_support.v2_command
@pytest.mark.asyncio
async def test_native_command_positive_control(tmp_path):
    handler = parse_handlers(
        [
            {
                "id": "native",
                "event": "PreToolUse",
                "type": "command",
                "effects": ["deny"],
                "argv": child_argv('print(\'{"version":2,"decision":"deny"}\')'),
            }
        ]
    )[0]
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        outcome = await engine.fire_async(tool_event())
        assert not outcome.allowed
        assert outcome.accepted[0][1].decision == "deny"
        assert not engine.processes.records
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dialect, source, response",
    [
        ("cursor/1", "preToolUse", {"permission": "deny"}),
        (
            "codex/1",
            "PreToolUse",
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                }
            },
        ),
    ],
)
async def test_vendor_denial_and_payload_use_existing_command_owner(
    tmp_path, dialect, source, response
):
    marker = tmp_path / "observed.json"
    code = f"import json,sys,pathlib;value=json.load(sys.stdin);pathlib.Path({str(marker)!r}).write_text(json.dumps(value));print({json.dumps(response)!r})"
    handler = vendor_command(tmp_path, code, dialect=dialect, source=source)
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        outcome = await engine.fire_async(tool_event())
        assert not outcome.allowed
        assert outcome.accepted[0][1].decision == "deny"
        payload = json.loads(marker.read_text())
        assert payload["tool_input"] == {"query": "candidate"}
        assert payload["tool_name"] == "lookup"
        assert payload["session_id"] == "runtime-a"
        assert not engine.processes.records
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["mcp", "plugin_mcp"])
async def test_before_mcp_payload_is_string_and_skips_other_providers(
    tmp_path, provider
):
    marker = tmp_path / "observed.json"
    handler = vendor_command(
        tmp_path,
        f'import json,sys,pathlib;value=json.load(sys.stdin);pathlib.Path({str(marker)!r}).write_text(json.dumps(value));print(\'{{"permission":"allow"}}\')',
        source="beforeMCPExecution",
    )
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        assert (await engine.fire_async(tool_event(provider="local"))).allowed
        assert not marker.exists()
        assert (await engine.fire_async(tool_event(provider=provider))).allowed
        payload = json.loads(marker.read_text())
        assert isinstance(payload["tool_input"], str)
        assert json.loads(payload["tool_input"]) == {"query": "candidate"}
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
async def test_cursor_post_success_only(tmp_path):
    marker = tmp_path / "observed"
    handler = vendor_command(
        tmp_path,
        f"from pathlib import Path;Path({str(marker)!r}).write_text('called');print('{{}}')",
        source="postToolUse",
    )
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        assert (
            await engine.fire_async(tool_event(name="PostToolUse", status="failed"))
        ).allowed
        assert not marker.exists()
        assert (
            await engine.fire_async(tool_event(name="PostToolUse", status="success"))
        ).allowed
        assert marker.read_text() == "called"
    finally:
        await engine.close()


@pytest.mark.parametrize(
    "command",
    [
        "echo hi | cat",
        "echo $(whoami)",
        "echo `whoami`",
        "echo hi > output",
        "echo one; echo two",
        "VAR=x command",
    ],
)
def test_shell_forms_are_refused(command):
    from tldw_chatbook.Plugins.adapters.vendor_hooks import normalize_vendor_hook

    with pytest.raises(ValueError):
        normalize_vendor_hook(
            {"id": "hook", "event": "preToolUse", "command": command}, "cursor/1"
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"matcher": "Shell|Read"},
        {"event": "beforeShellExecution"},
        {"event": "workspaceOpen"},
        {"type": "prompt"},
    ],
)
def test_unqualified_source_behavior_is_refused(fields):
    from tldw_chatbook.Plugins.adapters.vendor_hooks import normalize_vendor_hook

    with pytest.raises(ValueError):
        normalize_vendor_hook(
            {
                "id": "hook",
                "event": "preToolUse",
                "command": "python hook.py",
                **fields,
            },
            "cursor/1",
        )


@platform_support.vendor_command
@pytest.mark.parametrize(
    "source, final_guard", [("beforeMCPExecution", True), ("preToolUse", False)]
)
def test_source_guards_cannot_be_removed_by_component_selection(
    native_package, source, final_guard
):
    from Tests.Plugins.test_codex_adapter import write_json
    from tldw_chatbook.Plugins.inspection import inspect_package

    root = native_package()
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {
            "name": "guarded",
            "hooks": {
                "version": 1,
                "hooks": {source: [{"command": "python guard.py"}]},
            },
        },
    )
    result = inspect_package(root, dialect="cursor")
    guard = next(key for key in result.inventory if key.startswith("hook:"))
    assert result.inventory["skill:review"].dependencies == (guard,)
    assert (
        f"dependency_guard_unqualified:{guard}"
        in result.inventory["skill:review"].activation_blockers
    ) is (not final_guard)


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected",
    [
        ({"permission": "allow"}, "pass"),
        ({"permission": "ask"}, "refused"),
        ({"permission": "deny"}, "deny"),
        ({"permission": "allow", "updated_input": {"query": "changed"}}, "updated"),
    ],
)
async def test_foreign_permission_results_never_create_grants(
    tmp_path, response, expected
):
    handler = vendor_command(tmp_path, f"print({json.dumps(response)!r})")
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        result = await engine.fire_async(tool_event())
        if expected == "refused":
            assert not result.allowed and not result.accepted
        else:
            assert result.accepted[0][1].decision == (
                "deny" if expected == "deny" else "pass"
            )
            if expected == "updated":
                assert dict(result.accepted[0][1].updated_input) == {"query": "changed"}
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, stage, failure",
    [
        ("raise SystemExit(2)", "normal", None),
        ("raise SystemExit(2)", "reject_accept", "authority_refused"),
        (
            "import os;os.write(1,b'x'*16385);raise SystemExit(2)",
            "normal",
            "stdout_overflow",
        ),
        ("import time;time.sleep(30)", "normal", "timeout"),
        ("print('not JSON')", "normal", "invalid_output"),
    ],
)
async def test_vendor_exit_and_timeout_stay_under_native_acceptance(
    tmp_path, code, stage, failure
):
    import time

    handler = vendor_command(
        tmp_path, code, timeout=0.25 if failure == "timeout" else 10
    )
    engine = HookEngine(
        (handler,),
        lambda _h, _e, at: not (stage == "reject_accept" and at == "accept"),
        HookBudgetOwner(),
    )
    try:
        started = time.monotonic()
        result = await engine.fire_async(tool_event())
        assert not result.allowed
        if failure == "timeout":
            assert time.monotonic() - started >= 0.25
        if failure:
            assert result.failures[0].code in (
                {"timeout", "cancelled"} if failure == "timeout" else {failure}
            )
            assert not result.accepted
        else:
            assert result.accepted[0][1].decision == "deny"
        assert not engine.processes.records
        assert engine.budget_owner.snapshot()["tickets"] == 0
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
async def test_vendor_transform_runs_once_then_native_final_guard(tmp_path):
    from dataclasses import replace

    from Tests.Agents.test_agent_runtime import CALC
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import prepare_tool
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    calls = tmp_path / "calls"
    transform = vendor_command(
        tmp_path,
        f'from pathlib import Path;Path({str(calls)!r}).open(\'a\').write(\'T\');print(\'{{"permission":"allow","updated_input":{{"expression":"2+2"}}}}\')',
    )
    guard = parse_handlers(
        [
            {
                "id": "final",
                "event": "PreToolUse",
                "type": "command",
                "effects": ["deny"],
                "argv": child_argv(
                    "import json,sys;v=json.load(sys.stdin);assert v['data']['tool_args']=={'expression':'2+2'};print('{\"version\":2,\"decision\":\"deny\"}')"
                ),
            }
        ]
    )[0]
    original = tool_event(provider="local").model_dump()
    original = {k: v for k, v in original.items() if v is not None}
    original["data"].update(
        tool_id="local:test", tool_args={"expression": "1+1"}, tool_name="calculator"
    )
    from tldw_chatbook.Agents.hooks_v2 import parse_event
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import HookPreparationError

    engine = HookEngine((guard, transform), lambda *_: True, HookBudgetOwner())
    definition = ToolDefinitionSnapshot.from_schema(
        replace(CALC, id="local:test"), "local", 1, 1
    )
    try:
        with pytest.raises(HookPreparationError):
            await prepare_tool(parse_event(original), engine, definition=definition)
        assert calls.read_text() == "T"
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
async def test_admitted_vendor_hook_uses_original_native_process_owner_and_root(
    native_console, native_package, tmp_path
):
    import sys

    from Tests.Plugins.test_codex_adapter import write_json
    from Tests.Plugins.test_native_components import install_components
    from tldw_chatbook.Plugins.components import NativeHookProcessOwner, NativeHookSet

    package = native_package()
    (package / "plugin.json").unlink()
    marker = tmp_path / "native-observed.json"
    argv = child_argv(
        f"import os,sys,json;from pathlib import Path;value=json.load(sys.stdin);assert sys.argv[1]==os.getcwd()==os.environ['PLUGIN_ROOT'];assert value['cwd']==os.getcwd();assert sys.argv[2]=='literal ${{OTHER}}';Path({str(marker)!r}).write_text(json.dumps({{'cwd':os.getcwd(),'root':os.environ['PLUGIN_ROOT'],'args':value['tool_input']}}));print('{{\"permission\":\"deny\"}}')"
    )
    (package / "controlled.py").write_text(argv[-1])
    write_json(
        package,
        ".cursor-plugin/plugin.json",
        {
            "name": "owned-vendor",
            "variables": {
                "type": "object",
                "properties": {
                    "EXAMPLE": {"type": "string", "default": "literal ${OTHER}"}
                },
            },
            "hooks": {
                "version": 1,
                "hooks": {
                    "beforeMCPExecution": [
                        {
                            "command": shlex.join(
                                [
                                    sys.executable,
                                    "-I",
                                    "${CURSOR_PLUGIN_ROOT}/controlled.py",
                                    "${CURSOR_PLUGIN_ROOT}",
                                    "${EXAMPLE}",
                                ]
                            )
                        }
                    ]
                },
            },
        },
    )
    hook_id = "hook:cursor-0-0-0-beforeMCPExecution"
    installed = await install_components(
        native_console.service, package, ("skill:review", hook_id)
    )
    snapshot = await native_console.service.capture_mcp_snapshot(
        installed.installation_id, "workspace-a", "vendor-owned"
    )
    assert "skill:review" in snapshot.selection
    hooks = NativeHookSet(native_console.service, (snapshot,))
    owners = []

    def retain_owner(handler, occurrence, cancel):
        owner = hooks.process_owner(handler, occurrence, cancel)
        owners.append(owner)
        return owner

    engine = HookEngine(
        hooks.definitions,
        hooks.current,
        HookBudgetOwner(),
        process_owner_for=retain_owner,
        host_environment=hooks.host_environment,
        command_launch=hooks.command_launch,
        dependency_required=lambda handler, _event: handler.id in hooks.requirements,
    )
    try:
        result = await engine.fire_async(tool_event())
        assert result.accepted[0][1].decision == "deny"
        assert len(owners) == 1 and type(owners[0]) is NativeHookProcessOwner
        assert owners[0].snapshot is snapshot
        assert (
            json.loads(marker.read_text())["root"]
            == snapshot.inspection.materialized_identity
        )
        assert not engine.processes.records
        assert not native_console.service.live_runs()
        engine.enabled = False
        refused = await engine.fire_async(tool_event())
        assert not refused.allowed and not refused.accepted
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, selected, admitted",
    [
        ("beforeMCPExecution", False, False),
        ("beforeMCPExecution", True, True),
        ("preToolUse", True, False),
    ],
)
async def test_actual_admission_keeps_guard_selection_and_final_phase_constraints(
    native_console, native_package, source, selected, admitted
):
    from Tests.Plugins.test_codex_adapter import write_json
    from Tests.Plugins.test_native_components import install_components

    package = native_package()
    (package / "plugin.json").unlink()
    write_json(
        package,
        ".cursor-plugin/plugin.json",
        {
            "name": "guard-selection",
            "hooks": {
                "version": 1,
                "hooks": {source: [{"command": "python guard.py"}]},
            },
        },
    )
    hook_id = f"hook:cursor-0-0-0-{source}"
    selection = ("skill:review", hook_id) if selected else ("skill:review",)
    installed = await install_components(native_console.service, package, selection)
    if not selected:
        from tldw_chatbook.Plugins.admission import PluginUnavailable

        with pytest.raises(PluginUnavailable):
            await native_console.service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "guard-selection"
            )
    else:
        snapshot = await native_console.service.capture_mcp_snapshot(
            installed.installation_id, "workspace-a", "guard-selection"
        )
        assert ("skill:review" in snapshot.selection) is admitted
    assert (
        bool(native_console.service.capture_maximum("workspace-a")["available_skills"])
        is admitted
    )


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
async def test_vendor_allow_still_reaches_normal_permission_review(tmp_path, allowed):
    import asyncio
    from dataclasses import replace

    from Tests.Agents.test_agent_runtime import CFG, make_deps
    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.agent_models import ModelTurn, ToolCall, ToolResult
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    registry = ToolCatalogRegistry()
    dispatched, reviewed = [], []
    registry.register_provider(
        _Provider(lambda: dispatched.append(1) or ToolResult(ok=True, content="ok"))
    )
    handler = vendor_command(
        tmp_path,
        'print(\'{"permission":"allow","updated_input":{"review":"changed"}}\')',
    )
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda call: registry.snapshot_for_hook(call.name),
    )
    deps = make_deps(
        [ModelTurn(tool_calls=(ToolCall("probe", {}, "call"),)), ModelTurn(text="done")]
    )
    deps.prepare_hook_call = owner.prepare_call
    deps.validate_hook_dispatch = owner.validate_dispatch
    deps.install_tool_checkpoint = owner.install_result
    deps.await_hook_checkpoints = owner.admit_input
    deps.invoke_tool = lambda call: registry.invoke_by_name(call.name, call.args)

    def review(calls):
        reviewed.append(dict(calls[0].args))
        return {} if allowed else {calls[0].call_id: "normal permission refusal"}

    deps.review_tool_calls = review
    try:
        await asyncio.to_thread(
            run_agent_loop,
            replace(CFG, allowed_tools=("probe",)),
            [{"role": "user", "content": "go"}],
            [registry.load_schema("test:probe")],
            deps,
        )
        assert reviewed == [{"review": "changed"}]
        assert bool(dispatched) is allowed
    finally:
        await engine.close()


@platform_support.vendor_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dialect, source, host, response, denied",
    [
        (
            "cursor/1",
            "beforeSubmitPrompt",
            "UserPromptSubmit",
            {"continue": False},
            True,
        ),
        (
            "codex/1",
            "UserPromptSubmit",
            "UserPromptSubmit",
            {"decision": "block"},
            True,
        ),
        (
            "codex/1",
            "PostToolUse",
            "PostToolUse",
            {"hookSpecificOutput": {"additionalContext": "reviewed"}},
            False,
        ),
        (
            "cursor/1",
            "postToolUseFailure",
            "PostToolUseFailure",
            {"additional_context": "reviewed"},
            False,
        ),
    ],
)
async def test_qualified_event_payload_and_result_pairs(
    tmp_path, dialect, source, host, response, denied
):
    marker = tmp_path / "payload.json"
    handler = vendor_command(
        tmp_path,
        f"import json,sys;from pathlib import Path;value=json.load(sys.stdin);Path({str(marker)!r}).write_text(json.dumps(value));print({json.dumps(response)!r})",
        dialect=dialect,
        source=source,
    )
    if host == "UserPromptSubmit":
        from tldw_chatbook.Agents.hooks_v2 import parse_event

        raw = {k: v for k, v in event().model_dump().items() if v is not None}
        raw.update(event=host, data={"prompt": "exact untrusted prompt"})
        occurrence = parse_event(raw)
    else:
        occurrence = tool_event(name=host, status="failed")
    engine = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    try:
        outcome = await engine.fire_async(occurrence)
        assert bool(outcome.accepted)
        assert outcome.allowed is not denied
        payload = json.loads(marker.read_text())
        assert payload["hook_event_name"] == source
        if host == "UserPromptSubmit":
            assert payload["prompt"] == "exact untrusted prompt"
        else:
            assert payload["tool_response"] == {"text": "result"}
            assert outcome.accepted[0][1].context[0].text == "reviewed"
    finally:
        await engine.close()


@pytest.mark.parametrize(
    "change",
    [
        {"vendor_protocol": "cursor/2:preToolUse"},
        {"event": "SessionEnd"},
        {"effects": ["deny"]},
        {"type": "mcp_tool", "server": "server", "tool": "tool"},
    ],
)
def test_protocol_marker_cannot_change_type_event_or_effect_contract(change):
    raw = {
        "id": "hook",
        "event": "PreToolUse",
        "type": "command",
        "effects": ["deny", "updated_input"],
        "argv": ["python", "hook.py"],
        "vendor_protocol": "cursor/1:preToolUse",
    }
    assert parse_handlers([raw])[0].event == "PreToolUse"
    with pytest.raises(ValueError):
        parse_handlers([{**raw, **change}])


@platform_support.vendor_command
def test_unknown_variable_in_hook_blocks_its_native_dependents(native_package):
    from Tests.Plugins.test_codex_adapter import write_json
    from tldw_chatbook.Plugins.inspection import inspect_package

    root = native_package()
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {
            "name": "test",
            "hooks": {
                "version": 1,
                "hooks": {
                    "beforeMCPExecution": [{"command": "python guard.py ${MISSING}"}]
                },
            },
        },
    )
    result = inspect_package(root, dialect="cursor")
    hook = result.inventory["hook:cursor-0-0-0-beforeMCPExecution"]
    assert "variable_undeclared:MISSING" in hook.activation_blockers
    assert (
        f"dependency_blocked:{hook.component_id}"
        in result.inventory["skill:review"].activation_blockers
    )


@platform_support.package_capture
def test_malformed_unqualified_hook_fields_are_blocked_without_parser_crash(
    native_package,
):
    from Tests.Plugins.test_codex_adapter import write_json
    from tldw_chatbook.Plugins.inspection import inspect_package

    root = native_package()
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {
            "name": "test",
            "hooks": {
                "version": 1,
                "hooks": {
                    "preToolUse": [
                        {
                            "command": "python guard.py",
                            "effects": None,
                            "argv": None,
                            "env": None,
                        }
                    ]
                },
            },
        },
    )
    result = inspect_package(root, dialect="cursor")
    assert result.inventory["skill:review"].activation_blockers


@pytest.mark.skipif(
    platform_support.VENDOR_COMMAND, reason="native unsupported vendor control"
)
def test_native_unsupported_vendor_command_refuses():
    from tldw_chatbook.Plugins.adapters.vendor_hooks import normalize_vendor_hook

    with pytest.raises(ValueError, match="platform"):
        normalize_vendor_hook(
            {"id": "guard", "event": "preToolUse", "command": "python hook.py"},
            "cursor/1",
        )
