"""Source hooks need complete qualification; names and parser success grant nothing."""

import json

import pytest

from tldw_chatbook.Plugins.adapters.vendor_hooks import normalize_vendor_hook
from tldw_chatbook.Plugins.inspection import inspect_package

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.parametrize(
    "dialect,event",
    [
        ("codex-hooks/2026-10-01", "PreToolUse"),
        ("cursor-hooks/1@2026-10-01", "preToolUse"),
    ],
)
def test_simple_source_hook_never_borrows_native_payload_output_or_timeout(
    dialect, event
):
    value = normalize_vendor_hook(
        {"event": event, "command": 'python3 "./safe path/guard.py"', "timeout": 5},
        dialect,
    )
    assert value["proposed_argv"] == ["python3", "./safe path/guard.py"]
    assert value["target_event"] == "PreToolUse"
    assert value["guard_scope_unknown"]
    assert not value["qualification"]["payload"]
    assert not value["qualification"]["input_output"]
    assert not value["qualification"]["cwd"]
    assert not value["qualification"]["timeout"]


@pytest.mark.parametrize(
    "command",
    [
        "python3 guard.py && echo ok",
        "python3 $(pwd)/guard.py",
        "python3 guard.py | tee log",
        "python3 `pwd`/guard.py",
    ],
)
def test_source_shell_syntax_is_never_executed(command):
    value = normalize_vendor_hook(
        {"event": "PreToolUse", "command": command}, "codex-hooks/2026-10-01"
    )
    assert value["proposed_argv"] is None
    assert value["reason"] == "vendor_hook_shell_syntax_unsupported"


def test_regex_and_success_only_source_timing_stay_unqualified():
    value = normalize_vendor_hook(
        {
            "event": "postToolUse",
            "command": "python3 observe.py",
            "matcher": "Shell|Write",
        },
        "cursor-hooks/1@2026-10-01",
    )
    assert value["reason"] == "vendor_hook_regex_unqualified"
    assert value["target_event"] == "PostToolUse"
    assert not value["qualification"]["timing"]
    assert not value["guard_scope_unknown"]


def test_guard_blocks_scope_and_optional_observer_remains_visible(interop_package):
    guard = inspect_package(interop_package("cursor-guard"))
    assert "vendor_guard_scope_unknown" in guard.activation_blockers
    assert guard.inventory["skill:review"].activation_blockers
    observer = inspect_package(interop_package("cursor-observer"))
    assert not observer.activation_blockers
    assert not observer.inventory["skill:review"].activation_blockers
    (hook,) = [row for row in observer.inventory.values() if row.kind == "hook"]
    assert hook.support == "unsupported"
    assert not json.loads(hook.definition_json)["qualification"]["input_output"]


@pytest.mark.asyncio
async def test_actual_console_does_not_expand_or_execute_unqualified_guarded_material(
    native_console, interop_package
):
    rig = native_console
    root = interop_package("cursor-guard")
    review = await rig.service.review_install(
        root, selection=("skill:review",), workspace_id="workspace-a"
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enable = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enable, enable.operation_id)
    (row,) = [
        row
        for row in rig.service.list_components("workspace-a")
        if row["plugin_component_id"] == "skill:review"
    ]
    assert not row["plugin_available"] and row["plugin_blockers"]
    result = await rig.controller.submit_draft(
        "$cursor-guard:review", session_id=rig.session.id
    )
    assert result.accepted  # Unrelated ordinary messages remain available.
    assert "GUARDED_BODY" not in str(rig.gateway.payloads[-1])
    assert "$cursor-guard:review" in str(rig.gateway.payloads[-1])
    # A foreign allow/output path cannot be reached through an unavailable hook.
    assert all(
        row["plugin_kind"] != "hook"
        for row in rig.service.capture_maximum("workspace-a")["available_skills"]
    )


@pytest.mark.parametrize("location", ["root", "group", "handler"])
def test_observer_cannot_hide_required_or_unknown_guard_constraints(location):
    from tldw_chatbook.Plugins.adapters.vendor_hooks import inventory_vendor_hooks

    handler = {"command": "python3 observe.py"}
    group = {"hooks": [handler]}
    value = {"hooks": {"PostToolUse": [group]}}
    if location == "root":
        value["required"] = True
    elif location == "group":
        group["failClosed"] = True
    else:
        handler["unknownGuardPolicy"] = {"required": True}
    records, guarded = inventory_vendor_hooks(value, "hooks.json", "openai")
    assert guarded
    assert all(row.support == "unsupported" for row in records.values())
