"""Later native requests discard earlier transcript and continuation payloads."""

import json

import pytest

from Tests.Agents.test_goal_iteration_report import report
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture


@pytest.mark.asyncio
async def test_second_and_third_actual_requests_have_only_bounded_recent_memory(
    stores, monkeypatch
):
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": report(
                            summary=f"iteration-{count}",
                            candidate_draft=f"draft-{count}",
                            learnings=["é" * 12000],
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    import Tests.Chat.test_console_goal_dispatch as dispatch_module

    original_resolution = dispatch_module.resolution
    monkeypatch.setattr(
        dispatch_module,
        "resolution",
        lambda: original_resolution(
            provider="moonshot",
            execution_key="moonshot",
            model="kimi-k2",
            base_url="https://api.moonshot.ai/v1",
        ),
    )

    goal, _store, _session, _controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        for _ in range(3):
            result = await coordinator.dispatch_once(goal.id)
            checkpoint = coordinator.service.checkpoint(result)
            if result.ordinal == 1:
                from tldw_chatbook.Agents.agent_models import (
                    ContinuationEventContext,
                    FinalContinuation,
                )
                from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
                from tldw_chatbook.Chat.provider_continuation import (
                    ContinuationRound,
                    ProviderContinuationCheckpoint,
                )

                old = _store.append_message(
                    _session.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content="",
                    persist=True,
                )
                prior = ProviderContinuationCheckpoint(
                    schema_version=1,
                    checkpoint_revision=1,
                    provider=goal.request.provider.provider,
                    protocol="chat_completions",
                    model=goal.request.provider.model,
                    api_base_url="https://api.moonshot.ai/v1",
                    state="complete",
                    rounds=(
                        ContinuationRound(
                            assistant_content="OLD TRANSCRIPT PAYLOAD",
                            reasoning_blocks=("PRIOR CONTINUATION PRIVATE CANARY",),
                            calls=(),
                        ),
                    ),
                )
                _store.persist_provider_continuation_event(
                    FinalContinuation(
                        ContinuationEventContext(
                            old.id, result.native_run_id, "primary", "persistent"
                        ),
                        prior,
                        None,
                        "OLD TRANSCRIPT PAYLOAD",
                    )
                )
                assert (
                    "PRIOR CONTINUATION PRIVATE CANARY"
                    in stores[1].db.get_message_by_id(old.id)[
                        "provider_continuation_json"
                    ]
                )
        assert "PRIOR CONTINUATION PRIVATE CANARY" not in str(calls)
        assert "OLD TRANSCRIPT PAYLOAD" not in str(calls)
        assert len(calls) == 3
        assert checkpoint.iteration_count == 3
        for call in calls:
            assert goal.request.objective in str(call)
            assert goal.request.criteria in str(call)
        assert "iteration-1" not in str(calls[2])
        assert "iteration-2" in str(calls[2])
        assert "é" * 12000 not in str(calls[1])
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_repeated_real_fs_read_is_addressable_but_not_new_progress(
    stores, monkeypatch
):
    import re
    from pathlib import Path

    runs, persistence, registry, req = stores
    from tldw_chatbook.Agents.goal_models import GoalToolScope

    req = req.model_copy(
        update={"tool_scope": GoalToolScope(catalog_tools=("local:fs_read",))}
    )
    Path(req.binding.locator, "input.txt").write_text("unchanged source")
    call_number = 0
    ids = []

    def provider(**kwargs):
        nonlocal call_number
        call_number += 1
        if call_number % 2:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"read-{call_number}",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "input.txt"}),
                        },
                    }
                ],
            }
        else:
            found = re.findall(r"goal_evidence_id: ([a-f0-9]{32})", str(kwargs))
            ids.extend(found)
            message = {"content": report(evidence_ids=found[:1])}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _, _, controller, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService

    controller.app = SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    setting = controller_module.get_cli_setting
    monkeypatch.setattr(
        controller_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if (section, key) == ("console", "local_tools_enabled")
            else setting(section, key, default)
        ),
    )
    try:
        for _ in range(3):
            result = await coordinator.dispatch_once(goal.id)
            saved = coordinator.service.checkpoint(result)
        assert len(ids) == 3 and len(set(ids)) == 3
        assert len(calls) == 6
        assert saved.status == "paused"
        assert saved.checkpoints[-1].decision.no_progress_count == 2
        assert not any(c.satisfied for c in saved.checkpoints[-1].decision.checks)
        assert all(
            "unchanged source" in e.stdout for e in runs.goal_runs.evidence(goal.id)
        )
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_mandatory_goal_request_over_budget_never_dispatches_or_truncates_objective(
    stores, monkeypatch
):
    from tldw_chatbook.Agents.goal_models import GoalPolicy

    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={"objective": "é" * 4096, "policy": GoalPolicy(budget_tokens=1)}
    )
    goal, _, _, _, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        saved = coordinator.service.checkpoint(result)
        assert calls == []
        assert saved.request.objective == "é" * 4096
        assert saved.status in ("paused", "recovery_required"), result
        assert saved.accounting.used["generation"] == 1
    finally:
        await gateway.aclose()
