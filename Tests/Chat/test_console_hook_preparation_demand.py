"""Demand-driven hook setup retains real permission and owner boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import toml

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
hook_file = _hook_file


def _save_hooks(path, section):
    raw = toml.loads(path.read_text())
    raw["hooks"] = section
    path.write_text(dumps_cli_config(raw))


@pytest.fixture
async def hook_case(hook_file, monkeypatch):
    _save_hooks(hook_file, {})
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    runtime = ConsoleRuntime(app=None)
    owner = runtime.ensure_hook_permissions()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=None,
        hook_permissions_accessor=runtime.ensure_hook_permissions,
    )
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    reviews = []
    contexts = []
    original_review = owner.v2_configuration
    original_context = runtime._hooks_v2_context_key

    def read_review():
        result = original_review()
        reviews.append(result)
        return result

    def read_context(session_id):
        result = original_context(session_id)
        contexts.append((session_id, result))
        return result

    monkeypatch.setattr(owner, "v2_configuration", read_review)
    monkeypatch.setattr(runtime, "_hooks_v2_context_key", read_context)
    case = SimpleNamespace(
        runtime=runtime,
        store=store,
        session=session,
        owner=owner,
        path=hook_file,
        reviews=reviews,
        contexts=contexts,
        detached_engines=[],
    )
    try:
        yield case
    finally:
        for engine in case.detached_engines:
            await engine.close()
        await runtime.close_hooks_v2()
        await runtime.dispose()


async def test_cold_empty_hooks_reconcile_without_workspace_context(hook_case):
    case = hook_case

    assert await case.runtime.prepare_hooks_v2(case.session.id) is None

    assert len(case.reviews) == 1
    review, targets = case.reviews[0]
    assert review.ready
    assert targets == ()
    assert case.runtime.get_hooks_v2(case.session.id) is None
    assert case.session.id not in case.runtime._hooks_v2_lifecycles
    assert case.session.id not in case.runtime._hooks_v2_configured
    assert case.contexts == [], "No hook owner consumes this workspace context."


@pytest.mark.parametrize(
    "retained", ["engine", "closed_engine", "lifecycle", "configured"]
)
async def test_retained_hook_owner_still_reads_and_validates_context(
    hook_case, monkeypatch, retained
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    case = hook_case
    session_id = case.session.id
    if retained == "configured":
        case.runtime._hooks_v2_configured[session_id] = (
            load_hooks_config({"hooks": {}}),
            (),
            None,
        )
    else:
        engine = case.runtime.ensure_hooks_v2(session_id, (), lambda *_: True)
        if retained == "closed_engine":
            await engine.close()
        elif retained == "lifecycle":
            lifecycle = HookSessionLifecycle(engine, session_id)
            case.runtime._hooks_v2_lifecycles[session_id] = lifecycle
            case.runtime._hooks_v2_engines.pop(session_id)
            case.detached_engines.append(engine)

    original_context = case.runtime._hooks_v2_context_key

    def close_after_context(current_session_id):
        result = original_context(current_session_id)
        case.runtime._admission_fenced_sessions.add(current_session_id)
        return result

    monkeypatch.setattr(case.runtime, "_hooks_v2_context_key", close_after_context)

    with pytest.raises(RuntimeError):
        await case.runtime.prepare_hooks_v2(session_id)

    assert len(case.reviews) == 1
    assert case.reviews[0][0].ready
    assert len(case.contexts) == 1
    assert case.contexts[0][0] == session_id
    assert session_id in case.runtime._admission_fenced_sessions


@pytest.mark.parametrize("kind", ["unapproved", "invalid"])
async def test_unready_hook_configuration_keeps_original_refusal(hook_case, kind):
    case = hook_case
    handler = {
        "id": "pending",
        "event": "SessionStart",
        "type": "mcp_tool",
        "server": "local:fixture",
        "tool": "fixture",
        "effects": [],
    }
    if kind == "invalid":
        handler["type"] = "unsupported"
    _save_hooks(case.path, {"handler": [handler]})

    with pytest.raises(RuntimeError, match="Review enabled hooks before execution"):
        await case.runtime.prepare_hooks_v2(case.session.id)

    assert len(case.reviews) == 1
    assert not case.reviews[0][0].ready
    assert len(case.contexts) == 1
    assert case.runtime.get_hooks_v2(case.session.id) is None
    assert case.session.id not in case.runtime._hooks_v2_lifecycles


async def test_cold_empty_reconciliation_cannot_return_after_session_fence(
    hook_case, monkeypatch
):
    case = hook_case
    original_review = case.owner.v2_configuration

    def close_after_review():
        result = original_review()
        case.runtime._admission_fenced_sessions.add(case.session.id)
        return result

    monkeypatch.setattr(case.owner, "v2_configuration", close_after_review)

    with pytest.raises(RuntimeError):
        await case.runtime.prepare_hooks_v2(case.session.id)

    assert len(case.reviews) == 1
    assert case.reviews[0][0].ready
    assert case.contexts == []
    assert case.runtime.get_hooks_v2(case.session.id) is None
    assert case.session.id not in case.runtime._hooks_v2_lifecycles


@pytest.mark.parametrize("has_definitions", [False, True], ids=["empty", "definitions"])
async def test_native_plugin_capture_precedes_hook_absence_decision(
    hook_case, monkeypatch, has_definitions
):
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = hook_case
    maximum = {"available_skills": [{"plugin_owned": True}]}
    configuration = SimpleNamespace(skill_context_maximum=maximum)
    definitions = (
        parse_handlers(
            [
                {
                    "id": "native-stop",
                    "event": "Stop",
                    "type": "mcp_tool",
                    "server": "local:fixture",
                    "tool": "fixture",
                    "effects": [],
                }
            ]
        )
        if has_definitions
        else ()
    )
    native = SimpleNamespace(signature=("native",), definitions=definitions)
    captures = []

    async def hook_configuration(captured_maximum):
        captures.append(captured_maximum)
        return native

    # Only the plugin-service boundary is substituted. Permission reconciliation
    # and workspace capture still execute their original bodies.
    service = SimpleNamespace(hook_configuration=hook_configuration)
    skills = SimpleNamespace(local_service=SimpleNamespace(plugin_service=service))
    original_context = case.runtime._hooks_v2_context_key

    def close_after_context(session_id):
        result = original_context(session_id)
        case.runtime._admission_fenced_sessions.add(session_id)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(case.runtime._chat_controller, "_skills_service", skills)
        patch.setattr(case.runtime, "_hooks_v2_context_key", close_after_context)
        if has_definitions:
            # Fence after actual context capture to avoid starting a hook engine.
            # An incorrect absence return would neither capture nor refuse.
            with pytest.raises(RuntimeError):
                await case.runtime.prepare_hooks_v2(
                    case.session.id, configuration=configuration
                )
        else:
            assert (
                await case.runtime.prepare_hooks_v2(
                    case.session.id, configuration=configuration
                )
                is None
            )

    assert len(captures) == 1
    assert captures[0] is maximum
    assert len(case.reviews) == 1
    assert case.reviews[0][0].ready
    assert len(case.contexts) == int(has_definitions)
    assert case.runtime.get_hooks_v2(case.session.id) is None
    assert case.session.id not in case.runtime._hooks_v2_lifecycles
    assert case.session.id not in case.runtime._hooks_v2_configured
