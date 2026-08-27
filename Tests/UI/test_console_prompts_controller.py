"""Characterisation + contract tests for the Console prompt cluster.

Written BEFORE the wave-3 task-3 extraction of `ConsolePromptsController`
out of `ChatScreen`, so every assertion below is a statement about
behaviour that must survive the move unchanged:

* the real `_open_console_prompts_modal` path (a live screen, a live chat
  store, a real `ConsolePromptsModal` on the stack) and the SELECTION it
  persists -- `apply_improvement_result` writing the reviewed System
  prompt through `ConsoleChatStore.set_session_system_prompt`, plus its
  stale-guard refusing when the live System prompt moved underneath it;
* `/prompt <name>` and `/system <name>` name resolution, including their
  shared refuse-a-Recipe guard;
* the `/system` editor's save-to-Library outcome copy;
* the Library "Use in Console" staged-insert handoff;
* the lazily-created shared prompt-history store.

The final section pins the extraction's own contract: the controller is
wired in `ChatScreen.__init__`, owns zero DOM, and every screen-level name
a staying caller or a pre-existing test reaches by is still a real
delegation onto it.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    fingerprint_system_text,
)
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsResult
from Tests.console_provider_doubles import with_destination
from Tests.UI.app_factory import attach_chachanotes_db


class ConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    """A send-ready Console so the blocking first-run setup modal stays hidden."""
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": model}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": model}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


def _prompt_record(
    *,
    name: str = "Summarize",
    artifact_type: str = "prompt",
    system_prompt: str = "You are terse.",
    user_prompt: str = "Summarize the following.",
) -> dict[str, object]:
    return {
        "id": "local:prompt:11",
        "local_id": 11,
        "source_id": "11",
        "name": name,
        "artifact_type": artifact_type,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "version": 1,
    }


def _replacement_application(
    *,
    snapshot,
    session_id: str,
    user_text: str | None,
    system_text: str | None = None,
    system_fingerprint: str | None = None,
    created_monotonic: float | None = None,
) -> PromptVariableApplication:
    values = {
        "system_text": system_text,
        "user_text": user_text,
        "apply_system": system_text is not None,
        "apply_user": user_text is not None,
        "destination": "replace_snapshot",
        "target_session_id": session_id,
        "composer_fingerprint": snapshot.fingerprint,
        "system_fingerprint": system_fingerprint,
    }
    if created_monotonic is not None:
        values["created_monotonic"] = created_monotonic
    return PromptVariableApplication(**values)


def _append_application(session_id: str, user_text: str) -> PromptVariableApplication:
    return PromptVariableApplication(
        system_text=None,
        user_text=user_text,
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=session_id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )


class _PromptScopeService:
    """The `app_instance.prompt_scope_service` seam the cluster reads."""

    def __init__(self, record: dict[str, object] | None = None) -> None:
        self.record = record if record is not None else _prompt_record()
        self.search_calls: list[dict[str, object]] = []
        self.saved: list[dict[str, object]] = []
        self.existing: dict[str, object] | None = None
        self.save_result: object = {"local_id": 42}

    async def get_capabilities(self, *, mode: str):
        return SimpleNamespace(
            structured_kinds=frozenset(),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def list_prompts(self, *, mode: str, page: int, per_page: int):
        return {
            "items": [dict(self.record)],
            "page": page,
            "per_page": per_page,
            "total_items": 1,
            "total_pages": 1,
        }

    async def search_prompts(self, *, mode: str, query: str, limit: int, **kwargs):
        self.search_calls.append(
            {"mode": mode, "query": query, "limit": limit, **kwargs}
        )
        return [dict(self.record)]

    async def get_prompt(self, *, mode: str, prompt_identifier: str, **kwargs):
        return self.existing

    async def save_prompt(self, *, mode: str, **payload):
        self.saved.append({"mode": mode, **payload})
        return self.save_result


# ---------------------------------------------------------------------------
# The real modal-open path, and the selection it persists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_modal_open_persists_the_reviewed_system_selection() -> None:
    """Drive `_open_console_prompts_modal` for real, then apply through the
    modal's own host coordinator and assert the store PERSISTED it."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("draft body")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)

        result = ConsolePromptsResult(
            kind="apply",
            composer_snapshot=composer.capture_draft_snapshot(),
            user_text=None,
            system_text="You answer in one sentence.",
            apply_user=False,
            apply_system=True,
            captured_system_fingerprint=None,
        )
        outcome = await modal._apply_improvement_result(result, None)

        assert outcome.kind == "applied"
        settings = console._session._ensure_active_console_session_settings()
        assert settings.system_prompt == "You answer in one sentence."
        assert (
            store.session_settings(session_id).system_prompt
            == "You answer in one sentence."
        )


@pytest.mark.asyncio
async def test_prompts_modal_apply_refuses_when_the_system_prompt_moved() -> None:
    """The opening fingerprint is the guard: a System prompt changed behind
    the modal must make the apply stale rather than clobber it."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]

        # Somebody else moved the live System prompt after the modal opened.
        store.set_session_system_prompt(session_id, "Changed elsewhere.")

        result = ConsolePromptsResult(
            kind="apply",
            composer_snapshot=composer.capture_draft_snapshot(),
            user_text=None,
            system_text="Never lands.",
            apply_user=False,
            apply_system=True,
            captured_system_fingerprint=None,
        )
        outcome = await modal._apply_improvement_result(result, None)

        assert outcome.kind == "stale"
        assert store.session_settings(session_id).system_prompt == "Changed elsewhere."


@pytest.mark.asyncio
async def test_prompts_modal_reads_provider_recovery_off_the_screen_at_open() -> None:
    """The Configure-provider seam is resolved when the modal is BUILT, so a
    screen-level replacement made beforehand must reach the modal."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        recovery = AsyncMock()
        console._open_console_provider_recovery = recovery
        console._console_provider_blocker_copy = lambda: "No provider configured."

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]

        assert modal._configure_provider is recovery
        assert modal._improve_unavailable_reason == "No provider configured."


# ---------------------------------------------------------------------------
# The callback bundle the modal opener hands to `ConsolePromptsModal`
#
# Written BEFORE task-2766 decomposed `_open_console_prompts_modal`'s 17
# nested closures into named collaborators, so every assertion below is a
# statement about behaviour that must survive that change byte-identically:
# the exact prompt-source contract each data callable forwards, the copy each
# raises when the source cannot serve, the pinned-target lifecycle the
# improvement flow shares across callbacks, and the dismissal focus restore.
# ---------------------------------------------------------------------------


class _RecordingPromptScopeService(_PromptScopeService):
    """Records the exact keyword contract each source callable forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def get_capabilities(self, *, mode: str):
        self.calls.append(("get_capabilities", {"mode": mode}))
        return await super().get_capabilities(mode=mode)

    async def list_prompts(self, *, mode: str, page: int, per_page: int):
        self.calls.append(
            ("list_prompts", {"mode": mode, "page": page, "per_page": per_page})
        )
        return await super().list_prompts(mode=mode, page=page, per_page=per_page)

    async def search_prompts(self, *, mode: str, query: str, limit: int, **kwargs):
        self.calls.append(
            ("search_prompts", {"mode": mode, "query": query, "limit": limit, **kwargs})
        )
        return await super().search_prompts(
            mode=mode, query=query, limit=limit, **kwargs
        )

    async def get_prompt(self, *, mode: str, prompt_identifier: str, **kwargs):
        self.calls.append(
            (
                "get_prompt",
                {"mode": mode, "prompt_identifier": prompt_identifier, **kwargs},
            )
        )
        return await super().get_prompt(
            mode=mode, prompt_identifier=prompt_identifier, **kwargs
        )

    async def save_prompt(self, *, mode: str, **payload):
        self.calls.append(("save_prompt", {"mode": mode, **payload}))
        return await super().save_prompt(mode=mode, **payload)

    def payloads(self, name: str) -> list[dict[str, object]]:
        return [payload for called, payload in self.calls if called == name]


class _CountingResolutionGateway:
    """A ready provider whose `resolve_for_send` count is observable."""

    def __init__(self) -> None:
        self.resolve_calls = 0

    async def resolve_for_send(self, selection):
        self.resolve_calls += 1
        return with_destination(ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=selection.base_url or "http://127.0.0.1:9099",
            model=(
                selection.explicit_model or selection.configured_model or "local-model"
            ),
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        ))


async def _open_prompts_modal(host, pilot, console) -> ConsolePromptsModal:
    console._open_console_prompts_modal()
    await pilot.pause()
    modal = host.screen_stack[-1]
    assert isinstance(modal, ConsolePromptsModal)
    return modal


@pytest.mark.asyncio
async def test_prompt_source_callables_forward_the_exact_service_contract() -> None:
    """Page size, search limit and the `source`-to-`mode` rename are the
    contract the Prompt Library modal is built against."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    service = _RecordingPromptScopeService()
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        modal = await _open_prompts_modal(host, pilot, console)
        service.calls.clear()

        await modal._capabilities("server")
        await modal._list_page("server", 3)
        await modal._search("local", "terse")
        await modal._detail("local", "11")
        await modal._save(source="server", name="Fresh", system_prompt="Be terse.")

        assert service.payloads("get_capabilities") == [{"mode": "server"}]
        assert service.payloads("list_prompts") == [
            {"mode": "server", "page": 3, "per_page": 10}
        ]
        assert service.payloads("search_prompts") == [
            {"mode": "local", "query": "terse", "limit": 25}
        ]
        assert service.payloads("get_prompt") == [
            {"mode": "local", "prompt_identifier": "11"}
        ]
        # `save` routes the payload's own `source` into the service's `mode`.
        assert service.payloads("save_prompt") == [
            {"mode": "server", "name": "Fresh", "system_prompt": "Be terse."}
        ]


@pytest.mark.asyncio
async def test_prompt_source_callables_name_the_source_they_cannot_reach() -> None:
    """A scope service missing a method is a user-visible refusal, per
    callable and per source -- never an AttributeError."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = SimpleNamespace()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        modal = await _open_prompts_modal(host, pilot, console)

        with pytest.raises(ValueError, match="Local Prompt source is unavailable"):
            await modal._capabilities("local")
        with pytest.raises(ValueError, match="Server Prompt source is unavailable"):
            await modal._list_page("server", 1)
        with pytest.raises(ValueError, match="Local Prompt search is unavailable"):
            await modal._search("local", "terse")
        with pytest.raises(ValueError, match="Server Prompt source is unavailable"):
            await modal._detail("server", "11")
        with pytest.raises(ValueError, match="selected Prompt source cannot save"):
            await modal._save(source="local", name="Fresh")


@pytest.mark.asyncio
async def test_prompts_modal_dismissal_restores_composer_focus() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        modal = await _open_prompts_modal(host, pilot, console)
        focus_calls: list[dict[str, object]] = []
        console._focus_console_composer_if_needed = lambda **kwargs: focus_calls.append(
            kwargs
        )

        modal.dismiss(None)
        await pilot.pause()

        assert focus_calls == [{"force": True}]


@pytest.mark.asyncio
async def test_improvement_target_is_pinned_once_and_shared_across_callbacks() -> None:
    """`activate` pins the disclosed target; the model-free manual path must
    reuse that same pin rather than resolve a second, possibly different one."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    gateway = _CountingResolutionGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("draft body")

        modal = await _open_prompts_modal(host, pilot, console)

        first = await modal._capture_manual_resolution()
        resolved_once = gateway.resolve_calls
        second = await modal._capture_manual_resolution()

        assert first is second
        assert gateway.resolve_calls == resolved_once

        activated = await modal._activate_improvement_context()
        assert activated.pinned_resolution is not None
        assert await modal._capture_manual_resolution() is activated.pinned_resolution


@pytest.mark.asyncio
async def test_improvement_activation_refuses_a_moved_system_prompt() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    app.console_provider_gateway_factory = lambda: _CountingResolutionGateway()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        modal = await _open_prompts_modal(host, pilot, console)
        store.set_session_system_prompt(session_id, "Changed elsewhere.")

        with pytest.raises(ValueError, match="The Console System prompt changed"):
            await modal._activate_improvement_context()


@pytest.mark.asyncio
async def test_improvement_snapshot_refuses_an_unpinned_provider_target() -> None:
    """No disclosure, no request: the snapshot builder must not fall back to
    resolving a target the user was never shown."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    app.console_provider_gateway_factory = lambda: _CountingResolutionGateway()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("draft body")

        modal = await _open_prompts_modal(host, pilot, console)

        with pytest.raises(ValueError, match="no longer pinned"):
            await modal._build_improvement_snapshot(request_id="req-1", mode="auto")


@pytest.mark.asyncio
async def test_improvement_validation_prefers_the_captured_snapshot() -> None:
    """The reviewed working copy is validated against the snapshot the request
    captured; a captured object without one falls back to the opening draft."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("draft body")
        opening_snapshot = composer.capture_draft_snapshot()

        modal = await _open_prompts_modal(host, pilot, console)
        seen: list[tuple[object, str]] = []
        composer.validate_improvement = lambda snapshot, text: seen.append(
            (snapshot, text)
        )

        modal._validate_improvement(
            SimpleNamespace(composer_snapshot="captured"), "reviewed"
        )
        modal._validate_improvement(object(), "fallback")

        assert seen[0] == ("captured", "reviewed")
        assert seen[1] == (opening_snapshot, "fallback")


# ---------------------------------------------------------------------------
# `/prompt` and `/system` name resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_command_replaces_the_draft_with_the_resolved_body() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService(_prompt_record(system_prompt=""))
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("/prompt Summarize")

        await console._console_command_insert_prompt(SimpleNamespace(args="Summarize"))
        await pilot.pause()

        assert composer.draft_text() == "Summarize the following."


@pytest.mark.asyncio
async def test_prompt_application_replaces_complete_snapshot_and_persists_draft() -> (
    None
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("ordinary ")
        composer.insert_file_segment("secret file", "notes.txt · 11 B")
        composer.insert_pasted_text(" tail")
        snapshot = composer.capture_draft_snapshot()
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        pending = PendingAttachment(
            file_path="/tmp/photo.png",
            display_name="photo.png",
            file_type="image",
            insert_mode="attachment",
            data=b"image",
            mime_type="image/png",
            original_size=5,
            processed_size=5,
        )
        assert store.add_pending_attachment(session_id, pending) is True
        application = _replacement_application(
            snapshot=snapshot,
            session_id=session_id,
            user_text="replacement",
        )

        applied = console._prompts._apply_prompt_application(
            application,
            captured_snapshot=snapshot,
        )

        assert applied is True
        assert composer.draft_text() == "replacement"
        assert store.session_draft(session_id) == "replacement"
        assert store.pending_attachments(session_id) == [pending]
        assert [
            segment.origin for segment in composer.capture_draft_snapshot().segments
        ] == ["paste"]


@pytest.mark.asyncio
async def test_prompt_application_refuses_stale_draft_system_session_and_expiry() -> (
    None
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        notify = Mock()
        app.notify = notify

        composer.load_draft("draft")
        stale_draft_snapshot = composer.capture_draft_snapshot()
        stale_draft = _replacement_application(
            snapshot=stale_draft_snapshot,
            session_id=session_id,
            user_text="draft secret",
        )
        composer.insert_text(" changed")
        assert (
            console._prompts._apply_prompt_application(
                stale_draft,
                captured_snapshot=stale_draft_snapshot,
            )
            is False
        )
        assert composer.draft_text() == "draft changed"

        composer.load_draft("system draft")
        stale_system_snapshot = composer.capture_draft_snapshot()
        stale_system = _replacement_application(
            snapshot=stale_system_snapshot,
            session_id=session_id,
            user_text="user secret",
            system_text="system secret",
            system_fingerprint=fingerprint_system_text(""),
        )
        store.set_session_system_prompt(session_id, "changed elsewhere")
        assert (
            console._prompts._apply_prompt_application(
                stale_system,
                captured_snapshot=stale_system_snapshot,
            )
            is False
        )
        assert composer.draft_text() == "system draft"
        assert store.session_settings(session_id).system_prompt == "changed elsewhere"

        composer.load_draft("session draft")
        stale_session_snapshot = composer.capture_draft_snapshot()
        stale_session = _replacement_application(
            snapshot=stale_session_snapshot,
            session_id=session_id,
            user_text="session secret",
        )
        settings = store.session_settings(session_id)
        store.create_session(title="Other", settings=settings)
        assert (
            console._prompts._apply_prompt_application(
                stale_session,
                captured_snapshot=stale_session_snapshot,
            )
            is False
        )
        assert composer.draft_text() == "session draft"

        new_session_id = store.active_session_id
        assert new_session_id is not None
        expired_snapshot = composer.capture_draft_snapshot()
        expired = _replacement_application(
            snapshot=expired_snapshot,
            session_id=new_session_id,
            user_text="expired secret",
            created_monotonic=0.0,
        )
        assert (
            console._prompts._apply_prompt_application(
                expired,
                captured_snapshot=expired_snapshot,
            )
            is False
        )
        assert composer.draft_text() == "session draft"

        assert notify.call_count == 4
        notification_text = " ".join(str(call) for call in notify.call_args_list)
        assert "draft secret" not in notification_text
        assert "user secret" not in notification_text
        assert "system secret" not in notification_text
        assert "session secret" not in notification_text
        assert "expired secret" not in notification_text


@pytest.mark.asyncio
async def test_prompt_application_rolls_back_draft_when_system_mutation_raises(
    monkeypatch,
) -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        composer.load_draft("draft before")
        snapshot = composer.capture_draft_snapshot()
        store.set_session_draft(session_id, composer.draft_text())
        application = _replacement_application(
            snapshot=snapshot,
            session_id=session_id,
            user_text="user secret",
            system_text="system secret",
            system_fingerprint=fingerprint_system_text(""),
        )
        notify = Mock()
        app.notify = notify
        real_set_system = store.set_session_system_prompt
        calls = 0

        def fail_system(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                real_set_system(*args, **kwargs)
            raise RuntimeError("exception detail secret")

        monkeypatch.setattr(store, "set_session_system_prompt", fail_system)

        assert (
            console._prompts._apply_prompt_application(
                application,
                captured_snapshot=snapshot,
            )
            is False
        )
        restored = composer.capture_draft_snapshot()
        assert restored.segments == snapshot.segments
        assert restored.cursor_index == snapshot.cursor_index
        assert restored.selection == snapshot.selection
        assert store.session_draft(session_id) == "draft before"
        assert store.session_settings(session_id).system_prompt is None
        assert "secret" not in str(notify.call_args)


@pytest.mark.asyncio
async def test_prompt_application_reports_durable_system_failure_separately(
    monkeypatch,
) -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        composer.load_draft("draft before")
        snapshot = composer.capture_draft_snapshot()
        application = _replacement_application(
            snapshot=snapshot,
            session_id=session_id,
            user_text="user after",
            system_text="system after",
            system_fingerprint=fingerprint_system_text(""),
        )
        real_set_system = store.set_session_system_prompt

        def apply_but_report_unsaved(*args, **kwargs):
            session, _persisted = real_set_system(*args, **kwargs)
            return session, False

        monkeypatch.setattr(
            store,
            "set_session_system_prompt",
            apply_but_report_unsaved,
        )

        def fail_popup_sync() -> None:
            raise RuntimeError("popup exception secret")

        monkeypatch.setattr(
            console,
            "_sync_console_command_popup",
            fail_popup_sync,
        )
        notify = Mock()
        app.notify = notify

        assert (
            console._prompts._apply_prompt_application(
                application,
                captured_snapshot=snapshot,
            )
            is True
        )
        assert composer.draft_text() == "user after"
        assert store.session_settings(session_id).system_prompt == "system after"
        assert [item.args[0] for item in notify.call_args_list] == [
            ConsolePromptsController._PROMPT_DISPLAY_SYNC_FAILED_COPY,
            ConsolePromptsController._PROMPT_SYSTEM_PERSISTENCE_FAILED_COPY,
        ]
        assert all(
            item.kwargs == {"severity": "warning"} for item in notify.call_args_list
        )
        assert "secret" not in str(notify.call_args_list)


@pytest.mark.asyncio
async def test_prompt_application_keeps_durable_system_when_surface_sync_fails(
    monkeypatch,
) -> None:
    class RecordingPersistence:
        def __init__(self) -> None:
            self.system_updates: list[tuple[str, str | None]] = []

        def update_conversation_system_prompt(
            self,
            *,
            conversation_id: str,
            system_prompt: str | None,
        ) -> bool:
            self.system_updates.append((conversation_id, system_prompt))
            return True

    app = _build_test_app()

    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        session = next(item for item in store.sessions() if item.id == session_id)
        persistence = RecordingPersistence()
        session.persisted_conversation_id = "conversation-1"
        store.persistence = persistence
        composer.load_draft("draft before")
        snapshot = composer.capture_draft_snapshot()
        application = _replacement_application(
            snapshot=snapshot,
            session_id=session_id,
            user_text="user after",
            system_text="system after",
            system_fingerprint=fingerprint_system_text(""),
        )
        notify = Mock()
        app.notify = notify

        def fail_surface_sync() -> None:
            raise RuntimeError("surface exception secret")

        monkeypatch.setattr(
            console,
            "_sync_console_chat_core_state",
            fail_surface_sync,
        )

        assert (
            console._prompts._apply_prompt_application(
                application,
                captured_snapshot=snapshot,
            )
            is True
        )
        assert composer.draft_text() == "user after"
        assert store.session_draft(session_id) == "user after"
        assert store.session_settings(session_id).system_prompt == "system after"
        assert persistence.system_updates == [("conversation-1", "system after")]
        notify.assert_called_once_with(
            ConsolePromptsController._PROMPT_DISPLAY_SYNC_FAILED_COPY,
            severity="warning",
        )
        assert "secret" not in str(notify.call_args)


@pytest.mark.parametrize(
    ("system_fails", "expected_applied", "expected_undo"),
    [(False, True, False), (True, False, True)],
)
@pytest.mark.asyncio
async def test_system_only_empty_apply_settles_prior_prompt_undo(
    monkeypatch,
    system_fails: bool,
    expected_applied: bool,
    expected_undo: bool,
) -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        composer.load_draft("unrelated older draft")
        prior_snapshot = composer.capture_draft_snapshot()
        composer.replace_snapshot_as_paste(prior_snapshot, "")
        empty_snapshot = composer.capture_draft_snapshot()
        application = _replacement_application(
            snapshot=empty_snapshot,
            session_id=session_id,
            user_text=None,
            system_text="new system",
            system_fingerprint=fingerprint_system_text(""),
        )

        if system_fails:

            def fail_system(*_args, **_kwargs):
                raise RuntimeError("system mutation failed")

            monkeypatch.setattr(store, "set_session_system_prompt", fail_system)

        assert (
            console._prompts._apply_prompt_application(
                application,
                captured_snapshot=empty_snapshot,
            )
            is expected_applied
        )
        assert composer.draft_text() == ""
        assert store.session_settings(session_id).system_prompt == (
            "new system" if expected_applied else None
        )
        assert composer.improvement_undo_available is expected_undo
        assert composer.undo_improvement() is expected_undo
        assert composer.draft_text() == (
            "unrelated older draft" if expected_undo else ""
        )


@pytest.mark.asyncio
async def test_prompt_application_handles_composer_failure_without_secondary_error(
    monkeypatch,
) -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        composer.load_draft("draft before")
        snapshot = composer.capture_draft_snapshot()
        application = _replacement_application(
            snapshot=snapshot,
            session_id=session_id,
            user_text="replacement",
        )
        notify = Mock()
        app.notify = notify

        def fail_composer(*_args, **_kwargs):
            raise RuntimeError("composer exception secret")

        monkeypatch.setattr(composer, "replace_snapshot_as_paste", fail_composer)

        assert (
            console._prompts._apply_prompt_application(
                application,
                captured_snapshot=snapshot,
            )
            is False
        )
        assert composer.draft_text() == "draft before"
        assert "secret" not in str(notify.call_args)


@pytest.mark.asyncio
async def test_system_command_applies_and_persists_the_resolved_system_part() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("/system Summarize")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        await console._console_command_apply_system(SimpleNamespace(args="Summarize"))
        await pilot.pause()

        assert store.session_settings(session_id).system_prompt == "You are terse."
        # A handled command never leaves its own invocation text behind.
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_prompt_search_filters_recipes_and_uses_a_prefix_fts_query() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    service = _PromptScopeService(_prompt_record(artifact_type="recipe"))
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        records = await console._prompts._console_prompt_search("Summ")

        assert records == []
        assert service.search_calls[-1]["fts_match_query"] == '"Summ"*'
        assert await console._prompts._resolve_console_prompt_by_name("Summ") is None


def test_recipe_records_are_never_executable() -> None:
    is_recipe = ConsolePromptsController._is_recipe_prompt_record
    assert is_recipe({"artifact_type": "Recipe"}) is True
    assert is_recipe({"artifact_type": "prompt"}) is False
    assert is_recipe({}) is False


def test_prompt_prefix_fts_query_escapes_embedded_quotes() -> None:
    prefix_query = ConsolePromptsController._console_prompt_prefix_fts_query
    assert prefix_query('a"b') == '"a""b"*'


# ---------------------------------------------------------------------------
# Save-to-Library, handoff insert, prompt history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_save_to_library_reports_create_and_collision() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    service = _PromptScopeService()
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert (
            await console._prompts._save_console_system_prompt_to_library("", "body")
            == "Enter a name to save this system prompt to Library."
        )
        assert (
            await console._prompts._save_console_system_prompt_to_library("Name", "  ")
            == "Enter a system prompt to save."
        )
        assert (
            await console._prompts._save_console_system_prompt_to_library(
                "Fresh", "Be terse."
            )
            == "Saved."
        )
        assert service.saved[-1]["name"] == "Fresh"
        assert service.saved[-1]["system_prompt"] == "Be terse."

        service.existing = {"name": "Fresh", "id": 1}
        outcome = await console._prompts._save_console_system_prompt_to_library(
            "Fresh", "Be terse."
        )
        assert "already in use" in outcome


@pytest.mark.asyncio
async def test_prompt_history_store_is_lazy_shared_and_factory_seamed() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    sentinel = object()
    app.console_prompt_history_factory = lambda: sentinel
    screen = ChatScreen(app)

    first = screen._ensure_console_prompt_history()

    assert first is sentinel
    assert screen._ensure_console_prompt_history() is sentinel


@pytest.mark.asyncio
async def test_library_prompt_insert_handoff_appends_onto_the_live_draft() -> None:
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()

    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.clear_draft()
        composer.insert_text("existing")
        session_id = console._ensure_console_chat_store().active_session_id
        assert session_id is not None
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _append_application(session_id, "staged body"),
        )

        await console._consume_pending_console_prompt_insert()
        await pilot.pause()

        assert composer.draft_text() == "existing\nstaged body"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_library_prompt_insert_handoff_is_blocked_before_setup_completes() -> (
    None
):
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()

    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.clear_draft()
        console._console_setup_blocked_reason = lambda: "blocked"
        notify = Mock()
        app.notify = notify
        session_id = console._ensure_console_chat_store().active_session_id
        assert session_id is not None
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _append_application(session_id, "staged body"),
        )

        await console._consume_pending_console_prompt_insert()
        await pilot.pause()

        assert composer.draft_text() == ""
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        assert notify.called


# ---------------------------------------------------------------------------
# The extraction's own contract (wave-3 console decomposition, task 3)
# ---------------------------------------------------------------------------


def test_prompts_controller_is_wired_in_chat_screen_init() -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)

    screen = ChatScreen(app)

    assert isinstance(screen._prompts, ConsolePromptsController)
    assert screen._prompts._screen is screen
    assert screen._prompts.app_instance is app


def test_prompts_controller_owns_no_dom() -> None:
    """A controller owns behaviour and state, and zero pixels: no `query_one`
    /`query` anywhere in the module, matching every existing controller."""
    source = inspect.getsource(
        __import__("tldw_chatbook.UI.Console_Modules.prompts", fromlist=["prompts"])
    )
    tree = ast.parse(source)
    dom_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"query", "query_one", "query_exactly_one"}
    ]

    assert dom_calls == []


@pytest.mark.parametrize(
    "name",
    [
        "_open_console_prompts_modal",
        "_ensure_console_prompt_history",
        "_console_command_insert_prompt",
        "_console_command_apply_system",
        "_open_console_system_prompt_editor",
        "_consume_pending_console_prompt_insert",
    ],
)
def test_screen_keeps_a_real_delegation_for_every_outside_caller(name: str) -> None:
    """These six are reached from outside the cluster -- by a staying screen
    method, by Textual's `action_*` resolution, by the command-registry dict,
    or by a pre-existing test that replaces the exact screen attribute. Each
    must stay a thin forwarder onto the controller, never a re-implementation
    and never absent."""
    method = getattr(ChatScreen, name)
    body = inspect.getsource(method)

    assert "self._prompts." + name in body
    assert len(body.splitlines()) <= 4


def test_moved_methods_are_gone_from_the_screen() -> None:
    """The seven with no outside caller left no residue at all."""
    for name in (
        "_is_recipe_prompt_record",
        "_console_prompt_prefix_fts_query",
        "_console_prompt_search",
        "_resolve_console_prompt_by_name",
        "_open_console_prompt_picker_for_insert",
        "_open_console_prompt_picker_for_apply_system",
        "_save_console_system_prompt_to_library",
    ):
        assert not hasattr(ChatScreen, name), name
        assert hasattr(ConsolePromptsController, name), name


def test_stale_reason_reports_the_session_change_before_the_system_change() -> None:
    """Order matters here, and nothing else pins it.

    `_stale_reason` checks two facts pinned when the modal opened: the session
    it was opened over, and the System prompt it disclosed. When BOTH have
    moved the user sees only the first message, so the order chooses the copy.
    A task-2766 review mutated this ordering and all 28 controller tests plus
    all 304 native-chat-flow tests still passed -- the branch that reports the
    session change was reachable in no test where the fingerprint had also
    moved.

    Session-before-System is the right order because the session change is the
    larger fact: the System prompt the modal disclosed belongs TO a session, so
    naming the System prompt while the user has actually switched sessions
    describes a consequence rather than the cause.
    """
    from tldw_chatbook.UI.Console_Modules.prompts import (
        _ConsolePromptImprovementFlow,
    )

    flow = _ConsolePromptImprovementFlow.__new__(_ConsolePromptImprovementFlow)
    flow._session_id = "session-a"
    flow._current_system_fingerprint = "fingerprint-a"
    flow._store = SimpleNamespace(active_session_id="session-a")
    flow._active_system_fingerprint = lambda: "fingerprint-a"

    assert flow._stale_reason() == ""

    # Only the System prompt moved.
    flow._active_system_fingerprint = lambda: "fingerprint-b"
    assert flow._stale_reason() == "The Console System prompt changed."

    # Only the session moved.
    flow._active_system_fingerprint = lambda: "fingerprint-a"
    flow._store = SimpleNamespace(active_session_id="session-b")
    assert flow._stale_reason() == "The active Console session changed."

    # BOTH moved -- the session is reported, not the System prompt. This is
    # the assertion the mutation slipped past.
    flow._active_system_fingerprint = lambda: "fingerprint-b"
    assert flow._stale_reason() == "The active Console session changed."
