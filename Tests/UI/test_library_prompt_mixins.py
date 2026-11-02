"""Literal Library editor and body-free exact-target handoff regressions."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import fields, replace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, Select, TextArea

from Tests.Chat.test_console_prompt_mixin_admission import A, B
from Tests.Chat.test_console_prompt_mixin_admission import (
    case as case,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _open_prompts_list,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_worker_group_to_drain,
)
from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinPreferenceConflictError,
    MixinPreferences,
    MixinSourceChangedError,
    MixinSourceRef,
)
from tldw_chatbook.Library.library_prompts_state import (
    PromptBrowseScope,
    build_prompt_editor_state,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)
from tldw_chatbook.Prompt_Management.prompt_variables import APPLICATION_TTL_SECONDS
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleMixinEnableRequest,
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.Widgets.Prompts.prompt_mixin_editor import PromptMixinEditor

pytestmark = pytest.mark.bootstrap_profile


class EditorHarness(LibraryProductionCSSHarness):
    async def on_mount(self):
        pass

    def compose(self) -> ComposeResult:
        yield PromptMixinEditor(
            name="Scene",
            definition=LiteralMixinDefinition("{pose}\n## User\n", "after"),
        )


@pytest.mark.asyncio
async def test_editor_preserves_literal_body():
    app = EditorHarness(_build_test_app())
    async with app.run_test(size=(80, 24)):
        editor = app.query_one(PromptMixinEditor)
        assert editor.query_one("#mixin-body", TextArea).text == "{pose}\n## User\n"
        assert editor.read_definition() == LiteralMixinDefinition(
            "{pose}\n## User\n", "after"
        )
        assert editor.read_name() == "Scene"
        editor.query_one("#mixin-placement", Select).value = "before"
        assert editor.read_definition().placement == "before"
        assert not editor.query("#library-prompt-block-editor")


def test_type_filter_participates_in_browse_fingerprint():
    assert (
        PromptBrowseScope(artifact_type="mixin").fingerprint
        != PromptBrowseScope().fingerprint
    )
    with pytest.raises(ValueError):
        PromptBrowseScope(artifact_type="future")


def test_mixin_editor_state_is_positive_literal_dispatch():
    state = build_prompt_editor_state(
        {
            "artifact_type": "mixin",
            "prompt_format": "structured",
            "prompt_schema_version": 3,
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": "{pose}\n",
                "placement": "after",
            },
        }
    )
    assert state.mixin_definition == LiteralMixinDefinition("{pose}\n", "after")
    assert state.block_editor_state is None
    assert not state.compatibility_reason


def seed_mixin(db, name="Scene", body="{pose}\n## User\n", placement="after"):
    return db.add_prompt(
        name=name,
        author="Author",
        details="Literal instructions",
        system_prompt="",
        user_prompt=body,
        keywords=["scene"],
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": body,
            "placement": placement,
        },
    )[0]


@pytest.mark.asyncio
async def test_library_mixin_save_conflict_export_and_dirty_escape(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message="Literal editor did not arm",
            )
            editor = screen.query_one(PromptMixinEditor)
            assert not screen.query("#library-prompt-block-editor")
            body = "  {literal}\n## System\n[keep]\n"
            editor.query_one("#mixin-body", TextArea).load_text(body)
            editor.query_one("#mixin-placement", Select).value = "before"
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.dirty,
                message="Mix-in edit not dirty",
            )
            editor.query_one("#mixin-body", TextArea).focus()
            await pilot.press("escape")
            assert screen._prompts_state.view == "editor"
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._prompts_state.dirty,
                message="Mix-in not saved",
            )
            stored = db.fetch_prompt_details(prompt_id)
            assert json.loads(stored["prompt_definition"])["body"] == body
            assert json.loads(stored["prompt_definition"])["placement"] == "before"
            assert stored["system_prompt"] == ""
            assert (
                screen._prompts_controller._library_prompt_markdown_artifact_fields(
                    action="exporting"
                )["prompt_definition"]["body"]
                == body
            )
            work = screen.query_one("#library-prompt-work-pane")
            for mode in ("advanced", "info"):
                await work.set_editor_mode(mode)
                assert not screen.query_one("#library-prompt-basic-region").display
                assert not screen.query_one("#library-prompt-advanced-region").display
                assert (
                    screen.query_one(PromptMixinEditor).read_definition().body == body
                )
            editor.query_one("#mixin-name", Input).value = "Kept name"
            editor.query_one("#mixin-placement", Select).value = "after"
            editor.query_one("#mixin-body", TextArea).load_text("kept draft\n")
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.dirty,
                message="Second edit not dirty",
            )
            await service.save_prompt(
                mode="local",
                prompt_identifier=prompt_id,
                name="Scene",
                user_prompt="External",
                system_prompt="",
                artifact_type="mixin",
                prompt_format="structured",
                prompt_schema_version=3,
                prompt_definition={
                    "kind": "text_mixin",
                    "schema_version": 3,
                    "body": "External",
                    "placement": "after",
                },
                expected_version=stored["version"],
            )
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.conflict_snapshot is not None,
                message="Version conflict not surfaced",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen.query_one(PromptMixinEditor) is not editor
                ),
                message="Conflict editor not mounted",
            )
            assert (
                screen.query_one(PromptMixinEditor).read_definition().body
                == "kept draft\n"
            )
            assert screen.query_one(PromptMixinEditor).read_name() == "Kept name"
            assert (
                screen.query_one(PromptMixinEditor).read_definition().placement
                == "after"
            )
            await _wait_for_condition(
                pilot,
                lambda: bool(screen.query("#library-prompt-conflict-reload")),
                message="Conflict controls not mounted",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.query_one("#library-prompt-conflict-reload", Button).display
                    and not screen.query_one(
                        "#library-prompt-conflict-reload", Button
                    ).disabled
                    and not screen._prompts_controller._library_prompt_write_worker_is_active()
                ),
                message="Conflict reload still busy",
            )
            screen.query_one("#library-prompt-conflict-reload", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.conflict_snapshot is None,
                message="Conflict reload not applied",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen.query_one(PromptMixinEditor).read_definition().body
                    == "External"
                ),
                message="Reloaded literal editor not mounted",
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_literal_save_keeps_live_defining_module_policy_callbacks(
    tmp_path, monkeypatch
):
    from tldw_chatbook.UI.Library_Modules import library_prompts_controller as owner

    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    entered, release = threading.Event(), threading.Event()
    validator = owner.require_artifact_save_supported
    classifier = owner.classify_prompt_save_error
    save = service.save_prompt

    def refuse(*_args, **_kwargs):
        raise ValueError("Source policy refused this draft.")

    async def held_save(**kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("literal save was not released")
        return await save(**kwargs)

    def classify(*args):
        outcome = classifier(*args)
        return "name-in-use" if outcome == "ok" else outcome

    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.editor_armed,
                message="Literal editor did not arm",
            )
            editor = screen.query_one(PromptMixinEditor)
            editor.query_one("#mixin-body", TextArea).load_text("kept literal draft")
            await pilot.pause()
            monkeypatch.setattr(owner, "require_artifact_save_supported", refuse)
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.status == "Source policy refused this draft."
                ),
                message="Save bypassed the defining-module policy validator",
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 1
            assert editor.read_draft()[1] == "kept literal draft"
            monkeypatch.setattr(owner, "require_artifact_save_supported", validator)
            service.save_prompt = held_save
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot, entered.is_set, message="Save did not reach its source worker"
            )
            monkeypatch.setattr(owner, "classify_prompt_save_error", classify)
            release.set()
            await _wait_for_condition(
                pilot,
                lambda: "Name already in use" in screen._prompts_state.status,
                message="Save captured an obsolete result classifier before its await",
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 2
            assert screen._prompts_state.version == 1
            assert screen._prompts_state.dirty
            assert editor.read_draft()[1] == "kept literal draft"
    finally:
        release.set()
        db.close_connection()


@pytest.mark.asyncio
async def test_library_mixin_create_and_filtered_pagination_totals(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    for i in range(22):
        seed_mixin(db, name=f"Scene {i:02}")
    db.add_prompt(
        name="Ordinary", author="", details="", system_prompt="", user_prompt="plain"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompts_list(screen, pilot)
            controller = screen._library_prompt_browse_controller
            await _wait_for_condition(
                pilot,
                lambda: controller.result.status == "ready",
                message="Initial page not ready",
            )
            await pilot.pause()
            screen.query_one("#library-prompts-type-filter", Select).value = "mixin"
            await _wait_for_condition(
                pilot,
                lambda: (
                    controller.result.status == "ready"
                    and controller.result.scope.artifact_type == "mixin"
                ),
                message=lambda: (
                    f"Type filter not applied: {controller.scope} {controller.result.status} {controller.result.error}"
                ),
            )
            assert controller.result.total_items == 22
            assert len(controller.result.items) == 20
            screen._prompts_controller._request_library_prompts_browse(
                controller.scope_for_page(2), focus_identity=None
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    controller.result.status == "ready" and controller.result.page == 2
                ),
                message="Filtered page 2 not applied",
            )
            assert controller.result.total_items == 22
            assert len(controller.result.items) == 2
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query("#library-prompts-new-mixin"))
                    and screen.query_one(
                        "#library-prompts-new-mixin", Button
                    ).is_mounted
                    and not screen.query_one(
                        "#library-prompts-new-mixin", Button
                    ).disabled
                ),
                message="Filtered list controls not mounted",
            )
            screen.query_one("#library-prompts-new-mixin", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message="New literal editor not mounted",
            )
            editor = screen.query_one(PromptMixinEditor)
            browse_entered, browse_release = threading.Event(), threading.Event()
            original_browse = service.browse_prompts

            async def held_browse(**kwargs):
                browse_entered.set()
                if not browse_release.wait(5):
                    raise TimeoutError("held browse was not released")
                return await original_browse(**kwargs)

            service.browse_prompts = held_browse
            source_entered, source_release = asyncio.Event(), asyncio.Event()
            original_snapshot = screen._list_local_source_snapshot

            async def held_snapshot():
                source_entered.set()
                await source_release.wait()
                return await original_snapshot()

            screen._list_local_source_snapshot = held_snapshot
            editor.action_cancel()
            await asyncio.wait_for(source_entered.wait(), 5)
            assert await asyncio.to_thread(browse_entered.wait, 5)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.view == "list"
                    and not screen.query(PromptMixinEditor)
                ),
                message="Clean create Cancel did not exit",
            )
            await _wait_for_condition(
                pilot,
                lambda: bool(screen.query("#library-prompts-new-mixin")),
                message="Create controls not restored",
            )
            old_create = screen.query_one("#library-prompts-new-mixin", Button)
            assert old_create.disabled
            assert "Loading" in controller.pager.status_copy
            browse_release.set()
            await _wait_for_condition(
                pilot,
                lambda: controller.result.status == "ready",
                message="Cancel browse did not finish",
            )
            assert old_create.disabled
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen.query_one("#library-prompts-new-mixin", Button).disabled
                    and screen.query_one("#library-prompts-new-mixin", Button)
                    is not old_create
                    and screen.query_one(
                        "#library-prompts-new-mixin", Button
                    ).is_mounted
                ),
                message="Create stayed busy after browse refresh",
            )
            service.browse_prompts = original_browse
            assert not source_release.is_set()
            assert await pilot.click("#library-prompts-new-mixin")
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.view == "editor",
                message="Enabled Create click was not accepted",
            )
            source_release.set()
            await _wait_for_worker_group_to_drain(
                host, pilot, screen, "library_source_snapshot"
            )
            screen._list_local_source_snapshot = original_snapshot
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message=lambda: (
                    f"Create editor not reopened: view={screen._prompts_state.view} armed={screen._prompts_state.editor_armed} editors={len(screen.query(PromptMixinEditor))} workers={[(w.group, str(w.state)) for w in host.workers if w.node is screen]}"
                ),
            )
            editor = screen.query_one(PromptMixinEditor)
            assert editor.query_one("#mixin-body", TextArea).text == ""
            assert editor.query_one("#mixin-placement", Select).value == "before"
            editor.query_one("#mixin-name", Input).value = "New literal"
            editor.action_save()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.status
                    == "Instructions are required; enter your mix-in text."
                ),
                message="Empty literal instructions validation missing",
            )
            editor.query_one("#mixin-body", TextArea).load_text("{literal}\n")
            await pilot.pause()
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.selected_prompt_id is not None
                    and not screen._prompts_state.dirty
                ),
                message="Create mix-in did not persist",
            )
            assert (
                db.fetch_prompt_details(screen._prompts_state.selected_prompt_id)[
                    "artifact_type"
                ]
                == "mixin"
            )
    finally:
        db.close_connection()


def test_mixin_handoff_is_body_free_expires_and_latest_prompt_supersedes():
    from Tests.State.test_pending_handoff_store import _prompt_application

    source = MixinSourceRef(
        "local-profile", "local", "00000000-0000-4000-8000-000000000001"
    )
    request = ConsoleMixinEnableRequest(
        source, "chat", "incarnation", 0, (object(), object(), object()), 10.0
    )
    assert {field.name for field in fields(request)} == {
        "source",
        "target_session_id",
        "target_incarnation_id",
        "preference_revision",
        "source_fences",
        "created_monotonic",
    }
    now = [10.0]
    store = PendingHandoffStore(monotonic_clock=lambda: now[0])
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, request)
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim.status == "ready"
    assert all(a is b for a, b in zip(claim.value.source_fences, request.source_fences))
    assert store.release_prompt_claim(claim) == "ready"
    now[0] += APPLICATION_TTL_SECONDS
    expired = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert expired.status == "expired"
    assert store.acknowledge(expired)
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT, replace(request, created_monotonic=now[0])
    )
    old = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application(created_monotonic=now[0]),
    )
    assert not store.is_current_claim(old)
    assert store.release_prompt_claim(old) is None
    assert store.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT)


@pytest.mark.asyncio
async def test_enable_reference_keeps_existing_order_and_never_marks_used(case):
    store, session, scope, service = case.store, case.a, case.scope, case.service
    scope.release.set()
    revision = service.preferences(session.id).revision
    result = await service.enable_reference(
        session.id,
        source=B,
        target_incarnation_id=session.incarnation_id,
        expected_revision=revision,
    )
    assert tuple(row.source for row in result.entries) == (A, B)
    assert result.entries[1].enabled
    assert result.entries[1].last_used_version is None
    repeated = await service.enable_reference(
        session.id,
        source=B,
        target_incarnation_id=session.incarnation_id,
        expected_revision=result.revision,
    )
    assert repeated == result
    assert store.session_draft(session.id) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "incarnation", "tab", "preferences"])
async def test_enable_reference_refuses_changes_across_read(case, change):
    store, session, other, scope, service = (
        case.store,
        case.a,
        case.b,
        case.scope,
        case.service,
    )
    revision = service.preferences(session.id).revision
    current = [True]
    task = asyncio.create_task(
        service.enable_reference(
            session.id,
            source=B,
            target_incarnation_id=session.incarnation_id,
            expected_revision=revision,
            is_current=lambda: current[0],
        )
    )
    await scope.entered.wait()
    if change == "source":
        scope.source = object()
    elif change == "incarnation":
        session.incarnation_id = "replacement"
    elif change == "preferences":
        service.replace_preferences(
            session.id, expected_revision=revision, value=MixinPreferences(entries=())
        )
    else:
        store.switch_session(other.id)
        current[0] = False
    scope.release.set()
    with pytest.raises((MixinSourceChangedError, MixinPreferenceConflictError)):
        await task
    if change != "incarnation":
        assert not any(
            entry.enabled and entry.source == B
            for entry in service.preferences(session.id).entries
        )


@pytest.mark.asyncio
async def test_real_console_mixin_consumer_enables_without_touching_composer_or_system(
    tmp_path,
):
    from Tests.UI.app_factory import attach_chachanotes_db
    from Tests.UI.test_console_prompts_controller import (
        ConsoleHarness,
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    db, scope = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    record = db.fetch_prompt_details(prompt_id)
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.prompt_scope_service = scope

    class NativeConsoleHarness(ConsoleHarness):
        CSS_PATH = TldwCli.CSS_PATH

    host = NativeConsoleHarness(app)
    try:
        async with host.run_test(size=(140, 40)) as pilot:
            screen = host.screen_stack[-1]
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            store = screen._ensure_console_chat_store()
            session_id = store.active_session_id
            composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
            composer.insert_text("Keep draft")
            system = store.session_settings(session_id).system_prompt
            projection = screen.console_prompt_target_projection()
            assert projection.target_incarnation_id is not None
            request = ConsoleMixinEnableRequest(
                MixinSourceRef(
                    scope.mixin_source_identity(mode="local")[0],
                    "local",
                    record["uuid"],
                ),
                session_id,
                projection.target_incarnation_id,
                projection.preference_revision,
                (scope, *scope.mixin_source_identity(mode="local")[1:]),
            )
            app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, request)
            await screen._consume_pending_console_prompt_insert()
            preferences = screen._prompts._mixin_service().preferences(session_id)
            assert len(preferences.entries) == 1
            assert preferences.entries[0].enabled
            assert preferences.entries[0].last_used_version is None
            assert composer.draft_text() == "Keep draft"
            assert store.session_settings(session_id).system_prompt == system
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
            service = screen._prompts._mixin_service()
            disabled = service.replace_preferences(
                session_id,
                expected_revision=preferences.revision,
                value=replace(
                    preferences,
                    entries=(replace(preferences.entries[0], enabled=False),),
                ),
            )
            stale_request = replace(request, preference_revision=disabled.revision)
            # A reopened DB at the same lexical path has the same stable authority.
            replacement_db, replacement_scope = _real_prompt_scope_service(tmp_path)
            try:
                assert (
                    replacement_scope.mixin_source_identity(mode="local")[0]
                    == request.source.authority_key
                )
                app.pending_handoffs.stage(
                    HandoffChannel.CONSOLE_PROMPT_INSERT, stale_request
                )
                app.prompt_scope_service = replacement_scope
                await screen._consume_pending_console_prompt_insert()
                assert (
                    screen._prompts._mixin_service().preferences(session_id) == disabled
                )
            finally:
                app.prompt_scope_service = scope
                replacement_db.close_connection()
            # Scope-wrapper replacement reusing backend/DB is also a new origin.
            from tldw_chatbook.Prompt_Management.prompt_scope_service import (
                PromptScopeService,
            )

            wrapper = PromptScopeService(
                local_service=scope.mixin_source_identity(mode="local")[1],
                server_service=None,
            )
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT, stale_request
            )
            app.prompt_scope_service = wrapper
            await screen._consume_pending_console_prompt_insert()
            assert screen._prompts._mixin_service().preferences(session_id) == disabled
            app.prompt_scope_service = scope
            entered, release = asyncio.Event(), asyncio.Event()
            original_read = scope.read_mixin_records

            async def held_read(**kwargs):
                entered.set()
                await release.wait()
                return await original_read(**kwargs)

            scope.read_mixin_records = held_read
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT, stale_request
            )
            consuming = asyncio.create_task(
                screen._consume_pending_console_prompt_insert()
            )
            await asyncio.wait_for(entered.wait(), 5)
            app.prompt_scope_service = wrapper
            release.set()
            await consuming
            assert screen._prompts._mixin_service().preferences(session_id) == disabled
            assert composer.draft_text() == "Keep draft"
            assert store.session_settings(session_id).system_prompt == system
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
            scope.read_mixin_records = original_read
            app.prompt_scope_service = scope
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [3, 4])
async def test_unavailable_stored_mixin_never_opens_blank_create_editor(schema):
    from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
        LibraryPromptsListCanvas,
    )

    state = build_prompt_editor_state(
        {
            "local_id": 1,
            "name": "Unavailable",
            "version": 1,
            "artifact_type": "mixin",
            "prompt_format": "structured",
            "prompt_schema_version": schema,
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": schema,
                "body": "",
                "placement": "before",
            },
        }
    )

    class UnavailableHarness(EditorHarness):
        def compose(self):
            yield LibraryPromptsListCanvas(
                mode="editor", editor_state=state, can_update_original=False
            )

    app = UnavailableHarness(_build_test_app())
    async with app.run_test(size=(120, 40)):
        assert not app.query(PromptMixinEditor)
        assert app.query_one("#library-prompt-save", Button).disabled
        assert app.query_one("#library-prompt-insert-console", Button).disabled
        assert state.compatibility_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (170, 48)])
async def test_literal_editor_production_size_capture(tmp_path, size):
    """Keep a bounded visual evidence packet of the native Textual surface."""
    from pathlib import Path

    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(
        db,
        name="Literal scene direction",
        body="Use {pose} literally.\n## User\nKeep every section as written.\n",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    target = Path("/private/tmp/task7-captures")
    target.mkdir(mode=0o700, exist_ok=True)
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.editor_armed
                    and bool(screen.query(PromptMixinEditor))
                ),
                message="Capture editor not mounted",
            )
            editor = screen.query_one(PromptMixinEditor)
            editor.query_one("#mixin-name", Input).focus()
            await pilot.pause()
            host.save_screenshot(
                filename=f"mixin-{size[0]}x{size[1]}-clean.svg", path=str(target)
            )
            editor.query_one("#mixin-body", TextArea).load_text(
                "Keep {braces} literal.\n## System\n  Whitespace stays.\n"
            )
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.dirty,
                message="Capture editor not dirty",
            )
            save = screen.query_one("#library-prompt-save", Button)
            editor.query_one("#mixin-placement", Select).focus()
            await pilot.pause()
            assert save.display and not save.disabled
            host.save_screenshot(
                filename=f"mixin-{size[0]}x{size[1]}-dirty.svg", path=str(target)
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["backend", "database", "scope_wrapper"])
async def test_enable_refuses_same_authority_replacement_before_consumption(
    case, change
):
    from Tests.Chat.test_console_prompt_mixin_admission import Scope

    scope = case.scope
    fences = (scope, scope.source, scope.database)
    if change == "backend":
        scope.source = object()
    elif change == "database":
        scope.database = object()
    else:
        wrapper = Scope()
        wrapper.source, wrapper.database = scope.source, scope.database
        case.service.prompt_scope_service = wrapper
    with pytest.raises(MixinSourceChangedError):
        await case.service.enable_reference(
            case.a.id,
            source=B,
            target_incarnation_id=case.a.incarnation_id,
            expected_revision=case.service.preferences(case.a.id).revision,
            expected_source_fences=fences,
        )
    assert scope.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["save_as_new", "duplicate"])
async def test_invalid_mixin_copy_action_keeps_raw_draft_then_recovers(
    tmp_path, action
):
    """Strict validation cannot escape either real Library copy action."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message="Literal editor did not arm",
            )
            editor = screen.query_one(PromptMixinEditor)
            if action == "duplicate":
                screen.query_one("#library-prompt-more-actions", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen.query_one("#library-prompt-more-actions-region").display
                    ),
                    message="Duplicate actions did not open",
                )
            editor.query_one("#mixin-name", Input).value = "Recovered mix-in"
            editor.query_one("#mixin-placement", Select).value = "before"
            screen.query_one("#library-prompt-author", Input).value = "Kept author"
            screen.query_one(
                "#library-prompt-details", Input
            ).value = "Kept description"
            screen.query_one("#library-prompt-keywords", Input).value = "scene, kept"
            if action == "save_as_new":
                editor.query_one("#mixin-body", TextArea).load_text(
                    "Valid conflict draft"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: screen._prompts_state.dirty,
                    message="Conflict draft not dirty",
                )
                await service.save_prompt(
                    mode="local",
                    prompt_identifier=prompt_id,
                    name="Scene",
                    artifact_type="mixin",
                    prompt_format="structured",
                    prompt_schema_version=3,
                    prompt_definition={
                        "kind": "text_mixin",
                        "schema_version": 3,
                        "body": "External",
                        "placement": "after",
                    },
                    system_prompt="",
                    user_prompt="External",
                    expected_version=1,
                )
                screen.query_one("#library-prompt-save", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._prompts_state.conflict_snapshot is not None
                        and bool(screen.query(PromptMixinEditor))
                        and screen.query_one(PromptMixinEditor) is not editor
                        and not screen._prompts_controller._library_prompt_write_worker_is_active()
                    ),
                    message="Conflict editor did not mount",
                )
                editor = screen.query_one(PromptMixinEditor)
                button = screen.query_one("#library-prompt-conflict-save-new", Button)
                await _wait_for_condition(
                    pilot,
                    lambda: button.display and not button.disabled,
                    message="Save as new unavailable",
                )
            else:
                button = screen.query_one("#library-prompt-duplicate", Button)
            conflict = screen._prompts_state.conflict_snapshot
            version = screen._prompts_state.version
            source = screen._prompts_state.mixin_source_identity
            before = db.fetch_prompt_details(prompt_id)
            editor.query_one("#mixin-body", TextArea).load_text(" \n\t ")
            # Duplicate may already be queued when the child change hides More actions.
            button.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.status
                    == "Instructions are required; enter your mix-in text."
                ),
                message="Invalid copy action did not show Instructions validation",
            )
            assert host._exception is None
            assert screen.query_one(PromptMixinEditor) is editor
            assert editor.read_draft() == ("Recovered mix-in", " \n\t ", "before")
            assert screen._prompts_state.dirty
            assert screen._prompts_state.conflict_snapshot is conflict
            assert screen._prompts_state.selected_prompt_id == prompt_id
            assert screen._prompts_state.version == version
            assert screen._prompts_state.mixin_source_identity is source
            assert db.fetch_prompt_details(prompt_id) == before
            feedback = screen.query_one(
                "#library-prompt-conflict-copy"
                if conflict
                else "#library-prompt-save-status"
            )
            assert "Instructions are required" in str(feedback.render())
            body = "  Recovered {literal}\n## User\n"
            editor.query_one("#mixin-body", TextArea).load_text(body)
            await pilot.pause()
            button.press()
            if action == "duplicate":
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._prompts_state.selected_prompt_id is None
                        and bool(screen.query(PromptMixinEditor))
                        and screen.query_one(PromptMixinEditor) is not editor
                        and screen._prompts_state.editor_armed
                    ),
                    message="Valid Duplicate did not create its working copy",
                )
                assert (
                    screen.query_one(PromptMixinEditor).read_name()
                    == "Recovered mix-in (copy)"
                )
                screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.selected_prompt_id not in (None, prompt_id)
                    and not screen._prompts_state.dirty
                ),
                message="Valid copy action did not persist",
            )
            stored = db.fetch_prompt_details(screen._prompts_state.selected_prompt_id)
            assert stored["name"] == (
                "Recovered mix-in (copy)"
                if action == "duplicate"
                else "Recovered mix-in"
            )
            assert stored["artifact_type"] == "mixin"
            assert stored["prompt_format"] == "structured"
            assert stored["prompt_schema_version"] == 3
            assert json.loads(stored["prompt_definition"]) == {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": body,
                "placement": "before",
            }
            assert stored["system_prompt"] == ""
            assert stored["user_prompt"] == body
            assert stored["author"] == "Kept author"
            assert stored["details"] == "Kept description"
            assert set(stored["keywords"]) == {"scene", "kept"}
            assert (
                screen._prompts_controller._current_library_prompt_editor_state().source
                == "local"
            )
            assert all(
                a is b
                for a, b in zip(
                    screen._prompts_state.mixin_source_identity[1:],
                    service.mixin_source_identity(mode="local")[1:],
                )
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_library_mixin_save_ack_preserves_newer_literal_draft(tmp_path):
    """Current Library acknowledgement must retain edits accepted during Save."""
    import threading

    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    entered, release = threading.Event(), threading.Event()
    save = service.save_prompt

    async def held_save(**kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("held literal save was not released")
        return await save(**kwargs)

    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message="Literal editor did not arm",
            )
            editor = screen.query_one(PromptMixinEditor)
            editor.query_one("#mixin-body", TextArea).load_text("submitted literal")
            await pilot.pause()
            service.save_prompt = held_save
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot, entered.is_set, message="Save did not enter"
            )
            editor.query_one("#mixin-body", TextArea).load_text("")
            editor.query_one("#mixin-placement", Select).value = "before"
            release.set()
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_controller._library_prompt_version == 2,
                message="Save acknowledgement did not settle",
            )
            assert editor.read_draft()[1:] == ("", "before")
            assert screen._prompts_state.dirty
            assert (
                json.loads(db.fetch_prompt_details(prompt_id)["prompt_definition"])[
                    "body"
                ]
                == "submitted literal"
            )
            service.save_prompt = save
            editor.query_one("#mixin-body", TextArea).load_text("newer literal")
            await pilot.pause()
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._prompts_state.dirty,
                message="Second literal Save did not settle",
            )
            stored = json.loads(db.fetch_prompt_details(prompt_id)["prompt_definition"])
            assert (stored["body"], stored["placement"]) == ("newer literal", "before")
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retained_type", "current_type"),
    [
        ("mixin", "mixin"),
        ("prompt", "mixin"),
        ("recipe", "mixin"),
    ],
)
async def test_literal_history_restore_keeps_kind_definition_and_guards(
    tmp_path, retained_type, current_type
):
    from Tests.UI.test_console_prompts_modal import _definition
    from Tests.UI.test_library_prompt_action_journeys import _activate_more_action
    from tldw_chatbook.UI.Library_Modules.prompt_history_region import (
        LibraryPromptHistoryRegion,
    )
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    db, service = _real_prompt_scope_service(tmp_path)
    literal = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "  Original {literal}\n[before]\n",
        "placement": "before",
    }
    original_definition = (
        literal if retained_type == "mixin" else _definition("block_" + retained_type)
    )
    created = await service.save_prompt(
        mode="local",
        name="Retained artifact",
        author="Original author",
        details="Original details",
        keywords=["original"],
        artifact_type=retained_type,
        prompt_format="structured",
        prompt_schema_version=original_definition["schema_version"],
        prompt_definition=original_definition,
    )
    original = db.fetch_prompt_details(created["uuid"])
    prompt_id = original["id"]
    current_definition = (
        {**literal, "body": "Current instructions", "placement": "after"}
        if current_type == "mixin"
        else _definition("block_" + current_type)
    )
    await service.save_prompt(
        mode="local",
        prompt_identifier=prompt_id,
        expected_version=1,
        name="Retained artifact",
        author="Current author",
        details="Current details",
        artifact_type=current_type,
        prompt_format="structured",
        prompt_schema_version=current_definition["schema_version"],
        prompt_definition=current_definition,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)

    async def select_original(screen, pilot):
        await _activate_more_action(screen, pilot, "more-history")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_history_state.page_status == "loaded"
                and len(screen.query(".library-prompt-history-row")) == 2
            ),
            message="Literal retained history did not load",
        )
        next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        ).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_history_state.selected is not None,
            message="Retained v1 was not selected",
        )
        await pilot.pause()

    try:
        async with host.run_test(size=(140, 44)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await select_original(screen, pilot)
            restore = screen.query_one("#library-prompt-history-restore", Button)
            assert not restore.disabled
            assert screen.query_one("#library-prompt-history-user", TextArea).read_only
            state = screen._library_prompt_history_state
            stale = LibraryPromptHistoryRegion.RestoreRequested(
                prompt_uuid=state.prompt_uuid, scope_token=state.scope_token
            )
            controller = screen._prompts_controller
            capabilities = controller._library_prompt_capabilities
            controller._library_prompt_capabilities = replace(
                capabilities, conditional_update=False
            )
            controller._sync_library_prompt_history_region()
            await pilot.pause()
            assert screen.query_one("#library-prompt-history-restore", Button).disabled
            controller.handle_library_prompt_history_restore(stale)
            assert host.screen is screen
            controller._library_prompt_capabilities = capabilities
            controller._sync_library_prompt_history_region()
            field = screen.query_one(
                "#mixin-name" if current_type == "mixin" else "#library-prompt-name",
                Input,
            )
            field.value = "Unsaved name"
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.dirty,
                message="History edit did not become dirty",
            )
            assert screen.query_one("#library-prompt-history-restore", Button).disabled
            controller.handle_library_prompt_history_restore(stale)
            assert host.screen is screen
            screen.query_one("#library-prompt-discard", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.view == "list",
                message="Discard did not leave the editor",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_browse_controller.result.status == "ready"
                    and bool(screen.query(f"#library-prompt-row-{prompt_id}"))
                    and screen.query_one(
                        f"#library-prompt-row-{prompt_id}"
                    ).region.height
                    > 0
                ),
                message="Discarded editor did not reveal its settled Library row",
            )
            await pilot.pause()
            screen.query_one(f"#library-prompt-row-{prompt_id}", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.view == "editor"
                    and screen._prompts_state.detail is not None
                    and screen._prompts_state.editor_armed
                ),
                message="The settled Library row did not reopen its editor",
            )
            await select_original(screen, pilot)
            controller.handle_library_prompt_history_restore(stale)
            assert host.screen is screen
            screen.query_one("#library-prompt-history-restore", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: isinstance(host.screen, ConfirmationDialog),
                message="Literal restore confirmation did not open",
            )
            dialog = host.screen
            assert "retained v1" in dialog.message and "current v2" in dialog.message
            if retained_type != current_type:
                current_label = (
                    "Mix-in" if current_type == "mixin" else current_type.title()
                )
                retained_label = (
                    "Mix-in" if retained_type == "mixin" else retained_type.title()
                )
                assert f"{current_label} to {retained_label}" in dialog.message
            controller._library_prompt_capabilities = replace(
                capabilities, conditional_update=False
            )
            dialog.query_one("#confirm-button", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: host.screen is screen,
                message="Read-only confirmation did not return to the editor",
            )
            await pilot.pause()
            assert db.fetch_prompt_details(prompt_id)["version"] == 2
            controller._library_prompt_capabilities = capabilities
            controller._sync_library_prompt_history_region()
            await pilot.pause()
            screen.query_one("#library-prompt-history-restore", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: isinstance(host.screen, ConfirmationDialog),
                message="Restore could not retry after its source became writable",
            )
            dialog = host.screen
            dialog.query_one("#confirm-button", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: host.screen is screen and screen._prompts_state.version == 3,
                message="Retained definition did not become current v3",
            )
            restored = db.fetch_prompt_details(prompt_id)
            assert restored["version"] == 3
            for key in (
                "artifact_type",
                "author",
                "details",
                "keywords",
                "system_prompt",
                "user_prompt",
                "prompt_schema_version",
            ):
                assert restored[key] == original[key], key
            assert json.loads(restored["prompt_definition"]) == original_definition
            if retained_type == "mixin":
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        bool(screen.query(PromptMixinEditor))
                        and screen.query_one(PromptMixinEditor).read_draft()[1:]
                        == (literal["body"], "before")
                    ),
                    message="Restored literal editor did not mount",
                )
                assert screen.query_one(PromptMixinEditor).read_definition() == (
                    LiteralMixinDefinition(literal["body"], "before")
                )
                assert not screen.query("#library-prompt-block-editor")
            else:
                await _wait_for_condition(
                    pilot,
                    lambda: bool(screen.query("#library-prompt-block-editor")),
                    message="Restored block editor did not mount",
                )
                assert not screen.query(PromptMixinEditor)
                screen.query_one(
                    "#library-prompt-history-collapsible"
                ).collapsed = False
                await _wait_for_condition(
                    pilot,
                    lambda: len(screen.query(".library-prompt-history-row")) == 3,
                    message="Restored artifact history did not refresh",
                )
                next(
                    row
                    for row in screen.query(".library-prompt-history-row")
                    if row.source_version == 2
                ).press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_prompt_history_state.selected is not None
                        and screen._library_prompt_history_state.selected.source_version
                        == 2
                    ),
                    message="Retained mix-in v2 was not selected",
                )
                screen.query_one("#library-prompt-history-restore", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: isinstance(host.screen, ConfirmationDialog),
                    message="Block-to-mix-in restore confirmation did not open",
                )
                assert f"{retained_type.title()} to Mix-in" in host.screen.message
                host.screen.query_one("#confirm-button", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        host.screen is screen
                        and screen._prompts_state.version == 4
                        and bool(screen.query(PromptMixinEditor))
                        and screen.query_one(PromptMixinEditor).read_draft()[1:]
                        == ("Current instructions", "after")
                    ),
                    message="Historical mix-in was projected into a block editor",
                )
                restored_mixin = db.fetch_prompt_details(prompt_id)
                assert restored_mixin["artifact_type"] == "mixin"
                assert (
                    json.loads(restored_mixin["prompt_definition"])
                    == current_definition
                )
                assert screen.query_one(PromptMixinEditor).read_definition() == (
                    LiteralMixinDefinition("Current instructions", "after")
                )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_literal_editor_metadata_survives_the_saved_info_mode(tmp_path):
    from Tests.UI.test_library_prompt_action_journeys import _activate_more_action

    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id = seed_mixin(db)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "info"
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(screen.query(PromptMixinEditor))
                    and screen._prompts_state.editor_armed
                ),
                message="Info preference did not open the literal editor",
            )
            assert not screen.query_one("#library-prompt-mode-controls").display
            for selector in ("#library-prompt-author", "#library-prompt-keywords"):
                control = screen.query_one(selector, Input)
                assert control.display
                control.focus()
                await pilot.pause()
                assert screen.focused is control and control.region.height > 0
            screen.query_one("#library-prompt-author", Input).value = "Edited author"
            screen.query_one("#library-prompt-keywords", Input).value = "scene, edited"
            await _wait_for_condition(
                pilot,
                lambda: screen._prompts_state.dirty,
                message="Literal metadata edit did not become dirty",
            )
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.version == 2
                    and not screen._prompts_state.dirty
                ),
                message="Literal metadata did not save",
            )
            stored = db.fetch_prompt_details(prompt_id)
            assert stored["author"] == "Edited author"
            assert set(stored["keywords"]) == {"scene", "edited"}
            assert (
                json.loads(stored["prompt_definition"])["body"] == "{pose}\n## User\n"
            )
            await _activate_more_action(screen, pilot, "more-history")
            assert screen.query_one("#library-prompt-author", Input).display
            assert screen.query_one("#library-prompt-keywords", Input).display
            assert screen._prompts_controller._library_prompt_editor_mode == "info"
            assert app.app_config["library"]["prompt_editor_mode"] == "info"
    finally:
        db.close_connection()
