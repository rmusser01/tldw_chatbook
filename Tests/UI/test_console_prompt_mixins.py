"""Mounted Workbench controls use the production selection owner."""

from __future__ import annotations

from typing import ClassVar

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prompt_mixin_service import ConsolePromptMixinService
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.UI.Console_Modules.prompts import (
    _ConsolePromptMixinsFlow,
    _ConsolePromptSource,
)
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def runtime(tmp_path):
    database = PromptsDatabase(tmp_path / "prompts.db", client_id="workbench-test")
    scope = PromptScopeService(
        local_service=LocalPromptService(database), server_service=None
    )
    store = ConsoleChatStore()
    origin = store.create_session(session_id="origin", ephemeral=True)
    other = store.create_session(session_id="other", ephemeral=True)
    store.switch_session(origin.id)
    service = ConsolePromptMixinService(
        store=store, prompt_scope_service=scope, persistence=None
    )
    yield database, scope, store, origin, other, service
    for session in store.sessions():
        store.close_session(session.id)
    store.end_app_runtime()
    database.close_connection()


class WorkbenchHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = list(APP_STYLESHEETS)

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def open_workbench(self):
        _, scope, store, origin, _, service = self.runtime
        source = _ConsolePromptSource(scope)
        self.flow = _ConsolePromptMixinsFlow(
            service=service,
            session_id=origin.id,
            is_current=lambda: store.active_session_id == origin.id,
        )
        self.modal = ConsolePromptsModal(
            capabilities=source.capabilities,
            list_page=source.list_page,
            search=source.search,
            detail=source.detail,
            save=source.save,
            initial_mode="mixins",
            mixin_flow=self.flow,
        )
        self.push_screen(self.modal)

    def on_mount(self):
        self.open_workbench()


async def ready(app):
    await app.pilot.pause()
    await app.workers.wait_for_complete()
    await _wait_for_condition(
        app.pilot,
        lambda: bool(app.screen.query("#console-mixins-add")),
        message="Mix-ins did not load",
    )


async def add(app, name):
    from textual.widgets import Select

    select = app.screen.query_one("#console-mixins-library", Select)
    key = next(value for label, value in select._options if str(label) == name)
    select.value = key
    app.screen.query_one("#console-mixins-add", Button).press()
    await ready(app)
    assert any(
        row.name == name for row in app.runtime[-1].preferences("origin").entries
    ), str(app.screen.query_one("#console-mixins-status").render())


async def seed(scope):
    for name, placement in [
        ("Style", "before"),
        ("Summary", "after"),
        ("Diagram", "after"),
    ]:
        await scope.save_prompt(
            mode="local",
            name=name,
            artifact_type="mixin",
            prompt_format="structured",
            prompt_schema_version=3,
            prompt_definition={
                "kind": "text_mixin",
                "schema_version": 3,
                "body": name + " literal {body}\n",
                "placement": placement,
            },
        )


@pytest.mark.asyncio
async def test_mounted_selection_order_toggles_reopen_and_origin_fence(runtime):
    _, scope, store, origin, other, service = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        for name in ("Style", "Summary", "Diagram"):
            await add(app, name)
        checkbox = app.screen.query_one("#console-mixin-enabled-1")
        checkbox.focus()
        await pilot.press("space")
        await ready(app)
        up = app.screen.query_one("#console-mixin-up-2", Button)
        up.focus()
        await pilot.press("enter")
        await ready(app)
        prefs = service.preferences(origin.id)
        assert [row.name for row in prefs.entries] == ["Style", "Diagram", "Summary"]
        assert [row.enabled for row in prefs.entries] == [True, True, False]
        app.screen.query_one("#console-prompts-close", Button).press()
        await pilot.pause()
        app.open_workbench()
        await ready(app)
        assert service.preferences(origin.id) == prefs
        store.switch_session(other.id)
        app.screen.query_one("#console-mixin-remove-0", Button).press()
        await ready(app)
        assert service.preferences(origin.id) == prefs
        assert not service.preferences(other.id).entries


@pytest.mark.asyncio
async def test_deleted_disabled_and_enabled_choices_keep_count_and_preview_is_read_only(
    runtime,
):
    from dataclasses import replace

    from textual.widgets import Static

    database, scope, _, origin, _, service = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(80, 24)) as pilot:
        app.pilot = pilot
        await ready(app)
        for name in ("Style", "Summary"):
            await add(app, name)
        prefs = service.preferences(origin.id)
        service.replace_preferences(
            origin.id,
            expected_revision=prefs.revision,
            value=replace(
                prefs,
                entries=(prefs.entries[0], replace(prefs.entries[1], enabled=False)),
            ),
        )
        prefs = service.preferences(origin.id)
        before = prefs
        from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
            MixinWorkbenchIntent,
        )

        snapshot, _ = await app.flow.preview(
            MixinWorkbenchIntent(
                origin.id, origin.incarnation_id, prefs.revision, "inspect"
            )
        )
        assert snapshot.items[0].definition.body == "Style literal {body}\n"
        assert service.preferences(origin.id) == before
        for row in prefs.entries:
            database.delete_prompt(row.source.record_uuid)
        await app.modal._refresh_mixins()
        view = app.screen.query_one("ConsolePromptMixinsView")
        assert len(view.projection.unavailable) == 2
        assert (
            str(view.query_one("#console-mixins-count", Static).render())
            == "Mix-ins · 1 enabled"
        )
        view.query_one("#console-mixins-preview", Button).press()
        await ready(app)
        assert "unavailable" in str(
            view.query_one("#console-mixins-status", Static).render()
        )
        assert service.preferences(origin.id) == prefs


@pytest.mark.asyncio
async def test_shared_editor_dirty_close_invalid_save_and_version_change_preserve_draft(
    runtime,
):
    from textual.widgets import Input, Static, TextArea

    _, scope, _, origin, _, service = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        await add(app, "Style")
        app.screen.query_one("#console-mixin-inspect-0", Button).press()
        await ready(app)
        app.screen.query_one("#mixin-body", TextArea).load_text("")
        app.screen.query_one("#console-mixins-save", Button).press()
        await ready(app)
        assert "not saved" in str(
            app.screen.query_one("#console-mixins-status", Static).render()
        )
        assert app.screen.query_one("#mixin-body", TextArea).text == ""
        await pilot.press("escape")
        assert app.screen.query_one("#console-prompts-dirty-guard").display
        app.screen.query_one("#console-prompts-keep-editing", Button).press()
        await pilot.pause()
        app.screen.query_one("#mixin-body", TextArea).load_text(
            "Changed exact {body}\n"
        )
        app.screen.query_one("#mixin-name", Input).value = "Style updated"
        app.screen.query_one("#console-mixins-save", Button).press()
        await ready(app)
        source = service.preferences(origin.id).entries[0].source
        records = await scope.read_mixin_records(
            mode="local", record_uuids=(source.record_uuid,)
        )
        assert records[0]["name"] == "Style updated"
        assert records[0]["version"] == 2
        assert service.preferences(origin.id).entries[0].last_used_version is None


@pytest.mark.asyncio
async def test_source_replacement_and_stale_revision_refuse_captured_intent(runtime):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_prompt_mixins import MixinSourceChangedError

    _, scope, _, origin, _, service = runtime
    await seed(scope)
    flow = _ConsolePromptMixinsFlow(
        service=service, session_id=origin.id, is_current=lambda: True
    )
    projection = await flow.project()
    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        MixinWorkbenchIntent,
    )

    intent = MixinWorkbenchIntent(
        origin.id, origin.incarnation_id, 0, "add", projection.candidates[0][1]
    )
    await flow.apply(intent)
    with pytest.raises(MixinSourceChangedError):
        await flow.apply(intent)
    intent = replace(intent, revision=1)
    service.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(runtime[0]), server_service=None
    )
    with pytest.raises(MixinSourceChangedError):
        await flow.apply(intent)
    service.prompt_scope_service = scope


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (100, 30), (80, 24)])
async def test_real_console_prompts_action_retains_browse_and_controls_runtime(
    runtime, size
):
    from Tests.UI.app_factory import attach_chachanotes_db
    from Tests.UI.test_console_prompts_controller import (
        ConsoleHarness,
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_library_prompts_canvas import _build_test_app
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
    from tldw_chatbook.Widgets.Console.console_composer_menu_modal import ACTION_PROMPTS

    _, scope, _, _, _, _ = runtime
    await seed(scope)
    application = _build_test_app()
    attach_chachanotes_db(application)
    _configure_native_ready_console(application)
    application.prompt_scope_service = scope

    class NativeHarness(ConsoleHarness):
        CSS_PATH = TldwCli.CSS_PATH

    host = NativeHarness(application)
    try:
        async with host.run_test(size=size) as pilot:
            host.pilot = pilot
            console = host.screen
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            origin = store.active_session_id
            service = console._prompts._mixin_service()
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            composer.insert_text("Original draft")
            system = store.session_settings(origin).system_prompt
            console._handle_console_composer_menu_choice(ACTION_PROMPTS)
            await _wait_for_condition(
                pilot,
                lambda: bool(host.screen.query("#console-prompts-mixins")),
                message="Prompts action did not open Browse",
            )
            assert host.screen.state.mode == "browse"
            assert await pilot.click("#console-prompts-mixins")
            await ready(host)
            # Reuse the real runtime accessor when asserting Add controls.
            host.runtime = (None, scope, store, store._sessions[origin], None, service)
            select = host.screen.query_one("#console-mixins-library")
            select.value = next(
                value for label, value in select._options if str(label) == "Style"
            )
            host.screen.query_one("#console-mixins-add", Button).press()
            await ready(host)
            assert [row.name for row in service.preferences(origin).entries] == [
                "Style"
            ]
            for name in ("Summary", "Diagram"):
                select = host.screen.query_one("#console-mixins-library")
                select.value = next(
                    value for label, value in select._options if str(label) == name
                )
                host.screen.query_one("#console-mixins-add", Button).press()
                await ready(host)
            checkbox = host.screen.query_one("#console-mixin-enabled-1")
            checkbox.focus()
            await pilot.press("space")
            await ready(host)
            host.screen.query_one("#console-mixin-up-2", Button).focus()
            await pilot.press("enter")
            await ready(host)
            assert [row.name for row in service.preferences(origin).entries] == [
                "Style",
                "Diagram",
                "Summary",
            ]
            assert [row.enabled for row in service.preferences(origin).entries] == [
                True,
                True,
                False,
            ]
            assert composer.draft_text() == "Original draft"
            assert store.session_settings(origin).system_prompt == system
            before_preview = service.preferences(origin)
            host.screen.query_one("#console-mixins-preview", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    "Exact next-send preview"
                    in str(
                        host.screen.query_one(
                            "#console-mixins-preview-content"
                        ).render()
                    )
                ),
                message="Next-send preview did not finish",
            )
            preview = str(
                host.screen.query_one("#console-mixins-preview-content").render()
            )
            assert "Style literal" in preview and "Diagram literal" in preview
            assert "Summary literal" not in preview and "Original draft" in preview
            assert service.preferences(origin) == before_preview
            import os

            capture_dir = os.environ.get("TASK8_CAPTURE_DIR")
            if capture_dir:
                from pathlib import Path

                Path(capture_dir).mkdir(parents=True, exist_ok=True)
                host.save_screenshot(
                    path=capture_dir, filename=f"workbench-{size[0]}x{size[1]}.svg"
                )
            host.screen.query_one("#console-mixin-enabled-0").focus()
            await pilot.press("tab")
            assert host.focused.id == "console-mixin-inspect-0"
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: bool(host.screen.query("#mixin-name")),
                message="Shared editor did not open",
            )
            await host.workers.wait_for_complete()
            await pilot.pause()
            if capture_dir:
                host.save_screenshot(
                    path=capture_dir, filename=f"editor-{size[0]}x{size[1]}.svg"
                )
            await pilot.press("escape")
            await pilot.pause()
            assert host.screen is console
            await _wait_for_condition(
                pilot,
                lambda: host.focused is composer,
                message="Composer opener focus was not restored",
            )
    finally:
        await application._shutdown_app_owned_lifecycles()
        await application.evaluation_orchestrator.aclose()
        application.local_workspace_db.close()
        application.local_library_collections_db.close()
        application.subscriptions_db.close()


@pytest.mark.asyncio
async def test_live_unsaved_choices_retry_existing_settings_component(runtime):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_chat_store import ConsoleSettingsComponent

    _, scope, store, origin, _, service = runtime
    await seed(scope)

    class FailingPersistence:
        fail = True

        def save_conversation_mixin_preferences(
            self, *, conversation_id, expected_revision, value
        ):
            if self.fail:
                raise OSError("injected storage failure")
            return replace(value, revision=expected_revision + 1)

    persistence = FailingPersistence()
    store.persistence = persistence
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        await add(app, "Style")
        origin.ephemeral = False
        origin.persisted_conversation_id = "saved-origin"
        outcome = await store.persist_prompt_mixin_preferences(origin.id)
        assert ConsoleSettingsComponent.PROMPT_MIXINS in outcome.failed_components
        await app.modal._refresh_mixins()
        retry = app.screen.query_one("#console-mixins-retry", Button)
        assert retry.display
        prefs = service.preferences(origin.id)
        persistence.fail = False
        retry.focus()
        await pilot.press("enter")
        await ready(app)
        assert not app.screen.query_one("#console-mixins-retry", Button).display
        assert service.preferences(origin.id).entries == prefs.entries


@pytest.mark.asyncio
async def test_external_placement_edit_moves_disabled_choice_once_to_group_end(runtime):
    from dataclasses import replace

    _, scope, _, origin, _, service = runtime
    await seed(scope)
    flow = _ConsolePromptMixinsFlow(
        service=service, session_id=origin.id, is_current=lambda: True
    )
    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        MixinWorkbenchIntent,
    )

    for name in ("Style", "Summary", "Diagram"):
        projection = await flow.project()
        ref = next(ref for label, ref in projection.candidates if label == name)
        await flow.apply(
            MixinWorkbenchIntent(
                origin.id,
                origin.incarnation_id,
                projection.preferences.revision,
                "add",
                ref,
            )
        )
    prefs = service.preferences(origin.id)
    service.replace_preferences(
        origin.id,
        expected_revision=prefs.revision,
        value=replace(
            prefs,
            entries=(replace(prefs.entries[0], enabled=False), *prefs.entries[1:]),
        ),
    )
    prefs = service.preferences(origin.id)
    await scope.save_prompt(
        mode="local",
        prompt_identifier=prefs.entries[0].source.record_uuid,
        expected_version=1,
        name="Style",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "new body",
            "placement": "after",
        },
    )
    projected = await flow.project()
    assert [row.name for row in projected.preferences.entries] == [
        "Summary",
        "Diagram",
        "Style",
    ]
    assert not projected.preferences.entries[-1].enabled
    assert projected.preferences.entries[-1].last_used_version is None
    revision = projected.preferences.revision
    assert (await flow.project()).preferences.revision == revision


@pytest.mark.asyncio
async def test_held_refresh_cannot_overwrite_newer_toggle(runtime):
    import asyncio
    from dataclasses import replace

    _, scope, _, origin, _, service = runtime
    await seed(scope)
    flow = _ConsolePromptMixinsFlow(
        service=service, session_id=origin.id, is_current=lambda: True
    )
    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        MixinWorkbenchIntent,
    )

    projection = await flow.project()
    await flow.apply(
        MixinWorkbenchIntent(
            origin.id, origin.incarnation_id, 0, "add", projection.candidates[0][1]
        )
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = scope.read_mixin_records

    async def held(**kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return await original(**kwargs)

    scope.read_mixin_records = held
    pending = asyncio.create_task(flow.project())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        prefs = service.preferences(origin.id)
        wanted = service.replace_preferences(
            origin.id,
            expected_revision=prefs.revision,
            value=replace(prefs, entries=(replace(prefs.entries[0], enabled=False),)),
        )
        release.set()
        with pytest.raises(ValueError, match="mixin_workbench_revision_changed"):
            await pending
        assert service.preferences(origin.id) == wanted
    finally:
        release.set()
        scope.read_mixin_records = original
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_held_projection_cannot_replace_browse_after_navigation(runtime, failed):
    import asyncio

    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        ConsolePromptMixinsView,
    )
    from tldw_chatbook.Widgets.Console.console_prompts_browse import (
        ConsolePromptsBrowse,
    )

    _, scope, _, _, _, _ = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        entered, release = asyncio.Event(), asyncio.Event()
        original = app.flow.project

        async def held(**kwargs):
            projected = await original(**kwargs)
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            if failed:
                raise ValueError("injected projection failure")
            return projected

        app.flow.project = held
        worker = app.modal.run_worker(
            app.modal._refresh_mixins(), group="console-mixins"
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await app.modal.enter_mode("browse")
            browse = app.modal.query_one(ConsolePromptsBrowse)
            release.set()
            await worker.wait()
            assert app.modal.state.mode == "browse"
            assert app.modal.query_one(ConsolePromptsBrowse) is browse
            assert not app.modal.query(ConsolePromptMixinsView)
        finally:
            release.set()


@pytest.mark.asyncio
async def test_held_save_preserves_newer_raw_editor_draft_and_can_save_again(runtime):
    import asyncio

    from textual.widgets import Input, Select, TextArea

    from tldw_chatbook.Widgets.Prompts.prompt_mixin_editor import PromptMixinEditor

    _, scope, _, origin, _, service = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        await add(app, "Style")
        app.modal.query_one("#console-mixin-inspect-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(app.modal.query("#mixin-placement #label")),
            message="Editor placement control did not finish mounting",
        )
        editor = app.modal.query_one(PromptMixinEditor)
        editor.query_one("#mixin-name", Input).value = "Saved Style"
        editor.query_one("#mixin-placement", Select).value = "after"
        editor.query_one("#mixin-body", TextArea).load_text("saved first draft")
        entered, release = asyncio.Event(), asyncio.Event()
        original = app.flow.save

        async def held(*args):
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            return await original(*args)

        app.flow.save = held
        app.modal.query_one("#console-mixins-save", Button).press()
        try:
            await asyncio.wait_for(entered.wait(), 5)
            editor.query_one("#mixin-body", TextArea).load_text("newer unsaved draft")
            release.set()
            await app.workers.wait_for_complete()
            assert app.modal.query_one(PromptMixinEditor) is editor
            assert editor.read_draft()[1] == "newer unsaved draft"
            assert app.modal._holds_dirty_edit()
            reference = service.preferences(origin.id).entries[0].source
            records = await scope.read_mixin_records(
                mode="local", record_uuids=(reference.record_uuid,)
            )
            assert records[0]["version"] == 2
            assert records[0]["user_prompt"] == "saved first draft"
            app.flow.save = original
            app.modal.query_one("#console-mixins-save", Button).press()
            await ready(app)
            records = await scope.read_mixin_records(
                mode="local", record_uuids=(reference.record_uuid,)
            )
            assert records[0]["version"] == 3
            assert records[0]["user_prompt"] == "newer unsaved draft"
            assert not app.modal.query(PromptMixinEditor)
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["inspect", "save", "preview"])
@pytest.mark.parametrize("failed", [False, True])
async def test_held_completion_cannot_publish_into_new_workbench_mode(
    runtime, operation, failed
):
    import asyncio

    from textual.widgets import TextArea

    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        ConsolePromptMixinsView,
    )
    from tldw_chatbook.Widgets.Console.console_prompts_browse import (
        ConsolePromptsBrowse,
    )
    from tldw_chatbook.Widgets.Prompts.prompt_mixin_editor import PromptMixinEditor

    _, scope, _, _, _, _ = runtime
    await seed(scope)
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        await add(app, "Style")
        if operation == "save":
            app.modal.query_one("#console-mixin-inspect-0", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: bool(app.modal.query(PromptMixinEditor)),
                message="Editor did not mount",
            )
            await app.workers.wait_for_complete()
            app.modal.query_one("#mixin-body", TextArea).load_text(
                "Explicitly saved edit"
            )
        entered, release = asyncio.Event(), asyncio.Event()
        original = getattr(app.flow, operation)

        async def held(*args, **kwargs):
            try:
                result = await original(*args, **kwargs)
            except Exception as exc:
                app.delayed_error = exc
                entered.set()
                raise
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            if failed:
                raise ValueError("injected delayed completion failure")
            return result

        setattr(app.flow, operation, held)
        view = app.modal.query_one(ConsolePromptMixinsView)
        pending = (
            app.modal._save_mixin_editor()
            if operation == "save"
            else app.modal._preview_mixins()
            if operation == "preview"
            else app.modal._apply_mixin_intent(
                view.intent("inspect", view.projection.preferences.entries[0].source)
            )
        )
        worker = app.modal.run_worker(pending, group="console-mixins")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not hasattr(app, "delayed_error"), repr(
                getattr(app, "delayed_error", None)
            )
            await app.modal.enter_mode("browse")
            browse = app.modal.query_one(ConsolePromptsBrowse)
            release.set()
            await worker.wait()
            assert app.modal.state.mode == "browse"
            assert app.modal.query_one(ConsolePromptsBrowse) is browse
            assert not app.modal.query(ConsolePromptMixinsView)
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["server", "local"])
async def test_ordinary_browse_mixin_capability_refusal_and_local_open(runtime, source):
    from textual.widgets import Static

    from Tests.Prompt_Management.test_prompt_scope_service import (
        FakeServerPromptService,
        modern_prompt_health,
    )
    from tldw_chatbook.Widgets.Console.console_prompt_mixins_view import (
        ConsolePromptMixinsView,
    )
    from tldw_chatbook.Widgets.Console.console_prompts_browse import (
        ConsolePromptsBrowse,
    )

    db, scope, _, _, _, _ = runtime
    await seed(scope)
    record = db.get_prompt_by_id(db.get_all_active_prompt_ids()[0])
    health = modern_prompt_health(
        structured_kinds=[{"schema_version": 3, "kind": "text_mixin"}]
    )
    health["capabilities"]["artifact_types"].append("mixin")
    server = FakeServerPromptService(health=health)
    server.prompt = record
    scope.server_service = server
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        app.modal.state = app.modal.state.with_source(source)
        await app.modal.enter_mode("browse")
        await pilot.pause()
        browse = app.modal.query_one(ConsolePromptsBrowse)
        await app.modal.open_artifact(
            record["name"] if source == "local" else str(record["id"])
        )
        await pilot.pause()
        if source == "server":
            assert app.modal.state.mode == "browse"
            assert app.modal.state.source == "server"
            assert app.modal.query_one(ConsolePromptsBrowse) is browse
            assert "cannot capture Mix-ins" in str(
                browse.query_one("#console-prompts-browse-status", Static).render()
            )
            assert not app.modal.query(ConsolePromptMixinsView)
        else:
            assert app.modal.state.mode == "mixins"
            assert app.modal.query("PromptMixinEditor")


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["stale", "failure"])
async def test_browse_mixin_open_awaited_transition_is_fenced(
    runtime, monkeypatch, completion
):
    import asyncio

    from tldw_chatbook.Widgets.Console.console_prompts_browse import (
        ConsolePromptsBrowse,
    )

    db, scope, _, _, _, _ = runtime
    await seed(scope)
    record = db.get_prompt_by_id(db.get_all_active_prompt_ids()[0])
    app = WorkbenchHarness(runtime)
    async with app.run_test(size=(100, 30)) as pilot:
        app.pilot = pilot
        await ready(app)
        await app.modal.enter_mode("browse")
        entered, release = asyncio.Event(), asyncio.Event()
        original = app.flow.project

        async def held(**kwargs):
            entered.set()
            await release.wait()
            if completion == "failure":
                raise ValueError("source unavailable")
            return await original(**kwargs)

        monkeypatch.setattr(app.flow, "project", held)
        task = asyncio.create_task(app.modal.open_artifact(record["name"]))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if completion == "stale":
                await app.modal.enter_mode("browse")
        finally:
            release.set()
        await asyncio.wait_for(task, 5)
        assert not app.modal.query("PromptMixinEditor")
        if completion == "stale":
            assert app.modal.state.mode == "browse"
            assert app.modal.query(ConsolePromptsBrowse)
