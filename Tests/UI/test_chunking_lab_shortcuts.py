"""Mounted Lab hints/help follow state without taking over editor typing."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from Tests.private_profile import private_profile_test


@asynccontextmanager
async def mounted_lab(tmp_path, monkeypatch, size=(140, 45), *, fail_load=False):
    # Imports belong after private_profile_test has selected the child profile.
    from textual.binding import Binding

    from Tests.Chunking.test_lab_coordinator import setup
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from Tests.UI.test_chunking_lab_screen import settle_lab
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route

    coordinator, runner, writer, store, _events = setup(tmp_path)

    class LabHarness(ConsolidatedCSSApp):
        # Use the real global delegates without starting unrelated app services.
        BINDINGS = [
            Binding("f1", "show_workbench_help"),
            Binding("f6", "focus_next_workbench_pane"),
        ]
        action_show_workbench_help = LifecycleMixin.action_show_workbench_help
        action_focus_next_workbench_pane = (
            LifecycleMixin.action_focus_next_workbench_pane
        )
        _show_generic_screen_help = TldwCli._show_generic_screen_help

        def on_navigate_to_screen(self, event):
            self.navigation_requests.append(event.screen_name)

    inner = _build_test_app()
    app = LabHarness()
    app.navigation_requests = []

    async def owner():
        nonlocal fail_load
        if fail_load:
            fail_load = False
            raise OSError("checkpoint unavailable")
        return coordinator

    monkeypatch.setattr(inner, "get_chunking_lab_coordinator", owner)
    try:
        async with app.run_test(size=size) as pilot:
            screen = resolve_screen_route("chunking_lab").load_screen_class()(inner)
            await app.push_screen(screen)
            await screen.wait_until_ready()
            await settle_lab(app, screen, pilot)
            screen.query_one("#lab-show-sample").focus()
            await pilot.pause()
            yield app, screen, pilot, runner, writer, store
    finally:
        runner.release.set()
        store.fail_save = False
        await coordinator.close()


def hints(screen):
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

    footer = screen.query_one(AppFooterStatus)
    actions = {
        action.key: action.label
        for action in footer._context_actions
        if action.available
    }
    for key, label in (("r", "run B"), ("p", "pin A"), ("s", "save B")):
        if key not in actions:
            assert f"{key} {label}" not in footer.shortcut_text
    return actions


async def wait_for(pilot, predicate):
    async def changed():
        while not predicate():
            await pilot.pause()

    await asyncio.wait_for(changed(), 3)


@private_profile_test
async def test_busy_hints_refresh_without_focus_change(request, tmp_path, monkeypatch):
    from textual.widgets import Button

    async with mounted_lab(tmp_path, monkeypatch, (80, 24)) as (
        app,
        screen,
        pilot,
        runner,
        *_,
    ):
        focused = app.focused
        assert {"r", "p", "s"} <= hints(screen).keys()
        runner.release.clear()
        screen.action_lab_run()
        try:
            await asyncio.wait_for(runner.started.wait(), 3)
            # The manifest guard retires before the runner starts; wait for its
            # published authoring controls, not the transient guarded render.
            await wait_for(
                pilot,
                lambda: (
                    screen.query_one("#lab-run", Button).disabled
                    and not screen.query_one("#lab-save-b", Button).disabled
                ),
            )
            assert app.focused is focused
            assert "r" not in hints(screen)
            assert "s" in hints(screen)  # Saving B remains useful during a run.
            await pilot.press("f1")
            assert "r" not in dict(app.screen.state.shortcuts)
            assert "Preview running" in app.screen.state.render_text()
            await pilot.press("f1")
        finally:
            runner.release.set()
        await wait_for(pilot, lambda: not screen.coordinator.busy)
        await wait_for(pilot, lambda: "r" in hints(screen))
        assert app.focused is focused


@private_profile_test
async def test_recovery_and_leaving_refresh_hints_without_focus_change(
    request, tmp_path, monkeypatch
):
    from textual.widgets import Button

    from tldw_chatbook.Chunking.lab_recovery import export_recovery

    async with mounted_lab(tmp_path, monkeypatch) as (
        app,
        screen,
        pilot,
        _,
        writer,
        store,
    ):
        entered, release = asyncio.Event(), asyncio.Event()
        replace = writer.replace

        async def held_replace(*args, **kwargs):
            entered.set()
            await release.wait()
            return await replace(*args, **kwargs)

        monkeypatch.setattr(writer, "replace", held_replace)
        restoring = asyncio.create_task(
            screen.coordinator.replace_recovery(
                export_recovery(screen.coordinator.session)
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await wait_for(pilot, lambda: screen.query_one("#lab-run", Button).disabled)
            assert not {"r", "p", "s"} & hints(screen).keys()
            await pilot.press("f1")
            assert not {"r", "p", "s"} & dict(app.screen.state.shortcuts).keys()
            assert "transition" in app.screen.state.render_text()
            await pilot.press("f1")
        finally:
            release.set()
            await restoring
        await wait_for(pilot, lambda: "r" in hints(screen))
        drain = screen.drain_edits
        entered.clear()
        release.clear()

        async def held_drain():
            entered.set()
            await release.wait()
            await drain()

        monkeypatch.setattr(screen, "drain_edits", held_drain)
        leaving = asyncio.create_task(screen.confirm_navigation())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await pilot.pause()
            assert not {"r", "p", "s"} & hints(screen).keys()
            assert screen.query_one("#lab-run", Button).disabled
        finally:
            release.set()
            assert await leaving
        await wait_for(pilot, lambda: "r" in hints(screen))
        monkeypatch.setattr(screen, "drain_edits", drain)
        from tldw_chatbook.Chunking import lab_state

        store.fail_save = True
        screen.queue_edit(
            lambda session: lab_state.replace_sample(
                session, "unsaved", {"kind": "paste"}
            )
        )
        await screen.drain_edits()
        assert not await screen.confirm_navigation()
        assert "Could not save local recovery" in str(
            screen.query_one("#lab-message").content
        )
        assert not screen._leaving
        assert not screen.query_one("#lab-run", Button).disabled
        assert {"r", "s"} <= hints(screen).keys()
        store.fail_save = False


@private_profile_test
async def test_candidate_readiness_hides_pin_hint_but_keeps_refusal_guidance(
    request, tmp_path, monkeypatch
):
    from textual.widgets import Button

    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.Chunking.lab_recovery import export_recovery

    async with mounted_lab(tmp_path, monkeypatch) as (
        app,
        screen,
        pilot,
        _runner,
        _writer,
        store,
    ):
        await screen.coordinator.replace_recovery(
            export_recovery(lab_state.new_session("profile"))
        )
        # Wait for the replacement guard's idle control publication.
        await wait_for(
            pilot,
            lambda: (
                screen.query_one("#lab-both", Button).disabled
                and not screen.query_one("#lab-run", Button).disabled
            ),
        )
        assert "p" not in hints(screen)
        assert "r" in hints(screen) and "s" in hints(screen)
        assert not screen.query_one("#lab-pin", Button).disabled
        assert screen.check_action("lab_pin", ()) is not False
        await pilot.press("p")
        await wait_for(
            pilot,
            lambda: "Pin requires" in str(screen.query_one("#lab-message").content),
        )
        before = screen.coordinator.session
        screen._message("Waiting for direct Pin guidance")
        await screen._pin()
        assert (
            str(screen.query_one("#lab-message").content)
            == "Pin requires a completed current B result"
        )
        screen._message("Waiting for Pin button guidance")
        assert await pilot.click("#lab-pin")
        await wait_for(
            pilot,
            lambda: str(screen.query_one("#lab-message").content)
            == "Pin requires a completed current B result",
        )
        assert screen.coordinator.session.candidates == before.candidates
        assert screen.coordinator.session.results == before.results

        def read_checkpoint():
            from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore

            reader = CheckpointStore(store.path, store.profile_key)
            try:
                return reader.load()[0]
            finally:
                reader.close()

        persisted = await asyncio.to_thread(read_checkpoint)
        assert persisted.candidates == before.candidates
        assert persisted.results == before.results
        await screen.run_candidates()
        await wait_for(pilot, lambda: "p" in hints(screen))
        screen.queue_edit(
            lambda session: lab_state.replace_sample(
                session, "changed", {"kind": "paste"}
            )
        )
        await screen.drain_edits()
        await wait_for(pilot, lambda: "p" not in hints(screen))
        await pilot.press("f1")
        assert "p" not in dict(app.screen.state.shortcuts)
        assert "completed current B result" in app.screen.state.render_text()
        await pilot.press("f1")
        screen.queue_edit(
            lambda session: lab_state.edit_json(session, screen._b_id(session), "{")
        )
        await screen.drain_edits()
        assert {"r", "s"} <= hints(screen).keys()
        invalid = screen.coordinator.session
        run_refusal = f"Candidate {screen._b_id()} is not executable"
        await pilot.press("r")
        await wait_for(
            pilot, lambda: str(screen.query_one("#lab-message").content) == run_refusal
        )
        with pytest.raises(ValueError, match=f"^{run_refusal}$"):
            await screen.run_candidates()
        # A rejected capture can briefly publish the manifest guard; await its
        # idle control publication before activating the button.
        await wait_for(
            pilot,
            lambda: not screen.coordinator.busy
            and not screen.query_one("#lab-run", Button).disabled,
        )
        screen._message("Waiting for button validation")
        assert await pilot.click("#lab-run")
        await wait_for(
            pilot, lambda: str(screen.query_one("#lab-message").content) == run_refusal
        )
        save_refusal = "Correct or discard the current invalid edit before saving B"
        await pilot.press("s")
        await wait_for(
            pilot, lambda: str(screen.query_one("#lab-message").content) == save_refusal
        )
        with pytest.raises(ValueError, match=f"^{save_refusal}$"):
            await screen._save("B")
        screen._message("Waiting for save validation")
        assert await pilot.click("#lab-save-b")
        await wait_for(
            pilot, lambda: str(screen.query_one("#lab-message").content) == save_refusal
        )
        assert screen.coordinator.session.results == invalid.results
        assert screen.coordinator.session.candidates == invalid.candidates
        assert (
            screen.coordinator.session.candidates[screen._b_id()]["draft"]["raw_json"]
            == "{"
        )


@private_profile_test
async def test_failed_recovery_load_keeps_retry_and_read_only_help(
    request, tmp_path, monkeypatch
):
    from textual.widgets import Button

    async with mounted_lab(tmp_path, monkeypatch, fail_load=True) as (
        app,
        screen,
        pilot,
        *_,
    ):
        assert screen.coordinator is None
        assert not {"r", "p", "s"} & hints(screen).keys()
        assert not screen.query_one("#lab-retry", Button).disabled
        assert screen.query_one("#lab-run", Button).disabled
        await pilot.press("f1")
        assert "read-only" in app.screen.state.render_text()
        assert not {"r", "p", "s"} & dict(app.screen.state.shortcuts).keys()
        await pilot.press("f1")
        with pytest.raises(ValueError, match="Load local recovery"):
            await screen.run_candidates()
        assert await pilot.click("#lab-retry")
        await wait_for(pilot, lambda: not screen.query_one("#lab-run", Button).disabled)
        assert screen.coordinator is not None
        assert {"r", "p", "s"} <= hints(screen).keys()


@pytest.mark.parametrize("size", [(80, 24), (140, 45)], ids=["compact", "wide"])
@private_profile_test
async def test_editor_typing_and_f1_describe_scope_at_each_width(
    request, tmp_path, monkeypatch, size
):
    from textual.widgets import Input, Select, TextArea

    async with mounted_lab(tmp_path, monkeypatch, size) as (app, screen, pilot, *_):
        for selector, kind in (
            ("#lab-sample-text", TextArea),
            ("#lab-name", Input),
            ("#lab-menu", Select),
        ):
            if selector == "#lab-name":
                screen._active_region = "configure"
                screen._layout()
            editor = screen.query_one(selector, kind)
            editor.focus()
            await pilot.pause()
            assert app.focused is editor
            assert not {"r", "p", "s"} & hints(screen).keys()
            if kind is Select:
                await pilot.press("enter")
                assert editor.expanded
                await pilot.press("escape")
                assert not editor.expanded and app.screen is screen
                assert app.navigation_requests == []
            else:
                before = editor.text if kind is TextArea else editor.value
                await pilot.press("r", "p", "s")
                after = editor.text if kind is TextArea else editor.value
                assert len(after) == len(before) + 3 and "rps" in after
                await screen.drain_edits()
            await pilot.press("f1")
            assert app.screen.state.route_id == "chunking_lab"
            assert not {"r", "p", "s"} & dict(app.screen.state.shortcuts).keys()
            body = app.screen.state.render_text()
            assert "Input, TextArea and Select" in body
            assert "Tab" in body and "F1" in body
            assert (
                app.screen.query_one("#workbench-help-close").region
                in app.screen.content_region
            )
            await pilot.press("f1")
            assert app.screen is screen
        screen.query_one("#lab-show-sample").focus()
        await pilot.pause()
        assert "r" in hints(screen)
        await pilot.press("f1")
        assert "r" in dict(app.screen.state.shortcuts)
        await pilot.press("escape")
        assert app.screen is screen  # Escape closes help safely before Lab navigation.
        assert app.navigation_requests == []
        screen._active_region = "sample"
        screen._layout()
        screen.query_one("#lab-sample-text").focus()
        await pilot.pause()
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.navigation_requests == ["library"])
