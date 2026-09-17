"""Retained import options reject events superseded by reset or replacement."""

import pytest
from textual.widgets import Button, Checkbox, Input, Select, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_ingest_canvas import _CanvasHost, _MessageRecordingHost
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,widget_type,edited,reset_value",
    [
        ("chunk_size", Input, "2000", "1000"),
        ("analyze", Checkbox, True, False),
        ("encoding", Select, "utf-8", "auto"),
    ],
)
async def test_queued_option_change_cannot_restore_value_after_group_reset(
    name, widget_type, edited, reset_value
):
    """An old Changed snapshot must not undo a newer reset of its live field."""
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(title="Unrelated draft")
    )
    host = _MessageRecordingHost(state)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(LibraryIngestCanvas)
        control = canvas.query_one(f"#opt-generic-{name}", widget_type)
        title = canvas.query_one("#library-ingest-title", Input)
        assert host.option_changes == []

        # No yield between the edit and reset: the real widget's Changed
        # message is already queued when the reset changes its current value.
        control.value = edited
        canvas.sync_option_group("generic", state)
        assert control.value == reset_value
        await pilot.pause()

        assert host.option_changes == [], [
            (event.group, event.name, event.value) for event in host.option_changes
        ]
        assert canvas.query_one(f"#opt-generic-{name}") is control
        assert control.value == reset_value
        assert canvas.query_one("#library-ingest-title") is title
        assert title.value == "Unrelated draft"

        # Admission must still accept a genuine edit of the retained widget.
        control.value = edited
        await pilot.pause()
        assert [
            (event.group, event.name, event.value) for event in host.option_changes
        ] == [("generic", name, edited)]


@pytest.mark.asyncio
async def test_replaced_option_sender_cannot_edit_current_form():
    """A Changed message already bubbling during replacement owns no new field."""
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _MessageRecordingHost(state)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(LibraryIngestCanvas)
        previous = canvas.query_one("#opt-generic-chunk_size", Input)
        with previous.prevent(Input.Changed):
            previous.value = "2000"
        stale = Input.Changed(previous, "2000")

        canvas.sync_state(state)
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(canvas.query("#opt-generic-chunk_size"))
                and canvas.query_one("#opt-generic-chunk_size") is not previous
            ),
            message="Replacement option input did not mount",
        )
        await pilot.pause()
        current = canvas.query_one("#opt-generic-chunk_size", Input)
        assert current.value == "1000"
        assert not previous.is_attached
        assert host.option_changes == []

        # Deliver the real message at the receiving canvas after its sender
        # detached, as can happen when bubbling overlaps a recompose.
        canvas.post_message(stale)
        await pilot.pause()
        assert host.option_changes == [], [event.value for event in host.option_changes]
        assert current.value == "1000"

        current.value = "2500"
        await pilot.pause()
        assert [
            (event.group, event.name, event.value) for event in host.option_changes
        ] == [("generic", "chunk_size", "2500")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_backend,target_backend", [("local", "server"), ("server", "local")]
)
async def test_option_update_during_backend_recompose_settles_target_controls(
    initial_backend, target_backend
):
    """A pending backend recompose must not query new fields in the old body."""
    initial = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend=initial_backend,
        runtime_source="server",
        server_ingest_available=True,
    )
    assert initial.ingest_backend == initial_backend
    host = _CanvasHost(initial)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(LibraryIngestCanvas)
        target = build_library_ingest_state(
            (),
            form=LibraryIngestFormState(
                title="Current draft",
                analyze=True,
                chunk=False,
                type_options={"generic": {"analyze": True, "chunk": False}},
            ),
            ingest_backend=target_backend,
            runtime_source="server",
            server_ingest_available=True,
        )
        assert target.ingest_backend == target_backend

        # sync_state updates the snapshot before refresh(recompose=True)
        # replaces children. Deliver the group update inside that exact gap.
        canvas.sync_state(target)
        canvas.sync_option_group("generic", target)
        await pilot.pause()

        assert canvas.query_one("#opt-generic-analyze", Checkbox).value
        assert not canvas.query_one("#opt-generic-custom_prompt", TextArea).disabled
        assert not canvas.query_one("#opt-generic-chunk", Checkbox).value
        assert canvas.query_one("#opt-generic-chunk_size", Input).disabled
        assert canvas.query_one("#library-ingest-title", Input).value == "Current draft"
        assert bool(canvas.query("#opt-generic-keep_original_file")) == (
            target_backend == "server"
        )
        assert bool(canvas.query("#opt-generic-chunk_template")) == (
            target_backend == "local"
        )
        if target_backend == "local":
            assert canvas.query_one("#opt-generic-chunk_template", Select).disabled


@pytest.mark.asyncio
async def test_detached_canvas_reveal_callback_preserves_current_editor():
    """A reveal queued before leaving Import must not read a detached screen."""
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _CanvasHost(state)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(LibraryIngestCanvas)
        callback = canvas._reveal_focused_control
        current = Input("Current editor", id="current-editor")
        await host.screen.mount(current)
        current.focus()
        await pilot.pause()
        assert host.focused is current
        current.selection = type(current.selection)(1, 5)
        selection = current.selection

        await canvas.remove()
        assert not canvas.is_attached
        callback()

        assert host.focused is current
        assert current.selection == selection
        assert current.value == "Current editor"


@pytest.mark.asyncio
@private_profile_test
async def test_forwarded_option_edit_cannot_overtake_later_reset(request, monkeypatch):
    """Screen delivery must reject a real forwarded edit superseded by Reset."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryProductionCSSHarness,
        _seed_conversations,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        analyze = await _wait_for_selector(screen, pilot, "#opt-generic-analyze")
        await pilot.pause()
        canvas = screen.query_one(LibraryIngestCanvas)
        held = []
        forward = canvas.post_message

        def hold_option_edit(message):
            if isinstance(message, LibraryIngestCanvas.OptionValueChanged):
                held.append(message)
                return True
            return forward(message)

        # Hold only the new message produced by the real native Changed
        # handler. Reset and every other UI event still traverse production.
        monkeypatch.setattr(canvas, "post_message", hold_option_edit)
        analyze.value = True
        await _wait_for_condition(
            pilot,
            lambda: len(held) == 1,
            message="The real checkbox edit was not forwarded by the canvas",
        )
        assert (held[0].group, held[0].name, held[0].value) == (
            "generic",
            "analyze",
            True,
        )

        screen.query_one("#opt-generic-reset", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not analyze.value,
            message="Reset did not replace the pending checkbox value",
        )
        await pilot.pause()
        assert screen.query_one("#opt-generic-analyze") is analyze
        assert not screen._ingest_state.form.analyze

        # Deliver that same captured event after Reset reached the screen.
        monkeypatch.setattr(canvas, "post_message", forward)
        forward(held[0])
        await pilot.pause()
        assert not screen._ingest_state.form.analyze
        assert not screen._ingest_state.form.type_options["generic"].get(
            "analyze", False
        )
        assert not analyze.value

        # The identity/value fence must still admit the next genuine edit.
        analyze.value = True
        await _wait_for_condition(
            pilot,
            lambda: screen._ingest_state.form.analyze,
            message="A current checkbox edit was rejected after the stale event",
        )
        assert analyze.value


@pytest.mark.asyncio
@private_profile_test
async def test_option_dependency_refresh_keeps_later_pending_sibling_edit(
    request, monkeypatch
):
    """Delivering two real edits in order must retain the newer live editor."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryProductionCSSHarness,
        _seed_conversations,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        analyze = await _wait_for_selector(screen, pilot, "#opt-generic-analyze")
        await pilot.pause()
        canvas = screen.query_one(LibraryIngestCanvas)
        chunk_size = canvas.query_one("#opt-generic-chunk_size", Input)
        held = []
        forward = canvas.post_message

        def hold_option_edit(message):
            if isinstance(message, LibraryIngestCanvas.OptionValueChanged):
                held.append(message)
                return True
            return forward(message)

        monkeypatch.setattr(canvas, "post_message", hold_option_edit)
        analyze.value = True
        await _wait_for_condition(
            pilot,
            lambda: len(held) == 1,
            message="The first real option edit did not reach the forwarding gate",
        )
        chunk_size.focus()
        await pilot.pause()
        chunk_size.value = "2000"
        chunk_size.selection = type(chunk_size.selection)(1, 3)
        selection = chunk_size.selection
        await _wait_for_condition(
            pilot,
            lambda: len(held) == 2,
            message="The newer real editor change did not reach the forwarding gate",
        )
        assert [(event.name, event.value) for event in held] == [
            ("analyze", True),
            ("chunk_size", "2000"),
        ]

        # Preserve the actual emitted order. The first option's dependency
        # refresh must not overwrite the second edit before its delivery.
        monkeypatch.setattr(canvas, "post_message", forward)
        for event in held:
            forward(event)
        await pilot.pause()

        assert screen._ingest_state.form.analyze
        assert screen._ingest_state.form.chunk_size == "2000"
        assert screen._ingest_state.form.type_options["generic"]["chunk_size"] == "2000"
        assert canvas.query_one("#opt-generic-chunk_size") is chunk_size
        assert chunk_size.value == "2000"
        assert chunk_size.selection == selection
