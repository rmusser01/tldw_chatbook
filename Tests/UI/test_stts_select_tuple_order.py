"""STTS Select widgets must compose `(label, value)`, not `(id, label)` (task-15772).

Textual's `Select` interprets each options tuple as `(label, value)`. Three
`Select`s in `AudioBookGenerationWidget.compose` (`UI/STTS_Window.py`) were
composed backwards -- `("file", "Text File")` instead of
`("Text File", "file")` -- so `Select.value` returned the display label,
never the id every consumer (`_import_content`'s branch checks,
`_initialize_audiobook_defaults`'s provider/format assignment,
`_generate_audiobook`'s id-keyed `costs_per_1k`/`_get_model_for_provider`
lookups) was already written to expect. `set value = "<id>"` was therefore
illegal against the live options list, and `_initialize_audiobook_defaults`
silently swallowed the resulting `InvalidSelectValueError` into a debug log
on every mount.

Residual dependency (task-16471, NOT fixed here): `_import_from_notes` and
`_import_from_conversation` import four `ChaChaNotes_DB` helpers that do not
exist. That import happens outside their own `try/except`, so it raises
before either dialog opens. The dispatch tests below mock those two methods
at the boundary rather than letting the real (currently broken) import run,
so this suite stays honest about testing only the dispatch layer this task
owns.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from tldw_chatbook.UI import STTS_Window as stts_window_module
from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


def _options_as_pairs(select: Select) -> list[tuple[str, object]]:
    """Return the composed options as (label_text, value) pairs.

    Skips the auto-inserted blank option (`Select.NULL`) that Textual
    prepends whenever a `Select` is constructed without an explicit
    `value=`.
    """
    return [
        (str(label), value)
        for label, value in select._options
        if value is not Select.NULL
    ]


# ---------------------------------------------------------------------------
# Compose-shape assertions: (label, value), never (id, label).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_source_select_composes_label_value_order():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        select = app.query_one("#import-source-select", Select)
        assert _options_as_pairs(select) == [
            ("Text File", "file"),
            ("Notes", "notes"),
            ("Conversation", "conversation"),
            ("Paste Text", "paste"),
        ]


@pytest.mark.asyncio
async def test_audiobook_provider_select_composes_label_value_order():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        select = app.query_one("#audiobook-provider-select", Select)
        assert _options_as_pairs(select) == [
            ("OpenAI", "openai"),
            ("ElevenLabs", "elevenlabs"),
            ("Kokoro (Local)", "kokoro"),
            ("Chatterbox (Local)", "chatterbox"),
        ]


@pytest.mark.asyncio
async def test_audiobook_format_select_composes_label_value_order():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        select = app.query_one("#audiobook-format-select", Select)
        assert _options_as_pairs(select) == [
            ("MP3", "mp3"),
            ("M4B (AudioBook)", "m4b"),
            ("Opus", "opus"),
            ("AAC", "aac"),
            ("WAV", "wav"),
        ]


@pytest.mark.asyncio
async def test_the_real_ids_are_legal_select_values():
    """Direct demonstration of the label-vs-value confusion.

    Before the fix, the composed options' *values* are the display labels
    ("OpenAI", "Text File", ...), so assigning the real id raises
    `InvalidSelectValueError` -- the exact failure `_initialize_audiobook_
    defaults` was swallowing into a debug log on every mount.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        provider_select = app.query_one("#audiobook-provider-select", Select)
        provider_select.value = "openai"
        assert provider_select.value == "openai"

        format_select = app.query_one("#audiobook-format-select", Select)
        format_select.value = "m4b"
        assert format_select.value == "m4b"

        import_select = app.query_one("#import-source-select", Select)
        for source_id in ("file", "notes", "conversation", "paste"):
            import_select.value = source_id
            assert import_select.value == source_id


# ---------------------------------------------------------------------------
# `_initialize_audiobook_defaults`: AC #2 / #4.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_audiobook_defaults_lands_the_default_without_logging_a_swallow():
    app = _Host()
    logs: list[str] = []
    sink_id = stts_window_module.logger.add(lambda message: logs.append(str(message)))
    try:
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.pause()
            widget = app.query_one(AudioBookGenerationWidget)

            widget._initialize_audiobook_defaults()

            provider_select = app.query_one("#audiobook-provider-select", Select)
            format_select = app.query_one("#audiobook-format-select", Select)
            assert provider_select.value == "openai"
            assert format_select.value == "m4b"

            joined = "".join(logs)
            assert "Could not set audiobook provider" not in joined
            assert "Could not set audiobook format" not in joined
            assert "Failed to set audiobook defaults" not in joined
    finally:
        stts_window_module.logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Import-source dispatch, driven through the Select + button press (AC #3).
# ---------------------------------------------------------------------------


async def _dispatch_via_select(pilot, app, source_id: str) -> None:
    """Select `source_id` on the real Select widget, then press Import.

    "Import Content" is a `Collapsible` that starts collapsed, so its button
    has a zero-size region until it's expanded -- match what a real user
    does (open the section) rather than reaching past the collapsed state.
    """
    from textual.widgets import Collapsible

    for collapsible in app.query(Collapsible):
        if str(collapsible.title) == "Import Content":
            collapsible.collapsed = False
            break
    else:
        raise AssertionError("Could not find the 'Import Content' collapsible")
    await pilot.pause()

    import_select = app.query_one("#import-source-select", Select)
    import_select.value = source_id
    await pilot.pause()
    await pilot.click("#import-content-btn")
    await pilot.pause()


@pytest.mark.asyncio
async def test_selecting_file_and_pressing_import_calls_import_from_file():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        for name in (
            "_import_from_file",
            "_import_from_notes",
            "_import_from_conversation",
            "_import_from_paste",
        ):
            setattr(widget, name, Mock())

        await _dispatch_via_select(pilot, app, "file")

        widget._import_from_file.assert_called_once()
        widget._import_from_notes.assert_not_called()
        widget._import_from_conversation.assert_not_called()
        widget._import_from_paste.assert_not_called()


@pytest.mark.asyncio
async def test_selecting_paste_and_pressing_import_calls_import_from_paste():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        for name in (
            "_import_from_file",
            "_import_from_notes",
            "_import_from_conversation",
            "_import_from_paste",
        ):
            setattr(widget, name, Mock())

        await _dispatch_via_select(pilot, app, "paste")

        widget._import_from_paste.assert_called_once()
        widget._import_from_file.assert_not_called()
        widget._import_from_notes.assert_not_called()
        widget._import_from_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_selecting_notes_and_pressing_import_calls_import_from_notes():
    """Dispatch-layer coverage only -- see module docstring re: task-16471.

    `_import_from_notes` is mocked rather than exercised for real: its first
    statement imports `fetch_all_notes` from `ChaChaNotes_DB`, which does not
    exist (task-16471). This test proves the Select-driven dispatch routes
    to the right handler with the right value; it does not and should not
    claim the notes dialog itself opens.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        for name in (
            "_import_from_file",
            "_import_from_notes",
            "_import_from_conversation",
            "_import_from_paste",
        ):
            setattr(widget, name, Mock())

        await _dispatch_via_select(pilot, app, "notes")

        widget._import_from_notes.assert_called_once()
        widget._import_from_file.assert_not_called()
        widget._import_from_conversation.assert_not_called()
        widget._import_from_paste.assert_not_called()


@pytest.mark.asyncio
async def test_selecting_conversation_and_pressing_import_calls_import_from_conversation():
    """Dispatch-layer coverage only -- see module docstring re: task-16471."""
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        for name in (
            "_import_from_file",
            "_import_from_notes",
            "_import_from_conversation",
            "_import_from_paste",
        ):
            setattr(widget, name, Mock())

        await _dispatch_via_select(pilot, app, "conversation")

        widget._import_from_conversation.assert_called_once()
        widget._import_from_file.assert_not_called()
        widget._import_from_notes.assert_not_called()
        widget._import_from_paste.assert_not_called()


@pytest.mark.asyncio
async def test_import_content_dispatch_reads_the_value_the_select_actually_holds():
    """`_import_content` must switch on `.value`, i.e. the real id.

    Regression guard for the "consumer expects labels because the tuples
    were backwards" failure mode named in the task: if a future edit
    reverses the compose tuples again without updating `_import_content`,
    this fails because the Select-held id will not match any branch.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        widget._import_from_file = Mock()

        import_select = app.query_one("#import-source-select", Select)
        import_select.value = "file"
        assert import_select.value == "file"

        widget._import_content()

        widget._import_from_file.assert_called_once()
