"""`UI/*.py` top-level fixes from the tier-2 review (S20 P2s).

Gate-free: the markup and truncation tests are pure function calls, and the
worker test drives one method on a bare instance. None of them boots the app,
so `Backup_Recovery`'s ADR-126 recovery gate never runs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Static

from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbookCard

_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"


class _CardApp(App):
    """Minimal host so `ChatbookCard.compose` has an active app."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield ChatbookCard(self._data)


async def _card_description(data: dict[str, Any]) -> str:
    """The description text `ChatbookCard` actually renders."""
    async with _CardApp(data).run_test() as pilot:
        static = pilot.app.query_one(".chatbook-card-description", Static)
        return str(static.renderable)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chatbook_card_does_not_append_an_ellipsis_to_untruncated_text() -> None:
    """S20 P2: `x[:100] + "..."` appended the marker unconditionally.

    `_scan_chatbooks` sets the `"description"` key ONLY inside its
    `manifest.json` branch, so a zip with no manifest rendered
    `"No description..."` and a zip whose manifest carries an empty
    description rendered literally `"..."`.
    """
    assert await _card_description({"description": ""}) == "No description"
    assert await _card_description({}) == "No description"
    assert await _card_description({"description": "A short one"}) == "A short one"

    long_text = "x" * 400
    truncated = await _card_description({"description": long_text})
    assert len(truncated) == 100
    assert truncated.endswith("...")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module", "function", "argument"),
    (
        ("UI/STTS_Window.py", "_preview_chapter_audio", "chapter.title"),
        ("UI/Voice_Cloning_Window.py", "_test_generate_voice", "test_profile"),
    ),
)
def test_markup_on_richlog_writes_escape_their_interpolated_text(
    module: str, function: str, argument: str
) -> None:
    """S20 P2: untrusted names reached a `markup=True` `RichLog` raw.

    `RichLog._make_renderable` runs `Text.from_markup(content)` when
    `self.markup`, so "Chapter 3 [draft]" loses that segment and anything
    containing "[/" raises `MarkupError`. At the STTS site the raise is
    caught and the preview never generates; at the Voice Cloning site there
    is no handler at all, so the Test-voice button silently does nothing.
    """
    tree = ast.parse((_PACKAGE / module).read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    )

    writes = [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    ]
    interpolating = [
        node for node in writes if any(isinstance(v, ast.JoinedStr) for v in node.args)
    ]
    assert interpolating, f"{module}::{function} no longer writes interpolated text"

    for node in interpolating:
        rendered = ast.dump(node)
        assert "escape_markup" in rendered, (
            f"{module}::{function} line {node.lineno} interpolates "
            f"{argument} into a markup=True RichLog with no escaper"
        )


@pytest.mark.unit
def test_the_repo_escaper_actually_survives_a_bracketed_name() -> None:
    """The mechanism the test above pins, executed once.

    `rich.markup.escape` does NOT escape a `[TODO]`-shaped token; the repo's
    `Utils.input_validation.escape_markup` does.
    """
    from tldw_chatbook.Utils.input_validation import escape_markup

    assert Content.from_markup("Chapter 3 [draft]").plain == "Chapter 3 "
    assert (
        Content.from_markup(escape_markup("Chapter 3 [draft]")).plain
        == "Chapter 3 [draft]"
    )
    with pytest.raises(Exception):
        Content.from_markup("broken [/]")
    assert Content.from_markup(escape_markup("broken [/]")).plain == "broken [/]"


@pytest.mark.unit
def test_voice_cloning_actions_run_as_widget_owned_workers() -> None:
    """S20 P2: fire-and-forget `asyncio.create_task`, never cancelled.

    Five key bindings routed through `_spawn_action`, which used a bare
    `asyncio.create_task`. The file has no `on_unmount`, so nothing cancelled
    a live action on teardown -- and several of these coroutines park on
    `push_screen_wait` and then touch the DOM on resume. `run_worker` puts
    them in Textual's registry, where `Widget._on_unmount` ->
    `workers.cancel_node(self)` cancels them, and a failure is reported
    rather than dropped into asyncio's default handler.
    """
    from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

    window = VoiceCloningWindow.__new__(VoiceCloningWindow)
    recorded: dict[str, Any] = {}

    def _run_worker(work: Any, **kwargs: Any) -> str:
        recorded["work"] = work
        recorded["kwargs"] = kwargs
        return "worker"

    window.run_worker = _run_worker  # type: ignore[method-assign]

    async def _action() -> None:  # pragma: no cover - never awaited
        return None

    coroutine = _action()
    try:
        assert window._spawn_action(coroutine, "probe") == "worker"
    finally:
        coroutine.close()

    assert recorded["work"] is coroutine
    assert recorded["kwargs"]["group"] == "voice-cloning-actions"
    assert recorded["kwargs"]["exit_on_error"] is False, (
        "these are recoverable user actions; run_worker's default would turn "
        "a failed export into an app exit"
    )
    assert not hasattr(window, "_action_tasks"), (
        "the hand-rolled strong-reference set is redundant once WorkerManager "
        "owns the task"
    )
