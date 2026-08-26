"""Characterisation of what a Console composer draft edit must still trigger.

TASK-3749 moves six more `ChatScreen.on_key` branches into
`ConsoleComposerBar.handle_console_key` by replacing the screen's
"edit the draft, then call back into me" pattern with a
`ConsoleComposerBar.DraftChanged` message the screen subscribes to.

These tests are written and committed BEFORE that change, against the
current callback shape, and pin the three things it must not disturb:

1. **The Workbench actions row.** `#console-send-message` is a Workbench
   action whose `disabled` state is derived from the draft by
   `_sync_console_workbench_actions_from_draft`; every one of the six keys
   that edits the draft has to leave it correct.
2. **The slash-command popup**, which is opened by that same sync -- and
   opened *in the same key turn*, so that a following arrow key already
   finds it open. That timing is asserted adversarially (two keys queued
   with no event-loop drain between them), because a message-based
   notification is delivered asynchronously and would be free to arrive
   after the next key otherwise.
3. **The first-run guidance**, which is dismissed by the two keys that
   ADD text (a printable character, Shift+Enter's newline) and NOT by the
   four that remove it -- a distinction a naive "dismiss on every draft
   change" would erase.

They assert observable end state through a REAL key press on the real
Console screen, never that a particular method was called.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

APP_SIZE = (140, 42)


async def _console(host, pilot, text: str = ""):
    """Mount the ready Console, optionally seed the draft, focus the composer.

    `load_draft` is deliberately used to seed text: it is not one of the
    six branches under test, and (unlike typing the text) it does not
    itself dismiss the guidance, which the guidance tests below depend on.
    """
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    if text:
        composer.load_draft(text)
    composer.focus()
    await pilot.pause()
    return console, composer


def _send_action(console) -> Button:
    """The Workbench actions row's Send action."""
    return console.query_one("#console-send-message", Button)


# ---------------------------------------------------------------------------
# 1. The Workbench actions row tracks the draft across all six edit keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "keys", "expected_draft"),
    [
        ("x", ("backspace",), ""),
        ("x", ("ctrl+h",), ""),
        ("x", ("delete",), ""),
        ("hello", ("ctrl+w",), ""),
        ("hello there", ("ctrl+u",), ""),
    ],
)
@pytest.mark.asyncio
async def test_a_draft_emptying_key_disables_the_workbench_send_action(
    seed, keys, expected_draft
):
    """A key that empties the draft must disable the Workbench Send action."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot, seed)
        if "delete" in keys:
            composer.position_cursor_from_display_index(0)
            await pilot.pause()
        assert _send_action(console).disabled is False

        await pilot.press(*keys)
        await pilot.pause()

        assert composer.draft_text() == expected_draft
        assert _send_action(console).disabled is True


@pytest.mark.asyncio
async def test_a_printable_key_enables_the_workbench_send_action():
    """A printable key that adds text must enable the Workbench Send action."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        assert _send_action(console).disabled is True

        await pilot.press("a")
        await pilot.pause()

        assert composer.draft_text() == "a"
        assert _send_action(console).disabled is False


@pytest.mark.parametrize("key", ["shift+enter", "ctrl+j"])
@pytest.mark.asyncio
async def test_a_newline_key_resyncs_the_send_action_without_enabling_it(key):
    """A newline-only draft is still "nothing to send" -- and stays that way.

    Worth pinning precisely because it is the counter-example to "the sync
    just mirrors `draft_text() != ''`": the draft is non-empty here and the
    action stays disabled, so the sync really is re-deriving Workbench
    readiness, not flipping a boolean off the draft length.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        assert _send_action(console).disabled is True

        await pilot.press(key)
        await pilot.pause()

        assert composer.draft_text() == "\n"
        assert _send_action(console).disabled is True


@pytest.mark.parametrize("key", ["shift+enter", "ctrl+j"])
@pytest.mark.asyncio
async def test_a_newline_after_real_text_leaves_the_send_action_enabled(key):
    """shift+enter after real text keeps Send enabled (still a non-empty draft)."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot, "hi")
        assert _send_action(console).disabled is False

        await pilot.press(key)
        await pilot.pause()

        assert composer.draft_text() == "hi\n"
        assert _send_action(console).disabled is False


@pytest.mark.asyncio
async def test_a_partial_deletion_leaves_the_send_action_enabled():
    """The sync is not just an empty/non-empty flip -- it runs every edit."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot, "hello world")

        await pilot.press("ctrl+w")
        await pilot.pause()

        assert composer.draft_text() == "hello "
        assert _send_action(console).disabled is False


# ---------------------------------------------------------------------------
# 2. The slash-command popup opens on the keystroke that types "/"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_a_slash_opens_the_command_popup():
    """A lone '/' opens the slash-command popup once its DraftChanged lands."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        popup = console.query_one("#console-command-popup")
        assert popup.is_open is False

        await pilot.press("/")
        await pilot.pause()

        assert composer.draft_text() == "/"
        assert popup.is_open is True


@pytest.mark.asyncio
async def test_an_arrow_queued_behind_the_slash_still_navigates_the_popup():
    """The popup must be open by the time the NEXT key is handled.

    Was a strict xfail (the TASK-3790 ordering gap); un-xfailed by the
    generation-gated `_ensure_console_command_popup_current`, which
    re-derives the popup at the routing point ONLY when the draft moved
    since the last sync -- so a key queued behind the slash in the same
    driver read now finds the popup open, while an Escape-dismissed popup
    (no edit, no generation movement) stays dismissed.

    Both keys are posted to the focused composer's queue with no
    event-loop drain between them -- exactly how `App.on_event` routes two
    keystrokes that arrive in a single driver read (a key macro, a text
    expander, `tmux send-keys`; human typing is three orders of magnitude
    too slow to reach it). Before TASK-3749 the popup was opened
    synchronously inside the "/" key turn, so the Down that followed found
    `popup.is_open` already True and moved the highlight instead of
    falling through to the composer's own caret/history handling.

    Blast radius, measured as a two-arm A/B (this code vs. the baseline
    callback restored synchronously in its place): exactly one ignored
    keystroke. Batched "/"+Down -- baseline highlights the second entry,
    this ignores the Down. Batched "/"+Enter -- baseline accepts the
    highlighted suggestion into the draft, this ignores the Enter. In both
    arms the popup ends up open and the draft is "/", and in neither does
    the stray key send anything: Enter's own path re-reads the Send action
    AFTER stashing the draft, so it never depended on this sync's timing.
    The next keypress behaves normally.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        popup = console.query_one("#console-command-popup")

        composer.post_message(events.Key("slash", "/"))
        composer.post_message(events.Key("down", None))
        await pilot.pause()
        await pilot.pause()

        assert composer.draft_text() == "/"
        assert popup.is_open is True
        first, second = popup._suggestions[0], popup._suggestions[1]
        assert first.label != second.label
        # Down was consumed by the popup, not by the composer.
        assert popup.accept_selected().label == second.label


# ---------------------------------------------------------------------------
# 3. Guidance is dismissed by insertions only, never by deletions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keys", [("a",), ("shift+enter",), ("ctrl+j",)])
@pytest.mark.asyncio
async def test_an_insertion_key_dismisses_the_first_run_guidance(keys):
    """Text-ADDING keys retire the first-run guidance (is_insertion=True)."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)
        assert console._console_guidance_dismissed is False

        await pilot.press(*keys)
        await pilot.pause()

        assert console._console_guidance_dismissed is True


@pytest.mark.parametrize(
    ("seed", "keys"),
    [
        ("x", ("backspace",)),
        ("x", ("ctrl+h",)),
        ("x", ("delete",)),
        ("hello", ("ctrl+w",)),
        ("hello", ("ctrl+u",)),
    ],
)
@pytest.mark.asyncio
async def test_a_deletion_key_leaves_the_first_run_guidance_alone(seed, keys):
    """Deletions never dismiss guidance: only an insertion claims composing."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot, seed)
        if "delete" in keys:
            composer.position_cursor_from_display_index(0)
            await pilot.pause()
        assert console._console_guidance_dismissed is False

        await pilot.press(*keys)
        await pilot.pause()

        assert console._console_guidance_dismissed is False


@pytest.mark.asyncio
async def test_escape_dismissal_survives_a_following_arrow_key():
    """An Escape-dismissed popup must NOT be re-opened by mere navigation.

    The guard that makes `_ensure_console_command_popup_current` safe: it is
    generation-gated, and Escape edits nothing. An UNGATED re-sync at the
    routing point would re-derive suggestions from the still-slash draft and
    re-open the popup the user just closed -- turning the task-3790 fix into
    a worse regression than the one it fixes. This is the test that fails if
    anyone "simplifies" the gate away.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        popup = console.query_one("#console-command-popup")

        composer.post_message(events.Key("slash", "/"))
        await pilot.pause()
        await pilot.pause()
        assert popup.is_open is True

        # Escape reaches the popup through the screen's binding action, not
        # on_key (the popup branch never sees it), so invoke the real dismiss
        # route the Escape binding calls -- a synthetic Key posted to the
        # composer does not resolve screen bindings under this harness.
        console.action_focus_console_composer_home()
        await pilot.pause()
        assert popup.is_open is False

        composer.post_message(events.Key("down", None))
        await pilot.pause()
        await pilot.pause()
        assert popup.is_open is False, (
            "navigation re-opened a popup the user dismissed with Escape"
        )
        assert composer.draft_text() == "/"


@pytest.mark.asyncio
async def test_deferred_draft_changed_does_not_yank_the_moved_highlight():
    """The queued `DraftChanged` must not reset a highlight Down just moved.

    Same-read `/`+Down: the routing-point ensure opens the popup and Down
    moves the highlight to row 1 -- then the deferred `DraftChanged` finally
    delivers and re-runs the sync. `show_suggestions` is idempotent for an
    identical suggestion list precisely so that second run leaves the
    highlight alone; without it the user's Down would be silently undone a
    frame later.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        popup = console.query_one("#console-command-popup")

        composer.post_message(events.Key("slash", "/"))
        composer.post_message(events.Key("down", None))
        # Drain until the deferred DraftChanged has definitely delivered.
        for _ in range(4):
            await pilot.pause()

        assert popup.is_open is True
        second = popup._suggestions[1]
        assert popup.accept_selected().label == second.label, (
            "the deferred sync reset the highlight the Down had moved"
        )
