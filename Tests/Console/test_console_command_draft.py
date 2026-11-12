"""Decisions of the Console command-draft and command-origin helpers.

TASK-33622.16 review. ``command_draft`` decides whether a slash command takes
its draft, puts it back, or only says how to send it again;
``command_handoff`` decides whether a command still acts on the chat it was
sent from. The mounted tests (``test_console_video_send_freeze``,
``test_console_command_origin_chat``) drive them through the real Console;
these pin each branch against a composer double that keeps the composer's
draft-revision contract (``ConsoleComposerBar.commit_captured_draft``):
typing advances ``edit_serial``, a scope change (clear, load, commit,
restore) advances the draft generation.
"""

from __future__ import annotations

import contextvars
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.command_draft import (
    CAPTURED_COMMAND_DRAFT,
    TakenCommandDraft,
    restore_command_draft,
    take_command_draft,
    with_resend_hint,
)
from tldw_chatbook.UI.Console_Modules.command_handoff import (
    COMMAND_ORIGIN,
    CommandOrigin,
    append_command_output,
    command_origin_session,
    refuse_if_chat_changed,
)

COMMAND = "/generate-video a paper boat"
TYPED = "what about a sailboat"


class _Composer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.edit_serial = 0
        self.generation = 0
        self.restored: list = []

    # The user and the Console.
    def type(self, more: str) -> None:
        self.text += more
        self.edit_serial += 1

    def load(self, text: str) -> None:
        """A chat switch: the composer now shows another chat's draft."""
        self.text = text
        self.generation += 1

    def clear(self) -> None:
        self.text = ""
        self.generation += 1

    # The draft-revision API the helpers use.
    def draft_text(self) -> str:
        return self.text

    def capture_draft_for_send(self):
        if not self.text:
            return None
        return SimpleNamespace(
            text=self.text, edit_serial=self.edit_serial, generation=self.generation
        )

    def capture_draft_snapshot(self):
        return SimpleNamespace(generation=self.generation)

    def commit_captured_draft(self, stash) -> bool:
        if (
            self.generation != stash.generation
            or not self.text.startswith(stash.text)
            or (self.text == stash.text and self.edit_serial != stash.edit_serial)
        ):
            return False
        self.text = self.text[len(stash.text) :]
        self.generation += 1
        return True

    def restore_stashed_draft(self, stash) -> None:
        self.restored.append(stash)
        self.text = stash.text + self.text
        self.generation += 1


class _Clear:
    def __init__(self, composer: _Composer | None) -> None:
        self.calls = 0
        self._composer = composer

    def __call__(self) -> None:
        self.calls += 1
        if self._composer is not None:
            self._composer.clear()


def _take(composer: _Composer | None, captured=None):
    """``take_command_draft`` as a handed-off command runs it: inside a
    context holding the send's capture (None: a direct call)."""
    clear = _Clear(composer)

    def run():
        CAPTURED_COMMAND_DRAFT.set(captured)
        return take_command_draft(composer, clear)

    return contextvars.copy_context().run(run), clear


# -- take_command_draft ---------------------------------------------------------


def test_take_clears_a_composer_holding_exactly_the_captured_draft():
    composer = _Composer(COMMAND)
    captured = composer.capture_draft_for_send()

    taken, clear = _take(composer, captured)

    assert clear.calls == 1
    assert composer.text == ""
    assert taken.removed and taken.stash is captured
    assert taken.generation == composer.generation


def test_take_keeps_text_typed_after_the_captured_draft():
    composer = _Composer(COMMAND)
    captured = composer.capture_draft_for_send()
    composer.type(" " + TYPED)

    taken, clear = _take(composer, captured)

    assert clear.calls == 0
    assert composer.text == " " + TYPED
    assert taken.removed


@pytest.mark.parametrize("change", ["replaced", "switched"])
def test_take_removes_nothing_but_keeps_a_draft_that_changed(change):
    """Qodo #4: a take that removed nothing returned None, so a later
    failure had no command to offer for resend."""
    composer = _Composer(COMMAND)
    captured = composer.capture_draft_for_send()
    if change == "replaced":
        composer.clear()
        composer.type(TYPED)
        expected = TYPED
    else:
        composer.load(COMMAND)  # another chat, same text: still not ours
        expected = COMMAND

    taken, clear = _take(composer, captured)

    assert clear.calls == 0
    assert composer.text == expected, "the take touched a draft it does not own"
    assert taken is not None and taken.stash is captured
    assert not taken.removed


def test_take_without_a_capture_takes_the_live_draft():
    """A direct call (no hand-off) takes the whole draft, as before."""
    composer = _Composer(COMMAND)

    taken, clear = _take(composer)

    assert clear.calls == 1 and composer.text == ""
    assert taken.removed and taken.stash.text == COMMAND


def test_take_of_an_empty_draft_keeps_nothing():
    composer = _Composer("")

    taken, clear = _take(composer)

    assert taken is None and clear.calls == 1


def test_take_without_a_composer_still_keeps_the_captured_command():
    captured = _Composer(COMMAND).capture_draft_for_send()

    taken, clear = _take(None, captured)

    assert clear.calls == 1
    assert taken == TakenCommandDraft(None, captured, None)
    assert _take(None)[0] is None


# -- restore_command_draft --------------------------------------------------------


def _taken_from(composer: _Composer) -> TakenCommandDraft:
    taken, _clear = _take(composer, composer.capture_draft_for_send())
    assert taken.removed
    return taken


def test_restore_puts_the_command_back_into_its_own_empty_scope():
    composer = _Composer(COMMAND)
    taken = _taken_from(composer)

    assert restore_command_draft(composer, taken) == ""
    assert composer.restored == [taken.stash]
    assert composer.text == COMMAND


def test_restore_of_nothing_says_nothing():
    assert restore_command_draft(_Composer(), None) == ""


def _refused(composer, taken) -> str:
    before = composer.text if composer is not None else None
    hint = restore_command_draft(composer, taken)
    if composer is not None:
        assert composer.restored == [], "restored over a changed composer"
        assert composer.text == before
    return hint


def test_restore_never_overwrites_text_typed_since():
    composer = _Composer(COMMAND)
    taken = _taken_from(composer)
    composer.type(TYPED)

    assert COMMAND in _refused(composer, taken)


def test_restore_never_writes_into_the_chat_switched_to():
    composer = _Composer(COMMAND)
    taken = _taken_from(composer)
    composer.load("")  # the other chat's draft is empty

    assert COMMAND in _refused(composer, taken)


def test_restore_never_writes_into_another_composer():
    composer = _Composer(COMMAND)
    taken = _taken_from(composer)

    assert COMMAND in _refused(_Composer(), taken)
    assert COMMAND in _refused(None, taken)


def test_restore_of_a_command_the_take_left_behind_only_offers_a_resend():
    """Qodo #4: same composer, same generation, empty draft -- but the take
    never removed the command, so putting it back would duplicate or
    overwrite; the failure row offers it instead."""
    composer = _Composer(COMMAND)
    captured = composer.capture_draft_for_send()
    composer.clear()
    taken, _clear = _take(composer, captured)
    assert not taken.removed

    hint = _refused(composer, taken)

    assert hint.endswith(f"send: {COMMAND}")


# -- with_resend_hint -------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "hint", "expected"),
    [
        ("Video generation failed (X).", "", "Video generation failed (X)."),
        ("Image generation failed: boom", "", "Image generation failed: boom"),
        ("Video generation failed (X).", "Send: /c", "Video generation failed (X). Send: /c"),
        ("Image generation failed: boom", "Send: /c", "Image generation failed: boom. Send: /c"),
        ("Failed...", "Send: /c", "Failed. Send: /c"),
    ],
)
def test_with_resend_hint_joins_with_exactly_one_period(message, hint, expected):
    assert with_resend_hint(message, hint) == expected


# -- command_handoff origin helpers ------------------------------------------------


class _Screen:
    def __init__(self, active: str | None, visible: str | None) -> None:
        self.toasts: list[tuple[str, str | None]] = []
        store = SimpleNamespace(active_session_id=active)
        self._ensure_console_chat_store = lambda: store
        self._console_visible_draft_session_id = visible
        self.app_instance = SimpleNamespace(
            notify=lambda message, severity=None: self.toasts.append(
                (message, severity)
            )
        )


def _in_command(screen, origin: str, fn):
    def run():
        COMMAND_ORIGIN.set(CommandOrigin(screen, origin))
        return fn()

    return contextvars.copy_context().run(run)


def test_a_direct_call_has_no_origin_and_is_never_refused():
    screen = _Screen(active="b", visible="b")

    assert command_origin_session() is None
    assert refuse_if_chat_changed("system") is False
    assert screen.toasts == []


def test_a_command_from_the_chat_showing_runs():
    screen = _Screen(active="a", visible="a")

    assert _in_command(screen, "a", command_origin_session) == "a"
    assert _in_command(screen, "a", lambda: refuse_if_chat_changed("system")) is False
    assert screen.toasts == []


@pytest.mark.parametrize(
    ("active", "visible"),
    [("b", "b"), ("b", "a"), ("a", "b")],
    ids=["switched", "store-switched", "composer-switched"],
)
def test_a_command_whose_chat_no_longer_shows_is_refused_aloud(active, visible):
    screen = _Screen(active=active, visible=visible)

    assert _in_command(screen, "a", lambda: refuse_if_chat_changed("system")) is True
    assert len(screen.toasts) == 1
    message, severity = screen.toasts[0]
    assert "/system" in message and severity == "warning"


async def test_command_output_goes_to_the_origin_chat_only_inside_a_command():
    calls: list[tuple[str, dict]] = []

    async def append(message, **kwargs):
        calls.append((message, kwargs))

    await append_command_output(append, "direct")
    token = COMMAND_ORIGIN.set(CommandOrigin(_Screen("b", "b"), "a"))
    try:
        await append_command_output(append, "handed off")
    finally:
        COMMAND_ORIGIN.reset(token)

    assert calls == [("direct", {}), ("handed off", {"session_id": "a"})]
