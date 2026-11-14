"""A Console slash command that ran leaves an empty composer (TASK-33628.20).

Owner decision, 2026-11-12, "Clear it after running": a slash command that
ran (/help, /doctor) leaves an empty composer, the same as a sent message. A
refused or unknown command keeps its text so it can be corrected.

Before this, /help and /doctor left their own command text in the composer.
Since TASK-33628.16 their output rows are not loss, but the leftover text was
an ordinary draft, so Ctrl+Q right after /help asked "Quit Chatbook?" with
"Unsent drafts: 1", and tab close asked "Unsent draft: yes". /research,
/skills and /stream-video left theirs too (as did /rules, bare /system, /goal,
/workflow and /rewind with arguments, pinned in their own files).

These drive the real ``TldwCli`` on the real ``ChatScreen``, its real
controller and store, and press the real keys, as the TASK-33628.16 tests do.
Only the provider gateway (a fixed reply) and the irreversible shutdown at the
end of an approved quit are replaced.

PR #3458 review: a command takes only the draft its send captured. Text typed
right after Enter stays, and a direct call or a button has no captured draft,
so it takes nothing. /doctor and /skills take their text before their slow
work starts, so a re-send or Ctrl+Q while that work runs finds nothing left.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.events import Key
from textual.screen import Screen

from Tests.UI.test_console_command_rows_not_loss import (
    _app,
    _click_close,
    _close_dialog,
    _saved_chat,
    _system_rows,
)
from Tests.UI.test_console_command_origin_chat import (
    _AsyncGate,
    _commands_idle,
    _ThreadGate,
)
from Tests.UI.test_console_quit_loss_guard import (
    _mounted_console,
    _quit_dialogs,
    _type_into_composer,
    _until,
)
from tldw_chatbook.Chat.console_command_grammar import KIND_COMMAND, CommandParse
from tldw_chatbook.Media_Playback import stream_resolve
from tldw_chatbook.Utils import doctor
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    APPLY_BUTTON_ID,
    TEXT_AREA_ID,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Commands that run: what is typed, and how the row they post starts.
_RAN = {
    "help": ("/help", "Console commands:"),
    "help_detail": ("/help prefill", "/prefill"),
    "doctor": ("/doctor", "Doctor"),
    # No skills service in the test app: the list is empty, still a list.
    "skills": ("/skills", ""),
}

#: Commands that are refused or unknown: what is typed, and how the row that
#: says so starts. Each keeps its text so it can be corrected.
_KEPT = {
    "help_unknown_name": ("/help nosuchcommand", "/nosuchcommand is not a known command"),
    "prefill_usage": ("/prefill pin", "Usage: /prefill pin <text>."),
    "unknown_command": ("/nosuchcommand", "Unknown command /nosuchcommand"),
    "skills_name": ("/skills code-review", "Run skills by typing $code-review"),
    "research_usage": ("/research --policy web_only", "/research:"),
    "stream_video_usage": ("/stream-video", "Usage: /stream-video <url>"),
}


async def _run(pilot, console, store, session_id, typed: str, starts: str):
    """Type a command, press Enter, and wait for the row it posts.

    Returns:
        The composer, after the row landed.
    """
    # /doctor's checks take 10-25 s in the test app, longer under load.
    row_timeout = 90.0 if typed.startswith("/doctor") else 20.0
    before = len(_system_rows(store, session_id))
    composer = await _type_into_composer(pilot, console, typed)
    await pilot.press("enter")
    # A bare command name opens the command popup, whose first Enter only
    # completes the name; the next Enter runs it.
    await _until(
        pilot,
        lambda: (
            len(_system_rows(store, session_id)) > before
            or composer.draft_text() == f"{typed} "
        ),
        f"{typed} to run or complete",
    )
    if len(_system_rows(store, session_id)) == before:
        await pilot.press("enter")
    await _until(
        pilot,
        lambda: len(_system_rows(store, session_id)) == before + 1,
        f"the row for {typed}",
        timeout=row_timeout,
    )
    row = _system_rows(store, session_id)[-1]
    assert row.content.startswith(starts), row.content[:80]
    await pilot.pause()
    return composer


@pytest.mark.parametrize("kind", ["help", "help_detail", "skills"])
async def test_a_command_that_ran_leaves_an_empty_composer(monkeypatch, kind):
    """AC: a built-in command that ran clears its own text from the composer.

    /doctor is covered by the Ctrl+Q test below; its checks are slow here.
    """
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        typed, starts = _RAN[kind]

        composer = await _run(pilot, console, store, session_id, typed, starts)

        await _until(
            pilot,
            lambda: composer.draft_text() == "" and not store.session_draft(session_id),
            f"{typed} to clear the composer",
            timeout=5.0,
        )


@pytest.mark.parametrize("kind", list(_KEPT))
async def test_a_refused_or_unknown_command_keeps_its_text(monkeypatch, kind):
    """AC: a refused or unknown command keeps its text for correction."""
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        typed, starts = _KEPT[kind]

        composer = await _run(pilot, console, store, session_id, typed, starts)
        for _ in range(5):
            await pilot.pause()

        assert composer.draft_text().rstrip() == typed


@pytest.mark.parametrize("kind", ["help", "doctor"])
async def test_ctrl_q_right_after_a_command_quits_without_asking(monkeypatch, kind):
    """AC#2: in a saved chat with nothing else at stake, Ctrl+Q right after
    /help or /doctor quits without a question -- no Ctrl+U first."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        typed, starts = _RAN[kind]
        composer = await _run(pilot, console, store, session_id, typed, starts)
        await _until(pilot, lambda: composer.draft_text() == "", f"{typed} to clear")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: "quit" in events or bool(_quit_dialogs(app)),
            "Ctrl+Q to quit or ask",
            timeout=5.0,
        )
        dialogs = _quit_dialogs(app)
        assert not dialogs, f"Ctrl+Q asked after {typed}: {dialogs[0].message!r}"
        assert events == ["quit"]


async def test_tab_close_right_after_help_closes_without_asking(monkeypatch):
    """AC#2: closing the saved chat's tab right after /help closes it at once."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        typed, starts = _RAN["help"]
        composer = await _run(pilot, console, store, session_id, typed, starts)
        await _until(pilot, lambda: composer.draft_text() == "", f"{typed} to clear")

        await _click_close(pilot, app, session_id)
        await _until(
            pilot,
            lambda: (
                _close_dialog(app) is not None
                or session_id not in {session.id for session in store.sessions()}
            ),
            "the tab to close or ask",
        )
        dialog = _close_dialog(app)
        assert dialog is None, f"tab close asked after /help: {dialog.message!r}"
        assert session_id not in {session.id for session in store.sessions()}
        assert events == []


class _NoPlayer(Screen):
    """Stands in for the video player, which starts ffmpeg on mount."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()


async def test_stream_video_that_played_leaves_an_empty_composer(monkeypatch):
    """AC: /stream-video that resolved and opened its player takes its text."""
    from tldw_chatbook.UI.Screens import video_player_screen

    url = "https://example.com/clip.mp4"
    monkeypatch.setattr(
        stream_resolve,
        "resolve_stream_url",
        lambda raw: stream_resolve.StreamResolution(
            final_url=raw, content_type="video/mp4", seekable=True, via_ytdlp=False
        ),
    )
    monkeypatch.setattr(video_player_screen, "VideoPlayerScreen", _NoPlayer)
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        composer = await _type_into_composer(pilot, console, f"/stream-video {url}")
        await pilot.press("enter")
        await _until(pilot, lambda: isinstance(app.screen, _NoPlayer), "the player")
        await _until(pilot, lambda: composer.draft_text() == "", "the composer to clear")


async def test_research_that_started_leaves_an_empty_composer(monkeypatch):
    """AC: /research that started its run takes its text; the run itself is
    not under test, so its service refuses to launch (a logged warning)."""
    from tldw_chatbook.Tools import web_tool_impls

    def _launch_run(**_kwargs):
        raise RuntimeError("not launched in this test")

    monkeypatch.setattr(web_tool_impls, "deep_search_pipeline_params", lambda: {})
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        monkeypatch.setattr(
            app,
            "local_research_service",
            SimpleNamespace(launch_run=_launch_run),
            raising=False,
        )
        composer = await _run(
            pilot,
            console,
            store,
            session_id,
            "/research what is retrieval augmentation",
            "Deep research started: what is retrieval augmentation",
        )
        await _until(pilot, lambda: composer.draft_text() == "", "the composer to clear")


# -- PR #3458 review: only the captured draft, only inside a sent command -----


@pytest.mark.parametrize(
    "typed",
    ["/help prefill", "/prefill clear", "/stop", "/sessions"],
    ids=["help", "prefill", "stop", "action"],
)
async def test_text_typed_right_after_enter_stays(monkeypatch, typed):
    """A command takes only the draft its send captured: what the user typed
    between Enter and the command's first step stays. /prefill, /stop and the
    action commands (here /sessions) cleared the whole composer before."""
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft(typed)
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()
        # A bare name opens the popup, where Enter would complete it instead.
        console._dismiss_console_command_popup()
        # Enter without pumping messages, then type before the command's
        # worker gets its first step (as the /rewind restore test does).
        console.on_key(Key(key="enter", character="\r"))
        composer.insert_text("next")
        assert composer.draft_text() == f"{typed}next"

        await _until(
            pilot,
            lambda: composer.draft_text() == "next",
            f"{typed} to take only its own text",
            timeout=10.0,
        )
        await _until(pilot, lambda: _commands_idle(console), f"{typed} to finish")
        assert composer.draft_text() == "next"


async def test_a_direct_call_takes_nothing(monkeypatch):
    """Outside a sent command there is no captured draft: a direct call of a
    command handler, or a button that opens the same dialog, leaves the
    user's draft alone."""
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("my draft")

        await console._console_command_help(
            CommandParse(kind=KIND_COMMAND, name="help", args="")
        )
        await pilot.pause()
        assert composer.draft_text() == "my draft"

        # The System prompt editor opened from the rail line, then applied.
        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await _until(
            pilot,
            lambda: bool(app.screen.query(f"#{APPLY_BUTTON_ID}")),
            "the System prompt editor",
        )
        modal = app.screen
        modal.query_one(f"#{TEXT_AREA_ID}").text = "Be brief."
        modal.query_one(f"#{APPLY_BUTTON_ID}").press()
        await _until(pilot, lambda: app.screen is console, "the editor to close")
        settings = console._session._ensure_active_console_session_settings()
        assert settings.system_prompt == "Be brief."
        assert composer.draft_text() == "my draft"


async def test_doctor_takes_its_text_before_its_checks(monkeypatch):
    """/doctor's checks take 10-25 s. It took its text only after them, so
    for that whole time Ctrl+Q still asked about "/doctor", and a re-send
    raced it: "/doctor" sent, " network" typed, "/doctor network" sent -- the
    first take removed "/doctor" and the second found nothing to take, so
    " network" stayed. /doctor cannot be refused (every failure is a report
    row), so it takes its text before the checks start."""
    gate = _ThreadGate(result=[])
    monkeypatch.setattr(doctor, "run_doctor", gate)
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        before = len(_system_rows(store, session_id))
        try:
            composer = await _type_into_composer(pilot, console, "/doctor")
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: gate.entered.is_set() or composer.draft_text() == "/doctor ",
                "/doctor to run or complete",
            )
            if not gate.entered.is_set():
                await pilot.press("enter")
            await _until(pilot, gate.entered.is_set, "/doctor's checks to start")
            await _until(
                pilot,
                lambda: composer.draft_text() == "",
                "/doctor to take its text while its checks run",
                timeout=5.0,
            )

            # The re-send while the first checks still run.
            composer.focus()
            composer.load_draft("/doctor network")
            console._sync_console_command_popup()
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: composer.draft_text() == "",
                "/doctor network to take its text",
                timeout=5.0,
            )
        finally:
            gate.released.set()
        await _until(
            pilot,
            lambda: len(_system_rows(store, session_id)) == before + 2,
            "both reports",
            timeout=30.0,
        )
        await _until(pilot, lambda: _commands_idle(console), "both runs to end")
        assert composer.draft_text() == ""

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: "quit" in events or bool(_quit_dialogs(app)),
            "Ctrl+Q to quit or ask",
            timeout=5.0,
        )
        dialogs = _quit_dialogs(app)
        assert not dialogs, f"Ctrl+Q asked after /doctor: {dialogs[0].message!r}"
        assert events == ["quit"]


async def test_skills_takes_its_text_before_its_lookup(monkeypatch):
    """/skills cannot be refused (an empty list is still a list), so it takes
    its text before the skill lookup, which awaits, rather than after."""
    app = _app(monkeypatch, [])
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        gate = _AsyncGate(result={})
        monkeypatch.setattr(console._skill, "_fetch_console_skill_context", gate)
        try:
            composer = await _type_into_composer(pilot, console, "/skills")
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: gate.entered.is_set() or composer.draft_text() == "/skills ",
                "/skills to run or complete",
            )
            if not gate.entered.is_set():
                await pilot.press("enter")
            await _until(pilot, gate.entered.is_set, "the skill lookup to start")
            await _until(
                pilot,
                lambda: composer.draft_text() == "",
                "/skills to take its text during the lookup",
                timeout=5.0,
            )
        finally:
            gate.released.set()
        await _until(pilot, lambda: _commands_idle(console), "/skills to finish")
        assert composer.draft_text() == ""
