"""Per-keystroke work census for the Console composer (TASK-24300, TASK-24301).

Wall clock is not usable as evidence on this surface, twice over. Textual's
``Pilot`` posts one callback per mounted widget per ``pause()`` -- 81,862
dispatch lookups for 40 keystrokes on a 488-widget screen -- so press latency
tracks widget count rather than app work. And the machine this repo is
developed on routinely carries a load average of 5-10 from concurrent agent
sessions; the same unchanged tree measured 3.80 and 6.75 ms/key for the same
input twenty minutes apart, which is a 78% swing with no code between the two
runs.

So these guards count CALLS, which are deterministic. Every number below
reproduced exactly on every run during the 2026-08-28 review, on a machine
whose wall-clock numbers moved by 3.5x.

The census that motivated the file, measured on dev ``3a3383123e`` with a
400-message conversation:

    messages_for_session   3.27 calls/key  ->  1,310 message snapshots per key

``messages_for_session`` materialises every stream buffer and deep-snapshots
every message. Four call sites used it as a predicate ("does this session have
any messages?") and one of them is on the composer keystroke path, so typing
degraded linearly with conversation length: 1.31 ms/key empty, 13.46 ms/key at
400 messages, and the whole difference was this call.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any

import pytest
from Tests.private_profile import private_profile_test

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Printable keystrokes each census presses into the composer.
KEYSTROKES = 24

#: Messages seeded before the typing burst. Large enough that an O(N) term is
#: unmissable in a call count, small enough to keep the test quick.
SEEDED_MESSAGES = 200


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Seed the private-profile child's selected config with setup completed.

    A probe that skips this reads (and can write) the developer's real
    config; a probe that skips the API key lands Console in the SETUP state,
    where there is no composer to type into at all.

    Args:
        monkeypatch: pytest fixture used to set the environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    del tmp_path  # The private-profile wrapper owns the selected scratch tree.
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        '[general]\nusers_name = "census"\n\n'
        "[first_run]\nsetup_completed = true\n\n"
        "[_first_run]\nsetup_completed = true\n\n"
        "[splash_screen]\nenabled = false\n\n"
        "[api_settings.openai]\n"
        'api_key = "sk-census-000000000000000000000000000000000000"\n'
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "console_keystroke_census")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)


async def _settle(pilot: Any, passes: int = 30) -> None:
    """Let mount work finish so it is not billed to the typing burst."""
    for _ in range(passes):
        await asyncio.sleep(0.05)
        await pilot.pause()


async def _census(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seeded_messages: int
) -> dict[str, int]:
    """Boot Console, seed a transcript, type, and return a call census.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
        seeded_messages: How many messages to append before typing.

    Returns:
        Mapping of counter name to calls observed during the typing burst.
    """
    _scratch_env(monkeypatch, tmp_path)

    from textual.pilot import Pilot

    real_wait_for_screen = Pilot._wait_for_screen

    async def wait_for_screen(self: Any, timeout: float = 120.0) -> bool:
        # On the review Windows host the second full app's storage and
        # widget admission can exceed Textual's 30s default before typing
        # starts. This changes only the watchdog, never the work census.
        return await real_wait_for_screen(self, timeout=max(timeout, 120.0))

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait_for_screen)

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Console_Modules import session as session_module
    from tldw_chatbook.UI.Console_Modules import (
        console_spend_projection as spend_module,
    )
    from tldw_chatbook.UI.Screens import chat_screen as screen_module

    from tldw_chatbook.Chat import console_session_settings as settings_module

    counts: dict[str, int] = {
        "messages_for_session": 0,
        "snapshots": 0,
        "settings_readiness_builds": 0,
        "template_default_builds": 0,
        "snapshot_rows": 0,
        "spend_history_rows": 0,
        "cost_rows": 0,
        "context_rows": 0,
        "context_estimate_max_rows": 0,
    }
    counting = {"on": False}

    real_messages_for_session = ConsoleChatStore.messages_for_session

    def counted_messages_for_session(
        self: ConsoleChatStore, session_id: str
    ) -> list[Any]:
        result = real_messages_for_session(self, session_id)
        if counting["on"]:
            counts["messages_for_session"] += 1
            counts["snapshots"] += len(result)
            caller = inspect.currentframe().f_back
            if caller is not None:
                site = f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}"
                key = f"snapshot_caller:{site}"
                counts[key] = counts.get(key, 0) + 1
        return result

    monkeypatch.setattr(
        ConsoleChatStore, "messages_for_session", counted_messages_for_session
    )

    real_snapshot = ConsoleChatStore._snapshot

    def counted_snapshot(message: Any) -> Any:
        if counting["on"]:
            counts["snapshot_rows"] += 1
        return real_snapshot(message)

    monkeypatch.setattr(ConsoleChatStore, "_snapshot", staticmethod(counted_snapshot))

    def _count_projected_rows(module: Any, name: str, key: str) -> None:
        real = getattr(module, name)

        def counted(messages: Any, *args: Any, **kwargs: Any) -> Any:
            if counting["on"]:
                counts[key] += len(messages)
            return real(messages, *args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    _count_projected_rows(
        spend_module, "build_console_spend_history_projection", "spend_history_rows"
    )
    _count_projected_rows(
        spend_module, "build_console_current_cost_messages", "cost_rows"
    )
    _count_projected_rows(
        spend_module, "build_console_context_messages", "context_rows"
    )
    real_context_estimate = screen_module.build_console_context_estimate

    def counted_context_estimate(messages: Any, *args: Any, **kwargs: Any) -> Any:
        if counting["on"]:
            # Textual may coalesce a different number of one-row draft
            # repaint calls in each mounted app. The largest input to any
            # call is the deterministic O(N) signal: 400 means the whole
            # transcript returned to the typing path.
            counts["context_estimate_max_rows"] = max(
                counts["context_estimate_max_rows"], len(messages)
            )
        return real_context_estimate(messages, *args, **kwargs)

    monkeypatch.setattr(
        screen_module, "build_console_context_estimate", counted_context_estimate
    )

    def _count_calls(module: Any, name: str, key: str) -> None:
        real = getattr(module, name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            if counting["on"]:
                counts[key] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    # TASK-24301: the derivation legs. Patched on the modules the Console
    # session controller resolves them through, so a call that routes around
    # the memo is still seen.
    _count_calls(
        settings_module,
        "build_console_settings_readiness",
        "settings_readiness_builds",
    )
    _count_calls(
        session_module,
        "default_console_session_settings",
        "template_default_builds",
    )

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _settle(pilot)

        store = pilot.app.screen._ensure_console_chat_store()
        workspace_id = store.workspace_context.active_workspace_id
        session = store.ensure_session(title="census", workspace_id=workspace_id)
        # Restore the linear fixture in one pass. Appending 400 rows one at a
        # time repeatedly rebuilt the entire tree and made Windows setup
        # exceed the watchdog before the first measured keystroke.
        store._ingest_linear_messages(
            session.id,
            (
                ConsoleChatMessage(
                    role=(
                        ConsoleMessageRole.USER
                        if index % 2 == 0
                        else ConsoleMessageRole.ASSISTANT
                    ),
                    content=f"census message {index} " + ("lorem ipsum " * 6),
                )
                for index in range(seeded_messages)
            ),
        )
        assert store.message_count(session.id) == seeded_messages
        await _settle(pilot, passes=10)

        # The fixture itself changed the transcript. Pay the legitimate cold
        # projection rebuild before the measured unchanged typing burst; a
        # real restored conversation also paints context and cost before input.
        screen = pilot.app.screen
        screen._active_console_settings_context_estimate()
        screen._build_console_cost_state()

        # The composer is the DEFAULT focus at rest; never call focus() here.
        # The first Input in walk order is a settings field, and a probe that
        # focuses it types into the wrong widget and measures nothing.
        assert type(pilot.app.focused).__name__ == "ConsoleComposerBar", (
            "census is only meaningful with the composer focused; got "
            f"{type(pilot.app.focused).__name__}"
        )

        counting["on"] = True
        for _ in range(KEYSTROKES):
            await pilot.press("a")
        counting["on"] = False

    return counts


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_typing_never_snapshots_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """No keystroke materialises a transcript snapshot (TASK-24300).

    This is the guard the AC asks for: a predicate-shaped use of the
    snapshot API returning to the keystroke path fails here, and it fails on
    an EMPTY conversation too -- so the regression is caught before anyone
    has a long enough transcript to feel it.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    counts = await _census(monkeypatch, tmp_path, seeded_messages=SEEDED_MESSAGES)

    assert counts["messages_for_session"] == 0, (
        f"{counts['messages_for_session']} messages_for_session calls across "
        f"{KEYSTROKES} keystrokes ({counts['snapshots']} message snapshots; "
        f"callers={{{', '.join(f'{key}={value}' for key, value in counts.items() if key.startswith('snapshot_caller:'))}}} "
        "allocated). That call deep-copies the whole transcript, so any use "
        "of it on the keystroke path prices typing at O(conversation length) "
        "-- the TASK-24300 defect, which cost 12 ms per key at 400 messages. "
        "For an emptiness question use `has_messages`/`message_count`; to "
        "find the most recent match use `iter_messages_newest_first`."
    )


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_settled_400_message_typing_traverses_no_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Count actual mounted projection rows during unchanged long-chat typing."""
    counts = await _census(monkeypatch, tmp_path, seeded_messages=400)
    for key in (
        "messages_for_session",
        "snapshot_rows",
        "spend_history_rows",
        "cost_rows",
        "context_rows",
    ):
        assert counts[key] == 0, f"unchanged typing traversed {counts[key]} {key}"
    assert counts["context_estimate_max_rows"] <= 1


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_keystroke_work_does_not_scale_with_transcript_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Typing costs the same whether the conversation is empty or long.

    The absolute cost is deliberately not asserted -- see this module's
    docstring on why wall clock is not evidence here. What is asserted is
    that the census is IDENTICAL between the two transcript sizes, which is
    the property that actually broke: work that is O(N) in messages shows up
    as a difference between these two runs and nowhere else.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    empty = await _census(monkeypatch, tmp_path / "empty", seeded_messages=0)
    loaded = await _census(monkeypatch, tmp_path / "loaded", seeded_messages=400)

    assert empty == loaded, (
        f"per-keystroke work differs with transcript length: empty={empty}, "
        f"400 messages={loaded}. Something on the keystroke path is O(N) in "
        "the number of messages, which is what makes long conversations feel "
        "slower to type in than new ones."
    )

    for key in (
        "snapshot_rows",
        "spend_history_rows",
        "cost_rows",
        "context_rows",
    ):
        assert (
            loaded[key] == 0
        ), f"typing traversed settled transcript in {key}: {loaded[key]} rows"
    assert loaded["context_estimate_max_rows"] <= 1


#: Per-keystroke ceilings for the Console state derivation (TASK-24301),
#: measured on dev `3a3383123e` before/after. Template defaults reach ZERO
#: because that derivation is memoised across passes; readiness stays live on
#: purpose (it reads `os.environ` for credentials, and caching it against a
#: stale snapshot is the task-177 regression).
MAX_SETTINGS_READINESS_BUILDS_PER_KEY = 3
MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY = 0


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_typing_does_not_rebuild_the_provider_derivation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A keystroke re-derives the provider graph at most a bounded number of times.

    Before TASK-24301 a single printable key ran the template-defaults builder
    3.25 times and the readiness builder 4.35 times, and threw every result
    away: the equality gate that follows skips the DOM write but not the
    compute, and nothing in the derivation moves between two characters of a
    word.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    counts = await _census(monkeypatch, tmp_path, seeded_messages=0)

    template_per_key = counts["template_default_builds"] / KEYSTROKES
    readiness_per_key = counts["settings_readiness_builds"] / KEYSTROKES

    assert template_per_key <= MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY, (
        f"{template_per_key:.2f} template-default builds per keystroke "
        f"(budget {MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY}). This derivation is "
        "a pure function of (app_config, provider, model) and is memoised "
        "across passes; a non-zero count means something bypassed the memo."
    )
    assert readiness_per_key <= MAX_SETTINGS_READINESS_BUILDS_PER_KEY, (
        f"{readiness_per_key:.2f} readiness builds per keystroke (budget "
        f"{MAX_SETTINGS_READINESS_BUILDS_PER_KEY}). Readiness is deliberately "
        "NOT cached across passes -- it reads os.environ for credentials -- "
        "so the per-pass memo is the only thing keeping this bounded."
    )
