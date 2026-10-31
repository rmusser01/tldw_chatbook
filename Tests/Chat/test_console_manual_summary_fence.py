"""TASK-33621.4 AC#4: /rewind Summarize commits instead of always going stale.

``capture_skill_context_maximum`` mints a fresh ``pending:`` plugin run id on
every capture. The manual summary's commit fence captured the turn
configuration before and after the summary call and compared the whole
snapshots, so with any plugin skill installed (the app ships one) every
/rewind Summarize ended "Conversation changed while summarizing. No memory
was saved." -- reproduced through the real ChatScreen and controller. The
hold card points to /rewind for its preview, so that path has to work.
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    commit_fence_configuration,
)


def _snapshot(
    run_id: str, skills=("character-creator",)
) -> ConsoleTurnConfigurationSnapshot:
    return ConsoleTurnConfigurationSnapshot(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(provider="openai"),
        skill_context_maximum={
            "plugin_run_id": run_id,
            "available_skills": tuple({"name": name} for name in skills),
        },
    )


def test_captures_differing_only_by_a_pending_run_id_fence_equal() -> None:
    first, second = _snapshot("pending:1"), _snapshot("pending:2")

    assert first != second
    assert commit_fence_configuration(first) == commit_fence_configuration(second)


def test_a_real_skill_change_still_fences() -> None:
    before = _snapshot("pending:1")
    after = _snapshot("pending:2", skills=("character-creator", "researcher"))

    assert commit_fence_configuration(before) != commit_fence_configuration(after)


def test_a_bound_run_id_is_a_fact_and_still_fences() -> None:
    assert commit_fence_configuration(_snapshot("run-1")) != (
        commit_fence_configuration(_snapshot("run-2"))
    )
