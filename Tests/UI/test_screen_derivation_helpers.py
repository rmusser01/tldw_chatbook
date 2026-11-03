"""Moved derivations retain screen state and live dependency ownership."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_command_suggestions import (
    CommandSuggestion,
    count_suggestion_usage,
    rank_suggestions_by_usage,
)
from tldw_chatbook.UI.Console_Modules.command_handoff import (
    console_unknown_command_hint,
    rank_console_command_suggestions,
    record_console_command_dispatch,
)
from tldw_chatbook.UI.Library_Modules.screen_helpers import (
    source_study_context,
    study_handoff_copy,
)
from tldw_chatbook.UI.Screens.study_scope_models import (
    StudyScopeContext,
    StudySourceItem,
)

pytestmark = pytest.mark.bootstrap_profile


def test_command_ranking_reuses_only_the_current_history_revision():
    entries = ["/model", "/model", "/help"]
    history = SimpleNamespace(revision=1, entry_texts=lambda: tuple(entries))
    screen = SimpleNamespace(
        _ensure_console_prompt_history=lambda: history,
        _console_suggestion_usage_cache=None,
        _console_command_registry=SimpleNamespace(
            available_names=lambda: ("help", "new")
        ),
    )
    source = {
        "count_suggestion_usage": count_suggestion_usage,
        "rank_suggestions_by_usage": rank_suggestions_by_usage,
    }
    suggestions = [
        CommandSuggestion(f"/{name} ", f"/{name}") for name in ("help", "model", "new")
    ]
    assert [
        row.label
        for row in rank_console_command_suggestions(screen, suggestions, source=source)
    ] == ["/model", "/help", "/new"]
    entries[:] = ["/help", "/help", "/help"]
    assert [
        row.label
        for row in rank_console_command_suggestions(screen, suggestions, source=source)
    ] == ["/model", "/help", "/new"]
    history.revision += 1
    assert [
        row.label
        for row in rank_console_command_suggestions(screen, suggestions, source=source)
    ] == ["/help", "/model", "/new"]
    screen._console_command_registry.available_names = lambda: ("fresh",)
    assert (
        console_unknown_command_hint(screen, "missing")
        == "Unknown command /missing — available: /fresh. Press Enter again to send as text."
    )


@pytest.mark.asyncio
async def test_command_recording_retains_minimal_token_and_live_post_await_logger():
    started, release = asyncio.Event(), asyncio.Event()
    entries, messages = [], []

    async def append(text):
        entries.append(text)
        started.set()
        await release.wait()
        raise RuntimeError("PRIVATE COMMAND ARGS")

    source = {
        "asyncio": asyncio,
        "COMMAND_PREFIX": "/",
        "SKILLS_COMMAND_NAME": "skills",
        "logger": SimpleNamespace(debug=lambda message: pytest.fail("stale logger")),
    }
    screen = SimpleNamespace(
        _ensure_console_prompt_history=lambda: SimpleNamespace(append=append)
    )
    before = asyncio.all_tasks()
    record_console_command_dispatch(
        screen,
        SimpleNamespace(name="SKILLS", args="name PRIVATE COMMAND ARGS"),
        source=source,
    )
    (task,) = asyncio.all_tasks() - before
    await started.wait()
    source["logger"] = SimpleNamespace(debug=messages.append)
    release.set()
    await task
    assert entries == ["/skills name"]
    assert messages == ["Console command usage recording failed (best-effort)"]


def test_library_study_derivations_keep_scope_and_current_source_copy():
    has_sources, mode = [True], ["local"]
    screen = SimpleNamespace(
        _has_local_sources=lambda: has_sources[0],
        _runtime_active_source=lambda: mode[0],
        _local_source_records={
            "notes": ({"id": "n1", "title": "Note"}, {"title": "missing"}),
            "media": ({"id": "m1", "title": "Media"},),
        },
        _source_record_id=lambda row: row.get("id"),
        _source_title=lambda kind, row: row["title"],
        _source_sample_titles=lambda kind: {
            "notes": ("Note",),
            "media": ("Media",),
            "conversations": ("Chat",),
        }[kind],
        _source_snapshot_body=lambda: "summary",
    )
    source = {
        "StudyScopeContext": StudyScopeContext,
        "StudySourceItem": StudySourceItem,
        "MATERIAL_SOURCE_LIBRARY": "library",
        "MATERIAL_TITLE_LIBRARY_SOURCES": "Local Library Sources",
        "LIBRARY_STUDY_HANDOFF_MODES": {
            "study": {
                "header": "Study",
                "action_label": "Study",
                "button_label": "Continue",
                "purpose": "Learn",
            }
        },
        "_library_carries_forward_line": lambda titles: ", ".join(titles),
        "LIBRARY_STUDY_HANDOFF_OWNERSHIP_COPY": "owner",
        "LIBRARY_STUDY_HANDOFF_READY_COPY": "server ready",
        "LIBRARY_STUDY_HANDOFF_LOCAL_MODE_COPY": "local manual",
    }
    context = source_study_context(screen, source=source)
    assert context.material_titles == ("Note", "Media", "Chat")
    assert context.material_summary == "summary"
    assert context.source_items == (
        StudySourceItem("note", "n1", "Note"),
        StudySourceItem("media", "m1", "Media"),
    )
    assert (
        study_handoff_copy(screen, "unknown", source=source)["recovery"]
        == "local manual"
    )
    mode[0] = "server"
    source["LIBRARY_STUDY_HANDOFF_READY_COPY"] = "current server ready"
    assert (
        study_handoff_copy(screen, "study", source=source)["recovery"]
        == "current server ready"
    )
    has_sources[0] = False
    assert source_study_context(screen, source=source) is None
    copy = study_handoff_copy(screen, "study", source=source)
    assert copy["context"] == ""
    assert (
        copy["recovery"]
        == "Import sources or create notes first, or open Study globally without Library context."
    )
