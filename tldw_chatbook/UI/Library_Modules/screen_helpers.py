"""Library screen module-level helper functions.

Moved verbatim out of ``tldw_chatbook/UI/Screens/library_screen.py`` by PR 0a
of the Library screen decomposition
(``.superpowers/sdd/2026-09-01-library-decomposition-foundation``; see
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``).
``library_screen.py`` re-exports every name here so its import surface is
unchanged; later decomposition tasks import directly from this module.

NOT moved here despite being module-level `FunctionDef`s above
``class LibraryScreen``: ``_read_library_ingest_options_from_config`` and
``_library_ingest_options_for`` (with their ``_INGEST_OPTIONS_CACHE_ATTR``
key). Several tests (``Tests/UI/test_library_ingest_options_cache.py``,
``Tests/UI/test_library_screen.py::test_load_ingest_options_from_config`` and
the three ``test_task_33*_options_round_trip_persisted_config`` tests)
monkeypatch ``get_cli_setting`` / ``_read_library_ingest_options_from_config``
on the ``library_screen`` module object. Before this move both functions
shared ``library_screen.py``'s globals, so the patch reached the internal
``_library_ingest_options_for`` -> ``_read_library_ingest_options_from_config``
call; moving them into this module gives that call ITS OWN globals dict,
silently bypassing the patch (Python resolves a free name via
``func.__globals__``, fixed at the function's *defining* module, not
wherever it is re-exported to) -- 5 tests fail deterministically. This is
the exact "monkeypatch bypass breaks tests inside a 'pure move'" risk the
design doc names, whose stated mitigation for the analogous
``*_local_source_snapshot`` trio is "stays screen-routed" -- applied here to
the same class of problem. See PR 0a's task report for the full trace.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rich.markup import escape as escape_markup

from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...STT.transcribe_cpp_config import is_gguf_file
from ...Third_Party.textual_fspicker import Filters
from .screen_constants import (
    LIBRARY_NOTE_BLANK_SEED_TITLE,
    LIBRARY_STUDY_HANDOFF_TITLES_CAP,
)


def _library_screen_is_current(screen: Any) -> bool:
    """Reject delayed callbacks owned by a replaced Library screen."""
    try:
        runtime_app = screen.app
        current_screen = getattr(runtime_app, "screen", screen)
    except Exception:
        return True
    return current_screen is screen


def library_note_persisted_title(raw_title: str) -> str:
    """The exact title a note with ``raw_title`` is actually stored under.

    (P0, xhigh review + live-verify round) The save port substitutes the
    seed title for a blank one on the wire (task-2858's reviewed decision:
    an emptied-out title persists as "Untitled", never as a blank row
    name), but ``DatabaseNotePortSaveReply`` carries no title back -- so
    the session snapshot's baseline kept the blank the draft had while the
    DB row was named "Untitled", and every list row patched from that
    snapshot inherited the disagreement. The substitution is a pure
    function of the payload title, so both sides derive it from HERE
    instead of one side guessing: the port before the write, the list
    patch after it.

    Args:
        raw_title: The draft/payload title exactly as the user left it.

    Returns:
        ``raw_title`` when it carries any non-whitespace text, otherwise
        :data:`LIBRARY_NOTE_BLANK_SEED_TITLE`.
    """
    return raw_title if raw_title.strip() else LIBRARY_NOTE_BLANK_SEED_TITLE


def _ingestible_file_filters() -> Filters:
    """Filters that separate importable files from the rest.

    The picker previously listed every file regardless of whether ingest
    could do anything with it, so a user could pick something that was only
    ever going to fail. The supported set is taken from the ingest capability
    layer, so it cannot drift from what the pipeline actually accepts.
    """
    from ...Library.ingest_capabilities import UNSUPPORTED_GROUP, get_type_group

    def _is_ingestible(path: Path) -> bool:
        try:
            return get_type_group(str(path)) != UNSUPPORTED_GROUP
        except Exception:
            return False

    return Filters(
        ("Importable files", _is_ingestible),
        ("All files", lambda _path: True),
    )


def _transcribe_cpp_gguf_filters() -> Filters:
    """Restrict the direct-local model picker to GGUF files."""
    return Filters(("GGUF models", is_gguf_file))


def _library_carries_forward_line(titles: Sequence[str]) -> str:
    """Build the handoff canvas's capped, markup-escaped carries-forward line.

    Args:
        titles: Sampled source titles (notes/media/conversations) that will
            carry forward into Study. Must be non-empty -- callers render no
            line at all when there is no source context (see
            ``_study_handoff_copy``).

    Returns:
        ``"Carries forward: a, b, c"`` when there are at most
        ``LIBRARY_STUDY_HANDOFF_TITLES_CAP`` titles, else ``"Carries
        forward: a, b, c and N more."`` with the remaining count appended.
    """
    escaped_titles = [escape_markup(title) for title in titles]
    capped = escaped_titles[:LIBRARY_STUDY_HANDOFF_TITLES_CAP]
    joined = ", ".join(capped)
    remaining = len(escaped_titles) - len(capped)
    if remaining > 0:
        return f"Carries forward: {joined} and {remaining} more."
    return f"Carries forward: {joined}"


def _unbreakable_size_text(size_text: str) -> str:
    """Drop the space between a formatted size's number and its unit, so
    the rail's narrow Details column never wraps mid-unit (task-2859 item
    5: "Prompts 144.0 / KB").

    A non-breaking space (U+00A0) was the first thing tried here and does
    NOT work: Rich's own word-wrap splitter (``rich._wrap.words``, used by
    every plain ``Static``) tokenizes on ``re.compile(r"\\s*\\S+\\s*")``,
    and Python's ``re`` module's Unicode-aware ``\\s`` matches U+00A0 the
    same as an ordinary space -- confirmed by reproducing the exact wrap
    live (rail width ~24-26 cells still split "144.0" from "KB" with the
    NBSP already in place) and again directly against ``rich._wrap`` at
    that width. Removing the space entirely denies the wrapper any
    character to split on there at all -- verified stable across widths
    20-29.

    ``get_formatted_file_size``/``get_formatted_db_size_with_wal`` values
    (e.g. ``"144.0 KB"``, ``"512 B"``) carry exactly one space; a fallback
    value with no space at all (``"?"``, ``"N/A"``, ``"Error"``) passes
    through unchanged.
    """
    return size_text.replace(" ", "")


def _active_library_sync_scope(app_instance: Any) -> dict[str, str | None]:
    runtime_policy = getattr(app_instance, "runtime_policy", None)
    runtime_state = runtime_policy.state if runtime_policy is not None else None
    active_source = str(
        getattr(runtime_state, "active_source", "local") or "local"
    ).lower()
    server_profile_id = getattr(runtime_state, "active_server_id", None)
    source_authority = (
        "server" if active_source == "server" and server_profile_id else "local"
    )
    authenticated_principal_id = None
    if source_authority == "server":
        server_context_provider = getattr(app_instance, "server_context_provider", None)
        get_active_context = getattr(
            server_context_provider, "get_active_context", None
        )
        if callable(get_active_context):
            try:
                authenticated_principal_id = event_principal_id_from_active_context(
                    get_active_context()
                )
            except Exception:
                authenticated_principal_id = None
    workspace_scope = None
    workspace_service = getattr(app_instance, "workspace_registry_service", None)
    get_active_workspace = getattr(workspace_service, "get_active_workspace", None)
    if callable(get_active_workspace):
        try:
            active_workspace = get_active_workspace()
            workspace_scope = getattr(active_workspace, "workspace_id", None)
        except Exception:
            workspace_scope = None
    return {
        "source_authority": source_authority,
        "server_profile_id": str(server_profile_id) if server_profile_id else None,
        "authenticated_principal_id": authenticated_principal_id,
        "workspace_scope": workspace_scope,
    }


def _record_value(record: Any, key: str, fallback: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(key, fallback)
    return getattr(record, key, fallback)


def _library_collection_record_data(record: Any) -> dict[str, Any]:
    return {
        "collection_id": _record_value(record, "collection_id"),
        "name": _record_value(record, "name"),
        "description": _record_value(record, "description"),
        "item_count": _record_value(record, "item_count", 0),
        "source_authority": _record_value(record, "source_authority", "local"),
        "sync_status": _record_value(record, "sync_status", "local-only"),
        "created_at": _record_value(record, "created_at"),
        "updated_at": _record_value(record, "updated_at"),
    }


def _library_collection_browse_summary(record: Any) -> dict[str, Any]:
    """Project one committed record into the strict bounded-page row shape."""

    return {
        "collection_id": _record_value(record, "collection_id"),
        "name": _record_value(record, "name"),
        "description": _record_value(record, "description"),
        "item_count": _record_value(record, "item_count", 0),
        "created_at": _record_value(record, "created_at"),
        "updated_at": _record_value(record, "updated_at"),
    }


def _collection_scoped_mirror_report(
    report: Mapping[str, Any] | None,
    collection_id: str,
) -> dict[str, Any] | None:
    if not report:
        return None
    actions = tuple(
        action
        for action in report.get("actions", ())
        if isinstance(action, Mapping)
        and isinstance(action.get("identity"), Mapping)
        and str(action["identity"].get("local_entity_id", "")) == collection_id
    )
    if not actions:
        return None
    scoped_report = dict(report)
    scoped_report["actions"] = actions
    scoped_report["mapped_count"] = len(actions)
    scoped_report["dry_run"] = bool(report.get("dry_run", True))
    scoped_report["write_enabled"] = bool(report.get("write_enabled", False))
    return scoped_report


def _collection_scoped_conflicts(
    conflict_reports: Sequence[Mapping[str, Any]],
    collection_id: str,
) -> tuple[Mapping[str, Any], ...]:
    scoped: list[Mapping[str, Any]] = []
    local_side_suffix = f":local:{collection_id}"
    remote_side_suffix = f":remote:{collection_id}"
    for conflict in conflict_reports:
        local_side_key = str(conflict.get("local_side_key") or "")
        remote_side_key = str(conflict.get("remote_side_key") or "")
        if local_side_key or remote_side_key:
            if local_side_key.endswith(local_side_suffix) or remote_side_key.endswith(
                remote_side_suffix
            ):
                scoped.append(conflict)
            continue
        details = conflict.get("details", {})
        if isinstance(details, Mapping):
            local_entity_id = details.get("local_entity_id")
            if local_entity_id is not None and str(local_entity_id) != collection_id:
                continue
        scoped.append(conflict)
    return tuple(scoped)


def _canonical_shortcut_key(key: str) -> str:
    """Fold a shortcut key label to its canonical dedupe form.

    (task-3312 #1) The footer's shared shortcut sets use display spellings
    ("esc", "F6") while ``BINDINGS`` uses Textual key names ("escape",
    "f6"); the F1 panel merges the two sources and must treat those as the
    SAME key or it advertises one action twice.

    Args:
        key: A shortcut key label from either source.

    Returns:
        A casefolded key with the "escape"/"esc" spelling unified.
    """
    lowered = key.strip().casefold()
    return "esc" if lowered == "escape" else lowered
