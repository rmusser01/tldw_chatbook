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

import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tldw_chatbook.Utils.input_validation import escape_markup

from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...STT.transcribe_cpp_config import is_gguf_file
from ...Third_Party.textual_fspicker import Filters
from ...Library.library_notes_session import NoteFlushOutcomeKind
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


def _active_library_principal_id(app_instance: Any) -> str | None:
    """Authenticated principal scope for the active server context.

    Reads the auth token, which can be an OS-keyring read that blocks for
    seconds on Linux (TASK-32926): call it from a worker thread, never from
    the event loop.
    """
    server_context_provider = getattr(app_instance, "server_context_provider", None)
    get_active_context = getattr(server_context_provider, "get_active_context", None)
    if not callable(get_active_context):
        return None
    try:
        return event_principal_id_from_active_context(get_active_context())
    except Exception:
        return None


def _active_library_sync_scope(
    app_instance: Any, *, resolve_principal: bool = True
) -> dict[str, str | None]:
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
    if source_authority == "server" and resolve_principal:
        authenticated_principal_id = _active_library_principal_id(app_instance)
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


def _assign_library_reader_preferences_attribute(
    owner: Any, attribute: str, value: Any
) -> None:
    """Write through a possibly-dotted attribute path off ``owner``.

    Task 9 (Conversations cleanup) support: ``_replace_library_reader_preference``
    and ``_persist_library_reader_preference`` dispatch across every reader
    destination (media, collections, conversations, notes, notes_files,
    prompts, skills) through a ``{destination: attribute_name}`` dict, read
    with plain ``getattr``/``operator.attrgetter`` and written with plain
    ``setattr``. Every destination except conversations and collections still
    keeps its reader-preferences object as a flat screen attribute, so a bare
    attribute-name string has always been enough. Conversations' own
    ``reader_preferences`` field moved to ``self._conversations_state.reader_preferences``
    (Task 6/9) -- one extra hop the generic dispatch's plain ``setattr``
    cannot express. This resolves the last (dotted) segment's owner via
    ``operator.attrgetter`` and assigns onto it, and is a no-op passthrough
    (``setattr(owner, attribute, value)``) for every other, undotted,
    destination -- so the five not-yet-extracted subsystems are unaffected.
    Future subsystem extractions hit this exact same shape; this helper is
    meant to keep serving them, not to be re-derived per subsystem.

    Second use, added by Task 4 (Export cleanup): ``_close_open_library_
    choice_strip`` dispatches across a DIFFERENT dict-of-name-strings
    (media/prompts/skills/export choice-strip visibility, built by
    ``_library_open_choice_strip``) with the identical possibly-dotted-path
    shape -- Export's own visibility field moved to ``self._export_state.
    quality_choices_visible`` (Task 2/4), while media/prompts/skills keep
    flat screen attributes, so the same generic dotted-vs-flat passthrough
    this docstring already describes serves that dispatcher too, without a
    second near-identical helper.

    Third use, added by Task 7 (Collections cleanup): the same two dicts'
    ``"collections"`` entry moved from the flat ``_library_collections_
    reader_preferences`` name to ``self._collections_state.reader_preferences``
    (Task 5/7) -- exactly the same dotted-vs-flat shape Conversations already
    established, requiring no change to this helper's own logic.
    """
    head, _, tail = attribute.rpartition(".")
    target = operator.attrgetter(head)(owner) if head else owner
    setattr(target, tail, value)


def _library_note_editor_exit_veto_message(kind: NoteFlushOutcomeKind) -> str:
    """The user-facing "why" and "what to do" for one flush-veto kind.

    task-32133 AC#1 / fix round 1 Important 2: the mandated copy ("fix the
    title or press Discard new note") is specific to VALIDATION_VETO; the
    other four kinds each name their own real state and next step instead
    of reusing that sentence verbatim. ``NoteFlushOutcome.message`` already
    carries an accurate-but-technical string for these (meant for the
    status line, e.g. "A destructive action is in progress."); this is the
    same information reworded for a one-shot toast.
    """
    if kind is NoteFlushOutcomeKind.VALIDATION_VETO:
        return "Can't leave yet — fix the title or press Discard new note."
    if kind is NoteFlushOutcomeKind.FAILED:
        return "Can't leave yet — the save failed; press Save to retry or Discard."
    if kind is NoteFlushOutcomeKind.CONFLICTED:
        return (
            "Can't leave yet — this note changed elsewhere; "
            "choose Overwrite or Reload."
        )
    if kind is NoteFlushOutcomeKind.BLOCKED:
        return "Can't leave yet — another action is already in progress; wait for it to finish."
    return "Can't leave yet — the note changed while saving; try again."  # STALE


def _review_footer_entries(
    progress: str, *, at_last: bool = False
) -> tuple[tuple[str, str], ...]:
    """The Reader footer's review-set segment for one progress line.

    task-31225 (re-critique P2): on a COMPLETE set the final ``]`` is an
    idempotent no-op, so advertising it violates the honest-footer rule
    (task-28005). Completion keeps ``m`` (un-marking resumes the walk)
    and names ``R`` as the next step. The completion check reads the
    canonical ``format_review_progress`` "All N reviewed" form.

    task-31271 seam (c): on the LAST live item a forward ``]`` marks
    that item done in place rather than walking anywhere
    (``plan_walk``'s completion gesture), so "next in set" promised an
    item that does not exist (critique #4, B cap_50).

    Args:
        progress: The formatted live progress line.
        at_last: Whether the cursor sits on the last live item, so the
            next ``]`` completes the set instead of advancing.

    Returns:
        ``(key, label)`` entries for the footer.
    """
    if progress.startswith("All "):
        return (
            ("m", "toggle reviewed"),
            ("R", "finish review"),
            ("", progress),
        )
    return (
        # task-31635 (critique #5 item 9, ruling): a forward step marks
        # the item you leave done, and "next in set" hid that -- users
        # read the mark as an accident. The behaviour stays (it is the
        # set's contract since task-31233); the chip stops being coy.
        ("]", "finish review" if at_last else "next (marks reviewed)"),
        ("[", "prev in set"),
        ("m", "toggle reviewed"),
        ("R", "exit review"),
        ("", progress),
    )
