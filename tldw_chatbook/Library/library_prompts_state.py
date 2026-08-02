"""Pure display-state contracts for the Library prompts canvas.

Consumes record mappings shaped like ``PromptsDatabase.fetch_prompt_details``
/ ``list_prompts`` rows (keys: ``id``, ``name``, ``author``, ``details``,
``system_prompt``, ``user_prompt``, ``keywords``, ``last_modified`` /
``created_at``, ``version``). No Textual imports; the only DB import is the
``ConflictError`` exception type used to classify save outcomes.
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    ArtifactDefinitionState,
    ArtifactType,
    BlockArtifactDefinition,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
    decode_prompt_artifact,
)
from tldw_chatbook.Prompt_Management.prompt_legacy_decomposer import (
    decompose_legacy_lanes,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    CANONICAL_JSON_UTF8_V1,
    PromptCapabilityError,
    PromptSourceCapabilities,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
    set_artifact_type,
)

from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

_TIMESTAMP_KEYS = ("last_modified", "created_at")


@dataclass(frozen=True)
class PromptArtifactDraft:
    """Exact structured payload measurements used by Library save gates."""

    artifact_type: ArtifactType
    definition: BlockArtifactDefinition
    system_prompt: str
    user_prompt: str
    definition_bytes: bytes
    request_bytes: bytes


def _definition_mapping(definition: BlockArtifactDefinition) -> dict[str, Any]:
    """Serialize one validated block definition without optional null fields."""
    return {
        "kind": definition.kind,
        "schema_version": definition.schema_version,
        "lanes": [
            {
                "id": lane.id,
                "blocks": [
                    {
                        "id": block.id,
                        "title": block.title,
                        "syntax": block.syntax,
                        "content": block.content,
                        **(
                            {"xml_tag": block.xml_tag}
                            if block.xml_tag is not None
                            else {}
                        ),
                        **(
                            {"mapping_hint": block.mapping_hint}
                            if block.mapping_hint is not None
                            else {}
                        ),
                    }
                    for block in lane.blocks
                ],
            }
            for lane in definition.lanes
        ],
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prepare_prompt_artifact_save(
    state: PromptBlockEditorState,
    *,
    artifact_type: ArtifactType,
    include_recipe_starter_content: bool,
    request_fields: Mapping[str, Any],
) -> tuple[PromptArtifactDraft, dict[str, Any], PromptBlockEditorState]:
    """Build the exact structured save mapping and its measured working copy.

    Recipe starter content is opt-in. Turning it off clears only block content;
    stable IDs, lane order, titles, syntax, XML tags, and mapping hints remain.
    """
    if state.issues:
        raise ValueError("Fix block validation errors before saving.")
    prepared = set_artifact_type(state, artifact_type)
    if artifact_type == "recipe" and not include_recipe_starter_content:
        definition = replace(
            prepared.definition,
            lanes=tuple(
                replace(
                    lane,
                    blocks=tuple(replace(block, content="") for block in lane.blocks),
                )
                for lane in prepared.definition.lanes
            ),
        )
        prepared = PromptBlockEditorState.from_definition(
            artifact_type="recipe",
            definition=definition,
            dirty_block_ids=prepared.dirty_block_ids,
        )

    definition_mapping = _definition_mapping(prepared.definition)
    payload = {key: value for key, value in request_fields.items() if value is not None}
    payload.update(
        {
            "artifact_type": artifact_type,
            "prompt_format": "structured",
            "prompt_schema_version": prepared.definition.schema_version,
            "prompt_definition": definition_mapping,
            "system_prompt": prepared.compiled_system,
            "user_prompt": prepared.compiled_user,
        }
    )
    draft = PromptArtifactDraft(
        artifact_type=artifact_type,
        definition=prepared.definition,
        system_prompt=prepared.compiled_system,
        user_prompt=prepared.compiled_user,
        definition_bytes=_canonical_json_bytes(definition_mapping),
        request_bytes=_canonical_json_bytes(payload),
    )
    return draft, payload, prepared


def require_artifact_save_supported(
    draft: PromptArtifactDraft,
    capabilities: PromptSourceCapabilities,
    *,
    update_original: bool = False,
    expected_version: int | None = None,
) -> None:
    """Reject unsupported or oversized artifact saves without truncation."""
    expected_kind = (
        "block_prompt" if draft.artifact_type == "prompt" else "block_recipe"
    )
    if draft.definition.kind != expected_kind:
        raise ValueError("artifact_type and prompt definition kind must agree.")

    pair = (draft.definition.schema_version, draft.definition.kind)
    if pair not in capabilities.structured_kinds:
        raise PromptCapabilityError(capabilities.backend, f"structured kind {pair!r}")
    if draft.artifact_type not in capabilities.artifact_types:
        raise PromptCapabilityError(
            capabilities.backend, f"artifact type {draft.artifact_type!r}"
        )
    if capabilities.json_byte_measurement != CANONICAL_JSON_UTF8_V1:
        raise PromptCapabilityError(
            capabilities.backend, "canonical JSON byte measurement"
        )

    for field, value in (
        ("system_prompt", draft.system_prompt),
        ("user_prompt", draft.user_prompt),
    ):
        if len(value) > capabilities.compiled_lane_limit:
            raise ValueError(
                f"{field} exceeds {capabilities.compiled_lane_limit} characters; "
                "shorten this lane or choose a source with a larger limit."
            )
    for field, value, limit in (
        ("prompt_definition", draft.definition_bytes, capabilities.definition_limit),
        ("request", draft.request_bytes, capabilities.request_limit),
    ):
        if len(value) > limit:
            raise ValueError(
                f"{field} exceeds {limit} UTF-8 bytes; reduce that field or "
                "choose a source with a larger limit."
            )

    if not update_original:
        return
    if not capabilities.conditional_update:
        raise ValueError(
            "This source does not support conditional update; save as new."
        )
    if type(expected_version) is not int or expected_version < 1:
        raise ValueError(
            "Update original requires the captured current version; Reload or save as new."
        )


@dataclass(frozen=True)
class PromptListRow:
    """One row in the Library prompts canvas's list view.

    Attributes:
        prompt_id: The prompt's id.
        name: Display name, raw (the canvas escapes markup at render time).
        secondary: ``"<details> · <age>"`` -- the prompt's purpose, not
            ``author``/``keywords`` (Task 8b D2/U1; see ``_matches_query``'s
            comment for why keywords are never shown here) -- with either
            part (no details, or no timestamp) omitted, along with its
            separator.
    """

    prompt_id: int
    name: str
    secondary: str
    artifact_type: ArtifactType = "prompt"
    type_label: str = "Prompt"
    lane_summary: str = "Empty"
    source_label: str = "Local"


@dataclass(frozen=True)
class PromptsListState:
    """Display state for the Library prompts canvas's list view.

    Attributes:
        rows: The prompts to render, already filtered/sorted.
        count: ``len(rows)``.
        sort: The sort mode used to build ``rows`` (``"newest"`` or
            ``"name"``), echoed back for the caller's toggle label.
    """

    rows: tuple[PromptListRow, ...]
    count: int
    sort: str


@dataclass(frozen=True)
class PromptEditorState:
    """Display state for the Library prompts canvas's in-canvas editor.

    Attributes:
        prompt_id: The open prompt's id, or ``None`` when unknown/not yet
            saved.
        name: The prompt's name.
        author: The prompt's author.
        details: The prompt's description/details text.
        system_prompt: The prompt's system-prompt text.
        user_prompt: The prompt's user-prompt text.
        keywords_csv: The prompt's keywords as a single comma-separated
            string.
        version: The prompt's optimistic-lock version, or ``None`` when
            unknown.
        created: Raw ``created_at`` timestamp text, or ``""`` when absent.
        modified: Raw ``last_modified``/``created_at`` timestamp text, or
            ``""`` when absent.
    """

    prompt_id: int | None
    name: str
    author: str
    details: str
    system_prompt: str
    user_prompt: str
    keywords_csv: str
    version: int | None
    created: str
    modified: str
    artifact_type: ArtifactType = "prompt"
    definition_state: ArtifactDefinitionState = "legacy"
    block_editor_state: PromptBlockEditorState | None = None
    compiled_system_preview: str = ""
    compiled_user_preview: str = ""
    compatibility_stale: bool = False
    compatibility_reason: str = ""
    can_convert_as_new: bool = False
    source: str = "local"
    source_identity: str | None = None
    capabilities: PromptSourceCapabilities | None = None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_text(value: Any) -> str:
    """Like ``_text`` but preserves body text verbatim (no stripping)."""
    return "" if value is None else str(value)


def _timestamp_raw(record: Mapping[str, Any]) -> str:
    for key in _TIMESTAMP_KEYS:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _csv_from_keywords(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        items = []
        for item in value:
            if isinstance(item, Mapping):
                item = item.get("keyword") or item.get("text") or item.get("label")
            text = _text(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return ""


def _matches_query(record: Mapping[str, Any], query_lower: str) -> bool:
    if not query_lower:
        return True
    if query_lower in _text(record.get("name")).lower():
        return True
    # Task 8b D2: matches `details`, NOT `keywords` -- the raw local
    # `list_prompts` DB query has no per-page-keyword-join seam (only a
    # per-single-id `fetch_keywords_for_prompt`, an N+1 shape for a whole
    # page), so real list-page records never carry `keywords` at all (see
    # `_prompts_page_records_or_empty`'s docstring). Matching on it here
    # would silently promise a capability the filter could never actually
    # deliver. `details` IS present on list rows (the DB query now selects
    # it too), so this is the honest, still-cheap (no extra query) fix.
    # Keyword-in-list filtering awaits a batched per-page keyword-join DB
    # seam (backlog) if that capability is ever wanted.
    return query_lower in _text(record.get("details")).lower()


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_editor_prompt_id(detail: Mapping[str, Any]) -> int | None:
    """Resolve a prompt detail mapping's raw numeric id, robustly.

    The REAL production seam (``PromptScopeService.get_prompt`` ->
    ``normalize_prompt_record``, see
    ``tldw_chatbook/Prompt_Management/prompt_normalizers.py``) returns
    ``detail["id"]`` as a COMPOSITE STRING (``"<backend>:prompt:<uuid>"``)
    -- the raw local numeric id lives under ``detail["local_id"]`` instead.
    Preferring ``local_id`` here (when it resolves to an int) fixes that
    seam. Falling back to ``id`` keeps the other, non-composite-shaped
    callers working: the raw ``PromptsDatabase.fetch_prompt_details``/
    ``list_prompts`` row shape (``id`` IS the raw int, no ``local_id`` key
    at all) used directly by a handful of call sites/tests, and the
    post-save ``patched_detail`` the screen builds itself (which always
    writes a raw int straight into ``id``). ``_to_int`` on a composite
    string (or ``None``) naturally returns ``None``, so this never
    resolves a truly blank/new-editor detail (neither key present, e.g.
    the D1 create/Duplicate-action shapes) to anything but ``None``.

    Args:
        detail: A prompt detail mapping.

    Returns:
        The resolved int id, or ``None`` when unresolvable.
    """
    local_id = _to_int(detail.get("local_id"))
    if local_id is not None:
        return local_id
    return _to_int(detail.get("id"))


def _row(record: Mapping[str, Any], *, now: datetime) -> PromptListRow | None:
    prompt_id = _to_int(record.get("id"))
    if prompt_id is None:
        return None
    # Task 8b D2/U1: surfaces the prompt's PURPOSE (details) instead of
    # `author · age` -- author (and keywords, never present on list rows
    # anyway -- see `_matches_query`'s comment) are dropped from the
    # secondary line entirely.
    details = _text(record.get("details"))
    raw_timestamp = _timestamp_raw(record)
    age = format_console_relative_age(raw_timestamp, now=now) if raw_timestamp else ""
    secondary = " · ".join(part for part in (details, age) if part)
    artifact_type: ArtifactType = (
        "recipe" if record.get("artifact_type") == "recipe" else "prompt"
    )
    has_system = record.get("has_system_prompt")
    if not isinstance(has_system, bool):
        has_system = bool(_raw_text(record.get("system_prompt")).strip())
    has_user = record.get("has_user_prompt")
    if not isinstance(has_user, bool):
        has_user = bool(_raw_text(record.get("user_prompt")).strip())
    if has_system and has_user:
        lane_summary = "System + User"
    elif has_system:
        lane_summary = "System only"
    elif has_user:
        lane_summary = "User only"
    else:
        lane_summary = "Empty"
    source = _text(record.get("backend")) or "local"
    return PromptListRow(
        prompt_id=prompt_id,
        name=_text(record.get("name")),
        secondary=secondary,
        artifact_type=artifact_type,
        type_label=artifact_type.title(),
        lane_summary=lane_summary,
        source_label=source.title(),
    )


def build_prompts_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    query: str,
    sort: str,
    now: datetime,
) -> PromptsListState:
    """Build the Library prompts canvas's list-view display state.

    Records missing a mapping shape or a convertible ``id`` are silently
    dropped rather than raising, matching the Library notes state module's
    degrade-don't-crash behavior for malformed source records.

    Args:
        records: The prompts to render.
        query: Filter text, matched case-insensitively against name and
            details (Task 8b D2 -- not keywords, a field list-page records
            never actually carry; see ``_matches_query``); ``""`` disables
            filtering.
        sort: ``"name"`` sorts alphabetically case-insensitively; any other
            value (including ``"newest"``) sorts by most-recent
            modified/created timestamp, newest first.
        now: Reference time for the secondary line's relative-age part.

    Returns:
        The list view's display state.
    """
    query_lower = _text(query).lower()
    items = [
        record
        for record in (records or ())
        if isinstance(record, Mapping) and _matches_query(record, query_lower)
    ]
    if sort == "name":
        items.sort(key=lambda record: _text(record.get("name")).lower())
    else:
        items.sort(key=_timestamp_raw, reverse=True)
    rows = tuple(
        row for row in (_row(record, now=now) for record in items) if row is not None
    )
    return PromptsListState(rows=rows, count=len(rows), sort=sort)


def build_prompt_editor_state(
    detail: Mapping[str, Any],
    *,
    capabilities: PromptSourceCapabilities | None = None,
) -> PromptEditorState:
    """Build the prompt editor's display state from a prompt detail mapping.

    Args:
        detail: A prompt detail mapping -- either the raw
            ``fetch_prompt_details`` row shape (``id`` IS the raw int,
            ``keywords`` a list of strings), or the normalized
            ``PromptScopeService.get_prompt``/``normalize_prompt_record``
            shape (``id`` a composite ``"<backend>:prompt:<uuid>"``
            string, the raw int under ``local_id`` instead -- see
            ``_resolve_editor_prompt_id``), or a malformed/empty mapping.
            Tolerated to have missing/None fields.

    Returns:
        Immutable editor state, with keywords joined into a single
        comma-separated string.
    """
    if not isinstance(detail, Mapping):
        detail = {}
    try:
        decoded = decode_prompt_artifact(detail)
    except (TypeError, ValueError):
        decoded = None

    artifact_type: ArtifactType = (
        decoded.artifact_type
        if decoded is not None
        else cast(
            ArtifactType,
            "recipe" if detail.get("artifact_type") == "recipe" else "prompt",
        )
    )
    definition_state: ArtifactDefinitionState = (
        decoded.state if decoded is not None else "malformed"
    )
    compiled_system = (
        decoded.compiled_system
        if decoded is not None
        else _raw_text(detail.get("system_prompt"))
    )
    compiled_user = (
        decoded.compiled_user
        if decoded is not None
        else _raw_text(detail.get("user_prompt"))
    )
    block_state: PromptBlockEditorState | None = None
    if decoded is not None and decoded.state == "supported_v2":
        try:
            block_state = PromptBlockEditorState.from_definition(
                artifact_type=decoded.artifact_type,
                definition=decoded.definition,
            )
        except (TypeError, ValueError):
            definition_state = "malformed"
    elif (
        decoded is not None and decoded.state == "legacy" and artifact_type == "prompt"
    ):
        decomposition = decompose_legacy_lanes(compiled_system, compiled_user)
        block_state = PromptBlockEditorState.from_definition(
            artifact_type="prompt",
            definition=decomposition.definition,
            system_origin=decomposition.system_origin,
            user_origin=decomposition.user_origin,
        )

    compatibility_reason = ""
    if block_state is None:
        compatibility_reason = (
            f"{definition_state.replace('_', ' ')} artifact is read-only; "
            "use compatibility text and convert only as a new Prompt."
        )
    source = _text(detail.get("backend")) or "local"
    source_identity_value = detail.get("id", detail.get("uuid"))
    source_identity = (
        str(source_identity_value) if source_identity_value not in (None, "") else None
    )
    return PromptEditorState(
        prompt_id=_resolve_editor_prompt_id(detail),
        name=_text(detail.get("name")),
        author=_text(detail.get("author")),
        details=_raw_text(detail.get("details")),
        system_prompt=_raw_text(detail.get("system_prompt")),
        user_prompt=_raw_text(detail.get("user_prompt")),
        keywords_csv=_csv_from_keywords(detail.get("keywords")),
        version=_to_int(detail.get("version")),
        created=_text(detail.get("created_at")),
        modified=_timestamp_raw(detail),
        artifact_type=artifact_type,
        definition_state=definition_state,
        block_editor_state=block_state,
        compiled_system_preview=compiled_system,
        compiled_user_preview=compiled_user,
        compatibility_stale=bool(
            decoded.compatibility_stale if decoded is not None else False
        ),
        compatibility_reason=compatibility_reason,
        can_convert_as_new=bool(
            block_state is None and (compiled_system or compiled_user)
        ),
        source=source,
        source_identity=source_identity,
        capabilities=capabilities,
    )


def prompt_editor_meta_line(
    editor_state: PromptEditorState, *, now: datetime | None = None, dirty: bool = False
) -> str:
    """Render the prompt editor's muted meta line.

    Unlike the notes editor's ``meta_line`` (precomputed as part of
    ``LibraryNoteEditorState`` by ``build_library_note_editor_state``),
    ``PromptEditorState`` carries only raw ``modified``/``version`` fields
    -- shared here (rather than duplicated) so both the editor canvas's
    initial render and the screen's post-save targeted Static update agree
    on the exact same text. The Prompts table has no ``created_at`` column
    at all, so this renders only ``Modified <age>`` (never a "Created"
    part, and never a fake one) plus ``vN``.

    Args:
        editor_state: The prompt editor's current display state.
        now: Reference time for the relative-age part; defaults to the
            current UTC time.
        dirty: Task 8c U6: whether the editor has unsaved in-progress
            edits. A plain pure-function input (never derived from
            ``editor_state`` itself, which only ever reflects the
            last-saved record) -- callers thread the screen's own
            ``_library_prompt_dirty`` flag through. Defaults to ``False``
            so every pre-existing call site is unaffected.

    Returns:
        ``"New prompt"`` when ``editor_state.prompt_id`` is ``None`` (the
        Task 8b D1 create-flow sentinel: a blank, not-yet-saved record --
        see ``library_screen.py``'s ``_enter_library_prompt_create_editor``
        and the Duplicate action). Otherwise ``"Modified <age> · vN"``,
        with either part omitted (and its separator) when unknown. Either
        form gets a trailing ``"· • Unsaved changes"`` appended when
        ``dirty`` is ``True`` -- the only visible cue today that explicit
        Save/the nav-away dirty veto (``flush_pending_work``) has anything
        to act on.
    """
    if editor_state.prompt_id is None:
        base = "New prompt"
    else:
        reference_now = now if now is not None else datetime.now(timezone.utc)
        parts: list[str] = []
        if editor_state.modified:
            age = format_console_relative_age(editor_state.modified, now=reference_now)
            parts.append(f"Modified {age}")
        if editor_state.version is not None:
            parts.append(f"v{editor_state.version}")
        base = " · ".join(parts)
    if not dirty:
        return base
    return f"{base} · • Unsaved changes" if base else "• Unsaved changes"


def _is_name_conflict(exc: Exception | None, message_lower: str) -> bool:
    if isinstance(exc, sqlite3.IntegrityError) and "unique" in str(exc).lower():
        return True
    return "unique" in message_lower or "already exists" in message_lower


def classify_prompt_save_error(
    result_id: Any, message: str, exc: Exception | None
) -> str:
    """Classify the outcome of a prompt save (add/update) call.

    Args:
        result_id: The id the save call returned, or ``None`` when it did
            not produce a fresh saved row.
        message: Any accompanying human-readable message from the save
            call (e.g. the ``add_prompt`` tuple's message slot).
        exc: The exception raised by the save call, if any.

    Returns:
        One of ``"soft-deleted-name"``, ``"conflict"``, ``"name-in-use"``,
        ``"ok"``, or ``"error"``.
    """
    message_lower = _text(message).lower()
    if result_id is None and "soft-deleted" in message_lower:
        return "soft-deleted-name"
    if isinstance(exc, ConflictError):
        return "conflict"
    if _is_name_conflict(exc, message_lower):
        return "name-in-use"
    if exc is None and result_id is not None:
        return "ok"
    return "error"
