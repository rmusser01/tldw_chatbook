"""Pure display-state contracts for the Library Skills canvas.

Consumes record mappings shaped like ``LocalSkillsService.get_context``'s
``available_skills``/``blocked_skills`` envelopes (each a
``LocalSkillsService._summary_for_record`` dict: ``name``, ``description``,
``argument_hint``, ``user_invocable``, ``disable_model_invocation``,
``context``, plus the trust fields ``trust_status``/``trust_blocked``/
``trust_reason_code``/``trust_changed_files``/... from
``LocalSkillsService._trust_fields_for_record``) and detail mappings shaped
like ``LocalSkillsService.get_skill``'s response (a ``SkillResponse`` dump --
adds ``content``, ``supporting_files``, ``version`` -- plus the same trust
fields).

No Textual/DB/IO imports. The only non-stdlib imports are ``yaml`` (frontmatter
serialization -- matches the service's own use of it), the static
``LocalSkillsService._parse_front_matter`` frontmatter-splitting grammar (a
pure string function, reused rather than re-implemented so this module can
never drift from the service's actual parsing behavior), and the
``SkillTrustBlockedError`` exception type used to classify save outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from ..Skills_Interop.local_skills_service import LocalSkillsService
from ..Skills_Interop.skill_trust_models import SkillTrustBlockedError

# Task 2 spec (skills-200): names a fresh local skill would shadow if it were
# invoked by name -- the built-in agent tools (``spawn_subagent``,
# ``find_tools``, ``load_tools``, the Calculator/DateTime built-in tools) plus
# a handful of reserved slash-command-shaped words. Kept as a fixed literal
# set per the brief's explicit interface (rather than threaded in as
# caller-supplied "builtin tool names"/"registered command names" sets) --
# still a pure, deterministic predicate.
_SHADOWED_BUILTIN_NAMES = frozenset((
    "calculator",
    "get_current_datetime",
    "spawn_subagent",
    "find_tools",
    "load_tools",
    "prompt",
    "system",
    "skills",
))


@dataclass(frozen=True)
class SkillListRow:
    """One row in the Library Skills canvas's list view.

    Attributes:
        name: Display name, raw (the canvas escapes markup at render time).
        secondary: The flags line (``skill_flags_line``'s output) plus an
            optional description, joined with ``" · "`` (either part
            omitted, along with its separator, when absent).
        trust_glyph: ``"✓"`` when trusted, ``"⚠"`` when the skill needs
            trust review.
        blocked: Whether the skill is currently trust-blocked
            (``trust_blocked``) -- unusable until reviewed/re-trusted.
    """

    name: str
    secondary: str
    trust_glyph: str
    blocked: bool


@dataclass(frozen=True)
class SkillsListState:
    """Display state for the Library Skills canvas's list view.

    Attributes:
        rows: The skills to render, already filtered/sorted.
        count: ``len(rows)``.
        sort: The sort mode used to build ``rows`` (``"name"`` or
            ``"status"``), echoed back for the caller's toggle label.
    """

    rows: tuple[SkillListRow, ...]
    count: int
    sort: str


@dataclass(frozen=True)
class SkillEditorState:
    """Display state for the Library Skills canvas's in-canvas editor.

    Attributes:
        name: The skill's name.
        description: The skill's description.
        argument_hint: The skill's argument hint, or ``None`` when unset.
        allowed_tools_csv: The skill's ``allowed_tools`` list rendered as a
            single comma-separated string (``""`` when unset).
        user_invocable: Whether a user can invoke this skill directly.
        disable_model_invocation: Whether the agent is barred from
            invoking this skill on its own.
        context: The skill's execution context (``"inline"`` or
            ``"fork"``).
        model: The skill's model override, or ``None`` when unset.
        body: The skill's prompt body (the text after the frontmatter
            block), verbatim.
        supporting_files: The skill's supporting files as sorted
            ``(name, byte_length)`` pairs.
        version: The skill's optimistic-lock version, or ``None`` when
            unknown.
        trust_status: The skill's current trust status.
        trust_blocked: Whether the skill is currently trust-blocked.
        trust_changed_files: Files the trust service reports as changed
            since the last trusted snapshot.
    """

    name: str
    description: str
    argument_hint: str | None
    allowed_tools_csv: str
    user_invocable: bool
    disable_model_invocation: bool
    context: str
    model: str | None
    body: str
    supporting_files: tuple[tuple[str, int], ...]
    version: int | None
    trust_status: str
    trust_blocked: bool
    trust_changed_files: tuple[str, ...]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_text(value: Any) -> str:
    """Like ``_text`` but preserves body text verbatim (no stripping)."""
    return "" if value is None else str(value)


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _csv_from_list(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [text for item in value if (text := _text(item))]
        return ", ".join(items)
    return ""


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def skill_flags_line(user_invocable: bool, disable_model_invocation: bool) -> str:
    """Render the skill list/editor's user/agent invocability flags line.

    Args:
        user_invocable: Whether a user can invoke the skill directly.
        disable_model_invocation: Whether the agent is barred from
            invoking the skill on its own (note the inversion: the agent
            CAN invoke it when this is ``False``).

    Returns:
        ``"user · agent"`` when both a user and the agent can invoke it,
        ``"user"``/``"agent"`` when only one can, or ``"not invocable"``
        when neither can.
    """
    agent_invocable = not disable_model_invocation
    if user_invocable and agent_invocable:
        return "user · agent"
    if user_invocable:
        return "user"
    if agent_invocable:
        return "agent"
    return "not invocable"


def skill_name_shadows_builtin(name: str) -> str | None:
    """Return the builtin name ``name`` would shadow, or ``None``.

    Args:
        name: A candidate (or existing) local skill name.

    Returns:
        The normalized (stripped, lowercased) name when it collides with a
        reserved builtin tool/command name, else ``None``.
    """
    normalized = _text(name).lower()
    return normalized if normalized in _SHADOWED_BUILTIN_NAMES else None


def save_marks_needs_review(trust_status: str, trust_blocked: bool) -> bool:
    """Return whether saving the currently-open skill will re-quarantine it.

    Saving rewrites the skill's SKILL.md content, which changes its content
    hash -- so a skill that is trusted right now will always drop back to
    needs-review once the trust service re-fingerprints it post-save.

    Args:
        trust_status: The skill's trust status before the save.
        trust_blocked: Whether the skill is already trust-blocked before
            the save.

    Returns:
        ``True`` only when the skill is currently trusted and not already
        blocked (i.e. the save is the thing that will newly trigger
        needs-review).
    """
    return trust_status == "trusted" and not trust_blocked


def _matches_query(record: Mapping[str, Any], query_lower: str) -> bool:
    if not query_lower:
        return True
    if query_lower in _text(record.get("name")).lower():
        return True
    return query_lower in _text(record.get("description")).lower()


def _row(record: Mapping[str, Any], *, default_blocked: bool) -> SkillListRow | None:
    if not isinstance(record, Mapping):
        return None
    name = _text(record.get("name"))
    if not name:
        return None
    blocked = bool(record.get("trust_blocked", default_blocked))
    trust_glyph = "⚠" if blocked else "✓"
    flags = skill_flags_line(
        bool(record.get("user_invocable", True)),
        bool(record.get("disable_model_invocation", False)),
    )
    description = _text(record.get("description"))
    secondary = " · ".join(part for part in (flags, description) if part)
    return SkillListRow(name=name, secondary=secondary, trust_glyph=trust_glyph, blocked=blocked)


def build_skills_list_state(
    context_payload: Mapping[str, Any] | None,
    *,
    query: str,
    sort: str,
) -> SkillsListState:
    """Build the Library Skills canvas's list-view display state.

    Renders BOTH populations ``LocalSkillsService.get_context`` returns --
    ``available_skills`` (trusted, invocable today) and ``blocked_skills``
    (needs-review, per the Skills spec's blocked-skills visibility rule: a
    skill pending trust review is still a managed skill, just not
    invokable yet). Records missing a mapping shape or a name are silently
    dropped rather than raising.

    Args:
        context_payload: A mapping shaped like ``get_context``'s payload
            (``available_skills``/``blocked_skills`` lists of summary
            mappings). Malformed/``None`` degrades to an empty list.
        query: Filter text, matched case-insensitively against name and
            description; ``""`` disables filtering.
        sort: ``"status"`` sorts needs-review skills first, then
            alphabetically by name within each group. Any other value
            (including ``"name"``) sorts purely alphabetically
            case-insensitively.

    Returns:
        The list view's display state.
    """
    payload = context_payload if isinstance(context_payload, Mapping) else {}
    available = payload.get("available_skills") or ()
    blocked = payload.get("blocked_skills") or ()
    query_lower = _text(query).lower()

    rows: list[SkillListRow] = []
    for record in available:
        if isinstance(record, Mapping) and _matches_query(record, query_lower):
            row = _row(record, default_blocked=False)
            if row is not None:
                rows.append(row)
    for record in blocked:
        if isinstance(record, Mapping) and _matches_query(record, query_lower):
            row = _row(record, default_blocked=True)
            if row is not None:
                rows.append(row)

    if sort == "status":
        rows.sort(key=lambda row: (not row.blocked, row.name.lower()))
    else:
        rows.sort(key=lambda row: row.name.lower())

    rows_tuple = tuple(rows)
    return SkillsListState(rows=rows_tuple, count=len(rows_tuple), sort=sort)


def build_skill_editor_state(detail: Mapping[str, Any]) -> SkillEditorState:
    """Build the skill editor's display state from a ``get_skill`` detail mapping.

    Args:
        detail: A skill detail mapping shaped like
            ``LocalSkillsService.get_skill``'s response (or a
            malformed/empty mapping, tolerated). ``content`` is split into
            frontmatter/body using the exact same ``---\\n...\\n---``
            grammar the service itself parses with
            (``LocalSkillsService._parse_front_matter``), so this can
            never silently drift from what the service actually persists.

    Returns:
        Immutable editor state, with ``allowed_tools`` joined into a
        single comma-separated string and ``supporting_files`` reduced to
        sorted ``(name, byte_length)`` pairs.
    """
    if not isinstance(detail, Mapping):
        detail = {}
    content = _raw_text(detail.get("content"))
    _, body = LocalSkillsService._parse_front_matter(content)

    supporting_source = detail.get("supporting_files") or {}
    supporting_files = tuple(sorted(
        (name, len(str(text).encode("utf-8")))
        for name, text in supporting_source.items()
    ))

    changed_files = detail.get("trust_changed_files") or ()

    return SkillEditorState(
        name=_text(detail.get("name")),
        description=_text(detail.get("description")),
        argument_hint=_text(detail.get("argument_hint")) or None,
        allowed_tools_csv=_csv_from_list(detail.get("allowed_tools")),
        user_invocable=bool(detail.get("user_invocable", True)),
        disable_model_invocation=bool(detail.get("disable_model_invocation", False)),
        context=_text(detail.get("context")) or "inline",
        model=_text(detail.get("model")) or None,
        body=body,
        supporting_files=supporting_files,
        version=_to_int(detail.get("version")),
        trust_status=_text(detail.get("trust_status")) or "trusted",
        trust_blocked=bool(detail.get("trust_blocked", False)),
        trust_changed_files=tuple(str(item) for item in changed_files),
    )


def compose_skill_markdown(editor_state: SkillEditorState, *, body: str) -> str:
    """Assemble a SKILL.md string from editor state plus a (possibly edited) body.

    Emits only the frontmatter keys the editor owns -- ``name``,
    ``description``, ``argument_hint``, ``allowed_tools`` (as a list),
    ``user_invocable``, ``disable_model_invocation``, ``context``, and
    ``model`` (only when set) -- so this round-trips cleanly through the
    service's own ``_parse_front_matter``/``_metadata_from_content``
    parsing without ever emitting a key the service doesn't recognize.

    Args:
        editor_state: The editor's current display state (supplies every
            frontmatter field except the body).
        body: The (possibly just-edited) prompt body to write after the
            frontmatter block.

    Returns:
        A full SKILL.md string: ``---\\n<yaml frontmatter>\\n---\\n<body>``.
    """
    frontmatter: dict[str, Any] = {
        "name": editor_state.name,
        "description": editor_state.description,
        "argument_hint": editor_state.argument_hint,
        "allowed_tools": _split_csv(editor_state.allowed_tools_csv) or None,
        "user_invocable": editor_state.user_invocable,
        "disable_model_invocation": editor_state.disable_model_invocation,
        "context": editor_state.context,
    }
    if editor_state.model:
        frontmatter["model"] = editor_state.model
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False)
    return f"---\n{yaml_text}---\n{body}"


def classify_skill_save_error(result: Any, message: str, exc: Exception | None) -> str:
    """Classify the outcome of a local skill save (create/update) call.

    Args:
        result: The value the save call returned, or ``None``/falsy when
            it did not produce a fresh saved record.
        message: Any accompanying human-readable message/exception text
            from the save call.
        exc: The exception raised by the save call, if any.

    Returns:
        One of ``"exists"``, ``"version-conflict"``, ``"invalid-name"``,
        ``"trust-blocked"``, ``"ok"``, or ``"error"``.
    """
    message_text = _text(message)
    if "local_skill_exists:" in message_text:
        return "exists"
    if "local_skill_version_conflict:" in message_text:
        return "version-conflict"
    if isinstance(exc, ValueError) and "must contain only lowercase" in str(exc):
        return "invalid-name"
    if isinstance(exc, SkillTrustBlockedError):
        return "trust-blocked"
    if exc is None and isinstance(result, Mapping) and result.get("name") is not None:
        return "ok"
    return "error"
