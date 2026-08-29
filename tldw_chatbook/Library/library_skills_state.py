"""Pure display-state contracts for the Library Skills canvas.

Consumes record mappings shaped like ``LocalSkillsService.get_context``'s
``available_skills``/``blocked_skills`` envelopes (each a
``LocalSkillsService._summary_for_record`` dict: ``name``, ``description``,
``argument_hint``, ``user_invocable``, ``disable_model_invocation``,
``context``, plus the trust fields ``trust_status``/``trust_blocked``/
``trust_reason_code``/``trust_changed_files``/... from
``LocalSkillsService._trust_fields_for_record``) and detail mappings shaped
like ``LocalSkillsService.get_skill``'s response (a ``SkillResponse`` dump --
adds ``content``, ``supporting_files``, ``bundle_files``, ``version`` -- plus
the same trust fields).

No Textual/DB/IO imports. The only non-stdlib imports are ``yaml`` (frontmatter
serialization -- matches the service's own use of it), the static
``LocalSkillsService._parse_front_matter`` frontmatter-splitting grammar (a
pure string function, reused rather than re-implemented so this module can
never drift from the service's actual parsing behavior), and the
``SkillTrustBlockedError`` exception type used to classify save outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, cast

import yaml

from ..Skills_Interop.local_skills_service import LocalSkillsService
from ..Skills_Interop.skill_trust_models import SkillTrustBlockedError
from .library_pager_state import LibraryPagerDisplay

# Task 2 spec (skills-200): names a fresh local skill would shadow if it were
# invoked by name -- the built-in agent tools (``spawn_subagent``,
# ``find_tools``, ``load_tools``, the Calculator/DateTime built-in tools) plus
# a handful of reserved slash-command-shaped words. Kept as a fixed literal
# set per the brief's explicit interface (rather than threaded in as
# caller-supplied "builtin tool names"/"registered command names" sets) --
# still a pure, deterministic predicate.
_SHADOWED_BUILTIN_NAMES = frozenset(
    (
        "calculator",
        "get_current_datetime",
        "spawn_subagent",
        "find_tools",
        "load_tools",
        "fewer-permission-prompts",
        "prompt",
        # PR #729's /prefill command (sync test flagged the gap).
        "prefill",
        "system",
        "skills",
        # The skill_file runtime tool (reference-file reachability; the
        # drift-guard sync test flagged the gap when it joined
        # RUNTIME_TOOL_NAMES).
        "skill_file",
        # The install_skill runtime tool (same drift-guard rationale as
        # skill_file above).
        "install_skill",
        # The run_skill_script runtime tool (same drift-guard rationale as
        # skill_file/install_skill above).
        "run_skill_script",
        # The fleet's send_to_agent runtime tool (supervisor-fleet PR 3b) --
        # caught by the four-source guard TASK-13214 rebuilt, which is what
        # it was rebuilt FOR: the previous shape short-circuited on the first
        # failing subset, so each new name hid behind the last one.
        "send_to_agent",
        # The /research Console command (task-16481) -- the drift guard's
        # third sighting (TASK-13214): it was MASKED behind the video-command
        # gap until the guard learned to report all sources at once.
        "research",
        # The expand_document gated builtin (TASK-16174). Gated-OFF tools are
        # invisible to the live catalog, so the guard now reads the gate
        # TABLE (TASK-13214/F6) -- and every gateable tool is listed here
        # regardless of gate state.
        "expand_document",
        # The agent run-log search runtime tool must not be shadowed by an
        # installed skill with the same invocation name.
        "search_run_log",
        # The primary-agent run-log aggregation and contiguous-range tools
        # share the same reserved runtime namespace.
        "run_log_stats",
        "run_log_slice",
        # The fleet tools (PR2a): the supervisor collects concurrent
        # sub-agents with these, so a skill installed under either name
        # would shadow the runtime tool the moment it is invoked by name.
        # Same reserved-namespace rule as every other RUNTIME_TOOL_NAMES
        # entry above.
        "wait_agents",
        "check_agents",
        # task-580: console commands from the /rewind and image-generation
        # features. These were added to the command registry without updating
        # this set, so the drift guard below failed and was carried as an
        # accepted baseline across several branches -- which is exactly the
        # signal-erosion the guard exists to prevent.
        "rewind",
        "generate-image",
        # task-15210: and again, exactly as task-580 predicted -- the video
        # commands (commits d6c2e9756 /generate-video, 72a2ff3c5
        # /stream-video) were registered without updating this set, so the
        # drift guard sat red until something ran the file whole.
        "generate-video",
        "stream-video",
        # The sandbox-rooted file tools. These are CONFIG-GATED (off by
        # default), so the drift guard -- which builds a BuiltinToolProvider
        # with default config -- cannot see them and would not have caught
        # their absence. Listed explicitly: a skill named after one of them
        # still shadows a real builtin the moment a user enables the gate.
        "read_file",
        "list_directory",
        # TASK-545 P2's mutating tools. Same rationale as the two above:
        # CONFIG-GATED, so the drift guard (which builds a
        # BuiltinToolProvider with default config) cannot see them. A skill
        # named `write_file` shadows a real builtin the moment a user turns
        # the gate on.
        "write_file",
        "create_note",
        "update_note",
        # The sensitive-path-hardening glob_files/grep_files tools. Same
        # rationale as the file tools above: CONFIG-GATED, so the drift
        # guard (which builds a BuiltinToolProvider with default config)
        # cannot see them.
        "glob_files",
        "grep_files",
    )
)

SkillEditorMode = Literal["basic", "advanced"]
SkillReaderMode = Literal["overview", "edit", "trust", "files"]

DEFAULT_SKILL_BROWSE_PAGE_SIZE = 20
MAX_SKILL_BROWSE_PAGE_SIZE = 100
MAX_SKILL_BROWSE_PAGE = (2**63 - 1) // DEFAULT_SKILL_BROWSE_PAGE_SIZE + 1
SkillBrowseStatus = Literal["loading", "ready", "empty", "no_matches", "error"]


@dataclass(frozen=True)
class SkillBrowseScope:
    """One normalized local-only Library Skills page request."""

    backend: Literal["local"] = "local"
    query: str = ""
    sort: Literal["name", "status"] = "name"
    page: int = 1
    page_size: int = DEFAULT_SKILL_BROWSE_PAGE_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend.strip().lower() != "local":
            raise ValueError("Skill browsing is local-only.")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if not isinstance(self.sort, str):
            raise TypeError("sort must be a string.")
        normalized_sort = self.sort.strip().lower()
        if normalized_sort not in {"name", "status"}:
            raise ValueError("sort must be 'name' or 'status'.")
        if type(self.page) is not int or not 1 <= self.page <= MAX_SKILL_BROWSE_PAGE:
            raise ValueError("page is outside the supported range.")
        if (
            type(self.page_size) is not int
            or not 1 <= self.page_size <= MAX_SKILL_BROWSE_PAGE_SIZE
        ):
            raise ValueError("page_size must be between 1 and 100.")
        object.__setattr__(self, "backend", "local")
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "sort", normalized_sort)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            (self.backend, self.query, self.sort, self.page, self.page_size),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _skill_browse_integer(value: Any, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}.")
    return value


def _freeze_skill_browse_value(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Skill browse mappings must use string keys.")
        return MappingProxyType(
            {key: _freeze_skill_browse_value(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_skill_browse_value(item) for item in value)
    raise TypeError("Skill browse values must be JSON-like immutable data.")


def validate_skill_browse_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate and detach stable Skill page summaries."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise TypeError("Skill browse items must be a sequence.")
    identities: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("Skill browse items must be mappings.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Skill browse item name must be stable non-blank text.")
        if name in identities:
            raise ValueError("Skill browse item names must be unique.")
        if type(item.get("trust_blocked")) is not bool:
            raise ValueError("Skill browse item trust_blocked must be a boolean.")
        identities.add(name)
        validated.append(cast(Mapping[str, Any], _freeze_skill_browse_value(item)))
    return tuple(validated)


@dataclass(frozen=True)
class SkillBrowseResult:
    """Immutable loading, exact page, or failure state for Skills browsing."""

    scope: SkillBrowseScope
    items: tuple[Mapping[str, Any], ...]
    total_items: int
    page: int
    status: SkillBrowseStatus
    request_fingerprint: str
    request_token: int
    blocked_total: int = 0
    first_blocked_skill_name: str | None = None
    error: str = ""
    requested_page: int | None = None

    @property
    def total_pages(self) -> int:
        """Return the exact non-zero number of pages represented."""
        return max(
            1,
            (self.total_items + self.scope.page_size - 1) // self.scope.page_size,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.scope, SkillBrowseScope):
            raise TypeError("scope must be a SkillBrowseScope.")
        total = _skill_browse_integer(self.total_items, field="total_items", minimum=0)
        page = _skill_browse_integer(self.page, field="page", minimum=1)
        token = _skill_browse_integer(
            self.request_token, field="request_token", minimum=1
        )
        blocked_total = _skill_browse_integer(
            self.blocked_total, field="blocked_total", minimum=0
        )
        requested_page = (
            self.scope.page
            if self.requested_page is None
            else _skill_browse_integer(
                self.requested_page, field="requested_page", minimum=1
            )
        )
        expected_request = replace(self.scope, page=requested_page)
        if self.request_fingerprint != expected_request.fingerprint:
            raise ValueError("request_fingerprint does not match the request scope.")
        if blocked_total == 0 and self.first_blocked_skill_name is not None:
            raise ValueError("zero blocked_total cannot expose a review target.")
        if blocked_total > 0 and (
            not isinstance(self.first_blocked_skill_name, str)
            or not self.first_blocked_skill_name.strip()
            or self.first_blocked_skill_name != self.first_blocked_skill_name.strip()
        ):
            raise ValueError("blocked Skills require a stable review target.")
        if not isinstance(self.error, str):
            raise TypeError("error must be a string.")
        frozen_items = validate_skill_browse_items(self.items)
        object.__setattr__(self, "items", frozen_items)
        object.__setattr__(self, "total_items", total)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "request_token", token)
        object.__setattr__(self, "blocked_total", blocked_total)
        object.__setattr__(self, "requested_page", requested_page)
        object.__setattr__(self, "error", self.error.strip())

        if self.status in {"loading", "error"}:
            if frozen_items or total:
                raise ValueError(
                    "loading/error state cannot expose page rows or totals."
                )
            if page != self.scope.page or requested_page != self.scope.page:
                raise ValueError("loading/error state must retain the requested page.")
            if self.status == "error" and not self.error.strip():
                raise ValueError("error state requires error copy.")
            if self.status == "loading" and self.error:
                raise ValueError("loading state cannot expose error copy.")
            return
        if self.error:
            raise ValueError("settled state cannot expose error copy.")
        total_pages = max(1, (total + self.scope.page_size - 1) // self.scope.page_size)
        expected_page = min(requested_page, total_pages)
        if page != self.scope.page or page != expected_page:
            raise ValueError("settled page is outside exact result bounds.")
        expected_count = min(
            self.scope.page_size,
            max(0, total - (page - 1) * self.scope.page_size),
        )
        if len(frozen_items) != expected_count:
            raise ValueError("Skill browse result cardinality is not exact.")
        expected_status: SkillBrowseStatus = (
            "ready" if frozen_items else "no_matches" if self.scope.query else "empty"
        )
        if self.status != expected_status:
            raise ValueError("Skill browse status contradicts its rows and scope.")


def begin_skill_browse(
    scope: SkillBrowseScope, *, request_token: int = 1
) -> SkillBrowseResult:
    """Build loading state bound to one exact Skills request."""
    return SkillBrowseResult(
        scope=scope,
        items=(),
        total_items=0,
        page=scope.page,
        status="loading",
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
    )


def build_skill_browse_result(
    scope: SkillBrowseScope,
    record: Mapping[str, Any],
    *,
    request_token: int = 1,
) -> SkillBrowseResult:
    """Validate an exact `list_skills` response into immutable page state."""
    if not isinstance(record, Mapping):
        raise TypeError("Skill browse result must be a mapping.")
    raw_items = record.get("skills")
    if not isinstance(raw_items, list):
        raise TypeError("Skill browse skills must be a list.")
    items = tuple(raw_items)
    count = _skill_browse_integer(record.get("count"), field="count", minimum=0)
    total = _skill_browse_integer(record.get("total"), field="total", minimum=0)
    limit = _skill_browse_integer(record.get("limit"), field="limit", minimum=1)
    offset = _skill_browse_integer(record.get("offset"), field="offset", minimum=0)
    blocked_total = _skill_browse_integer(
        record.get("blocked_total"), field="blocked_total", minimum=0
    )
    if limit != scope.page_size:
        raise ValueError("limit must match the requested page size.")
    if offset % limit:
        raise ValueError("offset must identify a page boundary.")
    total_pages = max(1, (total + limit - 1) // limit)
    resolved_page = offset // limit + 1
    if resolved_page != min(scope.page, total_pages):
        raise ValueError("offset must match the requested or clamped page.")
    if count != len(items):
        raise ValueError("count must match the returned Skill rows.")
    expected_count = min(limit, max(0, total - offset))
    if len(items) != expected_count:
        raise ValueError("Skill page cardinality does not match its coordinates.")
    resolved_scope = replace(scope, page=resolved_page)
    status: SkillBrowseStatus = (
        "ready" if items else "no_matches" if resolved_scope.query else "empty"
    )
    return SkillBrowseResult(
        scope=resolved_scope,
        items=items,
        total_items=total,
        page=resolved_page,
        status=status,
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
        blocked_total=blocked_total,
        first_blocked_skill_name=record.get("first_blocked_skill_name"),
        requested_page=scope.page,
    )


def build_skill_browse_error(
    scope: SkillBrowseScope,
    *,
    request_token: int = 1,
    error: str = "Couldn't load Skills. Try again.",
) -> SkillBrowseResult:
    """Build a recoverable failure without forging an empty page."""
    return SkillBrowseResult(
        scope=scope,
        items=(),
        total_items=0,
        page=scope.page,
        status="error",
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
        error=error,
    )


def apply_skill_browse_result(
    state: SkillBrowseResult, result: SkillBrowseResult
) -> SkillBrowseResult:
    """Settle only the matching in-flight Skills scope and generation."""
    if (
        state.status != "loading"
        or result.status == "loading"
        or state.request_fingerprint != result.request_fingerprint
        or state.request_token != result.request_token
        or state.scope != replace(result.scope, page=result.requested_page)
    ):
        return state
    return result


def coerce_skill_reader_mode(value: Any) -> SkillReaderMode:
    """Return a supported Skills work-pane mode, defaulting to Overview.

    Args:
        value: Candidate reader-mode value.

    Returns:
        The supported mode, or ``"overview"`` for an unsupported value.
    """
    if value in {"edit", "trust", "files"}:
        return value
    return "overview"


def skill_review_identity_line(active_review: Mapping[str, Any] | None) -> str:
    """Identify the exact trust-service snapshot currently under review.

    Args:
        active_review: Trust-service review snapshot, when one is active.

    Returns:
        The review generation and digest label, or an empty string when the
        snapshot has no valid identity.
    """
    if not isinstance(active_review, Mapping):
        return ""
    generation = active_review.get("manifest_generation")
    digest = active_review.get("current_digest")
    if not isinstance(generation, int) or not isinstance(digest, str) or not digest:
        return ""
    return f"Reviewed files · trust generation {generation} · sha256:{digest}"


def coerce_skill_editor_mode(value: Any) -> SkillEditorMode:
    """Return a supported Skill editor mode, defaulting safely to Basic."""
    return "advanced" if value == "advanced" else "basic"


def skill_invocation_copy(user_invocable: bool, disable_model_invocation: bool) -> str:
    """Describe the independently configured user and agent invocation paths."""
    agent_invocable = not disable_model_invocation
    if user_invocable and agent_invocable:
        return "You and the agent can invoke this Skill."
    if user_invocable:
        return "Only you can invoke this Skill."
    if agent_invocable:
        return "Only the agent can invoke this Skill."
    return "Reference only — neither you nor the agent can invoke this Skill."


def skill_trust_requires_details(
    trust_status: str,
    trust_blocked: bool,
    changed_files: tuple[str, ...],
) -> bool:
    """Return whether safety details must remain expanded in either editor mode."""
    return trust_status != "trusted" or trust_blocked or bool(changed_files)


def skill_allowed_tools_sequence(value: str) -> tuple[str, ...]:
    """Parse the editor's captured tool list without sorting or deduplicating it."""
    return tuple(_split_csv(value))


def reconcile_skill_allowed_tools(
    captured: tuple[str, ...],
    *,
    selected: tuple[str, ...],
    catalog_order: tuple[str, ...],
    picker_changed: bool,
) -> tuple[str, ...]:
    """Apply an explicit picker edit without rewriting untouched Skill content."""
    if not picker_changed:
        return captured

    known = set(catalog_order)
    selected_known = set(selected) & known
    reconciled = [
        name for name in captured if name not in known or name in selected_known
    ]
    captured_names = set(captured)
    reconciled.extend(
        name
        for name in catalog_order
        if name in selected_known and name not in captured_names
    )
    return tuple(reconciled)


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
        selected: Whether this row owns the retained Work pane.
    """

    name: str
    secondary: str
    trust_glyph: str
    blocked: bool
    selected: bool = False


@dataclass(frozen=True)
class SkillsListState:
    """Display state for the Library Skills canvas's list view.

    Attributes:
        rows: The skills to render, already filtered/sorted.
        count: ``len(rows)``.
        sort: The sort mode used to build ``rows`` (``"name"`` or
            ``"status"``), echoed back for the caller's toggle label.
        pager: Source-owned exact paging display, when Skills browsing is
            backed by the bounded list service.
        blocked_total: Source-wide count of trust-blocked Skills. This is
            intentionally independent of the current page and filter.
        first_blocked_skill_name: Stable source-wide trust-review target.
        actions_disabled: Whether row actions must remain inert because the
            retained page is loading or stale.
        source_summary_fresh: Whether source-wide trust metadata is currently
            authoritative enough to display.
    """

    rows: tuple[SkillListRow, ...]
    count: int
    sort: str
    pager: LibraryPagerDisplay | None = None
    blocked_total: int = 0
    first_blocked_skill_name: str | None = None
    actions_disabled: bool = False
    source_summary_fresh: bool = True


@dataclass(frozen=True)
class SkillEditorSupportingFile:
    """One row in the skill editor's read-only supporting-files list.

    Attributes:
        name: The file's path, relative to the skill's directory. Nested
            paths (e.g. ``"references/api.md"``) render as-is.
        size: The file's size in bytes.
        is_text: Whether the file is text. ``False`` marks a binary file
            as view-only (it can't be edited as text). Defaults to
            ``True`` so the ``supporting_files``-only fallback path
            (which only ever carries decoded text) stays correct.
    """

    name: str
    size: int
    is_text: bool = True


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
            ``SkillEditorSupportingFile`` rows (built from ``bundle_files``
            when present -- covering nested paths and binaries -- falling
            back to ``supporting_files`` (text only) otherwise).
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
    supporting_files: tuple[SkillEditorSupportingFile, ...]
    version: int | None
    trust_status: str
    trust_blocked: bool
    trust_changed_files: tuple[str, ...]
    # task-419: the record's description was derived from the first body
    # line (no frontmatter description on disk) -- shown as a hint, never
    # echoed into the Description field.
    description_derived: bool = False


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
        A spelled-out invocability summary (task-418: the bare
        ``"user · agent"`` tokens had no legend anywhere in the UI):
        ``"invocable: user & agent"`` when both can invoke it,
        ``"invocable: user only"``/``"invocable: agent only"`` when one
        can, or ``"not invocable"`` when neither can.
    """
    agent_invocable = not disable_model_invocation
    if user_invocable and agent_invocable:
        return "invocable: user & agent"
    if user_invocable:
        return "invocable: user only"
    if agent_invocable:
        return "invocable: agent only"
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


def skill_trust_header_line(posture: str, blocked_count: int) -> tuple[str, str] | None:
    """Return (copy, action_id) for the Skills-list trust header, or None to hide.

    Args:
        posture: The trust service's ``trust_posture()`` value -- one of
            ``"needs_setup"``, ``"needs_resetup"``, ``"unavailable"``,
            ``"locked"``, ``"error"``, ``"ready"`` (Task 3). Any other
            unrecognized/empty value hides the header (``None``) -- the
            list canvas already degrades gracefully with no header, and
            surfacing raw trust-service errors here would be noise the user
            can't act on from this screen. ``"error"`` (a corrupt/tampered
            manifest) still gets a header, though: without one, blocked
            skills would render with no list-level recovery action at all,
            forcing the user to open a skill just to find the reset path.
        blocked_count: Number of rows in the current list state that are
            currently trust-blocked (``row.blocked``).

    Returns:
        ``(copy, action_id)`` where ``action_id`` is one of ``"setup"``,
        ``"resetup"``, ``"retry"``, ``"unlock"``, ``"review"``, or ``""``
        (posture is ``"ready"`` with nothing blocked -- shown but with no
        action), or ``None`` to hide the header entirely.
    """
    if posture == "needs_setup":
        return (
            "Skill trust isn't set up — set it up to review and use skills.",
            "setup",
        )
    if posture == "needs_resetup":
        return ("Skill trust needs to be set up again after an update.", "resetup")
    if posture == "unavailable":
        return ("Skill trust is temporarily unavailable — try again.", "retry")
    if posture == "locked":
        return ("Skill trust is locked for this session.", "unlock")
    if posture == "error":
        # Reuses the "resetup" action_id -- it already routes to
        # reset-then-bootstrap, the only recovery that makes sense when the
        # trust manifest itself can't be read/verified.
        return ("Skill trust can't be verified — set it up again.", "resetup")
    if posture == "ready":
        if blocked_count > 0:
            noun = "skill needs" if blocked_count == 1 else "skills need"
            return (f"{blocked_count} {noun} review before use.", "review")
        return ("Skill trust: ready.", "")
    return None


def _matches_query(record: Mapping[str, Any], query_lower: str) -> bool:
    if not query_lower:
        return True
    if query_lower in _text(record.get("name")).lower():
        return True
    return query_lower in _text(record.get("description")).lower()


def _row(
    record: Mapping[str, Any], *, default_blocked: bool, selected_name: str
) -> SkillListRow | None:
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
    return SkillListRow(
        name=name,
        secondary=secondary,
        trust_glyph=trust_glyph,
        blocked=blocked,
        selected=name == selected_name,
    )


def build_skills_list_state(
    context_payload: Mapping[str, Any] | None,
    *,
    query: str,
    sort: str,
    selected_name: str = "",
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
        selected_name: Skill currently projected in the retained Work pane.

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
            row = _row(
                record,
                default_blocked=False,
                selected_name=selected_name,
            )
            if row is not None:
                rows.append(row)
    for record in blocked:
        if isinstance(record, Mapping) and _matches_query(record, query_lower):
            row = _row(
                record,
                default_blocked=True,
                selected_name=selected_name,
            )
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
        sorted ``SkillEditorSupportingFile`` rows -- built from
        ``bundle_files`` (nested paths, binaries included) when present,
        falling back to ``supporting_files`` (text only, e.g. a
        remote/server skill that doesn't populate ``bundle_files``)
        otherwise.
    """
    if not isinstance(detail, Mapping):
        detail = {}
    content = _raw_text(detail.get("content"))
    front_matter, body = LocalSkillsService._parse_front_matter(content)

    bundle_source = detail.get("bundle_files")
    if bundle_source:
        supporting_files = tuple(
            sorted(
                (
                    SkillEditorSupportingFile(
                        name=_text(entry.get("path")),
                        size=_to_int(entry.get("size")) or 0,
                        is_text=bool(entry.get("is_text", True)),
                    )
                    for entry in bundle_source
                    if isinstance(entry, Mapping)
                ),
                key=lambda row: row.name,
            )
        )
    else:
        supporting_source = detail.get("supporting_files") or {}
        supporting_files = tuple(
            sorted(
                (
                    SkillEditorSupportingFile(
                        name=name, size=len(str(text).encode("utf-8"))
                    )
                    for name, text in supporting_source.items()
                ),
                key=lambda row: row.name,
            )
        )

    changed_files = detail.get("trust_changed_files") or ()

    # task-419: the service backfills a missing frontmatter description
    # from the first body line for LIST display. Echoing that into the
    # editor's Description field would misrepresent the file -- and a
    # later save would ratchet the derived text into the frontmatter.
    # Exact discriminator: derived iff the record carries a description
    # but the parsed frontmatter does not.
    record_description = _text(detail.get("description"))
    front_matter_description = (
        _text(front_matter.get("description"))
        if isinstance(front_matter, Mapping)
        else ""
    )
    description_derived = bool(record_description) and not front_matter_description
    return SkillEditorState(
        name=_text(detail.get("name")),
        description="" if description_derived else record_description,
        description_derived=description_derived,
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
