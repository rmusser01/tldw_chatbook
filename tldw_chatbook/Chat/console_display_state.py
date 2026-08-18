"""Pure display-state contracts for the native Console workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import PurePath
from typing import Any, Mapping, Optional, Sequence

from rich.cells import cell_len

from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.rag_scope import EffectiveScope, RagScope
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label
from tldw_chatbook.Workspaces.change_tracking import ChangedFile

CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID = "console-inspector-review-approval"
CONSOLE_INSPECTOR_REVIEW_APPROVAL_LABEL = "Review approval"
#: TASK-1972: the inspector's route to the Change Review screen -- the
#: honest replacement for the dead "Review tool call" action TASK-1843
#: removed. Enabled whenever change tracking is ON (git present, tracker
#: built): the SCREEN owns the empty state ("No file changes recorded"),
#: so enablement never needs a per-tick DB query.
CONSOLE_INSPECTOR_REVIEW_CHANGES_ID = "console-inspector-review-changes"
CONSOLE_INSPECTOR_REVIEW_CHANGES_LABEL = "Review changes"
CONSOLE_INSPECTOR_NO_CHANGE_TRACKING_REASON = (
    "Change tracking is off (git unavailable)."
)
CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID = "console-inspector-review-tool-call"
CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_LABEL = "Review tool call"
CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID = "console-inspector-save-chatbook"
CONSOLE_INSPECTOR_SAVE_CHATBOOK_LABEL = "Save as Chatbook"
CONSOLE_INSPECTOR_NO_APPROVAL_REASON = "No approval is pending."
CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON = "No tool calls are ready for review."
CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON = "No Chatbook artifact is available."

#: How many staged references the composer-level evidence strip lists before
#: it collapses the rest into a "+N more" line. The strip sits directly above
#: the composer (the status chips are below it now), so it must stay short
#: enough never to push the composer off a small terminal.
CONSOLE_STAGED_EVIDENCE_STRIP_MAX_ROWS = 3
CONSOLE_STAGED_EVIDENCE_UNSTAGE_ID = "console-unstage-evidence"
CONSOLE_STAGED_EVIDENCE_UNSTAGE_LABEL = "Un-stage"

_SOURCE_STATUS_CLASS_MAP = {
    "ready": {"ready", "available", "attached", "staged"},
    "running": {"retrieving", "running", "searching", "stale"},
    "blocked": {"blocked", "missing", "unavailable"},
}


def normalize_console_source_status(status: Any) -> str:
    """Map a raw source/launch status onto one of the UI status classes.

    Args:
        status: Raw status value from a display row, evidence reference, or
            live-work launch.

    Returns:
        One of ``ready``, ``running``, ``blocked``, or ``muted``.
    """
    normalized = str(status or "").strip().lower()
    for class_name, synonyms in _SOURCE_STATUS_CLASS_MAP.items():
        if normalized in synonyms:
            return class_name
    return "muted"


def _clean(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _safe_display_text(value: Any, fallback: str = "") -> str:
    """Normalize user/source text before exposing it in Console display rows."""
    return html_escape(_clean(value, fallback), quote=False)


def resolve_assistant_identity_label(
    *,
    character: Any = None,
    assistant_kind: Any = None,
    assistant_name: Any = None,
    assistant_id: Any = None,
) -> str:
    """Resolve the shared assistant identity label for Console and Chat.

    Args:
        character: Existing character display value, which always takes
            precedence when nonblank.
        assistant_kind: Optional presentation-only assistant kind.
        assistant_name: Optional presentation-only assistant display name.
        assistant_id: Optional presentation-only assistant identifier.

    Returns:
        ``Character: <value>``, ``Persona: <value>``, or the generic
        ``Assistant: General`` fallback.
    """
    character_text = sanitize_character_display_label(
        character,
        max_characters=180,
    )
    if character_text:
        return f"Character: {character_text}"

    assistant_kind_text = _clean(assistant_kind, "").lower()
    persona_text = sanitize_character_display_label(
        assistant_name,
        max_characters=180,
    ) or sanitize_character_display_label(
        assistant_id,
        max_characters=180,
    )
    if assistant_kind_text == "persona" and persona_text:
        return f"Persona: {persona_text}"
    return "Assistant: General"


def coerce_non_negative_int(value: Any) -> int:
    """Coerce a loose seam value into a non-negative integer.

    Args:
        value: Value from an app seam, test fixture, or serialized state.

    Returns:
        A non-negative integer, or 0 when the value is missing or invalid.
    """
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _is_blocked_rag_status(value: Any) -> bool:
    text = _clean(value, "").lower()
    return text.startswith("missing") or text in {"blocked", "unavailable"}


def _tools_ready_text(effective_tool_count: int) -> str:
    """Return the shared Tools copy for both the chip and the Inspector row.

    TASK-1843: these two surfaces sit in the same panel and previously derived
    their number independently, so one said "0 ready" while the other said
    "12 ready". One function, one wording, both callers -- fixing it per-site
    is exactly how it recurred after the first fix.

    Args:
        effective_tool_count: Built-in count plus MCP catalog size.

    Returns:
        "—" at zero -- a neutral placeholder, since this app never
        distinguishes "definitely zero" from "not counted yet" (and "0
        ready" reads as "no tools available" when built-ins are always
        registered). TASK-2154.12 (TX-04): "not loaded" exposed a
        lazy-loading implementation detail, so the chip now HIDES at zero
        and this row-level dash is all the Inspector shows. Otherwise
        "<n> ready".
    """
    return "—" if effective_tool_count == 0 else f"{effective_tool_count} ready"


def _mcp_inspector_row(
    tool_count: Any, not_connected_count: Any
) -> "ConsoleDisplayRow | None":
    """Build the "MCP" inspector row, or ``None`` when there is nothing to show.

    ``tool_count`` is ``None`` (the P5-T6 default) whenever the caller has
    no MCP composition to report at all -- no ``unified_mcp_service`` on
    the app, or the kill switch is on -- and the row is omitted entirely.
    Otherwise: a non-zero not-connected-server count ALWAYS wins with the
    blocked warning ("M servers enabled, not connected"), regardless of
    ``tool_count`` -- a stale (disconnected-with-snapshot) server still
    contributes its own tools to the catalog (see
    ``MCPToolProvider.compose_catalog``'s eligibility filter), so
    ``tool_count`` is essentially never 0 in the real mixed case; checking
    it first would make this affordance unreachable (Finding I2). Only
    when every enabled server is connected does a non-zero tool count
    render the ready row ("N tools ready"). Both zero omits the row just
    like the ``None`` case -- nothing worth telling the user about.
    """
    if tool_count is None:
        return None
    normalized_tool_count = coerce_non_negative_int(tool_count)
    normalized_not_connected_count = coerce_non_negative_int(not_connected_count)
    if normalized_not_connected_count > 0:
        server_word = "server" if normalized_not_connected_count == 1 else "servers"
        return ConsoleDisplayRow(
            "MCP",
            f"{normalized_not_connected_count} {server_word} enabled, not connected",
            status="blocked",
        )
    if normalized_tool_count > 0:
        tool_word = "tool" if normalized_tool_count == 1 else "tools"
        return ConsoleDisplayRow("MCP", f"{normalized_tool_count} {tool_word} ready")
    return None


def build_console_disabled_reason(
    *,
    action_id: str,
    has_draft: bool,
    send_blocked: bool,
    setup_blocked_reason: str = "",
    wake_turn_active: bool = False,
) -> str:
    """Return concise disabled copy for Console action controls.

    Args:
        action_id: Canonical action id, such as ``send``.
        has_draft: Whether the composer currently has message text.
        send_blocked: Whether sending is blocked by setup or run state.
        setup_blocked_reason: Provider/setup blocker copy, when present.
        wake_turn_active: Whether the active session is busy with a
            machine-injected auto-wake turn (task-15862 AC#3). Checked
            FIRST: during a wake the queue presentation's "wait to be
            accepted" tooltip rides the ``setup_blocked_reason`` slot (a
            chainless wake is never queue-accepted), and the setup
            fallback below would blame provider setup for it -- the
            observed live lie.

    Returns:
        A user-facing disabled reason, or an empty string when no conservative
        reason should be shown.
    """
    if action_id != "send":
        return ""

    if send_blocked and wake_turn_active:
        return "Send blocked — delivering a sub-agent result"

    setup_reason = _clean(setup_blocked_reason, "")
    setup_reason_lower = setup_reason.lower()
    if send_blocked and setup_reason:
        # TX-07 follow-up (TASK-2154.12): "api key"/"endpoint" must be
        # checked BEFORE "model" -- the real blocker copy points at the
        # "Settings > Providers & Models" screen (e.g. "Add API key in
        # Settings > Providers & Models before sending."), so a naive
        # "model" substring match swallowed the missing-key state and
        # named the wrong blocker.
        if "api key" in setup_reason_lower:
            return "Send blocked — add an API key to continue"
        if "endpoint" in setup_reason_lower:
            return "Send blocked — configure the endpoint to continue"
        if "model" in setup_reason_lower:
            return "Send blocked — choose a model to continue"
        if (
            "choose a provider" in setup_reason_lower
            or "missing provider" in setup_reason_lower
        ):
            return "Send blocked — choose a provider to continue"
        return "Send blocked — finish provider setup to continue"
    if send_blocked:
        # No setup copy means an active run is the blocker (the setup gate
        # always supplies its own reason).
        return "Send blocked — wait for the active run to finish"
    if not has_draft:
        return "Send disabled: type a message"
    return ""


@dataclass(frozen=True)
class ConsoleDisplayRow:
    """One user-visible Console display row."""

    label: str
    value: Any
    status: str = "ready"
    recovery: str = ""

    @property
    def text(self) -> str:
        suffix = f" - {self.recovery}" if self.recovery else ""
        return f"{self.label}: {self.value}{suffix}"


@dataclass(frozen=True)
class ConsoleEvidenceDisplayState:
    """Readable Console summary for one staged evidence bundle."""

    summary: str
    authority: str
    status: str
    recovery: str
    available_count: int
    total_count: int
    reference_rows: tuple[ConsoleDisplayRow, ...] = ()


def evidence_bundle_from_launch(
    launch: ConsoleLiveWorkLaunch | None,
) -> EvidenceBundle | None:
    """Parse a staged live-work evidence bundle without exposing raw payload text."""
    if launch is None:
        return None
    evidence_payload = launch.payload.get("evidence_bundle")
    if isinstance(evidence_payload, EvidenceBundle):
        return evidence_payload
    if not isinstance(evidence_payload, Mapping):
        return None
    try:
        return EvidenceBundle.from_payload(evidence_payload)
    except (TypeError, ValueError):
        return None


def build_console_evidence_display_state(
    launch: ConsoleLiveWorkLaunch | None,
) -> ConsoleEvidenceDisplayState | None:
    """Build the user-visible evidence summary for Console staged state."""
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return None

    available_count = 0
    blocked_count = 0
    stale_count = 0
    missing_count = 0
    reference_rows = []
    authority_values: list[str] = []
    seen_authorities: set[str] = set()
    for reference in bundle.references:
        if reference.status == "available":
            available_count += 1
        elif reference.status == "blocked":
            blocked_count += 1
        elif reference.status == "stale":
            stale_count += 1
        elif reference.status == "missing":
            missing_count += 1

        safe_authority = _safe_display_text(reference.authority_label)
        if safe_authority and safe_authority not in seen_authorities:
            authority_values.append(safe_authority)
            seen_authorities.add(safe_authority)

        reference_status = "blocked" if reference.status != "available" else "ready"
        reference_rows.extend(
            (
                ConsoleDisplayRow(
                    "Evidence source",
                    (
                        f"[{_safe_display_text(reference.evidence_id, 'unknown')}] "
                        f"{_safe_display_text(reference.title, 'Untitled source')}"
                    ),
                    status=reference_status,
                ),
                ConsoleDisplayRow(
                    "Evidence authority",
                    safe_authority or "unknown",
                    status=reference_status,
                ),
                ConsoleDisplayRow(
                    "Evidence status",
                    _safe_display_text(reference.status, "unknown"),
                    status=reference_status,
                ),
            )
        )
        if reference.snippet:
            reference_rows.append(
                ConsoleDisplayRow(
                    "Snippet",
                    _safe_display_text(reference.snippet),
                    status=reference_status,
                )
            )

    total_count = len(bundle.references)
    authority = ", ".join(authority_values) or "unknown"
    summary = f"{available_count}/{total_count} available ({bundle.status})"
    recovery = ""
    if total_count == 0:
        recovery = "No evidence references are attached."
    elif available_count == 0:
        recovery = "No available evidence. Review source authority before sending."
    elif blocked_count or stale_count or missing_count:
        warning_parts = []
        if blocked_count:
            warning_parts.append(f"{blocked_count} blocked")
        if stale_count:
            warning_parts.append(f"{stale_count} stale")
        if missing_count:
            warning_parts.append(f"{missing_count} missing")
        recovery = f"Some evidence needs review: {', '.join(warning_parts)}."

    row_status = "blocked" if available_count == 0 else "ready"
    return ConsoleEvidenceDisplayState(
        summary=summary,
        authority=authority,
        status=row_status,
        recovery=recovery,
        available_count=available_count,
        total_count=total_count,
        reference_rows=tuple(reference_rows),
    )


@dataclass(frozen=True)
class ConsoleInspectorAction:
    """One action exposed by the Console run inspector."""

    widget_id: str
    label: str
    enabled: bool
    disabled_reason: str = ""
    classes: str = "destination-action-button console-inspector-action"

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


# TASK-2154.5 (TX-06): name:value grammar like every sibling chip -- the old
# bare "System Prompt" read as a label with no state next to ": set".
CONSOLE_SYSTEM_PROMPT_LABEL_UNSET = "System Prompt: off"
CONSOLE_SYSTEM_PROMPT_LABEL_SET = "System Prompt: set"


@dataclass(frozen=True)
class ConsoleControlState:
    """Header/control labels for the Console-native workbench chrome."""

    provider_label: str
    model_label: str
    assistant_label: str
    rag_label: str
    sources_label: str
    tools_label: str
    approvals_label: str
    system_prompt_label: str = CONSOLE_SYSTEM_PROMPT_LABEL_UNSET
    sources_active: bool = False
    tools_active: bool = False
    approvals_active: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        provider: Any = None,
        model: Any = None,
        character: Any = None,
        assistant_kind: Any = None,
        assistant_name: Any = None,
        assistant_id: Any = None,
        rag_enabled: bool = False,
        staged_source_count: int = 0,
        tool_count: int = 0,
        mcp_tool_count: int | None = None,
        approval_count: int = 0,
        system_prompt_set: bool = False,
    ) -> "ConsoleControlState":
        """Build the Console control-bar chip state from raw run values.

        Args:
            provider: Active provider name, or falsy for "not selected".
            model: Active model name, or falsy for "not selected".
            character: Existing character presentation value; when present,
                renders as ``Character: <name>``.
            assistant_kind: Optional presentation-only assistant kind.
            assistant_name: Optional presentation-only assistant display name.
            assistant_id: Optional presentation-only assistant identifier.
            rag_enabled: Whether RAG is on for this send.
            staged_source_count: Number of staged context sources.
            tool_count: Built-in tools that can run.
            mcp_tool_count: MCP catalog size that can run, or ``None`` when no MCP
                seam is wired (chip then reflects built-in tools only).
            approval_count: Pending MCP approvals.
            system_prompt_set: Whether the active session has a system prompt;
                the chip then reads ``System Prompt: set``.

        Returns:
            A ``ConsoleControlState`` whose ``tools_label`` counts the tools that
            can actually run (built-in + MCP) -- or reads "Tools: —"
            at a zero count (task-1234/F7: a neutral placeholder, since this
            app never distinguishes "definitely zero" from "not counted yet";
            TASK-2154.12/TX-04 hides the chip at zero so even the dash stays
            off the strip) -- and whose ``*_active`` flags drive chip emphasis.
        """
        assistant_label = resolve_assistant_identity_label(
            character=character,
            assistant_kind=assistant_kind,
            assistant_name=assistant_name,
            assistant_id=assistant_id,
        )
        # TASK-350: the chip must reflect the tools that can ACTUALLY run — built-in
        # AND MCP. Counting only built-in read "Tools: 0 ready" while the inspector
        # showed "MCP: 10 tools ready". `mcp_tool_count is None` means no MCP seam
        # wired, so the chip falls back to built-in only.
        effective_tool_count = tool_count + (mcp_tool_count or 0)
        # Fleet-UX expert review F7 (task-1234): `tool_count` is sourced from
        # a getattr hook (`ChatScreen._console_tool_count`) that production
        # code never actually populates -- so a fresh app reads "Tools: 0
        # ready" forever, not just before the catalog lazily builds, and
        # that copy reads as "no tools available" even though built-ins
        # like calculator/get_current_datetime are always registered.
        # Eagerly counting the real enabled-builtin total was rejected: it
        # would also feed `ConsoleInspectorState`'s "Review tool call" gate
        # (a DIFFERENT concept -- "were any tool calls actually made this
        # run", not "how many tools are configured") and falsely mark it
        # actionable before any call ever happened. Scoped fix: a neutral,
        # honest placeholder for this chip alone at the zero count --
        # `tools_active` (dim/emphasis) is UNCHANGED, still `effective_tool_
        # count > 0`. TASK-2154.12 (TX-04): the placeholder is now an inert
        # dash and the chip HIDES at zero, so the lazy-loading detail
        # ("not loaded") no longer renders at all.
        tools_label = f"Tools: {_tools_ready_text(effective_tool_count)}"
        return cls(
            provider_label=f"Provider: {_clean(provider, 'not selected')}",
            model_label=f"Model: {_clean(model, 'not selected')}",
            assistant_label=assistant_label,
            rag_label=f"Library search: {'on' if rag_enabled else 'off'}",
            sources_label=f"Sources: {staged_source_count}",
            tools_label=tools_label,
            approvals_label=f"Approvals: {approval_count} pending",
            system_prompt_label=(
                CONSOLE_SYSTEM_PROMPT_LABEL_SET
                if system_prompt_set
                else CONSOLE_SYSTEM_PROMPT_LABEL_UNSET
            ),
            sources_active=staged_source_count > 0,
            tools_active=effective_tool_count > 0,
            approvals_active=approval_count > 0,
        )


@dataclass(frozen=True)
class ConsoleStagedContextState:
    """Display state for the Console staged-context tray."""

    heading: str
    summary: str
    rows: tuple[ConsoleDisplayRow, ...] = ()
    recovery: str = ""
    is_empty: bool = False
    #: True staged-source count for the tray's "Sources N" heading (D1a).
    #: NOT ``len(rows)`` -- one staged reference explodes into 3-4
    #: provenance rows (Evidence source/authority/status, optional
    #: Snippet), so a 5-reference bundle produced 17-22 rows and the tray
    #: rendered "Sources 18". ``None`` (matching this file's existing
    #: ``mcp_tool_count: int | None = None`` idiom) means "the caller did
    #: not supply one" -- every real production constructor (both
    #: ``from_live_work`` and ``empty()`` below) passes it explicitly, so
    #: this only fires for a state built directly without it (pre-fix
    #: tests, and any other hand-built state), where ``__post_init__``
    #: falls back to ``len(rows)`` so a single hand-built row still reads
    #: "1" -- weak but not wrong for that one-row shape. Deliberately NOT
    #: a negative-int sentinel: that would silently swallow a future
    #: production caller that forgets to pass ``source_count=``, regressing
    #: straight back to the row-count lie this task fixed with no signal.
    source_count: int | None = None

    def __post_init__(self) -> None:
        if self.source_count is None:
            object.__setattr__(self, "source_count", len(self.rows))

    @classmethod
    def from_live_work(
        cls,
        launch: ConsoleLiveWorkLaunch,
    ) -> "ConsoleStagedContextState":
        rows = []
        evidence_state = build_console_evidence_display_state(launch)
        if evidence_state is not None:
            rows.append(
                ConsoleDisplayRow(
                    "Evidence",
                    evidence_state.summary,
                    status=evidence_state.status,
                    recovery=evidence_state.recovery,
                )
            )
            rows.append(
                ConsoleDisplayRow(
                    "Authority",
                    evidence_state.authority,
                    status=evidence_state.status,
                )
            )
            rows.extend(evidence_state.reference_rows)
        rows.extend(
            ConsoleDisplayRow(label=key, value=value)
            for key, value in launch.payload_display_items()
        )
        # PR-T1 final review (I1): the summary interpolates raw launch
        # fields that originate in user data (a note title, a media
        # filename), so it goes through the same normalizer every other
        # display value in this module uses instead of being the one raw
        # f-string. This is defence in depth, not the crash fix -- HTML
        # escaping leaves Rich markup like `[/]` untouched, so the
        # MarkupError is fixed at the sink (`markup=False` on the staged-
        # context summary Static); this just stops the summary being the
        # module's lone unescaped exit.
        summary_title = _safe_display_text(launch.title, "Untitled source")
        summary_source = _safe_display_text(launch.source, "unknown")
        summary_status = _safe_display_text(launch.status, "unknown")
        return cls(
            heading="Staged Context",
            summary=f"{summary_title} ({summary_source}, {summary_status})",
            rows=tuple(rows),
            recovery=launch.recovery,
            source_count=console_staged_source_count(launch),
        )

    @classmethod
    def empty(cls) -> "ConsoleStagedContextState":
        """Return the no-sources-staged display state.

        Task-400: the empty state carries no summary line. The staged-context
        tray renders its own "No sources attached. Stage sources from
        Library." guidance Static when there are no rows, so a summary of
        "No sources attached." here rendered the same copy twice.

        Returns:
            Empty staged-context state with the semantic ``is_empty`` flag
            set and a blank summary.
        """
        return cls(
            heading="Staged Context",
            summary="",
            is_empty=True,
            source_count=0,
        )


def console_staged_source_count(launch: ConsoleLiveWorkLaunch | None) -> int:
    """Return how many sources a staged live-work launch actually carries.

    The Console "Sources: N staged" chip used to hardcode ``1`` for any
    staged launch while the staged bundle routinely carried several
    references, so a four-result Library RAG run advertised one source.

    Args:
        launch: Currently staged live-work launch, if any.

    Returns:
        The staged bundle's reference count; ``1`` for a launch with no (or
        an empty) evidence bundle, since the launch itself is one staged
        item; ``0`` when nothing is staged.
    """
    if launch is None:
        return 0
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return 1
    return len(bundle.references) or 1


def console_prompted_source_count(launch: ConsoleLiveWorkLaunch | None) -> int:
    """Return how many staged references a Console send will actually prompt.

    Distinct from :func:`console_staged_source_count`, which answers "how
    much is staged". This answers "how much reaches the model", and it
    applies exactly the filter
    ``capture_console_staged_evidence_for_chat`` applies before formatting
    the prompt blocks: available status (``EvidenceBundle.
    available_references``) AND ``source_owner == "local"``. A four-result
    bundle carrying two blocked references stages four and sends two.

    Args:
        launch: Currently staged live-work launch, if any.

    Returns:
        Count of references eligible to enter the prompt; ``0`` when nothing
        is staged or the launch carries no evidence bundle (a bundleless
        launch yields no prompt context at all).
    """
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return 0
    return sum(
        1
        for reference in bundle.available_references()
        if reference.source_owner.strip().lower() == "local"
    )


def console_prompted_evidence_text(launch: ConsoleLiveWorkLaunch | None) -> str:
    """Return the staged evidence text a Console send will actually prompt.

    task-6: the Console context/cost estimates used to report zero for
    staged evidence because nothing carried its TEXT that far --
    ``ConsoleStagedSource`` is label-only. This is the pure, zero-I/O
    source of truth for that text, read at estimate time (settings
    summary, cost chip) before any send happens.

    Applies exactly the same filter as :func:`console_prompted_source_count`
    (``EvidenceBundle.available_references`` AND ``source_owner == "local"``)
    because it answers the same question that function counts: "how much
    reaches the model". ``reference.snippet`` is the right field to read,
    not a full re-fetch, because the actual send path
    (``capture_console_staged_evidence_for_chat``) re-validates identity and
    authority but never re-fetches content -- it hands the provider exactly
    this (already length-limited, see ``EVIDENCE_SNIPPET_CHAR_LIMIT``)
    snippet verbatim. That length limit is also why an oversized source
    (e.g. a 942 KB document) still yields a bounded, non-zero estimate here:
    the snippet was already capped when the reference was staged.

    Args:
        launch: Currently staged live-work launch, if any.

    Returns:
        Prompt-eligible reference snippets joined with blank lines, in
        bundle order; ``""`` when nothing is staged or nothing is
        prompt-eligible.
    """
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return ""
    return "\n\n".join(
        reference.snippet
        for reference in bundle.available_references()
        if reference.source_owner.strip().lower() == "local" and reference.snippet
    )


@dataclass(frozen=True)
class ConsoleStagedEvidenceRow:
    """One compact staged-evidence line for the composer-level strip.

    ``title`` and ``source`` are already display-escaped; renderers must
    keep console markup parsing OFF so escaping stays the terminal step.
    """

    title: str
    source: str
    status: str = "ready"


@dataclass(frozen=True)
class ConsoleStagedEvidenceStripState:
    """Display state for the staged-evidence strip above the composer.

    Three mutually exclusive shapes, all built by
    :func:`build_console_staged_evidence_strip_state`:

    * hidden -- nothing staged and nothing just sent;
    * staged -- a heading, up to
      :data:`CONSOLE_STAGED_EVIDENCE_STRIP_MAX_ROWS` rows, an optional
      "+N more" overflow line, and the single un-stage action;
    * sent -- the one-send ``notice`` line, which is the only confirmation
      an unpersisted session ever gets that evidence went out with the
      message (the transcript's ``Sources (N)`` row needs persistence).
    """

    visible: bool = False
    heading: str = ""
    rows: tuple[ConsoleStagedEvidenceRow, ...] = ()
    overflow: str = ""
    notice: str = ""
    unstage_label: str = CONSOLE_STAGED_EVIDENCE_UNSTAGE_LABEL


def _source_noun(count: int) -> str:
    return "source" if count == 1 else "sources"


def build_console_staged_evidence_strip_state(
    launch: ConsoleLiveWorkLaunch | None,
    *,
    sent_source_count: int | None = None,
) -> ConsoleStagedEvidenceStripState:
    """Build the staged-evidence strip state for the current Console turn.

    Args:
        launch: Currently staged live-work launch, if any.
        sent_source_count: Source count of the evidence consumed by the most
            recent send, when that send has not yet been superseded.

    Returns:
        The hidden, staged, or just-sent strip state. Live staging always
        wins over a stale "sent" notice -- a strip that showed both would
        claim evidence was consumed while new evidence sits waiting.
    """
    if launch is not None:
        bundle = evidence_bundle_from_launch(launch)
        if bundle is not None and bundle.references:
            rows = tuple(
                ConsoleStagedEvidenceRow(
                    title=_safe_display_text(reference.title, "Untitled source"),
                    source=_safe_display_text(reference.source_type, "source"),
                    status=("ready" if reference.status == "available" else "blocked"),
                )
                for reference in bundle.references
            )
        else:
            # A launch with no bundle (a generic handoff, or a retrieval
            # still running / blocked) is still one staged item; show it
            # rather than rendering an empty strip that contradicts the chip.
            rows = (
                ConsoleStagedEvidenceRow(
                    title=_safe_display_text(launch.title, "Untitled source"),
                    source=_safe_display_text(launch.source, "unknown"),
                    status=normalize_console_source_status(launch.status),
                ),
            )
        total = len(rows)
        shown = rows[:CONSOLE_STAGED_EVIDENCE_STRIP_MAX_ROWS]
        hidden_count = total - len(shown)
        return ConsoleStagedEvidenceStripState(
            visible=True,
            heading=f"Staged for next send · {total} {_source_noun(total)}",
            rows=shown,
            overflow=f"+{hidden_count} more" if hidden_count > 0 else "",
        )

    if sent_source_count:
        count = int(sent_source_count)
        return ConsoleStagedEvidenceStripState(
            visible=True,
            notice=f"Evidence sent with this message · {count} {_source_noun(count)}",
        )

    return ConsoleStagedEvidenceStripState()


@dataclass(frozen=True)
class ConsoleRetrievalScopeState:
    """Display state for the Inspector's "Retrieval scope" row (task-9) and
    the header's "Scope" chip (task-10) -- both render from this SAME
    snapshot, never a second state source.

    Pure snapshot -- built from session-held state only (a persisted
    conversation's cached last-read scope, or an unpersisted session's
    ``SessionScopeHolder``), never a DB read at render/recompose time. A
    scope with zero items is never represented here as "scoped": both
    storage-layer entry points (``read_conversation_scope``,
    ``SessionScopeHolder.set``) already normalize a zero-item scope to
    ``None`` (unscoped) before this state is ever built from it.

    ``is_empty``/``cause`` mirror ``EffectiveScope``'s ``"empty"`` state
    (rag_scope.py) -- the configured scope(s) leave nothing to retrieve
    from (either every item in an active scope has since been deleted, or
    a conversation/workspace intersection with no overlap).

    ``conv_item_count``/``ws_item_count`` (task-13, Phase 3) carry the
    individual conversation-level and workspace-level scope counts
    alongside ``item_count`` (which is always the EFFECTIVE,
    post-intersection count) -- used only for the header chip's
    two-level breakdown tooltip ("Only searching: conversation scope (A
    items) and workspace scope (B items) — N in both.").
    ``None`` means that level has no active scope. Built via
    ``from_effective`` whenever a workspace scope might be in play;
    ``from_scope`` (the conversation-only shortcut still used before the
    off-loop effective resolution lands in the cache) sets
    ``conv_item_count`` to the same value as ``item_count`` and leaves
    ``ws_item_count`` unset.
    """

    is_scoped: bool
    item_count: int = 0
    is_empty: bool = False
    cause: Optional[str] = None
    conv_item_count: Optional[int] = None
    ws_item_count: Optional[int] = None

    @classmethod
    def unscoped(cls) -> "ConsoleRetrievalScopeState":
        """Return the "everything" (no active scope) display state."""
        return cls(is_scoped=False, item_count=0)

    @classmethod
    def from_scope(cls, scope: "RagScope | None") -> "ConsoleRetrievalScopeState":
        """Build the row's display state from a resolved scope, or ``None``."""
        if scope is None or not scope.items:
            return cls.unscoped()
        return cls(
            is_scoped=True,
            item_count=len(scope.items),
            conv_item_count=len(scope.items),
        )

    @classmethod
    def empty(cls, cause: Optional[str] = None) -> "ConsoleRetrievalScopeState":
        """Return the EMPTY (action-required) display state.

        Args:
            cause: Short machine-readable reason (e.g. ``"deleted-items"``
                or ``"no-workspace-overlap"``, mirroring
                ``EffectiveScope.cause``); surfaced in the chip's tooltip.
        """
        return cls(is_scoped=False, item_count=0, is_empty=True, cause=cause)

    @classmethod
    def from_effective(
        cls,
        effective: "EffectiveScope",
        *,
        conv_item_count: Optional[int] = None,
        ws_item_count: Optional[int] = None,
    ) -> "ConsoleRetrievalScopeState":
        """Build the row/chip display state from a resolved ``EffectiveScope``.

        Task-13: the Inspector row and header chip render the EFFECTIVE
        (post-intersection) state once a workspace scope is in play, not
        just the conversation's own scope -- this is the seam that carries
        that resolution into the display layer, mirroring
        ``rag_scope.resolve_effective_scope``'s three states exactly.

        Args:
            effective: The resolved effective scope (conversation
                intersected with the linked workspace's scope, if any).
            conv_item_count: The conversation-level scope's own item count,
                or ``None`` when the conversation has no scope. Carried
                through only for the chip's breakdown tooltip.
            ws_item_count: The linked workspace's scope item count, or
                ``None`` when unset/unlinked. Carried through only for the
                chip's breakdown tooltip.
        """
        if effective.state == "unscoped":
            return cls.unscoped()
        if effective.state == "empty":
            return cls(
                is_scoped=False,
                item_count=0,
                is_empty=True,
                cause=effective.cause,
                conv_item_count=conv_item_count,
                ws_item_count=ws_item_count,
            )
        total = sum(len(ids) for ids in effective.allowlist.values())
        return cls(
            is_scoped=True,
            item_count=total,
            conv_item_count=conv_item_count,
            ws_item_count=ws_item_count,
        )


@dataclass(frozen=True)
class ConsoleInspectorState:
    """Display state for Console run/readiness inspection."""

    rows: tuple[ConsoleDisplayRow, ...]
    actions: tuple[ConsoleInspectorAction, ...] = ()
    has_pending_approval: bool = False
    can_save_chatbook: bool = False
    dictionary_rows: tuple[ConsoleDisplayRow, ...] = ()
    dictionary_actions: tuple[ConsoleInspectorAction, ...] = ()
    world_book_rows: tuple[ConsoleDisplayRow, ...] = ()
    world_book_actions: tuple[ConsoleInspectorAction, ...] = ()
    #: TASK-347: whether a generation is actively running (thinking or
    #: streaming) — the status-summary/Live-work surfaces read this so they
    #: stop claiming "Ready" mid-run.
    run_active: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        live_work_title: Any = None,
        provider_label: Any = None,
        model_label: Any = None,
        provider_ready: bool = True,
        provider_recovery: Any = None,
        rag_status: Any = None,
        evidence_summary: Any = None,
        evidence_status: Any = None,
        evidence_recovery: Any = None,
        evidence_authority: Any = None,
        artifact_status: Any = None,
        tool_count: int = 0,
        approval_count: int = 0,
        mcp_tool_count: int | None = None,
        mcp_not_connected_count: int = 0,
        can_save_chatbook: bool = False,
        scope_item_count: int | None = None,
        run_active: bool = False,
        ephemeral: bool = False,
        change_review_available: bool = False,
    ) -> "ConsoleInspectorState":
        provider_status = "ready" if provider_ready else "blocked"
        # F2 (task-9 review): the inspector's Save Chatbook action is a
        # second door onto the same write the Console workbench action
        # already gates -- consult the same registry entry.
        chatbook_blocked = blocked_reason("save-chatbook", ephemeral=ephemeral)
        normalized_tool_count = coerce_non_negative_int(tool_count)
        effective_tool_count = normalized_tool_count + (mcp_tool_count or 0)
        normalized_approval_count = coerce_non_negative_int(approval_count)
        rag_value = _clean(rag_status, "not staged")
        provider_value = _clean(provider_label, "provider")
        model_value = _clean(model_label, "no model")
        source_summary = rag_value
        run_recipe = (
            f"{provider_value} / {model_value} / sources {source_summary} / "
            f"tools {normalized_tool_count} / approvals {normalized_approval_count}"
        )
        # task-9: an active conversation RAG retrieval scope surfaces on the
        # run recipe line ("... / scope N items"). ``None`` (unscoped, the
        # overwhelming common case) leaves the line unchanged; a scope with
        # zero items never reaches here (see ``ConsoleRetrievalScopeState``).
        if scope_item_count is not None and scope_item_count > 0:
            run_recipe = f"{run_recipe} / scope {scope_item_count} items"
        rows = [
            ConsoleDisplayRow("Run recipe", run_recipe),
            ConsoleDisplayRow(
                "Live work",
                # TASK-347: a running generation shows "Generating…"; else
                # the pending Library-RAG launch title, else no active work.
                "Generating…"
                if run_active
                else _clean(live_work_title, "No active work"),
            ),
            ConsoleDisplayRow(
                "Provider",
                provider_status,
                status=provider_status,
                recovery=_clean(provider_recovery, "") if not provider_ready else "",
            ),
            ConsoleDisplayRow(
                "Sources",
                source_summary,
                status="blocked" if _is_blocked_rag_status(source_summary) else "ready",
            ),
            # TASK-1843: same derivation as the chip. `tool_count` alone comes
            # from a getattr hook production never populates, so this row read
            # "0 ready" beside a chip reporting a real number -- the same bug
            # already fixed once on the chip and missed here. Both now share
            # `_tools_ready_text`, including the neutral zero placeholder
            # ("0 ready" reads as "no tools available" when built-ins like
            # calculator/get_current_datetime are always registered).
            # TASK-2154.12 (TX-04): the zero placeholder is now an inert
            # dash -- "not loaded" exposed a lazy-loading implementation
            # detail. The CHIP additionally hides at zero; this row stays
            # mounted so the Inspector's group structure doesn't reflow.
            ConsoleDisplayRow("Tools", _tools_ready_text(effective_tool_count)),
            ConsoleDisplayRow(
                "Approvals",
                f"{normalized_approval_count} pending",
                status="blocked" if normalized_approval_count > 0 else "ready",
            ),
        ]
        mcp_row = _mcp_inspector_row(mcp_tool_count, mcp_not_connected_count)
        if mcp_row is not None:
            rows.append(mcp_row)
        if _clean(evidence_summary, ""):
            rows.append(
                ConsoleDisplayRow(
                    "Evidence",
                    _clean(evidence_summary, ""),
                    status=_clean(evidence_status, "ready"),
                    recovery=_clean(evidence_recovery, ""),
                )
            )
        if _clean(evidence_authority, ""):
            rows.append(
                ConsoleDisplayRow(
                    "Authority",
                    _clean(evidence_authority, ""),
                    status=_clean(evidence_status, "ready"),
                )
            )
        rows.append(
            ConsoleDisplayRow("Artifacts", _clean(artifact_status, "unavailable"))
        )
        actions = [
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                label=CONSOLE_INSPECTOR_REVIEW_APPROVAL_LABEL,
                enabled=normalized_approval_count > 0,
                disabled_reason=CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
            ),
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_REVIEW_CHANGES_ID,
                label=CONSOLE_INSPECTOR_REVIEW_CHANGES_LABEL,
                enabled=change_review_available,
                disabled_reason=CONSOLE_INSPECTOR_NO_CHANGE_TRACKING_REASON,
            ),
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
                label=CONSOLE_INSPECTOR_SAVE_CHATBOOK_LABEL,
                enabled=can_save_chatbook and chatbook_blocked is None,
                disabled_reason=(
                    chatbook_blocked or CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON
                ),
            ),
        ]
        return cls(
            rows=tuple(rows),
            actions=tuple(actions),
            has_pending_approval=normalized_approval_count > 0,
            can_save_chatbook=can_save_chatbook,
            run_active=run_active,
        )

    def to_plain_text(self) -> str:
        return "\n".join(row.text for row in self.rows)


@dataclass(frozen=True)
class TurnFileEntry:
    """One changed file on a turn's transcript card (task: turn file card).

    ``label`` is what the row prints: the bare relpath for a single-root
    turn, ``<root-name>/<relpath>`` when the turn touched several roots.
    ``path``/``root`` stay separate because the diff loader needs the
    exact (row, path) pair the provider expects.
    """

    label: str
    path: str
    root: str
    status: str
    adds: int
    dels: int


def turn_file_entries(
    row_files: "Sequence[tuple[Mapping[str, Any], Sequence[Any]]]",
) -> "list[tuple[TurnFileEntry, Mapping[str, Any]]]":
    """Assemble a turn card's file rows from its snapshot rows.

    Pairs each entry with the EXACT row it came from, by position -- never
    keyed by root. A run's ``change_snapshots`` can hold rows from TWO
    windows on the SAME root (a turn's own window and its surviving
    sub-agents' post-turn window, PR3a-1 Task 6c; both markers carry the
    same ``change_review_run_id``), and a root-keyed dict silently drops
    one window's files and mispairs the rest. Building entries per row
    instead means each entry's diff is always read against the row that
    actually produced it, regardless of how many rows share a root.

    Semantics ruling (deliberate, mirrors the `v` Review screen): a card
    shows the UNION of ALL of its run's clean rows. When a turn and its
    post-turn window share a root, both of that run's markers therefore
    render the same union rather than each showing only its own slice --
    exactly like ``AgentRunsChangeReviewProvider.turns()`` groups every
    row for a ``run_id`` into one ``ReviewTurn`` regardless of window.

    Args:
        row_files: ``(row, changed_files)`` pairs in row order -- one pair
            per row of the run's ``change_snapshots``. ``changed_files`` is
            whatever ``AgentRunsChangeReviewProvider.changed_files(row)``
            returned for that exact row. Tracking-error rows contribute
            nothing -- the card degrades to the marker text for those (and
            may be passed in unfiltered; they are dropped below).

    Returns:
        ``(entry, row)`` pairs in row order then file order, labels
        root-prefixed only when more than one clean ROOT (not window)
        contributed.
    """
    clean = [
        (row, files) for row, files in row_files if not row.get("tracking_error")
    ]
    multi_root = len({str(row["root"]) for row, _ in clean}) > 1
    paired: list[tuple[TurnFileEntry, Mapping[str, Any]]] = []
    for row, files in clean:
        root = str(row["root"])
        prefix = f"{PurePath(root).name}/" if multi_root else ""
        for changed in files:  # ChangedFile
            paired.append(
                (
                    TurnFileEntry(
                        label=f"{prefix}{changed.path}",
                        path=changed.path,
                        root=root,
                        status=str(changed.status),
                        adds=int(changed.adds),
                        dels=int(changed.dels),
                    ),
                    row,
                )
            )
    return paired


@dataclass(frozen=True)
class ConversationFileEntry:
    """One file's cross-turn latest state in a conversation (review rail,
    TASK-18060 spec §1).

    ``label`` follows :class:`TurnFileEntry`'s exact convention: the bare
    relpath when every contributing row shares one root, ``<root-name>/
    <relpath>`` when they span several. ``run_id``/``snapshot_id`` name the
    NEWEST clean row that still covers this ``(root, path)`` -- the row
    whose diff the file's ``status``/``adds``/``dels`` come from, and the
    identity :func:`conversation_file_summary`'s caller (the rail's
    click-through) opens the Review screen against.

    **Counts honesty** (spec §1): ``adds``/``dels`` are that NEWEST row's
    own deltas for this file, not a sum across every turn that touched
    it -- callers must present them as "latest turn deltas", never as a
    cumulative total.
    """

    root: str
    path: str
    label: str
    status: str
    adds: int
    dels: int
    run_id: str
    snapshot_id: int
    note_count: int


def conversation_file_summary(
    rows_with_files: "Sequence[tuple[Mapping[str, Any], Sequence[ChangedFile]]]",
    note_counts: "Mapping[tuple[str, str], int]",
) -> "list[ConversationFileEntry]":
    """Cross-turn latest-state summary of a conversation's changed files.

    Pure assembly -- no I/O, no git, no DB. The caller (the provider's
    ``conversation_changed_files``) is responsible for filtering to CLEAN
    rows before calling this (``tracking_error`` falsy, ``end_sha``
    truthy -- the same guard :meth:`AgentRunsChangeReviewProvider.
    changed_files` applies) and for skipping any row whose diff raised
    ``ChangeTrackingError`` (retention-pruned history); a row that made it
    into ``rows_with_files`` is assumed to be fully readable.

    Latest-wins per ``(root, path)`` (spec §1): ``rows_with_files`` arrives
    OLDEST first (mirrors ``ORDER BY cs.id``), and each row's files simply
    overwrite whatever an earlier row recorded for the same path -- so a
    path deleted in an early turn and recreated in a later one correctly
    ends up "A", not "D", with no special-casing. A rename (``status
    "R"``) keys its entry by the NEW path (``changed.path``) and deletes
    any existing entry for the OLD path (``changed.old_path``) -- the old
    path stops existing as of that row.

    Args:
        rows_with_files: ``(row, changed_files)`` pairs, one per CLEAN
            snapshot row, oldest first. ``changed_files`` is whatever
            :meth:`AgentRunsChangeReviewProvider.changed_files` returned
            for that exact row.
        note_counts: ``{(root, path): count}`` from
            :meth:`AgentRunsDB.change_note_counts_for_conversation`,
            joined onto each surviving entry's CURRENT path (a rename's
            note count follows the note's own ``(root, path)`` key, which
            is unaffected by later renames -- callers accept that a note
            recorded against a path before it was renamed away no longer
            joins to the renamed entry).

    Returns:
        One :class:`ConversationFileEntry` per ``(root, path)`` still
        alive at the end of history, ordered NEWEST first by owning
        snapshot id, then by path.
    """
    multi_root = len({str(row["root"]) for row, _ in rows_with_files}) > 1
    latest: dict[tuple[str, str], ConversationFileEntry] = {}
    for row, files in rows_with_files:
        root = str(row["root"])
        run_id = str(row["run_id"])
        snapshot_id = int(row["id"])
        prefix = f"{PurePath(root).name}/" if multi_root else ""
        for changed in files:  # ChangedFile
            if changed.status == "R" and changed.old_path:
                latest.pop((root, str(changed.old_path)), None)
            path = str(changed.path)
            latest[(root, path)] = ConversationFileEntry(
                root=root,
                path=path,
                label=f"{prefix}{path}",
                status=str(changed.status),
                adds=int(changed.adds),
                dels=int(changed.dels),
                run_id=run_id,
                snapshot_id=snapshot_id,
                note_count=int(note_counts.get((root, path), 0)),
            )
    return sorted(
        latest.values(), key=lambda entry: (-entry.snapshot_id, entry.path)
    )


def _cell_trim_prefix(text: str, budget: int) -> str:
    """Keep as much of ``text``'s START as fits ``budget`` display cells.

    Drops trailing characters once the running cell width would exceed
    ``budget`` -- used to shorten the FIRST path component in
    :func:`middle_elide_path` (its directory hint lives at the front).
    Measured whole-character via ``cell_len``, so a double-width character
    that would not fully fit is dropped entirely rather than split.

    Args:
        text: The component text to trim.
        budget: Maximum display-cell width of the result.

    Returns:
        The longest prefix of ``text`` whose ``cell_len`` is ``<=
        budget``; ``""`` when ``budget <= 0``.
    """
    if budget <= 0:
        return ""
    kept: list[str] = []
    used = 0
    for char in text:
        width = cell_len(char)
        if used + width > budget:
            break
        kept.append(char)
        used += width
    return "".join(kept)


def _cell_trim_suffix(text: str, budget: int) -> str:
    """Keep as much of ``text``'s END as fits ``budget`` display cells.

    The mirror of :func:`_cell_trim_prefix`, used to shorten the LAST path
    component in :func:`middle_elide_path` -- its recognizable tail (often
    a file extension) lives at the end, so leading characters are dropped
    instead.

    Args:
        text: The component text to trim.
        budget: Maximum display-cell width of the result.

    Returns:
        The longest suffix of ``text`` whose ``cell_len`` is ``<=
        budget``; ``""`` when ``budget <= 0``.
    """
    if budget <= 0:
        return ""
    kept: list[str] = []
    used = 0
    for char in reversed(text):
        width = cell_len(char)
        if used + width > budget:
            break
        kept.append(char)
        used += width
    return "".join(reversed(kept))


def middle_elide_path(path: str, budget: int) -> str:
    """Middle-elide a path to fit a display budget, preserving both ends.

    Keeps the first and last path components intact -- the two fragments a
    user actually recognizes a file by (its directory of origin and its
    own name) -- and collapses everything between them into a single "…"
    placeholder component. Splits on "/" rather than going through
    `pathlib`: every path this renders (``TurnFileEntry.label``) is
    already a root-relative git path, not a local filesystem path to
    resolve, and git always uses "/" regardless of host OS.

    TASK-17611 (AC#5): budgeted in terminal display CELLS via
    ``rich.cells.cell_len``, not raw ``len()`` -- a path carrying
    double-width (CJK etc.) characters can fit comfortably within a
    character-count budget while still overflowing the actual row width
    by several cells; ``cell_len`` is the same width function
    Rich/Textual use to lay out text, so this budget check matches what
    actually gets painted. ASCII paths are unaffected: ``cell_len(text)
    == len(text)`` for any text with no wide/zero-width characters, so
    every existing ASCII-path caller/test keeps its exact prior result.

    Qodo round (same task): the ``"<first>/…/<last>"`` candidate is now
    itself MEASURED, not just assumed to fit -- a wide first/last
    component (or just a long one) can overflow the budget even after
    dropping every middle component, which the original AC#5 fix left
    unaddressed. When it does, both endpoint components are further
    trimmed, cell-aware (:func:`_cell_trim_prefix`/:func:`_cell_trim_
    suffix`), splitting the remaining budget between them (a component
    that already fits its half-share donates the rest to the other side)
    so the FINAL result never exceeds ``budget`` whenever ``budget`` is at
    least the ellipsis's own cell width -- only a budget too small even
    for the bare "…" placeholder is allowed to overflow, since there is
    nothing narrower left to offer.

    Args:
        path: The path to elide.
        budget: Maximum display-cell width of the result.

    Returns:
        ``path`` unchanged when it already fits within ``budget``, or when
        it has two or fewer components -- there is no middle left to drop
        without mangling the one meaningful fragment that remains (a bare
        filename, or a directory/filename pair where both ends already
        ARE the whole path). Otherwise ``"<first>/…/<last>"`` when that
        fits; when it doesn't, the same shape with one or both endpoint
        components further cell-trimmed to fit ``budget`` exactly (or as
        close as the budget allows) -- see the Qodo-round note above for
        the one case still allowed to overflow.
    """
    if cell_len(path) <= budget:
        return path
    parts = path.split("/")
    if len(parts) <= 2:
        return path
    first, last = parts[0], parts[-1]
    candidate = f"{first}/…/{last}"
    if cell_len(candidate) <= budget:
        return candidate

    ellipsis_width = cell_len("…")
    if budget < ellipsis_width:
        # Not even the bare placeholder fits -- nothing honest to return
        # that respects the budget; this is the one case allowed to
        # overflow (an unusably small budget).
        return "…"
    slash_width = cell_len("/")
    endpoints_budget = budget - ellipsis_width - (2 * slash_width)
    if endpoints_budget <= 0:
        # Room for the ellipsis (and maybe the slashes) but nothing left
        # for either endpoint component's own text.
        return "…"

    first_width = cell_len(first)
    last_width = cell_len(last)
    first_budget = endpoints_budget // 2
    last_budget = endpoints_budget - first_budget
    # A component that already fits its half-share doesn't need to eat
    # into the other's -- redistribute the unused allowance so the
    # tighter side gets more room instead of being trimmed needlessly.
    if first_width <= first_budget:
        last_budget += first_budget - first_width
        first_budget = first_width
    elif last_width <= last_budget:
        first_budget += last_budget - last_width
        last_budget = last_width

    trimmed_first = _cell_trim_prefix(first, first_budget)
    trimmed_last = _cell_trim_suffix(last, last_budget)
    return f"{trimmed_first}/…/{trimmed_last}"


# --------------------------------------------------------------------------
# Diff hunk segmentation + annotate/feedback loop (task: turn-file-card
# annotate loop, TASK-16800)
# --------------------------------------------------------------------------

#: Matches a unified-diff hunk header line verbatim, e.g. "@@ -1,4 +1,6 @@"
#: or "@@ -1,4 +1,6 @@ def foo():" (git's optional trailing function
#: context). Adapted from ``Tools/patch_tool_impls.py:58``'s
#: ``_HUNK_HEADER`` -- copied locally rather than imported so this module's
#: segmentation stays independent of the patch tool's own parser, which is
#: not to be modified for this feature.
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")

#: The delivery block's heading (spec §4). Shared verbatim between
#: ``render_diff_feedback_block`` and its consumer (the bridge attach seam,
#: Task 5) so the format can't drift between definition and use.
_DIFF_FEEDBACK_HEADING = "## Diff feedback from the user (on your earlier file changes)"


@dataclass(frozen=True)
class DiffHunk:
    """One hunk of a single file's unified diff, plus its shared prelude.

    ``file_prelude`` (the ``diff --git``/``index``/``---``/``+++`` lines)
    is identical across every hunk of the same file -- it is repeated on
    each ``DiffHunk`` rather than factored out so a hunk is a
    self-contained unit callers can pass around individually (e.g. one
    hunk per note).
    """

    header: str
    body_lines: tuple[str, ...]
    file_prelude: str


def split_unified_diff(text: str) -> list[DiffHunk]:
    """Segment one file's unified-diff text into per-hunk blocks.

    Always runs over the FULL diff text (spec §2) -- never a
    display-truncated slice -- so hunk indices are stable regardless of any
    display cap a caller later applies. Expects ``text`` to be a single
    file's diff output (e.g. ``provider.diff_text(row, path)``); a
    multi-file diff is not a supported input shape.

    Args:
        text: The unified diff text for one file, verbatim.

    Returns:
        One ``DiffHunk`` per ``@@ ... @@`` header found, in order. When
        the diff has no hunk headers at all (a binary file, or a clean
        rename with no content change), returns a single fallback
        ``DiffHunk`` with an empty ``header``/``file_prelude`` and every
        line of ``text`` as ``body_lines`` -- this keeps such diffs
        annotatable as one unit instead of vanishing from segmentation.
    """
    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if _HUNK_HEADER.match(line)]
    if not header_indices:
        return [DiffHunk(header="", body_lines=tuple(lines), file_prelude="")]

    file_prelude = "\n".join(lines[: header_indices[0]])
    hunks: list[DiffHunk] = []
    for position, start in enumerate(header_indices):
        end = (
            header_indices[position + 1]
            if position + 1 < len(header_indices)
            else len(lines)
        )
        hunks.append(
            DiffHunk(
                header=lines[start],
                body_lines=tuple(lines[start + 1 : end]),
                file_prelude=file_prelude,
            )
        )
    return hunks


#: Byte cap applied to a captured hunk excerpt (Qodo #5, PR #1779 fix
#: round). The line cap (``hunk_excerpt``'s ``cap`` parameter) alone is not
#: enough: a minified single-line file's ONE body line can carry far more
#: bytes than the whole delivery block's cap, and `render_diff_feedback_
#: block`'s per-note inclusion loop treats any note whose entry alone
#: exceeds the block cap as an unconditional queue-blocker (see that
#: function's own fix below). Bounding excerpt bytes at CAPTURE time keeps
#: newly-saved notes well clear of that failure mode; the render-time
#: truncation-to-fit guard below is what protects notes captured before
#: this cap existed.
_EXCERPT_BYTE_CAP = 4096
_EXCERPT_BYTE_CAP_TAIL = "… truncated"
#: Tail appended when `render_diff_feedback_block` truncates the OLDEST
#: pending note's excerpt to guarantee it is always deliverable (Qodo #5).
_EXCERPT_TRUNCATED_TO_FIT_TAIL = "… excerpt truncated to fit"


def _byte_safe_truncate(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes.

    Never splits a multi-byte codepoint: backs off byte-by-byte from a
    raw slice of the UTF-8 encoding until the remainder decodes cleanly.

    Args:
        text: The text to truncate.
        max_bytes: The maximum UTF-8 byte length of the result.

    Returns:
        The longest prefix of ``text`` that both decodes cleanly and fits
        in ``max_bytes`` bytes. Empty when ``max_bytes <= 0``.
    """
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


def _cap_text_to_byte_budget(text: str, budget_bytes: int, tail: str) -> str:
    """Truncate ``text`` to fit ``budget_bytes`` UTF-8 bytes, tail included.

    Prefers a line boundary: keeps whole lines from the start for as long
    as they fit, then appends ``"\\n" + tail``. The one line that does NOT
    fit whole is not simply dropped, though -- whatever budget remains
    after the last whole line is spent on a byte-safe PARTIAL prefix of
    it, so a body that is one huge line (e.g. a minified file's single
    diff line -- the motivating case for this cap) still yields a useful,
    budget-respecting excerpt instead of empty content past the header.

    Args:
        text: The text to cap.
        budget_bytes: Maximum UTF-8 byte length of the result, tail
            included.
        tail: An honest elision marker appended (on its own line) when
            truncation actually happens.

    Returns:
        ``text`` unchanged when it already fits ``budget_bytes``;
        otherwise a truncated prefix plus ``"\\n" + tail``, guaranteed to
        encode to at most ``budget_bytes`` UTF-8 bytes.
    """
    if len(text.encode("utf-8")) <= budget_bytes:
        return text

    tail_line = f"\n{tail}"
    tail_bytes = len(tail_line.encode("utf-8"))
    content_budget = budget_bytes - tail_bytes
    if content_budget <= 0:
        # No room for the tail alongside any content -- best effort: a
        # bare hard truncation, no tail, still honoring the byte budget.
        return _byte_safe_truncate(text, max(budget_bytes, 0))

    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    for line in lines:
        sep = "\n" if kept else ""
        sep_bytes = len(sep)  # sep is ASCII ("" or "\n") -- 1 byte or 0
        line_bytes = len(line.encode("utf-8"))
        if used + sep_bytes + line_bytes <= content_budget:
            kept.append(line)
            used += sep_bytes + line_bytes
            continue
        # This line doesn't fit whole -- spend whatever budget remains on
        # a byte-safe PARTIAL prefix of it rather than dropping its
        # content outright.
        remaining = content_budget - used - sep_bytes
        if remaining > 0:
            partial = _byte_safe_truncate(line, remaining)
            if partial:
                kept.append(partial)
        break

    return "\n".join(kept) + tail_line


def hunk_excerpt(hunk: DiffHunk, cap: int = 40, byte_cap: int = _EXCERPT_BYTE_CAP) -> str:
    """Render a capped, self-contained excerpt of one hunk.

    This is the retention safety net (spec §1): captured once at note
    creation, it keeps a note's display and delivery self-contained even
    after the shadow repo prunes the snapshots the hunk came from.

    Two independent caps apply, in order: ``cap`` bounds the number of
    body LINES (as before); ``byte_cap`` then bounds the whole rendered
    excerpt's UTF-8 BYTE size (Qodo #5, PR #1779 fix round) -- a line cap
    alone does not bound a minified single-line file's excerpt, which can
    carry far more bytes in that one line than the entire delivery
    block's cap.

    Args:
        hunk: The hunk to excerpt.
        cap: Maximum number of body lines to include before eliding.
        byte_cap: Maximum UTF-8 byte size of the rendered excerpt.

    Returns:
        The header (when non-empty) followed by up to ``cap`` body lines,
        newline-joined. When the body is longer than ``cap``, an honest
        "… N more lines" tail line is appended. When the result (line cap
        already applied) still exceeds ``byte_cap`` bytes, it is further
        truncated at a line boundary where possible with an honest
        "… truncated" tail.
    """
    parts: list[str] = []
    if hunk.header:
        parts.append(hunk.header)
    body = hunk.body_lines
    parts.extend(body[:cap])
    if len(body) > cap:
        parts.append(f"… {len(body) - cap} more lines")
    text = "\n".join(parts)
    return _cap_text_to_byte_budget(text, byte_cap, _EXCERPT_BYTE_CAP_TAIL)


def _diff_feedback_note_entry(note: Mapping[str, Any]) -> str:
    """Render one note's block entry (spec §4), sans the shared heading.

    The excerpt is fenced with FOUR backticks, not the usual three
    (final-review fix wave): the excerpt is a verbatim hunk body, and a
    hunk from a markdown-file diff can itself contain a triple-backtick
    line -- with a three-backtick fence that would prematurely close the
    fence mid-excerpt and corrupt the rest of the model payload. A bare
    diff line can start with a literal backtick but a *fenced code block*
    delimiter inside a diff of a markdown file is exactly the case this
    guards; four backticks is the standard "fence one level up" escape
    used for exactly this nesting problem.
    """
    short_id = str(note["run_id"])[:8]
    return (
        f"### {note['path']} — {note['hunk_header']}   [run {short_id}]\n"
        f"> {note['note']}\n"
        f"````\n{note['hunk_excerpt']}\n````"
    )


def _oldest_note_entry_truncated_to_fit(
    note: Mapping[str, Any], *, cap_bytes: int, held_after: int
) -> "str | None":
    """Shrink ONLY this note's excerpt so its entry fits under ``cap_bytes``.

    Queue-blocker guard (Qodo #5, PR #1779 fix round): the oldest pending
    note must always be deliverable, even one whose captured excerpt (a
    legacy row from before ``hunk_excerpt`` grew its own byte cap) is
    larger than the whole block cap. Only the excerpt is shrunk -- path,
    hunk header, and note text are never touched -- and the shrink budget
    already reserves room for the "… N more notes held" line the caller
    will need to append when ``held_after`` notes remain uninspected.

    Args:
        note: The oldest pending note's row dict.
        cap_bytes: The block's overall byte cap.
        held_after: How many notes after this one will be left pending
            (the caller always stops considering further notes once this
            guard engages, so this count is fixed at call time).

    Returns:
        The rendered entry (heading NOT included) when a truncation makes
        it fit; ``None`` when even a zero-length excerpt can't -- the
        note's own fixed metadata alone already exceeds the budget, so
        the caller falls back to the pre-fix excluded/held behavior.
    """
    heading_bytes = len(_DIFF_FEEDBACK_HEADING.encode("utf-8"))
    sep_bytes = 1  # the "\n" joining the heading and this entry
    if held_after > 0:
        holdover_bytes = len(
            f"\n\n… {held_after} more notes held for the next message".encode(
                "utf-8"
            )
        )
    else:
        holdover_bytes = 0

    # -1 for a strict-inequality safety margin, matching the rest of this
    # module's "strictly under cap" per-note convention.
    entry_budget = cap_bytes - heading_bytes - sep_bytes - holdover_bytes - 1
    skeleton = _diff_feedback_note_entry({**note, "hunk_excerpt": ""})
    skeleton_bytes = len(skeleton.encode("utf-8"))
    excerpt_budget = entry_budget - skeleton_bytes
    if excerpt_budget <= 0:
        return None

    truncated_excerpt = _cap_text_to_byte_budget(
        str(note["hunk_excerpt"]), excerpt_budget, _EXCERPT_TRUNCATED_TO_FIT_TAIL
    )
    return _diff_feedback_note_entry({**note, "hunk_excerpt": truncated_excerpt})


def render_diff_feedback_block(
    notes: Sequence[dict], *, cap_bytes: int = 16384
) -> "tuple[str, list[int]]":
    """Render the auto-attached diff-feedback block (spec §4).

    Notes are included oldest-first (callers pass ``ORDER BY id``) while
    the running UTF-8 size of the block-so-far stays under ``cap_bytes``:
    a note is included only if adding its full rendering keeps the total
    strictly under the cap. The first note that would push the block over
    the cap, and every note after it, are excluded and NOT stamped
    delivered by the caller -- they stay pending and ride the next send.

    Queue-blocker guard (Qodo #5, PR #1779 fix round): when the very
    FIRST (oldest) note alone doesn't fit -- typically a legacy row whose
    excerpt predates ``hunk_excerpt``'s own byte cap, since a freshly
    captured excerpt is now bounded well under this block's cap -- its
    excerpt is truncated to fit instead of excluding it outright, so the
    oldest pending note is (almost) always deliverable and can never
    permanently block every note behind it. Every note after it still
    follows the pre-existing break-at-cap behavior (they stay pending,
    riding the next send, holdover line as today).

    The cap covers the WHOLE rendered block, including the trailing
    "… N more notes held for the next message" line when one is needed --
    that line is never allowed to push the total over ``cap_bytes`` on its
    own. When nothing is excluded, no such line is appended and nothing is
    reserved for one, so a cap that exactly fits every note's own bytes is
    never needlessly short by the holdover line's size.

    Args:
        notes: ``change_notes`` row dicts, oldest first.
        cap_bytes: Maximum UTF-8 byte size of the rendered block.

    Returns:
        A ``(block, included_ids)`` pair. ``included_ids`` holds exactly
        the ``id`` of every note that made it into ``block``, in the same
        order. When any notes were excluded by the cap, ``block`` ends
        with a "… N more notes held for the next message" line. Empty
        ``notes`` returns ``("", [])``.
    """
    if not notes:
        return "", []

    lines: list[str] = [_DIFF_FEEDBACK_HEADING]
    included_count = 0
    floor = 0
    for index, note in enumerate(notes):
        entry = _diff_feedback_note_entry(note)
        candidate = "\n".join(lines + [entry])
        if len(candidate.encode("utf-8")) < cap_bytes:
            lines.append(entry)
            included_count += 1
            continue

        if index == 0:
            # Queue-blocker guard: this oldest note doesn't fit even alone
            # -- try shrinking ITS excerpt (only) so it is deliverable
            # anyway, rather than leaving it (and everything behind it)
            # pending forever.
            held_after = len(notes) - 1
            truncated_entry = _oldest_note_entry_truncated_to_fit(
                note, cap_bytes=cap_bytes, held_after=held_after
            )
            if truncated_entry is not None:
                lines.append(truncated_entry)
                included_count = 1
                # Guaranteed (by construction, see the helper's budget
                # math) to already fit under cap_bytes with the holdover
                # line included -- never evict it below this floor.
                floor = 1
        # Every note after the oldest keeps the pre-existing break-at-cap
        # behavior regardless of whether the guard above engaged.
        break

    if included_count == len(notes):
        # Everything fit -- no holdover line, so nothing needed to be
        # reserved for one either.
        return "\n".join(lines), [int(note["id"]) for note in notes]

    # At least one note was excluded by the loop above (which did not yet
    # account for the holdover line's own bytes). Evict from the tail
    # until the holdover-inclusive block actually fits under cap_bytes --
    # each eviction both shrinks the notes portion and (usually) shrinks
    # "held"'s digit count, so this converges quickly. ``floor`` is the
    # irreducible bound (0 normally; 1 when the queue-blocker guard above
    # already placed a guaranteed-to-fit truncated oldest note that must
    # never be evicted) and is returned even if it still exceeds
    # cap_bytes -- there is nothing left that may be evicted.
    while True:
        held = len(notes) - included_count
        block = "\n".join(lines) + f"\n\n… {held} more notes held for the next message"
        if len(block.encode("utf-8")) <= cap_bytes or included_count <= floor:
            included_ids = [int(note["id"]) for note in notes[:included_count]]
            return block, included_ids
        included_count -= 1
        lines.pop()


def format_diff_feedback_disclosure(notes: Sequence[dict]) -> str:
    """Render the disclosure text for delivered diff-feedback notes.

    Shared verbatim by live emission at run completion and by resume
    re-derivation from delivered ``change_notes`` rows (spec §4) -- both
    callers must render identical text for the same notes.

    Args:
        notes: ``change_notes`` row dicts to disclose, one line each.

    Returns:
        One "📝 Diff feedback attached — ``<path>`` ``<hunk_header>``:
        ``"<note>"``" line per note, newline-joined. Empty ``notes``
        returns ``""``.
    """
    return "\n".join(
        f'📝 Diff feedback attached — {note["path"]} {note["hunk_header"]}: "{note["note"]}"'
        for note in notes
    )
