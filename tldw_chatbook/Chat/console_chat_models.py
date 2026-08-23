"""Pure Console-native chat state contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Mapping
from uuid import uuid4

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchCheckpoint,
    )
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.Chat.provider_continuation import ProviderContinuationCheckpoint
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


class ConsoleMessageRole(str, Enum):
    """Roles used by the native Console transcript."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConsoleRunStatus(str, Enum):
    """Lifecycle states for a Console send or recovery run."""

    IDLE = "idle"
    VALIDATING = "validating"
    STREAMING = "streaming"
    CHECKING_CITATIONS = "checking_citations"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    FAILED = "failed"
    RETRYING = "retrying"


#: Console selection phase 3: run statuses during which an agent run counts
#: as active for selection review feedback -- Request changes / LGTM arm
#: (feedback queues behind the run via the prompt-queue seam); anything
#: else leaves those two menu actions gated (Comment never gates).
#: Single source of truth: both consumers derive from this set -- the
#: transcript's selection gating (a string frozenset of the values) and
#: the screen's active-run constant (used for the transcript poll timer,
#: the sub-agent badge cache, and the same feedback gating). New active
#: statuses must be added here, never at a consumer.
FEEDBACK_ACTIVE_RUN_STATUSES: frozenset["ConsoleRunStatus"] = frozenset(
    {
        ConsoleRunStatus.VALIDATING,
        ConsoleRunStatus.STREAMING,
        ConsoleRunStatus.CHECKING_CITATIONS,
        ConsoleRunStatus.RETRYING,
    }
)


class ConsoleSubmissionOrigin(str, Enum):
    """Origin of a Console turn entering the accepted-send boundary.

    ``AGENT_WAKE`` (PR3a-2 Task 5) is a machine-injected auto-wake turn:
    it requires a coordinator-issued ``AgentWakeAuthorization`` (the
    queue-token precedent), writes NO USER transcript row (a SYSTEM-class
    machine-origin notice instead), never clears the composer, and its
    notice reaches the model as a payload-only trailing user-role entry.
    """

    MANUAL = "manual"
    QUEUED = "queued"
    AGENT_WAKE = "agent_wake"


@dataclass(frozen=True, slots=True)
class ConsoleQueuedAcceptanceEvent:
    """Content-free notice that one claimed queue entry became a real turn."""

    session_id: str
    entry_id: str


@dataclass(frozen=True, slots=True)
class ConsoleControllerActivity:
    """Single content-free lifecycle projection for Console fleet consumers."""

    session_id: str
    occupies_slot: bool
    preparing_before_acceptance: bool
    accepted_live_turn: bool
    needs_approval: bool
    queued_count: int
    queue_paused: bool
    terminal_notification_eligible: bool

    @property
    def has_queued_work(self) -> bool:
        """Return whether queue-owned future work controls this session."""

        return self.queued_count > 0


@dataclass(frozen=True, slots=True)
class ConsoleLifecycleImpact:
    """Revisioned, content-free loss impact derived from Console activity."""

    revision: int
    live_run_count: int
    queued_session_count: int
    unsent_prompt_count: int

    @property
    def has_loss_risk(self) -> bool:
        """Return whether leaving would discard or cancel Console work."""

        return bool(
            self.live_run_count
            or self.queued_session_count
            or self.unsent_prompt_count
        )


class ConsoleRunMarker(str, Enum):
    """Fleet-visible run marker for a session (parallel-agents spec §6).

    Derived (never stored raw) by ``ConsoleChatController.run_marker_for``
    from a session's live run state, pending-approval flag, and unvisited
    terminal outcome. ``NONE`` is the steady state; the other four values
    are what a tab/fleet-summary glyph renders for a session that is not
    currently being viewed.
    """

    NONE = "none"
    RUNNING = "running"
    NEEDS_APPROVAL = "needs-approval"
    FINISHED_OK = "finished-ok"
    FINISHED_FAILED = "finished-failed"
    #: PR3a-2 Task 4: a background SUB-AGENT of this conversation settled
    #: after its turn had already returned, and the user has not viewed
    #: the conversation since. Deliberately distinct from FINISHED_OK/
    #: FINISHED_FAILED, which announce a TURN's unvisited outcome -- a
    #: survivor finishing is a different event (the turn it belonged to
    #: ended long ago, possibly in a previous app run: this marker is
    #: backed by the durable ``fleet_unseen`` conversation-local mark, so
    #: it survives restart where the turn markers cannot). Derived in the
    #: screen layer (the controller has no marks service); lowest
    #: precedence of the non-NONE markers -- any live or unvisited TURN
    #: state outranks it.
    SUBAGENT_UNSEEN = "subagent-unseen"


#: Glyph shown for each `ConsoleRunMarker` on Console session tabs and
#: sidebar conversation-browser rows (parallel-agents spec §6, PA-T8). NONE
#: maps to the empty string so an unmarked tab/row gets no glyph and no
#: stray leading space -- callers must guard the space themselves, e.g.
#: ``f"{glyph} {label}" if glyph else label``.
CONSOLE_RUN_MARKER_GLYPHS: dict[ConsoleRunMarker, str] = {
    ConsoleRunMarker.NONE: "",
    ConsoleRunMarker.RUNNING: "●",
    ConsoleRunMarker.NEEDS_APPROVAL: "◆",
    ConsoleRunMarker.FINISHED_OK: "✓",
    ConsoleRunMarker.FINISHED_FAILED: "✗",
    ConsoleRunMarker.SUBAGENT_UNSEEN: "◈",
}


#: Human-readable meaning for each `ConsoleRunMarker`, for tooltips that
#: decode the fleet glyph in context rather than leaving a reader to infer
#: ● / ◆ / ✓ / ✗ from shape alone (fleet-UX expert review F4, task-1233).
#: `NONE` maps to the empty string -- same "guard with `if meaning:`"
#: contract `CONSOLE_RUN_MARKER_GLYPHS` already documents for its own NONE
#: entry, so an unmarked tab/row's tooltip gets no stray suffix.
#:
#: TWIN CONSTANT -- see `CONSOLE_FLEET_MARKER_LEGEND` in
#: `tldw_chatbook/UI/Screens/chat_screen.py` (the F1 Help "Agents" section's
#: legend line, task-1232). That legend deliberately uses its OWN shorter
#: per-glyph wording ("running"/"needs approval"/"finished"/"failed") in one
#: combined scannable line, distinct from this dict's fuller in-context
#: phrasing ("agent running"/"waiting for approval"/"finished — unseen") --
#: a deliberate register split (task-1233 review round 1), not drift. If you
#: change what a glyph MEANS, update both.
CONSOLE_RUN_MARKER_MEANINGS: dict[ConsoleRunMarker, str] = {
    ConsoleRunMarker.NONE: "",
    ConsoleRunMarker.RUNNING: "agent running",
    ConsoleRunMarker.NEEDS_APPROVAL: "waiting for approval",
    ConsoleRunMarker.FINISHED_OK: "finished — unseen",
    ConsoleRunMarker.FINISHED_FAILED: "failed — unseen",
    # Deliberately "ended", not "finished": the durable mark behind this
    # marker is set for error/cancelled survivor settles too, and the
    # tooltip must not promise success the run log may contradict.
    ConsoleRunMarker.SUBAGENT_UNSEEN: "sub-agent ended in background — unseen",
}

#: Reverse lookup from rendered glyph to its meaning, for callers along the
#: sidebar conversation-browser pipeline that thread the resolved glyph
#: *string* rather than the `ConsoleRunMarker` enum itself (the pipeline
#: deliberately stores glyphs so `Workspaces/conversation_browser_state.py`
#: stays free of a Chat-layer model import -- see that module's own
#: `run_marker` docstrings). The empty NONE glyph is excluded so a lookup
#: miss (no marker) and an explicit `""` marker both fall back the same way
#: via `.get(glyph, "")`.
CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH: dict[str, str] = {
    glyph: CONSOLE_RUN_MARKER_MEANINGS[marker]
    for marker, glyph in CONSOLE_RUN_MARKER_GLYPHS.items()
    if glyph
}


@dataclass(frozen=True, slots=True)
class ConsoleFleetCompletionTarget:
    """Deep-link payload for a background sub-agent completion (PR3a-2 Task 4).

    Staged on the ``HandoffChannel.CONSOLE_FLEET_COMPLETION`` single-slot
    channel by the fleet drain consumer when a survivor settles while
    Console is NOT the active screen; the next Console mount claims it and
    switches to the named conversation's session. ``conversation_id`` is
    the bridge's durable id (the persisted conversation id when the
    session was saved, the native session id otherwise), so the claimer
    matches it against both ``session.persisted_conversation_id`` and
    ``session.id``.

    Attributes:
        conversation_id: The settled conversation's durable id (required).
        session_id: The native Console session id the drain event carried,
            when known -- a faster exact match for a still-open session.
    """

    conversation_id: str
    session_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not isinstance(
            self.session_id, str
        ):
            raise TypeError("Console fleet completion target ids must be text")
        normalized = self.conversation_id.strip()
        if not normalized:
            raise ValueError("Console fleet completion target needs a conversation id")
        object.__setattr__(self, "conversation_id", normalized)
        object.__setattr__(self, "session_id", self.session_id.strip())


ConsoleMessageStatus = Literal[
    "complete", "pending", "streaming", "stopped", "failed", "discarded"
]
ConsoleMessageFeedback = Literal["up", "down"]
ConsoleActivityKind = Literal[
    "thinking",
    "tool",
    "spawn",
    "tasks",
    "changes",
    "feedback",
    "warning",
    "activity",
]
ConsoleActivityStatus = Literal["success", "blocked", "failed", "done"]

_CONSOLE_ACTIVITY_KINDS = frozenset(
    {
        "thinking",
        "tool",
        "spawn",
        "tasks",
        "changes",
        "feedback",
        "warning",
        "activity",
    }
)
_CONSOLE_ACTIVITY_STATUSES = frozenset({"success", "blocked", "failed", "done"})


CONSOLE_DISPATCH_UNRECONSTRUCTABLE_REASON = (
    "Retry response is unavailable because one-shot prefill or transient evidence "
    "cannot be reconstructed exactly."
)
CONSOLE_DISPATCH_IN_FLIGHT_REASON = "Recovery action is already in progress."
CONSOLE_DISPATCH_DUPLICATE_WARNING = (
    "Retry anyway may send a duplicate request because delivery status is unknown."
)
CONSOLE_DISPATCH_DISCARDED_COPY = "Response discarded."
CONSOLE_EPHEMERAL_PROMOTION_BLOCK_COPY = (
    "Finish or discard the pending turn before saving."
)


class ConsoleDispatchRecoveryActionId(str, Enum):
    """Explicit actions available for a device-local dispatch owner."""

    RETRY_RESPONSE = "retry_response"
    RETRY_ANYWAY = "retry_anyway"
    DISCARD = "discard"


class ConsoleDispatchRecoveryKind(str, Enum):
    """Bounded loader and runtime outcomes for one assistant owner."""

    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    EPHEMERAL_ACCEPTED = "ephemeral_accepted"
    EPHEMERAL_DISPATCH_STARTED = "ephemeral_dispatch_started"
    REMOTE_ACCEPTED = "remote_accepted"
    REMOTE_DISPATCH_STARTED = "remote_dispatch_started"
    CONTINUATION = "continuation"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class ConsoleDispatchRecoveryAction:
    """Literal, UI-neutral action state for dispatch recovery."""

    action_id: ConsoleDispatchRecoveryActionId
    label: str
    enabled: bool
    disabled_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, ConsoleDispatchRecoveryActionId):
            raise TypeError("action_id must be a ConsoleDispatchRecoveryActionId")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("recovery action label must be non-empty text")
        if type(self.enabled) is not bool:
            raise TypeError("recovery action enabled must be a bool")
        if not isinstance(self.disabled_reason, str):
            raise TypeError("recovery action disabled_reason must be text")
        if self.enabled and self.disabled_reason:
            raise ValueError("enabled recovery actions cannot have a disabled reason")


@dataclass(frozen=True, slots=True)
class ConsoleDispatchRecoveryState:
    """One body-free, app-lifetime recovery projection for an assistant."""

    kind: ConsoleDispatchRecoveryKind
    assistant_message_id: str
    conversation_id: str
    visible_copy: str
    actions: tuple[ConsoleDispatchRecoveryAction, ...]
    warning: str = ""
    error_code: str | None = None
    checkpoint: "ConsoleDispatchCheckpoint | None" = field(
        default=None,
        repr=False,
    )
    queue_entry_id: str | None = None
    preparation_id: str | None = None
    in_flight: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConsoleDispatchRecoveryKind):
            raise TypeError("kind must be a ConsoleDispatchRecoveryKind")
        if not isinstance(self.assistant_message_id, str):
            raise TypeError("assistant_message_id must be text")
        if not isinstance(self.conversation_id, str):
            raise TypeError("conversation_id must be text")
        if not isinstance(self.visible_copy, str) or not self.visible_copy:
            raise ValueError("recovery visible_copy must be non-empty text")
        if type(self.actions) is not tuple or any(
            not isinstance(action, ConsoleDispatchRecoveryAction)
            for action in self.actions
        ):
            raise TypeError("actions must contain recovery actions")
        if not isinstance(self.warning, str):
            raise TypeError("warning must be text")
        if type(self.in_flight) is not bool:
            raise TypeError("in_flight must be a bool")
        if self.error_code is not None and (
            not isinstance(self.error_code, str)
            or not self.error_code
            or len(self.error_code) > 64
        ):
            raise ValueError("recovery error_code is invalid")

    def with_in_flight(self, in_flight: bool) -> "ConsoleDispatchRecoveryState":
        """Return an exact action-disabled/enabled projection for one intent."""

        if type(in_flight) is not bool:
            raise TypeError("in_flight must be a bool")
        if in_flight == self.in_flight:
            return self
        actions = tuple(
            replace(
                action,
                enabled=False,
                disabled_reason=CONSOLE_DISPATCH_IN_FLIGHT_REASON,
            )
            for action in self.actions
        )
        if not in_flight:
            truth = getattr(self.checkpoint, "reconstructability", None)
            reconstructable = bool(
                truth is not None
                and truth.attachments_reconstructable
                and truth.evidence_reconstructable
                and truth.prefill_reconstructable
            )
            actions = _console_dispatch_actions(
                self.kind,
                reconstructable=reconstructable,
                in_flight=False,
            )
        return replace(self, actions=actions, in_flight=in_flight)


def _console_dispatch_actions(
    kind: ConsoleDispatchRecoveryKind,
    *,
    reconstructable: bool,
    in_flight: bool,
) -> tuple[ConsoleDispatchRecoveryAction, ...]:
    if kind not in {
        ConsoleDispatchRecoveryKind.ACCEPTED,
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
    }:
        return ()
    started = kind in {
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
    }
    retry_id = (
        ConsoleDispatchRecoveryActionId.RETRY_ANYWAY
        if started
        else ConsoleDispatchRecoveryActionId.RETRY_RESPONSE
    )
    retry_enabled = reconstructable and not in_flight
    retry_reason = ""
    if in_flight:
        retry_reason = CONSOLE_DISPATCH_IN_FLIGHT_REASON
    elif not reconstructable:
        retry_reason = CONSOLE_DISPATCH_UNRECONSTRUCTABLE_REASON
    discard_enabled = not in_flight
    return (
        ConsoleDispatchRecoveryAction(
            retry_id,
            "Retry anyway" if started else "Retry response",
            retry_enabled,
            retry_reason,
        ),
        ConsoleDispatchRecoveryAction(
            ConsoleDispatchRecoveryActionId.DISCARD,
            "Discard",
            discard_enabled,
            "" if discard_enabled else CONSOLE_DISPATCH_IN_FLIGHT_REASON,
        ),
    )


def console_dispatch_recovery_from_checkpoint(
    checkpoint: "ConsoleDispatchCheckpoint",
    *,
    ephemeral: bool = False,
    in_flight: bool = False,
) -> ConsoleDispatchRecoveryState:
    """Derive exact local actions from one validated dispatch owner."""

    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchCheckpoint,
        ConsoleDispatchCheckpointState,
    )

    if not isinstance(checkpoint, ConsoleDispatchCheckpoint):
        raise TypeError("checkpoint must be a ConsoleDispatchCheckpoint")
    started = checkpoint.state is ConsoleDispatchCheckpointState.DISPATCH_STARTED
    if ephemeral:
        kind = (
            ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED
            if started
            else ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
        )
    else:
        kind = (
            ConsoleDispatchRecoveryKind.DISPATCH_STARTED
            if started
            else ConsoleDispatchRecoveryKind.ACCEPTED
        )
    truth = checkpoint.reconstructability
    reconstructable = bool(
        truth.attachments_reconstructable
        and truth.evidence_reconstructable
        and truth.prefill_reconstructable
    )
    return ConsoleDispatchRecoveryState(
        kind=kind,
        assistant_message_id=checkpoint.assistant_message_id,
        conversation_id=checkpoint.conversation_id,
        visible_copy=(
            "Response delivery status is unknown on the source device."
            if started
            else "Response accepted; waiting for dispatch."
        ),
        warning=CONSOLE_DISPATCH_DUPLICATE_WARNING if started else "",
        actions=_console_dispatch_actions(
            kind,
            reconstructable=reconstructable,
            in_flight=in_flight,
        ),
        checkpoint=checkpoint,
        queue_entry_id=checkpoint.queue_entry_id,
        preparation_id=checkpoint.preparation_id,
        in_flight=in_flight,
    )


@dataclass(frozen=True)
class ConsoleActivityPresentation:
    """Bounded, session-only presentation facts for one activity marker."""

    kind: ConsoleActivityKind
    label: str
    status: ConsoleActivityStatus

    def __post_init__(self) -> None:
        """Reject unbounded labels and values outside the public vocabulary."""
        if self.kind not in _CONSOLE_ACTIVITY_KINDS:
            raise ValueError("activity kind is invalid")
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > 200
            or "\n" in self.label
            or "\r" in self.label
        ):
            raise ValueError(
                "activity label must be a non-empty single line <= 200 chars"
            )
        if self.status not in _CONSOLE_ACTIVITY_STATUSES:
            raise ValueError("activity status is invalid")


CONSOLE_GLOBAL_WORKSPACE_ID = "global"
DEFAULT_CONSOLE_SESSION_TITLE = "Chat 1"

CONSOLE_AUTO_TITLE_MAX_LENGTH = 30
_DEFAULT_CONSOLE_SESSION_TITLE_RE = re.compile(r"^Chat \d+$")

# Parallel-agents spec S4 (task-5): user-adjustable global cap on
# simultaneous Console runs. Single source of truth for the default --
# ConsoleChatController.max_parallel_runs reads it as the get_cli_setting
# fallback, and settings_screen.py's DEFAULT_CONSOLE_MAX_PARALLEL_RUNS
# aliases it so the settings UI and the controller can never drift apart.
CONSOLE_DEFAULT_MAX_PARALLEL_RUNS = 3
# send_refusal_copy names at most this many busy sessions before folding
# the rest into an "and N more" suffix.
CONSOLE_CAP_REFUSAL_TITLE_LIMIT = 3


class ConsoleCitationPhase(str, Enum):
    """Approved structural phases for transient citation presentation."""

    CHECKING = "checking"
    REPAIRING = "repairing"
    SELECTED = "selected"


class ConsoleCitationNoticeCode(str, Enum):
    """Approved structural citation-result notices."""

    REPAIRED = "repaired"
    UNAVAILABLE = "unavailable"
    CANCELED = "canceled"


@dataclass(frozen=True)
class ConsoleCitationPresentation:
    """Content-free transient citation presentation metadata."""

    phase: ConsoleCitationPhase
    notice_code: ConsoleCitationNoticeCode | None = None
    original_attempt_available: bool = False

    def __post_init__(self) -> None:
        """Reject unbounded or non-structural presentation values."""
        if not isinstance(self.phase, ConsoleCitationPhase):
            raise ValueError("phase must be an approved ConsoleCitationPhase")
        if self.notice_code is not None and not isinstance(
            self.notice_code, ConsoleCitationNoticeCode
        ):
            raise ValueError(
                "notice_code must be an approved ConsoleCitationNoticeCode"
            )
        if type(self.original_attempt_available) is not bool:
            raise ValueError("original_attempt_available must be a bool")


def is_default_console_session_title(title: str) -> bool:
    """Return whether a session title is an auto-numbered default like ``Chat 3``."""
    return bool(_DEFAULT_CONSOLE_SESSION_TITLE_RE.match(str(title or "").strip()))


def derive_console_session_title(
    draft: str,
    *,
    max_length: int = CONSOLE_AUTO_TITLE_MAX_LENGTH,
) -> str:
    """Derive a conversation title from the first user message.

    Args:
        draft: Validated composer draft text.
        max_length: Maximum title length including the ellipsis suffix.

    Returns:
        A collapsed, truncated title, or an empty string for blank drafts.
    """
    collapsed = " ".join(str(draft or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= max_length:
        return collapsed
    if max_length < 3:
        return collapsed[:max_length]
    return f"{collapsed[: max_length - 3].rstrip()}..."


@dataclass(frozen=True)
class ConsoleStagedSource:
    """A source currently staged for use by Console."""

    source_id: str
    label: str
    source_type: str
    workspace_id: str | None = None


@dataclass(frozen=True)
class ConsoleWorkspaceContext:
    """Workspace and source policy state used before sending to a provider."""

    active_workspace_id: str = CONSOLE_GLOBAL_WORKSPACE_ID
    staged_sources: tuple[ConsoleStagedSource, ...] = ()
    active_run_id: str | None = None
    handoff_id: str | None = None

    @property
    def blocked_sources(self) -> list[ConsoleStagedSource]:
        """Return staged sources that cannot be used in the active workspace."""
        return [
            source
            for source in self.staged_sources
            if source.workspace_id not in (None, self.active_workspace_id)
        ]

    @property
    def allowed_sources(self) -> list[ConsoleStagedSource]:
        """Return staged sources available to the active workspace."""
        blocked = {source.source_id for source in self.blocked_sources}
        return [
            source for source in self.staged_sources if source.source_id not in blocked
        ]

    @property
    def has_policy_blocks(self) -> bool:
        """Return whether any staged source is blocked by workspace policy."""
        return bool(self.blocked_sources)

    @property
    def recovery_copy(self) -> str:
        """Human-readable recovery text for workspace policy blocks."""
        labels = ", ".join(source.label for source in self.blocked_sources)
        return f"Workspace policy blocked sources outside {self.active_workspace_id}: {labels}"


@dataclass(frozen=True)
class ConsoleProviderSelection:
    """Effective provider/model/base URL selected for a Console send."""

    provider: str
    base_url: str | None = None
    explicit_model: str | None = None
    configured_model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    streaming: bool = True
    #: Optional per-session system prompt to prepend to provider messages.
    #: Not used for readiness resolution; carried through so the controller
    #: can build provider messages from a single selection snapshot.
    system_prompt: str | None = None
    workspace_context: ConsoleWorkspaceContext = field(
        default_factory=ConsoleWorkspaceContext
    )


@dataclass(frozen=True)
class ConsoleRunState:
    """Visible run state surfaced in Console controls and inspector."""

    status: ConsoleRunStatus = ConsoleRunStatus.IDLE
    visible_copy: str = ""

    @classmethod
    def blocked(cls, visible_copy: str) -> "ConsoleRunState":
        """Build a blocked run state with visible recovery copy."""
        return cls(ConsoleRunStatus.BLOCKED, visible_copy)

    @classmethod
    def retrying(
        cls, visible_copy: str = "Retrying failed response"
    ) -> "ConsoleRunState":
        """Build a retrying run state."""
        return cls(ConsoleRunStatus.RETRYING, visible_copy)

    @property
    def is_send_allowed(self) -> bool:
        """Return whether Console can accept a new send from this state."""
        return self.status in {
            ConsoleRunStatus.IDLE,
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }

    @property
    def is_stop_allowed(self) -> bool:
        """Return whether Console can stop an active stream from this state."""
        return self.status in {
            ConsoleRunStatus.STREAMING,
            ConsoleRunStatus.CHECKING_CITATIONS,
        }


@dataclass(frozen=True)
class MessageAttachment:
    """One attachment carried by a Console message (position 0 = legacy slot)."""

    data: bytes | None
    mime_type: str
    display_name: str
    position: int


@dataclass(frozen=True)
class GenerationVariantMeta:
    """Per-variant image-generation metadata (mirrors a ``message_generation_metadata`` row).

    Position is deliberately NOT stored on the instance -- callers track it
    externally via index alignment with the owning message's attachments:
    index i of ``ConsoleChatMessage.generation_metadata`` always describes
    ``attachments[i]`` (attachment position i). ``to_row``/``from_row``
    convert to/from the DB sidecar row shape, which DOES carry an explicit
    ``position`` column (``ChaChaNotes_DB.set_message_generation_metadata`` /
    ``get_generation_metadata_for_messages``).
    """

    prompt: str
    negative_prompt: str
    backend: str
    model: str | None
    seed: int | None
    style: str | None
    params: dict[str, Any]

    def to_row(self, position: int) -> dict[str, Any]:
        """Convert to a ``message_generation_metadata`` row dict for ``position``.

        Args:
            position: The attachment position this variant's metadata
                belongs to (not carried on the instance itself).

        Returns:
            A dict shaped for
            ``CharactersRAGDB.set_message_generation_metadata``/
            ``ChatPersistenceService.create_message(generation_metadata=...)``.
        """
        return {
            "position": position,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "backend": self.backend,
            "model": self.model,
            "seed": self.seed,
            "style": self.style,
            "params_json": json.dumps(self.params),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "GenerationVariantMeta":
        """Build from a DB sidecar row dict (``position``/``created_at`` ignored).

        Args:
            row: A row dict as returned by
                ``get_generation_metadata_for_messages`` (or an equivalent
                mapping built by a test fake).

        Returns:
            The decoded metadata. An unparseable ``params_json`` degrades to
            an empty ``params`` dict rather than raising.
        """
        raw_params = row.get("params_json") or "{}"
        try:
            params = (
                json.loads(raw_params)
                if isinstance(raw_params, str)
                else dict(raw_params)
            )
        except (TypeError, ValueError):
            params = {}
        return cls(
            prompt=row["prompt"],
            negative_prompt=row.get("negative_prompt", ""),
            backend=row["backend"],
            model=row.get("model"),
            seed=row.get("seed"),
            style=row.get("style"),
            params=params,
        )


@dataclass
class ConsoleChatMessage:
    """A native Console transcript message."""

    role: ConsoleMessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    status: ConsoleMessageStatus = "complete"
    persisted_message_id: str | None = None
    #: Persisted id of this node's PARENT in the conversation tree (None for a
    #: root / not-yet-known parent). Distinct from ``persisted_message_id``
    #: (this node's own persisted id). Used to reconstruct the active path.
    parent_message_id: str | None = None
    #: Transient (non-persisted) sibling-navigation hints the store fills in on
    #: active-path snapshots so the renderer can show `<`/`>` + an `n/m` counter
    #: without reaching into store internals. Default 0/1 = "no siblings".
    sibling_index: int = 0
    sibling_count: int = 1
    variants: "ConsoleVariantSet | None" = None
    feedback: ConsoleMessageFeedback | None = None
    image_data: bytes | None = None
    image_mime_type: str | None = None
    attachment_label: str | None = None
    attachments: tuple["MessageAttachment", ...] = ()
    #: Per-variant image-generation metadata, index-aligned with
    #: ``attachments`` (index i describes attachment position i). An empty
    #: tuple (the default) means this is NOT a generation message.
    generation_metadata: tuple["GenerationVariantMeta", ...] = ()
    #: Captured provider exchanges for this turn (Conversation Inspector).
    #: Tuple for snapshot-safety; the store replaces, never mutates.
    exchanges: tuple["ExchangeCapture", ...] = ()
    #: Safe current-session citation UI state. Never persisted or restored.
    citation_presentation: ConsoleCitationPresentation | None = None
    #: Structured activity-header facts. Session-only; never persisted,
    #: restored, sent to a provider, or written to the agent run log.
    activity_presentation: ConsoleActivityPresentation | None = None
    #: TASK-1860: the FULL, untruncated tool result behind a TOOL marker.
    #: ``content`` is a preview capped by the Console display setting, so
    #: without this the whole result was unreachable from the transcript --
    #: the user could not tell a complete result from its first N characters,
    #: and a failed call showed only its failure line. None for every message
    #: that is not a tool marker, and for a marker whose result was short
    #: enough that ``content`` already shows all of it.
    tool_output_full: str | None = None
    #: TASK-1366: the raw (file_path, before, after) contents behind a
    #: file-writing TOOL marker, captured live at the provider's strip seam
    #: (``BuiltinToolProvider.invoke``) for the transcript's inline diff
    #: row. Session-only display state: TOOL markers return from
    #: ``append_message`` before the persistence path, so this is NEVER
    #: persisted, and it is never echoed to the model or the run log --
    #: ``content``/``tool_output_full`` (post-strip text) are the only
    #: forms those consumers see. None for non-diff rows and for
    #: resume-rebuilt markers (AgentRunsDB holds only the stripped record,
    #: so a resumed run renders exactly as pre-diff).
    tool_diff: tuple[str, str, str] | None = None
    #: TASK-1972: set on a change-summary transcript row -- the agent run
    #: whose diff the row reviews. Session-only, never persisted; resume
    #: re-derives it from change_snapshots. The `v` action needs to know
    #: WHICH turn it opens, not guess from row position.
    change_review_run_id: str | None = None
    # Normalized token usage for THIS generation (None for user rows, legacy
    # rows, and providers that reported nothing). Persisted as usage_json.
    usage: "ProviderUsage | None" = None
    # TASK-2364: structured facts ABOUT the turn -- engine provenance, the
    # interrupted flag, and a voice row's transcript status. None for rows
    # that predate the field and for every turn with nothing to record.
    # Persisted as the local-only metadata_json column; the point of the
    # field is that machine consumers (reseed, exports, summaries) read it
    # instead of string-matching UI copy in ``content``.
    metadata: "MessageMetadata | None" = None
    # Private provider state owned by this exact assistant generation. It is
    # deliberately excluded from repr/render content while remaining part of
    # ordinary dataclass equality/copy semantics.
    provider_continuation: "ProviderContinuationCheckpoint | None" = field(
        default=None,
        repr=False,
    )
    # Safe restore/display facts kept separate from the private checkpoint.
    provider_continuation_warning: str | None = None
    provider_continuation_remote: bool = False
    provider_continuation_message_version: int | None = None
    # task-3401.4: structured facts about a generated VIDEO (slug name,
    # prompt/backend/seed/shape) -- the tombstone card's payload after the
    # ephemeral bytes are gone (ADR-044). Persisted as a namespaced key in
    # the same local-only metadata_json column. Mutually exclusive with
    # ``metadata`` by construction (a video row never carries turn
    # provenance and vice versa; persistence prefers this when set so a
    # resume+edit can never clobber the video payload). None for every
    # non-video message.
    video_metadata: "VideoGenerationMetadata | None" = None
    #: Render-only live activity line for an IN-FLIGHT assistant row ("⚙
    #: read_file · 4s"). Set by nothing that owns messages: the store never
    #: writes it, no persistence path reads it, and it is never echoed to a
    #: model or a run log. ``ConsoleTranscript`` alone stamps it, on a
    #: throwaway ``replace()`` copy made inside its row-planning walk (the
    #: same seam ``tool_output_full`` expansion uses), from a value the
    #: 0.2s Console poll re-derives every tick. It is therefore always
    #: ``""`` on every message the rest of the app ever sees.
    live_activity: str = ""


@dataclass(frozen=True)
class ConsoleVariant:
    """One regenerated variant for a turn."""

    content: str
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ConsoleVariantSet:
    """Regenerated variants for one turn with current selection state."""

    turn_id: str
    variants: list[ConsoleVariant]
    selected_index: int = 0

    @classmethod
    def from_contents(
        cls,
        *,
        turn_id: str,
        contents: list[str],
        selected_index: int = 0,
    ) -> "ConsoleVariantSet":
        """Build a variant set from raw message contents."""
        if not contents:
            raise ValueError("ConsoleVariantSet requires at least one variant")
        if selected_index < 0 or selected_index >= len(contents):
            raise ValueError("selected_index must reference an existing variant")
        return cls(
            turn_id=turn_id,
            variants=[ConsoleVariant(content) for content in contents],
            selected_index=selected_index,
        )

    @property
    def current(self) -> ConsoleVariant:
        """Return the currently selected variant."""
        return self.variants[self.selected_index]

    @property
    def can_go_previous(self) -> bool:
        """Return whether a previous variant exists."""
        return self.selected_index > 0

    @property
    def can_go_next(self) -> bool:
        """Return whether a next variant exists."""
        return self.selected_index < len(self.variants) - 1


@dataclass(frozen=True)
class ProjectInstructionPreview:
    """Disposable, user-requested preview of automatic project context.

    Only ``next_send_payload`` may contain the instruction body. The remaining
    fields are content-free metadata suitable for the Context diagnostics UI.
    """

    relative_source: str | None
    scope: str
    byte_count: int
    outcomes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    next_send_payload: dict[str, Any]


@dataclass(frozen=True)
class ProjectInstructionActivationEvent:
    """Content-free notice that project guidance changed for one run."""

    relative_sources: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    outcome_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = (
            self.relative_sources,
            self.scopes,
            self.outcome_codes,
        )
        if any(
            not isinstance(value, str) or "\n" in value or "\r" in value
            for values in fields
            for value in values
        ):
            raise ValueError("project instruction event values must be single-line text")


@dataclass(frozen=True)
class ConsoleContextSnapshot:
    """Independent snapshot of current transcript and next-send provider payload.

    ``frozen=True`` prevents reassigning the snapshot's top-level fields.
    ``independent`` means the snapshot is safe from store mutation: the
    ``current_messages`` and ``next_send_payload`` structures are copied at
    creation time, so mutating them does not change the underlying store.
    It does *not* promise deep immutability of nested values.
    """

    current_messages: list[ConsoleChatMessage]
    next_send_payload: dict[str, Any]
    project_instruction_preview: ProjectInstructionPreview | None = None


def fold_greeting_into_system_prompt(system_prompt: str, greeting: str) -> str:
    """Return the system content carrying a seeded assistant greeting.

    Strict providers (Anthropic, Gemini) reject an assistant-first message
    array, so a seeded character greeting cannot ride in the message list
    (task-427) -- but dropping it entirely makes the model contradict the
    greeting the user already read in the transcript (task-1531). Folding
    the greeting into the system row delivers it to every provider while
    keeping the message array user-first. The configured system prompt is
    kept verbatim at the start; the greeting block is appended after it.

    Args:
        system_prompt: The session's configured system prompt ("" for none).
        greeting: The seeded assistant greeting text ("" for none).

    Returns:
        The combined system content; "" when both inputs are blank.
    """
    greeting_text = (greeting or "").strip()
    if not greeting_text:
        return system_prompt
    opener_block = (
        "You already opened this conversation with the following message, "
        f"which the user has seen:\n{greeting_text}"
    )
    if not (system_prompt or "").strip():
        return opener_block
    return f"{system_prompt}\n\n{opener_block}"
