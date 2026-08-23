"""Pure, bounded, process-memory state for Console prompt queues.

The registry is deliberately synchronous and confined to the thread that creates
it.  It owns text-only queue entries and exposes body-free render snapshots; worker,
provider, persistence, and widget behavior belong to later layers.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from rich.cells import cell_len, split_graphemes
from rich.markup import escape as escape_markup

from tldw_chatbook.Utils.input_validation import validate_text_input


MAX_CONSOLE_QUEUE_ENTRIES = 10
MAX_CONSOLE_QUEUED_PROMPT_LENGTH = 100_000
PROMPT_PREVIEW_CELL_BUDGET = 96
_ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"

# OSC comes first so the generic two-byte escape form cannot consume only its
# introducer.  Incomplete OSC strings are stripped through end-of-input as a
# fail-closed preview rule.  CSI covers both ESC-[ and its one-byte C1 form.
_ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|$))"
    r"|(?:\x1b\[[0-?]*[ -/]*[@-~])"
    r"|(?:\x9b[0-?]*[ -/]*[@-~])"
    r"|(?:\x1b[@-_])"
)
_WHITESPACE_RE = re.compile(r"\s+")
_BIDI_CONTROLS = frozenset(
    chr(codepoint)
    for codepoint in (
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    )
)


class PromptQueueMode(str, Enum):
    """How a session's queue should behave after the current turn."""

    DRAINING = "draining"
    PAUSE_AFTER_TURN = "pause_after_turn"
    PAUSED = "paused"


class PromptQueuePauseReason(str, Enum):
    """Why automatic queue progression is paused."""

    MANUAL = "manual"
    FAILED = "failed"
    STOPPED = "stopped"
    CONTEXT_CHANGED = "context_changed"
    DISPATCH_REFUSED = "dispatch_refused"


class PromptQueueReservation(str, Enum):
    """Whether the session currently occupies its agent slot."""

    RELEASED = "released"
    HELD = "held"


class PromptQueueEntryPhase(str, Enum):
    """Body-free lifecycle phase exposed to renderers."""

    WAITING = "waiting"
    STARTING = "starting"


class QueueMutationStatus(str, Enum):
    """Content-free outcome for a registry intent."""

    APPLIED = "applied"
    UNCHANGED = "unchanged"
    STALE_REVISION = "stale_revision"
    INVALID = "invalid"
    FULL = "full"
    NOT_FOUND = "not_found"
    LOCKED = "locked"
    REROUTE_NORMAL_SEND = "reroute_normal_send"
    CLOSING = "closing"
    SHUTTING_DOWN = "shutting_down"


class QueueThreadViolation(RuntimeError):
    """Raised when queue state is touched outside its creating thread."""


def _strip_terminal_controls(text: str) -> str:
    text = _ANSI_ESCAPE_RE.sub("", text)
    safe: list[str] = []
    for character in text:
        if character in _BIDI_CONTROLS:
            continue
        if character.isspace():
            safe.append(" ")
            continue
        if unicodedata.category(character) in {"Cc", "Cs"}:
            continue
        safe.append(character)
    return "".join(safe)


def _truncate_cells(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if cell_len(text) <= budget:
        return text
    ellipsis_width = cell_len(_ELLIPSIS)
    if budget <= ellipsis_width:
        return _ELLIPSIS if budget == ellipsis_width else ""

    spans, _cell_offsets = split_graphemes(text)
    available = budget - ellipsis_width
    used = 0
    end = 0
    for _start, next_end, width in spans:
        if used + width > available:
            break
        used += width
        end = next_end
    prefix = text[:end].rstrip()
    while prefix and cell_len(prefix) + ellipsis_width > budget:
        prefix_spans, _ = split_graphemes(prefix)
        if not prefix_spans:
            prefix = ""
            break
        prefix = prefix[: prefix_spans[-1][0]].rstrip()
    return f"{prefix}{_ELLIPSIS}"


def make_prompt_preview(
    text: str,
    *,
    cell_budget: int = PROMPT_PREVIEW_CELL_BUDGET,
) -> str:
    """Return a one-line, terminal-safe, Rich-markup-safe prompt preview.

    The budget applies to rendered terminal cells.  Rich escaping happens only
    after fitting, so escape syntax does not consume the visible-cell budget.
    """

    if not isinstance(text, str):
        raise TypeError("queued prompt text must be a string")
    if not isinstance(cell_budget, int) or isinstance(cell_budget, bool):
        raise TypeError("preview cell budget must be an integer")
    normalized = _WHITESPACE_RE.sub(" ", _strip_terminal_controls(text)).strip()
    return escape_markup(_truncate_cells(normalized, max(0, cell_budget)))


@dataclass(frozen=True, slots=True, repr=False)
class QueuedPrompt:
    """One immutable canonical prompt; body-bearing fields are repr-hidden."""

    entry_id: str
    text: str = field(repr=False)
    preview: str = field(repr=False)
    insertion_order: int
    admitted_at: float

    def __repr__(self) -> str:
        return (
            "QueuedPrompt("
            f"entry_id={self.entry_id!r}, text=<redacted>, preview=<redacted>, "
            f"insertion_order={self.insertion_order}, admitted_at={self.admitted_at!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PromptQueueClaim:
    """A locked pre-accept queue entry."""

    prompt: QueuedPrompt = field(repr=False)
    claimed_at: float

    def __repr__(self) -> str:
        return (
            "PromptQueueClaim("
            f"entry_id={self.prompt.entry_id!r}, prompt=<redacted>, "
            f"claimed_at={self.claimed_at!r})"
        )


@dataclass(frozen=True, slots=True)
class PromptQueueEntrySnapshot:
    """Body-free render projection for one waiting or starting entry."""

    entry_id: str
    preview: str
    insertion_order: int
    position: int
    phase: PromptQueueEntryPhase


@dataclass(frozen=True, slots=True)
class PromptQueueSnapshot:
    """Immutable, body-free state consumed by controllers and widgets."""

    session_id: str
    revision: int
    entries: tuple[PromptQueueEntrySnapshot, ...]
    waiting_count: int
    claimed_count: int
    total_count: int
    mode: PromptQueueMode
    pause_reason: PromptQueuePauseReason | None
    reservation: PromptQueueReservation
    expected_context_epoch: int | None
    closing: bool


@dataclass(frozen=True, slots=True, repr=False)
class PromptQueueMutationResult:
    """Content-free transition result, optionally carrying one redacted claim."""

    status: QueueMutationStatus
    snapshot: PromptQueueSnapshot
    entry_id: str | None = None
    claim: PromptQueueClaim | None = field(default=None, repr=False)
    detail: str | None = None

    @property
    def applied(self) -> bool:
        return self.status is QueueMutationStatus.APPLIED

    def __repr__(self) -> str:
        claim_id = self.claim.prompt.entry_id if self.claim is not None else None
        return (
            "PromptQueueMutationResult("
            f"status={self.status!r}, session_id={self.snapshot.session_id!r}, "
            f"revision={self.snapshot.revision}, total_count={self.snapshot.total_count}, "
            f"entry_id={self.entry_id!r}, claim_entry_id={claim_id!r}, "
            f"detail={self.detail!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PromptQueueTextResult:
    """Revision-checked full text for one selected waiting entry only."""

    status: QueueMutationStatus
    session_id: str
    revision: int
    entry_id: str | None = None
    text: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            "PromptQueueTextResult("
            f"status={self.status!r}, session_id={self.session_id!r}, "
            f"revision={self.revision}, entry_id={self.entry_id!r}, "
            "text=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class PromptQueueShutdownResult:
    """Aggregate, body-free result of the global shutdown transition."""

    status: QueueMutationStatus
    registry_revision: int
    removed_sessions: int = 0
    removed_prompts: int = 0


@dataclass(slots=True)
class _SessionQueueState:
    session_id: str
    waiting: list[QueuedPrompt] = field(default_factory=list)
    claimed: PromptQueueClaim | None = None
    claimed_preparation_id: str | None = None
    revision: int = 0
    mode: PromptQueueMode = PromptQueueMode.DRAINING
    pause_reason: PromptQueuePauseReason | None = None
    reservation: PromptQueueReservation = PromptQueueReservation.RELEASED
    expected_context_epoch: int | None = None
    closing: bool = False
    reroute_admission_revision: int | None = None
    snapshot_cache: PromptQueueSnapshot | None = None

    @property
    def total_count(self) -> int:
        return len(self.waiting) + int(self.claimed is not None)


class ConsolePromptQueueRegistry:
    """Synchronous owner of bounded, revisioned per-session prompt queues."""

    DURABLE_ACCEPTANCE_TOMBSTONE_CAP = 256

    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._owner_thread_id = threading.get_ident()
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.RLock()
        self._states: dict[str, _SessionQueueState] = {}
        self._empty_snapshots: dict[str, PromptQueueSnapshot] = {}
        # Entry identities need only be unique while they can still be
        # referenced by a queue mutation or coordinator callback. Retaining
        # historical IDs forever would make this process-memory queue grow
        # with lifetime throughput instead of its bounded active contents.
        self._active_entry_ids: set[str] = set()
        self._durable_acceptance_tombstones: OrderedDict[tuple[str, str], str] = (
            OrderedDict()
        )
        self._next_insertion_order = 0
        self._registry_revision = 0
        self._shutting_down = False

    @property
    def registry_revision(self) -> int:
        self._assert_owner_thread()
        return self._registry_revision

    @property
    def shutting_down(self) -> bool:
        self._assert_owner_thread()
        return self._shutting_down

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise QueueThreadViolation(
                "Console prompt queue access must be marshalled to its owner thread"
            )

    @staticmethod
    def _session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        return session_id

    @staticmethod
    def _expected_revision(expected_revision: int) -> int:
        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        return expected_revision

    @staticmethod
    def _context_epoch(context_epoch: int) -> int:
        if (
            not isinstance(context_epoch, int)
            or isinstance(context_epoch, bool)
            or context_epoch < 0
        ):
            raise ValueError("context_epoch must be a non-negative integer")
        return context_epoch

    @staticmethod
    def _valid_text(text: str) -> bool:
        return (
            isinstance(text, str)
            and bool(text.strip())
            and validate_text_input(
                text,
                max_length=MAX_CONSOLE_QUEUED_PROMPT_LENGTH,
                allow_html=False,
            )
        )

    def _empty_snapshot(self, session_id: str) -> PromptQueueSnapshot:
        cached = self._empty_snapshots.get(session_id)
        if cached is None:
            cached = PromptQueueSnapshot(
                session_id=session_id,
                revision=0,
                entries=(),
                waiting_count=0,
                claimed_count=0,
                total_count=0,
                mode=PromptQueueMode.DRAINING,
                pause_reason=None,
                reservation=PromptQueueReservation.RELEASED,
                expected_context_epoch=None,
                closing=False,
            )
            self._empty_snapshots[session_id] = cached
        return cached

    @staticmethod
    def _entry_view_key(
        prompt: QueuedPrompt,
        phase: PromptQueueEntryPhase,
        position: int,
    ) -> tuple[str, str, int, PromptQueueEntryPhase, int]:
        return (
            prompt.entry_id,
            prompt.preview,
            prompt.insertion_order,
            phase,
            position,
        )

    def _snapshot(self, state: _SessionQueueState) -> PromptQueueSnapshot:
        cached = state.snapshot_cache
        if cached is not None and cached.revision == state.revision:
            return cached

        reusable: dict[
            tuple[str, str, int, PromptQueueEntryPhase, int],
            PromptQueueEntrySnapshot,
        ] = {}
        if cached is not None:
            for entry in cached.entries:
                reusable[
                    (
                        entry.entry_id,
                        entry.preview,
                        entry.insertion_order,
                        entry.phase,
                        entry.position,
                    )
                ] = entry

        projected: list[PromptQueueEntrySnapshot] = []
        prompts: list[tuple[QueuedPrompt, PromptQueueEntryPhase]] = []
        if state.claimed is not None:
            prompts.append((state.claimed.prompt, PromptQueueEntryPhase.STARTING))
        prompts.extend(
            (prompt, PromptQueueEntryPhase.WAITING) for prompt in state.waiting
        )
        for position, (prompt, phase) in enumerate(prompts):
            key = self._entry_view_key(prompt, phase, position)
            projected.append(
                reusable.get(key)
                or PromptQueueEntrySnapshot(
                    entry_id=prompt.entry_id,
                    preview=prompt.preview,
                    insertion_order=prompt.insertion_order,
                    position=position,
                    phase=phase,
                )
            )

        snapshot = PromptQueueSnapshot(
            session_id=state.session_id,
            revision=state.revision,
            entries=tuple(projected),
            waiting_count=len(state.waiting),
            claimed_count=int(state.claimed is not None),
            total_count=state.total_count,
            mode=state.mode,
            pause_reason=state.pause_reason,
            reservation=state.reservation,
            expected_context_epoch=state.expected_context_epoch,
            closing=state.closing,
        )
        state.snapshot_cache = snapshot
        return snapshot

    def snapshot(self, session_id: str) -> PromptQueueSnapshot:
        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        state = self._states.get(session_id)
        return (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )

    def read_waiting_text(
        self,
        session_id: str,
        *,
        entry_id: str,
        expected_revision: int,
    ) -> PromptQueueTextResult:
        """Materialize only the selected waiting entry's canonical text."""

        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return PromptQueueTextResult(
                status=refusal.status,
                session_id=session_id,
                revision=refusal.snapshot.revision,
                entry_id=entry_id,
            )
        assert state is not None
        if self._is_claimed(state, entry_id):
            return PromptQueueTextResult(
                status=QueueMutationStatus.LOCKED,
                session_id=session_id,
                revision=state.revision,
                entry_id=entry_id,
            )
        index = self._waiting_index(state, entry_id)
        if index is None:
            return PromptQueueTextResult(
                status=QueueMutationStatus.NOT_FOUND,
                session_id=session_id,
                revision=state.revision,
                entry_id=entry_id,
            )
        return PromptQueueTextResult(
            status=QueueMutationStatus.APPLIED,
            session_id=session_id,
            revision=state.revision,
            entry_id=entry_id,
            text=state.waiting[index].text,
        )

    def _result(
        self,
        status: QueueMutationStatus,
        session_id: str,
        *,
        state: _SessionQueueState | None = None,
        entry_id: str | None = None,
        claim: PromptQueueClaim | None = None,
        detail: str | None = None,
    ) -> PromptQueueMutationResult:
        snapshot = (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )
        return PromptQueueMutationResult(
            status=status,
            snapshot=snapshot,
            entry_id=entry_id,
            claim=claim,
            detail=detail,
        )

    def _check(
        self,
        session_id: str,
        expected_revision: int,
        *,
        allow_closing: bool = False,
    ) -> tuple[str, _SessionQueueState | None, PromptQueueMutationResult | None]:
        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        expected_revision = self._expected_revision(expected_revision)
        state = self._states.get(session_id)
        if self._shutting_down:
            return (
                session_id,
                state,
                self._result(
                    QueueMutationStatus.SHUTTING_DOWN, session_id, state=state
                ),
            )
        snapshot = (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )
        if expected_revision != snapshot.revision:
            return (
                session_id,
                state,
                self._result(
                    QueueMutationStatus.STALE_REVISION, session_id, state=state
                ),
            )
        if state is None:
            return (
                session_id,
                None,
                self._result(QueueMutationStatus.NOT_FOUND, session_id),
            )
        if state.closing and not allow_closing:
            return (
                session_id,
                state,
                self._result(QueueMutationStatus.CLOSING, session_id, state=state),
            )
        return session_id, state, None

    def _bump(
        self,
        state: _SessionQueueState,
        *,
        reroute_admission_from: int | None = None,
    ) -> None:
        state.revision += 1
        state.reroute_admission_revision = reroute_admission_from
        self._registry_revision += 1
        self._assert_invariants(state)

    @staticmethod
    def _assert_invariants(state: _SessionQueueState) -> None:
        if state.total_count > MAX_CONSOLE_QUEUE_ENTRIES:
            raise RuntimeError("Console prompt queue capacity invariant violated")
        identifiers = [prompt.entry_id for prompt in state.waiting]
        if state.claimed is not None:
            identifiers.append(state.claimed.prompt.entry_id)
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError("Console prompt queue entry identity invariant violated")
        # A paused+held state may exist only between the coordinator's synchronous
        # ``reserve`` and ``resume`` calls.  No await or widget mutation can occur
        # between those event-loop-thread-confined transitions.

    def _new_prompt(self, text: str) -> QueuedPrompt | None:
        entry_id = self._id_factory()
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or entry_id in self._active_entry_ids
        ):
            return None
        admitted_at = self._monotonic()
        if not isinstance(admitted_at, (int, float)) or isinstance(admitted_at, bool):
            return None
        self._next_insertion_order += 1
        prompt = QueuedPrompt(
            entry_id=entry_id,
            text=text,
            preview=make_prompt_preview(text),
            insertion_order=self._next_insertion_order,
            admitted_at=float(admitted_at),
        )
        self._active_entry_ids.add(entry_id)
        return prompt

    def _claim_time(self) -> float | None:
        claimed_at = self._monotonic()
        if not isinstance(claimed_at, (int, float)) or isinstance(claimed_at, bool):
            return None
        return float(claimed_at)

    @staticmethod
    def _waiting_index(state: _SessionQueueState, entry_id: str) -> int | None:
        for index, prompt in enumerate(state.waiting):
            if prompt.entry_id == entry_id:
                return index
        return None

    @staticmethod
    def _is_claimed(state: _SessionQueueState, entry_id: str) -> bool:
        return state.claimed is not None and state.claimed.prompt.entry_id == entry_id

    @staticmethod
    def _finalize_released_empty_state(state: _SessionQueueState) -> None:
        if state.total_count != 0 or state.reservation is PromptQueueReservation.HELD:
            return
        state.expected_context_epoch = None
        state.mode = PromptQueueMode.DRAINING
        state.pause_reason = None

    def begin_chain(
        self,
        session_id: str,
        *,
        context_epoch: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        expected_revision = self._expected_revision(expected_revision)
        context_epoch = self._context_epoch(context_epoch)
        state = self._states.get(session_id)
        if self._shutting_down:
            return self._result(
                QueueMutationStatus.SHUTTING_DOWN, session_id, state=state
            )
        current = (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )
        if expected_revision != current.revision:
            return self._result(
                QueueMutationStatus.STALE_REVISION, session_id, state=state
            )
        if state is None:
            state = _SessionQueueState(session_id=session_id)
            self._states[session_id] = state
        elif state.closing:
            return self._result(QueueMutationStatus.CLOSING, session_id, state=state)
        elif (
            state.total_count != 0
            or state.expected_context_epoch is not None
            or state.reservation is PromptQueueReservation.HELD
        ):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.expected_context_epoch = context_epoch
        state.reservation = PromptQueueReservation.HELD
        state.mode = PromptQueueMode.DRAINING
        state.pause_reason = None
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def admit(
        self,
        session_id: str,
        *,
        text: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        expected_revision = self._expected_revision(expected_revision)
        state = self._states.get(session_id)
        if self._shutting_down:
            return self._result(
                QueueMutationStatus.SHUTTING_DOWN, session_id, state=state
            )
        current = (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )
        if expected_revision != current.revision:
            if state is not None and (
                expected_revision == state.reroute_admission_revision
                and state.total_count == 0
                and state.expected_context_epoch is None
                and state.reservation is PromptQueueReservation.RELEASED
                and not state.closing
            ):
                return self._result(
                    QueueMutationStatus.REROUTE_NORMAL_SEND,
                    session_id,
                    state=state,
                )
            return self._result(
                QueueMutationStatus.STALE_REVISION, session_id, state=state
            )
        if state is None or (
            state.total_count == 0 and state.expected_context_epoch is None
        ):
            return self._result(
                QueueMutationStatus.REROUTE_NORMAL_SEND, session_id, state=state
            )
        if state.closing:
            return self._result(QueueMutationStatus.CLOSING, session_id, state=state)
        if not self._valid_text(text):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.total_count >= MAX_CONSOLE_QUEUE_ENTRIES:
            return self._result(QueueMutationStatus.FULL, session_id, state=state)
        prompt = self._new_prompt(text)
        if prompt is None:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.waiting.append(prompt)
        self._bump(state)
        return self._result(
            QueueMutationStatus.APPLIED,
            session_id,
            state=state,
            entry_id=prompt.entry_id,
        )

    def edit(
        self,
        session_id: str,
        *,
        entry_id: str,
        text: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if self._is_claimed(state, entry_id):
            return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
        index = self._waiting_index(state, entry_id)
        if index is None:
            return self._result(QueueMutationStatus.NOT_FOUND, session_id, state=state)
        if not self._valid_text(text):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        existing = state.waiting[index]
        if text == existing.text:
            return self._result(
                QueueMutationStatus.UNCHANGED,
                session_id,
                state=state,
                entry_id=entry_id,
            )
        state.waiting[index] = QueuedPrompt(
            entry_id=existing.entry_id,
            text=text,
            preview=make_prompt_preview(text),
            insertion_order=existing.insertion_order,
            admitted_at=existing.admitted_at,
        )
        self._bump(state)
        return self._result(
            QueueMutationStatus.APPLIED,
            session_id,
            state=state,
            entry_id=entry_id,
        )

    def move(
        self,
        session_id: str,
        *,
        entry_id: str,
        new_index: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if self._is_claimed(state, entry_id):
            return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
        if not isinstance(new_index, int) or isinstance(new_index, bool):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        current_index = self._waiting_index(state, entry_id)
        if current_index is None:
            return self._result(QueueMutationStatus.NOT_FOUND, session_id, state=state)
        if new_index < 0 or new_index >= len(state.waiting):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if new_index == current_index:
            return self._result(
                QueueMutationStatus.UNCHANGED,
                session_id,
                state=state,
                entry_id=entry_id,
            )
        prompt = state.waiting.pop(current_index)
        state.waiting.insert(new_index, prompt)
        self._bump(state)
        return self._result(
            QueueMutationStatus.APPLIED,
            session_id,
            state=state,
            entry_id=entry_id,
        )

    def remove(
        self,
        session_id: str,
        *,
        entry_id: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if self._is_claimed(state, entry_id):
            return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
        index = self._waiting_index(state, entry_id)
        if index is None:
            return self._result(QueueMutationStatus.NOT_FOUND, session_id, state=state)
        removed = state.waiting.pop(index)
        self._active_entry_ids.discard(removed.entry_id)
        self._finalize_released_empty_state(state)
        self._bump(state)
        return self._result(
            QueueMutationStatus.APPLIED,
            session_id,
            state=state,
            entry_id=entry_id,
        )

    def clear_waiting(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if not state.waiting:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        self._active_entry_ids.difference_update(
            prompt.entry_id for prompt in state.waiting
        )
        state.waiting.clear()
        self._finalize_released_empty_state(state)
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def claim_next(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        with self._lock:
            session_id, state, refusal = self._check(session_id, expected_revision)
            if refusal is not None:
                return refusal
            assert state is not None
            if state.claimed is not None:
                return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
            if (
                state.mode is not PromptQueueMode.DRAINING
                or state.reservation is not PromptQueueReservation.HELD
                or state.expected_context_epoch is None
            ):
                return self._result(
                    QueueMutationStatus.INVALID, session_id, state=state
                )
            if not state.waiting:
                return self._result(
                    QueueMutationStatus.NOT_FOUND, session_id, state=state
                )
            claimed_at = self._claim_time()
            if claimed_at is None:
                return self._result(
                    QueueMutationStatus.INVALID, session_id, state=state
                )
            prompt = state.waiting.pop(0)
            state.claimed = PromptQueueClaim(prompt=prompt, claimed_at=claimed_at)
            state.claimed_preparation_id = None
            self._bump(state)
            return self._result(
                QueueMutationStatus.APPLIED,
                session_id,
                state=state,
                entry_id=prompt.entry_id,
                claim=state.claimed,
            )

    def bind_claimed_preparation(
        self,
        session_id: str,
        *,
        entry_id: str,
        preparation_id: str,
    ) -> PromptQueueMutationResult:
        """Bind one exact live claim to its preparation before durable acceptance."""

        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        if type(entry_id) is not str or not entry_id:
            raise ValueError("entry_id must be non-empty text")
        if type(preparation_id) is not str or not preparation_id:
            raise ValueError("preparation_id must be non-empty text")
        with self._lock:
            state = self._states.get(session_id)
            if state is None or state.claimed is None:
                return self._result(
                    QueueMutationStatus.NOT_FOUND,
                    session_id,
                    state=state,
                    entry_id=entry_id,
                )
            if state.claimed.prompt.entry_id != entry_id:
                return self._result(
                    QueueMutationStatus.LOCKED,
                    session_id,
                    state=state,
                    entry_id=entry_id,
                )
            existing = state.claimed_preparation_id
            if existing is not None:
                status = (
                    QueueMutationStatus.UNCHANGED
                    if existing == preparation_id
                    else QueueMutationStatus.LOCKED
                )
                return self._result(status, session_id, state=state, entry_id=entry_id)
            state.claimed_preparation_id = preparation_id
            self._bump(state)
            return self._result(
                QueueMutationStatus.APPLIED,
                session_id,
                state=state,
                entry_id=entry_id,
            )

    def settle_claim(
        self,
        session_id: str,
        *,
        entry_id: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        with self._lock:
            session_id, state, refusal = self._check(session_id, expected_revision)
            if refusal is not None:
                return refusal
            assert state is not None
            if state.claimed is None:
                return self._result(
                    QueueMutationStatus.NOT_FOUND, session_id, state=state
                )
            if state.claimed.prompt.entry_id != entry_id:
                return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
            self._active_entry_ids.discard(state.claimed.prompt.entry_id)
            state.claimed = None
            state.claimed_preparation_id = None
            self._bump(state)
            return self._result(
                QueueMutationStatus.APPLIED,
                session_id,
                state=state,
                entry_id=entry_id,
            )

    def settle_durable_acceptance(
        self,
        session_id: str,
        *,
        entry_id: str,
        preparation_id: str,
    ) -> PromptQueueMutationResult:
        """Settle an exact committed claim even after its live chain vanished."""

        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        if type(entry_id) is not str or not entry_id:
            raise ValueError("entry_id must be non-empty text")
        if type(preparation_id) is not str or not preparation_id:
            raise ValueError("preparation_id must be non-empty text")
        with self._lock:
            key = (session_id, entry_id)
            prior = self._durable_acceptance_tombstones.get(key)
            state = self._states.get(session_id)
            if prior is not None:
                status = (
                    QueueMutationStatus.UNCHANGED
                    if prior == preparation_id
                    else QueueMutationStatus.LOCKED
                )
                return self._result(status, session_id, state=state, entry_id=entry_id)
            if state is None or state.claimed is None:
                return self._result(
                    QueueMutationStatus.NOT_FOUND,
                    session_id,
                    state=state,
                    entry_id=entry_id,
                )
            if (
                state.claimed.prompt.entry_id != entry_id
                or state.claimed_preparation_id != preparation_id
            ):
                return self._result(
                    QueueMutationStatus.LOCKED,
                    session_id,
                    state=state,
                    entry_id=entry_id,
                )
            self._active_entry_ids.discard(entry_id)
            state.claimed = None
            state.claimed_preparation_id = None
            if state.waiting:
                state.mode = PromptQueueMode.PAUSED
                state.pause_reason = PromptQueuePauseReason.FAILED
                state.reservation = PromptQueueReservation.RELEASED
            else:
                state.reservation = PromptQueueReservation.RELEASED
                self._finalize_released_empty_state(state)
            self._bump(state)
            self._durable_acceptance_tombstones[key] = preparation_id
            while (
                len(self._durable_acceptance_tombstones)
                > self.DURABLE_ACCEPTANCE_TOMBSTONE_CAP
            ):
                self._durable_acceptance_tombstones.popitem(last=False)
            return self._result(
                QueueMutationStatus.APPLIED,
                session_id,
                state=state,
                entry_id=entry_id,
            )

    def return_claim_to_head(
        self,
        session_id: str,
        *,
        entry_id: str,
        reason: PromptQueuePauseReason,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        with self._lock:
            session_id, state, refusal = self._check(session_id, expected_revision)
            if refusal is not None:
                return refusal
            assert state is not None
            if not isinstance(reason, PromptQueuePauseReason):
                return self._result(
                    QueueMutationStatus.INVALID, session_id, state=state
                )
            if state.claimed is None:
                return self._result(
                    QueueMutationStatus.NOT_FOUND, session_id, state=state
                )
            if state.claimed.prompt.entry_id != entry_id:
                return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
            state.waiting.insert(0, state.claimed.prompt)
            state.claimed = None
            state.claimed_preparation_id = None
            state.mode = PromptQueueMode.PAUSED
            state.pause_reason = reason
            state.reservation = PromptQueueReservation.RELEASED
            self._bump(state)
            return self._result(
                QueueMutationStatus.APPLIED,
                session_id,
                state=state,
                entry_id=entry_id,
            )

    def request_pause_after_turn(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if (
            state.mode is not PromptQueueMode.DRAINING
            or state.reservation is not PromptQueueReservation.HELD
            or state.total_count == 0
        ):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.mode = PromptQueueMode.PAUSE_AFTER_TURN
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def keep_draining(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if state.mode is not PromptQueueMode.PAUSE_AFTER_TURN:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.mode = PromptQueueMode.DRAINING
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def pause(
        self,
        session_id: str,
        *,
        reason: PromptQueuePauseReason,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if not isinstance(reason, PromptQueuePauseReason) or state.total_count == 0:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.claimed is not None:
            return self._result(QueueMutationStatus.LOCKED, session_id, state=state)
        if (
            state.mode is PromptQueueMode.PAUSED
            and state.pause_reason is reason
            and state.reservation is PromptQueueReservation.RELEASED
        ):
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.mode = PromptQueueMode.PAUSED
        state.pause_reason = reason
        state.reservation = PromptQueueReservation.RELEASED
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def reserve(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if state.total_count == 0 or state.expected_context_epoch is None:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.reservation is PromptQueueReservation.HELD:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.reservation = PromptQueueReservation.HELD
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def release_reservation(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if state.reservation is PromptQueueReservation.RELEASED:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        if state.mode is not PromptQueueMode.PAUSED and state.total_count > 0:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.reservation = PromptQueueReservation.RELEASED
        self._finalize_released_empty_state(state)
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def resume(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if state.mode is not PromptQueueMode.PAUSED:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.reservation is not PromptQueueReservation.HELD:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        state.mode = PromptQueueMode.DRAINING
        state.pause_reason = None
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def adopt_context_baseline(
        self,
        session_id: str,
        *,
        context_epoch: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        context_epoch = self._context_epoch(context_epoch)
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if (
            state.mode is not PromptQueueMode.PAUSED
            or state.pause_reason is not PromptQueuePauseReason.CONTEXT_CHANGED
        ):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.expected_context_epoch == context_epoch:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.expected_context_epoch = context_epoch
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def adopt_recovery_context_baseline(
        self,
        session_id: str,
        *,
        context_epoch: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Adopt an epoch changed by one coordinator-authorized recovery turn."""

        context_epoch = self._context_epoch(context_epoch)
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if (
            state.mode is not PromptQueueMode.DRAINING
            or state.reservation is not PromptQueueReservation.HELD
            or state.claimed is not None
            or state.total_count == 0
        ):
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if state.expected_context_epoch == context_epoch:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.expected_context_epoch = context_epoch
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def finalize_empty_chain(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(session_id, expected_revision)
        if refusal is not None:
            return refusal
        assert state is not None
        if state.total_count != 0:
            return self._result(QueueMutationStatus.INVALID, session_id, state=state)
        if (
            state.expected_context_epoch is None
            and state.reservation is PromptQueueReservation.RELEASED
        ):
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.expected_context_epoch = None
        state.reservation = PromptQueueReservation.RELEASED
        state.mode = PromptQueueMode.DRAINING
        state.pause_reason = None
        self._bump(state, reroute_admission_from=expected_revision)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def mark_closing(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        session_id, state, refusal = self._check(
            session_id,
            expected_revision,
            allow_closing=True,
        )
        if refusal is not None:
            return refusal
        assert state is not None
        if state.closing:
            return self._result(QueueMutationStatus.UNCHANGED, session_id, state=state)
        state.closing = True
        state.reservation = PromptQueueReservation.RELEASED
        self._bump(state)
        return self._result(QueueMutationStatus.APPLIED, session_id, state=state)

    def remove_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        self._assert_owner_thread()
        session_id = self._session_id(session_id)
        expected_revision = self._expected_revision(expected_revision)
        state = self._states.get(session_id)
        if self._shutting_down:
            return self._result(
                QueueMutationStatus.SHUTTING_DOWN, session_id, state=state
            )
        current = (
            self._snapshot(state)
            if state is not None
            else self._empty_snapshot(session_id)
        )
        if expected_revision != current.revision:
            return self._result(
                QueueMutationStatus.STALE_REVISION, session_id, state=state
            )
        if state is None:
            return self._result(QueueMutationStatus.UNCHANGED, session_id)
        self._active_entry_ids.difference_update(
            prompt.entry_id for prompt in state.waiting
        )
        if state.claimed is not None:
            self._active_entry_ids.discard(state.claimed.prompt.entry_id)
        del self._states[session_id]
        self._registry_revision += 1
        return self._result(QueueMutationStatus.APPLIED, session_id)

    def shutdown(self, *, expected_registry_revision: int) -> PromptQueueShutdownResult:
        self._assert_owner_thread()
        expected_registry_revision = self._expected_revision(expected_registry_revision)
        if self._shutting_down:
            return PromptQueueShutdownResult(
                status=QueueMutationStatus.UNCHANGED,
                registry_revision=self._registry_revision,
            )
        if expected_registry_revision != self._registry_revision:
            return PromptQueueShutdownResult(
                status=QueueMutationStatus.STALE_REVISION,
                registry_revision=self._registry_revision,
            )
        removed_sessions = len(self._states)
        removed_prompts = sum(state.total_count for state in self._states.values())
        self._shutting_down = True
        self._states.clear()
        self._active_entry_ids.clear()
        self._registry_revision += 1
        return PromptQueueShutdownResult(
            status=QueueMutationStatus.APPLIED,
            registry_revision=self._registry_revision,
            removed_sessions=removed_sessions,
            removed_prompts=removed_prompts,
        )

    def reopen(self) -> None:
        """Clear the shutdown latch so a NEXT Console visit can admit again.

        task-15860 (the teardown split). ``shutdown()`` remains the
        per-visit tombstone -- it still clears every state and bumps the
        revision -- but its ``_shutting_down`` latch is now per-visit too,
        because one app-owned registry serves every visit. Deliberately
        does NOT restore anything the tombstone removed.
        """

        self._assert_owner_thread()
        if not self._shutting_down:
            return
        self._shutting_down = False
        self._registry_revision += 1


__all__ = [
    "MAX_CONSOLE_QUEUE_ENTRIES",
    "MAX_CONSOLE_QUEUED_PROMPT_LENGTH",
    "PROMPT_PREVIEW_CELL_BUDGET",
    "ConsolePromptQueueRegistry",
    "PromptQueueClaim",
    "PromptQueueEntryPhase",
    "PromptQueueEntrySnapshot",
    "PromptQueueMode",
    "PromptQueueMutationResult",
    "PromptQueuePauseReason",
    "PromptQueueReservation",
    "PromptQueueShutdownResult",
    "PromptQueueSnapshot",
    "PromptQueueTextResult",
    "QueueMutationStatus",
    "QueueThreadViolation",
    "QueuedPrompt",
    "make_prompt_preview",
]
