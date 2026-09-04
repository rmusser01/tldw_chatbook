"""Run-owned Canvas staging and source-private turn contributions."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.limits import sha256_utf8
from tldw_chatbook.Canvas.models import (
    CanvasConflictResult,
    CanvasListItem,
    CanvasMutationResult,
    CanvasOrigin,
    CanvasReadResult,
    CanvasRevisionInfo,
    CanvasScope,
)
from tldw_chatbook.Canvas.repository import (
    CanvasImportBatch,
    CanvasImportDocument,
    CanvasImportRevision,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleExactNativeIdTransactionContribution,
    ConsoleTransactionWriter,
)
from tldw_chatbook.Chat.message_metadata import (
    CanvasCardMetadata,
    CanvasCardOriginMetadata,
)

_MAX_RETAINED_CLOSED_RUNS = 256


class CanvasRunState(str, Enum):
    """Closed lifecycle states for one assistant run's Canvas mutations."""

    OPEN = "open"
    READY = "ready"
    COMMITTED = "committed"
    DISCARDED = "discarded"


@dataclass(frozen=True, slots=True, eq=False)
class CanvasSessionOwner:
    """Process-only incarnation fence for one temporary Console session."""

    session_id: str
    incarnation: int
    _nonce: object = field(default_factory=object, repr=False, compare=False)


@dataclass(frozen=True, slots=True, eq=False)
class CanvasRunOwner:
    """Opaque process-only identity for one exact registered run stage."""

    run_id: str
    _nonce: object = field(default_factory=object, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _StagedRevision:
    info: CanvasRevisionInfo
    source: str = field(repr=False, compare=False)
    created_at: str
    creates_document: bool


@dataclass(frozen=True, slots=True)
class CanvasTurnContribution(ConsoleExactNativeIdTransactionContribution):
    """Source-bearing rows usable only inside the caller-owned turn transaction."""

    assistant_message_id: str
    revisions: tuple[_StagedRevision, ...] = field(repr=False)
    require_active_path: bool = field(default=True, repr=False)

    @property
    def revision_count(self) -> int:
        return len(self.revisions)

    @property
    def origin_message_id(self) -> str:
        return self.assistant_message_id

    def __repr__(self) -> str:
        return (
            "CanvasTurnContribution("
            f"assistant_message_id={self.assistant_message_id!r}, "
            f"revision_count={len(self.revisions)})"
        )

    def durable_acceptance_fingerprint(self) -> dict[str, object]:
        """Return a content-free identity for retry/canonicalization."""

        return {
            "assistant_message_id": self.assistant_message_id,
            "revisions": tuple(
                {
                    "canvas_id": row.info.canvas_id,
                    "revision_id": row.info.revision_id,
                    "parent_revision_id": row.info.parent_revision_id,
                    "sequence": row.info.sequence,
                    "digest": row.info.content_sha256,
                }
                for row in self.revisions
            ),
        }

    def write_exact(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        native_message_ids: Mapping[str, str],
    ) -> None:
        """Append through Canvas' canonical transaction-aware repository API."""

        documents: list[CanvasImportDocument] = []
        revisions: list[CanvasImportRevision] = []
        for row in self.revisions:
            origin_id = native_message_ids.get(row.info.origin.message_id)
            if not isinstance(origin_id, str) or not origin_id:
                raise RuntimeError("canvas_origin_mapping_missing")
            if row.creates_document:
                documents.append(
                    CanvasImportDocument(
                        canvas_id=row.info.canvas_id,
                        conversation_id=conversation_id,
                        created_at=row.created_at,
                    )
                )
            revisions.append(
                CanvasImportRevision(
                    revision_id=row.info.revision_id,
                    canvas_id=row.info.canvas_id,
                    parent_revision_id=row.info.parent_revision_id,
                    sequence=row.info.sequence,
                    title=row.info.title,
                    runtime_profile=row.info.runtime_profile,
                    source=row.source,
                    content_sha256=row.info.content_sha256,
                    source_bytes=row.info.source_bytes,
                    actor_kind="assistant",
                    origin_message_id=origin_id,
                    origin_turn_id=row.info.origin.run_id,
                    created_at=row.created_at,
                )
            )
        anchor_id = native_message_ids.get(self.assistant_message_id)
        if not isinstance(anchor_id, str) or not anchor_id:
            # Promotion aggregates many assistant origins; use the last mapped
            # revision origin only as a transaction anchor for path validation.
            anchor_id = revisions[-1].origin_message_id
        writer.append_canvas_batch(
            CanvasImportBatch(
                conversation_id=conversation_id,
                documents=tuple(documents),
                revisions=tuple(revisions),
            ),
            anchor_message_id=anchor_id,
            require_active_path=self.require_active_path,
        )


@dataclass(frozen=True, slots=True)
class CanvasSessionPromotionContribution(ConsoleExactNativeIdTransactionContribution):
    """Exact frozen set of committed temporary runs for one promotion lease."""

    session_id: str
    generation: int
    run_ids: tuple[str, ...]
    turn: CanvasTurnContribution = field(repr=False)
    _owner: CanvasSessionOwner = field(repr=False, compare=False)
    _lease: object = field(repr=False, compare=False)

    @property
    def revision_count(self) -> int:
        return self.turn.revision_count

    def write_exact(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        native_message_ids: Mapping[str, str],
    ) -> None:
        self.turn.write_exact(
            writer=writer,
            conversation_id=conversation_id,
            native_message_ids=native_message_ids,
        )


@dataclass(frozen=True, slots=True)
class CanvasRunSettlement:
    """Immutable handoff from run lifecycle to message settlement."""

    run_id: str
    assistant_message_id: str
    state: CanvasRunState
    cards: tuple[CanvasCardMetadata, ...]
    contribution: CanvasTurnContribution | None = field(repr=False)
    _run_owner: CanvasRunOwner = field(repr=False, compare=False)

    @property
    def metadata_json(self) -> str:
        return json.dumps(
            {"canvas_cards": [asdict(card) for card in self.cards]},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(slots=True)
class _RunStage:
    scope: CanvasScope
    assistant_message_id: str
    temporary: bool
    owner: CanvasSessionOwner | None
    run_owner: CanvasRunOwner
    state: CanvasRunState = CanvasRunState.OPEN
    revisions: list[_StagedRevision] = field(default_factory=list)
    by_revision_id: dict[str, _StagedRevision] = field(default_factory=dict)
    latest_by_canvas_id: dict[str, _StagedRevision] = field(default_factory=dict)
    replays: dict[
        tuple[str, str], tuple[str, CanvasMutationResult | CanvasConflictResult]
    ] = field(default_factory=dict)
    settlement: CanvasRunSettlement | None = None


class CanvasRunCoordinator:
    """Coordinator view permanently fenced to one opaque run owner."""

    __slots__ = ("_run_owner", "controller")

    def __init__(self, controller: ConsoleCanvasController, owner: CanvasRunOwner):
        self.controller = controller
        self._run_owner = owner

    def is_scope_current(self, scope: CanvasScope) -> bool:
        return self.controller.is_scope_current(scope, _run_owner=self._run_owner)

    def list_canvases(self, scope: CanvasScope) -> tuple[CanvasListItem, ...]:
        return self.controller.list_canvases(scope, _run_owner=self._run_owner)

    def read_canvas(self, scope: CanvasScope, canvas_id: str) -> CanvasReadResult:
        return self.controller.read_canvas(scope, canvas_id, _run_owner=self._run_owner)

    def create_canvas(self, scope: CanvasScope, **kwargs: Any) -> CanvasMutationResult:
        return self.controller.create_canvas(
            scope, _run_owner=self._run_owner, **kwargs
        )

    def update_canvas(
        self, scope: CanvasScope, **kwargs: Any
    ) -> CanvasMutationResult | CanvasConflictResult:
        return self.controller.update_canvas(
            scope, _run_owner=self._run_owner, **kwargs
        )

    def finish_assistant_run(
        self,
        assistant_message_id: str,
        *,
        actual_run_id: str,
        terminal_status: str,
    ) -> CanvasRunSettlement | None:
        return self.controller.finish_assistant_run(
            assistant_message_id,
            actual_run_id=actual_run_id,
            terminal_status=terminal_status,
            _run_owner=self._run_owner,
        )


class ConsoleCanvasController:
    """Coordinate exactly one mutation stage per registered assistant run."""

    def __init__(self, *, durable_service: object | None = None) -> None:
        self._durable_service = durable_service
        self._runs: dict[str, _RunStage] = {}
        self._assistant_runs: dict[str, str] = {}
        self._closed = False
        self._session_owners: dict[str, CanvasSessionOwner] = {}
        self._session_incarnations: dict[str, int] = {}
        self._promotion_leases: dict[str, object] = {}
        self._session_generations: dict[str, int] = {}
        self._lock = threading.RLock()

    def activate_session(self, session_id: str) -> CanvasSessionOwner:
        """Activate a fresh temporary incarnation and retire same-ID predecessors."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("canvas_runtime_closed")
            if session_id in self._promotion_leases:
                raise RuntimeError("canvas_promotion_in_flight")
            incarnation = self._session_incarnations.get(session_id, 0) + 1
            owner = CanvasSessionOwner(session_id, incarnation)
            self._session_incarnations[session_id] = incarnation
            self._session_owners[session_id] = owner
            for run_id, stage in tuple(self._runs.items()):
                if stage.scope.session_id == session_id:
                    self._release_assistant_run(run_id, stage)
                    self._runs.pop(run_id, None)
            self._session_generations[session_id] = (
                self._session_generations.get(session_id, 0) + 1
            )
            return owner

    def register_run(
        self,
        scope: CanvasScope,
        *,
        assistant_message_id: str,
        temporary: bool,
    ) -> CanvasRunCoordinator:
        """Register exact server-owned scope before any tool is advertised."""

        if not isinstance(scope, CanvasScope):
            raise TypeError("scope must be a CanvasScope")
        if not assistant_message_id:
            raise ValueError("assistant_message_id must be non-empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("canvas_runtime_closed")
            if temporary and scope.session_id not in self._session_owners:
                raise RuntimeError("canvas_session_not_active")
            if temporary and scope.session_id in self._promotion_leases:
                raise RuntimeError("canvas_promotion_in_flight")
            current = self._runs.get(scope.run_id)
            if current is not None:
                if (
                    current.scope != scope
                    or current.assistant_message_id != assistant_message_id
                    or current.temporary != temporary
                ):
                    raise RuntimeError("canvas_run_owner_changed")
                return CanvasRunCoordinator(self, current.run_owner)
            if assistant_message_id in self._assistant_runs:
                raise RuntimeError("canvas_assistant_owner_already_registered")
            run_owner = CanvasRunOwner(scope.run_id)
            self._runs[scope.run_id] = _RunStage(
                scope=scope,
                assistant_message_id=assistant_message_id,
                temporary=temporary,
                owner=self._session_owners.get(scope.session_id) if temporary else None,
                run_owner=run_owner,
            )
            self._assistant_runs[assistant_message_id] = scope.run_id
            return CanvasRunCoordinator(self, run_owner)

    def is_scope_current(
        self, scope: CanvasScope, *, _run_owner: CanvasRunOwner | None = None
    ) -> bool:
        with self._lock:
            stage = self._runs.get(scope.run_id)
            return bool(
                not self._closed
                and stage is not None
                and stage.state is CanvasRunState.OPEN
                and (_run_owner is None or stage.run_owner is _run_owner)
                and self._stage_owner_current(stage)
                and self._same_authority(stage.scope, scope)
            )

    def list_canvases(
        self, scope: CanvasScope, *, _run_owner: CanvasRunOwner | None = None
    ) -> tuple[CanvasListItem, ...]:
        with self._lock:
            stage = self._require_open(scope, _run_owner)
            committed = (
                ()
                if stage.temporary
                else self._service_call("list_canvases", scope, default=())
            )
            items = {item.canvas_id: item for item in committed}
            if stage.temporary:
                for canvas_id, row in self._temporary_latest_rows(scope).items():
                    item = self._list_item(row.info, scope)
                    if (
                        item.is_selected
                        and row.info.sequence
                        < self._temporary_max_sequence(scope, canvas_id)
                    ):
                        item = replace(item, is_historical_selection=True)
                    items[canvas_id] = item
            for canvas_id, row in stage.latest_by_canvas_id.items():
                items[canvas_id] = self._list_item(row.info, scope)
            return tuple(items[key] for key in sorted(items))

    def read_canvas(
        self,
        scope: CanvasScope,
        canvas_id: str,
        *,
        _run_owner: CanvasRunOwner | None = None,
    ) -> CanvasReadResult:
        with self._lock:
            stage = self._require_open(scope, _run_owner)
            row = stage.latest_by_canvas_id.get(canvas_id)
            if row is not None:
                return CanvasReadResult(revision=row.info, source=row.source)
            if stage.temporary:
                row = self._temporary_latest_rows(scope).get(canvas_id)
                if row is not None:
                    return CanvasReadResult(revision=row.info, source=row.source)
                raise RuntimeError("canvas_base_unavailable")
            return self._service_call("read_canvas", scope, canvas_id)

    def create_canvas(
        self,
        scope: CanvasScope,
        *,
        tool_call_id: str,
        title: str,
        html: str,
        _run_owner: CanvasRunOwner | None = None,
    ) -> CanvasMutationResult:
        with self._lock:
            stage = self._require_scope(scope, _run_owner)
            request_digest = sha256_utf8(f"create\0{title}\0{html}")
            replay = self._replay(stage, tool_call_id, request_digest)
            if replay is not None:
                return replay
            if stage.state is not CanvasRunState.OPEN:
                raise RuntimeError("canvas_scope_unavailable")
            plan = compile_canvas_document(html)
            now = datetime.now(UTC).isoformat()
            info = CanvasRevisionInfo(
                canvas_id=str(uuid4()),
                revision_id=str(uuid4()),
                parent_revision_id=None,
                title=title,
                runtime_profile="canvas-v1",
                content_sha256=plan.source_identity.sha256,
                source_bytes=plan.source_identity.source_bytes,
                sequence=1,
                origin=CanvasOrigin(stage.assistant_message_id, scope.run_id),
            )
            result = CanvasMutationResult(info, plan.compatibility_issues)
            self._append(stage, _StagedRevision(info, html, now, True))
            stage.replays[(scope.run_id, tool_call_id)] = (request_digest, result)
            return result

    def update_canvas(
        self,
        scope: CanvasScope,
        *,
        tool_call_id: str,
        canvas_id: str,
        expected_parent_revision_id: str,
        html: str,
        _run_owner: CanvasRunOwner | None = None,
    ) -> CanvasMutationResult | CanvasConflictResult:
        with self._lock:
            stage = self._require_scope(scope, _run_owner)
            request_digest = sha256_utf8(
                f"update\0{canvas_id}\0{expected_parent_revision_id}\0{html}"
            )
            replay = self._replay(stage, tool_call_id, request_digest)
            if replay is not None:
                return replay
            if stage.state is not CanvasRunState.OPEN:
                raise RuntimeError("canvas_scope_unavailable")
            latest = stage.latest_by_canvas_id.get(canvas_id)
            if latest is not None:
                if latest.info.revision_id != expected_parent_revision_id:
                    return self._cache_conflict(
                        stage,
                        tool_call_id,
                        request_digest,
                        latest.info,
                        "ambiguous_ancestry",
                    )
                parent = latest.info
            elif (
                other := self._other_uncommitted_latest(stage, canvas_id)
            ) is not None:
                return self._cache_conflict(
                    stage,
                    tool_call_id,
                    request_digest,
                    other.info,
                    "ambiguous_ancestry",
                )
            elif (
                stage.temporary
                and (latest := self._temporary_latest_rows(scope).get(canvas_id))
                is not None
            ):
                if latest.info.revision_id != expected_parent_revision_id:
                    return self._cache_conflict(
                        stage, tool_call_id, request_digest, latest.info, "stale_parent"
                    )
                parent = latest.info
            else:
                if stage.temporary:
                    raise RuntimeError("canvas_base_unavailable")
                read = self._service_call("read_canvas", scope, canvas_id)
                parent = read.revision
                if parent.revision_id != expected_parent_revision_id:
                    return self._cache_conflict(
                        stage, tool_call_id, request_digest, parent, "stale_parent"
                    )
            plan = compile_canvas_document(html)
            info = CanvasRevisionInfo(
                canvas_id=canvas_id,
                revision_id=str(uuid4()),
                parent_revision_id=parent.revision_id,
                title=parent.title,
                runtime_profile=parent.runtime_profile,
                content_sha256=plan.source_identity.sha256,
                source_bytes=plan.source_identity.source_bytes,
                sequence=self._next_sequence(stage, canvas_id, parent),
                origin=CanvasOrigin(stage.assistant_message_id, scope.run_id),
            )
            result = CanvasMutationResult(info, plan.compatibility_issues)
            self._append(
                stage,
                _StagedRevision(info, html, datetime.now(UTC).isoformat(), False),
            )
            stage.replays[(scope.run_id, tool_call_id)] = (request_digest, result)
            return result

    def finish_run(
        self, run_id: str, terminal_status: str
    ) -> CanvasRunSettlement | None:
        """Freeze success or discard failure once; duplicate callbacks are inert."""

        with self._lock:
            stage = self._runs.get(run_id)
            if stage is None:
                return None
            if stage.settlement is not None:
                return stage.settlement
            if not self._stage_owner_current(stage) or (
                stage.temporary and stage.scope.session_id in self._promotion_leases
            ):
                return None
            success = terminal_status == "done"
            state = CanvasRunState.READY if success else CanvasRunState.DISCARDED
            success_status = "temporary" if stage.temporary else "updated"
            cards = tuple(
                self._card(
                    row.info,
                    success_status if success else "discarded",
                    success,
                )
                for row in stage.revisions
            )
            contribution = (
                CanvasTurnContribution(
                    stage.assistant_message_id, tuple(stage.revisions)
                )
                if success and stage.revisions
                else None
            )
            stage.state = state
            if not success:
                stage.by_revision_id.clear()
                stage.latest_by_canvas_id.clear()
                stage.revisions.clear()
            stage.settlement = CanvasRunSettlement(
                run_id,
                stage.assistant_message_id,
                state,
                cards,
                contribution,
                stage.run_owner,
            )
            if not success:
                self._release_assistant_run(run_id, stage)
                self._prune_closed_stages()
            return stage.settlement

    def finish_assistant_run(
        self,
        assistant_message_id: str,
        *,
        actual_run_id: str,
        terminal_status: str,
        _run_owner: CanvasRunOwner | None = None,
    ) -> CanvasRunSettlement | None:
        """Settle the registered assistant only when the Agent run ID agrees."""

        with self._lock:
            registered_run_id = self._assistant_runs.get(assistant_message_id)
            if registered_run_id is None:
                return None
            stage = self._runs.get(registered_run_id)
            if (
                stage is None
                or (_run_owner is not None and stage.run_owner is not _run_owner)
                or not self._stage_owner_current(stage)
            ):
                return None
            if registered_run_id != actual_run_id:
                self.abort_settlement(registered_run_id, "run_identity_changed")
                return self._runs[registered_run_id].settlement
            return self.finish_run(registered_run_id, terminal_status)

    def resume_run(
        self, scope: CanvasScope, *, assistant_message_id: str
    ) -> CanvasRunSettlement | None:
        """Return existing settlement for exact continuation/re-entry."""

        with self._lock:
            stage = self._runs.get(scope.run_id)
            if (
                stage is None
                or stage.assistant_message_id != assistant_message_id
                or not self._same_authority(stage.scope, scope)
            ):
                return None
            return stage.settlement

    def settlement_for_assistant(
        self, assistant_message_id: str
    ) -> CanvasRunSettlement | None:
        with self._lock:
            run_id = self._assistant_runs.get(assistant_message_id)
            stage = self._runs.get(run_id or "")
            if stage is not None:
                return stage.settlement
            for retired in reversed(tuple(self._runs.values())):
                if retired.assistant_message_id == assistant_message_id:
                    return retired.settlement
            return None

    def confirm_settlement(self, run_id: str) -> bool:
        with self._lock:
            stage = self._runs.get(run_id)
            if (
                stage is None
                or stage.state is not CanvasRunState.READY
                or not self._stage_owner_current(stage)
                or (
                    stage.temporary and stage.scope.session_id in self._promotion_leases
                )
            ):
                return False
            stage.state = CanvasRunState.COMMITTED
            assert stage.settlement is not None
            stage.settlement = CanvasRunSettlement(
                run_id,
                stage.assistant_message_id,
                CanvasRunState.COMMITTED,
                stage.settlement.cards,
                None,
                stage.run_owner,
            )
            if stage.temporary:
                self._session_generations[stage.scope.session_id] = (
                    self._session_generations.get(stage.scope.session_id, 0) + 1
                )
            else:
                stage.by_revision_id.clear()
                stage.latest_by_canvas_id.clear()
                stage.revisions.clear()
            self._prune_closed_stages()
            return True

    def abort_settlement(self, run_id: str, error_code: str = "commit_failed") -> bool:
        with self._lock:
            stage = self._runs.get(run_id)
            if (
                stage is None
                or stage.state not in {CanvasRunState.OPEN, CanvasRunState.READY}
                or not self._stage_owner_current(stage)
                or (
                    stage.temporary and stage.scope.session_id in self._promotion_leases
                )
            ):
                return False
            rows = tuple(stage.revisions)
            cards = tuple(
                replace(
                    self._card(row.info, "discarded", False),
                    error_code=error_code[:64],
                )
                for row in rows
            )
            stage.state = CanvasRunState.DISCARDED
            stage.revisions.clear()
            stage.by_revision_id.clear()
            stage.latest_by_canvas_id.clear()
            stage.settlement = CanvasRunSettlement(
                run_id,
                stage.assistant_message_id,
                CanvasRunState.DISCARDED,
                cards,
                None,
                stage.run_owner,
            )
            self._release_assistant_run(run_id, stage)
            self._prune_closed_stages()
            return True

    def confirm_exact_settlement(self, settlement: CanvasRunSettlement) -> bool:
        """Commit only the stage represented by this immutable settlement."""

        with self._lock:
            stage = self._runs.get(settlement.run_id)
            if stage is None or stage.run_owner is not settlement._run_owner:
                return False
            if stage.state is CanvasRunState.COMMITTED:
                return True
            if stage.settlement is not settlement:
                return False
            return self.confirm_settlement(settlement.run_id)

    def abort_exact_settlement(
        self,
        settlement: CanvasRunSettlement,
        error_code: str = "commit_failed",
    ) -> bool:
        """Discard only the stage represented by this immutable settlement."""

        with self._lock:
            stage = self._runs.get(settlement.run_id)
            if (
                stage is None
                or stage.run_owner is not settlement._run_owner
                or stage.settlement is not settlement
            ):
                return False
            return self.abort_settlement(settlement.run_id, error_code)

    def discard_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._promotion_leases:
                return
            for run_id, stage in tuple(self._runs.items()):
                if stage.scope.session_id != session_id:
                    continue
                if stage.state in {CanvasRunState.OPEN, CanvasRunState.READY}:
                    self.abort_settlement(run_id, "session_closed")
                elif stage.state is CanvasRunState.COMMITTED and stage.temporary:
                    self._release_assistant_run(run_id, stage)
                    self._runs.pop(run_id, None)
            self._session_owners.pop(session_id, None)
            self._promotion_leases.pop(session_id, None)

    def discard_all(self) -> None:
        with self._lock:
            for run_id, stage in tuple(self._runs.items()):
                if stage.scope.session_id in self._promotion_leases:
                    continue
                if stage.state in {CanvasRunState.OPEN, CanvasRunState.READY}:
                    self.abort_settlement(run_id, "state_replaced")
                elif stage.state is CanvasRunState.COMMITTED and stage.temporary:
                    self._release_assistant_run(run_id, stage)
                    self._runs.pop(run_id, None)
            leased_sessions = set(self._promotion_leases)
            self._session_owners = {
                session_id: owner
                for session_id, owner in self._session_owners.items()
                if session_id in leased_sessions
            }

    def promotion_contribution(
        self, session_id: str
    ) -> CanvasSessionPromotionContribution | None:
        """Freeze committed temporary runs for the existing promotion transaction."""

        with self._lock:
            owner = self._session_owners.get(session_id)
            session_stages = tuple(
                stage
                for stage in self._runs.values()
                if stage.scope.session_id == session_id
                and stage.temporary
                and stage.owner is owner
            )
            if any(
                stage.state in {CanvasRunState.OPEN, CanvasRunState.READY}
                for stage in session_stages
            ):
                raise RuntimeError("canvas_turns_not_settled")
            stages = tuple(
                stage
                for stage in session_stages
                if stage.state is CanvasRunState.COMMITTED and stage.revisions
            )
            if owner is None or not stages:
                return None
            if session_id in self._promotion_leases:
                raise RuntimeError("canvas_promotion_in_flight")
            lease = object()
            self._promotion_leases[session_id] = lease
            revisions = tuple(row for stage in stages for row in stage.revisions)
            return CanvasSessionPromotionContribution(
                session_id=session_id,
                generation=self._session_generations.get(session_id, 0),
                run_ids=tuple(stage.scope.run_id for stage in stages),
                turn=CanvasTurnContribution(
                    "promotion", revisions, require_active_path=False
                ),
                _owner=owner,
                _lease=lease,
            )

    def confirm_contribution(
        self, session_id: str, contribution: CanvasSessionPromotionContribution
    ) -> bool:
        with self._lock:
            if not self._promotion_matches(session_id, contribution):
                return False
            for run_id in contribution.run_ids:
                stage = self._runs.get(run_id)
                if stage is None or stage.owner is not contribution._owner:
                    return False
            for run_id in contribution.run_ids:
                stage = self._runs.pop(run_id)
                self._release_assistant_run(run_id, stage)
            self._promotion_leases.pop(session_id, None)
            self._session_owners.pop(session_id, None)
            return True

    def abort_contribution(
        self, session_id: str, contribution: CanvasSessionPromotionContribution
    ) -> bool:
        with self._lock:
            if not self._promotion_matches(session_id, contribution):
                return False
            self._promotion_leases.pop(session_id, None)
            if self._closed:
                for run_id in contribution.run_ids:
                    stage = self._runs.pop(run_id, None)
                    if stage is not None:
                        self._release_assistant_run(run_id, stage)
                self._session_owners.pop(session_id, None)
            return True

    def retire_contribution(
        self, session_id: str, contribution: CanvasSessionPromotionContribution
    ) -> bool:
        return self.confirm_contribution(session_id, contribution)

    def _promotion_matches(
        self, session_id: str, contribution: CanvasSessionPromotionContribution
    ) -> bool:
        return bool(
            isinstance(contribution, CanvasSessionPromotionContribution)
            and contribution.session_id == session_id
            and self._session_owners.get(session_id) is contribution._owner
            and self._promotion_leases.get(session_id) is contribution._lease
            and self._session_generations.get(session_id, 0) == contribution.generation
            and tuple(
                run_id
                for run_id, stage in self._runs.items()
                if stage.scope.session_id == session_id
                and stage.temporary
                and stage.owner is contribution._owner
                and stage.state is CanvasRunState.COMMITTED
                and stage.revisions
            )
            == contribution.run_ids
        )

    def close_runtime(self) -> None:
        with self._lock:
            for run_id, stage in tuple(self._runs.items()):
                if stage.scope.session_id in self._promotion_leases:
                    continue
                self.abort_settlement(run_id, "runtime_closed")
                if stage.state is CanvasRunState.COMMITTED and stage.temporary:
                    self._release_assistant_run(run_id, stage)
                    self._runs.pop(run_id, None)
            self._closed = True
            leased_sessions = set(self._promotion_leases)
            self._session_owners = {
                session_id: owner
                for session_id, owner in self._session_owners.items()
                if session_id in leased_sessions
            }

    def run_revision_count(self, run_id: str) -> int:
        with self._lock:
            stage = self._runs.get(run_id)
            if stage is None:
                return 0
            if (
                stage.settlement is not None
                and stage.settlement.contribution is not None
            ):
                return stage.settlement.contribution.revision_count
            return len(stage.revisions)

    @staticmethod
    def _same_authority(left: CanvasScope, right: CanvasScope) -> bool:
        return (
            left.session_id,
            left.conversation_id,
            left.active_message_ids,
            left.run_id,
        ) == (
            right.session_id,
            right.conversation_id,
            right.active_message_ids,
            right.run_id,
        )

    def _require_open(
        self, scope: CanvasScope, run_owner: CanvasRunOwner | None = None
    ) -> _RunStage:
        if not self.is_scope_current(scope, _run_owner=run_owner):
            raise RuntimeError("canvas_scope_unavailable")
        return self._runs[scope.run_id]

    def _require_scope(
        self, scope: CanvasScope, run_owner: CanvasRunOwner | None = None
    ) -> _RunStage:
        stage = self._runs.get(scope.run_id)
        if (
            self._closed
            or stage is None
            or (run_owner is not None and stage.run_owner is not run_owner)
            or not self._stage_owner_current(stage)
            or (stage.temporary and stage.scope.session_id in self._promotion_leases)
            or not self._same_authority(stage.scope, scope)
        ):
            raise RuntimeError("canvas_scope_unavailable")
        return stage

    @staticmethod
    def _append(stage: _RunStage, row: _StagedRevision) -> None:
        stage.revisions.append(row)
        stage.by_revision_id[row.info.revision_id] = row
        stage.latest_by_canvas_id[row.info.canvas_id] = row

    @staticmethod
    def _replay(
        stage: _RunStage, tool_call_id: str, request_digest: str
    ) -> CanvasMutationResult | CanvasConflictResult | None:
        prior = stage.replays.get((stage.scope.run_id, tool_call_id))
        if prior is None:
            return None
        if prior[0] != request_digest:
            raise RuntimeError("canvas_idempotency_conflict")
        return prior[1]

    def _service_call(self, name: str, *args: object, default: object = None) -> Any:
        method = getattr(self._durable_service, name, None)
        if not callable(method):
            if default is not None:
                return default
            raise RuntimeError("canvas_base_unavailable")
        return method(*args)

    def _temporary_latest_rows(self, scope: CanvasScope) -> dict[str, _StagedRevision]:
        latest: dict[str, _StagedRevision] = {}
        path_positions = {
            message_id: index
            for index, message_id in enumerate(scope.active_message_ids)
        }
        for stage in self._runs.values():
            if (
                stage.scope.session_id != scope.session_id
                or not stage.temporary
                or stage.state is not CanvasRunState.COMMITTED
                or not self._stage_owner_current(stage)
            ):
                continue
            for canvas_id, row in stage.latest_by_canvas_id.items():
                if row.info.origin.message_id not in path_positions:
                    continue
                current = latest.get(canvas_id)
                if current is None or (
                    path_positions[row.info.origin.message_id],
                    row.info.sequence,
                ) > (
                    path_positions[current.info.origin.message_id],
                    current.info.sequence,
                ):
                    latest[canvas_id] = row
        if scope.selected_canvas_id and scope.selected_revision_id:
            for stage in self._runs.values():
                row = stage.by_revision_id.get(scope.selected_revision_id)
                if (
                    row is not None
                    and row.info.canvas_id == scope.selected_canvas_id
                    and row.info.origin.message_id in path_positions
                    and stage.temporary
                    and stage.state is CanvasRunState.COMMITTED
                    and self._stage_owner_current(stage)
                ):
                    latest[row.info.canvas_id] = row
                    break
        return latest

    def _stage_owner_current(self, stage: _RunStage) -> bool:
        return not stage.temporary or (
            stage.owner is not None
            and self._session_owners.get(stage.scope.session_id) is stage.owner
        )

    def _temporary_max_sequence(self, scope: CanvasScope, canvas_id: str) -> int:
        path = set(scope.active_message_ids)
        return max(
            (
                row.info.sequence
                for stage in self._runs.values()
                if stage.scope.session_id == scope.session_id
                and stage.temporary
                and stage.state is CanvasRunState.COMMITTED
                and self._stage_owner_current(stage)
                for row in stage.revisions
                if row.info.canvas_id == canvas_id
                and row.info.origin.message_id in path
            ),
            default=0,
        )

    def _other_uncommitted_latest(
        self, stage: _RunStage, canvas_id: str
    ) -> _StagedRevision | None:
        for other in self._runs.values():
            if (
                other is not stage
                and other.scope.session_id == stage.scope.session_id
                and other.scope.conversation_id == stage.scope.conversation_id
                and other.state in {CanvasRunState.OPEN, CanvasRunState.READY}
                and self._stage_owner_current(other)
                and canvas_id in other.latest_by_canvas_id
            ):
                return other.latest_by_canvas_id[canvas_id]
        return None

    def _next_sequence(
        self, stage: _RunStage, canvas_id: str, parent: CanvasRevisionInfo
    ) -> int:
        maximum = parent.sequence
        for candidate in stage.revisions:
            if candidate.info.canvas_id == canvas_id:
                maximum = max(maximum, candidate.info.sequence)
        if stage.temporary:
            for candidate_stage in self._runs.values():
                if (
                    candidate_stage.scope.session_id == stage.scope.session_id
                    and candidate_stage.temporary
                    and candidate_stage.state is CanvasRunState.COMMITTED
                    and self._stage_owner_current(candidate_stage)
                ):
                    for row in candidate_stage.revisions:
                        if row.info.canvas_id == canvas_id:
                            maximum = max(maximum, row.info.sequence)
        else:
            next_sequence = self._service_call(
                "next_revision_sequence", stage.scope, canvas_id, default=0
            )
            if type(next_sequence) is int and next_sequence > 0:
                maximum = max(maximum, next_sequence - 1)
            unselected = replace(
                stage.scope, selected_canvas_id=None, selected_revision_id=None
            )
            for item in self._service_call("list_canvases", unselected, default=()):
                if item.canvas_id == canvas_id:
                    maximum = max(maximum, item.sequence)
        return maximum + 1

    def _cache_conflict(
        self,
        stage: _RunStage,
        tool_call_id: str,
        request_digest: str,
        info: CanvasRevisionInfo,
        code: str,
    ) -> CanvasConflictResult:
        result = self._conflict(info, code)
        stage.replays[(stage.scope.run_id, tool_call_id)] = (request_digest, result)
        return result

    def _prune_closed_stages(self) -> None:
        closed = [
            (run_id, stage)
            for run_id, stage in self._runs.items()
            if stage.state is CanvasRunState.DISCARDED
            or (stage.state is CanvasRunState.COMMITTED and not stage.temporary)
        ]
        for run_id, stage in closed[:-_MAX_RETAINED_CLOSED_RUNS]:
            self._runs.pop(run_id, None)
            self._release_assistant_run(run_id, stage)

    def _release_assistant_run(self, run_id: str, stage: _RunStage) -> None:
        """Release assistant ownership only when it still names this exact run."""

        if self._assistant_runs.get(stage.assistant_message_id) == run_id:
            self._assistant_runs.pop(stage.assistant_message_id, None)

    @staticmethod
    def _list_item(info: CanvasRevisionInfo, scope: CanvasScope) -> CanvasListItem:
        return CanvasListItem(
            canvas_id=info.canvas_id,
            revision_id=info.revision_id,
            parent_revision_id=info.parent_revision_id,
            title=info.title,
            runtime_profile=info.runtime_profile,
            content_sha256=info.content_sha256,
            source_bytes=info.source_bytes,
            sequence=info.sequence,
            origin=info.origin,
            is_selected=scope.selected_canvas_id == info.canvas_id,
            is_historical_selection=False,
        )

    @staticmethod
    def _conflict(info: CanvasRevisionInfo, code: str) -> CanvasConflictResult:
        return CanvasConflictResult(
            code=code,
            canvas_id=info.canvas_id,
            current_revision_id=info.revision_id,
            content_sha256=info.content_sha256,
            title=info.title,
            sequence=info.sequence,
            origin=info.origin,
        )

    @staticmethod
    def _card(
        info: CanvasRevisionInfo, status: str, reopenable: bool
    ) -> CanvasCardMetadata:
        return CanvasCardMetadata(
            canvas_id=info.canvas_id,
            revision_id=info.revision_id,
            title=info.title,
            sequence=info.sequence,
            digest=info.content_sha256,
            status=status,
            origin=CanvasCardOriginMetadata(
                message_id=info.origin.message_id, run_id=info.origin.run_id
            ),
            reopenable=reopenable,
        )


__all__ = [
    "CanvasCardMetadata",
    "CanvasRunCoordinator",
    "CanvasRunOwner",
    "CanvasRunSettlement",
    "CanvasRunState",
    "CanvasSessionOwner",
    "CanvasSessionPromotionContribution",
    "CanvasTurnContribution",
    "ConsoleCanvasController",
]
