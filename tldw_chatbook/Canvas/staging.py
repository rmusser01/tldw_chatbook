"""Process-only temporary Canvas histories and atomic promotion snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionWriter,
)

from .compiler import compile_canvas_document
from .limits import (
    CanvasLimitError,
    CanvasRepositoryLimits,
    sha256_utf8,
    utf8_byte_length,
    validate_opaque_identifier,
    validate_utf8_text,
)
from .models import (
    CanvasMutationResult,
    CanvasOrigin,
    CanvasRenderPlan,
    CanvasRevisionInfo,
)

MAX_TEMPORARY_STAGED_SOURCE_BYTES_PER_SESSION = 8 * 1024 * 1024


class CanvasStagingError(RuntimeError):
    """Bounded staging failure that never includes generated source."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StagedCanvasRead:
    """Explicit exact read of one staged revision and its closed render plan."""

    revision: CanvasRevisionInfo
    source: str = field(repr=False)
    render_plan: CanvasRenderPlan = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class CanvasStagingOwner:
    """Process-only incarnation fence for one temporary Console session."""

    session_id: str
    incarnation: int
    _nonce: object = field(default_factory=object, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _StagedRevision:
    info: CanvasRevisionInfo
    source: str = field(repr=False)
    render_plan: CanvasRenderPlan = field(repr=False)
    actor_kind: str
    created_at: str


@dataclass(frozen=True, slots=True)
class _PromotionDocument:
    canvas_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class _PromotionRevision:
    info: CanvasRevisionInfo
    source: str = field(repr=False)
    actor_kind: str
    created_at: str


@dataclass(frozen=True, slots=True)
class CanvasPromotionContribution:
    """Frozen source-bearing graph used only inside one promotion attempt."""

    session_id: str
    generation: int
    documents: tuple[_PromotionDocument, ...]
    revisions: tuple[_PromotionRevision, ...] = field(repr=False)
    reopen_canvas_id: str | None
    _owner: CanvasStagingOwner = field(repr=False, compare=False)
    _lease: object = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "CanvasPromotionContribution("
            f"session_id={self.session_id!r}, generation={self.generation}, "
            f"document_count={len(self.documents)}, "
            f"revision_count={len(self.revisions)}, "
            f"reopen_canvas_id={self.reopen_canvas_id!r})"
        )

    def write_exact(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        native_message_ids: Mapping[str, str],
    ) -> None:
        """Insert the complete graph through the caller-owned transaction."""

        try:
            revision_rows: list[tuple[object, ...]] = []
            for revision in self.revisions:
                durable_origin = native_message_ids.get(revision.info.origin.message_id)
                if not isinstance(durable_origin, str) or not durable_origin:
                    raise ValueError("origin_mapping_missing")
                revision_rows.append(
                    (
                        revision.info.revision_id,
                        revision.info.canvas_id,
                        revision.info.parent_revision_id,
                        revision.info.sequence,
                        revision.info.title,
                        revision.info.runtime_profile,
                        revision.source,
                        revision.info.content_sha256,
                        revision.info.source_bytes,
                        revision.actor_kind,
                        durable_origin,
                        revision.info.origin.run_id,
                        revision.created_at,
                        None,
                    )
                )

            for document in self.documents:
                writer.execute(
                    "INSERT INTO canvas_documents("
                    "id, conversation_id, created_at, deleted, deleted_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        document.canvas_id,
                        conversation_id,
                        document.created_at,
                        0,
                        None,
                    ),
                )
            for row in revision_rows:
                writer.execute(
                    "INSERT INTO canvas_revisions("
                    "id, canvas_id, parent_revision_id, sequence, title, "
                    "runtime_profile, html, content_sha256, html_bytes, actor_kind, "
                    "origin_message_id, origin_turn_id, created_at, deleted_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            if self.reopen_canvas_id is not None:
                writer.execute(
                    "INSERT INTO canvas_conversation_hints("
                    "conversation_id, last_canvas_id, updated_at"
                    ") VALUES (?, ?, ?)",
                    (conversation_id, self.reopen_canvas_id, _utc_now()),
                )
        except Exception:  # noqa: BLE001 - sanitize every ordinary writer failure.
            raise CanvasStagingError("canvas_promotion_failed") from None


@dataclass(slots=True)
class _SessionState:
    documents: dict[str, str] = field(default_factory=dict)
    revisions: dict[str, _StagedRevision] = field(default_factory=dict)
    revision_ids_by_canvas: dict[str, list[str]] = field(default_factory=dict)
    idempotency: dict[
        tuple[str, str], tuple[tuple[object, ...], CanvasMutationResult]
    ] = field(default_factory=dict)
    source_bytes: int = 0
    selected_canvas_id: str | None = None
    generation: int = 0


class CanvasStagingStore:
    """Own bounded temporary Canvas source and render plans in process memory."""

    def __init__(
        self,
        *,
        compiler: Callable[[str], CanvasRenderPlan] = compile_canvas_document,
        repository_limits: CanvasRepositoryLimits | None = None,
        max_staged_source_bytes: int = MAX_TEMPORARY_STAGED_SOURCE_BYTES_PER_SESSION,
    ) -> None:
        if not callable(compiler):
            raise TypeError("compiler must be callable")
        self._compiler = compiler
        self._limits = repository_limits or CanvasRepositoryLimits()
        if not isinstance(self._limits, CanvasRepositoryLimits):
            raise TypeError("repository_limits must be CanvasRepositoryLimits")
        if type(max_staged_source_bytes) is not int or max_staged_source_bytes <= 0:
            raise ValueError("max_staged_source_bytes must be a positive integer")
        self._max_staged_source_bytes = min(
            max_staged_source_bytes,
            self._limits.max_source_bytes_per_conversation,
        )
        self._sessions: dict[str, _SessionState] = {}
        self._active_owners: dict[str, CanvasStagingOwner] = {}
        self._incarnations: dict[str, int] = {}
        self._promotion_leases: dict[str, object] = {}
        self._runtime_closed = False
        self._lock = RLock()

    def activate_session(self, session_id: str) -> CanvasStagingOwner:
        """Activate one server-owned temporary session incarnation."""

        self._id(session_id, "session_id")
        with self._lock:
            if self._runtime_closed:
                raise CanvasStagingError("runtime_closed")
            incarnation = self._incarnations.get(session_id, 0) + 1
            owner = CanvasStagingOwner(session_id, incarnation)
            self._incarnations[session_id] = incarnation
            self._active_owners[session_id] = owner
            self._sessions.pop(session_id, None)
            self._promotion_leases.pop(session_id, None)
            return owner

    def session_owner(self, session_id: str) -> CanvasStagingOwner:
        """Return the current process-only owner to trusted server code."""

        self._id(session_id, "session_id")
        with self._lock:
            if self._runtime_closed:
                raise CanvasStagingError("runtime_closed")
            owner = self._active_owners.get(session_id)
            if owner is None:
                raise CanvasStagingError("session_retired")
            return owner

    def create_canvas(
        self,
        *,
        owner: CanvasStagingOwner,
        run_id: str,
        tool_call_id: str,
        title: str,
        source: str,
        origin_message_id: str,
    ) -> CanvasMutationResult:
        """Compile and stage one temporary Canvas root revision."""

        self._require_owner_active(owner)
        self._id(run_id, "run_id")
        self._id(tool_call_id, "tool_call_id")
        title = self._title(title)
        self._source_bytes(source)
        self._id(origin_message_id, "origin_message_id")
        request = self._request("create", title, source, origin_message_id)
        with self._lock:
            state = self._mutable_state(owner, create=True)
            replay = self._replay(state, run_id, tool_call_id, request)
            if replay is not None:
                return replay
            if len(state.documents) >= self._limits.max_canvases_per_conversation:
                raise CanvasStagingError("canvas_count")
            plan = self._compile(source)
            source_bytes = plan.source_identity.source_bytes
            self._require_session_capacity(state, source_bytes)
            timestamp = _utc_now()
            canvas_id = str(uuid4())
            revision_id = str(uuid4())
            info = self._info(
                canvas_id=canvas_id,
                revision_id=revision_id,
                parent_revision_id=None,
                title=title,
                source=source,
                source_bytes=source_bytes,
                sequence=1,
                origin_message_id=origin_message_id,
                run_id=run_id,
            )
            result = CanvasMutationResult(
                revision=info,
                compatibility_issues=plan.compatibility_issues,
            )
            state.documents[canvas_id] = timestamp
            state.revisions[revision_id] = _StagedRevision(
                info, source, plan, "assistant", timestamp
            )
            state.revision_ids_by_canvas[canvas_id] = [revision_id]
            self._commit_mutation(
                state, run_id, tool_call_id, request, result, source_bytes, canvas_id
            )
            return result

    def update_canvas(
        self,
        *,
        owner: CanvasStagingOwner,
        run_id: str,
        tool_call_id: str,
        canvas_id: str,
        expected_parent_revision_id: str,
        source: str,
        origin_message_id: str,
    ) -> CanvasMutationResult:
        """Compile and append one complete replacement from an exact parent."""

        self._require_owner_active(owner)
        self._id(run_id, "run_id")
        self._id(tool_call_id, "tool_call_id")
        self._uuid(canvas_id, "canvas_id")
        self._uuid(expected_parent_revision_id, "revision_id")
        self._source_bytes(source)
        self._id(origin_message_id, "origin_message_id")
        request = self._request(
            "update",
            canvas_id,
            expected_parent_revision_id,
            source,
            origin_message_id,
        )
        with self._lock:
            state = self._mutable_state(owner)
            replay = self._replay(state, run_id, tool_call_id, request)
            if replay is not None:
                return replay
            parent = self._parent(state, canvas_id, expected_parent_revision_id)
            self._require_revision_capacity(state, canvas_id)
            plan = self._compile(source)
            source_bytes = plan.source_identity.source_bytes
            self._require_session_capacity(state, source_bytes)
            return self._append(
                state=state,
                run_id=run_id,
                tool_call_id=tool_call_id,
                request=request,
                parent=parent,
                title=parent.info.title,
                source=source,
                plan=plan,
                actor_kind="assistant",
                origin_message_id=origin_message_id,
            )

    def rename_canvas(
        self,
        *,
        owner: CanvasStagingOwner,
        run_id: str,
        tool_call_id: str,
        canvas_id: str,
        expected_parent_revision_id: str,
        title: str,
        origin_message_id: str,
    ) -> CanvasMutationResult:
        """Append a title-only revision from an exact selected parent."""

        self._require_owner_active(owner)
        self._id(run_id, "run_id")
        self._id(tool_call_id, "tool_call_id")
        self._uuid(canvas_id, "canvas_id")
        self._uuid(expected_parent_revision_id, "revision_id")
        title = self._title(title)
        self._id(origin_message_id, "origin_message_id")
        request = self._request(
            "rename",
            canvas_id,
            expected_parent_revision_id,
            title,
            origin_message_id,
        )
        with self._lock:
            state = self._mutable_state(owner)
            replay = self._replay(state, run_id, tool_call_id, request)
            if replay is not None:
                return replay
            parent = self._parent(state, canvas_id, expected_parent_revision_id)
            self._require_revision_capacity(state, canvas_id)
            self._require_session_capacity(state, parent.info.source_bytes)
            return self._append(
                state=state,
                run_id=run_id,
                tool_call_id=tool_call_id,
                request=request,
                parent=parent,
                title=title,
                source=parent.source,
                plan=parent.render_plan,
                actor_kind="user_rename",
                origin_message_id=origin_message_id,
            )

    def read_revision(self, *, session_id: str, revision_id: str) -> StagedCanvasRead:
        """Return one exact staged source only to an explicit trusted caller."""

        with self._lock:
            revision = self._sessions.get(session_id, _SessionState()).revisions.get(
                revision_id
            )
            if revision is None:
                raise CanvasStagingError("revision_not_found")
            return StagedCanvasRead(
                revision=revision.info,
                source=revision.source,
                render_plan=revision.render_plan,
            )

    def promotion_contribution(
        self, session_id: str
    ) -> CanvasPromotionContribution | None:
        """Freeze the exact graph currently owned by one temporary session."""

        with self._lock:
            if self._runtime_closed:
                raise CanvasStagingError("runtime_closed")
            owner = self._active_owners.get(session_id)
            state = self._sessions.get(session_id)
            if owner is None or state is None or not state.revisions:
                return None
            if session_id in self._promotion_leases:
                raise CanvasStagingError("promotion_in_flight")
            lease = object()
            self._promotion_leases[session_id] = lease
            documents = tuple(
                _PromotionDocument(canvas_id, created_at)
                for canvas_id, created_at in sorted(state.documents.items())
            )
            revisions = tuple(
                _PromotionRevision(
                    state.revisions[revision_id].info,
                    state.revisions[revision_id].source,
                    state.revisions[revision_id].actor_kind,
                    state.revisions[revision_id].created_at,
                )
                for canvas_id in sorted(state.revision_ids_by_canvas)
                for revision_id in state.revision_ids_by_canvas[canvas_id]
            )
            return CanvasPromotionContribution(
                session_id=session_id,
                generation=state.generation,
                documents=documents,
                revisions=revisions,
                reopen_canvas_id=state.selected_canvas_id,
                _owner=owner,
                _lease=lease,
            )

    def confirm_contribution(
        self, session_id: str, contribution: CanvasPromotionContribution
    ) -> bool:
        """Discard only a still-current snapshot after its transaction commits."""

        with self._lock:
            state = self._sessions.get(session_id)
            if (
                state is None
                or contribution.session_id != session_id
                or state.generation != contribution.generation
                or self._active_owners.get(session_id) is not contribution._owner
                or self._promotion_leases.get(session_id) is not contribution._lease
            ):
                return False
            self._sessions.pop(session_id, None)
            self._active_owners.pop(session_id, None)
            self._promotion_leases.pop(session_id, None)
            return True

    def abort_contribution(
        self, session_id: str, contribution: CanvasPromotionContribution
    ) -> bool:
        """Release only the exact failed promotion lease, retaining staged state."""

        with self._lock:
            if (
                contribution.session_id != session_id
                or self._active_owners.get(session_id) is not contribution._owner
                or self._promotion_leases.get(session_id) is not contribution._lease
            ):
                return False
            self._promotion_leases.pop(session_id, None)
            return True

    def discard_session(self, session_id: str) -> None:
        """Destroy all process-local Canvas state for one ended session."""

        with self._lock:
            self._sessions.pop(session_id, None)
            self._active_owners.pop(session_id, None)
            self._promotion_leases.pop(session_id, None)

    def discard_all(self) -> None:
        """Retire every current owner while leaving the runtime reusable."""

        with self._lock:
            self._sessions.clear()
            self._active_owners.clear()
            self._promotion_leases.clear()

    def close_runtime(self) -> None:
        """Permanently retire this staging runtime and every current owner."""

        with self._lock:
            self._runtime_closed = True
            self._sessions.clear()
            self._active_owners.clear()
            self._promotion_leases.clear()

    def staged_revision_count(self, session_id: str) -> int:
        """Return a content-free diagnostic count for lifecycle coordination."""

        with self._lock:
            state = self._sessions.get(session_id)
            return len(state.revisions) if state is not None else 0

    def _append(
        self,
        *,
        state: _SessionState,
        run_id: str,
        tool_call_id: str,
        request: tuple[object, ...],
        parent: _StagedRevision,
        title: str,
        source: str,
        plan: CanvasRenderPlan,
        actor_kind: str,
        origin_message_id: str,
    ) -> CanvasMutationResult:
        canvas_id = parent.info.canvas_id
        revision_id = str(uuid4())
        sequence = len(state.revision_ids_by_canvas[canvas_id]) + 1
        timestamp = _utc_now()
        info = self._info(
            canvas_id=canvas_id,
            revision_id=revision_id,
            parent_revision_id=parent.info.revision_id,
            title=title,
            source=source,
            source_bytes=plan.source_identity.source_bytes,
            sequence=sequence,
            origin_message_id=origin_message_id,
            run_id=run_id,
        )
        result = CanvasMutationResult(
            revision=info,
            compatibility_issues=plan.compatibility_issues,
        )
        state.revisions[revision_id] = _StagedRevision(
            info, source, plan, actor_kind, timestamp
        )
        state.revision_ids_by_canvas[canvas_id].append(revision_id)
        self._commit_mutation(
            state,
            run_id,
            tool_call_id,
            request,
            result,
            info.source_bytes,
            canvas_id,
        )
        return result

    @staticmethod
    def _request(operation: str, *values: object) -> tuple[object, ...]:
        safe_values = tuple(
            sha256_utf8(value) if isinstance(value, str) else value for value in values
        )
        return (operation, *safe_values)

    def _replay(
        self,
        state: _SessionState,
        run_id: str,
        tool_call_id: str,
        request: tuple[object, ...],
    ) -> CanvasMutationResult | None:
        self._id(run_id, "run_id")
        self._id(tool_call_id, "tool_call_id")
        prior = state.idempotency.get((run_id, tool_call_id))
        if prior is None:
            return None
        if prior[0] != request:
            raise CanvasStagingError("idempotency_conflict")
        return prior[1]

    def _parent(
        self, state: _SessionState, canvas_id: str, revision_id: str
    ) -> _StagedRevision:
        self._uuid(canvas_id, "canvas_id")
        self._uuid(revision_id, "revision_id")
        revision = state.revisions.get(revision_id)
        if revision is None or revision.info.canvas_id != canvas_id:
            raise CanvasStagingError("parent_not_found")
        return revision

    def _mutable_state(
        self, owner: CanvasStagingOwner, *, create: bool = False
    ) -> _SessionState:
        if self._runtime_closed:
            raise CanvasStagingError("runtime_closed")
        if (
            not isinstance(owner, CanvasStagingOwner)
            or self._active_owners.get(owner.session_id) is not owner
        ):
            raise CanvasStagingError("session_retired")
        if owner.session_id in self._promotion_leases:
            raise CanvasStagingError("promotion_in_flight")
        state = self._sessions.get(owner.session_id)
        if state is None:
            if not create:
                raise CanvasStagingError("session_empty")
            state = _SessionState()
            self._sessions[owner.session_id] = state
        return state

    def _require_owner_active(self, owner: CanvasStagingOwner) -> None:
        with self._lock:
            if self._runtime_closed:
                raise CanvasStagingError("runtime_closed")
            if (
                not isinstance(owner, CanvasStagingOwner)
                or self._active_owners.get(owner.session_id) is not owner
            ):
                raise CanvasStagingError("session_retired")
            if owner.session_id in self._promotion_leases:
                raise CanvasStagingError("promotion_in_flight")

    def _compile(self, source: str) -> CanvasRenderPlan:
        source_bytes = self._source_bytes(source)
        try:
            plan = self._compiler(source)
            if not isinstance(plan, CanvasRenderPlan):
                raise TypeError("invalid render plan")
            plan.source_identity.verify_source(source)
            if plan.source_identity.source_bytes != source_bytes:
                raise CanvasStagingError("source_identity_mismatch")
            if (
                plan.source_identity.source_bytes
                > self._limits.max_source_bytes_per_revision
            ):
                raise CanvasStagingError("revision_source_bytes")
            return plan
        except CanvasStagingError:
            raise
        except Exception:  # noqa: BLE001 - compiler failures are untrusted text.
            raise CanvasStagingError("compile_failed") from None

    def _title(self, title: str) -> str:
        if not isinstance(title, str) or not title.strip():
            raise CanvasStagingError("invalid_title")
        try:
            validate_utf8_text(
                title, limit=self._limits.max_title_bytes, field_name="title"
            )
        except CanvasLimitError:
            raise CanvasStagingError("title_bytes") from None
        return title

    def _source_bytes(self, source: str) -> int:
        try:
            source_bytes = utf8_byte_length(source)
        except CanvasLimitError:
            raise CanvasStagingError("invalid_source") from None
        if source_bytes > self._limits.max_source_bytes_per_revision:
            raise CanvasStagingError("revision_source_bytes")
        return source_bytes

    def _require_session_capacity(self, state: _SessionState, added: int) -> None:
        if state.source_bytes + added > self._max_staged_source_bytes:
            raise CanvasStagingError("session_source_bytes")

    def _require_revision_capacity(self, state: _SessionState, canvas_id: str) -> None:
        if (
            len(state.revision_ids_by_canvas[canvas_id])
            >= self._limits.max_revisions_per_canvas
        ):
            raise CanvasStagingError("revision_count")

    @staticmethod
    def _commit_mutation(
        state: _SessionState,
        run_id: str,
        tool_call_id: str,
        request: tuple[object, ...],
        result: CanvasMutationResult,
        source_bytes: int,
        canvas_id: str,
    ) -> None:
        state.source_bytes += source_bytes
        state.selected_canvas_id = canvas_id
        state.generation += 1
        state.idempotency[(run_id, tool_call_id)] = (request, result)

    @staticmethod
    def _info(
        *,
        canvas_id: str,
        revision_id: str,
        parent_revision_id: str | None,
        title: str,
        source: str,
        source_bytes: int,
        sequence: int,
        origin_message_id: str,
        run_id: str,
    ) -> CanvasRevisionInfo:
        CanvasStagingStore._id(origin_message_id, "origin_message_id")
        return CanvasRevisionInfo(
            canvas_id=canvas_id,
            revision_id=revision_id,
            parent_revision_id=parent_revision_id,
            title=title,
            runtime_profile="canvas-v1",
            content_sha256=sha256_utf8(source),
            source_bytes=source_bytes,
            sequence=sequence,
            origin=CanvasOrigin(message_id=origin_message_id, run_id=run_id),
        )

    @staticmethod
    def _id(value: str, field_name: str) -> str:
        try:
            return validate_opaque_identifier(value, field_name=field_name)
        except CanvasLimitError:
            raise CanvasStagingError(f"invalid_{field_name}") from None

    @staticmethod
    def _uuid(value: str, field_name: str) -> str:
        try:
            if str(UUID(value)) != value:
                raise ValueError
        except (TypeError, ValueError, AttributeError):
            raise CanvasStagingError(f"invalid_{field_name}") from None
        return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
