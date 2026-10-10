"""Detached received inputs and exact session revision checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

if TYPE_CHECKING:
    from .attachment_core import PendingAttachment
    from .console_chat_store import ConsoleChatSession, ConsoleChatStore
    from .console_configuration_preparation import ConsoleTurnCaptureSelection
    from .console_session_settings import ConsoleSessionSettings


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleSessionInputSnapshot:
    """One live session's input/source witness; no presentation owner is retained."""

    session_id: str
    incarnation_id: str
    workspace_id: str
    settings_revision: int
    generation_settings_revision: int
    context_policy_revision: int
    identity_revision: int
    conversation_binding_revision: int
    ephemeral: bool
    draft: str = field(repr=False)
    draft_revision: int
    attachments: tuple[PendingAttachment, ...] = field(repr=False)
    attachment_revision: int
    one_shot_prefill: str | None = field(repr=False)
    prefill_revision: int
    _store_ref: ReferenceType[ConsoleChatStore] = field(repr=False)
    _session_ref: ReferenceType[ConsoleChatSession] = field(repr=False)
    _settings: ConsoleSessionSettings | None = field(repr=False)
    _attachment_ids: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConsoleReceivedTurnIntent:
    """Received selected values before complete configuration or input transfer."""

    turn_id: str
    session_id: str
    inputs: ConsoleSessionInputSnapshot = field(repr=False)
    selection: ConsoleTurnCaptureSelection = field(repr=False)
    staged_evidence_launch: Any | None = field(repr=False)
    staged_evidence_revision: int
    view_attachment_generation: int
    queue_revision: int | None = None
    _pressed_inputs: ConsoleSessionInputSnapshot | None = field(default=None, repr=False)
    _pressed_stash: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .console_configuration_preparation import ConsoleTurnCaptureSelection
        from .console_turn_preparation import _validate_identifier

        _validate_identifier(self.turn_id, "received turn ID")
        _validate_identifier(self.session_id, "received session ID")
        if type(self.inputs) is not ConsoleSessionInputSnapshot:
            raise TypeError("inputs must be ConsoleSessionInputSnapshot")
        if self._pressed_inputs is not None and self._pressed_inputs is not self.inputs:
            raise ValueError("Pressed inputs must be the original received snapshot.")
        if self.inputs.session_id != self.session_id:
            raise ValueError("Received inputs belong to another session.")
        if not isinstance(self.selection, ConsoleTurnCaptureSelection):
            raise TypeError("selection must be ConsoleTurnCaptureSelection")
        for name in (
            "staged_evidence_revision",
            "view_attachment_generation",
            "queue_revision",
        ):
            value = getattr(self, name)
            if name == "queue_revision" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _replace_session_draft_locked(
    session: ConsoleChatSession,
    draft: str,
    authored_token: tuple[int, int] | None = None,
) -> bool:
    """Update pure session input state while its existing admission lock is held."""
    changed = session.draft != draft
    authored = (
        authored_token is not None and session._draft_authored_token != authored_token
    )
    if changed or authored:
        session.draft_revision += 1
    session.draft = draft
    if authored_token is not None:
        session._draft_authored_token = authored_token
    if draft:
        session.has_user_work = True
    return changed


def _input_session_locked(
    store: ConsoleChatStore, snapshot: ConsoleSessionInputSnapshot
) -> ConsoleChatSession | None:
    if type(snapshot) is not ConsoleSessionInputSnapshot:
        return None
    session = store._sessions.get(snapshot.session_id)
    if (
        snapshot._store_ref() is not store
        or session is None
        or snapshot._session_ref() is not session
        or session.incarnation_id != snapshot.incarnation_id
        or session.conversation_binding_revision
        != snapshot.conversation_binding_revision
        or session.ephemeral is not snapshot.ephemeral
    ):
        return None
    return session


class ConsoleReceivedIntentInputMixin:
    """Input checks on the store's existing session objects and admission lock."""

    def _set_session_draft_inputs(
        self,
        session_id: str,
        draft: str,
        *,
        authored_token: tuple[int, int] | None = None,
    ) -> ConsoleChatSession:
        """Mirror draft edits before queued UI events; retain handoff write-through.

        Args:
            session_id: The live composer session.
            draft: Current authored text.
            authored_token: Optional composer generation/edit serial identity.

        Returns:
            The updated live session.

        Raises:
            KeyError: If the session is unknown.
            ValueError: If an authored token is malformed.
        """
        if authored_token is not None and (
            type(authored_token) is not tuple
            or len(authored_token) != 2
            or any(
                not isinstance(part, int) or isinstance(part, bool) or part < 0
                for part in authored_token
            )
        ):
            raise ValueError("authored_token must contain generation and edit serial")
        with self._preparation_lock:
            session = self._session_or_raise(session_id)
            authored_changed = (
                authored_token is not None
                and session._draft_authored_token != authored_token
            )
            _replace_session_draft_locked(session, draft, authored_token)
            authored_revision = session.draft_revision if authored_changed else None
        self._publish_session_draft_handoff(
            session, draft, authored_revision=authored_revision
        )
        return session

    def _publish_session_draft_handoff(
        self,
        session: ConsoleChatSession,
        draft: str,
        *,
        authored_revision: int | None = None,
    ) -> None:
        """Publish only the latest original session draft outside its input lock."""
        session_id = session.id
        with self._preparation_lock:
            pending = self._agent_handoff_writes.get(session_id)
            if (
                self._sessions.get(session_id) is not session
                or session.draft != draft
                or pending is None
                or session.agent_handoff_state != "pending"
                or (
                    authored_revision is not None
                    and session.draft_revision != authored_revision
                )
                or (pending["draft"] == draft and authored_revision is None)
            ):
                return
            pending["revision"] += 1
            pending["draft"] = draft
            session.agent_handoff_revision = pending["revision"]
        if self._agent_handoff_changed is not None:
            self._agent_handoff_changed(session_id, "draft_changed")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_agent_handoff_revision(pending)
        else:
            if pending["task"] is None or pending["task"].done():
                pending["task"] = loop.create_task(
                    self._drain_agent_handoff_writer(pending)
                )

    def session_input_snapshot(
        self: ConsoleChatStore, session_id: str
    ) -> ConsoleSessionInputSnapshot:
        """Capture in-memory input and source revisions without native work.

        Args:
            session_id: The original live session.

        Returns:
            Frozen input values and exact source witnesses.

        Raises:
            KeyError: If the session is unknown.
        """
        with self._preparation_lock:
            session = self._session_or_raise(session_id)
            attachments = tuple(session.pending_attachments)
            return ConsoleSessionInputSnapshot(
                session_id=session.id,
                incarnation_id=session.incarnation_id,
                workspace_id=session.workspace_id,
                settings_revision=session.settings_revision,
                generation_settings_revision=session.generation_settings_revision,
                context_policy_revision=session.context_policy_revision,
                identity_revision=session.identity_revision,
                conversation_binding_revision=session.conversation_binding_revision,
                ephemeral=session.ephemeral,
                draft=session.draft,
                draft_revision=session.draft_revision,
                attachments=attachments,
                attachment_revision=session.attachment_revision,
                one_shot_prefill=session.one_shot_prefill,
                prefill_revision=session.one_shot_prefill_revision,
                _store_ref=ref(self),
                _session_ref=ref(session),
                _settings=session.settings,
                _attachment_ids=tuple(item.attachment_id for item in attachments),
            )

    def session_inputs_are_current(
        self: ConsoleChatStore, snapshot: ConsoleSessionInputSnapshot,
        *, include_draft: bool = True,
    ) -> bool:
        """Check receipt-time source inputs before complete request promotion.

        Args:
            snapshot: The original received input snapshot.

        Returns:
            Whether the original live source and every captured input still match.
        """
        with self._preparation_lock:
            session = _input_session_locked(self, snapshot)
            if session is None:
                return False
            pending = session.pending_attachments
            return (
                session.workspace_id == snapshot.workspace_id
                and session.settings is snapshot._settings
                and session.settings_revision == snapshot.settings_revision
                and session.generation_settings_revision
                == snapshot.generation_settings_revision
                and session.context_policy_revision == snapshot.context_policy_revision
                and session.identity_revision == snapshot.identity_revision
                and (
                    not include_draft
                    or (session.draft_revision == snapshot.draft_revision
                        and session.draft == snapshot.draft)
                )
                and session.attachment_revision == snapshot.attachment_revision
                and len(pending) == len(snapshot.attachments)
                and all(
                    current is original and current.attachment_id == identifier
                    for current, original, identifier in zip(
                        pending, snapshot.attachments, snapshot._attachment_ids
                    )
                )
                and session.one_shot_prefill_revision == snapshot.prefill_revision
                and session.one_shot_prefill == snapshot.one_shot_prefill
            )

    def commit_session_input_draft(
        self: ConsoleChatStore, snapshot: ConsoleSessionInputSnapshot
    ) -> bool:
        """Clear only the exact accepted authored draft, after input transfer.

        Args:
            snapshot: The original received snapshot at domain acceptance.

        Returns:
            True on the exact draft CAS; False for changed source or newer edits.
        """
        with self._preparation_lock:
            session = _input_session_locked(self, snapshot)
            if (
                session is None
                or session.draft_revision != snapshot.draft_revision
                or session.draft != snapshot.draft
            ):
                return False
            changed = _replace_session_draft_locked(session, "")
            if not changed:
                session.draft_revision += 1
        # Original handoff effects stay outside the input/admission lock.
        self._publish_session_draft_handoff(session, "")
        return True
