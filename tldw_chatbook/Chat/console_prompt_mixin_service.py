"""Operational mix-in contribution to existing Console transaction owners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from weakref import ref

from tldw_chatbook.Chat.console_prompt_mixins import MixinAttemptWrite, MixinForkState
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleExactNativeIdTransactionContribution,
    ConsoleTransactionWriter,
)

from .console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinPreferences,
    MixinSourceRef,
)


@dataclass(frozen=True)
class MixinAttemptContribution(ConsoleExactNativeIdTransactionContribution):
    """An immutable write plan; its frozen fields participate in acceptance identity."""

    batch: MixinAttemptWrite = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.batch) is not MixinAttemptWrite:
            raise ValueError("mixin_invalid_attempt")

    def durable_acceptance_fingerprint(self) -> str:
        """Hash the entire immutable batch without exposing body-bearing values."""
        import hashlib
        import json
        from dataclasses import asdict

        return hashlib.sha256(
            json.dumps(asdict(self.batch), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Use only explicitly supplied native identities, never role aliases."""
        self.write_exact(
            writer=writer,
            conversation_id=conversation_id,
            native_message_ids=message_ids,
        )

    def write_exact(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        native_message_ids: Mapping[str, str],
    ) -> None:
        """Append through the capability bound to the accepted conversation."""
        writer.append_mixin_attempt(
            batch=self.batch, native_message_ids=native_message_ids
        )


@dataclass(frozen=True)
class MixinHistoryContribution(ConsoleExactNativeIdTransactionContribution):
    """Promotion imports fork-owned history before appending fresh live attempts."""

    state: MixinForkState = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.state) is not MixinForkState:
            raise ValueError("mixin_invalid_fork_state")

    def durable_acceptance_fingerprint(self) -> str:
        import hashlib
        import json
        from dataclasses import asdict

        return hashlib.sha256(
            json.dumps(asdict(self.state), sort_keys=True).encode()
        ).hexdigest()

    def write(self, *, writer, conversation_id, message_ids) -> None:
        self.write_exact(
            writer=writer,
            conversation_id=conversation_id,
            native_message_ids=message_ids,
        )

    def write_exact(self, *, writer, conversation_id, native_message_ids) -> None:
        writer.append_mixin_history(
            state=self.state, native_message_ids=native_message_ids
        )


class ConsolePromptMixinService:
    """Chat-owned choices and exact-source admission; creates no worker owner."""

    def __init__(
        self,
        *,
        store: Any,
        prompt_scope_service: Any,
        persistence: Any,
        preparation_owner: Any = None,
        settings_durability_owner: Any = None,
    ) -> None:
        self.store = store
        self.prompt_scope_service = prompt_scope_service
        self.persistence = persistence
        self.preparation_owner = preparation_owner
        self.settings_durability_owner = settings_durability_owner
        self._origins: dict[str, tuple[Any, str, int]] = {}

    def _session(self, session_id: str):
        from .console_prompt_mixins import MixinPreferenceConflictError

        session = self.store._sessions.get(session_id)
        if session is None:
            raise MixinPreferenceConflictError("mixin_session_changed")
        origin = self._origins.setdefault(
            session_id,
            (
                ref(session),
                session.incarnation_id,
                session.conversation_binding_revision,
            ),
        )
        if (
            origin[0]() is not session
            or origin[1] != session.incarnation_id
            or origin[2] != session.conversation_binding_revision
        ):
            raise MixinPreferenceConflictError("mixin_session_changed")
        return session

    def preferences(self, session_id: str) -> MixinPreferences:
        """Return understood live choices; preserve unavailable formats."""
        from .console_prompt_mixins import MixinPreferenceConflictError

        preferences = self._session(session_id).prompt_mixin_preferences.preferences
        if preferences is None:
            raise MixinPreferenceConflictError("mixin_preferences_unavailable")
        return preferences

    def replace_preferences(
        self, session_id: str, *, expected_revision: int, value: MixinPreferences
    ) -> MixinPreferences:
        """Commit an exact live intent and join the existing settings save owner."""
        from .console_prompt_mixins import (
            MixinPreferenceConflictError,
            MixinSnapshotUnavailableError,
            MixinSourceChangedError,
        )

        preferences = self.preferences(session_id)
        if (
            type(value) is not MixinPreferences
            or type(expected_revision) is not int
            or preferences.revision != expected_revision
        ):
            raise MixinPreferenceConflictError("mixin_preference_conflict")
        prior = {row.source: row.enabled for row in preferences.entries}
        for row in value.entries:
            if row.enabled and not prior.get(row.source):
                try:
                    identity = self.prompt_scope_service.mixin_source_identity(
                        mode=row.source.backend
                    )
                except Exception:  # noqa: BLE001 - source uncertainty must refuse exact capture.
                    raise MixinSnapshotUnavailableError(
                        "mixin_source_unsupported"
                    ) from None
                if identity[0] != row.source.authority_key:
                    raise MixinSourceChangedError("mixin_source_changed")
        from .console_prompt_mixins import MixinPreferenceConflictError

        owner = self.settings_durability_owner
        session = self._session(session_id)
        needs_save = (
            not session.ephemeral and session.persisted_conversation_id is not None
        )
        lease = owner.try_acquire() if needs_save and owner is not None else None
        if needs_save and owner is not None and lease is None:
            raise MixinPreferenceConflictError("mixin_save_admission_closed")
        try:
            updated = self.store.replace_prompt_mixin_preferences(
                session_id,
                expected_revision=expected_revision,
                value=value,
                _origin=self._origins[session_id],
            )
            self.store.schedule_prompt_mixin_preferences_save(
                session_id, owner=owner, lease=lease
            )
        except BaseException:
            if lease is not None:
                lease.release()
            raise
        return updated

    async def enable_reference(
        self,
        session_id: str,
        *,
        source: MixinSourceRef,
        target_incarnation_id: str,
        expected_revision: int,
        is_current: Callable[[], bool] = lambda: True,
        expected_source_fences: tuple[object, object, object] | None = None,
    ) -> MixinPreferences:
        """Resolve one exact source, then enable via the existing live CAS."""
        import asyncio
        from dataclasses import replace

        from ..Prompt_Management.prompt_artifact_codec import decode_prompt_artifact
        from .console_prompt_mixins import (
            MixinPreferenceConflictError,
            MixinSelection,
            MixinSnapshot,
            MixinSnapshotUnavailableError,
            MixinSourceChangedError,
        )

        if (
            type(source) is not MixinSourceRef
            or type(expected_revision) is not int
            or expected_revision < 0
        ):
            raise MixinPreferenceConflictError("mixin_invalid_enable")
        current = self._capture_currentness(
            session_id, expected_revision=expected_revision
        )
        scope = self.prompt_scope_service
        session = self._session(session_id)
        if session.incarnation_id != target_incarnation_id or not is_current():
            raise MixinSourceChangedError("mixin_session_changed")
        try:
            identity = scope.mixin_source_identity(mode=source.backend)
            if expected_source_fences is not None and (
                type(expected_source_fences) is not tuple
                or len(expected_source_fences) != 3
                or any(
                    a is not b
                    for a, b in zip((scope, *identity[1:]), expected_source_fences)
                )
            ):
                raise MixinSourceChangedError("mixin_source_changed")
            if identity[0] != source.authority_key:
                raise MixinSourceChangedError("mixin_source_changed")
            async with asyncio.timeout(5.0):
                records = await scope.read_mixin_records(
                    mode=source.backend, record_uuids=(source.record_uuid,)
                )
            current()
            after = scope.mixin_source_identity(mode=source.backend)
            if after[0] != identity[0] or any(
                a is not b for a, b in zip(after[1:], identity[1:])
            ):
                raise MixinSourceChangedError("mixin_source_changed")
            if not is_current() or session.incarnation_id != target_incarnation_id:
                raise MixinSourceChangedError("mixin_session_changed")
            if len(records) != 1 or records[0]["uuid"] != source.record_uuid:
                raise ValueError("mixin_record_unavailable")
            record = records[0]
            decoded = decode_prompt_artifact(record)
            if decoded.state != "supported_v3":
                raise ValueError("mixin_definition_unavailable")
            snapshot = MixinSnapshot(
                source, record["version"], record["name"], decoded.definition
            )
        except (MixinPreferenceConflictError, MixinSourceChangedError):
            raise
        except Exception:  # noqa: BLE001 - source uncertainty must refuse enable.
            raise MixinSnapshotUnavailableError("mixin_source_unavailable") from None
        preferences = self.preferences(session_id)
        entries = list(preferences.entries)
        for index, entry in enumerate(entries):
            if entry.source == source:
                entries[index] = replace(entry, enabled=True)
                break
        else:
            selection = MixinSelection(
                source, True, snapshot.name, snapshot.definition.placement
            )
            index = max(
                (
                    i + 1
                    for i, row in enumerate(entries)
                    if row.placement == selection.placement
                ),
                default=(0 if selection.placement == "before" else len(entries)),
            )
            entries.insert(index, selection)
        if tuple(entries) == preferences.entries:
            return preferences
        return self.replace_preferences(
            session_id,
            expected_revision=expected_revision,
            value=replace(preferences, entries=tuple(entries)),
        )

    def _capture_currentness(self, session_id: str, *, expected_revision: int):
        """Retain body-free source/session fences until direct admission wins."""
        from .console_prompt_mixins import (
            MixinPreferenceConflictError,
            MixinSnapshotUnavailableError,
            MixinSourceChangedError,
        )

        store, scope = self.store, self.prompt_scope_service
        session = self._session(session_id)
        incarnation, binding = (
            session.incarnation_id,
            session.conversation_binding_revision,
        )
        preferences = self.preferences(session_id)
        enabled = tuple(row for row in preferences.entries if row.enabled)
        if preferences.revision != expected_revision:
            raise MixinPreferenceConflictError("mixin_preference_conflict")
        identities = {}
        try:
            for row in enabled:
                identity = scope.mixin_source_identity(mode=row.source.backend)
                if identity[0] != row.source.authority_key:
                    raise MixinSourceChangedError("mixin_source_changed")
                identities[row.source.backend] = identity
        except MixinSourceChangedError:
            raise
        except Exception:  # noqa: BLE001 - source uncertainty must refuse exact capture.
            raise MixinSnapshotUnavailableError("mixin_source_unavailable") from None

        owner_app = getattr(self.preparation_owner, "app", None)

        def current():
            if (
                self.store is not store
                or store._sessions.get(session_id) is not session
                or session.incarnation_id != incarnation
                or session.conversation_binding_revision != binding
            ):
                raise MixinSourceChangedError("mixin_session_changed")
            if self.preferences(session_id).revision != expected_revision:
                raise MixinPreferenceConflictError("mixin_preference_conflict")
            if (
                getattr(self.preparation_owner, "app", None) is not owner_app
                or self.prompt_scope_service is not scope
            ) or (
                owner_app is not None
                and getattr(owner_app, "prompt_scope_service", None) is not scope
            ):
                raise MixinSourceChangedError("mixin_source_changed")
            for backend, identity in identities.items():
                try:
                    now = scope.mixin_source_identity(mode=backend)
                except Exception:  # noqa: BLE001 - source uncertainty must refuse exact capture.
                    raise MixinSourceChangedError("mixin_source_changed") from None
                if now[0] != identity[0] or any(
                    a is not b for a, b in zip(now[1:], identity[1:])
                ):
                    raise MixinSourceChangedError("mixin_source_changed")

        current()
        return current

    async def capture(
        self, session_id: str, *, expected_revision: int, publish_facts: bool = True
    ) -> MixinAttemptSnapshot:
        """Freeze references before awaiting policy-checked coherent source reads."""
        import asyncio
        from functools import partial

        from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
            decode_prompt_artifact,
        )
        from tldw_chatbook.Prompt_Management.prompt_scope_service import (
            PromptScopeService,
        )

        from .console_preparation_reads import run_preparation_read
        from .console_prompt_mixins import (
            MixinAttemptSnapshot,
            MixinPreferenceConflictError,
            MixinPreferenceRead,
            MixinSnapshot,
            MixinSnapshotUnavailableError,
            MixinSourceChangedError,
            reconcile_mixin_placements,
        )

        current = self._capture_currentness(
            session_id, expected_revision=expected_revision
        )
        store, scope = self.store, self.prompt_scope_service
        session = self._session(session_id)
        preferences = self.preferences(session_id)
        enabled = tuple(row for row in preferences.entries if row.enabled)
        identities = dict.fromkeys(row.source.backend for row in enabled)

        snapshots = []
        try:
            async with asyncio.timeout(5.0):
                for backend in identities:
                    refs = tuple(
                        row.source for row in enabled if row.source.backend == backend
                    )
                    kwargs = {}
                    owner = self.preparation_owner
                    if isinstance(scope, PromptScopeService) and owner is not None:
                        kwargs["_native_runner"] = partial(
                            run_preparation_read,
                            creator=owner,
                            session_id=session_id,
                            reads=owner._preparation_reads,
                            require_current=current,
                        )
                    records = await scope.read_mixin_records(
                        mode=backend,
                        record_uuids=tuple(ref.record_uuid for ref in refs),
                        **kwargs,
                    )
                    current()
                    by_uuid = {record["uuid"]: record for record in records}
                    if len(by_uuid) != len(refs) or set(by_uuid) != {
                        ref.record_uuid for ref in refs
                    }:
                        raise ValueError("mixin_record_set_changed")
                    for ref in refs:
                        record = by_uuid[ref.record_uuid]
                        decoded = decode_prompt_artifact(record)
                        if decoded.state != "supported_v3":
                            raise ValueError("mixin_definition_unavailable")
                        snapshots.append(
                            MixinSnapshot(
                                ref,
                                record["version"],
                                record["name"],
                                decoded.definition,
                            )
                        )
        except (MixinPreferenceConflictError, MixinSourceChangedError):
            raise
        except Exception:  # noqa: BLE001 - source uncertainty must refuse exact capture.
            raise MixinSnapshotUnavailableError("mixin_snapshot_unavailable") from None
        current()
        reconciled = reconcile_mixin_placements(
            self.preferences(session_id), tuple(snapshots)
        )
        if publish_facts:
            session.prompt_mixin_preferences = MixinPreferenceRead(
                "supported", reconciled, None
            )
            if reconciled != preferences:
                store.schedule_prompt_mixin_preferences_save(
                    session_id, owner=self.settings_durability_owner
                )
        indexed = {item.source: item for item in snapshots}
        return MixinAttemptSnapshot(
            tuple(indexed[row.source] for row in reconciled.entries if row.enabled)
        )

    async def accept_attempt(self, session_id: str, batch: MixinAttemptWrite) -> None:
        """Retain the exact operational record before any provider can be entered."""
        import asyncio
        from functools import partial

        from tldw_chatbook.DB.base_db import run_owned_db_call

        from .console_prompt_mixins import MixinSnapshotUnavailableError

        session = self._session(session_id)
        binding = (
            session.incarnation_id,
            session.conversation_binding_revision,
            session.persisted_conversation_id,
        )
        persistence = self.persistence
        assistant = self.store.get_message(batch.assistant_message_id)
        user = self.store.get_message(batch.user_message_id)
        if not session.ephemeral and session.persisted_conversation_id is not None:
            if (
                persistence is None
                or not callable(
                    getattr(persistence, "save_conversation_mixin_attempt", None)
                )
                or getattr(persistence, "db", None) is None
            ):
                raise MixinSnapshotUnavailableError("mixin_persistence_unavailable")
            pending_assistant = (
                (
                    self.store.durable_parent_for_message(assistant.id),
                    assistant.metadata.to_json()
                    if assistant.metadata is not None
                    else None,
                )
                if assistant.persisted_message_id is None
                else None
            )
            callback = partial(
                self.persistence.save_conversation_mixin_attempt,
                session.persisted_conversation_id,
                batch=batch,
                pending_assistant=pending_assistant,
                native_message_ids={
                    user.id: user.persisted_message_id or user.id,
                    assistant.id: assistant.persisted_message_id or assistant.id,
                },
            )
            worker = asyncio.create_task(run_owned_db_call(persistence.db, callback))
            cancelled = False
            try:
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        cancelled = True
                worker.result()
            except Exception:  # noqa: BLE001 - uncertain persistence refuses dispatch.
                raise MixinSnapshotUnavailableError(
                    "mixin_attempt_save_failed"
                ) from None
        else:
            cancelled = False
        if (
            self.store._sessions.get(session_id) is not session
            or binding
            != (
                session.incarnation_id,
                session.conversation_binding_revision,
                session.persisted_conversation_id,
            )
            or self.store.persistence is not persistence
        ):
            raise MixinSnapshotUnavailableError("mixin_attempt_owner_changed")
        self.store.publish_prompt_mixin_attempt(session_id, batch)
        if cancelled:
            raise asyncio.CancelledError

    async def retry_snapshot(
        self, session_id: str, *, assistant_message_id: str
    ) -> MixinAttemptSnapshot:
        """Replay recorded instructions, mapping only genuinely legacy data empty."""
        import asyncio

        from .console_prompt_mixins import (
            MixinAttemptSnapshot,
            MixinSnapshotUnavailableError,
        )

        session = self._session(session_id)
        incarnation, binding = (
            session.incarnation_id,
            session.conversation_binding_revision,
        )
        message = next(
            (
                row
                for row in self.store.messages_for_session(session_id)
                if row.id == assistant_message_id
            ),
            None,
        )
        from .console_chat_models import ConsoleMessageRole

        if message is None or message.role is not ConsoleMessageRole.ASSISTANT:
            raise MixinSnapshotUnavailableError("mixin_retry_owner_unavailable")
        metadata = message.metadata
        marked = (
            metadata is not None and metadata.mixin_snapshot_schema_version is not None
        )
        if marked and metadata.mixin_attempt_id in session.prompt_mixin_attempts:
            batch = session.prompt_mixin_attempts[metadata.mixin_attempt_id]
            if batch.assistant_message_id != assistant_message_id:
                raise MixinSnapshotUnavailableError("mixin_retry_unavailable")
            return batch.snapshot
        if session.persisted_conversation_id is None:
            if (
                session.prompt_mixin_history is not None
                and assistant_message_id in dict(session.prompt_mixin_history.markers)
            ):
                replay = session.prompt_mixin_history.replay(assistant_message_id)
                if (
                    replay.status == "recorded"
                    and replay.snapshot is not None
                    and marked
                ):
                    marker = dict(session.prompt_mixin_history.markers).get(
                        assistant_message_id
                    )
                    import json

                    if (
                        json.loads(marker or "{}").get("mixin_attempt_id")
                        == metadata.mixin_attempt_id
                    ):
                        return replay.snapshot
                if replay.status != "legacy":
                    raise MixinSnapshotUnavailableError("mixin_retry_unavailable")
            if not marked:
                return MixinAttemptSnapshot()
            raise MixinSnapshotUnavailableError("mixin_retry_unavailable")
        if self.persistence is None:
            raise MixinSnapshotUnavailableError("mixin_retry_unavailable")
        from functools import partial

        from .console_preparation_reads import run_preparation_read

        persistence = self.persistence

        def current():
            if (
                self.store._sessions.get(session_id) is not session
                or session.incarnation_id != incarnation
                or session.conversation_binding_revision != binding
                or self.store.persistence is not persistence
            ):
                raise MixinSnapshotUnavailableError("mixin_retry_unavailable")

        callback = partial(
            persistence.get_conversation_mixin_attempt,
            session.persisted_conversation_id,
            attempt_id=metadata.mixin_attempt_id if marked else None,
            assistant_message_id=message.persisted_message_id or message.id,
        )
        try:
            async with asyncio.timeout(5.0):
                owner = self.preparation_owner
                if owner is None:
                    result = await asyncio.to_thread(callback)
                else:
                    result = await run_preparation_read(
                        callback,
                        creator=owner,
                        session_id=session_id,
                        reads=owner._preparation_reads,
                        require_current=current,
                    )
            current()
        except Exception:  # noqa: BLE001 - source uncertainty must refuse exact capture.
            raise MixinSnapshotUnavailableError("mixin_retry_unavailable") from None
        if result.status == "legacy" and not marked:
            return MixinAttemptSnapshot()
        if result.status != "recorded" or result.snapshot is None:
            raise MixinSnapshotUnavailableError("mixin_retry_unavailable")
        return result.snapshot


@dataclass(frozen=True)
class MixinPreferenceContribution:
    """Seed staged selections in a newly promoted conversation transaction."""

    preferences: MixinPreferences = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.preferences) is not MixinPreferences
            or self.preferences.revision < 1
        ):
            raise ValueError("mixin_invalid_preferences")

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        from .console_prompt_mixins import encode_mixin_preferences

        writer.execute(
            "INSERT INTO console_prompt_mixin_preferences(conversation_id,schema_version,revision,selection_json) VALUES(?,?,?,?)",
            (
                conversation_id,
                1,
                self.preferences.revision,
                encode_mixin_preferences(self.preferences),
            ),
        )
