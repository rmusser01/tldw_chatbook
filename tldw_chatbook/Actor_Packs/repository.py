"""Portable Actor identity and bounded Persona intent persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

from .contracts import canonical_json_bytes


_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REASON = re.compile(r"[a-z0-9_]{1,64}\Z")
_OPERATIONS = frozenset({"create", "copy", "update"})
_STATES = frozenset({"prepared", "committed", "quarantined"})


class ActorPackRepositoryError(ValueError):
    """One fixed-category Actor Pack repository failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PortableActorIdentity:
    """Stable portable identity for one local Character or Persona."""

    actor_kind: str
    local_actor_id: str
    portable_uuid: str
    source_portable_uuid: str | None
    version: int


@dataclass(frozen=True, slots=True)
class PersonaActorPackIntent:
    """Bounded private coordinator state; profile JSON stays out of repr."""

    intent_id: str
    persona_id: str
    operation: str
    state: str
    old_profile_json: str | None = field(repr=False)
    new_profile_json: str = field(repr=False)
    old_profile_sha256: str | None
    new_profile_sha256: str
    old_store_sha256: str
    new_store_sha256: str
    old_registry_uuid: str | None
    new_registry_uuid: str
    quarantine_reason: str | None


class ActorPackRepository:
    """Persist portable identities and Persona coordination intents."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self.db = db
        self._uuid_factory = uuid_factory

    def get_identity(
        self, actor_kind: str, local_actor_id: object
    ) -> PortableActorIdentity | None:
        """Return one exact local identity or ``None``."""

        actor_id = _actor_id(actor_kind, local_actor_id)
        try:
            row = self.db.execute_query(
                """
                SELECT actor_kind, local_actor_id, portable_uuid,
                       source_portable_uuid, version
                  FROM actor_portable_identities
                 WHERE actor_kind = ? AND local_actor_id = ?
                """,
                (actor_kind, actor_id),
            ).fetchone()
            if row is None:
                return None
            try:
                return _decode_identity(row)
            except (
                ActorPackRepositoryError,
                TypeError,
                ValueError,
                UnicodeError,
                OverflowError,
            ):
                raise ActorPackRepositoryError(
                    "actor_pack_repository_corrupt"
                ) from None
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_read_failed"
            ) from None
        except (TypeError, ValueError, UnicodeError, OverflowError):
            raise ActorPackRepositoryError("actor_pack_repository_corrupt") from None

    def assign_identity(
        self,
        actor_kind: str,
        local_actor_id: object,
        *,
        source: str = "local",
        source_portable_uuid: str | None = None,
    ) -> PortableActorIdentity:
        """Assign one stable local UUID under an owned reserved transaction."""

        _actor_id(actor_kind, local_actor_id)
        if type(source) is not str or source != "local":
            raise ActorPackRepositoryError("actor_pack_source_not_local")
        provenance = (
            None
            if source_portable_uuid is None
            else _portable_uuid(source_portable_uuid)
        )
        self._require_no_transaction()
        try:
            with self.db.transaction(immediate=True):
                return self._assign_identity_in_transaction(
                    actor_kind,
                    local_actor_id,
                    source=source,
                    source_portable_uuid=provenance,
                )
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_write_failed"
            ) from None
        except (TypeError, ValueError, UnicodeError, OverflowError):
            raise ActorPackRepositoryError("actor_pack_identity_invalid") from None

    def prepare_persona_intent(
        self,
        *,
        persona_id: str,
        operation: str,
        old_profile: Mapping[str, Any] | None,
        new_profile: Mapping[str, Any],
        old_store_sha256: str,
        new_store_sha256: str,
        old_registry_uuid: str | None,
        new_registry_uuid: str,
        intent_id: str | None = None,
    ) -> PersonaActorPackIntent:
        """Durably write one prepared intent before Persona JSON changes."""

        values = _intent_input(
            persona_id=persona_id,
            operation=operation,
            old_profile=old_profile,
            new_profile=new_profile,
            old_store_sha256=old_store_sha256,
            new_store_sha256=new_store_sha256,
            old_registry_uuid=old_registry_uuid,
            new_registry_uuid=new_registry_uuid,
            intent_id=intent_id,
        )
        self._require_no_transaction()
        try:
            with self.db.transaction(immediate=True):
                self.db.execute_query(
                    """
                    INSERT INTO actor_pack_persona_intents(
                        intent_id, persona_id, operation, state,
                        old_profile_json, new_profile_json,
                        old_profile_sha256, new_profile_sha256,
                        old_store_sha256, new_store_sha256,
                        old_registry_uuid, new_registry_uuid
                    ) VALUES (?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                    redact_params=True,
                )
                return self._get_intent_in_transaction(values[0])
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_write_failed"
            ) from None

    def commit_persona_intent(
        self,
        intent_id: str,
        *,
        authority_guard: Callable[[], bool] | None = None,
    ) -> tuple[PortableActorIdentity, PersonaActorPackIntent]:
        """Atomically assign its Persona UUID and mark an intent committed."""

        intent_key = _intent_id(intent_id)
        self._require_no_transaction()
        try:
            with self.db.transaction(immediate=True):
                intent = self._get_intent_in_transaction(intent_key)
                if intent.state != "prepared":
                    raise ActorPackRepositoryError("actor_pack_intent_state_changed")
                if authority_guard is not None:
                    connection = self.db.get_connection()
                    before_changes = connection.total_changes
                    try:
                        allowed = authority_guard()
                    except Exception:
                        allowed = False
                    depth = getattr(self.db._local, "transaction_depth", 0)
                    if (
                        allowed is not True
                        or not connection.in_transaction
                        or depth != 1
                        or connection.total_changes != before_changes
                    ):
                        raise ActorPackRepositoryError(
                            "actor_pack_intent_state_changed"
                        )
                existing = self._get_identity_in_transaction(
                    "persona", intent.persona_id
                )
                if existing is None:
                    identity = self._insert_identity_in_transaction(
                        "persona",
                        intent.persona_id,
                        intent.new_registry_uuid,
                        intent.old_registry_uuid
                        if intent.operation == "copy"
                        else None,
                    )
                elif existing.portable_uuid == intent.new_registry_uuid:
                    identity = existing
                else:
                    raise ActorPackRepositoryError("actor_pack_intent_state_changed")
                changed = self.db.execute_query(
                    """
                    UPDATE actor_pack_persona_intents
                       SET state = 'committed', updated_at = CURRENT_TIMESTAMP
                     WHERE intent_id = ? AND state = 'prepared'
                    """,
                    (intent_key,),
                )
                if changed.rowcount != 1:
                    raise ActorPackRepositoryError("actor_pack_intent_state_changed")
                return identity, self._get_intent_in_transaction(intent_key)
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_write_failed"
            ) from None

    def list_persona_intents(self) -> tuple[PersonaActorPackIntent, ...]:
        """Return all bounded intents in deterministic recovery order."""

        try:
            rows = self.db.execute_query(
                """
                SELECT intent_id, persona_id, operation, state,
                       old_profile_json, new_profile_json,
                       old_profile_sha256, new_profile_sha256,
                       old_store_sha256, new_store_sha256,
                       old_registry_uuid, new_registry_uuid, quarantine_reason
                  FROM actor_pack_persona_intents
                 ORDER BY created_at, intent_id
                """
            ).fetchall()
            try:
                return tuple(_decode_intent(row) for row in rows)
            except (
                ActorPackRepositoryError,
                TypeError,
                ValueError,
                UnicodeError,
                OverflowError,
                json.JSONDecodeError,
            ):
                raise ActorPackRepositoryError(
                    "actor_pack_repository_corrupt"
                ) from None
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_read_failed"
            ) from None
        except (
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
            json.JSONDecodeError,
        ):
            raise ActorPackRepositoryError("actor_pack_repository_corrupt") from None

    def cleanup_persona_intent(self, intent_id: str, *, expected_state: str) -> None:
        """Delete exactly one converged intent under its expected state."""

        intent_key = _intent_id(intent_id)
        if expected_state not in _STATES:
            raise ActorPackRepositoryError("actor_pack_intent_invalid")
        self._require_no_transaction()
        try:
            with self.db.transaction(immediate=True):
                changed = self.db.execute_query(
                    "DELETE FROM actor_pack_persona_intents WHERE intent_id = ? AND state = ?",
                    (intent_key, expected_state),
                )
                if changed.rowcount != 1:
                    raise ActorPackRepositoryError("actor_pack_intent_state_changed")
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_write_failed"
            ) from None

    def quarantine_persona_intent(
        self, intent_id: str, reason: str
    ) -> PersonaActorPackIntent:
        """Fail closed by retaining one intent with a fixed reason category."""

        intent_key = _intent_id(intent_id)
        if type(reason) is not str or _REASON.fullmatch(reason) is None:
            raise ActorPackRepositoryError("actor_pack_intent_invalid")
        self._require_no_transaction()
        try:
            with self.db.transaction(immediate=True):
                current = self._get_intent_in_transaction(intent_key)
                if current.state == "quarantined":
                    if current.quarantine_reason != reason:
                        raise ActorPackRepositoryError(
                            "actor_pack_intent_state_changed"
                        )
                    return current
                changed = self.db.execute_query(
                    """
                    UPDATE actor_pack_persona_intents
                       SET state = 'quarantined', quarantine_reason = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE intent_id = ? AND state = ?
                    """,
                    (reason, intent_key, current.state),
                )
                if changed.rowcount != 1:
                    raise ActorPackRepositoryError("actor_pack_intent_state_changed")
                return self._get_intent_in_transaction(intent_key)
        except ActorPackRepositoryError:
            raise
        except (sqlite3.Error, CharactersRAGDBError):
            raise ActorPackRepositoryError(
                "actor_pack_repository_write_failed"
            ) from None

    def _get_identity_in_transaction(
        self, actor_kind: str, actor_id: str
    ) -> PortableActorIdentity | None:
        row = self.db.execute_query(
            """
            SELECT actor_kind, local_actor_id, portable_uuid,
                   source_portable_uuid, version
              FROM actor_portable_identities
             WHERE actor_kind = ? AND local_actor_id = ?
            """,
            (actor_kind, actor_id),
        ).fetchone()
        if row is None:
            return None
        try:
            return _decode_identity(row)
        except (
            ActorPackRepositoryError,
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
        ):
            raise ActorPackRepositoryError("actor_pack_repository_corrupt") from None

    def _assign_identity_in_transaction(
        self,
        actor_kind: str,
        local_actor_id: object,
        *,
        source: str = "local",
        source_portable_uuid: str | None = None,
        portable_uuid: str | None = None,
    ) -> PortableActorIdentity:
        """Package seam for an already-owned single SQLite transaction."""

        connection = self.db.get_connection()
        depth = getattr(self.db._local, "transaction_depth", 0)
        if not connection.in_transaction or depth != 1:
            raise ActorPackRepositoryError("actor_pack_repository_transaction_active")
        actor_id = _actor_id(actor_kind, local_actor_id)
        if type(source) is not str or source != "local":
            raise ActorPackRepositoryError("actor_pack_source_not_local")
        provenance = (
            None
            if source_portable_uuid is None
            else _portable_uuid(source_portable_uuid)
        )
        existing = self._get_identity_in_transaction(actor_kind, actor_id)
        if existing is not None:
            return existing
        assigned = (
            _generated_uuid(self._uuid_factory)
            if portable_uuid is None
            else _portable_uuid(portable_uuid)
        )
        if provenance == assigned:
            raise ActorPackRepositoryError("actor_pack_identity_invalid")
        return self._insert_identity_in_transaction(
            actor_kind,
            actor_id,
            assigned,
            provenance,
        )

    def _insert_identity_in_transaction(
        self,
        actor_kind: str,
        actor_id: str,
        portable_uuid: str,
        source_portable_uuid: str | None,
    ) -> PortableActorIdentity:
        self.db.execute_query(
            """
            INSERT INTO actor_portable_identities(
                actor_kind, local_actor_id, portable_uuid, source_portable_uuid
            ) VALUES (?, ?, ?, ?)
            """,
            (actor_kind, actor_id, portable_uuid, source_portable_uuid),
        )
        identity = self._get_identity_in_transaction(actor_kind, actor_id)
        if identity is None:
            raise ActorPackRepositoryError("actor_pack_repository_corrupt")
        return identity

    def _get_intent_in_transaction(self, intent_id: str) -> PersonaActorPackIntent:
        row = self.db.execute_query(
            """
            SELECT intent_id, persona_id, operation, state,
                   old_profile_json, new_profile_json,
                   old_profile_sha256, new_profile_sha256,
                   old_store_sha256, new_store_sha256,
                   old_registry_uuid, new_registry_uuid, quarantine_reason
              FROM actor_pack_persona_intents
             WHERE intent_id = ?
            """,
            (intent_id,),
            redact_params=True,
        ).fetchone()
        if row is None:
            raise ActorPackRepositoryError("actor_pack_intent_state_changed")
        try:
            return _decode_intent(row)
        except (
            ActorPackRepositoryError,
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
            json.JSONDecodeError,
        ):
            raise ActorPackRepositoryError("actor_pack_repository_corrupt") from None

    def _require_no_transaction(self) -> None:
        depth = getattr(self.db._local, "transaction_depth", 0)
        if self.db.get_connection().in_transaction or depth:
            raise ActorPackRepositoryError("actor_pack_repository_transaction_active")


def _actor_id(actor_kind: str, value: object) -> str:
    if actor_kind == "character":
        if type(value) is not int or value <= 0:
            raise ActorPackRepositoryError("actor_pack_identity_invalid")
        return str(value)
    if actor_kind == "persona":
        if type(value) is not str or not value or len(value) > 200 or "\x00" in value:
            raise ActorPackRepositoryError("actor_pack_identity_invalid")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ActorPackRepositoryError("actor_pack_identity_invalid") from None
        return value
    raise ActorPackRepositoryError("actor_pack_identity_invalid")


def _portable_uuid(value: object) -> str:
    if type(value) is not str or value != value.lower():
        raise ActorPackRepositoryError("actor_pack_identity_invalid")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError, TypeError):
        raise ActorPackRepositoryError("actor_pack_identity_invalid") from None
    if str(parsed) != value or parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ActorPackRepositoryError("actor_pack_identity_invalid")
    return value


def _generated_uuid(factory: Callable[[], uuid.UUID]) -> str:
    value = factory()
    if type(value) is not uuid.UUID:
        raise ActorPackRepositoryError("actor_pack_identity_invalid")
    return _portable_uuid(str(value))


def _intent_id(value: object) -> str:
    if type(value) is not str or _HEX32.fullmatch(value) is None:
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    return value


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    return value


def _profile_json(value: object) -> tuple[str, str]:
    if type(value) is not dict:
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    try:
        encoded = canonical_json_bytes(value)
    except ValueError:
        raise ActorPackRepositoryError("actor_pack_intent_invalid") from None
    if len(encoded) > 2 * 1024 * 1024:
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


def _intent_input(
    *,
    persona_id: str,
    operation: str,
    old_profile: Mapping[str, Any] | None,
    new_profile: Mapping[str, Any],
    old_store_sha256: str,
    new_store_sha256: str,
    old_registry_uuid: str | None,
    new_registry_uuid: str,
    intent_id: str | None,
) -> tuple[object, ...]:
    persona_key = _actor_id("persona", persona_id)
    if operation not in _OPERATIONS:
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    key = _intent_id(intent_id) if intent_id is not None else uuid.uuid4().hex
    new_json, new_sha = _profile_json(new_profile)
    if old_profile is None:
        old_json = old_sha = None
    else:
        old_json, old_sha = _profile_json(old_profile)
    old_store = _sha256(old_store_sha256)
    new_store = _sha256(new_store_sha256)
    old_uuid = None if old_registry_uuid is None else _portable_uuid(old_registry_uuid)
    new_uuid = _portable_uuid(new_registry_uuid)
    if operation == "create" and (old_json is not None or old_uuid is not None):
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    if operation == "copy" and (
        old_json is not None or old_uuid is None or old_uuid == new_uuid
    ):
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    if operation == "update" and (old_json is None or old_uuid != new_uuid):
        raise ActorPackRepositoryError("actor_pack_intent_invalid")
    return (
        key,
        persona_key,
        operation,
        old_json,
        new_json,
        old_sha,
        new_sha,
        old_store,
        new_store,
        old_uuid,
        new_uuid,
    )


def _decode_identity(row: sqlite3.Row) -> PortableActorIdentity:
    kind = row["actor_kind"]
    actor_id = row["local_actor_id"]
    if type(kind) is not str or type(actor_id) is not str:
        raise ValueError
    if kind == "character":
        if not actor_id.isascii() or not actor_id.isdigit() or int(actor_id) <= 0:
            raise ValueError
    else:
        _actor_id(kind, actor_id)
    portable = _portable_uuid(row["portable_uuid"])
    source = row["source_portable_uuid"]
    provenance = None if source is None else _portable_uuid(source)
    if provenance == portable:
        raise ValueError
    version = row["version"]
    if type(version) is not int or version <= 0:
        raise ValueError
    return PortableActorIdentity(kind, actor_id, portable, provenance, version)


def _decode_intent(row: sqlite3.Row) -> PersonaActorPackIntent:
    intent_id = _intent_id(row["intent_id"])
    persona_id = _actor_id("persona", row["persona_id"])
    operation = row["operation"]
    state = row["state"]
    if operation not in _OPERATIONS or state not in _STATES:
        raise ValueError
    old_json = row["old_profile_json"]
    new_json = row["new_profile_json"]
    if old_json is not None and type(old_json) is not str:
        raise ValueError
    if type(new_json) is not str:
        raise ValueError
    old_sha = row["old_profile_sha256"]
    expected_old_sha = None
    if old_json is not None:
        old_document = json.loads(old_json)
        canonical_old, expected_old_sha = _profile_json(old_document)
        if canonical_old != old_json:
            raise ValueError
    new_document = json.loads(new_json)
    canonical_new, expected_new_sha = _profile_json(new_document)
    if canonical_new != new_json:
        raise ValueError
    if old_sha != expected_old_sha or row["new_profile_sha256"] != expected_new_sha:
        raise ValueError
    old_store = _sha256(row["old_store_sha256"])
    new_store = _sha256(row["new_store_sha256"])
    old_uuid_value = row["old_registry_uuid"]
    old_uuid = None if old_uuid_value is None else _portable_uuid(old_uuid_value)
    new_uuid = _portable_uuid(row["new_registry_uuid"])
    reason_value = row["quarantine_reason"]
    if reason_value is not None and (
        type(reason_value) is not str or _REASON.fullmatch(reason_value) is None
    ):
        raise ValueError
    if state == "quarantined" and reason_value is None:
        raise ValueError
    if state != "quarantined" and reason_value is not None:
        raise ValueError
    if operation == "create" and (old_json is not None or old_uuid is not None):
        raise ValueError
    if operation == "copy" and (
        old_json is not None or old_uuid is None or old_uuid == new_uuid
    ):
        raise ValueError
    if operation == "update" and (old_json is None or old_uuid != new_uuid):
        raise ValueError
    return PersonaActorPackIntent(
        intent_id,
        persona_id,
        operation,
        state,
        old_json,
        new_json,
        old_sha,
        expected_new_sha,
        old_store,
        new_store,
        old_uuid,
        new_uuid,
        reason_value,
    )
