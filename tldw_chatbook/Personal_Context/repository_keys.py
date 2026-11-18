"""Which keyring account holds a Personal Context profile's key (TASK-40133).

Schema v9 stores the account in ``profile_meta.key_account``; empty means the
legacy path-derived account. A key is used only after it authenticates the
database's manifest head (``_authenticate_keys``, kept in ``repository.py`` with
the other protected native routes), never by comparing ``key_version``: unrelated
keys usually share version 1. New and re-created profiles get a new, unused
account; link and Remove act only on the account the database names.
Migrations commit before any key is read, so a locked database is still
brought current; it is just never stamped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager, suppress

from .key_protector import (
    ProfileKeyMaterial,
    ProfileKeyMaterialInvalidError,
    ProfileLockedError,
    _serialize_material,
    decode_keyring_material,
)
from .runtime_policy import GLOBAL_POLICY_ID, AgentAuthority

# Any other stored value (a recovery marker, say) names no usable account.
KEY_ACCOUNT = re.compile(r"personal-context:[0-9a-z_]{1,128}")
# A removed row's record of what its Remove still owes: the account and a tag
# bound to the key Remove authenticated (never the key itself).
REMOVAL = re.compile(r"removal:(personal-context:[0-9a-z_]{1,128}):([0-9a-f]{64})")


def _removal_tag(keys: ProfileKeyMaterial, account: str) -> str:
    return hmac.new(
        _serialize_material(keys),
        b"personal-context-removal:" + account.encode(),
        hashlib.sha256,
    ).hexdigest()


# TASK-40136: a restore's marker names why it stays locked, across restarts.
RESTORE_MARKER = "recovery:setup_required:"
RESTORE_REASONS = frozenset(
    {
        "personal_context_key_not_in_backup",
        "personal_context_keyring_unavailable",
        "personal_context_key_install_unavailable",
    }
)
KEY_MISMATCH = "personal_context_key_mismatch"
# TASK-41121, owner decision 2026-11-15 ("no": damaged objects other than
# records must not block the restored-profile review). Each kind re-encryption
# removes when no key reads a version of it, with the local row that names that
# version (table, id column, version column). A damaged current manifest never
# reaches it (the key does not authenticate). Owner decisions 2026-11-16: a
# damaged current scope is rebuilt, and a dropped agent access setting is Off.
# Pointers stay a literal mapping of tuples: the semantic mutation inventory
# derives the dynamic SQL tables from it.
_LOCAL_POINTERS = {
    "outbox": ("encrypted_outbox", "outbox_id", "envelope_version"),
    "undo": ("local_undo", "undo_id", "encrypted_undo_version"),
    "record_link": ("local_record_links", "record_id", "encrypted_link_version"),
    "scope_binding": ("local_scope_bindings", "scope_id", "encrypted_binding_version"),
    "runtime_policy": ("local_runtime_policy", "scope_id", "encrypted_policy_version"),
}
_DROPPABLE = frozenset(
    ("record", "proposal", "manifest", "scope", *_LOCAL_POINTERS.keys())
)
# The review counts an older (not current) version of these kinds apart.
_HISTORY = frozenset({"manifest", "scope", "scope_binding", "runtime_policy"})
REBUILT_SCOPE_RECORDS = "rebuilt_scope_records"
# Fix round 3 (coordinator ruling): damaged scopes whose kind no evidence
# settles, more than one with no known global scope, refuse; never guessed.
AMBIGUOUS_SCOPES = "personal_context_damaged_scopes_ambiguous"
_REVIEW_ONLY = (REBUILT_SCOPE_RECORDS, AMBIGUOUS_SCOPES)
RECOVERED_SCOPE_LABEL = "Recovered scope"


def _key(row) -> tuple[str, str, str]:
    return (row["object_type"], row["object_id"], row["version_id"])


def _is_current(connection, row) -> bool:
    """Whether a version (a row, or an ``(type, id, version)`` key) is its head."""
    key = row if isinstance(row, tuple) else _key(row)
    return (
        connection.execute(
            "SELECT 1 FROM object_heads WHERE object_type = ? "
            "AND object_id = ? AND version_id = ?",
            key,
        ).fetchone()
        is not None
    )


def _locked(message: str, reason: str) -> ProfileLockedError:
    error = ProfileLockedError(message)
    error.reason_code = reason
    return error


class RepositoryKeyCustody:
    """Key-account lookup, provisioning and removal for the repository."""

    def _named_key_account(self, stored: object) -> str:
        """Return the keyring account a live profile row names, or lock."""

        if stored == "":
            return self._profile_ref
        if isinstance(stored, str) and KEY_ACCOUNT.fullmatch(stored):
            return stored
        if isinstance(stored, str) and stored.startswith(RESTORE_MARKER):
            reason = stored.removeprefix(RESTORE_MARKER)
            if reason in RESTORE_REASONS:
                raise _locked("Restored profile key is not set up.", reason)
        raise ProfileLockedError("Profile key account is invalid.")

    def _or_marker(self, keyring_protector):
        """The default keyring protector, or the reason a restore left.

        Without a secure keyring (an isolated restored copy always runs a null
        one) the protector refuses to construct. A restore's marker names the
        real reason, so it is read first, read-only, without creating the
        database (TASK-40136); anything else keeps the keyring's refusal.
        """
        try:
            return keyring_protector()
        except ProfileLockedError as refusal:
            from .recovery import stored_restore_marker

            stored = stored_restore_marker(self.db_path)
            if stored is not None:
                self._named_key_account(stored)  # raises the marker's reason
            raise refusal

    def _account_on(self, connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT key_account FROM profile_meta WHERE singleton = 1"
        ).fetchone()
        return self._named_key_account(None if row is None else row["key_account"])

    def _authenticated_account(
        self, connection: sqlite3.Connection, keys: ProfileKeyMaterial | None
    ) -> str:
        """The account the row names now, only if it holds exactly these keys.

        The caller's ``keys`` already authenticated this transaction's manifest;
        a reference rewritten after open (another database's account) refuses
        before link or Remove touches that entry.
        """

        account = self._account_on(connection)
        if keys is None or self._protector.load(account) != keys:
            raise ProfileLockedError("Profile key account changed.")
        return account

    def _names_account(self, account: str) -> bool:
        with closing(self._connect()) as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM profile_meta WHERE key_account = ?", (account,)
                ).fetchone()
                is not None
            )

    def _open_keys(self, connection, recovery_integrity_key):
        """Load the named key at open; adopt it only once it authenticates.

        Returns:
            ``(account, protected_keys, keys)``; keys combine the recovery
            integrity key when an interrupted link is being activated.
        """

        meta = connection.execute(
            "SELECT destroyed, key_account FROM profile_meta WHERE singleton = 1"
        ).fetchone()
        account = protected = None
        if meta is not None and not bool(meta["destroyed"]):
            account = self._named_key_account(meta["key_account"])
            protected = self._protector.load(account)
        if recovery_integrity_key is None:
            if protected is not None:
                from .repository import ProfileIntegrityError

                try:
                    self._authenticate_keys(connection, protected)
                except ProfileLockedError as exc:
                    # The manifest head failed AEAD/HMAC under the stored key:
                    # that key belongs to another copy (or the head was
                    # tampered with, which fails the same way). TASK-40136.
                    if isinstance(exc.__context__, ProfileIntegrityError):
                        raise _locked("Profile key does not match.", KEY_MISMATCH) from None
                    raise
                if meta["key_account"] == "":
                    connection.execute(
                        "UPDATE profile_meta SET key_account = ? "
                        "WHERE singleton = 1 AND key_account = ''",
                        (account,),
                    )
            return account, protected, protected
        if protected is None:
            raise ProfileLockedError("Destroyed profile keys cannot be recovered.")
        versions = {
            int(row["key_version"])
            for row in connection.execute(
                "SELECT DISTINCT key_version FROM encrypted_objects"
            )
        }
        if len(versions) != 1:
            raise ProfileLockedError("Interrupted profile link rebaseline is incomplete.")
        return (
            account,
            protected,
            ProfileKeyMaterial(
                encryption_key=protected.encryption_key,
                integrity_key=recovery_integrity_key,
                key_version=versions.pop(),
            ),
        )

    def _adopt_named_keys(self) -> None:
        """Let a handle opened before its profile existed read it once it does.

        Read-only: the key is adopted only after it authenticates the manifest
        head, and stamping stays with a fresh open.
        """

        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            try:
                meta = connection.execute(
                    "SELECT destroyed FROM profile_meta WHERE singleton = 1"
                ).fetchone()
                if meta is None or bool(meta["destroyed"]):
                    return
                account = self._account_on(connection)
                self._authenticate_keys(connection, self._protector.load(account))
            finally:
                connection.rollback()

    @contextmanager
    def _new_profile_key(self) -> Iterator[tuple[str, ProfileKeyMaterial]]:
        """Write fresh material to a new, unused account; on failure delete only it."""

        account = "personal-context:profile_" + secrets.token_hex(16)
        keys = self._protector.create(account)
        previous = self._keys
        try:
            yield account, keys
        except BaseException:
            # A failure after commit must not delete the key the row now names;
            # if that cannot be read, keep the key (an orphan entry is harmless).
            try:
                named = self._names_account(account)
            except Exception:  # noqa: BLE001
                named = True
            self._keys = keys if named else previous
            if not named:
                with suppress(Exception):
                    self._protector.delete(account)
            raise

    def damaged_versions(self) -> dict[str, int]:
        """How many stored versions no key can read, by kind (TASK-40140,
        TASK-41121). The review shows these counts, and re-encryption removes
        them. Read-only.
        """

        self._require_keys()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            try:
                rows = connection.execute("SELECT * FROM encrypted_objects").fetchall()
                damaged, scopes = [], {}
                for row in rows:
                    plaintext = self._reads(connection, row)
                    if plaintext is None:
                        damaged.append((self._damage_kind(connection, row), _key(row)))
                    elif row["object_type"] == "scope":
                        scopes[_key(row)[1:]] = plaintext
                counts = Counter(kind for kind, _ in damaged)
                rebuilt = {key[1] for kind, key in damaged if kind == "scope"}
                if rebuilt:
                    counts[REBUILT_SCOPE_RECORDS] = len(
                        self._scope_records(connection, rebuilt)
                        - {key for _, key in damaged}
                    )
                    _, ambiguous = self._scope_kinds(connection, rebuilt, scopes)
                    if ambiguous:
                        counts[AMBIGUOUS_SCOPES] = len(ambiguous)
                return dict(counts)
            finally:
                connection.rollback()

    def _reads(self, connection, row) -> bytes | None:
        """The row's plaintext, or None when it fails authentication and its
        kind can go (``_DROPPABLE``). A damaged current manifest, or an unknown
        kind, still refuses."""
        from .repository import ProfileIntegrityError

        try:
            return self._decrypt_row_authenticated(row)
        except ProfileIntegrityError:
            kind = row["object_type"]
            if kind not in _DROPPABLE or (
                kind == "manifest" and _is_current(connection, row)
            ):
                raise
            return None

    @staticmethod
    def _damage_kind(connection, row) -> str:
        """What the review calls a damaged row: ``<kind>_history`` for an
        older version of a kind that keeps history (review r15pc-rev1)."""
        kind = row["object_type"]
        if kind == "runtime_policy" and row["object_id"] == GLOBAL_POLICY_ID:
            kind = "runtime_switch"  # the global on/off setting
        if (kind in _HISTORY or kind == "runtime_switch") and not _is_current(
            connection, row
        ):
            return kind + "_history"
        return kind

    @staticmethod
    def _scope_records(connection, scope_ids) -> set[tuple[str, str, str]]:
        """The current record versions in these scopes."""
        rows = connection.execute(
            "SELECT o.object_type, o.object_id, o.version_id, o.scope_id "
            "FROM object_heads h JOIN encrypted_objects o "
            "USING (object_type, object_id, version_id) WHERE h.object_type = 'record'"
        ).fetchall()
        return {tuple(row)[:3] for row in rows if row["scope_id"] in scope_ids}

    def reencrypt_profile(
        self,
        *,
        keep_old_account: bool,
        expected_damaged: Mapping[str, int] | None = None,
        on_kept: Callable[[str], None] | None = None,
    ) -> str:
        """Re-encrypt every row under a fresh encryption key (TASK-40140).

        The integrity key and link state stay; ``key_version`` goes up. The new
        key goes into a new, unused account first; one transaction then rewrites
        every row and names that account, so an interruption leaves either the
        old rows naming the old key or the new rows naming the new one. Versions
        no key can read are removed in that same transaction, but only as many of
        each kind as the review showed (``expected_damaged``; owner decisions
        2026-11-14 and 2026-11-15). The old account is deleted only after the new
        one opens the database, and only when the caller knows no retained
        restore copy can name it; a kept one is passed to ``on_kept``.

        Returns:
            The new account.
        """

        from .crypto import EnvelopeCipher

        old = self._require_keys()
        if old.key_version >= 255:
            raise ProfileLockedError("Profile key version is exhausted.")
        new = ProfileKeyMaterial(
            secrets.token_bytes(32), old.integrity_key, old.key_version + 1
        )
        with self._transaction() as connection:
            old_account = self._reencryption_source(connection, old)
        account = "personal-context:profile_" + secrets.token_hex(16)
        self._protector.install(account, new)
        try:
            with self._transaction() as connection:
                if self._reencryption_source(connection, old) != old_account:
                    raise ProfileLockedError("Profile key account changed.")
                cipher = EnvelopeCipher(new.encryption_key, key_version=new.key_version)
                rows = connection.execute(
                    "SELECT * FROM encrypted_objects"
                ).fetchall()
                damaged, kinds, scopes = [], Counter(), {}
                for row in rows:
                    key = _key(row)
                    plaintext = self._reads(connection, row)
                    if plaintext is None:
                        damaged.append(key)
                        kinds[self._damage_kind(connection, row)] += 1
                        continue
                    if key[0] == "scope":
                        scopes[key[1:]] = plaintext
                    sealed = cipher.encrypt(plaintext, self._aad(*key))
                    connection.execute(
                        "UPDATE encrypted_objects SET algorithm = ?, nonce = ?, "
                        "ciphertext = ?, wrapped_dek = ?, key_version = ? "
                        "WHERE object_type = ? AND object_id = ? AND version_id = ?",
                        (
                            sealed.algorithm,
                            sealed.nonce,
                            sealed.ciphertext,
                            sealed.wrap_nonce + sealed.wrapped_dek,
                            sealed.key_version,
                            *key,
                        ),
                    )
                rebuilt, ambiguous = self._scope_kinds(
                    connection,
                    {
                        key[1]
                        for key in damaged
                        if key[0] == "scope" and _is_current(connection, key)
                    },
                    scopes,
                )
                if ambiguous:
                    raise _locked("Damaged scopes cannot be told apart.", AMBIGUOUS_SCOPES)
                expected = Counter(expected_damaged or {})
                for review_only in _REVIEW_ONLY:
                    expected.pop(review_only, None)
                if kinds != expected:
                    raise ProfileLockedError("Damaged objects changed since review.")
                off = [
                    key[1]
                    for key in damaged
                    if key[0] == "runtime_policy"
                    and key[1] != GLOBAL_POLICY_ID
                    and _is_current(connection, key)
                ]
                self._remove_damaged(connection, damaged)
                connection.execute(
                    "UPDATE profile_meta SET key_account = ? WHERE singleton = 1",
                    (account,),
                )
                self._authenticate_keys(connection, new)
                # Written under the new key, in the same transaction.
                self._rebuild_scopes(connection, rebuilt)
                for scope_id in off:
                    self._put_local_body(
                        connection,
                        "runtime_policy",
                        scope_id,
                        {"version": 1, "authority": AgentAuthority.OFF.value},
                    )
        except BaseException:
            self._keys = old
            with suppress(Exception):
                if not self._names_account(account):
                    self._protector.delete(account)
            raise
        # The new account must open the committed database before the old goes.
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            try:
                self._authenticate_keys(connection, self._protector.load(account))
            finally:
                connection.rollback()
        if not keep_old_account:
            self._protector.delete(old_account)
        elif on_kept is not None:
            on_kept(old_account)
        return account

    def _remove_damaged(self, connection, damaged) -> None:
        """Drop unreadable versions with what points at them: a head (the
        object goes; it already reads under no key), queued envelopes with their
        encrypted bodies and heads, quarantine notes, and the local row naming
        the version (an outbox entry, undo, record link, workspace mapping or
        agent access setting). Readable history and notes that name no version
        stay."""
        for key in damaged:
            if key[0] in _LOCAL_POINTERS:
                self._remove_local_pointer(connection, key)
            for outbox_id, envelope_version in connection.execute(
                "SELECT outbox_id, envelope_version FROM encrypted_outbox "
                "WHERE object_type = ? AND object_id = ? AND version_id = ?",
                key,
            ).fetchall():
                self._shred_outbox_body(connection, outbox_id, envelope_version)
            for table in ("object_heads", "encrypted_outbox", "quarantine"):
                connection.execute(
                    f"DELETE FROM {table} WHERE object_type = ? "  # nosec B608 - fixed names
                    "AND object_id = ? AND version_id = ?",
                    key,
                )
            connection.execute(
                "DELETE FROM encrypted_objects WHERE object_type = ? "
                "AND object_id = ? AND version_id = ?",
                key,
            )

    @staticmethod
    def _remove_local_pointer(connection, key) -> None:
        """Drop the local row naming a damaged version (``_LOCAL_POINTERS``)."""
        table, id_column, version_column = _LOCAL_POINTERS[key[0]]
        connection.execute(
            f"DELETE FROM {table} WHERE {id_column} = ? "  # nosec B608 - fixed names
            f"AND {version_column} = ?",
            key[1:],
        )

    @staticmethod
    def _scope_kinds(connection, damaged_ids, scopes) -> tuple[dict, list[str]]:
        """The kind each damaged current scope is rebuilt with, and the ids no
        evidence settles (fix round 3).

        The kind is sealed in the damaged version. Evidence, in order: a
        readable older version of the scope; a workspace mapping or unlinked
        row (only workspaces have one); a `scope-global-` id (local global
        scopes are created so). A profile has one global scope, so once one is
        known (readable, or settled above) the rest are workspaces, and when
        none is known a single unsettled scope is it. More than one unsettled
        with no known global is ambiguous and refuses: guessing could serve a
        workspace's records as global context in every chat.
        """
        from tldw_profile_core import ProfileScope, ScopeKind

        kinds: dict[str, ScopeKind] = {}
        globals_ = set()
        for (scope_id, version), plaintext in scopes.items():
            kinds[scope_id] = ProfileScope.model_validate_json(plaintext).kind
            if kinds[scope_id] is ScopeKind.GLOBAL and _is_current(
                connection, ("scope", scope_id, version)
            ):
                globals_.add(scope_id)
        settled, unsettled = {}, []
        for scope_id in sorted(damaged_ids):
            kind = kinds.get(scope_id)
            if kind is None and connection.execute(
                "SELECT 1 FROM local_scope_bindings WHERE scope_id = ? UNION ALL "
                "SELECT 1 FROM local_unlinked_scopes WHERE scope_id = ?",
                (scope_id, scope_id),
            ).fetchone():
                kind = ScopeKind.WORKSPACE
            if kind is None and scope_id.startswith("scope-global-"):
                kind = ScopeKind.GLOBAL
            if kind is None:
                unsettled.append(scope_id)
                continue
            settled[scope_id] = kind
            if kind is ScopeKind.GLOBAL:
                globals_.add(scope_id)
        if len(globals_) > 1:
            return settled, sorted(damaged_ids)
        if globals_ or len(unsettled) == 1:
            last = ScopeKind.WORKSPACE if globals_ else ScopeKind.GLOBAL
            settled.update(dict.fromkeys(unsettled, last))
            return settled, []
        return settled, unsettled

    def _rebuild_scopes(self, connection, rebuilt) -> None:
        """Owner decision 2026-11-16 ("Rebuild the scope"): a fresh version of
        each damaged current scope under its own id, so its records stay
        attached. The scope object's own fields reset (fresh times; only the id
        and kind carry over); a workspace keeps its mapping and its readable
        label, and one with no label is named "Recovered scope". Its agent
        access row is a separate object: a
        readable one stays exactly as it was (Off stays Off); only a damaged one
        is replaced, by Off (coordinator ruling, fix round 2)."""
        from tldw_profile_core import ProfileScope

        from .repository import _now, _uuid

        if not rebuilt:
            return
        (profile_id,) = connection.execute(
            "SELECT profile_id FROM profile_meta WHERE singleton = 1"
        ).fetchone()
        for scope_id, kind in rebuilt.items():
            now = _now()
            self._insert_scope(
                connection,
                ProfileScope(
                    scope_id=scope_id,
                    profile_id=profile_id,
                    kind=kind,
                    version_id=_uuid("scope-version"),
                    created_at=now,
                    updated_at=now,
                ),
                expected_version_id=None,
            )
            body = self._local_body(
                connection, "local_scope_bindings", "scope_binding", scope_id
            )
            # Fix round 3: a readable label is kept; only a mapping with none
            # is named "Recovered scope".
            if body is not None and not str(body.get("label") or "").strip():
                self._put_local_body(
                    connection,
                    "scope_binding",
                    scope_id,
                    {**body, "label": RECOVERED_SCOPE_LABEL},
                )

    def _local_body(self, connection, table, object_type, scope_id):
        """The current local body for a scope (already under the new key), or
        None when there is none."""
        column = _LOCAL_POINTERS[object_type][2]
        row = connection.execute(
            f"SELECT o.* FROM {table} t JOIN encrypted_objects o "  # nosec B608 - fixed names
            f"ON o.object_type = ? AND o.object_id = t.scope_id "
            f"AND o.version_id = t.{column} WHERE t.scope_id = ?",
            (object_type, scope_id),
        ).fetchone()
        return None if row is None else json.loads(self._decrypt_row_authenticated(row))

    def _put_local_body(self, connection, object_type, scope_id, body) -> None:
        """Name a fresh version of a scope's local body, as ``_commit_local_body``
        does, inside the caller's transaction."""
        from .repository import _uuid

        table, _, column = _LOCAL_POINTERS[object_type]
        version_id = _uuid(f"{object_type}-version")
        self._insert_encrypted(
            connection,
            object_type=object_type,
            object_id=scope_id,
            version_id=version_id,
            scope_id=scope_id,
            value=body,
        )
        connection.execute(
            "DELETE FROM object_heads WHERE object_type = ? AND object_id = ?",
            (object_type, scope_id),
        )
        connection.execute(
            "INSERT INTO object_heads VALUES (?, ?, ?)", (object_type, scope_id, version_id)
        )
        connection.execute(
            f"INSERT INTO {table}(scope_id, {column}) VALUES (?, ?) "  # nosec B608 - fixed names
            f"ON CONFLICT(scope_id) DO UPDATE SET {column} = excluded.{column}",
            (scope_id, version_id),
        )

    def _reencryption_source(self, connection, keys) -> str:
        """The live account holding exactly ``keys``; never mid-link."""
        if connection.execute(
            "SELECT 1 FROM first_link_freeze UNION ALL "
            "SELECT 1 FROM first_link_rebaseline_commit LIMIT 1"
        ).fetchone():
            raise ProfileLockedError("A first link is in progress.")
        self._authenticate_keys(connection, keys)
        return self._authenticated_account(connection, keys)

    def _add_key_account_column(self, connection: sqlite3.Connection) -> None:
        """v8 -> v9: the empty reference; stamping waits for an authenticated key.

        A removed v8 row may still owe its path-account key a deletion (a failed
        secure removal), so it records that account for the retry.
        """

        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(profile_meta)")
        }
        if "key_account" not in columns:
            connection.execute(
                "ALTER TABLE profile_meta ADD COLUMN key_account TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "UPDATE profile_meta SET key_account = ? WHERE destroyed = 1",
                (self._profile_ref,),
            )

    def _removal_record(
        self, connection: sqlite3.Connection, keys: ProfileKeyMaterial | None
    ) -> str:
        """What Remove owes: the named account, bound to the key it authenticated."""

        account = self._authenticated_account(connection, keys)
        return f"removal:{account}:{_removal_tag(keys, account)}"

    def _delete_removed_account(self, record: object) -> None:
        """Delete only the key this database's Remove authenticated, then forget it.

        The entry must still reproduce the record's tag. A bare path account is a
        v8 removal still owed, deleted as dev's retry did. Anything else, or an
        entry now holding another key (a rewritten row), is never deleted. An
        unavailable keyring raises and keeps the record; only an absent entry
        or one that is not a profile key clears it without a deletion.
        """

        if not isinstance(record, str) or not record:
            return
        removal = REMOVAL.fullmatch(record)
        account = removal.group(1) if removal else record
        if removal is None:
            if record == self._profile_ref:
                self._protector.delete(account)  # Absent entry: a no-op.
        elif KEY_ACCOUNT.fullmatch(account):
            # One read on the way in: exists() also reads the secret.
            try:
                material = self._protector.load(account)
            except ProfileKeyMaterialInvalidError:
                material = None  # An entry that is not a profile key stays.
            except ProfileLockedError:
                if self._protector.exists(account):  # Present but unreadable now.
                    raise
                material = None  # Already gone: nothing is owed.
            if material is not None and hmac.compare_digest(
                _removal_tag(material, account), removal.group(2)
            ):
                self._protector.delete(account)
        with self._transaction() as connection:
            connection.execute(
                "UPDATE profile_meta SET key_account = '' "
                "WHERE singleton = 1 AND destroyed = 1 AND key_account = ?",
                (record,),
            )


def profile_key_opens(connection: sqlite3.Connection, account: str, value) -> bool:
    """Whether a stored keyring value authenticates this database; no writes.

    Backup uses it on a captured copy before it records the key (TASK-40134).
    """

    return profile_key_authenticates(connection, account, value) is True


def profile_key_authenticates(
    connection: sqlite3.Connection, account: str, value
) -> bool | None:
    """True or False when the manifest head says so; None when it could not tell.

    False is definite: the value is not key material, or the head fails
    AEAD/HMAC under it. Anything else (a read error, say) is None, so a caller
    never deletes a possibly good key on it (TASK-40135).
    """

    from .repository import PersonalContextRepository, ProfileIntegrityError

    try:
        keys = decode_keyring_material(value)
    except (ProfileLockedError, ValueError):
        return False
    live = "SELECT 1 FROM profile_meta WHERE singleton = 1 AND destroyed = 0"
    if connection.execute(live).fetchone() is None:
        return False
    probe = PersonalContextRepository.__new__(PersonalContextRepository)
    probe._keys, probe._profile_ref = None, account
    factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        probe._authenticate_keys(connection, keys)
        return True
    except ProfileIntegrityError:
        return False
    except ProfileLockedError as error:
        return False if isinstance(error.__context__, ProfileIntegrityError) else None
    except Exception:  # noqa: BLE001 - could not tell; never a verdict
        return None
    finally:
        connection.row_factory = factory
