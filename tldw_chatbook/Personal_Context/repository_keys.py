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
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager, suppress
from pathlib import Path

from .key_protector import (
    ProfileKeyMaterial,
    ProfileKeyMaterialInvalidError,
    ProfileLockedError,
    _serialize_material,
    decode_keyring_material,
)

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


def _locked(message: str, reason: str) -> ProfileLockedError:
    error = ProfileLockedError(message)
    error.reason_code = reason
    return error


def stored_restore_marker(db_path: Path) -> str | None:
    """A live profile's restore marker, read-only; None for anything else.

    The repository reads it when no secure keyring exists, so a restore's
    reason shows instead of the keyring's refusal (TASK-40136). It never
    creates the database.
    """
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    if not Path(db_path).exists():
        return None
    try:
        with closing(
            connect_private_sqlite(
                "recovery.personal_context.repository", db_path, read_only=True
            )
        ) as connection:
            row = connection.execute(
                "SELECT key_account FROM profile_meta "
                "WHERE singleton = 1 AND destroyed = 0"
            ).fetchone()
    except (OSError, ValueError, sqlite3.Error):
        return None
    stored = None if row is None else row[0]
    return stored if isinstance(stored, str) and stored.startswith(RESTORE_MARKER) else None


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

    def damaged_record_versions(self) -> int:
        """How many stored record versions no key can read (TASK-40140).

        Owner decision 2026-11-14: the review shows this count, and
        re-encryption removes them. Read-only.
        """

        self._require_keys()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            try:
                rows = connection.execute(
                    "SELECT * FROM encrypted_objects WHERE object_type = 'record'"
                ).fetchall()
                return sum(1 for row in rows if self._reads(row) is None)
            finally:
                connection.rollback()

    def _reads(self, row) -> bytes | None:
        """The row's plaintext, or None when it fails authentication (a record
        only; any other damaged object still refuses)."""
        from .repository import ProfileIntegrityError

        try:
            return self._decrypt_row_authenticated(row)
        except ProfileIntegrityError:
            if row["object_type"] != "record":
                raise
            return None

    def reencrypt_profile(
        self, *, keep_old_account: bool, expected_damaged: int = 0
    ) -> str:
        """Re-encrypt every row under a fresh encryption key (TASK-40140).

        The integrity key and link state stay; ``key_version`` goes up. The new
        key goes into a new, unused account first; one transaction then rewrites
        every row and names that account, so an interruption leaves either the
        old rows naming the old key or the new rows naming the new one. Record
        versions no key can read are removed in that same transaction, but only
        as many as the review showed (``expected_damaged``; owner decision
        2026-11-14). The old account is deleted only after the new one opens the
        database, and only when the caller knows no retained restore copy can
        name it.

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
                damaged = []
                for row in rows:
                    key = (row["object_type"], row["object_id"], row["version_id"])
                    plaintext = self._reads(row)
                    if plaintext is None:
                        damaged.append(key)
                        continue
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
                if len(damaged) != expected_damaged:
                    raise ProfileLockedError("Damaged records changed since review.")
                self._remove_damaged(connection, damaged)
                connection.execute(
                    "UPDATE profile_meta SET key_account = ? WHERE singleton = 1",
                    (account,),
                )
                self._authenticate_keys(connection, new)
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
        return account

    def _remove_damaged(self, connection, damaged) -> None:
        """Drop unreadable record versions with what points at them: a head
        (the record goes; it already reads under no key), queued envelopes with
        their encrypted bodies and heads, and quarantine notes. Readable history
        and notes that name no version stay."""
        for key in damaged:
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
