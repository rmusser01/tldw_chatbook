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


class RepositoryKeyCustody:
    """Key-account lookup, provisioning and removal for the repository."""

    def _named_key_account(self, stored: object) -> str:
        """Return the keyring account a live profile row names, or lock."""

        if stored == "":
            return self._profile_ref
        if isinstance(stored, str) and KEY_ACCOUNT.fullmatch(stored):
            return stored
        raise ProfileLockedError("Profile key account is invalid.")

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
                self._authenticate_keys(connection, protected)
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

    from .repository import PersonalContextRepository

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
    except Exception:  # noqa: BLE001 - any failure means the key does not open it
        return False
    finally:
        connection.row_factory = factory
