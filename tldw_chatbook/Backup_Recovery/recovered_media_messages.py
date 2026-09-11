"""Current-profile chat tombstones reconcile the existing recovered-media owner."""

import hashlib
import json
import stat
import tomllib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import bootstrap, recovered_media
from .generation_witnesses import _witnesses
from .profile_paths import database_path, lexical_path, user_data_dir
from .storage_admission import _read_recovery_file, acquire_storage

CLEANUP_PENDING = "Message deleted; recovered-media reference cleanup is pending."


@dataclass(frozen=True)
class _Source:
    config: Path
    digest: str
    database: Path
    database_identity: tuple[int, ...]
    root: Path
    catalog_identity: tuple | None
    generations: bytes


def _db_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.resolve() != path:
        raise ValueError("recovered_message_source_changed")
    return recovered_media._native_identity(info)


@contextmanager
def _source_scope(db):
    selected = bootstrap.effective_config_path()
    with ExitStack() as stack:
        config_lease = stack.enter_context(acquire_storage(selected))
        generations = [(str(selected), _witnesses(selected, config_lease))]
        raw = _read_recovery_file("config", selected, max_bytes=16 * 1024**2)
        parsed = tomllib.loads(raw.decode("utf-8"))
        path = database_path(parsed, "chachanotes_db_path")
        if lexical_path(db.db_path) != path:
            raise ValueError("recovered_message_source_changed")
        db_lease = stack.enter_context(acquire_storage(path))
        generations.append((str(path), _witnesses(path, db_lease)))
        db_identity = _db_identity(path)
        root = user_data_dir(parsed) / "recovered_media"
        catalog = None
        root_lease = None
        if recovered_media.list_recovered_media(root, limit=1) is not None:
            root_lease = stack.enter_context(acquire_storage(root))
            generations.append((str(root), _witnesses(root, root_lease)))
            catalog = recovered_media._message_catalog_identity(
                recovered_media._existing_store(root)
            )
        source = _Source(
            selected,
            hashlib.sha256(raw).hexdigest(),
            path,
            db_identity,
            root,
            catalog,
            json.dumps(generations, sort_keys=True).encode(),
        )
        yield source
        if (
            bootstrap.effective_config_path() != selected
            or lexical_path(db.db_path) != path
            or _db_identity(path) != db_identity
            or _read_recovery_file("config", selected, max_bytes=16 * 1024**2) != raw
            or _witnesses(selected, config_lease) != generations[0][1]
            or _witnesses(path, db_lease) != generations[1][1]
            or root_lease is not None
            and (
                _witnesses(root, root_lease) != generations[2][1]
                or recovered_media._message_catalog_identity(
                    recovered_media._existing_store(root)
                )
                != catalog
            )
        ):
            raise ValueError("recovered_message_source_changed")


class RecoveredMessageReferences:
    """One composition's actual DB and selected source, never a retargetable root."""

    def __init__(self, db):
        self.db = db
        with _source_scope(db) as source:
            self.source = source
        self.profile = hashlib.sha256(str(source.config).encode()).hexdigest()[:24]

    @contextmanager
    def _checked(self):
        with _source_scope(self.db) as current:
            if current != self.source:
                raise ValueError("recovered_message_source_changed")
            yield

    def release(self, message_ids):
        """Reconfirm positive committed tombstones before the owner transaction."""
        with self._checked():
            if self.source.catalog_identity is None:
                return
            for start in range(0, len(message_ids), 200):
                rows = self.db.get_message_tombstones(message_ids[start : start + 200])
                ids = tuple(row["message_id"] for row in rows)
                recovered_media.release_message_references(
                    self.source.root,
                    self.profile,
                    ids,
                    expected_identity=self.source.catalog_identity,
                )

    def retry(self):
        """Keyset pages of existing refs use only positive chat tombstone evidence."""
        with self._checked():
            if self.source.catalog_identity is None:
                return
            after = ""
            while True:
                ids = recovered_media.message_reference_page(
                    self.source.root, self.profile, after=after
                )
                if not ids:
                    return
                rows = self.db.get_message_tombstones(ids)
                recovered_media.release_message_references(
                    self.source.root,
                    self.profile,
                    tuple(row["message_id"] for row in rows),
                    expected_identity=self.source.catalog_identity,
                )
                after = ids[-1]
