"""Bounded disposable-root validation for the installed Chroma 1.5.8 Rust format."""

import hashlib
import json
import os

# Inspect opcodes only; never unpickle candidate data.
import pickletools  # nosec B403
import shutil
import signal
import sqlite3
import stat

# Fixed installed validator argv, never a shell.
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

from .archive_reader import _check, _regular
from .native_files import create_private_directory, create_private_file
from .space import require_capacity

_SCHEMA_DIGEST = "0eadac740dc45d9a7984fdeefce9cff811052c32c565bb113fa476bbdff79bc7"
_SEGMENT_FILES = frozenset(
    {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin"}
)
_MAX_BYTES = 2 * 1024**3
_MAX_RECORDS = 100_000
_SECONDS = 30


def _uuid(value):
    return type(value) is str and str(UUID(value)) == value


def recognized_path(relative, kind):
    """Only canonical installed engine filenames can become candidate paths."""
    parts = Path(relative).parts
    if not parts:
        return kind == "directory"
    if len(parts) == 1 and parts[0] in {
        "chroma.sqlite3",
        "chroma.sqlite3-wal",
        "chroma.sqlite3-shm",
    }:
        return kind == "file"
    try:
        if not _uuid(parts[0]):
            return False
    except ValueError:
        return False
    return (len(parts) == 1 and kind == "directory") or (
        len(parts) == 2
        and kind == "file"
        and parts[1] in _SEGMENT_FILES | {"index_metadata.pickle"}
    )


def validate_groups(items, candidates, parent, cancel, limits, budget):
    """Copy each complete engine group privately; never lend payloads to native code."""
    roots = {}
    for item in items:
        if item.owner == "rag.projections" and item.status in {
            "included",
            "included_directory",
        }:
            if item.metadata is None:
                raise ValueError("rag_projection_topology_required")
            roots.setdefault(item.metadata.root_id, []).append(item)
    for root_id, group in roots.items():
        _check(cancel)
        member_ids = {item.logical_id for item in group}
        if any(
            dependency.split(":")[2:3] == ["rag.projections"]
            and dependency not in member_ids
            for item in group
            for dependency in item.dependencies
        ):
            raise ValueError("rag_projection_root_incomplete")
        if not any(
            i.logical_id == root_id and i.metadata.relative_path == "" for i in group
        ):
            raise ValueError("rag_projection_root_incomplete")
        if any(
            not recognized_path(i.metadata.relative_path, i.metadata.kind)
            for i in group
        ):
            raise ValueError("rag_projection_format_unsupported")
        files = [i for i in group if i.status == "included"]
        if not files:
            if len(group) != 1:
                raise ValueError("rag_projection_root_incomplete")
            continue
        sizes = {i.logical_id: candidates[i.logical_id].stat().st_size for i in files}
        total = sum(sizes.values())
        if (
            total > min(budget, limits.expanded_bytes, _MAX_BYTES)
            or any(size > limits.member_bytes for size in sizes.values())
            or len(group) > limits.members
        ):
            raise ValueError("rag_projection_validation_budget")
        # One independent copy plus at most its own size of native replay growth.
        require_capacity({parent: total * 2})
        scratch = parent / ("rag-validation-" + uuid4().hex)
        create_private_directory(scratch)
        try:
            copied = 0
            for item in sorted(
                group,
                key=lambda i: (
                    len(Path(i.metadata.relative_path).parts),
                    i.metadata.relative_path,
                ),
            ):
                relative = item.metadata.relative_path
                if not relative:
                    continue
                target = scratch / relative
                if item.metadata.kind == "directory":
                    create_private_directory(target)
                    continue
                with (
                    _regular(candidates[item.logical_id]) as source,
                    create_private_file(target) as fd,
                ):
                    written = 0
                    while chunk := source.read(1024**2):
                        _check(cancel)
                        copied += len(chunk)
                        written += len(chunk)
                        if copied > total or written > sizes[item.logical_id]:
                            raise ValueError("rag_projection_candidate_changed")
                        view = memoryview(chunk)
                        while view:
                            count = os.write(fd, view)
                            if count <= 0:
                                raise OSError("rag_projection_copy_failed")
                            view = view[count:]
                    if written != sizes[item.logical_id]:
                        raise ValueError("rag_projection_candidate_changed")
            _run_validator(scratch, cancel, byte_budget=min(total * 2, _MAX_BYTES))
        finally:
            # _run_validator always reaps its child before returning or raising.
            shutil.rmtree(scratch)


def _run_validator(root, cancel, *, byte_budget):
    """Bound one native child, retaining it through cancellation and termination."""
    _check(cancel)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # Fixed module and generated private path only.
    process = subprocess.Popen(  # nosec B603
        [sys.executable, "-m", __name__, str(root), str(byte_budget)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def check_growth():
        size = 0
        for path in root.rglob("*"):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue  # Native journal removal is ordinary replay cleanup.
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("rag_projection_candidate_kind_changed")
            if size > byte_budget:
                raise ValueError("rag_projection_validation_budget")

    try:
        deadline = time.monotonic() + _SECONDS
        while process.poll() is None:
            _check(cancel)
            if time.monotonic() >= deadline:
                raise ValueError("rag_projection_validation_timeout")
            check_growth()
            cancel.wait(0.02)
        check_growth()
        failures = {
            2: "rag_projection_foreign_execution_configuration",
            3: "rag_projection_validation_budget",
            4: "rag_projection_format_unsupported",
        }
        for name, failure in (
            ("SIGXCPU", "rag_projection_validation_timeout"),
            ("SIGXFSZ", "rag_projection_validation_budget"),
        ):
            if number := getattr(signal, name, None):
                failures[-number] = failure
        if process.returncode:
            raise ValueError(
                failures.get(process.returncode, "rag_projection_candidate_invalid")
            )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _json_config(text):
    if text is None:
        return
    if len(text) > 1024**2:
        raise ValueError("rag_projection_configuration_limit")
    value = json.loads(text)
    allowed = {
        "defaults",
        "keys",
        "string",
        "float_list",
        "sparse_vector",
        "int",
        "float",
        "bool",
        "fts_index",
        "string_inverted_index",
        "vector_index",
        "sparse_vector_index",
        "int_inverted_index",
        "float_inverted_index",
        "bool_inverted_index",
        "enabled",
        "config",
        "space",
        "source_key",
        "hnsw",
        "bm25",
        "ef_construction",
        "max_neighbors",
        "ef_search",
        "num_threads",
        "batch_size",
        "sync_threshold",
        "resize_factor",
    }
    pending = [(value, False)]
    while pending:
        part, metadata_names = pending.pop()
        if isinstance(part, dict):
            for key, child in part.items():
                if not metadata_names and key == "embedding_function":
                    if child not in (
                        {"type": "unknown"},
                        {"type": "known", "name": "default", "config": {}},
                    ):
                        raise ValueError(
                            "rag_projection_foreign_execution_configuration"
                        )
                    continue
                if not metadata_names and key not in allowed:
                    raise ValueError("rag_projection_configuration_unsupported")
                pending.append((child, key == "keys" and not metadata_names))
        elif isinstance(part, list):
            raise TypeError("rag_projection_configuration_unsupported")


def _preflight(root):
    """Reject schema and locator authority before native code sees candidate bytes."""
    from .sqlite_validation import _Restrictions

    path = root / "chroma.sqlite3"
    if not path.is_file():
        raise ValueError("rag_projection_database_missing")
    # Only the child-created, fixed private candidate enters this WAL-visible open.
    connection = connect_private_sqlite(
        "recovery.rag_projection_validation",
        path,
        read_only=True,
        must_exist=True,
        immutable=False,
        isolation_level=None,
        cached_statements=0,
        timeout=0,
    )
    try:
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8 * 1024**2)
        _Restrictions(connection)
        schema = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"
        ).fetchmany(1000)
        digest = hashlib.sha256(
            json.dumps(schema, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != _SCHEMA_DIGEST:
            raise ValueError("rag_projection_schema_unsupported")
        if connection.execute("PRAGMA quick_check").fetchmany(2) != [("ok",)]:
            raise ValueError("rag_projection_database_invalid")
        collections = connection.execute(
            "SELECT id,name,dimension,database_id,config_json_str,schema_str FROM collections"
        ).fetchmany(1001)
        if len(collections) > 1000:
            raise ValueError("rag_projection_collection_limit")
        ids = set()
        for identity, name, dimension, database, config, schema in collections:
            if not _uuid(identity) or not _uuid(database) or not isinstance(name, str):
                raise ValueError("rag_projection_identifier_invalid")
            if dimension is not None and not 0 < dimension <= 65536:
                raise ValueError("rag_projection_dimension_invalid")
            ids.add(identity)
            if config is not None and json.loads(config) != {}:
                raise ValueError("rag_projection_foreign_execution_configuration")
            _json_config(schema)
        databases = dict(
            connection.execute("SELECT id,name FROM databases").fetchmany(1001)
        )
        if databases != {
            "00000000-0000-0000-0000-000000000000": "default_database"
        } or connection.execute("SELECT id FROM tenants").fetchmany(2) != [
            ("default_tenant",)
        ]:
            raise ValueError("rag_projection_namespace_unsupported")
        segments = connection.execute(
            "SELECT id,type,scope,collection FROM segments"
        ).fetchmany(2001)
        if len(segments) != 2 * len(collections):
            raise ValueError("rag_projection_segments_invalid")
        vectors = set()
        metadata_ids = {}
        vector_ids = {}
        seen = set()
        for identity, kind, scope, collection in segments:
            if (
                not _uuid(identity)
                or collection not in ids
                or (collection, scope) in seen
            ):
                raise ValueError("rag_projection_segment_identifier_invalid")
            seen.add((collection, scope))
            if (kind, scope) == (
                "urn:chroma:segment/vector/hnsw-local-persisted",
                "VECTOR",
            ):
                vectors.add(identity)
                vector_ids[collection] = identity
            elif (kind, scope) != ("urn:chroma:segment/metadata/sqlite", "METADATA"):
                raise ValueError("rag_projection_segment_type_unsupported")
            else:
                metadata_ids[collection] = identity
        for collection in ids:
            if not (root / vector_ids[collection]).is_dir() and (
                connection.execute(
                    "SELECT count(*) FROM embeddings WHERE segment_id=?",
                    (metadata_ids[collection],),
                ).fetchone()[0]
                or connection.execute(
                    "SELECT count(*) FROM embeddings_queue WHERE topic=?",
                    ("persistent://default/default/" + collection,),
                ).fetchone()[0]
            ):
                raise ValueError("rag_projection_segment_files_missing")
        for child in root.iterdir():
            if child.is_dir():
                if child.name not in vectors or not _SEGMENT_FILES <= {
                    p.name for p in child.iterdir()
                }:
                    raise ValueError("rag_projection_segment_files_missing")
                metadata = child / "index_metadata.pickle"
                if metadata.exists():
                    if metadata.stat().st_size > 32 * 1024**2:
                        raise ValueError("rag_projection_metadata_limit")
                    allowed = {
                        "PROTO",
                        "FRAME",
                        "EMPTY_DICT",
                        "MARK",
                        "NONE",
                        "BININT",
                        "BININT1",
                        "BININT2",
                        "LONG1",
                        "BINUNICODE",
                        "SHORT_BINUNICODE",
                        "SETITEMS",
                        "SETITEM",
                        "STOP",
                        "BINPUT",
                        "LONG_BINPUT",
                        "BINGET",
                        "LONG_BINGET",
                        "MEMOIZE",
                    }
                    if any(
                        op.name not in allowed
                        for op, _, _ in pickletools.genops(metadata.read_bytes())
                    ):
                        raise ValueError("rag_projection_pickle_unsupported")
        return collections
    finally:
        connection.close()


def _native_validate(root):
    collections = _preflight(root)
    import chromadb
    from chromadb.api.rust import RustBindingsAPI
    from chromadb.config import Settings

    if chromadb.__version__ != "1.5.8":
        raise ValueError("rag_projection_version_unsupported")
    client = chromadb.PersistentClient(
        path=str(root),
        settings=Settings(anonymized_telemetry=False, migrations="validate"),
    )
    system = client._system
    try:
        server = client._server
        if type(server) is not RustBindingsAPI:
            raise ValueError("rag_projection_backend_unsupported")
        total = 0
        for identity, _, dimension, _, _, _ in collections:
            count = server._count(UUID(identity))
            total += count
            if total > _MAX_RECORDS:
                raise ValueError("rag_projection_record_limit")
            observed = 0
            for offset in range(0, count, 128):
                batch = server._get(
                    UUID(identity),
                    limit=128,
                    offset=offset,
                    include=["documents", "embeddings", "metadatas"],
                )
                observed += len(batch["ids"])
                if any(len(vector) != dimension for vector in batch["embeddings"]):
                    raise ValueError("rag_projection_vector_invalid")
            if observed != count:
                raise ValueError("rag_projection_count_invalid")
    finally:
        client.close()
    if system._running or "bindings" in vars(client._server):
        raise ValueError("rag_projection_retirement_unavailable")


def _main(root):
    """Expose only fixed known qualification failures across the child boundary."""
    try:
        _native_validate(root)
    except (TypeError, ValueError) as error:
        if str(error) in {
            "rag_projection_foreign_execution_configuration",
            "rag_projection_configuration_unsupported",
        }:
            return 2
        if str(error) in {
            "rag_projection_record_limit",
            "rag_projection_collection_limit",
            "rag_projection_configuration_limit",
            "rag_projection_metadata_limit",
        }:
            return 3
        if str(error) in {
            "rag_projection_version_unsupported",
            "rag_projection_schema_unsupported",
            "rag_projection_backend_unsupported",
            "rag_projection_namespace_unsupported",
            "rag_projection_segment_type_unsupported",
            "rag_projection_pickle_unsupported",
        }:
            return 4
        raise
    return 0


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (_SECONDS, _SECONDS))
    file_limit = min(int(sys.argv[2]), _MAX_BYTES)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    sys.exit(_main(Path(sys.argv[1])))
