"""Private paste artifacts bound to durable Research source operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_text,
    lexical_path,
    open_private_binary,
    secure_private_directory,
    unlink_private_file,
)

from .source_operations import SourceOperationStatus, validate_source_operation_id


MAX_STAGING_ARTIFACTS = 100
MAX_STAGING_INDEX_BYTES = 64 * 1024
MAX_PASTE_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PasteStagingSweepResult:
    deleted: int
    retained: int
    incomplete: bool = False


class ResearchPasteStagingStore:
    """Own bounded private payload files and a path-free operation index."""

    def __init__(self, root: str | Path) -> None:
        self.root = lexical_path(root)
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()
        secure_private_directory(
            self.root,
            create=True,
            application_owned=True,
        )

    def artifact_path(self, operation_id: str) -> Path:
        """Derive the one application-owned artifact path for an operation."""

        normalized = validate_source_operation_id(operation_id)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.txt"

    def stage(self, operation_id: str, *, title: str, body: str) -> Path:
        """Write one private payload, then bind it in the private index."""

        normalized = validate_source_operation_id(operation_id)
        if not isinstance(title, str) or not isinstance(body, str) or not body:
            raise ValueError("paste title/body must be text and body must be nonblank")
        payload = f"{title.strip() or 'Pasted research source'}\n\n{body}"
        if len(payload.encode("utf-8")) > MAX_PASTE_BYTES:
            raise ValueError("pasted source exceeds the staging bound")
        path = self.artifact_path(normalized)
        digest = path.stem
        with self._lock:
            operations, precondition = self._load_index()
            if digest in operations:
                if operations[digest] != normalized or not path.exists():
                    raise ValueError("paste staging index is inconsistent")
                return path
            if len(operations) >= MAX_STAGING_ARTIFACTS:
                raise ValueError("paste staging exceeds the artifact bound")
            atomic_private_write_text(
                path,
                payload,
                application_owned_directory=self.root,
                target_precondition=PrivateFileWritePrecondition.missing(),
            )
            try:
                operations[digest] = normalized
                self._write_index(operations, precondition=precondition)
            except Exception:
                unlink_private_file(path, application_owned_directory=self.root)
                raise
        return path

    def delete(self, operation_id: str) -> bool:
        """Delete only the derived artifact bound to the exact operation."""

        normalized = validate_source_operation_id(operation_id)
        path = self.artifact_path(normalized)
        digest = path.stem
        with self._lock:
            operations, precondition = self._load_index()
            indexed = operations.get(digest)
            if indexed not in {None, normalized}:
                raise ValueError("paste staging index is inconsistent")
            deleted = unlink_private_file(
                path,
                application_owned_directory=self.root,
            )
            if indexed is not None:
                del operations[digest]
                self._write_index(operations, precondition=precondition)
            return deleted

    def sweep(
        self,
        operation_store: Any,
        *,
        job_registry: Any | None = None,
        limit: int = MAX_STAGING_ARTIFACTS,
    ) -> PasteStagingSweepResult:
        """Bound startup cleanup while retaining retryable catalog failures."""

        if type(limit) is not int or not 1 <= limit <= MAX_STAGING_ARTIFACTS:
            raise ValueError("limit is outside the staging sweep bound")
        with self._lock:
            operations, precondition = self._load_index()
            deleted = 0
            retained = 0
            visited = 0
            for digest, operation_id in tuple(operations.items()):
                if visited >= limit:
                    break
                visited += 1
                try:
                    operation = operation_store.get(operation_id)
                except Exception:
                    retained += 1
                    continue
                held_job_exists = False
                if operation is None and job_registry is not None:
                    held_job_exists = any(
                        job.research_source_operation_id == operation_id
                        and str(getattr(job.state, "value", "")) == "queued"
                        and bool(getattr(job, "dispatch_held", False))
                        for job in job_registry.jobs()
                    )
                should_delete = operation is None and not held_job_exists
                if operation is not None:
                    should_delete = (
                        operation.catalog_status is SourceOperationStatus.SUCCEEDED
                    )
                    if (
                        not should_delete
                        and job_registry is not None
                        and operation.ingest_job_id
                    ):
                        job = job_registry.get_job(operation.ingest_job_id)
                        state = str(getattr(getattr(job, "state", None), "value", ""))
                        should_delete = state in {"cancelled", "skipped"}
                if should_delete:
                    unlink_private_file(
                        self.root / f"{digest}.txt",
                        application_owned_directory=self.root,
                    )
                    del operations[digest]
                    deleted += 1
                else:
                    retained += 1

            indexed_names = {f"{digest}.txt" for digest in operations}
            for candidate in sorted(
                self.root.glob("*.txt"), key=lambda item: item.name
            ):
                if visited >= limit:
                    break
                if candidate.name in indexed_names:
                    continue
                visited += 1
                if unlink_private_file(
                    candidate,
                    application_owned_directory=self.root,
                ):
                    deleted += 1
            self._write_index(operations, precondition=precondition)
            incomplete = visited >= limit and (
                len(operations) > retained
                or any(
                    candidate.name not in indexed_names
                    for candidate in self.root.glob("*.txt")
                )
            )
            return PasteStagingSweepResult(deleted, retained, incomplete)

    def _load_index(
        self,
    ) -> tuple[dict[str, str], PrivateFileWritePrecondition]:
        try:
            with open_private_binary(self.index_path) as opened:
                precondition = PrivateFileWritePrecondition.from_opened(opened)
                raw = opened.stream.read(MAX_STAGING_INDEX_BYTES + 1)
        except FileNotFoundError:
            return {}, PrivateFileWritePrecondition.missing()
        if len(raw) > MAX_STAGING_INDEX_BYTES:
            raise ValueError("paste staging index exceeds its bound")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "operations",
        }:
            raise ValueError("paste staging index is invalid")
        if payload["schema_version"] != 1 or not isinstance(
            payload["operations"], dict
        ):
            raise ValueError("paste staging index is invalid")
        if len(payload["operations"]) > MAX_STAGING_ARTIFACTS:
            raise ValueError("paste staging index exceeds the artifact bound")
        operations: dict[str, str] = {}
        for digest, operation_id in payload["operations"].items():
            normalized = validate_source_operation_id(operation_id)
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise ValueError("paste staging index contains an invalid key")
            if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != digest:
                raise ValueError("paste staging index operation binding is invalid")
            operations[digest] = normalized
        return operations, precondition

    def _write_index(
        self,
        operations: dict[str, str],
        *,
        precondition: PrivateFileWritePrecondition,
    ) -> None:
        atomic_private_write_text(
            self.index_path,
            json.dumps(
                {"schema_version": 1, "operations": operations},
                sort_keys=True,
                separators=(",", ":"),
            ),
            application_owned_directory=self.root,
            target_precondition=precondition,
        )
