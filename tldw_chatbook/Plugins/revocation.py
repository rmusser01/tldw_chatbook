"""Immediate scoped fences and retained cancellation, independent of storage."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Literal

from .authority import PluginMarker
from .review import OperationReceipt


@dataclass(frozen=True)
class RevocationTarget:
    """A named workspace, explicit global-default edit, or all scopes."""

    installation_id: str
    workspace_id: str | None
    everywhere: bool
    global_default: bool = False

    def __post_init__(self) -> None:
        PluginMarker(
            generation=1,
            operation_id=self.installation_id,
            recovery_snapshot_digest="0" * 64,
        )
        if (
            "/" in self.installation_id
            or "\\" in self.installation_id
            or self.installation_id in {".", ".."}
        ):
            raise ValueError("installation identity must be one path segment")
        if type(self.everywhere) is not bool or type(self.global_default) is not bool:
            raise ValueError("revocation mode must be explicit")
        if self.workspace_id is not None:
            if (
                self.everywhere
                or self.global_default
                or not isinstance(self.workspace_id, str)
                or not self.workspace_id.strip()
                or self.workspace_id in {"global", "workspace-default"}
            ):
                raise ValueError("invalid named workspace revocation")
        elif self.everywhere == self.global_default:
            raise ValueError("choose everywhere or global_default explicitly")

    @property
    def scope(self) -> tuple[str, str, str]:
        return (
            self.installation_id,
            (
                "installation"
                if self.everywhere
                else "global_default"
                if self.global_default
                else "workspace"
            ),
            self.workspace_id or "",
        )

    def matches(self, record) -> bool:
        return record.installation_id == self.installation_id and (
            self.everywhere
            or (
                self.global_default
                and any(kind == "global_default" for kind, _, _ in record.generations)
            )
            or (not self.global_default and record.workspace_id == self.workspace_id)
        )


def revocation_for_review(review) -> RevocationTarget | None:
    """Find activation reviews that remove effective authority at issuance."""
    if review.kind != "activate":
        return None
    closes = review.intent == "disabled"
    if review.intent == "inherit":
        installed = next(
            row
            for row in json.loads(review.authority_json)["installations"]
            if row["installation_id"] == review.installation_id
        )
        closes = not installed["activation_default"]
    if not closes:
        return None
    return RevocationTarget(
        review.installation_id,
        review.workspace_id,
        False,
        global_default=review.workspace_id is None,
    )


@dataclass(frozen=True)
class RevocationRequest:
    """Host request identity; only retained current-session custody may mutate."""

    request_id: str

    def __post_init__(self):
        if re.fullmatch(r"lr1\.[0-9a-f]{32}\.[0-9a-f]{32}", self.request_id) is None:
            raise ValueError("invalid revocation request identity")


class RevocationConflict(ValueError):
    """A retry cannot reuse another mutation's immutable identity."""


class RevocationFailure(RuntimeError):
    """Content-free public error with original exception retained for its owner."""

    def __init__(self, receipt: OperationReceipt, error: BaseException):
        super().__init__("plugin_revocation_persistence_failed")
        self.receipt = receipt
        self.original_error = error


@dataclass
class RevocationOperation:
    """Retained session custody; never authenticated commitment evidence."""

    target: RevocationTarget
    kind: Literal["revoke", "uninstall", "activate"]
    operation_id: str
    records: tuple
    receipt: OperationReceipt
    cleanup_tasks: set = field(default_factory=set)
    cleanup_errors: list[BaseException] = field(default_factory=list)
    task: asyncio.Task | None = None
    review: object | None = None
    files_pending: bool = False
    unresolved_tokens: tuple[str, ...] = ()
    runtime_observed: bool = False
    scope_versions: tuple[tuple[tuple[str, str, str], int], ...] = ()
    expires_at: float = field(default_factory=lambda: time.monotonic() + 900)
    durable_id: str | None = None

    def status(self) -> OperationReceipt:
        stopped = not self.unresolved_tokens and all(
            record.completed.is_set() for record in self.records
        )
        if stopped and not self.runtime_observed:
            stopped = None
        return replace(
            self.receipt,
            request_id=self.operation_id,
            operation_id=self.durable_id,
            runtime_stopped=stopped,
            cleanup_pending=stopped is not True
            or bool(self.cleanup_tasks)
            or self.files_pending,
            cleanup_errors=tuple(type(error).__name__ for error in self.cleanup_errors),
        )

    def cancel_owned(self) -> None:
        """Invoke exact host callbacks now; retain awaitable cleanup separately."""
        seen = set()
        for record in self.records:
            if record.run_id in seen or record.completed.is_set():
                continue
            seen.add(record.run_id)
            try:
                result = record.cancel()
                if isinstance(result, Future):
                    result = asyncio.wrap_future(result)
                if inspect.isawaitable(result):
                    task = asyncio.ensure_future(result)
                    self.cleanup_tasks.add(task)
                    task.add_done_callback(self._cleanup_done)
            # A host callback failure cannot abandon the remaining owners.
            except BaseException as error:  # noqa: BLE001
                self.cleanup_errors.append(error)

    def _cleanup_done(self, task):
        self.cleanup_tasks.discard(task)
        try:
            task.result()
        except BaseException as error:  # noqa: BLE001
            self.cleanup_errors.append(error)
