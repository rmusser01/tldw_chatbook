"""Dispatch/TTL/backoff orchestrator for the Console Environment panel.

Pure orchestration: no Textual imports here. Every I/O seam (worker
dispatch, UI-thread marshalling, workspace-root/rail-open accessors, the
snapshot callback, and the clock) is injected so this class can be
exercised with synchronous fakes and no running app -- see
``Tests/UI/test_console_environment_controller.py``.

Two worker tiers:

- LOCAL (``LOCAL_WORKER_GROUP``): git status + backlog task scan. Cheap,
  local-only I/O; dispatched on every 10s poll tick.
- NET (``NET_WORKER_GROUP``): the ``gh`` PR/checks fetch. Expensive
  network I/O; TTL'd to 60s per ``(root, branch)`` and only dispatched on
  demand (rail opened, explicit refresh, or ``force_net``). Kept in a
  *separate* worker group from the local tier so a 10s local poll never
  cancels an in-flight ``gh`` fetch.

Each tier tracks consecutive failures and pauses itself after
``_MAX_FAILURES`` in a row, until a workspace-root change resets it (both
tiers) or, for the net tier only, a ``force_net`` refresh.

The gatherers (``gather_git_env``, ``gather_pr_env``, ``BacklogTaskScanner``)
are imported as module attributes -- not called through an indirection
layer -- so tests can monkeypatch them directly on this module.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from tldw_chatbook.Chat.console_environment_state import (
    EnvironmentSnapshot,
    EnvSourceAvailability,
)
from tldw_chatbook.Workspaces.environment_status import (
    BacklogTaskScanner,
    gather_git_env,
    gather_pr_env,
)

_LOCAL_TIER = "local"
_NET_TIER = "net"


class ConsoleEnvironmentController:
    """Cadence/TTL/backoff orchestrator for the Environment panel's data."""

    LOCAL_WORKER_GROUP = "console-environment-local"
    NET_WORKER_GROUP = "console-environment-net"

    _MAX_FAILURES = 3
    _NET_TTL = timedelta(seconds=60)

    def __init__(
        self,
        *,
        run_worker: Callable[..., Any],
        marshal_to_ui: Callable[..., None],
        workspace_root_accessor: Callable[[], str | None],
        rail_open_accessor: Callable[[], bool],
        on_snapshot: Callable[[EnvironmentSnapshot], None],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._run_worker = run_worker
        self._marshal_to_ui = marshal_to_ui
        self._workspace_root_accessor = workspace_root_accessor
        self._rail_open_accessor = rail_open_accessor
        self._on_snapshot = on_snapshot
        self._now = now or (lambda: datetime.now(timezone.utc))

        self.snapshot = EnvironmentSnapshot()
        self._scanner = BacklogTaskScanner()
        # (root, branch, fetched_at) for the last dispatched net fetch.
        self._net_fetched_at: tuple[str, str | None, datetime] | None = None
        self._failures: dict[str, int] = {_LOCAL_TIER: 0, _NET_TIER: 0}

    # -- public API -------------------------------------------------------

    def request_refresh(self, *, include_net: bool = False, force_net: bool = False) -> None:
        """Dispatch a local refresh (and optionally a net refresh)."""
        root = self._workspace_root_accessor()
        if root is None or not self._rail_open_accessor():
            return
        scope_root = root

        if self._failures[_LOCAL_TIER] < self._MAX_FAILURES:
            self._dispatch_local(scope_root)

        if include_net and (force_net or self._failures[_NET_TIER] < self._MAX_FAILURES):
            self._dispatch_net(scope_root, force_net=force_net)

    def poll_tick(self) -> None:
        """Called every 10s by the screen timer: local tier only.

        Detects a workspace-root change since the last landed snapshot and,
        when one happened, resets backoff/TTL state and does a full
        (both-tier) refresh instead of a local-only poll.
        """
        root = self._workspace_root_accessor()
        if root is None or not self._rail_open_accessor():
            return
        last_root = self.snapshot.git.root or None
        if last_root is not None and root != last_root:
            self._failures = {_LOCAL_TIER: 0, _NET_TIER: 0}
            self._net_fetched_at = None
            self.request_refresh(include_net=True)
            return
        self.request_refresh(include_net=False)

    def notify_rail_opened(self) -> None:
        """Called when the Inspect rail opens: both tiers, TTL-respecting."""
        self.request_refresh(include_net=True)

    # -- dispatch -----------------------------------------------------------

    def _dispatch_local(self, scope_root: str) -> None:
        previous_git = self.snapshot.git

        def job() -> None:
            git_result = gather_git_env(Path(scope_root), previous=previous_git)
            tasks_result = self._scanner.scan(Path(scope_root), git_result.branch)
            self._marshal_to_ui(
                self._land, scope_root, _LOCAL_TIER, (git_result, tasks_result)
            )

        self._run_worker(job, thread=True, exclusive=True, group=self.LOCAL_WORKER_GROUP)

    def _dispatch_net(self, scope_root: str, *, force_net: bool) -> None:
        branch = self.snapshot.git.branch
        key = (scope_root, branch)
        if not force_net and self._net_fetched_at is not None:
            prev_root, prev_branch, fetched_at = self._net_fetched_at
            if (prev_root, prev_branch) == key and (self._now() - fetched_at) < self._NET_TTL:
                return  # TTL still holds
        previous_pr = self.snapshot.pr
        self._net_fetched_at = (scope_root, branch, self._now())

        def job() -> None:
            pr_result = gather_pr_env(Path(scope_root), branch, previous=previous_pr)
            self._marshal_to_ui(self._land, scope_root, _NET_TIER, pr_result)

        self._run_worker(job, thread=True, exclusive=True, group=self.NET_WORKER_GROUP)

    # -- landing (runs on the UI thread, via marshal_to_ui) ------------------

    def _land(self, scope_root: str, tier: str, result: Any) -> None:
        if scope_root != self._workspace_root_accessor():
            return  # stale-scope guard: a newer refresh already superseded this one

        if tier == _LOCAL_TIER:
            git_result, tasks_result = result
            self._record_outcome(_LOCAL_TIER, git_result.availability)
            self.snapshot = dataclasses.replace(self.snapshot, git=git_result, tasks=tasks_result)
        else:
            self._record_outcome(_NET_TIER, result.availability)
            self.snapshot = dataclasses.replace(self.snapshot, pr=result)

        self._on_snapshot(self.snapshot)

    def _record_outcome(self, tier: str, availability: EnvSourceAvailability) -> None:
        if availability is EnvSourceAvailability.ERROR:
            self._failures[tier] += 1
        else:
            self._failures[tier] = 0
