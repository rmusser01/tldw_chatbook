"""Dispatch/TTL/backoff orchestrator for the Console Environment panel.

Pure orchestration: no Textual imports here. Every I/O seam (worker
dispatch, UI-thread marshalling, workspace-root/rail-open accessors, the
snapshot callback, and the clock) is injected so this class can be
exercised with fakes and no running app -- see
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

Correctness notes (hardened after a deferred-worker review found these
only break under REAL async ordering, which the original inline-fake
tests could not exercise since jobs ran synchronously at dispatch time):

- The net tier's branch comes from the last *landed* local result
  (``_landed_root`` matching the current scope), never from whatever
  ``snapshot.git.branch`` happens to hold at dispatch time. On first rail
  open the local job may not have landed yet, so a net dispatch made
  before that is *deferred* and re-issued from ``_land`` once the local
  tier lands for the same root -- otherwise the branch is silently ``None``
  and the TTL key gets poisoned with a fetch that never ran ``gh``.
- All root-change comparisons (in ``poll_tick``) use ``_landed_root`` --
  the accessor's own raw string, recorded at dispatch time -- never
  ``GitEnvState.root``, which is git's *resolved* toplevel path (can
  differ in spelling from the accessor, e.g. ``/tmp`` vs ``/private/tmp``)
  and is ``""`` on an ERROR result. Comparing against the resolved path
  either loops the reset branch forever (spelling mismatch: backoff never
  arms) or never fires it (an errored tier's blank root reads as "no
  change": backoff never resumes).
- Landings carry a per-tier monotonic dispatch token. ``exclusive=True``
  on ``run_worker`` cancels the *awaiting* asyncio task when a newer
  dispatch of the same tier supersedes it, but the underlying OS thread
  keeps running and still calls back into ``_land`` -- so an older
  dispatch's result can arrive after a newer one already landed. A
  landing is honored only when its token still matches the tier's latest
  issued token; otherwise it's a stale, superseded result and is dropped.
- The deferred net re-dispatch re-checks ``rail_open_accessor`` (the rail
  can close during the deferral), and a pending net request whose scope is
  dropped mid-flight is cleared rather than latched -- otherwise its
  accumulated ``force_net`` leaks into the next request that re-keys the
  slot and silently bypasses the TTL. (task-13 addition B.)
- A local landing whose branch differs from the one the current PR state
  was fetched for escalates a net refresh while the rail is open. The TTL
  key is ``(root, branch)`` so the new branch was always TTL-clean --
  nothing *triggered* the refetch, so a checkout kept painting the previous
  branch's PR and checks as if they were the new branch's. (task-13
  addition C.)

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
        # (root, branch, fetched_at) for the last DISPATCHED net fetch.
        self._net_fetched_at: tuple[str, str | None, datetime] | None = None
        self._failures: dict[str, int] = {_LOCAL_TIER: 0, _NET_TIER: 0}

        # The root of the most recently LANDED local result, recorded as
        # the accessor's own raw string (never GitEnvState.root -- see
        # module docstring). All root-change comparisons must use this.
        self._landed_root: str | None = None

        # A net refresh requested before the local tier has landed for the
        # CURRENT root defers until `_land` supplies a known branch (C1).
        self._net_pending = False
        self._net_pending_force = False
        self._net_pending_scope: str | None = None

        # Per-tier monotonic dispatch counters (I4): a landing is honored
        # only if its token still matches the tier's latest dispatch.
        self._dispatch_tokens: dict[str, int] = {_LOCAL_TIER: 0, _NET_TIER: 0}

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

        Detects a workspace-root change since the last LANDED local
        result (comparing against ``_landed_root``, never the
        git-resolved ``snapshot.git.root``) and, when one happened, resets
        backoff/TTL state and does a full (both-tier) refresh instead of a
        local-only poll.
        """
        root = self._workspace_root_accessor()
        if root is None or not self._rail_open_accessor():
            return
        last_root = self._landed_root
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

    def _next_token(self, tier: str) -> int:
        self._dispatch_tokens[tier] += 1
        return self._dispatch_tokens[tier]

    def _dispatch_local(self, scope_root: str) -> None:
        previous_git = self.snapshot.git
        token = self._next_token(_LOCAL_TIER)

        def job() -> None:
            git_result = gather_git_env(Path(scope_root), previous=previous_git)
            tasks_result = self._scanner.scan(Path(scope_root), git_result.branch)
            self._marshal_to_ui(
                self._land, scope_root, _LOCAL_TIER, (git_result, tasks_result), token
            )

        self._run_worker(job, thread=True, exclusive=True, group=self.LOCAL_WORKER_GROUP)

    def _dispatch_net(self, scope_root: str, *, force_net: bool) -> None:
        if self._landed_root != scope_root:
            # The local tier hasn't landed for this root yet, so the
            # current branch is unknown (or stale from a different root).
            # Defer: `_land` re-issues this once local lands for `scope_root`.
            if self._net_pending_scope != scope_root:
                self._net_pending_force = force_net
            else:
                self._net_pending_force = self._net_pending_force or force_net
            self._net_pending = True
            self._net_pending_scope = scope_root
            return

        branch = self.snapshot.git.branch
        key = (scope_root, branch)
        if not force_net and self._net_fetched_at is not None:
            prev_root, prev_branch, fetched_at = self._net_fetched_at
            if (prev_root, prev_branch) == key and (self._now() - fetched_at) < self._NET_TTL:
                return  # TTL still holds
        previous_pr = self.snapshot.pr
        self._net_fetched_at = (scope_root, branch, self._now())
        token = self._next_token(_NET_TIER)

        def job() -> None:
            pr_result = gather_pr_env(Path(scope_root), branch, previous=previous_pr)
            self._marshal_to_ui(self._land, scope_root, _NET_TIER, pr_result, token)

        self._run_worker(job, thread=True, exclusive=True, group=self.NET_WORKER_GROUP)

    # -- landing (runs on the UI thread, via marshal_to_ui) ------------------

    def _clear_net_pending(self) -> None:
        self._net_pending = False
        self._net_pending_force = False
        self._net_pending_scope = None

    def _land(self, scope_root: str, tier: str, result: Any, token: int) -> None:
        if scope_root != self._workspace_root_accessor():
            # Stale-scope guard: a newer refresh already superseded this one.
            # B2: a net request deferred against THIS scope was waiting on
            # exactly this landing to supply its branch, so nothing can ever
            # re-issue it -- and leaving it latched lets its accumulated
            # `force_net` leak into whichever request re-keys the slot next,
            # silently bypassing the 60s TTL. Drop it; the next rail
            # open/refresh for the live scope requests its own fetch.
            if self._net_pending and self._net_pending_scope == scope_root:
                self._clear_net_pending()
            return
        if token != self._dispatch_tokens[tier]:
            return  # stale-dispatch guard: a newer dispatch of this tier already landed/is in flight

        if tier == _LOCAL_TIER:
            git_result, tasks_result = result
            self._record_outcome(_LOCAL_TIER, git_result.availability)
            self.snapshot = dataclasses.replace(self.snapshot, git=git_result, tasks=tasks_result)
            self._landed_root = scope_root
            if self._net_pending and self._net_pending_scope == scope_root:
                force = self._net_pending_force
                self._clear_net_pending()
                # B1: the rail-open guard `request_refresh` applied when this
                # was requested has to be re-applied HERE -- the deferral
                # means an unknown amount of time passed, and the user may
                # have collapsed the Inspect rail since. Without this a
                # deferred `gh` fetch fires for a panel nobody is looking at.
                if self._rail_open_accessor() and (
                    force or self._failures[_NET_TIER] < self._MAX_FAILURES
                ):
                    self._dispatch_net(scope_root, force_net=force)
            elif self._branch_change_needs_net(scope_root):
                self._dispatch_net(scope_root, force_net=False)
        else:
            self._record_outcome(_NET_TIER, result.availability)
            self.snapshot = dataclasses.replace(self.snapshot, pr=result)

        self._on_snapshot(self.snapshot)

    def _branch_change_needs_net(self, scope_root: str) -> bool:
        """Whether a just-landed local branch invalidates the fetched PR state.

        C (task-13): the net TTL is keyed ``(root, branch)``, so a branch
        change is already TTL-clean -- the defect was that *nothing
        triggered* the refetch. A checkout with the rail open therefore kept
        painting the previous branch's PR number, title, and check results
        as if they belonged to the new branch, until something else (rail
        reopen, explicit Refresh) happened to bust it. Showing another
        branch's PR unmarked is worse than showing none.

        Only escalates when a net fetch has actually been made (there is a
        branch to compare against); the first fetch stays owned by the
        rail-open/refresh paths, and the rail-open and backoff guards still
        apply here exactly as they do on the request path.

        Args:
            scope_root: The workspace root this landing belongs to.

        Returns:
            ``True`` when a net refresh should be dispatched now.
        """
        if self._net_fetched_at is None:
            return False
        if self.snapshot.git.availability is not EnvSourceAvailability.OK:
            # An ERROR/NOT_APPLICABLE local result carries ``branch=None``,
            # which is a missing answer, not a branch change: escalating on
            # it would spend a `gh` call describing nothing.
            return False
        fetched_root, fetched_branch, _ = self._net_fetched_at
        if (fetched_root, fetched_branch) == (scope_root, self.snapshot.git.branch):
            return False
        if not self._rail_open_accessor():
            return False
        return self._failures[_NET_TIER] < self._MAX_FAILURES

    def _record_outcome(self, tier: str, availability: EnvSourceAvailability) -> None:
        if availability is EnvSourceAvailability.ERROR:
            self._failures[tier] += 1
        else:
            self._failures[tier] = 0
