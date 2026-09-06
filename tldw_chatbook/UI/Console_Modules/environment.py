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
``_MAX_FAILURES`` in a row, until a workspace-root change or an explicit
user refresh (``force_net``) resets it. Both of those reset BOTH tiers:
the counters exist to stop an automatic 10s flap loop, not to make a
deliberate keypress inert, and a local tier that could only be revived by
switching workspaces left the panel stuck on its error row with a Refresh
slot that did nothing.

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
- A ``None`` workspace root is a landable state, not a skip (TASK-31660).
  ``poll_tick``/``request_refresh`` used to return early on it, so nothing
  landed and the last paint stood: after a switch to a workspace that binds
  no folder the panel kept the PREVIOUS repository's branch and counts and
  still offered "Commit or push · N files" against it, permanently, with an
  inert Refresh. They now land an explicit UNBOUND snapshot (all three
  tiers, so no half-state can keep another repo's PR), through the ordinary
  ``_land``/``on_snapshot`` path with ``scope_root=None``. Because ``None``
  is a real landed scope, "nothing has landed yet" needs its own flag
  (``_has_landed``) -- see ``_landed_root``'s comment.
- A local landing whose branch differs from the one the current PR state
  was fetched for escalates a net refresh while the rail is open. The TTL
  key is ``(root, branch)`` so the new branch was always TTL-clean --
  nothing *triggered* the refetch, so a checkout kept painting the previous
  branch's PR and checks as if they were the new branch's. (task-13
  addition C.)
- A bound -> bound root change resets the PR tier to ``PENDING`` in the
  same landing that replaces git/tasks (TASK-31665 AC#10). The per-field
  ``dataclasses.replace`` used to keep ``pr``, so between the fast local
  landing and the slow ``gh`` one the panel showed the NEW root's branch
  and counts beside the OLD root's PR number and checks.
- ``UNKNOWN_ROOT`` still skips -- but no longer forever (TASK-31665
  AC#11). ``poll_tick`` counts consecutive undetermined observations and,
  once ``_MAX_UNKNOWN_TICKS`` have passed with nothing EVER having landed,
  lands an explicit ``EnvSourceAvailability.UNKNOWN`` snapshot once
  (``_land_unknown``); an explicit Refresh in that same never-landed state
  lands it immediately. This is a THIRD state, distinct from both
  neighbours: ``PENDING`` promises an answer is coming (something was
  dispatched), ``UNBOUND`` asserts a determined "nothing is bound", and
  ``UNKNOWN`` says the root itself could not be named -- no chat
  controller, or no active session. A panel that has real data keeps it:
  an undetermined root is not evidence the last answer went stale, and
  neither the failure counters nor ``_landed_root``/``_has_landed`` are
  touched by the unknown landing.

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
    PrEnvState,
    unbound_snapshot,
    unknown_snapshot,
)
from tldw_chatbook.Workspaces.environment_status import (
    BacklogTaskScanner,
    gather_git_env,
    gather_pr_env,
)

_LOCAL_TIER = "local"
_NET_TIER = "net"


class UnknownRoot:
    """Type of the :data:`UNKNOWN_ROOT` sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNKNOWN_ROOT"


#: The workspace root could not be DETERMINED -- distinct from ``None``,
#: which means it was determined to be "no folder is bound".
#:
#: TASK-31660 round 1 (review): the root accessor chain already draws this
#: line and the screen seam was throwing it away. `resolve_turn_execution_
#: context(...).workspace_roots` returns ``()`` for a genuinely unbound
#: workspace, but the two layers above it return ``None`` for "I cannot
#: tell" -- an exception inside the roots accessor
#: (`review_selection.py`), or the chat controller not being built yet /
#: having no active session (`wiring.py`). Collapsing both into ``None``
#: was survivable while ``None`` merely SKIPPED; once ``None`` started
#: landing an emphatic on-screen "No folder is bound to this
#: conversation's workspace", a transient failure or a pre-mount
#: not-ready state would assert something nothing had established -- and
#: would additionally zero the local tier's 3-strike backoff via
#: ``_record_outcome``, which is the counter that stops a 10s flap loop.
#:
#: An accessor returning this keeps the OLD behaviour exactly: no
#: dispatch, no landing, no counter touched, previous paint stands.
UNKNOWN_ROOT = UnknownRoot()


class ConsoleEnvironmentController:
    """Cadence/TTL/backoff orchestrator for the Environment panel's data."""

    LOCAL_WORKER_GROUP = "console-environment-local"
    NET_WORKER_GROUP = "console-environment-net"

    _MAX_FAILURES = 3
    _NET_TTL = timedelta(seconds=60)
    #: Consecutive UNKNOWN_ROOT polls before the panel says so out loud
    #: (TASK-31665 AC#11). Three 10s ticks ~= 30s: long enough that a
    #: pre-mount/not-ready blip on the accessor chain never reaches the
    #: screen, short enough that a genuinely session-less Console stops
    #: promising "Checking workspace…" inside a minute.
    _MAX_UNKNOWN_TICKS = 3

    def __init__(
        self,
        *,
        run_worker: Callable[..., Any],
        marshal_to_ui: Callable[..., None],
        workspace_root_accessor: Callable[[], "str | None | UnknownRoot"],
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
        #
        # ``None`` is now a REAL landed value ("the workspace binds no
        # folder"), so it can no longer double as "nothing has landed yet"
        # -- `_has_landed` carries that. Conflating them would make the
        # unbound->bound recovery unreachable: the root-change branch in
        # `poll_tick` would read the rebind as "first landing" and never
        # reset the net TTL that still belongs to the old root.
        self._landed_root: str | None = None
        self._has_landed = False

        # A net refresh requested before the local tier has landed for the
        # CURRENT root defers until `_land` supplies a known branch (C1).
        self._net_pending = False
        self._net_pending_force = False
        self._net_pending_scope: str | None = None

        # Per-tier monotonic dispatch counters (I4): a landing is honored
        # only if its token still matches the tier's latest dispatch.
        self._dispatch_tokens: dict[str, int] = {_LOCAL_TIER: 0, _NET_TIER: 0}

        # TASK-31665 AC#11: consecutive UNKNOWN_ROOT observations. An
        # undetermined root still SKIPS (see `UNKNOWN_ROOT`'s comment), but
        # skipping forever left the panel on PENDING's "Checking workspace…"
        # with an inert Refresh for the life of the screen whenever the
        # cause was structural rather than transient (no chat controller, no
        # active session). After `_MAX_UNKNOWN_TICKS` polls with nothing
        # ever having landed, an explicit UNKNOWN state lands ONCE.
        self._unknown_ticks = 0
        self._unknown_landed = False

        # TASK-31664 AC#3, round-1 review I1/I2: tiers the most recent
        # EXPLICIT (``force_net``) refresh is still owed. Populated by
        # ``request_refresh`` with exactly the tiers it decided to pursue,
        # BEFORE dispatching them; discarded in ``_land`` as each one
        # genuinely lands (or is abandoned -- see the deferred-net-dropped
        # comment in ``_land``). Empty almost all the time; the screen
        # reads ``pending_ack_tiers`` to know when to clear the
        # "Refreshing…" acknowledgment, which must survive until the SLOW
        # tier (``gh``, up to the measured ~12s) lands -- not just the
        # fast local one.
        self._pending_ack_tiers: set[str] = set()

    # -- public API -------------------------------------------------------

    @property
    def pending_ack_tiers(self) -> frozenset[str]:
        """Tiers the most recent explicit refresh press is still owed.

        Empty except during the window between an explicit Refresh press
        and every tier it dispatched actually landing (or being abandoned
        -- see ``_land``). The screen clears its transient "Refreshing…"
        acknowledgment only once this is empty again.
        """
        return frozenset(self._pending_ack_tiers)

    def request_refresh(self, *, include_net: bool = False, force_net: bool = False) -> None:
        """Dispatch a local refresh (and optionally a net refresh).

        Args:
            include_net: Whether the expensive ``gh`` tier is in scope.
            force_net: This is an EXPLICIT user refresh (the Environment
                header's "Refresh" slot). It busts the net TTL *and*
                revives both tiers' backoff pauses -- see below.

        ``force_net`` clears the pause counter for BOTH tiers, not just the
        net one. The spec and the shipped user guide both promise a paused
        source stops polling "until manual refresh or scope change", but the
        force path used to bypass only the net tier's counter: a local tier
        that had hit 3 consecutive ERRORs (a slow `git status` in a large
        tree timing out three times) was unrecoverable for the life of the
        screen unless the workspace root changed, and the panel sat on the
        ERROR row with a Refresh slot that did nothing. The counters exist
        to stop a 10s *automatic* flap loop; a deliberate keypress is not
        that loop.

        A ``None`` root is an ANSWER ("no folder is bound"), not a reason to
        skip: it lands an explicit UNBOUND snapshot. The old early return
        meant nothing landed and the LAST PAINT STOOD -- after a switch to
        an unbound workspace the panel kept the previous repository's branch
        and counts and still offered "Commit or push · N files" against it,
        permanently, because this method (and `poll_tick`) were the only
        things that could ever have replaced them. It also makes the
        Refresh slot re-check the binding rather than be a visible no-op
        (TASK-31660 AC #4).

        TASK-31664 round-1 review I1/I2: ``force_net`` is also the signal
        this is the ack-worthy explicit press (every production caller of
        ``force_net=True`` is the Refresh button; ``poll_tick`` and
        ``notify_rail_opened`` never pass it). ``pending_ack_tiers`` is
        recorded here, BEFORE any dispatch below, with EXACTLY the tiers
        this call decided to pursue -- never guessed after the fact from
        whether a dispatch "looks like" it happened, so a call that turns
        out to do nothing (``UNKNOWN_ROOT``, rail closed, both tiers
        backed off) simply never touches it, and the screen's post-call
        check of ``pending_ack_tiers`` correctly finds nothing to wait for.
        """
        root = self._workspace_root_accessor()
        if not self._rail_open_accessor():
            return
        if root is UNKNOWN_ROOT:
            # "Cannot tell" is not an answer: skip silently, exactly as this
            # method did for every falsy root before TASK-31660. Landing
            # UNBOUND here would assert "no folder is bound" on the strength
            # of a swallowed exception or a not-yet-built chat controller,
            # AND would reset the local tier's failure counter.
            #
            # TASK-31665 AC#11: a DELIBERATE Refresh press against a panel
            # that has never had an answer is the one case where staying
            # silent is worse than saying so -- the user just asked, and
            # the honest reply is "there is no session to look at". Gated
            # on `not self._has_landed` so a press during a transient blip
            # never wipes a good paint, and it still touches no counter.
            if force_net and not self._has_landed:
                self._land_unknown()
            return
        self._unknown_ticks = 0
        self._unknown_landed = False
        if root is None:
            if force_net:
                self._failures = {_LOCAL_TIER: 0, _NET_TIER: 0}
                # I2: `_land_unbound` lands SYNCHRONOUSLY (see its own
                # docstring), so `_land` will already have discarded this
                # again by the time this method returns -- the screen's
                # post-call check of `pending_ack_tiers` correctly finds it
                # empty and never arms an ack nothing will ever clear.
                self._pending_ack_tiers = {_LOCAL_TIER}
            self._land_unbound()
            return
        scope_root = root

        if force_net:
            self._failures = {_LOCAL_TIER: 0, _NET_TIER: 0}

        dispatch_local = self._failures[_LOCAL_TIER] < self._MAX_FAILURES
        dispatch_net = include_net and (
            force_net or self._failures[_NET_TIER] < self._MAX_FAILURES
        )
        if force_net:
            pending: set[str] = set()
            if dispatch_local:
                pending.add(_LOCAL_TIER)
            if dispatch_net:
                pending.add(_NET_TIER)
            self._pending_ack_tiers = pending

        if dispatch_local:
            self._dispatch_local(scope_root)

        if dispatch_net:
            self._dispatch_net(scope_root, force_net=force_net)

    def poll_tick(self) -> None:
        """Called every 10s by the screen timer: local tier only.

        Detects a workspace-root change since the last LANDED local
        result (comparing against ``_landed_root``, never the
        git-resolved ``snapshot.git.root``) and, when one happened, resets
        backoff/TTL state and does a full (both-tier) refresh instead of a
        local-only poll.

        TASK-31660: the change detection is gated on ``_has_landed`` rather
        than on ``_landed_root is not None``, so BOTH directions across the
        unbound boundary are genuine root changes -- ``root -> None`` (which
        must wipe the previous repository's paint within one tick, AC #3)
        and ``None -> root`` (which must retire the old root's ``gh`` TTL
        and re-fetch both tiers).
        """
        root = self._workspace_root_accessor()
        if not self._rail_open_accessor():
            return
        if root is UNKNOWN_ROOT:
            # See `request_refresh`: an undetermined root must not be
            # mistaken for a root CHANGE either, or a transient accessor
            # failure would wipe the net TTL and re-fetch `gh` on every
            # 10s tick for as long as it lasts.
            #
            # TASK-31665 AC#11: but it must not be mistaken for a temporary
            # condition FOREVER either. Count the ticks; once the condition
            # has persisted and nothing has ever landed, say so once. A
            # panel that HAS landed real data keeps it -- an undetermined
            # root is not evidence the last answer went stale.
            self._unknown_ticks += 1
            if (
                self._unknown_ticks >= self._MAX_UNKNOWN_TICKS
                and not self._has_landed
                and not self._unknown_landed
            ):
                self._land_unknown()
            return
        self._unknown_ticks = 0
        self._unknown_landed = False
        if self._has_landed and root != self._landed_root:
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

    def _land_unbound(self) -> None:
        """Land the "no folder is bound" state through the normal path.

        Deliberately calls ``_land`` DIRECTLY rather than going through
        ``_marshal_to_ui``: this runs on the UI thread already (it is
        reached from ``poll_tick``/``request_refresh``, both timer/handler
        callbacks), and the production marshal is
        ``app.call_from_thread``, which raises when called from the app's
        own thread. Everything else about the landing is the ordinary path
        -- the stale-scope guard, the dispatch token, ``_record_outcome``,
        and ``on_snapshot`` -- so an UNBOUND state is superseded and
        supersedes exactly like a gathered one.

        The scope is ``None``, which the stale-scope guard now accepts as a
        landable scope (it compares against the accessor, which is what
        just returned ``None``).
        """
        unbound = unbound_snapshot()
        self._land(
            None,
            _LOCAL_TIER,
            (unbound.git, unbound.tasks),
            self._next_token(_LOCAL_TIER),
        )

    def _land_unknown(self) -> None:
        """Paint the "root could not be determined" state, once (AC#11).

        Deliberately does NOT go through ``_land``: there is no scope to
        pass its stale-scope guard (the accessor returns the
        ``UNKNOWN_ROOT`` sentinel, not a ``str | None``), no dispatch to
        supersede, and nothing about this landing should touch
        ``_landed_root``/``_has_landed`` -- it is a statement that nothing
        has been established, so it must not read as an establishment. The
        failure counters are likewise untouched, for the same reason the
        skip path never touched them: this is not a gatherer outcome.

        Idempotent within one UNKNOWN spell: ``_unknown_landed`` is cleared
        the moment the accessor returns a real answer again.
        """
        self._unknown_landed = True
        self.snapshot = unknown_snapshot(target=self.snapshot.target)
        self._on_snapshot(self.snapshot)

    def _land(self, scope_root: str | None, tier: str, result: Any, token: int) -> None:
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
            # TASK-31664 round-1 review I2: the scope an outstanding
            # explicit refresh targeted has moved on -- nothing will ever
            # land for it through THIS guard again, so any acknowledgment
            # still waiting on it would otherwise wedge forever. The next
            # legitimate landing (for the NEW scope, imminent -- a root
            # change always triggers its own immediate refresh) will find
            # this already empty and clear the screen's ack right away.
            self._pending_ack_tiers.clear()
            return
        if token != self._dispatch_tokens[tier]:
            return  # stale-dispatch guard: a newer dispatch of this tier already landed/is in flight

        # I1: discard BEFORE the tier-specific branch below, unconditionally
        # -- a landing counts for the acknowledgment regardless of what it
        # lands AS (UNBOUND, ERROR, or a real result all satisfy "this
        # tier is now current").
        self._pending_ack_tiers.discard(tier)

        if tier == _LOCAL_TIER:
            git_result, tasks_result = result
            self._record_outcome(_LOCAL_TIER, git_result.availability)
            if git_result.availability is EnvSourceAvailability.UNBOUND:
                # No root means no repository, so the PR tier's data belongs
                # to the root that just went away: a per-field
                # `dataclasses.replace` would mark git UNBOUND while leaving
                # ANOTHER repository's PR number and check results painted.
                # Replace the whole snapshot (keeping the exec target, which
                # is a session property, not a workspace one) and retire the
                # net tier's TTL/pending bookkeeping with it, so a later
                # rebind re-fetches instead of inheriting a stale 60s window
                # keyed on the old (root, branch).
                self.snapshot = unbound_snapshot(target=self.snapshot.target)
                self._net_fetched_at = None
                self._clear_net_pending()
            elif self._has_landed and scope_root != self._landed_root:
                # TASK-31665 AC#10 (TASK-31660 round-1 review finding). A
                # bound -> bound switch takes this per-field branch, which
                # replaced git and tasks and KEPT `pr`. The local tier is
                # fast and the `gh` tier is not (measured ~12s), so for the
                # whole deferred-fetch window the panel painted the NEW
                # root's branch and counts beside the OLD root's PR number,
                # title and check results -- the same "another repository's
                # data, unmarked" defect TASK-31660 fixed for the
                # bound -> unbound direction, one direction over. The PR
                # tier goes back to PENDING (which `project_environment_
                # section` renders as no PR rows at all, never as "no PR"),
                # and its TTL/pending bookkeeping retires with it so the
                # new root re-fetches instead of inheriting the old 60s
                # window keyed on the old `(root, branch)`.
                self.snapshot = dataclasses.replace(
                    self.snapshot,
                    git=git_result,
                    tasks=tasks_result,
                    pr=PrEnvState(availability=EnvSourceAvailability.PENDING),
                )
                self._net_fetched_at = None
            else:
                self.snapshot = dataclasses.replace(
                    self.snapshot, git=git_result, tasks=tasks_result
                )
            self._landed_root = scope_root
            self._has_landed = True
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
                else:
                    # TASK-31664 round-1 review I2: the deferred net fetch
                    # is being ABANDONED (rail closed, or backed off since
                    # it was requested) -- nothing will ever land for it,
                    # so an outstanding acknowledgment must not wait
                    # forever for a fetch that is never coming.
                    self._pending_ack_tiers.discard(_NET_TIER)
            elif self._branch_change_needs_net(scope_root):
                self._dispatch_net(scope_root, force_net=False)
        else:
            self._record_outcome(_NET_TIER, result.availability)
            self.snapshot = dataclasses.replace(self.snapshot, pr=result)

        self._on_snapshot(self.snapshot)

    def _branch_change_needs_net(self, scope_root: str | None) -> bool:
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
            scope_root: The workspace root this landing belongs to, or
                ``None`` for the UNBOUND landing.

        Returns:
            ``True`` when a net refresh should be dispatched now.
        """
        if scope_root is None:
            # The UNBOUND landing reaches here through `_land`'s `elif`.
            # There is no root, so there is no branch that could have
            # changed and no `gh` fetch that could describe one. Stated
            # explicitly rather than left to the `_net_fetched_at is None`
            # check below: that only holds because `_land` nulls
            # `_net_fetched_at` two lines earlier, which makes this
            # function's safety a property of its CALLER's statement order
            # (review finding, TASK-31660 round 1).
            return False
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
