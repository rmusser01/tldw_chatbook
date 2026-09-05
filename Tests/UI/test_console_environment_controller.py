# Tests/UI/test_console_environment_controller.py
"""Controller cadence/TTL/backoff tests — no Textual app, synchronous fakes."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import tldw_chatbook.UI.Console_Modules.environment as env_mod
from tldw_chatbook.Chat.console_environment_state import (
    EnvironmentSnapshot, EnvSourceAvailability, GitEnvState, PrEnvState, TasksEnvState,
)
from tldw_chatbook.UI.Console_Modules.environment import ConsoleEnvironmentController
from tldw_chatbook.Workspaces.environment_status import BacklogTaskScanner


class Fixture:
    def __init__(self, monkeypatch, *, root="/w/repo", rail_open=True):
        self.dispatched: list[dict] = []
        self.snapshots: list[EnvironmentSnapshot] = []
        self.root: str | None = root
        self.rail_open = rail_open
        self.clock = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        self.git_calls = 0
        self.pr_calls = 0

        def fake_git(path, previous=None):
            self.git_calls += 1
            return GitEnvState(availability=EnvSourceAvailability.OK,
                               root=str(path), branch="feat/task-1-x")

        def fake_pr(path, branch, runner=None, previous=None):
            self.pr_calls += 1
            return PrEnvState(availability=EnvSourceAvailability.OK, number=7,
                              title="T", state="OPEN", url="https://x/pull/7")

        monkeypatch.setattr(env_mod, "gather_git_env", fake_git)
        monkeypatch.setattr(env_mod, "gather_pr_env", fake_pr)
        monkeypatch.setattr(
            BacklogTaskScanner, "scan",
            lambda scanner, ws, branch: TasksEnvState(
                availability=EnvSourceAvailability.NOT_APPLICABLE),
        )

        def run_worker(fn, **kwargs):
            self.dispatched.append(kwargs)
            fn()  # synchronous: the "worker" runs inline

        self.controller = ConsoleEnvironmentController(
            run_worker=run_worker,
            marshal_to_ui=lambda fn, *a: fn(*a),
            workspace_root_accessor=lambda: self.root,
            rail_open_accessor=lambda: self.rail_open,
            on_snapshot=self.snapshots.append,
            now=lambda: self.clock,
        )


def test_no_dispatch_while_rail_closed(monkeypatch):
    fx = Fixture(monkeypatch, rail_open=False)
    fx.controller.request_refresh(include_net=True)
    assert fx.dispatched == []


def test_local_and_net_use_distinct_worker_groups(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    groups = {d["group"] for d in fx.dispatched}
    assert groups == {ConsoleEnvironmentController.LOCAL_WORKER_GROUP,
                      ConsoleEnvironmentController.NET_WORKER_GROUP}
    assert all(d["thread"] is True and d["exclusive"] is True for d in fx.dispatched)


def test_net_ttl_suppresses_refetch_within_60s_and_force_busts_it(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    fx.clock += timedelta(seconds=30)
    fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 1  # TTL held
    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.pr_calls == 2
    fx.clock += timedelta(seconds=61)
    fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 3


def test_three_failures_pause_the_local_tier_until_forced(monkeypatch):
    fx = Fixture(monkeypatch)
    monkeypatch.setattr(
        env_mod, "gather_git_env",
        lambda path, previous=None: GitEnvState(availability=EnvSourceAvailability.ERROR),
    )
    for _ in range(3):
        fx.controller.poll_tick()
    dispatched_before = len(fx.dispatched)
    fx.controller.poll_tick()  # paused: no new dispatch
    assert len(fx.dispatched) == dispatched_before


def _fail_git(fx, monkeypatch):
    """Replace the git gatherer with an always-ERROR one that still counts.

    ``Fixture``'s own fake owns ``git_calls``; a bare ``monkeypatch.setattr``
    lambda silently stops incrementing it, so a test asserting on the count
    reads 0 and proves nothing about the pause.
    """
    def fail_git(path, previous=None):
        fx.git_calls += 1
        return GitEnvState(availability=EnvSourceAvailability.ERROR)

    monkeypatch.setattr(env_mod, "gather_git_env", fail_git)


def test_forced_refresh_revives_a_paused_local_tier(monkeypatch):
    """A backed-off LOCAL tier must come back on the Refresh slot (F3a).

    The Refresh tail posts ``request_refresh(include_net=True,
    force_net=True)``. Before this fix ``force_net`` bypassed only the net
    tier's counter, so three consecutive local ERRORs left the panel stuck
    on its error row for the life of the screen -- contradicting both the
    spec and the shipped user guide ("until manual refresh or scope
    change").
    """
    fx = Fixture(monkeypatch)
    _fail_git(fx, monkeypatch)
    for _ in range(3):
        fx.controller.poll_tick()
    assert fx.git_calls == 3
    fx.controller.poll_tick()
    assert fx.git_calls == 3  # paused

    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.git_calls == 4  # revived by the explicit refresh
    # ... and an ordinary poll works again from there (counter really reset,
    # not merely bypassed for the one forced call).
    def ok_git(path, previous=None):
        fx.git_calls += 1
        return GitEnvState(availability=EnvSourceAvailability.OK,
                           root=str(path), branch="feat/x")

    monkeypatch.setattr(env_mod, "gather_git_env", ok_git)
    fx.controller.poll_tick()
    assert fx.git_calls == 5


def test_forced_refresh_revives_a_paused_net_tier_too(monkeypatch):
    fx = Fixture(monkeypatch)

    def fail_pr(path, branch, runner=None, previous=None):
        fx.pr_calls += 1
        return PrEnvState(availability=EnvSourceAvailability.ERROR)

    monkeypatch.setattr(env_mod, "gather_pr_env", fail_pr)
    for _ in range(3):
        fx.clock += timedelta(seconds=61)
        fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 3
    fx.clock += timedelta(seconds=61)
    fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 3  # paused
    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.pr_calls == 4


def test_unforced_refresh_does_not_revive_a_paused_tier(monkeypatch):
    """Negative control: only the FORCED path clears the counters."""
    fx = Fixture(monkeypatch)
    _fail_git(fx, monkeypatch)
    for _ in range(4):
        fx.controller.poll_tick()
    assert fx.git_calls == 3
    fx.controller.request_refresh(include_net=True)  # no force
    assert fx.git_calls == 3


def test_stale_scope_snapshot_is_dropped_when_root_changes_mid_flight(monkeypatch):
    fx = Fixture(monkeypatch)

    def run_worker_scope_shift(fn, **kwargs):
        fx.root = "/other/repo"  # root changes while the "worker" runs
        fn()

    fx.controller._run_worker = run_worker_scope_shift
    fx.controller.request_refresh()
    assert fx.snapshots == []  # landed result discarded by the stale-scope guard


def test_poll_tick_dispatches_local_only(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.poll_tick()
    assert fx.git_calls == 1 and fx.pr_calls == 0


def test_rail_open_dispatches_both_tiers(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.notify_rail_opened()
    assert fx.git_calls == 1 and fx.pr_calls == 1


# ---------------------------------------------------------------------------
# Deferred-fake tests (controller ruling: authorized extension beyond the
# plan's verbatim set). `Fixture` above runs each dispatched job SYNCHRONOUSLY
# at dispatch time, which hides bugs that only surface under real async
# ordering -- a job landing after a NEWER dispatch of the same tier, or a net
# dispatch being evaluated before the local tier has landed. `DeferredFixture`
# queues jobs instead of running them, so tests can land them in whatever
# order they choose.
# ---------------------------------------------------------------------------


class DeferredFixture:
    def __init__(self, monkeypatch, *, root="/w/repo", rail_open=True):
        self.dispatched: list[dict] = []
        self.jobs: list = []  # queued job callables, in dispatch order
        self.snapshots: list[EnvironmentSnapshot] = []
        self.root: str | None = root
        self.rail_open = rail_open
        self.clock = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        self.git_calls = 0
        self.pr_calls = 0
        self.pr_branches_seen: list[str | None] = []

        def fake_git(path, previous=None):
            self.git_calls += 1
            return GitEnvState(availability=EnvSourceAvailability.OK,
                               root=str(path), branch=f"feat/call-{self.git_calls}")

        def fake_pr(path, branch, runner=None, previous=None):
            self.pr_calls += 1
            self.pr_branches_seen.append(branch)
            return PrEnvState(availability=EnvSourceAvailability.OK, number=7,
                              title="T", state="OPEN", url="https://x/pull/7")

        monkeypatch.setattr(env_mod, "gather_git_env", fake_git)
        monkeypatch.setattr(env_mod, "gather_pr_env", fake_pr)
        monkeypatch.setattr(
            BacklogTaskScanner, "scan",
            lambda scanner, ws, branch: TasksEnvState(
                availability=EnvSourceAvailability.NOT_APPLICABLE),
        )

        def run_worker(fn, **kwargs):
            self.dispatched.append(kwargs)
            self.jobs.append(fn)  # queued -- NOT run inline

        self.controller = ConsoleEnvironmentController(
            run_worker=run_worker,
            marshal_to_ui=lambda fn, *a: fn(*a),
            workspace_root_accessor=lambda: self.root,
            rail_open_accessor=lambda: self.rail_open,
            on_snapshot=self.snapshots.append,
            now=lambda: self.clock,
        )

    def run_job(self, index: int) -> None:
        self.jobs[index]()


def test_net_dispatch_waits_for_landed_branch_on_first_open(monkeypatch):
    """C1: on first rail open the local job hasn't landed when net would be
    dispatched, so the branch is unknown. The net dispatch must defer until
    local lands, then fire with the freshly-landed branch (never None)."""
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    assert len(fx.jobs) == 1  # net can't dispatch yet -- branch unknown
    assert fx.pr_calls == 0

    fx.run_job(0)  # land the local job: branch becomes known
    assert len(fx.jobs) == 2  # the deferred net job is now queued
    assert fx.pr_calls == 0  # not run yet

    fx.run_job(1)  # land the (previously deferred) net job
    assert fx.pr_calls == 1
    assert fx.pr_branches_seen == ["feat/call-1"]  # landed branch, never None
    assert fx.controller.snapshot.pr.availability is EnvSourceAvailability.OK
    assert fx.controller.snapshot.pr.number == 7


def test_root_spelling_difference_does_not_defeat_ttl(monkeypatch):
    """I2: root-change detection must compare against the accessor's own raw
    string (`_landed_root`), never git's RESOLVED toplevel (`GitEnvState.root`),
    or a spelling difference (e.g. /tmp vs /private/tmp) makes every poll
    misdetect a root change, wipe `_net_fetched_at`, and re-dispatch net."""
    fx = DeferredFixture(monkeypatch, root="/tmp/wt")

    def fake_git_resolved(path, previous=None):
        fx.git_calls += 1
        return GitEnvState(availability=EnvSourceAvailability.OK,
                           root="/private/tmp/wt",  # resolved -- different
                           branch="feat/task-1-x")  # spelling than the accessor

    monkeypatch.setattr(env_mod, "gather_git_env", fake_git_resolved)

    fx.controller.notify_rail_opened()
    fx.run_job(0)  # land local -> branch known -> deferred net job queued
    assert len(fx.jobs) == 2
    fx.run_job(1)  # land net
    assert fx.pr_calls == 1

    def net_dispatch_count() -> int:
        return sum(
            1 for d in fx.dispatched if d["group"] == ConsoleEnvironmentController.NET_WORKER_GROUP
        )

    net_dispatches_before = net_dispatch_count()
    for _ in range(5):
        fx.controller.poll_tick()
    assert net_dispatch_count() == net_dispatches_before  # no false root-change reset fired
    assert fx.pr_calls == 1  # gh called at most once inside the TTL window


def test_local_tier_resumes_after_root_change_following_pause(monkeypatch):
    """I3: an ERROR GitEnvState carries root="" so a paused local tier can
    only resume via a genuine root change, detected against `_landed_root`
    (set unconditionally on every local landing, success or failure)."""
    fx = Fixture(monkeypatch)
    monkeypatch.setattr(
        env_mod, "gather_git_env",
        lambda path, previous=None: GitEnvState(availability=EnvSourceAvailability.ERROR),
    )
    for _ in range(3):
        fx.controller.poll_tick()
    assert fx.controller._failures["local"] == 3

    def local_dispatch_count() -> int:
        return sum(
            1 for d in fx.dispatched if d["group"] == ConsoleEnvironmentController.LOCAL_WORKER_GROUP
        )

    local_dispatches_before = local_dispatch_count()
    fx.root = "/other/repo"
    fx.controller.poll_tick()
    assert local_dispatch_count() > local_dispatches_before  # pause lifted by the root change
    assert fx.controller._failures["local"] < 3  # counters were reset, not left at the cap


def test_stale_local_landing_is_dropped_by_dispatch_token(monkeypatch):
    """I4: a per-tier monotonic dispatch token drops a landing that arrives
    after a NEWER dispatch of the same tier -- regardless of which job's OS
    thread happens to finish (i.e. land) first."""
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh()  # dispatch #1 (local only)
    fx.controller.request_refresh()  # dispatch #2 (local only)
    assert len(fx.jobs) == 2

    fx.run_job(1)  # land the SECOND (newer, highest-token) dispatch first
    newer_branch = fx.controller.snapshot.git.branch

    fx.run_job(0)  # land the FIRST (older, now-stale) dispatch second
    assert fx.controller.snapshot.git.branch == newer_branch  # stale landing dropped
    assert fx.git_calls == 2  # both gathers ran; only the newer one's landing stuck


# ---------------------------------------------------------------------------
# task-13 hardening (additions B and C). B closes two gaps in the deferred
# net path that only exist because the dispatch is deferred at all; C makes a
# branch change actually TRIGGER the net refetch its own TTL key already
# allows for.
# ---------------------------------------------------------------------------


def _net_dispatch_count(fx) -> int:
    return sum(
        1
        for d in fx.dispatched
        if d["group"] == ConsoleEnvironmentController.NET_WORKER_GROUP
    )


def test_deferred_net_dispatch_is_dropped_when_the_rail_closed_meanwhile(monkeypatch):
    """B1: the rail-open guard must be re-checked at RE-dispatch time.

    `request_refresh` refuses to dispatch behind a closed rail, but a net
    request deferred while the branch was unknown is re-issued later, from
    `_land` -- and the rail can have closed in between. Without a second
    check that deferred `gh` fetch fires for a panel nobody is looking at.
    """
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    assert len(fx.jobs) == 1 and _net_dispatch_count(fx) == 0  # net deferred

    fx.rail_open = False  # the user collapsed the Inspect rail meanwhile
    fx.run_job(0)  # local lands: the deferred net would be re-issued here
    assert _net_dispatch_count(fx) == 0
    assert fx.pr_calls == 0
    assert fx.controller._net_pending is False  # dropped, not left latched

    fx.rail_open = True  # reopening still fetches normally
    fx.controller.notify_rail_opened()
    assert _net_dispatch_count(fx) == 1


def test_pending_net_request_is_cleared_when_its_scope_is_dropped(monkeypatch):
    """B2: a pending net request whose local landing is scope-dropped is orphaned.

    `_land`'s stale-scope guard returns before the pending-net block, so the
    request that was waiting on that landing can never be re-issued -- and
    its accumulated `force_net` then leaks into whatever request re-keys the
    slot next, silently bypassing the 60s TTL.
    """
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.controller._net_pending is True
    assert fx.controller._net_pending_force is True

    fx.root = "/other/repo"  # workspace switched before the local job landed
    fx.run_job(0)  # this landing is scope-dropped
    assert fx.controller._net_pending is False
    assert fx.controller._net_pending_force is False
    assert fx.controller._net_pending_scope is None

    # ...so a later PLAIN refresh cannot inherit the dropped request's force.
    fx.controller.request_refresh(include_net=True)
    assert fx.controller._net_pending_scope == "/other/repo"
    assert fx.controller._net_pending_force is False


def test_pending_net_force_accumulates_within_one_scope(monkeypatch):
    """The `_net_pending_scope`/force block's same-scope arm: force is sticky."""
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    assert fx.controller._net_pending_scope == "/w/repo"
    assert fx.controller._net_pending_force is False

    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.controller._net_pending_force is True

    fx.controller.request_refresh(include_net=True)  # must not DOWNGRADE it
    assert fx.controller._net_pending_force is True

    # An in-window TTL entry for the branch that is about to land (the fake
    # gatherer numbers branches by CALL, not by dispatch, so the first job
    # actually run yields `feat/call-1`): only a genuinely FORCED deferred
    # dispatch busts it.
    fx.controller._net_fetched_at = ("/w/repo", "feat/call-1", fx.clock)
    fx.run_job(2)  # land the newest local dispatch (older tokens are stale)
    assert fx.controller._net_pending is False
    fx.run_job(len(fx.jobs) - 1)
    assert fx.pr_calls == 1
    assert fx.pr_branches_seen == ["feat/call-1"]


def test_pending_net_force_is_rekeyed_not_inherited_on_a_new_scope(monkeypatch):
    """The other arm: a DIFFERENT scope resets force rather than inheriting it."""
    fx = DeferredFixture(monkeypatch)
    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.controller._net_pending_force is True

    fx.root = "/other/repo"
    fx.controller.request_refresh(include_net=True)  # new scope, not forced
    assert fx.controller._net_pending_scope == "/other/repo"
    assert fx.controller._net_pending_force is False


def test_branch_change_on_a_local_landing_escalates_a_net_refresh(monkeypatch):
    """C: showing another branch's PR unmarked is the defect this closes.

    The net TTL is keyed `(root, branch)`, so a branch change is already
    TTL-clean -- nothing merely *triggered* the refetch, so the panel kept
    painting the previous branch's PR/checks until the next rail open.
    """
    fx = DeferredFixture(monkeypatch)
    fx.controller.notify_rail_opened()
    fx.run_job(0)  # local lands feat/call-1; the deferred net job is queued
    fx.run_job(1)  # net lands for feat/call-1
    assert fx.pr_branches_seen == ["feat/call-1"]

    jobs_before = len(fx.jobs)
    fx.controller.poll_tick()  # local tier only
    fx.run_job(jobs_before)  # lands feat/call-2 -- a different branch
    assert len(fx.jobs) == jobs_before + 2  # the escalated net job is queued
    fx.run_job(jobs_before + 1)
    assert fx.pr_branches_seen == ["feat/call-1", "feat/call-2"]


def test_branch_change_does_not_escalate_while_the_rail_is_closed(monkeypatch):
    """C's negative control: no `gh` fetch for a panel nobody is looking at."""
    fx = DeferredFixture(monkeypatch)
    fx.controller.notify_rail_opened()
    fx.run_job(0)
    fx.run_job(1)
    assert fx.pr_calls == 1

    fx.controller.request_refresh()  # queue one more local job while open
    fx.rail_open = False  # rail closes before it lands
    jobs_before = len(fx.jobs)
    fx.run_job(jobs_before - 1)
    assert len(fx.jobs) == jobs_before  # nothing escalated
    assert fx.pr_calls == 1
