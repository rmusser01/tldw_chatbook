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
            env_mod.BacklogTaskScanner, "scan",
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
