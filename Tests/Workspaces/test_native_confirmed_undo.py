"""Confirmed Undo keeps its original runtime, history and physical worker."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.UI.test_change_review_screen import _Harness, _record_turn, _wait_for
from tldw_chatbook.Agents.execution_capacity import WorkOrigin, current_execution_owner
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    ChangeRevertConfirmModal,
    ChangeReviewScreen,
)
from tldw_chatbook.Workspaces.change_revert import RevertRefusedError
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.fixture
async def undo_host(tmp_path):
    app = _Harness(None)
    app.app_config = {}
    app.chachanotes_db = CharactersRAGDB(tmp_path / "chat.db", "native-undo")
    runtime = app.console_runtime = ConsoleRuntime(app)
    store = runtime.ensure_chat_store()
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store,
        provider_gateway_factory=lambda: object(),
        skills_service=None,
    )
    tracker = bridge._change_tracker
    tracker.service = ShadowRepoService(tmp_path / "shadow")
    db = bridge._db
    run = db.create_run(conversation_id="undo-conversation", agent_kind="primary")
    root = tmp_path / "root"
    root.mkdir()
    (root / "edit.txt").write_text("before\n")
    _record_turn(
        db, tracker, root, run, lambda: (root / "edit.txt").write_text("after\n")
    )
    provider = app._provider = bridge.change_review_provider("undo-conversation")
    row = provider.turn_for_run(run).rows[0]
    h = SimpleNamespace(**locals())
    try:
        yield h
    finally:
        await runtime.dispose()
        app.chachanotes_db.close_connection()


async def open_confirmation(h, pilot):
    screen = await _wait_for(
        pilot,
        lambda: h.app.screen if isinstance(h.app.screen, ChangeReviewScreen) else None,
        "original Review",
    )
    await _wait_for(pilot, lambda: screen._leaves, "original file selection")
    screen.select_file("edit.txt")
    screen.action_revert_file()
    await _wait_for(
        pilot,
        lambda: isinstance(h.app.screen, ChangeRevertConfirmModal),
        "actual user confirmation",
    )
    await _wait_for(
        pilot,
        lambda: list(h.app.screen.query("#change-revert-yes")),
        "confirmation controls",
    )
    return screen


async def test_app_provider_cannot_revert_without_original_confirmation(undo_host):
    h = undo_host
    with pytest.raises(RevertRefusedError):
        await asyncio.to_thread(h.provider.revert, h.row, ["edit.txt"])
    assert (h.root / "edit.txt").read_text() == "after\n"

    from tldw_chatbook.Workspaces.change_revert_ownership import capture_undo

    h.provider._undo_native = None
    with pytest.raises(RevertRefusedError):
        capture_undo(h.provider)
    with pytest.raises(RevertRefusedError):
        await asyncio.to_thread(h.provider.revert, h.row, ["edit.txt"])


async def test_review_engine_rejects_preconfirmation_mutation(undo_host, monkeypatch):
    from tldw_chatbook.Workspaces.change_revert import revert_paths
    from tldw_chatbook.Workspaces.change_revert_ownership import native_undo_call

    h = undo_host
    original = h.provider.preflight_revert
    original_repo = h.tracker.service.repo_for_root
    observations, repos = [], []

    def repo(root):
        repos.append(root)
        return original_repo(root)

    def preflight(row, paths):
        call = native_undo_call()
        assert call is not None and not call.applying
        before = len(repos)
        try:
            result = revert_paths(
                h.tracker.service, h.db, row, paths, run_active=lambda: False
            )
        except RevertRefusedError:
            observations.append(("refused", len(repos) - before))
        else:
            observations.append(
                ([outcome.ok for outcome in result], len(repos) - before)
            )
        return original(row, paths)

    monkeypatch.setattr(h.tracker.service, "repo_for_root", repo)
    monkeypatch.setattr(h.provider, "preflight_revert", preflight)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: (
                h.app.screen if isinstance(h.app.screen, ChangeReviewScreen) else None
            ),
            "original Review",
        )
        await _wait_for(pilot, lambda: screen._leaves, "original selection")
        screen.select_file("edit.txt")
        screen.action_revert_file()
        await _wait_for(
            pilot, lambda: observations, "actual preconfirmation engine call"
        )
        await screen.workers.wait_for_complete()
        assert (h.root / "edit.txt").read_text() == "after\n", observations
        assert observations == [("refused", 0)]


@pytest.mark.parametrize("attempt", ["row", "paths", "duplicate"])
async def test_applying_engine_requires_exact_unconsumed_selection(
    undo_host, monkeypatch, attempt
):
    from tldw_chatbook.Workspaces.change_revert import revert_paths
    from tldw_chatbook.Workspaces.change_revert_ownership import native_undo_call

    h = undo_host
    original = h.provider.revert
    original_repo = h.tracker.service.repo_for_root
    observations, repos, calls = [], [], []

    def repo(root):
        repos.append(root)
        return original_repo(root)

    def apply(row, paths):
        call = native_undo_call()
        calls.append(call)
        assert call.applying and call.source.confirmed
        result = original(row, paths) if attempt == "duplicate" else None
        attempted_row = (
            dict(row, baseline_sha=row["end_sha"]) if attempt == "row" else row
        )
        attempted_paths = ["outside-confirmation.txt"] if attempt == "paths" else paths
        before = len(repos)
        try:
            revert_paths(
                h.tracker.service,
                h.db,
                attempted_row,
                attempted_paths,
                run_active=lambda: False,
            )
        except RevertRefusedError:
            observations.append(("refused", len(repos) - before))
        else:
            observations.append(("accepted", len(repos) - before))
        return result if attempt == "duplicate" else original(row, paths)

    monkeypatch.setattr(h.tracker.service, "repo_for_root", repo)
    monkeypatch.setattr(h.provider, "revert", apply)
    async with h.app.run_test(size=(120, 40)) as pilot:
        await open_confirmation(h, pilot)
        assert await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot,
            lambda: calls and calls[0].retired.is_set(),
            "original apply retirement",
        )
        assert observations == [("refused", 0)]
        assert (h.root / "edit.txt").read_text() == "before\n"
        assert '"edit.txt"' in h.db.change_snapshots_for_run(h.run)[0]["reverted"]


async def test_actual_preflight_has_finite_native_owner_before_modal(
    undo_host, monkeypatch
):
    h = undo_host
    observed = []
    original = h.provider.preflight_revert

    def preflight(*args):
        owner = current_execution_owner()
        observed.append(owner)
        assert owner is not None, "original Undo preflight is unowned"
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        return original(*args)

    monkeypatch.setattr(h.provider, "preflight_revert", preflight)
    async with h.app.run_test(size=(120, 40)) as pilot:
        await open_confirmation(h, pilot)
        assert observed and observed[0].origin is WorkOrigin.MANUAL
        assert h.runtime.execution_capacity.snapshot().executions == ()
        await pilot.click("#change-revert-no")


async def test_writer_admitted_after_modal_refuses_whole_undo(undo_host):
    h = undo_host
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            owner = h.runtime.execution_capacity.begin_execution(
                origin=WorkOrigin.MANUAL, conversation_id="actual-writer"
            )
            claim.bind_owner(owner)
            try:
                await pilot.click("#change-revert-yes")
                await screen.workers.wait_for_complete()
                assert (h.root / "edit.txt").read_text() == "after\n", (
                    "Undo wrote through live native claim"
                )
            finally:
                owner.finish_root()
                claim.settle(durable=True)


async def test_confirmed_restore_records_original_row_and_retires(undo_host):
    h = undo_host
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        await screen.workers.wait_for_complete()
        assert (h.root / "edit.txt").read_text() == "before\n"
        assert '"edit.txt"' in h.db.change_snapshots_for_run(h.run)[0]["reverted"]
        assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize("change", ["disk", "row", "provider", "capacity"])
async def test_confirmation_never_reselects_changed_sources(undo_host, change):
    h = undo_host
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        restore = None
        if change == "disk":
            (h.root / "edit.txt").write_text("user edit while modal open\n")
        elif change == "row":
            with h.db.transaction() as conn:
                conn.execute(
                    "UPDATE change_snapshots SET end_sha=baseline_sha WHERE id=?",
                    (h.row["id"],),
                )
        elif change == "provider":
            restore = h.provider._service
            h.provider._service = ShadowRepoService(h.tmp_path / "replacement")
        else:
            from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

            restore = h.runtime._execution_capacity
            h.runtime._execution_capacity = RuntimeCapacity()
        try:
            await pilot.click("#change-revert-yes")
            await screen.workers.wait_for_complete()
            assert (h.root / "edit.txt").read_text() == (
                "user edit while modal open\n" if change == "disk" else "after\n"
            )
            assert not h.runtime._change_revert_native.pending
        finally:
            if change == "provider":
                h.provider._service = restore
            elif change == "capacity":
                h.runtime._execution_capacity = restore


async def test_native_ticket_exists_before_original_worker_enqueue(
    undo_host, monkeypatch
):
    from tldw_chatbook.Workspaces.change_revert_ownership import _UndoCall

    h = undo_host
    original = asyncio.to_thread
    observations = []

    async def enqueue(callback, *args, **kwargs):
        if isinstance(getattr(callback, "__self__", None), _UndoCall):
            call = callback.__self__
            if callback.__name__ == "run":
                observations.append(call)
                assert call in h.runtime._change_revert_native.pending
                assert call.operation is not None
                assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        return await original(callback, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", enqueue)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        await screen.workers.wait_for_complete()
    assert len(observations) == 2
    assert observations[0] is not observations[1]
    assert all(call.retired.is_set() for call in observations)


async def test_commit_then_raise_is_observed_on_original_sqlite(undo_host, monkeypatch):
    h = undo_host
    original = h.db.update_change_snapshot_reverted

    def committed(row, paths):
        original(row, paths)
        raise RuntimeError("actual DB commit returned through an error")

    monkeypatch.setattr(h.db, "update_change_snapshot_reverted", committed)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        await screen.workers.wait_for_complete()
        assert (h.root / "edit.txt").read_text() == "before\n"
        assert not h.runtime._change_revert_native.pending
        assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_finish_return_retains_original_worker_db_and_scratch(
    undo_host, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation
    from tldw_chatbook.Workspaces.change_revert_ownership import native_undo_call

    h = undo_host
    snapshot = h.runtime.scratch_spaces.snapshot("original-existing-scratch")
    root = snapshot.root
    (root / "edit.txt").write_text("before\n")
    run = h.db.create_run(conversation_id="undo-conversation", agent_kind="primary")
    _record_turn(
        h.db, h.tracker, root, run, lambda: (root / "edit.txt").write_text("after\n")
    )
    h.app._initial_run_id = run
    entered, release = threading.Event(), threading.Event()
    original = OwnedOperation.finish
    observed = []

    def held(operation):
        call = native_undo_call()
        if call is not None and call.applying:
            conn = getattr(h.db._thread_local, "conn", None)
            observed.append((call, conn, threading.get_ident()))
            entered.set()
            assert release.wait(10)
            assert current_execution_owner() is call.owner
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            assert root.exists(), "scratch retired before operation.finish return"
        return original(operation)

    monkeypatch.setattr(OwnedOperation, "finish", held)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        try:
            await pilot.click("#change-revert-yes")
            await _wait_for(pilot, entered.is_set, "actual finish entry")
            dispose = asyncio.create_task(h.runtime.dispose())
            await _wait_for(pilot, lambda: h.runtime._disposed, "runtime sealed")
            dispose.cancel()
            dispose.cancel()
            call, conn, thread = observed[0]
            assert thread != threading.get_ident() and conn is not None
            assert not call.retired.is_set() and root.exists()
            assert call in call.source.issuer.pending
            assert h.runtime._first_send_sources == 1
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
        finally:
            release.set()
        await screen.workers.wait_for_complete()
        try:
            await dispose
        except asyncio.CancelledError:
            pass
        assert call.retired.is_set()
        assert call not in call.source.issuer.pending
        assert h.runtime._first_send_sources == 0
        assert await asyncio.to_thread(h.runtime.scratch_spaces.wait_for_cleanup, 3)
        assert not root.exists()


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
async def test_historical_smudge_helper_retains_writer_after_ui_unmount(
    undo_host, monkeypatch, mode
):
    import sys

    from tldw_chatbook.Notes.git_process_containment import ProcessTreeController
    from tldw_chatbook.Workspaces import change_tracking

    h = undo_host
    entered, release = h.tmp_path / "helper-entered", h.tmp_path / "helper-release"
    script = h.tmp_path / "smudge.py"
    script.write_text(
        "import sys,time\nfrom pathlib import Path\n"
        f"Path({str(entered)!r}).touch()\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read().replace(b'before',b'FILTERED'))\n"
    )
    repo = h.tracker.service.repo_for_root(h.root)
    repo._run("config", "filter.original.smudge", f"{sys.executable} {script}")
    repo._run("config", "filter.original.required", "true")
    repo._run("config", "filter.original.clean", "cat")
    (h.root / ".gitattributes").write_text("edit.txt filter=original\n")
    async with h.app.run_test(size=(120, 40)) as pilot:
        await open_confirmation(h, pilot)
        if mode == "timeout":
            monkeypatch.setattr(change_tracking, "_GIT_TIMEOUT_SECONDS", 0.2)
            monkeypatch.setattr(ProcessTreeController, "terminate", lambda *args: None)
            monkeypatch.setattr(ProcessTreeController, "kill", lambda *args: None)
        try:
            await pilot.click("#change-revert-yes")
            await _wait_for(pilot, entered.exists, "actual historical smudge")
            call = next(
                call
                for call in h.runtime._change_revert_native.pending
                if call.applying
            )
            h.app.pop_screen()
            if mode == "timeout":
                await _wait_for(
                    pilot, lambda: call.operation._stopping, "actual process timeout"
                )
            assert not call.retired.is_set()
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
        finally:
            release.touch()
        await _wait_for(pilot, call.retired.is_set, "original helper and worker return")
        assert (h.root / "edit.txt").read_text() == "FILTERED\n"


async def test_multiroot_undo_has_one_writer_for_rename_and_uncreate(
    undo_host, monkeypatch
):
    from tldw_chatbook.Workspaces.change_revert_ownership import native_undo_call

    h = undo_host
    second = h.tmp_path / "second"
    second.mkdir()
    (second / "old.txt").write_text("stable rename\n" * 20)

    def mutate():
        (second / "old.txt").rename(second / "new-name.txt")
        (second / "created.txt").write_text("agent-created\n")

    _record_turn(h.db, h.tracker, second, h.run, mutate)
    calls = []
    original = h.provider.revert

    def record(row, paths):
        call = native_undo_call()
        calls.append(call)
        assert call is not None and call.applying
        return original(row, paths)

    monkeypatch.setattr(h.provider, "revert", record)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: (
                h.app.screen if isinstance(h.app.screen, ChangeReviewScreen) else None
            ),
            "review",
        )
        await _wait_for(pilot, lambda: len(screen._leaves) == 3, "all roots")
        screen.action_undo_all()
        await _wait_for(
            pilot,
            lambda: isinstance(h.app.screen, ChangeRevertConfirmModal),
            "all-root confirmation",
        )
        assert h.runtime.execution_capacity.snapshot().executions == ()
        await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot, lambda: calls and calls[0].retired.is_set(), "whole writer return"
        )
        assert len(calls) == 2 and calls[0] is calls[1]
        assert (h.root / "edit.txt").read_text() == "before\n"
        assert (second / "old.txt").is_file()
        assert not (second / "new-name.txt").exists()
        assert not (second / "created.txt").exists()


@pytest.mark.parametrize("kind", ["nested", "alias", "recovery", "capacity"])
async def test_post_modal_admission_honors_original_conflicts(undo_host, kind):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeScope

    h = undo_host
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        root = h.root
        if kind == "alias":
            alias = h.tmp_path / "alias"
            alias.symlink_to(root, target_is_directory=True)
            root = alias.resolve()
        scope = WorktreeScope(root / "edit.txt", False, capture_directory_chain(root))
        footprint = WorktreeFootprint(writes=(scope,))
        owner = claim = operation = None
        if kind == "recovery":
            h.runtime.worktree_activity.protect_recovery(
                "original-unsettled-workflow", footprint
            )
        else:
            owner = h.runtime.execution_capacity.begin_execution(
                origin=WorkOrigin.MANUAL, conversation_id="writer"
            )
            if kind == "capacity":
                h.runtime.execution_capacity._read_settings = False
                h.runtime.execution_capacity.set_tool_limits(1, 0)
                operation = owner.reserve_tool()
            else:
                claim = h.runtime.worktree_activity.try_acquire(
                    owner=owner, footprint=footprint
                )
                assert claim is not None
        try:
            await pilot.click("#change-revert-yes")
            await screen.workers.wait_for_complete()
            assert (h.root / "edit.txt").read_text() == "after\n"
            assert not h.runtime._change_revert_native.pending
        finally:
            if operation is not None:
                operation.finish()
            if owner is not None:
                owner.finish_root()
            if claim is not None:
                claim.settle(durable=True)


@pytest.mark.parametrize("failure", ["snapshot", "db"])
async def test_unknown_required_settlement_retains_original_scopes(
    undo_host, monkeypatch, failure
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    receipt = os.environ.get("UNDO_REQUIRED_RECEIPT")
    if receipt is None:
        result_path = undo_host.tmp_path / "required.json"
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_unknown_required_settlement_retains_original_scopes[{failure}]",
                "--basetemp=" + str(undo_host.tmp_path / "child"),
            ],
            env={**os.environ, "UNDO_REQUIRED_RECEIPT": str(result_path)},
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result_path.read_text()) == {
            "failure": failure,
            "retained": True,
        }
        return
    try:
        from tldw_chatbook.Agents.activation import _guard
        from tldw_chatbook.Backup_Recovery import storage_admission
        from tldw_chatbook.Workspaces.change_revert import revert_paths
        from tldw_chatbook.Workspaces.change_revert_ownership import native_undo_call
        from tldw_chatbook.Workspaces.change_tracking import (
            ChangeTrackingError,
            ShadowRepo,
        )

        h = undo_host
        second = h.tmp_path / "zz-second"
        second.mkdir()
        (second / "edit.txt").write_text("before\n")
        _record_turn(
            h.db,
            h.tracker,
            second,
            h.run,
            lambda: (second / "edit.txt").write_text("after\n"),
        )
        observed = {}
        original_snapshot = ShadowRepo.snapshot
        original_update = h.db.update_change_snapshot_reverted
        original_revert = h.provider.revert
        original_repo = h.tracker.service.repo_for_root
        repos = []

        def repo(root):
            repos.append(root)
            return original_repo(root)

        def apply(row, paths):
            result = original_revert(row, paths)
            call = native_undo_call()
            assert not call.required_proven
            next_row, next_paths, _ = next(
                value for value in call.source.rows if value[0]["id"] != row["id"]
            )
            before = len(repos)
            try:
                revert_paths(
                    h.tracker.service,
                    h.db,
                    next_row,
                    next_paths,
                    run_active=lambda: False,
                )
            except RevertRefusedError:
                observed["next_engine"] = ("refused", len(repos) - before)
            else:
                observed["next_engine"] = ("accepted", len(repos) - before)
            observed["mutated_root"] = row["root"]
            observed["untouched_root"] = next_row["root"]
            return result

        def snapshot(repo, message, **kwargs):
            result = original_snapshot(repo, message, **kwargs)
            if message == "revert":
                observed["state"] = _guard.context.get()
                observed["connection"] = h.db._thread_local.conn
                raise ChangeTrackingError("original snapshot committed then raised")
            return result

        def update(row, paths):
            observed["state"] = _guard.context.get()
            observed["connection"] = h.db._thread_local.conn
            with h.db.transaction():
                original_update(row, paths)
                raise RuntimeError("actual required DB rollback")

        if failure == "snapshot":
            monkeypatch.setattr(ShadowRepo, "snapshot", snapshot)
        else:
            monkeypatch.setattr(h.db, "update_change_snapshot_reverted", update)
        monkeypatch.setattr(h.tracker.service, "repo_for_root", repo)
        monkeypatch.setattr(h.provider, "revert", apply)
        async with h.app.run_test(size=(120, 40)) as pilot:
            screen = await _wait_for(
                pilot,
                lambda: (
                    h.app.screen
                    if isinstance(h.app.screen, ChangeReviewScreen)
                    else None
                ),
                "original Review",
            )
            await _wait_for(
                pilot, lambda: len(screen._leaves) == 2, "both original rows"
            )
            screen.action_undo_all()
            await _wait_for(
                pilot,
                lambda: (
                    isinstance(h.app.screen, ChangeRevertConfirmModal)
                    and list(h.app.screen.query("#change-revert-yes"))
                ),
                "original all-row confirmation",
            )
            outcomes = []
            original_report = screen._report_outcomes
            monkeypatch.setattr(
                screen,
                "_report_outcomes",
                lambda value: outcomes.extend(value) or original_report(value),
            )
            await pilot.click("#change-revert-yes")
            await _wait_for(pilot, lambda: outcomes, "honest partial host result")
            call = next(iter(h.runtime._change_revert_native.pending))
            await _wait_for(
                pilot,
                lambda: call.operation._stopping,
                "required uncertainty retention",
            )
            assert (
                Path(observed["mutated_root"]) / "edit.txt"
            ).read_text() == "before\n"
            assert (
                Path(observed["untouched_root"]) / "edit.txt"
            ).read_text() == "after\n"
            assert observed["next_engine"] == ("refused", 0)
            assert outcomes[0].ok and not outcomes[0].settled
            assert not call.required_proven and call.host_proven
            assert not call.retired.is_set() and observed["state"].live
            assert observed["connection"] is not None
            with storage_admission._lock:
                assert (
                    set(observed["state"].leases.values())
                    <= storage_admission._live_leases
                )
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
            Path(receipt).write_text(json.dumps({"failure": failure, "retained": True}))
            os._exit(0)
    except BaseException:  # noqa: BLE001 -- bounded subprocess owns deliberately retained worker
        import traceback

        Path(receipt).write_text(traceback.format_exc())
        os._exit(1)


async def test_failed_path_after_actual_checkout_still_snapshots(
    undo_host, monkeypatch
):
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepo

    h = undo_host
    original_restore, original_snapshot = ShadowRepo.restore_paths, ShadowRepo.snapshot
    snapshots = []

    def restore(repo, commit, paths):
        original_restore(repo, commit, paths)
        raise RuntimeError("checkout wrote before returning an error")

    def snapshot(repo, message, **kwargs):
        result = original_snapshot(repo, message, **kwargs)
        snapshots.append(message)
        return result

    monkeypatch.setattr(ShadowRepo, "restore_paths", restore)
    monkeypatch.setattr(ShadowRepo, "snapshot", snapshot)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        outcomes = []
        original_report = screen._report_outcomes

        def report(result):
            outcomes.extend(result)
            original_report(result)

        monkeypatch.setattr(screen, "_report_outcomes", report)
        assert await pilot.click("#change-revert-yes")
        await _wait_for(pilot, lambda: outcomes, "actual checkout failure outcome")
        await screen.workers.wait_for_complete()
        assert (h.root / "edit.txt").read_text() == "before\n"
        assert snapshots == ["revert"]
        assert not h.runtime._change_revert_native.pending
        assert not outcomes[0].ok and outcomes[0].settled
        assert h.db.change_snapshots_for_run(h.run)[0]["reverted"] in (None, "", "[]")


async def test_original_provider_refusal_reaches_review_as_per_path_outcome(
    undo_host, monkeypatch
):
    h = undo_host

    def refuse(*args):
        raise RevertRefusedError("original provider refused before mutation")

    monkeypatch.setattr(h.provider, "revert", refuse)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        outcomes = []
        original_report = screen._report_outcomes

        def report(result):
            outcomes.extend(result)
            original_report(result)

        monkeypatch.setattr(screen, "_report_outcomes", report)
        assert await pilot.click("#change-revert-yes")
        await _wait_for(pilot, lambda: outcomes, "actual provider refusal outcome")
        await screen.workers.wait_for_complete()
        assert len(outcomes) == 1 and not outcomes[0].ok
        assert outcomes[0].error == "original provider refused before mutation"
        assert (h.root / "edit.txt").read_text() == "after\n"
        assert not h.runtime._change_revert_native.pending


async def test_cancel_between_admission_and_enqueue_retires_only_unused_ticket(
    undo_host, monkeypatch
):
    from tldw_chatbook.Workspaces.change_revert_ownership import _UndoCall

    h = undo_host
    entered = asyncio.Event()
    original = _UndoCall.run_async
    calls = []

    async def gap(call, *args, **kwargs):
        if call.applying:
            calls.append(call)
            entered.set()
            await asyncio.Event().wait()
        return await original(call, *args, **kwargs)

    monkeypatch.setattr(_UndoCall, "run_async", gap)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        await _wait_for(pilot, entered.is_set, "admitted enqueue gap")
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        screen.workers.cancel_group(screen, "change-review-revert")
        await _wait_for(pilot, calls[0].retired.is_set, "unused ticket retirement")
        assert (h.root / "edit.txt").read_text() == "after\n"
        assert not calls[0]._started


async def test_real_console_card_uses_native_confirmation_and_original_provider(
    tmp_path,
):
    from textual.widgets import Button

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Widgets.Console.console_turn_file_card import ConsoleTurnFileCard

    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(tmp_path / "card-chat.db", "native-undo-card")
    host = ConsoleHarness(app)
    runtime = None
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            screen = host.screen_stack[-1]
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            store = screen._ensure_console_chat_store()
            session = store.create_session()
            store.persist_session_if_needed(session.id)
            screen._ensure_console_chat_controller()
            bridge = screen._ensure_console_agent_bridge()
            runtime = app.console_runtime
            assert bridge._change_revert_native is runtime._change_revert_native
            tracker, db = bridge._change_tracker, bridge._db
            tracker.service = ShadowRepoService(tmp_path / "card-shadow")
            root = tmp_path / "card-root"
            root.mkdir()
            (root / "edit.txt").write_text("before\n")
            run = db.create_run(
                conversation_id=session.persisted_conversation_id, agent_kind="primary"
            )
            _record_turn(
                db,
                tracker,
                root,
                run,
                lambda: (root / "edit.txt").write_text("after\n"),
            )
            store.append_message(
                session.id,
                role=ConsoleMessageRole.TOOL,
                content="Edited 1 file",
                change_review_run_id=run,
            )
            await screen._sync_native_console_chat_ui()
            card = await _wait_for(
                pilot,
                lambda: next(iter(screen.query(ConsoleTurnFileCard)), None),
                "real turn card",
            )
            button = card.query_one(".console-turn-file-undo-all-btn", Button)
            await _wait_for(pilot, lambda: not button.disabled, "card Undo")
            card.scroll_visible(animate=False)
            await pilot.pause()
            assert await pilot.click(button), "actual card button was not hit"
            await _wait_for(
                pilot,
                lambda: isinstance(host.screen, ChangeRevertConfirmModal),
                "card confirmation",
            )
            assert runtime.execution_capacity.snapshot().executions == ()
            await pilot.click("#change-revert-yes")
            await _wait_for(
                pilot, lambda: str(button.label) == "Undone", "original card acceptance"
            )
            await _wait_for(
                pilot,
                lambda: not runtime._change_revert_native.pending,
                "card worker retirement",
            )
            assert (root / "edit.txt").read_text() == "before\n"
            assert '"edit.txt"' in db.change_snapshots_for_run(run)[0]["reverted"]
    finally:
        if runtime is not None:
            await runtime.dispose()
        app.chachanotes_db.close_connection()


async def test_actual_host_return_keeps_original_worker_and_claim(
    undo_host, monkeypatch
):
    h = undo_host
    entered, release = threading.Event(), threading.Event()
    original = h.app.call_from_thread
    observed = []

    def marshal(callback, *args, **kwargs):
        call = next(
            (
                call
                for call in h.runtime._change_revert_native.pending
                if call.applying and call.thread == threading.get_ident()
            ),
            None,
        )
        result = original(callback, *args, **kwargs)
        if call is not None and call.applying:
            conn = h.db._thread_local.conn
            observed.append(call)
            entered.set()
            assert release.wait(10)
            assert call.owner.is_admitted
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        return result

    monkeypatch.setattr(h.app, "call_from_thread", marshal)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        try:
            await pilot.click("#change-revert-yes")
            await _wait_for(pilot, entered.is_set, "original host return")
            assert not observed[0].host_proven
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
        finally:
            release.set()
        await screen.workers.wait_for_complete()
        assert observed[0].retired.is_set()


async def test_revoked_original_scratch_never_becomes_unmanaged(undo_host):
    h = undo_host
    snapshot = h.runtime.scratch_spaces.snapshot("historical-original")
    root = snapshot.root
    (root / "edit.txt").write_text("before\n")
    run = h.db.create_run(conversation_id="undo-conversation", agent_kind="primary")
    _record_turn(
        h.db, h.tracker, root, run, lambda: (root / "edit.txt").write_text("after\n")
    )
    h.app._initial_run_id = run
    with h.runtime.scratch_spaces.lease(snapshot):
        async with h.app.run_test(size=(120, 40)) as pilot:
            screen = await open_confirmation(h, pilot)
            h.runtime.scratch_spaces.close("historical-original")
            assert h.runtime.scratch_spaces.existing_snapshot_for_root(root) is snapshot
            await pilot.click("#change-revert-yes")
            await screen.workers.wait_for_complete()
            assert (root / "edit.txt").read_text() == "after\n"
            assert not h.runtime._change_revert_native.pending


@pytest.mark.parametrize("unused", [False, True])
async def test_activation_entry_refusal_retires_without_effect_then_allows_later_work(
    undo_host,
    monkeypatch,
    unused,
):
    from contextlib import contextmanager

    from tldw_chatbook.Workspaces import change_revert_ownership as ownership

    h = undo_host
    original_scope = ownership.operation_owned_connection
    original_async = ownership._UndoCall.run_async
    entered = asyncio.Event()
    refused = []

    @contextmanager
    def scope(db):
        call = next(
            (
                call
                for call in h.runtime._change_revert_native.pending
                if call.applying and current_execution_owner() is call.owner
            ),
            None,
        )
        if call is not None and not refused:
            refused.append(call)
            raise RuntimeError(
                "original connection scope refused before effect callback"
            )
        with original_scope(db):
            yield

    async def gap(call, *args, **kwargs):
        if unused and call.applying and not refused:
            entered.set()
            await asyncio.Event().wait()
        return await original_async(call, *args, **kwargs)

    monkeypatch.setattr(ownership, "operation_owned_connection", scope)
    monkeypatch.setattr(ownership._UndoCall, "run_async", gap)
    async with h.app.run_test(size=(120, 40)) as pilot:
        screen = await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        if unused:
            await _wait_for(pilot, entered.is_set, "unused admission")
            screen.workers.cancel_group(screen, "change-review-revert")
        await _wait_for(
            pilot,
            lambda: refused and refused[0].retired.is_set(),
            "no-effect retirement",
        )
        assert (h.root / "edit.txt").read_text() == "after\n"
        assert h.runtime.execution_capacity.snapshot().executions == ()
        await open_confirmation(h, pilot)
        await pilot.click("#change-revert-yes")
        await screen.workers.wait_for_complete()
        assert (h.root / "edit.txt").read_text() == "before\n"


async def test_undo_capture_refuses_before_generation_capture_during_publication(
    undo_host,
):
    from tldw_chatbook.Workspaces.change_revert_ownership import capture_undo

    h = undo_host
    h.runtime._first_send_publishing = True
    try:
        with pytest.raises(RuntimeError, match="being saved"):
            capture_undo(h.provider)
        assert not h.provider._undo_native.pending
        assert h.runtime._first_send_sources == 0
        assert not h.runtime.execution_capacity.snapshot().executions
        assert (h.root / "edit.txt").read_text() == "after\n"
    finally:
        h.runtime._first_send_publishing = False
