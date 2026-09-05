"""Recovery flow uses real temporary SQLite and the local child process."""

import asyncio
import os
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


@pytest.fixture
def lab_app(tmp_path, monkeypatch):
    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: tmp_path)
    return _build_test_app()


@pytest.mark.asyncio
async def test_recovery_fallback_explains_rollback_without_private_content(
    lab_app, tmp_path
):
    import sqlite3

    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore

    def seed():
        store = CheckpointStore(tmp_path / "chunking_lab.sqlite3", str(tmp_path))
        older = lab_state.replace_sample(
            lab_state.new_session(str(tmp_path)),
            "private older sample",
            {"kind": "paste"},
        )
        token = store.save(older, expected=None)
        newer = lab_state.replace_sample(
            older, "private newer sample", {"kind": "paste"}
        )
        store.save(newer, expected=token)
        store.close()

    await asyncio.to_thread(seed)
    with sqlite3.connect(tmp_path / "chunking_lab.sqlite3") as connection:
        connection.execute(
            "UPDATE lab_checkpoints SET document = ? WHERE id = (SELECT current_checkpoint FROM lab_state)",
            ("broken",),
        )
    async with lab_app.run_test(size=(80, 24)):
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        message = str(screen.query_one("#lab-message").content)
        assert "previous" in message.lower()
        assert "private older" not in message and "private newer" not in message
        assert (
            screen.coordinator.session.samples[
                screen.coordinator.session.view["sample_hash"]
            ]["text"]
            == "private older sample"
        )


@pytest.mark.asyncio
async def test_saved_result_is_not_unsaved_when_only_view_checkpoint_fails(lab_app):
    from Tests.UI.test_chunking_lab_screen import settle_lab
    from tldw_chatbook.Chunking import lab_state

    screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
    async with lab_app.run_test(size=(80, 24)) as pilot:
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        screen.queue_edit(
            lambda session: lab_state.replace_sample(
                session, "one two three", {"kind": "paste"}
            )
        )
        await screen.run_candidates()
        await screen._refresh_session()
        await settle_lab(lab_app, screen, pilot)
        assert "Saved locally" in str(screen.query_one("#lab-status").content)
        with patch.object(
            screen.coordinator._writer._store,
            "save",
            side_effect=OSError("view write failure"),
        ):
            screen.queue_edit(
                lambda session: lab_state.update_view(session, {"region": "results"})
            )
            await screen.drain_edits()
            assert not await screen.confirm_navigation()
            await screen._refresh_session()
            await settle_lab(lab_app, screen, pilot)
            assert "Save failed" in str(screen.query_one("#lab-status").content)
            assert "Unsaved result" not in str(screen.query_one("#lab-status").content)
        await screen.coordinator.cancel()
        await lab_app.pop_screen()
        await screen.coordinator.close()
        lab_app._chunking_lab_coordinator = None
        reopened = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(reopened)
        await reopened.wait_until_ready()
        await reopened._refresh_session()
        await settle_lab(lab_app, reopened, pilot)
        assert "Unsaved result" not in str(reopened.query_one("#lab-status").content)
        store = reopened.coordinator._writer._store
        original_save = store.save

        def fail_new_result(session, *args, **kwargs):
            if len(session.results) > 1:
                raise OSError("result write failure")
            return original_save(session, *args, **kwargs)

        with patch.object(store, "save", fail_new_result):
            with pytest.raises(OSError, match="result write failure"):
                await reopened.run_candidates()
            await reopened._refresh_session()
            await settle_lab(lab_app, reopened, pilot)
            assert "Unsaved result" in str(reopened.query_one("#lab-status").content)
        await reopened.coordinator.cancel()


@pytest.mark.asyncio
async def test_save_capture_cannot_associate_an_import_during_preparation(
    lab_app, tmp_path
):
    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.UI.Screens import chunking_lab_screen as module

    lab_app.media_db = MediaDatabase(
        str(tmp_path / "media.sqlite3"), client_id="save-race"
    )
    async with lab_app.run_test(size=(80, 24)):
        screen = module.ChunkingLabScreen(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        screen.queue_edit(
            lambda session: lab_state.edit_record_fields(
                session,
                screen._b_id(session),
                {"name": "Captured save", "description": "captured"},
            )
        )
        await screen.drain_edits()
        entered, release = threading.Event(), threading.Event()
        calls = 0
        prepare = module._save_payload

        def delayed(session, role):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                assert release.wait(5)
            return prepare(session, role)

        async def answer(dialog):
            return {"name": "Captured save", "description": "captured", "tags": ""}

        with (
            patch.object(module, "_save_payload", delayed),
            patch.object(screen, "_dialog", answer),
        ):
            saving = asyncio.create_task(screen._save("B"))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                screen.queue_edit(
                    lambda session: lab_state.replace_template(
                        session,
                        screen._b_id(session),
                        {"chunking": {"method": "fixed_size"}},
                        record_fields={"name": "Unrelated import"},
                        expected_record=None,
                    )
                )
                await screen.drain_edits()
            finally:
                release.set()
            await saving
        draft = screen.coordinator.session.candidates[screen._b_id()]["draft"]
        assert draft["record_fields"]["name"] == "Unrelated import"
        assert draft["expected_record"] is None


@pytest.mark.asyncio
async def test_missing_library_source_does_not_mislabel_loaded_recovery(lab_app):
    screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
    screen.apply_navigation_context({"local_media_id": 42})
    async with lab_app.run_test(size=(80, 24)) as pilot:
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await pilot.pause()
        assert screen.coordinator is not None
        assert "Recovery load failed" not in str(
            screen.query_one("#lab-status").content
        )
        assert "source" in str(screen.query_one("#lab-message").content).lower()


@pytest.mark.asyncio
async def test_failed_initial_load_is_read_only_and_retry_reads_again(lab_app):
    from tldw_chatbook.Chunking.lab_coordinator import LabCoordinator

    screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
    async with lab_app.run_test(size=(80, 24)) as pilot:
        with patch.object(
            LabCoordinator, "load", side_effect=OSError("unreadable checkpoint")
        ):
            await lab_app.push_screen(screen)
            await screen.wait_until_ready()
            assert screen.coordinator is None
            assert screen.query_one("#lab-sample").disabled
            assert screen.query_one("#lab-editor").disabled
            assert screen.query_one("#lab-run").disabled
        await screen._action("lab-retry")
        await screen.wait_until_ready()
        await pilot.pause()
        assert screen.coordinator is not None
        assert not screen.query_one("#lab-sample").disabled


@pytest.mark.asyncio
async def test_typing_during_delayed_edit_and_completion_preserves_inputs(lab_app):
    from Tests.Chunking.test_lab_coordinator import outcome
    from tldw_chatbook.Chunking import lab_state

    route = resolve_screen_route("chunking_lab")
    async with lab_app.run_test(size=(120, 40)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        screen = lab_app.screen
        await screen.wait_until_ready()
        screen.query_one("#lab-sample-text").load_text("one two three")
        await pilot.pause()
        await screen.drain_edits()
        original = screen.coordinator.session
        requests = await asyncio.to_thread(
            lab_state.capture_batch, original, (screen._b_id(),)
        )
        screen.coordinator.set_session(
            await asyncio.to_thread(lab_state.install_batch, original, requests)
        )
        entered, release = threading.Event(), threading.Event()
        edit = lab_state.edit_control

        def delayed(session, candidate_id, field, value):
            entered.set()
            assert release.wait(5)
            return edit(session, candidate_id, field, value)

        with patch.object(lab_state, "edit_control", delayed):
            screen.query_one("#lab-max-size").value = "3"
            await asyncio.to_thread(entered.wait, 3)
            # A notifier render may finish while the first edit is still off-loop.
            await screen._refresh_session()
            assert screen.query_one("#lab-max-size").value == "3"
            screen.query_one("#lab-overlap").value = "1"
            await pilot.pause()
            await screen.coordinator._accept(
                await asyncio.to_thread(outcome, requests[0])
            )
            release.set()
            await screen.drain_edits()
        import json

        draft = screen.coordinator.session.candidates[screen._b_id()]["draft"]
        config = json.loads(draft["raw_json"])["chunking"]["config"]
        assert config["max_size"] == 3 and config["overlap"] == 1
        assert requests[0].run_id in screen.coordinator.session.results


@pytest.mark.asyncio
async def test_failed_checkpoint_blocks_navigation_export_survives(
    lab_app, tmp_path, monkeypatch
):
    from Tests.UI.test_chunking_lab_screen import settle_lab

    route = resolve_screen_route("chunking_lab")
    async with lab_app.run_test(size=(80, 24)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        screen = lab_app.screen
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        assert not screen.query_one("#lab-sample").disabled
        screen.query_one("#lab-sample-text").load_text("retained private sample")
        await pilot.pause()
        await screen.drain_edits()
        assert (
            screen.coordinator.session.samples[
                screen.coordinator.session.view["sample_hash"]
            ]["text"]
            == "retained private sample"
        )
        store = screen.coordinator._writer._store
        with patch.object(store, "save", side_effect=OSError("injected disk failure")):
            assert not await screen.confirm_navigation()
            await screen.file_operation(
                "export-recovery", {"path": str(tmp_path / "failed-export.json")}
            )
            assert (tmp_path / "failed-export.json").is_file()
            assert "Retry" in str(
                screen.query_one("#lab-message").content
            ) or "Exported" in str(screen.query_one("#lab-message").content)
        await screen._action("lab-retry")
        assert await screen.confirm_navigation()
        await lab_app.pop_screen()
        fresh_path = tmp_path / "transfer-profile"
        fresh_path.mkdir(mode=0o700)
        monkeypatch.setattr(
            "tldw_chatbook.config.get_user_data_dir", lambda: fresh_path
        )
        destination = route.load_screen_class()(lab_app)
        await lab_app.push_screen(destination)
        await destination.wait_until_ready()
        await destination.file_operation(
            "restore", {"path": str(tmp_path / "failed-export.json")}
        )
        assert destination.coordinator.session.profile_key == str(fresh_path)
        assert (
            destination.coordinator.session.samples[
                destination.coordinator.session.view["sample_hash"]
            ]["text"]
            == "retained private sample"
        )
        assert await destination.confirm_navigation()


@pytest.mark.asyncio
async def test_profile_switch_failure_retains_old_owner(lab_app, tmp_path, monkeypatch):
    owner = await lab_app.get_chunking_lab_coordinator()
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: other)
    with patch.object(owner, "close", side_effect=OSError("injected failure")):
        with pytest.raises(OSError):
            await lab_app.get_chunking_lab_coordinator()
        assert lab_app._chunking_lab_coordinator is owner
        assert not (other / "chunking_lab.sqlite3").exists()
    fresh = await lab_app.get_chunking_lab_coordinator()
    assert fresh is not owner
    assert fresh.session.profile_key == str(other)
    await fresh.close()


@pytest.mark.asyncio
async def test_cancel_button_reaps_real_noncooperative_child(
    lab_app, tmp_path, monkeypatch
):
    from tldw_chatbook.Chunking import lab_runner

    pidfile = tmp_path / "cancel-child.pid"
    child = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        + f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(100)"
    )
    monkeypatch.setattr(
        lab_runner, "_worker_command", lambda: [sys.executable, "-c", child]
    )
    route = resolve_screen_route("chunking_lab")
    async with lab_app.run_test(size=(80, 24)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        screen = lab_app.screen
        await screen.wait_until_ready()
        run = asyncio.create_task(screen.run_candidates())
        try:
            for _ in range(200):
                if pidfile.exists():
                    break
                await asyncio.sleep(0.01)
            assert pidfile.exists()
            await screen._refresh_session()
            assert await pilot.click("#lab-cancel")
            await asyncio.wait_for(run, 5)
            with pytest.raises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)
            assert not screen.coordinator.busy
            assert {
                r["status"] for r in screen.coordinator.session.results.values()
            } == {"canceled"}
        finally:
            await screen.coordinator.cancel()
            await run


@pytest.mark.asyncio
async def test_forced_app_crash_recovers_committed_invalid_draft_and_result(
    lab_app, tmp_path, monkeypatch
):
    profile = tmp_path / "crash-profile"
    profile.mkdir(mode=0o700)
    config_root = tmp_path / "subprocess-isolation"
    config_root.mkdir(mode=0o700)
    marker = tmp_path / "checkpoint-committed"
    code = r"""
import Tests.conftest
import asyncio, os
from pathlib import Path
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook import config
from tldw_chatbook.UI.Screens.chunking_lab_screen import ChunkingLabScreen
config.get_user_data_dir = lambda: Path(os.environ["LAB_CRASH_PROFILE"])
async def main():
    app = _build_test_app()
    async with app.run_test(size=(80,24)) as pilot:
        screen = ChunkingLabScreen(app)
        await app.push_screen(screen)
        await screen.wait_until_ready()
        screen.query_one("#lab-sample-text").load_text("committed exact sample")
        await pilot.pause()
        await screen.drain_edits()
        await screen.run_candidates()
        screen.query_one("#lab-json").load_text('{"unfinished":')
        await pilot.pause()
        await screen.drain_edits()
        assert await screen.confirm_navigation()
        Path(os.environ["LAB_CRASH_MARKER"]).write_text("committed")
        await asyncio.Event().wait()
asyncio.run(main())
"""
    environment = {
        **os.environ,
        "TLDW_TEST_CONFIG_ROOT": str(config_root),
        "LAB_CRASH_PROFILE": str(profile),
        "LAB_CRASH_MARKER": str(marker),
    }
    process = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(600):
            if marker.exists() or process.poll() is not None:
                break
            await asyncio.sleep(0.05)
        assert marker.exists(), f"child exited {process.poll()} before checkpoint"
        process.kill()
        assert await asyncio.to_thread(process.wait, 5) == -9
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait, 5)
    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: profile)
    coordinator = await lab_app.get_chunking_lab_coordinator()
    draft = next(
        c["draft"] for c in coordinator.session.candidates.values() if c["role"] == "B"
    )
    assert draft["raw_json"] == '{"unfinished":'
    assert len(coordinator.session.results) == 1
    assert next(iter(coordinator.session.results.values()))["status"] == "completed"
    assert not coordinator.busy
    await coordinator.close()


@pytest.mark.asyncio
async def test_completed_preview_and_invalid_json_survive_reopen(lab_app):
    route = resolve_screen_route("chunking_lab")
    assert route is not None
    async with lab_app.run_test(size=(120, 40)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        screen = lab_app.screen
        await screen.wait_until_ready()
        screen.query_one("#lab-sample-text").load_text("one two three four")
        await pilot.pause()
        await screen.drain_edits()
        await screen.run_candidates()
        screen.query_one("#lab-json").load_text('{"broken":')
        await pilot.pause()
        await screen.drain_edits()
        assert await screen.confirm_navigation()
        coordinator = await lab_app.get_chunking_lab_coordinator()
        assert len(coordinator.session.results) == 1
        await coordinator.close()
        lab_app._chunking_lab_coordinator = None
        restored = await lab_app.get_chunking_lab_coordinator()
        draft = next(
            c["draft"] for c in restored.session.candidates.values() if c["role"] == "B"
        )
        assert draft["raw_json"] == '{"broken":'
        assert draft["parse_error"]
        assert len(restored.session.results) == 1
        assert not restored.busy
