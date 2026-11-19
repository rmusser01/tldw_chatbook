"""Actual complete-profile packaging of native wave evidence."""

import asyncio
import json
import os
import threading
from pathlib import Path


def durable_facts(connection, workflow_id):
    return {
        table: [list(row) for row in connection.execute(query, (workflow_id,))]
        for table, query in {
            "windows": "SELECT id,phase_attempt_id,ordinal,anchor_attempt_id,roots_json,state,baseline,ending,snapshot_count,reserved_bytes FROM workflow_change_windows WHERE workflow_id=? ORDER BY ordinal",
            "refs": "SELECT r.window_id,r.snapshot_id,s.run_id,s.root,s.baseline_sha,s.end_sha,s.kind FROM workflow_change_window_refs r JOIN workflow_change_windows w ON w.id=r.window_id JOIN change_snapshots s ON s.id=r.snapshot_id WHERE w.workflow_id=? ORDER BY r.snapshot_id",
            "checkpoints": "SELECT attempt_id,payload_hash,payload_json FROM workflow_checkpoints WHERE workflow_id=? ORDER BY attempt_id",
            "results": "SELECT attempt_id,payload_hash,payload_json FROM workflow_results WHERE workflow_id=? ORDER BY attempt_id",
        }.items()
    }


async def capture_profile(app, destination):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app

    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    options = {"staging_parent": Path.home()}
    preview = preview_capture((selector,), options=options)
    assert preview.complete, preview.issues
    monitor = asyncio.create_task(monitor_app(app))
    cancel = threading.Event()
    watchdog = asyncio.get_running_loop().call_later(30, cancel.set)
    try:
        captured = await asyncio.to_thread(
            capture,
            (selector,),
            preview.scope_digest,
            destination,
            options=options,
            cancel=cancel,
        )
        async with asyncio.timeout(10):
            while (
                storage._pause is not None
                or app._backup_runtime_maintenance is not None
            ):
                await asyncio.sleep(0.01)
        manifest = json.loads(captured.manifest_bytes)
        assert captured.inventory.complete and manifest["consistency"] == "coherent"
        assert any(item["owner_id"] == "db.agent_runs" for item in manifest["files"])
        await asyncio.to_thread(
            write_archive, captured, destination, password=None, cancel=cancel
        )
        assert destination.is_file()
    finally:
        watchdog.cancel()
        cancel.set()
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass


REOPEN = r"""
import asyncio,json,os,sqlite3
from pathlib import Path
from Tests import network_guard,real_profile_guard
network_guard.install();real_profile_guard.install()
from Tests.windows_private_fixture_runner import user_fixture_default_owner
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
home=Path.home()
receipt=json.loads((home/'restored.json').read_text())
select_profile(receipt['profile'],home/'control')
from tldw_chatbook import app as app_source
from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from Tests.Chat.workflow_wave_archive import capture_profile,durable_facts
async def main():
    creators=OriginalPrivateAppCreators(app_source,home.parent)
    creators.install()
    app=app_source.TldwCli()
    creators.finish_construction(app)
    try:
        facts=json.loads((home/'wave.json').read_text())
        with sqlite3.connect(Path(receipt['data'])/'agent_runs.db') as conn:
            assert durable_facts(conn,facts['workflow_id'])==facts['facts']
        assert not app.console_runtime._workflow_waves
        assert not app.console_runtime.execution_capacity.snapshot().executions
        await capture_profile(app,home/'second.tldw-backup.zip')
        print('WAVE_RESTORED_INERT_REQUIRED_EVIDENCE_REBACKED_UP',flush=True)
    finally:
        await app.console_runtime.dispose()
        creators.retire()
with user_fixture_default_owner():
    asyncio.run(main())
print('retired and reopened')
"""


def restore_and_rebackup(original):
    """Restore the complete native profile in a separate process and package it again."""
    import os
    import subprocess
    import sys

    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_temporary_media_capture import _RESTORE

    restored = original.parent / "restored"
    restored.mkdir()
    restore = _RESTORE.replace(
        "source.rename(source.with_name('source-home-removed'))",
        "(home/'wave.json').write_bytes((source/'wave.json').read_bytes())\nsource.rename(source.with_name('source-home-removed'))",
    )
    restore = restore.replace(
        "'agents.history':'tool_sandbox'",
        "'workspaces.change_tracking':'change_review','tokenizers.custom':'tokenizers','agents.history':'tool_sandbox'",
    )
    restore = restore.replace(
        "mapping[f'profile:{profile}:paths.data_dir']",
        "(home / '.config/tldw_cli/tokenizers').mkdir(parents=True, mode=0o700, exist_ok=True)\nfor row in doc.directories:\n    if row.parent_id is None and not row.synthetic and producer[row.logical_id].owner_id == 'tokenizers.custom':\n        mapping[row.logical_id] = home / '.config/tldw_cli/tokenizers'\nmapping[f'profile:{profile}:paths.data_dir']",
    )
    _run(restored, str(original / "home"), "isolated", script=restore, timeout=60)
    environment = dict(
        os.environ,
        HOME=str(restored / "home"),
        USERPROFILE=str(restored / "home"),
        XDG_CONFIG_HOME=str(restored / "config"),
        XDG_DATA_HOME=str(restored / "data"),
        TLDW_CONFIG_PATH=str(restored / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", REOPEN],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (original.parent / "wave-rebackup.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-4000:]
    assert (restored / "home/second.tldw-backup.zip").is_file()
