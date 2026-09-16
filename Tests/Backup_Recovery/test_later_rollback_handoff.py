"""Later rollback leaves ordinary writers before obtaining a fresh review."""

import pytest


@pytest.mark.parametrize("value", [None, 0, 1, "copies"])
def test_recovery_copies_hint_rejects_non_bool(tmp_path, value):
    from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart

    with pytest.raises(ValueError, match="invalid_recovery_restart"):
        RecoveryRestart(None, tmp_path / "config.toml", recovery_copies=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["review", "start", "handoff"])
async def test_ordinary_later_rollback_requires_fresh_recovery_host(
    tmp_path, monkeypatch, action
):
    from textual.app import App
    from textual.widgets import Button, Checkbox, Input, Static

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")
    calls = []
    forbidden = []

    def unexpected(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError("ordinary app must not review or execute later rollback")

    class Host(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

        def request_recovery_restart(self, archive, target, *, recovery_copies=False):
            calls.append((archive, target, recovery_copies))

    app = Host()
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            screen = app.screen
            screen._show_mode("copies")
            screen._select_rollback(Button.Pressed(Button("copy", name="old-copy")))
            selected = tmp_path / "config.toml"
            screen.query_one("#backup-later-target", Input).value = str(selected)
            for key in (
                "backup-copy-password",
                "backup-safety-password",
                "backup-safety-confirm",
            ):
                screen.query_one("#" + key, Input).value = "fixture-secret"
            await pilot.pause()
            # Simulate an otherwise fully reviewed stale callback, not just a
            # disabled button. The guard must precede every service call.
            with screen.query_one("#backup-later-confirm", Checkbox).prevent(
                Checkbox.Changed
            ):
                screen.query_one("#backup-later-confirm", Checkbox).value = True
            screen._rollback_plan = object()
            screen._rollback_availability = (True, None)
            monkeypatch.setattr(screen, "_preview_rollback", unexpected)
            monkeypatch.setattr(screen, "_start_rollback_operation", unexpected)
            if action == "review":
                screen._review_rollback()
            elif action == "start":
                screen._start_rollback()
            else:
                button = screen.query_one("#backup-later-restart", Button)
                button.focus()
                await pilot.press("enter")
                assert calls == [(None, selected, True)]
                assert screen._rollback_plan is None
                assert screen._rollback_availability is None
                assert not screen.query_one("#backup-later-confirm", Checkbox).value
                assert all(
                    screen.query_one("#" + key, Input).value == ""
                    for key in (
                        "backup-copy-password",
                        "backup-safety-password",
                        "backup-safety-confirm",
                    )
                )
            assert not forbidden
            assert service.current() is None
            assert screen.query_one("#backup-later-review", Button).disabled
            assert screen.query_one("#backup-later-restart", Button).display
            assert (
                "recovery mode"
                in str(
                    screen.query_one("#backup-later-restart-note", Static).render()
                ).lower()
            )
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["copies", "invalid"])
def test_fresh_child_copies_hint_requires_selection_and_review(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "restart", mode, script=_COPIES_CHILD)


_COPIES_CHILD = r"""
import asyncio, os, sys
from pathlib import Path
from textual.app import App
from textual.widgets import Button, Input, Static
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import recovery_restart
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_bytes(b'invalid existing configuration')
original=selector.read_bytes()
mode=sys.argv[2]
entered=[]
def headless(app):
 entered.append(True)
 async def main():
  async with app.run_test(size=(100,36)) as pilot:
   screen=app.screen
   assert screen._mode=='copies'
   assert screen.query_one('#backup-later-target',Input).value==str(selector)
   assert screen._rollback_copy_id is None and screen._rollback_plan is None
   assert not screen.query_one('#backup-later-form').display
   assert screen.query_one('#backup-later-start',Button).disabled
   assert not screen.query_one('#backup-later-restart').display
   assert all(screen.query_one('#'+key,Input).value=='' for key in (
    'backup-copy-password','backup-safety-password','backup-safety-confirm'))
   await screen.workers.wait_for_complete()
   await pilot.pause()
   rows=list(screen.query_one('#backup-list').query(Static))
   assert len(rows)==1 and 'No local entries.' in str(rows[0].render()), ([str(row.render()) for row in rows],screen._list_delivery)
   revision=screen._revision
   screen._rollback_plan=object()
   screen.query_one('#backup-later-target',Input).value=str(selector.with_name('other.toml'))
   await pilot.pause()
   assert screen._revision>revision and screen._rollback_plan is None
  assert app.recovery_service._closed
 asyncio.run(main())
App.run=headless
sys.argv=['probe','',str(selector),mode]
try:
 exec(recovery_restart._ENTRY)
except ValueError as error:
 assert mode=='invalid' and str(error)=='invalid_recovery_restart'
 assert not entered
except SystemExit as error:
 assert mode=='copies' and error.code==0 and entered==[True]
else:
 raise AssertionError('entry must exit or reject')
assert selector.read_bytes()==original
assert 'tldw_chatbook.app' not in sys.modules
assert 'tldw_chatbook.config' not in sys.modules
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


def test_installed_excluded_log_append_invalidates_before_later_copy_access(tmp_path):
    from dataclasses import replace
    from threading import Event

    from tldw_chatbook.Backup_Recovery.config_adapter import _Diagnostics
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_directory,
        create_private_file,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        LocalSnapshotSource,
        RestorePlan,
        _fingerprint,
        _paths,
        recheck_targets,
    )

    data = tmp_path / "data"
    create_private_directory(data)
    user = data / "fixture"
    create_private_directory(user)
    logfile = user / "tldw_cli_app.log"
    with create_private_file(logfile) as fd:
        import os

        os.write(fd, b"Before review\n")
    selected = tmp_path / "config.toml"
    document = {
        "general": {"users_name": "fixture"},
        "paths": {"data_dir": str(data)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selected, "fixture"),
    }
    (item,) = _Diagnostics("diagnostics.logs").discover(document)
    assert item.path == logfile and item.status == "intentionally_excluded"
    target = classify_entries((item,))
    control = tmp_path / "control-never-opened"
    plan = RestorePlan(
        "unused-archive-digest",
        "replace",
        (),
        (),
        ((item.logical_id, logfile),),
        "",
        target=target,
        local_snapshot=LocalSnapshotSource(
            control, "fixture-operation", "unused-copy-digest"
        ),
    )
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), plan.target))
    recheck_targets(plan)
    original = logfile.stat()
    with logfile.open("ab") as stream:
        stream.write(b"UI event_loop_stall after review\n")
    assert logfile.stat().st_ino == original.st_ino
    (current,) = _Diagnostics("diagnostics.logs").discover(document)
    assert (current.owner, current.logical_id, current.path, current.status) == (
        item.owner,
        item.logical_id,
        item.path,
        item.status,
    )
    with pytest.raises(ValueError, match="^target_changed$"):
        execute_rollback(
            "fixture-operation",
            control_root=control,
            old_password=b"old-fixture-password",
            new_password=b"new-fixture-password",
            cancel=Event(),
            approved_plan=plan,
        )
    assert not control.exists()
