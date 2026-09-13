"""Actual retained F9 copy, explicit later safety omissions and new recovery copy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)
from Tests.Backup_Recovery.test_f9_replacement_workflow import (
    test_full_f9_replacement_after_explicit_safety_and_credential_review as _earn_replacement,
)

_LATER = r"""
from Tests.network_guard import install, blocked_attempts
install()
import asyncio
import json
import os
from pathlib import Path
import sys
import time
home = Path.home()
saved = json.loads((home / "probe-mapping.json").read_text())
previous = json.loads((home / "probe-result.json").read_text())
operation = previous["result"]["journal_operation_id"]
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed = Path(os.environ["TLDW_TEST_INSTALLED_PACKAGE"])
assert installed == Path(saved["package"])
assert Path(tldw_chatbook.__file__).resolve() == installed / "tldw_chatbook" / "__init__.py"
from textual.widgets import Button, Input, Checkbox, Static
from tldw_chatbook.Backup_Recovery.launcher import recovery_app

assert "tldw_chatbook.app" not in sys.modules
assert "tldw_chatbook.config" not in sys.modules

def retain(checkpoint, **values):
    record = {"checkpoint": checkpoint, "time": time.time(), **values}
    (home / "later-ui-probe-result.json").write_text(
        json.dumps(record, default=str)
    )
    with (home / "later-ui-checkpoints.jsonl").open("a") as output:
        output.write(json.dumps(record, default=str) + "\n")
    print(checkpoint, values, flush=True)

async def main():
    app = recovery_app("replacement_requested")
    async with app.run_test(size=(120, 42)) as pilot:
        screen = app.screen
        screen.query_one("#backup-open-copies", Button).focus()
        await pilot.press("enter")
        async with asyncio.timeout(90 if sys.platform == "win32" else 15):
            while not list(screen.query(".backup-review-rollback")):
                await asyncio.sleep(.03)
        choices = [b for b in screen.query(".backup-review-rollback") if b.name == operation]
        assert len(choices) == 1 and not choices[0].disabled
        retain("verified_copy_listed", operation=operation)
        choices[0].focus()
        await pilot.press("enter")
        screen.query_one("#backup-later-target", Input).value = os.environ["TLDW_CONFIG_PATH"]
        screen.query_one("#backup-copy-password", Input).value = "test-only-new-safety-password"
        await pilot.pause()
        # Hold the first actual native preview result, then change the input.
        # The old result must be discarded rather than authorizing that new input.
        import threading
        from tldw_chatbook.Backup_Recovery import later_rollback
        original_preview=later_rollback.preview_rollback
        entered,release=threading.Event(),threading.Event()
        def held_preview(*args,**kwargs):
            result=original_preview(*args,**kwargs)
            entered.set()
            assert release.wait(60 if sys.platform == "win32" else 10)
            return result
        later_rollback.preview_rollback=held_preview
        try:
            screen.query_one("#backup-later-review", Button).focus()
            await pilot.press("enter")
            async with asyncio.timeout(180 if sys.platform == "win32" else 15):
                while not entered.is_set():await asyncio.sleep(.03)
            screen.query_one('#backup-later-target',Input).value=os.environ['TLDW_CONFIG_PATH']+'.changed'
            await pilot.pause()
        finally:
            release.set()
            later_rollback.preview_rollback=original_preview
        await screen.workers.wait_for_complete()
        assert screen._rollback_plan is None and screen.query_one('#backup-later-start',Button).disabled
        assert not screen._later_review_codes_seen
        retain('stale_native_preview_discarded')
        screen.query_one('#backup-later-target',Input).value=os.environ['TLDW_CONFIG_PATH']
        screen.query_one('#backup-copy-password',Input).value='test-only-new-safety-password'
        await pilot.pause()
        screen.query_one("#backup-later-review", Button).focus()
        await pilot.press("enter")
        async with asyncio.timeout(180 if sys.platform == "win32" else 45):
            while screen.query_one("#backup-later-start", Button).disabled:
                text = str(screen.query_one("#backup-later-preview", Static).render())
                if "refused" in text:
                    state = app.recovery_service.current()
                    retain("review_refused", text=text,
                           state={key: state[key] for key in ("state", "phase", "issues", "review_issues")})
                    raise AssertionError(text)
                await asyncio.sleep(.03)
        assert screen._rollback_plan is not None
        retain("later_rollback_reviewed")
        screen.query_one("#backup-copy-password", Input).value = "test-only-new-safety-password"
        screen.query_one("#backup-safety-password", Input).value = "test-only-later-safety-password"
        screen.query_one("#backup-safety-confirm", Input).value = "test-only-later-safety-password"
        screen.query_one("#backup-later-confirm", Checkbox).value = True
        await pilot.pause()
        previous_operation = app.recovery_service.current()["operation_id"]
        screen.query_one("#backup-later-start", Button).focus()
        await pilot.press("enter")
        async with asyncio.timeout(300 if sys.platform == "win32" else 65):
            await pilot.pause()
            await asyncio.gather(*(worker.wait() for worker in list(app.workers) if worker.group == "backup-later-start"))
            current = app.recovery_service.current()
            assert current["kind"] == "later_rollback" and current["operation_id"] != previous_operation, dict(current)
            state = await asyncio.to_thread(app.recovery_service.wait, current["operation_id"], timeout=300 if sys.platform == "win32" else 65)
        retain("later_rollback_terminal",
               state={key: state[key] for key in ("state", "phase", "issues", "review_issues")}, result=dict(state["result"]))
        assert state['state']=='recovery_required',dict(state)
        expected=tuple(state['review_issues'])
        assert expected and all(code.startswith('credential_') for code in expected)
        async with asyncio.timeout(60 if sys.platform == "win32" else 10):
            while len(list(screen.query('.backup-acknowledge-later-credential')))!=len(expected):
                await asyncio.sleep(.03)
        boxes=list(screen.query('.backup-acknowledge-later-credential'))
        assert {box.name for box in boxes}==set(expected) and not any(box.value for box in boxes)
        retain('later_omissions_visible_unchecked',count=len(boxes))
        async with asyncio.timeout(90 if sys.platform == "win32" else 15):
            while not list(screen.query('.backup-recover-abort')):
                await asyncio.sleep(.03)
        screen.query_one('.backup-recover-abort',Button).focus()
        await pilot.press('enter')
        aborted=await asyncio.to_thread(app.recovery_service.wait,app.recovery_service.current()['operation_id'],timeout=90 if sys.platform == "win32" else 20)
        assert aborted['result']['aborted'],dict(aborted)
        retain('later_aborted_untouched')
        for box in boxes:box.value=True
        screen.query_one('#backup-copy-password',Input).value='test-only-new-safety-password'
        await pilot.pause()
        assert screen._rollback_plan is None
        screen.query_one('#backup-later-review',Button).focus()
        await pilot.press('enter')
        async with asyncio.timeout(180 if sys.platform == "win32" else 45):
            while screen.query_one('#backup-later-start',Button).disabled:
                text=str(screen.query_one('#backup-later-preview',Static).render())
                assert 'refused' not in text,text
                await asyncio.sleep(.03)
        assert set(screen._rollback_plan.acknowledged_credential_issues)==set(expected)
        retain('omissions_reviewed')
        reviewed=screen._rollback_plan
        boxes[0].value=False
        await pilot.pause()
        assert screen._rollback_plan is None and not screen.query_one('#backup-later-confirm',Checkbox).value
        assert set(reviewed.acknowledged_credential_issues)==set(expected)
        boxes[0].value=True
        screen.query_one('#backup-copy-password',Input).value='test-only-new-safety-password'
        await pilot.pause()
        screen.query_one('#backup-later-review',Button).focus()
        await pilot.press('enter')
        async with asyncio.timeout(180 if sys.platform == "win32" else 45):
            while screen.query_one('#backup-later-start',Button).disabled:await asyncio.sleep(.03)
        assert set(screen._rollback_plan.acknowledged_credential_issues)==set(expected)
        retain('changed_acknowledgement_reviewed_again')
        before_confirm=app.recovery_service.current()['operation_id']
        screen.query_one('#backup-later-start',Button).focus()
        await pilot.press('enter')
        assert app.recovery_service.current()['operation_id']==before_confirm
        screen.query_one('#backup-copy-password',Input).value='test-only-new-safety-password'
        screen.query_one('#backup-safety-password',Input).value='test-only-later-safety-password'
        screen.query_one('#backup-safety-confirm',Input).value='test-only-later-safety-password'
        screen.query_one('#backup-later-confirm',Checkbox).value=True
        await pilot.pause()
        # Textual ignores a repeated Enter while the prior refusal is still active.
        async with asyncio.timeout(30 if sys.platform == "win32" else 1):
            while screen.query_one('#backup-later-start',Button).has_class('-active'):
                await asyncio.sleep(.01)
        screen.query_one('#backup-later-start',Button).focus()
        await pilot.press('enter')
        async with asyncio.timeout(90 if sys.platform == "win32" else 15):
            while app.recovery_service.current()['operation_id']==before_confirm:
                await asyncio.sleep(.03)
        accepted=app.recovery_service.current()
        assert accepted['kind']=='later_rollback',dict(accepted)
        state=await asyncio.to_thread(app.recovery_service.wait,accepted['operation_id'],timeout=360 if sys.platform == "win32" else 65)
        retain('later_rollback_complete',state=state['state'],result=dict(state['result']))
        assert state['state']=='succeeded' and state['result']['restoration_validated'],dict(state)
        copies=await asyncio.to_thread(app.recovery_service.recovery_copies)
        assert any(row.operation_id==state['result']['journal_operation_id'] and row.status=='verified' for row in copies)
        assert any(row.operation_id==operation and row.status=='verified' for row in copies)
        new_copy=state['result']['journal_operation_id']
        async with asyncio.timeout(90 if sys.platform == "win32" else 15):
            while not [b for b in screen.query('.backup-review-rollback') if b.name==new_copy]:await asyncio.sleep(.03)
        next_copy=next(b for b in screen.query('.backup-review-rollback') if b.name==new_copy)
        next_copy.focus();await pilot.press('enter')
        assert screen._rollback_copy_id==new_copy and screen._rollback_plan is None
        assert not screen._later_review_codes_seen and not any(box.value for box in boxes)
        inspected=app.recovery_service.start_copy_inspection(new_copy,password=b'test-only-later-safety-password')
        checked=await asyncio.to_thread(app.recovery_service.wait,inspected,timeout=180 if sys.platform == "win32" else 20)
        assert checked['state']=='succeeded' and checked['result']['archive_verified'],dict(checked)
        from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
        assert verify_sealed(app.recovery_service.inspection(inspected)).credential_policy=='rollback'
    assert not blocked_attempts(), blocked_attempts()

asyncio.run(main())
assert not blocked_attempts(), blocked_attempts()
"""


def test_f9_later_rollback_requires_explicit_credential_review(
    tmp_path, native_package
):
    _earn_replacement(tmp_path, native_package)
    test_root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join((str(native_package), str(test_root))),
        TLDW_TEST_INSTALLED_PACKAGE=str(native_package),
    )
    log = tmp_path / "later-child.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", _LATER],
            cwd=tmp_path,
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
            # Four native reviews, two execution attempts, Abort, and readback.
            # Their individual operation/UI deadlines remain independently bounded.
            timeout=900 if sys.platform == "win32" else 180,
            check=False,
        )
    assert result.returncode == 0, log.read_text()[-10000:]


def test_rollback_preview_transport_preserves_nonsecret_review(tmp_path, monkeypatch):
    """Component transport check; the full UI case supplies native omission proof."""
    from tldw_chatbook.Backup_Recovery import later_rollback
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    issues = ("credential_format_unreadable", "credential_missing:test-slot")

    def require_review(*args, **kwargs):
        raise CaptureReviewRequired(issues)

    monkeypatch.setattr(later_rollback, "preview_rollback", require_review)
    service = RecoveryService(tmp_path / "control")
    try:
        with pytest.raises(CaptureReviewRequired) as refusal:
            service.preview_rollback(
                "test-copy", old_password=b"test-only", target=None
            )
        assert refusal.value.issues == issues
        assert service.current()["review_issues"] == issues
    finally:
        service.close()
