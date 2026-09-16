"""Retained encrypted copies remain protected by actual operation evidence."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery import recovery_copies as copies


@pytest.mark.parametrize(
    "pending,held,selected",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_pending_or_held_copy_cannot_be_deleted(pending, held, selected):
    assert not copies.deletion_allowed(
        pending_operation=pending, active_hold=held, user_selected=selected
    )


def test_completed_copy_can_be_explicitly_deleted():
    assert copies.deletion_allowed(
        pending_operation=False, active_hold=False, user_selected=True
    )


def test_real_completed_copy_list_hold_and_exact_deletion(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        control = tmp_path / "control"
        operation = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"private rollback password",
            cancel=Event(),
        )
        entries = copies.list_recovery_copies(control)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.operation_id == operation
        assert entry.status == "verified" and not entry.pending_operation
        archive = Path(entry.path)
        before = archive.read_bytes()
        assert before.startswith(b"age-encryption.org/v1")
        assert entry.size == len(before) and entry.coverage
        unrelated = archive.with_name("unrelated.age")
        unrelated.write_bytes(b"leave unrelated bytes")
        with copies.hold_recovery_copy(control, operation) as held:
            assert held == entry
            with pytest.raises(ValueError, match="recovery_copy_held"):
                copies.delete_recovery_copy(control, operation, user_selected=True)
            assert archive.read_bytes() == before
        with pytest.raises(ValueError, match="recovery_copy_delete_not_selected"):
            copies.delete_recovery_copy(control, operation, user_selected=False)
        archive.write_bytes(b"changed ciphertext")
        assert copies.list_recovery_copies(control)[0].status == "changed"
        with pytest.raises(ValueError, match="recovery_copy_changed"):
            copies.delete_recovery_copy(control, operation, user_selected=True)
        archive.write_bytes(before)
        saved = archive.with_name("saved-original.age")
        archive.rename(saved)
        archive.symlink_to(unrelated)
        assert copies.list_recovery_copies(control)[0].status == "changed"
        with pytest.raises(ValueError, match="recovery_copy_changed"):
            copies.delete_recovery_copy(control, operation, user_selected=True)
        archive.unlink()
        saved.rename(archive)
        script = """
import sys
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery.recovery_copies import hold_recovery_copy
with hold_recovery_copy(Path(sys.argv[1]), sys.argv[2]):
 print('held', flush=True)
 sys.stdin.readline()
"""
        with subprocess.Popen(
            [sys.executable, "-c", script, str(control), operation],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as child:
            try:
                assert child.stdout.readline().strip() == "held"
                with pytest.raises(ValueError, match="recovery_copy_held"):
                    copies.delete_recovery_copy(control, operation, user_selected=True)
            finally:
                child.communicate("release\n", timeout=10)
            assert child.returncode == 0
        copies.delete_recovery_copy(control, operation, user_selected=True)
        assert not archive.exists()
        assert unrelated.read_bytes() == b"leave unrelated bytes"
        assert source.exists() and selector.exists()
        assert copies.list_recovery_copies(control)[0].status == "missing"
        from tldw_chatbook.Backup_Recovery.activation import replacement_installation_id

        assert replacement_installation_id() is not None


def test_real_pending_copy_and_corrupt_journal_are_not_deletable(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, _, _ = case
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        control = tmp_path / "control"
        retire = publication._retire

        def interrupted(item):
            retire(item)
            raise InterruptedError("after native retirement")

        monkeypatch.setattr(publication, "_retire", interrupted)
        with pytest.raises(InterruptedError):
            replacement.replace(
                plan,
                candidate,
                control_root=control,
                rollback_password=b"rollback",
                cancel=Event(),
            )
        operation = bootstrap._records(tmp_path / "bootstrap")[0][0]["operation_id"]
        entry = copies.list_recovery_copies(control)[0]
        assert entry.pending_operation and entry.status == "verified"
        original = entry.path.read_bytes()
        with pytest.raises(ValueError, match="recovery_copy_pending"):
            copies.delete_recovery_copy(control, operation, user_selected=True)
        assert entry.path.read_bytes() == original
        journal = copies._journal(control, operation)
        (journal.root / "000000.json").write_bytes(b"invalid evidence")
        assert copies.list_recovery_copies(control)[0].status == "recovery_required"
        with pytest.raises(ValueError):
            copies.delete_recovery_copy(control, operation, user_selected=True)
        assert entry.path.read_bytes() == original
