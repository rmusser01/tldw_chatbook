"""Pre-safety cancellation clears only proven untouched replacement targets."""

from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import bootstrap, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal


@pytest.mark.parametrize("boundary", ["pending", "prepared"])
def test_cancelled_pre_safety_replacement_can_abort_unchanged_targets(
    tmp_path, monkeypatch, boundary
):
    from tldw_chatbook.Backup_Recovery import control_records

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        originals = {path: path.read_bytes() for path in source.parent.iterdir()}
        cancel = Event()
        if boundary == "pending":
            register = control_records.register_pending

            def interrupted(*args, **kwargs):
                register(*args, **kwargs)
                cancel.set()

            monkeypatch.setattr(control_records, "register_pending", interrupted)
        else:

            def interrupted(*args, **kwargs):
                cancel.set()
                raise InterruptedError("cancel before safety capture")

            monkeypatch.setattr(replacement, "capture_verify_rollback", interrupted)
        with pytest.raises(InterruptedError):
            replacement.replace(
                replace(
                    plan,
                    acknowledged_credential_issues=("credential_format_unreadable",),
                ),
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"unused",
                cancel=cancel,
            )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        assert not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        result = replacement.recover_replacement(
            operation,
            control_root=tmp_path / "control",
            action="abort",
            rollback_password=None,
            cancel=Event(),
        )
        assert result == "aborted"
        assert all(path.read_bytes() == data for path, data in originals.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        assert rows[-1].event == "prepublication_aborted"
        assert not any(row.event == "rollback_verified" for row in rows)


def test_actual_rollback_credential_review_exposes_safe_issues_and_allows_abort(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        with pytest.raises(CaptureReviewRequired) as caught:
            replacement.replace(
                case[1],
                case[0],
                control_root=tmp_path / "control",
                rollback_password=b"unused",
                cancel=Event(),
            )
        assert caught.value.args == ("rollback_credential_coverage_changed",)
        assert caught.value.issues == ("credential_format_unreadable",)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        assert (
            replacement.recover_replacement(
                pending[0]["operation_id"],
                control_root=tmp_path / "control",
                action="abort",
                rollback_password=None,
                cancel=Event(),
            )
            == "aborted"
        )


_ABORT_CHILD = r"""
import os,sys
from pathlib import Path
from threading import Event
from dataclasses import replace
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap,replacement,control_records
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
root,mode,boundary,operation=sys.argv[1:]
root=Path(root)
bootstrap.default_bootstrap_root=lambda:root/'bootstrap'
if mode=='start':
 if boundary=='pending':
  original=control_records.register_pending
  def stop(*a,**kw):original(*a,**kw);os._exit(91)
  control_records.register_pending=stop
 else:
  def stop(*a,**kw):os._exit(91)
  replacement.capture_verify_rollback=stop
 journal=Journal(root/'control','held-sqlite')
 plan=load_plan(journal)
 with journal._locked(exclusive=False) as parent:
  candidate=Path(journal._records(parent)[0].evidence['stage']['path'])
 replacement.replace(replace(plan,acknowledged_credential_issues=('credential_format_unreadable',)),candidate,
  control_root=root/'control',rollback_password=b'unused',cancel=Event())
else:
 if boundary=='terminal':
  original=Journal._append
  def stop(self,parent,event,evidence):
   original(self,parent,event,evidence)
   if event=='prepublication_aborted':os._exit(91)
  Journal._append=stop
 replacement.validate_replacement_abort(operation,control_root=root/'control',cancel=Event())
 assert replacement.recover_replacement(operation,control_root=root/'control',action='abort',rollback_password=None,cancel=Event())=='aborted'
assert not blocked_attempts(),blocked_attempts()
"""


def _abort_child(tmp_path, mode, boundary, operation="", expected=0):
    import subprocess
    import sys

    with (tmp_path / f"abort-{mode}-{boundary}.log").open("w+") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _ABORT_CHILD,
                str(tmp_path),
                mode,
                boundary,
                operation,
            ],
            stdout=log,
            stderr=log,
            timeout=25,
            check=False,
        )
        log.seek(0)
        assert result.returncode == expected, log.read()[-7000:]


@pytest.mark.parametrize("boundary", ["pending", "prepared"])
def test_fresh_process_abort_and_terminal_retry_preserve_originals(
    tmp_path, monkeypatch, boundary
):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        originals = {path: path.read_bytes() for path in case[4].parent.iterdir()}
        _abort_child(tmp_path, "start", boundary, expected=91)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        _abort_child(tmp_path, "abort", "terminal", operation, expected=91)
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        _abort_child(tmp_path, "abort", "finish", operation)
        assert all(path.read_bytes() == data for path, data in originals.items())
        assert bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]


@pytest.mark.parametrize(
    "change", ["bytes", "metadata", "inode", "unknown_child", "candidate", "pending"]
)
def test_abort_refuses_changed_native_proof_without_clearing_fence(
    tmp_path, monkeypatch, change
):
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _abort_child(tmp_path, "start", "prepared", expected=91)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        if change == "bytes":
            case[4].write_bytes(case[4].read_bytes() + b"changed")
        elif change == "metadata":
            case[4].chmod(0o400)
        elif change == "inode":
            data = case[4].read_bytes()
            case[4].unlink()
            case[4].write_bytes(data)
        elif change == "unknown_child":
            (case[4].parent / "unexpected").write_bytes(b"present")
        elif change == "candidate":
            (case[0] / "candidate.json").write_bytes(b"changed")
        else:
            name = "pending-" + bootstrap._key(operation) + ".json"
            changed = dict(pending[0], selectors=[str(case[4])])
            with pinned_directory(tmp_path / "bootstrap") as parent:
                import json
                import os

                os.unlink(name, dir_fd=parent)
                Admission._write_new_record(parent, name, json.dumps(changed).encode())
        with pytest.raises(ValueError):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="abort",
                rollback_password=None,
                cancel=Event(),
            )
        assert (
            tmp_path / "bootstrap" / ("pending-" + bootstrap._key(operation) + ".json")
        ).exists()
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "prepared"


def test_verified_safety_copy_cannot_use_prepublication_abort(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import crypto, publication

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:

        def stop(*args, **kwargs):
            raise InterruptedError("verified safety boundary")

        monkeypatch.setattr(publication, "publish_candidate", stop)
        with pytest.raises(InterruptedError):
            replacement.replace(
                replace(
                    case[1],
                    acknowledged_credential_issues=("credential_format_unreadable",),
                ),
                case[0],
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        with pytest.raises(ValueError, match="prepublication_abort_unavailable"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="abort",
                rollback_password=None,
                cancel=Event(),
            )
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        # The qualified existing reverse route is still available with actual decryption.
        assert (
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="rollback",
                rollback_password=b"rollback",
                cancel=Event(),
            )
            == "rolled_back"
        )


def test_abort_validator_does_not_clear_pending_or_mint_public_proof(
    tmp_path, monkeypatch
):
    with replacement_case(tmp_path, monkeypatch, prepared=False):
        _abort_child(tmp_path, "start", "pending", expected=91)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        journal = Journal(tmp_path / "control", operation)
        replacement.validate_replacement_abort(
            operation, control_root=tmp_path / "control", cancel=Event()
        )
        assert bootstrap._records(tmp_path / "bootstrap")[0] == pending
        with pytest.raises(ValueError, match="recovery_execution_required"):
            journal.record("prepublication_aborted", {})
        for action in ("finish", "rollback"):
            with pytest.raises(ValueError, match="prepublication_abort_required"):
                replacement.recover_replacement(
                    operation,
                    control_root=tmp_path / "control",
                    action=action,
                    rollback_password=b"unused",
                    cancel=Event(),
                )
        with journal._locked(exclusive=False) as parent:
            assert [row.event for row in journal._records(parent)] == [
                "candidate_staged"
            ]


def test_aborted_unchanged_operation_can_retry_as_a_new_verified_replacement(
    tmp_path, monkeypatch, helper_resource_root
):
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        descriptor = replacement._descriptor(case[0], case[1])
        expected_config = Path(
            next(
                row["candidate"]
                for row in descriptor["artifacts"]
                if row["destination"] == str(case[5])
            )
        ).read_bytes()
        _abort_child(tmp_path, "start", "pending", expected=91)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        replacement.recover_replacement(
            operation,
            control_root=tmp_path / "control",
            action="abort",
            rollback_password=None,
            cancel=Event(),
        )
        completed = replacement.replace(
            replace(
                case[1],
                acknowledged_credential_issues=("credential_format_unreadable",),
            ),
            case[0],
            control_root=tmp_path / "control",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        assert completed != operation
        assert case[5].read_bytes() == expected_config
        journal = Journal(tmp_path / "control", completed)
        with journal._locked(exclusive=False) as parent:
            events = [row.event for row in journal._records(parent)]
        assert "rollback_verified" in events and events[-1] == "committed"


def test_target_change_after_abort_terminal_keeps_fence(tmp_path, monkeypatch):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _abort_child(tmp_path, "start", "pending", expected=91)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        append = Journal._append

        def mutate(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == "prepublication_aborted":
                case[4].chmod(0o400)

        monkeypatch.setattr(Journal, "_append", mutate)
        with pytest.raises(ValueError, match="target_changed"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="abort",
                rollback_password=None,
                cancel=Event(),
            )
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]


def test_abort_preserves_explicit_unselected_sibling_outside_held_scope(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import control_records
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        sibling = tmp_path / "other-profile.toml"
        sibling.write_bytes(b"untouched")
        sibling.chmod(0o600)
        original = sibling.stat()
        archive = replacement._acquired_source(case[0], case[1], Event())
        target = replace(
            case[1].target,
            items=(
                *case[1].target.items,
                StorageItem("config", "profile:other:config", sibling, "included", ()),
            ),
        )
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dict((*case[1].destinations, *case[1].selectors)),
            target=target,
            profile_names=dict(case[1].profile_names),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        assert ("profile:other:config", sibling) in plan.preserve
        with control_records.admission_authority(tmp_path / "bootstrap").maintenance(
            ("bootstrap.unbound",), 3
        ) as session:
            candidate = stage_restore(
                archive, plan, tmp_path / "sibling-stage", Event(), session=session
            )
        register = control_records.register_pending
        cancel = Event()

        def stop(*args, **kwargs):
            register(*args, **kwargs)
            cancel.set()

        monkeypatch.setattr(control_records, "register_pending", stop)
        with pytest.raises(InterruptedError):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"unused",
                cancel=cancel,
            )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        assert pending[0]["namespaces"] == ["profile"]
        assert (
            replacement.recover_replacement(
                pending[0]["operation_id"],
                control_root=tmp_path / "control",
                action="abort",
                rollback_password=None,
                cancel=Event(),
            )
            == "aborted"
        )
        assert sibling.read_bytes() == b"untouched" and sibling.stat() == original
