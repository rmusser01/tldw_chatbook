"""Stable file namespaces survive a real interrupted replacement rename."""

import tomllib
from pathlib import Path

import pytest

from Tests.Backup_Recovery import test_replacement_recovery as recovery_fixture
from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from Tests.Backup_Recovery.test_replacement_recovery import _child
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.journal import Journal

_SELECTED_CRASH = r"""
if mode=='start':
 if action_boundary=='retire_native':
  original=publication._retire
  def selected_retire(item):
   original(item)
   if item.target==str(crash_target):os._exit(91)
  publication._retire=selected_retire
 else:
  original=publication._complete_move
  def selected_observed(journal,parent,prepared,intent,**kwargs):
   original(journal,parent,prepared,intent,**kwargs)
   selected=intent.destination if action_boundary=='published' else intent.source.path
   step='publish' if action_boundary=='published' else 'retire'
   if intent.step==step and selected==str(crash_target):os._exit(91)
  publication._complete_move=selected_observed
"""


def _selected_crash(monkeypatch, boundary):
    child_program = recovery_fixture._CHILD.replace(
        "if mode=='start':",
        "action_boundary="
        + repr(boundary)
        + "\ncrash_target=root/"
        + repr("live/config.toml")
        + "\n"
        + _SELECTED_CRASH
        + "\nif mode=='start':",
    )
    monkeypatch.setattr(recovery_fixture, "_CHILD", child_program)


@pytest.mark.parametrize("action", ["finish", "rollback"])
@pytest.mark.parametrize("boundary", ["retire_native", "move_observed"])
@pytest.mark.parametrize("tree", [False, True])
def test_registered_file_namespace_recovers_native_rename_gap(
    tmp_path, monkeypatch, helper_resource_root, action, boundary, tree
):
    with replacement_case(tmp_path, monkeypatch, prepared=False, tree=tree) as case:
        _selected_crash(monkeypatch, boundary)
        selector = case[5]
        original = selector.read_bytes()
        authority = admission_authority(tmp_path / "bootstrap")
        authority.register("aaa.selected.config.file", (selector,))
        alias = tmp_path / "config-alias"
        alias.symlink_to(selector)
        authority.register("aaa.selected.config.alias", (alias,))
        _child(tmp_path, helper_resource_root, "start", action, "none", expected=91)
        assert not selector.exists()
        with (
            pytest.raises(FileNotFoundError),
            authority.normal(("aaa.selected.config.file",)),
        ):
            pytest.fail("Ordinary admission must not accept a missing file")
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        assert len(pending) == 1
        _child(
            tmp_path,
            helper_resource_root,
            "recover",
            action,
            "none",
            pending[0]["operation_id"],
        )
        assert selector.is_file()
        if action == "rollback":
            assert selector.read_bytes() == original
        else:
            assert (
                tomllib.loads(selector.read_text())["general"]["users_name"]
                == dict(case[1].profile_names)["profile"]
            )
        with authority.normal(("aaa.selected.config.file",)):
            assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


@pytest.mark.parametrize(
    "damage", ["retained", "unrelated", "pending", "parent", "published"]
)
def test_recovery_admission_refuses_unproven_missing_roots(
    tmp_path, monkeypatch, helper_resource_root, damage
):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _selected_crash(
            monkeypatch, "published" if damage == "published" else "move_observed"
        )
        selector = case[5]
        root = tmp_path / "bootstrap"
        authority = admission_authority(root)
        authority.register("aaa.selected.config.file", (selector,))
        unrelated = tmp_path / "unrelated.txt"
        unrelated.write_text("untouched unrelated source")
        unrelated.chmod(0o600)
        authority.register("unrelated.file", (unrelated,))
        _child(tmp_path, helper_resource_root, "start", "finish", "none", expected=91)
        pending, _ = bootstrap._records(root)
        operation = pending[0]["operation_id"]
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            before = journal._records(parent)
        prepared = next(row.evidence for row in before if row.event == "prepared")
        item = next(
            row for row in prepared["artifacts"] if row["target"] == str(selector)
        )
        if damage == "retained":
            Path(item["retained"]).write_bytes(b"changed retained content")
        elif damage == "unrelated":
            unrelated.unlink()
        elif damage == "pending":
            pointer = root / ("pending-" + bootstrap._key(operation) + ".json")
            pointer.rename(pointer.with_suffix(".saved"))
        elif damage == "parent":
            selector.parent.rename(tmp_path / "moved-live")
            selector.parent.mkdir(mode=0o700)
        else:
            assert selector.is_file()
            selector.unlink()
        with (
            pytest.raises((OSError, ValueError, RuntimeError)),
            authority._replacement_recovery(journal, 3),
        ):
            pytest.fail("Changed evidence must not grant recovery admission")
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent) == before
        assert not selector.exists()


def test_recovery_staging_keeps_source_exclusion_under_journal_lock(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _selected_crash(monkeypatch, "move_observed")
        selector = case[5]
        root = tmp_path / "bootstrap"
        authority = admission_authority(root)
        authority.register("aaa.selected.config.file", (selector,))
        _child(tmp_path, helper_resource_root, "start", "finish", "none", expected=91)
        pending, _ = bootstrap._records(root)
        journal = Journal(tmp_path / "control", pending[0]["operation_id"])
        private = tmp_path / "private-readback"
        private.mkdir(mode=0o700)
        limits = ArchiveLimits()
        with (
            authority._replacement_recovery(journal, 3) as session,
            journal._locked(exclusive=True),
        ):
            with session._capture_bound_sources(
                (), private, limits, limits.expanded_bytes
            ):
                assert not selector.exists()
            with (
                pytest.raises(
                    bootstrap.RecoveryRequired, match="capture_staging_overlaps_source"
                ),
                session._capture_bound_sources(
                    (), selector.parent, limits, limits.expanded_bytes
                ),
            ):
                pytest.fail("Recovery must still exclude original source directories")
