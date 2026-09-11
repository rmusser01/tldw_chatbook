"""Actual credential remapping survives interruption with exact file namespaces."""

import json
from pathlib import Path

import pytest

from Tests.Backup_Recovery import test_rollback_credentials as fixture
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.journal import Journal

_CRASH = r"""
step,point=boundary.split(':')
if point=='observed':
 original=publication._complete_move
 def move(journal,parent,prepared,intent,**kwargs):
  original(journal,parent,prepared,intent,**kwargs)
  if intent.step==step:os._exit(91)
 publication._complete_move=move
else:
 original=publication._reverse_native_move
 def move(intent):
  if intent.step==step and point=='before':os._exit(91)
  original(intent)
  if intent.step==step:os._exit(91)
 publication._reverse_native_move=move
"""


def _stop_originals(monkeypatch, tmp_path, operation, event):
    append = Journal._append

    def stop(self, parent, name, evidence):
        append(self, parent, name, evidence)
        if name == event:
            raise KeyboardInterrupt("actual rollback boundary")

    with monkeypatch.context() as patch:
        patch.setattr(Journal, "_append", stop)
        with pytest.raises(KeyboardInterrupt):
            fixture._recover(tmp_path, operation)


@pytest.mark.parametrize(
    ("boundary", "damage"),
    [
        ("credential_retire:native", None),
        ("credential_retire:observed", None),
        ("credential_publish:before", None),
        ("credential_unpublish:native", None),
        ("credential_unpublish:observed", None),
        ("credential_unpublish:observed", "held"),
        ("credential_unpublish:observed", "parent"),
    ],
)
def test_exact_credential_file_namespace_recovers_actual_move(
    tmp_path, monkeypatch, helper_resource_root, boundary, damage
):
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    with fixture._interrupted(
        tmp_path, monkeypatch, helper_resource_root, second=True
    ) as (case, store, backend, targets, original, operation):
        authority = admission_authority(tmp_path / "bootstrap")
        authority.register("aaa.credential.targets.file", (targets,))
        alias = tmp_path / "targets-alias"
        alias.symlink_to(targets)
        authority.register("aaa.credential.targets.alias", (alias,))
        if boundary.startswith("credential_retire"):
            _stop_originals(monkeypatch, tmp_path, operation, "originals_validated")
        store.set_secret("peer", "api_key", "first-foreign")
        if boundary.startswith("credential_unpublish"):
            _stop_originals(
                monkeypatch, tmp_path, operation, "rollback_activation_recorded"
            )
            store.set_secret("second", "api_key", "second-foreign")
        saved = tmp_path / "backend.json"
        saved.write_text(
            json.dumps([[*key, value] for key, value in backend.values.items()])
        )
        saved.chmod(0o600)
        program = fixture._CHILD.replace(
            "if boundary.startswith('credential_'):", "if False:"
        ).replace(
            "result=replacement.recover_replacement",
            _CRASH + "\nresult=replacement.recover_replacement",
        )
        with monkeypatch.context() as patch:
            patch.setattr(fixture, "_CHILD", program)
            fixture._child(tmp_path, helper_resource_root, operation, boundary, 91)
        assert not targets.exists()
        if damage:
            before = fixture._records(tmp_path, operation)
            intent = next(
                row.evidence for row in reversed(before) if row.event == "move_intended"
            )
            held = Path(intent["destination"])
            if damage == "held":
                held.write_bytes(b"changed remapped credentials")
            else:
                moved_parent = held.parent.with_name(held.parent.name + "-moved")
                held.parent.rename(moved_parent)
                held.parent.mkdir(mode=0o700)
                (moved_parent / held.name).rename(held)
            journal = Journal(tmp_path / "control", operation)
            with (
                pytest.raises((OSError, ValueError, RuntimeError)),
                authority._replacement_recovery(journal, 3),
            ):
                pytest.fail("Changed credential artifact must not grant admission")
            assert fixture._records(tmp_path, operation) == before
            assert not targets.exists()
            return
        fixture._child(tmp_path, helper_resource_root, operation, "", 0)
        active = json.loads(targets.read_bytes())["targets"]
        actual = KeyringServerCredentialStore(
            keyring_backend=fixture._PersistentKeyring(saved)
        )
        assert [
            actual.get_secret(
                row["server_id"], row["auth_reference"].removeprefix("keyring:")
            )
            for row in active
        ] == ["current-shared-secret", "second-captured-secret"]
        assert actual.get_secret("peer", "api_key") == "first-foreign"
        if boundary.startswith("credential_unpublish"):
            assert actual.get_secret("second", "api_key") == "second-foreign"
        rows = fixture._records(tmp_path, operation)
        prepared = next(row.evidence for row in rows if row.event == "prepared")
        item = next(
            row for row in prepared["artifacts"] if row["target"] == str(targets)
        )
        assert Path(item["retained"]).read_bytes() == original
        assert rows[-1].event == "rolled_back"
        with authority.normal(("aaa.credential.targets.file",)):
            assert bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
