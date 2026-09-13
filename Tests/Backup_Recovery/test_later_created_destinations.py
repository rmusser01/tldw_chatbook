"""Created inactive destinations remain recoverable during later rollback."""

import hashlib
import json
import shutil
import zipfile
from threading import Event

import pytest

from Tests.Backup_Recovery.test_builtin_later_snapshot import (
    _current,
)
from Tests.Backup_Recovery.test_builtin_later_snapshot import (
    complete_builtin_case as _complete_builtin_case,
)
from tldw_chatbook.Backup_Recovery import archive_reader, bootstrap, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore

complete_builtin_case = _complete_builtin_case


@pytest.fixture
def created_destinations(complete_builtin_case, tmp_path, request):
    archive, plan_for, members, selected = complete_builtin_case
    if getattr(request, "param", None) == "valid_original":
        # Only the two-cycle case needs an ordinary readable original profile.
        # Seed it before any plan or safety capture; other cases retain the
        # inherited damaged-input fixture and its credential omission coverage.
        (tmp_path / "live/config.toml").write_text(
            'api_key="test-only-original-secret"\n[database]\n'
            "chachanotes_db_path=" + json.dumps(str(tmp_path / "live/core.db")) + "\n"
            "research_db_path=" + json.dumps(str(tmp_path / "live/research.db")) + "\n"
        )
    original = plan_for(item.logical_id for item in members)
    document = json.loads(archive.manifest_bytes)
    with zipfile.ZipFile(archive.path) as packed:
        payloads = {
            row["payload"]: packed.read(row["payload"]) for row in document["files"]
        }
    created = {}
    manual_parent = tmp_path / "selected-manual-parent"
    manual_parent.mkdir(mode=0o700)
    for owner, root_key, relative, data in (
        (
            "persona.visual_identity_builtin",
            "profile:profile:persona.visual_identity_builtin",
            "kept.png",
            b"inactive artwork",
        ),
        ("eval.definitions", "eval-container", "eval_config.yaml", b"tasks: {}\n"),
    ):
        file_key = "profile:profile:" + owner + ":incoming"
        document["owners"].append(
            {"owner_id": owner, "schema_version": 1, "capabilities": []}
        )
        document["directories"].append(
            {
                "logical_id": root_key,
                "root_id": root_key,
                "parent_id": None,
                "relative_path": "",
                "synthetic": owner == "eval.definitions",
                "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
            }
        )
        payload = "payload/" + owner
        payloads[payload] = data
        document["files"].append(
            {
                "logical_id": file_key,
                "root_id": root_key,
                "parent_id": root_key,
                "relative_path": relative,
                "owner_id": owner,
                "payload": payload,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "metadata": {"version": 1, "mode": 0o600, "mtime_ns": 0},
            }
        )
        dependencies = [file_key, "profile:profile:config"]
        if owner == "persona.visual_identity_builtin":
            dependencies.append("profile:profile:db.chachanotes.primary")
        document["producer_inventory"].extend(
            (
                {
                    "logical_id": root_key,
                    "owner_id": owner,
                    "status": "included_directory",
                    "dependencies": dependencies,
                    "shared_group": None,
                },
                {
                    "logical_id": file_key,
                    "owner_id": owner,
                    "status": "included",
                    "dependencies": [root_key, "profile:profile:config"],
                    "shared_group": None,
                },
            )
        )
        document["dependency_groups"][0]["members"].extend((root_key, file_key))
        created[root_key] = manual_parent / (
            "inactive-artwork" if owner.startswith("persona") else "inactive-eval"
        )
    source = tmp_path / "created-input.zip"
    with zipfile.ZipFile(source, "w") as packed:
        packed.writestr("manifest.json", json.dumps(document))
        for key, data in payloads.items():
            packed.writestr(key, data)
    acquired = archive_reader.acquire(
        source, tmp_path / "created-input", ArchiveLimits(), None, Event()
    )
    destinations = {
        **dict(original.destinations),
        **dict(original.selectors),
        **created,
    }
    issues = original.acknowledged_credential_issues
    for attempt in range(2):
        plan = plan_restore(
            acquired,
            mode="replace",
            destinations=destinations,
            target=original.target,
            profile_names=dict(original.profile_names),
            safety_scope=original.safety_scope,
            acknowledged_credential_issues=issues,
        )
        candidate = stage_restore(
            acquired, plan, tmp_path / f"created-stage-{attempt}", Event()
        )
        try:
            operation = replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"old",
                cancel=Event(),
            )
            break
        except replacement.RollbackCredentialReviewRequired as review:
            issues = review.issues
            pending = bootstrap._records(tmp_path / "bootstrap")[0]
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
    else:
        pytest.fail("credential review failed to settle")
    shutil.rmtree(candidate)
    assert not candidate.exists()
    source.unlink()
    journal = Journal(tmp_path / "control", operation)
    assert (journal.root / "verified-manifest.json").is_file()
    assert selected.read_bytes() == b"selected builtin bytes"
    yield operation, created, selected


def test_later_preview_classifies_actual_created_inactive_roots(
    created_destinations, tmp_path
):
    operation, created, selected = created_destinations
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"old",
        target=_current(tmp_path),
        cancel=Event(),
    )
    assert set(created.values()) <= set(dict(reviewed.retire).values())
    assert selected not in dict(reviewed.retire).values()
    assert selected.read_bytes() == b"selected builtin bytes"


def _execute_reviewed(operation, old_password, new_password, tmp_path):
    """Use the actual omission review and abort APIs for this native fixture."""
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    issues = ()
    for _ in range(2):
        reviewed = preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=old_password,
            target=_current(tmp_path),
            cancel=Event(),
            acknowledged_credential_issues=issues,
        )
        try:
            result = execute_rollback(
                operation,
                control_root=tmp_path / "control",
                old_password=old_password,
                new_password=new_password,
                cancel=Event(),
                approved_plan=reviewed,
            )
        except replacement.RollbackCredentialReviewRequired as review:
            issues = review.issues
            pending = bootstrap._records(tmp_path / "bootstrap")[0]
            assert len(pending) == 1
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
        else:
            return result, reviewed
    pytest.fail("actual credential omission review did not settle")


@pytest.mark.parametrize("created_destinations", ["valid_original"], indirect=True)
def test_later_execution_saves_created_edits_before_retirement_and_recovers_them(
    created_destinations, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.recovery_copies import (
        delete_recovery_copy,
        list_recovery_copies,
    )

    operation, created, selected = created_destinations
    artwork = created["profile:profile:persona.visual_identity_builtin"] / "kept.png"
    evaluation = created["eval-container"] / "eval_config.yaml"
    edited = {
        artwork: b"post-replacement inactive artwork",
        evaluation: b"tasks: {post_restore: {}}\n",
    }
    for path, data in edited.items():
        path.write_bytes(data)
    preserved = selected.read_bytes(), selected.stat().st_ino
    capture = replacement.capture_verify_rollback
    observed = []

    def verified_before_publication(*args, **kwargs):
        encrypted = capture(*args, **kwargs)
        assert encrypted.read_bytes().startswith(b"age-encryption.org/v1")
        assert all(path.read_bytes() == data for path, data in edited.items())
        journal = args[2]
        with journal._locked(exclusive=False) as parent:
            events = [row.event for row in journal._records(parent)]
        assert events[-1] == "rollback_verified"
        assert "publication_started" not in events
        observed.append(encrypted)
        return encrypted

    with monkeypatch.context() as patch:
        patch.setattr(
            replacement, "capture_verify_rollback", verified_before_publication
        )
        result, reviewed = _execute_reviewed(operation, b"old", b"new", tmp_path)
    assert len(observed) == 1 and result != operation
    assert not any(path.exists() for path in created.values())
    assert (selected.read_bytes(), selected.stat().st_ino) == preserved
    saved = next(
        row
        for row in list_recovery_copies(tmp_path / "control")
        if row.operation_id == result
    )
    acquired = archive_reader.acquire(
        saved.path, tmp_path / "new-safety-readback", ArchiveLimits(), b"new", Event()
    )
    document = archive_reader.verify_sealed(acquired)
    logical = {item.path: item.logical_id for item in reviewed.target.items}
    assert logical[artwork.parent] != "profile:profile:persona.visual_identity_builtin"
    with zipfile.ZipFile(acquired.path) as packed:
        for path, data in edited.items():
            payload = next(
                row for row in document.files if row.logical_id == logical[path]
            )
            assert packed.read(payload.payload) == data
            assert payload.sha256 == hashlib.sha256(data).hexdigest()

    # The newer copy must not require the older payload ciphertext to survive.
    original_copy = next(
        row
        for row in list_recovery_copies(tmp_path / "control")
        if row.operation_id == operation
    )
    delete_recovery_copy(tmp_path / "control", operation, user_selected=True)
    assert not original_copy.path.exists()
    assert (
        Journal(tmp_path / "control", operation).root / "verified-manifest.json"
    ).is_file()
    second, _ = _execute_reviewed(result, b"new", b"third", tmp_path)
    assert second not in {operation, result}
    assert all(path.read_bytes() == data for path, data in edited.items())
    assert (selected.read_bytes(), selected.stat().st_ino) == preserved


@pytest.mark.parametrize(
    "change", ["unknown", "missing", "alias", "replaced_root", "manifest", "pair"]
)
def test_created_root_requires_exact_members_and_current_local_history(
    created_destinations, tmp_path, change
):
    operation, created, _ = created_destinations
    root = created["profile:profile:persona.visual_identity_builtin"]
    if change == "unknown":
        (root / "unknown.txt").write_bytes(b"not declared by this publication")
    elif change == "missing":
        (root / "kept.png").unlink()
    elif change == "alias":
        (root / "kept.png").rename(tmp_path / "outside.png")
        (root / "kept.png").symlink_to(tmp_path / "outside.png")
    elif change == "replaced_root":
        root.rename(tmp_path / "old-root")
        shutil.copytree(tmp_path / "old-root", root)
    elif change == "manifest":
        (
            Journal(tmp_path / "control", operation).root / "verified-manifest.json"
        ).write_bytes(b"{}")
    else:
        next((tmp_path / "bootstrap").glob("activation-*.json")).write_bytes(b"{}")
    with pytest.raises(
        (ValueError, RuntimeError),
        match=(
            "^record_version$"
            if change == "pair"
            else "local_snapshot_created|verified_manifest|activation|projection_generation|bootstrap"
        ),
    ):
        preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"old",
            target=_current(tmp_path),
            cancel=Event(),
        )
    assert root.exists()
