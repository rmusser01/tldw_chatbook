"""Authenticated rollback can restore an absence without publishing any file."""

from copy import deepcopy
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_retained_config import (  # noqa: F401
    retained_case,
    retained_plan,
)
from tldw_chatbook.Backup_Recovery import journal


@pytest.fixture
def retirement_evidence(tmp_path):
    """Build valid prior records around a real file moved to retained storage."""
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    descriptor = stage / "candidate.json"
    descriptor.write_bytes(b"{}")
    descriptor.chmod(0o600)
    target = tmp_path / "created-state.toml"
    target.write_bytes(b'[state]\nselected="later edit"\n')
    target.chmod(0o600)
    retained = tmp_path / "retained-state.toml"
    artifact = {
        "logical_id": "created-state",
        "action": "retire",
        "candidate": None,
        "target": str(target),
        "previous": journal.observe_artifact(target),
        "previous_metadata": journal.observe_artifact(target, metadata=True),
        "retained": str(retained),
    }
    receipt = {
        "stage": journal.observe_artifact(stage),
        "descriptor": journal.observe_artifact(descriptor),
        "archive_digest": "a" * 64,
        "manifest_digest": "b" * 64,
        "plan_digest": "c" * 64,
    }
    prepared = {
        "generation": "retirement",
        "mode": "replace",
        "artifacts": [artifact],
        "installed_paths": [],
        "publication": {
            "bootstrap_root": str(tmp_path / "bootstrap"),
            "namespaces": ["local"],
            "selectors": [str(tmp_path / "config.toml")],
            "archive_digest": receipt["archive_digest"],
            "plan_digest": receipt["plan_digest"],
            "descriptor": receipt["descriptor"],
        },
    }
    rows = []

    def record(event, evidence):
        rows.append(
            journal._Event(
                operation_id="retire-only",
                sequence=len(rows),
                previous="0" * 64,
                event=event,
                evidence=journal._validate(event, evidence, rows),
            )
        )

    record("candidate_staged", receipt)
    record("prepared", prepared)
    record(
        "rollback_verified",
        {
            "ciphertext": journal.observe_artifact(descriptor),
            "sealed_digest": "d" * 64,
            "manifest_digest": "e" * 64,
            "coverage": {"created-state": "saved-state"},
        },
    )
    record("publication_started", {})
    target.rename(retained)
    record(
        "artifact_retired",
        {
            "logical_id": "created-state",
            "observed": journal.observe_artifact(retained, metadata=True),
        },
    )
    proof = {
        "plan_digest": receipt["plan_digest"],
        "manifest_digest": receipt["manifest_digest"],
        "descriptor_digest": receipt["descriptor"]["sha256"],
        "artifacts": [],
    }
    return rows, proof


def test_completed_retirement_accepts_empty_installed_evidence(retirement_evidence):
    rows, proof = retirement_evidence
    assert journal._validate("installed_validated", proof, rows) == proof


@pytest.mark.parametrize(
    "damage",
    [
        "no_artifacts",
        "isolated",
        "publish",
        "missing_previous",
        "missing_metadata",
        "missing_progress",
        "missing_coverage",
        "extra_coverage",
        "wrong_digest",
        "missing_publication",
    ],
)
def test_empty_installed_evidence_requires_complete_retirement_proof(
    retirement_evidence, damage
):
    rows, proof = deepcopy(retirement_evidence)
    prepared = next(row.evidence for row in rows if row.event == "prepared")
    rollback = next(row.evidence for row in rows if row.event == "rollback_verified")
    if damage == "no_artifacts":
        prepared["artifacts"] = []
    elif damage == "isolated":
        prepared["mode"] = "isolated"
    elif damage == "publish":
        prepared["artifacts"][0]["action"] = "publish"
        prepared["artifacts"][0]["candidate"] = next(
            row.evidence["descriptor"]
            for row in rows
            if row.event == "candidate_staged"
        )
    elif damage == "missing_previous":
        prepared["artifacts"][0]["previous"] = None
    elif damage == "missing_metadata":
        prepared["artifacts"][0]["previous_metadata"] = None
    elif damage == "missing_progress":
        rows = [row for row in rows if row.event != "artifact_retired"]
    elif damage == "missing_coverage":
        rollback["coverage"] = {}
    elif damage == "extra_coverage":
        rollback["coverage"]["unrelated"] = "unrelated-source"
    elif damage == "wrong_digest":
        proof["plan_digest"] = "f" * 64
    elif damage == "missing_publication":
        prepared["publication"] = None
    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        journal._validate("installed_validated", proof, rows)


def test_native_later_rollback_retires_first_restoration_and_preserves_current_bytes(
    retained_case,  # noqa: F811
    tmp_path,
    monkeypatch,
    helper_resource_root,
):
    from keyring.backends.null import Keyring

    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        credentials,
        crypto,
        later_rollback,
        recovery_copies,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    store = KeyringServerCredentialStore(keyring_backend=Keyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    archive, target, selector, state, authority = retained_case
    # Created-file absence proofs consult the real selected profile's native
    # generation witness, unlike replacement of an already-present state file.
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    owners = {owner.owner_id: owner for owner in recovery_adapters()}
    context = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            selector, target.items[0].logical_id.split(":")[1]
        )
    }
    unselected = selector.parent / "recent_emojis.json"
    unselected.write_bytes(b'["retained"]')
    unselected.chmod(0o600)
    state.unlink()

    def observed():
        return replace(
            target,
            items=tuple(
                owners[owner].discover(context)[0]
                for owner in ("config", "ui.state", "ui.emoji_recents")
            ),
        )

    target = observed()
    assert (
        next(row for row in target.items if row.owner == "ui.state").status == "unused"
    )
    plan = retained_plan((archive, target, selector, state, authority))
    before = {
        path: (path.read_bytes(), path.stat().st_ino) for path in (selector, unselected)
    }
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "first-stage", Event(), session=session
        )
    control = tmp_path / "control"
    original_operation = replacement.replace(
        plan,
        candidate,
        control_root=control,
        rollback_password=b"before-first-group",
        cancel=Event(),
    )
    assert state.read_bytes() == b'[state]\nselected="incoming"\n'
    edited = b'[state]\nselected="after first restoration"\n'
    state.write_bytes(edited)
    current = observed()
    rollback = later_rollback.preview_rollback(
        original_operation,
        control_root=control,
        old_password=b"before-first-group",
        target=current,
        cancel=Event(),
    )
    assert rollback.restore == ()
    assert rollback.retire == (
        (
            next(row.logical_id for row in current.items if row.owner == "ui.state"),
            state,
        ),
    )
    removed_operation = later_rollback.execute_rollback(
        original_operation,
        control_root=control,
        old_password=b"before-first-group",
        new_password=b"before-removing-group",
        cancel=Event(),
        approved_plan=rollback,
    )
    assert not state.exists()
    assert all(
        (path.read_bytes(), path.stat().st_ino) == value
        for path, value in before.items()
    )
    removed_journal = journal.Journal(control, removed_operation)
    with removed_journal._locked(exclusive=False) as parent:
        rows = removed_journal._records(parent)
    assert rows[-1].event == "committed"
    prepared = next(row.evidence for row in rows if row.event == "prepared")
    assert prepared["installed_paths"] == []
    assert len(prepared["artifacts"]) == 1
    assert prepared["artifacts"][0]["action"] == "retire"
    assert (
        next(row.evidence for row in rows if row.event == "installed_validated")[
            "artifacts"
        ]
        == []
    )
    saved = next(
        row
        for row in recovery_copies.list_recovery_copies(control)
        if row.operation_id == removed_operation
    )
    safety_copy = archive_reader.acquire(
        saved.path,
        tmp_path / "new-copy-check",
        ArchiveLimits(),
        b"before-removing-group",
        Event(),
    )
    document = archive_reader.verify_sealed(safety_copy)
    assert any(row.owner_id == "ui.state" for row in document.files)
    redo = later_rollback.preview_rollback(
        removed_operation,
        control_root=control,
        old_password=b"before-removing-group",
        target=observed(),
        cancel=Event(),
    )
    later_rollback.execute_rollback(
        removed_operation,
        control_root=control,
        old_password=b"before-removing-group",
        new_password=b"before-returning-group",
        cancel=Event(),
        approved_plan=redo,
    )
    assert state.read_bytes() == edited
    assert all(
        (path.read_bytes(), path.stat().st_ino) == value
        for path, value in before.items()
    )
