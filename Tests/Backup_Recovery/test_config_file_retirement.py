"""Receipt-bound config file absences survive imported/local profile ID changes."""

import hashlib
import json
import zipfile
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case


@pytest.fixture(params=["ui.state", "ui.emoji_recents", "runtime.source_state"])
def completed_config_file(tmp_path, monkeypatch, helper_resource_root, request):
    owner_id = request.param
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        credentials,
        crypto,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.journal import Journal, _Prepared
    from tldw_chatbook.Backup_Recovery.later_rollback import (
        _created_manifest,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    from tldw_chatbook.runtime_policy.recovery import (
        recovery_adapters as runtime_adapters,
    )

    owner = next(
        x for x in (*recovery_adapters(), *runtime_adapters()) if x.owner_id == owner_id
    )

    def extras(live):
        path = owner.discover(
            {DISCOVERY_CONTEXT_KEY: DiscoveryContext(live / "config.toml", "profile")}
        )[0].path
        path.write_text(
            '[sidebar]\nsearch_query="incoming"\n' if owner_id == "ui.state" else "{}"
        )
        path.chmod(0o600)
        return owner.discover(
            {DISCOVERY_CONTEXT_KEY: DiscoveryContext(live / "config.toml", "profile")}
        )

    with replacement_case(tmp_path, monkeypatch, prepared=False, extras=extras) as case:
        _, original, _, _, _, selector = case
        local_id = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
        imported_id = hashlib.sha256(
            str(tmp_path / "imported" / "config.toml").encode()
        ).hexdigest()[:24]
        path = owner.discover(
            {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, local_id)}
        )[0].path
        path.unlink()
        context = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, local_id)}
        absent = owner.discover(context)[0]
        assert absent.status == "unused"
        target = replace(
            original.target,
            items=tuple(
                replace(
                    x,
                    logical_id=x.logical_id.replace(
                        "profile:profile:", f"profile:{local_id}:"
                    ),
                    dependencies=tuple(
                        k.replace("profile:profile:", f"profile:{local_id}:")
                        for k in x.dependencies
                    ),
                )
                for x in original.target.items
                if x.owner != owner_id
            )
            + (absent,),
        )
        archive_path = tmp_path / "incoming-ids.zip"
        with (
            zipfile.ZipFile(tmp_path / "replacement.zip") as old,
            zipfile.ZipFile(archive_path, "w") as new,
        ):
            for name in old.namelist():
                payload = old.read(name)
                if name == "manifest.json":
                    document = json.loads(
                        payload.replace(
                            b"profile:profile:", f"profile:{imported_id}:".encode()
                        )
                    )
                    document["profile_ids"] = [imported_id]
                    payload = json.dumps(document).encode()
                new.writestr(name, payload)
        archive = archive_reader.acquire(
            archive_path,
            tmp_path / "cross-profile-input",
            ArchiveLimits(),
            None,
            Event(),
        )
        dest = {
            key.replace("profile:profile:", f"profile:{imported_id}:"): value
            for key, value in (*original.destinations, *original.selectors)
        }
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dest,
            target=target,
            profile_names={imported_id: "Local"},
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        candidate = stage_restore(archive, plan, tmp_path / "cross-candidate", Event())
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "cross-control",
            rollback_password=b"original",
            cancel=Event(),
        )
        journal = Journal(tmp_path / "cross-control", operation)
        saved = load_plan(journal)
        with journal._locked(exclusive=False) as fd:
            rows = journal._records(fd)
        prepared = _Prepared.model_validate(
            next(r.evidence for r in rows if r.event == "prepared")
        )
        doc, _digest = _created_manifest(journal, saved, prepared, rows)
        artifact = next(x for x in prepared.artifacts if x.target == str(path))
        incoming = next(
            x for x in doc.producer_inventory if x.logical_id == artifact.logical_id
        )
        assert imported_id != local_id
        assert incoming.owner_id == owner_id
        assert incoming.dependencies == (f"profile:{imported_id}:config",)
        assert artifact.previous is None and artifact.candidate.kind == "file"
        edited = (
            b'[sidebar]\nsearch_query="post-restore edit"\n'
            if owner_id == "ui.state"
            else b'{"probe":"post-restore edit"}'
        )
        path.write_bytes(edited)
        path.chmod(0o600)
        current = owner.discover(context)[0]
        observed = replace(
            target,
            items=tuple(
                current if x.logical_id == absent.logical_id else x
                for x in target.items
            ),
        )
        yield {
            "owner": owner,
            "context": context,
            "path": path,
            "selector": selector,
            "operation": operation,
            "journal": journal,
            "original": saved,
            "prepared": prepared,
            "rows": rows,
            "document": doc,
            "target": observed,
            "absent": absent,
            "current": current,
            "edited": edited,
            "control": tmp_path / "cross-control",
        }


def test_actual_config_file_retirement_preserves_edits_then_restores_absence(
    completed_config_file, tmp_path
):
    from tldw_chatbook.Backup_Recovery import archive_reader, recovery_copies
    from tldw_chatbook.Backup_Recovery.later_rollback import (
        _config_file_retirement_scopes,
        preview_rollback,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    case = completed_config_file
    approved = preview_rollback(
        case["operation"],
        control_root=case["control"],
        old_password=b"original",
        target=case["target"],
        cancel=Event(),
    )
    assert (case["current"].logical_id, case["path"]) in approved.retire
    expected_scope = ((case["current"].logical_id, case["path"], case["selector"]),)
    assert _config_file_retirement_scopes(approved) == expected_scope
    assert dict(approved.profile_names) == {
        case["current"].logical_id.split(":")[1]: "Local"
    }
    result = recovery_copies.rollback(
        case["operation"],
        control_root=case["control"],
        old_password=b"original",
        new_password=b"new safety",
        cancel=Event(),
        approved_plan=approved,
    )
    assert not case["path"].exists()
    # Receipt authority survives native moves; the caller separately proves its new phase.
    assert _config_file_retirement_scopes(approved) == expected_scope
    saved = next(
        x
        for x in recovery_copies.list_recovery_copies(case["control"])
        if x.operation_id == result
    )
    archive = archive_reader.acquire(
        saved.path, tmp_path / "readback", ArchiveLimits(), b"new safety", Event()
    )
    doc = archive_reader.verify_sealed(archive)
    row = next(x for x in doc.files if x.owner_id == case["current"].owner)
    with zipfile.ZipFile(archive.path) as packed:
        assert packed.read(row.payload) == case["edited"]


@pytest.mark.parametrize("completed_config_file", ["ui.state"], indirect=True)
@pytest.mark.parametrize(
    "damage",
    [
        "original_status",
        "original_dependency",
        "current_owner",
        "current_id",
        "duplicate_current",
        "incoming_owner",
        "incoming_dependency",
        "candidate_kind",
    ],
)
def test_config_file_join_refuses_unclassified_provenance(
    completed_config_file, damage
):
    from tldw_chatbook.Backup_Recovery.later_rollback import _config_file_relations

    case = completed_config_file
    original, prepared, document, target = (
        case["original"],
        case["prepared"],
        case["document"],
        case["target"],
    )
    key = next(
        x.logical_id for x in prepared.artifacts if x.target == str(case["path"])
    )
    if damage.startswith("original_"):
        changes = (
            {"status": "included"}
            if damage == "original_status"
            else {"dependencies": ()}
        )
        original = replace(
            original,
            target=replace(
                original.target,
                items=tuple(
                    replace(x, **changes)
                    if x.logical_id == case["absent"].logical_id
                    else x
                    for x in original.target.items
                ),
            ),
        )
    elif damage in {"current_owner", "current_id"}:
        changes = (
            {"owner": "ui.emoji_recents"}
            if damage == "current_owner"
            else {"logical_id": "forged-local-id"}
        )
        target = replace(
            target,
            items=tuple(
                replace(x, **changes) if x.path == case["path"] else x
                for x in target.items
            ),
        )
    elif damage == "duplicate_current":
        target = replace(target, items=(*target.items, case["current"]))
    elif damage == "incoming_owner":
        document = document.model_copy(
            update={
                "files": tuple(
                    x.model_copy(update={"owner_id": "ui.emoji_recents"})
                    if x.logical_id == key
                    else x
                    for x in document.files
                )
            }
        )
    elif damage == "incoming_dependency":
        document = document.model_copy(
            update={
                "producer_inventory": tuple(
                    x.model_copy(update={"dependencies": ()})
                    if x.logical_id == key
                    else x
                    for x in document.producer_inventory
                )
            }
        )
    else:
        prepared = prepared.model_copy(
            update={
                "artifacts": tuple(
                    x.model_copy(
                        update={
                            "candidate": x.candidate.model_copy(
                                update={"kind": "directory"}
                            )
                        }
                    )
                    if x.logical_id == key
                    else x
                    for x in prepared.artifacts
                )
            }
        )
    with pytest.raises(ValueError, match="local_snapshot_created_scope_unverified"):
        _config_file_relations(original, prepared, document, target)
    assert case["path"].read_bytes() == case["edited"]


@pytest.mark.parametrize("completed_config_file", ["ui.state"], indirect=True)
@pytest.mark.parametrize("damage", ["retire_key", "retire_path", "receipt"])
def test_config_file_retirement_scope_requires_exact_local_receipt(
    completed_config_file, damage
):
    from tldw_chatbook.Backup_Recovery.later_rollback import (
        _config_file_retirement_scopes,
        preview_rollback,
    )

    case = completed_config_file
    plan = preview_rollback(
        case["operation"],
        control_root=case["control"],
        old_password=b"original",
        target=case["target"],
        cancel=Event(),
    )
    if damage == "retire_key":
        plan = replace(plan, retire=(("forged", case["path"]),))
    elif damage == "retire_path":
        plan = replace(
            plan,
            retire=((case["current"].logical_id, case["path"].with_name("unrelated")),),
        )
    else:
        saved = case["journal"].root / "restore-plan.json"
        saved.write_bytes(saved.read_bytes() + b" ")
    with pytest.raises(
        ValueError,
        match="local_snapshot_created_history_unverified|recovery_plan_changed",
    ):
        _config_file_retirement_scopes(plan)
    assert case["path"].read_bytes() == case["edited"]


@pytest.mark.parametrize("completed_config_file", ["ui.state"], indirect=True)
@pytest.mark.parametrize("damage", ["foreign", "activation"])
def test_created_config_file_live_witness_refuses_changed_authority(
    completed_config_file, monkeypatch, damage
):
    from copy import deepcopy

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.later_rollback import (
        _created_config_file_targets,
    )

    case = completed_config_file
    if damage == "foreign":
        admission_authority(bootstrap.default_bootstrap_root()).register(
            "foreign", (case["path"],)
        )
    else:
        records = bootstrap._control_records

        def changed(root):
            pending, profiles, associations = deepcopy(records(root))
            for row in profiles:
                if row["selector"] == str(case["selector"]):
                    row["activation"]["generation"] = "other-generation"
            return pending, profiles, associations

        monkeypatch.setattr(bootstrap, "_control_records", changed)
    with pytest.raises((ValueError, bootstrap.RecoveryRequired)):
        _created_config_file_targets(
            case["journal"],
            case["original"],
            case["prepared"],
            case["rows"],
            case["target"],
        )
    assert case["path"].read_bytes() == case["edited"]
