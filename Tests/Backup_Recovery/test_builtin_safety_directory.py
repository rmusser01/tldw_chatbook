"""Finite builtin safety sources retain metadata and explicitly selected bytes."""

import os
from pathlib import Path

import pytest

from Tests.Backup_Recovery import test_held_sqlite_rollback as _native_fixture
from tldw_chatbook.Backup_Recovery import publication
from tldw_chatbook.Backup_Recovery.journal import _Object, observe_artifact


def test_directory_safety_witness_never_reads_unselected_descendants(tmp_path):
    root = tmp_path / "assets"
    root.mkdir(mode=0o700)
    ignored = root / "not-selected"
    ignored.symlink_to(tmp_path / "absent")
    before = root.stat()
    observed = _Object.model_validate(publication._observe_safety_source(root))
    assert observed.kind == "directory" and observed.size == 0
    assert publication._safety_source_matches(observed)
    # Even an unsafe unselected physical descendant is not part of this finite
    # metadata witness; the selected files have separate full observations.
    ignored.unlink()
    ignored.write_bytes(b"unselected")
    os.utime(root, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not publication._safety_source_matches(observed)


@pytest.mark.parametrize("change", ["mode", "replacement", "symlink"])
def test_directory_safety_witness_refuses_metadata_or_identity_drift(tmp_path, change):
    root = tmp_path / "assets"
    root.mkdir(mode=0o700)
    observed = _Object.model_validate(publication._observe_safety_source(root))
    if change == "mode":
        root.chmod(0o750)
    else:
        root.rename(tmp_path / "old")
        if change == "symlink":
            root.symlink_to(tmp_path / "old", target_is_directory=True)
        else:
            root.mkdir(mode=0o700)
    assert not publication._safety_source_matches(observed)


def test_regular_safety_file_keeps_existing_full_object_witness(tmp_path):
    path = tmp_path / "selected"
    path.write_bytes(b"selected bytes")
    path.chmod(0o600)
    actual = publication._observe_safety_source(path)
    assert actual == observe_artifact(path, metadata=True)
    assert publication._safety_source_matches(_Object.model_validate(actual))
    path.write_bytes(b"different bytes")
    assert not publication._safety_source_matches(_Object.model_validate(actual))


@pytest.fixture
def builtin_case(tmp_path, monkeypatch, helper_resource_root, request):
    import hashlib
    from dataclasses import replace
    from threading import Event

    import keyring.core
    from keyring.backends.null import Keyring

    replacement_case = _native_fixture.replacement_case
    from tldw_chatbook.Backup_Recovery import crypto, replacement
    from tldw_chatbook.Backup_Recovery.inventory import _sqlite_sidecars
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        StorageItem,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Persona_Visual.recovery import _Assets, recovery_adapters

    monkeypatch.setattr(keyring.core, "_keyring_backend", Keyring())
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    owner_id = getattr(request, "param", "persona.visual_identity_builtin")
    core_key = "profile:profile:db.chachanotes.primary"

    def core_source(live):
        core = live / "core.db"
        db = CharactersRAGDB(core, "fixture")
        db.close()
        return (StorageItem("db.chachanotes.primary", core_key, core, "included", ()),)

    with replacement_case(
        tmp_path, monkeypatch, prepared=False, extras=core_source
    ) as case:
        candidate, original, _, _, _, selector = case
        archive = replacement._acquired_source(candidate, original, Event())
        root = selector.parent / (
            "persona_visual" if owner_id == "persona.assets" else "package-assets"
        )
        selected = root / "characters" / "selected.png"
        selected.parent.mkdir(parents=True, mode=0o700)
        root.chmod(0o700)
        selected.write_bytes(b"selected builtin bytes")
        selected.chmod(0o600)
        if owner_id == "persona.visual_identity_builtin":
            (root / "unselected").symlink_to(tmp_path / "absent")
        core = selector.parent / "core.db"
        db = CharactersRAGDB(core, "fixture")
        try:
            if owner_id == "persona.assets":
                from Tests.Persona_Visual.test_persona_visual_publication import (
                    _snapshot,
                )
                from tldw_chatbook.Persona_Visual.publication import (
                    publish_persona_visual,
                )
                from tldw_chatbook.Persona_Visual.repository import (
                    PersonaVisualRepository,
                )

                selected.unlink()
                selected.parent.rmdir()
                source = tmp_path / "legacy-source"
                source.mkdir(mode=0o700)
                publish_persona_visual(
                    PersonaVisualRepository(db),
                    _snapshot(source),
                    source_root=source,
                    profile_root=selector.parent,
                    authority_guard=lambda: True,
                )
                selected = (
                    selector.parent
                    / db.get_connection()
                    .execute("SELECT storage_relpath FROM persona_visual_assets")
                    .fetchone()[0]
                )
            else:
                actor = db.add_character_card({"name": "Original builtin"})
                VisualIdentityRepository(db).activate_pack(
                    pack={
                        "title": "builtin",
                        "default_expression_key": "neutral",
                        "source_kind": "builtin",
                    },
                    manifest={},
                    assets=[
                        {
                            "expression_key": "neutral",
                            "original_expression_key": "neutral",
                            "source_filename": selected.name,
                            "storage_relpath": selected.relative_to(root).as_posix(),
                            "content_type": "image/png",
                            "bytes": selected.stat().st_size,
                            "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
                            "width": 1,
                            "height": 1,
                        }
                    ],
                    actor_kind="character",
                    actor_id=actor,
                )
        finally:
            db.close()
        # Relocate only the fixture's installed source root. Discovery, finite
        # reference selection, native core/asset validation and capture are real.
        monkeypatch.setattr(_Assets, "_root", lambda self, config: root)
        monkeypatch.setattr(_Assets, "_definition_path", lambda self, config: root)
        config = {
            "database": {"chachanotes_db_path": str(core)},
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "profile"),
        }
        owner = next(a for a in recovery_adapters() if a.owner_id == owner_id)
        members = owner.discover(config)
        assert all(i.status in {"included", "included_directory"} for i in members)
        files = {i.path for i in members if i.status == "included"}
        assert selected in files
        assert len(files) == (1 if owner_id == "persona.visual_identity_builtin" else 3)
        root_key = next(i.logical_id for i in members if i.path == root)
        transients = _sqlite_sidecars(
            core_adapters()[0].discover(config), core_adapters()
        )
        # Actual DB setup may load/save the selected ordinary config. Restore the
        # existing fixture's deliberate damaged input before reviewing targets.
        selector.write_bytes(b'api_key="test-only-original-secret"\nbroken = [')
        target = replace(
            original.target,
            items=tuple(
                replace(i, dependencies=(root_key,)) if i.logical_id == core_key else i
                for i in original.target.items
            )
            + members
            + transients,
        )

        def plan_for(keys, issues=("credential_format_unreadable",)):
            return plan_restore(
                archive,
                mode="replace",
                destinations=dict((*original.destinations, *original.selectors)),
                target=target,
                profile_names=dict(original.profile_names),
                safety_scope=tuple(keys),
                acknowledged_credential_issues=issues,
            )

        yield archive, plan_for, members, selected


@pytest.mark.parametrize(
    "builtin_case", ["persona.visual_identity_builtin", "persona.assets"], indirect=True
)
@pytest.mark.parametrize("corrupt_readback", [False, True])
def test_actual_builtin_finite_safety_copy_is_authenticated_without_unselected_tree(
    builtin_case, tmp_path, monkeypatch, corrupt_readback
):
    import zipfile
    from threading import Event

    from tldw_chatbook.Backup_Recovery import archive_reader, bootstrap, replacement
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive, plan_for, members, selected = builtin_case
    if corrupt_readback:
        native_copy = replacement._copy_verified_payload

        def changed(archive, payload, destination, cancel):
            native_copy(archive, payload, destination, cancel)
            if payload.owner_id == members[0].owner:
                destination.write_bytes(b"changed authenticated private candidate")

        monkeypatch.setattr(replacement, "_copy_verified_payload", changed)
    plan = plan_for(i.logical_id for i in members)
    candidate = stage_restore(archive, plan, tmp_path / "builtin-stage", Event())
    before = selected.read_bytes(), selected.stat().st_ino
    try:
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"test-only",
            cancel=Event(),
        )
    except replacement.RollbackCredentialReviewRequired as review:
        pending = bootstrap._records(tmp_path / "bootstrap")[0]
        assert (selected.read_bytes(), selected.stat().st_ino) == before
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
        assert (selected.read_bytes(), selected.stat().st_ino) == before
        plan = plan_for((i.logical_id for i in members), issues=review.issues)
        candidate = stage_restore(
            archive, plan, tmp_path / "reviewed-builtin-stage", Event()
        )
        if corrupt_readback:
            with pytest.raises(ValueError, match="asset_digest_mismatch"):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=tmp_path / "control",
                    rollback_password=b"test-only",
                    cancel=Event(),
                )
            assert (selected.read_bytes(), selected.stat().st_ino) == before
            pending = bootstrap._records(tmp_path / "bootstrap")[0]
            journal = Journal(tmp_path / "control", pending[0]["operation_id"])
            with journal._locked(exclusive=False) as parent:
                assert "publication_started" not in {
                    row.event for row in journal._records(parent)
                }
            return
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"test-only",
            cancel=Event(),
        )
    assert (selected.read_bytes(), selected.stat().st_ino) == before
    journal = Journal(tmp_path / "control", operation)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    proof = next(row.evidence for row in rows if row.event == "rollback_verified")
    sealed = archive_reader.acquire(
        Path(proof["ciphertext"]["path"]),
        tmp_path / "readback",
        ArchiveLimits(),
        b"test-only",
        Event(),
    )
    doc = archive_reader.verify_sealed(sealed)
    owner_files = [row for row in doc.files if row.owner_id == members[0].owner]
    expected = {
        i.logical_id: i.path.read_bytes() for i in members if i.status == "included"
    }
    assert {row.logical_id for row in owner_files} == expected.keys()
    with zipfile.ZipFile(sealed.path) as packed:
        assert all(
            packed.read(row.payload) == expected[row.logical_id] for row in owner_files
        )
    assert {row["logical_id"] for row in proof["safety_sources"]} == {
        i.logical_id for i in members
    }
    assert {row.logical_id for row in doc.directories if not row.synthetic} >= {
        i.logical_id for i in members if i.status == "included_directory"
    }


@pytest.mark.parametrize(
    "builtin_case", ["persona.visual_identity_builtin", "persona.assets"], indirect=True
)
@pytest.mark.parametrize("omitted", ["root", "ancestor", "file", "foreign_owner"])
def test_builtin_safety_requires_explicit_finite_member_closure(
    builtin_case, tmp_path, omitted
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery import bootstrap, replacement
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive, plan_for, members, _selected = builtin_case
    missing = next(
        i
        for i in members
        if (
            i.metadata.parent_id is None
            if omitted == "root"
            else i.status == "included"
            if omitted == "file"
            else i.status == "included_directory" and i.metadata.parent_id is not None
        )
    )
    if omitted == "foreign_owner":
        from dataclasses import replace

        plan = plan_for(i.logical_id for i in members)
        from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

        target = replace(
            plan.target,
            items=tuple(
                replace(item, owner="persona.visual_identity")
                if item in members
                else item
                for item in plan.target.items
            ),
        )
        plan = plan_restore(
            archive,
            mode="replace",
            target=target,
            destinations=dict((*plan.destinations, *plan.selectors)),
            profile_names=dict(plan.profile_names),
            safety_scope=plan.safety_scope,
            acknowledged_credential_issues=plan.acknowledged_credential_issues,
        )
    else:
        plan = plan_for(i.logical_id for i in members if i is not missing)
    candidate = stage_restore(archive, plan, tmp_path / "builtin-stage", Event())
    before = {
        i.path: i.path.read_bytes() for i in plan.target.items if i.status == "included"
    }
    expected_error = (
        "safety_scope_capability_unavailable"
        if omitted == "foreign_owner"
        else "safety_scope_incomplete"
    )
    with pytest.raises(ValueError, match=expected_error):
        replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"test-only",
            cancel=Event(),
        )
    pending = bootstrap._records(tmp_path / "bootstrap")[0]
    journal = Journal(tmp_path / "control", pending[0]["operation_id"])
    with journal._locked(exclusive=False) as parent:
        assert "publication_started" not in {r.event for r in journal._records(parent)}
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize(
    "builtin_case", ["persona.visual_identity_builtin", "persona.assets"], indirect=True
)
def test_builtin_semantic_validation_checks_actual_core_references(builtin_case):
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.replacement import _validate_builtin_safety

    _archive, _plan_for, members, selected = builtin_case
    plan = _plan_for(item.logical_id for item in members)
    candidates = {i.logical_id: i.path for i in members if i.status == "included"}
    candidates["profile:profile:db.chachanotes.primary"] = next(
        item.path
        for item in plan.target.items
        if item.logical_id == "profile:profile:db.chachanotes.primary"
    )
    owners = {a.owner_id: a for a in install_adapters()}
    _validate_builtin_safety(members, candidates, owners, plan)
    selected.write_bytes(b"different selected bytes")
    with pytest.raises(ValueError, match="asset_digest_mismatch"):
        _validate_builtin_safety(members, candidates, owners, plan)
