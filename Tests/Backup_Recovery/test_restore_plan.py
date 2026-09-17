"""Restore preparation uses explicit local targets and private candidates."""

import pytest


def test_replace_requires_independent_target_inventory(tmp_path):
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = SealedArchive(tmp_path / "archive", "digest", b"{}")
    with pytest.raises(ValueError, match="target_unverified"):
        plan_restore(archive, mode="replace", destinations={}, target=None)


import hashlib
import json
import zipfile
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem


def sealed(tmp_path, *, partial=False, unknown=False, mutate=None, data=b"durable"):
    from Tests.Backup_Recovery.test_archive_reader import manifest

    doc = manifest(data)
    doc["owners"][0]["owner_id"] = "unknown.newer" if unknown else "ui.state"
    doc["files"][0]["owner_id"] = doc["owners"][0]["owner_id"]
    doc["consistency"] = "partial" if partial else "coherent"
    if mutate:
        mutate(doc)
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(doc))
        for payload in doc["files"]:
            archive.writestr(payload["payload"], data)
    return acquire(path, tmp_path / "archive-work", ArchiveLimits(), None, Event())


def test_isolated_maps_explicit_new_root_without_loading_current_config(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    broken = tmp_path / "broken.toml"
    broken.write_text("not = [valid")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(broken))
    archive = sealed(tmp_path)
    destination = tmp_path / "new-profile"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    assert dict(plan.restore) == {"root": destination, "file": destination / "note.txt"}
    assert not destination.exists()
    assert broken.read_text() == "not = [valid"


@pytest.mark.parametrize("kind", ["relative", "existing", "symlink", "archive"])
def test_isolated_rejects_unsafe_destinations(tmp_path, kind):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "new"
    if kind == "relative":
        destination = Path("relative")
    elif kind == "existing":
        destination.mkdir()
    elif kind == "symlink":
        destination.symlink_to(archive.path.parent, target_is_directory=True)
    else:
        destination = archive.path.parent
    with pytest.raises(ValueError):
        plan_restore(
            archive, mode="isolated", destinations={"root": destination}, target=None
        )


def test_rejects_unknown_installed_owner(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, unknown=True)
    with pytest.raises(ValueError, match="unsupported_owner"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        )


def test_partial_archive_cannot_replace(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, partial=True)
    with pytest.raises(ValueError, match="partial_replacement"):
        plan_restore(
            archive,
            mode="replace",
            destinations={},
            target=Inventory((), True, "local", ()),
        )


def test_staging_rejects_changed_destination_and_preserves_live_bytes(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    destination.mkdir()
    (destination / "arrived").write_text("new live work")
    with pytest.raises(ValueError, match="target_changed"):
        stage_restore(archive, plan, tmp_path / "work", Event())
    assert (destination / "arrived").read_text() == "new live work"


def test_stage_real_bytes_without_creating_destination(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((candidate / "candidate.json").read_bytes())
    record = next(row for row in descriptor["artifacts"] if row["logical_id"] == "file")
    assert Path(record["candidate"]).read_bytes() == b"durable"
    assert record["sha256"] == hashlib.sha256(b"durable").hexdigest()
    assert not destination.exists()


def producer(doc):
    doc["producer_inventory"] = [
        {
            "logical_id": "root",
            "owner_id": "ui.state",
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        },
        {
            "logical_id": "file",
            "owner_id": "ui.state",
            "status": "included",
            "dependencies": ["root"],
            "shared_group": None,
        },
    ]


def test_replace_explicit_tree_retires_obsolete_and_preserves_outside(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, mutate=producer)
    destination = tmp_path / "live"
    destination.mkdir()
    old = destination / "obsolete"
    old.write_bytes(b"rollback needed")
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserved")
    target = Inventory(
        (
            StorageItem("ui.state", "root", destination, "included_directory", ()),
            StorageItem("ui.state", "old", old, "included", ("root",)),
            StorageItem("ui.state", "outside", outside, "included", ()),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    assert plan.retire == (("old", old),)
    assert plan.preserve == (("outside", outside),)
    assert old.read_bytes() == b"rollback needed"


def test_replace_unknown_current_child_is_not_retired(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, mutate=producer)
    destination = tmp_path / "live"
    destination.mkdir()
    unknown = destination / "unknown"
    unknown.write_text("do not delete")
    target = Inventory(
        (StorageItem("ui.state", "root", destination, "included_directory", ()),),
        True,
        "local",
        (),
    )
    with pytest.raises(ValueError, match="target_owner_unclassified"):
        plan_restore(
            archive, mode="replace", destinations={"root": destination}, target=target
        )
    assert unknown.read_text() == "do not delete"


def test_partial_complete_group_can_stage_and_incomplete_group_refuses(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, partial=True)
    assert (
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        ).mode
        == "isolated"
    )
    bad = tmp_path / "bad"
    bad.mkdir()
    archive = sealed(
        bad,
        partial=True,
        mutate=lambda doc: doc["dependency_groups"][0].update(complete=False),
    )
    with pytest.raises(ValueError, match="dependency_group_incomplete"):
        plan_restore(
            archive, mode="isolated", destinations={"root": bad / "new"}, target=None
        )


def test_cancelled_staging_never_creates_live_target(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        stage_restore(archive, plan, tmp_path / "work", cancel)
    assert not destination.exists()
    assert not list(tmp_path.glob(".chatbook-restore-*"))


def test_transitive_producer_dependency_cannot_be_omitted(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def missing(doc):
        producer(doc)
        doc["producer_inventory"][1]["dependencies"].append("missing")
        doc["producer_inventory"].append(
            {
                "logical_id": "missing",
                "owner_id": "ui.state",
                "status": "unsupported",
                "dependencies": [],
                "shared_group": None,
            }
        )
        doc["exclusions"].append({"logical_id": "missing", "reason": "unsupported"})

    archive = sealed(tmp_path, partial=True, mutate=missing)
    with pytest.raises(ValueError, match="dependency_group_incomplete"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        )


def test_config_remaps_explicit_local_selector_and_freezes_fresh_profile(tmp_path):
    import tomllib

    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def config(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["files"][0].update(owner_id="config", logical_id="profile:profile:config")
        doc["dependency_groups"][0]["members"] = ["profile:profile:config"]

    archive = sealed(
        tmp_path,
        mutate=config,
        data=b'[general]\nusers_name="original"\n[paths]\ndata_dir="/never/use/original"\n',
    )
    destination = tmp_path / "config-root"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={
            "root": destination,
            "profile:profile:paths.data_dir": tmp_path / "new-data",
        },
        target=None,
        profile_names={"profile": "chosen_recovered_name"},
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((candidate / "candidate.json").read_bytes())
    artifact = next(row for row in descriptor["artifacts"] if row["kind"] == "file")
    config_data = tomllib.loads(Path(artifact["candidate"]).read_text())
    assert config_data["paths"]["data_dir"] == str(tmp_path / "new-data")
    assert config_data["general"]["users_name"] == dict(plan.profile_names)["profile"]
    assert config_data["general"]["users_name"] != "original"


def test_shared_alias_cannot_be_split_by_forged_independent_groups(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def aliases(doc):
        producer(doc)
        doc["directories"].append(
            {
                **doc["directories"][0],
                "logical_id": "other-root",
                "root_id": "other-root",
            }
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "logical_id": "alias",
                "root_id": "other-root",
                "parent_id": "other-root",
                "payload": "payload/2",
            }
        )
        doc["producer_inventory"][1]["shared_group"] = "shared"
        doc["producer_inventory"].extend(
            [
                {
                    "logical_id": "other-root",
                    "owner_id": "ui.state",
                    "status": "included_directory",
                    "dependencies": [],
                    "shared_group": None,
                },
                {
                    "logical_id": "alias",
                    "owner_id": "ui.state",
                    "status": "included",
                    "dependencies": ["other-root"],
                    "shared_group": "shared",
                },
            ]
        )
        doc["dependency_groups"].append(
            {"group_id": "alias-group", "members": ["alias"], "complete": True}
        )

    archive = sealed(tmp_path, mutate=aliases)
    with pytest.raises(ValueError, match="shared_scope_expansion_required"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        )


def test_empty_root_stages_supported_directory_metadata(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def empty(doc):
        doc["files"] = []
        doc["dependency_groups"] = []
        doc["directories"][0]["metadata"].update(mtime_ns=1_234_000_000, mode=448)
        producer(doc)
        doc["producer_inventory"] = doc["producer_inventory"][:1]

    archive = sealed(tmp_path, mutate=empty)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    stage = stage_restore(archive, plan, tmp_path / "work", Event())
    row = json.loads((stage / "candidate.json").read_bytes())["artifacts"][0]
    assert Path(row["candidate"]).stat().st_mtime_ns == 1_234_000_000


def test_newer_schema_refuses_before_destination_creation(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(
        tmp_path, mutate=lambda doc: doc["owners"][0].update(schema_version=999)
    )
    with pytest.raises(ValueError, match="unsupported_schema_version"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        )
    assert not (tmp_path / "new").exists()


def test_real_older_sqlite_migrates_only_private_candidate(tmp_path):
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters

    owner = recovery_adapters()[0]
    source = tmp_path / "older.db"
    with closing(sqlite3.connect(source)) as db:
        for sql in owner.schema_policy().schema_sql[0][1]:
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                db.execute(sql)
        db.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('kept','nebula','now','now')"
        )
        db.commit()
    original = source.read_bytes()

    def research(doc):
        doc["owners"][0].update(owner_id=owner.owner_id, schema_version=0)
        doc["files"][0]["owner_id"] = owner.owner_id

    archive = sealed(tmp_path, mutate=research, data=original)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((candidate / "candidate.json").read_bytes())["artifacts"]
    path = Path(next(row for row in rows if row["kind"] == "file")["candidate"])
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (1,)
        assert db.execute(
            "SELECT query,lease_attempts FROM research_runs"
        ).fetchall() == [("nebula", 0)]
    assert source.read_bytes() == original


def test_synthetic_container_does_not_authorize_sibling_retirement(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def synthetic(doc):
        producer(doc)
        doc["directories"][0]["synthetic"] = True

    archive = sealed(tmp_path, mutate=synthetic)
    destination = tmp_path / "live"
    destination.mkdir()
    sibling = destination / "unrelated"
    sibling.write_text("preserved")
    target = Inventory(
        (
            StorageItem("ui.state", "root", destination, "included_directory", ()),
            StorageItem("ui.state", "sibling", sibling, "included", ()),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    assert plan.retire == ()
    assert ("sibling", sibling) in plan.preserve
    assert "root" not in dict(plan.restore)


def test_multiple_synthetic_roots_share_new_local_parent_without_file_collision(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def synthetic(doc):
        doc["directories"][0]["synthetic"] = True
        doc["directories"].append(
            {
                **doc["directories"][0],
                "logical_id": "other-root",
                "root_id": "other-root",
            }
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "logical_id": "other-file",
                "root_id": "other-root",
                "parent_id": "other-root",
                "relative_path": "other.txt",
                "payload": "payload/2",
            }
        )
        doc["dependency_groups"][0]["members"].append("other-file")

    archive = sealed(tmp_path, mutate=synthetic)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={"root": destination, "other-root": destination},
        target=None,
    )
    stage = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((stage / "candidate.json").read_bytes())["artifacts"]
    assert sum(row["publication_unit"] for row in rows) == 1
    assert {Path(row["candidate"]).name for row in rows if row["kind"] == "file"} == {
        "note.txt",
        "other.txt",
    }
    assert not destination.exists()


def test_replace_includes_exact_sqlite_sidecars_in_rollback_scope(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def database(doc):
        producer(doc)
        doc["directories"][0]["synthetic"] = True
        doc["owners"][0].update(owner_id="research.local", schema_version=1)
        doc["files"][0]["owner_id"] = "research.local"
        for row in doc["producer_inventory"]:
            row["owner_id"] = "research.local"

    archive = sealed(tmp_path, mutate=database)
    destination = tmp_path / "live"
    destination.mkdir()
    database_path = destination / "note.txt"
    database_path.write_bytes(b"original data")
    wal = destination / "note.txt-wal"
    wal.write_bytes(b"committed WAL must reach safety copy")
    target = Inventory(
        (
            StorageItem("research.local", "file", database_path, "included", ()),
            StorageItem(
                "sqlite.transient", "wal", wal, "intentionally_excluded", ("file",)
            ),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    assert plan.retire == (("wal", wal),)
    assert wal.read_bytes() == b"committed WAL must reach safety copy"


def test_unqualified_target_volume_creates_no_candidates(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import qualification
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    monkeypatch.setattr(
        qualification,
        "qualified_for",
        lambda *args: (False, "native_platform_unqualified"),
    )
    with pytest.raises(ValueError, match="native_platform_unqualified"):
        stage_restore(archive, plan, tmp_path / "work", Event())
    assert not list(tmp_path.glob(".chatbook-restore-*"))


def test_metadata_normalization_and_legacy_omission_are_explicit(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    assert "metadata_unavailable:file" in plan.issues
    second = tmp_path / "second"
    second.mkdir()
    archive = sealed(
        second,
        mutate=lambda doc: doc["files"][0].update(
            metadata={"version": 1, "mode": 420, "mtime_ns": 1234000000}
        ),
    )
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": second / "new"}, target=None
    )
    assert "metadata_normalized:file" in plan.issues
    candidate = stage_restore(archive, plan, second / "work", Event())
    row = next(
        row
        for row in json.loads((candidate / "candidate.json").read_bytes())["artifacts"]
        if row["logical_id"] == "file"
    )
    assert row["desired_metadata"]["mode"] == 0o644
    assert row["applied_metadata"]["mode"] == 0o600
    assert Path(row["candidate"]).stat().st_mtime_ns == 1234000000


def test_credential_material_is_validated_privately_without_profile_destination(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_archive_reader import manifest
    from tldw_chatbook.Backup_Recovery import archive_reader, credentials
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    doc = manifest(b"durable")
    doc["owners"] = [
        {"owner_id": "ui.state", "schema_version": 1, "capabilities": []},
        {"owner_id": "recovery.credentials", "schema_version": 1, "capabilities": []},
    ]
    doc["files"][0]["owner_id"] = "ui.state"
    material = json.dumps({"version": 1, "mode": "include", "records": []}).encode()
    doc["credential_policy"] = "include"
    doc["directories"].append(
        {
            **doc["directories"][0],
            "logical_id": "credential-root",
            "root_id": "credential-root",
            "synthetic": True,
        }
    )
    doc["files"].append(
        {
            **doc["files"][0],
            "logical_id": "credentials",
            "root_id": "credential-root",
            "parent_id": "credential-root",
            "relative_path": "credential-recovery.json",
            "owner_id": "recovery.credentials",
            "payload": "payload/credential-recovery.json",
            "size": len(material),
            "sha256": hashlib.sha256(material).hexdigest(),
        }
    )
    doc["dependency_groups"].append(
        {"group_id": "credentials", "members": ["credentials"], "complete": True}
    )
    private = tmp_path / "sealed"
    private.mkdir(mode=0o700)
    source = private / "decrypted.zip"
    with zipfile.ZipFile(source, "w") as container:
        container.writestr("manifest.json", json.dumps(doc))
        container.writestr("payload/1", b"durable")
        container.writestr("payload/credential-recovery.json", material)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical = archive_reader._inspect(source, ArchiveLimits(), True, Event(), digest)
    archive = SealedArchive(source, digest, canonical)
    monkeypatch.setattr(
        credentials,
        "_credential_store",
        lambda: pytest.fail("no keyring needed for empty material"),
    )
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((candidate / "candidate.json").read_bytes())
    assert descriptor["credential_scopes"] == {}
    assert "credentials" not in {row["logical_id"] for row in descriptor["artifacts"]}
    assert (candidate / "credential-recovery.json").read_bytes() == material


def test_producer_optional_exclusion_is_preserved_inside_existing_tree(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def excluded(doc):
        producer(doc)
        doc["producer_inventory"].append(
            {
                "logical_id": "optional",
                "owner_id": "ui.state",
                "status": "intentionally_excluded",
                "dependencies": [],
                "shared_group": None,
            }
        )
        doc["exclusions"].append(
            {"logical_id": "optional", "reason": "intentionally_excluded"}
        )

    archive = sealed(tmp_path, mutate=excluded)
    destination = tmp_path / "live"
    destination.mkdir()
    optional = destination / "optional"
    optional.write_bytes(b"keep outside replacement")
    target = Inventory(
        (
            StorageItem("ui.state", "root", destination, "included_directory", ()),
            StorageItem("ui.state", "optional", optional, "included", ()),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    assert plan.retire == ()
    assert plan.preserve == (("optional", optional),)
    stage = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((stage / "candidate.json").read_bytes())["artifacts"]
    assert not next(row for row in rows if row["logical_id"] == "root")[
        "publication_unit"
    ]
    assert optional.read_bytes() == b"keep outside replacement"


def test_cancel_after_private_copy_cleans_candidates_without_touching_live_data(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import staging
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    cancel = Event()
    original = staging._copy

    def copied(*args):
        result = original(*args)
        cancel.set()
        return result

    monkeypatch.setattr(staging, "_copy", copied)
    with pytest.raises(InterruptedError):
        staging.stage_restore(archive, plan, tmp_path / "work", cancel)
    assert not destination.exists()
    assert not list(tmp_path.glob(".chatbook-restore-*"))
    assert not list((tmp_path / "work").iterdir())


def test_insufficient_space_refuses_before_candidate_writes(tmp_path, monkeypatch):
    from collections import namedtuple

    from tldw_chatbook.Backup_Recovery import staging
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: usage(100, 100, 0))
    with pytest.raises(ValueError, match="insufficient_space"):
        staging.stage_restore(archive, plan, tmp_path / "work", Event())
    assert not (tmp_path / "work").exists()


def test_config_identity_is_explicit_and_mapped_raw_store_is_reachable(tmp_path):
    import tomllib

    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def config_with_state(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["owners"].append(
            {"owner_id": "ui.state", "schema_version": 1, "capabilities": []}
        )
        doc["files"][0].update(
            owner_id="config",
            logical_id="profile:profile:config",
            relative_path="config.toml",
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "owner_id": "ui.state",
                "logical_id": "profile:profile:ui.state",
                "relative_path": "ui_state.toml",
                "payload": "payload/2",
            }
        )
        doc["dependency_groups"][0]["members"] = [
            "profile:profile:config",
            "profile:profile:ui.state",
        ]

    archive = sealed(
        tmp_path, mutate=config_with_state, data=b'[general]\nusers_name="original"\n'
    )
    destination = tmp_path / "config-root"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={
            "root": destination,
            "profile:profile:paths.data_dir": tmp_path / "data",
        },
        target=None,
        profile_names={"profile": "chosen_local_name"},
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((candidate / "candidate.json").read_bytes())["artifacts"]
    actual = tomllib.loads(
        Path(
            next(row for row in rows if row["logical_id"] == "profile:profile:config")[
                "candidate"
            ]
        ).read_text()
    )
    assert actual["general"]["users_name"] == "chosen_local_name"
    assert (
        dict(plan.restore)["profile:profile:ui.state"] == destination / "ui_state.toml"
    )


def test_selected_shared_files_remain_one_physical_candidate(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def aliases(doc):
        producer(doc)
        doc["directories"][0]["synthetic"] = True
        doc["directories"].append(
            {
                **doc["directories"][0],
                "logical_id": "alias-root",
                "root_id": "alias-root",
            }
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "logical_id": "alias",
                "root_id": "alias-root",
                "parent_id": "alias-root",
                "payload": "payload/2",
            }
        )
        doc["producer_inventory"][1]["shared_group"] = "shared"
        doc["producer_inventory"].extend(
            [
                {
                    "logical_id": "alias-root",
                    "owner_id": "ui.state",
                    "status": "included_directory",
                    "dependencies": [],
                    "shared_group": None,
                },
                {
                    "logical_id": "alias",
                    "owner_id": "ui.state",
                    "status": "included",
                    "dependencies": ["alias-root"],
                    "shared_group": "shared",
                },
            ]
        )
        doc["dependency_groups"][0]["members"].append("alias")

    archive = sealed(tmp_path, mutate=aliases)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={"root": destination, "alias-root": destination},
        target=None,
    )
    candidate = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((candidate / "candidate.json").read_bytes())["artifacts"]
    files = [Path(row["candidate"]) for row in rows if row["kind"] == "file"]
    assert len(files) == 2
    assert files[0] == files[1]
    assert files[0].read_bytes() == b"durable"


from Tests.Backup_Recovery.test_file_inventory import (
    installed_model as installed_model,  # noqa: PLC0414
)


def archive_entries(tmp_path, config, entries, *, include_config=True, mutate=None):
    """Use the current producer serializer with real installed-owner payloads."""
    import shutil

    import toml

    from tldw_chatbook.Backup_Recovery.capture import _manifest_for
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    selector = config[DISCOVERY_CONTEXT_KEY].config_path
    selector.write_text(
        toml.dumps({k: v for k, v in config.items() if k != DISCOVERY_CONTEXT_KEY})
    )
    entries = tuple(
        item for item in entries if not item.logical_id.endswith(":participant_pending")
    )
    if include_config:
        entries = (*config_adapter().discover(config), *entries)
    inventory = Inventory(tuple(entries), True, "scope", ())
    stage = tmp_path / "serialized"
    (stage / "payload").mkdir(parents=True)
    staged = []
    for index, item in enumerate(entries):
        if item.status == "included":
            path = stage / "payload" / str(index)
            shutil.copyfile(item.path, path)
            staged.append((item, path))
    manifest = _manifest_for(
        inventory,
        staged,
        {},
        {
            "root": stage,
            "versions": {},
            "cancel": Event(),
            "mode": "exclude",
            "encrypted": False,
            "limits": ArchiveLimits(),
        },
        (),
    )
    doc = json.loads(manifest)
    if mutate is not None:
        mutate(doc)
        manifest = json.dumps(doc).encode()
    archive_path = tmp_path / "actual.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", manifest)
        for _, path in staged:
            archive.write(path, path.relative_to(stage))
    return acquire(
        archive_path, tmp_path / "sealed", ArchiveLimits(), None, Event()
    ), doc


@pytest.mark.parametrize("mode", ["isolated", "replace"])
def test_real_model_bundle_config_uses_only_selected_static_paths(
    installed_model, tmp_path, mode
):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    _store, _desc, config, adapter = installed_model
    config["database"] = {"media_db_path": "/old-machine/unselected.db"}
    archive, doc = archive_entries(tmp_path, config, adapter.discover(config))
    new_data = tmp_path / "new-data"
    destinations = {
        row["logical_id"]: (
            tmp_path / "new-config"
            if row.get("synthetic")
            else new_data / "Local" / "models"
        )
        for row in doc["directories"]
        if row["parent_id"] is None
    }
    destinations["profile:p:paths.data_dir"] = new_data
    plan = plan_restore(
        archive,
        mode=mode,
        destinations=destinations,
        target=None if mode == "isolated" else Inventory((), True, "target", ()),
        profile_names={"p": "Local"},
    )
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    records = json.loads((result / "candidate.json").read_bytes())["artifacts"]
    assert any(
        Path(row["candidate"]).read_bytes() == b"actual model payload"
        for row in records
        if row["kind"] == "file"
    )
    assert not new_data.exists()


def test_missing_local_parents_are_explicit_private_container_candidates(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    destination = tmp_path / "missing" / "user" / "root"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((result / "candidate.json").read_bytes())
    assert [Path(row["destination"]) for row in descriptor["containers"]] == [
        tmp_path / "missing",
        tmp_path / "missing" / "user",
    ]
    assert all(
        not list(Path(row["candidate"]).iterdir()) for row in descriptor["containers"]
    )
    assert not (tmp_path / "missing").exists()


def test_unreadable_metadata_is_disclosed_and_private_candidates_remain_usable(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def no_access(doc):
        doc["directories"][0]["metadata"]["mode"] = 0
        doc["files"][0]["metadata"] = {"version": 1, "mode": 0, "mtime_ns": 0}

    archive = sealed(tmp_path, mutate=no_access)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    rows = json.loads((result / "candidate.json").read_bytes())["artifacts"]
    assert all(
        row["applied_metadata"]["mode"]
        & (0o700 if row["kind"] == "directory" else 0o600)
        == (0o700 if row["kind"] == "directory" else 0o600)
        for row in rows
    )
    assert "metadata_normalized:file" in plan.issues


@pytest.mark.parametrize("empty", [False, True])
def test_coherent_external_owner_still_requires_new_destination(tmp_path, empty):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def external(doc):
        doc["owners"][0]["owner_id"] = "external.files"
        doc["files"][0]["owner_id"] = "external.files"
        producer(doc)
        for item in doc["producer_inventory"]:
            item["owner_id"] = "external.files"
        if empty:
            doc["files"] = []
            doc["producer_inventory"] = doc["producer_inventory"][:1]
            doc["dependency_groups"] = []

    archive = sealed(tmp_path, mutate=external)
    destination = tmp_path / "existing"
    destination.mkdir()
    target = Inventory(
        (StorageItem("external.files", "root", destination, "included_directory", ()),),
        True,
        "target",
        (),
    )
    with pytest.raises(ValueError, match="external_destination_exists"):
        plan_restore(
            archive, mode="replace", destinations={"root": destination}, target=target
        )


def test_work_directory_cannot_be_a_preserved_live_target(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    preserved = tmp_path / "preserved"
    preserved.mkdir(mode=0o700)
    target = Inventory(
        (StorageItem("ui.state", "preserved", preserved, "included_directory", ()),),
        True,
        "target",
        (),
    )
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=target
    )
    before = preserved.stat()
    with pytest.raises(ValueError, match="staging_target_alias"):
        stage_restore(archive, plan, preserved, Event())
    assert preserved.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("mode", ["isolated", "replace"])
@pytest.mark.parametrize("forged_synthetic", [None, False, True])
def test_real_persona_core_bundle_never_discovers_unpublished_target(
    tmp_path, mode, monkeypatch, forged_synthetic
):
    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
    from tldw_chatbook.Persona_Visual.recovery import _Assets, recovery_adapters
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

    data = tmp_path / "data" / "Ada"
    data.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    database = tmp_path / "core.db"
    db = CharactersRAGDB(database, "recovery")
    try:
        publish_persona_visual(
            PersonaVisualRepository(db),
            _snapshot(source),
            source_root=source,
            profile_root=data,
            authority_guard=lambda: True,
        )
        asset_path = (
            data
            / db.get_connection()
            .execute("SELECT storage_relpath FROM persona_visual_assets LIMIT 1")
            .fetchone()[0]
        )
    finally:
        db.close()
    selector = tmp_path / "config.toml"
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        "database": {"chachanotes_db_path": str(database)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    entries = (
        *recovery_adapters()[0].discover(config),
        StorageItem(
            "db.chachanotes.primary",
            "profile:p:db.chachanotes.primary",
            database,
            "included",
            ("profile:p:config", "profile:p:persona.assets"),
        ),
    )
    if forged_synthetic is not None:
        from dataclasses import replace

        # A valid archive digest is insufficient: these bytes disagree with the
        # referenced asset digest retained inside the real core database.
        asset_path.write_bytes(b"authentically archived but wrong persona bytes")
        entries = tuple(
            replace(
                item,
                dependencies=tuple(
                    key for key in item.dependencies if key != "profile:p:config"
                ),
            )
            for item in entries
        )

    def forge(doc):
        next(
            row
            for row in doc["directories"]
            if row["logical_id"] == "profile:p:persona.assets"
        )["synthetic"] = forged_synthetic

    archive, doc = archive_entries(
        tmp_path,
        config,
        entries,
        include_config=forged_synthetic is None,
        mutate=forge if forged_synthetic is not None else None,
    )
    new_data = tmp_path / "new-data"
    destinations = {}
    for root in (row for row in doc["directories"] if row["parent_id"] is None):
        destinations[root["logical_id"]] = (
            (tmp_path / "new-config")
            if root["logical_id"] != "profile:p:persona.assets"
            else new_data / "Local" / "persona_visual"
        )
    destinations["profile:p:paths.data_dir"] = new_data
    plan = plan_restore(
        archive,
        mode=mode,
        destinations=destinations,
        target=None if mode == "isolated" else Inventory((), True, "target", ()),
        profile_names={"p": "Local"},
    )

    def forbidden(*args):
        raise AssertionError("staging must not discover unpublished target DB")

    monkeypatch.setattr(_Assets, "discover", forbidden)
    if forged_synthetic is not None:
        with pytest.raises(
            ValueError, match="asset_digest_mismatch|invalid_synthetic_asset_root"
        ):
            stage_restore(archive, plan, tmp_path / "work", Event())
        assert not new_data.exists()
        return
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    records = json.loads((result / "candidate.json").read_bytes())["artifacts"]
    asset = next(
        row
        for row in doc["files"]
        if row["relative_path"]
        == asset_path.relative_to(data / "persona_visual").as_posix()
    )
    restored = next(row for row in records if row["logical_id"] == asset["logical_id"])
    assert Path(restored["candidate"]).read_bytes() == asset_path.read_bytes()
    assert not new_data.exists()


def test_legacy_empty_root_cannot_invent_installed_ownership(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def empty(doc):
        doc["files"] = []
        doc["dependency_groups"] = []

    archive = sealed(tmp_path, mutate=empty)
    with pytest.raises(ValueError, match="producer_inventory_required"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / "new"},
            target=None,
        )


from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_recovered_media import media as media  # noqa: PLC0414


@pytest.mark.parametrize("mode", ["isolated", "replace"])
def test_real_recovered_catalog_and_payload_stage_together(media, tmp_path, mode):
    from Tests.Backup_Recovery.test_recovered_media import retain
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.recovered_media import _RecoveredAdapter
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    store, source = media
    identity = retain(store, source)
    selector = tmp_path / "config.toml"
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    archive, doc = archive_entries(
        tmp_path, config, _RecoveredAdapter(store.root).discover(config)
    )
    new_data = tmp_path / "new-data"
    destinations = {
        row["logical_id"]: (
            tmp_path / "new-config"
            if row.get("synthetic")
            else new_data / "Local" / "recovered_media"
        )
        for row in doc["directories"]
        if row["parent_id"] is None
    }
    destinations["profile:p:paths.data_dir"] = new_data
    plan = plan_restore(
        archive,
        mode=mode,
        destinations=destinations,
        target=None if mode == "isolated" else Inventory((), True, "target", ()),
        profile_names={"p": "Local"},
    )
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    records = json.loads((result / "candidate.json").read_bytes())["artifacts"]
    payload = next(
        row for row in records if Path(row["destination"]).name == identity + ".payload"
    )
    assert Path(payload["candidate"]).read_bytes() == source.read_bytes()
    assert not new_data.exists()


@pytest.mark.parametrize("refuse", [False, True])
def test_staging_receipt_observes_complete_descriptor_before_return(tmp_path, refuse):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    observed = []

    class Receipt:
        def record_candidate(self, stage, actual_plan, actual_archive):
            descriptor = json.loads((stage / "candidate.json").read_bytes())
            observed.append((actual_plan, actual_archive, descriptor["archive_digest"]))
            if refuse:
                raise ValueError("receipt_refused")

    if refuse:
        with pytest.raises(ValueError, match="receipt_refused"):
            stage_restore(archive, plan, tmp_path / "work", Event(), journal=Receipt())
        assert not list((tmp_path / "work").iterdir())
    else:
        assert stage_restore(
            archive, plan, tmp_path / "work", Event(), journal=Receipt()
        ).exists()
    assert observed == [(plan, archive, archive.digest)]


@pytest.mark.parametrize(
    "owner_id,leaf,location",
    [
        ("runtime.event_state", "tldw_chatbook_event_state.db", "data"),
        ("db.agent_runs", "tldw_chatbook_agent_runs.db", "core"),
        ("mcp.permissions", "mcp_permissions.json.bak", "data"),
        ("runtime.source_state", "runtime_policy.json", "config"),
        ("workspaces.change_tracking", "change_review", "data"),
    ],
)
@pytest.mark.parametrize("wrong", [False, True])
def test_installed_simple_selectors_are_validated_without_discovery(
    tmp_path, monkeypatch, owner_id, leaf, location, wrong
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
    from tldw_chatbook.Backup_Recovery.staging import _config_targets

    owners = {owner.owner_id: owner for owner in install_adapters()}
    owner = owners[owner_id]

    def forbidden(*args):
        raise AssertionError("no target discovery")

    monkeypatch.setattr(type(owner), "discover", forbidden)
    config_target = tmp_path / "config" / "config.toml"
    data = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Local"},
        "database": {"chachanotes_db_path": str(tmp_path / "database" / "core.db")},
    }
    base = {
        "config": config_target.parent,
        "data": tmp_path / "data" / "Local",
        "core": tmp_path / "database",
    }[location]
    if owner_id == "db.agent_runs":
        leaf = owner.leaf
    destination = base / leaf
    if wrong:
        destination = tmp_path / "wrong" / leaf
    logical = "profile:p:" + owner_id
    payload = SimpleNamespace(logical_id=logical, owner_id=owner_id, root_id="root")
    root = SimpleNamespace(logical_id="root", synthetic=True)
    doc = SimpleNamespace(files=(payload,), directories=(root,))
    plan = RestorePlan(
        "digest",
        "isolated",
        ((logical, destination),),
        (),
        (),
        "fingerprint",
        (("root", destination.parent),),
    )
    if wrong:
        with pytest.raises(ValueError, match="owner_relocation_unverified"):
            _config_targets(data, "p", config_target, doc, plan, owners)
    else:
        _config_targets(data, "p", config_target, doc, plan, owners)


def test_partial_selection_plans_only_credentials_for_selected_payloads(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_archive_reader import manifest
    from tldw_chatbook.Backup_Recovery import archive_reader, credentials
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    values = {
        name: json.dumps(
            {"targets": [{"server_id": name, "auth_reference": "keyring:api_key"}]}
        ).encode()
        for name in ("selected", "other")
    }
    records = [
        {
            "id": hashlib.sha256(name.encode()).hexdigest(),
            "kind": "server",
            "file": "payload/" + name,
            "server_id": name,
            "purpose": "api_key",
            "remappable": True,
            "status": "captured",
            "value": "fixture-secret-" + name,
        }
        for name in values
    ]
    material = json.dumps(
        {"version": 1, "mode": "include", "records": records}
    ).encode()
    doc = manifest(values["selected"])
    doc["credential_policy"] = "include"
    doc["consistency"] = "partial"
    doc["owners"] = [
        {"owner_id": "mcp.targets", "schema_version": 1, "capabilities": []},
        {"owner_id": "recovery.credentials", "schema_version": 1, "capabilities": []},
    ]
    template = doc["files"][0]
    directory = doc["directories"][0]
    payloads = {**values, "credentials": material}
    doc["directories"] = [
        {**directory, "logical_id": name, "root_id": name, "synthetic": True}
        for name in payloads
    ]
    doc["files"] = [
        {
            **template,
            "logical_id": name + "-file",
            "root_id": name,
            "parent_id": name,
            "owner_id": "recovery.credentials"
            if name == "credentials"
            else "mcp.targets",
            "relative_path": "credential-recovery.json"
            if name == "credentials"
            else "mcp_server_targets.json",
            "payload": "payload/credential-recovery.json"
            if name == "credentials"
            else "payload/" + name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in payloads.items()
    ]
    doc["dependency_groups"] = [
        {"group_id": name, "members": [name, name + "-file"], "complete": True}
        for name in payloads
    ]
    private = tmp_path / "sealed"
    private.mkdir(mode=0o700)
    path = private / "decrypted.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(doc))
        for row in doc["files"]:
            archive.writestr(row["payload"], payloads[row["root_id"]])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    archive = SealedArchive(
        path,
        digest,
        archive_reader._inspect(path, ArchiveLimits(), True, Event(), digest),
    )
    reads = []
    monkeypatch.setattr(credentials, "_credential_store", lambda: object())

    def read_scope(record, store):
        reads.append(record["server_id"])
        return record["value"]

    monkeypatch.setattr(credentials, "_read_scope", read_scope)
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={"selected": tmp_path / "new"},
        target=None,
    )
    result = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((result / "candidate.json").read_bytes())
    assert set(descriptor["credential_scopes"]) == {records[0]["id"]}
    assert reads == ["selected"]
    assert (
        json.loads((result / "credential-recovery.json").read_bytes())["records"]
        == records[:1]
    )


@pytest.mark.parametrize("mode", ["isolated", "replace"])
@pytest.mark.parametrize(
    "owner_id", ["eval.definitions", "tts.voices", "agents.history"]
)
def test_known_deferred_owners_preserve_bytes_at_new_roots_with_setup_issue(
    tmp_path, mode, owner_id
):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    source = tmp_path / "retained.yaml"
    source.write_bytes(b"retained: value\n")
    config = {
        "general": {"users_name": "Old"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "p"),
    }
    item = StorageItem(
        owner_id, "profile:p:" + owner_id, source, "included", ("profile:p:config",)
    )
    archive, doc = archive_entries(tmp_path, config, (item,))
    roots = {row["root_id"]: row["owner_id"] for row in doc["files"]}
    destinations = {
        key: tmp_path / ("new-config" if owner == "config" else "inert-owner")
        for key, owner in roots.items()
    }
    destinations["profile:p:paths.data_dir"] = tmp_path / "new-data"
    plan = plan_restore(
        archive,
        mode=mode,
        destinations=destinations,
        target=None if mode == "isolated" else Inventory((), True, "target", ()),
        profile_names={"p": "Local"},
    )
    stage = stage_restore(archive, plan, tmp_path / "work", Event())
    descriptor = json.loads((stage / "candidate.json").read_bytes())
    assert "owner_setup_required:" + owner_id in descriptor["issues"]
    row = next(
        row for row in descriptor["artifacts"] if row["logical_id"] == item.logical_id
    )
    assert Path(row["candidate"]).read_bytes() == source.read_bytes()
    assert not (tmp_path / "inert-owner").exists()
