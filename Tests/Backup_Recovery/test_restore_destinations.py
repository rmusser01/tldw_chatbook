"""Profile destination choices bind installed paths before confirmation."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore


def profile_archive(tmp_path, profiles=("source",), *, custom=False, external=False):
    """A sealed archive with native config and declared shared database owners."""
    doc = manifest()
    doc.update(
        profile_ids=list(profiles),
        files=[],
        directories=[],
        producer_inventory=[],
        dependency_groups=[],
    )
    content = {}
    owners = {
        "config": 1,
        "db.chachanotes.primary": 37,
        "db.agent_runs": 1,
        "chat.attachments": 37,
        "runtime.source_state": 1,
    }
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    installed = {a.owner_id: a for a in install_adapters()}
    for profile in profiles:
        for owner, leaf in (
            ("config", "config.toml"),
            (
                "db.chachanotes.primary",
                "custom.db" if custom else "tldw_chatbook_ChaChaNotes.db",
            ),
            (
                "chat.attachments",
                "custom.db" if custom else "tldw_chatbook_ChaChaNotes.db",
            ),
            ("db.agent_runs", "agent_runs.db"),
            ("runtime.source_state", "runtime_policy.json"),
        ):
            key = f"profile:{profile}:{owner}"
            root = f"root-{profile}-{owner}"
            data = (
                b'[general]\nusers_name="Old"\n[paths]\ndata_dir="/archive/source/data"\n'
                if owner == "config"
                else b"native-fixture"
            )
            doc["directories"].append(
                {
                    "logical_id": root,
                    "root_id": root,
                    "parent_id": None,
                    "relative_path": "",
                    "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
                    "synthetic": True,
                }
            )
            doc["files"].append(
                {
                    "logical_id": key,
                    "root_id": root,
                    "parent_id": root,
                    "relative_path": leaf,
                    "owner_id": owner,
                    "payload": "payload/" + str(len(content)),
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            content[doc["files"][-1]["payload"]] = data
            shared = (
                f"shared-{profile}"
                if owner in {"db.chachanotes.primary", "chat.attachments"}
                else None
            )
            doc["producer_inventory"].extend(
                [
                    {
                        "logical_id": root,
                        "owner_id": owner,
                        "status": "included_directory",
                        "dependencies": [],
                        "shared_group": None,
                    },
                    {
                        "logical_id": key,
                        "owner_id": owner,
                        "status": "included",
                        "dependencies": [],
                        "shared_group": shared,
                    },
                ]
            )
    if external:
        root = "external"
        doc["directories"].append(
            {
                "logical_id": root,
                "root_id": root,
                "parent_id": None,
                "relative_path": "",
                "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": root,
                "owner_id": "external.files",
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            }
        )
        owners["external.files"] = 1
    doc["dependency_groups"] = [
        {
            "group_id": "all",
            "members": [f["logical_id"] for f in doc["files"]],
            "complete": True,
        }
    ]

    doc["owners"] = [
        {
            "owner_id": o,
            "schema_version": installed[o].schema_policy().versions[-1],
            "capabilities": [],
        }
        for o in owners
    ]
    source = tmp_path / "incoming.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("manifest.json", json.dumps(doc))
        for key, data in content.items():
            archive.writestr(key, data)
    return source


@pytest.fixture
def inspected(tmp_path):
    service = RecoveryService(tmp_path / "control")
    (tmp_path / "restore-locations").mkdir(mode=0o700)

    def inspect(**kwargs):
        op = service.start_inspection(
            profile_archive(tmp_path, **kwargs), password=None
        )
        state = service.wait(op)
        assert state["state"] == "succeeded", dict(state)
        return service, op

    yield inspect
    service.close()


def test_summary_asks_one_profile_base_and_independent_external_root(inspected):
    service, op = inspected(external=True)
    slots = service.summary(op)["destination_slots"]
    assert [(s["logical_id"], s["kind"]) for s in slots] == [
        ("source", "profile_base"),
        ("external", "external_root"),
    ]


def test_profile_choice_colocates_shared_db_and_fixed_owners(inspected, tmp_path):
    service, op = inspected(custom=True)
    base = tmp_path / "restore-locations" / "new"
    plan = service.preview_restore(
        op,
        mode="isolated",
        profile_bases={"source": base},
        external_destinations={},
        target=None,
        profile_names={"source": "New User"},
    )
    paths = dict(plan.restore)
    assert (
        paths["profile:source:db.chachanotes.primary"]
        == paths["profile:source:chat.attachments"]
        == base / "data" / "New_User" / "custom.db"
    )
    assert (
        paths["profile:source:db.agent_runs"]
        == base / "data" / "New_User" / "agent_runs.db"
    )
    assert (
        paths["profile:source:runtime.source_state"]
        == base / "config" / "runtime_policy.json"
    )
    assert dict(plan.selectors)["profile:source:paths.data_dir"] == base / "data"


def test_preview_rejects_late_owner_failure_for_explicit_api(inspected, tmp_path):
    service, op = inspected()
    doc = service.summary(op)
    destinations = {
        r["logical_id"]: tmp_path / "roots" / str(i) for i, r in enumerate(doc["roots"])
    }
    archive_doc = json.loads(service.inspection(op).manifest_bytes)
    for row in archive_doc["files"]:
        if row["owner_id"] in {"db.chachanotes.primary", "chat.attachments"}:
            destinations[row["root_id"]] = tmp_path / "shared"
    destinations["profile:source:paths.data_dir"] = tmp_path / "data"
    plan_restore(
        service.inspection(op),
        mode="isolated",
        target=None,
        destinations=destinations,
        profile_names={"source": "New"},
    )
    with pytest.raises(ValueError, match="owner_relocation_unverified"):
        service.preview_restore(
            op,
            mode="isolated",
            target=None,
            destinations=destinations,
            profile_names={"source": "New"},
        )


@pytest.mark.parametrize("kind", ["relative", "existing", "alias", "overlap"])
def test_profile_bases_refuse_unsafe_paths(inspected, tmp_path, kind):
    service, op = inspected(profiles=("one", "two"))
    first, second = tmp_path / "new", tmp_path / "other"
    if kind == "relative":
        first = Path("relative")
    elif kind == "existing":
        first.mkdir()
    elif kind == "alias":
        first.symlink_to(tmp_path, target_is_directory=True)
    else:
        second = first / "inside"
    with pytest.raises(ValueError):
        service.preview_restore(
            op,
            mode="isolated",
            profile_bases={"one": first, "two": second},
            external_destinations={},
            target=None,
            profile_names={"one": "One", "two": "Two"},
        )


def test_multiple_profiles_have_distinct_new_layouts(inspected, tmp_path):
    service, op = inspected(profiles=("one", "two"))
    plan = service.preview_restore(
        op,
        mode="isolated",
        profile_bases={p: tmp_path / "restore-locations" / p for p in ("one", "two")},
        external_destinations={},
        target=None,
        profile_names={"one": "Same name", "two": "Same name"},
    )
    paths = dict(plan.restore)
    assert paths["profile:one:db.agent_runs"].is_relative_to(
        tmp_path / "restore-locations" / "one"
    )
    assert paths["profile:two:db.agent_runs"].is_relative_to(
        tmp_path / "restore-locations" / "two"
    )


@pytest.mark.parametrize("broken", [False, True])
def test_replacement_uses_independent_target_custom_paths_and_identity(
    inspected, tmp_path, broken
):
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    service, op = inspected(custom=True)
    live = tmp_path / "live"
    live.mkdir()
    config = live / "config.toml"
    config.write_text(
        "broken = ["
        if broken
        else f'[general]\nusers_name="live"\n[paths]\ndata_dir="{tmp_path}"\n[database]\nchachanotes_db_path="{live / "custom.db"}"\n'
    )
    items = []
    for owner, name in (
        ("config", "config.toml"),
        ("db.chachanotes.primary", "custom.db"),
        ("chat.attachments", "custom.db"),
        ("db.agent_runs", "agent_runs.db"),
        ("runtime.source_state", "runtime_policy.json"),
    ):
        path = live / name
        if owner != "config":
            path.write_bytes(b"existing target")
        items.append(
            StorageItem(
                owner,
                f"profile:local:{owner}",
                path,
                "included",
                (),
                shared_group="local-shared"
                if owner in {"db.chachanotes.primary", "chat.attachments"}
                else None,
            )
        )
    inventory = Inventory(tuple(items), True, "independent", ())
    plan = service.preview_restore(
        op,
        mode="replace",
        profile_bases={},
        external_destinations={},
        target_configs={"source": config},
        target=inventory,
        profile_names={},
    )
    assert dict(plan.restore)["profile:source:db.agent_runs"] == live / "agent_runs.db"
    assert (
        dict(plan.restore)["profile:source:db.chachanotes.primary"]
        == live / "custom.db"
    )
    assert dict(plan.profile_names) == {"source": "live"}


def test_extraction_summary_names_contents_and_counts(inspected):
    service, op = inspected()
    group = service.summary(op)["dependency_groups"][0]
    assert group["file_count"] == 5
    assert group["payload_bytes"] > 0
    assert "config.toml" in group["sample_files"]
    assert "Notes and chats" in group["label"]
    assert "root-" not in group["label"]


@pytest.mark.parametrize(
    "code",
    [
        "destination_exists",
        "destination_alias",
        "profile_identity_required",
        "owner_relocation_unverified:db.agent_runs",
        "shared_target_split",
        "target_unverified",
    ],
)
def test_known_destination_errors_have_bounded_actionable_messages(code):
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_code, issue_message

    visible = issue_message(issue_code(ValueError(code)))
    assert visible != "backup_operation_failed"
    assert any(word in visible.lower() for word in ("choose", "review", "enter"))
    assert "/archive/source" not in visible


def test_profiles_sharing_notes_keep_one_database_and_agent_history(tmp_path):
    from threading import Event

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    source = profile_archive(tmp_path, profiles=("one", "two"))
    with zipfile.ZipFile(source) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    doc = json.loads(content["manifest.json"])
    for row in doc["producer_inventory"]:
        if row["status"] == "included":
            if row["owner_id"] in {"db.chachanotes.primary", "chat.attachments"}:
                row["shared_group"] = "shared-notes"
            elif row["owner_id"] == "db.agent_runs":
                row["shared_group"] = "shared-agents"
    content["manifest.json"] = json.dumps(doc).encode()
    source.unlink()
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in content.items():
            archive.writestr(name, data)
    acquired = archive_reader.acquire(
        source, tmp_path / "acquired", ArchiveLimits(), None, Event()
    )
    plan = resolve_destinations(
        acquired,
        mode="isolated",
        profile_bases={p: tmp_path / p for p in ("one", "two")},
        external_destinations={},
        target=None,
        profile_names={"one": "One", "two": "Two"},
    )
    restored = dict(plan.restore)
    assert (
        restored["profile:one:db.chachanotes.primary"]
        == restored["profile:two:db.chachanotes.primary"]
    )
    assert (
        restored["profile:one:db.agent_runs"] == restored["profile:two:db.agent_runs"]
    )


def test_malformed_config_identity_remains_inspectable_for_manual_extraction(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed

    def malformed(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["files"][0]["owner_id"] = "config"

    archive = sealed(
        tmp_path, mutate=malformed, data=b'[general]\nusers_name="Example"\n'
    )
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(archive.path, password=None)
        assert service.wait(op)["state"] == "succeeded"
        assert service.summary(op)["destination_slots"] == ()
        with pytest.raises(ValueError, match="config_profile_unverified"):
            service.preview_restore(
                op,
                mode="isolated",
                profile_bases={},
                external_destinations={},
                target=None,
                profile_names={},
            )
    finally:
        service.close()


def _rewrite_archive(source, mutate):
    """Rewrite fixture metadata and payloads before verified acquisition."""
    with zipfile.ZipFile(source) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    doc = json.loads(content["manifest.json"])
    mutate(doc, content)
    content["manifest.json"] = json.dumps(doc).encode()
    source.unlink()
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in content.items():
            archive.writestr(name, data)


@pytest.mark.parametrize(
    "data",
    [b"general=1\n", b"AppRAGSearchConfig=1\n", b"[AppRAGSearchConfig]\nrag=1\n"],
)
def test_imported_scalar_config_tables_refuse_safely(tmp_path, data):
    source = profile_archive(tmp_path)

    def malformed(doc, content):
        row = next(row for row in doc["files"] if row["owner_id"] == "config")
        row.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        content[row["payload"]] = data

    _rewrite_archive(source, malformed)
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(source, password=None)
        assert service.wait(op)["state"] == "succeeded"
        with pytest.raises(ValueError, match="invalid_config_shape"):
            service.preview_restore(
                op,
                mode="isolated",
                profile_bases={"source": tmp_path / "new"},
                external_destinations={},
                target=None,
                profile_names={"source": "New"},
            )
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["isolated", "replace"])
def test_shared_deferred_voices_keep_one_inert_destination(tmp_path, mode):
    source = profile_archive(tmp_path, profiles=("one", "two"))

    def voices(doc, content):
        from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

        owner = next(a for a in install_adapters() if a.owner_id == "tts.voices")
        doc["owners"].append(
            {
                "owner_id": "tts.voices",
                "schema_version": owner.schema_policy().versions[-1],
                "capabilities": [],
            }
        )
        for profile in ("one", "two"):
            root = f"profile:{profile}:tts.voices"
            key = root + ":voice"
            doc["directories"].append(
                {
                    "logical_id": root,
                    "root_id": root,
                    "parent_id": None,
                    "relative_path": "",
                    "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
                }
            )
            doc["files"].append(
                {
                    "logical_id": key,
                    "root_id": root,
                    "parent_id": root,
                    "relative_path": "voice.wav",
                    "owner_id": "tts.voices",
                    "payload": "payload/voice-" + profile,
                    "size": 5,
                    "sha256": hashlib.sha256(b"voice").hexdigest(),
                }
            )
            content["payload/voice-" + profile] = b"voice"
            doc["producer_inventory"].extend(
                [
                    {
                        "logical_id": root,
                        "owner_id": "tts.voices",
                        "status": "included_directory",
                        "dependencies": [],
                        "shared_group": "shared-voice-root",
                    },
                    {
                        "logical_id": key,
                        "owner_id": "tts.voices",
                        "status": "included",
                        "dependencies": [root],
                        "shared_group": "shared-voice-file",
                    },
                ]
            )
            doc["dependency_groups"][0]["members"].extend([root, key])

    _rewrite_archive(source, voices)
    (tmp_path / "restore-locations").mkdir(mode=0o700)
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(source, password=None)
        assert service.wait(op)["state"] == "succeeded"
        choices = {
            "mode": mode,
            "profile_bases": {
                p: tmp_path / "restore-locations" / p for p in ("one", "two")
            },
            "external_destinations": {},
            "target": None,
            "profile_names": {"one": "One", "two": "Two"},
        }
        if mode == "replace":
            from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

            items, configs = [], {}
            for profile in ("one", "two"):
                live = tmp_path / ("live-" + profile)
                user_root = live / "data" / profile
                user_root.mkdir(mode=0o700, parents=True)
                config = live / "config.toml"
                config.write_text(
                    f'[general]\nusers_name="{profile}"\n[paths]\ndata_dir="{live / "data"}"\n'
                )
                configs[profile] = config
                for owner, path in (
                    ("config", config),
                    ("runtime.source_state", live / "runtime_policy.json"),
                    (
                        "db.chachanotes.primary",
                        user_root / "tldw_chatbook_ChaChaNotes.db",
                    ),
                    ("chat.attachments", user_root / "tldw_chatbook_ChaChaNotes.db"),
                    ("db.agent_runs", user_root / "agent_runs.db"),
                ):
                    if owner != "config":
                        path.write_bytes(b"existing")
                    items.append(
                        StorageItem(
                            owner,
                            f"profile:{profile}:{owner}",
                            path,
                            "included",
                            (),
                            shared_group="local-notes-" + profile
                            if owner in {"db.chachanotes.primary", "chat.attachments"}
                            else None,
                        )
                    )
            setup = tmp_path / "files-needing-setup"
            setup.mkdir(mode=0o700)
            choices.update(
                profile_bases={},
                target_configs=configs,
                target=Inventory(tuple(items), True, "local", ()),
                setup_parent=setup,
            )
        plan = service.preview_restore(op, **choices)
        paths = dict(plan.restore)
        assert (
            paths["profile:one:tts.voices:voice"]
            == paths["profile:two:tts.voices:voice"]
        )
        voice = paths["profile:one:tts.voices:voice"]
        if mode == "replace":
            assert voice.parent.parent == setup
            assert setup.is_dir() and not voice.parent.exists()
        else:
            assert "inert" in voice.parts
        assert "owner_setup_required:tts.voices" in plan.issues
    finally:
        service.close()


def test_preview_refuses_parent_that_contains_recovery_storage(inspected, tmp_path):
    service, op = inspected()
    with pytest.raises(
        ValueError, match="isolated_destination_parent_overlaps_control"
    ):
        service.preview_restore(
            op,
            mode="isolated",
            profile_bases={"source": tmp_path / "new"},
            external_destinations={},
            target=None,
            profile_names={"source": "New"},
        )


@pytest.mark.parametrize("data", ["notes=1\n", "[notes]\nsync_directory=1\n"])
def test_invalid_replacement_selector_shapes_refuse_safely(inspected, tmp_path, data):
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    service, op = inspected()
    target_config = tmp_path / "target.toml"
    target_config.write_text(data)
    inventory = Inventory(
        (
            StorageItem(
                "config", "profile:target:config", target_config, "included", ()
            ),
        ),
        True,
        "local",
        (),
    )
    with pytest.raises(ValueError, match="invalid_config_shape"):
        service.preview_restore(
            op,
            mode="replace",
            profile_bases={},
            external_destinations={},
            target_configs={"source": target_config},
            target=inventory,
            profile_names={},
        )


def test_unused_deferred_owner_needs_no_setup_directory(tmp_path):
    source = profile_archive(tmp_path)

    def unused(doc, content):
        doc["owners"].append(
            {"owner_id": "tts.voices", "schema_version": 1, "capabilities": []}
        )
        doc["producer_inventory"].append(
            {
                "logical_id": "profile:source:tts.voices",
                "owner_id": "tts.voices",
                "status": "unused",
                "dependencies": [],
                "shared_group": None,
            }
        )
        doc["exclusions"].append(
            {"logical_id": "profile:source:tts.voices", "reason": "unused"}
        )

    _rewrite_archive(source, unused)
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(source, password=None)
        assert service.wait(op)["state"] == "succeeded"
        assert not service.summary(op)["setup_destination_required"]
    finally:
        service.close()


@pytest.mark.parametrize("kind", ["missing", "public", "overlap", "alias"])
def test_files_needing_setup_parent_must_be_private_and_independent(tmp_path, kind):
    from tldw_chatbook.Backup_Recovery.destinations import check_setup_parent

    protected = tmp_path / "profile"
    protected.mkdir(mode=0o700)
    parent = tmp_path / "setup"
    if kind == "public":
        parent.mkdir(mode=0o755)
    elif kind == "overlap":
        parent = protected
    elif kind == "alias":
        parent.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError):
        check_setup_parent(parent, (protected,))
