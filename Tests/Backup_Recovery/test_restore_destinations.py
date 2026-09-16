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
    from tldw_chatbook.Utils.platform_files import os

    protected = tmp_path / "profile"
    protected.mkdir(mode=0o700)
    parent = tmp_path / "setup"
    if kind == "public":
        parent.mkdir(mode=0o755)
        if os.name == "nt":
            import subprocess

            subprocess.run(
                ["icacls", str(parent), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
        else:
            os.chmod(parent, 0o755)
    elif kind == "overlap":
        parent = protected
    elif kind == "alias":
        parent.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError):
        check_setup_parent(parent, (protected,))


@pytest.mark.parametrize(
    "relative", [None, "saved/image.png", "temp/image.png", "saved"]
)
def test_generated_root_derivation_preserves_saved_file_boundary(tmp_path, relative):
    source = profile_archive(tmp_path)
    root = "profile:source:generation.assets"

    def generated(doc, content):
        doc["owners"].append(
            {"owner_id": "generation.assets", "schema_version": 1, "capabilities": []}
        )
        doc["directories"].append(
            {
                "logical_id": root,
                "root_id": root,
                "parent_id": None,
                "relative_path": "",
                "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
                "synthetic": False,
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": root,
                "owner_id": "generation.assets",
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            }
        )
        doc["dependency_groups"][0]["members"].append(root)
        if relative is not None:
            data = b"saved-image-fixture"
            key = root + ":image"
            parent_id = root
            if "/" in relative:
                directory = relative.split("/")[0]
                parent_id = root + ":" + directory
                doc["directories"].append(
                    {
                        "logical_id": parent_id,
                        "root_id": root,
                        "parent_id": root,
                        "relative_path": directory,
                        "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
                        "synthetic": False,
                    }
                )
                doc["producer_inventory"].append(
                    {
                        "logical_id": parent_id,
                        "owner_id": "generation.assets",
                        "status": "included_directory",
                        "dependencies": [],
                        "shared_group": None,
                    }
                )
                doc["dependency_groups"][0]["members"].append(parent_id)
            doc["files"].append(
                {
                    "logical_id": key,
                    "root_id": root,
                    "parent_id": parent_id,
                    "relative_path": relative,
                    "owner_id": "generation.assets",
                    "payload": "payload/generated",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            content["payload/generated"] = data
            doc["producer_inventory"].append(
                {
                    "logical_id": key,
                    "owner_id": "generation.assets",
                    "status": "included",
                    "dependencies": [],
                    "shared_group": None,
                }
            )
            doc["dependency_groups"][0]["members"].append(key)

    _rewrite_archive(source, generated)
    parent = tmp_path / "restore-locations"
    parent.mkdir(mode=0o700)
    base = parent / "new"
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(source, password=None)
        assert service.wait(op)["state"] == "succeeded"
        choices = {
            "mode": "isolated",
            "profile_bases": {"source": base},
            "external_destinations": {},
            "target": None,
            "profile_names": {"source": "New"},
        }
        if relative not in (None, "saved/image.png"):
            with pytest.raises(
                ValueError, match="owner_relocation_unverified:generation.assets"
            ):
                service.preview_restore(op, **choices)
        else:
            plan = service.preview_restore(op, **choices)
            assert (
                dict(plan.destinations)[root]
                == base / "data" / "New" / "generated_images"
            )
        assert not base.exists()
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["isolated", "replace"])
@pytest.mark.parametrize("optional", ["absent", "blank", "sentinel", "custom", "local"])
def test_restore_keeps_unused_optional_selectors_optional(tmp_path, mode, optional):
    import tomllib
    from threading import Event

    import toml

    from Tests.Backup_Recovery.test_file_inventory import VOICE_LOCATION_KEYS
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    source = profile_archive(tmp_path)
    data = {"general": {"users_name": "Original"}}
    if optional != "absent":
        data["database"] = {
            "research_db_path": {
                "blank": "",
                "sentinel": "~/.local/share/tldw_cli/tldw_chatbook_research.db",
                "custom": "/archive/source/research.db",
                "local": "/archive/source/research.db",
            }[optional]
        }
        data["app_tts"] = {
            "CHATTERBOX_VOICE_DIR": "/archive/source/voices"
            if optional in {"custom", "local"}
            else ""
        }
        for location in VOICE_LOCATION_KEYS:
            table = data if len(location) == 1 else data.setdefault(location[0], {})
            table[location[-1]] = (
                "/archive/source/voices"
                if optional in {"custom", "local"}
                else ""
            )

    def config_only(doc, content):
        row = next(row for row in doc["files"] if row["owner_id"] == "config")
        payload = toml.dumps(data).encode()
        row.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        content.clear()
        content[row["payload"]] = payload
        doc["files"] = [row]
        doc["directories"] = [
            d for d in doc["directories"] if d["root_id"] == row["root_id"]
        ]
        ids = {row["logical_id"], row["root_id"]}
        doc["producer_inventory"] = [
            r for r in doc["producer_inventory"] if r["logical_id"] in ids
        ]
        doc["dependency_groups"][0]["members"] = [row["logical_id"]]
        doc["owners"] = [r for r in doc["owners"] if r["owner_id"] == "config"]

    _rewrite_archive(source, config_only)
    parent = tmp_path / "restore-locations"
    parent.mkdir(mode=0o700)
    service = RecoveryService(tmp_path / "control")
    try:
        op = service.start_inspection(source, password=None)
        assert service.wait(op)["state"] == "succeeded"
        choices = {
            "mode": mode,
            "profile_bases": {"source": parent / "new"},
            "external_destinations": {},
            "target": None,
            "profile_names": {"source": "New"},
        }
        if mode == "replace":
            config = parent / "existing" / "config.toml"
            config.parent.mkdir(mode=0o700)
            local = {
                "general": {"users_name": "Existing"},
                "paths": {"data_dir": str(parent / "data")},
            }
            if optional == "local":
                for location in VOICE_LOCATION_KEYS:
                    table = (
                        local if len(location) == 1 else local.setdefault(location[0], {})
                    )
                    table[location[-1]] = str(
                        parent.joinpath("local-voices", *location)
                    )
            config.write_text(toml.dumps(local))
            choices.update(
                profile_bases={},
                target_configs={"source": config},
                target=Inventory(
                    (
                        StorageItem(
                            "config", "profile:local:config", config, "included", ()
                        ),
                    ),
                    True,
                    "local",
                    (),
                ),
            )
        plan = service.preview_restore(op, **choices)
        stage = stage_restore(service.inspection(op), plan, tmp_path / "work", Event())
        descriptor = json.loads((stage / "candidate.json").read_text())
        artifact = next(row for row in descriptor["artifacts"] if row["kind"] == "file")
        restored = tomllib.loads(Path(artifact["candidate"]).read_text())
        restored[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
            dict(plan.restore)["profile:source:config"], "source"
        )
        user_data_dir(restored).mkdir(parents=True, mode=0o700)
        adapters = {a.owner_id: a for a in install_adapters()}
        if optional in {"custom", "local"}:
            assert "/archive/source" not in Path(artifact["candidate"]).read_text()
            for location in VOICE_LOCATION_KEYS:
                table = restored if len(location) == 1 else restored[location[0]]
                expected = (
                    parent.joinpath("local-voices", *location)
                    if mode == "replace" and optional == "local"
                    else user_data_dir(restored).joinpath(*location)
                )
                assert table[location[-1]] == str(expected)
                assert (
                    dict(plan.selectors)["profile:source:" + ".".join(location)]
                    == expected
                )
        else:
            for owner in (
                "notifications.client",
                "research.local",
                "writing.local",
                "tts.profile_store",
            ):
                assert all(
                    row.status in {"unused", "intentionally_excluded"}
                    for row in adapters[owner].discover(restored)
                ), owner
            assert restored.get("database", {}) == data.get("database", {})
            assert restored.get("app_tts", {}) == data.get("app_tts", {})
            for location in VOICE_LOCATION_KEYS:
                old = data if len(location) == 1 else data.get(location[0], {})
                new = restored if len(location) == 1 else restored.get(location[0], {})
                assert new.get(location[-1]) == old.get(location[-1])
                assert "profile:source:" + ".".join(location) not in dict(
                    plan.selectors
                )
    finally:
        service.close()


@pytest.mark.parametrize(
    "location",
    [
        ("global_tts_settings", "CHATTERBOX_VOICE_DIR"),
        ("global_tts_settings", "KOKORO_VOICE_BLENDS_DIR"),
        ("local_chatterbox_default", "CHATTERBOX_VOICE_DIR"),
        ("local_kokoro_default_onnx", "KOKORO_VOICE_BLENDS_DIR"),
        ("local_kokoro_default_pytorch", "KOKORO_VOICE_BLENDS_DIR"),
        ("local_higgs_default", "HIGGS_VOICE_SAMPLES_DIR"),
        ("local_higgs_v2", "HIGGS_VOICE_SAMPLES_DIR"),
        ("HIGGS_VOICE_SAMPLES_DIR",),
    ],
)
def test_voice_selector_shape_rejects_nonstring_paths(location):
    from tldw_chatbook.Backup_Recovery.destinations import _config_tables

    data = {location[-1]: 42}
    if len(location) == 2:
        data = {location[0]: data}
    with pytest.raises(ValueError, match="invalid_config_shape"):
        _config_tables(data)
