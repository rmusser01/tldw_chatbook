"""An authenticated empty owner remains part of the selected restore scope."""

import hashlib
import json

import pytest

from Tests.Backup_Recovery.test_restore_data_groups import _document
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.archive_reader import _manifest, verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_groups import (
    available_group_ids,
    resolve_archive_groups,
)

CONFIG = b'[general]\nusers_name="default_user"\n'
WRITING_ID = "profile:profile:writing.local"


def _empty_document(*, version=2, status="unused", groups=("settings", "writing")):
    document = _document(("config",)).model_dump(mode="json")
    document["format_version"] = version
    document["files"][0].update(
        relative_path="config.toml",
        size=len(CONFIG),
        sha256=hashlib.sha256(CONFIG).hexdigest(),
    )
    document["owners"].append(
        {"owner_id": "writing.local", "schema_version": 0, "capabilities": []}
    )
    document["producer_inventory"].append(
        {
            "logical_id": WRITING_ID,
            "owner_id": "writing.local",
            "status": status,
            "dependencies": ["profile:profile:config"],
        }
    )
    document["exclusions"] = [{"logical_id": WRITING_ID, "reason": status}]
    if status in {"missing_required", "unavailable", "unsupported"}:
        document["consistency"] = "partial"
    if version == 2:
        document["group_scope"] = {
            "requested_groups": groups,
            "effective_groups": groups,
            "support_ids": [],
        }
    return document


def _read(document):
    return _manifest(json.dumps(document).encode(), ArchiveLimits(), encrypted=False)


def _archive(tmp_path, **choices):
    document = _empty_document(**choices)
    return sealed(
        tmp_path,
        data=CONFIG,
        mutate=lambda value: (value.clear(), value.update(document)),
    )


@pytest.mark.parametrize("version", [1, 2])
def test_authenticated_unused_owner_is_an_available_group(version):
    doc = _read(_empty_document(version=version))
    assert available_group_ids(doc) == ("settings", "writing")


@pytest.mark.parametrize("selection", [None, ("writing",), ("settings", "writing")])
def test_default_and_explicit_selection_keep_empty_writing(selection):
    doc = _read(_empty_document())
    scope = resolve_archive_groups(doc, selection)
    expected = ("settings", "writing") if selection is None else selection
    assert scope.effective_groups == expected
    assert WRITING_ID in scope.member_ids


def test_unused_owner_keeps_only_its_direct_same_profile_config_support():
    from tldw_chatbook.Backup_Recovery.data_groups import resolve_inventory_groups
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    config = StorageItem("config", "profile:source:config", None, "included", ())
    unused = StorageItem(
        "writing.local",
        "profile:source:writing.local",
        None,
        "unused",
        (config.logical_id, "profile:source:db.media.primary"),
    )
    scope = resolve_inventory_groups((config, unused), ("writing",))
    assert scope.effective_groups == ("writing",)
    assert scope.support_ids == (config.logical_id,)


def test_unused_owner_does_not_borrow_another_profiles_config():
    from tldw_chatbook.Backup_Recovery.data_groups import resolve_inventory_groups
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    config = StorageItem("config", "profile:foreign:config", None, "included", ())
    unused = StorageItem(
        "writing.local",
        "profile:source:writing.local",
        None,
        "unused",
        (config.logical_id,),
    )
    assert not resolve_inventory_groups((config, unused), ("writing",)).support_ids


@pytest.mark.parametrize("owner", [None, "db.media.primary"])
def test_unused_config_dependency_requires_the_actual_config_owner(owner):
    from tldw_chatbook.Backup_Recovery.data_groups import resolve_inventory_groups
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    key = "profile:source:config"
    config = () if owner is None else (StorageItem(owner, key, None, "included", ()),)
    unused = StorageItem(
        "writing.local", "profile:source:writing.local", None, "unused", (key,)
    )
    with pytest.raises(ValueError, match="dependency_unavailable"):
        resolve_inventory_groups((*config, unused), ("writing",))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("status", ["intentionally_excluded", "missing_required"])
def test_excluded_or_missing_owner_does_not_advertise_empty_group(version, status):
    doc = _read(_empty_document(version=version, status=status))
    assert available_group_ids(doc) == ("settings",)
    with pytest.raises(ValueError, match="archive_group_unavailable"):
        resolve_archive_groups(doc, ("writing",))


def test_unused_owner_outside_version_two_scope_stays_unavailable():
    doc = _read(_empty_document(groups=("settings",)))
    assert available_group_ids(doc) == ("settings",)


def test_legacy_archive_without_producer_evidence_does_not_infer_empty_group():
    document = _empty_document(version=1)
    document["producer_inventory"] = []
    assert available_group_ids(_read(document)) == ("settings",)


def test_verified_service_summary_shows_empty_writing_with_zero_payload(tmp_path):
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    archive = _archive(tmp_path)
    service = RecoveryService(tmp_path / "control")
    try:
        inspection = service.start_inspection(archive.path, password=None)
        assert service.wait(inspection)["state"] == "succeeded"
        summary = service.summary(inspection)
        groups = {row["group_id"]: row for row in summary["data_groups"]}
        assert set(groups) == {"settings", "writing"}
        assert groups["writing"]["file_count"] == 0
        assert groups["writing"]["payload_bytes"] == 0
    finally:
        service.close()


def test_empty_sqlite_locator_refusal_explains_current_settings_review():
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_code, issue_message

    code = issue_code(ValueError("selected_absence_locator_changed"))
    assert code == "selected_absence_locator_changed"
    assert "Settings" in issue_message(code)
    assert "review" in issue_message(code).lower()


@pytest.fixture
def target_case(tmp_path):
    """Plan real paths without claiming domain validation or native publication."""
    from tldw_chatbook.Backup_Recovery.models import (
        FileMetadata,
        Inventory,
        StorageItem,
    )

    archive = _archive(tmp_path)
    assert verify_sealed(archive).group_scope.effective_groups == (
        "settings",
        "writing",
    )
    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    current_config = (
        CONFIG
        + (
            "\n[database]\nwriting_db_path = "
            + json.dumps(str(live / "Writing.db"))
            + "\n"
        ).encode()
    )
    profile = hashlib.sha256(str(live / "config.toml").encode()).hexdigest()[:24]
    config_id = "profile:" + profile + ":config"
    items = []
    for owner, filename, content in (
        ("config", "config.toml", current_config),
        ("writing.local", "Writing.db", b"new local writing after backup"),
        ("db.media.primary", "Media.db", b"unselected local media"),
    ):
        path = live / filename
        path.write_bytes(content)
        path.chmod(0o600)
        key = "profile:" + profile + ":" + owner
        items.append(
            StorageItem(
                owner,
                key,
                path,
                "included",
                () if owner == "config" else (config_id,),
                metadata=None
                if owner == "writing.local"
                else FileMetadata(
                    1,
                    key,
                    filename,
                    None,
                    "file",
                    0o600,
                    path.stat().st_mtime_ns,
                    "private",
                ),
            )
        )
    before = {
        row.path: (row.path.read_bytes(), row.path.stat().st_ino) for row in items
    }
    target = Inventory(tuple(items), True, "independent-local-review", ())
    return archive, live, tuple(items), target, before


@pytest.mark.parametrize(
    "selection", [None, ("settings", "writing"), ("settings",), ("writing",)]
)
def test_plan_retires_only_exact_selected_empty_owner(target_case, selection):
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RetainedConfig,
        plan_restore,
        retained_config_observation,
    )

    archive, live, items, target, before = target_case
    retained = (
        (
            RetainedConfig(
                "profile:profile:config",
                items[0].logical_id,
                items[0].path,
                retained_config_observation(items[0].path),
            ),
        )
        if selection == ("writing",)
        else ()
    )
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={"root:config": live},
        target=target,
        data_groups=selection,
        retained_configs=retained,
        profile_names={"profile": "default_user"},
    )
    writing = (items[1].logical_id, items[1].path)
    selected = selection != ("settings",)
    assert plan.retire == ((writing,) if selected else ())
    assert (writing in plan.preserve) is not selected
    assert (items[2].logical_id, items[2].path) in plan.preserve
    assert plan.restore == (
        () if retained else (("profile:profile:config", items[0].path),)
    )
    assert all(
        (path.read_bytes(), path.stat().st_ino) == saved
        for path, saved in before.items()
    )


@pytest.mark.parametrize(
    "change",
    ["no_relation", "no_target_data", "foreign_relation", "missing", "excluded"],
)
def test_empty_only_replacement_requires_exact_nonempty_review(
    target_case, tmp_path, change
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RetainedConfig,
        plan_restore,
        retained_config_observation,
    )

    archive, _, items, target, _ = target_case
    retained = (
        RetainedConfig(
            "profile:profile:config",
            items[0].logical_id,
            items[0].path,
            retained_config_observation(items[0].path),
        ),
    )
    if change == "no_relation":
        retained = ()
    elif change == "no_target_data":
        target = replace(target, items=(items[0], items[2]))
    elif change == "foreign_relation":
        retained = (replace(retained[0], archive_config_id="profile:foreign:config"),)
    else:
        source = tmp_path / "other-archive"
        source.mkdir()
        archive = _archive(
            source,
            status="missing_required"
            if change == "missing"
            else "intentionally_excluded",
        )
    with pytest.raises(ValueError):
        plan_restore(
            archive,
            mode="replace",
            destinations={},
            target=target,
            retained_configs=retained,
            data_groups=("writing",),
        )


@pytest.mark.parametrize("change", ["directory", "different_logical_id", "raw_owner"])
def test_missing_metadata_cannot_authorize_a_tree_another_slot_or_raw_file(
    target_case, change
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.restore_groups import selected_absent_target_ids
    from tldw_chatbook.Backup_Recovery.restore_plan import RetainedConfig

    archive, _, items, target, _ = target_case
    document = verify_sealed(archive)
    writing = items[1]
    group = "writing"
    if change == "directory":
        writing = replace(writing, status="included_directory")
    elif change == "different_logical_id":
        writing = replace(writing, logical_id=writing.logical_id + ":other")
    else:
        group = "prompts"
        owner = "chat.prompt_history"
        raw = _empty_document(groups=("prompts", "settings"))
        raw["owners"][-1]["owner_id"] = owner
        source_id = "profile:profile:" + owner
        raw["producer_inventory"][-1].update(owner_id=owner, logical_id=source_id)
        raw["exclusions"][0]["logical_id"] = source_id
        document = _read(raw)
        writing = replace(
            writing,
            owner=owner,
            logical_id=items[0].logical_id.removesuffix("config") + owner,
        )
    target = replace(target, items=(items[0], writing, items[2]))
    scope = resolve_archive_groups(document, (group,))
    relation = RetainedConfig(
        "profile:profile:config", items[0].logical_id, items[0].path, "0" * 64
    )
    assert not selected_absent_target_ids(document, scope, target, (), (relation,))


@pytest.mark.parametrize("selection", [None, ("writing",)])
@pytest.mark.parametrize("change", ["different_locator", "locator_after_fingerprint"])
def test_empty_sqlite_retirement_requires_current_locator_at_review(
    target_case, monkeypatch, selection, change
):
    from tldw_chatbook.Backup_Recovery import restore_plan as module

    archive, live, items, target, _ = target_case
    config = items[0].path

    def move_locator():
        other = live / "OtherWriting.db"
        other.write_bytes(items[1].path.read_bytes())
        other.chmod(0o600)
        config.write_bytes(
            CONFIG
            + (
                "\n[database]\nwriting_db_path = " + json.dumps(str(other)) + "\n"
            ).encode()
        )

    if change == "different_locator":
        move_locator()
    else:
        original = module._fingerprint

        def fingerprint(paths, inventory):
            result = original(paths, inventory)
            move_locator()
            return result

        monkeypatch.setattr(module, "_fingerprint", fingerprint)
    retained = (
        (
            module.RetainedConfig(
                "profile:profile:config",
                items[0].logical_id,
                config,
                module.retained_config_observation(config),
            ),
        )
        if selection == ("writing",)
        else ()
    )
    with pytest.raises(
        ValueError,
        match="selected_absence_locator_changed|retained_config_changed|target_changed",
    ):
        module.plan_restore(
            archive,
            mode="replace",
            destinations={"root:config": live},
            target=target,
            data_groups=selection,
            retained_configs=retained,
            profile_names={"profile": "default_user"},
        )


@pytest.mark.parametrize("selection", [None, ("writing",)])
@pytest.mark.parametrize("change", ["content", "inode"])
def test_empty_sqlite_config_drift_after_review_is_rejected(
    target_case, selection, change
):
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RetainedConfig,
        plan_restore,
        recheck_targets,
        retained_config_observation,
    )

    archive, live, items, target, _ = target_case
    config = items[0].path
    retained = (
        (
            RetainedConfig(
                "profile:profile:config",
                items[0].logical_id,
                config,
                retained_config_observation(config),
            ),
        )
        if selection == ("writing",)
        else ()
    )
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={"root:config": live},
        target=target,
        data_groups=selection,
        retained_configs=retained,
        profile_names={"profile": "default_user"},
    )
    assert (items[1].logical_id, items[1].path) in plan.retire
    if change == "content":
        config.write_bytes(config.read_bytes() + b"\n# preference edit after review\n")
    else:
        replacement = live / "changed-config.toml"
        replacement.write_bytes(config.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(config)
    with pytest.raises(ValueError, match="retained_config_changed|target_changed"):
        recheck_targets(plan)


_EMPTY_WRITING_SERVICE = r'''
capture_groups = CAPTURE_GROUPS
restore_groups = RESTORE_GROUPS
prompt_process('prepare', 'prepare', 'Unselected saved prompt')
service = RecoveryService(Path.home() / 'recovery-control')
options = {'data_groups': capture_groups}
destination = Path.home() / 'empty-writing.tldw-backup.zip'
try:
    preview = service.preview_backup((source,), options=options)
    assert preview.complete, preview.issues
    empty = next(row for row in preview.items if row.owner == 'writing.local')
    assert empty.status == 'unused'
    succeeded(service, service.start_backup((source,), preview.scope_digest,
                                          destination, options=options, password=None))
    inspection = service.start_inspection(destination, password=None)
    succeeded(service, inspection)
    summary = service.summary(inspection)
    assert {row['group_id'] for row in summary['data_groups']} == set(capture_groups)
    writing = next(row for row in summary['data_groups'] if row['group_id'] == 'writing')
    assert writing['file_count'] == writing['payload_bytes'] == 0
    writer = """
import os, sys, tomllib
from pathlib import Path
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService
path = database_path(tomllib.loads(Path(os.environ['TLDW_CONFIG_PATH']).read_text()), 'writing_db_path')
if sys.argv[1] == 'read':
    assert path.is_file(), 'readback must not create writing data'
store = LocalWritingService(path)
try:
    if sys.argv[1] == 'create':
        store.create_project(title='Writing created after backup')
    else:
        assert any(row['title'] == 'Writing created after backup' for row in store.list_projects())
finally:
    store.close()
"""
    def writing_process(action):
        result = subprocess.run([sys.executable, '-c', writer, action],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr[-4000:]
    writing_process('create')
    writing_process('read')
    target = service.preview_backup((source,), options={})
    current_writing = next(row for row in target.items if row.owner == 'writing.local')
    assert current_writing.status == 'included'
    assert current_writing.metadata is None
    selected = set(capture_groups if restore_groups is None else restore_groups)
    preserved = {row.path: observed(row.path) for row in target.items
                 if row.status == 'included' and row.path is not None
                 and group_for_owner(row.owner) not in selected | {None}}
    assert any(row.owner == 'db.prompts.primary' and row.path in preserved for row in target.items)
    profile = summary['profile_ids'][0]
    plan = service.preview_restore(inspection, mode='replace', profile_bases={},
        target_configs={profile: source}, external_destinations={}, profile_names={},
        target=target, data_groups=restore_groups)
    assert set(plan.effective_groups) == selected
    assert plan.retire == ((current_writing.logical_id, current_writing.path),), plan.retire
    if 'settings' not in selected:
        assert not plan.restore and plan.retained_configs
    result = succeeded(service, service.start_restore(inspection, plan,
        rollback_password=b'empty-writing-safety'))
    assert not current_writing.path.exists()
    assert_preserved()
    operation = result['result']['journal_operation_id']
    current = service.preview_backup((source,), options={})
    rollback = service.preview_rollback(operation, old_password=b'empty-writing-safety', target=current)
    assert current_writing.path in {path for _, path in rollback.restore}
    result = service.wait(service.start_rollback(operation, rollback,
        old_password=b'empty-writing-safety', new_password=b'writing-undo-safety'))
    if result['state'] == 'recovery_required' and result['review_issues']:
        assert 'settings' in selected
        issues = tuple(result['review_issues'])
        assert all(issue.startswith('credential_') for issue in issues)
        pending = result['result']['journal_operation_id']
        succeeded(service, service.start_recovery(pending, action='abort'))
        current = service.preview_backup((source,), options={})
        rollback = service.preview_rollback(operation, old_password=b'empty-writing-safety',
            target=current, acknowledged_credential_issues=issues)
        result = service.wait(service.start_rollback(operation, rollback,
            old_password=b'empty-writing-safety', new_password=b'writing-undo-safety'))
    assert result['state'] == 'succeeded', dict(result)
    writing_process('read')
    assert_preserved()
    read_prompts({'prepare': 'Unselected saved prompt'})
finally:
    service.close()
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "capture_groups,restore_groups",
    [
        (("settings", "writing"), None),
        (("settings", "writing"), ("writing",)),
        (("writing",), ("writing",)),
    ],
)
def test_native_empty_writing_replacement_and_later_rollback(
    tmp_path, capture_groups, restore_groups
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_selective_restore_service import _SELECTED_SERVICE

    prefix = _SELECTED_SERVICE.split("saved_text =", 1)[0].replace(
        "SCENARIO", repr("settings")
    )
    script = prefix + _EMPTY_WRITING_SERVICE.replace(
        "CAPTURE_GROUPS", repr(capture_groups)
    ).replace("RESTORE_GROUPS", repr(restore_groups))
    _run(tmp_path, "empty-writing", "success", script=script, timeout=180)
