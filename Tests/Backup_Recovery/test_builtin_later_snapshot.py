"""Later rollback retains only authenticated finite builtin sources."""

import tomllib
from threading import Event

import pytest

from Tests.Backup_Recovery.test_builtin_safety_directory import (
    builtin_case as _builtin_fixture,
)
from Tests.Backup_Recovery.test_builtin_safety_directory import (
    test_actual_builtin_finite_safety_copy_is_authenticated_without_unselected_tree as _complete_original,
)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.inventory import _sqlite_sidecars
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    Inventory,
    StorageItem,
)
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
from tldw_chatbook.Persona_Visual.recovery import recovery_adapters


def _current(tmp_path, owner_id="persona.visual_identity_builtin"):
    selector = tmp_path / "live/config.toml"
    config = tomllib.loads(selector.read_text())
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(selector, "profile")
    owner = next(a for a in recovery_adapters() if a.owner_id == owner_id)
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    declarations = install_adapters()
    coupled = tuple(
        item
        for adapter in declarations
        if adapter.owner_id
        in {"db.chachanotes.primary", "chat.attachments", "notes.sync_bindings"}
        for item in adapter.discover(config)
    )
    research = StorageItem(
        "research.local",
        "profile:profile:research.local",
        selector.parent / "research.db",
        "included",
        (),
    )
    selected = StorageItem("config", "profile:profile:config", selector, "included", ())
    return Inventory(
        (
            selected,
            research,
            *coupled,
            *_sqlite_sidecars((research, *coupled), declarations),
            *owner.discover(config),
        ),
        True,
        "current-fixture-footprint",
        (),
    )


@pytest.fixture
def complete_builtin_case(tmp_path, monkeypatch, helper_resource_root, request):
    # The earlier finite-source fixture used only a primary SQLite role. This
    # later-operation fixture must capture the actual full coupled core footprint.
    from contextlib import contextmanager

    from Tests.Backup_Recovery import test_held_sqlite_rollback as native_fixture
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    native_case = native_fixture.replacement_case

    @contextmanager
    def coupled_case(path, patch, *, extras, **kwargs):
        def coupled(live):
            extras(live)
            config = {
                "database": {"chachanotes_db_path": str(live / "core.db")},
                DISCOVERY_CONTEXT_KEY: DiscoveryContext(
                    live / "config.toml", "profile"
                ),
            }
            return tuple(
                item
                for adapter in install_adapters()
                if adapter.owner_id
                in {"db.chachanotes.primary", "chat.attachments", "notes.sync_bindings"}
                for item in adapter.discover(config)
            )

        with native_case(path, patch, extras=coupled, **kwargs) as case:
            yield case

    monkeypatch.setattr(native_fixture, "replacement_case", coupled_case)
    yield from _builtin_fixture.__wrapped__(
        tmp_path, monkeypatch, helper_resource_root, request
    )


@pytest.fixture
def later_case(complete_builtin_case, tmp_path, monkeypatch):
    _complete_original(complete_builtin_case, tmp_path, monkeypatch, False)
    selector = tmp_path / "live/config.toml"
    profile = next(
        r
        for r in bootstrap._records(tmp_path / "bootstrap")[1]
        if r["selector"] == str(selector)
    )
    operation = profile["activation"]["operation_id"]
    original = load_plan(Journal(tmp_path / "control", operation))
    current = _current(tmp_path)
    assert [
        i.status for i in current.items if i.owner == "persona.visual_identity_builtin"
    ] == ["unused"]
    return operation, original, current


def test_later_preview_retains_actual_unreferenced_builtin_members(
    later_case, tmp_path
):
    operation, original, current = later_case
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    assert set(reviewed.safety_scope) == set(original.safety_scope)
    assert {
        i.path for i in reviewed.target.items if i.logical_id in reviewed.safety_scope
    } == {
        i.path for i in original.target.items if i.logical_id in original.safety_scope
    }
    assert not set(reviewed.safety_scope).intersection(dict(reviewed.restore))


def _execute_current(
    operation,
    current,
    tmp_path,
    reviewed=None,
    *,
    selected=None,
    owner_id="persona.visual_identity_builtin",
):
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    selected = selected or tmp_path / "live/package-assets/characters/selected.png"
    before = selected.read_bytes(), selected.stat().st_ino
    if reviewed is None:
        reviewed = preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            target=current,
            cancel=Event(),
        )
    from tldw_chatbook.Backup_Recovery import replacement

    try:
        result = execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
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
        reviewed = preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            target=_current(tmp_path, owner_id),
            cancel=Event(),
            acknowledged_credential_issues=review.issues,
        )
        result = execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
        )
    return result


def test_later_execution_restores_old_core_and_preserves_current_files(
    later_case, tmp_path
):
    import sqlite3

    operation, _original, current = later_case
    from contextlib import closing

    selected = tmp_path / "live/package-assets/characters/selected.png"
    unselected = selected.parents[1] / "new-unselected.txt"
    unselected.write_bytes(b"current unrelated package content")
    with closing(sqlite3.connect(tmp_path / "live/research.db")) as db:
        db.execute("UPDATE research_runs SET query='post-replacement edit'")
        db.commit()
    before = selected.read_bytes(), selected.stat().st_ino
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    result = _execute_current(operation, current, tmp_path, reviewed)
    assert result != operation
    assert (selected.read_bytes(), selected.stat().st_ino) == before
    with closing(sqlite3.connect(tmp_path / "live/core.db")) as db:
        assert (
            db.execute("SELECT COUNT(*) FROM visual_identity_assets").fetchone()[0] == 1
        )
    assert unselected.read_bytes() == b"current unrelated package content"
    assert (selected.parents[1] / "unselected").is_symlink()
    import zipfile

    from tldw_chatbook.Backup_Recovery import archive_reader, recovery_copies
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    saved = next(
        row
        for row in recovery_copies.list_recovery_copies(tmp_path / "control")
        if row.operation_id == result
    )
    acquired = archive_reader.acquire(
        saved.path,
        tmp_path / "new-current-copy",
        ArchiveLimits(),
        b"new-test-only",
        Event(),
    )
    document = archive_reader.verify_sealed(acquired)
    payload = next(row for row in document.files if row.owner_id == "research.local")
    copied = tmp_path / "current-research.db"
    with zipfile.ZipFile(acquired.path) as packed:
        copied.write_bytes(packed.read(payload.payload))
    with closing(sqlite3.connect(copied)) as db:
        assert (
            db.execute("SELECT query FROM research_runs").fetchone()[0]
            == "post-replacement edit"
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "symlink",
        "hardlink",
        "wrong_root",
        "conflicting_owner",
        "missing_current_dependency",
    ],
)
def test_later_preview_refuses_unverified_current_builtin_sources(
    later_case, tmp_path, monkeypatch, damage
):
    from dataclasses import replace

    from tldw_chatbook.Persona_Visual.recovery import _Assets

    operation, _original, current = later_case
    selected = tmp_path / "live/package-assets/characters/selected.png"
    if damage == "missing":
        selected.unlink()
    elif damage in {"symlink", "hardlink"}:
        import os

        selected.rename(tmp_path / "foreign")
        if damage == "symlink":
            selected.symlink_to(tmp_path / "foreign")
        else:
            os.link(tmp_path / "foreign", selected)
    elif damage == "wrong_root":
        monkeypatch.setattr(
            _Assets, "_root", lambda self, config: tmp_path / "foreign-root"
        )
    elif damage == "conflicting_owner":
        current = replace(
            current,
            items=(
                *current.items,
                StorageItem("ui.state", "conflict", selected, "included", ()),
            ),
        )
    else:
        current = replace(
            current,
            items=tuple(
                i for i in current.items if i.owner != "db.chachanotes.primary"
            ),
        )
    with pytest.raises((ValueError, bootstrap.RecoveryRequired)):
        preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            target=current,
            cancel=Event(),
        )
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


def test_later_execute_refuses_file_change_after_review(later_case, tmp_path):
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    operation, _original, current = later_case
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    selected = tmp_path / "live/package-assets/characters/selected.png"
    selected.write_bytes(b"new current contents")
    with pytest.raises(ValueError, match="target_changed"):
        execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
        )
    assert selected.read_bytes() == b"new current contents"
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


def test_changed_selected_asset_refuses_old_core_semantics_before_publication(
    later_case, tmp_path
):
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    operation, _original, current = later_case
    selected = tmp_path / "live/package-assets/characters/selected.png"
    selected.write_bytes(b"deliberate current incompatible content")
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    with pytest.raises(ValueError, match="asset_digest_mismatch"):
        execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
        )
    assert selected.read_bytes() == b"deliberate current incompatible content"
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


@pytest.mark.parametrize("omission", ["root", "file"])
def test_later_execute_refuses_removed_authenticated_safety_member(
    later_case, tmp_path, omission
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    operation, _original, current = later_case
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    missing = next(
        i.logical_id
        for i in reviewed.target.items
        if i.logical_id in reviewed.safety_scope
        and (
            i.status == "included"
            if omission == "file"
            else i.metadata.parent_id is None
        )
    )
    reviewed = replace(
        reviewed,
        safety_scope=tuple(key for key in reviewed.safety_scope if key != missing),
    )
    with pytest.raises(ValueError, match="target_changed"):
        execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
        )
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


def test_later_preview_refuses_generation_evidence_change_during_observation(
    later_case, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Persona_Visual.recovery import _Assets

    operation, _original, current = later_case
    witness = bootstrap._records(tmp_path / "bootstrap")[1][0]["activation"]
    store = ActivationStore(witness["store_root"])
    required = store._generation(witness["generation"]) / "required.json"
    retained = required.with_name("required-held.json")
    native = _Assets._tree

    def change(self, *args, **kwargs):
        result = native(self, *args, **kwargs)
        required.rename(retained)
        return result

    monkeypatch.setattr(_Assets, "_tree", change)
    try:
        with pytest.raises((ValueError, bootstrap.RecoveryRequired, FileNotFoundError)):
            preview_rollback(
                operation,
                control_root=tmp_path / "control",
                old_password=b"test-only",
                target=current,
                cancel=Event(),
            )
    finally:
        if retained.exists():
            retained.rename(required)
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


def test_fresh_process_rederives_finite_sources_and_executes_later_rollback(
    later_case, tmp_path, helper_resource_root
):
    import subprocess
    import sys

    operation, _original, _current_inventory = later_case
    code = r"""
import os,sys
from pathlib import Path
os.umask(0o077)
root, helper, operation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
os.environ['TLDW_CONFIG_PATH'] = str(root/'live/config.toml')
import keyring.core
from keyring.backends.null import Keyring
keyring.core._keyring_backend = Keyring()
from tldw_chatbook.Backup_Recovery import bootstrap,crypto
bootstrap.default_bootstrap_root = lambda: root/'bootstrap'
crypto._package_resource_root = lambda: helper
from tldw_chatbook.Persona_Visual.recovery import _Assets
_Assets._root = lambda self, config: root/'live/package-assets'
from Tests.Backup_Recovery.test_builtin_later_snapshot import _current,test_later_execution_restores_old_core_and_preserves_current_files
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
original = load_plan(Journal(root/'control',operation))
test_later_execution_restores_old_core_and_preserves_current_files((operation,original,_current(root)),root)
print('FRESH_NATIVE_LATER_COMPLETE')
"""
    log = tmp_path / "fresh-later.log"
    with log.open("w") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(tmp_path),
                str(helper_resource_root),
                operation,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, log.read_text()[-7000:]
    assert "FRESH_NATIVE_LATER_COMPLETE" in log.read_text()


@pytest.mark.parametrize("damage", ["bytes", "symlink"])
def test_final_private_builtin_validation_refuses_drift_and_alias(
    later_case, tmp_path, monkeypatch, damage
):
    from tldw_chatbook.Backup_Recovery import storage_admission

    operation, _original, current = later_case
    selected = tmp_path / "live/package-assets/characters/selected.png"
    before = selected.read_bytes(), selected.stat().st_ino
    native = storage_admission.copy_capture_file
    reached = []

    def changed(owner, source, destination, cancel, **kwargs):
        native(owner, source, destination, cancel, **kwargs)
        if owner == "persona.visual_identity_builtin" and any(
            part.startswith("installed-check-") for part in destination.parts
        ):
            reached.append(destination)
            if damage == "bytes":
                destination.write_bytes(b"invalid private current asset")
            else:
                destination.unlink()
                destination.symlink_to(selected)

    monkeypatch.setattr(storage_admission, "copy_capture_file", changed)
    with pytest.raises(ValueError, match="replacement_rolled_back"):
        _execute_current(operation, current, tmp_path)
    assert reached
    assert (selected.read_bytes(), selected.stat().st_ino) == before
    assert not bootstrap._records(tmp_path / "bootstrap")[0]


@pytest.mark.parametrize("timing", ["entry", "observation"])
def test_stage_hold_rechecks_actual_paired_generation(
    later_case, tmp_path, monkeypatch, timing
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.later_rollback import execute_rollback

    operation, _original, current = later_case
    reviewed = preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-only",
        target=current,
        cancel=Event(),
    )
    root = tmp_path / "bootstrap"
    association = next(
        path
        for path in root.glob("activation-*.json")
        if not path.name.startswith("activation-update-")
    )
    retained = tmp_path / "held-association.json"
    from tldw_chatbook.Persona_Visual.recovery import _Assets

    native = Admission.maintenance
    native_tree = _Assets._tree
    reached = []
    held = []

    def observed(self, *args, **kwargs):
        result = native_tree(self, *args, **kwargs)
        if held and timing == "observation":
            association.rename(retained)
            reached.append(True)
        return result

    @contextmanager
    def changed(self, *args, **kwargs):
        with native(self, *args, **kwargs) as session:
            held.append(True)
            if timing == "entry":
                association.rename(retained)
                reached.append(True)
            try:
                yield session
            finally:
                held.clear()
                if retained.exists():
                    retained.rename(association)

    monkeypatch.setattr(Admission, "maintenance", changed)
    monkeypatch.setattr(_Assets, "_tree", observed)
    with pytest.raises(
        ValueError, match="local_snapshot_builtin_(scope_unverified|generation_changed)"
    ):
        execute_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            new_password=b"new-test-only",
            cancel=Event(),
            approved_plan=reviewed,
        )
    assert reached
    assert not bootstrap._records(root)[0]


@pytest.mark.parametrize("complete_builtin_case", ["persona.assets"], indirect=True)
@pytest.mark.parametrize("damage", [None, "bytes", "extra_member", "foreign_owner"])
def test_legacy_persona_safety_later_rollback_preserves_exact_owned_tree(
    complete_builtin_case, tmp_path, monkeypatch, damage
):
    import sqlite3
    from contextlib import closing
    from dataclasses import replace

    _complete_original(complete_builtin_case, tmp_path, monkeypatch, False)
    _archive, _plan_for, members, selected = complete_builtin_case
    selector = tmp_path / "live/config.toml"
    profile = next(
        row
        for row in bootstrap._records(tmp_path / "bootstrap")[1]
        if row["selector"] == str(selector)
    )
    operation = profile["activation"]["operation_id"]
    original = load_plan(Journal(tmp_path / "control", operation))
    current = _current(tmp_path, "persona.assets")
    core = tmp_path / "live/core.db"
    with closing(sqlite3.connect(core)) as db:
        assert (
            db.execute("SELECT count(*) FROM persona_visual_assets").fetchone()[0] == 0
        )
    if damage == "bytes":
        selected.write_bytes(b"changed preserved asset")
    elif damage == "extra_member":
        (selected.parent / "unreviewed.png").write_bytes(b"unreviewed")
    elif damage == "foreign_owner":
        current = replace(
            current,
            items=tuple(
                replace(item, owner="persona.visual_identity")
                if item.owner == "persona.assets"
                else item
                for item in current.items
            ),
        )
    before = {
        item.path: (item.path.read_bytes(), item.path.stat().st_ino)
        for item in members
        if item.status == "included"
    }

    def run():
        reviewed = preview_rollback(
            operation,
            control_root=tmp_path / "control",
            old_password=b"test-only",
            target=current,
            cancel=Event(),
        )
        assert set(reviewed.safety_scope) == set(original.safety_scope)
        assert not set(reviewed.safety_scope).intersection(dict(reviewed.restore))
        return _execute_current(
            operation,
            current,
            tmp_path,
            reviewed,
            selected=selected,
            owner_id="persona.assets",
        )

    if damage is None:
        assert run() != operation
        with closing(sqlite3.connect(core)) as db:
            assert (
                db.execute("SELECT count(*) FROM persona_visual_assets").fetchone()[0]
                == 1
            )
    else:
        with pytest.raises(
            ValueError, match="local_snapshot_builtin_|asset_digest_mismatch"
        ):
            run()
        with closing(sqlite3.connect(core)) as db:
            assert (
                db.execute("SELECT count(*) FROM persona_visual_assets").fetchone()[0]
                == 0
            )
    assert {path: (path.read_bytes(), path.stat().st_ino) for path in before} == before
