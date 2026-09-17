"""Selective native generations preserve inactive Eval files and mutable settings."""

import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery import test_eval_retained_definitions
from Tests.Backup_Recovery.test_restore_data_groups import _document


@pytest.fixture
def completed_replacement(tmp_path_factory, helper_resource_root):
    """Give each mutating lifecycle its own actual completed generation."""
    yield from test_eval_retained_definitions.completed_replacement.__wrapped__(
        tmp_path_factory, helper_resource_root
    )


@pytest.mark.parametrize(
    "missing_canonical",
    [False, True],
    ids=["custom-carry-and-undo", "missing-canonical"],
)
def test_selective_generations_keep_inactive_evals_after_config_edit_and_undo(
    completed_replacement, monkeypatch, missing_canonical, request
):
    """Use actual receipts through two replacements, later rollback, and undo."""
    from keyring.backends.null import Keyring

    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        credentials,
        recovery_copies,
    )
    from tldw_chatbook.Backup_Recovery.activation import activation_permission
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RetainedConfig,
        retained_config_observation,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Evals import _default_config_path
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    root, _, original_context, selected = completed_replacement
    selector = original_context.config_path
    control = root / "control"
    if missing_canonical:
        import tldw_chatbook.Evals as evals

        # Model a missing packaged resource privately, without removing the
        # repository's YAML or replacing discovery/admission with a fake result.
        package = selector.parent / "missing-eval-package"
        package.mkdir(mode=0o700)
        monkeypatch.setattr(evals, "__file__", str(package / "__init__.py"))
        assert not _default_config_path().exists()
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    context = DiscoveryContext(selector, profile)
    config = tomllib.loads(selector.read_text())
    prompt_path = database_path(config, "prompts_db_path")
    missing_path = database_path(config, "scheduled_tasks_db_path")
    prompt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    empty_credentials = KeyringServerCredentialStore(keyring_backend=Keyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: empty_credentials)
    owners = {owner.owner_id: owner for owner in install_adapters()}

    def add_prompt(name):
        store = PromptsDatabase(prompt_path, "eval-selective-retention")
        try:
            store.add_prompt(name, "fixture", "Retention proof", user_prompt=name)
        finally:
            store.close()

    def assert_prompt(name, *, present):
        assert prompt_path.is_file()
        store = PromptsDatabase(prompt_path, "eval-selective-readback")
        try:
            assert (store.get_prompt_by_name(name) is not None) is present
        finally:
            store.close()

    def discover():
        configured = {
            **tomllib.loads(selector.read_text()),
            DISCOVERY_CONTEXT_KEY: context,
        }
        with _preview_reads():
            target = classify_entries(
                tuple(
                    item
                    for owner in (
                        "config",
                        "db.prompts.primary",
                        "db.scheduled_tasks",
                        "eval.definitions",
                    )
                    for item in owners[owner].discover(configured)
                )
            )
        assert target.issues == ("missing_required",)
        assert not target.complete and not missing_path.exists()
        return target

    def observed(path):
        info = path.stat()
        return path.read_bytes(), info.st_dev, info.st_ino

    preserved = {path: observed(path) for path in selected}

    def assert_retained():
        with _preview_reads():
            rows = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
        assert {item.path for item in rows} == {_default_config_path(), *selected}
        assert len({item.logical_id for item in rows}) == len(rows) == 3
        assert all(observed(path) == before for path, before in preserved.items())
        assert not activation_permission("eval.definitions", config_selector=selector)

    add_prompt("saved")
    payloads = {
        "config": selector.read_bytes(),
        "db.prompts.primary": prompt_path.read_bytes(),
    }
    document = _document(tuple(payloads)).model_dump(mode="json")
    for row in document["files"]:
        data = payloads[row["owner_id"]]
        row.update(
            relative_path=selector.name
            if row["owner_id"] == "config"
            else prompt_path.name,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    for row in document["owners"]:
        row["schema_version"] = max(owners[row["owner_id"]].schema_policy().versions)
    source = root / "selective-prompts.zip"
    with zipfile.ZipFile(source, "w") as packed:
        packed.writestr("manifest.json", json.dumps(document))
        for row in document["files"]:
            packed.writestr(row["payload"], payloads[row["owner_id"]])
    archive = acquire(
        source, root / "selective-acquired", ArchiveLimits(), None, Event()
    )
    service = RecoveryService(control)
    request.addfinalizer(service.close)
    inspection = service.start_inspection(archive.path, password=None)
    inspected = service.wait(inspection)
    assert inspected["state"] == "succeeded", dict(inspected)
    add_prompt("first-local")
    assert_retained()

    def replace_prompts():
        target = discover()
        relation = RetainedConfig(
            "profile:profile:config",
            f"profile:{profile}:config",
            selector,
            retained_config_observation(selector),
        )
        plan = service.preview_restore(
            inspection,
            mode="replace",
            target=target,
            destinations={"root:db.prompts.primary": prompt_path.parent},
            retained_configs=(relation,),
            data_groups=("prompts",),
        )
        assert plan.effective_groups == ("prompts",)
        assert {path for _, path in plan.restore} == {prompt_path}
        eval_rows = [item for item in target.items if item.path in selected]
        assert len(eval_rows) == 2
        assert all((item.logical_id, item.path) in plan.preserve for item in eval_rows)
        assert not ({item.logical_id for item in eval_rows} & set(plan.safety_scope))
        unchanged_config = observed(selector)
        completed = service.wait(
            service.start_restore(
                inspection,
                plan,
                rollback_password=b"selective-retention-password",
            )
        )
        assert completed["state"] == "succeeded", dict(completed)
        operation = completed["result"]["journal_operation_id"]
        journal = Journal(control, operation)
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        assert records[-1].event == "committed"
        assert observed(selector) == unchanged_config
        assert not any(
            row["owner_id"] == "eval.definitions"
            for row in json.loads(
                (journal.root / "verified-manifest.json").read_bytes()
            )["files"]
        )
        shutil.rmtree(Path(records[0].evidence["stage"]["path"]))
        assert_retained()
        return operation

    replace_prompts()
    assert_prompt("first-local", present=False)
    if missing_canonical:
        with _preview_reads():
            rows = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
        assert (
            next(item.status for item in rows if item.path == _default_config_path())
            == "missing_required"
        )
        return
    # The actual config writer atomically publishes preferences and advances
    # only this existing binding. Each child exits before native recovery runs.
    def edit_config(section, key, value):
        old_inode = selector.stat().st_ino
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys\n"
                    "from Tests import network_guard; network_guard.install()\n"
                    "from tldw_chatbook.Backup_Recovery import bootstrap\n"
                    "root = Path(sys.argv[1])\n"
                    "bootstrap.default_bootstrap_root = lambda: root\n"
                    "from tldw_chatbook import config\n"
                    "assert config.save_setting_to_cli_config(sys.argv[2], sys.argv[3], sys.argv[4])\n"
                    "assert not network_guard.blocked_attempts()\n"
                ),
                str(bootstrap.default_bootstrap_root()),
                section,
                key,
                value,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert child.returncode == 0, child.stderr[-5000:] + child.stdout[-1000:]
        assert selector.stat().st_ino != old_inode
        assert_retained()

    for theme in ("textual-light", "textual-dark"):
        edit_config("general", "default_theme", theme)
    add_prompt("second-local")
    operation = replace_prompts()
    assert_prompt("second-local", present=False)
    # A legitimate owned config write may point at a different active store,
    # but an old copy must never roll back the now-inactive original location.
    alternate = prompt_path.with_name("alternate-prompts.db")
    shutil.copy2(prompt_path, alternate)
    edit_config("database", "prompts_db_path", str(alternate))
    moved_target = discover()
    assert next(
        item.path for item in moved_target.items if item.owner == "db.prompts.primary"
    ) == alternate
    unchanged_stores = {path: observed(path) for path in (prompt_path, alternate)}
    unchanged_config = observed(selector)
    with pytest.raises(ValueError):
        preview_rollback(
            operation,
            control_root=control,
            old_password=b"selective-retention-password",
            target=moved_target,
            cancel=Event(),
        )
    assert observed(selector) == unchanged_config
    assert {path: observed(path) for path in unchanged_stores} == unchanged_stores
    edit_config("database", "prompts_db_path", str(prompt_path))
    alternate.unlink()
    # Later rollback is a fresh review: an ordinary preference edit after the
    # selected commit must survive without changing any data location setting.
    previous_settings = tomllib.loads(selector.read_text())
    edit_config("general", "default_theme", "textual-light")
    current_settings = tomllib.loads(selector.read_text())
    assert previous_settings["general"].pop("default_theme") == "textual-dark"
    assert current_settings["general"].pop("default_theme") == "textual-light"
    assert current_settings == previous_settings
    source.unlink()
    shutil.rmtree(archive.path.parent)
    service.inspection(inspection).path.unlink()
    assert_retained()

    def rollback(operation):
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"selective-retention-password",
            target=discover(),
            cancel=Event(),
        )
        assert approved.effective_groups == ("prompts",)
        assert all(path not in dict(approved.restore).values() for path in selected)
        unchanged_config = observed(selector)
        result = recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"selective-retention-password",
            new_password=b"selective-retention-password",
            cancel=Event(),
            approved_plan=approved,
        )
        assert observed(selector) == unchanged_config
        assert_retained()
        return result

    undo = rollback(operation)
    assert_prompt("second-local", present=True)
    rollback(undo)
    assert_prompt("second-local", present=False)
    assert_prompt("saved", present=True)
