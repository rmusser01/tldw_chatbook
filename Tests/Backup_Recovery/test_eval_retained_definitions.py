"""Completed local generations retain inactive eval sources across rebackup."""

import os
import subprocess
import sys
from pathlib import Path

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_temporary_media_capture import (
    _PUBLIC,
    _REOPEN,
    _RESTORE,
)


def _replace(source, old, new):
    assert source.count(old) == 1
    return source.replace(old, new)


_RETAIN_RESTORE = _replace(
    _RESTORE,
    "source.rename(source.with_name('source-home-removed'))",
    """import shutil
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.journal import Journal
_, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
witness = next(row['activation'] for row in profiles if row.get('activation'))
journal = Journal(home / 'control', witness['operation_id'])
with journal._locked(exclusive=False) as parent:
    rows = journal._records(parent)
assert rows[-1].event == 'committed'
candidate = Path(rows[0].evidence['stage']['path'])
shutil.rmtree(candidate)
assert not candidate.exists()
assert (journal.root / 'verified-manifest.json').is_file()
assert (journal.root / 'restore-plan.json').is_file()
shutil.rmtree(source)
assert not source.exists()""",
)
_RETAIN_REOPEN = _REOPEN


def test_complete_rebackup_retains_eval_after_source_and_candidate_removal(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _run(original, "temporary", "complete", script=_PUBLIC)
    restored = tmp_path / "restored"
    restored.mkdir()
    _run(restored, str(original / "home"), "isolated", script=_RETAIN_RESTORE)
    environment = os.environ.copy()
    environment.update(
        HOME=str(restored / "home"),
        XDG_CONFIG_HOME=str(restored / "config"),
        XDG_DATA_HOME=str(restored / "data"),
        TLDW_CONFIG_PATH=str(restored / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _RETAIN_REOPEN],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    (tmp_path / "reopen.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "retired and reopened" in result.stdout


import json
from threading import Event

import pytest


@pytest.fixture(scope="module")
def completed_replacement(tmp_path_factory, helper_resource_root):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import crypto, replacement
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
    from tldw_chatbook.Evals import _default_config_path

    root = tmp_path_factory.mktemp("retained-eval-replacement")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
        with replacement_case(
            root,
            patch,
            prepared=False,
            extras=lambda live: (live / "data").mkdir(mode=0o700) or (),
        ) as case:
            candidate, plan, _, _, _, selector = case
            # Deferred owner setup uses a newly selected directory, including in
            # replace mode. The existing source fixture only owns config/research.
            import hashlib
            import zipfile

            from tldw_chatbook.Backup_Recovery.archive_reader import (
                acquire,
                verify_sealed,
            )
            from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
            from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
            from tldw_chatbook.Backup_Recovery.staging import stage_restore

            source = replacement._acquired_source(candidate, plan, Event())
            doc = verify_sealed(source).model_dump(mode="json")
            with zipfile.ZipFile(source.path) as packed:
                payloads = {
                    row["payload"]: packed.read(row["payload"]) for row in doc["files"]
                }
            doc["owners"].append(
                {
                    "owner_id": "eval.definitions",
                    "schema_version": 1,
                    "capabilities": [],
                }
            )
            doc["directories"].append(
                {
                    **doc["directories"][0],
                    "logical_id": "eval-root",
                    "root_id": "eval-root",
                }
            )
            doc["producer_inventory"].append(
                {
                    "logical_id": "eval-root",
                    "owner_id": "eval.definitions",
                    "status": "included_directory",
                    "dependencies": [],
                    "shared_group": None,
                }
            )
            selected = []
            for number in range(2):
                key = f"profile:profile:eval.definitions:item{number}"
                payload = f"payload/eval-{number}"
                data = _default_config_path().read_bytes()
                payloads[payload] = data
                selected.append(
                    selector.parent / "inactive-evals" / f"reviewed-{number}.yaml"
                )
                doc["files"].append(
                    {
                        "logical_id": key,
                        "root_id": "eval-root",
                        "parent_id": "eval-root",
                        "relative_path": f"reviewed-{number}.yaml",
                        "owner_id": "eval.definitions",
                        "payload": payload,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                doc["producer_inventory"].append(
                    {
                        "logical_id": key,
                        "owner_id": "eval.definitions",
                        "status": "included",
                        "dependencies": ["profile:profile:config"],
                        "shared_group": None,
                    }
                )
                doc["dependency_groups"][0]["members"].append(key)
            archive = root / "eval-incoming.zip"
            with zipfile.ZipFile(archive, "w") as packed:
                packed.writestr("manifest.json", json.dumps(doc))
                for name, data in payloads.items():
                    packed.writestr(name, data)
            sealed = acquire(
                archive, root / "eval-input", ArchiveLimits(), None, Event()
            )
            plan = plan_restore(
                sealed,
                mode="replace",
                destinations={
                    **dict(plan.destinations),
                    **dict(plan.selectors),
                    "eval-root": selector.parent / "inactive-evals",
                },
                target=plan.target,
                profile_names=dict(plan.profile_names),
                acknowledged_credential_issues=("credential_format_unreadable",),
            )
            candidate = stage_restore(sealed, plan, root / "eval-candidate", Event())
            operation = replacement.replace(
                plan,
                candidate,
                control_root=root / "control",
                rollback_password=b"test-only-password",
                cancel=Event(),
            )
            journal = Journal(root / "control", operation)
            with journal._locked(exclusive=False) as parent:
                rows = journal._records(parent)
            assert rows[-1].event == "committed"
            import shutil

            shutil.rmtree(candidate)
            yield (
                root,
                journal,
                DiscoveryContext(selector, "destination"),
                tuple(selected),
            )


def test_committed_replacement_preserves_multiple_inactive_eval_sources(
    completed_replacement,
):
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY
    from tldw_chatbook.Evals import _default_config_path
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    _, _, context, selected = completed_replacement
    items = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
    assert {item.path for item in items} == {_default_config_path(), *selected}
    assert len({item.logical_id for item in items}) == 3
    assert all(item.dependencies == ("profile:destination:config",) for item in items)


@pytest.mark.parametrize(
    "change",
    ["manifest", "plan", "association", "nonterminal", "missing", "alias", "unrelated"],
)
def test_retained_mapping_requires_exact_current_local_evidence(
    completed_replacement, change
):
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    root, journal, context, selected = completed_replacement
    config = {DISCOVERY_CONTEXT_KEY: context}
    if change == "unrelated":
        unrelated = selected[0].parent / "eval_config.yaml"
        unrelated.write_bytes(selected[0].read_bytes())
        try:
            assert unrelated not in {
                item.path for item in _DefinitionsAdapter().discover(config)
            }
        finally:
            unrelated.unlink()
        return
    if change in ("missing", "alias"):
        original = selected[0]
        saved = original.with_name("saved-definition")
        original.rename(saved)
        try:
            if change == "alias":
                original.symlink_to(saved)
                with pytest.raises(ValueError):
                    _DefinitionsAdapter().discover(config)
            else:
                items = _DefinitionsAdapter().discover(config)
                assert (
                    next(item for item in items if item.path == original).status
                    == "missing_required"
                )
        finally:
            if original.is_symlink():
                original.unlink()
            saved.rename(original)
        return
    if change == "manifest":
        path = journal.root / "verified-manifest.json"
    elif change == "plan":
        path = journal.root / "restore-plan.json"
    elif change == "association":
        path = next((root / "bootstrap").glob("activation-*.json"))
    else:
        path = max(journal.root.glob("[0-9]*.json"))
    before = path.read_bytes()
    if change == "nonterminal":
        saved = root / "saved-terminal"
        path.rename(saved)
    else:
        path.write_bytes(before + b" " if change in ("manifest", "plan") else b"{}")
    try:
        with pytest.raises((ValueError, OSError)):
            _DefinitionsAdapter().discover(config)
    finally:
        if change == "nonterminal":
            saved.rename(path)
        else:
            path.write_bytes(before)


def test_pending_recovery_refuses_retained_definition_lookup(completed_replacement):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Evals.recovery import _retained_definition_paths

    root, _, context, _ = completed_replacement
    _, profiles, _ = bootstrap._control_records(root / "bootstrap")
    profile = next(
        row for row in profiles if row["selector"] == str(context.config_path)
    )
    operation = "retained-eval-pending-test"
    register_pending(
        root / "bootstrap",
        operation,
        tuple(profile["namespaces"]),
        root / "control",
        (context.config_path,),
    )
    path = root / "bootstrap" / ("pending-" + bootstrap._key(operation) + ".json")
    try:
        with pytest.raises(ValueError, match="recovery_pending"):
            _retained_definition_paths(context)
    finally:
        path.unlink()


def test_ordinary_profile_does_not_create_retained_authority(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Evals import _default_config_path
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    root = tmp_path / "absent-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n")
    unrelated = tmp_path / "eval_config.yaml"
    unrelated.write_bytes(b"jobs: []\n")
    context = DiscoveryContext(selector, "ordinary")
    assert [
        item.path
        for item in _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
    ] == [_default_config_path()]
    assert not root.exists()
