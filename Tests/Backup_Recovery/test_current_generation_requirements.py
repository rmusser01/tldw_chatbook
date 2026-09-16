"""Current requirements follow real replacement history, not isolated catalogs."""

from contextlib import contextmanager
from threading import Event

import pytest

from Tests.Backup_Recovery.test_later_rollback import _completed
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService


@contextmanager
def _healthy_replacement(tmp_path, monkeypatch, helper):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        credentials,
        crypto,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        # Select healthy original bytes before the fresh actual plan. This fixture
        # deliberately leaves the committed config unchanged after replacement.
        import json
        import tomllib

        case[-1].write_text(
            '[general]\nusers_name="old"\n[database]\nresearch_db_path='
            + json.dumps(str(case[-2]))
            + "\n[paths]\ndata_dir="
            + json.dumps(str(case[-1].parent / "data"))
            + "\n"
        )
        archive = archive_reader.acquire(
            tmp_path / "replacement.zip",
            tmp_path / "healthy-input",
            ArchiveLimits(),
            None,
            Event(),
        )
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dict((*case[1].destinations, *case[1].selectors)),
            target=case[1].target,
            profile_names=dict(case[1].profile_names),
        )
        candidate = stage_restore(
            archive, plan, tmp_path / "healthy-candidate", Event()
        )
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"original",
            cancel=Event(),
        )
        installed = tomllib.loads(case[-1].read_text())
        assert installed["database"]["research_db_path"] == str(case[-2])
        assert installed["paths"]["data_dir"] == str(case[-1].parent / "data")
        yield (candidate, plan, *case[2:]), operation, None


def test_actual_current_replacement_reports_persistent_requirements(
    tmp_path, monkeypatch, helper_resource_root
):
    with _healthy_replacement(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        _,
    ):
        service = RecoveryService(tmp_path / "control")
        try:
            from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
            from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

            with acquire_storage(case[-1]) as lease:
                assert _witnesses(case[-1], lease)
            row = service.current_requirements()
            assert row["requirements_checked"] and row["needs_setup"], dict(row)
            assert row["config"] == str(case[-1])
            assert row["operation_id"] == operation
            assert (
                row["required_owners"]
                and row["pending_owners"] == row["required_owners"]
            )
            assert service.profiles() == ()
        finally:
            service.close()


def _fresh_summary(selector, root, control):
    import json
    import os
    import subprocess
    import sys

    script = r"""
import json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
service=RecoveryService(Path(sys.argv[2]))
try:print(json.dumps(dict(service.current_requirements())))
finally:service.close()
assert not blocked_attempts(),blocked_attempts()
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not any(name.startswith(('tldw_chatbook.LLM_Calls.','tldw_chatbook.RAG_Search.')) for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["TLDW_CONFIG_PATH"] = str(selector)
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), str(control)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    return json.loads(result.stdout)


def test_actual_later_generation_supersedes_old_copy_requirements(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore

    with _healthy_replacement(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        original,
        _,
    ):
        service = RecoveryService(tmp_path / "control")
        try:
            before = service.current_requirements()
            # Fixture-only permission isolates the read-only presentation behavior.
            store = ActivationStore(tmp_path / "control" / "activation")
            store.approve(before["generation"], "skills")
            assert "skills" not in service.current_requirements()["pending_owners"]
            assert "config" in service.current_requirements()["pending_owners"]
            import sqlite3
            from contextlib import closing

            with closing(sqlite3.connect(case[-2])) as database:
                database.execute("UPDATE research_runs SET query='new current edit'")
                database.commit()
            old_status = service.status(original)
            plan = service.preview_rollback(
                original, old_password=b"original", target=case[1].target
            )
            job = service.start_rollback(
                original, plan, old_password=b"original", new_password=b"new"
            )
            state = service.wait(job, timeout=40)
            assert state["state"] == "succeeded", dict(state)
            after = service.current_requirements()
            assert after["requirements_checked"] and after["needs_setup"], dict(after)
            assert after["generation"] != before["generation"]
            assert after["operation_id"] == state["result"]["journal_operation_id"]
            assert (
                "skills" in after["pending_owners"]
                and "config" in after["pending_owners"]
            )
            assert service.status(original) == old_status
            assert any(
                row.operation_id == original and row.status == "verified"
                for row in service.recovery_copies()
            )
            child = _fresh_summary(
                case[-1], tmp_path / "bootstrap", tmp_path / "control"
            )
            assert child["generation"] == after["generation"]
            assert child["pending_owners"] == list(after["pending_owners"])
        finally:
            service.close()


def _metadata_bytes(base, selector):
    return {
        str(path): path.read_bytes()
        for path in (
            *base.joinpath("bootstrap").glob("*.json"),
            *base.joinpath("control", "activation").rglob("*.json"),
            selector,
        )
    }


@pytest.mark.parametrize(
    "damage",
    ["missing_pair", "mismatch", "missing_required", "corrupt_required", "during_read"],
)
def test_actual_current_metadata_failures_are_unknown(
    tmp_path, monkeypatch, helper_resource_root, damage
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore

    with _healthy_replacement(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        _,
        _,
    ):
        service = RecoveryService(tmp_path / "control")
        try:
            healthy = service.current_requirements()
            assert healthy["requirements_checked"]
            pair = (
                tmp_path
                / "bootstrap"
                / ("activation-" + bootstrap._key(str(case[-1])) + ".json")
            )
            store = ActivationStore(tmp_path / "control" / "activation")
            required = store._generation(healthy["generation"]) / "required.json"
            if damage == "missing_pair":
                pair.unlink()
            elif damage == "mismatch":
                pair.write_text(
                    pair.read_text().replace(healthy["generation"], "other-generation")
                )
            elif damage == "missing_required":
                required.unlink()
            elif damage == "corrupt_required":
                required.write_bytes(b"{")
            else:
                original = ActivationStore.allowed

                def changed_after_read(self, generation, owner):
                    result = original(self, generation, owner)
                    pair.unlink(missing_ok=True)
                    return result

                monkeypatch.setattr(ActivationStore, "allowed", changed_after_read)
            row = service.current_requirements()
            assert not row["requirements_checked"] and row["needs_setup"] is None, dict(
                row
            )
            assert row["generation"] is None and row["pending_owners"] is None
        finally:
            service.close()


@pytest.mark.parametrize("view", ["current", "missing_pair", "listing_error"])
def test_actual_current_summary_is_read_only_and_mounted_independently(
    tmp_path, monkeypatch, helper_resource_root, view
):
    import asyncio

    from textual.app import App
    from textual.widgets import Static

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    with _healthy_replacement(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        _,
        _,
    ):
        service = RecoveryService(tmp_path / "control")
        try:
            if view == "missing_pair":
                (
                    tmp_path
                    / "bootstrap"
                    / ("activation-" + bootstrap._key(str(case[-1])) + ".json")
                ).unlink()
            elif view == "listing_error":

                def broken_listing():
                    raise ValueError("isolated fixture unavailable")

                monkeypatch.setattr(service, "profiles", broken_listing)
            before = _metadata_bytes(tmp_path, case[-1])
            expected = service.current_requirements()

            class Harness(App):
                def on_mount(self):
                    self.push_screen(BackupRestoreScreen(service))

            async def inspect():
                app = Harness()
                async with app.run_test(size=(100, 40)) as pilot:
                    for _ in range(2):
                        await pilot.click("#backup-open-profiles")
                        await app.workers.wait_for_complete()
                        await pilot.pause()
                        text = str(
                            app.screen.query_one(
                                "#backup-current-requirements", Static
                            ).render()
                        )
                        assert "Current profile" in text and str(case[-1]) in text
                        assert (
                            "Setup requirements unavailable"
                            if view == "missing_pair"
                            else "Needs setup"
                        ) in text
                        assert "Opened successfully" not in text
                        assert not app.screen.query(".backup-open-profile")
                    assert service.current_requirements() == expected

            asyncio.run(inspect())
            assert _metadata_bytes(tmp_path, case[-1]) == before
        finally:
            service.close()


def test_intact_ordinary_current_profile_has_no_invented_generation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    selector = live / "config.toml"
    selector.write_text('[general]\nusers_name="ordinary"\n')
    selector.chmod(0o600)
    root = tmp_path / "bootstrap"
    admission_authority(root).register("ordinary", (live,))
    bind_profile(root, selector, ("ordinary",), root / "admission")
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    service = RecoveryService(tmp_path / "control")
    try:
        row = service.current_requirements()
        assert row["status"] == "no_verified_generation", dict(row)
        assert row["needs_setup"] is None and not row["requirements_checked"]
        assert row["generation"] is None and row["required_owners"] is None
    finally:
        service.close()


def test_edited_unenrolled_current_selector_remains_unknown(
    tmp_path, monkeypatch, helper_resource_root
):
    with _completed(tmp_path, monkeypatch, helper_resource_root):
        service = RecoveryService(tmp_path / "control")
        try:
            row = service.current_requirements()
            assert not row["requirements_checked"] and row["needs_setup"] is None
            assert row["status"] == "requirements_unavailable"
        finally:
            service.close()
