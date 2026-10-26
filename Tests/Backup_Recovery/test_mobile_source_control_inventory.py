"""Native claim controls preserve discovery scope without granting authority."""

import os
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from tldw_chatbook.Backup_Recovery.inventory import discover
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
    _captured_mobile,
    _claim_source,
)
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections


def _prepare(values):
    session, journal, source, _, preview, inventory, _ = values
    session.prepare_mobile_consolidation(
        journal,
        source_config=source,
        target_config=preview.target_selector,
        selections=DiscoverySelections(),
        approved_scope=inventory.scope_digest,
        source_fingerprint=preview.source_fingerprint,
        destination_witnesses=preview.destination_witnesses,
        destination_sidecars=preview.destination_sidecars,
        reviewed_preview=preview,
    )


@pytest.mark.parametrize(
    "change",
    ("safe", "lock-mode", "lock-symlink", "lock-hardlink", "invalid-json", "unknown"),
)
def test_native_alias_control_transition_keeps_only_safe_scope(
    tmp_path, monkeypatch, change
):
    with _capture_fixture(tmp_path, monkeypatch, distinct_selectors=True) as values:
        session, journal, source, _, preview, approved, _ = values
        _prepare(values)
        with _claim_source(session, journal, preview) as claim:
            with session._discovery_reads(private_sqlite=True):
                current = discover((source, preview.target_selector))
            assert current.complete
            assert current.scope_digest == approved.scope_digest
            assert {item.logical_id for item in current.items} == {
                item.logical_id for item in approved.items
            }
            control_rows = [
                item
                for item in current.items
                if item.path in (claim.path, claim.lock_path)
            ]
            assert len(control_rows) == 4
            assert all(item.status == "intentionally_excluded" for item in control_rows)
            claim.check()
            if change == "safe":
                return
            changed = claim.path if change == "invalid-json" else claim.lock_path
            retained = changed.with_name(changed.name + "-test-retained")
            extra = None
            if change == "lock-mode":
                changed.chmod(0o644)
            elif change == "invalid-json":
                original = changed.read_bytes()
                changed.write_text('{"unknown":true}')
            elif change == "unknown":
                extra = claim.path.with_name(".mobile-consolidation-unknown.json")
                extra.write_text("unknown durable control")
                extra.chmod(0o600)
            elif change == "lock-hardlink":
                extra = retained
                os.link(changed, extra)
            else:
                changed.rename(retained)
                changed.symlink_to(retained)
            try:
                with session._discovery_reads(private_sqlite=True):
                    refused = discover((source, preview.target_selector))
                assert not refused.complete
                assert refused.scope_digest != approved.scope_digest
                if change == "unknown":
                    assert any(
                        item.owner == "unknown" and item.path == extra
                        for item in refused.items
                    )
                else:
                    assert all(
                        item.status == "unsupported"
                        for item in refused.items
                        if item.path == changed
                    )
            finally:
                # Restore only this test's fault after observing native refusal.
                if change == "lock-mode":
                    changed.chmod(0o600)
                elif change == "invalid-json":
                    changed.write_bytes(original)
                elif change == "lock-symlink":
                    changed.unlink()
                    retained.rename(changed)
                if extra is not None:
                    extra.unlink()


@pytest.mark.parametrize("member", ("claim", "lock"))
def test_safe_inventory_never_adopts_substituted_native_control(
    tmp_path, monkeypatch, member
):
    with _capture_fixture(tmp_path, monkeypatch, distinct_selectors=True) as values:
        session, journal, _, _, preview, _, _ = values
        _prepare(values)
        with _claim_source(session, journal, preview) as claim:
            changed = claim.path if member == "claim" else claim.lock_path
            original = changed.read_bytes()
            retained = changed.with_name(changed.name + "-test-retained")
            changed.rename(retained)
            changed.write_bytes(original)
            changed.chmod(0o600)
            try:
                with pytest.raises(ValueError, match=f"mobile_source_{member}_changed"):
                    claim.check()
            finally:
                changed.unlink()
                retained.rename(changed)


def test_distinct_alias_capture_refuses_changed_source_payload(tmp_path, monkeypatch):
    with _capture_fixture(tmp_path, monkeypatch, distinct_selectors=True) as values:
        session, journal, _, _, preview, inventory, _ = values
        source = preview.source_dir / "chat.db"
        with source.open("ab") as stream:
            stream.write(b"unreviewed fixture payload")
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            pytest.raises(ValueError, match="mobile_preview_authority_changed"),
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ),
        ):
            pytest.fail("changed source gained capture authority")
        assert not tuple(output.iterdir())


def test_ordinary_shared_owner_change_still_changes_discovery_scope(tmp_path):
    from Tests.Backup_Recovery.test_inventory import config_file, real_db
    from tldw_chatbook.Backup_Recovery.capture import compare_scope

    database = real_db(tmp_path / "shared.db")
    alias = tmp_path / "alias.db"
    os.link(database, alias)
    first = config_file(
        tmp_path, "first", "one", f'[database]\nmedia_db_path = "{database}"\n'
    )
    second = config_file(
        tmp_path, "second", "two", f'[database]\nmedia_db_path = "{alias}"\n'
    )
    before = discover((first, second))
    shared = [item for item in before.items if item.owner == "db.media.primary"]
    assert len(shared) == 2 and shared[0].shared_group == shared[1].shared_group
    assert shared[0].shared_group is not None
    alias.unlink()
    real_db(alias)
    after = discover((first, second))
    assert all(
        item.shared_group is None
        for item in after.items
        if item.owner == "db.media.primary"
    )
    assert compare_scope(before, after) == ("scope_changed",)


def test_native_alias_members_require_every_producer_payload_and_physical_proof(
    tmp_path, monkeypatch
):
    import sys

    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_consolidation,
    )
    from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _MobileCapturePartition,
    )

    with _capture_fixture(tmp_path, monkeypatch, distinct_selectors=True) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        # Capture custody deliberately encloses disposable read custody.
        with _captured_mobile(  # noqa: SIM117
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                partition = next(
                    row
                    for row in session._scopes
                    if type(row) is _MobileCapturePartition
                    and row.purpose == "legacy"
                    and row.claim is captured.claim
                    and row.active
                )
                document = ArchiveManifest.model_validate_json(
                    captured.source.manifest_bytes
                )
                aliases = [
                    row
                    for row in document.producer_inventory
                    if row.owner_id == "db.workspaces" and row.status == "included"
                ]
                assert len(aliases) == 2
                native = mobile_captured_sqlite._members(reads, partition, document)
                representative = next(
                    row[3] for row in native if row[0] == "workspaces"
                )
                assert representative.logical_id == min(
                    row.logical_id for row in aliases
                )
                for alias in aliases:
                    for changes in (
                        {"owner_id": "db.agent_runs"},
                        {"status": "unused"},
                        {"dependencies": ()},
                        {"shared_group": None},
                        {"shared_group": "unproved-group"},
                    ):
                        altered = alias.model_copy(update=changes)
                        document_copy = document.model_copy(
                            update={
                                "producer_inventory": tuple(
                                    altered
                                    if row.logical_id == alias.logical_id
                                    else row
                                    for row in document.producer_inventory
                                )
                            }
                        )
                        with pytest.raises(ValueError, match="inventory_changed"):
                            mobile_captured_sqlite._members(
                                reads, partition, document_copy
                            )
                    missing = document.model_copy(
                        update={
                            "producer_inventory": tuple(
                                row
                                for row in document.producer_inventory
                                if row.logical_id != alias.logical_id
                            )
                        }
                    )
                    with pytest.raises(ValueError, match="inventory_changed"):
                        mobile_captured_sqlite._members(reads, partition, missing)
                    member = next(
                        row
                        for row in document.files
                        if row.logical_id == alias.logical_id
                    )
                    for changes, reason in (
                        ({"owner_id": "db.agent_runs"}, "member_changed"),
                        ({"size": member.size + 1}, "alias_payload_changed"),
                        ({"sha256": "0" * 64}, "alias_payload_changed"),
                    ):
                        altered = member.model_copy(update=changes)
                        changed = document.model_copy(
                            update={
                                "files": tuple(
                                    altered
                                    if row.logical_id == member.logical_id
                                    else row
                                    for row in document.files
                                )
                            }
                        )
                        with pytest.raises(ValueError, match=reason):
                            mobile_captured_sqlite._members(reads, partition, changed)
                    missing = document.model_copy(
                        update={
                            "files": tuple(
                                row
                                for row in document.files
                                if row.logical_id != alias.logical_id
                            )
                        }
                    )
                    with pytest.raises(ValueError, match="member_changed"):
                        mobile_captured_sqlite._members(reads, partition, missing)
                path = preview.source_dir / "workspaces.db"
                retained = path.with_name("test-retained-workspaces.db")
                identity = mobile_consolidation._resource_identity
                observed = []

                def replace_after_first_native_observation(selected):
                    actual = identity(selected)
                    if (
                        selected == path
                        and sys._getframe(1).f_code
                        is mobile_captured_sqlite._members.__code__
                    ):
                        observed.append(actual)
                        if len(observed) == 1:
                            original = path.read_bytes()
                            path.rename(retained)
                            path.write_bytes(original)
                            path.chmod(0o600)
                    return actual

                try:
                    with monkeypatch.context() as fault:
                        fault.setattr(
                            mobile_consolidation,
                            "_resource_identity",
                            replace_after_first_native_observation,
                        )
                        with pytest.raises(ValueError, match="alias_unproven"):
                            mobile_captured_sqlite._members(reads, partition, document)
                    assert len(observed) == 2 and observed[0] != observed[1]
                finally:
                    if retained.exists():
                        path.unlink()
                        retained.rename(path)


def test_distinct_alias_capture_preserves_source_and_canonical_bytes(
    tmp_path, monkeypatch
):
    with _capture_fixture(tmp_path, monkeypatch, distinct_selectors=True) as values:
        session, journal, _, _, preview, inventory, _ = values
        paths = (*preview.capture_scope, *preview.rollback_scope)
        before = {
            path: (path.read_bytes(), path.stat().st_mode) if path.is_file() else None
            for path in paths
        }
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads,
        ):
            assert reads.versions == (
                ("legacy", "chat", 81),
                ("legacy", "workspaces", 8),
                ("legacy", "runs", 25),
                ("canonical", "chat", 82),
                ("canonical", "workspaces", 9),
                ("canonical", "runs", 26),
            )
            columns, rows = reads.page("legacy", "chat", "messages")
            assert "private fixture prompt" in [
                row[columns.index("content")] for row in rows
            ]
        assert before == {
            path: (path.read_bytes(), path.stat().st_mode) if path.is_file() else None
            for path in paths
        }
        assert not tuple(output.glob(".mobile-read-*"))


def test_distinct_alias_capture_preserves_native_optional_runs_absence(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite
    from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _MobileCapturePartition,
    )

    with _capture_fixture(
        tmp_path,
        monkeypatch,
        distinct_selectors=True,
        canonical_runs_absent=True,
        legacy_artifact_root=tmp_path / "profile" / "legacy-artifacts",
    ) as values:
        session, journal, _, stores, preview, inventory, _ = values
        absent = [
            row
            for row in inventory.items
            if row.owner == "db.agent_runs" and row.path == stores.run_path
        ]
        assert len(absent) == 2 and all(row.status == "unused" for row in absent)
        assert not stores.run_path.exists()
        before = {
            path: path.read_bytes()
            for path in (stores.chat_path, stores.workspace_path)
        }
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads,
        ):
            assert ("canonical", "chat", 82) in reads.versions
            assert ("canonical", "workspaces", 9) in reads.versions
            assert not any(row[:2] == ("canonical", "runs") for row in reads.versions)
            with pytest.raises(ValueError, match="member_absent"):
                reads.page("canonical", "runs", "agent_runs")
            partition = next(
                row
                for row in session._scopes
                if type(row) is _MobileCapturePartition
                and row.purpose == "canonical"
                and row.claim is captured.claim
                and row.active
            )
            document = ArchiveManifest.model_validate_json(
                captured.canonical.manifest_bytes
            )
            producers = [
                row
                for row in document.producer_inventory
                if row.logical_id in {item.logical_id for item in absent}
            ]
            assert len(producers) == 2 and all(
                row.status == "unused" for row in producers
            )
            assert not any(
                row.logical_id in {item.logical_id for item in absent}
                for row in document.files
            )
            assert {
                row[0]
                for row in mobile_captured_sqlite._members(reads, partition, document)
            } == {"chat", "workspaces"}
            for producer in producers:
                mixed = document.model_copy(
                    update={
                        "producer_inventory": tuple(
                            row.model_copy(update={"status": "included"})
                            if row.logical_id == producer.logical_id
                            else row
                            for row in document.producer_inventory
                        )
                    }
                )
                with pytest.raises(ValueError, match="inventory_changed"):
                    mobile_captured_sqlite._members(reads, partition, mixed)
        assert not stores.run_path.exists()
        assert before == {path: path.read_bytes() for path in before}
        assert not tuple(output.glob(".mobile-read-*"))


@pytest.mark.parametrize("change", ("required-chat-missing", "optional-runs-residual"))
def test_unused_runs_never_excuses_missing_required_or_residual_state(
    tmp_path, monkeypatch, change
):
    with _capture_fixture(
        tmp_path,
        monkeypatch,
        distinct_selectors=True,
        canonical_runs_absent=True,
        legacy_artifact_root=tmp_path / "profile" / "legacy-artifacts",
    ) as values:
        session, journal, source, stores, preview, inventory, _ = values
        if change == "required-chat-missing":
            stores.chat_path.rename(stores.chat_path.with_name("test-retained-chat.db"))
        else:
            residual = stores.run_path.with_name(stores.run_path.name + "-wal")
            residual.write_bytes(b"retained SQLite state")
            residual.chmod(0o600)
        with session._discovery_reads(private_sqlite=True):
            changed = discover((source, preview.target_selector))
        assert not changed.complete
        owner = (
            "db.chachanotes.primary"
            if change == "required-chat-missing"
            else "db.agent_runs"
        )
        assert any(
            row.owner == owner and row.status in {"missing_required", "unavailable"}
            for row in changed.items
        )
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            pytest.raises(ValueError),
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ),
        ):
            pytest.fail("changed required or residual state must refuse before capture")
        assert not tuple(output.iterdir())


def test_native_retained_default_logs_require_otherwise_optional_runs(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    with prepared(tmp_path, monkeypatch, canonical_runs_absent=True) as values:
        session, journal, _source, stores, preview, inventory, _ = values
        unused = next(
            row
            for row in inventory.items
            if row.owner == "db.agent_runs" and row.path == stores.run_path
        )
        assert unused.status == "unused" and not stores.run_path.exists()
        assert any(
            row.owner == "agents.history"
            and row.status == "missing_required"
            and unused.logical_id in row.dependencies
            for row in inventory.items
        )
        assert not inventory.complete
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            pytest.raises(ValueError),
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ),
        ):
            pytest.fail("retained logs must prevent absent run-store capture")
        assert not tuple(output.iterdir())
