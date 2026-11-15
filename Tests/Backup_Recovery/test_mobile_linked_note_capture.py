"""Actual mobile capture entry for historical linked-note metadata (R133)."""

import hashlib
import json
from pathlib import Path
from threading import Event

from Tests.Backup_Recovery import mobile_consolidation_fixture
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
from Tests.network_guard import blocked_attempts
from Tests.real_profile_guard import take_violations
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _captured_mobile


def test_actual_mobile_capture_accepts_native_linked_note_metadata(
    tmp_path, monkeypatch
):
    seeded = {}

    def seed(stores, *, legacy, **options):
        result = _small_native_seed(
            stores,
            legacy=legacy,
            source_case="linked_note_capture",
            note_root=tmp_path / "external-legacy-notes" if legacy else None,
            **options,
        )
        if legacy:
            seeded.update(result["operator_note"])
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    captured_facts = []
    events = []
    before = {}
    after = {}
    try:
        with _capture_fixture(tmp_path, monkeypatch) as values:
            session, journal, _, _, preview, inventory, _ = values
            # Existing native seed deliberately overlaps IDs; those are remapped,
            # not a persona-name or capture refusal. Source persona was renamed
            # by its real owner before downgrade.
            assert preview.identity_conflicts
            assert preview.has_data and not preview.issues
            assert inventory.complete
            linked = Path(seeded["path"])
            assert linked.parent == tmp_path / "external-legacy-notes"
            assert not linked.is_relative_to(tmp_path / "profile")
            assert hashlib.sha256(linked.read_bytes()).hexdigest() == seeded["sha256"]
            paths = (*preview.capture_scope, linked)
            before = {
                str(path): (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_mode,
                    path.stat().st_ino,
                )
                for path in paths
                if path.is_file()
            }
            destination = tmp_path / "linked-captures"
            destination.mkdir(mode=0o700)
            with _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=b"isolated-linked-note-capture",
                cancel=Event(),
            ) as captured:
                for purpose, archive in (
                    ("legacy", captured.source),
                    ("canonical", captured.canonical),
                ):
                    assert archive.path.is_file()
                    document = json.loads(archive.manifest_bytes)
                    assert any(
                        row["owner_id"] == "db.chachanotes.primary"
                        for row in document["files"]
                    )
                    captured_facts.append(
                        {
                            "purpose": purpose,
                            "manifest_sha256": hashlib.sha256(
                                archive.manifest_bytes
                            ).hexdigest(),
                        }
                    )
            with journal._locked(exclusive=False) as parent:
                events = [row.event for row in journal._records(parent)]
            assert events.count("mobile_capture_verified") == 2
            assert not any(
                event in events
                for event in (
                    "mobile_owner_opened",
                    "mobile_owner_committed",
                    "mobile_committed",
                )
            )
        # The actual maintenance/capture scopes have physically retired here.
        after = {
            path: (
                hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                Path(path).stat().st_mode,
                Path(path).stat().st_ino,
            )
            for path in before
        }
        assert after == before
    finally:
        profile_violations, network_violations = take_violations(), blocked_attempts()
        (tmp_path / "linked-note-capture.json").write_text(
            json.dumps(
                {
                    "seeded_link": seeded,
                    "captures": captured_facts,
                    "events": events,
                    "before": before,
                    "after": after,
                    "profile_violations": profile_violations,
                    "network_violations": network_violations,
                    "scope": "native captures only; no import or App startup qualification",
                },
                indent=2,
            )
        )
        assert not profile_violations and not network_violations
