"""Standalone complete backup, isolated from ordinary application startup leases."""


def _capture_workflow_profile(seed, expected):
    """Capture through the installed inventory after ordinary source producers exit."""
    import json
    import os
    import sqlite3
    from contextlib import closing
    from pathlib import Path
    from threading import Event, Timer

    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture

    home = Path.home()
    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    options = {"staging_parent": home}
    from tldw_chatbook.Backup_Recovery import storage_admission

    assert not storage_admission._startups, (
        "capture process entered an ordinary startup lease"
    )
    preview = preview_capture((selector,), options=options)
    assert preview.complete, (
        preview.issues,
        [
            (item.owner, item.status)
            for item in preview.items
            if item.status in ("unsupported", "unavailable", "missing_required")
        ],
    )
    destination = home / (
        "retained-temporary.tldw-backup.zip" if seed else "second.tldw-backup.zip"
    )
    cancel = Event()
    watchdog = Timer(30, cancel.set)
    watchdog.start()
    try:
        captured = capture(
            (selector,),
            preview.scope_digest,
            destination,
            options=options,
            cancel=cancel,
        )
    finally:
        watchdog.cancel()
        watchdog.join()
    manifest = json.loads(captured.manifest_bytes)
    assert captured.inventory.complete and manifest["consistency"] == "coherent"
    version = 30
    if seed:
        with closing(sqlite3.connect(seed)) as source:
            version = source.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
    assert any(
        row["owner_id"] == "db.agent_runs" and row["schema_version"] == version
        for row in manifest["owners"]
    )
    (member,) = (row for row in manifest["files"] if row["owner_id"] == "db.agent_runs")
    candidate = captured.root / member["payload"]
    from Tests.Backup_Recovery.test_workflow_recovery_schema import (
        _payload_witness,
        owner,
    )
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    assert validate_candidate(owner(), candidate, Event(), migrate=False) == ()
    with closing(sqlite3.connect(candidate)) as staged:
        assert _payload_witness(staged) == expected
    write_archive(captured, destination, password=None, cancel=Event())
    assert destination.is_file()
