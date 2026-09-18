"""Read-only post-exit verification. Usage: verify_profile.py PROFILE."""

import hashlib
import json
import re
import runpy
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "2026-09-16-ingest-lifecycle/native_check.py"
        )
    )["validate_profile"](root)
    evidence = root / "evidence"
    result = json.loads((evidence / "result.json").read_text())
    assert result["passed"] and result["app_run_returned"]
    assert int((root / "exit-status.txt").read_text()) == 0
    assert len(result["steps"]) == 4
    before = json.loads((evidence / "source-before.json").read_text())
    assert before == json.loads((evidence / "source-after.json").read_text())
    integrity = {}
    for path in sorted((root / "data").rglob("*.db")):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            checks = [row[0] for row in db.execute("PRAGMA integrity_check")]
            assert checks == ["ok"], (path, checks)
            integrity[str(path.relative_to(root))] = checks

    def canonical(key, value):
        if value and key in {"created_at", "last_modified", "timestamp"}:
            return datetime.fromisoformat(value).isoformat()
        return value

    def compare(expected, actual):
        assert {k: canonical(k, v) for k, v in expected.items()} == {
            k: canonical(k, actual[k]) for k in expected
        }

    counts = []
    path = root / "data/db/chachanotes.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        identities = [item["conversation"]["id"] for item in before]
        assert sorted(
            row[0] for row in db.execute("SELECT id FROM conversations")
        ) == sorted(identities)
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 10
        for item in before:
            identity = item["conversation"]["id"]
            row = dict(
                db.execute(
                    "SELECT * FROM conversations WHERE id=?", (identity,)
                ).fetchone()
            )
            compare(item["conversation"], row)
            messages = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp,id",
                    (identity,),
                )
            ]
            expected = sorted(
                item["messages"], key=lambda row: (row["timestamp"], row["id"])
            )
            assert len(messages) == len(expected) == 2
            for saved, actual in zip(expected, messages, strict=True):
                compare(saved, actual)
            counts.append(
                {
                    "id": identity,
                    "message_count": len(messages),
                    "active_leaf_message_id": row["active_leaf_message_id"],
                    "version": row["version"],
                }
            )
    workspace_path = root / "data/db/workspaces.db"
    with sqlite3.connect(f"file:{workspace_path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        memberships = [
            dict(row) for row in db.execute("SELECT * FROM workspace_memberships")
        ]
    assert memberships == json.loads((evidence / "foreign-membership.json").read_text())
    fingerprints = json.loads((root / "default-before.json").read_text())
    after = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if Path(name).exists()
        else None
        for name in fingerprints
    }
    assert after == fingerprints
    log = (root / "data/default_user/tldw_cli_app.log").read_text()
    errors = [
        line for line in log.splitlines() if re.search(r"\b(ERROR|CRITICAL)\b", line)
    ]
    tracebacks = log.count("Traceback (most recent call last)")
    fault_bytes = (root / "data/default_user/faulthandler.log").stat().st_size
    assert not errors and not tracebacks and not fault_bytes
    receipt = {
        "profile": str(root),
        "database_integrity": integrity,
        "conversations": counts,
        "no_extra_conversations_or_messages": True,
        "only_original_foreign_membership_remains": True,
        "memberships": memberships,
        "all_original_conversation_and_message_fields_unchanged": True,
        "timestamp_normalization": "ISO datetime equivalence only",
        "default_profile_hashes_unchanged": True,
        "default_profile_files_checked": list(fingerprints),
        "error_lines": errors,
        "traceback_headers": tracebacks,
        "faulthandler_bytes": fault_bytes,
        "sqlite_connection_mode": "ro",
        "sqlite_version": sqlite3.sqlite_version,
    }
    (evidence / "persistence.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "databases": len(integrity),
                "conversations": len(counts),
                "messages": 10,
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
