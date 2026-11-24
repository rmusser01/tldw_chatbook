"""Installed subscription migrations and exact ordered catalog additions."""

from pathlib import Path
import re
import sqlite3

_STEPS = {
    2: "subscriptions_v2_to_v3_url_provenance.sql",
    3: "subscriptions_v3_to_v4_article_listening.sql",
}


def statements(script: str):
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending.strip()
            pending = ""
    if pending.strip():
        raise ValueError("incomplete_subscriptions_migration")


def migration(version: int) -> tuple[str, ...]:
    path = Path(__file__).with_name("migrations") / _STEPS[version]
    return tuple(statements(path.read_text(encoding="utf-8")))


def catalog_key(sql: str) -> tuple[str, str]:
    match = re.match(
        r"""CREATE (?:VIRTUAL |UNIQUE )?(TABLE|INDEX|VIEW|TRIGGER) ["'`\[]?([^\s"'`\[(\]]+)""",
        sql,
    )
    if not match:
        raise ValueError("invalid_installed_schema")
    return match[1], match[2]


def listening_catalog(
    schema: tuple[str, ...], additions: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(sorted((*schema, *additions), key=catalog_key))


def listening_issues(connection) -> tuple[str, ...]:
    """Validate bounded v4 values and cross-table ownership after exact catalog match."""
    from dataclasses import asdict

    from tldw_chatbook.Subscriptions.article_listening_models import (
        AudioMetadata,
        EntryCandidate,
        FrozenVoice,
        ListeningAuthority,
        MAX_ENTRIES,
        MAX_SESSION_BYTES,
        MAX_TEXT_BYTES,
        _digest,
        _encoded,
        _hash,
        _id,
        _int,
        _KEY,
    )

    invalid = ("invalid_listening_state",)
    cur = connection.cursor()
    cur.row_factory = sqlite3.Row
    sessions = connection.cursor()
    try:
        # Bound retained values before materializing imported SQLite text in Python.
        if cur.execute(
            "SELECT 1 FROM listening_session WHERE length(CAST(default_voice AS BLOB))>8192 OR "
            "length(CAST(incarnation AS BLOB))>128 OR length(CAST(profile AS BLOB))>128"
        ).fetchone():
            return invalid
        if cur.execute(
            "SELECT 1 FROM listening_entries WHERE slot >= ? OR length(CAST(text AS BLOB))>? "
            "OR length(CAST(title AS BLOB))>2048 OR length(CAST(voice AS BLOB))>8192 "
            "OR length(CAST(entry_id AS BLOB))>128 OR length(CAST(converter AS BLOB))>128 "
            "OR length(CAST(speech_hash AS BLOB))>64 OR length(CAST(voice_hash AS BLOB))>64 OR length(CAST(dependency_hash AS BLOB))>64 "
            "OR length(CAST(bookmark_hash AS BLOB))>64 LIMIT 1",
            (MAX_ENTRIES, MAX_TEXT_BYTES),
        ).fetchone():
            return invalid
        sessions.row_factory = sqlite3.Row
        for s in sessions.execute("SELECT * FROM listening_session"):
            total = (
                cur.execute(
                    "SELECT sum(length(CAST(text AS BLOB))) FROM listening_entries WHERE session_id=? AND slot>=0",
                    (s["id"],),
                ).fetchone()[0]
                or 0
            )
            if total > MAX_SESSION_BYTES:
                return invalid
            _int(s["revision"])
            _int(s["playback_generation"])
            entries = cur.execute(
                "SELECT e.*, i.subscription_id FROM listening_entries e "
                "LEFT JOIN subscription_items i ON i.id=e.item_id WHERE session_id=? ORDER BY slot",
                (s["id"],),
            ).fetchall()
            if s["transport"] == "empty":
                if s["profile"] is not None:
                    ListeningAuthority(s["profile"], "local", s["policy_epoch"])
                elif s["revision"] or s["policy_epoch"] is not None:
                    return invalid
                if (
                    entries
                    or any(
                        s[k] is not None
                        for k in ("default_voice", "brief_id", "cursor")
                    )
                    or s["resume_queue"]
                ):
                    return invalid
            else:
                _id(s["incarnation"])
                ListeningAuthority(s["profile"], "local", s["policy_epoch"])
                FrozenVoice.from_json(s["default_voice"])
                queue = [e for e in entries if e["slot"] >= 0]
                if (
                    not entries
                    or [e["slot"] for e in queue] != list(range(len(queue)))
                    or (
                        bool(queue) != (s["brief_id"] is not None)
                        or (s["cursor"] is None if queue else s["cursor"] is not None)
                        or queue
                        and not 0 <= s["cursor"] < len(queue)
                        or s["resume_queue"]
                        and (not queue or entries[0]["slot"] != -1)
                    )
                ):
                    return invalid
                if (
                    queue
                    and cur.execute(
                        "SELECT 1 FROM briefing_items WHERE briefing_id=? AND provenance_version<>1 LIMIT 1",
                        (s["brief_id"],),
                    ).fetchone()
                ):
                    ids = [
                        r[0]
                        for r in cur.execute(
                            "SELECT live_item_id FROM briefing_items WHERE briefing_id=? "
                            "AND cited=1 ORDER BY citation_position,item_id LIMIT ?",
                            (s["brief_id"], MAX_ENTRIES + 1),
                        )
                    ]
                    if len(queue) != len(ids) or any(
                        e["item_id"] is not None and e["item_id"] != i
                        for e, i in zip(queue, ids, strict=False)
                    ):
                        return invalid
            for e in entries:
                voice = (
                    FrozenVoice.from_json(e["voice"])
                    if e["voice"] is not None
                    else None
                )
                entry = EntryCandidate(
                    e["item_id"],
                    e["source_id"],
                    e["title"],
                    e["text"],
                    e["converter"],
                    voice,
                )
                _id(e["entry_id"])
                if e["item_id"] is not None:
                    if (
                        e["subscription_id"] != e["source_id"]
                        or e["speech_hash"] != _hash(entry.text)
                        or (
                            e["voice_hash"] != _hash(_encoded(asdict(voice)))
                            or e["dependency_hash"]
                            != _hash(_encoded(asdict(voice)["dependencies"]))
                        )
                    ):
                        return invalid
                elif any(
                    e[k] is not None
                    for k in (
                        "speech_hash",
                        "voice_hash",
                        "dependency_hash",
                        "artifact",
                    )
                ):
                    return invalid
                if e["artifact"] is None:
                    if any(
                        e[k] is not None
                        for k in ("bookmark_hash", "bookmark_rate", "bookmark_frame")
                    ):
                        return invalid
                elif not cur.execute(
                    "SELECT 1 FROM article_audio WHERE generation=? AND profile=? AND "
                    "(item_id,speech_hash,converter,voice_hash,dependency_hash)=(?,?,?,?,?) "
                    "AND status IN ('ready','missing') AND sha256=? AND sample_rate=? AND ? BETWEEN 0 AND content_frames",
                    (
                        e["artifact"],
                        s["profile"],
                        e["item_id"],
                        e["speech_hash"],
                        e["converter"],
                        e["voice_hash"],
                        e["dependency_hash"],
                        e["bookmark_hash"],
                        e["bookmark_rate"],
                        e["bookmark_frame"],
                    ),
                ).fetchone():
                    return invalid
        if cur.execute(
            "SELECT 1 FROM article_audio WHERE length(CAST(file_key AS BLOB))<>36 OR "
            "length(CAST(profile AS BLOB))>128 OR length(CAST(converter AS BLOB))>128 OR "
            "length(CAST(target_entry AS BLOB))>128 OR length(CAST(sha256 AS BLOB))>64 OR "
            "length(CAST(speech_hash AS BLOB))>64 OR length(CAST(voice_hash AS BLOB))>64 OR "
            "length(CAST(dependency_hash AS BLOB))>64 LIMIT 1"
        ).fetchone():
            return invalid
        for r in cur.execute("SELECT * FROM article_audio"):
            _int(r["generation"], 1)
            if not _KEY.fullmatch(r["file_key"]):
                return invalid
            ListeningAuthority(r["profile"], "local", 0)
            if r["status"] == "pending_delete":
                if any(
                    r[k] is not None
                    for k in r.keys()
                    if k not in {"generation", "file_key", "status", "profile"}
                ):
                    return invalid
                continue
            _id(r["converter"])
            _int(r["item_id"], 1)
            for name in ("speech_hash", "voice_hash", "dependency_hash"):
                _digest(r[name])
            if r["status"] == "pending":
                _id(r["target_entry"])
                _int(r["target_revision"])
                if r["regenerate"] not in (0, 1) or any(
                    r[k] is not None
                    for k in (
                        "sha256",
                        "sample_rate",
                        "content_frames",
                        "file_frames",
                        "byte_size",
                    )
                ):
                    return invalid
            else:
                if any(
                    r[k] is not None
                    for k in ("target_entry", "target_revision", "regenerate")
                ):
                    return invalid
                AudioMetadata(
                    r["sha256"],
                    r["sample_rate"],
                    r["content_frames"],
                    r["file_frames"],
                    r["byte_size"],
                )
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return invalid
    finally:
        sessions.close()
        cur.close()
    return ()
