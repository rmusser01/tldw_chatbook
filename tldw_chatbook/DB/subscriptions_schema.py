"""Installed subscription migrations and exact ordered catalog additions."""

from itertools import islice
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


def listening_brief_items(cursor, brief_id: int) -> tuple[int | None, ...]:
    """Resolve a completed brief's ordered citations without reading article bodies."""
    from tldw_chatbook.Subscriptions.article_listening_models import (
        ListeningConflict,
        MAX_ENTRIES,
        MAX_SESSION_BYTES,
    )
    from tldw_chatbook.Utils.briefing_citations import iter_citation_ids

    brief = cursor.execute(
        "SELECT status,length(CAST(body_markdown AS BLOB)) FROM briefings WHERE id=?",
        (brief_id,),
    ).fetchone()
    if not brief or brief[0] != "complete":
        raise ListeningConflict("listening_brief_unavailable")
    # A saved historical ID alone cannot authorize a lookup in a reused live row.
    members = (
        "SELECT bi.item_id,i.id FROM briefing_items bi LEFT JOIN subscription_items i "
        "ON i.id=bi.live_item_id AND i.id=bi.item_id AND i.subscription_id=bi.source_id "
        "WHERE bi.briefing_id=? "
    )
    modern = cursor.execute(
        "SELECT 1 FROM briefing_items WHERE briefing_id=? AND provenance_version<>1 LIMIT 1",
        (brief_id,),
    ).fetchone()
    if modern:
        ids = tuple(
            row[1]
            for row in cursor.execute(
                members
                + "AND bi.cited=1 ORDER BY bi.citation_position,bi.item_id LIMIT ?",
                (brief_id, MAX_ENTRIES + 1),
            )
        )
        if len(ids) > MAX_ENTRIES:
            raise ValueError("listening_brief_limit")
        return ids
    if (brief[1] or 0) > MAX_SESSION_BYTES:
        raise ValueError("listening_brief_limit")
    body = cursor.execute(
        "SELECT body_markdown FROM briefings WHERE id=?", (brief_id,)
    ).fetchone()[0]
    ids = tuple(islice(iter_citation_ids(body), MAX_ENTRIES + 1))
    if len(ids) > MAX_ENTRIES:
        raise ValueError("listening_brief_limit")
    # Oversize citation integers cannot name a SQLite row and remain placeholders.
    selected = tuple(i for i in ids if 0 < i < 2**63)
    by_id = {}
    if selected:
        marks = ",".join("?" for _ in selected)
        by_id = dict(
            cursor.execute(
                members + f"AND bi.item_id IN ({marks})", (brief_id, *selected)
            )
        )
    return tuple(by_id.get(i) for i in ids)


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
            "length(CAST(incarnation AS BLOB))>128 OR length(CAST(profile AS BLOB))>128 "
            "OR typeof(revision)<>'integer' OR typeof(policy_epoch)<>'integer' "
            "OR typeof(playback_generation)<>'integer' OR typeof(brief_id) NOT IN ('null','integer') "
            "OR length(CAST(transport AS BLOB))>16 OR typeof(cursor) NOT IN ('null','integer') OR typeof(resume_queue)<>'integer' "
            "OR typeof(created_at)<>'integer' OR typeof(updated_at)<>'integer' LIMIT 1"
        ).fetchone():
            return invalid
        if cur.execute(
            "SELECT 1 FROM listening_entries WHERE typeof(slot)<>'integer' OR slot NOT BETWEEN -1 AND ? "
            "OR typeof(session_id)<>'integer' OR typeof(item_id) NOT IN ('null','integer') "
            "OR typeof(source_id) NOT IN ('null','integer') OR typeof(artifact) NOT IN ('null','integer') "
            "OR typeof(bookmark_frame) NOT IN ('null','integer') OR typeof(bookmark_rate) NOT IN ('null','integer') "
            "OR length(CAST(unavailable_reason AS BLOB))>32 OR length(CAST(text AS BLOB))>? "
            "OR length(CAST(title AS BLOB))>2048 OR length(CAST(voice AS BLOB))>8192 "
            "OR length(CAST(entry_id AS BLOB))>128 OR length(CAST(converter AS BLOB))>128 "
            "OR length(CAST(speech_hash AS BLOB))>64 OR length(CAST(voice_hash AS BLOB))>64 OR length(CAST(dependency_hash AS BLOB))>64 "
            "OR length(CAST(bookmark_hash AS BLOB))>64 LIMIT 1",
            (MAX_ENTRIES - 1, MAX_TEXT_BYTES),
        ).fetchone():
            return invalid
        sessions.row_factory = sqlite3.Row
        for s in sessions.execute("SELECT * FROM listening_session"):
            count, total = cur.execute(
                "SELECT count(*),sum(CASE WHEN slot>=0 THEN length(CAST(text AS BLOB)) ELSE 0 END) "
                "FROM listening_entries WHERE session_id=?",
                (s["id"],),
            ).fetchone()
            if count > MAX_ENTRIES + 1 or (total or 0) > MAX_SESSION_BYTES:
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
                if queue:
                    ids = listening_brief_items(cur, s["brief_id"])
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
                    unavailable_reason=e["unavailable_reason"],
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
            "SELECT 1 FROM article_audio WHERE typeof(last_used) NOT IN ('integer','null') "
            "OR typeof(created_at) NOT IN ('integer','null') OR typeof(regenerate) NOT IN ('integer','null') "
            "OR typeof(item_id) NOT IN ('integer','null') OR typeof(sample_rate) NOT IN ('integer','null') "
            "OR typeof(content_frames) NOT IN ('integer','null') OR typeof(file_frames) NOT IN ('integer','null') "
            "OR typeof(byte_size) NOT IN ('integer','null') OR typeof(target_revision) NOT IN ('integer','null') "
            "OR length(CAST(status AS BLOB))>16 OR length(CAST(file_key AS BLOB))<>36 OR "
            "(custody_witness IS NOT NULL AND (typeof(custody_witness)<>'text' "
            "OR length(CAST(custody_witness AS BLOB))<>64 OR length(custody_witness)<>64 "
            "OR custody_witness GLOB '*[^0-9a-f]*')) OR "
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
                    if k
                    not in {
                        "generation",
                        "file_key",
                        "status",
                        "profile",
                        "custody_witness",
                    }
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
