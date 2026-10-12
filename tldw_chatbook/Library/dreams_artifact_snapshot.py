"""Bounded Library projections over the existing local Dreams owner."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from types import MappingProxyType

from .library_artifacts_state import (
    ArtifactDetail,
    ArtifactKey,
    ArtifactScope,
    ArtifactSourceWindow,
    ArtifactSummary,
    validate_artifact_window,
)


@contextmanager
def dreams_artifact_snapshot(getter):
    """Share one read snapshot; retain a caller's existing transaction."""
    db = getter()
    if db is None:
        yield []
        return
    with db.connection() as conn:
        owned = not conn.in_transaction
        if owned:
            conn.execute("BEGIN")
        try:
            yield [
                (source, DreamsArtifactSnapshot(db, conn, source))
                for source in ("dream_story", "dream_collection", "dream_track")
            ]
        finally:
            if owned:
                conn.rollback()


class DreamsArtifactSnapshot:
    """Metadata and selected-body queries within a caller-owned snapshot."""

    def __init__(self, db, connection, source):
        if source not in ("dream_story", "dream_collection", "dream_track"):
            raise ValueError("Unknown Dreams artifact namespace")
        self.db = db
        self.connection = connection
        self.source = source

    def _projection(self, scope):
        if self.source == "dream_track":
            title = "COALESCE(NULLIF(t.query_template, ''), s.title, 'Tracked page')"
            order = (
                f"LOWER({title})"
                if scope.sort == "title"
                else "COALESCE(-CAST(strftime('%s', t.created_at) AS INTEGER), 9223372036854775807)"
            )
            return (
                f"""SELECT t.id, {title} AS title, t.status AS tracking_status,
                        COALESCE((SELECT status FROM dream_track_runs WHERE tracked_item_id=t.id
                        ORDER BY created_at DESC, id DESC LIMIT 1), t.status) AS status, 0 AS kept,
                        t.created_at, t.updated_at, t.origin_story_id, t.suspension,
                        c.local_date, c.status AS collection_status, c.degradation_notes,
                        (SELECT id FROM dream_track_runs WHERE tracked_item_id=t.id
                         ORDER BY created_at DESC, id DESC LIMIT 1) AS last_run_id,
                        {order} AS order_value FROM dream_tracked_items t
                        LEFT JOIN dream_stories s ON s.id=t.origin_story_id
                        LEFT JOIN dreams_collections c ON c.id=s.collection_id
                        WHERE ? AND instr(LOWER({title}), LOWER(?)) > 0""",
                (
                    int(scope.view in ("dreams", "all") and not scope.kept_only),
                    scope.query,
                ),
            )
        story = self.source == "dream_story"
        title = "s.title" if story else "'Cycle ' || c.local_date"
        order = (
            f"LOWER({title})"
            if scope.sort == "title"
            else "COALESCE(-CAST(strftime('%s', c.created_at) AS INTEGER), 9223372036854775807)"
        )
        identity = "s.id" if story else "c.id"
        status = "s.status" if story else "c.status"
        kept = "s.kept" if story else "0"
        join = "JOIN dream_stories s ON s.collection_id=c.id" if story else ""
        empty = (
            ""
            if story
            else "AND NOT EXISTS (SELECT 1 FROM dream_stories s WHERE s.collection_id=c.id)"
        )
        return (
            f"""SELECT {identity} AS id, {title} AS title, {status} AS status,
                    {kept} AS kept, c.created_at, c.local_date, c.status AS collection_status,
                    c.degradation_notes, {order} AS order_value
                    FROM dreams_collections c {join}
                    WHERE ? AND instr(LOWER({title}), LOWER(?)) > 0 {empty}""",
            (int(scope.view in ("dreams", "all") and not scope.kept_only), scope.query),
        )

    def _summary(self, row):
        row = dict(row)
        key = ArtifactKey(self.source, row["id"])
        revision = hashlib.sha256(
            json.dumps(
                {k: v for k, v in row.items() if k != "order_value"}, sort_keys=True
            ).encode()
        ).hexdigest()
        return ArtifactSummary(
            key,
            (row["order_value"], self.source, key.native_id),
            row["title"],
            "Dreams",
            "Tracked"
            if self.source == "dream_track"
            else "Kept"
            if row["kept"]
            else "Dreams",
            row["status"],
            "Dream story"
            if self.source == "dream_story"
            else "Tracked watch"
            if self.source == "dream_track"
            else "Discovery cycle",
            revision,
            row["created_at"],
        )

    def read_artifact_window(
        self, scope, *, boundary=None, direction="after", limit=20, inclusive=False
    ):
        """Return bounded metadata candidates and exact ranks, never story bodies."""
        validate_artifact_window(scope, boundary, direction, limit, inclusive)
        projection, params = self._projection(scope)
        # Namespace comes only from the closed construction constants above.
        key_sql = f"(order_value, '{self.source}', id)"
        if boundary is None:
            total = self.connection.execute(
                f"WITH metadata AS ({projection}) SELECT COUNT(*) FROM metadata", params
            ).fetchone()[0]
            before, equal = (total if direction == "before" else 0), 0
        else:
            total, before, equal = self.connection.execute(
                f"""WITH metadata AS ({projection}) SELECT COUNT(*),
                COALESCE(SUM({key_sql} < (?, ?, ?)), 0),
                COALESCE(SUM({key_sql} = (?, ?, ?)), 0) FROM metadata""",
                (*params, *boundary, *boundary),
            ).fetchone()
        predicate = ""
        if boundary is not None:
            operator = ">" if direction == "after" else "<"
            if inclusive:
                operator += "="
            predicate = f"WHERE {key_sql} {operator} (?, ?, ?)"
            params = (*params, *boundary)
        ordering = "ASC" if direction == "after" else "DESC"
        rows = self.connection.execute(
            f"""WITH metadata AS ({projection}) SELECT * FROM metadata {predicate}
            ORDER BY order_value {ordering}, id {ordering} LIMIT ?""",
            (*params, limit),
        ).fetchall()
        items = tuple(self._summary(row) for row in rows)
        return ArtifactSourceWindow(
            items if direction == "after" else tuple(reversed(items)),
            total,
            before,
            equal,
        )

    def get_artifact_summary(self, scope, key):
        if key.source != self.source:
            return None
        projection, params = self._projection(scope)
        row = self.connection.execute(
            f"WITH metadata AS ({projection}) SELECT * FROM metadata WHERE id=?",
            (*params, key.native_id),
        ).fetchone()
        return self._summary(row) if row else None

    def read_detail(self, key):
        """Read one selected story and its persisted collection explanation."""
        scope = ArtifactScope(view="dreams")
        summary = self.get_artifact_summary(scope, key)
        if summary is None:
            return None
        projection, params = self._projection(scope)
        row = self.connection.execute(
            f"WITH metadata AS ({projection}) SELECT * FROM metadata WHERE id=?",
            (*params, key.native_id),
        ).fetchone()
        details = [
            ("Collection", row["local_date"]),
            ("Collection status", row["collection_status"]),
        ]
        details = [(label, value) for label, value in details if value is not None]
        if row["degradation_notes"]:
            details.append(("Degradation", row["degradation_notes"]))
        story_id = (
            key.native_id
            if self.source == "dream_story"
            else row["origin_story_id"]
            if self.source == "dream_track"
            else None
        )
        story = self.db.get_story(story_id) if story_id is not None else None
        if self.source == "dream_track":
            details.append(("Tracking", row["tracking_status"]))
            if row["last_run_id"] is not None:
                run = self.connection.execute(
                    "SELECT status, verdict_note FROM dream_track_runs WHERE id=?",
                    (row["last_run_id"],),
                ).fetchone()
                details.extend(
                    (("Last check", run["status"]), ("Verdict", run["verdict_note"]))
                )
            from ..Dreams.dreams_view import _held_reason

            held = _held_reason(dict(row))
            if held:
                details.append(
                    (
                        "Paused",
                        "Page address could not be resolved"
                        if held == "dns_failure"
                        else "Page address is not a public web address",
                    )
                )
        if story is not None:
            story["collection_date"] = row["local_date"]
            story["label"] = summary.title
            story["tracked"] = (
                self.connection.execute(
                    "SELECT 1 FROM dream_tracked_items WHERE origin_story_id=? AND status='active' LIMIT 1",
                    (story_id,),
                ).fetchone()
                is not None
            )
        return ArtifactDetail(
            key,
            summary.revision,
            story["body"] if story else "",
            False,
            False,
            False,
            False,
            False,
            False,
            tuple(details),
            dream_story=MappingProxyType(story) if story else None,
        )
