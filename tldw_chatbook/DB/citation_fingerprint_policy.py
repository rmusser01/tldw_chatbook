"""Inert native citation-history invariant, shared by provisioning and recovery."""

import sqlite3


def fingerprint_bearing_rows_exist(connection: sqlite3.Connection) -> bool:
    """Check all installed fingerprint history; unknown schemas/errors propagate."""
    row = connection.execute(
        """
SELECT
              EXISTS(SELECT 1 FROM rag_message_trace_owners)
              OR EXISTS(SELECT 1 FROM rag_payload_tombstones)
              OR EXISTS(SELECT 1 FROM rag_legacy_migration_journal)
              OR EXISTS(
                SELECT 1 FROM rag_citation_traces
                WHERE origin = 'imported' OR import_package_fingerprint IS NOT NULL
              )
              OR EXISTS(
                SELECT 1 FROM rag_evidence_snapshots
                WHERE content_hash IS NOT NULL OR comparison_fingerprint IS NOT NULL
              )
        """
    ).fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int or row[0] not in (0, 1):
        raise ValueError("citation_history_unverified")
    return bool(row[0])
