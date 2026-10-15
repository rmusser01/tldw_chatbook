# transaction_helpers.py (TASK-34905)
#########################################
# Shared flat-store transaction context manager (ADR-177 helper, not a base
# class).
#
# The store-template DBs (AgentRuns, Library_Collections, Dreams,
# automatic_work, Library_Ingest_Jobs) each carried a byte-equivalent
# ``transaction()``: hold the thread's connection, ``BEGIN IMMEDIATE`` so
# the write lock is acquired up front (a deferred read-then-write block can
# fail to upgrade under a concurrent writer), yield, roll back on
# ``BaseException`` (TASK-32801.5: a CancelledError/KeyboardInterrupt is
# not an ``Exception``; skipping rollback left the transaction open and
# every later write on that thread rode it uncommitted), commit on clean
# exit. This helper owns that mechanism once; each store delegates with a
# one-liner.
#
# Deliberately NOT unified here (per the task's scope note):
# - ChaChaNotes' depth-counted TransactionContextManager and Subscriptions'
#   inline depth tracking (borrowing semantics, not flat).
# - Client_Media/Prompts' borrow-style managers (task-22204 legacy-
#   isolation exceptions).
# - Workflows_DB's cursor-yielding write= variant (different contract).
# - Library_Collections' ``read_transaction`` (read-only snapshot with a
#   total_changes write-detection guard) stays store-local; only its
#   nesting-RAISE contract is documented here.
####

import sqlite3
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def flat_store_transaction(
    held_connection, *, immediate: bool = True
) -> Iterator[sqlite3.Connection]:
    """Yield a held connection inside a write transaction.

    Args:
        held_connection: The store's per-thread held connection (already
            live; liveness pings and pragma setup are the store's business).
        immediate: ``True`` (default) issues ``BEGIN IMMEDIATE`` so the
            write lock is acquired up front; ``False`` issues a deferred
            ``BEGIN`` (the Library_Ingest_Jobs flavor, where any lock
            upgrade is resolved by its single-writer admission).

    Yields:
        The connection with a write transaction already started.

    Raises:
        BaseException: Re-raised after rolling back, on any error or
            interruption inside the ``with`` block. On clean exit the
            transaction commits.
    """
    conn = held_connection
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
