"""A fresh connection's ``PRAGMA journal_mode`` statement must be finished.

``PRAGMA journal_mode`` is a writer statement to SQLite. Left active (an
unfetched cursor kept alive), every COMMIT on that connection fails with
"cannot commit transaction - SQL statements in progress" and BEGIN IMMEDIATE
becomes a busy-handler-free read-to-write upgrade. The connection setup used to
rely on the cursor being garbage collected at once; anything that retains the
frame that ran it (a debugger, a held traceback, a stack sampler) kept the
statement active and refused real Console Sends. Retain those frames here and
require that a write transaction on the fresh connection still commits.
"""

from __future__ import annotations

import sys
import threading

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _commit_on_fresh_thread_retaining_frames(open_and_write):
    retained, outcome = [], {}

    def profile(frame, event, arg):
        if event == "return":
            retained.append(frame)  # keep every returned frame's locals alive

    def worker():
        sys.setprofile(profile)
        try:
            open_and_write()
            outcome["ok"] = True
        except Exception as error:  # noqa: BLE001 - reported to the main thread
            outcome["error"] = error
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(60)
    assert not thread.is_alive()
    assert retained, "profile hook observed no returned frames"
    assert outcome.get("ok"), outcome.get("error")


def test_chachanotes_fresh_connection_commits_with_retained_setup_frames(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "c.db"), client_id="test-client")

    def open_and_write():
        try:
            # A new thread opens a new thread-local connection (runs the PRAGMA).
            with db.transaction(immediate=True) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS _pragma_probe(x INTEGER)")
                conn.execute("INSERT INTO _pragma_probe(x) VALUES (1)")
        finally:
            db.close_connection()

    try:
        _commit_on_fresh_thread_retaining_frames(open_and_write)
    finally:
        db.close_connection()
