"""Collections/Tags keyword handlers: off-loop DB calls and lookup dedupe.

task-15471 coverage. Two properties are pinned here:

1. **The handlers actually work.** Every handler in
   `collections_tag_events.py` used to call ``app.run_in_thread`` -- a method
   neither Textual 8.x's ``App`` nor ``TldwCli`` defines -- so rename/merge/
   delete all raised ``AttributeError`` straight into their own error toast.
   The success-path assertions below fail against that code.

2. **Delete resolves each keyword name exactly once.** The old delete
   handler ran the same ``SELECT`` twice per keyword id (once for the
   notification, once again before the delete), synchronously on the event
   loop. The batch lookup must issue one SELECT per id.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Event_Handlers.collections_tag_events import (
    KeywordDeleteEvent,
    KeywordRenameEvent,
    handle_keyword_delete,
    handle_keyword_rename,
)


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeMediaDB:
    # False on purpose: exercises the real asyncio.to_thread path.
    is_memory_db = False

    def __init__(self):
        self.keywords = {1: "alpha", 2: "beta"}
        self.select_calls = 0
        self.deleted: list[str] = []
        self.renames: list[tuple[int, str]] = []

    def execute_query(self, query, params):
        assert "SELECT keyword FROM Keywords" in query
        self.select_calls += 1
        keyword = self.keywords.get(params[0])
        return _FakeCursor({"keyword": keyword} if keyword else None)

    def soft_delete_keyword(self, keyword):
        self.deleted.append(keyword)
        return True

    def rename_keyword(self, keyword_id, new_name):
        self.renames.append((keyword_id, new_name))
        return True


class _FakeApp:
    """Duck-typed TldwCli stand-in: media_db + notify recorder.

    Deliberately has no ``query_one`` -- the handlers' window-refresh blocks
    swallow that, and no ``run_in_thread`` -- the real app never had one
    either, which is exactly what property 1 above pins.
    """

    def __init__(self, media_db):
        self.media_db = media_db
        self.notifications: list[tuple[str, str]] = []

    def notify(self, message, severity="information"):
        self.notifications.append((str(message), severity))


@pytest.mark.asyncio
async def test_keyword_delete_resolves_each_name_once_and_deletes_off_loop():
    db = _FakeMediaDB()
    app = _FakeApp(db)

    await handle_keyword_delete(app, KeywordDeleteEvent([1, 2, 999]))

    # One SELECT per requested id -- not two (the pre-task-15471 shape).
    assert db.select_calls == 3
    assert db.deleted == ["alpha", "beta"]

    severities = {severity for _message, severity in app.notifications}
    assert "error" not in severities
    success = [m for m, s in app.notifications if s == "information"]
    assert success and "alpha" in success[0] and "beta" in success[0]
    # id 999 resolved to nothing, so one warning about the shortfall.
    assert any(s == "warning" for _m, s in app.notifications)


@pytest.mark.asyncio
async def test_keyword_rename_succeeds_without_run_in_thread():
    db = _FakeMediaDB()
    app = _FakeApp(db)

    await handle_keyword_rename(app, KeywordRenameEvent(1, "gamma"))

    assert db.renames == [(1, "gamma")]
    assert app.notifications, "rename must confirm"
    message, severity = app.notifications[0]
    assert severity == "information" and "gamma" in message
