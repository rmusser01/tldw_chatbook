# Console Multiple Attachments per Message (TASK-217) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Console messages carry up to 5 attachments — staged append-wise in the composer, persisted in a new `message_attachments` table (positions ≥ 1; legacy columns keep #0), sent as multiple image parts within an image-counting budget, and rendered as one chip per attachment.

**Architecture:** v18→v19 migration adds a bare association table matching the `conversation_keywords` precedent. The in-memory model is list-first (`ConsoleChatMessage.attachments` tuple) with the Phase-1 scalar fields auto-mirrored from `attachments[0]` through a single store helper, so every existing reader keeps working. The persistence adapter owns the split addressing (legacy columns = position 0, table = positions ≥ 1) in one transaction; reads batch-fetch. UI layers migrate to the list where they need N.

**Tech Stack:** Python ≥3.11, SQLite (existing migration framework), Textual 8.2.7, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-13-console-multi-attachment-design.md` — read first; decisions settled.

## Global Constraints

- Run all tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q --no-header` from the worktree root (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-multi-attach-217`).
- Modify ONLY files each task names. NEVER touch files outside the worktree. Never use git stash.
- Existing tests read-only; additions append-only at true EOF (verify `git diff HEAD -- <testfile> | grep -c '^-[^-]'` = 0). Exception: none.
- Exact values (verbatim from spec): staging cap **5** (`MAX_PENDING_ATTACHMENTS = 5`, named constant); cap toast `"Attachment limit reached (5 per message)."`; count label `"📎 {N} files"`; truncated multi-drop toast `"Attached first {n} of {m} dropped files."`; Save-all toast `"Saved {N} images to {directory}"`; table DDL exactly as in the spec (composite PK `(message_id, position)`, `CHECK (position >= 1)`); version bump `UPDATE db_schema_version SET version = 19 WHERE schema_name = 'rag_char_chat_schema' AND version = 18;`.
- Mirror invariant: `image_data`/`image_mime_type`/`attachment_label` scalars ALWAYS equal `attachments[0]`'s fields (or None/empty tuple together); all mutation flows through the store's `_set_message_attachments`.
- Attachments update rule: `None` = don't touch (extends the #621 omit-when-None fix); attachment rows change only when explicitly provided.
- No sync triggers on `message_attachments` (deliberate — TASK-220); no raw bytes in screen-state serialization (labels only).
- Legacy chat untouched; legacy image regression gate green unedited.
- CI intentionally cancelled remotely — verify locally. End commits with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: DB — v19 migration + attachment accessors

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (`_CURRENT_SCHEMA_VERSION` at 142; new `_MIGRATE_V18_TO_V19_SQL` class attr beside `_MIGRATE_V17_TO_V18_SQL`; new `_migrate_from_v18_to_v19` beside `_migrate_from_v17_to_v18` (~3091); migration_steps dict entry at ~3204; new public methods near `get_message_by_id`)
- Create: `tldw_chatbook/DB/migrations/chachanotes_v18_to_v19_message_attachments.sql` (documentation mirror, matching sibling files' header style)
- Test: `Tests/ChaChaNotesDB/test_chachanotes_db.py` (append-only)

**Interfaces:**
- Produces (Tasks 3, 5 consume):
  - `set_message_attachments(conn_or_none, message_id: str, rows: list[dict]) -> None` — public method `set_message_attachments(message_id, rows)` executing DELETE + INSERTs inside `self.transaction()`; each row dict: `{"position": int (>=1), "data": bytes, "mime_type": str, "display_name": str}`.
  - `get_attachments_for_messages(message_ids: Sequence[str]) -> dict[str, list[dict]]` — single `SELECT ... WHERE message_id IN (...)` (chunk the IN list at 500 ids), rows ordered by position, each `{"position", "data", "mime_type", "display_name"}`.

- [ ] **Step 1: Write the failing tests** (append at true EOF of `Tests/ChaChaNotesDB/test_chachanotes_db.py`; follow the file's fixture idiom for constructing a `CharactersRAGDB` on a tmp path — read the first existing test's setup and mirror it)

```python
def _make_conversation_with_message(db):
    conv_id = db.add_conversation({"title": "att", "client_id": db.client_id})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "hello",
            "client_id": db.client_id,
        }
    )
    return conv_id, msg_id


class TestMessageAttachmentsTable:
    def test_schema_v19_creates_empty_attachments_table(self, db_instance):
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_attachments'"
            )
            assert cursor.fetchone() is not None
            cursor.execute("SELECT COUNT(*) FROM message_attachments")
            assert cursor.fetchone()[0] == 0

    def test_set_and_batch_get_attachments(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        rows = [
            {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "a.png"},
            {"position": 2, "data": b"img-2", "mime_type": "image/jpeg", "display_name": "b.jpg"},
        ]
        db_instance.set_message_attachments(msg_id, rows)

        fetched = db_instance.get_attachments_for_messages([msg_id])
        assert list(fetched.keys()) == [msg_id]
        assert [r["position"] for r in fetched[msg_id]] == [1, 2]
        assert fetched[msg_id][0]["data"] == b"img-1"
        assert fetched[msg_id][1]["display_name"] == "b.jpg"

        # Replace semantics: a second set replaces, not appends.
        db_instance.set_message_attachments(
            msg_id,
            [{"position": 1, "data": b"img-3", "mime_type": "image/png", "display_name": "c.png"}],
        )
        fetched = db_instance.get_attachments_for_messages([msg_id])
        assert [r["display_name"] for r in fetched[msg_id]] == ["c.png"]

    def test_position_zero_rejected(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        import sqlite3 as _sqlite3

        import pytest as _pytest

        with _pytest.raises((ValueError, _sqlite3.IntegrityError, Exception)):
            db_instance.set_message_attachments(
                msg_id,
                [{"position": 0, "data": b"x", "mime_type": "image/png", "display_name": "z.png"}],
            )

    def test_hard_delete_cascades_attachments(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        db_instance.set_message_attachments(
            msg_id,
            [{"position": 1, "data": b"img", "mime_type": "image/png", "display_name": "a.png"}],
        )
        with db_instance.transaction() as cursor:
            cursor.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
            cursor.execute(
                "SELECT COUNT(*) FROM message_attachments WHERE message_id = ?", (msg_id,)
            )
            assert cursor.fetchone()[0] == 0

    def test_get_attachments_empty_and_unknown_ids(self, db_instance):
        assert db_instance.get_attachments_for_messages([]) == {}
        assert db_instance.get_attachments_for_messages(["nope"]) == {}
```

(`db_instance` — use the file's existing fixture name for a fresh DB; read the file first and adapt the fixture name/idiom, disclosing if it differs.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ChaChaNotesDB/test_chachanotes_db.py -q --no-header -k "MessageAttachments"`
Expected: FAIL — table missing / methods missing.

- [ ] **Step 3: Implement**

1. `_CURRENT_SCHEMA_VERSION = 19` (update the comment: `# Adds message_attachments (positions >= 1; position 0 stays in messages.image_data).`)
2. Class attr (beside `_MIGRATE_V17_TO_V18_SQL`):

```python
    _MIGRATE_V18_TO_V19_SQL = """
CREATE TABLE IF NOT EXISTS message_attachments(
  message_id   TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position     INTEGER NOT NULL CHECK (position >= 1),
  data         BLOB    NOT NULL,
  mime_type    TEXT    NOT NULL,
  display_name TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (message_id, position)
);
CREATE INDEX IF NOT EXISTS idx_message_attachments_message ON message_attachments(message_id);
UPDATE db_schema_version
   SET version = 19
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 18;
"""
```

3. `_migrate_from_v18_to_v19(self, conn)` — copy `_migrate_from_v17_to_v18`'s exact structure (docstring, executescript, version check expects 19, error wrapping).
4. migration_steps dict: add `18: self._migrate_from_v18_to_v19,`.
5. Documentation mirror SQL file with the sibling files' comment header ("-- Migration: ChaChaNotes V18 to V19 message attachments ..." + note about positions >= 1 and no sync triggers, citing TASK-220).
6. Public methods (Google docstrings per repo style):

```python
    def set_message_attachments(self, message_id: str, rows: list[dict]) -> None:
        """Replace the extra attachments (positions >= 1) for a message.

        Position 0 lives in ``messages.image_data``/``image_mime_type``; this
        table only holds positions >= 1. Runs DELETE + INSERT in one
        transaction.

        Args:
            message_id: Target message UUID.
            rows: Dicts with ``position`` (>= 1), ``data``, ``mime_type``,
                ``display_name``.

        Raises:
            ValueError: If any row has position < 1.
            CharactersRAGDBError: On database errors.
        """
        for row in rows:
            if int(row.get("position", 0)) < 1:
                raise ValueError("message_attachments positions start at 1.")
        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM message_attachments WHERE message_id = ?", (message_id,)
            )
            cursor.executemany(
                "INSERT INTO message_attachments (message_id, position, data, mime_type, display_name)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        message_id,
                        int(row["position"]),
                        row["data"],
                        row["mime_type"],
                        row.get("display_name", ""),
                    )
                    for row in rows
                ],
            )

    def get_attachments_for_messages(
        self, message_ids: "Sequence[str]"
    ) -> dict[str, list[dict]]:
        """Batch-fetch extra attachments (positions >= 1) for messages.

        Args:
            message_ids: Message UUIDs to fetch for.

        Returns:
            Mapping of message_id to position-ordered attachment row dicts
            (``position``, ``data``, ``mime_type``, ``display_name``); ids
            with no rows are absent.
        """
        ids = [str(m) for m in message_ids if m]
        if not ids:
            return {}
        result: dict[str, list[dict]] = {}
        with self.transaction() as cursor:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    "SELECT message_id, position, data, mime_type, display_name"
                    f" FROM message_attachments WHERE message_id IN ({placeholders})"
                    " ORDER BY message_id, position",
                    chunk,
                )
                for row in cursor.fetchall():
                    result.setdefault(row["message_id"], []).append(
                        {
                            "position": row["position"],
                            "data": row["data"],
                            "mime_type": row["mime_type"],
                            "display_name": row["display_name"],
                        }
                    )
        return result
```

(`Sequence` — add to the module's typing imports if absent. Row access by name assumes the connection's row_factory — verify how other methods read rows (dict-style vs index) and match.)

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ChaChaNotesDB/test_chachanotes_db.py -q --no-header`
Expected: all pass (new class + all pre-existing — the migration chain must not break fresh-create; the whole file exercises it).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v18_to_v19_message_attachments.sql Tests/ChaChaNotesDB/test_chachanotes_db.py
git commit -m "feat(db): message_attachments table (v19) with batch accessors

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Model + store — attachments tuple, mirror invariant, pending list

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (`ConsoleChatMessage` fields), `tldw_chatbook/Chat/console_chat_store.py` (session field 280-330 region, `append_message` ~374, new `_set_message_attachments` helper)
- Test: `Tests/Chat/test_console_chat_store.py` (append-only)

**Interfaces:**
- Produces (Tasks 3–6 consume):
  - `@dataclass(frozen=True) MessageAttachment(data: bytes | None, mime_type: str, display_name: str, position: int)` in `console_chat_models.py`.
  - `ConsoleChatMessage.attachments: tuple[MessageAttachment, ...] = ()`.
  - `MAX_PENDING_ATTACHMENTS = 5` (store module constant).
  - Store: `add_pending_attachment(session_id, attachment) -> bool` (False + no-op when at cap), `pending_attachments(session_id) -> list[PendingAttachment]`, `clear_pending_attachments(session_id)`; legacy `pending_attachment()` returns first-or-None; legacy `set_pending_attachment()` = clear+add; legacy `clear_pending_attachment()` aliases clear-all.
  - `append_message(..., attachments: Sequence[MessageAttachment] = (), image_data=None, image_mime_type=None, attachment_label=None)` — scalar kwargs converted to a one-element tuple when `attachments` empty; store helper `_set_message_attachments(message, attachments)` enforces the mirror (scalars = attachments[0] or all-None).

- [ ] **Step 1: Write the failing tests** (append at true EOF)

```python
from tldw_chatbook.Chat.console_chat_models import MessageAttachment
from tldw_chatbook.Chat.console_chat_store import MAX_PENDING_ATTACHMENTS


def _att(name="a.png", data=b"img", position=1):
    return MessageAttachment(
        data=data, mime_type="image/png", display_name=name, position=position
    )


def test_append_message_with_attachments_mirrors_first_into_scalars():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="look",
        attachments=(
            _att("a.png", b"img-1", 0),
            _att("b.jpg", b"img-2", 1),
        ),
    )
    assert len(message.attachments) == 2
    assert message.image_data == b"img-1"
    assert message.image_mime_type == "image/png"
    assert message.attachment_label and "a.png" in message.attachment_label


def test_append_message_scalar_kwargs_become_single_attachment():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"img",
        image_mime_type="image/png",
        attachment_label="pic.png · 3 B",
    )
    assert len(message.attachments) == 1
    assert message.attachments[0].data == b"img"
    assert message.image_data == b"img"


def test_pending_list_appends_caps_and_clears():
    store = ConsoleChatStore()
    session = store.ensure_session()
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    def _pending(name):
        return PendingAttachment(
            file_path=f"/tmp/{name}", display_name=name, file_type="image",
            insert_mode="attachment", data=b"x", mime_type="image/png",
            original_size=1, processed_size=1,
        )

    for index in range(MAX_PENDING_ATTACHMENTS):
        assert store.add_pending_attachment(session.id, _pending(f"f{index}.png")) is True
    assert store.add_pending_attachment(session.id, _pending("overflow.png")) is False
    assert len(store.pending_attachments(session.id)) == MAX_PENDING_ATTACHMENTS

    # Legacy single accessors still work over the list.
    assert store.pending_attachment(session.id).display_name == "f0.png"
    store.clear_pending_attachments(session.id)
    assert store.pending_attachments(session.id) == []
    assert store.pending_attachment(session.id) is None


def test_legacy_set_pending_attachment_replaces_all():
    store = ConsoleChatStore()
    session = store.ensure_session()
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    def _pending(name):
        return PendingAttachment(
            file_path=f"/tmp/{name}", display_name=name, file_type="image",
            insert_mode="attachment", data=b"x", mime_type="image/png",
            original_size=1, processed_size=1,
        )

    store.add_pending_attachment(session.id, _pending("a.png"))
    store.add_pending_attachment(session.id, _pending("b.png"))
    store.set_pending_attachment(session.id, _pending("only.png"))
    names = [p.display_name for p in store.pending_attachments(session.id)]
    assert names == ["only.png"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -q --no-header`
Expected: new tests FAIL (imports/attrs missing); pre-existing pass.

- [ ] **Step 3: Implement**

`console_chat_models.py` (near `ConsoleChatMessage`):

```python
@dataclass(frozen=True)
class MessageAttachment:
    """One attachment carried by a Console message (position 0 = legacy slot)."""

    data: bytes | None
    mime_type: str
    display_name: str
    position: int
```

`ConsoleChatMessage` gains `attachments: tuple["MessageAttachment", ...] = ()` (after `attachment_label`).

`console_chat_store.py`:
1. `MAX_PENDING_ATTACHMENTS = 5` near the top constants; import `MessageAttachment`; import `Sequence` if absent.
2. `ConsoleChatSession.pending_attachment: PendingAttachment | None` → `pending_attachments: list[PendingAttachment] = field(default_factory=list)`.
3. Replace/extend the pending accessors (keeping the three legacy names working):

```python
    def pending_attachments(self, session_id: str) -> list[PendingAttachment]:
        """Return the staged attachments for a session (stage order)."""
        return list(self._session_or_raise(session_id).pending_attachments)

    def add_pending_attachment(
        self, session_id: str, attachment: PendingAttachment
    ) -> bool:
        """Append a staged attachment; False (no-op) when at the cap.

        Args:
            session_id: Native Console session ID.
            attachment: Processed attachment to stage.

        Returns:
            True when staged; False when MAX_PENDING_ATTACHMENTS reached.

        Raises:
            KeyError: If the session is unknown.
        """
        session = self._session_or_raise(session_id)
        if len(session.pending_attachments) >= MAX_PENDING_ATTACHMENTS:
            return False
        session.pending_attachments.append(attachment)
        return True

    def clear_pending_attachments(self, session_id: str) -> ConsoleChatSession:
        """Remove all staged attachments from a session."""
        session = self._session_or_raise(session_id)
        session.pending_attachments.clear()
        return session

    def pending_attachment(self, session_id: str) -> PendingAttachment | None:
        """Return the first staged attachment (legacy single accessor)."""
        pending = self._session_or_raise(session_id).pending_attachments
        return pending[0] if pending else None

    def set_pending_attachment(
        self, session_id: str, attachment: PendingAttachment
    ) -> ConsoleChatSession:
        """Replace all staged attachments with one (legacy semantics)."""
        session = self._session_or_raise(session_id)
        session.pending_attachments[:] = [attachment]
        return session

    def clear_pending_attachment(self, session_id: str) -> ConsoleChatSession:
        """Alias of clear_pending_attachments (legacy name)."""
        return self.clear_pending_attachments(session_id)
```

4. Mirror helper + `append_message`:

```python
    @staticmethod
    def _set_message_attachments(
        message: ConsoleChatMessage,
        attachments: Sequence[MessageAttachment],
    ) -> None:
        """Set a message's attachments tuple and mirror #0 into the scalars.

        Every attachments mutation MUST flow through here — the scalar
        image fields are a read-compatibility mirror of attachments[0].
        Positions are re-based sequentially from 0 in the given order.
        """
        rebased = tuple(
            replace(attachment, position=index)
            for index, attachment in enumerate(attachments)
        )
        message.attachments = rebased
        first = rebased[0] if rebased else None
        message.image_data = first.data if first else None
        message.image_mime_type = first.mime_type if first else None
        message.attachment_label = (
            first.display_name if first and first.display_name else None
        )
```

Mirror-label rule (state it in the task report): the persisted/chip label is the display NAME (sizes only exist at stage time — the composer indicator keeps using `PendingAttachment.label`'s name · size). The chip helper already falls back to mime · size when bytes exist, so both label shapes stay honest.

```python
    def append_message(
        self,
        session_id: str,
        *,
        role: ConsoleMessageRole,
        content: str,
        persist: bool = False,
        attachments: Sequence[MessageAttachment] = (),
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
        attachment_label: str | None = None,
    ) -> ConsoleChatMessage:
        """Append a message; scalar image kwargs become a one-item tuple."""
        self._session_or_raise(session_id)
        effective = tuple(attachments)
        if not effective and image_data is not None:
            effective = (
                MessageAttachment(
                    data=image_data,
                    mime_type=image_mime_type or "image/png",
                    display_name=attachment_label or "",
                    position=0,
                ),
            )
        message = ConsoleChatMessage(
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
        )
        self._set_message_attachments(message, effective)
        if attachment_label and effective and not effective[0].display_name:
            message.attachment_label = attachment_label
        self._messages_by_session[session_id].append(message)
        self._sessions[session_id].updated_at = _utc_now_iso()
        self._message_session_index[message.id] = session_id
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        return self._snapshot(message)
```

(Also: normalize positions — `_set_message_attachments` re-numbers `position` sequentially from 0 by stage order regardless of input positions, via `dataclasses.replace(att, position=i)`. Add that; the model's position field then always reflects reality.)
5. Persist-defer condition (from #621): update `_persist_new_message_or_defer`'s check to `if not message.content and not message.attachments:`.

- [ ] **Step 4: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py -q --no-header`
Expected: all pass (controller suite proves the scalar mirror keeps Phase-1 flows green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): list-first message attachments with mirrored scalars and capped pending list

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Persistence — split-addressed write + batch read

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (`create_message` ~253, `update_message_content` ~213), `tldw_chatbook/Chat/console_chat_store.py` (`_persist_new_message` ~648, `_persist_existing_message` ~667, protocol at top)
- Test: `Tests/Chat/test_chat_persistence_service.py` (append-only; file exists from the #621 pre-merge work), `Tests/Chat/test_console_chat_store.py` (append-only)

**Interfaces:**
- Consumes: Task 1 DB methods; Task 2 model.
- Produces (Task 5 consumes): service `create_message(..., attachments: Sequence[Mapping] | None = None)` / `update_message_content(..., attachments: ... | None = None)` where each mapping is `{"data", "mime_type", "display_name", "position"}` covering ALL positions (0..N-1): the service writes position 0 into the legacy columns and positions ≥ 1 via `set_message_attachments` (positions ≥ 1 re-based as given), inside its existing transaction flow; `None` = attachments untouched (legacy image kwargs keep working when attachments is None). Service `get_attachments_for_messages(message_ids)` passthrough to the DB method.

- [ ] **Step 1: Write the failing tests** (append; use the recording-fake-db idiom the file established in the #621 pre-merge tests — read it first)

```python
def test_create_message_splits_position_zero_and_rest(real_db_service):
    service, db = real_db_service  # adapt to the file's fixture idiom
    conv_id = service.create_conversation(
        assistant_kind="generic", assistant_id="console",
        conversation_title="t", workspace_id=None, scope_type="global",
    )
    attachments = [
        {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
        {"position": 1, "data": b"img-1", "mime_type": "image/jpeg", "display_name": "b.jpg"},
        {"position": 2, "data": b"img-2", "mime_type": "image/png", "display_name": "c.png"},
    ]
    msg_id = service.create_message(
        conversation_id=conv_id, sender="user", content="multi",
        image_data=None, image_mime_type=None, attachments=attachments,
    )
    row = db.get_message_by_id(msg_id)
    assert row["image_data"] == b"img-0"
    assert row["image_mime_type"] == "image/png"
    extra = db.get_attachments_for_messages([msg_id])[msg_id]
    assert [r["position"] for r in extra] == [1, 2]
    assert extra[0]["data"] == b"img-1"


def test_update_without_attachments_leaves_table_and_columns_alone(real_db_service):
    service, db = real_db_service
    conv_id = service.create_conversation(
        assistant_kind="generic", assistant_id="console",
        conversation_title="t", workspace_id=None, scope_type="global",
    )
    msg_id = service.create_message(
        conversation_id=conv_id, sender="user", content="multi",
        image_data=None, image_mime_type=None,
        attachments=[
            {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
            {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
        ],
    )
    service.update_message_content(
        message_id=msg_id, content="edited",
        image_data=None, image_mime_type=None,
    )
    row = db.get_message_by_id(msg_id)
    assert row["content"] == "edited"
    assert row["image_data"] == b"img-0"
    assert db.get_attachments_for_messages([msg_id])[msg_id][0]["data"] == b"img-1"
```

Store-level (append to test_console_chat_store.py): extend `RecordingPersistence` usage — a new recording fake accepting `attachments` kwargs; assert `_persist_new_message` passes the full attachments mapping list (positions 0..N-1, data/mime/name from the tuple) and `_persist_existing_message` passes `attachments=None` on plain edits.

```python
def test_persist_new_message_sends_full_attachment_list():
    class RecordingAttachmentPersistence(RecordingPersistence):
        pass  # create_message already records kwargs

    persistence = RecordingAttachmentPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="multi",
        attachments=(
            _att("a.png", b"img-0"),
            _att("b.png", b"img-1"),
        ),
        persist=True,
    )
    sent = persistence.created[-1]["attachments"]
    assert [a["position"] for a in sent] == [0, 1]
    assert sent[0]["data"] == b"img-0"
    assert sent[1]["display_name"] == "b.png"


def test_persist_edit_leaves_attachments_none():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="x",
        attachments=(_att("a.png", b"img"),), persist=True,
    )
    store.update_message_content(message.id, "edited")
    assert persistence.updated[-1]["attachments"] is None
```

(RecordingPersistence gains `attachments=None` params on both methods — it's a test fake defined in this test file; extending it is a sanctioned append-adjacent edit ONLY if it was defined in an appended region we own — it was (Phase 1). Since existing tests assert on `.created[...]` keys unrelated to attachments, adding the kwarg with a default is compatible; if the fake lives in the read-only region, subclass instead and disclose.)

- [ ] **Step 2: Run to verify RED**, as usual, with `-k "splits_position or leaves_table or full_attachment_list or attachments_none"`.

- [ ] **Step 3: Implement**

`chat_persistence_service.py`:
- `create_message(..., attachments: "Sequence[Mapping[str, Any]] | None" = None)`: when `attachments` is not None, derive `image_data`/`image_mime_type` from the position-0 entry (overriding the scalar kwargs) and, after the `add_message` call returns the id, call `self.db.set_message_attachments(message_id, [rows with position >= 1])` (empty list fine — writes nothing but clears stale rows on retry paths).
- `update_message_content(..., attachments=None)`: when None → current behavior exactly (the #621/#628-era omit rules). When provided → set legacy columns from position 0 (include image keys in the update payload) and `set_message_attachments(message_id, rows >= 1)`.
- `get_attachments_for_messages(self, message_ids)` → `return self.db.get_attachments_for_messages(message_ids)`.

`console_chat_store.py`:
- Protocol: add `attachments` kwarg to both methods (default None) + `get_attachments_for_messages` (make it optional via `getattr` at call sites in Task 5 instead — protocol gains it, fakes may omit; document).
- `_persist_new_message`: build `attachments=[{"position": a.position, "data": a.data, "mime_type": a.mime_type, "display_name": a.display_name} for a in message.attachments if a.data is not None]` when `message.attachments` else None; drop the scalar image kwargs when sending attachments (service derives them).
- `_persist_existing_message`: pass `attachments=None` always (edits never change attachments; scalar image kwargs continue to carry the #0 mirror for the pre-existing preserve semantics).

- [ ] **Step 4: Run** the two Chat test files + `Tests/ChaChaNotesDB/test_chachanotes_db.py -k MessageAttachments`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): persist multi-attachments — legacy columns hold #0, table holds the rest

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Controller — image-counting payload budget + send staging

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`submit_draft` pending block, `_provider_message_payloads` ~773), `tldw_chatbook/Chat/attachment_core.py` (add `image_url_part`)
- Test: `Tests/Chat/test_console_chat_controller.py` (append-only), `Tests/Chat/test_attachment_core.py` (append-only)

**Interfaces:**
- Consumes: Task 2 model/store.
- Produces: `attachment_core.image_url_part(image_data: bytes, mime_type: str) -> dict` (single image part; `image_content_parts` reimplemented over it); payload builder counts IMAGES newest-message-first: per message include images in position order until the remaining budget is exhausted; over-budget images fall back per-message to text-only inclusion of remaining text (text parts always included once per message); dataless attachments skipped. `submit_draft` stages ALL pendings as `MessageAttachment` tuple (positions re-based by `_set_message_attachments`) and calls `clear_pending_attachments`.

- [ ] **Step 1: Failing tests** (append)

`test_attachment_core.py`:

```python
def test_image_url_part_and_content_parts_agree():
    from tldw_chatbook.Chat.attachment_core import image_content_parts, image_url_part

    part = image_url_part(b"\x89PNG", "image/png")
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    combined = image_content_parts("hi", b"\x89PNG", "image/png")
    assert combined[-1] == part
```

`test_console_chat_controller.py`:

```python
def test_submit_stages_all_pendings_and_clears(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="vision-model")
    session = store.ensure_session()
    store.add_pending_attachment(session.id, _pending_image("a.png"))
    store.add_pending_attachment(session.id, _pending_image("b.png"))

    result = asyncio.run(controller.submit_draft("two pics"))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    image_parts = [p for p in user_payload["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2
    assert store.pending_attachments(session.id) == []
    messages = store.messages_for_session(session.id)
    user_message = [m for m in messages if m.role is ConsoleMessageRole.USER][-1]
    assert len(user_message.attachments) == 2
    assert user_message.image_data is not None  # mirror holds


def test_image_budget_counts_images_newest_first(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 3)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway, model="vision-model")
    session = store.ensure_session()
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    def _atts(n, tag):
        return tuple(
            MessageAttachment(data=f"{tag}-{i}".encode(), mime_type="image/png",
                              display_name=f"{tag}{i}.png", position=i)
            for i in range(n)
        )

    store.append_message(session.id, role=ConsoleMessageRole.USER,
                         content="older", attachments=_atts(2, "old"))
    store.append_message(session.id, role=ConsoleMessageRole.USER,
                         content="newer", attachments=_atts(2, "new"))

    asyncio.run(controller.submit_draft("go"))

    user_payloads = [m for m in gateway.messages_seen if m["role"] == "user"]
    # newest ("newer") gets both images; "older" gets 1 (budget 3), oldest first-dropped.
    newer = user_payloads[1]
    older = user_payloads[0]
    newer_images = [p for p in newer["content"] if p["type"] == "image_url"] if isinstance(newer["content"], list) else []
    older_images = [p for p in older["content"] if p["type"] == "image_url"] if isinstance(older["content"], list) else []
    assert len(newer_images) == 2
    assert len(older_images) == 1
```

(Budget partial rule: when a message is partially budgeted, include its NEWEST-position images first? Spec says "include what fits" walking images in position order — resolve as: per message, include images in position order until budget runs out; messages walked newest-first for budget RESERVATION, but emission order stays chronological. For "older" with budget 1 left: include position 0. Assert `older_images[0]["image_url"]["url"]` decodes from `old-0`. Implementer: verify the assertion matches the implementation rule and disclose.)

- [ ] **Step 2: RED** with `-k "image_url_part or stages_all or counts_images"`.

- [ ] **Step 3: Implement**

`attachment_core.py`:

```python
def image_url_part(image_data: bytes, mime_type: str) -> dict[str, Any]:
    """Build one OpenAI-style image_url content part (base64 data URL).

    Args:
        image_data: Raw image bytes.
        mime_type: MIME type for the data URL.

    Returns:
        A single ``{"type": "image_url", ...}`` content part dict.
    """
    encoded = base64.b64encode(image_data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
```

and `image_content_parts` body becomes `parts.append(image_url_part(image_data, mime_type))` for the image part (behavior identical; test asserts agreement).

`console_chat_controller.py`:
- Import `image_url_part` and `MessageAttachment` as needed.
- `submit_draft` pending block: `pendings = self.store.pending_attachments(session.id)`; image gate if any pending has `insert_mode == "attachment"` and data; build `attachments=tuple(MessageAttachment(data=p.data, mime_type=p.mime_type or "image/png", display_name=p.display_name, position=i) for i, p in enumerate(attachment_mode_pendings))`; `append_message(..., attachments=attachments)`; `clear_pending_attachments`.
- `_provider_message_payloads`: replace the message-id budget with per-image reservation:

```python
        # Reserve the image budget newest-message-first, counting IMAGES.
        budget = max_history_images(self.provider, model) if vision else 0
        allowed_counts: dict[str, int] = {}
        for message in reversed(session_messages):
            if budget <= 0:
                break
            if message.role is not ConsoleMessageRole.USER:
                continue
            usable = [a for a in message.attachments if a.data is not None]
            if not usable:
                continue
            take = min(len(usable), budget)
            allowed_counts[message.id] = take
            budget -= take
```

and in the emission loop, for a message with `allowed_counts.get(message.id, 0) > 0`: parts = optional text part + `image_url_part` for the first `take` usable attachments in position order; else existing text fallback.

- [ ] **Step 4: Run** controller + store + attachment_core suites in full.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/attachment_core.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_attachment_core.py
git commit -m "feat(console): image-counting payload budget and multi-attachment send

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Screen — staging flows, labels, Save-all, serialization, resume

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (staging tails of `_process_console_attachment` ~7489 and `_paste_console_clipboard_image` ~7430; on_paste multi-drop toast; pending label build in `_sync_console_composer_action_state`; `_save_console_message_image` ~7811 → save-all; `_serialize_console_message` ~5858 + restore; `_rehydrate_console_message_image` ~6038 → batch attachments; `_console_messages_from_conversation_tree` region)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append-only)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: staging uses `add_pending_attachment` (False → cap toast `"Attachment limit reached (5 per message)."` and stop); multi-drop truncation toast `"Attached first {n} of {m} dropped files."`; composer label: 1 → `pending.label` (unchanged), N → `"📎 {N} files"` with tooltip listing display names — passed via the existing `set_pending_attachment_label` (label string) plus tooltip; Save-all writes every attachment with data (in-memory first, DB batch fallback), summary toast `"Saved {N} images to {directory}"`; serialization allowlist gains `attachment_labels: list[str]`; restore/resume rebuilds tuples via `get_attachments_for_messages` batch + legacy columns.

- [ ] **Step 1: Failing tests** (append; harness idiom per neighbors)

```python
async def test_staging_appends_and_caps_at_five(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    paths = []
    for index in range(6):
        p = tmp_path / f"img{index}.png"
        PILImage.new("RGB", (8, 8), (index * 30, 9, 9)).save(p, format="PNG")
        paths.append(p)

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        import tldw_chatbook.Chat.attachment_core as attachment_core

        original_load = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original_load(file_path, allowed_root=str(tmp_path))

        monkeypatch.setattr(attachment_core, "load_processed_file", _rooted)
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )

        store = console._ensure_console_chat_store()
        for index in range(5):
            await console._process_console_attachment(str(paths[index]))
            session_id = store.active_session_id
            assert len(store.pending_attachments(session_id)) == index + 1

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer._pending_attachment_label == "📎 5 files"

        await console._process_console_attachment(str(paths[5]))
        assert len(store.pending_attachments(store.active_session_id)) == 5
        assert any("Attachment limit reached (5 per message)." in n for n in notifications)


async def test_save_image_saves_all_attachments(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.chat_screen.get_cli_setting",
            lambda section, key, default=None: str(tmp_path)
            if (section, key) == ("chat.images", "save_location")
            else default,
        )
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="three",
            attachments=(
                MessageAttachment(data=b"img-0", mime_type="image/png", display_name="a.png", position=0),
                MessageAttachment(data=b"img-1", mime_type="image/png", display_name="b.png", position=1),
                MessageAttachment(data=b"img-2", mime_type="image/jpeg", display_name="c.jpg", position=2),
            ),
        )

        await console._save_console_message_image(message.id)

        saved = sorted(tmp_path.glob("console_image_*"))
        assert len(saved) == 3
        assert any("Saved 3 images to" in n for n in notifications)
```

Also append a serialization round-trip test: build a 2-attachment message, assert `ChatScreen._serialize_console_message(message)["attachment_labels"] == ["a.png", "b.png"]` and that no bytes-bearing key exists; then `_restore_console_message` yields a 2-item metadata-only tuple (data None, names preserved). Adapt harness/fixture idioms from neighbors (`test_console_message_serialization_carries_image_metadata_not_bytes`); no `...` may remain in the committed tests.

- [ ] **Step 2: RED** on the new selections.

- [ ] **Step 3: Implement**

1. Staging tails (`_process_console_attachment` attachment branch, `_paste_console_clipboard_image` image/paths branches): `store.add_pending_attachment(...)`; on False → `self.app_instance.notify("Attachment limit reached (5 per message).", severity="warning")` and skip label/toast updates. Multi-drop in `on_paste` and clipboard-paths: attach up to remaining capacity across the dropped list (loop the paths through the same worker sequentially — keep it simple: attach only the FIRST as today, but when `total_dropped > 1` AND capacity remains, keep the existing first-only behavior and emit `"Attached first 1 of {m} dropped files."`? NO — spec says append-wise staging with cap; implement: iterate dropped paths, `await self._process_console_attachment(path)` for each until `add_pending_attachment` returns False or list ends; toast `"Attached first {n} of {m} dropped files."` only when truncated (n < m); when all attach, per-file toasts already fire).
2. Composer label build in `_sync_console_composer_action_state`: `pendings = store.pending_attachments(...)`; label = `pendings[0].label` if len==1 else `f"📎 {len(pendings)} files"` if pendings else None; tooltip via `set_pending_attachment_label` stays label-based — extend the composer method? NO composer changes allowed per spec ("composer API unchanged"): the tooltip for N files goes on the attach button via the label string only; full name list goes in the indicator tooltip — the composer sets tooltip from the label already ("Attached: {label}"). Names list: append to the notify on stage instead. Keep composer untouched.
3. `_console_pending_image_attachment` predicate → any staged attachment-mode item with data (rename NOT needed; adjust body to scan the list).
4. Save-all: loop `message.attachments` (data present), reuse the collision-safe writer per attachment (extension per mime); DB fallback: when tuple has dataless entries and `persisted_message_id` exists → batch fetch via `getattr(db, "get_attachments_for_messages", None)` + legacy row for #0; summary toast `f"Saved {count} images to {save_location}"` (escape path per #628 convention? path is config-derived — match the existing single-save toast style).
5. Serialization: `"attachment_labels": [a.display_name for a in message.attachments]` in `_serialize_console_message`; restore rebuilds metadata-only tuples (`MessageAttachment(data=None, mime_type=message.image_mime_type or "", display_name=label, position=i)`) — scalars restored as today.
6. Rehydration: `_rehydrate_console_message_image` generalizes — after restore, for messages with `persisted_message_id`, batch-fetch attachments table rows once per restore pass (collect ids first, one `get_attachments_for_messages` call), rebuild full tuples (legacy row #0 via existing `get_message_by_id` fallback + table rows), through the store mirror helper (`store` not available at that point? it operates pre-restore_state on message objects — set fields directly maintaining the mirror manually, with a comment referencing the store helper rule).
7. Resume tree builder: same batch approach for `_console_messages_from_conversation_tree` (fetch after building the list, before returning).

- [ ] **Step 4: Run** the flow file in full + Tests/Chat/.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): multi-attachment staging, save-all, serialization, resume

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Transcript — one chip per attachment

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`_message_image_chip` ~110 → `_message_attachment_chips`, call site ~150)
- Test: `Tests/UI/test_console_native_transcript.py` (append-only)

**Interfaces:**
- Consumes: `ConsoleChatMessage.attachments` (Task 2); scalar mirror keeps single-attachment behavior byte-identical.
- Produces: multi-attachment messages render one `🖼 {name-or-mime-size}` line per attachment (position order); single/zero behavior unchanged (existing chip tests must stay green).

- [ ] **Step 1: Failing test** (append)

```python
def test_multi_attachment_message_renders_chip_per_attachment():
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="three pics",
    )
    message.attachments = (
        MessageAttachment(data=b"1", mime_type="image/png", display_name="a.png", position=0),
        MessageAttachment(data=b"22", mime_type="image/jpeg", display_name="b.jpg", position=1),
        MessageAttachment(data=None, mime_type="image/png", display_name="", position=2),
    )
    message.image_data = b"1"
    message.image_mime_type = "image/png"
    message.attachment_label = "a.png"

    rendered = _message_render_text(message, selected=False)
    plain = rendered.plain
    assert "🖼 a.png" in plain
    assert "🖼 b.jpg" in plain
    assert plain.count("🖼") == 3  # dataless third falls back to mime label
```

- [ ] **Step 2: RED** (`count("🖼") == 3` fails — only one chip today).

- [ ] **Step 3: Implement** — generalize the helper:

```python
def _message_attachment_chips(message: ConsoleChatMessage) -> list[str]:
    """Return one placeholder chip line per attachment (position order)."""
    attachments = getattr(message, "attachments", ()) or ()
    if not attachments:
        legacy = _message_image_chip_legacy(message)
        return [legacy] if legacy else []
    chips: list[str] = []
    for attachment in attachments:
        if attachment.display_name:
            chips.append(f"🖼 {attachment.display_name}")
        elif attachment.data is not None:
            chips.append(
                f"🖼 {attachment.mime_type or 'image'} · {_human_size(len(attachment.data))}"
            )
        else:
            chips.append(f"🖼 {attachment.mime_type or 'image'}")
    return chips
```

(rename the existing `_message_image_chip` to `_message_image_chip_legacy` UNCHANGED as the zero-attachments fallback — metadata-only restored rows pre-dating the tuple; call site joins `chips` with newlines where the single chip was appended. The 215 image-row and signature logic are untouched — first-image rendering keys off the scalar mirror.)

- [ ] **Step 4: Run** the transcript file in full (existing chip tests green = single behavior preserved).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_transcript.py
git commit -m "feat(console): transcript chip per attachment

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Full verification + live QA + approval gate

**Files:** none expected (fix-forward only).

- [ ] **Step 1: Full affected surface**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/ Tests/ChaChaNotesDB/test_chachanotes_db.py \
  Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_chat_image_attachment.py Tests/Event_Handlers/Chat_Events/test_chat_image_events.py \
  Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py Tests/unit/test_chat_image_unit.py \
  Tests/DB/test_chat_image_db_compatibility.py Tests/Widgets/test_chat_message_enhanced.py \
  -q --no-header
```

Expected: 0 real failures.

- [ ] **Step 2: Live QA** — rig per `Docs/superpowers/qa/console-paste-dnd-2026-07/README.md`. Captures → `Docs/superpowers/qa/console-multi-attach-2026-07/`:
  1. `staged-three-files.png` — three attaches → `📎 3 files` indicator.
  2. `cap-toast-sixth.png` — sixth attach → cap toast, count stays 5.
  3. `sent-chips-per-attachment.png` — real send (vision-override recipe) → one 🖼 chip per attachment + first image inline (215 row).
  4. `multi-drop-truncation.png` — paste more paths than remaining capacity → "Attached first n of m dropped files."
  5. `save-all.png` — Save Image on the multi message → summary toast; `ls` evidence of N files.
  6. `resume-multi-rehydrated.png` — relaunch + resume → all chips back (names from the table!), first image re-renders.
  Also: run the app once against a COPY of a pre-migration (v18) DB fixture to capture `migration-upgrade-clean.png` (app boots, old single-image conversation intact) — create the fixture by checking out nothing: build it with a scratch script pinning `_CURRENT_SCHEMA_VERSION` — if impractical, document honestly and rely on the migration unit test.
- [ ] **Step 3: Visual approval gate** — user approval required before PR.
- [ ] **Step 4: Wrap-up** — finishing flow (TASK-217 backlog Done + notes riding the branch; PR on approval). TASK-222 starts next per the user's queue.

---

## Deferred (do not implement)

Per-item pending removal; multi-image inline rendering; Sync v2 attachment sync (TASK-220); Chatbook export (TASK-221); config-driven caps (TASK-222 — `MAX_PENDING_ATTACHMENTS` stays a named constant here).
