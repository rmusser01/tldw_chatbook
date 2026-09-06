-- ChaChaNotes v66 -> v67: conversation-owned immutable Canvas revision graphs.
--
-- Canvas rows are deliberately local-only: no sync-log triggers are installed.
-- Titles live on revisions so branch-local renames remain immutable history.

CREATE TABLE canvas_documents(
  id TEXT PRIMARY KEY NOT NULL
    CHECK(length(id) = 36),
  conversation_id TEXT NOT NULL
    REFERENCES conversations(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL
    CHECK(length(created_at) > 0),
  deleted INTEGER NOT NULL DEFAULT 0
    CHECK(deleted IN (0, 1)),
  deleted_at TEXT DEFAULT NULL,
  CHECK(
    (deleted = 0 AND deleted_at IS NULL)
    OR (deleted = 1 AND length(deleted_at) > 0)
  )
);

CREATE UNIQUE INDEX uq_canvas_documents_id_conversation
  ON canvas_documents(id, conversation_id);

CREATE INDEX idx_canvas_documents_conversation
  ON canvas_documents(conversation_id, deleted, created_at, id);

CREATE TABLE canvas_revisions(
  id TEXT PRIMARY KEY NOT NULL
    CHECK(length(id) = 36),
  canvas_id TEXT NOT NULL
    REFERENCES canvas_documents(id) ON DELETE CASCADE,
  parent_revision_id TEXT DEFAULT NULL,
  sequence INTEGER NOT NULL
    CHECK(sequence > 0),
  title TEXT NOT NULL
    CHECK(length(title) > 0),
  runtime_profile TEXT NOT NULL
    CHECK(runtime_profile = 'canvas-v1'),
  html TEXT NOT NULL,
  content_sha256 TEXT NOT NULL
    CHECK(
      length(content_sha256) = 64
      AND content_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
  html_bytes INTEGER NOT NULL
    CHECK(html_bytes >= 0 AND html_bytes = length(CAST(html AS BLOB))),
  actor_kind TEXT NOT NULL
    CHECK(actor_kind IN ('assistant', 'user_rename', 'user_import')),
  origin_message_id TEXT NOT NULL
    REFERENCES messages(id) ON DELETE RESTRICT,
  origin_turn_id TEXT NOT NULL
    CHECK(length(origin_turn_id) > 0),
  created_at TEXT NOT NULL
    CHECK(length(created_at) > 0),
  deleted_at TEXT DEFAULT NULL,
  CHECK(
    typeof(html) = 'text'
    AND canvas_revision_payload_valid(
      CAST(html AS BLOB), content_sha256, html_bytes
    ) = 1
  ),
  CHECK(
    (sequence = 1 AND parent_revision_id IS NULL)
    OR (sequence > 1 AND parent_revision_id IS NOT NULL)
  ),
  FOREIGN KEY(parent_revision_id, canvas_id)
    REFERENCES canvas_revisions(id, canvas_id)
);

CREATE UNIQUE INDEX uq_canvas_revisions_id_canvas
  ON canvas_revisions(id, canvas_id);

CREATE UNIQUE INDEX idx_canvas_revisions_canvas_sequence
  ON canvas_revisions(canvas_id, sequence);

CREATE INDEX idx_canvas_revisions_parent
  ON canvas_revisions(canvas_id, parent_revision_id, sequence);

CREATE INDEX idx_canvas_revisions_origin_message
  ON canvas_revisions(origin_message_id, canvas_id, sequence);

CREATE TABLE canvas_conversation_hints(
  conversation_id TEXT PRIMARY KEY NOT NULL
    REFERENCES conversations(id) ON DELETE CASCADE,
  last_canvas_id TEXT NOT NULL,
  updated_at TEXT NOT NULL
    CHECK(length(updated_at) > 0),
  FOREIGN KEY(last_canvas_id, conversation_id)
    REFERENCES canvas_documents(id, conversation_id) ON DELETE CASCADE
);

CREATE TRIGGER canvas_documents_ownership_immutable
BEFORE UPDATE OF id, conversation_id, created_at ON canvas_documents
WHEN
  OLD.id IS NOT NEW.id
  OR OLD.conversation_id IS NOT NEW.conversation_id
  OR OLD.created_at IS NOT NEW.created_at
BEGIN
  SELECT RAISE(ABORT, 'canvas document ownership is immutable');
END;

CREATE TRIGGER canvas_revisions_origin_owner_guard
BEFORE INSERT ON canvas_revisions
WHEN NOT EXISTS (
  SELECT 1
    FROM canvas_documents AS document
    JOIN messages AS message
      ON message.id = NEW.origin_message_id
   WHERE document.id = NEW.canvas_id
     AND message.conversation_id = document.conversation_id
)
BEGIN
  SELECT RAISE(ABORT, 'canvas revision origin owner mismatch');
END;

CREATE TRIGGER canvas_origin_message_owner_guard
BEFORE UPDATE OF conversation_id ON messages
WHEN OLD.conversation_id IS NOT NEW.conversation_id
  AND EXISTS (
    SELECT 1
      FROM canvas_revisions AS revision
      JOIN canvas_documents AS document
        ON document.id = revision.canvas_id
     WHERE revision.origin_message_id = OLD.id
       AND document.conversation_id IS NOT NEW.conversation_id
  )
BEGIN
  SELECT RAISE(ABORT, 'canvas revision origin owner mismatch');
END;

CREATE TRIGGER canvas_revisions_parent_guard
BEFORE INSERT ON canvas_revisions
WHEN NEW.parent_revision_id IS NOT NULL AND NOT EXISTS (
  SELECT 1
    FROM canvas_revisions AS parent
   WHERE parent.id = NEW.parent_revision_id
     AND parent.canvas_id = NEW.canvas_id
     AND parent.sequence < NEW.sequence
)
BEGIN
  SELECT RAISE(ABORT, 'canvas revision parent mismatch');
END;

CREATE TRIGGER canvas_revisions_no_update
BEFORE UPDATE ON canvas_revisions
BEGIN
  SELECT RAISE(ABORT, 'canvas revisions are immutable');
END;

CREATE TRIGGER canvas_revisions_no_delete
BEFORE DELETE ON canvas_revisions
WHEN canvas_revision_delete_authorized(OLD.canvas_id) <> 1
BEGIN
  SELECT RAISE(ABORT, 'canvas revision deletion authorization required');
END;
