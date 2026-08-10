-- V32 -> V33: local-only Console context policy, derived memory, and
-- content-free auxiliary-attempt ownership (ADR-052 / TASK-14811.1).
--
-- These tables intentionally have no sync columns and no sync triggers.
-- Conversation transcripts remain authoritative; memory is private derived
-- data, and auxiliary rows contain only bounded operational/usage metadata.

CREATE TABLE IF NOT EXISTS console_conversation_context_policy(
  conversation_id      TEXT PRIMARY KEY
                            REFERENCES conversations(id)
                            ON DELETE CASCADE ON UPDATE CASCADE,
  budget_mode          TEXT CHECK(budget_mode IN ('automatic','custom')),
  custom_budget_tokens INTEGER CHECK(custom_budget_tokens > 0),
  compaction_mode      TEXT CHECK(compaction_mode IN ('ask','automatic','off')),
  trigger_ratio        REAL CHECK(trigger_ratio > 0 AND trigger_ratio <= 0.95),
  target_ratio         REAL CHECK(target_ratio > 0 AND target_ratio < 1),
  summary_max_tokens   INTEGER CHECK(summary_max_tokens > 0),
  failure_behavior     TEXT CHECK(failure_behavior IN ('stop_and_ask','omit_older_context')),
  carry_forward_mode   TEXT CHECK(carry_forward_mode IN ('memory_with_recent_turns','memory_with_latest_exchange')),
  policy_revision      INTEGER NOT NULL DEFAULT 1 CHECK(policy_revision > 0),
  updated_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Composite memory FKs must prove that branch pointers belong to the same
-- conversation as the memory row, not merely that a globally unique message
-- ID exists somewhere in the database.
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conversation_id_id
  ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS console_conversation_memories(
  id                       TEXT PRIMARY KEY,
  conversation_id          TEXT NOT NULL
                               REFERENCES conversations(id)
                               ON DELETE CASCADE ON UPDATE CASCADE,
  boundary_message_id      TEXT,
  captured_leaf_message_id TEXT,
  lineage_json             TEXT NOT NULL DEFAULT '[]',
  summary_text             TEXT NOT NULL,
  provider                 TEXT,
  model                    TEXT,
  prompt_id                TEXT,
  prompt_revision          INTEGER CHECK(prompt_revision > 0),
  prompt_digest            TEXT,
  selected_units_json      TEXT NOT NULL DEFAULT '[]',
  summarized_prefix_digest TEXT,
  input_tokens             INTEGER CHECK(input_tokens >= 0),
  output_tokens            INTEGER CHECK(output_tokens >= 0),
  before_tokens            INTEGER CHECK(before_tokens >= 0),
  after_tokens             INTEGER CHECK(after_tokens >= 0),
  created_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revision                 INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
  active                   INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
  source_kind              TEXT NOT NULL DEFAULT 'generated'
                                CHECK(source_kind IN ('generated','legacy')),
  reset_at                 DATETIME,
  FOREIGN KEY (conversation_id, boundary_message_id)
    REFERENCES messages(conversation_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (conversation_id, captured_leaf_message_id)
    REFERENCES messages(conversation_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_console_memories_conversation_active
  ON console_conversation_memories(conversation_id, active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_console_memories_boundary
  ON console_conversation_memories(conversation_id, boundary_message_id);

CREATE TABLE IF NOT EXISTS console_auxiliary_attempts(
  operation_id             TEXT PRIMARY KEY,
  conversation_id          TEXT NOT NULL
                               REFERENCES conversations(id)
                               ON DELETE CASCADE ON UPDATE CASCADE,
  purpose                  TEXT NOT NULL,
  provider                 TEXT NOT NULL,
  model                    TEXT NOT NULL,
  requested_output_cap     INTEGER NOT NULL CHECK(requested_output_cap > 0),
  estimated_input_tokens   INTEGER NOT NULL CHECK(estimated_input_tokens >= 0),
  status                   TEXT NOT NULL
                                CHECK(status IN ('started','succeeded','failed','cancelled','stale')),
  started_at               DATETIME NOT NULL,
  finished_at              DATETIME,
  elapsed_ms               INTEGER CHECK(elapsed_ms >= 0),
  pricing_provenance_json  TEXT,
  provider_usage_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_console_aux_attempts_conversation_started
  ON console_auxiliary_attempts(conversation_id, started_at DESC);

-- Preserve legacy /rewind summaries as reviewable derived memory, but keep
-- them inactive: v26 columns do not carry lineage/prefix digests, so they
-- are not safe for automatic request selection. The original columns remain
-- in place for manual-rewind compatibility during the staged rollout.
INSERT OR IGNORE INTO console_conversation_memories(
  id, conversation_id, boundary_message_id, captured_leaf_message_id,
  lineage_json, summary_text, prompt_id, selected_units_json,
  created_at, revision, active, source_kind
)
SELECT 'legacy-context-summary:' || c.id,
       c.id,
       c.summary_boundary_message_id,
       c.summary_boundary_message_id,
       '[]',
       c.context_summary,
       'console.rewind_summarize',
       '[]',
       CURRENT_TIMESTAMP,
       1,
       0,
       'legacy'
  FROM conversations AS c
 WHERE c.deleted = 0
   AND c.context_summary IS NOT NULL
   AND c.summary_boundary_message_id IS NOT NULL
   AND EXISTS (
       SELECT 1
         FROM messages AS m
        WHERE m.id = c.summary_boundary_message_id
          AND m.conversation_id = c.id
          AND m.deleted = 0
   );

UPDATE db_schema_version
   SET version = 33
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 32;
