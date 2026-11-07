BEGIN TRANSACTION;
CREATE TABLE agent_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    instructions TEXT NOT NULL DEFAULT '',
                    tool_allowlist TEXT NOT NULL DEFAULT '[]',
                    model TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
CREATE TABLE agent_run_steps (
                    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );
CREATE TABLE agent_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    agent_kind TEXT NOT NULL,
                    task TEXT,
                    status TEXT NOT NULL,
                    steps TEXT NOT NULL DEFAULT '[]',
                    result TEXT,
                    budget TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    assistant_message_id TEXT,
                    agent_definition TEXT,
                    definition_fingerprint TEXT,
                    wake_delivered_at TEXT,
                    -- v11 (fleet PR3b Task 4, spec SS6): when this run is
                    -- a CONTINUATION of a finished sub-agent -- a NEW run
                    -- seeded from the old one's retained in-memory
                    -- transcript via send_to_agent -- this records the
                    -- run it resumed from. NULL for every ordinary run.
                    -- Lineage only: parent_run_id still points at the
                    -- RESUMING turn's primary, never at the old run.
                    resumed_from_run_id TEXT,
                    -- v14 (ADR-080): the stable parent Trace event that
                    -- caused this run to exist. NULL for primary and
                    -- legacy runs whose precise cause was not captured.
                    spawn_event_id TEXT
                );
CREATE TABLE change_notes (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    root TEXT NOT NULL,
                    path TEXT NOT NULL,
                    hunk_index INTEGER NOT NULL,
                    hunk_header TEXT NOT NULL,
                    hunk_excerpt TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    -- v9 (TASK-16800 Task 6 fix round): which run's
                    -- COMPLETION actually stamped this note delivered --
                    -- distinct from `run_id` above, which anchors the note
                    -- to the run whose DIFF it critiques. A note written
                    -- against an earlier run's diff is commonly delivered
                    -- on a LATER run's completion, so resume re-derivation
                    -- needs this to place the disclosure row at the
                    -- delivering run's position (matching live emission)
                    -- instead of fragmenting/mis-anchoring at the
                    -- annotated run. NULL until stamped, and NULL forever
                    -- on rows stamped by code that predates this column.
                    delivered_by_run_id TEXT,
                    -- v10 (Qodo #6, PR #1779 fix round): the exact
                    -- change_snapshots row (its own DB `id`) this note's
                    -- hunk was read from. Two windows on the SAME run+
                    -- root+path (a turn's own window and its surviving
                    -- sub-agents' post-turn window, PR3a-1 Task 6c) can
                    -- theoretically carry the same hunk_header at the
                    -- same hunk_index (identical position/line-count
                    -- edits against different baselines produce
                    -- byte-identical "@@ ... @@" headers, since a header
                    -- encodes only positions/counts, never content) --
                    -- (run_id, hunk_index, hunk_header) alone can't then
                    -- say which window's diff was actually annotated.
                    -- NULL for legacy rows saved before this column
                    -- existed; the card falls back to hunk_index+
                    -- hunk_header matching for those.
                    snapshot_id INTEGER,
                    -- v12 (TASK-18060 Task 1, review-rail spec §4): anchor
                    -- kind + diff-line anchoring, for the Review screen's
                    -- plannotator-style comments alongside the card's
                    -- existing hunk notes. 'hunk' is every pre-v11 row's
                    -- true, honest kind (nothing else existed before this
                    -- column), so the DEFAULT both back-declares existing
                    -- rows correctly and keeps every V1.5 caller (which
                    -- never passes anchor_kind) byte-compatible.
                    -- diff_line_index is 0-based over the file's FULL diff
                    -- text (same semantics as hunk_index, a distinct
                    -- axis), NULL except for 'diff_line' rows.
                    -- diff_line_text is the anchored line, captured
                    -- verbatim at note-creation time -- same retention
                    -- posture as hunk_excerpt (self-contained even after
                    -- shadow-repo snapshot pruning), NULL except for
                    -- 'diff_line' rows. For 'diff_line' rows the hunk
                    -- fields ABOVE are ALSO populated (the hunk the line
                    -- falls in) -- a deliberate convergence so the card's
                    -- existing hunk-note filter renders line comments
                    -- under their hunk with no card changes. For 'file'
                    -- rows (whole-file comments) hunk_index=-1,
                    -- hunk_header='', hunk_excerpt='' -- sentinels that
                    -- can never match a real hunk, keeping them out of
                    -- the card's hunk-note filter.
                    anchor_kind TEXT NOT NULL DEFAULT 'hunk',
                    diff_line_index INTEGER,
                    diff_line_text TEXT
                );
CREATE TABLE change_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    root TEXT NOT NULL,
                    baseline_sha TEXT NOT NULL,
                    end_sha TEXT NOT NULL,
                    files_changed INTEGER NOT NULL DEFAULT 0,
                    adds INTEGER NOT NULL DEFAULT 0,
                    dels INTEGER NOT NULL DEFAULT 0,
                    reverted TEXT NOT NULL DEFAULT '',
                    tracking_error TEXT NOT NULL DEFAULT '',
                    untracked_oversize INTEGER NOT NULL DEFAULT 0,
                    nested_repos TEXT NOT NULL DEFAULT '[]',
                    -- v5->v6 (PR3a-1 Task 6c): which WINDOW this row
                    -- covers. 'turn' is the turn's own B/E window;
                    -- 'turn_concurrent_subagent' is the same window taken
                    -- while a sub-agent from an EARLIER turn was writing
                    -- (so the diff may include changes this turn's agent
                    -- did not make -- disclosed, never implied);
                    -- 'subagent_post_turn' is the window AFTER a turn's E
                    -- during which its surviving sub-agents were still
                    -- working. The default keeps every pre-existing row
                    -- reading as what it was.
                    kind TEXT NOT NULL DEFAULT 'turn',
                    created_at TEXT NOT NULL
                );
CREATE TABLE console_activity_receipts (
                activity_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL
                    CHECK(origin IN ('ordinary', 'fleet_survivor')),
                logical_outcome_id TEXT NOT NULL,
                transition_revision INTEGER NOT NULL
                    CHECK(transition_revision > 0),
                session_id TEXT,
                conversation_id TEXT,
                run_id TEXT,
                assistant_message_id TEXT,
                status TEXT NOT NULL CHECK(status IN
                    ('done', 'failed', 'stuck', 'stopped', 'cancelled')),
                created_at TEXT NOT NULL,
                acknowledged_at TEXT,
                superseded_at TEXT,
                CHECK(session_id IS NOT NULL OR conversation_id IS NOT NULL),
                UNIQUE(origin, logical_outcome_id, transition_revision)
            );
CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
INSERT INTO "schema_version" VALUES(4);
INSERT INTO "schema_version" VALUES(5);
INSERT INTO "schema_version" VALUES(6);
INSERT INTO "schema_version" VALUES(7);
INSERT INTO "schema_version" VALUES(8);
INSERT INTO "schema_version" VALUES(9);
INSERT INTO "schema_version" VALUES(10);
INSERT INTO "schema_version" VALUES(11);
INSERT INTO "schema_version" VALUES(12);
INSERT INTO "schema_version" VALUES(13);
INSERT INTO "schema_version" VALUES(14);
INSERT INTO "schema_version" VALUES(15);
CREATE INDEX idx_agent_runs_conversation
                    ON agent_runs(conversation_id);
CREATE INDEX idx_agent_runs_parent
                    ON agent_runs(parent_run_id);
CREATE INDEX idx_change_snapshots_run
                    ON change_snapshots(run_id);
CREATE UNIQUE INDEX idx_agent_definitions_name
                    ON agent_definitions(name) WHERE deleted = 0;
CREATE INDEX idx_change_notes_pending
                    ON change_notes(run_id) WHERE delivered_at IS NULL;
CREATE INDEX idx_console_activity_receipts_unseen
                ON console_activity_receipts(created_at DESC, activity_id)
                WHERE acknowledged_at IS NULL AND superseded_at IS NULL
            ;
DELETE FROM "sqlite_sequence";
COMMIT;
