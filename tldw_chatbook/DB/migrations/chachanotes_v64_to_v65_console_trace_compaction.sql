-- ChaChaNotes v64 -> v65: bounded, content-free trace compaction status.

CREATE TABLE console_trace_compaction_state(
  singleton_id INTEGER PRIMARY KEY NOT NULL CHECK(singleton_id = 1),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'running', 'complete')),
  reason_code TEXT NOT NULL DEFAULT 'awaiting_gc'
    CHECK(length(reason_code) BETWEEN 1 AND 64),
  last_gc_request_id TEXT DEFAULT NULL
    CHECK(last_gc_request_id IS NULL OR length(last_gc_request_id) BETWEEN 1 AND 128),
  attempt_id TEXT DEFAULT NULL
    CHECK(attempt_id IS NULL OR length(attempt_id) BETWEEN 1 AND 128),
  retry_count INTEGER NOT NULL DEFAULT 0 CHECK(retry_count BETWEEN 0 AND 32),
  next_retry_at TEXT DEFAULT NULL,
  progress_basis_points INTEGER NOT NULL DEFAULT 0
    CHECK(progress_basis_points BETWEEN 0 AND 10000),
  allocated_bytes_before INTEGER NOT NULL DEFAULT 0 CHECK(allocated_bytes_before >= 0),
  allocated_bytes_after INTEGER NOT NULL DEFAULT 0 CHECK(allocated_bytes_after >= 0),
  freelist_bytes_before INTEGER NOT NULL DEFAULT 0 CHECK(freelist_bytes_before >= 0),
  freelist_bytes_after INTEGER NOT NULL DEFAULT 0 CHECK(freelist_bytes_after >= 0),
  wal_bytes_before INTEGER NOT NULL DEFAULT 0 CHECK(wal_bytes_before >= 0),
  wal_bytes_after INTEGER NOT NULL DEFAULT 0 CHECK(wal_bytes_after >= 0),
  logical_live_bytes INTEGER NOT NULL DEFAULT 0 CHECK(logical_live_bytes >= 0),
  started_at TEXT DEFAULT NULL,
  completed_at TEXT DEFAULT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO console_trace_compaction_state(singleton_id) VALUES (1);

CREATE TRIGGER console_trace_compaction_state_immutable_key
BEFORE UPDATE ON console_trace_compaction_state
WHEN OLD.singleton_id IS NOT NEW.singleton_id
BEGIN
  SELECT RAISE(ABORT, 'trace compaction singleton identity is immutable');
END;

CREATE TRIGGER console_trace_compaction_state_no_delete
BEFORE DELETE ON console_trace_compaction_state BEGIN
  SELECT RAISE(ABORT, 'console_trace_compaction_state deletion prohibited');
END;
