# Private Persistent Artifact Lifecycles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TASK-490 by making application logs, MCP execution logs,
and the optional tool-result cache private, no-follow, bounded, and
non-executable on disk.

**Architecture:** Extend the existing dependency-leaf `private_paths` module
with descriptor-anchored append and atomic-replacement primitives. Reuse those
primitives in a small private rotating-file handler, `MCPExecutionLog`, and the
existing `ToolResultCache`; do not introduce a second filesystem policy layer.
The tool cache uses a strict versioned JSON envelope and leaves unsupported
results memory-only.

**Tech Stack:** Python 3.11+, stdlib `os`/`json`/`logging`, Loguru, pytest.

---

### Task 1: Lock the artifact lifecycle with failing tests

**Files:**
- Create: `Tests/Utils/test_private_persistent_artifacts.py`
- Modify: `Tests/MCP/test_execution_log.py`
- Create: `Tests/Tools/test_tool_result_cache_persistence.py`
- Create: `Tests/test_logging_private_files.py`

- [x] Add tests proving basename-only application log selection rejects
  absolute paths, separators, dot entries, and traversal without creating an
  outside directory.
- [x] Add POSIX tests proving new and eligible existing application log files
  and rotated generations become `0600` beneath a `0700` application data
  directory.
- [x] Add log-handler tests proving a symlinked active or rotated generation
  disables only the file sink.
- [x] Extend MCP tests for private creation, existing-file hardening, rotation,
  no-follow count/read/append, corrupt-line tolerance, and unsafe target
  containment.
- [x] Add cache tests proving pickle is never invoked, legacy pickle remains
  inert, JSON corruption is ignored, unsupported results remain memory-only,
  strict size/schema validation applies, and a valid cache round-trips.
- [x] Add target/parent replacement and simulated unverified-platform tests
  around the shared private append/replace seam.
- [x] Run the new tests and record the expected failures against the current
  pathname-based logging, MCP, and pickle implementation.

### Task 2: Add descriptor-anchored private append and replacement

**Files:**
- Modify: `tldw_chatbook/Utils/private_paths.py`
- Test: `Tests/Utils/test_private_persistent_artifacts.py`

- [x] Add a private append opener that verifies or exclusively creates a
  current-user regular file, applies `0600`, pins identity, and never follows
  the leaf.
- [x] Add atomic private byte/text replacement using a `0600` sibling created
  relative to the verified parent descriptor, target identity revalidation,
  descriptor-relative `os.replace`, fsync, and a verified postcondition.
- [x] Preserve the existing Windows `unverified_platform` contract without
  claiming POSIX mode or ACL verification.
- [x] Run the focused utility tests until green.

### Task 3: Secure the rotating application file sink

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Logging_Config.py`
- Test: `Tests/test_logging_private_files.py`

- [x] Make `get_cli_log_file_path()` accept only a non-empty basename and
  return a child of the secured user data directory without creating
  user-controlled parents.
- [x] Add a `RotatingFileHandler` subclass whose active stream uses the shared
  no-follow private append opener.
- [x] Before installation and after rollover, verify or harden every eligible
  active/rotated generation; reject links and non-regular entries.
- [x] On an unsafe file target, omit only the file handler and emit a bounded
  metadata-only warning while Rich/UI/terminal handlers remain installed.
- [x] Run the focused logging tests until green.

### Task 4: Secure MCP execution-log storage

**Files:**
- Modify: `tldw_chatbook/MCP/execution_log.py`
- Test: `Tests/MCP/test_execution_log.py`

- [x] Secure the application-owned parent before every operation.
- [x] Count and read generations through the pinned private binary opener.
- [x] Append through the private append seam.
- [x] Rotate by safely reading the active generation and atomically replacing
  the rotated and active generations; never follow or overwrite an unsafe
  target.
- [x] Keep torn/corrupt final-line tolerance and the two-generation bound.
- [x] Run the focused MCP tests until green.

### Task 5: Replace pickle persistence with bounded versioned JSON

**Files:**
- Modify: `tldw_chatbook/Tools/tool_executor.py`
- Test: `Tests/Tools/test_tool_result_cache_persistence.py`

- [x] Remove the pickle import and define one versioned JSON cache envelope
  with explicit entry count, byte-size, key, expiry, finite-number, depth, and
  JSON-value validation.
- [x] Load through the private binary opener; harden but ignore legacy,
  corrupt, oversized, or invalid files without deserializing executable data.
- [x] Persist only JSON-compatible entries through atomic private replacement;
  leave unsupported results in the bounded in-memory cache.
- [x] Preserve LRU order, TTL expiry, cache-hit return values, and clear
  behavior.
- [x] Run the focused cache tests until green.

### Task 6: Verify and close TASK-490

**Files:**
- Modify: `backlog/tasks/task-490 - Harden-persistent-log-and-tool-cache-file-lifecycles.md`

- [x] Run `pytest Tests/Utils/test_private_persistent_artifacts.py Tests/MCP/test_execution_log.py Tests/Tools/test_tool_result_cache_persistence.py Tests/test_logging_private_files.py -q`.
- [x] Run relevant broader MCP, config/private-path, and tool tests.
- [x] Run Ruff on changed Python files, Python compilation, and
  `git diff --check`.
- [x] Run a real sentinel/mode probe against all three persistent artifacts.
- [x] Self-review the complete diff for unsafe path races, payload diagnostics,
  and scope creep.
- [x] Check all TASK-490 acceptance criteria, add concise implementation notes
  and verification evidence, set the task Done, and commit only TASK-490
  files.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/022-local-private-data-boundary.md`

Reason: TASK-490 directly implements ADR-022's accepted persistent-log and
tool-cache lifecycle policy without changing the decision.
