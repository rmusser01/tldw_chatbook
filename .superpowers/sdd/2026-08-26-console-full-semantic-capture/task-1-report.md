# Task 1 report — Safe-First Capture Provenance and Persistence

## Implementation

- Added Safe/Full capture-detail contracts, fail-safe policy resolution, shared 64 MiB uncompressed budget, endpoint identity sanitization, structural credential exclusion, nested binary stubbing, bounded blob encode/decode, and legacy-Safe/blob-column provenance validation.
- Added the local-only v50→v51 migration: `message_exchanges.capture_detail` with a Safe default and checked sparse per-conversation capture-policy storage.
- Threaded immutable capture detail through DB upsert/select, persistence/store writes, and Inspector load so mismatches are skipped with content-free diagnostic categories.
- Added `exchange_capture_detail = "safe"` as the application default. No user-visible Full control was added.

## Files

Production: `console_exchange_capture.py`, `console_project_instructions.py`, `console_capture_policy_repository.py`, `config.py`, `ChaChaNotes_DB.py`, v50→v51 migration SQL, `chat_persistence_service.py`, `console_chat_store.py`, and `chat_screen.py`.

Tests: capture, repository, migration, message-exchange, store, and Inspector-loader tests listed in the task brief.

## TDD evidence

RED observed before production implementation:

1. `Tests/Chat/test_console_exchange_capture.py -q` failed collection with `ImportError: cannot import name 'CaptureDetail'`.
2. After the policy implementation, the Safe/Full construction test pass established the first GREEN; adding response construction tests then failed collection with `ImportError: cannot import name 'build_response_capture'`.
3. `Tests/DB/test_chachanotes_full_capture_migration.py Tests/Chat/test_console_capture_policy_repository.py -q` failed collection with `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_capture_policy_repository'`.
4. Provenance seam tests failed with the expected missing store key and unsafe loader behavior: `KeyError: 'capture_detail'` and a mismatched DB row decoded instead of returning `[]`.

GREEN evidence:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py \
  Tests/DB/test_chachanotes_full_capture_migration.py \
  Tests/Chat/test_console_capture_policy_repository.py \
  Tests/DB/test_chachanotes_message_exchanges.py \
  Tests/Chat/test_console_chat_store_exchanges.py \
  Tests/UI/test_chat_screen_console_inspector_loader.py -q
67 passed, 1 warning in 7.92s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_controller_exchanges.py -q
9 passed, 1 warning in 0.53s
```

`py_compile` also completed successfully for the three changed core Python modules, and `git diff --check` was clean.

## Self-review

- Safe is the trailing/default detail in both object and DB migration paths; absent legacy blob field is Safe.
- Blob and DB detail must match; invalid/mismatched provenance fails closed in Inspector loading without blob-content logging.
- Migration adds no sync triggers, FTS, server payload, metadata, or Trace projection. Compression remains only compression, never encryption.
- Policy repository uses parameterized immediate transactions and returns bounded status categories without exception-body logging.

## Concern / required remaining gate

The required complete DB migration command was attempted three times:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ChaChaNotesDB Tests/DB -q
```

The execution harness terminates/returns the command at its fixed 30-second boundary after progress around 7% (`Running 1828 items in this shard`) without a resumable session or an exit status. A detached `tmux` fallback is disallowed by its socket permission (`error creating /private/tmp/tmux-501/default (Operation not permitted)`). Therefore this report cannot claim the complete DB gate passed, and the Backlog task intentionally remains In Progress with unchecked ACs. The focused affected tests are green; a host that permits a long-lived test process must run the exact command above before task closeout.

## Commit

`311f8cff0 feat(console): persist bounded capture provenance`

## Concern resolution — schema inventory follow-up

Controller RED evidence completed the formerly blocked DB gate: the required
command collected 1,828 items and failed exactly the eight nodes recorded in
`.pytest_cache/v/cache/lastfailed`. The failures were schema-bookkeeping gaps
introduced by v50, rather than capture-policy behavior failures.

The follow-up pins `idx_message_exchanges_capture_detail` as the non-unique
`(capture_detail, message_id)` index, admits the local-only capture-policy
table through the independent ChaChaNotes SQL allowlist, and updates only
fully-migrated/current-version assertions in the v49 migration contract. The
v48-to-v49 step assertions remain pinned to version 49.

Files changed:

- `Tests/ChaChaNotesDB/test_index_census.py`
- `Tests/DB/test_chachanotes_v49_messages_fts_update_scope.py`
- `tldw_chatbook/DB/sql_validation.py`

GREEN evidence:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest --lf -q
7 passed, 403 deselected, 10 warnings in 8.83s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/ChaChaNotesDB/test_index_census.py \
  Tests/DB/test_chachanotes_v49_messages_fts_update_scope.py \
  Tests/DB/test_schema_table_allowlist_guard.py -q
43 passed, 1 warning in 6.12s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/DB/test_sql_validation.py -q
25 passed, 1 warning in 1.21s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py \
  Tests/DB/test_chachanotes_full_capture_migration.py \
  Tests/Chat/test_console_capture_policy_repository.py \
  Tests/DB/test_chachanotes_message_exchanges.py \
  Tests/Chat/test_console_chat_store_exchanges.py \
  Tests/UI/test_chat_screen_console_inspector_loader.py -q
67 passed, 1 warning in 8.30s
```

`git diff --check` passed. The known pytest cleanup warnings concern sandboxed
temporary `published` directories and do not affect test results. The
controller will rerun the complete DB migration command before the Backlog is
marked Done, so the task remains In Progress.

Follow-up commit: `bd6fe35536 test(db): complete capture schema inventory`.

## Fix round 1/5 — capture privacy review

Addressed all three Important findings from `task-1-review.md`:

- Both `api_endpoint` and `api_base_url` now use the incumbent canonical
  endpoint identity helper before capture retention.
- Fixed-name credential aliases now include `access_token` and
  `client_secret`. Recognized provider tool JSON string fields
  (`arguments`, `input`, `result`, and `output`) are parsed recursively with
  stdlib JSON, then use the incumbent credential-removal and binary-stubbing
  paths; ordinary non-JSON text remains unchanged.
- Response `content` now passes through the existing binary stubber before
  it consumes the shared capture budget.

Files changed:

- `tldw_chatbook/Chat/console_exchange_capture.py`
- `Tests/Chat/test_console_exchange_capture.py`

TDD RED against `bd6fe35536` before production edits:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py -q \
  -k 'api_endpoint_identity or request_tool_argument_json or response_tool_json or response_base64_content or response_data_uri_content'
5 failed, 26 deselected, 1 warning in 0.61s
```

The failures showed the credential-bearing `api_endpoint` unchanged,
`access_token`/`client_secret` plus nested binary payloads intact in JSON tool
strings, and raw base64/data-URI response content retained.

GREEN evidence:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py -q \
  -k 'api_endpoint_identity or request_tool_argument_json or response_tool_json or response_base64_content or response_data_uri_content'
5 passed, 26 deselected, 1 warning in 0.40s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py -q
31 passed, 1 warning in 2.79s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_exchange_capture.py \
  Tests/DB/test_chachanotes_full_capture_migration.py \
  Tests/Chat/test_console_capture_policy_repository.py \
  Tests/DB/test_chachanotes_message_exchanges.py \
  Tests/Chat/test_console_chat_store_exchanges.py \
  Tests/UI/test_chat_screen_console_inspector_loader.py -q
72 passed, 1 warning in 6.89s
```

The complete DB suite was not rerun for this round, per instruction; the
controller owns amended-code coverage for that gate.

Fix-round commit: `16915283a4 fix(console): sanitize capture payload gaps`.

## Final whole-branch correction

The final review found that sub-threshold streamed chunks could each pass
sanitation and then rejoin into raw data-URI/plain-base64 content. The gateway
now sanitizes final accumulated response and tool content before immutable
capture, without spending the shared budget twice. Typed persisted-policy read
outcomes also prevent schema/error/corrupt rows from being mistaken for absence
under Global Full, and the lowest SQLite write boundary is content-free.
Regression coverage drives split chunks through the real SQLite and Full-export
owners. See `final-review-fix-report.md` for exact RED/GREEN evidence. Task 1
remains **In Progress** and affected AC #2 remains unchecked for independent
re-review.

## Controller verification after the follow-up

The controller reran the exact mandatory gate from `bd6fe35536` in a resumable
session:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ChaChaNotesDB Tests/DB -q
1827 passed, 1 skipped, 4 warnings in 204.31s (0:03:24)
exit code 0
```

The four suite warnings are incumbent dependency/SyntaxWarning noise; pytest
also emitted sandbox-only temporary-directory cleanup warnings after the
successful exit. No DB or migration test failed.

## Final-review fix round 2

Independent re-review found the sanitized DB error still explicitly chained
the raw SQLite exception. RED: the real-seam test failed once because
`CharactersRAGDBError.__cause__` retained the semantic/path/binary canaries.
Implementation `873972639f` constructs the sanitized error during handling and
raises it only after leaving the active source exception, with cause/context
suppressed. GREEN: focused seam `1 passed`; message-exchange file `12 passed`;
corrected-area cumulative `423 passed, 2 sandbox loopback skips`; full DB area
`1831 passed, 1 Windows-only skip`; Ruff/py_compile/diff-check clean. Task 1
remains **In Progress** and affected AC #2 stays unchecked for independent
re-review.
