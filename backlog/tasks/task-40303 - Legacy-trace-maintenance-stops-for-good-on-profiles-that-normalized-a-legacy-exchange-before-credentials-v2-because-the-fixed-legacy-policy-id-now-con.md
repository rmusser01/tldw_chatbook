---
id: TASK-40303
title: "Legacy trace maintenance stops for good on profiles that normalized a legacy exchange before credentials-v2, because the fixed legacy policy id now conflicts"
status: Done
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 02:37
assignee:
- '@claude'
labels:
- follow-up
- console
- trace
- recovery
priority: medium
references:
- tldw_chatbook/Chat/console_trace_legacy.py
- tldw_chatbook/Chat/console_trace_repository.py
- tldw_chatbook/Chat/console_trace_maintenance_runtime.py
- tldw_chatbook/Chat/console_trace_redaction.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security stream review). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**What happens.** A profile that normalized at least one legacy message exchange before 2026-11-06 holds the legacy policy row `...000000000097` tagged credentials-v1. When it later gets a new `message_exchanges` row (trace capture and legacy compatibility writes both on; `console_chat_controller.py:5789-5791`, default off) or still has unmigrated rows, every batch of legacy maintenance rolls back. The runtime loop logs "legacy trace maintenance paused after TraceIdentityConflict" and retries forever (`console_trace_maintenance_runtime.py:62-74`). Those exchanges never migrate, and GC and compaction, which wait for `logical_complete`, never run.

**Why.** `console_trace_legacy.py:44` hard-codes `_LEGACY_POLICY_ID`, and `:256-263` calls `ensure_policy` with `credential_filter_version=CREDENTIAL_FILTER_VERSION`, now `"credentials-v2"` (`console_trace_redaction.py:20`, bfe3fd290a, which did not touch the legacy file). `ensure_policy` (`console_trace_repository.py:2360-2378`) raises `TraceIdentityConflict("policy_id")` when the stored row differs, and the policies table is append-only (an UPDATE trigger aborts, `recovery_core_schema.py:449`). Every other policy uses a fresh id, so only legacy import is affected. No test covers it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A profile whose ledger holds the credentials-v1 legacy policy normalizes a new legacy exchange under credentials-v2 without a conflict, and maintenance reaches logical_complete; a test is red before the fix
- [x] #2 Calls already bound to the v1 legacy policy still bind, and re-mask under v2 on read
- [x] #3 No stored policy row is rewritten (for example, the legacy policy id is per filter version)
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd (`_LEGACY_POLICY_ID` fixed, `ensure_policy` conflict on a stored credentials-v1 row).
2. Write a red test: a ledger holding the v1 legacy policy row, a new `message_exchanges` row, `LegacyTraceMaintenance.run_batch()` reaches `logical_complete` with no `TraceIdentityConflict`.
3. Fix: the legacy policy id is per credential filter version (v1 keeps `...097`, v2 gets its own fixed id); no stored row is rewritten, so no data migration is needed.
4. Test AC#2: a call already bound to the v1 legacy policy still reads, and its read re-masks under v2.

## Implementation Notes

Legacy trace maintenance no longer stalls on a profile whose ledger holds the credentials-v1 legacy policy.
- **Premise held on base 26b2e677dd**: `console_trace_legacy.py` still built every legacy call's policy with the one fixed id `...000000000097` and `CREDENTIAL_FILTER_VERSION` (`credentials-v2`), and `ensure_policy` raises `TraceIdentityConflict("policy_id")` when the stored row differs.
- **Fix**: `_LEGACY_POLICY_IDS` maps each credential filter version to its own fixed id (v1 keeps `...097`, v2 is `...098`), read at normalize time. The policies table is append-only, so no stored row is rewritten and no data migration is needed: a profile that already holds the v1 row keeps it (its calls still bind and read), and the first v2 legacy exchange inserts the v2 row, which later exchanges reuse (`ensure_policy` returns the stored row when it is equal). Nothing runs twice, so the already-normalized profile needs no special path.
- **Test** `Tests/Chat/test_console_trace_legacy_migration.py::test_a_v1_legacy_ledger_keeps_migrating_new_exchanges_under_v2`: captures and normalizes an exchange as the app did before 2026-11-06 (credentials-v1 sanitizers patched in, so its stored response keeps a fake bare `sk_live_` key v1 never masked), then writes a new exchange under the real v2 code. Asserts the batch reaches `logical_complete`, no exchange is left, the v1 row is unchanged and the new call binds a separate v2 row (AC#1, AC#3); the v1-bound call still reads and both viewer profiles mask the key (AC#2); and `_LEGACY_POLICY_IDS` covers exactly `READABLE_CREDENTIAL_FILTER_VERSIONS`, so a future credentials-v3 must add its own id.
  - Red with the fix reverted: `assert (None is not None)` (the batch raised `TraceIdentityConflict`, caught as the runtime loop catches it). Green after.
  - AC#2 negative control: with `project_capture_for_viewer` temporarily returning the stored capture unchanged, the re-mask assertion failed; reverted.
- **Lane**: the node joins the recovered-suites step after TASK-39149's nodes (`.github/workflows/derived-artifacts.yml`); 5.8 s serial with that step's flags (`-p no:xdist --storage-lease-guard --timeout=180`), load 66 on 18 CPUs.
- **Runs** (run.sh, `-n 4`, load 57-70): the legacy migration, legacy normalizer, maintenance parking and trace projection files plus `Tests/Benchmarks/test_console_trace_legacy_migration.py`: 99 passed, 12 failed. The 12 (8 parking/runtime nodes and 3 projection runtime nodes, an `ImportError` of `maintenance_error_must_propagate` inside the runtime task, and the v53 benchmark's `processed_rows 0`) fail the same way at base with the same command (12 failed, 74 passed over the three files that hold them).
- Files: `tldw_chatbook/Chat/console_trace_legacy.py`, `Tests/Chat/test_console_trace_legacy_migration.py`, `.github/workflows/derived-artifacts.yml`.
