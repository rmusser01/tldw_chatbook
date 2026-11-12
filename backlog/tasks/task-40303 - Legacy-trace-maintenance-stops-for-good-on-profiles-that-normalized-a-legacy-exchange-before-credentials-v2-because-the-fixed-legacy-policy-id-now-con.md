---
id: TASK-40303
title: "Legacy trace maintenance stops for good on profiles that normalized a legacy exchange before credentials-v2, because the fixed legacy policy id now conflicts"
status: To Do
created_date: 2026-11-12 19:36
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
- [ ] #1 A profile whose ledger holds the credentials-v1 legacy policy normalizes a new legacy exchange under credentials-v2 without a conflict, and maintenance reaches logical_complete; a test is red before the fix
- [ ] #2 Calls already bound to the v1 legacy policy still bind, and re-mask under v2 on read
- [ ] #3 No stored policy row is rewritten (for example, the legacy policy id is per filter version)
<!-- AC:END -->
