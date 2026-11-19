# Scoped Task8 I1 re-review

This is a scoped fix re-review, not a fresh broad feature/branch audit. Read the separate fix1 brief/report and immutable fix package once. Original review task-8-review.md and source40b9 remain immutable. Fix source base72c67b80 is source-identical to40b9; only plan/Backlog/ADR source-observation clarification changed in72c. Confirm that identity in the package receipt.

Findings to verdict:
- I1 Important: _chat_creation_record_locked performs SQLite get_run under _pending_chat_create_lock, potentially blocking actual Close/cancellation. Verify per-call A/B/C removes all such IO while preserving exact record and row predicates, current actor/source/bridge/database/incarnation/conversation/workspace/current-primary/assistant/cancel/Close/revocation checks and atomic approval/grant insertion.
- M1 Minor: explicitly disclose +265 and+233 Task8 child descriptor warnings with exact receipts, separate from prior+223; no cause attribution or suppression.
- M2 Minor: retain the three final58-case budget warnings and exact qualification limits; do not turn them into a claimed clean sweep.

Binding source-observation contract: read a new phase-local row candidate outside the registry lock, then recheck actual record and current runtime ownership under the existing plain lock. The old lock did not serialize independent DB writers. No evidence reuse across human waits, durable writes, owner scheduling or acceptance, no cache/authority boolean/newlock/RLock/fresh-getter substitution, and no claim of universal latest-row atomicity. Verify immutable/corrected direct-DB-only mutation characterization and separately the real final pre-save refusal/no durable row. Canonical runtime Close/Stop/revocation must still deny between B/C; genuine surviving-child positives must remain valid.

Verify report/source/test/maps against isolated logs/XML and hashes. Do not rerun tests unless the code raises a specific unanswered doubt; then only that precise control. No broad startup/provider/schema/encryption replay. Do not inspect private profile/config/DB/cache directories. Use only the previous immutable dependency/get_run excerpts if needed; ask root for a missing exact excerpt rather than scanning unrelated plan workspaces.

The newly discovered named controller cap is a real remaining gate:40b9=32910 and incoming5e=32020 against29367; I1 adds about110. Root records it for separate bounded Task9 structural repair and will not publish/merge with it unresolved. Do not misstate full static or source-gate completion; include any issue introduced by fix1 honestly. Structural work is not yet part of the review package.

Read-only checkout: no tracked/index/HEAD/Git mutations, agents or suites. Save per-finding ADDRESSED/NOT ADDRESSED evidence, new Critical/Important/Minor fix breakage, out-of-scope observations, and both scoped spec/quality verdicts to task-8-fix1-review.md. Return concise verdicts and report path only.
