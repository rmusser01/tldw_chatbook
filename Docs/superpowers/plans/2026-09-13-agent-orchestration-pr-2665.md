# PR2665 integration and review implementation plan

**Goal:** Integrate the remaining orchestration work with current dev and resolve every Qodo comment with checked evidence.
**Architecture:** Keep ordinary selected-authority Git, configured destination merge identity, protected uncertain checkouts, bounded webhook admission and existing approval/runtime ownership. Review changes are minimal corrections within existing decisions.
**Tech stack:** Python >=3.12, Textual8, SQLite, ordinary Git.
**Scope:** TASK13154 review criteria; Qodo review https://github.com/rmusser01/tldw_chatbook/pull/2665#issuecomment-5651226487.

ADR required: no new ADR for mechanical integration and routine corrections.
ADR paths: backlog/decisions/153-bounded-run-webhook-delivery.md; backlog/decisions/154-agent-denial-streak-boundary.md; backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/148-console-run-hooks.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md.
Reason: preserve accepted runtime and authority boundaries; improve refusal clarity, validation coverage and bounded delivery. Amend an existing ADR before any material lifetime-contract correction discovered during testing.

- [x] Preserve original head08037e50; rebase84 commits onto dev3e28c9e7. Result73fba740 retains upstream run hooks, fork/new-chat disclosure, confirmation state and modular stylesheet loading. The reviewed candidate was subsequently replayed onto documentation-only dev a51edff97f; production/test bytes were unchanged. Generated diagnostic inventory has been reviewed and regenerated.
- [x] Webhook finding2: reproduce stalled lookup followed by queued healthy delivery using releasable gates; normalize captured timeout and apply asyncio.timeout across the full coroutine. Assert queue progress before fallback cleanup and exact owned-thread joins afterward. Preserve explicit OS DNS cancellation limitations.
- [x] Recovery findings1/10/11/12/13: retain configured user identity (no fabricated author), prove missing identity before source/destination mutation and improve actionable refusal. Add malformed run-ID no-effect cases, trace selected-authority and strict-Allow validation, and disclose retained checkout on post-create admission failure. Reject review suggestions that weaken accepted authority or pretend failed persistence succeeded.
- [x] Configuration findings3/9: use TLDW_AGENTS_DENIAL_CIRCUIT_BREAKER_LIMIT before TOML and the shared named default; parameterized precedence/invalid/zero cases prove behavior.
- [x] Maintainability findings4-8: concise accurate API/class docs, recovery return type and import grouping. No behavior changes or new wrapper hierarchy for style-only feedback.
- [x] Verify rebase seams: affected run-hook/review/denial precedence, model tool names, fork/new-chat and preview parity, worktree lifecycle/SQLite/Git, Settings and token governance. Preserve exact requesting-run attribution for worktree ApprovalRequested hooks and explicit chat-create flags in disposable previews. Canonicalize app-created temporary paths before recording them, never loaded authority. Use repository test isolation, owned basetemps and existing interpreter. No full-suite opt-in is inferred.
- [x] Pin the new worktree scope index against the actual paged query with sqlite_stat1 absent; add the missing index-census row. Regenerate and semantically review diagnostic inventory; resolve generated CSS according to current upstream loading/build contract. Run scoped static comparisons and guard checks without loosening baselines. Record all failures accurately.
- [x] Independent final review, confirmed corrections, task/evidence/PR update and exact force-with-lease publication; all13 original review threads addressed and resolved.
- [ ] Observe the required remote gate on the final published head, confirm current dev/PR state, then merge. Implementation completion does not assert this final operation has occurred.

All test evidence and raw Qodo comments are retained under .superpowers/sdd/2026-09-13-agent-orchestration-pr-2665; durable final dispositions belong in Docs/superpowers/reviews/2026-09-13-agent-orchestration-pr-2665.md. No evidence/worktree cleanup or foreign process retirement.
