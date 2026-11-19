# PR3312 current-dev integration

Integrate TASK-33621.5/.6 onto `820ea70cc5b14d2809764abf7a15fc132db34430`, preserving current native custody, physical-call trace and usage accounting, exact recovery identity and provider-failure presentation.

ADR required: no new ADR.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/211-ephemeral-provider-failure-presentation.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md.
Reason: compose the existing per-physical-call reservation and conservative recovery contracts with the already implemented stream fallback. No new replay authority, storage schema, provider policy or execution boundary.

1. Rebase the refreshed non-draft source head `9fabd0f9aca6a244545aea112641e8dce236ae19` onto current dev. Preserve both sides of controller, gateway, trace-runtime, session, documentation and test overlaps. Keep current size pins and diagnostic inventory until final source review.
2. Reproduce the independent review's integration finding through the real stream-fallback path: an empty HTTP200 followed by HTTP429 and then a retry reservation failure must not qualify as refusal-only safe replay. Add a both-calls-definitively-refused control and retain mixed answer/unknown-outcome controls.
3. Count physical adapter admissions and their outcomes inside the existing aggregate preparation scope. Record the first call before fallback; keep refusal/dispatch equality and exact-owner single-use proof strict. Preserve every physical call's trace/usage lineage and native lifetime.
4. Run focused provider-error, trace-recovery and fallback checks, including relevant real mounted paths with only the HTTP edge replaced. Compare unrelated existing failures with exact dev when needed. No full sweep, hosted CI or size-limit gate.
5. Verify the ADR029 diagnostic inventory against integrated source; regenerate only reviewed changes. Record current evidence separately from historical live results. Retain TASK-33621.70-.73 and the incomplete TASK-33621.6 criterion explicitly.
6. Obtain independent requesting-code-review signoff on the final integrated head. Refresh live head/base/draft/threads/auto-merge, publish with exact force-with-lease, merge under the user's CI waiver, verify actual merge and identical branch-protection restoration. Drafts and PR3267/3297 remain excluded; no release work.
