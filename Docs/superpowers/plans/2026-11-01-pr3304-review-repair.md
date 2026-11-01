# PR3304 review repairs

ADR required: no new ADR
ADR paths: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md; backlog/decisions/131-durable-agent-budget-accounting.md; backlog/decisions/156-live-per-run-stream-usage-attribution.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: restore existing physical-call budget, usage, transport and trace contracts; no new persistence or authority boundary.

- [x] Rebase all functional changes onto dev f34510953e5d174e74625fef6fbc33f9ba87f197, preserving current workflow/census/credential/owner changes.
- [x] Add failing real-adapter regressions for Custom/vLLM JSON-on-stream, physical retry admission, and disjoint usage retention.
- [x] Scope automatic admission/settlement separately around each physical generation. Unknown usage remains uncertain.
- [x] Normalize a successful application/json completion in the shared local HTTP adapter without a second request, retaining text, tools and usage.
- [x] Retain each physical usage payload and trace settlement; expose ordered raw logical legs to the bridge, validating each before normalizing/summing. Preserve strict rejection of malformed counts and one-time aggregate publication.
- [x] Exercise capture-on retry using the real trace factory/SQLite, targeted Stop/credential/budget/config/UI contracts, and measure metadata pins.
- [x] Update existing task notes and integration QA with actual evidence and limitations; obtain fresh independent review and address blockers.
- [ ] Refresh live state/head/base/threads/draft flag, publish with exact force-with-lease, merge under the user's hosted-CI waiver, restore protection and confirm merge commit.

Drafts and PR3267/3297 are excluded. No release work. Named custom-hosted fallback, the existing cloud OpenAI exception, and shared first-token deadline remain their existing separate tasks.
