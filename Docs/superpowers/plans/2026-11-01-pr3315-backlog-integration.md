# PR3315 backlog integration plan

> For agentic workers: use superpowers:executing-plans for this authorized integration and superpowers:requesting-code-review for independent final review.

Goal: integrate TASK-36381 Settings test isolation and missing Tools inspector guidance onto current dev without dropping upstream repairs.
Architecture: reuse existing private-profile children and module-local mocks. Keep the production recovery/egress guards and Settings ownership contracts intact.
Tech Stack: Python, pytest, Textual, Git.
Spec: backlog/tasks/task-36381 - Settings-provider-and-discovery-test-files-that-no-PR-lane-runs-are-red-on-dev.md

ADR required: no new ADR.
ADR paths: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/240-mcp-destination-retirement-tools-hub-in-settings.md.
Reason: existing profile isolation and Settings ownership boundaries; test repairs and inspector copy only.

## Constraints and review focus
- Source 98cd750928af894f5fe8c1044d07ed96dab20211; base 3e55f1f3aa1396ab0ae2fe29a19f397cda88c239.
- Preserve dev Dreams/Guardian ownership, exact search ranking, token assertions, scrolled clicks, and selected-profile diagnostics seeding.
- Shared bootstrap tests must not mutate config; private seeding must verify child ownership.
- JSON fakes stay module-local; DNS mocks preserve egress checks.
- Preserve census entries/order and append the eight new selections; size limits and hosted CI are waived by user.
- Drafts and PR3267/3297 excluded. No release or main promotion.

## Integration and verification
- [x] Rebase source commits onto the exact base; resolve overlaps by reviewing each difference, keeping stronger dev assertions.
- [x] Reconcile census floor/list and measured Settings size pin. Keep all source failure history and append current integration evidence.
- [x] Run the endpoint probe and local discovery suites, discovery journeys, changed hub paths and Tools guidance coverage through the private-fixture runner, serial mounted UI execution. Run census contracts and scoped static checks.
- [x] Investigate any failure against exact base where necessary; retain causal evidence and do not weaken assertions.
- [ ] Commit final QA/task notes, obtain independent review of the immutable head, and publish with exact force-with-lease after live draft/head/thread checks.
- [ ] Verify actual merged commit and identical protection restoration, then continue oldest eligible backlog.
### Frozen targeted-run follow-through

The 234-case frozen run completed with 229 passes and five failures. Two discovery expectations predate the now-paired localhost/IP defaults (both reproduced on dev); pin all twelve ordered candidates and the exact two llama.cpp URLs while still forbidding Ollama fallback. Two schedule sizes read too early or awaited a cancelled unrelated prompt-summary worker; wait for each intended visible gate result instead, retaining painted, button, live-projection and durable-config assertions. The appearance save was still saving at the test-only five-second deadline; retain its save/durable assertions with a bounded thirty-second functional completion wait, without claiming latency qualification. Remove the unused first seed result only. No production behavior or retirement guard changes. Rerun the local discovery module and these three UI cases; preserve all other frozen test definitions byte/AST-equivalently.

The first repair rerun passed all59 discovery cases and appearance persistence, but exposed schedule hit-testing failures after layout changed (one explicit click returned False, one never enabled). Reuse the existing scrolled-click helper for both actions and query current mounted controls after each completed save; keep both viewport sizes and all painted/durable assertions.

### Final repair result

The scrolled mouse and center-hit attempts did not reliably activate the toggle; an instrumented run passed, so they do not establish a production root cause. Existing guidance documents the Pilot coordinate lifetime. Use the current visible, enabled, focused button and its real Enter binding, preserving every paint/runtime/durable assertion and both viewports. Both final cases pass. The earlier cancellation/read-timing explanation was provisional and is not treated as a proven mouse cause. The two sibling mouse failure-path tests remain unchanged. See the integration QA for frozen runs, incremental passing evidence and explicit limits. No new production change was needed.
