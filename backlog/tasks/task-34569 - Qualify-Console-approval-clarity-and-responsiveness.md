---
id: TASK-34569
title: Qualify Console approval clarity and responsiveness
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 06:42'
updated_date: '2026-10-06 20:38'
labels: []
dependencies:
  - TASK-34564
  - TASK-34565
  - TASK-34566
  - TASK-34567
  - TASK-34568
documentation:
  - >-
    Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
  - backlog/decisions/221-console-approval-interaction-and-feedback.md
  - Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the reviewed permission interaction and measure the resulting native and browser responsiveness before claiming completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single, mixed, grouped, raw-shell, same-tool and grant-failure journeys preserve exact decisions through actual invocation paths.
- [ ] #2 Actions, warnings, focus and disclosures remain usable at 80x24, 120x40 and 170x48 with Inspect open and closed in dark and light themes.
- [ ] #3 Qualified native and browser measurements compare identical boundaries and check the 100 ms feedback and 200 ms actionable-card p95 targets on recorded hardware.
- [x] #4 Targeted checks, generated artifacts, documentation and evidence reflect the shipped behavior; unqualified metrics and any remaining measured gaps are explicitly reported.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 6 Qualify the complete interaction and publish evidence

**Backlog:** TASK-34569. **ADR:** ADR-221 with the existing UI and security decisions. **Consumes:** Tasks 1-5 and their passing targeted receipts; Task 1's comparable baseline.

**Files:** Create `Tests/UI/test_console_approval_ux_journeys.py` and `Tests/Chat/test_console_approval_scope_journeys.py`. Extend the Task 1 probe with new owner observations while retaining old boundary names. Update `Docs/User_Guide/console/agent-runs-and-tools.md`, `Docs/User_Guide/Console.md`, `Docs/User_Guide/mcp.md`, and the existing `scripts/regen_approval_card_svg.py` fixture only where its controls/copy changed. Regenerate `Docs/User_Guide/images/console/approval-card.svg`. Save actual evidence and qualification notes under the already declared QA directory.

**Interfaces:** Produce final timing receipts in the same Task 1 schema with source hashes, actual sample distributions and qualified/unverified status. No new runtime interface is introduced here.

- [ ] **Step 1: Write the integrated regressions.** Real-gate journeys must pin once, denial reason, exact-profile temporary grants, Default inheritance, raw chat/Disarm lifetime, ordinary and explicit-review batches, grouped arguments, same-tool matching withholding, and grant failure. Assert actual dispatch/refusal and stored/cache state, not selected labels. UI journeys exercise painted actions, Alt+A then Enter, stale disclosure actions, same-round snapshot replacement, late pages/summaries, navigation, FIFO promotion, timeout and rapid double clicks. Use the real owners with private SQLite and harmless invocation fixtures. Run the two new files to establish meaningful failures if anything is missing.
- [ ] **Step 2: Qualify targeted automated behavior.** Run the new journeys and the task-specific tests from Tasks 2-5. Run token, component-pattern and CSS bundle guards. Check formatter/linter only on changed Python files. Since new imports and CSS matching can affect first-use speed, run the relevant existing UI latency/startup guard cases; do not broaden into a full UI or application suite. Attribute any baseline failure by its actual cause and preserve it rather than lowering assertions.
- [ ] **Step 3: Inspect native and browser journeys.** Use the real private app and served UI, harmless local fixtures, and actual keyboard/mouse gestures. Inspect all six size/Inspect combinations in both theme families. Confirm complete targets and source scope, disabled/applying readability, pinned actions, large-page continuation and preserved focus. Read the entire screen after each action for contradictory run/composer/card copy. A widget query or export alone cannot replace native/browser inspection.
- [ ] **Step 4: Measure the final flow.** Repeat at least 40 warm single/batch/raw-deny/large scenarios using the same baseline hardware, load, transport and boundary definitions; report first-use costs separately. Record input queueing and complete last-output-to-card cost as well as the two local targets. Publish separate native/browser results and cross-clock error bounds. If data or clock qualification is missing, label that result unverified. If a measured local target is missed, add the exact attributed reproduction/files/fix to the relevant task and this plan before a repair, then requalify; do not claim Done from faster-looking copy.
- [ ] **Step 5: Update user documentation and artwork.** Document immediate actions, More options, actual temporary/persistent scopes, counted batch consent, staged Apply and feedback/failure meanings. Replace old Approve all then Submit instructions. Run `Tests/Scripts/test_regen_approval_card_svg.py` and inspect the regenerated SVG. An optional owner-led comprehension check asks what action, target and scope a representative card conveys; no messages to other users are sent by the agent without authorization.
- [ ] **Step 6: Review and close out.** Self-review the complete diff, then use the requesting-code-review skill for one independent final review. Resolve material findings and rerun only affected checks. Update ACs, Implementation Notes, relevant docs, ADR links and real evidence; add a lesson only if a specific incident generalizes. Use the Backlog CLI to mark each verified task Done. Commit owned files only. PR creation, publishing or merging follows any later user instruction; this plan does not require a full sweep or imply an unrequested release.

ADR required: yes
ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md
Reason: Existing ADR221 with ADR150/161/195/210 governs verification/documentation of the implemented interaction; no new runtime interface or permission policy.

Execution clarification: author independent user-guide/artwork and integration tests while Task5 final commit/review completes; run final integration/qualification only against its reviewed commit. Native/browser presented-frame gate remains open under the recorded capability limitation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented captured-card guides and reproducible artwork with production theme fallback, destination-validation/bare-theme regressions, finite targeted launcher additions and final content-free receipts under Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/. New journeys connect painted consent/controller arbitration to real permission/cache owners and private SQLite using harmless external transport doubles (scope9/UI4). Existing stale-generation, disclosure, Details, FIFO/deadline and Alt+A composer cases are credited separately. Final targeted presentation18/feedback29/Details18/interaction21/artwork5/recorder16/bundle5 pass. Tokens7pass1existingfail, component13pass5existingfails with matching BASE messages; startup lazy imports2pass, optional-work wait fails identically on BASE. Owned root Ruff/format pass. Native/browser presented-frame observers unavailable, zero qualified samples, actual Windows local/virtual dispatch root-pin failures and complete transport visual matrix remain open. No freeze/speed claim or full sweep. ADR-221; all13 rulings/costs preserved; final independent review pending; In Progress, not Done.

Final bundled fix4542052fb5..9d8a1bd29a: batch broader scopes behind More options, neutral Review individually route, bounded complete-identifier preview with explicit omissions and complete captured target Details. Actual80x24 Console clipping reproduction repaired with existing batch-only4/2 viewport tokens and three-action toolbar. Interaction25/Details19/ownership25/compact1/journeys4/budget28/CSS5/token1/refinement1 passed. Independent scoped review all3 addressed, no new material issue; hashes verified. Evidence Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/final-review.md and final-fix/. Native/browser timing/fullmatrix, actual Windows dispatch and broader baseline failures remain open; In Progress, not Done. ADR-221.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Originally TASK-34416 in the reviewed approval checkout. Renumbered to TASK-34569 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.
