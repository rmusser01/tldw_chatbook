# MCP component review

## PR2722 merged; saved raw-response review next — 2026-09-22

[PR2722](https://github.com/rmusser01/tldw_chatbook/pull/2722) merged as `ed2a579062` after owner visual approval, all six Qodo findings resolved and final-head Qodo zero findings. CI passed1,152 contract and123 admission cases with one expected failure; all remaining checks pass. Its actual tree exactly matches the qualified head/candidate and native source hashes. TASK-32836 is Done. [Closeout](../qa/2026-09-18-mcp-inspector-guidance/merge-closeout/README.md).

A fresh follow-up branch starts from that merged dev. Next is saved PR2719: replace stale result details after local validation and keep raw responses readable in compact inspectors. Its scrolling prerequisite already merged in PR2770/TASK-32882; older PR2718 is historical. Requalify the saved slice against current runtime and obtain its own visual approval before merge. Other MCP and destination reviews remain open; the heartbeat stays paused. Earlier pending/next checkpoints are historical.

## Current continuation — PR2722 inspector guidance

Approved PR2721 merged at `3722a857480b94b30fd4755f3f8e3002bd163ec3` with passing current-head CI, zero Qodo findings and an actual tree identical to its qualified candidate. TASK-32835 is Done; [closeout](../qa/2026-09-18-mcp-audit-filters/merge-closeout/README.md).

PR2722 / TASK-32836 resumes from that actual merge. Readiness badge, explanation and actions now share the existing detail visibility owner. [201 targeted passes and 16 current native captures](../qa/2026-09-18-mcp-inspector-guidance/current-dev/README.md) qualify the bounded slice. One inherited dimension ratchet is documented; no allowances changed. Both ledger conflict histories were retained, with the saved inspector checkpoint labeled historical; no product conflict occurred. The owner has approved the gallery. [Closeout qualification](../qa/2026-09-18-mcp-inspector-guidance/current-dev/approved-closeout/README.md) records a conflict-free rebase onto dev5cdc9ddd, a test-only CI mouse-scroll repair, 590 targeted passes and unchanged inspector captures. This slice remains In Progress pending current-head CI/Qodo and final dev review. No full suite ran; the wider workstream remains open and the old heartbeat remains paused.


## PR2721 current-dev Audit filters ready for visual review — 2026-09-21

The saved compact-filter slice now includes merged PR2720 dev `7a758b8196`.
Product/source CSS apply without conflicts; both review-document histories remain.
[Current qualification](../qa/2026-09-18-mcp-audit-filters/current-dev/README.md):
294 targeted passes, one inherited dimension-governance failure (all twelve flagged
declarations unchanged from dev), nine artifact guards, no new Ruff diagnostics
and independent review clear. Eighteen current native captures verify readable
filters, keyboard menus, retained values and last-row access in both themes/sizes,
with clean private-profile lifecycle and source provenance.

TASK-32835 remains In Progress pending its own fresh visual approval, current-head
CI, accumulated Qodo review and final dev/conflict checks. PR2720 is merged and
TASK-32834 is Done. PR2722 inspector guidance ownership remains the next separate
slice; other component/destination reviews remain open. Earlier checkpoints below
are historical, and the post-PR2707 heartbeat remains paused.

## PR2720 merged; compact Audit filters resumed — 2026-09-21

PR2720 merged as `7a758b8196` after owner approval and resolved Qodo review.
CI passed 1,152 main and 123 admission cases (one expected failure). A performance
PR landed eight seconds before merge; the actual tree was separately qualified
with 382 targeted passes, all nine artifact guards, independent integration review
and twenty approved-equivalent native captures. Five upstream size-budget failures
are documented separately; Audit introduces none. TASK-32834 is Done.
[Exact closeout](../qa/2026-09-18-mcp-audit-selection/merge-closeout/README.md).

Saved PR2721 / TASK-32835 resumes from that merged state on a fresh branch.
Both documentation histories are retained; product and source CSS merge cleanly.
Current tests, native qualification and its own visual approval remain before its
merge. Earlier filter evidence below is historical. PR2722 guidance ownership and
other component reviews remain separate; the heartbeat remains paused.

## PR2769 merged; integration repair complete — 2026-09-21

[PR2769](https://github.com/rmusser01/tldw_chatbook/pull/2769) merged as
`85e7153158` after successful current-head CI (1,152 main passes, 123 admission
passes and one existing expected failure), zero accumulated Qodo findings,
resolved review threads and independent review of the final import repair.
The actual tree exactly matches the locally tested combination with dev
`6622115c27`; no conflicts or UI changes were introduced. TASK-32825 is Done.

The compact Test Tool inspector is the next separate visual review, TASK-32882:
510 targeted cases, four clean native dark/light compact/wide journeys and
16 inspected captures. It remains In Progress pending its own visual approval,
PR CI/review and final current-dev checks. Other MCP and destination reviews
remain open. Earlier checkpoints below are historical.

## PR2712 merged; bounded integration repair — 2026-09-21

[PR2712](https://github.com/rmusser01/tldw_chatbook/pull/2712) merged into dev
as `4b61a5ca8f` after fresh visual approval, successful final-head CI, resolved
Qodo review with zero findings, and conflict-free integration checks. The 14
captured native source hashes match the actual merge.
[Receipt](../qa/2026-09-18-mcp-server-actions/current-dev/merge-closeout.json)
records separate verification for the later dev changes.

PR2744 landed at merge time and introduced a Workbench size limit: post-merge
architecture verification found 66 passes and one ten-line overage. A bounded
follow-up moves existing mode-dependent inspector clearing into MCPInspector,
preserving worker ownership, cancellation group and await order; the ratchet
is tightened to 6,760 lines. All 106 targeted checks pass and independent review
is clear. TASK-32825 remains In Progress until this integration follow-up merges.

Compact Test Tool inspector reachability at 80×24 is next on a fresh follow-up
branch. Other MCP and destination reviews remain open; older checkpoints below
are retained as history.


## PR2712 server actions ready for fresh visual review — 2026-09-21

Existing draft [PR2712](https://github.com/rmusser01/tldw_chatbook/pull/2712)
resumed on merged PR2716 dev `9cf5ba67b2`, without conflicts. Toolbar actions
retain displayed profile identity and reject retired/unavailable controls.
Mode departure revokes queued presses immediately; delayed refreshes preserve
newer input, and overlapping rendering cannot interrupt accepted deletion.
Compact actions use a token-backed two-column grid; wide layout is unchanged.

[Current qualification](../qa/2026-09-18-mcp-server-actions/current-dev/README.md):
232 distinct targeted cases, eight artifact guards, no introduced Ruff findings,
independent review clear, four native theme/size journeys and 16 inspected captures.
Real private profile persistence verifies Keep/Escape, ignored retired presses
and exact-target deletion. Clean shutdown, healthy private databases, unchanged
defaults, matching provenance and zero network attempts pass. The obsolete saved
runner is retired behind its immutable reference.

TASK-32825 remains In Progress. Fresh owner visual approval, current-head CI,
accumulated Qodo review and current-dev/conflict checks gate merge. PR2716's
approval does not approve this slice. Compact rail truncation, compact Test Tool
reachability, Audit and other screens remain separate bounded reviews.

## PR2716 merged; server-action review resumes — 2026-09-21

[PR2716](https://github.com/rmusser01/tldw_chatbook/pull/2716) merged into dev
as `9cf5ba67b2` after owner visual approval, all 3 resolved Qodo
threads, and zero remaining Qodo bugs/rule violations on the final head.
Current-head CI passed 1,152 main cases and 123 admission cases with one existing
expected failure; artifact, CSS, backlog and performance guards passed. The
actual merge tree matches the tested current-dev merge candidate exactly.

TASK-32831 is Done. The bounded result-error repair has 87 distinct targeted MCP
passes and the approved wide dark/light failure/retry evidence. Later changes
repair CI startup/timing and add a client docstring/direct model tests; executable
MCP behavior and all native evidence remain identical to the approved head.
The compact Test Tool limitation remains explicitly unqualified.
[Merge receipt](../qa/2026-09-18-mcp-tool-errors/current-dev/merge-closeout.json).

Existing draft PR2712 is the next bounded slice: server-action target ownership
and compact toolbar usability. Resume it from this merged dev on a fresh branch,
with current regressions, native qualification and its own visual approval.
Other inspector, Audit, permission and destination review limits stay open.

## PR2716 tool-error qualification — 2026-09-20

PR2714 merged into dev as `e4096e2059` after owner approval and all CI/review gates;
TASK-32830 is Done. The fresh follow-up branch integrates existing PR2716 without
conflicts. Server-reported tool errors now reach the existing failed result and
Audit path; successful retry uses the same connection. Production changes cover client result handling
and strict flag validation in the existing shared input-validation module, with
existing ADR-111/161 and no new architecture.

75 distinct targeted cases, eight artifact guards and independent reviews pass.
Qodo's malformed-flag finding is repaired: eight real-stdio cases reproduce the
false success before repair, then reject malformed values without body leakage
and successfully retry on the same session.
Fresh native dark/light 170×48 passes with eight inspected SVGs, real error/success
audit records, clean app/child shutdown, healthy private databases, unchanged
defaults/sentinels and matching source hashes. The separate 80×24 attempt fails
before execution because Test Tool is offscreen even after focus/scroll; it exits
1 with verified clean shutdown. Compact execution remains unqualified, and its
layout repair stays separate from this client-only PR.

[Current evidence](../qa/2026-09-18-mcp-tool-errors/current-dev/README.md).
Owner approved the PR2716 gallery at `19ceab4dc3`. A test/CI-only follow-up
repairs the splash-dependent watcher fixture and extends the bounded serial job
to 30 minutes; production/native hashes are unchanged. Its 123 admission cases
(plus one existing expected failure), 26 CI contracts and eight artifact checks
pass. TASK-32831 remains In Progress pending current-head CI, accumulated review
and merge. Full Audit navigation, raw-response expansion,
complete permission flows and remaining screen reviews stay open.

Qodo's accumulated follow-up added the public tool-call contract and twelve
pure boundary tests. All 31 affected cases pass, bringing the distinct MCP
inventory to 87. Independent review and baseline-relative static checks pass;
executable client AST is unchanged apart from the docstring, so the recorded
visual approval applies. [Evidence](../qa/2026-09-18-mcp-tool-errors/current-dev/review-followup/README.md).

## PR2714 merged; PR2716 resumed — 2026-09-20

PR2714 merged into dev at `e4096e2059` after owner visual approval, zero open Qodo
findings, all nine review threads resolved, and current-head CI (1,152 Fast Lane
cases plus artifact gate). The conflict-free final rebase preserves the approved
MCP implementation and native source hashes. TASK-32830 is Done; its current QA
merge receipt records the exact tree, tests and review evidence.

PR2716 tool-result errors resumes separately from this merged state. Further
compact inspector/notification work and remaining screen reviews stay open.

## Current checkpoint — PR2714 owner approval and review follow-ups

Owner approved the catalog-refresh gallery at `f8d1731abc`. PR2714 is rebased
without conflicts onto dev `7bfd330046`. Qodo follow-ups validate fixture paths
and malformed requests, document the API, add isolated tests, use canonical IDs
for cleanup and report failed temporary-session teardown honestly. The real
control plane records `ok=False` while an owned process remains alive; retry
reaps it. The original race claim was independently disproved and Qodo dismissed it.

[Current evidence](../qa/2026-09-18-mcp-connection-refresh/current-dev/README.md)
includes 47 distinct affected cases, seven artifact guards, independent review,
and final native004's four cells/32 captures with exact source hashes and clean
shutdown/defaults/database/lock checks. The approved controls/styles are unchanged;
compact toast overlap, toolbar clipping and catalog scrolling remain disclosed.
Current-head CI/Qodo and final dev review still gate merge. PR2716 tool execution
is the next saved bounded slice; it has not been integrated. PR2707 heartbeat
remains paused. Older checkpoints below are historical.

## Current checkpoint — PR2713 merged; catalog refresh resumes

[PR2713](https://github.com/rmusser01/tldw_chatbook/pull/2713) merged as
`5e0f9f82c3` on 2026-09-20 after owner approval of twelve native captures,
zero remaining Qodo findings, all five review threads resolved, and current-head
CI: 1,152 Fast Lane cases plus artifact, performance, CSS and backlog checks.
The final merge tree exactly matches CI's merge with current dev `c768376092`.
[TASK-32880 closeout receipt](../qa/2026-09-18-mcp-lifecycle-cancellation/current-dev/merge-closeout.json).

Existing draft [PR2714](https://github.com/rmusser01/tldw_chatbook/pull/2714)
resumes from this merged dev on a fresh branch. TASK-32830 covers actual local
catalog discovery during refresh, original connection state, permission denials,
failure/retry and owned-process cleanup. Current integration has 49 distinct focused passes, seven artifact guards,
independent review, and four real native cells with 32 inspected captures.
[Current QA](../qa/2026-09-18-mcp-connection-refresh/current-dev/README.md).
Immediate compact notifications temporarily cover inspector actions; feedback
captures preserve this UI follow-up alongside PR2712 toolbar clipping and
below-fold catalog scrolling. Current-head review/CI and owner visual approval
remain before its merge. Connected
execution (PR2716), compact actions (PR2712) and wider screen reviews remain
separate. Earlier checkpoints below are historical.

## Current checkpoint — PR2713 cancellation integration ready for review

Saved [PR2713](https://github.com/rmusser01/tldw_chatbook/pull/2713) resumes on
current dev `45d67a6704`. Its old TASK-32829 number collided with an unrelated
landed task; this work is now TASK-32880. Cancellation retains per-server
ownership until cleanup and final readiness collection finish, rejects retired
Cancel controls, and supports immediate native worker completion.

[96 targeted passes, seven guards, independent review and twelve inspected
native captures](../qa/2026-09-18-mcp-lifecycle-cancellation/current-dev/README.md)
qualify the bounded integration. The one conflict retains both the current
recovery-token checks and the saved displayed-operation identity. No CSS changes.
Owner visual approval is recorded. The conflict-free latest-dev rebase leaves
MCP source and all twelve captures unchanged; 35 targeted reruns and native
verification pass. Current-head CI/review remain before merge.
Connected catalog refresh (PR2714), execution (PR2716), compact server actions
(PR2712), and the other saved MCP slices remain separate. Earlier checkpoints
below are historical; PR2707's heartbeat stays paused.

## Current checkpoint — PR2759 merged; connected runtime next

[PR2759](https://github.com/rmusser01/tldw_chatbook/pull/2759) merged into dev at
`8d110a06e384add07a72e4d96f7e1e0aa4b7a144`. Its actual tree equals the verified
head. Owner visual approval, 116 distinct local targeted passes, 16 inspected
native captures, current-head CI (1,152 Fast Lane passes and all required guards),
resolved Qodo findings and unchanged dev/conflict review qualify this bounded
compact root-review repair. The two reproduced baseline test failures remain
recorded; no full suite ran. [Merge receipt](../qa/2026-09-20-mcp-compact-review/pr2759-closeout.json).
TASK-32879 is closed.

The fresh `codex/mcp-connected-runtime-review-20260920` branch begins at the
merged dev state. Next bounded scope is the MCP connection lifecycle and
connected inspector journey. Other unqualified controls/screens remain open;
no additional screen is declared qualified. PR2707’s heartbeat stays paused.
Earlier checkpoints below are historical.

## Current checkpoint — PR2759 visual approval received

[PR2759](https://github.com/rmusser01/tldw_chatbook/pull/2759) / TASK-32879
repairs the restored MCP root review at 80×24: keyboard-scrollable,
left-aligned complete paths and persistent Cancel/Use fresh MCP defaults actions.
[116 targeted passes, two reproduced dev failures, seven preflight guards and
16 inspected native captures](../qa/2026-09-20-mcp-compact-review/README.md)
qualify this bounded slice. The catalog test now waits for actual review completion.
Existing approval ownership and native recovery writes remain unchanged.
The owner approved the [visual gallery](../qa/2026-09-20-mcp-compact-review/GALLERY.md).
The Qodo follow-up adds a return docstring and clearer scroll-progress assertions;
product behavior and styles remain identical to the approved native capture.
All four affected tests pass again. Current-head CI, accumulated review and
current dev/conflict review remain before merge. Connected-runtime journeys and
other unqualified MCP controls remain separate; PR2707’s heartbeat stays paused.

## Current checkpoint — PR2757 merged; compact review next

[PR2757](https://github.com/rmusser01/tldw_chatbook/pull/2757) merged into dev at
`62d43190ce3ca21bda0f0ff03e5148971eedc7ca`. Its actual tree equals the verified
head. Current-head CI passed, including 1,152 Fast Lane tests; 316 local targeted
checks, native lifecycle, owner visual approval and accumulated review qualify
the change. [Merge receipt](../qa/2026-09-20-mcp-inspector-refresh/pr2757-closeout.json).
TASK-32823 and the matching legacy TASK-235 are closed.

A fresh follow-up branch starts from that merge. Next bounded scope is compact
MCP action reachability and long-path readability; connected-runtime journeys
remain separate. No additional screen is declared qualified. PR2707’s heartbeat
remains paused. Earlier checkpoints below are historical.

## Current checkpoint — PR2757 approved closeout

The owner approved [PR2757](https://github.com/rmusser01/tldw_chatbook/pull/2757)
and its eight-capture gallery. Conflict-free rebases include current dev
`6e9e94c794`; MCP product code, styles and native source hashes remain unchanged.
[All 13 catalog-refresh regressions pass again](../qa/2026-09-20-mcp-inspector-refresh/approved-integration.json).
Qodo review, current-head CI and final live-dev/merge-tree verification remain
before the authorized merge. [Qodo follow-up](../qa/2026-09-20-mcp-inspector-refresh/qodo/README.md)
now includes changed permission facts in refresh equality, completes API docs
and verifies the real mount boundary. All 316 current-source targeted cases
and the fresh native journey pass; approved visuals differ only by caret blink.
Remaining component scope is unchanged.
Earlier checkpoints below are historical.

## Current checkpoint — PR2730 merged; inspector refresh resumed

[PR2730](https://github.com/rmusser01/tldw_chatbook/pull/2730) merged at
`802809947b0161f9d589d73412fb497516a87f27` after owner visual approval,
current-head CI and resolved Qodo review. Its actual tree matches the verified
head. [Merge receipt](../qa/2026-09-20-mcp-inspector-refresh/pr2730-closeout.json).

[Draft PR2757](https://github.com/rmusser01/tldw_chatbook/pull/2757) resumes
saved TASK-32823 on a fresh branch from that merge. Selected MCP
inspector details now follow catalog changes while equal definitions retain
argument drafts, cursor, focus and previews. Synchronous form ownership rejects
late previews during teardown; newer selection and focus win. [475 targeted
passes, one reproduced Workflows baseline failure, seven guards, independent
review and eight inspected native captures](../qa/2026-09-20-mcp-inspector-refresh/README.md)
qualify this bounded follow-up. Its own current-head CI/Qodo and final visual
approval remain before merge. Compact/long-path presentation, connected-runtime
journeys and remaining screens stay open. PR2707's heartbeat remains paused.
Earlier checkpoints below are historical.

## Current checkpoint — PR2730 approved closeout

The owner approved PR2730’s Console gallery and merge. Its rebase onto current
dev `d1a0649cd2` (PR2754 Library/Artifacts integration) had no conflicts and keeps
the approved approval logic and Console controls unchanged. [211 targeted passes,
18 reproduced baseline failures, seven guards and fresh native comparison](../qa/2026-09-19-approval-action-ownership/CLOSEOUT.md)
qualify the combined state. Only upstream main-navigation pixels changed.
Current-head CI/review and actual merge-tree verification remain before the
authorized closeout. Saved MCP inspector work resumes after confirmed merge.
PR2707’s heartbeat stays paused; earlier checkpoints are historical.

## Current checkpoint — PR2728 merged; PR2730 integrated

[PR2728](https://github.com/rmusser01/tldw_chatbook/pull/2728) merged at
`de10a62e67124a2b21b78edf1a4887cea03ff139` after owner visual approval,
current-head CI and accumulated review; its actual tree equals the approved tree.
[Merge receipt](../qa/2026-09-19-approval-action-ownership/integration/pr2728-closeout.json).

Saved PR2730 / TASK-32841 is integrated onto that merged dev. Both report histories
are retained; there were no product conflicts. [210 distinct targeted passes,
19 reproduced baseline failures, seven guards and eight inspected current-source
native captures](../qa/2026-09-19-approval-action-ownership/CURRENT-DEV-REVIEW.md)
qualify the bounded Console approval ownership repair. Current-head CI/review and
this PR's own final visual approval remain before merge. MCP inspector refresh,
long-path/compact presentation and connected-runtime journeys remain follow-ups.
PR2707's heartbeat stays paused. Earlier checkpoints below are historical.

## Current checkpoint — PR2727 merged; PR2728 integrated

[PR2727](https://github.com/rmusser01/tldw_chatbook/pull/2727) merged at
`ebee42fab8a7f55def6a03fc7b3301935940eebf` after owner visual approval,
current-head CI and accumulated review. Concurrent PR2751 changed unrelated
Chatbook/audio/Evals components during merge. The actual merged app passed all
85 affected checks and the 16-capture native journey; MCP sources/styles match
the approved head. [Actual merge receipt](../qa/2026-09-19-mcp-recovery-catalog/integration/pr2727-closeout.json).

Saved PR2728 / TASK-32840 is integrated onto that verified merged state.
The report conflict retains both histories; the diagnostic inventory was rebuilt
after reviewing its sole added redacted warning. No product conflict occurred.
[182 targeted passes, seven guards and eight inspected native captures](../qa/2026-09-19-mcp-recovery-catalog/CURRENT-DEV-REVIEW.md)
qualify passive catalog repopulation and accurate retained-history guidance.
This PR's own current-head CI/review and final visual approval remain before merge.
Long-path presentation, inspector refresh, connected-runtime journeys and saved
Console approval draft PR2730 remain follow-ups. PR2707's heartbeat stays paused.
Earlier checkpoints below are historical.

## Current checkpoint — PR2749 merged; PR2727 integrated

[PR2749](https://github.com/rmusser01/tldw_chatbook/pull/2749) merged at
`65d79cc2b7bd8b7b1d86ee87cc31f1823475ab3a` after owner approval, current-head
CI and accumulated review. The actual merge tree equals the verified head.
[Merge receipt](../qa/2026-09-18-mcp-restored-roots/integration/pr2749-closeout.json).

Existing PR2727 / TASK-32839 is rebased onto that merged state. Both report
histories are retained; no product conflict occurred. Four new real-owner
keyboard regressions reproduce and fix late completion overwriting a newer
Permissions row/profile selection, including round trips, while the native
write still completes. [85 targeted passes, seven guards, conflict choices and
16 fresh native captures](../qa/2026-09-18-mcp-restored-roots/CURRENT-DEV-REVIEW.md)
qualify the integrated repair. Current-head CI/review and this PR's own final
visual approval remain before merge. Catalog repopulation/status guidance,
long-path presentation, inspector refresh and connected-runtime journeys remain
follow-ups. PR2707's heartbeat stays paused. Earlier checkpoints are historical.

## Current checkpoint — PR2749 approved closeout

The owner approved the Tools header gallery. The final dev rebase onto
`b91340a5db` preserves both sides of a documentation-only conflict; Tools sources
are unchanged. [Fresh 152-case validation, seven guards, native lifecycle and
four pixel-identical approved views](../qa/2026-09-19-mcp-tools-header/CLOSEOUT.md)
qualify the integration. Current-head CI/review and live-dev inspection remain
before the authorized merge. Existing PR2727 is next; PR2707's heartbeat stays
paused. Earlier pending-approval checkpoints below are historical.

## Current checkpoint — PR2726 merged; Tools header follow-up

[PR2726](https://github.com/rmusser01/tldw_chatbook/pull/2726) merged at
`ad0f76e23b8737904f24eb34760bbee9ac01a04c`. Its final-head CI passed, accumulated
Qodo review was clean, independent review covered the final selector repair, and
the actual merged tree equals the tested tree. [Merge receipt](../qa/2026-09-19-mcp-tools-header/pr2726-closeout.json).

TASK-32868 reproduces the retained Tools header race on that merged baseline.
Local cache invalidation at measurement restores the composed header without
changing layout or selection. [152 targeted passes, seven guards, independent
review and native before/after evidence](../qa/2026-09-19-mcp-tools-header/README.md)
qualify the bounded fix; [visual approval](../qa/2026-09-19-mcp-tools-header/GALLERY.md)
and this PR's own current-head CI/review remain before merge. Next is existing
PR2727's Permissions restored-root review, then remaining MCP component work.
PR2707's heartbeat remains paused. Earlier pending/next statements below are
historical checkpoints.

## PR2726 approved closeout — selector-budget repair

The owner approved PR2726's integrated conflict choices and gallery. Qodo found
zero issues on approved head `a4158a6a9a`. The Perf Guard then exposed three
inherited wide-modal selectors from PR2742 (277/274). Class-qualified subjects
preserve their exact targets and token values while restoring the unchanged
ratchet. [203 passing targeted cases, seven guards and independent review](../qa/2026-09-18-mcp-audit-catalog-freshness/CLOSEOUT.md)
verify the closeout; all twelve modal before/after captures and all eight
approved Audit captures are pixel-identical. Final-head CI/review and fresh live
dev inspection remain before the authorized merge. The header rendering race
and remaining component reviews are separate follow-ups.


## Current checkpoint — PR2724 merged; PR2726 integrated

PR2724 merged at `cccf0acdad8e939a55cb003588ff1406cef5d1f4` after owner
approval, current-head CI and Qodo closeout. Concurrent PR2742 changed the merge
tree outside MCP; the actual merged app passed 32 Audit cases and its 24 native
captures matched approved terminal content apart from synthetic timestamps.
PR2707's continuation heartbeat stays paused.

Existing draft PR2726 / TASK-32838 is rebased onto that verified merged state.
Both report histories are retained. The source conflict combines the fresh
same-ID lookup with dev's row-selection and post-selection profile checks.
[Conflict choices, 168 targeted passes, seven guards, independent review and
fresh native visuals](../qa/2026-09-18-mcp-audit-catalog-freshness/CURRENT-DEV-REVIEW.md)
are ready for this PR's own final visual approval and current-head CI/review.
No merge approval transfers from PR2724.

An [intermittent painted Tools header mismatch](../qa/2026-09-18-mcp-audit-catalog-freshness/HEADER-FOLLOWUP.md)
was observed in one native run and absent in the unchanged-source replay.
Attribution needs baseline reproduction; this remains a bounded follow-up before
Permissions restored roots. Already-open inspector refresh, connected-runtime
journeys and the wider screen review remain open. Older pending/next statements
below are historical checkpoints.


This ledger separates qualified repairs from remaining destination behavior.
Token migration alone does not qualify a complete workflow.

## Current checkpoint — PR2724 integration

PR2731 (session revocation), PR2734 (permission rule actions) and PR2740
(permission navigation) have merged. PR2707's continuation heartbeat remains
paused. Existing draft PR2724 / TASK-32837 is rebased onto dev
`29b0a31df4701160a3c805e1bf490c76b9353964`, preserving all three merged repairs.
Only this ledger and the completion report conflicted; both histories were kept.
[Exact conflict choices and fresh visuals](../qa/2026-09-18-mcp-audit-navigation/CURRENT-DEV-REVIEW.md)
cover the integration and rejection of queued Audit controls whose owning view
has become unavailable. Current-head CI and PR2724's own final visual approval
remain required. Same-ID catalog freshness is saved separately in PR2726;
connected-runtime journeys and the wider destination review remain open.
All pending/next statements below are historical checkpoints.

## Earlier closeout checkpoint

PR2707 is merged and its heartbeat remains paused. The owner approved the
PR2731/PR2734 continuation and visual evidence. Both follow-ups are ready, with
Qodo's accumulated findings resolved; runner validation and documentation fixes
preserve the approved application behavior. Both now target dev for required CI,
with PR2731 explicitly required to merge before PR2734. Historical draft/pending
approval statements below describe earlier checkpoints, not current status.
The next bounded review is remaining permission/Audit navigation ownership,
followed by connected-runtime journeys; neither is part of these PRs.

## Post-merge continuation: session-grant revocation

PR2707 is merged; its open-PR statements below are historical. The continuation
heartbeat remains paused. TASK-32865 is an independent bounded follow-up from
dev, separate from drafts PR2727/2728/2730.

Revoke binds to its mounted button's grant and profile, invalidates replaced
controls before removal, and admits each control once. Failed revocation can be
retried; late completion preserves a newer profile's listing.
[26 passing targeted checks and eight inspected native captures](../qa/2026-09-19-mcp-session-revocation/README.md)
cover the repair. All seven derived guards pass. Native keyboard revocation uses
the real service and runtime gate: only the selected grant clears, another tool
and profile retain their grants, and calculator asks again. No tool executes.

The follow-up remains draft pending current-head CI/review and owner visual
approval. Exact-input rule removal, Re-allow and other permission actions remain
next; connected external runtime journeys and the wider destination review stay
open.

## Post-merge continuation

PR #2707 merged into dev at `149acda36be8939fe8cd5e589bf77d13462257e7`
on 2026-09-18. The closeout checkpoint below is historical. Component review
has resumed on independent, bounded follow-up branches; the resume heartbeat
is paused. Separate draft PRs preserve Audit selection (#2720), filter layout
(#2721), and inspector guidance (#2722). PR #2722 checks pass on saved head
`4b77c4b33c21affd1f258b7c084490cd65e566c3`; that result does not qualify this
branch. None of those changes is included in this fresh-dev follow-up.
Current-head CI and final visual approval are required before its merge.

## Audit navigation ownership — TASK-32837

[64 targeted cases, seven preflight guards and 16 native captures](../qa/2026-09-18-mcp-audit-navigation/README.md)
qualify mounted Audit action identity, profile capture, destination-filter
reveal and missing-row handling. Retired controls cannot target replacement
records; vanished destinations warn with detail cleared. Dark/light journeys
at 120×40 and 170×48 use the real private catalog and two synthetic metadata
records, without execution or policy mutation. Independent review found no
blocker; the private app exited cleanly with defaults unchanged.

Next: tool-definition freshness when a catalog replaces a same-ID target during
an in-flight drilldown. Missing rows are qualified here; catalog revisions,
80×24 action reachability and connected-runtime behavior are separate bounds.
This follow-up remains subject to current-head CI and final visual approval.

## Post-merge Audit catalog freshness — TASK-32838

PR #2707 merged into dev at `149acda36be8939fe8cd5e589bf77d13462257e7`.
Its unmerged closeout checkpoint below is historical; component review resumed
on bounded follow-up branches and the resume heartbeat is paused.

[62 targeted cases, seven preflight guards and eight inspected native captures](../qa/2026-09-18-mcp-audit-catalog-freshness/README.md)
qualify both Audit drilldowns against same-ID catalog replacements, removals
and pending publication. Navigation re-resolves under the existing publication
lock before selecting/rendering detail; captured profile checks remain.
The real private app uses a controlled collector replacement, without tool
execution or policy mutation. It exited cleanly with defaults unchanged.

PR #2724 separately saves retired-control identity and missing-row handling.
This fresh-dev branch does not include that or the inspector refresh/layout/
guidance follow-ups. Refresh of an already-open inspector, 80×24 reachability
and connected-runtime qualification remain separate. Current-head CI and final
visual approval still gate merging. Next: Permissions restored-roots review.

## Post-review MCP catalog and guidance — TASK-32840

[49 targeted cases, seven preflight guards and eight inspected native captures](../qa/2026-09-19-mcp-recovery-catalog/README.md)
qualify passive repopulation of reviewed local definitions, current-view
publication through navigation/service changes, successful approval with a
catalog retry warning, and accurate retained-history guidance. No discovery,
connection or grants occur. The private native app exits normally with unchanged
user defaults; one unrelated Evals enrollment diagnostic is recorded.

This follow-up stacks on PR #2727's ownership guards and closes its deferred
catalog/guidance items. Retarget its draft to dev after the parent merges.
Next review: bulk permission actions and approval workflows, then remaining
connected-runtime journeys. Current-head CI and final visual approval still
gate merging. The resume heartbeat remains paused.

## Post-merge restored MCP review completion — TASK-32839

PR #2707 merged into dev at `149acda36be8939fe8cd5e589bf77d13462257e7`.
Its unmerged closeout checkpoint below is historical; component review resumed
on bounded follow-up branches and the resume heartbeat remains paused.

[45 targeted cases, seven preflight guards and 16 inspected native captures](../qa/2026-09-18-mcp-restored-roots/README.md)
qualify accepted-review receipt ownership through native completion/rendering,
mode and screen round trips, service replacement, fresh Ask/local defaults and
clearing old passive catalog displays. Real owner approval, cancellation and
changed-root rejection remain intact. The private app exited cleanly with user
defaults unchanged. One unrelated restored Evals enrollment diagnostic is recorded.

This fresh-dev branch is independent of Audit PRs #2724/#2726 and earlier
inspector/lifecycle follow-ups. It does not reload discovery or qualify connected
runtime use. Next review: post-review catalog repopulation/status guidance,
then bulk permission actions and approval workflows. Current-head CI and final
visual approval still gate merging this draft.

## Post-merge continuation: Console approval actions

PR2707 has merged; its closeout statements below are historical checkpoints.
The continuation heartbeat is paused. TASK-32841 is an independent bounded
follow-up from merged dev, separate from draft PR2727/PR2728.

Queued Console Approve all, Deny all and Submit actions now retain their displayed
batch identity, and each batch can submit only once. Unchanged resyncs preserve
choices; changed/cleared/finishing batches invalidate old actions and fresh rounds
remain usable. [123 passing targeted cases, 19 identical baseline failures and
eight inspected native captures](../qa/2026-09-19-approval-action-ownership/README.md)
cover the existing controls. All seven derived guards pass, including a separate
Canvas input-fetch retry. Native uses synthetic pending calls with the real
controller and never dispatches a tool. No connected-runtime claim is made.

The proposed MCP permission-matrix bulk-actions ADR is not implemented here.
Next: session-grant review/revocation and remaining approval/connected-runtime
journeys. This draft requires current-head CI/review and final visual approval
before merge; the wider component review remains open.

## Saved Audit selection qualification — 2026-09-18

PR #2707 merged into dev at `149acda36be8939fe8cd5e589bf77d13462257e7`.
The closeout notes below are historical; feature reviews have resumed on bounded
follow-up branches. TASK-32834 qualifies Audit execution selection/filter
invalidation and exact same-name tool drilldown with [102 targeted cases and
18 native captures](../qa/2026-09-18-mcp-audit-selection/README.md).

Next: compact Audit filter readability. At 80×24 the current fixed filter slots
squeeze away the text value and clip the initiator prompt. Also review inspector
guidance ownership: built-in readiness guidance remains above a selected local
execution/tool. The separate inspector reachability repair is saved in PR #2718.
These open visual issues are not qualified by the selection repair.

## Saved filter checkpoint — 2026-09-18 (historical)

## Current continuation — TASK-32835

PR #2707 merged into `dev` at `149acda36be8939fe8cd5e589bf77d13462257e7`
on 2026-09-18. Its closeout notes below are historical. The post-merge review
has resumed; the backup heartbeat remains paused. Follow-up PRs retain their
own current-head checks and final visual approval before merge.

TASK-32835 repairs Audit filter readability and keyboard reachability from
fresh `dev` (`cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`). Full-width stacked
filters and execution-pane scrolling keep labels and focused controls visible;
resize also reveals the retained table cursor. The
[targeted checks and native gallery](../qa/2026-09-18-mcp-audit-filters/README.md)
record the evidence and limitations. Rendered-layout assertions replace the
obsolete Audit CSS literal pins recorded by TASK-32796.

This branch does not include the separate Audit selection repair saved in
[draft PR #2720](https://github.com/rmusser01/tldw_chatbook/pull/2720).
Next bounded review: inspector guidance ownership, where built-in readiness
guidance appears above unrelated local Audit/tool details. Remaining connected
runtime, server lifecycle, permission and destination work is still open.


## Saved inspector integration history (historical)

## Saved inspector checkpoint — 2026-09-18 (historical)

PR #2707 merged at `149acda36be8939fe8cd5e589bf77d13462257e7` on 2026-09-18.
Its closeout notes below are historical; post-merge work is underway and the
backup heartbeat remains paused. Follow-ups keep their own visual approval gate.

TASK-32836 extends the inspector's existing visibility owner to the complete
server readiness block. The badge, explanation and actions hide together while
tool, permission, Audit or finding detail is present; clearing the last detail
reveals current server guidance. [62 targeted checks and 16 native captures](../qa/2026-09-18-mcp-inspector-guidance/README.md)
verify that boundary on fresh `dev` (`cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`).
The native gallery still shows the baseline compact Audit filter and inspector
layout; those independent repairs remain in their own follow-ups.

Audit selection and filter layout are saved separately in
[PR #2720](https://github.com/rmusser01/tldw_chatbook/pull/2720) and
[PR #2721](https://github.com/rmusser01/tldw_chatbook/pull/2721), with green
checks at `4c5cfaebe9696780908219b6d5cd023a44fe0909` and
`8ba5f72f0b6c045756e6b1761e83bbf819410744`. They remain drafts awaiting visual
approval. Next bounded review: Audit's Open tool / Adjust permission drilldowns,
including missing or stale tool context. Remaining connected-runtime and
permission workflows, and the wider destination review, remain open.

## Qualified

| Surface | Evidence | Bounds |
| --- | --- | --- |
| Settings → permission Edit | TASK-32785, [129 targeted cases and native gallery](../qa/2026-09-18-mcp-permission-handoff/README.md) | Exact profile/revision, visible controls/rows and read-only canvas scrolling |
| Compact introduction, Source and permission matrix | TASK-32788, [134 targeted cases and native gallery](../qa/2026-09-18-mcp-compact-readability/README.md) | Complete guidance and Local/Server labels; permission Tool/State together, tags accessible; no connected external server qualification |
| Tools controls and catalog access | TASK-32789, [88 distinct targeted cases and native gallery](../qa/2026-09-18-mcp-tools-access/README.md) | Full on/off label, focused filters/rows, retained cursor through resize, real private toggle persistence and exact row inspection; no tool execution |
| Tools name/state readability and identity | TASK-32790, [88 final targeted cases and native gallery](../qa/2026-09-18-mcp-tools-readability/README.md) | Complete names and State together, metadata reachable, identity retained through resize/filter/refresh, independent Enter and short/long scrollbar transitions; no tool execution |
| Root drafts, ordered saves and truthful scope | TASK-32791, [targeted checks and ten native captures](../qa/2026-09-18-mcp-root-settings/README.md) | Exact draft ownership, app lifetime/shutdown, cache warnings and later config/external-file supersession; original Permissions resize failure retained and repaired below |
| Permissions final viewport and selected-row continuity | TASK-32792, [targeted checks and twelve native captures](../qa/2026-09-18-mcp-permission-reflow/README.md) | Child width/height reflow, full Tool/State paint, selected row/focus/filter retention and fresh-context Enter; no policy changes or tool execution |
| Shared local-tools master controls | TASK-32793, [177 distinct targeted cases and sixteen native captures](../qa/2026-09-18-mcp-master-settings/README.md) | Ordered saves across both entry points, activation/config identity, pending refresh/recreation, truthful partial receipts and shutdown; no runtime authority changes |
| Tools filter and focus continuity | TASK-32794, [127 distinct targeted cases and native gallery](../qa/2026-09-18-mcp-tools-refresh/README.md) | Open menu/highlight, real pointer/keyboard admission, delayed filter/drill ordering, empty/recovery focus; controlled empty projection, no connected-server or execution qualification |
| Workbench status lifetime | TASK-32795, [79 distinct targeted cases and eight native captures](../qa/2026-09-18-mcp-workbench-lifetime/README.md) | Pre-compose/prune polling, receipt replay across canvas/Workbench replacement, actual loading boundary; native navigation and saved receipts, no generic Select shutdown claim |
| Quiet table redraw and inspector clearing | TASK-32796, [275 distinct targeted cases and eight native captures](../qa/2026-09-18-mcp-table-selection/README.md) | Publication-time row/cell suppression, table-scoped gesture dedup and quiet external drills; real keyboard refresh/navigation, no schema/execution qualification |
| Compact rail navigation | TASK-32812, [119 targeted cases and eight native captures](../qa/2026-09-18-mcp-rail-navigation/README.md) | Full All servers/Source paint, literal Unicode names/counts, ordinary-refresh/resize focus, exact row identity and scrollbars; structural catalog replacement focus is not qualified |
| Complete tool-switch labels | TASK-32822, [37 targeted cases and eight native captures](../qa/2026-09-18-mcp-gate-labels/README.md) | All eleven gate labels, focus/scroll reveal, disabled dependencies and real private Deep research save/reversal; no broader save concurrency or execution qualification |

## PR #2707 closeout boundary

The owner approved moving remaining MCP reviews to follow-up PRs. TASK-32823
is preserved on `codex/mcp-inspector-refresh-followup` at `135f226888`, outside
PR #2707. Its inspector refresh prototype has an unresolved delayed-preview
ownership race; it is not qualified or complete. Resume MCP lifecycle/inspector
work from merged `dev` after the current PR receives final visual approval and
its merge is confirmed.

## Remaining review

- **Remote verification:** Fast Lane, CSS, latency, backlog and all platform GGUF
  checks pass on saved head d7b4c10a7d. The derived-artifact aggregate is still
  running at this checkpoint; checks on the next pushed head are separate.
  TASK-32814 repaired the earlier rail ellipsis assertion failure.
- **Existing test debt:** TASK-32835 replaces the obsolete Audit CSS literal
  assertions with rendered geometry checks. Two Speech harness cases failed
  during profile setup before UI creation; that boundary remains recorded in
  TASK-32796 evidence. TASK-32812 also
  recorded 26 baseline CSS-consolidation ratchet offenders and a destination-tour
  profile-setup failure; TASK-32813 now closes those two boundaries with
  [computed-style, guard and native evidence](../qa/2026-09-18-css-consolidation/README.md).
- **Inspector refresh and execution:** selected tool definition/currentness,
  argument drafts across background refresh, diagnostic empty-state action routing,
  disconnected/stale tools, schema/raw arguments, execution and inspector recovery.
- **Servers:** source transitions, add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery. TASK-32822 repairs the
  compact non-master gate label clipping recorded in TASK-32793. The All servers
  rail clipping recorded in TASK-32796 is repaired by TASK-32812 above.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. Each follow-up PR retains its own current-head CI,
integration and final visual approval before merge.

## Post-merge continuation: exact-input Remove and Re-allow

TASK-32866 follows PR2731 on a separate stacked branch, `codex/mcp-rule-action-review`.
Both controls now capture their mounted target and submit once; obsolete controls
are rejected. Re-allow retains the reviewed definition fingerprint across cached
refreshes, retry and successful completion. Late completion preserves newer
selections, and failed writes remain retryable. Existing permission policy and
layout are preserved.

[56 passing targeted checks and twelve inspected native captures](../qa/2026-09-19-mcp-rule-actions/README.md)
qualify this bounded repair in both themes at 120×40 and 170×48. Real private
store/service writes remove only the selected rule and persist the reviewed
definition; another profile's rule survives. No connection or tool execution.
Seven preflight guards, unchanged static-analysis baseline, independent review,
normal shutdown, lock release, database health and source hashes pass.
Current-head CI/review and owner visual approval remain merge gates. Other
permission controls, connected runtime journeys and the wider review remain open.

## Tool permission navigation continuation — TASK-32867

PR2731 (session Revoke) and PR2734 (exact-input Remove/Re-allow) are now merged
into dev, at `7a695b5e73` and `6095db2f5f` respectively. This continuation starts
from that merged state on `codex/mcp-navigation-action-review`.

Both Tools inspector and Test Tool “Change in Permissions” controls now retain
the displayed tool/profile and reject retired, hidden, disabled or covered-screen
presses. Live navigation remains retryable. [75 targeted passing cases, independent
review and twelve inspected native captures](../qa/2026-09-19-mcp-permission-navigation/README.md)
verify the repair. Eight native keyboard routes select the expected permission row
without changing its record; no server connects or tool executes. All seven
preflight guards and private lifecycle checks pass.

Audit navigation is already saved in PR2724 / TASK-32837; it was not duplicated.
That existing draft currently conflicts with dev and still requires integration
review. Other saved MCP drafts, connected-runtime journeys and remaining screen
reviews stay open. This new repair is a separate draft against dev and requires
current-head CI/review plus its own final owner visual approval before merge.


### PR2724 approved closeout follow-up

The owner approved PR2724's current-dev conflict choices and gallery. Qodo's
five findings are addressed: structured method docs, test import grouping, and
serialization of both destination row checks with the existing publication lock.
The final targeted run passes 147 cases; all seven guards pass and independent
review finds no blockers. The fresh native replay matches all 24 approved terminal
captures except fixture timestamps. PR2724 is ready for review and awaits
current-head CI/review before its authorized merge. Same-ID definition freshness
remains in PR2726.


## PR2770 merged; saved Audit selection PR2720 resumed — 2026-09-21

PR2770 merged as `ba5aa6e9ec60bb93aec4adc6aae219deb9266b78` after owner
visual approval, resolved Qodo review and all current-head required checks.
The merge tree exactly matches qualified head `d6c15f9e4a`; CI passed 1,152
fast-contract cases and 123 admission cases (one expected failure).
[Closeout receipt](../qa/2026-09-21-mcp-inspector-compact/pr2770-closeout.json).

Saved PR2720 / TASK-32834 now follows that merged state. The selection repair
also clears ambiguous duplicate contraction and retires gestures across mode
and subview round trips. [Current qualification](../qa/2026-09-18-mcp-audit-selection/current-dev/README.md)
records 305 targeted passing cases, four known unchanged CSS assertions excluded,
nine guards, twenty inspected native captures and clean private lifecycle.
Two documentation conflicts retain both histories; no product conflicts occurred.
Its own current-head CI/review and fresh visual approval remain merge gates.
Next are saved PR2721 (compact filters) and PR2722 (guidance ownership).
The broader component workstream remains open; the heartbeat remains paused.
