---
id: TASK-31552
title: llama.cpp manual prompt-cache snapshot manager
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:15'
updated_date: '2026-09-05 18:51'
labels: []
dependencies: []
references:
  - backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md
documentation:
  - Docs/superpowers/specs/2026-09-04-llamacpp-slot-snapshots-design.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-slot-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users manually preserve and reload processed llama.cpp context, including supported image and audio context, from a server launched inside Chatbook. Provide predictable private storage and configurable retention without implying conversation recovery or guaranteed cache reuse.

### Design status

The user selected manual management before automatic per-conversation persistence,
Chatbook-launched servers, timestamp-generated names, and configurable retention
with a default of 10. The reviewed specification and ADR are linked above.
Implementation units 1–5 are implemented and independently reviewed. Task6 adds
an opt-in live harness and [per-criterion evidence](../../Docs/superpowers/reviews/2026-09-04-llamacpp-slot-snapshots-verification.md).
The existing b10816/Gemma 4/vision-projector assets were used in live UAT on
2026-09-05. The confirmed-Restore readiness race was fixed and independently
reviewed. Final normal-UI UAT passes without diagnostic suppression, including
measured text/image reuse and real retention/Delete. AC1/AC5 now have live evidence.
PR review remediation and scoped local verification are complete; GitHub merge
remains gated on current-head checks. See the
[UAT record](../../Docs/superpowers/reviews/2026-09-05-llamacpp-slot-snapshots-uat.md).

The user approved integration of the follow-up review: compatibility-gated save
publication, integrity checks before restore, proxy-free loopback transport,
terminal working-file cleanup, separate probe/operation deadlines, and visible
cross-model retention wording. These now have targeted automated contract evidence,
not a claim of real-model or audio reuse. ADR-119 records the accepted amendments.

ADR required: yes

ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md

Reason: new private snapshot files, automatic deletion, and a llama-server
management boundary. Existing ADR-029 and ADR-036 also apply.

ID allocation: the CLI offered 31429; refs and 64 worktrees contained task IDs
through 31551, so this record was moved to 31552 before linking it elsewhere.
Recheck allocation before integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can save a selected slot with an automatically generated timestamp name and restore a saved snapshot to an eligible slot on a Chatbook-launched server.
- [x] #2 The manager retains the newest 10 complete snapshots per profile by default, supports a validated configurable count, and prunes only after a fully committed successful save.
- [x] #3 Snapshot operations honor launch identity, endpoint readiness, compatibility evidence, private file ownership, and uncertain operation outcomes across navigation and restart.
- [x] #4 The UI explains cache-only restore semantics, exposes actionable failure and partial-success states, and remains keyboard usable in the production Models screen.
- [x] #5 Targeted automated checks and an isolated real-server save/restart/restore test prove persistence and actual same-image prefix reuse with an eligible model.
- [x] #6 Save is disabled without complete required compatibility evidence; evidence invalidated before publication prevents retaining the new snapshot or pruning older ones, including with keep count 1.
- [x] #7 Restore verifies staged byte length and SHA-256 before any Restore POST; truncated or same-length corrupted input leaves the destination slot untouched.
- [x] #8 All management and readiness traffic uses a validated numeric loopback destination with proxies and redirects disabled; proxy environment variables and redirect responses cannot forward requests or credentials elsewhere.
- [x] #9 Successful and acknowledged terminal operations and proven pre-submission failures release safe working files; repeated restores do not accumulate copies, cleanup failures expose residual bytes, and uncertain operations retain files until safe.
- [x] #10 Five-second probe deadlines are separate from explicit ten-minute Save/Restore submission deadlines; preparation and elapsed operation status remain visible and slow valid operations are not failed at the probe deadline.
- [x] #11 The Save area visibly states the effective newest-N retention limit across all models, including count changes and narrow terminal layouts.
- [x] #12 The integrated Console keeps Environment collectors off closed-Inspect startup while first-open, refresh, and reopen behavior remain usable
- [x] #13 Suspended reusable Console screens do not dispatch Environment collectors, and returning to an open Inspect rail refreshes using the retained owner
- [x] #14 Snapshot settings remain outside the whole-registry pre-import closure until the provider settings surface is used
- [x] #15 The complete registered-screen pre-import census meets the existing module and LOC limits, without moving cost onto boot, and deferred Library and Settings features remain usable on first open.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner-approved inherited pre-import repair: execute
Docs/superpowers/plans/2026-09-05-preimport-budget-paydown.md. Defer the measured
Library controller/note-import edges and Settings RAG/Tool Pack services to their
existing runtime consumers. Preserve screen registration, event classes, patch
seams, ownership and all budget limits. Verify cold-import RED/GREEN, mounted
first use, the unchanged full census and the other boot guards before review.
ADR required: no new ADR. ADR path: backlog/decisions/097-boot-budget-ratchets.md.
Reason: routine import deferral preserves existing runtime boundaries.

Pre-import contribution plan: pin snapshot-settings absence in a fresh whole-route
walk, defer the canonical F9 preference imports to their existing use sites, and
verify mounted F9 behavior plus paired census against untouched dev. ADR required:
no new ADR; ADR-097 applies. Do not raise limits or refactor unrelated routes.

Latest-dev integration plan: reproduce Environment dispatch while a reusable
Console is covered, gate the controller's existing rail accessor on screen
activity, and refresh the retained owner on ordinary resume. Verify mounted
suspend/resume, existing deferred-controller guards, startup census and live UAT.
ADR required: no new ADR. Existing ADR-097 applies; this preserves visible-only
collection across the newly inherited Console screen-reuse lifecycle.

ADR required: yes. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md. Reason: accepted snapshot ownership and retention contract; ADR-029 and ADR-036 also apply. Execute Docs/superpowers/plans/2026-09-04-llamacpp-slot-snapshots.md in six reviewed units: (1) strict settings and effective launch admission; (2) private transactional storage and integrity; (3) bounded loopback-only management HTTP; (4) app-owned operation and subprocess lifecycle; (5) manual Models widget and canonical F9 settings; (6) isolated real-server reuse evidence and closeout. Units 1–5 are implemented and reviewed. Task6 supplies safeguards, an opt-in production-path harness, and honest evidence documentation; its required real-server execution remains pending. Use targeted RED/GREEN tests and check criteria only when their evidence exists.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner-approved inherited pre-import repair supersedes the historical pause below.
Deferred Library runtime controllers/note-import helpers and Settings RAG/Tool
Pack services at existing use sites; patch seams and Textual event classes remain
intact. Complete census is 490 modules/363740 LOC (was 547/422544), Library113319.
ADR-097 banks the savings at 378740 total/123319 per-route LOC; module limit500
unchanged. No budget increase. New closure RED/GREEN, 299 affected Settings checks,
13 other boot/closure checks and independent reviews pass. Existing broad-file
lint has no new diagnostics; derived artifacts reproduce. Real b10816/Gemma4
vision UAT passes267.44s with text22/23, sameimage105/106, changedimage19/106,
retention/Delete and owned-child cleanup. See the current UAT report for complete
Library repeat and retained source-inspection failure evidence. No full sweep.
Final complete unchanged-source Library verification: 125 passed in94.36s;
final tightened census/closure repeat: 5 passed in9.17s.
Plan: Docs/superpowers/plans/2026-09-05-preimport-budget-paydown.md.
ADR path: backlog/decisions/097-boot-budget-ratchets.md; no new ADR required.

Merge paused after latest-dev `22006e84d` integration. 71 startup/UI checks pass;
the wider pre-import guard fails on both PR and untouched dev (549 vs 547 modules,
limit500; both exceed LOC caps too). Deferred F9 snapshot-preference imports at
their three use sites, removing the PR's two modules: final547/422544LOC versus
baseline547/422128LOC. New whole-route closure RED/GREEN, 20 affected tests pass,
independent review clear, no new lint debt. ADR-097 limits unchanged. Broader
inherited multi-route paydown requires owner direction; status stays In Progress.

Latest-dev reuse integration: Library baseline failures were paired against an
untouched archive; 106 focused plus two changed lifecycle tests passed. Console
reuse passed 131 targeted checks, then independent review exposed Environment
collection while covered. A mounted RED proved four dispatches; top-screen
identity gates the shared accessor (including deferred network dispatch), and
ordinary resume refreshes the retained owner. Final affected verification:
55 passed; independent re-review clear, no new baseline-relative lint findings,
whitespace clean. Both rebased live text/vision UAT runs passed (252.25s/252.67s).
ADR-097 applies; no new owner, policy, dependency or raised budget.

Implemented app-owned manual snapshots with private atomic publication and
commit-before-prune retention, strict launch compatibility and bounded loopback
HTTP, retained file workers and exact-claim Stop barriers, and Models/F9 controls
with stale-draft-safe canonical persistence. Existing ADR-119, ADR-029 and ADR-036
govern the ownership boundaries; no new config owner/database/dependency was added.

The opt-in test drives actual Models launch/actions and production service/store/client,
then ordinary no-slot-forcing chat requests. Native image A→B controls bound the
different-media prefix using the pinned runtime's SHA-256 media identities. The
RAM-cache admission allowlist was narrowly completed to isolate this measurement;
ordinary launch defaults and compatibility identity are unchanged.

See [user guidance](../../Docs/LLMs/llamacpp-snapshots.md) and the
[verification record](../../Docs/superpowers/reviews/2026-09-04-llamacpp-slot-snapshots-verification.md)
for exact scope, tests, known environment/inventory limitations, and missing live
counters. No real model/audio reuse, Windows ACL equivalence, or feature completion
is claimed. No Done transition, merge, push, or asset download was performed.

Integrated final review I1/I2/I3/M1/M2 were addressed in one bounded wave:
preference admission errors are safe before reservation with Models recovery;
entry/init reconciles only terminal work and valid deletion tombstones while
preserving unproven writers; nested JSON obeys existing conservative recovery
policies and unexpected reconciliation failure cannot bypass Stop teardown.
Details now shows absolute observation time, and the aggregate-deadline test no
longer depends on a narrow scheduling margin. Final affected verification:
156 passed, 1 existing RequestsDependencyWarning; scoped lint/format/compile/diff
checks pass. Independent scoped re-review closed all five findings at `a5268225df`
with no new breakage. Live AC5 and status In Progress
are unchanged; no real-model or audio reuse evidence was added.

2026-09-05 UAT supersedes the preceding live status: existing b10816, Gemma 4 and
the adjacent BF16 vision projector were used. Three normal-UI attempts failed at
confirmed Restore; traced modal-return refresh temporarily clears readiness during
Restore staging. AC1 was reopened and AC5 remains open. Suppressing only that
callback in a scratch diagnostic plugin yielded measured text reuse (22/23),
same-image reuse (105/106), different-image text-prefix-only reuse (19/106), and
real newest-10/lowered-to-2 retention plus cancel/confirmed Delete. These controls
are not a passing production UAT. Corrected the live harness/docs CPU projector
argument to none and retained screenshots/status/counters; no production code
changed. All UAT-owned children exited. See the 2026-09-05 UAT record for exact
failures, diagnostic boundaries and supplementary F9 test failure/rerun. Existing
ADR-119 applies; status remains In Progress.

UAT remediation completed: pending readiness refreshes retain the last completed
observation; actual failures still revoke readiness. Six barrier regressions cover
Save/Restore, with positive RED/GREEN and failed-probe mutation evidence. Independent
review and its scoped re-review are closed. Normal Models live UAT now passes with
no diagnostic override (274.86s), including all measured image controls, newest-10
retention, lower-count-after-save pruning and cancel/confirmed Delete. The rapid
keyboard-press harness issue was corrected without changing production UI behavior.
Targeted checks: 95 passed; final affected service/live-helper repeat: 64 passed;
scoped lint/format/whitespace pass. AC1 and AC5 are checked from this evidence.
No full repository sweep, merge or push; the worktree is retained for integration.

PR #2419 remediation: addressed six Qodo validation/documentation/style findings
and proved the reported endpoint fall-through does not occur with three real-worker
regressions and a failing unsafe mutation. Strict launch validation preserves
Mapping inputs; both UI keep-count forms use the shared integer validator. Fixed
CI's 981-module startup breach by composing snapshots on first use, preserving
one owner, explicit roots and shutdown settlement. ADR-119 and ADR-097 apply;
no budget was raised. Final evidence: 460 combined tests passed, 3 mounted Models
checks passed, startup census passed twice at 972/972, and fresh normal b10816/
Gemma vision UAT passed in 258.67s. Scoped lint/format, CSS reproduction, inventory
and whitespace checks pass; unchanged broad-file lint and dependency warnings
remain documented. The current UAT report supersedes historical pending statements.

Rebased onto dev e990738b2812 (Chunking Lab) without code conflicts. Regenerated
only the merged diagnostic sink count (11 to 12), and paid down three snapshot
bare-type CSS rules using precise existing subjects while preserving inherited
Checkbox/CollapsibleTitle styling and button precedence. The initial 80-column
regressions and their corrections are retained in the UAT report; the lesson is
recorded in lessons-testing-evidence.md. Post-rebase snapshot verification passed
460 checks plus 3 socket checks after granting local socket permissions; final
combined mounted layout/boot verification passed all 25 checks in 48.83s.
CSS/inventory reproduction and whitespace checks pass. Existing ADR-097/ADR-119
apply; no budget change or broad test sweep. GitHub review/check settlement and
the requested remote merge remain the external integration steps.
<!-- SECTION:NOTES:END -->

Owner-approved inherited Console fix: first-use Environment ownership keeps the
four collectors/projection modules off closed-Inspect startup. First-open paints
the no-workspace state; reopens retain controller state; callbacks to an existing
owner preserve their original policy. Warm census restored from 976 to 972 with
an explicit absence guard. Scoped evidence: 226 passed plus two corrected
callback failures; final complete wiring 24 passed, boot/snapshot service 59
passed, projection/census 32 passed. Independent review and re-review clear;
generated artifacts, whitespace, and baseline-relative lint verified. ADR-097
applies, no budget increase. Latest additional dev delta e49a7a16d is docs/assets
only. GitHub check settlement and requested merge remain integration steps.

UAT remediation plan (2026-09-05): reproduce overlapping background readiness and
Save/Restore staging with deterministic barriers; distinguish a pending refresh
from failed/invalidated readiness without removing each operation's fresh probe;
run service/UI regressions and the complete real b10816/Gemma 4 vision/retention
UAT with no diagnostic callback suppression; update evidence from observed results.
ADR required: no new ADR. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md.
Reason: routine concurrency fix within the accepted ownership/readiness contract.

PR #2419 remediation plan (2026-09-05): validate Qodo's seven findings against
the current code; add shared key-path and UI-integer boundaries, strict Pydantic
launch-input validation, named credential limits and missing API documentation;
prove ambiguous endpoint probes fail closed, then run focused regression and
required CI checks before replying to reviews and merging the current head.
The CI startup-census breach (981 > 972 modules) requires deferring snapshot
composition off the unused startup path, with owner initialization/shutdown and
real Models UAT reverified. Preserve the ADR-097 ratchet without raising its limit.
ADR required: no new ADR. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md.
Also applies: backlog/decisions/097-boot-budget-ratchets.md.
Reason: boundary hardening and lazy composition implement the accepted contracts.

Post-rebase CI plan: replace the three snapshot ancestor-scoped bare-type CSS
rules exposed by the boot ratchet (277 > 274) with selectors keyed to existing
snapshot IDs/classes, rebuild generated CSS, and rerun the boot budgets and
mounted Models/F9 tests. Regenerate the merged diagnostic inventory summary
(12 sink files: dev's 11 plus snapshot_store) without changing sink rows.
ADR required: no new ADR; existing ADR-097 and ADR-119 apply.

Owner-approved Console startup remediation plan (2026-09-05): defer the
Environment controller and projections until Inspect first opens; retain the
same controller and snapshot thereafter. Keep closed-panel focus/fleet/poll
callbacks cold. Place the existing rail section IDs with rail state and preserve
their environment-state exports. Verify the existing failing warm census,
first-open empty-state rendering, reopening, controller/wiring tests, and PR CI.
ADR required: no new ADR. ADR path: backlog/decisions/097-boot-budget-ratchets.md.
Reason: first-use deferral restores TASK-31450 AC6 within existing ownership;
no storage, network, display-policy or boot-budget change.
