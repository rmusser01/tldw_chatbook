---
id: TASK-31552
title: llama.cpp manual prompt-cache snapshot manager
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 01:15'
updated_date: '2026-09-05 01:43'
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
Status remains In Progress for branch integration/remaining repository gates. See the
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md. Reason: accepted snapshot ownership and retention contract; ADR-029 and ADR-036 also apply. Execute Docs/superpowers/plans/2026-09-04-llamacpp-slot-snapshots.md in six reviewed units: (1) strict settings and effective launch admission; (2) private transactional storage and integrity; (3) bounded loopback-only management HTTP; (4) app-owned operation and subprocess lifecycle; (5) manual Models widget and canonical F9 settings; (6) isolated real-server reuse evidence and closeout. Units 1–5 are implemented and reviewed. Task6 supplies safeguards, an opt-in production-path harness, and honest evidence documentation; its required real-server execution remains pending. Use targeted RED/GREEN tests and check criteria only when their evidence exists.
<!-- SECTION:PLAN:END -->

UAT remediation plan (2026-09-05): reproduce overlapping background readiness and
Save/Restore staging with deterministic barriers; distinguish a pending refresh
from failed/invalidated readiness without removing each operation's fresh probe;
run service/UI regressions and the complete real b10816/Gemma 4 vision/retention
UAT with no diagnostic callback suppression; update evidence from observed results.
ADR required: no new ADR. ADR path: backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md.
Reason: routine concurrency fix within the accepted ownership/readiness contract.

## Implementation Notes

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
