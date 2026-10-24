---
id: TASK-36698
title: Ignore superseded Image Gen demo worker results and errors
status: In Progress
created_date: 2026-10-24 17:34
labels:
- textual-patterns
- reliability
priority: low
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
assignee:
- '@codex'
updated_date: 2026-10-24 20:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Image Gen demo uses an exclusive thread worker, but the canceled old worker still publishes success or failure. A slow earlier request can replace the newest image, metadata or success status.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Real Generate clicks admit OLD then NEW. After NEW publishes, OLD has actual Worker.is_cancelled=True; releasing it replaces metadata/image with OLD or changes Done to OLD adapter failure. Serial control retains NEW. This is the live palette-accessible developer demo, not the production media-generation pipeline.

Book basis: PDF335 / printed326 explicitly checks the current worker cancellation state before posting a thread result to the UI. Both success and failure publication must remain applicable to the current demo request.

Affected owners: [image_gen_demo_screen.py](../../tldw_chatbook/UI/Screens/image_gen_demo_screen.py), [image_gen_command_provider.py](../../tldw_chatbook/UI/image_gen_command_provider.py).

### Existing ownership and decisions

TASK-497 is Done; it marshals status to the UI thread and intentionally defers other demo polish. TASK-34935 offloads image decoding and TASK-36098 owns Image/Video Settings drafts. None owns canceled demo request publication.

Related tasks: [TASK-497](../tasks/task-497%20-%20Image-generation-Phase-1-polish-follow-ups.md), [TASK-34935](../tasks/task-34935%20-%20Move-image-gen-demo-PIL-decode-off-the-UI-thread.md), [TASK-36098](../tasks/task-36098%20-%20Preserve-newer-Image-and-Video-Settings-drafts-after-pending-saves.md).

ADR required: no.
ADR paths: [176-one-media-generation-core](../decisions/176-one-media-generation-core.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: Routine applicability checks within the existing demo worker and media-generation core; no new runtime or provider boundary.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A superseded or canceled demo request cannot replace the current image, metadata or status on success or failure.
- [x] #2 The newest still-applicable request publishes its normal success or error outcome.
- [x] #3 Ordinary serial generation and the existing command-palette entry keep working without real API calls in regression tests.
- [x] #4 Mounted held-old/new controls exercise the actual exclusive worker and UI publication for both old success and old failure, with a serial control.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required:no. ADR paths:[ADR-176](../decisions/176-one-media-generation-core.md) and [ADR-033](../decisions/033-application-session-state-ownership.md). Reason:current-request UI publication in the existing demo worker owner; unchanged media core/provider/runtime boundaries.
1. Add mounted real Generate/exclusive-thread Worker controls for older success/error after newer completion, queued older UI publication after new admission, explicit cancellation, normal serial success/newest error and live palette reachability. Bound provider/catalog I/O and exercise real PNG decode off the UI thread.
2. Retain stale-publication RED with actual Worker.is_cancelled observed. Fence success and error at final UI publication using current worker/request authority, preserving existing status/render behavior and low-level decode/apply contracts.
3. Preserve off-thread request/generation/decode and current UI-thread-only widget publication. Wait distinct Generate button activation before the second request. Do not use ever-mounted state as attachment authority; retain safe cancellation/retirement semantics.
4. Run new and directly relevant existing demo/decode controls with actual private-child XML inspection and focused lint/format. No full suite, real provider calls or unrelated service fixes.
5. Obtain task-scoped Superpowers requesting-code-review, register every new UI regression in existing required CI, update media architecture/notes and then perform overall integration/rebase/review/PR/CI qualification.
Master plan:Docs/superpowers/plans/2026-10-24-textual-lifecycle-and-import-followups.md (Task6).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Image Gen demo success and error callbacks now carry the originating actual Worker through a shared UI-thread publication guard. Canceled workers and detached screens cannot update widgets; checks occur after queued callbacks resume. Existing exclusive cancellation provides request authority without another counter. Request/validation/provider execution and real PNG/Pixels decoding remain off-thread; low-level decode/apply and palette contracts are unchanged.

Verification:10 new real palette/Generate/worker/provider/decode controls plus2 unchanged decode/apply controls pass, with all10 private children clean. Six preimplementation failures establish stale provider completion, queued callback and explicit-cancel success/error defects. The parent13-case selection has12passes/1existing legacy profile-setup failure, not all13green. Its unchanged real-host body passes separately in an untracked private-profile diagnostic. Root verified actual XMLs and5final hashes; reviewer inspected all37 historical receipts. Ruff/format/whitespace pass. No full suite, optional skips or real provider API.

Superpowers task review approved with no blocking findings. All10 new exact cases are registered in required UI gate (floor839). Overall hosted qualification remains pending. ADR required:no; existing ADR-176/033 preserve demo/media-core ownership. Updated Docs/Architecture/media-generation.md and implementation plan. Source/tests:UI/Screens/image_gen_demo_screen.py and Tests/UI/test_image_gen_demo_request_lifetime.py. Existing dependency warnings and direct legacy profile fixture limitation remain explicit.
Final whole-branch review follow-up: native removal marks the screen pruning before attachment loss or Worker cancellation. The shared publication guard now rejects that interval. Two actual palette/Generate/provider/Worker/native-Unmount controls reproduce unwanted success/error status writes before the guard and pass afterward; no application crash is claimed. Final selected run has 6 parent passes and 6 clean private children, zero failures/errors/skips (72.279s), including both ordinary serial outcomes and two unrelated Chunking positive controls. This task now contributes 12 exact UI cases; overall census floor843 with 81 new UI and20 parser cases. The four final review regressions are registered without shard/timeout/allowance changes. One scoped final re-review and hosted qualification remain pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->