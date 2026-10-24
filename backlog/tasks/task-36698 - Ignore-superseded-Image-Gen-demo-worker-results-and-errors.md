---
id: TASK-36698
title: Ignore superseded Image Gen demo worker results and errors
status: To Do
created_date: 2026-10-24 17:34
labels:
- textual-patterns
- reliability
priority: low
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
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
- [ ] #1 A superseded or canceled demo request cannot replace the current image, metadata or status on success or failure.
- [ ] #2 The newest still-applicable request publishes its normal success or error outcome.
- [ ] #3 Ordinary serial generation and the existing command-palette entry keep working without real API calls in regression tests.
- [ ] #4 Mounted held-old/new controls exercise the actual exclusive worker and UI publication for both old success and old failure, with a serial control.
<!-- AC:END -->
