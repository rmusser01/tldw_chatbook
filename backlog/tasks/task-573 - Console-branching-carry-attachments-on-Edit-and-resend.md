---
id: TASK-573
title: 'Console branching: carry attachments across Edit & resend'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-24'
labels:
  - console
  - chat
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(Renumbered from task-521 in the PR #803 rework: that ID collided with an unrelated Done task on dev, which keeps the ID per the Done-tasks-never-move rule.)

Phase B's "Edit & resend" (PR #811) forks a new user-message branch from an edited user message, but the edit modal is text-only: `edit_and_resend_message` calls `create_sibling(role=USER, content=...)` and synthesizes a text-only provider dict, so if the anchor user message carried attachments (an image), the new branch loses them in BOTH the persisted sibling and the provider payload. The old branch keeps them off-path, so nothing is destroyed — but a user who edits the caption of an image prompt silently re-sends without the image. Fix: copy the anchor's `attachments` tuple onto the resent sibling (and include them in the provider payload the same way `submit_draft` embeds staged attachments), or — at minimum — surface the text-only limitation in the edit modal copy when the anchor has attachments. Respect the vision-capability gate the send path applies (`vision_block_reason`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Edit & resend on a user message with attachments carries the attachments onto the new sibling (persisted) and into the provider payload, or clearly informs the user they will be dropped
- [x] #2 The vision-capability block applies the same way it does on a fresh send
- [x] #3 Plain in-place Save continues to leave attachments untouched
<!-- AC:END -->

## Implementation Plan

1. Vision gate in `edit_and_resend_message` before any mutation, identical to
   `submit_draft` (fires only when the anchor carries data-bearing attachments).
2. Replace the hand-rolled text-only synthesized dict with ONE
   `_provider_message_payloads` pass over ancestors + a synthesized
   `ConsoleChatMessage` carrying the anchor's attachments, so image budget /
   vision degradation / mime fallback all apply exactly as on a fresh send.
3. `ConsoleChatStore.create_sibling` gains `attachments=` (same
   `_set_message_attachments` seam as `append_message`); the resent sibling is
   created with the anchor's tuple and persists through the existing path.
4. TDD: carry-over (node + multimodal payload), non-vision block with no
   orphan nodes (mutate-last), text-only anchor keeps plain-string payload.

## Implementation Notes

- The synthesized turn is now a `ConsoleChatMessage` (not a bare dict) run
  through the same payload builder as everything else -- newest-first image
  budget reservation covers the resent turn, and a non-vision model that
  slipped past the gate degrades identically to a fresh send.
- The vision gate reuses `vision_block_reason(..., is_capable=is_vision_capable)`
  (the documented monkeypatch seam), fires pre-mutation, and produces the same
  copy as `submit_draft` -- blocked resends leave no forked sibling and no
  pending assistant node.
- Plain in-place Save is untouched (`update_message_content` without
  `attachments=`), pinned by the existing wiring/persistence tests.
- Files: `tldw_chatbook/Chat/console_chat_controller.py`,
  `tldw_chatbook/Chat/console_chat_store.py`,
  `Tests/Chat/test_console_edit_resend.py`.
