---
id: TASK-521
title: 'Console branching: carry attachments across Edit & resend'
status: To Do
assignee: []
created_date: '2026-07-24'
labels:
  - console
  - chat
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase B's "Edit & resend" (PR #811) forks a new user-message branch from an edited user message, but the edit modal is text-only: `edit_and_resend_message` calls `create_sibling(role=USER, content=...)` and synthesizes a text-only provider dict, so if the anchor user message carried attachments (an image), the new branch loses them in BOTH the persisted sibling and the provider payload. The old branch keeps them off-path, so nothing is destroyed — but a user who edits the caption of an image prompt silently re-sends without the image. Fix: copy the anchor's `attachments` tuple onto the resent sibling (and include them in the provider payload the same way `submit_draft` embeds staged attachments), or — at minimum — surface the text-only limitation in the edit modal copy when the anchor has attachments. Respect the vision-capability gate the send path applies (`vision_block_reason`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Edit & resend on a user message with attachments carries the attachments onto the new sibling (persisted) and into the provider payload, or clearly informs the user they will be dropped
- [ ] #2 The vision-capability block applies the same way it does on a fresh send
- [ ] #3 Plain in-place Save continues to leave attachments untouched
<!-- AC:END -->
