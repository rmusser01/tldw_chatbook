---
id: TASK-376
title: Distinguish attach vs insert-content behavior for text files in the composer
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Picking test-image.png staged a right-side chip ('test-image.png - 341 B', button became paperclip+check). Picking notes.txt instead inserted the file's content into the draft as an inline pill '(doc) notes.txt - 115 B' glued to the typed text with no separator; the only differentiator is a transient toast 'notes.txt content inserted'. The pill's '115 B' does not match the 60 B file (it is the wrapped-content size), and the pill is then subject to the unfurl flow (see j3-enter-hijacked). Users reasonably believe they attached a file; they actually pasted its body.

**Repro:** Attach test-image.png then notes.txt via the composer Attach picker. Compare: image becomes a right-side chip; txt becomes an inline draft pill reading '115 B' for a 60-byte file, announced only by a 5s toast.

**Verifier note:** Facts confirmed in code: text files route insert_mode='inline' and splice content as a pill (chat_screen.py:9946-9963), announced only by a 5s toast; the pill size uses processed_size = len(wrapped content) not the file size (attachment_core.py:290 + label property line 218), so '115 B' for a 60 B file is a real mislabel. The dual routing itself is the shipped attachment_core architecture (phase-1 #621) and task-230 adjudicated only excluded-image formats, so part of this restates design — but the wrong size, the no-separator splice, and zero at-selection differentiation are unrecorded defects. P2→P3: confusion/polish, no data loss, toast does differentiate.

**Source:** Console UX expert review 2026-07-20 (finding j3-attach-txt-actually-inserts-content; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-42-staged-img-and-txt.png`, `j3-43-typed-before-send.png`, `j3-22-two-staged.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One affordance, one mental model: either stage text files as attachments too, or label the action distinctly at selection time ('Insert as text') and show the real file size
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept the shipped dual-routing architecture (phase-1 #621) and fixed the confirmed
mislabels. The `PendingAttachment.label` property now shows the real FILE size
(`original_size`) for inserted text (`insert_mode == "inline"`) instead of the
wrapped-content length -- so a 60-byte file no longer reads "115 B". Real
attachments still show their processed (attached) byte size. The insert toast is
now distinct at selection time: "<name> inserted as text (not attached)" so the
user isn't misled into thinking they attached a file. RED->GREEN pure label test.
<!-- SECTION:NOTES:END -->
