---
id: TASK-4022
title: Soft-deleted media is permanently un-importable and the trash is unreachable
status: To Do
assignee: []
created_date: '2026-08-09 20:30'
labels:
  - library
  - media
  - data-loss
  - recritique-2026-08-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09 (RC-04/RC-05), reproduced by the mechanical arm at dev `4d0232358`.

Repro: import a file → Media ▸ Select → check it → Delete selected → confirm. Then re-import the
same file. Result:

    ≡ matched · short.txt
    Already in Library — matched an existing item; nothing new was imported.

…while `Media (1)` and the item is absent from every list. The import dedup matches
**soft-deleted** rows, so a deleted file can never be re-added. Meanwhile the confirmation dialog
promises `This moves them to trash.` and there is no trash anywhere in the product — not in the
rail, not in the `type:` filter (which offers only `All` and the ingested types), not on any canvas.

Net effect: the user's content is neither present nor restorable through the UI, and the one action
that promised reversibility is the one that makes it unreachable.

Two coupled defects, both in scope here:
1. Dedup must not match soft-deleted rows (or must offer to restore the existing row instead of
   silently refusing the import).
2. Bulk delete completes with no receipt and no undo. Compare the asymmetry: creating one item
   yields `✓ done · file · 1s` plus an `Open in Library` jump; destroying two yields silence.

Either ship the trash the copy promises (a `type: Trash` value or a rail row, with restore), or
change the copy to state what actually happens — but the current combination of a reversibility
promise, no destination, and a permanent import block is the worst of the three options.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A file deleted from Media can be re-imported, or the duplicate-match path offers restore instead of silently refusing
- [ ] #2 Bulk delete emits a receipt naming the count, with an undo affordance at the point of action
- [ ] #3 The confirmation copy and the product agree: either the trash is reachable and restores, or the copy stops promising it
- [ ] #4 Live verification of the full cycle: import → delete → re-import → the item is present exactly once
<!-- AC:END -->
