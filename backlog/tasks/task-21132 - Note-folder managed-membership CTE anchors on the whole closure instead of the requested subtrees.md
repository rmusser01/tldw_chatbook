---
id: TASK-21132
title: >-
  Note-folder managed-membership CTE anchors on the whole closure instead of the requested subtrees
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notes
  - database
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21132).

`Notes/note_folder_repository.py:1831-1861`: the recursive CTE's anchor ignores the requested
folder_ids and filters only at the end, so every Notes-tree interaction walks the entire
managed-membership closure - twice per tree refresh (library_screen.py:11805,11943; off-loop,
so latency not freeze).

## Acceptance Criteria

- [ ] The CTE anchor is seeded from the requested ids' subtrees; results identical on existing fixtures
- [ ] A timing probe on a deep synthetic tree shows the reduction

## Re-verification against dev 2be18842a (2026-08-23) — RECOMMEND CANCEL

An independent read-only pass re-checked this finding. **The mechanism is real; the cost is
effectively zero for the default user; and the rewrite is a query inversion, not a filter.**

**Confirmed**: `Notes/note_folder_repository.py:1889-1919` — the recursive anchor selects all
managed memberships and applies the requested ids only at the end. Called twice per tree refresh
(`:383-385`, `:654-656`, with `library_screen.py:12096-12180` issuing two `load_batch` calls).

**But the magnitude does not hold up.** The recursion walks **upward** (child → parent), so the
closure is bounded by *distinct managed folders x tree depth* — not by the folder tree and not by
note count. Managed memberships exist only where a notes-sync owner has placed notes. For a user
with no sync owners — the default, and TASK-21112 has since made zero-profile boots create no sync
state at all — the anchor is **empty**, served by the partial index
`idx_note_folder_memberships_managed_owner`, and the query costs one index probe. It is also
already off the event loop: `Notes/notes_scope_service.py:248-250` wraps the call in
`asyncio.to_thread`. So "every Notes-tree interaction walks the entire managed-membership closure"
is true only for a heavy sync user, and even then it is milliseconds on a worker thread.

**And the fix is riskier than billed.** Because the recursion runs upward, seeding it with the
requested ids means *inverting* it to descend via `parent_id` — a different query, not a filtered
one. Equivalence hinges on a quirk: the current query tests `folder.deleted = 0` on the *source*
row, so a deleted parent still enters the result set but cannot recurse further. A naive downward
rewrite will not reproduce that, and "results identical on existing fixtures" is too weak a bar —
it would need fixtures with a deleted intermediate folder, a deleted leaf-managed folder, and a
managed membership under a deleted ancestor.

**Recommendation**: cancel, or downgrade to a "clean this up when someone is next in this file"
note. This is a correctness-of-shape improvement, not a performance fix.

## Closure recommendation (2026-08-24, burn-down close-out)

**Recommend closing without work.** Effectively free for the default user. The recursion walks UPWARD, so it is bounded by managed folders x tree depth, not by the folder tree; with no sync owners the anchor is empty and served by a partial index. It is already off the event loop via asyncio.to_thread. The rewrite is a query INVERSION, not a filter, and equivalence hinges on a quirk (deleted parents enter the result set but cannot recurse) that a naive downward rewrite would not reproduce.

Left open rather than closed unilaterally: retiring a filed finding is the owner's call. The
evidence above is what a re-verification pass measured against dev before dispatch; if it is
accepted, close this as "retired on evidence" rather than "won't fix", because the mechanism was
real and only the cost or the prescribed fix was wrong.

