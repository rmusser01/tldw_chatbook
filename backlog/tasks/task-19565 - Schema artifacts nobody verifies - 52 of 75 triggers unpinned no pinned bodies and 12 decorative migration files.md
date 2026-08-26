---
id: TASK-19565
title: >-
  Schema artifacts nobody verifies — 52 of 75 triggers unpinned, no trigger has
  a pinned body, and 12 migration .sql files are decorative
status: To Do
assignee: []
created_date: '2026-08-21 20:15'
labels:
  - db
  - testing
  - schema
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F2** and **F10**. Grouped: both are schema artifacts that look
authoritative and are checked by nothing. Re-verified at this branch base.

**A — the index census's method was never extended to triggers.** The lane
measured this with the index census's own method: **indexes 96, zero unnamed in
`Tests/`. Triggers 75, of which 52 (69%) are named nowhere in `Tests/` at
all.** Unpinned families include every `notes_sync_*`, `character_cards_sync_*`,
`keywords_sync_*`, `world_books_sync_*`, and `messages_ai`/`messages_ad`.

**Worse: even a *named* trigger has no pinned body.** That is not
hypothetical — `ChaChaNotes_DB.py:13165-13178` is a **runtime self-heal that
`DROP`s and re-`CREATE`s `notes_au` on every single database open**, because
the trigger shipped without its `deleted = 0` guards. A trigger whose body was
wrong shipped to every user, and the repair is a permanent startup fixup rather
than a migration. Nothing would have caught the body being wrong.

This directly enables TASK-19566: the FTS soft-delete guard lives in a trigger
body, and no test asserts any trigger body.

Column coverage is only half-covered, and for a specific reason worth carrying
forward: the parity sweep compares **chain-derived sides against each other**,
which is an identity comparison and cannot detect a shared error.

**B — 12 of 26 `DB/migrations/*.sql` files are decorative.** The migration step
executes an **embedded Python constant**; the on-disk `.sql` twin is never
opened. They are kept aligned only by a comment. Meanwhile the packaging test
**pins nine of them as shipped wheel content**, which makes them look
authoritative to anyone reading the repo. **No test compares any file to its
constant**, so they can silently diverge — and a future maintainer editing the
`.sql` file would change nothing at all.

## Acceptance Criteria

- [ ] Trigger existence is pinned the way index existence already is: a census
      test fails when a trigger is added, removed or renamed without the test
      being updated
- [ ] Trigger **bodies** are pinned, not just their names — the `notes_au`
      incident is the proof this is needed
- [ ] The `notes_au` runtime self-heal is replaced by a real migration, so a
      correct trigger body ships rather than being re-patched on every database
      open
- [ ] The column parity sweep no longer compares two chain-derived sides
      against each other; it compares against an independently-declared
      expectation
- [ ] Each `DB/migrations/*.sql` file either becomes the actual source the
      migration executes, or is deleted — a file that is shipped in the wheel,
      pinned by the packaging test, and never opened is worse than no file
- [ ] If any `.sql` files are kept alongside embedded constants, a test
      compares each file to its constant and fails on divergence
