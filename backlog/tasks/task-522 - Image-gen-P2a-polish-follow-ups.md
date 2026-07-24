---
id: TASK-522
title: >-
  Image-gen P2a polish follow-ups
status: To Do
assignee: []
created_date: '2026-07-24 09:05'
updated_date: '2026-07-24 09:05'
labels:
  - image-generation
  - console
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking findings from the whole-branch review of the image-gen Console card + variants feature (P2a; spec `Docs/superpowers/specs/2026-07-23-image-gen-console-card-variants-design.md`). None block the P2a PR; group into one cleanup pass. Distinct from [[task-497]] (P1 polish) and [[task-498]] (egress/SSRF), and from the deferred P2b feature slice (TTS, Style-preset picker, prompt-from-context).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] In-memory screen-state restore path (`ChatScreen.restore_state` → `_restore_console_message`) rehydrates `generation_metadata` the same way it already rehydrates attachments, so a tab-switch-restored generation message keeps its card (today only the DB-resume path hydrates).
- [ ] Generation records capture the resolved seed/model where the backend reports them (card currently shows the requested seed or "random" and no model; extend `run_generation_batch`/`GenerationVariantMeta` population when `ImageGenResult` grows the fields).
- [ ] `/generate-image` clamps the initial batch to `max_variants_per_message` (today only regenerate enforces the cap; a misconfigured `default_batch` can exceed it).
- [ ] Stale narrative fixed in `test_console_generation_store.py`'s round-trip test (comments claim `restore_persisted_session` doesn't hydrate — false since the resume fix; drop the redundant manual hydrate calls) and `restore_persisted_session`'s docstring documents the hydration it now performs.
- [ ] Draft is restored to the composer when the generation batch RAISES (today only the zero-success return path restores it).
- [ ] Test-coverage nits closed: generation-vs-sibling precedence pinned with `sibling_count=3`; exact-limit (80-char) content-marker boundary asserted; empty-negative-prompt card branch covered.
- [ ] Cosmetics: `console-generation-card*` CSS classes either get TCSS rules or are removed (currently inert; styling is set in Python); new DB ops' docstrings stop promising `CharactersRAGDBError` for raw `sqlite3.IntegrityError` (or wrap it), matching whichever convention the sibling methods adopt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
