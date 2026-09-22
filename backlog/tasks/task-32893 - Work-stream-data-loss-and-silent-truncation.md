---
id: TASK-32893
title: "Work stream: data loss and silent truncation"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-data-loss
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six paths that destroy or silently drop user data while reporting success: a rechunk that hard-DELETEs
every chunk and reports `"rechunked"` when the replacement is empty, an extraction refusal swallowed
because `"interrupted"` is missing from the failure-reason set, an `expected_version` parameter accepted
and never threaded through, two voice-profile stores that cannot tell "absent" from "unreadable", and
three silent export truncations.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No path deletes user rows before its replacement is known non-empty
- [ ] #2 Every swallowed failure in the children either surfaces or is logged with its reason
- [ ] #3 `expected_version` is either honoured or removed from the signature
- [ ] #4 Each child has a test that reproduces the loss before the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-dataloss` as `90c45d67b2`, then merged with `origin/dev` and reworked as
`4eb7213492`. Not pushed. All six real.

**The rework is the interesting part.** The branch's TTS fixes assumed the two voice managers were not a
shared hierarchy, so they added three module-level helpers. While the branch was being written,
`51abc0dc9a` "refactor(tts): HiggsVoiceProfileManager adopts VoiceManagerBase (TASK-32863)" landed on dev
and made both managers subclass `VoiceManagerBase` -- invalidating that premise. Reworking onto the base
shrank the TTS diff from **+175/-42 to +126/-45** and deleted all three helpers. `VoiceManagerBase` now owns
`load_profiles` (`{}` when absent, **raises** when unreadable), `_backup_before_save` (copy + prune, gated on
`keep_backups`), and `_list_backups` (ordered by **parsed timestamp**, mtime fallback for legacy naive-local
names), with `BACKUP_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"` -- ADR-173 UTC+Z, filename-safe since the canonical `:`
form cannot be a path. Chatterbox's entire fix is now `keep_backups = True`.

**Dev's refactor fixed none of the four durability properties.** It relocated the defect: the swallowing
`load_profiles` moved from two managers into one base method. And it **added a test pinning the defect** --
see TASK-32906 item 8.

Decisions made where the brief offered a choice:
- **`_EXTRACTION_FAILURE_REASONS` was kept, not replaced.** It is a hand-maintained allowlist with no
  enumerating producer: only 2 of its 11 members are ever emitted, the other 9 have zero producers anywhere.
  So it *will* drift again. Mitigated rather than redesigned -- added the missing `"interrupted"`, replaced
  two bare `except CollectionsCaptureError: pass` with one logging helper, and added a test that AST-scans
  the service for its `reason=` literals and asserts they are a subset of the set.
- **`expected_version` was removed, not threaded.** The client's `delete_workspace_note` has no version
  parameter and the `expected-version` header exists only on `/api/v1/notes/*` and
  `/api/v1/writing/manuscripts/*` -- nothing under `/api/v1/workspaces/*`, which locks via a `version` field
  in the request *body*, and a DELETE has no body. Threading a header there would recreate the same false
  guarantee. Removing makes the absence visible.

Also corrected: item 2's symptom is not "recorded as a success" as filed -- the row is left **wedged at
`processing` holding its lease**, contradicting `cancel_extractions`' own docstring.

Item 6 turned out to have a fourth angle: `status`!=saved and `favorite=True` each produced a well-formed
**empty** export byte-identical to "you have nothing saved", and `domain` was applied in Python *after*
`search_media(limit=size)` so it only ever saw one page. All three now raise; the domain refusal fires only
when the underlying page came back full, so a page that saw the whole scope still exports.

Gates: preflight GREEN; size ratchet failing node-id **set identical to `origin/dev`** (5 reds -- an
in-flight report of 6 was a miscount, re-measured); full `Tests/TTS/` 356 red node ids, sets identical to
dev, zero regressions, +9 passing which are exactly the new safety cases.

Two notes for elsewhere:
- **An ADR-126 workaround worth keeping.** `Tests/Media/test_local_media_reading_service.py` raises
  `RecoveryRequired: raw_source_selection_changed` in a clean worktree -- not from the fixture, but because
  a media-DB migration lazily loads config *inside* an active raw-participant scope. Forcing the load first
  (`import tldw_chatbook.Chunking` before invoking pytest) makes the whole file runnable locally, turning a
  "CI-only" file into a local red->green loop. Belongs in `backlog/docs/lessons-testing-evidence.md`, filed
  once for the whole tier-2 wave rather than per-branch (those files conflict on every dev merge).
- **Pre-existing wart on dev, deliberately not fixed here:** `voice_manager_base.py:373-376` has four
  stacked `@abstractmethod` decorators on `export_profile`, orphaned by `3dc3727629`. Harmless and
  idempotent, but it belongs to a dev-side commit, not inside this merge.
<!-- SECTION:NOTES:END -->
