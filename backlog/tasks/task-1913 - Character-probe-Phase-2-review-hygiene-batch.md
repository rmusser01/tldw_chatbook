---
id: TASK-1913
title: Character-probe Phase 2 review hygiene batch
status: To Do
assignee: []
created_date: '2026-08-02 04:15'
labels:
  - evals
  - character-probe
  - hygiene
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small findings deferred during the character-probe Phase 2 authoring UI review
loop (`feat/character-probe-phase2-1691`). None is a defect a user would hit as a
trap; each is an inconsistency worth closing while the code is fresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `CharacterBenchEstimate` applies one honesty rule: it either prints a zero count or refuses to, consistently across a 0-card draft and a missing probe set
- [ ] #2 The `character_bench` Delete branch is gated on a resolved row, matching the `bench` branch beside it
- [ ] #3 `character_probe.storage` is imported at module scope in `evals_state.py`, matching every other consumer
- [ ] #4 Probe-set selection does not depend on a `created_at DESC` tie at one-second granularity
- [ ] #5 A target row carrying BOTH `prefix` and `system_prompt` is flagged as corrupt in the targets listing rather than silently degrading to an unsuffixed label
- [ ] #6 `EvalsViewModel.character_benches()` and `probe_sets()` have direct unit tests
- [ ] #7 `CardRow.__init__` calls `super().__init__()` before setting instance attributes, matching `Widgets/emoji_picker.py`
- [ ] #8 `SnippetEditor._handle_import_file_selected` reads its file off the UI thread, matching the shared `LibraryRail._read_import_file_off_thread` seam its two siblings now use
<!-- AC:END -->
