---
id: TASK-1467
title: >-
  Fix the order-dependent tests exposed by the parallel outcome diff (pass in some collection orders, fail in isolation)
status: To Do
assignee: []
created_date: '2026-07-30 11:35'
labels:
  - testing
  - bug
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-07-30 audit's parallel A/B (`backlog/docs/test-suite-audit-2026-07-30.md` §8) surfaced tests whose outcome depends on collection order: they passed in the full `-n 8 --dist loadscope` run on origin/dev, but **fail in isolation with the identical failure set on both clean dev and the quick-wins branch** — i.e. they depend on state seeded by whichever tests happen to share their worker bucket. Any change to file counts reshuffles loadscope buckets and flips them, so they will keep producing phantom "regressions" in every future outcome diff until fixed.

Confirmed order-dependent (fail alone on clean dev):
- `Tests/Performance/test_rag_citation_provenance_benchmark.py::test_local_runner_is_machine_readable_and_never_opens_a_socket`
- `Tests/Performance/test_rag_citation_provenance_benchmark.py::test_runner_consumes_each_representative_corpus_family`
- `Tests/Performance/test_rag_citation_provenance_benchmark.py::test_changing_selected_corpus_cases_changes_exercised_seam_inputs`
- `Tests/Performance/test_rag_citation_provenance_benchmark.py::test_repository_storage_candidate_runs_only_in_qualification_mode`
- `Tests/UI/test_library_prompts_canvas.py::test_library_prompt_editing_shows_unsaved_marker_and_save_clears_it`
- plus one of `Tests/Audio/test_audio_integration.py::TestEndToEndDictation::test_dictation_with_mock_transcription` / `Tests/UI/test_console_parallel_runs.py::test_navigating_away_with_busy_fleet_confirms_and_records_teardown` (the isolation batch had 6 failures; re-derive the exact sixth when starting — the other of the pair is contention-flaky rather than order-dependent)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Each listed test passes when run alone on a clean checkout AND in a full parallel run (no hidden dependence on sibling tests' state)
- [ ] The root state dependency for each is identified in the fix commit (fixture seeding, module import side effect, cached global, etc.)
- [ ] A junit outcome diff between two parallel runs with different `--dist` orderings shows none of them flipping
