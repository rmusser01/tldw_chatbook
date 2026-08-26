---
id: TASK-17066
title: Source-type-aware relevance gate for non-paper evidence
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-17 03:58'
updated_date: '2026-08-17 03:59'
labels:
  - research
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The recorded repositories-lane baseline (gate_pass 0.29 vs 0.72 for papers) shows the relevance gate's single usefulness prompt is calibrated for papers: dataset/software/figure records and scholarly metadata records fail the comprehensively-answers bar even when topically on-point. Teach the gate what kind of evidence it is judging, using the catalog's own categories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A source-kind classifier maps catalog categories to evidence kinds (repositories to repository records, open_research_graph to metadata records, everything else and unknowns to the paper default) driven by the result's provider metadata,Repository and metadata records are evaluated with a source-type note that counts topically-related supporting evidence (data, methods, artifacts) as relevant without needing to directly answer the question,Paper-classified and unclassified results keep the current prompt byte-for-byte,Tests pin the classifier mapping, the note's presence per kind, and unchanged prompts for papers/unclassified,The repositories lane is re-measured live and the baseline doc records the gate_pass comparison against the recorded 0.29
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD source_kind_for_provider in research_source_catalog (category-driven: repositories -> repository, open_research_graph -> metadata, else paper default for unknowns)
2. TDD the gate prompt variant in search_result_relevance: results whose metadata.provider classifies as non-paper carry a source-type note in the eval input; papers and unclassified results keep the current input unchanged
3. Full suites plus lint
4. Live re-measurement of the repositories lane against the recorded 0.29 gate_pass; doc update
5. PR, Qodo loop, merge
ADR required: no - prompt calibration within the existing gate contract; the fallback and honesty footer unchanged
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `source_kind_for_provider(provider)` in the catalog module: category-driven (repositories -> "repository", open_research_graph -> "metadata", everything else including unknowns/None -> "paper" -- the strict default the gate prompt was calibrated for). Classification keys off the result's `metadata.provider`, which `papers_to_evidence` already stamps.
- `search_result_relevance` prepends a source-type note to the eval input for non-paper kinds: repository records count as relevant when "topically related and could serve as supporting evidence (data, methods, or artifacts)" without needing to directly answer; metadata records when they "describe a work topically related". The prompt template itself is untouched (note rides the input line), and paper/unclassified results get the byte-identical input as before (pinned by test).
- Measured justification: repositories lane gate_pass 0.29 vs 0.72 (graph) / 0.93 (papers) in the task-17385 baselines; the note targets exactly that gap. The zero-relevant fallback remains the backstop either way.
- **Live re-measurement blocked at ship time**: both local llama.cpp endpoints (9191 and 52864) refused connections. Re-run once the endpoint is back and record in the baseline doc:
  `python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --providers repositories --llm-base-url http://127.0.0.1:<port>/v1`
  Expected signal: gate_pass_rate meaningfully above the recorded 0.29 with citation_accuracy holding at 1.00.
- Verified TDD: 2 classifier tests + 3 gate-prompt tests (repository note, metadata note, byte-identical prompts for papers/web) written first and watched failing; sweep 100 passed; ruff clean on touched files (remaining findings in the pipeline test file are pre-existing drift).

## Re-measurement (2026-08-17, endpoint on :9191)

- gate_pass_rate **0.29 → 0.42** (+45% relative); citation_accuracy **held at 1.00** (72/72 markers); claim_support 0.97 → 1.00; cited_sentence_ratio 0.52 → 0.75; quote_grounding 0.33 (one run quoted, all verified).
- Honest residual: repositories still pass at under half the paper rate (0.93) — partly genuine (repository records are supporting material, not answers, for general-purpose questions). Fully closing would need a per-kind threshold or category-tuned question sets; the top-3 flagged fallback covers the remainder. Comparison table in the baseline doc.
