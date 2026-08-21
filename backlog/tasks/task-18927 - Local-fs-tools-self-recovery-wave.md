---
id: TASK-18927
title: 'fs_* local tools: self-recovery wave'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - agents
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's tool self-recovery philosophy (2026-08-19 hermes-release review) to chatbook's local fs_* tools (Tools/local_tool_impls.py): when a tool call almost-succeeds, return an actionable recovery result instead of a bare error, so the agent wastes fewer turns — and fewer approval cycles — on friction. Specifics, mirroring hermes's wave: fs_edit detects already-applied edits (success no-op) and diagnoses whitespace mismatches with nearest-match candidates; fs_write verifies on-disk content after writing; fs_grep/fs_glob zero-match results probe near-misses (closest filenames/patterns); large outputs always carry pre-truncation size plus the spill path. Recovery messages are bounded, clearly machine-generated hints — never instructions the model must obey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 fs_edit: an edit whose change is already present in the target returns success with a no-op note instead of a match error
- [ ] #2 fs_edit: a failed match returns a bounded diagnosis that visualizes whitespace differences against the nearest candidate locations in the file
- [ ] #3 fs_write: post-write verification compares the written bytes against the intended content and reports any mismatch honestly
- [ ] #4 fs_grep/fs_glob: a zero-match result includes bounded near-miss suggestions (e.g. closest existing filenames) rather than a bare empty result
- [ ] #5 Every recovery/hint message is length-bounded and labeled as a tool-generated hint; unit tests include adversarial inputs (huge files, binary content, path edge cases) and assert bounds
- [ ] #6 No permission-model change: recovery never bypasses Allow/Ask/Off or any approval gate — verified by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: behavior improvements within the existing local-tool boundary (ADRs 030/032/033 govern the boundary itself); no schema or interface change.

1. fs_edit: idempotence detection + whitespace-visualized no-match diagnosis
2. fs_write: read-back verification
3. fs_grep/fs_glob: near-miss probing with bounded suggestions
4. Bounds/hint-labeling tests, adversarial cases, docs note in agent-runs-and-tools.md
<!-- SECTION:PLAN:END -->
