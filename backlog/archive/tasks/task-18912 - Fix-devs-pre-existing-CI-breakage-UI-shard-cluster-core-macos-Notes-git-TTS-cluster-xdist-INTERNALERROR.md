---
id: TASK-18912
title: 'Fix dev''s pre-existing CI breakage: UI-shard cluster + core-macos Notes-git/TTS cluster + xdist INTERNALERROR'
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-20'
updated_date: '2026-08-20'
labels:
  - ci
  - tests
  - tech-debt
dependencies: []
priority: high
archived_reason: duplicate of TASK-18609/18610/18611 triage chain
---

## Description

Dev's Tests workflow fails on most PRs regardless of their content — every
PR since 2026-08-19 has had to merge over red with manual provenance
comments (PRs #1829, #1834, #1835).

## Resolution

CLOSED AS DUPLICATE (2026-08-20): this breakage is already tracked and
actively worked by the existing triage chain — TASK-18609 (pass 1, merged
14f398673), TASK-18610 (pass 2, In Progress: git-push and TTS clusters
fixed via PR #1833; UI-119/wizards/git-integration/stragglers open),
TASK-18611 (library-prompts-canvas cluster, To Do). This task was filed
before discovering that chain.

## Retained value

This task's notes carried a fresh failure inventory harvested from PR
#1835's core-test-results artifact plus all 12 shard logs (post-pass-2
residue): the 44 core-macos failures with zero console overlap; the local
trio vs CI-only-24 distinction in library_prompts_canvas; the GGUF-windows
flake pattern; the personas flake passing 4/4 locally on both branches.
That inventory informed the current-state view of 18610's remaining ACs.
Work continues under 18610/18611.
