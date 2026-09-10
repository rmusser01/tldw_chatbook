---
id: TASK-32148
title: Add repeatable opt-in live TTS cancellation and playback validation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:34'
updated_date: '2026-09-09 14:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the saved speech validation probes into a reproducible developer tool, and close real Kokoro cancellation and sustained-playback coverage gaps on macOS ARM.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The opt-in command accepts explicit local assets, runtime and new private output paths; help and cheap tests cause no inference, audio, network downloads or user-configuration writes.
- [x] #2 Actual delegated PyTorch and ONNX native calls expose monotonic entry and exit evidence so cancellation is proved to overlap inference, followed by successful successor playback.
- [x] #3 Repeated synthesis and playback retain complete audio and record bounded observer memory, resource snapshots and cleanup assertions rather than unconditional success flags.
- [x] #4 Validation artifacts identify tested application, packages, models, audio bytes and physical playback outcomes; full-content transcription remains separate from transport and cleanup evidence.
- [x] #5 Targeted harness regressions and real macOS CPU, MPS and ONNX runs pass; any unavailable cases have concrete backlog follow-ups and prerequisite evidence.
- [x] #6 CLI and evidence input paths pass centralized validation; audio references remain within the selected evidence run, malformed evidence yields a consistent validation error, and optional recognizer absence has actionable guidance before work starts.
- [x] #7 New validation helpers document and type their callable contracts while help and import remain inert.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 TTS provider ownership, ADR-039 Global/Studio ownership and ADR-040 Lab audition ownership apply.
Reason: Test-only promotion of existing live validation seams; production fixes discovered by the probe are tracked separately before implementation.

1. Add cheap failing negative-control tests for opt-in admission, real inference overlap, playback duration/content evidence and joined ownership.
2. Promote the existing validators into import-safe local-asset runner and separate local-ASR verifier, with an isolated private worker, bounded observations and explicit package provenance; provide a concise runbook.
3. Observe unchanged real PyTorch forward and ONNX session calls, prove cancellation falls between entry and exit, and verify terminal lifecycle plus successor playback and bounded repeated runs.
4. Execute serialized macOS CPU, MPS and ONNX qualification with complete source/playback artifacts and separate content results. Record any production defect before fixing it.
5. Run targeted harness and affected lifecycle tests, verify real cleanup and unchanged user configuration, and file remaining hardware/provider prerequisites.

6. PR #2545 review: add failure-first negative controls for central CLI path validation, evidence shape and run confinement, missing optional ASR dependencies, and inert import/help. Repair those admission boundaries and complete helper docstrings/type hints without weakening content criteria; rerun targeted harness checks and a bounded real verifier/runner compatibility check. ADR required: no; apply existing path/input/dependency policies within this test-only tool, with no new trust root or provider contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an opt-in local-asset validation runner and separate full-content ASR verifier, with a documented private-profile workflow. The mounted Speech Lab and trusted Console path use real inference, codecs and physical playback. Negative controls cover admission, actual inference overlap, full-duration playback, bounded observations, source/asset identity and retained cleanup; imports/help remain inert. Existing ADR-023/039/040 apply; no new architecture decision is needed for the test-only tool.

The original CPU/MPS/ONNX English matrix passed all scenarios with 33 normalized-exact complete clips, three observed native-overlap Stops, successor recovery and bounded repeats. The final installed wheel passed another nine complete clips and three Stop/recovery controls. Failed earlier probes remain distinct. Targeted harness and affected integration regressions pass; the user config hash is unchanged and all recorded owned processes exited. Evidence and scope limits are in Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,integration}/README.md and Docs/Development/TTS/Live_Validation.md. This does not claim acoustic loopback, full-shell navigation or indefinite memory-leak freedom.

PR #2545 review hardened centralized CLI path checks, strict consumed evidence validation, complete successful-clip accounting, descriptor-relative run-confined audio reads, pre/post file identity, optional ASR guidance and temporary profile isolation. Public runner/verifier contracts now have types and documentation; negative controls include path swaps, symlinks, malformed data and private-profile restoration. A stale setuptools build was rejected by full file-set verification; its premature playback attempt is excluded from final wheel evidence and a clean rebuild was requalified. Post-review evidence: Docs/QA/tts-macos-burndown-2026-09-09/review/README.md. The post-rebase targeted selection passed 405 tests with one explicit MPS skip; the final migration/resolver/provider selection passed 81 overlapping tests. The clean installed wheel matches 2,275 Python files (2,266 application plus nine profile-core files). Five serialized tuples passed generation/playback and joined cleanup; English CPU/MPS/ONNX produced nine exact full transcripts and three real Stop/recovery controls. Japanese/Mandarin raw content differences remain review-required. All 23 recorded owned PIDs exited and user config stayed unchanged. No new maintained-source Ruff diagnostics; existing debt and frozen QA snapshots are documented separately. Existing ADRs remain applicable.
<!-- SECTION:NOTES:END -->
