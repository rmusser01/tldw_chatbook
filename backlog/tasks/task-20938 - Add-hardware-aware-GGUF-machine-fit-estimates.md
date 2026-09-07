---
id: TASK-20938
title: Add hardware-aware GGUF machine-fit estimates
status: Done
assignee: []
created_date: '2026-08-22 19:44'
updated_date: '2026-08-22 22:44'
labels:
  - models
  - ui
  - ux
dependencies:
  - TASK-20935
references:
  - backlog/decisions/080-model-machine-memory-fit-estimation.md
  - Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md
  - Docs/superpowers/plans/2026-08-22-remote-model-machine-fit-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build on deterministic Remote variant guidance with transparent 32,768- and 65,536-token memory scenarios that compare a GGUF allowance with local RAM without implying model-context support, runtime compatibility, or successful inference.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Machine facts are collected through a provider-neutral, bounded, off-loop capability seam with independent system-memory and accelerator evidence states, fixed reason codes, and exact input/output limits.
- [x] #2 Each candidate shows a text-labeled memory-scenario classification, both estimated loads, working-budget margin, and adjacent limitations; no label claims that the model supports 32K/64K or that a runtime will load successfully.
- [x] #3 Unsupported platforms and incomplete CPU, RAM, GPU, or unified-memory evidence fall back to deterministic guidance without blocking browsing or installation.
- [x] #4 LLMScreen owns accepted machine facts, observation time, worker, and generation across body recomposition; RemoteView requests rechecks and renders hydrated immutable state without stale generations replacing newer facts.
- [x] #5 The estimation policy and platform-specific probes have focused boundary, lifecycle, process-cleanup, failure, privacy, and Linux, macOS, and Windows evidence before the feature is enabled.
- [x] #6 Projections use exactly 32,768 and 65,536 tokens, lead with the 65,536-token scenario, expose both estimated loads and the RAM working budget, and show current available-memory pressure separately without changing the stable classification.
- [x] #7 Observed VRAM is shown per device when bounded platform evidence is available, Apple unified memory is shown once, multiple devices are never blindly summed, and the UI states that accelerator evidence does not change the runtime-neutral RAM rating.
- [x] #8 Below 72 RemoteView content cells the repository workflow becomes a keyboard-complete one-pane drill-down with Back and collapsed estimate details; production 80×24 evidence covers both rail states, long names, overflow, focus restoration, Recheck, and Install.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/080-model-machine-memory-fit-estimation.md
Reason: This feature establishes a long-lived provider-neutral capability boundary, privacy contract, bounded platform-probe contract, and recomposition-stable Models-screen ownership.

1. Add immutable machine-memory domain values and exact pure 32,768-/65,536-token projection tests.
2. Add injected, bounded macOS/Linux/Windows RAM and optional VRAM probes with cleanup/privacy tests.
3. Add pure presentation copy and LLMScreen-owned generation, refresh, and recomposition hydration.
4. Add RemoteView machine evidence, current-pressure warnings, stable in-place candidate updates, and the 72-cell drill-down.
5. Prove the feature in production 80x24 layout, run targeted verification, self-review against ADR-080, and record exact task evidence.

Detailed plan: Docs/superpowers/plans/2026-08-22-remote-model-machine-fit-implementation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-080 provider-neutral machine-memory capability, bounded
macOS/Linux/Windows probe, pure 32,768-/65,536-token projection policy,
runtime-neutral presenter, LLMScreen-owned refresh/generation lifecycle, and
RemoteView machine evidence plus adaptive one-pane workflow. The no-header
tradeoff is intentional: estimates use exact catalog candidate bytes and a
visible heuristic instead of adding remote GGUF range reads or implying
model-context/runtime support.

Core implementation and evidence live in
`tldw_chatbook/Model_Artifacts/machine_memory.py`,
`machine_memory_probe.py`, `UI/Screens/model_memory_presenter.py`,
`model_remote_view.py`, `llm_screen.py`, their four focused feature test files,
and `Tests/UI/test_llm_screen_lab_adoption.py`. Production-width evidence also
surfaced and fixed three bounded integration defects: the governed Remote CSS
sheet was regenerated from existing source, narrow machine actions now stack,
and completion reapplies the existing pane visibility policy after its internal
recompose. No new architecture decision was required beyond accepted ADR-080.

The final-review fix wave removed the blocking command-reader thread in favor
of deadline-polled owned-pipe reads and bounded both child-reap waits; completed
the snapshot state/reason/source/unified-memory matrix; made projection creation
policy-owned while validating exact estimate allowances and RAM budget; removed
duplicate AMD/NVIDIA prefixes at the real probe/presenter boundary; added exact
LLMScreen wall/monotonic clock seams that survive failed refresh and real
recomposition; and replaced the long-filename hidden-text oracle with exact
painted compositor evidence in both rail states. Full RED/GREEN details are in
`.superpowers/sdd/2026-08-22-remote-model-machine-fit-implementation/final-review-fix-report.md`.

The user-authorized second final-review fix wave closes the two remaining
review blockers. On Windows, command output now uses a unique, local-only,
single-instance named pipe whose server read handle is created with
`PIPE_NOWAIT`; direct `ReadFile` polling therefore has the documented immediate
empty-read behavior and no longer calls `PeekNamedPipe` on a synchronous
`subprocess.PIPE`. The parent owns and closes both pipe ends, the shared
collector retains the exact output/deadline limits, and injected cross-platform
tests cover blocked legacy readiness and Win32/CRT handle cleanup, including a
failed descriptor conversion. This macOS host did not claim a real Windows run.
The production 80×24 filename oracle now
reacquires and scrolls the currently mounted filename after finite machine-state
refreshes replace candidate children, then retains its exact compositor-cell,
multi-row, no-ellipsis, no-overflow, current-identity checks in both rail states.
This routine boundedness/test-harness correction implements ADR-080 without a
new architecture decision.

The PR review follow-up validated and addressed all three Qodo findings. A
non-forced request received during an active probe now hydrates the currently
mounted RemoteView with retained screen-owned evidence without launching a
second worker. ADR-080's RAM reserve calculation now has one validated public
domain helper consumed by both projection and presentation, with regression
coverage preventing presenter drift. The eight public callables identified by
the compliance review now document their arguments, returns, and applicable
errors using the required Google-style sections. These are lifecycle,
policy-ownership, and documentation corrections within the accepted ADR; no
new architecture decision was required.

Authoritative targeted evidence (run from the feature worktree):

```text
../../.venv/bin/pytest -q Tests/test_probe_import_provenance.py -s
1 passed, 1 dependency warning; imports resolved from this worktree

../../.venv/bin/pytest -q Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py Tests/UI/test_model_memory_presenter.py Tests/UI/test_model_remote_view.py
215 passed, 1 dependency warning

../../.venv/bin/pytest -q Tests/UI/test_llm_screen_lab_adoption.py -k "machine_memory or memory_clocks or remote_drill_down_install_action or remote_memory_scenarios_survive_recompose or remote_completion"
15 passed, 126 deselected, 1 dependency warning

for run_index in 1 2 3 4 5; do ../../.venv/bin/pytest -q Tests/UI/test_llm_screen_lab_adoption.py::test_remote_memory_scenarios_survive_recompose_at_80_columns --tb=short || break; done
5/5 serial repetitions passed, each with 1 existing dependency warning

../../.venv/bin/pytest -q Tests/UI/test_widget_css_consolidation.py
31 passed, 1 existing dependency warning

../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
passed; all five generated CSS artifacts reproduce from source

../../.venv/bin/ruff check tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py tldw_chatbook/UI/Screens/model_memory_presenter.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py tldw_chatbook/UI/Screens/model_browser_state.py Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py Tests/UI/test_model_memory_presenter.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_model_browser_state.py
passed; All checks passed

../../.venv/bin/ruff format --check tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py tldw_chatbook/UI/Screens/model_memory_presenter.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py tldw_chatbook/UI/Screens/model_browser_state.py Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py Tests/UI/test_model_memory_presenter.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_model_browser_state.py
passed; 12 files already formatted

../../.venv/bin/python -m compileall -q tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py tldw_chatbook/UI/Screens/model_memory_presenter.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py
passed

git diff --check
passed
```

The planned `Tests/UI/test_ui_css_parse.py` path does not exist, so the full
canonical `Tests/UI/test_widget_css_consolidation.py` suite plus the bundle-sync
guard was used under controller ruling. A once-only, worktree-first, scratch
HOME/XDG/config-isolated diagnostic passed on platform class `Darwin-arm64`:
unified memory had a positive total, exactly one Apple shared marker was
present, and an injected guard proved no discrete accelerator command was
attempted. No observed capacity/device values were persisted. The Task 5 report
retains the RED/GREEN production diagnoses; this task record contains the
authoritative completion commands and results.
<!-- SECTION:NOTES:END -->
