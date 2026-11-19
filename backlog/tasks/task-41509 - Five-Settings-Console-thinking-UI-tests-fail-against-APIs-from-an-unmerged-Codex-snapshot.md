---
id: TASK-41509
title: "Five Settings Console-thinking UI tests fail against APIs from an unmerged Codex snapshot"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:31
labels:
- follow-up
- tests
- settings
priority: medium
references:
- Tests/UI/test_settings_console_thinking.py
dependencies:
- TASK-41031
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15). `Tests/UI/test_settings_console_thinking.py` has 5 failed nodes that call APIs which exist only in the unmerged Codex snapshot bf364a4bf7. This is separate from the two reasoning tests TASK-41031 already lists (`test_local_qwen_reasoning`, `test_local_reasoning_toggle`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failing node is repointed at the current API, or deleted with the reason recorded
- [x] #2 The file runs on a PR lane and passes
<!-- AC:END -->

## Implementation Plan

1. Run the file plainly and under the bootstrap profile; check each failing node's API against dev and the Codex snapshot bf364a4bf7.
2. Repoint nodes whose API exists; delete nodes whose API never shipped, naming the tests that cover the shipped behaviour.
3. Lane the file (UI census tail) with its measured serial time.

## Implementation Notes

At base the file had 5 red nodes. Plain, 4 failed with `RecoveryRequired('raw_source_selection_changed')` and 1 with an ImportError. Under the bootstrap profile 2 stayed red. The APIs those two need exist in no product tree: `git grep` on the Codex snapshot bf364a4bf7 finds `ConsoleThinkingBlock`, `append_thinking_chunk` and `settings-console-show-thinking` only in its test files.

Per node:
- `test_thinking_default_on_saves_off_and_reopens_off[show_thinking]`: **deleted**. Its `console.show_thinking` setting, saved by the category Save, never shipped. The shipped setting is `console.show_model_thinking` (`#settings-console-show-model-thinking`), saved at once rather than by Save. Its three claims are pinned for that setting already:
  - default on: `test_settings_context_memory_controls.py::test_model_thinking_visibility_defaults_on_and_fails_safe`;
  - saves off: `::test_show_model_thinking_is_canonical_immediate_and_rolls_back`;
  - reopens with the saved value: `test_settings_thinking_visibility_journeys.py::test_thinking_visibility_rebases_after_config_reload`.
- `test_console_screen_refreshes_thinking_before_answer_text`: **deleted**. Its `ConsoleThinkingBlock` widget and `ConsoleChatStore.append_thinking_chunk` never shipped. Live thinking is the transcript's activity disclosure, pinned by `test_console_thinking_disclosures.py::test_first_live_evidence_expands_and_delta_updates_same_widgets` and `::test_visibility_gate_hides_only_thinking_and_restores_it_collapsed`. The hide-and-show through Settings is pinned by `test_settings_thinking_visibility_journeys.py::test_thinking_setting_updates_mounted_console_without_altering_evidence`.
- `test_reasoning_history_defaults_auto_and_remembers_current_model_override[llama_cpp|local_vllm|local_ollama]`: **kept**. Its APIs are current, and it failed only on the config read under the per-test sandbox. It is now `@private_profile_test`, the lane rule for a gated Tests/UI test that reads config (TASK-34982's lesson). Red plain at base (RecoveryRequired), green now.

The module docstring names the deletions and where their coverage lives. With the kept file at `-n 4` (load ~25), the four Settings covering nodes passed plainly; the two disclosure nodes fail plainly here with the same RecoveryRequired, and `test_console_thinking_disclosures.py` with `test_settings_context_memory_controls.py` gave 59 passed under a temporary bootstrap-profile plugin (removed). None of the covering files is on a lane (proposed follow-up in the round report).

**AC#2.** The file is a whole entry at the UI census tail: 3 passed in 23.1 s serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), load ~11.

Files: `Tests/UI/test_settings_console_thinking.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
