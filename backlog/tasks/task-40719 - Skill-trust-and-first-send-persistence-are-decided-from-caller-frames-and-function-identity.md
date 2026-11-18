---
id: TASK-40719
title: "Skill trust and first-send persistence are decided from caller frames and function identity, against ADR-356 rule 4"
status: Done
created_date: 2026-11-13 06:28
labels:
- follow-up
- architecture
- agents
- console
priority: medium
references:
- backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
- tldw_chatbook/app_skill_wiring.py
- tldw_chatbook/Chat/console_received_dispatch.py
- Tests/Architecture/test_chat_code_identity_ratchet.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** ADR-356 rule 4 says: "Orchestration must not inspect function objects, code/defaults/closures, descriptors, caller stacks/locals or coroutine implementation to establish authority." #3480 brought that pattern to the ordinary chat path, with the workflow flag on or off. None of the three sites below exists at e10ce63fe8^1:
- `app_skill_wiring.py:30-51` takes `sys._getframe(1)`. It trusts a skill candidate only if the caller's `f_code` is the code of the `local_skill_trust_service` getter or of `ensure_local_skill_trust_service`, and the caller's locals match (:36-46).
- `Chat/console_received_dispatch.py:892-907` saves the first-send setting only while `_function_unchanged(_NATIVE_CONFIG_CHECK)` holds. That helper is `Agents/activation.py`'s `callable_source_current` (:151), imported under that name via `Chat/console_ordinary_tree.py:10-11`. It compares `__code__`, defaults, kwdefaults and closure cells.
- `Chat/console_received_dispatch.py:789-792` also gates `_hook_read` on `submit_draft.__func__ is ConsoleChatController.submit_draft`.

If those functions are wrapped, reloaded or patched, skill trust is silently not recorded, or the first-send setting is not saved.

**Count.** `git grep -E -n '__code__|_getframe|\bcr_code\b|\bf_code\b' <rev> -- 'tldw_chatbook/*.py'` gives:
- 369 before #3480 (e10ce63fe8^1);
- 411 at its merge, unchanged at 7579f75ea9.

`Tests/Architecture/test_chat_code_identity_ratchet.py` (TASK-36944.9) covers only Chat/, where the count fell from 57 to 56. The growth is elsewhere:
- Agents/ 25 → 36;
- Tools/ 7 → 16;
- Workspaces/ 8 → 14;
- MCP/ 37 → 41;
- runtime_policy 0 → 4;
- `app*.py` 26 → 29;
- Backup_Recovery/ 30 → 32;
- Personal_Context/ 0 → 2;
- Library, Notes and Prompt_Management up by one each.

`config.py` went from 9 to 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skill trust is recorded from runtime-owned registration, the same way whoever calls it, with no caller-frame inspection.
- [x] #2 The first-send setting is saved whether or not the config check function is wrapped, and no Send path is gated on function or code identity.
- [x] #3 An architecture ratchet covering all of `tldw_chatbook/`, not only Chat/, is pinned at the current count and fails on any increase in sites that read caller frames or function, code, default or closure identity.
<!-- AC:END -->

## Implementation Plan

ADR required: no. ADR-356 rule 4 already decides that function/code identity and caller frames establish no authority; this applies it to three remaining sites.

1. Behaviour tests first, shown failing on the unmodified tree:
   - skill trust: a build reached through a wrapped trust getter or a wrapped first-use initializer is recorded for the active native publication; a setter-injected service, a build on another thread, a build outside a publication and a candidate that is not the published slot are not;
   - first send: the setting reaches config.toml when the config writer is wrapped;
   - Send: a wrapped `submit_draft` (and a wrapped `prepare_hooks_v2`) still shares the attempt's one hook read.
2. Replace the checks with the runtime state already there: the active publication marker (same thread/task, same app) and the published slot for skill trust; the installed config module, config identity, app/runtime/controller/store, quiescence and the mutation CAS for first send (drop the now-unused config-function identity records); pass the attempt's hook read whenever it exists.
3. Add a whole-package AST ratchet over `tldw_chatbook/` for caller-frame and function/code/default/closure identity reads, pinned at the post-change counts, with a negative control.

## Implementation Notes

All three named sites now decide from runtime-owned registration and current state (ADR-356 rule 4); no ADR change.

- **Skill trust** (`app_skill_wiring.record_native_skill_trust_candidate`, internal): the `sys._getframe(1)` / `f_code` / `f_locals` match is gone. Any caller on the publication's thread and task could pass the slot's current value; ADR-356 rule 4 accepts that, and the docstring says so. A candidate is recorded when the publication registered for this app is active on the same thread and task and the app's trust slot holds that candidate. Only the two first-use builders call it, after filling the empty slot; the setter never does. A wrapped getter or initializer is now recorded like the stock one. A setter-installed service, a build on another thread, a build for another app or after the publication, and a non-slot candidate are still not recorded (all pinned).
- **First-send save** (`Chat/console_received_dispatch._persist_first_send_when_quiescent`): dropped the captured-checker identity gate and the checker itself, which only compared the code/defaults/closures of 20 config writers. The save keeps its current-state guards: installed config module, `current_config_identity()`, app/runtime/controller/store, quiescence and the mutation CAS. Removed the only-use `record_first_send_sources`, `first_send_persistence_source_current` and `_FIRST_SEND_PERSISTENCE_SOURCE`. A wrapped `apply_settings_mutation_to_cli_config` now saves.
- **Hook-read sharing**: received dispatch (`submit_draft.__func__`) and `console_automatic_submission.prepare_submission_hooks` (`prepare_hooks_v2.__func__`, the same pattern on every Send) now pass the attempt's hook read whenever it exists. Before, a wrapped collaborator made the Send read hook consent a second time. Both gates predate #3480: 077e5bfa09 (2026-10-31) added them, and at e10ce63fe8^1 they sit in `Chat/console_runtime.py` (`submit_draft`, ~3189) and `Chat/console_automatic_submission.py:247-248` (`prepare_hooks_v2`).
- **Ratchet**: `Tests/Architecture/test_code_identity_ratchet.py` scans `tldw_chatbook/**/*.py` and counts, per name, every `Attribute` and exact string `Constant` naming a frame/coroutine field (`_getframe`, `currentframe`, `_current_frames`, `getouterframes`, `extract_stack`, `f_back`, `f_code`, `f_locals`, `f_globals`, `cr_/gi_/ag_code|frame`) or a function identity field (`__func__`, `__code__`, `__defaults__`, `__kwdefaults__`, `__closure__`, `cell_contents`, `__globals__`, `__wrapped__`). Pinned at the post-fix counts (1,938 total; this change removed 17). Includes shape, lookalike, injected-site and whole-package controls. It runs in the required arch lane: it is listed after `test_chat_code_identity_ratchet.py` in `.github/workflows/derived-artifacts.yml` (`pr-fast-lane-arch`) and in `ARCH_REVIEW_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`). Its docstring states two limits: names built at runtime are not seen, and per-name package totals let a removal in one file pay for an addition in another (`ponytail:` note: pin Agents/ or Chat/ separately if needed).
- **Tests**: new `test_first_use_build_is_recorded_however_the_trust_service_is_reached` plus four untrusted-path controls (`Tests/Chat/test_console_trust_initializer_ownership.py`), `test_a_wrapped_collaborator_still_shares_the_attempts_one_read` (`Tests/Chat/test_console_send_hook_read_sharing.py`), and `test_viewless_first_send_is_saved_through_a_wrapped_config_writer` (`Tests/UI/test_console_runtime_ownership.py`). All five failing nodes fail on base and pass on head. The survivor test that pinned refusal of a replaced checker now expects the save to go through a wrapped writer. Like every variant in that file, it fails before reaching the save on both trees on this machine (the child never enters), so CI must qualify `test_runtime_first_send_saves_through_a_wrapped_config_writer`.
- **Scope**: AC #2's "no Send path" was applied to the gates this task names plus the matching `prepare_hooks_v2` gate. Send preparation still has other identity checks, e.g. `_capture_console_skill_trust_source`, `_console_skill_metadata_current` and `console_configuration_preparation`'s `_function_unchanged` guards. The new ratchet caps them; it does not remove them.
- Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `tldw_chatbook/app_skill_wiring.py`, `tldw_chatbook/Chat/console_received_dispatch.py`, `tldw_chatbook/Chat/console_automatic_submission.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Backup_Recovery/config_participants.py`, the four test files above and `Tests/Chat/test_console_first_send_survivor_native.py`.

