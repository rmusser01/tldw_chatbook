---
id: TASK-41644
title: "Console configuration preparation still compares MCP capture and bound methods by identity"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- architecture
priority: low
references:
- tldw_chatbook/Chat/console_configuration_preparation.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** These are the last two TASK-41319 follow-ups (ADR-356 rule 4), in `Chat/console_configuration_preparation.py`:
- `_composition_mcp_capture_current` (`:1498`) checks the checker's `__code__`/`__globals__`;
- the bound-method comparisons at `:2473` and `:2670` test `__self__`/`__func__` identity, where `==` on the bound method expresses the same thing without reading code identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Neither site reads `__code__`, `__globals__` or `__func__` identity
- [x] #2 Existing preparation tests pass, and the code-identity ratchet is lowered accordingly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (ADR-356 rule 4 already decides this; the task applies it to two leftover sites).

1. `_composition_mcp_capture_current`: drop the `checker.__code__`/`checker.__globals__` reads and the recorded-namespace check; keep the controller type, the module's registered checker binding and the checker's own current-state verdict.
2. The two bound-method comparisons (`same` in the mounted selection capture, `same_callback` in the resident turn capture): replace the `__self__`/`__func__` identity reads with `==` on the bound methods (bound-method equality compares `__self__` by identity and `__func__` by equality, which for functions is identity).
3. Lower `Tests/Architecture/test_code_identity_ratchet.py` (`__code__` -1, `__globals__` -1, `__func__` -2) and `test_chat_code_identity_ratchet.py` (`__code__` -1).
4. Run the preparation/composition test files and `Tests/Architecture` serially.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both sites now decide without reading code identity (ADR-356 rule 4).

- `_composition_mcp_capture_current` keeps the controller type, the module's registered checker binding (`_stock_console_tool_composition_current is _CONSOLE_TOOL_COMPOSITION_CHECKER[0]`) and the checker's own verdict, and no longer reads the checker's `__code__`/`__globals__` or compares the recorded namespace. The controller's own copy of this check (`console_chat_controller.py`, the `deferred` branch before `_compose_mcp_provider`) is out of scope and unchanged.
- `same` (in `capture_mounted_configuration_owned`) and `same_callback` (in `_controller_capture_turn_configuration_snapshot`) compare bound methods with `==`, which in CPython compares `__self__` by identity and `__func__` by equality (identity for functions), so the verdict is unchanged.
- Ratchets lowered by exactly what was removed: `test_code_identity_ratchet.py` `__func__` 145 -> 141 (two reads per site), `__code__` 389 -> 388, `__globals__` 209 -> 208; `test_chat_code_identity_ratchet.py` `__code__` 49 -> 48. Measured with each test's own counter before and after.
- Evidence (2026-11-18, load 35-60 on 18 CPUs; base = dev d0486978dd in a detached worktree). Covering files (58 files over three `-n 4` runs: Chat/MCP/Plugins/Guardian/Agents preparation and queue files, the Console UI send/queue files, and the heavy controller/ordinary-tree/runtime-ownership/native-receipt files): every head failure also fails on base when its node ids are rerun there, except three that pass when rerun alone on head (`test_console_ssh_custody.py::test_original_virtual_model_operation_waits_for_real_ssh_stderr_return[ssh_console0]`, `test_console_send_admission_off_pump.py::test_a_second_enter_after_an_unknown_command_hint_sends_it_as_text`, `test_console_runtime_ownership.py::test_cold_native_start_commits_both_receipts[source-stop]`). `Tests/Architecture` serially: 1476 passed, 3 failed (`test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only`, two `test_timer_path_static_update_inventory` tests), all three fail the same on base. `./scripts/preflight.sh` passed.
- Files: `tldw_chatbook/Chat/console_configuration_preparation.py`, `Tests/Architecture/test_code_identity_ratchet.py`, `Tests/Architecture/test_chat_code_identity_ratchet.py`.
- **Review round (2026-11-18).** `==` alone was not enough: CPython's bound-method `==` compares `__func__` with `==`, so `MethodType(EqualBuilder(), owner)` (an always-equal callable) was accepted where base refused it. Both sites now go through `_same_bound_attribute(holder, name, original, slot)`: a stored attribute must be the same object; a method read must also have no per-instance override (`name not in vars(holder)`) and the class slot recorded with the original (`inspect.getattr_static(type(holder), name)`, taken at capture entry for the owner's two methods, before any producer runs), and only then does `==` decide. No `__func__`/`__code__` read came back, so the ratchet rows stay as lowered above. Site 2's seven callbacks are per-instance attributes, so for them this is identity, as before for the stock path.
- New tests in `Tests/Chat/test_console_configuration_preparation.py`: `test_mounted_capture_refuses_an_always_equal_bound_impostor` drives the real `capture_mounted_configuration_owned` and installs the impostor per instance (builder, selector, screen runtime accessor) or through the class (a descriptor that binds it on every read); all five refuse with `console_snapshot_owner_changed`, the stock case captures. Red on the previous head 378077650c (5 failed, "DID NOT RAISE"). `test_composition_mcp_capture_refuses_a_rebound_checker` shows a different module binding than the registered checker is refused. Mutations: dropping the per-instance check fails the three instance cases; dropping the class-slot check fails the two class cases; dropping the registered-checker binding check fails the rebound-checker test.
- **Accepted on purpose under ADR-356 rule 4.** `_composition_mcp_capture_current` no longer refuses an in-place `checker.__code__` swap, nor a rebinding of the registration tuple and the module attribute together (the rebound-checker test's positive control is that second case): the registration and current state decide, not code identity.
- Review-round evidence: the 58 covering files rerun on the new head at `-n 4` show no failure outside the earlier base-equal set except three nodes that pass alone (`test_console_default_send_capture.py::test_default_config_second_chat_reaches_provider_while_first_reply_runs[writable]`, `test_console_archive_recovery_flow.py::test_native_rows_number_immutable_page_and_label_inactive_match`, `test_console_send_acknowledgement_extended.py::test_failed_durable_commit_releases_the_row_and_restores_send`), and two teardown errors ("Native app workers remain uncertain; config retirement refused") in `test_console_send_acknowledgement*.py::test_enter_paints_a_sending_row_before_admission_*` that came once under load and not in two more alone runs on head (base also clean twice). `Tests/Architecture` serially: the same 3 base-equal failures.
<!-- SECTION:NOTES:END -->
