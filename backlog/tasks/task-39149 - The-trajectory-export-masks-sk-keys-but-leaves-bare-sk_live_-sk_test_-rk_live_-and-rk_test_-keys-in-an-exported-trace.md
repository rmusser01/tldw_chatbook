---
id: TASK-39149
title: "The trajectory export masks sk- keys but leaves bare sk_live_, sk_test_, rk_live_ and rk_test_ keys in an exported trace"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:27
labels:
- follow-up
- security
- privacy
- trace
- export
priority: medium
references:
- tldw_chatbook/Chat/trajectory_export.py
- Tests/Chat/test_trajectory_export.py
- tldw_chatbook/Utils/log_sanitizer.py
dependencies:
- TASK-39123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (backup stream: TASK-39123's notes on round-12 branch fu/r12-backup list this as not covered). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev 3a6498ae14's (re-read 2026-11-09); the regex is the same on all three trees.

- `_CREDENTIAL_VALUE_RES` (`Chat/trajectory_export.py:300-317`) has one key-prefix pattern, `\bsk-[A-Za-z0-9_-]{8,}` (`:307`). `_credential_text` (`:339-343`) applies these patterns to exported strings, and `_has_credential` decides with it.
- A probe on dev and on the stack used synthetic values and no network:
  - `_credential_text("sk-abcdefgh12345678")` returns `[credential redacted]`;
  - the same call returns `sk_live_abcdefgh12345678`, `sk_test_…`, `rk_live_…` and `rk_test_…` unchanged.
- This is separate from Console trace storage. TASK-39123 (round 12) moved stored traces to `credentials-v2`, whose patterns live in `Utils/log_sanitizer.py` (`trace_credentials_v2_spans`, round-12 branch). The export does not read trace storage and keeps its own regex.
- No PR lane runs `Tests/Chat/test_trajectory_export.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exported trajectory masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys wherever it masks `sk-` keys; a test is red on dev, where the value is returned unchanged
- [x] #2 `sk-` keys, private keys, bearer and basic tokens, AWS key ids and URL user info are still masked as before
- [x] #3 The new tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->

## Implementation Plan

1. Add one key-prefix pattern for bare `sk_`/`rk_` live and test keys to `_CREDENTIAL_VALUE_RES` in `Chat/trajectory_export.py`, with the same 8-character floor as the `sk-` pattern and a body without `_` (so identifiers such as `sk_live_mode_enabled` stay readable).
2. Tests in `Tests/Chat/test_trace_v2_collaboration.py` (the pure Trace v2 bundle tests, which already cover the other credential classes in every profile): the four bare key shapes in a preview, a payload value and a nested key, for every profile (red on dev); and a pin that `sk-`, PEM, Bearer, Basic, AKIA and URL user info are still masked (shown red under a temporary source edit).
3. Lane the new nodes on the dreams-boundary step (not the fast PR contract) and record three timings of that step.

## Implementation Notes

The Trace export masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys wherever it masks `sk-` keys.
- **Fix.** One pattern joins `_CREDENTIAL_VALUE_RES` (`Chat/trajectory_export.py`): `\b[rs]k_(?:live|test)_[A-Za-z0-9]{8,}`, case-insensitive like its neighbours. Its floor is the `sk-` pattern's 8 characters rather than `log_sanitizer`'s 24, so the task's probe value `sk_live_abcdefgh12345678` is masked; the body takes no `_`, so an identifier such as `sk_live_mode_enabled` stays readable. Every masking site reads `_credential_text`: string values and key names, identifier aliasing, structural metadata, and the final `_has_credential_material` gate, which `trajectory_import` also applies. So importing an older bundle that still holds such a key is now refused, as one holding an `sk-` key already was.
- **Where the tests live.** The ADR-313 file export (`build_trajectory_export`, tested in `Tests/Chat/test_trajectory_export.py`) applies no credential patterns; it keeps previews only. The patterns belong to the Trace v2 bundle (`build_trace_export`), whose pure tests in `Tests/Chat/test_trace_v2_collaboration.py` already pin the other credential classes for every profile, so the new tests are there.
- **AC#1.** `test_bare_underscore_keys_are_masked_wherever_sk_dash_keys_are` (4 keys × 3 profiles): the key in a preview, a payload value and a payload key; the serialized bundle never holds it, a credential provenance row exists, and `_credential_text` treats it as it treats `sk-abcdefgh12345678`. Red at base ae9f9e8d21 (run.sh, `-n 4`): 12 failed, e.g. `assert b'rk_live_abcdefgh12345678' not in b'{"format": "tldw-trace", "version": 2, ...'`. Green after.
- **AC#2.** `test_the_credential_classes_masked_before_are_still_masked` (`sk-`, PEM, Bearer, Basic, AKIA, URL user info × 3 profiles): green at base and after. Under a temporary edit keeping only the `sk-` pattern the other 15 cases failed, and keeping only the new pattern the 3 `sk-` cases failed; reverted.
- **AC#3.** Both nodes run in the dreams-boundary job's "Run Dreams and Guardian boundary contracts" step (never the fast PR contract), among its sandboxed targets because they read no config, and in `DREAMS_BOUNDARY_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`). The same step runs TASK-39150's and TASK-39151's new tests (`Tests/Chat/test_local_server_discovery.py`, whole). `jobstep.sh` (the step's 35 targets, its order and flags, serial) three times: 1126 passed each, in 1742.6 s, 1055.7 s and 967.1 s, at 1-minute load 84, 65 and 29 when each started (18 CPUs, other agents running). In CI the job took 10.9-14.1 min of its 20 over ten PR runs on 2026-11-09/10 (the step 7.7-9.8 min); these nodes add seconds.
- **Runs** (run.sh, `-n 4` unless noted, 1-minute load 30-85 on 18 CPUs with other agents running; shared by TASK-39149, TASK-39150 and TASK-39151 on branch fu/r13-sec, base ae9f9e8d21, 2026-11-10):
  - Final core set (discovery, project instructions, Trace v2, trajectory export and import, the endpoint contract, custom endpoint registry, exchange capture, trace final values, the provider gateway): 1380 passed, 3 failed. The 3 (`test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`, `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]`, `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`) fail at base with the same command.
  - Wider covering run: the 96 non-UI files that name a changed helper (`normalize_llamacpp_base_url`, the readiness and summary builders, the custom endpoint registry, project instructions, trajectory export, `resolve_llamacpp`) plus `Tests/LLM_Provider_Catalog/`: 5054 passed, 167 failed, 29 errors; 14 UI and performance files (Chat settings, custom endpoints, discovery, switcher, project instructions, trace export and import, keystroke census): 814 passed, 152 failed. Every failing node was rerun with the four changed source files at base: all fail there too, except the 127.x gateway pin rewritten on purpose (TASK-39151), 10 goal and agent-swap nodes and 1 Chat-settings node, which passed in that one base batch. Rerun on their own, alternating head and base twice, the same 4 of the 10 failed on both trees and the other 6 passed on both (base once failed a fifth, a capacity-wait test), and the Chat-settings node failed on both with `RecoveryRequired: raw_source_selection_changed`. Most reds in both runs are that local `RecoveryRequired` trap.
  - `Tests/LLM_Calls` llama files: 175 passed. CI workflow contracts (`test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py`, `test_console_p0_regression_gate.py`): 365 passed. Serial: `Tests/Architecture/test_module_size_ratchet.py` 201 passed (gateway 8,488 lines under its 8,493 pin); the screen-size ratchet, chat code identity ratchet, Agents/Chat boundary and agent runtime import footprint, 33 passed. `./scripts/preflight.sh`: all checks passed.

Files: `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
