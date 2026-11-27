---
id: TASK-40413
title: >-
  Three provider-gateway unit tests are red on dev: DeepSeek is pinned a base
  URL nobody configured, and the OpenAI Stop test never sees content
status: Done
assignee:
  - '@claude'
created_date: '2026-11-12 13:45'
updated_date: '2026-11-26 15:43'
labels:
  - console
  - tests
  - ci
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three tests in `Tests/Chat/test_console_provider_gateway.py` fail on dev every time, so the file cannot tell a new provider-gateway regression from these old ones. Found while rebasing PR #3456 (TASK-37762), where they were the only reds in that file. They failed twice in isolation on dev `4b90eb860b` and failed the same way in a whole-file run (3 failed, 493 passed). Dev's own task records already show them (TASK-39149, TASK-39150 and TASK-39151 each say the three "fail at base"), but no task tracked them. The file is not named in the merge-blocking PR lanes (`derived-artifacts.yml`). It runs only in the full-tree runs: `test.yml` on push to `main` and `nightly-deep.yml`. That would explain why the PR check never went red.

- `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]` and `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base` (its DeepSeek case): with no `api_base_url` configured, the adapter kwargs carry `api_base_url='https://api.deepseek.com'`. Both already fail at `8ae979d81b` (2026-11-06). The kwargs builder pins `api_base_url` for every `{"qwencloud", "deepseek"}` resolution, and blame dates `deepseek` in that set to `f5a63125ff` (2026-10-13). That points to the TASK-2117 AC#3 expectation ("nothing is pinned by default") being stale for DeepSeek. Nobody has yet decided whether the test or the pin is the one that's wrong.
- `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`: the content-recorded event never fires within 5 s. It passes at `8ae979d81b`, and `git bisect` names `c1cad552f4` (TASK-38589, "decode legacy SSE streams as UTF-8 explicitly at four iter_lines sites") as the first bad commit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For DeepSeek with no configured base URL, the expected adapter kwargs are decided (pinned builtin base, or no pin), and the gateway and both tests agree with that decision
- [x] #2 The OpenAI Stop test sees streamed content recorded and the transport closed through the gateway boundary again, or the test is rewritten to match the adapter's current streaming read, with the reason recorded
- [x] #3 `Tests/Chat/test_console_provider_gateway.py` passes as a whole on dev, twice in a row
- [x] #4 The file runs in a CI lane that blocks merges to dev, or the task records why it does not
<!-- AC:END -->

## Implementation Plan

1. Reproduce the three reds on the base.
2. DeepSeek: read the registry record, the endpoint contract (`effective_provider_endpoint`, `meaningful_adapter_base_url`), the three gateway kwargs builders and the trace verifier's reconstruction; decide pin or no pin and make both tests agree.
3. OpenAI Stop: read c1cad552f4 and fix whichever side drifted.
4. Whole file twice at `-n 4`; remove the three deselects from the architecture-review step and `ARCH_REVIEW_DESELECTED`; run the step.

## Implementation Notes

The three nodes pass, and the file runs whole in its lane. Fixing the three nodes needed test changes only. The review round's `/image` pin (60c23f17cd) did change product code, `Chat/console_provider_endpoints.py`. Base 6e41919880 at `-n 4`: 3 failed, 499 passed.

- **DeepSeek base URL (AC#1). Decision: pin the builtin base.** With no configured URL, DeepSeek's adapter kwargs pin the endpoint Console resolved, which is the registry's `default_base_url` `https://api.deepseek.com`. The evidence:
  - The pin has been the product since f5a63125ff (2026-10-12, ADR-064's Chat-Completions amendment). That commit put DeepSeek in QwenCloud's branch of all three gateway kwargs builders so the explicit `api_mode` reaches the adapter, and the pin came with it.
  - The trace verifier's independent reconstruction (`console_trace_final_values.py`) pins it the same way, so Capture-On shadow verification already expects it.
  - The endpoint contract's `effective_provider_endpoint` is documented as "the effective endpoint to pin into the provider resolution". Pinning sends the endpoint Console's readiness and `endpoint_refusal` checked, which is HuggingFace's rule (TASK-40300 r14ep-rev5). Unpinned, the adapter would re-resolve from its own runtime config read.
  - With no configured URL, the pinned value equals the adapter's own fallback (`builtin_provider_endpoint`), so the wire is unchanged.
  - Leaving it unpinned would mean changing three builders and the verifier, for no change on the wire.

  The `/image` conversation-context call still treated DeepSeek as conditionally pinned, through `CONDITIONAL_BASE_URL_EXECUTION_KEYS`. That was the one place the set still applied to DeepSeek; the gateway and the verifier never reach that branch for it. DeepSeek is now removed from the set (`Chat/console_provider_endpoints.py`), so `/image` pins what a send pins. `test_the_image_context_call_pins_what_a_console_send_pins` (`Tests/Chat/test_local_server_discovery.py`, already laned) compares the image options with the gateway's own kwargs for DeepSeek and OpenAI, each with a builtin and a custom URL. Before the fix it failed only on DeepSeek's builtin case (`None == 'https://api.deepseek.com'`); after it, all 4 pass.

  TASK-2117's "nothing is pinned by default" is therefore stale for DeepSeek. `test_console_send_leaves_base_url_unpinned_in_default_case` now expects DeepSeek pinned, beside HuggingFace, and its docstring says why. The DeepSeek case of `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base` expects the pin on the primary, prepared and auxiliary paths. Every other assertion in both tests is unchanged.
- **OpenAI Stop (AC#2): the test was wrong.** c1cad552f4 (TASK-38589) made `chat_with_openai` iterate BYTES with `iter_lines()` and decode UTF-8 itself, because `decode_unicode=True` decodes with the header charset and mangles non-ASCII text. The fake's `iter_lines(self, *, decode_unicode)` required that keyword. The call raised `TypeError` before any content, so nothing was recorded and the 5 s wait expired. The fake now has requests' own signature, yields bytes and asserts `decode_unicode is False`, which pins TASK-38589. The capture, cancel, transport-closed, `stopped` status and `"he"` content assertions are kept.
- **AC#3:** the whole file at `-n 4` gave 502 passed twice (27.3 s and 47.1 s at load 29-40), plus the soak in TASK-42552's notes: 20 runs at `-n 4`, 20/20 green.
- **AC#4:** the file already ran whole in the architecture-review step (`pr-fast-lane-arch`, TASK-41136), with these three nodes deselected. The deselects are gone from `.github/workflows/derived-artifacts.yml`, and so are `ARCH_REVIEW_DESELECTED` and its use in `Tests/CI/test_ci_queue_pressure_contract.py`. The step ("Run the architecture-review tests", 47 targets, 0 deselected, serial, `--timeout=180`) gave 1566 passed, 2 skipped and 1 xfailed in 1761.6 s, at load 26-59 with other runs beside it. The three nodes take under 5 s together.

TASK-41053 tracks the same three nodes and is resolved by this work.

Files:
- `Tests/Chat/test_console_provider_gateway.py`;
- `Tests/Chat/test_local_server_discovery.py` and `tldw_chatbook/Chat/console_provider_endpoints.py` (review round);
- `.github/workflows/derived-artifacts.yml`;
- `Tests/CI/test_ci_queue_pressure_contract.py`.
