---
id: TASK-40413
title: >-
  Three provider-gateway unit tests are red on dev: DeepSeek is pinned a base
  URL nobody configured, and the OpenAI Stop test never sees content
status: To Do
assignee: []
created_date: '2026-11-12 13:45'
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
- [ ] #1 For DeepSeek with no configured base URL, the expected adapter kwargs are decided (pinned builtin base, or no pin), and the gateway and both tests agree with that decision
- [ ] #2 The OpenAI Stop test sees streamed content recorded and the transport closed through the gateway boundary again, or the test is rewritten to match the adapter's current streaming read, with the reason recorded
- [ ] #3 `Tests/Chat/test_console_provider_gateway.py` passes as a whole on dev, twice in a row
- [ ] #4 The file runs in a CI lane that blocks merges to dev, or the task records why it does not
<!-- AC:END -->
