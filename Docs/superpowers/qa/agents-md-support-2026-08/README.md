# Console AGENTS.md support — verification evidence

Date: 2026-08-20

Design base: `5047b6962`

Architecture decision: [ADR-069](../../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md)

## Scope

This evidence covers the Console AGENTS.md implementation: secure root discovery,
local-only binding state, startup and lazy nested delivery, tool-batch deferral,
provider transport grammar, metadata-only UI/persistence surfaces, and the final
Console recovery/documentation UX.

## Deterministic performance

`Tests/Agents/test_project_instruction_performance.py` builds a 32-directory target
chain plus 32 two-level decoy branches.

- Startup inspected exactly one unique directory (the binding root).
- First nested activation inspected exactly the 32 root-to-target directories and
  no decoy.
- Tests assert operation counts, not machine-dependent elapsed time.

Reproduce the recorded timing evidence (the values are informational, not asserted):

```console
$ python -m pytest Tests/Agents/test_project_instruction_performance.py -q -s
startup_elapsed_ns=491750
nested_elapsed_ns=35353416
2 passed in 0.37s
```

## Focused automated verification

The final change-related gate covered the project-instruction resolver/runtime,
registry ownership, AgentService/AgentRuntime delivery, native/fenced provider
grammar, local persistence migration, Console state/UI, sentinel boundaries, the
performance test, and three legacy Console helper files affected by the new-session
default:

- 1,229 passed, 2 deselected, 3 warnings in 106.16 seconds in the final run.
- The two deselected nodes are localhost-only client-loop tests in
  `test_console_provider_gateway.py`. In the sandbox they error at fixture setup with
  `PermissionError: [Errno 1] Operation not permitted` while binding `127.0.0.1`;
  earlier delivery closeouts reproduced the identical nodes on the clean base.
- The first unfiltered focused run produced 1,229 passes and exactly those two setup
  errors, with no test failures.

Exact final command (using the populated repository virtual environment because the
worktree environment lacks the test dependencies):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Agents/test_agent_runtime_preparation.py \
  Tests/Agents/test_agent_runtime_review_hook.py \
  Tests/Agents/test_agent_service.py \
  Tests/Agents/test_agent_service_review_state_scope.py \
  Tests/Agents/test_project_instruction_concurrency.py \
  Tests/Agents/test_project_instruction_path_targets.py \
  Tests/Agents/test_project_instruction_resolver.py \
  Tests/Agents/test_project_instruction_resolver_properties.py \
  Tests/Agents/test_project_instruction_runtime.py \
  Tests/Agents/test_project_instruction_performance.py \
  Tests/Agents/test_run_log_eviction.py \
  Tests/Agents/test_tool_catalog.py \
  Tests/Agents/test_tool_catalog_owner_cache.py \
  Tests/Chat/test_anthropic_native_tools.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_agent_project_instructions.py \
  Tests/Chat/test_console_chat_store_project_instructions.py \
  Tests/Chat/test_console_project_instruction_persistence_boundary.py \
  Tests/Chat/test_console_project_instruction_provider_grammar.py \
  Tests/Chat/test_console_project_instructions.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/DB/test_chachanotes_console_project_context_migration.py \
  Tests/UI/test_console_context_modal.py \
  Tests/UI/test_console_project_instructions.py \
  Tests/UI/test_console_right_rail.py \
  --deselect Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_survives_agent_bridge_style_loop_swap \
  --deselect Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop \
  -q
```

Final output: `1229 passed, 2 deselected, 3 warnings in 106.16s`.

The new default project-instruction state exposed three pre-feature test helper
families which created real Console sessions without declaring their intended legacy
mode. The narrow fixes are test-only and leave production defaults unchanged:

- `test_console_agent_swap.py`: 44 passed.
- `test_console_chat_controller.py`: 165 passed.
- `test_console_local_citation_boundary.py`: 95 passed.

Task 13 UI verification separately recorded 79 passes and responsive Textual Pilot
evidence at 80x24, 100x30, and 140x40. The Impeccable detector ran exactly once and
returned `[]`.

## Static, security, and licence checks

- Ruff check: clean for the Task 14 performance and fixture changes.
- Ruff format: the new performance file is formatted. The three pre-existing legacy
  files are whole-file formatter baselines on both `HEAD` and the changed tree, so
  they were not broadly reformatted.
- mypy: clean for the resolver, runtime, and Console project-instruction controller.
- Bandit: clean for those same three security/runtime files.
- Licence metadata: `AGPL-3.0-or-later` assertion passed.
- `git diff --check 5047b6962...HEAD` and the working diff passed.
- `pip check` is not green in the shared development environment: installed
  `textual-web 0.8.0` requires Textual `<0.44` and uvloop `<0.20`, while this project
  uses Textual 8.2.8 and uvloop 0.22.1. This environment baseline is unrelated to the
  AGENTS.md change and is not reported as a pass.

## Privacy and persistence

Focused persistence-boundary tests use automatic-channel sentinels and verify that
the body reaches the provider request but not the Console database, AgentRunsDB,
run logs, transcript/event metadata, diagnostic callback failures, or exported
Context JSON. Explicit user reads and model quotations retain their normal durable
behavior. Delivery 1 and Delivery 2 scratch audits likewise found their sentinel only
in the provider spy.

The isolated Delivery 3 profile was prepared under
`/tmp/chatbook-agents-md-delivery3`; all config/data paths were set before import and
resolved inside that boundary.

## Live UAT status

The fenced/local-model half of live UAT passed on 2026-08-20 against the actual
`llama_cpp` listener at `127.0.0.1:9099`, using
`gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` and an isolated profile
rooted at `/private/tmp/chatbook-agents-md-uat`:

- consent accepted one root `AGENTS.md` source;
- the first `fs_read` call was atomically deferred before review/execution when
  `pkg/AGENTS.md` activated;
- the retry was reviewed once, executed once, and returned the explicit file body;
- the fenced tool-results block closed before the nested project-context block;
- three live provider calls completed with final text `UAT_LOCAL_SUCCESS`;
- the automatic root and nested sentinels appeared only in the provider-wire capture,
  not AgentRunsDB, the transcript, activation/review metadata, application logs, or
  the SQLite dump; and
- the explicit user-requested file body retained normal durable tool-result behavior.

The live run summary was:

```json
{
  "status": "done",
  "final_text": "UAT_LOCAL_SUCCESS",
  "provider_calls": 3,
  "nested_request_index": 1,
  "reviews": [["fs_read"]],
  "activation_sources": ["pkg/AGENTS.md"],
  "activation_scopes": ["pkg"]
}
```

The related user-visible Console checks also passed at 80x24, 100x30, and 140x40:
Context containment, warning/recovery metadata and focus, consent usability, and the
30-column rail status row (`11 passed`).

Credentialed native-cloud UAT also passed on 2026-08-21 against OpenAI
`gpt-4.1-mini`. The user-supplied folder credential was read only by the launch shell,
injected as `OPENAI_API_KEY`, and never written to Chatbook config or evidence. The
request used a synthetic 32x32 checkerboard PNG generated inside the scratch profile:

- the native streaming tool path completed three provider calls;
- the multimodal request retained the image through the actual provider boundary;
- root instructions reached the first request and `pkg/AGENTS.md` activated before
  the first `fs_read` could be reviewed or executed;
- the reconsidered native tool call was reviewed once, executed once, and returned
  the explicit file body;
- the model completed with exact text `UAT_CLOUD_SUCCESS`; and
- a second three-call run replaced the nested override with an invalid directory,
  delivered content-free outcome `invalid`, then recovered and completed the same
  reviewed tool call and final response without transmitting the nested body.

The successful native run summary was:

```json
{
  "status": "done",
  "final_text": "UAT_CLOUD_SUCCESS",
  "provider_calls": 3,
  "nested_request_index": 1,
  "reviews": [["fs_read"]],
  "activation_sources": ["pkg/AGENTS.md"],
  "activation_scopes": ["pkg"]
}
```

The warning/recovery run had the same status, final text, provider-call count, and
single reviewed `fs_read`, with `activation_outcomes: ["invalid"]` and no activated
nested source. The final combined sentinel audit found automatic root/nested bodies
only in the local/native provider request captures. They were absent from AgentRunsDB,
the Console transcript, activation/review metadata, application logs, and SQLite
dumps. The explicit file-read result remained durable by design. An exact-value scan
also found no OpenAI key in any evidence file or git diff.

## Scope decision

At the user's direction, closeout verification was limited to tests related to the
modified functionality and changed code. A partially completed broad repository run
was stopped and is not presented as evidence.
