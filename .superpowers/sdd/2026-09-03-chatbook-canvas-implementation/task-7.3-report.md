# Task 7.3 implementation report

## Result

- Added a focused Canvas user workflow covering create/update, multiple named
  Canvases, branch and exact-revision navigation, Temporary/save behavior,
  immutable undo, source export, confirmed submit/download, compatibility
  refusal, scripts-disabled recovery, and deferred V2/V3/synchronization work.
- Corrected served-mode operations guidance to the implemented dedicated-token,
  TLS/trusted-proxy, session expiry/restart, kill-switch, and incident-response
  boundaries. It now states that served Chatbook has host-process authority and
  is neither an OS sandbox nor a multi-user isolation system; browser-bound
  Canvas URLs cannot borrow another browser session's capability.
- Added schema-sensitive model guidance beside the Canvas schemas. It names only
  disclosed tools, includes the V1 authoring/runtime contract when either
  mutation tool is disclosed, and appears in preview, live, progressive-load,
  first-request, and personal-context budget paths. Progressive discovery gets
  one Canvas cue only when all four tools are actually allowed; disabled and
  no-tool paths remain unchanged.
- Aligned the design spec with inert `.html.txt` revision exports and locked,
  complete same-identity graph/metadata/source revalidation; digest equality
  alone is not idempotence. Linked the final workflow, compatibility, and
  operations documents from ADR-115.

## RED/GREEN evidence

- RED focused guidance collection (before implementation):
  `../../.venv/bin/python -m pytest -q` with the new provider-guidance,
  request-guidance, progressive-load, discovery-gate, and profile-budget node
  IDs exited non-zero because the shared Canvas guidance builder/constants and
  injection seams did not exist.
- RED budget boundary: with the new first-request budgeting append temporarily
  removed,
  `../../.venv/bin/python -m pytest -q Tests/Agents/test_agent_service.py::test_first_request_plan_counts_canvas_guidance_before_direct_disclosure`
  failed because direct disclosure remained selected instead of falling back to
  progressive discovery.
- GREEN focused contract:
  `../../.venv/bin/python -m pytest -q Tests/Agents/test_canvas_tool_provider.py::test_runtime_guidance_names_only_disclosed_tools_and_keeps_mutation_apis Tests/Agents/test_canvas_tool_provider.py::test_catalog_descriptions_explain_canvas_selection_and_replacement_contract Tests/Agents/test_agent_service.py::test_model_request_guidance_tracks_the_exact_disclosed_canvas_schema_set Tests/Agents/test_agent_service.py::test_first_request_plan_counts_canvas_guidance_before_direct_disclosure Tests/Agents/test_agent_service.py::test_load_tools_adds_canvas_guidance_on_the_next_budgeted_request Tests/Chat/test_console_agent_bridge.py::test_canvas_discovery_hint_requires_the_actual_complete_run_allow_list Tests/Chat/test_console_personal_context_snapshot.py::test_first_request_profile_budget_reserves_canvas_runtime_guidance Tests/Agents/test_agent_service.py::test_first_request_plan_drops_discovery_tools_when_only_no_tool_request_fits Tests/Chat/test_console_agent_bridge.py::test_compose_appends_discovery_hint_only_when_find_load_offered`
  passed 10 tests.
- Targeted regression runs:
  - `../../.venv/bin/python -m pytest -q Tests/Agents/test_canvas_tool_provider.py`
    — 99 passed.
  - `../../.venv/bin/python -m pytest -q Tests/Agents/test_agent_service.py -k 'canvas or first_request_plan or load_tools'`
    — 21 passed, 100 deselected.
  - `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_agent_bridge.py -k 'canvas_discovery or compose_appends_discovery'`
    — 2 passed, 274 deselected.
  - `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_personal_context_snapshot.py`
    — 11 passed.
- Every pytest invocation emitted the existing `RequestsDependencyWarning` for
  the installed urllib3/chardet/charset-normalizer versions; no test failed.
- `ruff format --check --range=<changed-range>` passed for all 23 changed
  logical Python ranges. Full-rule `ruff check` over the seven changed Python
  files reports 103 inherited whole-file findings; checking the same seven
  files from exact base `205b2f8f2c` through `git show ... | ruff check
  --stdin-filename ... -` also reports 103. No lint finding was added.
  `ruff check --select E9,F63,F7,F82` over the changed files passed;
  `compileall -q` over those files passed; `git diff --check` passed.
- A local Markdown-link check resolved 68 links across the seven changed
  documentation entry points with zero missing targets. A stale-claim search
  found no remaining firewall-only, restricted-subprocess, old `.html` export,
  digest-only idempotence, or pre-shipment Canvas wording.

No provider calls, browser launches, full-suite run, new runtime capability, or
server synchronization work was performed.

## Self-review

- Compared model copy with the actual four schemas and compatibility facade;
  partial create-only and update-only disclosure cannot advertise the other
  tool names, but both receive the supported V1 authoring APIs.
- Compared operational copy with implemented config names, credential
  precedence, login/bootstrap routes, expiry/restart behavior, and Canvas kill
  switch. No logout/revoke UI, server sandbox, account isolation, or transcript
  portability claim was added.
- The only qualification is the pre-existing dependency-version warning above;
  there is no Task 7.3 implementation concern.
