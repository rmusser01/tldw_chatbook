---
id: TASK-41708
title: "Three sensitive-logging tests let real DNS lookups out because they stub the wrong session factory and the network guard ignores getaddrinfo"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 19:04
labels:
- follow-up
- tests
- security
priority: high
references:
- Tests/Chat/test_sensitive_llm_logging.py
- Tests/network_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-41522 implementer and confirmed by its review (round 16). `chat_with_openai` has used `create_owned_default_session` since 2026-10-12, but `Tests/Chat/test_sensitive_llm_logging.py` (`test_sensitive_openai_http_error_log_and_exception_are_metadata_only`, `test_auxiliary_pins_configured_endpoint_when_selection_url_is_empty` and `test_auxiliary_pins_default_openai_endpoint_before_config_changes`) stubs only `create_default_session` and `create_pooled_default_session`. The adapter builds a real session; the socket guard blocks the connection, but `Tests/network_guard.py` does not guard `getaddrinfo`, so real DNS lookups of `api.openai.test`, `pinned.example.test` and `api.openai.com` ran during the covering run (no HTTP request was sent). The three tests are red on dev. TASK-32907 (two TTS tests that make real network calls) is the same guard gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The three tests stub the session factory the adapter actually uses and pass
- [x] #2 The network guard refuses a DNS lookup of a non-loopback name during tests, and a test shows it red before the change
- [x] #3 No test in the suite needs a real DNS lookup to pass, including TASK-32907's two TTS nodes, or each one that does is listed in the notes with why
<!-- AC:END -->

## Implementation Plan

1. `Tests/network_guard.py`: guard `socket.getaddrinfo`. A name that is not `None`, empty, `localhost` or a numeric IP literal is refused unless the test is `allow_network`/`live`. The refusal is recorded like a blocked connect, and its error is also a `socket.gaierror`, so callers take their resolver-failure path.
2. `Tests/test_network_guard.py`: pin the refusal with the C resolver (`_socket.getaddrinfo`) replaced by a recorder, so the red run before the guard change reaches no real resolver.
3. Under the new guard, show the three sensitive-logging tests red with the lookups blocked, then stub `create_owned_default_session` too and show them green.
4. Run the suite under the guard and fix, or list with a reason, every test the guard fails for a lookup (TASK-32907's TTS nodes included).

## Implementation Notes

- **Guard.** `Tests/network_guard.py` now patches `socket.getaddrinfo`. Unless the test is `allow_network`/`live`, a lookup of any name other than `None`, `""`, `localhost` or a numeric IP literal is refused before it reaches the resolver, recorded like a blocked connect (so the autouse fixture fails the test at teardown, naming `socket.getaddrinfo -> host:port`), and raised as `BlockedDnsLookup`, which is both `BlockedNetworkAccess` and `socket.gaierror`, so callers take their unresolvable-name path. asyncio's `loop.getaddrinfo`, anyio/httpx, urllib3 and aiohttp all go through `socket.getaddrinfo`; uvloop and aiodns, which would not, are not used. `gethostbyname*` and reverse lookups stay unguarded (no package caller).
- **AC #2 red/green.** `test_a_dns_lookup_of_a_remote_name_is_refused_before_the_resolver` swaps the C resolver (`_socket.getaddrinfo`) for a recorder, so no run of it reaches a real one: before the guard change it failed `DID NOT RAISE socket.gaierror` (the recorder took the lookup); after it, it passes and the recorder saw nothing. `test_local_and_numeric_lookups_and_opted_in_tests_reach_the_resolver` pins the pass-through cases. `Tests/test_network_guard.py`: 26 passed, 1 skipped.
- **AC #1.** With the guard in place and the tests unchanged, the three named tests failed and two more erred, all with the lookup blocked, not sent: `api.openai.test:443` (x4, including `test_openai_responses_api_input_field_is_never_logged[True/False]` and `test_tool_definitions_log_names_only_never_schema_or_description`, which passed on dev only because the real lookup failed), `pinned.example.test:443`, `api.openai.com:443`. A helper, `_stub_session_factories`, now points all three egress factories (`create_default_session`, `create_owned_default_session`, `create_pooled_default_session`) at the fake session at all 17 sites in the file; 108 passed under the guard.
- **AC #3, sweep under the guard.** Every run went through the wrapper at `-n 4` (load 20-65), with each lookup refused, never sent. Runs: the non-UI suite in order until stopped (4,880 tests); then every test file naming a non-loopback URL host, plus all of `Tests/TTS` (983 files, 43,667 tests with `-m "not allow_network and not live"`), in four runs. Found and fixed, each shown red under the guard first:
  - `Tests/Chat/test_console_agent_bridge.py::test_routed_adapter_real_gateway_keeps_saved_url_after_registry_edit` (2 nodes, `saved-server`): stubs `egress._resolve`/`_resolve_async` to raise `gaierror`, which is what CI's resolver answers.
  - `Tests/Chat/test_console_provider_gateway.py` (34 nodes in 11 tests, custom-endpoint hosts such as `owner-custom.example.test`, `api.example.com`): the context-window probe's egress check resolved them. An autouse fixture stubs the same two seams.
  - `Tests/UI/test_library_collection_reader_journeys.py::test_collections_actions_repaint_their_panes_without_a_library_recompose` (`example.test`, the quick capture's background extraction): same stub in the test.
  No other lookup was recorded.
- **TASK-32907's two TTS nodes** (`test_alltalk_uses_immutable_leased_default_for_actual_request[...]`) already use `httpx.MockTransport`: under the guard they recorded no lookup and passed 2 of 2 under a temporary `@private_profile_test` (removed); plain, they fail locally on the `RecoveryRequired` trap.
- **Tests that may still resolve names, and why:** only `allow_network`/`live` tests, which opt into unrestricted sockets: Agents/test_preset_fallback.py, Chat/test_goal_cli_verification.py, Chat/test_kimi_zai_native_tools.py, Chat/test_live_databricks_api.py, Chat/test_live_inference_cloud_api.py, Chat/test_live_moonshot_zai_api.py, Chat/test_live_qwencloud_api.py, Chat/test_qwencloud_native_tools.py, Chat/test_response_rules_live_local.py, Chat/test_sensitive_llm_logging.py (two nodes, numeric `127.0.0.1:1`), integration/test_first_run_pocket_tts_flow.py, Library/test_library_ingest_runner.py, LLM_Calls/test_hosted_chat.py, LLM_Calls/test_provider_session_reuse.py, LLM_Calls/test_qwencloud.py, Local_Ingestion/test_parakeet_v2_artifact.py, Model_Artifacts (8 files), QA/test_console_watchlists_workflow_uat.py, Subscriptions/test_feed_server.py, TTS (4 files), UI (9 files), Watchlists/test_watchlists_artifacts_pane.py, Wizards (2 files); plus Tools/test_remote_session_live.py and Web_Scraping/test_search_backends.py (`live`). The ones read use loopback servers or skip without an opt-in variable; their lookups are not observed, by design.
- **Not covered by the sweep:** test files that name no non-loopback URL outside the first 4,880 non-UI tests, and about 3,800 nodes that failed locally on the `RecoveryRequired: raw_source_selection_changed` trap before reaching their network code (mostly Tests/UI, Watchlists, Web_Scraping). Any of them that resolves a name now fails at teardown naming `socket.getaddrinfo`, so the first full CI run (`test.yml`) will list them.
- **Trap I hit:** the first two sweeps did not deselect `allow_network` tests, which the guard exempts; see the lesson added to `backlog/docs/lessons-testing-evidence.md`.
- Files: `Tests/network_guard.py`, `Tests/test_network_guard.py`, `Tests/Chat/test_sensitive_llm_logging.py`, `Tests/Chat/test_console_agent_bridge.py`, `Tests/Chat/test_console_provider_gateway.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `backlog/docs/lessons-testing-evidence.md`.
- Other failures seen in those files are not this task's: `test_console_provider_gateway.py`'s three are TASK-40413; the rest are the local `RecoveryRequired` trap.

