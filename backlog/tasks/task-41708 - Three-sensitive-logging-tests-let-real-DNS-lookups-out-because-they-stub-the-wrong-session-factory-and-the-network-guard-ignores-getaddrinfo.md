---
id: TASK-41708
title: "Three sensitive-logging tests let real DNS lookups out because they stub the wrong session factory and the network guard ignores getaddrinfo"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-20 09:50
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
- **AC #3, the guarded sweep (first pass).** Through the wrapper at `-n 4`, every lookup refused, never sent: the non-UI suite in order until stopped (4,880 tests), then every test file naming a non-loopback URL host plus all of `Tests/TTS` (983 files, 43,667 tests). Found and fixed: `test_console_agent_bridge.py::test_routed_adapter_real_gateway_keeps_saved_url_after_registry_edit` (2 nodes, `saved-server`), `test_console_provider_gateway.py` (34 nodes in 11 tests; the context-window probe's egress check resolved custom-endpoint hosts such as `owner-custom.example.test`), and `test_library_collection_reader_journeys.py::test_collections_actions_repaint_their_panes_without_a_library_recompose` (`example.test`, the quick capture's background fetch).
- **Review r17lg-rev1 (Critical): the first pass missed tests.** Fixed, each shown red under the guard first:
  - `Tests/Web_Server/test_artifact_share_controller.py`: `test_compute_display_urls_loopback_and_lan` (`IndexError` at `lan[1]` plus a teardown error) and `test_start_share_non_loopback_with_credentials_starts` (teardown error), both recording `socket.getaddrinfo -> <this machine's name>.local:None` and `socket.connect -> 10.255.255.255:1` from `_primary_route_ip`. A `fixed_lan_address` fixture now answers `gethostname`/`getaddrinfo` with `127.0.1.1` and `192.168.7.23`, so the first test pins the exact display URLs; 11 of 11 pass. `_primary_route_ip`'s docstring no longer says the lookup sends nothing or that the guard skips it.
  - `Tests/UI/test_library_collections_capture_reader.py`: `test_real_library_route_quick_capture_persists_and_selects_capture` and `test_quick_capture_hostile_fields_are_rejected_and_valid_still_saves`. Under a temporary `@private_profile_test` (since removed) both failed with the child recording `socket.getaddrinfo -> example.test:None` 4 times; with the fixture, 2 of 2 passed.
  - One shared fixture, `unresolved_egress_dns` in `Tests/conftest.py`, now replaces the three hand-written stubs: it patches `egress._resolve`/`_resolve_async`, raises `socket.gaierror` for a non-local name (what an unregistered test name gets on CI) and passes `localhost` and numeric IPs to the real resolver (review minor 2).
- **AC #3, the full sweep (second pass).** All 5,816 test files ran with `-m "not allow_network and not live"`, under the guard, through the wrapper at `-n 4` (`Tests/Architecture` serial), checking `diskutil` free space before each batch (223 GB at the start, 30 GB at the lowest; the drop was not this run's, whose basetemp stayed under 100 MB):

  | Batch | Files | Result | Refused lookups |
  |---|---|---|---|
  | Tests/Chat (no host URL) | 583 | 599 failed, 12,844 passed, 136 skipped, 11 errors | 0 |
  | Tests/Backup_Recovery | 349 | 282 failed, 5,782 passed, 54 skipped, 15 errors | 0 |
  | Agents, DB, Library, Utils | 509 | 374 failed, 11,450 passed, 26 skipped, 36 errors | 0 |
  | Tests/Architecture (serial) | 80 | 3 failed, 1,463 passed, 5 skipped | 0 |
  | Other non-UI | 1,738 | 1,901 failed, 32,027 passed, 256 skipped, 278 errors | 2 nodes |
  | Tests/UI, three batches | 1,551 | 5,143 failed, 22,707 passed, 269 errors | 0 |
  | Host-naming files, non-UI (rerun on final code) | 848 | 1,977 failed, 33,670 passed, 81 skipped, 181 errors | 0 |
  | Host-naming files, UI (rerun) | 158 | 1,822 failed, 5,950 passed | 0 |

  The 2 nodes were `Tests/Model_Artifacts/test_provision_serialization.py::test_provision_releases_session_lease_when_cancelled_during_acquire` and `::test_provision_not_yet_installed_reaches_fetch_phase` (`example.test`); both now use the fixture (7 of 7 pass).
- **Nodes the local trap hid, rerun.** 9,898 nodes in 859 files failed locally on `RecoveryRequired: raw_source_selection_changed` before their network code. I reran every one of them under the guard on the bootstrap profile, through a temporary pytest plugin that marked each item `bootstrap_profile` (removed, never committed): non-UI 3,445 nodes (377 failed, 2,949 passed, 6 skipped, 130 errors; 15 nodes refused), UI 3,300 (400 failed, 2,853 passed, 50 errors; 0) and UI 3,153 (171 failed, 2,956 passed, 27 errors; 1 node). Fixed:
  - `Tests/Chat/test_chat_functions.py`, 12 `TestProviderRequestPayloads` nodes (`api.openai.test`, `canonical.test`): the same stale-stub cause as AC #1. An autouse fixture sends `create_owned_default_session` to the test's stubbed `create_default_session`; the 12 now also pass plainly (they were red at base, locally).
  - `Tests/Subscriptions/test_watchlist_check_now_source_id.py`, 2 nodes (`example.invalid`), and `Tests/Web_Scraping/test_deep_search_pipeline.py::test_relevance_scrape_failure_keeps_result_with_fallback` (`kept.example`): the egress check; they use the fixture.
  - `Tests/UI/test_watchlists_check_now_failure.py::test_an_unexpected_exception_in_the_fetch_path_logs_above_debug` (`summitroute.com`, a real domain): the file now uses the fixture through `pytestmark`; 27 of 27 pass on the bootstrap profile.
- **Still unswept, and why:** 1,155 nodes in 260 files also failed on the bootstrap profile, from other local causes (missing optional modules, local config, TASK-40413), so whether they reach a resolver after that point is unobserved. Static check of those files: none calls `gethostname`, `getaddrinfo` or `gethostbyname` except `test_deep_search_pipeline.py`, whose calls are its own resolver stubs; 23 name a host outside the reserved test domains. I ran the six of those most likely to fetch (`UI/test_watchlists_inspector.py`, `UI/test_watchlists_content_pane.py`, `UI/test_console_native_chat_flow.py`, `UI/test_first_run_wizard_live_contract.py`, `integration/test_console_thinking_end_to_end.py`, `App/test_submit_library_ingest_job.py`) again on the bootstrap profile under the guard: 743 tests, 0 refusals. Any lookup still hidden in those 1,155 nodes fails at teardown naming `socket.getaddrinfo` in CI, which has no trap.
- **TASK-32907's two TTS nodes** (`test_alltalk_uses_immutable_leased_default_for_actual_request[...]`) already use `httpx.MockTransport`: under the guard they recorded no lookup and passed 2 of 2 under a temporary `@private_profile_test` (removed); plain, they fail locally on the `RecoveryRequired` trap.
- **Tests that may still resolve names, and why:** only `allow_network`/`live` tests, which opt into unrestricted sockets: Agents/test_preset_fallback.py, Chat/test_goal_cli_verification.py, Chat/test_kimi_zai_native_tools.py, Chat/test_live_databricks_api.py, Chat/test_live_inference_cloud_api.py, Chat/test_live_moonshot_zai_api.py, Chat/test_live_qwencloud_api.py, Chat/test_qwencloud_native_tools.py, Chat/test_response_rules_live_local.py, Chat/test_sensitive_llm_logging.py (two nodes, numeric `127.0.0.1:1`), integration/test_first_run_pocket_tts_flow.py, Library/test_library_ingest_runner.py, LLM_Calls/test_hosted_chat.py, LLM_Calls/test_provider_session_reuse.py, LLM_Calls/test_qwencloud.py, Local_Ingestion/test_parakeet_v2_artifact.py, Model_Artifacts (8 files), QA/test_console_watchlists_workflow_uat.py, Subscriptions/test_feed_server.py, TTS (4 files), UI (9 files), Watchlists/test_watchlists_artifacts_pane.py, Wizards (2 files); plus Tools/test_remote_session_live.py and Web_Scraping/test_search_backends.py (`live`). The ones read use loopback servers or skip without an opt-in variable; their lookups are not observed, by design.
- **Trap I hit:** the first two sweeps did not deselect `allow_network` tests, which the guard exempts; see the lesson added to `backlog/docs/lessons-testing-evidence.md`.
- **Review minors.** (1) A lookup recorded on an `asyncio_N` thread is now named as the event loop's executor, not blamed on an earlier test; `test_an_executor_thread_lookup_is_not_blamed_on_an_earlier_test` was red before (`"event loop's executor" in ...`). (4) `test_loopback_only_mode_refuses_a_remote_name_lookup` was red with loopback mode's refusal removed (temporary edit; `DID NOT RAISE BlockedDnsLookup`). Minors 2 and 3 are above and in TASK-41714. `Tests/test_network_guard.py`: 28 passed, 1 skipped.
- Files: `Tests/network_guard.py`, `Tests/test_network_guard.py`, `Tests/conftest.py`, `Tests/Chat/test_sensitive_llm_logging.py`, `Tests/Chat/test_console_agent_bridge.py`, `Tests/Chat/test_console_provider_gateway.py`, `Tests/Chat/test_chat_functions.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `Tests/UI/test_library_collections_capture_reader.py`, `Tests/UI/test_watchlists_check_now_failure.py`, `Tests/Web_Server/test_artifact_share_controller.py`, `Tests/Model_Artifacts/test_provision_serialization.py`, `Tests/Subscriptions/test_watchlist_check_now_source_id.py`, `Tests/Web_Scraping/test_deep_search_pipeline.py`, `tldw_chatbook/Web_Server/artifact_share.py` (docstring), `backlog/docs/lessons-testing-evidence.md`.
- Other failures seen in those files are not this task's: `test_console_provider_gateway.py`'s three are TASK-40413; the rest are the local `RecoveryRequired` trap.

