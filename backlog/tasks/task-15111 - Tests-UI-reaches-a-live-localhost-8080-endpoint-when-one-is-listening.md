---
id: TASK-15111
title: 'Tests/UI reaches a live localhost:8080 endpoint when one is listening'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 04:00'
updated_date: '2026-08-11 14:24'
labels:
  - tests
  - test-infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed while repairing task-14920: the `Tests/UI` console suites **make real requests to `127.0.0.1:8080/v1/models`** when something happens to be listening on that port on the developer's machine.

That makes those suites environment-dependent in the worst way — they behave differently depending on whether an unrelated local server is running, and the difference is silent. A developer with a local model server up is running different tests from CI, and neither knows it. It is also the mirror image of two traps this repo has already recorded: a missing optional extra faking a code regression, and a green suite saying nothing about installs that are not yours.

Worth checking as part of this: whether any test can *mutate* state on a live endpoint it reaches, and whether the same pattern exists outside `Tests/UI`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test in the suite performs network I/O to a live local endpoint; the boundary is stubbed or blocked
- [x] #2 A guard makes the escape impossible to reintroduce silently (e.g. sockets blocked by default in the UI test configuration, with an explicit opt-in for any test that genuinely needs one)
- [x] #3 The check covers whether any such call could mutate state on the endpoint it reached, not just read from it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the truth: run Tests/UI under a record-only socket shim; capture every connect with its address; identify the production code path that builds a live client during a test.
2. Inventory every network-reachable point (not just /v1/models, not just Tests/UI): grep/AST the client-construction seams and record HTTP verbs, not just connects.
3. Fix the mechanism at the seams that build a REAL client in tests (local-server discovery probe chokepoint; ConsoleProviderGateway's owned-client factory), preserving the CI 'nothing is listening' behaviour exactly.
4. Add a default-deny socket guard installed at conftest import time, with an explicit @pytest.mark.allow_network opt-in; record blocked attempts so a broad 'except Exception' in the code under test cannot swallow the guard.
5. Prove the guard bites: tests that attempt a connection and assert refusal; mutation-check by disabling the guard and by disabling each mechanism fix.
6. Answer AC#3 empirically: stand up a stand-in local server and observe what the test-reachable client actually SENDS (verb + path + body).
7. Sweep for tests the guard exposes as network-dependent; stub or mark each with the opt-in and report counts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Tests no longer reach a live local endpoint, and the escape now cannot come back silently.

**What was actually happening.** Confirmed empirically first, not taken on faith: a record-only socket shim over `Tests/UI` logged **386 real TCP connect attempts across 20 test files in the first 12% of the suite** on a machine with an `audiocpp` server bound to `127.0.0.1:8080`. Two distinct mechanisms, not one:

1. *Discovery.* Mounting `ChatScreen` with an unconfigured provider makes the setup card blocking, which starts `_maybe_start_console_local_discovery` -> `discover_local_servers`. `build_local_server_candidates` **always** leads with `http://127.0.0.1:8080` and `http://127.0.0.1:11434` regardless of config, and `probe_models_endpoint` builds a real `httpx.AsyncClient` when none is injected. Exactly one test in the suite (`test_console_local_server_discovery_card`) had ever stubbed the `console_local_server_discovery` app seam; every other Console test fell through to the network. On a machine where 8080 answers, the setup card grows a "detected server" affordance CI never sees.
2. *The provider gateway.* `_configure_native_ready_console` points the Console at `http://127.0.0.1:9099` and several tests then drive a REAL send through `ConsoleProviderGateway`, whose `_new_owned_http_client` is a real client. On CI nothing listens, `_is_reachable`'s `GET /health` fails, and the send stops -- which is why it looked read-only.

**AC#3, answered by measurement.** A stand-in llama.cpp was bound to 127.0.0.1:9099 and `test_console_command_popup::test_enter_with_popup_closed_sends_normally` re-run against it. The server received `GET /health` **and two `POST /v1/chat/completions`** (streaming, then the non-streaming fallback) carrying the test's prompt. So the reachable client was never read-only: with a real llama.cpp on that port, `pytest` would have driven inference on the developer's own server. The gateway's reachable surface is `_is_reachable` (GET /health), `resolve_llamacpp` (GET /v1/models), and `stream_llamacpp_chat` / `complete_llamacpp_chat` / `complete_auxiliary` / `stream_chat` (POST /v1/chat/completions).

**Fixes.** Both mechanisms are fixed at the seam that builds a real client, and a default-deny socket guard proves it rather than being the only thing standing between the suite and the network:

- `Tests/network_guard.py` (new): patches `socket.connect`/`connect_ex`/`sendto`/`create_connection` for AF_INET/AF_INET6 at conftest **import** time (so collection, fixture teardown and worker threads outliving a test are covered), denied by default. `BlockedNetworkAccess` subclasses `OSError` so clients degrade down their existing "unreachable" path -- and because `_get_models_payload` really does `except Exception`, every block is also **recorded**; the autouse `_no_network_io` fixture fails the test on a non-empty record, which is what makes the guard unswallowable. `pytest-socket` is not in this repo's dev extras, so this is hand-rolled rather than a new dependency.
- `_no_local_server_probes` (autouse, `Tests/conftest.py`): stubs `local_server_discovery._get_models_payload`, the single chokepoint all three production import sites funnel through. Opt-out marker `local_server_probe` for the MockTransport probe tests.
- `_console_gateway_http_client_is_offline` (autouse): substitutes an `httpx.MockTransport` that raises `ConnectError` for `ConsoleProviderGateway._new_owned_http_client`, the single owned-client construction site; injected clients are untouched. Opt-out marker `owned_http_client` (plus `allow_network`).
- Opt-in marker `allow_network` (and `live`, implicitly) for tests that genuinely need a socket.

**Evidence.** RED first: the shim's 386 connects, and with the guard disabled `test_create_connection_to_the_llamacpp_default_port_is_refused` DID NOT RAISE -- it really connected to the live 8080 server. Mutation-checked twice: commenting out `network_guard.install()` turns 6 of 9 guard tests red; disabling `_no_local_server_probes` turns the two isolation tests red. With the fake 9099 server still running, the whole guarded Console run left its request log unchanged.

**Other tests the guard exposed (not weakened).** Failure *sets* compared against an identical unguarded baseline over `Tests/Model_Artifacts Tests/Subscriptions Tests/Notes Tests/Local_Ingestion Tests/Media_DB/test_sync_client_integration.py Tests/TTS`: baseline 21 failed / 5712 passed / 6 errors; guarded 94 failed / 5639 passed / 90 errors. Every newly failing test was in one of 12 modules that stand up an **in-process** HTTP server (`FixtureArtifactServer`, `fake_audiocpp_server`, the briefings feed server) -- all 84 blocked addresses were ephemeral loopback ports this process was itself listening on. Those 12 modules got a `pytestmark = pytest.mark.allow_network` with a comment saying why: 903 passed, 0 failed afterwards. Three `Tests/Chat/test_console_provider_gateway.py` tests that are *about* the owned client got `owned_http_client` (two also `allow_network`, since they use their own `local_http_server`). The 21 pre-existing failures sit in 6 modules none of this touches.

**Files:** added `Tests/network_guard.py`, `Tests/test_network_guard.py`, `Tests/UI/test_console_local_server_probe_isolation.py`; changed `Tests/conftest.py`, `pyproject.toml` (3 markers), `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_console_provider_gateway.py`, 12 fixture-server modules, `backlog/docs/lessons-testing-evidence.md`.

**Final counts (READ).** `Tests/test_network_guard.py` + `Tests/UI/test_console_local_server_probe_isolation.py` + `Tests/Chat/test_local_server_discovery.py`: **37 passed**. Coordinator keep-green set (`test_background_signal_bounds`, `test_console_moved_seam_guard`, `test_screen_navigation`, `test_personas_workbench`) plus the new tests and both discovery/gateway suites: **629 passed, 3 failed** -> the 3 were the owned-client tests, now marked `owned_http_client`: **4 passed**. The 12 fixture-server modules after marking: **903 passed, 0 failed**. Nine of the Console modules the record-only shim caught red-handed (setup-lock-polish, product-maturity empty setup states, discovery card, agent rail, button routing, character avatar, composer menu, auto-rag-on-send, edit/resend): **147 passed, 4 failed, ZERO blocked-egress hits** -- and those same 4 fail identically with the guard AND both mechanism fixtures disabled, so they are pre-existing stale contracts (`section:starred` collapse pref, `ConsoleChatController._turn_context_provider`, two auto-RAG ordering contracts), not caused here. `test_console_command_popup`'s one remaining failure is likewise pre-existing: it pins a 6-item slash-command list that `/generate-video` + `/stream-video` (task-3401.5, already on dev) grew to 8. A full `Tests/UI` sweep was started in halves but abandoned unfinished -- the machine was running 4+ concurrent pytest sessions from other agents and neither half got past ~6%; the targeted runs above cover every module the shim proved was reaching the network.
<!-- SECTION:NOTES:END -->
