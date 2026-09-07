# First-Chat UAT Remediation Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver and verify all 17 findings from the fresh-install llama.cpp and custom OpenAI-compatible UAT without introducing a provider migration or destabilizing active Console sessions.

**Architecture:** Execute four ordered, independently reviewable slices: shared provider contracts, first-run handoff, Settings/Console reliability, and launch diagnostics. Each slice has its own TDD plan and commit gates; this master plan owns cross-plan sequencing, conflict management, the end-to-end mock harness, visual evidence, and the final acceptance ledger.

**Tech Stack:** Python 3.11+, Textual/textual-serve, Rich, httpx, pytest, Playwright/browser capture, TOML configuration, setuptools wheel builds.

## Global Constraints

- Baseline is `origin/dev` at `5414d811b8720c1c32c5813f96925a82c60c5f72`.
- Preserve current configuration and session ownership; no named connection registry or automatic migration.
- Complete each child-plan gate before beginning the next slice.
- No UAT harness request may leave loopback; browser automation aborts non-loopback HTTP(S) requests.
- Screenshots supplement text/state assertions and never replace them.
- Use isolated `HOME`, config, data, ports, and mock processes for every live replay.
- Keep the separate first-run reliability and TTS plans unmodified; coordinate shared files through the integration order below.
- Use `Tests/...` as the canonical test path spelling.

---

## Child Plans

1. `Docs/superpowers/plans/2026-08-12-provider-connection-foundation-implementation-plan.md`
2. `Docs/superpowers/plans/2026-08-12-first-run-provider-handoff-implementation-plan.md`
3. `Docs/superpowers/plans/2026-08-12-settings-console-reliability-implementation-plan.md`
4. `Docs/superpowers/plans/2026-08-12-launch-diagnostics-trust-implementation-plan.md`

## Shared-File Integration Order

| Shared file | Required order |
|---|---|
| `FirstRunSetupWizard.py`, `first_run_setup_state.py`, `_wizards.tcss` | First-run reliability Tasks 1/3/4/5, provider foundation, first-run handoff, then TTS Voice-step work. Rebase and rerun both wizard suites after each layer. |
| `settings_endpoint_probe.py` | Provider foundation first; TTS configuration must consume its structured outcome or add a TTS-specific adapter without restoring route guessing. |
| `console_session_settings.py` | Provider/model precedence from provider foundation first, Settings context presentation second, TTS compatibility work last. |
| `pending_handoff_store.py`, `chat_screen.py` | First-chat intent before Settings apply intent; both retain separate typed channels and exact claim semantics. |
| `config.py`, `openai_tts_mappings.json`, `pyproject.toml` | The mapping-resource task is executed once. Speech Lab/TTS plans detect and test the existing canonical file instead of recreating it. |
| `app.py` | First-run startup/recovery work first, notification generation work second, metrics main-block cleanup last. |

### Task 1: Execute and review the four implementation slices

**Files:**
- Follow the four child plans in the listed order.

- [ ] **Step 1: Execute provider connection foundation**

Follow every task and gate in `2026-08-12-provider-connection-foundation-implementation-plan.md`.

Expected: shared endpoint derivation, structured readiness/evidence, and atomic provider/default persistence are committed and green.

- [ ] **Step 2: Execute first-run provider handoff**

First satisfy the reliability prerequisite named by the child plan, then follow every task and gate in `2026-08-12-first-run-provider-handoff-implementation-plan.md`.

Expected: manual setup and exact-draft model discovery work, navigation remains visible, and Start chatting targets an eligible/new Console session.

- [ ] **Step 3: Execute Settings and Console reliability**

Follow every task and gate in `2026-08-12-settings-console-reliability-implementation-plan.md`.

Expected: Settings is task-oriented, save/apply are distinct, refused sends create no history, and retries create no duplicate user turns.

- [ ] **Step 4: Execute launch diagnostics and trust**

Follow every task and gate in `2026-08-12-launch-diagnostics-trust-implementation-plan.md`.

Expected: splash rendering is typed/fenced, notifications are generation-owned, metrics are explicit, and the installed wheel contains the TTS mapping.

- [ ] **Step 5: Run the cross-slice regression command**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/Chat/test_provider_endpoint_contract.py \
  Tests/Chat/test_local_server_discovery.py \
  Tests/Chat/test_provider_readiness.py \
  Tests/Chat/test_provider_setup_persistence.py \
  Tests/Wizards/test_first_run_setup_state.py \
  Tests/Wizards/test_first_run_setup_wizard.py \
  Tests/Wizards/test_first_run_setup_integration.py \
  Tests/UI/test_first_run_wizard_live_contract.py \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_settings_provider_test_draft.py \
  Tests/UI/test_settings_provider_switch_atomic.py \
  Tests/UI/test_settings_apply_current_conversation.py \
  Tests/State/test_pending_handoff_store.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_send_draft_snapshot.py \
  Tests/Widgets/test_splash_frames.py \
  Tests/Widgets/test_splash_lifecycle.py \
  Tests/UI/test_startup_notifications.py \
  Tests/Metrics/test_metrics_startup.py \
  Tests/Packaging/test_built_wheel_resources.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit cross-slice corrections when needed**

```bash
git add tldw_chatbook pyproject.toml Tests
git commit -m "test: integrate first-chat remediation slices"
```

Skip this commit when no integration corrections are needed.

### Task 2: Build the deterministic first-chat acceptance harness

**Files:**
- Create: `Tests/UAT/openai_compatible_mock.py`
- Create: `Tests/UAT/test_first_chat_fresh_profile.py`
- Create: `Tests/UAT/conftest.py`

**Interfaces:**
- `OpenAICompatibleMock(models_status, model_ids, chat_responses)` serves only loopback.
- Captures sanitized request method/path/model; never stores Authorization values or prompt bodies.
- Provides llama mode (`GET /v1/models` 200) and custom mode (`GET /v1/models` 404, `POST /v1/chat/completions` 200).

- [ ] **Step 1: Write the mock server contract tests**

```python
async def test_custom_mock_models_404_but_chat_succeeds(custom_mock):
    async with httpx.AsyncClient() as client:
        models = await client.get(f"{custom_mock.base_url}/v1/models")
        chat = await client.post(
            f"{custom_mock.base_url}/v1/chat/completions",
            json={"model": "custom-model", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert models.status_code == 404
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "custom reply"
```

Add llama models/chat success, streaming response, accepted failure, delayed cancellation, and an egress guard rejecting non-loopback hosts.

- [ ] **Step 2: Run the harness tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/UAT/test_first_chat_fresh_profile.py -k "mock" -v`

Expected: FAIL because the UAT harness does not exist.

- [ ] **Step 3: Implement the loopback-only mock**

Use an in-process ASGI transport for automated tests and a loopback `uvicorn` fixture for live replay. Bind to port `0` where supported. Record only bounded route/status metadata; replace Authorization with a boolean `authorization_present` and do not retain message content.

- [ ] **Step 4: Add automated fresh-profile journeys**

Create two Pilot journeys:

1. llama.cpp entered as a full `/v1/chat/completions` URL, models discovered, setup completed, first Console chat succeeds.
2. custom OpenAI-compatible entered as a base URL, `/models` returns 404, model entered manually, save remains “listing unavailable/chat untested,” new Console session uses the saved pair, and chat succeeds after an app restart.

For each journey assert config sections, confirmation metadata, active session provider/model, exactly one user/assistant pair, and no contradictory verdict text.

- [ ] **Step 5: Add failure/retry/cancel journey assertions**

From the custom-provider session, exercise preflight refusal, accepted failure, retry, and cancellation. Assert zero rows for refusal, one user plus one failed assistant for failure, still one user after retry, and a distinct cancelled assistant state.

- [ ] **Step 6: Run and commit the acceptance harness**

Run: `.venv/bin/python -m pytest Tests/UAT/test_first_chat_fresh_profile.py -v`

Expected: PASS with zero external network calls.

```bash
git add Tests/UAT/openai_compatible_mock.py Tests/UAT/conftest.py Tests/UAT/test_first_chat_fresh_profile.py
git commit -m "test: automate fresh-profile first-chat UAT"
```

### Task 3: Perform terminal and browser live replay

**Files:**
- Create: `Docs/superpowers/qa/first-chat-uat-remediation-2026-08/README.md`
- Create PNG evidence under `Docs/superpowers/qa/first-chat-uat-remediation-2026-08/`.

- [ ] **Step 1: Prepare an isolated profile and mock endpoints**

Use a new directory under `/private/tmp/tldw-first-chat-uat-20260812`, set `HOME` and all supported Chatbook config/data overrides to that directory, and start the two loopback mock modes on separate ports. Confirm no existing user config is read.

- [ ] **Step 2: Replay llama.cpp setup in a native terminal**

Enter the full chat URL, expand optional auth without supplying a key, test, select a discovered model, complete setup, start chatting, and receive the deterministic reply. Record visible endpoint/model/verdict copy and message counts in the QA README.

- [ ] **Step 3: Replay custom OpenAI-compatible setup and restart**

Enter the base URL and optional credential, observe models-listing unavailable, manually enter `custom-model`, save, start a new Console conversation, chat successfully, restart, and chat successfully again. Confirm the configured state after restart is **connection not tested**, not Verified.

- [ ] **Step 4: Capture browser evidence at three viewports**

Start the served app:

```bash
HOME=/private/tmp/tldw-first-chat-uat-20260812 PYTHONPATH=. .venv/bin/python -m tldw_chatbook.app --serve --host 127.0.0.1 --port 9178
```

Use Playwright bundled Chromium, abort non-loopback HTTP(S), and capture first-run Provider, Model, Summary, Settings Connection/Models/Generation/Context, Console refusal, failed attempt, retry success, and restart splash at 700x480, 1200x800, and 2050x1240. Assert DOM/xterm text before each capture and confirm no clipping or overlapping controls.

- [ ] **Step 5: Review startup logs and listeners**

Assert exactly one truthful outcome for TTS mapping load and metrics. With no metrics opt-in, verify no metrics listener. Repeat with `METRICS_PORT=9179` and verify one listener on `127.0.0.1:9179` plus an `Already running` result for duplicate initialization.

- [ ] **Step 6: Commit live evidence**

```bash
git add Docs/superpowers/qa/first-chat-uat-remediation-2026-08
git commit -m "docs: record first-chat remediation UAT"
```

### Task 4: Close the 17-finding acceptance ledger

**Files:**
- Modify: `Docs/superpowers/qa/first-chat-uat-remediation-2026-08/README.md`

| ID | Owning plan/task | Required evidence |
|---|---|---|
| UAT-01 | First-run Tasks 2-3 | llama manual Endpoint, optional auth, base/full URL tests, live chat. |
| UAT-02 | Foundation Tasks 3-4 | Fresh template is incomplete/unconfirmed until explicit acceptance; no false ready state. |
| UAT-03 | Foundation Tasks 2-3 | Models 404 produces one listing-unavailable/chat-untested verdict. |
| UAT-04 | First-run Task 5; Settings Task 3 | New session snapshots saved defaults; user-owned session remains unchanged. |
| UAT-05 | Foundation Task 4 | Atomic paired save, matching-default precedence, explicit repair only. |
| UAT-06 | Launch Tasks 1-2 | Typed frames and restart captures in terminal/browser. |
| UAT-07 | Settings/Console Tasks 4-5 | Zero history on refusal; one user through failure/retry. |
| UAT-08 | Foundation Tasks 3-4 | Equivalent successful save preserves exact process-local evidence. |
| UAT-09 | First-run integration gate/Task 3 | Pinned visible navigation at compact, normal, and wide sizes. |
| UAT-10 | First-run Task 4 | Exact draft discovery, endpoint-scoped cache, manual fallback. |
| UAT-11 | Settings Task 1 | Search/group/unknown/manual provider picker tests and capture. |
| UAT-12 | Settings Task 1 | Configuration/test/storage/privacy-first overview assertion and capture. |
| UAT-13 | Settings Task 2 | Capability groups, scoped reset, compact layout. |
| UAT-14 | First-run Task 5 | One state-dependent primary and unique secondary/tertiary actions. |
| UAT-15 | Launch Task 3; first-run integration | Inline setup status, generation fencing, no obscuring startup toast. |
| UAT-16 | Settings Task 2 | Unknown capacity state, source precedence, expanded calculation details. |
| UAT-17 | Launch Tasks 4-5 | Installed-wheel mapping, lazy assets, default-off/idempotent metrics, truthful logs. |

- [ ] **Step 1: Attach automated evidence to every row**

For each ID, list exact test node IDs and result. No row may cite only a screenshot.

- [ ] **Step 2: Attach live evidence to user-visible rows**

For UAT-01, 03, 04, 06, 07, and 09-16, link the relevant terminal observation and browser PNG.

- [ ] **Step 3: Record residual risks and deferrals**

The only allowed product deferral in this scope is named provider connections. Record that its future design begins by inspecting `tldw_server` connection ownership and migration semantics. Any other unmet row keeps this plan open.

- [ ] **Step 4: Run final quality checks**

Run: `git diff --check origin/dev...HEAD`

Expected: no whitespace errors.

Run: `.venv/bin/python -m pytest Tests/UAT/test_first_chat_fresh_profile.py -v`

Expected: PASS.

- [ ] **Step 5: Commit ledger corrections when needed**

```bash
git add Docs/superpowers/qa/first-chat-uat-remediation-2026-08 Tests/UAT
git commit -m "docs: close first-chat UAT acceptance ledger"
```

Skip this commit when Task 3's evidence commit already contains the final ledger.
