# Real-provider Three-turn Console Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a revision-pinned, mounted-Console benchmark that measures the real three-turn/local-tool path through the local llama.cpp provider without touching user state.

**Architecture:** A single standalone script has parent and child modes. Parent mode remains application-import-free and owns immutable revision resolution, detached-control lifecycle, balanced ordering, subprocess watchdogs, raw validation, statistics, and reports; each child installs its isolated environment before importing the selected revision, mounts that revision's production-shaped Console harness, drives the real composer and prompt queue, and records content-free timestamps around existing seams. Pure helpers live in the runner so a child never imports the candidate `Tests` package before target-root validation.

**Tech Stack:** Python 3.11+, Textual Pilot, httpx, pytest, SQLite, Git shadow repositories, JSONL, standard-library subprocess/statistics/random.

---

## File map

- Create `Tests/Performance/run_console_three_turn_profile.py`: application-import-free parent CLI, pure statistics/contracts, child bootstrap, target adapter, mounted sample driver, watchdog, and report writer.
- Create `Tests/Performance/test_console_three_turn_profile.py`: pure and subprocess-level RED/GREEN coverage for every reusable runner contract; no network calls.
- Create `Docs/superpowers/qa/console-three-turn-real-provider/README.md`: exact reproduction command and evidence inventory after the smoke succeeds.
- Generate `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.raw.jsonl`: continuously flushed boundary/failure records and terminal heartbeat vectors.
- Generate `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.manifest.json`: revisions, model/runtime metadata, fixture hashes, arm order, and safe host facts.
- Generate `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.summary.json`: recomputable per-arm distributions, paired confidence bounds, validity gates, and verdicts.
- Generate `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn-summary.md`: concise human interpretation separating provider and application latency.
- Modify `backlog/tasks/task-20009 - Measure-real-provider-three-turn-Console-latency.md`: check acceptance criteria, add implementation/evidence notes, retain ADR-079 linkage, and mark Done only after the full evidence review.

ADR required: no
ADR path: `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md` (existing governing ADR)
Reason: the task adds opt-in benchmark tooling and evidence only; it does not change production ownership, storage, provider contracts, or user-visible behavior.

### Task 1: Establish the pure benchmark contract

**Files:**
- Create: `Tests/Performance/run_console_three_turn_profile.py`
- Create: `Tests/Performance/test_console_three_turn_profile.py`

- [ ] **Step 1: Write failing rotation and nearest-rank tests**

```python
def test_balanced_arm_order_rotates_complete_triples():
    assert profile.balanced_arm_order(0) == ("control", "disabled", "enabled")
    assert profile.balanced_arm_order(1) == ("disabled", "enabled", "control")
    assert profile.balanced_arm_order(2) == ("enabled", "control", "disabled")
    assert profile.balanced_arm_order(3) == profile.balanced_arm_order(0)


def test_nearest_rank_percentile_uses_one_based_ceiling():
    values = list(range(1, 31))
    assert profile.nearest_rank_percentile(values, 0.95) == 29
```

- [ ] **Step 2: Run the tests and verify behavioral RED**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`

Expected: collected tests fail because `balanced_arm_order` and `nearest_rank_percentile` are absent; the test module itself imports successfully.

- [ ] **Step 3: Implement the smallest pure helpers**

```python
ARMS = ("control", "disabled", "enabled")


def balanced_arm_order(iteration: int) -> tuple[str, str, str]:
    offset = iteration % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires values and 0 < fraction <= 1")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]
```

- [ ] **Step 4: Write RED tests for paired-block bootstrap and heartbeat reduction**

Cover deterministic seed stability, complete-triple resampling, one-sided/two-sided quantiles, zero-control refusal, and reduction of one heartbeat p95 per sample before arm aggregation.

- [ ] **Step 5: Implement `paired_p95_ratio_bounds` and `sample_heartbeat_p95_ns` minimally**

`paired_p95_ratio_bounds(blocks, candidate, *, resamples=10_000, seed=19641)` resamples whole iteration dictionaries, computes a nearest-rank p95 for control and candidate on each resample, and returns `two_sided_95`, `one_sided_lower_95`, and `one_sided_upper_95`. It raises on missing arms, incomplete blocks, nonpositive control p95, or fewer than two blocks.

- [ ] **Step 6: Run focused tests GREEN**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`

Expected: all Task 1 tests pass with no warnings.

- [ ] **Step 7: Commit Task 1**

```bash
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "test(perf): define three-turn benchmark contract"
```

### Task 2: Add fail-closed sample, privacy, and report validation

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py`
- Modify: `Tests/Performance/test_console_three_turn_profile.py`

- [ ] **Step 1: Write RED tests for arm-specific sample validation**

Build one minimal valid row per arm and mutants for:

- missing terminal third assistant/provider;
- provider round counts other than `1/3/1`;
- missing or extra `load_tools`/`fs_write` calls;
- wrong `local:fs_write` id, path, payload hash, or permission hash;
- `third_send_requested_ns >= turn_2_release_ns`;
- disabled-arm review events present;
- control legacy baseline/E boundaries missing;
- enabled candidate schedule/start/complete boundaries missing;
- E occurring before, across, or after the third send without invalidating an otherwise complete tracked sample.

Use `validate_sample(row) -> tuple[str, ...]`; tests assert stable category codes rather than prose.

- [ ] **Step 2: Verify the validation tests fail for missing behavior**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'sample or arm_contract'`

Expected: assertions fail because arm schemas and validators are absent.

- [ ] **Step 3: Implement immutable arm contracts and validation**

```python
ARM_CONTRACTS = {
    "control": ArmContract(
        required_review=("baseline_started", "baseline_ready", "review_e_started", "review_e_completed"),
        prohibited_review=("finalization_scheduled",),
    ),
    "disabled": ArmContract(required_review=(), prohibited_review=ALL_REVIEW_EVENTS),
    "enabled": ArmContract(
        required_review=("baseline_started", "baseline_ready", "finalization_scheduled", "review_e_started", "review_e_completed"),
        prohibited_review=(),
    ),
}
```

The validator reads timestamps/counts/hashes only. It never needs prompt, response, tool-result, or file content.

- [ ] **Step 4: Write RED tests for run completeness and decision rules**

Cover exactly 30 samples per arm, 90 unique sample ids, 30 complete rotation blocks, invalid-overrides-pass, both non-regression gates required, `pass`/`regression`/`inconclusive` confidence-bound cases, and no critical-path improvement claim from provider or whole-conversation time.

- [ ] **Step 5: Implement `validate_run` and `build_summary`**

The summary must preserve all failure counts and return overall `invalid` unless every sample contract passes. A candidate arm passes only when both p95-ratio upper bounds are `<= 1.10`; lower bound `> 1.10` is `regression`, otherwise `inconclusive`.

- [ ] **Step 6: Write RED tests for privacy and JSONL flushing**

Cover recursive rejection of credential-shaped keys, headers, prompt/response/tool bodies, absolute home/worktree/venv paths, and unnormalized exception paths. Verify `write_boundary_event` calls `flush()` immediately, while `HeartbeatBuffer.record()` performs no writes and `write_terminal_sample` emits its integer vector once.

- [ ] **Step 7: Implement privacy scan, path normalization, and event writers**

Use fixed replacement roots (`$CONTROL`, `$CANDIDATE`, `$RUN`, `$VENV`, `$HOME`) before serialization. Do not retain arbitrary environment mappings or command lines.

- [ ] **Step 8: Run Task 2 tests GREEN and commit**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`

```bash
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "test(perf): validate three-turn benchmark evidence"
```

### Task 3: Build isolated target bootstrap and bounded parent ownership

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py`
- Modify: `Tests/Performance/test_console_three_turn_profile.py`

- [ ] **Step 1: Write RED tests for the child environment**

Call `build_child_environment(base_env, sample_root)` with fake cloud keys, proxy variables, `PYTHONPATH`, and unrelated host paths. Assert the result contains only the fixed safe allowlist plus sample-scoped `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, `TLDW_CONFIG_PATH`, `TLDW_TEST_MODE=1`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONUNBUFFERED=1`; provider credentials/proxies and inherited repository paths are absent.

- [ ] **Step 2: Implement environment construction without importing app code**

The parent creates the config file with explicit `[paths] data_dir`, completed first-run state, disabled splash/background refresh, local tools enabled, `workspace_root` blank, and llama.cpp endpoint/model. The child asserts all paths before its first target import.

- [ ] **Step 3: Write RED subprocess tests for import-root validation**

Build tiny temporary `target_a`/`target_b` packages. Launch child bootstrap from A targeting B and assert `tldw_chatbook`, `Tests.UI.test_destination_shells`, and `Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation` resolve below B. A deliberate candidate-path import must fail with `target_import_mismatch`.

- [ ] **Step 4: Implement `install_target_root` and `assert_target_modules`**

Before target imports: set `sys.dont_write_bytecode = True`, remove already-loaded `tldw_chatbook`/`Tests` modules, insert the resolved target at `sys.path[0]`, invalidate import caches, and require every resolved module path to be below the target.

- [ ] **Step 5: Write RED watchdog tests with a real disposable child process**

Cover normal exit, nonzero exit preserving the last flushed event, deadline expiry sending TERM, TERM-resistant child receiving KILL after the fixed grace period, and process-group cleanup. Keep deadlines injectable so tests finish in under two seconds.

- [ ] **Step 6: Implement `run_child_with_watchdog`**

Launch each child in its own process group/session. On deadline: TERM, bounded wait, KILL, bounded reap. Return a typed result with status/category/returncode; never silently discard partial JSONL.

- [ ] **Step 7: Add parent CLI/revision tests and implementation**

Parse `--endpoint`, `--model`, `--iterations`, `--output-root`, `--control-sha`, `--candidate-sha`, `--sample-timeout`, and internal `--child-spec`. Resolve both hashes once, require the control hash to equal `5f720a40417eaa78f33619d5cbc82effc470104b`, create one detached control worktree below the run root, and record hashes before any sample. Test commands through injected `run_command`; do not create real worktrees in unit tests.

The parent runs one explicitly tagged untimed warmup per arm before measured iteration zero, then schedules only complete measured rotation blocks. Tests prove warmups are excluded from the ninety-sample validator, summaries, heartbeat distributions, and confidence-bound inputs while their success remains a fail-closed precondition for measurement.

- [ ] **Step 8: Run Task 3 tests GREEN and commit**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`

```bash
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "feat(perf): isolate revision-pinned benchmark children"
```

### Task 4: Drive one real mounted sample through the target revision

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py`
- Modify: `Tests/Performance/test_console_three_turn_profile.py`

- [ ] **Step 1: Write RED adapter-fingerprint tests**

Against source fixtures representing control and candidate shapes, require:

- control: no consent service, legacy `ChangeTurnTracker.end_turn` owned synchronously by the bridge;
- candidate disabled/enabled: explicit consent service and `ChangeReviewFinalizationCoordinator.finalize` schedule/start/completion seams;
- a mismatch in required/prohibited attributes fails before provider contact.

- [ ] **Step 2: Implement the small `TargetAdapter`**

Use public/semipublic runtime seams where shared and isolate revision differences in methods named `configure_control_review`, `configure_candidate_disabled_review`, `configure_candidate_enabled_review`, `install_timing_wrappers`, and `review_events`. No revision check is scattered through the mounted driver.

- [ ] **Step 3: Write RED tests for workspace/tool preparation using real temporary DBs**

Build a real `WorkspaceDB`, `LocalWorkspaceRegistryService`, workspace, and one `allow_write=True` folder binding. For candidate enabled, call `set_change_review_enabled`, construct `ChangeReviewConsentService`, and wait for `RootReadinessState.READY` before timing. For disabled, assert missing consent creates no shadow repository. Construct `UnifiedMCPControlPlaneService`, derive the live `fs_write` `HubTool` from `LocalToolProvider`, call `set_tool_state(..., "allow", tool=hub)`, and assert `gate_tool_test(hub)` is exact `allow` with the stored definition hash.

- [ ] **Step 4: Implement deterministic corpus and snapshot preparation**

Generate 1,024 files from a fixed index-derived byte pattern, one fixed 8 MiB blob, and the pre-created `measured/` directory. Record relative path, byte length, and SHA-256 in a manifest; compare content-tree digests across arms, never commit ids.

- [ ] **Step 5: Write a no-network mounted scripted-gateway test for the common trigger**

Use the target's real `ConsoleHarness`, real composer key entry, real prompt queue, real local `fs_write`, and a deterministic gateway that emits the exact `1/3/1` rounds. The turn-two terminal provider completion sets a Textual-loop event; the sample coroutine then presses Enter on the already-typed third draft. Assert:

```python
assert third_send_requested_ns < turn_2_release_ns
assert provider_round_counts == {1: 1, 2: 3, 3: 1}
assert tool_calls == ["load_tools", "fs_write"]
assert third_provider_started_ns is not None
assert (workspace / "measured/turn-two.txt").read_bytes() == FIXED_MUTATION
```

The test must fail if the mounted driver calls the controller/bridge directly instead of typing through the composer and pressing Enter.

- [ ] **Step 6: Implement `run_mounted_sample`**

Import target helpers only after bootstrap. Build a fresh file-backed `CharactersRAGDB`, workspace DB/service, real `ConsoleProviderGateway(config_provider=lambda: app.app_config, environ=isolated_env)`, and `ConsoleHarness`. Wrap and immediately delegate at these existing seams:

- `ConsolePromptQueueUIController.dispatch` for composer/send request;
- `ConsolePromptQueueCoordinator.turn_accepted`, `_after_turn`, and `_drain_waiting` for admission/release/claim;
- `ConsoleChatController._run_agent_reply` for worker entry;
- `ConsoleProviderGateway.stream_chat` for per-round start/first-chunk/completion and the common call-four trigger;
- `ConsoleChatStore` terminal persistence or `_record_run_assistant_message` for durable assistant anchoring;
- `ChangeTurnTracker` baseline/end methods and candidate coordinator finalize/worker methods for arm-specific review timing.

Drive prompts with Pilot key events. Type turn three while turn two is accepted, wait for the call-four completion event, timestamp the request, then press Enter without waiting for turn-two settlement. Wait for terminal third output, coordinator idle, heartbeat settlement, and owned resource teardown.

- [ ] **Step 7: Add child final ownership and write-inventory assertions**

Assert no live benchmark-owned threads, workers, provider clients, SQLite connections, or shadow operations remain. Compare before/after inventories outside sample/run roots and reject any write. Emit terminal sample only after these checks.

- [ ] **Step 8: Run Task 4 tests GREEN and commit**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`

Run: `.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py -q -k 'third_turn_starts_while_second_review_e_is_held or prompt_queue'`

```bash
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "feat(perf): drive mounted three-turn Console samples"
```

### Task 5: Fail-fast real-provider smoke

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider/README.md`

- [ ] **Step 1: Run static/focused verification before network contact**

Run:

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/ruff check Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
```

- [ ] **Step 2: Run the application-import-free endpoint/model preflight**

Run the runner's `--preflight-only` mode against `http://127.0.0.1:9099`. Require the expected model, temperature-zero completion, and no credentials.

- [ ] **Step 3: Run one mounted sample per arm in rotated order**

```bash
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --iterations 1 \
  --output-root /tmp/tldw-console-three-turn-smoke
```

Expected: three complete valid samples; exact `1/3/1` rounds; one `load_tools`, one confined `fs_write`, terminal third assistants; common trigger before turn-two release; disabled arm has no review work; tracked arms have complete E observations.

- [ ] **Step 4: Inspect smoke evidence before allowing the long run**

Independently recalculate counts/timestamps and response-token accounting from raw JSONL, scan paths/keys, inspect filesystem inventories, and confirm target module paths/hashes. Verify the manifest records the fixed temperature, 512-token cap, prompt/schema fixture ids and hashes, dependency/runtime versions, sanitized llama-server metadata, host load, and listener resource samples. If any gate fails, stop and fix with a new RED regression test.

- [ ] **Step 5: Document the exact reproduction command and smoke result**

The README records prerequisites, endpoint/model, 512-token cap, synthetic fixture ids, expected duration, output files, and the fact that no cloud credential or user content is used.

- [ ] **Step 6: Commit the smoke-ready harness**

```bash
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py Docs/superpowers/qa/console-three-turn-real-provider/README.md
git commit -m "test(perf): validate real-provider benchmark smoke"
```

### Task 6: Collect and independently verify 30 balanced samples per arm

**Files:**
- Generate: `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.raw.jsonl`
- Generate: `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.manifest.json`
- Generate: `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.summary.json`
- Generate: `Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn-summary.md`
- Modify: `Docs/superpowers/qa/console-three-turn-real-provider/README.md`

- [ ] **Step 1: Freeze the candidate hash and launch the complete run**

Do not edit imported source while the long run is active. Run:

```bash
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --iterations 30 \
  --output-root Docs/superpowers/qa/console-three-turn-real-provider
```

- [ ] **Step 2: Monitor without changing the tree**

Use the runner's unbuffered progress lines and raw boundary tail. Do not infer health from buffered stdout or edit files imported by a running single-process child.

- [ ] **Step 3: Recompute every statistic independently from raw JSONL**

Use a read-only one-off verifier with a different code path for sample counts, round counts, medians, nearest-rank p95s, heartbeat reduction, application intervals, paired bootstrap inputs, failure counts, and final verdicts. Compare exact JSON values to the generated summary.

- [ ] **Step 4: Run privacy and isolation scans**

Require zero absolute host/home/worktree/venv paths, zero credential/header/environment fields, zero prompt/response/tool/file bodies, zero writes outside allowlisted run/sample roots, and zero user DB/config/shadow-repository fingerprint changes.

- [ ] **Step 5: Write the human summary conservatively**

Report provider latency separately. Claim application improvement only from `assistant_durable -> turn_release` and `terminal_provider_complete -> third_worker/provider` when the pre-registered interval permits it. Report total conversation wall time descriptively only. Preserve `invalid` or `inconclusive` without override.

- [ ] **Step 6: Commit retained evidence**

```bash
git add Docs/superpowers/qa/console-three-turn-real-provider
git commit -m "perf(console): retain real-provider three-turn evidence"
```

### Task 7: Close verification and backlog hygiene

**Files:**
- Modify: `backlog/tasks/task-20009 - Measure-real-provider-three-turn-Console-latency.md`
- Modify if generalized knowledge was learned: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run final focused and changed-surface gates**

Run:

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_change_review_finalization.py -q
.venv/bin/ruff check Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
```

- [ ] **Step 2: Self-review the complete diff against the spec and ADR-079**

Check that no production file changed, the disabled path creates no shadow state, the trigger is identical across arms, wrappers are observational, every required boundary is real, and evidence claims match raw data.

- [ ] **Step 3: Update the five-digit task file directly**

Check all seven ACs only when supported, add concise Implementation Notes with modified files/evidence/verdict/ADR check, and set `status: Done`. Do not use the Backlog CLI edit command for this five-digit id.

- [ ] **Step 4: Add a lesson only if this run produced a new reusable incident**

Record the incident and evidence, not a generic rule. If no new lesson emerged, state that no lessons file was changed.

- [ ] **Step 5: Commit closeout**

```bash
git add 'backlog/tasks/task-20009 - Measure-real-provider-three-turn-Console-latency.md' backlog/docs
git commit -m "docs(perf): close real-provider Console benchmark"
```

- [ ] **Step 6: Final verification-before-completion audit**

Confirm clean worktree, exact committed hashes, all commands' fresh outputs, complete evidence inventory, and task DoD. Report any Git loose-object maintenance warning separately; do not run destructive pruning as part of this benchmark.
