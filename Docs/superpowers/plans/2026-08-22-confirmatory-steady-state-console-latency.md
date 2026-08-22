# Confirmatory Steady-state Console Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing real-provider Console benchmark with a review-bound, first-valid-attempt confirmation that excludes five predeclared balanced burn-in blocks and measures 30 fresh blocks per arm.

**Architecture:** Keep the existing parent/child runner and tests as the only executable files. Add a thin phase/campaign wrapper around the proven sample path, load the digest-verified original runner for validation/statistics, and use standard-library JSONL, hashing, directory locking, copy, and atomic rename for attempt governance and publication. The mounted Console path and all production modules remain unchanged.

**Tech Stack:** Python 3.11+, pytest, Textual Pilot, SQLite, Git detached worktrees, JSON/JSONL, `hashlib`, `pathlib`, `shutil`, `subprocess`, and the local llama.cpp OpenAI-compatible endpoint.

---

## File map

- Modify `Tests/Performance/run_console_three_turn_profile.py`: schedule phases, exact raw-sequence validation, original-runner loading, target/protocol preflight, campaign ledger/lock, review digest, and atomic promotion.
- Modify `Tests/Performance/test_console_three_turn_profile.py`: all RED/GREEN unit and subprocess coverage; no live network calls.
- Rename `backlog/tasks/task-19641 - Measure-real-provider-three-turn-Console-latency.md` to `backlog/tasks/task-20009 - Measure-real-provider-three-turn-Console-latency.md`: remove the latest-dev task-ID collision without changing retained evidence bytes.
- Rename `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md` to `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md`: preserve dev's canonical ADR-077 and renumber this branch's Change Review ADR.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/README.md`.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.raw.jsonl`.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.manifest.json`.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.summary.json`.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn-summary.md`.
- Generate only after approved promotion `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/confirmatory-review-receipt.json`.
- Modify `backlog/tasks/task-20010 - Confirm-steady-state-three-turn-Console-latency.md`: implementation notes, AC completion, ADR result, evidence/verdict, and final status.
- Modify a lessons file only if execution produces a new reusable incident.

ADR required: no

ADR path: `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md` after latest-dev integration (existing governing ADR, renumbered from the branch-local ADR-077)

Reason: TASK-20010 changes benchmark-only tooling and retained evidence. It does not change production storage, provider/runtime boundaries, privacy ownership, security policy, or user-visible behavior; the renumbered ADR-079 already governs the Change Review behavior being measured.

### Task 0: Integrate the work with the latest dev without rewriting benchmark evidence

**Files:**
- Rename: `backlog/tasks/task-19641 - Measure-real-provider-three-turn-Console-latency.md` → `backlog/tasks/task-20009 - Measure-real-provider-three-turn-Console-latency.md`
- Rename: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md` → `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md`
- Modify: branch-owned specs, plans, tasks, decision index, and lessons that point to those two identities
- Preserve byte-for-byte: `Docs/superpowers/qa/console-three-turn-real-provider/`
- Preserve protocol fixture strings and original runner digest: `Tests/Performance/run_console_three_turn_profile.py`

- [ ] **Step 1: Refresh refs and prove the replacement identities are free**

```bash
git fetch --all --prune
git rev-parse origin/dev
git ls-tree -r --name-only origin/dev backlog/tasks backlog/decisions | rg 'task-20009|task-20010|/079-'
git cat-file -e 'eb8225a32f88ea43c337aff99804d360384e7668^{commit}'
git update-ref refs/benchmarks/task-20009-candidate eb8225a32f88ea43c337aff99804d360384e7668
```

Also scan every fetched ref, not only `origin/dev`. Require TASK-20009, TASK-20010, and ADR-079 to be unused outside this branch before proceeding. Record the current original QA hashes from the approved spec. The private benchmark ref keeps the digest-pinned pre-rebase candidate object reachable while branch commits are rewritten; verify it still resolves to the exact commit after rebase.

- [ ] **Step 2: Rebase onto the exact refreshed `origin/dev`**

```bash
git rebase origin/dev
```

Resolve the known identity conflicts explicitly: retain dev's unrelated TASK-19641 and canonical server-offload ADR-077, while preserving this branch's benchmark task as TASK-20009 and Change Review ADR as ADR-079. Never resolve by dropping either side's work. Re-run the all-ref uniqueness scan after the rebase.

- [ ] **Step 3: Update only mutable identity references**

Set the renamed task frontmatter to `TASK-20009`, renumber the Change Review decision heading/index/links to ADR-079, and update branch-owned specs, plans, tasks, and lessons accordingly. In the confirmation spec, state that the immutable original artifacts internally retain their pre-integration `TASK-19641` label. Do not edit the original QA directory or the original runner's task-labeled fixture strings, because both are digest-pinned protocol inputs.

- [ ] **Step 4: Revalidate and pin the exact post-integration baseline**

Recompute all original runner/evidence hashes, verify both task IDs and ADR numbers are unique, and run the existing benchmark and changed-surface focused tests. Commit only the renumber/reference integration if it is not already represented by the rebase conflict resolution. After all pre-implementation integration commits, pin that exact `HEAD` as `refs/benchmarks/task-20010-implementation-base`; later code-scope checks must resolve this ref rather than recomputing a merge base.

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_change_review_finalization.py -q
git diff --check
git add backlog/decisions backlog/tasks Docs/superpowers/specs Docs/superpowers/plans backlog/docs
git commit -m "chore: align Console confirmation with latest dev"
git update-ref refs/benchmarks/task-20010-implementation-base HEAD
git rev-parse refs/benchmarks/task-20010-implementation-base
```

Do not create an empty commit if the rebase already recorded every required resolution. Record the resolved implementation-base SHA in the campaign manifest and TASK-20010 closeout notes; do not move the ref after Task 1 begins.

### Task 1: Add the burn-in schedule and exact pre-filter gate

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py:140-147,430-433,503-684,1312-1322`
- Test: `Tests/Performance/test_console_three_turn_profile.py:207-384,483-495`

- [ ] **Step 1: Write RED schedule tests**

```python
def test_confirmatory_schedule_continues_rotation_after_five_burn_in_blocks():
    schedule = profile.sample_schedule(30, burn_in_blocks=5)
    assert [(row.phase, row.arm, row.iteration) for row in schedule[:3]] == [
        ("warmup", "control", -1),
        ("warmup", "disabled", -1),
        ("warmup", "enabled", -1),
    ]
    assert len([row for row in schedule if row.phase == "burn_in"]) == 15
    measured = [row for row in schedule if row.phase == "measured"]
    assert len(measured) == 90
    assert [row.arm for row in measured[:3]] == list(profile.balanced_arm_order(5))
    assert [row.iteration for row in measured[:3]] == [0, 0, 0]


def test_zero_burn_in_keeps_the_existing_schedule():
    assert profile.sample_schedule(4) == profile.sample_schedule(4, burn_in_blocks=0)
```

- [ ] **Step 2: Run the schedule tests and verify RED**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'schedule or burn_in'`

Expected: FAIL because `sample_schedule` does not accept `burn_in_blocks`.

- [ ] **Step 3: Implement the minimum schedule extension**

```python
def sample_schedule(
    iterations: int, *, burn_in_blocks: int = 0
) -> tuple[SamplePlan, ...]:
    if iterations < 1 or burn_in_blocks < 0:
        raise ValueError("schedule counts must be nonnegative with measured iterations")
    schedule = [SamplePlan("warmup", arm, -1) for arm in ARMS]
    for block in range(burn_in_blocks):
        schedule.extend(
            SamplePlan("burn_in", arm, block)
            for arm in balanced_arm_order(block)
        )
    for iteration in range(iterations):
        schedule.extend(
            SamplePlan("measured", arm, iteration)
            for arm in balanced_arm_order(burn_in_blocks + iteration)
        )
    return tuple(schedule)
```

- [ ] **Step 4: Write RED exact-sequence tests**

Build 108 valid terminal rows from the schedule, each with its exact zero-based `schedule_position`. Cover exact success, reordered/missing/extra rows, missing/mismatched schedule positions, an unknown phase, within-phase and cross-phase duplicate IDs, and a burn-in row rejected by the injected original `validate_sample`.

- [ ] **Step 5: Run the exact-sequence tests and verify RED**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'confirmation_rows or cross_phase'`

Expected: FAIL because `validate_confirmation_rows` is absent.

- [ ] **Step 6: Implement the exact wrapper without changing statistics**

```python
def validate_confirmation_rows(
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[SamplePlan],
    *,
    validate_sample: Callable[[Mapping[str, Any]], tuple[str, ...]],
) -> tuple[tuple[str, ...], list[Mapping[str, Any]]]:
    expected = [
        (
            f"{plan.phase}-{plan.iteration}-{plan.arm}",
            plan.phase,
            plan.iteration,
            plan.arm,
            position,
        )
        for position, plan in enumerate(schedule)
    ]
    observed = [
        (
            row.get("sample_id"),
            row.get("phase"),
            row.get("iteration"),
            row.get("arm"),
            row.get("schedule_position"),
        )
        for row in rows
    ]
    errors: list[str] = []
    if observed != expected:
        errors.append("confirmation_schedule_contract")
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("confirmation_sample_id_duplicate")
    if any(validate_sample(row) for row in rows):
        errors.append("confirmation_sample_contract")
    filtered = [row for row in rows if row.get("phase") != "burn_in"]
    return tuple(errors), filtered
```

Do not modify `nearest_rank_percentile`, `paired_p95_ratio_bounds`, `validate_sample`, `validate_run`, or `build_summary`.

- [ ] **Step 7: Run Task 1 GREEN and commit**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'schedule or confirmation_rows or cross_phase'
git diff --check
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "test(perf): define confirmatory burn-in schedule"
```

### Task 2: Pin the original harness and fail closed on protocol drift

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py:32-137,1166-1310,1312-1482,2809-3036`
- Test: `Tests/Performance/test_console_three_turn_profile.py:483-735,811-919,1150-1224`

- [ ] **Step 1: Write RED original-runner and evidence-guard tests**

Pin:

```python
ORIGINAL_HARNESS_SHA = "eb8225a32f88ea43c337aff99804d360384e7668"
ORIGINAL_RUNNER_SHA256 = "fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae"
CANDIDATE_SHA = "eb8225a32f88ea43c337aff99804d360384e7668"
```

Test all five original QA artifact hashes, altered-byte rejection, isolated module loading, direct calls to the original module's validators/summary, and summary byte-equivalence when only burn-in metrics change.

- [ ] **Step 2: Verify the original-guard tests fail**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'original_runner or original_evidence or burn_in_summary'`

Expected: FAIL because constants/loaders/guards are absent.

- [ ] **Step 3: Generalize detached target ownership and load the original module**

Replace the control-only helper with `prepare_target_worktree(repository_root, run_root, *, name, revision, run_command=subprocess.run)`. Create fixed `control` and `candidate` worktrees. Verify the original runner inside the candidate worktree, load it as `task_19641_original_runner`, and invoke its `validate_sample`, `validate_run`, and `build_summary` rather than copying its transitive statistics path.

- [ ] **Step 4: Write RED protocol-equivalence tests**

Mutate one field at a time and require exact equality for revisions, provider kind, every `provider_server` field, complete `runtime`, model alias, request settings, fixture IDs/hashes, six metric names, two primary gates, 30 blocks, 10,000 resamples, seed `19_641`, 95% bounds, 1.10 non-regression, and 1.00 improvement. Prove no Markdown file is parsed.

- [ ] **Step 5: Implement pure protocol comparison and preflight child**

Return one `confirmation_protocol(...)` mapping and stable mismatch codes. Derive prompt/mutation hashes from current constants, corpus digest from the existing generator, and per-arm tool-definition hashes through an isolated preflight child using each pinned target and existing `prepare_workspace_runtime`. Add exact child-spec `mode: "sample" | "protocol_preflight"`; preflight never mounts a conversation and fully tears down.

- [ ] **Step 6: Write RED clean-harness/listener tests and implement guards**

Cover dirty/untracked refusal, full current revision/runner digest, listener PID/start converted to a retained SHA-256 fingerprint, continuity at every sample boundary, changed-listener invalidation, and continued omission of raw PID/command/model path. Keep `listener_resource_snapshot` content-free; use a separate injected-command identity helper.

- [ ] **Step 7: Run Task 2 GREEN and commit**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'original or protocol or revision or worktree or listener or runtime'
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "test(perf): pin confirmatory benchmark protocol"
```

### Task 3: Add the append-only campaign and acquisition lock

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py:805-883,1366-1456,2821-3036`
- Test: `Tests/Performance/test_console_three_turn_profile.py:398-449,555-735,940-1133`

- [ ] **Step 1: Write RED ledger-transition tests**

Cover sequential `attempt-NNNN` IDs; flushed/fsynced append-only JSONL; acquisition blocked by `running`, `complete_pending_review`, and `changes_required`; retry only after `failed`/uncorrectable `invalid`; all measured verdicts entering pending review; correctable derived defects preserving the raw hash; and complete lineage.

- [ ] **Step 2: Run ledger tests and verify RED**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'campaign or attempt or changes_required'`

Expected: FAIL because campaign helpers are absent.

- [ ] **Step 3: Implement minimal JSONL state helpers**

```python
BLOCKING_ATTEMPT_STATES = frozenset(
    {"running", "complete_pending_review", "changes_required"}
)


def append_attempt_state(ledger: Path, event: Mapping[str, Any]) -> None:
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
```

Reject unknown fields/states and sensitive values. Never overwrite/truncate the ledger.

- [ ] **Step 4: Write RED atomic-lock/interruption tests**

Inject process-identity probes. Cover atomic `.campaign-lock` creation, second-owner refusal, owner-token release, abrupt death, live-owner recovery refusal, dead exact-owner `failed:interrupted`, and PID reuse with a different start identity.

- [ ] **Step 5: Implement the standard-library lock and staging**

Use `Path.mkdir()` without `exist_ok`. Keep PID/start/owner token only in the campaign-private lock. Recovery never completes/repairs raw evidence. Create `campaign/attempts/attempt-NNNN/`; preserve failure evidence and clean only detached target worktrees.

- [ ] **Step 6: Run Task 3 GREEN and commit**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'campaign or attempt or lock or interrupted'
git diff --check
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "feat(perf): govern confirmatory acquisition attempts"
```

### Task 4: Wire confirmatory acquisition end to end

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py:126-155,964-1010,1312-1482,2740-3052`
- Test: `Tests/Performance/test_console_three_turn_profile.py:483-735,940-1133,1150-1455`

- [ ] **Step 1: Write RED CLI truth-table tests**

Add `--campaign-root`, `--burn-in-blocks` (default 0), `--campaign-action acquire|recover|digest|register-review|promote` (default `acquire`), `--attempt-id`, `--review-receipt`, and `--destination`. Acquisition/preflight requires endpoint/model; other actions do not contact the provider; legacy zero-burn-in/output-root remains valid; burn-in requires the fixed candidate.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'parse_arguments or main_dispatches or campaign_action'`

Expected: FAIL because new modes are absent.

- [ ] **Step 3: Implement fail-closed acquisition ordering**

For `acquire`: lock/create attempt; verify clean harness/current digest/original evidence/revisions; create both target worktrees; run protocol-preflight children; compare server/runtime contract; record listener fingerprint; only then start conversations. A pre-conversation failure is `invalid` with a stable code.

- [ ] **Step 4: Wire the scheduled child loop**

Use `sample_schedule(..., burn_in_blocks=...)` for progress and manifest. Pass and retain the exact zero-based `schedule_position` in every child-start and terminal row. Control uses `control_root`; disabled/enabled use fixed `candidate_root`. Verify listener identity before every child. Keep watchdog, mounted path, immediate validation/privacy, cleanup, and terminal ownership unchanged.

- [ ] **Step 5: Validate raw and call the original statistics module**

```python
terminal_rows = read_terminal_samples(raw_path)
errors, statistics_rows = validate_confirmation_rows(
    terminal_rows, schedule, validate_sample=original_runner.validate_sample
)
if errors:
    raise RuntimeError("confirmation_run_invalid")
if original_runner.validate_run(statistics_rows, expected_iterations=args.iterations):
    raise RuntimeError("original_run_validation_failed")
summary = (
    original_runner.build_summary(statistics_rows)
    if args.iterations >= 2
    else smoke_summary(statistics_rows)
)
```

Add only excluded burn-in count/contract status. Record attempt lineage through pending review, protocol equality, original/current harness identity, listener fingerprint, and model-weight limitation.

- [ ] **Step 6: Write parent integration tests**

Assert 108 official samples, target root by arm, preflight before warmup, listener checks, burn-in fail-fast, summary invariance, worktree cleanup, and pending-review blocking. Use real temp JSONL/directories with provider/child calls monkeypatched.

- [ ] **Step 7: Run Task 4 GREEN and commit**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "feat(perf): run steady-state confirmation acquisitions"
```

### Task 5: Bind independent review to the exact artifacts and publish atomically

**Files:**
- Modify: `Tests/Performance/run_console_three_turn_profile.py`
- Test: `Tests/Performance/test_console_three_turn_profile.py`

- [ ] **Step 1: Write RED canonical-artifact-digest tests**

Define the exact reviewed artifact allowlist as `README.md`, the raw JSONL, manifest JSON, machine summary JSON, and human summary Markdown. Test that the digest is the SHA-256 of a sorted canonical-JSON map from relative filename to file SHA-256. Reject missing or extra entries, symlinks, and absolute paths. Keep campaign state, locks, and review receipts outside the artifact set.

- [ ] **Step 2: Write RED review-receipt and registration tests**

Require an exact receipt schema containing the attempt ID, canonical artifact-set digest, decision (`approved` or `changes_required`), reviewer identity, timestamp, privacy confirmation, and content-free findings. Reject unknown or sensitive fields. Test that `changes_required` keeps the attempt non-publishable and does not authorize reacquisition; corrected derived artifacts preserve the raw acquisition hash and require a fresh digest and receipt.

- [ ] **Step 3: Write RED promotion tests**

Cover missing or non-approved receipts, attempt/digest mismatch, an already-existing destination (including a directory containing only `README.md`), changed source artifacts, destination-copy corruption, and source mutation between copy and verification. Assert that success publishes exactly the five reviewed artifacts plus the approving receipt.

- [ ] **Step 4: Implement digest, receipt registration, and verified atomic promotion**

Use only the standard library. Rehash the source artifact set before copying, copy it with `shutil.copy2()` into a deterministic absent sibling temporary directory, copy the approving receipt, rehash both the source and destination copy after copying, and atomically rename the sibling temporary directory to the absent final destination only when both still equal the reviewed digest. On every mismatch, fail closed and leave the reviewed attempt intact. Do not add a resumable publication protocol or destructive cleanup.

- [ ] **Step 5: Run Task 5 GREEN and commit**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q -k 'artifact or digest or receipt or review or promot'
.venv/bin/ruff check Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
git add Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
git commit -m "test(perf): bind confirmation review and publication"
```

### Task 6: Verify the committed harness and run a disposable smoke campaign

**Files:**
- Verify: `Tests/Performance/run_console_three_turn_profile.py`
- Verify: `Tests/Performance/test_console_three_turn_profile.py`
- Do not retain: disposable smoke campaign directory

- [ ] **Step 1: Run the committed harness verification suite**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/ruff check Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
git diff --check
git status --short
```

Expected: tests, lint, compilation, and whitespace checks pass; status contains only intentional task/spec/plan changes before their documentation commit, or is clean after that commit.

- [ ] **Step 2: Reverify immutable inputs and the live endpoint**

Recompute the original runner and TASK-19641 evidence hashes recorded in the spec. Query `http://127.0.0.1:9099` and require the exact model alias, provider server, and runtime contract before creating any benchmark conversation. Stop if any identity differs.

- [ ] **Step 3: Run one disposable end-to-end smoke campaign**

```bash
confirm_smoke_root="$(mktemp -d /tmp/tldw-console-three-turn-confirmatory-smoke.XXXXXX)"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --candidate-sha eb8225a32f88ea43c337aff99804d360384e7668 \
  --iterations 1 \
  --burn-in-blocks 1 \
  --campaign-root "$confirm_smoke_root"
```

Expected: exactly 3 warmup, 3 burn-in, and 3 measured conversations complete with the required tool, ownership, listener, cleanup, and privacy contract. This smoke is never promoted and is not statistical evidence.

Exercise recovery and digest explicitly against the disposable campaign:

```bash
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --campaign-root "$confirm_smoke_root" \
  --campaign-action recover
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --campaign-root "$confirm_smoke_root" \
  --campaign-action digest \
  --attempt-id attempt-0001
```

Expected: `recover` reports no interrupted owner for a clean completed smoke. `digest` exits nonzero with the stable missing-reviewed-artifact error because a disposable smoke intentionally has no README or human report; this proves the five-file allowlist fails closed without contacting the provider. Task 7 alone prepares all five reviewed artifacts and expects digest success.

- [ ] **Step 4: Inspect the smoke result and correct harness defects test-first**

Validate the smoke raw rows, manifest, state history, sensitive-field scan, target cleanup, and terminal ownership. If the smoke exposes a harness defect, first add a failing focused test, apply the smallest fix, rerun Task 6, and commit that fix. Do not create an empty commit when no correction is needed.

### Task 7: Acquire, independently review, and publish the definitive confirmation

**Files:**
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/README.md`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.raw.jsonl`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.manifest.json`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn.summary.json`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/real-provider-three-turn-summary.md`
- Create: `Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/confirmatory-review-receipt.json`

- [ ] **Step 1: Acquire one official attempt**

```bash
confirm_campaign_root="$(mktemp -d /tmp/tldw-console-three-turn-confirmatory-task-20010.XXXXXX)"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --candidate-sha eb8225a32f88ea43c337aff99804d360384e7668 \
  --iterations 30 \
  --burn-in-blocks 5 \
  --campaign-root "$confirm_campaign_root"
```

Expected: 108 conversations and 540 provider calls: 3 warmups, 15 excluded burn-in conversations, and 90 measured conversations, with five provider calls per conversation. The attempt ends `complete_pending_review` regardless of the measured verdict.

- [ ] **Step 2: Independently recompute and inspect the retained evidence**

From raw JSONL, verify 108 terminal rows and 216 start-plus-terminal rows; the 3/15/90 phase split; exact global order, identity, and stored schedule positions before filtering; tool calls, ownership, listener identity, and cleanup; measured medians, p95s, bootstrap intervals, gates, and token totals through the digest-verified original module; and privacy, path, import, hash, and revision isolation. Burn-in is validation-only and must not appear in performance calculations.

- [ ] **Step 3: Prepare the five reviewed artifacts in attempt staging**

Ensure `README.md` records exact reproduction and verification commands. The human report must disclose the unavailable historical model-weight digest, exclude burn-in from performance claims, and preserve the computed verdict without reinterpretation. Generate the canonical artifact-set digest and a receipt template under `reviews/review-001.json`.

- [ ] **Step 4: Obtain an independent artifact-bound review**

The reviewer checks the raw acquisition and all derived outputs, then records either `approved` or `changes_required` against the exact attempt ID and artifact-set digest. If derived artifacts need correction, retain the acquisition and raw hash, regenerate only the affected derived artifacts, produce a new digest, and use a later review receipt. Only an uncorrectable acquisition/raw defect may lead to a new attempt.

- [ ] **Step 5: Register the approving receipt and promote once**

```bash
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --campaign-root "$confirm_campaign_root" \
  --campaign-action register-review \
  --attempt-id attempt-0001 \
  --review-receipt "$confirm_campaign_root/attempts/attempt-0001/reviews/review-001.json"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --campaign-root "$confirm_campaign_root" \
  --campaign-action promote \
  --attempt-id attempt-0001 \
  --review-receipt "$confirm_campaign_root/attempts/attempt-0001/reviews/review-001.json" \
  --destination Docs/superpowers/qa/console-three-turn-real-provider-confirmatory
```

If review required a corrected receipt, substitute its exact later path. Promotion must fail if the destination already exists or any source, copied artifact, digest, receipt, or attempt identity differs.

- [ ] **Step 6: Verify and commit the promoted evidence**

Re-run the privacy scan, artifact-set rehash, JSON/JSONL parsing, exact allowlist check, and receipt binding from the final directory.

```bash
git diff --check
git add Docs/superpowers/qa/console-three-turn-real-provider-confirmatory
git commit -m "perf(console): retain steady-state confirmation evidence"
```

### Task 8: Perform final regression, evidence, and Backlog closeout review

**Files:**
- Modify: `backlog/tasks/task-20010 - Confirm-steady-state-three-turn-Console-latency.md`
- Optionally modify only when a real reusable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the focused and full regression gates**

```bash
.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q
.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_change_review_finalization.py -q
.venv/bin/ruff check Tests/Performance/run_console_three_turn_profile.py Tests/Performance/test_console_three_turn_profile.py
.venv/bin/python -m py_compile Tests/Performance/run_console_three_turn_profile.py
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
git diff --name-only "$(git rev-parse refs/benchmarks/task-20010-implementation-base)"..HEAD | rg '^tldw_chatbook/'
```

Expected: all tests and static checks pass, and the final command prints nothing because this task changes no production module.

- [ ] **Step 2: Verify the definitive evidence package**

Confirm the exact published inventory, valid JSON/JSONL, the artifact-set digest, the approving receipt, the attempt lineage, raw hash preservation, sample/call counts, recalculated metrics, privacy scan, and immutable original-evidence hashes. Preserve canonical TASK-20009's historical inconclusive verdict and its evidence files' internal pre-integration TASK-19641 label.

- [ ] **Step 3: Request final code-and-evidence review**

Give the reviewer the spec, plan, task, harness/tests, campaign lineage, published evidence, and verification output. Correct derived-evidence findings against the same acquisition and obtain a newly bound approving receipt before republishing; correct code findings test-first and rerun all affected checks.

- [ ] **Step 4: Complete TASK-20010 hygiene only after every gate passes**

Check all eight acceptance criteria, add concise Implementation Notes with the commands and measured verdict, document the ADR-079 link/no-new-ADR decision, add a lessons entry only for a genuinely reusable incident, and set TASK-20010 to Done with Backlog CLI. Keep it In Progress if any review, evidence, test, documentation, or publication requirement remains.

- [ ] **Step 5: Commit closeout and verify a clean branch**

```bash
git add 'backlog/tasks/task-20010 - Confirm-steady-state-three-turn-Console-latency.md'
git add backlog/docs/lessons-testing-evidence.md backlog/docs/lessons-live-verification.md  # only if intentionally changed
git commit -m "docs(perf): close steady-state confirmation task"
git status --short
```

Expected: the branch is clean. Do not prune the retained evidence or rewrite the original benchmark history.
