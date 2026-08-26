# TASK-603 Cross-Platform Dictation Evidence Plan

> **Execution:** Follow `superpowers:executing-plans`, strict TDD, and the
> repository evidence lessons. Keep the implementation lean: this plan adds
> evidence plumbing only and does not change production dictation behavior.

**Goal:** Close TASK-603 AC6 with deterministic latency, backpressure,
cancellation, shutdown, batch-coexistence, limit, explicit-resume, and
unsupported-streaming contract evidence on the same five supported platform
lanes used by TASK-602, while retaining the existing macOS physical Mic smoke
as the real-device proof.

**Architecture:** Run an exact bounded pytest selection on Linux x86_64,
Linux arm64, Windows x86_64, macOS arm64, and macOS x86_64. Normalize each
lane's JUnit result into strict path-private JSON bound to the tested commit,
workflow run, platform identity, exact required node IDs, and successful pytest
outcome. Aggregate only individually valid passing lane documents. Hosted
runners prove deterministic control-plane behavior; they do not claim physical
microphone capture or repeat TASK-602's native ONNX model smoke.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already defines bounded buffer admission, dictation-next
ordering, non-preemption, cancellation, shutdown, and release gates. This work
records cross-platform evidence for that existing boundary.

## Global constraints

- Do not modify production dictation, executor, coordinator, Library, or UI
  behavior unless a genuine platform RED exposes a real defect and the user
  approves a reviewed corrective plan.
- Do not download or run Parakeet models in this workflow; TASK-602 already
  supplies the five-platform native ONNX evidence.
- Do not claim hosted audio hardware or microphone capture. Retain the existing
  macOS physical Mic smoke as that proof.
- Use exact test node IDs. A successful pytest process with a missing, skipped,
  duplicated, wrong-module, or failed required node is invalid evidence.
- Keep JSON recursively path-private. Permit only canonical GitHub run URLs and
  exact repository-relative test node IDs.
- A dependency, test, normalization, validation, upload, or aggregation failure
  keeps the lane/task red.
- Run only focused tests and static checks; no full repository suite.

## Task 1: Add a strict TASK-603 evidence normalizer

**Files:**

- Create: `.github/scripts/task603_dictation_evidence.py`
- Create: `Tests/CI/test_task603_dictation_evidence.py`

**Required contract nodes:**

- coordinator derived byte ceiling and exact-limit retention;
- one pending/coalesced dictation request with ordered boundaries;
- pending cancellation clears the gate without preempting active batch;
- delayed active batch does not block the dictation processing-thread join;
- Library heavy work is gated while light work remains dispatchable;
- Library terminal callback hands the executor to pending dictation first;
- shutdown cooperatively cancels dictation before executor close;
- Parakeet streaming factory reports unsupported through the existing fallback;
- mounted Console exposes the limit transition and explicit Mic resume;
- hands-free mode does not auto-reopen or auto-send after the limit.

**Steps:**

1. Write strict tests for CLI mode exclusivity, initialization/failure docs,
   JUnit exact-node validation, pytest outcome non-vacuity, numeric bounds,
   recursive privacy, per-lane validation, and five-lane aggregation.
2. Run the focused test file and capture genuine missing-script RED evidence.
3. Implement one flat standard-library normalizer with explicit schemas and no
   reusable framework or dependencies.
4. Mutation-check pytest-outcome enforcement, exact classname/name matching,
   and aggregate commit/run/platform equality; restore each mutation.
5. Run the full focused normalizer tests plus Ruff, format-check, `py_compile`,
   and `git diff --check`.
6. Commit: `test(stt): normalize task 603 platform evidence`.

## Task 2: Add the five-platform evidence workflow

**Files:**

- Create: `.github/workflows/task-603-platform-evidence.yml`
- Modify: `Tests/CI/test_task603_dictation_evidence.py`

**Steps:**

1. Add semantic workflow tests first: exact `workflow_dispatch` and labeled-PR
   trigger, contents-read permissions, five pinned runner lanes, Python 3.12,
   bounded timeout, exact pytest nodes, JUnit output, outcome forwarding,
   validate-always, and JSON-only upload-always.
2. Run the workflow-focused selection and capture the missing-workflow RED.
3. Add the minimal matrix workflow, following the proven TASK-601/602 failure
   initialization and normalization pattern.
4. Mutation-check that replacing the real pytest outcome with literal success
   makes the semantic test fail; restore it.
5. Run the full focused test file, Ruff, format-check, YAML parse, and diff-check.
6. Commit: `ci(stt): add task 603 platform evidence`.

## Task 3: Freeze, run, and review platform-contract evidence

**Files:** no planned source edits

**Steps:**

1. Rebase onto current `origin/dev` and rerun the exact local focused gate plus
   static checks.
2. Freeze the executable commit, push the branch, open a PR, and trigger a new
   TASK-603 workflow run for that exact SHA.
3. Wait for all five lanes. If any lane is red, download and validate its
   artifact, classify the exact failure, and stop before edits or aggregation.
4. For a genuine product or portability defect, use systematic debugging and
   TDD, obtain approval for any production-scope expansion, then trigger a
   brand-new five-lane run on the new frozen SHA.
5. Address verified PR review comments, resolve threads, and repeat focused
   verification after each fix.

## Task 4: Aggregate evidence and close TASK-603

**Files:**

- Modify: `Docs/STT_Evaluation/task-603/README.md`
- Create: `Docs/STT_Evaluation/task-603/platform-evidence.json`
- Modify: `backlog/tasks/task-603 - Restore-bounded-Parakeet-ONNX-dictation-buffers.md`

**Steps:**

1. Download each named lane artifact into separate scratch directories and
   validate each document independently.
2. Aggregate only through the normalizer and validate the aggregate again.
3. Update the README with exact frozen SHA, run URL/ID, five lane outcomes,
   exact local commands, retained macOS physical Mic evidence, limitations, and
   the distinction between deterministic hosted tests and native ONNX evidence.
4. Through Backlog CLI, check AC6, add concise implementation/evidence notes,
   and mark TASK-603 Done only after every DoD item is actually satisfied.
5. Verify the evidence-only boundary, JSON schema/privacy, Ruff, format-check,
   `py_compile`, diff-check, ancestry, task status, and clean worktree.
6. Commit: `docs(stt): record task 603 platform evidence`.
7. Push, address any final verified review comments, rebase if `origin/dev`
   moved, rerun affected exact gates, and merge only with a clean reviewed PR.

## Completion boundary

TASK-603 closes only when the same release-candidate commit has five valid
passing platform documents, the aggregate validates, the existing macOS Mic
smoke remains accurately qualified, all acceptance criteria and DoD items are
complete, and the PR review is clean. TASK-605 default promotion and legacy
provider removal remain separate work.
