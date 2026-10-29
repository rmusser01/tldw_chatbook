# Windows recovery regression fixes

> **For agentic workers:** Use superpowers:executing-plans for the coordinator and superpowers:dispatching-parallel-agents for independent failure domains. Track steps below.

**Goal:** Resolve the Windows/helper failures recorded by TASK-37205 before opening the portable-bundle PR.

**Spec:** TASK-37269 acceptance criteria and the existing [portable relocation contract](../specs/2026-10-28-portable-profile-relocation-design.md).

**Architecture:** Preserve installed helper integrity, native no-follow/private metadata and committed-profile authority. Correct demonstrated runtime defects at their source; correct fixtures where they assume POSIX metadata, privileged symlinks, obsolete schemas or a different Windows home. Existing failing cases provide the initial regressions.

**Tech stack:** Python 3.12, pytest, native Windows filesystem wrappers, existing package-owned age worker.

ADR required: no
ADR paths: existing backlog/decisions/125-lock-safe-private-sqlite-validation.md, 126-complete-local-backup-and-recovery.md, 127-fresh-install-private-data-root-recovery.md and 256-detached-portable-profile-import.md.
Reason: restore established behavior and qualify its Windows tests; introduce no new runtime/security boundary. Reassess if investigation requires one.

## Constraints and review focus

- Reuse this managed worktree and its private test environment; retain the existing ZIP/evidence.
- Targeted tests only; repository policy requires explicit opt-in for a full sweep.
- Do not skip safety cases to obtain green results or relax admission, helper integrity, activation, alias or metadata checks.
- Verify actual isolated child dependencies/environment and package-worker bytes before changing helper code.
- Exercise real unprivileged NTFS aliases; verify original target bytes survive refusals.
- Keep the shipped old-schema fixture historical while comparing a migrated candidate with the installed schema.
- Bind Windows HOME/USERPROFILE/XDG/bootstrap selectors consistently in fresh children; keep hostile provider/app override checks.
- Requalify the installed portable runtime if any product code changes, then review before publishing the PR.

## Tasks

### 1. Helper qualification

Files: `Backup_Recovery/crypto.py`, `age_worker.py` only if necessary; `Tests/Backup_Recovery/test_crypto.py`, helper fixture if demonstrated.

- [x] Reproduce `helper_unavailable` in encrypted archive/isolated cases and trace the isolated child failure.
- [x] Apply the smallest verified correction, retaining protocol/hash/authenticated-EOF and fixed diagnostic behavior.
- [x] Run helper tests and encrypted caller regressions; verify wrong-password, tamper, bounds and cancellation behavior.

### 2. Windows archive/planner fixtures

Files: `Tests/Backup_Recovery/test_archive_reader.py`, `test_restore_plan.py`, a shared native test alias helper only if reused.

- [x] Reproduce POSIX-mode, symlink privilege and schema-77 assumptions.
- [x] Use the installed native metadata view, actual Windows directory reparse aliases and installed target schema without weakening refusal assertions.
- [x] Run the complete affected files, including unchanged-source and no-live-destination controls.

### 3. Fresh isolated startup

Files: `Tests/Backup_Recovery/test_isolated_restore.py`; shared fresh-child helper only if the root cause requires it.

- [x] Reproduce the fresh child activation failure under private native directories.
- [x] Trace all bootstrap/home/environment selectors and test the minimal correction.
- [x] Exercise the directory-alias refusal with a real unprivileged NTFS alias and run isolated restore tests with the corrected helper.

### 4. Integration and PR

- [x] Run the affected recovery files and portable cohort together, then scoped lint/format/manifest/whitespace checks.
- [x] Rebuild/requalify the portable artifact if runtime bytes change; preserve old proof as historical evidence.
- [x] Review the combined diff, record exact results and update TASK-37269/37205.
- [ ] Create a `codex/` branch, commit and open the portable-bundle PR against the repository's verified base; attach it to this chat.

## Execution ledger

- User requested these fixes and subsequent portable PR on 2026-10-28. Continue the established execution workflow without redundant implementation-method approval.
- Initial inventory: ten recorded failures across archive reader, restore planner and isolated restore. The private Python environment now contains Cryptodome 3.24.1 under `-I`; output has a native owner/SYSTEM/Administrators-only ACL. Reproduce current results before assuming the earlier helper failure persists.
- Root trace: inherited global temporary storage had a shared ACL. Preview SQLite capacity qualification correctly refused it before domain validation. The recovery test fixture now creates a native private temporary directory and binds TMPDIR/TEMP/TMP plus the cached tempfile selector; no runtime gate changed.
- Fresh-child RED then GREEN: Windows Path.home used inherited USERPROFILE while HOME pointed at the fixture, causing isolated_activation_required. Both selectors now point at the same fixture. Windows junction refusal occurs earlier as native ELOOP/windows_reparse_point_refused; the test asserts that exact error and retained target bytes. Two affected cases pass in 65.53s.
- Cold help diagnostic: a delayed child stack showed ordinary app imports blocked in importlib bytecode writes inside the disposable source copy. The fixture's two fresh interpreters use -B; original 30/45-second child deadlines remain unchanged. No production launch or admission bypass was introduced.
- Independent whole-branch review: no Critical, Important or Minor findings. Its deferred native-artifact claims are covered by the recorded exact-ZIP proof; clean-machine/cross-account/non-NTFS/all-optional behavior remains explicitly unqualified. Exhaustive fault injection at every OS write boundary is not claimed; targeted sealed-source, resume and publication-barrier cases remain required. Fresh-start checks passed after dispatch; combined cohort remains the final test gate.
- First combined run: 325 passed, one failed and 35 later setup errors in 441.96s. Native inspection found the task worktree ancestor had become shared-writable (0766) while root-file edits were made during verification; output and test directories remained private. The source and subsequent private-temp fixture correctly refused that ancestor. Ruling: finish edits first, restore only this managed worktree root to native owner/SYSTEM/Administrators privacy, then run a frozen-tree verification. Do not relax guards or report the interrupted run as green.
- Git checkout policy: the exact native patch bytes are now pinned to LF in .gitattributes, matching its existing provenance SHA-256 cd9196d6b7df2e620d906d9e748654ddef604b28672f5f72fcb7b4fc5112669b.


### Helper qualification evidence (2026-10-28)

- Saved `output/recovery-regressions.log` (12:55) and `output/recovery-isolated-regressions.log` (13:02) show fixed `helper_unavailable` failures before `output/portable-test-dependencies.log` (13:28) installed the declared `pycryptodomex` 3.24.1 dependency into the private test interpreter. The earlier test setup grafted dependencies through `PYTHONPATH`; the intentionally isolated helper ignores it.
- Controlled reproduction in disposable `output/helper-isolation-control-20261028-01`: the same package worker in a dependency-free interpreter with only `SYSTEMROOT` and a dependency `PYTHONPATH` returns protocol 2/age-v1 with rc 0 without isolation, then rc 1, empty stdout and fixed `transform_failed` stderr under `-I`. The fully provisioned interpreter returns the expected isolated info, no stderr and `(True, "available")`; package worker SHA-256 equals its existing anchor.
- No changes to `crypto.py` or `age_worker.py` were required. `test_crypto.py` now exercises real unprivileged NTFS directory junctions and a real Windows named pipe, native privacy metadata, platform text newlines, actual runtime-descendant memory, and a real file-sharing cleanup denial. POSIX symlink/FIFO/permission cases remain. Alias targets survive unchanged; no refusal or cryptographic safety case is skipped.
- Initial helper file plus actual-profile encrypted case: **9 failed, 34 passed in 68.23 s** (`output/helper-fixes-full-red-20261028-01.log`): eight Windows fixture assumptions and the actual-profile case's separately diagnosed shared-TEMP refusal. The runtime helper was already available.
- Encrypted archive quota, retained ciphertext acquisition and streamed ciphertext digest regressions: **3 passed in 19.66 s** before helper test edits.
- Native refusal/diagnostics/cleanup fixture subset: **18 passed in 5.90 s**.
- Complete helper file plus `test_actual_bound_profile_stages_new_isolated_config_without_source_changes`, with a private native `TEMP`/`TMP`/`TMPDIR` selected before launch: **43 passed in 121.68 s** (`output/helper-fixes-full-green-20261028-01.log`). Covers wrong password, truncated/tampered ciphertext, integrity/protocol rejection, byte and memory bounds, secret-free environment/diagnostics, cancellation, crash and queued-job cleanup.
- Final alias/source preservation assertions: **14 passed in 2.33 s**. Scoped Ruff lint, Ruff format check and `git diff --check` pass for the owned helper test file. Only targeted tests were run.
- Test invocation: private `output/portable-build-env/Scripts/python.exe`, source `PYTHONPATH`, `-o addopts='' -o asyncio_default_fixture_loop_scope=function --confcutdir=Tests/Backup_Recovery -p no:cacheprovider --basetemp=output/helper-fixes-<unique> --tb=short`.

- Final helper coverage correction: the former byte-budget test reduced `_MAX_CONTAINER` while leaving already-bound default parameters at 2 TiB, so it reached `invalid_budget` before streaming. Separate explicit zero-budget checks now retain that refusal; two 2000-byte stream cases independently constrain input or output to 1000 bytes (the other limit is 4096), require `transform_failed`, and require no published output or temporary plaintext. A subprocess-only in-memory mutation replacing `_pipes` input/output limits with the maximum made both stream cases fail `DID NOT RAISE` (**2 failed in 10.97 s**, `output/helper-fixes-budgets-mutation-red-20261028-01.log`); no product file was changed for the mutation.
- Final complete `test_crypto.py` census, including the two added actual streaming cases: **44 passed in 68.31 s** (`output/helper-fixes-final-green-20261028-01.log`). Scoped Ruff lint/format and whitespace checks pass after the final test edits. The earlier 43-test pass still records the complete then-current 42-case helper file plus the actual-profile caller; the final helper census is 44.

- Frozen combined verification: 361 passed, one intentional duplicate-ZIP warning in1210.67s. Packaging/native-builder cohort:18 passed plus5 subtests. Native ancestor mode remained0700; sandbox-owned new source files were explicitly made readable by the workspace owner before this successful run. Runtime guards were never relaxed.
- Canonical ADR numbering: dev already contains252/253, so this feature's provisional252/253/254 decisions became254/255/256 with references updated. Content/approval unchanged; the pinned Rust patch has no ADR-number reference and its hash is unchanged.
- Integration ruling: rebase this feature onto current dev before publishing. Upstream changes no recovery,DB,configuration or dependency files; re-run fresh CLI/profile checks after rebase. Existing ZIP remains the historical exact798f8fca75+portable qualification artifact; test-only fixes and ADR docstring renumbering do not require a new distribution build. No new-tip ZIP or clean-machine qualification is claimed.

- Dev integration: rebased onto3efe8476ce; resolved the single lesson conflict by retaining both entries. Recovery/DB/config/dependency/packaging/test bytes were identical to tested bc612f30f0 across the rebase. Native patch LF/hash and packaging18+5 subtests passed again.
- Post-rebase functional fixture correction: the old30-second cold-help and45-second restore budgets could still expire during ordinary native work. The test now runs one actual cold CLI (90s bound), selects/admit-reads through the public packaged-probe path in the parent (150s bound), and gives native seed/restore120s. This removes duplicate TUI imports without a startup SLA claim. Full isolated+model-import file20 passed in178.14s.
- CLI routing coverage retained explicitly: private sitecustomize captures expected selectors before the parser consumes argv and verifies the child selected that exact profile/control tuple. A no-op child that consumes argv returned0 with the old observer (RED), then74 with the fixed observer (GREEN). The final genuine CLI plus real Notes/chat/media/client-identity assertions passed in64.13s. No runtime permission, admission or crypto code changed.
- Final review gate satisfied: independent review had no actionable findings; remaining native/qualification limits stand as documented. Full suite was not requested/run. Existing binary qualification remains historical; no current-tip GUI rebuild claim.

- Publication gate: automatic approval review rejected the combined commit/push request because it requires explicit human authorization to publish the source/task changes to github.com/rmusser01/tldw_chatbook. Local commit and a concrete PR draft remain permitted; no alternate push path is attempted.
