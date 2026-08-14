# Windows audio.cpp Lifecycle and Clone Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Follow
> strict RED/GREEN TDD and stop at every review checkpoint.

**Goal:** Deliver truthful Guided audio.cpp and clone-reference parity on
Windows 10+ x86/x64 with Python 3.12+, including native path/ACL safety, exact
process-handle settlement, hosted Windows coverage, and a parameterized real
UAT handoff.

**Architecture:** Keep `AudioCppSupervisor`, the bounded package scanner,
`AudioCppGeneratedLaunchArtifact`, and `TTSCloneReferenceMaterializer` as the
only lifecycle owners. Add one TTS-scoped stdlib Win32 filesystem capability
under them, retain all blocking/native work through existing async ownership
patterns, and enable the existing Settings/Speech Lab UI only when the Windows
capability baseline is satisfied.

**Tech Stack:** Python 3.12+, `asyncio` Proactor subprocesses, stdlib `ctypes`
and `msvcrt`, Textual 8.x, pytest, GitHub Actions `windows-latest`, Windows
PowerShell 5.1+/PowerShell 7, Ruff, mypy

## Global constraints

- Supported Windows baseline: Windows 10+, x86 and x64, Python 3.12+.
- ARM64 is out of scope and must remain labeled unsupported.
- Do not add `pywin32`, another dependency, another process supervisor, a
  process-tree registry, a background retry loop, or a second configuration
  source.
- Do not use Job Objects or claim ownership of descendants/daemonizing builds.
- Do not weaken the existing POSIX descriptor-relative implementation.
- Do not broaden ADR-029's Windows privacy claim beyond generated audio.cpp
  launch artifacts and operation-scoped clone materializations.
- Every native call must be hidden behind stable path-free outcomes; no Win32
  error text, SID, username, executable/model/runtime path, transcript, WAV
  bytes, child output, or environment value may enter public errors/logs.
- All native work expected to exceed 100 ms runs off the Textual event loop.
- Cleanup authority is retained until exact handles/locks/transports settle;
  cancellation never abandons a thread or a returned owner.
- Normal CI is hermetic. It never downloads audio.cpp or a model package.
- The provisioned UAT takes parameters and contains no workstation-specific
  paths, usernames, versions, or device assumptions.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md` amendment

Reason: TASK-13208 establishes a verified Windows DACL posture for two private
plaintext artifact classes. ADR-023, ADR-050, and ADR-051 already govern the
process, generated-configuration, and clone owners; no new lifecycle boundary
is introduced.

---

### Task 1: Add the narrow native Windows artifact filesystem capability

**Files:**

- Create: `tldw_chatbook/TTS/windows_artifact_fs.py`
- Create: `Tests/TTS/test_windows_artifact_fs.py`
- Reference only: `tldw_chatbook/Notes/note_import_windows_fs.py`

**Public/internal interfaces:**

- `WindowsArtifactFilesystem` protocol for deterministic non-Windows tests.
- `NativeWindowsArtifactFilesystem` stdlib implementation selected only on
  native Windows.
- Opaque `WindowsPinnedHandle` and frozen `WindowsFileIdentity` values; no raw
  handle value appears in `repr`.
- Narrow operations: normalize an admitted drive/UNC path, open/create without
  following a reparse point, read/write through the retained handle, acquire or
  release an exact `LockFileEx` lock, install/verify the approved DACL, enumerate
  exact owned children, mark the exact handle for deletion, and close once.
- `windows_audio_cpp_platform_supported()` for Windows version, Python version,
  and process architecture gating.

- [ ] **Step 1: Write RED path and capability tests**

Cover absolute drive and UNC paths, Unicode, extended-length conversion,
drive-relative paths, device namespaces, reserved names, x86/x64 pointer
widths, Python <3.12, Windows <10, and ARM64. Assert unsupported tuples fail
closed without importing or calling Win32 on POSIX.

Run:

```bash
python -m pytest Tests/TTS/test_windows_artifact_fs.py -q
```

Expected RED: module/API missing.

- [ ] **Step 2: Write RED native handle and DACL tests**

Using an injectable fake kernel on every host and `pytest.mark.skipif` native
probes on Windows, cover:

- `CreateFileW` with `FILE_FLAG_OPEN_REPARSE_POINT` and directory backup
  semantics;
- non-inheritable handles;
- volume/file identity equality and substitution;
- reparse, junction, mount-point, special-file, and case-collision rejection;
- protected DACL installation and re-read verification for only current token
  user SID, LocalSystem, and Builtin Administrators;
- null/invalid DACL, unexpected allow trustee, and native failure;
- exclusive nonblocking `LockFileEx` ownership;
- exact handle-directed delete, idempotent close, retryable close failure, and
  control-flow preservation; and
- exception-graph/log canaries containing private paths, SIDs, and error text.

- [ ] **Step 3: Implement the smallest stdlib capability**

Use explicit `ctypes.WinDLL(..., use_last_error=True)` signatures with
pointer-sized `wintypes.HANDLE`. Validate ordinary absolute paths before adding
the `\\?\` or `\\?\UNC\` prefix. Keep all raw structures and codes private.
Return stable errors such as `unavailable`, `changed`, `privacy_unavailable`,
`busy`, and `cleanup_failed`.

- [ ] **Step 4: Verify GREEN on the host-independent matrix**

```bash
python -m pytest Tests/TTS/test_windows_artifact_fs.py -q
python -m ruff check tldw_chatbook/TTS/windows_artifact_fs.py Tests/TTS/test_windows_artifact_fs.py
python -m ruff format --check tldw_chatbook/TTS/windows_artifact_fs.py Tests/TTS/test_windows_artifact_fs.py
python -m mypy tldw_chatbook/TTS/windows_artifact_fs.py
```

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_chatbook/TTS/windows_artifact_fs.py Tests/TTS/test_windows_artifact_fs.py
git commit -m "feat(tts): add native Windows artifact handles"
```

---

### Task 2: Pin Windows package scans without changing recipe matching

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_package_scanner.py`
- Modify: `Tests/TTS/test_audio_cpp_package_scanner.py`
- Use: `tldw_chatbook/TTS/windows_artifact_fs.py`

**Interfaces:** Preserve `scan_audio_cpp_package_root()` and
`scan_audio_cpp_package_root_async()` signatures for callers. Add only one
optional internal filesystem seam for deterministic tests. Matching,
budgets, result types, and managed-root equality remain shared.

- [ ] **Step 1: Write RED Windows scanner regressions**

Cover drive/UNC/Unicode/long roots, disclosed top-level link review, nested
symlink/junction/reparse rejection, case-insensitive duplicate names,
directory substitution between enumeration phases, file substitution before
read, short reads, close failure, cancellation, and exact managed-root
revalidation. Prove one selected root is scanned and no drive/profile search
occurs.

Run:

```bash
python -m pytest Tests/TTS/test_audio_cpp_package_scanner.py -k windows -q
```

Expected RED: Windows falls into `NO_FOLLOW_UNAVAILABLE` or fails the new
native pin assertions.

- [ ] **Step 2: Add the Windows pin/read path**

On native Windows, retain a no-reparse directory handle while each `scandir`
iterator is live, compare pathname identity to the handle before and after
enumeration, and read allowlisted metadata through a pinned no-reparse file
handle. Keep the existing POSIX `O_NOFOLLOW` path byte-for-byte equivalent.

- [ ] **Step 3: Mutation-check the two ownership guards**

Temporarily remove (a) the directory identity recheck and (b) the file identity
recheck. Each dedicated race test must fail. Restore both guards.

- [ ] **Step 4: Verify scanner GREEN**

```bash
python -m pytest Tests/TTS/test_audio_cpp_package_scanner.py -q
python -m ruff check tldw_chatbook/TTS/audio_cpp_package_scanner.py Tests/TTS/test_audio_cpp_package_scanner.py
python -m mypy tldw_chatbook/TTS/audio_cpp_package_scanner.py
```

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_chatbook/TTS/audio_cpp_package_scanner.py Tests/TTS/test_audio_cpp_package_scanner.py
git commit -m "feat(tts): pin audio cpp scans on Windows"
```

---

### Task 3: Create and clean generated launch artifacts with verified DACLs

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_guided_launch.py`
- Modify: `Tests/TTS/test_audio_cpp_guided_launch.py`
- Use: `tldw_chatbook/TTS/windows_artifact_fs.py`

**Interfaces:** Keep `AudioCppGeneratedLaunchArtifact`,
`materialize_audio_cpp_guided_launch()`, and cleanup-owner transfer unchanged
for callers. The artifact privately retains either the current POSIX
descriptor owner or one Windows handle owner. Add a bounded read-only
`privacy_posture` projection that becomes Windows-account-protected only after
the DACL is installed and reverified.

- [ ] **Step 1: Write RED generated-artifact tests**

Cover Windows creation, canonical JSON bytes, protected DACL verification,
digest/size/identity validation, reparse and name substitution, unexpected
entries, partial creation, cancellation during creation, cleanup fail/retry,
managed-lease close failure, control flow, and no path/SID/native error leaks.

- [ ] **Step 2: Add the Windows storage branch**

Create `generation-<32 hex>/server.json` only beneath the app-owned generated
root. Install and verify the DACL before publication, write and flush through
the retained file handle, make the configuration read-only for the owning
lifecycle, and keep the exact directory/file handles and identities until
cleanup settles.

- [ ] **Step 3: Preserve exact cleanup semantics**

Delete only the recognized `server.json` and generation directory after exact
identity/DACL/name/entry checks. Unknown or replaced contents must remain.
Failed close/delete remains attached to the existing cleanup owner and is
retryable by the adapter/supervisor lifecycle.

- [ ] **Step 4: Verify both platform branches GREEN**

```bash
python -m pytest Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_windows_artifact_fs.py -q
python -m ruff check tldw_chatbook/TTS/audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_guided_launch.py
python -m mypy tldw_chatbook/TTS/audio_cpp_guided_launch.py
```

- [ ] **Step 5: Commit Task 3**

```bash
git add tldw_chatbook/TTS/audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_guided_launch.py
git commit -m "feat(tts): protect Windows guided launch artifacts"
```

---

### Task 4: Materialize clone references under the same native owner

**Files:**

- Modify: `tldw_chatbook/TTS/profile_reference_materialization.py`
- Modify: `Tests/TTS/test_profile_reference_materialization.py`
- Use: `tldw_chatbook/TTS/windows_artifact_fs.py`

**Interfaces:** Preserve `TTSCloneReferenceMaterializer`,
`TTSCloneReferenceMaterialization`, `voice_ref`, `validated_voice_ref()`, and
`aclose()`. Internally admit a Windows materialization record alongside the
existing POSIX record; do not create a second materializer.

- [ ] **Step 1: Replace the Windows module skip with RED Windows contracts**

Keep POSIX `fcntl` tests capability-gated, then add host-independent Windows
record tests plus native Windows probes for protected root/owner/lock/asset
DACLs, `LockFileEx`, exact identity validation, one active owner, orphan sweep,
reparse/unknown-entry preservation, cancellation, shutdown, close retry, and
WAV/transcript privacy.

- [ ] **Step 2: Implement Windows create/validate/cleanup dispatch**

Reuse the existing async retained-worker sets and locks. The synchronous
Windows branch must create the recognized names, hold an exact nonblocking
owner lock, flush the WAV, validate every retained handle and DACL, and delete
only exact recognized entries. Do not use POSIX mode bits on Windows.

- [ ] **Step 3: Mutation-check the live lock and identity guards**

Removing the `LockFileEx` ownership check or accepting a replaced asset must
fail its regression. Restore both guards before continuing.

- [ ] **Step 4: Verify materialization GREEN**

```bash
python -m pytest Tests/TTS/test_profile_reference_materialization.py -q
python -m ruff check tldw_chatbook/TTS/profile_reference_materialization.py Tests/TTS/test_profile_reference_materialization.py
python -m mypy tldw_chatbook/TTS/profile_reference_materialization.py
```

- [ ] **Step 5: Commit Task 4**

```bash
git add tldw_chatbook/TTS/profile_reference_materialization.py Tests/TTS/test_profile_reference_materialization.py
git commit -m "feat(tts): materialize clone references on Windows"
```

---

### Task 5: Admit Windows binaries and select only evidenced backends

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_guided_launch.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_managed_config.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_recipes.py`
- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py`
- Modify: `Tests/TTS/test_audio_cpp_guided_launch.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_config.py`
- Modify: `Tests/TTS/test_audio_cpp_recipes.py`
- Modify: `Tests/UI/test_settings_audio_cpp_experience_model.py`

**Interfaces:** Keep manual selection and `detect_audio_cpp_server_binary()`.
Detection adds Windows `PATHEXT` behavior for `audiocpp_server.exe`; it does
not execute or recursively search. Normalize Windows process architectures to
`x86` or `x86_64`. Add Expected CPU evidence for both exact Windows tuples;
only provisioned evidence may become Verified.

- [ ] **Step 1: Write RED binary/path tests**

Cover PATH/PATHEXT discovery, manual `.exe`, drive/UNC/Unicode/long paths,
relative and drive-relative rejection, directory/reparse/device-namespace
rejection, non-PE files, x86/x64 PE machine evidence, and no execution or
environment mutation during detection/Save.

- [ ] **Step 2: Write RED backend tests**

Cover Windows x86/x64 CPU Auto/explicit selection, unsupported ARM64, exact
accelerated evidence, ambiguous device evidence, and no projection of x64
evidence onto x86. Preserve macOS/Linux ordering.

- [ ] **Step 3: Implement side-effect-free Windows admission**

Use the native no-reparse handle for validation and a bounded PE header read
for machine type. Add `windows` to the existing backend selector and retain
the current recipe intersection semantics. Do not start the binary or probe a
device during Save.

- [ ] **Step 4: Verify admission/backend GREEN**

```bash
python -m pytest Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_managed_config.py Tests/TTS/test_audio_cpp_recipes.py Tests/UI/test_settings_audio_cpp_experience_model.py -q
python -m ruff check tldw_chatbook/TTS/audio_cpp_guided_launch.py tldw_chatbook/TTS/audio_cpp_managed_config.py tldw_chatbook/TTS/audio_cpp_recipes.py tldw_chatbook/UI/Screens/settings_speech_tts.py
python -m mypy tldw_chatbook/TTS/audio_cpp_guided_launch.py tldw_chatbook/TTS/audio_cpp_managed_config.py tldw_chatbook/TTS/audio_cpp_recipes.py
```

- [ ] **Step 5: Commit Task 5**

```bash
git add tldw_chatbook/TTS/audio_cpp_guided_launch.py tldw_chatbook/TTS/audio_cpp_managed_config.py tldw_chatbook/TTS/audio_cpp_recipes.py tldw_chatbook/UI/Screens/settings_speech_tts.py Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_managed_config.py Tests/TTS/test_audio_cpp_recipes.py Tests/UI/test_settings_audio_cpp_experience_model.py
git commit -m "feat(tts): admit evidenced Windows audio cpp runtimes"
```

---

### Task 6: Settle the exact Windows process handle under the existing supervisor

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py`
- Modify: `Tests/TTS/test_audio_cpp_supervisor.py`
- Modify: `Tests/TTS/fixtures/fake_audiocpp_server.py`

**Interfaces:** Extend private `_OwnedAudioCppProcess` with one idempotent
`close_native_transport` callback. `AudioCppSupervisor` remains the sole
owner. Extend `AudioCppProcessSnapshot` only with the bounded generated
artifact privacy posture so Speech Lab can distinguish planned protection from
an actually verified live artifact.

- [ ] **Step 1: Write RED ownership/race tests**

Cover cancellation and timeout while `create_subprocess_exec` is returning,
start/stop, restart, crash, close-during-start, close-during-stop, repeated
cancellation, inherited pipes, graceful timeout then force terminate, process
wait failure, transport-close failure/retry, and one shared outer deadline.
Assert no second waiter/reaper and no descendant/process-tree operation.

- [ ] **Step 2: Retain spawn through ownership publication**

Create one explicit launcher task and settle it through caller cancellation or
timeout. If the process appears after the lifecycle epoch changed, immediately
publish an internal generation owner, terminate/join it, close its pipes and
transport, then re-raise the original cancellation. A returned process may
never live only inside an abandoned future.

- [ ] **Step 3: Close the exact transport last**

After the exact child wait, output drains, hooks/client cleanup, generated
artifact, clone materializations, and retained tasks settle, invoke the owned
transport close exactly once. A failure keeps the generation and cleanup
failure visible so `wait_closed()` cannot claim success and a later close may
retry.

- [ ] **Step 4: Add native Windows helper-process proof**

On Windows 3.12, launch the existing short-lived fixture without a shell,
inspect the exact native process handle in the test, terminate/wait when
needed, close the owned transport, and prove the handle is invalid afterward.
Prove an independently spawned descendant/sibling is never terminated or
claimed.

- [ ] **Step 5: Mutation-check spawn retention and close ordering**

Removing spawn settlement must fail the cancellation race. Moving transport
close before child wait/artifact cleanup must fail the ordering test. Restore
both.

- [ ] **Step 6: Verify supervisor GREEN**

```bash
python -m pytest Tests/TTS/test_audio_cpp_supervisor.py -q
python -m ruff check tldw_chatbook/TTS/audio_cpp_supervisor.py Tests/TTS/test_audio_cpp_supervisor.py Tests/TTS/fixtures/fake_audiocpp_server.py
python -m mypy tldw_chatbook/TTS/audio_cpp_supervisor.py
```

- [ ] **Step 7: Commit Task 6**

```bash
git add tldw_chatbook/TTS/audio_cpp_supervisor.py Tests/TTS/test_audio_cpp_supervisor.py Tests/TTS/fixtures/fake_audiocpp_server.py
git commit -m "feat(tts): settle owned audio cpp processes on Windows"
```

---

### Task 7: Enable truthful Windows Settings and Speech Lab parity

**Files:**

- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify: `tldw_chatbook/UI/Speech/speech_clone_setup.py`
- Modify: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py` only if the
  existing projection lacks the required bounded posture row
- Modify: `Tests/UI/test_settings_speech_tts_panel.py`
- Modify: `Tests/UI/test_speech_playground_pane.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`

**Interfaces:** Replace `_AUDIO_CPP_MANAGED_UI_SUPPORTED = os.name != "nt"`
with the shared capability predicate. Preserve all existing IDs, bindings,
saved/applied/process projections, lifecycle actions, focus locators, and
result ownership.

- [ ] **Step 1: Write RED mounted Settings tests**

On a supported Windows capability, Managed/Guided controls are available,
PATH/manual selection works, Save remains side-effect free, invalid capability
states explain why Managed is unavailable, and the privacy row says account
protected / administrators and system retain access / plaintext / not
encryption. Before launch it must say protection will be applied; only the
verified live process snapshot may say it was applied. Test keyboard traversal
and focus through scan/Save recompose.

- [ ] **Step 2: Write RED mounted Speech Lab tests**

Cover Start/Test/Restart/Shutdown, text WAV, clone WAV, last-valid-sample
retention, profile Save, cancellation, crash, retry, app close, and focus at
80x24. Assert saved/applied/process truth remains independent and every
Windows failure uses bounded recovery copy.

- [ ] **Step 3: Enable the existing surfaces**

Use only the shared capability predicate and existing lifecycle events. Update
clone privacy copy to distinguish POSIX mode posture from verified Windows
DACL posture without promising exclusivity from administrators or encryption.
Do not add controls, settings, bindings, or a Windows-only workflow.

- [ ] **Step 4: Verify UI GREEN**

```bash
python -m pytest Tests/UI/test_settings_speech_tts_panel.py Tests/UI/test_speech_playground_pane.py Tests/UI/test_speech_playground_pane_lifecycle.py Tests/TTS/test_stts_audio_cpp_generation.py -q
python -m ruff check tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tldw_chatbook/UI/Speech/speech_clone_setup.py tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py
```

- [ ] **Step 5: Commit Task 7**

```bash
git add tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tldw_chatbook/UI/Speech/speech_clone_setup.py tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py Tests/UI/test_settings_speech_tts_panel.py Tests/UI/test_speech_playground_pane.py Tests/UI/test_speech_playground_pane_lifecycle.py Tests/TTS/test_stts_audio_cpp_generation.py
git commit -m "feat(ui): enable guided audio cpp on Windows"
```

---

### Task 8: Add a hermetic Windows 3.12 CI gate

**Files:**

- Modify: `.github/workflows/test.yml`
- Modify: `Tests/CI/test_github_actions_test_workflow.py`

**Interfaces:** Add one PR-required `windows-latest` / Python 3.12 job matrix
for `x86` and `x64`, running only the Windows capability, scanner, artifact,
clone, supervisor, recipe, and focused mounted parity tests. Add its stable
gate to `test-summary`.

- [ ] **Step 1: Write RED workflow-shape assertions**

Assert the job is on `windows-latest`, uses Python 3.12 with setup-python's
`x86` and `x64` architectures, installs only project and test requirements,
runs the exact owned test list with a timeout, does not download
audio.cpp/models, and is required by the summary gate.

- [ ] **Step 2: Add the focused Windows job**

Use `shell: pwsh` where quoting differs. Keep fixtures local and use the Python
helper process for subprocess tests. Do not place provisioned UAT secrets or
paths in Actions.

- [ ] **Step 3: Verify workflow GREEN**

```bash
python -m pytest Tests/CI/test_github_actions_test_workflow.py -q --confcutdir=Tests/CI
python -m ruff check Tests/CI/test_github_actions_test_workflow.py
```

- [ ] **Step 4: Commit Task 8**

```bash
git add .github/workflows/test.yml Tests/CI/test_github_actions_test_workflow.py
git commit -m "ci(tts): gate Windows audio cpp lifecycle"
```

---

### Task 9: Provide the parameterized one-command Windows UAT

**Files:**

- Create: `scripts/uat_audio_cpp_windows.ps1`
- Create: `scripts/uat_audio_cpp_windows.py`
- Create: `Tests/TTS/test_audio_cpp_windows_uat_harness.py`
- Create: `Docs/superpowers/qa/audio-cpp-windows-2026-08-14/README.md`

**Interfaces:** The PowerShell entry point accepts user-provisioned server,
text package, clone package/reference, and exact expected identity parameters.
It detects and records the exact x86/x64 tuple, creates disposable
config/data/model/runtime roots, invokes the Python harness, and requires an
explicit human audible confirmation. No parameter default identifies a
particular machine.

- [ ] **Step 1: Write RED harness tests**

Cover required parameters, Windows/Python/architecture rejection, root
isolation, environment restoration, no secret/path output, exact identity
comparison, structural WAV validation, failed audible confirmation, cleanup
failure, and stable JSON evidence. Use fixture binaries/packages only.

- [ ] **Step 2: Implement the PowerShell wrapper**

The wrapper remains compatible with Windows PowerShell 5.1 and PowerShell 7.
It validates only parameter presence, creates one disposable root, sets
process-local Chatbook environment variables, invokes the checked-in Python
harness, asks the operator to confirm playback, and always runs cleanup. It
emits a nonzero exit code for skipped, partial, failed, inaudible, or dirty
teardown outcomes.

- [ ] **Step 3: Implement the production-path Python journey**

Exercise:

1. explicit PATH/manual binary review without launch;
2. one local package and one Model Library exact-root return;
3. Guided Save with zero process creation;
4. generated JSON, health, catalog, and exact model identities;
5. one text WAV and one clone-reference WAV;
6. restart, cancellation, forced crash, and recovery;
7. exact runtime lease/removal blocking while live;
8. final app shutdown; and
9. no owned process/handle/task/client/endpoint/generated artifact/clone
   materialization remaining.

The Python harness returns sanitized structural evidence; the PowerShell layer
records only `pass`, `fail`, `partial`, or `inaudible` plus bounded identities.

- [ ] **Step 4: Verify the hermetic harness GREEN**

```bash
python -m pytest Tests/TTS/test_audio_cpp_windows_uat_harness.py -q
python -m ruff check scripts/uat_audio_cpp_windows.py Tests/TTS/test_audio_cpp_windows_uat_harness.py
python -m mypy scripts/uat_audio_cpp_windows.py
```

- [ ] **Step 5: Run the provisioned Windows UAT**

From each claimed Windows architecture checkout, run the single parameterized
command documented in the QA README. Do not copy machine-specific arguments
into the repository. Require objective pass evidence plus the operator's
explicit audible confirmation. If either x86 or x64 lacks a compatible server,
or any journey step is unavailable, record that tuple as PARTIAL and keep
TASK-13208 In Progress rather than projecting evidence across architectures.

- [ ] **Step 6: Commit Task 9**

```bash
git add scripts/uat_audio_cpp_windows.ps1 scripts/uat_audio_cpp_windows.py Tests/TTS/test_audio_cpp_windows_uat_harness.py Docs/superpowers/qa/audio-cpp-windows-2026-08-14/README.md
git commit -m "test(tts): add provisioned Windows audio cpp UAT"
```

---

### Task 10: Run the release matrix and close TASK-13208 truthfully

**Files:**

- Modify: `backlog/tasks/task-13208 - Add-Windows-parity-for-guided-audio.cpp-lifecycle-and-cloning.md`
- Modify: `Docs/superpowers/qa/audio-cpp-windows-2026-08-14/README.md`
- Modify: `backlog/docs/lessons-live-verification.md` only if this task
  reproduces a new generalizable incident not already documented

- [ ] **Step 1: Run the focused cross-platform matrix**

```bash
python -m pytest \
  Tests/TTS/test_windows_artifact_fs.py \
  Tests/TTS/test_audio_cpp_package_scanner.py \
  Tests/TTS/test_audio_cpp_guided_launch.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_recipes.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_settings_speech_tts_panel.py \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/CI/test_github_actions_test_workflow.py -q
```

- [ ] **Step 2: Run full static and formatting gates**

```bash
python -m ruff check <all changed Python files>
python -m ruff format --check <all changed Python files>
python -m mypy \
  tldw_chatbook/TTS/windows_artifact_fs.py \
  tldw_chatbook/TTS/audio_cpp_package_scanner.py \
  tldw_chatbook/TTS/audio_cpp_guided_launch.py \
  tldw_chatbook/TTS/profile_reference_materialization.py \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  tldw_chatbook/TTS/audio_cpp_recipes.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py
git diff --check
```

- [ ] **Step 3: Verify hosted Windows and provisioned UAT evidence**

Require both Windows 3.12 CI architectures to pass. Record the checked-out
commit, each exact architecture tuple, objective UAT outcomes, clean-root
proof, and human audible confirmation without storing private arguments or
artifacts. Do not close the x86/x64 support claim with evidence from only one
tuple.

- [ ] **Step 4: Self-review ownership and privacy**

Walk every native handle, lock, retained task, cleanup carrier, process
transport, generated artifact, and materialization from acquisition through
success, ordinary failure, control flow, cancellation, unmount, and app
shutdown. Search the complete exception/log graph with private canaries.

- [ ] **Step 5: Update Backlog truth**

Only after every automated gate and the provisioned UAT pass:

- check all seven acceptance criteria;
- add concise Implementation Notes, including the ADR-029 amendment and any
  plan deviations;
- set TASK-13208 to Done; and
- add a lesson only if a genuinely new recurring trap was reproduced.

If UAT is partial, keep the task In Progress and document the exact bounded
release gate instead of claiming support.

- [ ] **Step 6: Commit closeout**

```bash
git add 'backlog/tasks/task-13208 - Add-Windows-parity-for-guided-audio.cpp-lifecycle-and-cloning.md' Docs/superpowers/qa/audio-cpp-windows-2026-08-14/README.md
git commit -m "docs(tts): close Windows audio cpp parity"
```
