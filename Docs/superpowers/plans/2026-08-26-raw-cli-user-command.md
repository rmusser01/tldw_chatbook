# Raw CLI Executor and Console User Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the false-by-default, per-launch-armed raw one-shot shell executor and the direct Console `! ` user command without making a provider request or adding command output to model context.

**Architecture:** A synchronous, UI-free `RawShellExecutor` owns validation, fixed shell argv, scrubbed environment, admitted worker containment, bounded streaming, timeout, and cleanup. An app-owned `RawCliRuntime` owns the process-memory arm bit and active invocation registry. A wired `ConsoleRawCliController` under `UI/Console_Modules/` tracks submission/projection work without growing `chat_screen.py`; composer state retains physically typed prefix provenance. The controller starts a non-exclusive worker, projects events into a display-only TOOL marker, and writes a `local_command` AgentRunsDB/run-log record. Settings owns the persistent unlock and immediate Arm/Disarm controls. Raw CLI remains full host-user authority; workspace selection is only an initial directory.

**Tech Stack:** Python 3.11, Textual 8, `multiprocessing`/`subprocess`/`threading` from the standard library, existing `ExecutorProcessTree`, `AgentRunsDB`, `RunLogWriter`, Console store/transcript widgets, pytest.

**Backlog task:** `TASK-18926`

**ADR required:** yes

**ADR path:** `backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md`

**Reason:** ADR-093 authorizes the host-authority shell boundary, partially supersedes ADR-033, and fixes the two-gate, no-sandbox, process-ownership, and persistence contracts used here.

---

## Task 1: Pin raw policy, request, result, and configuration contracts

**Files:**

- Create: `tldw_chatbook/Tools/raw_cli_executor.py`
- Modify: `tldw_chatbook/config.py`
- Create: `Tests/Tools/test_raw_cli_executor_contract.py`
- Modify: `Tests/test_config_console_defaults.py`

- [ ] **Step 1: Write failing contract tests**

Cover exact types and limits before process work:

```python
def test_raw_request_rejects_nul_and_more_than_16_kib(tmp_path): ...
def test_raw_request_requires_existing_absolute_initial_directory(tmp_path): ...
def test_timeout_may_lower_but_not_exceed_300_seconds(tmp_path): ...
def test_build_scrubbed_environment_starts_empty(monkeypatch): ...
def test_bash_argv_disables_profiles(): ...
def test_powershell_argv_disables_profiles(): ...
def test_cmd_argv_disables_autorun(): ...
def test_console_raw_cli_permitted_defaults_false(): ...
```

Pin the public vocabulary rather than implementation details:

```python
RawCliCaller = Literal["user", "model"]
RawCliShell = Literal["auto", "bash", "powershell", "cmd"]
RawCliTerminalState = Literal[
    "refused", "shell_unavailable", "spawn_failed", "containment_unavailable",
    "exited", "timed_out", "cancelled", "cleanup_unproven",
]

@dataclass(frozen=True, slots=True)
class RawCliRequest:
    invocation_id: str
    caller: RawCliCaller
    command: str
    shell: RawCliShell
    initial_directory: Path
    timeout_seconds: float
    console_session_id: str
    transcript_anchor_id: str | None
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing behavior**

Run:

```bash
pytest -q Tests/Tools/test_raw_cli_executor_contract.py Tests/test_config_console_defaults.py -k raw_cli
```

Expected: FAIL because the raw contracts and config key do not exist.

- [ ] **Step 3: Implement immutable contracts and boundary validation**

In `raw_cli_executor.py`, add only the constants and value objects needed by all callers:

- `MAX_RAW_COMMAND_BYTES = 16 * 1024`
- `MAX_RAW_TIMEOUT_SECONDS = 300.0`
- `MAX_RAW_PREVIEW_BYTES = 32 * 1024`
- `RawCliStreamEvent(stream, text, total_bytes, truncated)` with `stream` limited to `stdout`/`stderr`
- `RawCliResult` carrying caller, resolved shell, directory, elapsed, previews, bounded `record_output`, exit code, terminal state, `truncated`, and `cleanup_proven`
- `validate_raw_cli_request()` rejecting empty/whitespace-only commands, NUL, oversized UTF-8 payloads, invalid timeout, and non-absolute/non-directory cwd

Do not accept environment overrides in `RawCliRequest`.

- [ ] **Step 4: Implement fixed shell selection and a copied allowlist environment**

Expose pure helpers so tests do not spawn processes:

```python
def resolve_shell_argv(selector: RawCliShell, command: str) -> tuple[str, ...]:
    """Return code-owned argv; the command is the only dynamic element."""

def build_scrubbed_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy only approved shell-essential variables from source/os.environ."""
```

Use these exact launch shapes:

- Bash: `bash --noprofile --norc -c <command>`
- POSIX fallback: `sh -c <command>`
- PowerShell: `pwsh|powershell -NoLogo -NoProfile -NonInteractive -Command <command>`
- CMD: `cmd.exe /D /S /C <command>`

`auto` resolves Bash then `sh` on POSIX and `pwsh`/Windows PowerShell then CMD on Windows. Explicit Bash on Windows is supported only when an ordinary `bash` executable is discoverable; do not add WSL path translation in v1.

Build the output environment from `{}` and copy only the platform-relevant subset of `PATH`, `HOME`, `USERPROFILE`, `TMPDIR`, `TEMP`, `TMP`, locale variables, `SYSTEMROOT`, `WINDIR`, `COMSPEC`, and `PATHEXT`. Add a truth-table test containing representative API keys, proxy variables, tracing credentials, and Python injection variables and assert none survive.

- [ ] **Step 5: Normalize and document the persisted unlock**

Add `raw_cli_permitted = false` to the shipped `[console]` template and normalize it with `coerce_bool_setting(..., False)`. Do not add an `armed` config key.

- [ ] **Step 6: Re-run the focused tests**

Run:

```bash
pytest -q Tests/Tools/test_raw_cli_executor_contract.py Tests/test_config_console_defaults.py -k raw_cli
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Tools/raw_cli_executor.py tldw_chatbook/config.py Tests/Tools/test_raw_cli_executor_contract.py Tests/test_config_console_defaults.py
git commit -m "feat: define raw CLI execution contracts"
```

## Task 2: Build the admitted one-shot executor and bounded stream pipeline

**Files:**

- Modify: `tldw_chatbook/Tools/raw_cli_executor.py`
- Modify: `tldw_chatbook/STT/executor_process_tree.py`
- Create: `Tests/Tools/test_raw_cli_executor_process.py`
- Modify: `Tests/STT/test_executor_process_tree.py`

- [ ] **Step 1: Write failing admission and lifecycle tests**

Add tests for:

- the worker reports `WorkerContainmentIdentity`, then blocks on `admission_event` before calling `subprocess.Popen`;
- failed admission terminates the untrusted worker without launching the shell and returns `containment_unavailable`;
- outer launch always uses `shell=False`, `stdin=subprocess.DEVNULL`, and separate stdout/stderr pipes;
- stdout/stderr remain distinguishable, invalid UTF-8 becomes U+FFFD, and a nonzero exit remains `terminal_state="exited"` with the real exit code;
- timeout and cancellation use the same idempotent process-tree cleanup;
- a flood larger than preview/spool caps is fully drained, bounded, and reports `truncated=True`;
- a late cancel cannot rewrite an exited result.

The admission-race test must coordinate with events, not sleep:

```python
assert worker_identity_received.wait(2)
assert not shell_started.is_set()
assert admit_worker(tree) is True
assert shell_started.wait(2)
```

- [ ] **Step 2: Run the process tests and confirm failure**

Run:

```bash
pytest -q Tests/Tools/test_raw_cli_executor_process.py Tests/STT/test_executor_process_tree.py
```

Expected: new executor tests FAIL; existing containment tests PASS.

- [ ] **Step 3: Generalize the existing containment module without duplicating it**

Change the module/class docstrings in `executor_process_tree.py` from STT-only wording to local worker-generation wording. Preserve its public API and existing STT behavior. Add no second Job Object wrapper.

- [ ] **Step 4: Implement the worker admission handshake**

Use `multiprocessing.get_context("spawn")`. The worker entry point must:

1. call `enter_worker_containment()`;
2. send the identity to the parent;
3. wait for `admission_event` or abort;
4. only then resolve/launch the shell;
5. drain stdout and stderr concurrently;
6. send a single terminal payload and close its queue handles.

The parent must construct `ExecutorProcessTree(process, admission_event, identity)` and pass it to a required `admit_worker(tree)` callback. The executor never calls `tree.admit()` directly. Task 2 tests may supply a minimal callback that admits immediately; Task 3 supplies the runtime-owned callback that atomically rechecks authority and admits under the runtime lock. Start the execution clock only after the callback reports successful admission.

- [ ] **Step 5: Bound worker-to-parent output and sanitize at one choke point**

Implement a small standard-library accumulator rather than a terminal emulator:

- a bounded IPC queue;
- per-stream coalescing by byte threshold or a short monotonic interval;
- concurrent readers that always continue draining after caps are reached;
- UTF-8 replacement decoding;
- removal of CSI/ANSI, OSC (BEL and ST terminated), C0/C1 controls except normalized newline/tab;
- Rich/markup rendered as literal text;
- 32 KiB combined transcript preview;
- a bounded secure spool capped at `configured_max_record_bytes()` with restrictive permissions.

The executor is the sole spool owner. It writes stream-tagged sanitized records, reads at most the configured cap into `RawCliResult.record_output`, and closes/deletes the spool in its own `finally` before returning or raising. No spool path crosses into the runtime, Console controller, model provider, AgentRunsDB, or `RunLogWriter`; those consumers receive only bounded text. Add tests for success, spawn/admission failure, cancellation, callback failure, and run-log write failure proving no temporary artifact remains.

Generic logger calls may include only invocation id, shell, elapsed time, byte counts, terminal state, truncation, and cleanup certainty—never command text, cwd, stdout, or stderr.

- [ ] **Step 6: Implement monotonic cancellation, timeout, and cleanup reporting**

`RawShellExecutor.execute(request, *, cancel_event, on_event, admit_worker)` is synchronous so both a Textual worker and a model-tool worker can call the same implementation. It returns exactly one `RawCliResult`. Every terminal path—including an ordinary shell exit—closes the process-tree owner so accidental background children are terminated and cleanup is proven where possible. `cancel_event`, timeout, or parent shutdown call `ExecutorProcessTree.terminate_tree()`; if death cannot be proven, preserve the trigger (`cancelled`/`timed_out`) in result details and set `cleanup_proven=False`, using `cleanup_unproven` only when no more specific terminal state exists.

- [ ] **Step 7: Add real platform evidence**

POSIX test: launch a shell that creates a grandchild, cancel, and prove the owned process group disappears; separately let a shell exit after starting a redirected background child and prove normal finalization removes it.

Windows test: run in the repository's existing `.github/workflows/test.yml` `windows-latest` matrix, assign the waiting worker to a kill-on-close Job Object, release admission, create a child, and prove both cancellation and ordinary finalization empty the job. Mocked Windows tests remain useful for errors but do not satisfy this step.

- [ ] **Step 8: Run focused process tests**

Run on POSIX:

```bash
pytest -q Tests/Tools/test_raw_cli_executor_process.py Tests/STT/test_executor_process_tree.py
```

Expected: PASS with native POSIX evidence; Windows-only tests SKIP with an explicit reason. Run the same files on native Windows and expect the Job Object evidence to PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Tools/raw_cli_executor.py tldw_chatbook/STT/executor_process_tree.py Tests/Tools/test_raw_cli_executor_process.py Tests/STT/test_executor_process_tree.py
git commit -m "feat: execute admitted bounded raw shell commands"
```

## Task 3: Add the app-owned arm state and active-command lifecycle

**Files:**

- Create: `tldw_chatbook/Chat/console_raw_cli.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Chat/test_console_raw_cli_runtime.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

- [ ] **Step 1: Write failing runtime tests**

Pin these invariants:

```python
def test_runtime_starts_unarmed_even_when_persistent_unlock_is_true(): ...
def test_arm_refuses_until_saved_unlock_is_true(): ...
def test_execute_rechecks_unlock_and_arm_before_admission(): ...
def test_disarm_clears_session_grants_and_cancels_every_active_invocation(): ...
def test_shutdown_is_idempotent_and_bounded(): ...
def test_terminal_result_wins_over_late_disarm(): ...
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Chat/test_console_raw_cli_runtime.py Tests/UI/test_console_runtime_ownership.py -k raw_cli
```

Expected: FAIL because `RawCliRuntime` is absent.

- [ ] **Step 3: Implement one process-lifetime owner**

Add `RawCliRuntime` with a lock-protected `_armed = False`, active invocation map, and methods:

```python
def arm(self) -> RawCliArmResult: ...
def disarm(self) -> tuple[str, ...]: ...
def execute(self, request: RawCliRequest, on_event: RawCliEventSink) -> RawCliResult: ...
def cancel(self, invocation_id: str) -> bool: ...
def shutdown(self) -> RawCliShutdownResult: ...
```

Inject a callable that reads the latest saved `raw_cli_permitted`; do not snapshot it at construction. Admission order is: validate request → verify permitted/armed → register invocation → ask the executor to spawn its waiting worker → enter the runtime's admission guard → recheck permitted/armed under the runtime lock → call `ExecutorProcessTree.admit()` while that lock is still held. The executor must not call `admit()` directly; it receives a runtime-owned `admit_worker(tree)` callback. If Disarm linearizes first, the guard refuses and the executor terminates the unadmitted worker with `containment_unavailable`; if admission linearizes first, Disarm sees the registered active invocation and signals its cancellation before returning. This lock/guard is the proof that no shell can begin after Disarm has completed. Disarm atomically sets unarmed, clears future model session grants (an empty hook in this task), snapshots active cancellation events, then signals them outside the lock.

- [ ] **Step 4: Wire ownership into `TldwCli`**

Construct `self.raw_cli_runtime` immediately after `self.app_config = load_settings()`. Add `_shutdown_raw_cli_runtime()` and invoke it before `_shutdown_console_runtime()` so no command callback can target a disposed Console store. Keep shutdown idempotent across `_shutdown()` and `on_unmount()` fallback paths.

- [ ] **Step 5: Run focused lifecycle tests**

```bash
pytest -q Tests/Chat/test_console_raw_cli_runtime.py Tests/UI/test_console_runtime_ownership.py -k raw_cli
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/app.py Tests/Chat/test_console_raw_cli_runtime.py Tests/UI/test_console_runtime_ownership.py
git commit -m "feat: own raw CLI arming for one app launch"
```

## Task 4: Add the unmistakable Settings danger gate

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Create: `Tests/UI/test_settings_raw_cli.py`

- [ ] **Step 1: Write mounted Settings tests**

Test all three states: Locked; Unlocked, not armed; Armed for this launch. Assert:

- the persistent checkbox participates in ordinary Settings draft/save/revert behavior;
- Arm is disabled until a changed unlock has been saved;
- first unlock and each launch Arm require separate confirmations;
- Disarm acts immediately and does not modify the settings draft;
- saving unlock Off disarms and starts cleanup;
- the armed state retains a visible danger class/label after transient notifications disappear;
- confirmation focus defaults to Cancel and no single shortcut auto-accepts host authority;
- the disabled Arm action names its recovery (`Save unlock first`) and remains readable at the repo's measured disabled-control contrast floor;
- state and recovery remain readable without color and at the smallest supported terminal size.

Pin disclosure substrings for full OS-user file/process/network authority, credential-file access despite scrubbing, local command/output persistence, and possibly surviving detached descendants. Assert the copy does not contain “sandboxed” or “confined.”

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_settings_raw_cli.py
```

Expected: FAIL because the Settings card is absent.

- [ ] **Step 3: Add the Privacy & Security card through existing Settings patterns**

Add one `Raw CLI — unsafe` operating panel to the canonical Privacy & Security section. Privacy & Security is currently read-only, so make the mixed contract explicit rather than placing a mutating control under a `Read-only here` badge:

- add `SettingsCategoryId.PRIVACY_SECURITY` to `GUIDED_SETTINGS_MUTATION_CATEGORIES`;
- extend that category's draft/load/change/save/revert seams with only `console.raw_cli_permitted`;
- change its State/Scope and Impact copy to `Draft — save with s` and say the raw unlock is the sole editable value while posture/secrets remain read-only;
- keep `Check Privacy` and existing jump actions behavior unchanged.

Use the existing settings adapter for `console.raw_cli_permitted`. Keep the category's pinned save-model State bar authoritative: the unlock is a draft until Save, while the Arm/Disarm row says `Applies immediately`. Use explicit confirmation modals for unlock and arm; never place `raw_cli_armed` in a draft, adapter, config, snapshot, or screen state.

The card must clearly render:

```text
DANGER!!! RAW CLI HOST ACCESS
Commands run with every permission of the OS user running Chatbook.
This is not a sandbox and is not limited to your workspace.
```

Keep the complete disclosure visible in/adjacent to the control, not only in a modal.

- [ ] **Step 4: Add restrained but persistent danger styling**

Use existing semantic `$error`/surface variables and a dedicated danger class; do not introduce a second stylesheet. Use a full state border/background plus a non-color `ARMED — HOST ACCESS` label/icon—never a decorative side stripe or color alone. Focus/hover must not change dimensions, and the narrow layout must wrap/scroll disclosure without pushing Arm/Disarm out of reach.

- [ ] **Step 5: Run the mounted Settings tests**

```bash
pytest -q Tests/UI/test_settings_raw_cli.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_settings_raw_cli.py
git commit -m "feat: add raw CLI danger gate to Settings"
```

## Task 5: Track physically typed `! ` provenance in the composer

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `Tests/UI/test_console_command_composer.py`

- [ ] **Step 1: Write failing provenance tests**

Cover:

- physically typing `!` then Space at offsets 0 and 1 sets `raw_cli_prefix_typed=True`;
- pasting `! command` never sets it;
- programmatic draft replacement with `! command` never sets it;
- after physical prefix activation, pasting only the command body preserves it;
- clicking the visible Send action after physically typing the prefix preserves and consumes the trusted stash exactly like pressing Enter;
- editing/deleting/replacing either prefix character clears it and cannot reconstruct it from text alone;
- stashing and restoring a refused send preserves the trusted latch;
- ordinary history restore/undo does not mint the latch;
- `\! ` is classified as escaped chat and removes one leading backslash.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_console_command_composer.py -k raw_cli
```

Expected: FAIL.

- [ ] **Step 3: Implement a non-serializable typed-prefix latch**

Add `raw_cli_prefix_typed: bool = False` to `ConsoleDraftStash`. Keep private composer progress that can only advance in the physical printable-key handler:

- physical `!` into an empty draft at index 0 records stage one;
- physical Space immediately after that exact `!` records the trusted prefix;
- all mutation paths may invalidate the latch but may never create it;
- `insert_pasted_text`, `set_draft_text`, dictation, history restore, session restore, and inline-file insertion cannot advance it;
- `restore_stashed_draft` restores the stash bit because this is the exact rejected keypress payload.

Expose a pure classifier:

```python
@dataclass(frozen=True, slots=True)
class ConsoleRawDraft:
    kind: Literal["chat", "escaped_chat", "raw"]
    text: str

def classify_console_raw_draft(stash: ConsoleDraftStash) -> ConsoleRawDraft:
    if stash.text.startswith(r"\! "):
        return ConsoleRawDraft("escaped_chat", stash.text[1:])
    if stash.raw_cli_prefix_typed and stash.text.startswith("! "):
        return ConsoleRawDraft("raw", stash.text[2:])
    return ConsoleRawDraft("chat", stash.text)
```

Do not infer raw authorization from `has_paste`; a typed prefix followed by pasted body is explicitly valid.

- [ ] **Step 4: Drive a persistent composer danger class from the trusted latch**

When the latch is active, show `RAW CLI · HOST ACCESS` in the composer and apply its danger class. Reuse the one-row composer edge/background conventions, keep the state text-labeled, and avoid any height/focus layout shift. The display is informational; the runtime still rechecks permitted/armed on submit.

- [ ] **Step 5: Run the composer tests**

```bash
pytest -q Tests/UI/test_console_command_composer.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_command_composer.py
git commit -m "feat: require typed provenance for raw Console commands"
```

## Task 6: Route direct user commands before every provider and queue seam

**Files:**

- Modify: `tldw_chatbook/Chat/console_raw_cli.py`
- Create: `tldw_chatbook/UI/Console_Modules/raw_cli.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_raw_cli_send.py`
- Modify: `Tests/Chat/test_console_send_gate_queue_race.py`

- [ ] **Step 1: Write failing send-path tests**

Use provider, token-accounting, slash-parser, and prompt-queue spies. Prove an armed trusted raw stash:

- is intercepted before slash parsing and provider readiness;
- executes while an ordinary model run is active;
- makes zero provider requests and records zero model tokens;
- never enters the prompt queue;
- leaves staged attachments untouched and never treats them as command input;
- refuses an empty command, Locked state, or Unlocked-but-unarmed state inline and restores the exact draft;
- does not open/change Settings automatically;
- treats pasted prefix text as chat and escaped `\! ` as ordinary chat text.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_console_raw_cli_send.py Tests/Chat/test_console_send_gate_queue_race.py -k raw_cli
```

Expected: FAIL.

- [ ] **Step 3: Add the pre-slash dispatch branch**

At the start of `_send_console_message_from_visible_action`, immediately after obtaining the `ConsoleDraftStash`, classify it. For `escaped_chat`, replace only the stashed text and continue through the normal chat path. For `raw`, delegate to `self._raw_cli.start_user_command(stash)` and return before slash parsing, readiness, attachments, or `_dispatch_console_draft_send`.

Build direct-user requests with `shell="auto"`. Resolve the initial directory once at submission from the selected local-filesystem binding, falling back to that Chat's private scratch, and capture the current persisted transcript leaf as the anchor. Neither value is a confinement boundary. V1 deliberately adds no alternate `!bash`/`!powershell` composer grammar or persisted shell preference.

Construct `ConsoleRawCliController` in `UI/Console_Modules/wiring.py`. Its keyword-only constructor names late-bound callables for the app runtime, active session/anchor/root, composer restore, store marker operations, AgentRunsDB/run-log access, worker start, and UI-thread marshal; do not make it reach sibling controllers through screen attributes. Keep only the existing screen proxy/call site in `chat_screen.py` so the screen-size ratchet does not rise.

The controller starts a named non-exclusive Textual worker because command execution must not serialize behind the provider prompt queue. The worker calls the app-owned runtime synchronously; UI updates marshal through injected thread-safe screen seams. A locked/unarmed refusal remains a local transcript/composer error and restores the trusted stash.

- [ ] **Step 4: Re-run the send-path tests**

```bash
pytest -q Tests/UI/test_console_raw_cli_send.py Tests/Chat/test_console_send_gate_queue_race.py -k raw_cli
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_raw_cli_send.py Tests/Chat/test_console_send_gate_queue_race.py
git commit -m "feat: run typed Console commands without a provider"
```

## Task 7: Render a live bounded command marker with Stop

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Modify: `tldw_chatbook/UI/Console_Modules/raw_cli.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_raw_cli_transcript.py`

- [ ] **Step 1: Write failing store and mounted transcript tests**

Pin one stable marker id from start to finish. Assert separate stdout/stderr labels, shell, initial directory, elapsed, exit code, truncation, and cleanup certainty. Assert text—not color—distinguishes Running/Stopped/Timed out/Cleanup unproven. Assert `Stop` appears only while active, is reachable in ordinary focus order without a terminal-convention keybinding, calls `runtime.cancel(invocation_id)` once, and disables immediately with an explicit `Stopping…` state. Include ANSI/OSC/Rich-markup payloads and prove rendered text is literal and safe.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_console_raw_cli_transcript.py
```

Expected: FAIL.

- [ ] **Step 3: Add a narrow TOOL-marker update seam**

Add `ConsoleChatStore.update_tool_marker(session_id, message_id, **bounded_fields)` that updates only the registered display-only marker, preserves its anchor/id, recomputes the active-path projection, and returns an immutable snapshot. Add a narrow `record_trajectory=False` option when creating direct raw markers so `_record_trajectory_tool_marker` does not misclassify a user-authored local action as an agent tool call. Do not turn TOOL markers into conversation tree nodes, trajectory records, or provider-history messages.

Extend activity presentation only as needed for raw command `running`/terminal state. Do not make generic message actions capable of running arbitrary callbacks; resolve one `raw-cli-stop` action only for a marker carrying a bounded `RawCliPresentation` with an active invocation id.

- [ ] **Step 4: Coalesce UI repaint without losing stream separation**

Update the marker no more frequently than the executor event cadence. Keep complete bounded stdout/stderr in `tool_output_full`; keep the compact preview in `content`. Render cwd and command with markup disabled. The command row owns the running elapsed timer; no app-wide timer is introduced.

- [ ] **Step 5: Re-run transcript tests**

```bash
pytest -q Tests/UI/test_console_raw_cli_transcript.py Tests/UI/test_console_native_transcript.py -k "raw_cli or tool_marker"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_raw_cli_transcript.py
git commit -m "feat: stream raw commands into stoppable Console markers"
```

## Task 8: Persist and restore `local_command` records without agent side effects

**Files:**

- Modify: `tldw_chatbook/Chat/console_raw_cli.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py`
- Modify: `tldw_chatbook/UI/Console_Modules/raw_cli.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `Tests/DB/test_agent_runs_db.py`
- Create: `Tests/Chat/test_console_raw_cli_persistence.py`
- Modify: `Tests/UI/test_console_resume_active_path.py`
- Modify: `Tests/Agents/test_run_log_writer.py`
- Modify: `Tests/Agents/test_run_log_cross_run_search.py`

- [ ] **Step 1: Write failing persistence and exclusion tests**

Prove:

- every accepted user command creates one `agent_kind="local_command"` run with a generic task label and current persisted leaf locator;
- exact command and bounded sanitized result are steps/run-log contents, not run metadata or generic diagnostics;
- completion status and exit code survive restart;
- resume recreates one TOOL marker at the recorded anchor;
- provider history before/after resume is byte-identical to the history without the marker;
- model-facing run-log search/slice/stats cannot discover a `local_command` command or output;
- `latest_primary_run`, subagent counts, fleet state, assistant-turn grouping, rails, and cost summaries ignore local commands.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_raw_cli_persistence.py Tests/UI/test_console_resume_active_path.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_cross_run_search.py -k local_command
```

Expected: FAIL.

- [ ] **Step 3: Persist through existing DB and run-log APIs**

Use the unconstrained `agent_kind` column; do not add a migration. Update `AgentRunsDB` docstrings/type hints that falsely imply only primary/subagent. `ConsoleRawCliController` creates the run before execution, appends bounded `STEP_TOOL_CALL`/`STEP_TOOL_RESULT`-compatible records (or dedicated local-command step kinds if the formatter cannot represent shell fields without ambiguity), finalizes exactly once, and writes `RawCliResult.record_output` through `RunLogWriter` rooted in an app-private `local-command-runs` directory that is never registered with model-facing run-log search/slice/stats providers. The executor has already closed/deleted its private spool before returning; persistence never receives a path to clean up. Keep the controller's DB/log dependencies in `wiring.py` as late-bound callables.

- [ ] **Step 4: Restore only local display markers**

Extend `ConsoleAgentBridge.resume_marker_messages()` or add a tightly scoped companion that queries `agent_kind="local_command"`, formats one bounded marker per run, and injects it at `assistant_message_id`. Keep existing primary-run reconstruction unchanged. Missing/deleted anchors fail quietly without reparenting a conversation.

- [ ] **Step 5: Make exclusions explicit where queries are broader than exact kinds**

Most current queries already select `agent_kind='primary'` or `'subagent'`. Audit every run-count/fleet/cost/rail query touched by the resume path and add explicit kind predicates only where a broad query would now include local commands. Add regression tests at those seams; do not build a new run taxonomy layer.

- [ ] **Step 6: Re-run persistence tests**

```bash
pytest -q Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_raw_cli_persistence.py Tests/UI/test_console_resume_active_path.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_cross_run_search.py -k local_command
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/DB/AgentRuns_DB.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/UI/Console_Modules/wiring.py Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_raw_cli_persistence.py Tests/UI/test_console_resume_active_path.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_cross_run_search.py
git commit -m "feat: persist resumable local command markers"
```

## Task 9: Document and verify the raw user-command vertical slice

**Files:**

- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console.md`
- Modify: `backlog/decisions/033-local-agent-process-execution-boundary.md`
- Modify: `backlog/tasks/task-18926 - Raw-CLI-executor-and-Console-user-command.md`

- [ ] **Step 1: Update user and authority documentation**

Document exact `! ` syntax, `\! ` escape, typed-prefix/paste rule, supported shells, non-interactive stdin, timeout, Stop, per-launch re-arm, config key, local persistence, environment scrubbing limitations, and cleanup uncertainty. State plainly that raw CLI has OS-user authority and is not confined to a workspace. Add ADR-093's partial-supersession link to ADR-033.

- [ ] **Step 2: Run the focused raw suite**

```bash
pytest -q \
  Tests/Tools/test_raw_cli_executor_contract.py \
  Tests/Tools/test_raw_cli_executor_process.py \
  Tests/STT/test_executor_process_tree.py \
  Tests/Chat/test_console_raw_cli_runtime.py \
  Tests/Chat/test_console_raw_cli_persistence.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_raw_cli_send.py \
  Tests/UI/test_console_raw_cli_transcript.py
```

Expected: PASS on POSIX with Windows-native tests skipped; PASS on native Windows with Job Object evidence.

- [ ] **Step 3: Run static and whitespace checks scoped to changed files**

```bash
ruff check tldw_chatbook/Tools/raw_cli_executor.py tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py
git diff --check
```

Expected: PASS with no warnings/errors. Do not run the full pytest suite unless the user explicitly asks.

- [ ] **Step 4: Perform live mounted verification with isolated state**

Launch Chatbook with isolated config/data directories. Verify Locked refusal, saved unlock, Arm, red composer state, a command with separate stdout/stderr, Stop on a long command, Disarm cleanup, navigation away/back, and restart returning to Unlocked/not armed. Capture rendered frames or screenshots as evidence; widget existence alone is not visual verification.

- [ ] **Step 5: Self-review against ADR-093 and TASK-18926**

Search for accidental command/output diagnostic logging, ambient `os.environ.copy()`, `shell=True`, persisted arm/session grants, “sandbox/confined” raw copy, unbounded queues/buffers, provider-history insertion, and generic agent queries that include `local_command`.

- [ ] **Step 6: Finish Backlog hygiene only after all evidence exists**

Check every acceptance criterion, add concise Implementation Notes listing the exact files/decisions/evidence, keep ADR-093 linked, record any actual new lesson only if the work exposed one, and set TASK-18926 Done through the Backlog CLI.

- [ ] **Step 7: Commit documentation and task completion**

```bash
git add Docs backlog/decisions/033-local-agent-process-execution-boundary.md "backlog/tasks/task-18926 - Raw-CLI-executor-and-Console-user-command.md"
git commit -m "docs: explain raw CLI host authority"
```
