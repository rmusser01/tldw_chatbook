# Persistent Terminal Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add up to four user-only, app-global interactive PTY/ConPTY terminal sessions inside Console without changing raw `!`, model `shell_exec`, or `virtual_cli` authority, persistence, or context boundaries.

**Architecture:** One app-owned `TerminalSessionManager` holds volatile authority, session records, terminal models, and platform backends; Textual widgets subscribe to immutable projections and never own processes. POSIX launches through an admission-gated controlling PTY, Windows through an admitted low-level pywinpty ConPTY worker in a kill-on-close Job Object, and all terminal bytes pass through a bounded pre-parser and qualified pyte adapter before safe cells reach the UI.

**Tech Stack:** Python 3.11+, Textual 8.2.8, pyte 0.8.2, pywinpty 3.0.5 on Windows only, psutil, POSIX PTY APIs, Windows ConPTY/Job Objects, pytest, Ruff

---

## Governing documents and delivery gate

- Approved design: `Docs/superpowers/specs/2026-08-28-persistent-terminal-sessions-design.md`
- Accepted ADR: `backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`
- Backlog task: `backlog/tasks/task-22512 - Persistent-interactive-PTY-and-ConPTY-terminal-sessions.md`
- Mandatory qualification evidence: `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`

- ADR required: yes
- ADR path: `backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`
- Reason: ADR-099 fixes the long-lived PTY/ConPTY owner, parser and dependency boundary, launch-local authority, user-only privacy contract, resource limits, and cleanup proof introduced by this work.

The qualification artifact is a hard gate. Do not add pyte/pywinpty to project dependencies, write product parser/backend code, or begin UI integration until every mandatory row for the affected parser/backend passes. A failed pyte row stops the feature. A failed Windows row makes Windows Terminal fail closed and blocks claiming Windows acceptance. Changing either pinned version or the low-level API boundary requires a new or superseding ADR before continuing.

Focused tests are the default. Do not run the full suite unless the user explicitly opts in, as required by `AGENTS.md`.

## File map

### New runtime package

- `tldw_chatbook/Terminal/__init__.py`: intentionally small public exports.
- `tldw_chatbook/Terminal/contracts.py`: limits, enums, immutable projections, transition validation, cleanup deadlines.
- `tldw_chatbook/Terminal/backend.py`: platform-neutral backend protocol and event contracts.
- `tldw_chatbook/Terminal/launch.py`: names, shell discovery, argv, starting-directory resolution, and terminal-specific scrubbed environments.
- `tldw_chatbook/Terminal/protocol_gate.py`: bounded cross-chunk terminal-control pre-parser.
- `tldw_chatbook/Terminal/screen_model.py`: incremental UTF-8, qualified pyte adapter, safe cells, alternate screen, and bounded scrollback.
- `tldw_chatbook/Terminal/io_actors.py`: ordered input, atomic paste, output credit/backpressure, resize coalescing, and priority cleanup signal.
- `tldw_chatbook/Terminal/session_manager.py`: app-global authority, four-slot admission, session ownership, subscriptions, selection, lifecycle, and parallel cleanup coordination.
- `tldw_chatbook/Terminal/posix_backend.py`: admission-gated controlling PTY, nonblocking I/O, process identity, one reaper, and POSIX cleanup proof.
- `tldw_chatbook/Terminal/posix_launcher.py`: fresh executable gated helper that performs session/TTY setup and then becomes the shell PID.
- `tldw_chatbook/Terminal/windows_backend.py`: parent-side admitted worker, credit protocol, Job ownership, and Windows cleanup proof.
- `tldw_chatbook/Terminal/windows_job.py`: narrow ctypes Job Object and process-handle operations.
- `tldw_chatbook/Terminal/windows_worker.py`: fresh-process low-level `winpty.PTY(..., backend=Backend.ConPTY)` owner.

### New Console presentation

- `tldw_chatbook/UI/Console_Modules/terminal.py`: `ChatScreen`-facing controller, projection subscription, view-generation tokens, and workspace routing.
- `tldw_chatbook/UI/Console_Modules/wiring.py`: centralized late-bound construction of `screen._terminal` and selected-local-root access.
- `tldw_chatbook/Widgets/Console/console_terminal_workspace.py`: safe-cell viewport, session/action strip, status, receipts, and keyboard-local scrollback mode.
- `tldw_chatbook/Widgets/Console/console_terminal_session_modal.py`: New/Rename form with name, shell, and starting-directory fields.
- `Tests/Terminal/`: focused domain, parser, actor, manager, POSIX, Windows, and qualification tests.
- `Tests/fixtures/terminal/`: real executable descendants, profile probes, output/EOF fixtures, and app-crash probes.
- `scripts/terminal_qualification/`: repeatable dependency/backend qualification probes.

### Existing integration points

- `tldw_chatbook/app.py`: construct and shut down the one app-owned manager.
- `tldw_chatbook/UI/Screens/settings_screen.py`: shared saved unlock plus independent raw-CLI and Terminal launch arms.
- `tldw_chatbook/UI/Screens/chat_screen.py`: typed Terminal request/action handling and center-region selection only; controller construction stays in wiring.
- `tldw_chatbook/UI/Console_Modules/left_rail.py`: visible Terminal action in Sessions.
- `tldw_chatbook/UI/console_command_provider.py`: `Console: Open Terminal` command.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: Terminal workspace and persistent danger treatment.
- `tldw_chatbook/css/tldw_cli_modular.tcss`: regenerated output only.
- `pyproject.toml` and `requirements.txt`: exact qualified dependency pins.
- `Tests/UI/test_console_runtime_ownership.py`, `Tests/UI/test_console_controller_wiring.py`, `Tests/UI/test_settings_raw_cli.py`, `Tests/UI/test_console_left_rail.py`, `Tests/UI/test_console_internals_decomposition.py`, and `Tests/UI/test_css_bundle_sync_guard.py`: existing ownership, controller-wiring, and UI contracts extended in place.
- `Tests/Architecture/test_persistent_terminal_privacy_boundary.py`: no tool, provider, persistence, export, run-log, or model-context projection.
- `Docs/User_Guide/console/sessions-tabs-workspaces.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`, `Docs/User_Guide/settings.md`, `Docs/User_Guide/console.md`, and `README.md`: user and platform documentation.
- `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md` and `backlog/decisions/README.md`: relationship/index metadata only; ADR-094 semantics stay unchanged.

## Plan handoff prerequisite

Before implementation, link this approved plan from TASK-22512, commit the two planning documents together, and confirm the worktree is clean. Do not use `backlog task edit` for the five-digit task ID; edit the task file directly.

```bash
git add Docs/superpowers/plans/2026-08-28-persistent-terminal-sessions-implementation.md \
  "backlog/tasks/task-22512 - Persistent-interactive-PTY-and-ConPTY-terminal-sessions.md"
git commit -m "docs: plan persistent terminal sessions"
git status --short
```

Expected: the planning commit succeeds and `git status --short` prints nothing.

Before Task 1, update the integration base and replay the planning/design commits onto the exact latest remote development branch:

```bash
git fetch origin
git rebase origin/dev
git status --short --branch
```

Expected: rebase succeeds, the worktree is clean, and the branch is no longer behind `origin/dev`. Task 1 resolves `origin/dev` once, stores that immutable commit SHA in `format-baseline.json`, and every later formatter verification reads the stored SHA rather than a moving ref. If a later pre-merge rebase changes the base, regenerate the baseline from the new exact `origin/dev` SHA before rerunning the ratchet.

For every later commit block, inspect the index before committing:

```bash
git diff --cached --name-only
git diff --cached
```

Expected: only the exact paths named by that task and only intended hunks are staged. If anything else appears, stop and unstage that path without discarding the user's working-tree changes.

Before product code, Task 1 captures a machine-readable formatter ratchet for every existing Python file this plan expects to modify. The current planning-base probe reports six already-red files: `app.py`, `settings_screen.py`, `wiring.py`, `test_console_runtime_lifetime.py`, `test_console_internals_decomposition.py`, and `test_console_workbench_contract.py`. The authoritative snapshot is generated after the implementation branch's initial rebase and lives in `Docs/superpowers/reviews/evidence/task-22512/format-baseline.json`, not in Implementation Notes (which AGENTS.md reserves for post-implementation closeout).

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/console_command_provider.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  Tests/Chat/test_console_runtime_lifetime.py \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_css_bundle_sync_guard.py
```

Expected on the current planning base: exactly the six files named above are reported. Task 1's ratchet stores normalized `ruff format --check --diff` hashes for the base blobs; final verification fails if a task-changed line overlaps a formatter-required hunk or if normalized formatter debt grows in an already-red file.

Every task that creates Python files must run `ruff format --check` on those exact new paths in its GREEN verification block before committing, even where the abbreviated task command below shows only `ruff check`. The final formatting matrix is a backstop, not permission to carry unformatted intermediate commits.

For every TDD slice, a missing module, import error, collection error, or unrelated failing test does not count as RED. After writing the tests, add only the minimal importable API skeleton/signatures for that slice, returning explicit neutral/refusal placeholders with no production behavior. Run the new test node(s) independently and require an assertion-level failure on the intended behavior before implementation. Then implement the smallest behavior needed for GREEN. The task-specific RED commands below inherit this rule even where they abbreviate the skeleton step.

### Task 1: Qualify dependencies and platform boundaries before product code

**Files:**
- Create: `scripts/terminal_qualification/common.py`
- Create: `scripts/terminal_qualification/pyte_probe.py`
- Create: `scripts/terminal_qualification/pywinpty_probe.py`
- Create: `scripts/terminal_qualification/environment_probe.py`
- Create: `scripts/terminal_qualification/format_ratchet.py`
- Create: `scripts/terminal_qualification/README.md`
- Create: `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`
- Create: `Docs/superpowers/reviews/evidence/task-22512/format-baseline.json`
- Create: `Tests/Terminal/test_dependency_qualification.py`
- Create: `Tests/Terminal/test_format_ratchet.py`
- Reference: `backlog/docs/lessons-testing-evidence.md`
- Reference: `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Prove the interpreter imports this worktree before collecting evidence**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/test_probe_import_provenance.py -q
```

Expected: PASS and the imported `tldw_chatbook` path resolves under this worktree. If it resolves to the main checkout, do not repair or retarget the shared editable install. Stop and rerun exactly with `PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/test_probe_import_provenance.py -q`; do not collect evidence until that worktree-qualified command passes.

- [ ] **Step 2: Write qualification contract tests and importable refusal skeletons, then verify RED**

Write `test_dependency_qualification.py` first so it requires the artifact's exact version/hash/license/wheel/platform/API/parser/environment/I/O/EOF/memory sections, requires every mandatory row to be `PASS` or an explicit fail-closed unsupported-platform result, requires uniquely named raw row evidence and all 16 formatter-baseline paths, and scans the Windows probe for forbidden APIs. Write `test_format_ratchet.py` with synthetic repositories proving inherited formatter debt outside changed lines passes while a new or expanded formatter hunk on a changed line fails.

Create only enough of each qualification script for imports and command parsing to succeed. The refusal skeletons must return an explicit `unimplemented` status and nonzero exit; `format_ratchet.py` exposes `snapshot` and `verify` entry points but does not create or accept a baseline. They must not import product code.

```python
def test_dependency_qualification_records_all_binding_rows() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        assert heading in evidence
    assert "pyte==0.8.2" in evidence
    assert "pywinpty==3.0.5" in evidence
    assert "PENDING" not in evidence
    assert "UNKNOWN" not in evidence
```

Run each contract independently so one intended failure cannot mask another:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_format_ratchet.py -q
../../.venv/bin/python -B -m pytest Tests/Terminal/test_dependency_qualification.py -q
```

Expected: both tests collect and reach assertions. The formatter test fails because the refusal skeleton neither snapshots immutable debt nor rejects changed-line formatter growth; the evidence test fails on the first missing required artifact fact. Import, collection, provenance, or unrelated failures do not count.

- [ ] **Step 3: Implement the qualification scripts and immutable formatter baseline**

`common.py` should record OS/version/architecture/Python, package version and file hash, wheel metadata, elapsed time, and peak/RSS facts without recording environment values, shell profile content, or terminal output. All probes accept `--json-out PATH`, refuse an existing output path unless `--replace` is explicit, and exit nonzero when a mandatory row fails. `environment_probe.py` accepts `--shell default|bash|zsh|powershell|cmd`; `pyte_probe.py` owns the parser matrix; `pywinpty_probe.py` owns low-level API identity, ConPTY-only construction, one-credit reads, concurrent write/resize/cancel/close, Job membership, EOF, and output-integrity rows.

The pywinpty probe must fail if any of these symbols/paths appear:

```python
FORBIDDEN_WINDOWS_APIS = (
    "PtyProcess",
    "PtyProcessUnicode",
    "Backend.WinPTY",
    "subprocess.PIPE",
)
```

`format_ratchet.py snapshot` resolves `--base` to one commit SHA, stores that immutable SHA, materializes each named base blob into a temporary directory, runs the repository Ruff version with `format --check --diff`, normalizes temporary paths and hunk offsets, and writes source/normalized-diff hashes plus the exact already-red file set. `verify` reads the stored SHA rather than resolving `origin/dev` again, repeats the process for `HEAD`, reads the branch's zero-context changed-line ranges, and fails if task-changed lines overlap a formatter-required hunk or normalized formatter debt grows.

Run the pre-code snapshot with every existing modified Python path:

```bash
../../.venv/bin/python scripts/terminal_qualification/format_ratchet.py snapshot \
  --base origin/dev \
  --output Docs/superpowers/reviews/evidence/task-22512/format-baseline.json \
  --path tldw_chatbook/app.py \
  --path tldw_chatbook/UI/Screens/settings_screen.py \
  --path tldw_chatbook/UI/Screens/chat_screen.py \
  --path tldw_chatbook/UI/Console_Modules/left_rail.py \
  --path tldw_chatbook/UI/Console_Modules/wiring.py \
  --path tldw_chatbook/UI/console_command_provider.py \
  --path tldw_chatbook/Widgets/Console/__init__.py \
  --path Tests/Chat/test_console_runtime_lifetime.py \
  --path Tests/UI/test_console_runtime_ownership.py \
  --path Tests/UI/test_settings_raw_cli.py \
  --path Tests/UI/test_console_left_rail.py \
  --path Tests/UI/test_console_internals_decomposition.py \
  --path Tests/UI/test_console_controller_wiring.py \
  --path Tests/UI/test_console_workbench_contract.py \
  --path Tests/UI/test_console_shell_regions.py \
  --path Tests/UI/test_css_bundle_sync_guard.py
```

Expected: the JSON records the fresh rebased base, Ruff version, all 16 paths, normalized formatter-diff hashes, and the exact baseline-red set without modifying any source file.

Verify the formatter ratchet independently before any host qualification, then confirm the evidence contract remains RED only because native rows have not been collected:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_format_ratchet.py -q
../../.venv/bin/python -B -m pytest Tests/Terminal/test_dependency_qualification.py -q
```

Expected: the formatter-ratchet test passes. The evidence test collects and fails an artifact-content assertion because native qualification rows are not yet complete; no import, collection, or unrelated failure is accepted.

- [ ] **Step 4: Run pyte qualification in an isolated temporary environment**

Use a disposable virtual environment, install exactly `pyte==0.8.2`, record the downloaded artifact SHA-256 and license, then run the parser matrix against captured output from the default account shell, Bash/Zsh where available, PowerShell/CMD output fixtures, Vim/Nano, Less, top/htop-class programs, Unicode/wide/combining cells, alternate screen, resize, bracketed paste, terminal queries, malformed controls, and every bounded incomplete-sequence class.

`common.py prepare-row` runs under the selected host interpreter, creates the row's venv, downloads every named requirement with dependencies into the row directory, hashes the exact artifacts, installs only from those downloaded files with `--no-index`, records the fully resolved version set, and writes a manifest consumed by later probes. Use one unique directory per OS/Python/architecture row.

Run on each POSIX row from the worktree, replacing the example row ID and interpreter with that claimed host's exact values:

```bash
TASK22512_ROW_DIR=$(mktemp -d /tmp/tldw-task-22512-macos-arm64-py311.XXXXXX)
python3.11 scripts/terminal_qualification/common.py prepare-row \
  --row-id macos-arm64-py311 \
  --row-dir "$TASK22512_ROW_DIR" \
  --requirement pyte==0.8.2 \
  --requirement "wcwidth>=0.2.14,<1" \
  --json-out "$TASK22512_ROW_DIR/artifacts.json"
"$TASK22512_ROW_DIR/venv/bin/python" \
  scripts/terminal_qualification/pyte_probe.py \
  --artifact-manifest "$TASK22512_ROW_DIR/artifacts.json" \
  --json-out "$TASK22512_ROW_DIR/pyte.json"
```

Expected: the unique row directory contains the exact downloaded artifacts, SHA-256 manifest, resolved dependency versions including wcwidth, installed venv, and passing pyte result. The probe verifies it consumed the pyte artifact named and hashed by `artifacts.json`.

Expected: all mandatory parser rows pass with `TERM=linux`; every mutable pyte collection is classified as static, viewport-bounded, or adapter-capped. Otherwise stop and revisit ADR-099 before adding the dependency.

- [ ] **Step 5: Run real POSIX environment/profile and PTY precursor probes**

On macOS and Linux qualification hosts, prove standard account startup files run from the exact allowed environment, ambient provider/proxy/tracing/Python-injection/credential-agent values are absent before the profile, command discovery works, and profile code can intentionally repopulate values. Record only booleans/counts and package/platform facts, never values or profile output.

Run each available shell explicitly:

```bash
"$TASK22512_ROW_DIR/venv/bin/python" \
  scripts/terminal_qualification/environment_probe.py \
  --shell default --json-out "$TASK22512_ROW_DIR/env-default.json"
"$TASK22512_ROW_DIR/venv/bin/python" \
  scripts/terminal_qualification/environment_probe.py \
  --shell bash --json-out "$TASK22512_ROW_DIR/env-bash.json"
"$TASK22512_ROW_DIR/venv/bin/python" \
  scripts/terminal_qualification/environment_probe.py \
  --shell zsh --json-out "$TASK22512_ROW_DIR/env-zsh.json"
```

Expected: available shells pass; an unavailable optional shell is recorded with a content-free `unavailable` row and does not silently substitute another shell.

Expected: Bash/Zsh and the default account shell satisfy the approved startup contract; a missing optional shell is recorded as unavailable, not as a product fallback.

- [ ] **Step 6: Run native Windows pywinpty qualification**

On each supported Windows/Python/architecture row, use a fresh spawned interpreter with fd-backed standard streams. Prove Windows 10 1809/Server 2019 floor detection, exact `winpty.Backend.ConPTY`, Job admission before ConPTY creation, Job membership of worker/shell/helpers, parent-only non-inheritable Job handle, one unacknowledged 64 KiB chunk, bounded internal reads, concurrent write/resize/`cancel_io`/priority close, profile/module discovery, Unicode, alternate screen, app crash, descendant cleanup, and bounded missing-EOF behavior.

Run in PowerShell for each claimed Python/architecture row. Repeat the command with the matching Python launcher selector and row ID; the example is CPython 3.11 x64:

```powershell
$Task22512Row = Join-Path $env:TEMP ("tldw-task-22512-win-amd64-py311-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Task22512Row | Out-Null
py -3.11-64 scripts/terminal_qualification/common.py prepare-row --row-id win-amd64-py311 --row-dir $Task22512Row --requirement pywinpty==3.0.5 --requirement pyte==0.8.2 --requirement "wcwidth>=0.2.14,<1" --json-out "$Task22512Row\artifacts.json"
& "$Task22512Row\venv\Scripts\python.exe" scripts/terminal_qualification/environment_probe.py --shell powershell --json-out "$Task22512Row\env-powershell.json"
& "$Task22512Row\venv\Scripts\python.exe" scripts/terminal_qualification/environment_probe.py --shell cmd --json-out "$Task22512Row\env-cmd.json"
& "$Task22512Row\venv\Scripts\python.exe" scripts/terminal_qualification/pyte_probe.py --artifact-manifest "$Task22512Row\artifacts.json" --json-out "$Task22512Row\pyte.json"
& "$Task22512Row\venv\Scripts\python.exe" scripts/terminal_qualification/pywinpty_probe.py --artifact-manifest "$Task22512Row\artifacts.json" --json-out "$Task22512Row\winpty.json"
```

Expected: each unique row directory retains the downloaded artifacts, SHA-256 manifest, exact resolved dependency set, and separately named environment/pyte/winpty results. Installs resolve exactly to pyte 0.8.2 and pywinpty 3.0.5, all mandatory ConPTY rows pass, and no legacy/high-level/pipe fallback is observed.

Expected: every supported row passes. Missing wheels or a failed mandatory behavior must be recorded as fail-closed and blocks claiming that platform/architecture. Pilot or mocked evidence does not qualify.

- [ ] **Step 7: Complete the artifact and verify GREEN**

For every row, run `common.py collect-row --row-dir <unique-row-dir> --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw`; it validates the content-free schema and copies the uniquely named JSON into `raw/<row-id>/` without copying wheels, environments, output, or profiles. The Markdown evidence builder records each raw file's SHA-256. The artifact must include exact commands, dates, hosts, hashes, versions, licenses/notices, wheel matrix, API source references, environment key sets, result tables, and limitations. This plan prohibits adapting `textual-terminal` source in v1; record `textual-terminal source adaptation: none` so attribution cannot become a late, implicit decision. It must contain no terminal content or secret values.

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal/test_dependency_qualification.py \
  Tests/Terminal/test_format_ratchet.py -q
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 8: Commit the qualification gate**

```bash
git add scripts/terminal_qualification/common.py \
  scripts/terminal_qualification/pyte_probe.py \
  scripts/terminal_qualification/pywinpty_probe.py \
  scripts/terminal_qualification/environment_probe.py \
  scripts/terminal_qualification/format_ratchet.py \
  scripts/terminal_qualification/README.md \
  Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md \
  Docs/superpowers/reviews/evidence/task-22512/format-baseline.json \
  Docs/superpowers/reviews/evidence/task-22512/raw \
  Tests/Terminal/test_dependency_qualification.py \
  Tests/Terminal/test_format_ratchet.py
git commit -m "test: qualify persistent terminal dependencies"
```

### Task 2: Admit only the qualified dependency versions

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `Tests/Terminal/test_dependency_qualification.py`

- [ ] **Step 1: Extend the dependency contract test and verify RED**

Parse `pyproject.toml` with `tomllib` and `requirements.txt` as requirement lines. Require one unconditional exact pyte pin and one Windows-only exact pywinpty pin:

```python
assert "pyte==0.8.2" in core_dependencies
assert any(
    value.startswith("pywinpty==3.0.5") and "sys_platform == 'win32'" in value
    for value in core_dependencies
)
```

Also assert the artifact names the same versions and hashes.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_dependency_qualification.py -q
```

Expected: the test collects and reaches the exact metadata assertion for the absent pyte or Windows-marked pywinpty pin. Import, collection, or evidence-artifact failures do not count.

- [ ] **Step 2: Add the exact pins**

Add `pyte==0.8.2` to core dependencies and `pywinpty==3.0.5; sys_platform == 'win32'` as a Windows-only core dependency in both dependency sources. Do not add a POSIX pywinpty fallback or a high-level PTY wrapper.

- [ ] **Step 3: Install only the admitted packages without retargeting the shared editable checkout**

The repository development venv is shared by sibling worktrees. Never run `pip install -e .` from this worktree. Install only the new exact package into that interpreter, keep the existing editable target untouched, and select this worktree explicitly for provenance checks.

On POSIX:

```bash
../../.venv/bin/python -m pip install pyte==0.8.2 "wcwidth>=0.2.14,<1"
../../.venv/bin/python -m pip check
PYTHONPATH="$PWD" ../../.venv/bin/python -c "from importlib.metadata import version; from pathlib import Path; import pyte, tldw_chatbook; assert version('pyte') == '0.8.2'; assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd().resolve()); print(Path(pyte.__file__).resolve())"
PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/test_probe_import_provenance.py -q
```

Expected: install and `pip check` pass, pyte is exactly 0.8.2, Chatbook imports from this worktree, and the shared environment's editable Chatbook target has not changed. Task 1's isolated POSIX row—not this potentially pre-populated shared venv—proves the Windows marker does not install pywinpty.

On Windows, run the equivalent from the worktree and require both exact versions:

```powershell
..\..\.venv\Scripts\python.exe -m pip install pyte==0.8.2 pywinpty==3.0.5 "wcwidth>=0.2.14,<1"
..\..\.venv\Scripts\python.exe -m pip check
$env:PYTHONPATH = (Get-Location).Path
..\..\.venv\Scripts\python.exe -c "from importlib.metadata import version; from pathlib import Path; import tldw_chatbook; assert version('pyte') == '0.8.2'; assert version('pywinpty') == '3.0.5'; assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd().resolve())"
..\..\.venv\Scripts\python.exe -B -m pytest Tests/test_probe_import_provenance.py -q
```

Expected: both packages resolve exactly and Chatbook imports from the Windows worktree.

- [ ] **Step 4: Verify packaging GREEN**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal/test_dependency_qualification.py \
  Tests/CI/test_textual_runtime_contract.py -q
```

Expected: focused dependency metadata tests pass.

- [ ] **Step 5: Commit the admitted pins**

```bash
git add pyproject.toml requirements.txt Tests/Terminal/test_dependency_qualification.py
git commit -m "build: admit qualified terminal dependencies"
```

### Task 3: Define terminal contracts, limits, transitions, and backend protocol

**Files:**
- Create: `tldw_chatbook/Terminal/__init__.py`
- Create: `tldw_chatbook/Terminal/contracts.py`
- Create: `tldw_chatbook/Terminal/backend.py`
- Create: `Tests/Terminal/test_contracts.py`

- [ ] **Step 1: Write all contract, transition, and deadline tests before implementation**

Pin the constants and prove lifecycle, terminal reason, exit code, `stream_closed`, and `output_complete` do not imply one another. In the same RED slice, cover reservation failure release, running-to-draining shell exit, nonzero ordinary exit, parser failure reason, closing-to-closed/cleanup-unproven, receipt Retry, forbidden transitions, and proof that only explicit Retry creates a new T0:

```python
def test_terminal_limits_match_adr_099() -> None:
    assert MAX_SESSION_RECORDS == 4
    assert (MIN_COLUMNS, MAX_COLUMNS) == (5, 300)
    assert (MIN_ROWS, MAX_ROWS) == (2, 120)
    assert MAX_SCROLLBACK_LINES == 5_000
    assert MAX_SCROLLBACK_BYTES == 4 * 1024 * 1024
    assert MAX_PENDING_INPUT_BYTES == 512 * 1024
    assert MAX_PENDING_OUTPUT_BYTES == 512 * 1024
    assert MAX_PASTE_BYTES == 256 * 1024
    assert MAX_IO_CHUNK_BYTES == 64 * 1024
    assert MAX_PARSER_TURN_BYTES == 256 * 1024
    assert MAX_PARSER_TURN_SECONDS == 0.008
    assert CleanupSchedule().proof_reserve_seconds == 1.25


def test_lifecycle_and_terminal_reason_vocabularies_match_the_design() -> None:
    assert {item.value for item in TerminalLifecycle} == {
        "reserved", "creating", "admitting", "running", "draining",
        "exited", "closing", "closed", "cleanup_unproven",
    }
    assert {item.value for item in TerminalReason} == {
        "locked", "unarmed", "session_limit", "invalid_name",
        "invalid_start_directory", "shell_unavailable", "backend_unavailable",
        "admission_failed", "spawn_failed", "input_backpressure",
        "terminal_protocol_failed", "io_failed", "worker_failed",
        "output_incomplete", "cleanup_unproven",
    }


def test_exited_does_not_claim_stream_or_output_completion() -> None:
    projection = replace(
        running_projection(),
        lifecycle=TerminalLifecycle.EXITED,
        exit_code=0,
        stream_closed=False,
        output_complete=False,
    )
    assert projection.exit_code == 0
    assert projection.stream_closed is False
    assert projection.output_complete is False
```

- [ ] **Step 2: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
```

Expected: the test collects against the minimal contracts skeleton and reaches an assertion failure because transition validation, cleanup offsets, or immutable value semantics still return neutral placeholders. Import or collection failure does not count.

- [ ] **Step 3: Implement minimal value contracts**

Use `str, Enum` values for `TerminalLifecycle` and `TerminalReason`, frozen slotted dataclasses for request/event/projection values, and one pure transition validator. Include the approved cleanup boundaries as absolute offsets:

```python
@dataclass(frozen=True, slots=True)
class CleanupSchedule:
    deadline_seconds: float = 5.0
    hangup_no_later_than: float = 0.75
    terminate_no_later_than: float = 2.25
    force_kill_no_later_than: float = 3.75
    proof_reserve_seconds: float = 1.25


class TerminalBackend(Protocol):
    def start(self, request: TerminalLaunchRequest, admission: AdmissionGate) -> BackendIdentity: ...
    def write(self, data: bytes) -> None: ...
    def resize(self, columns: int, rows: int) -> None: ...
    def request_priority_close(self) -> None: ...
    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof: ...
```

Do not put subprocess handles, raw bytes, environment mappings, or mutable screen objects in UI projections.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Terminal Tests/Terminal/test_contracts.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/Terminal/__init__.py \
  tldw_chatbook/Terminal/contracts.py \
  tldw_chatbook/Terminal/backend.py \
  Tests/Terminal/test_contracts.py
git commit -m "feat: define persistent terminal contracts"
```

### Task 4: Build shell, name, directory, argv, and scrubbed-environment boundaries

**Files:**
- Create: `tldw_chatbook/Terminal/launch.py`
- Create: `Tests/Terminal/test_launch.py`
- Reference: `tldw_chatbook/Tools/raw_cli_executor.py`
- Reference: `tldw_chatbook/Chat/console_project_instructions.py`
- Reference: `tldw_chatbook/UI/Console_Modules/wiring.py::_raw_cli_selected_local_root`

- [ ] **Step 1: Write RED tests for names and fixed-family shell discovery**

Cover NFC normalization, trim, 1-64 display characters, control/markup refusal, Unicode-casefold uniqueness, POSIX account-shell fallback to Bash then `sh`, Windows `pwsh` then Windows PowerShell then CMD, and no arbitrary executable picker entry. Assert argv contains no command string, `-NoProfile`, `-NonInteractive`, `/C`, or caller-supplied argument.

- [ ] **Step 2: Write RED tests for starting-directory resolution**

The existing Console wiring seam already resolves the current session's selected `working_folder_binding_id` to a still-usable local folder. Generalize `_raw_cli_selected_local_root` to a neutral `_selected_console_local_root` and keep raw CLI using it. The pure launch resolver receives that late-bound `Path | None`; it uses the selected root when present and otherwise the real OS account home. The New Session form may supply another path, but launch revalidates the final absolute existing directory immediately before backend admission.

```python
def test_selected_ready_local_root_precedes_real_home(tmp_path, fake_home) -> None:
    assert resolve_start_directory(tmp_path, account_home=fake_home) == tmp_path


def test_missing_or_nonlocal_selection_falls_back_to_account_home(fake_home) -> None:
    assert resolve_start_directory(None, account_home=fake_home) == fake_home
```

Task 13 adds the wiring behavior test that proves only a selected, same-workspace, ready local-filesystem directory reaches this function. This is convenience only. Do not call workspace file-tool confinement helpers and do not retain a claim that later `cd` stays under the starting directory.

- [ ] **Step 3: Write RED tests for the dedicated environment allowlist**

Parameterize POSIX and Windows. Seed ambient mappings with provider keys, proxy values, tracing fields, `PYTHONPATH`, `PYTHONHOME`, credential-agent sockets, and unrelated values; prove none enter the result. Prove account/OS values come from injected platform readers rather than the caller mapping. Assert `TERM=linux`, no `COLORTERM`, no inherited rows/columns, and no ambient `PSModulePath`.

- [ ] **Step 4: Write fresh-profile integration RED tests**

Use temporary HOME/profile fixtures in fresh subprocesses to prove the standard Bash/Zsh profile path runs, the starting directory is correct, and scrubbed ambient sentinels are absent until explicitly restored by the profile. Add Windows-native PowerShell/CMD profile/module rows to the qualification probe rather than pretending to verify them on POSIX.

- [ ] **Step 5: Verify the complete launch slice RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_launch.py -q
```

Expected: the test collects against the launch skeleton and reaches assertions showing that environment stripping, starting-directory resolution, or profile launch behavior still returns the explicit refusal placeholder. Import or collection failure does not count.

- [ ] **Step 6: Implement the smallest launch boundary**

Keep this module pure and dependency-injected. It may reuse audited validation/identity primitives but must not call `_build_shell_environment` or import the one-shot executor as its environment builder. Export immutable `ShellChoice` and `ResolvedLaunch` values; all argv remains code-owned.

- [ ] **Step 7: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_launch.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/launch.py Tests/Terminal/test_launch.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/Terminal/launch.py Tests/Terminal/test_launch.py
git commit -m "feat: define terminal launch boundary"
```

### Task 5: Implement the bounded terminal protocol gate and safe screen model

**Files:**
- Create: `tldw_chatbook/Terminal/protocol_gate.py`
- Create: `tldw_chatbook/Terminal/screen_model.py`
- Create: `Tests/Terminal/test_protocol_gate.py`
- Create: `Tests/Terminal/test_screen_model.py`

- [ ] **Step 1: Write cross-chunk hostile-sequence RED tests**

Test complete and incomplete 4 KiB caps, CSI 32-parameter/four-digit/9,999 clamps, 16-byte private/intermediate cap, 16-byte non-CSI ESC cap, OSC/DCS/PM/APC termination by ST/BEL/CAN/SUB/reset, and recovery when a terminator arrives in a later chunk. Assert rejected payload text never appears in reasons or logs.

- [ ] **Step 2: Write safe-screen and qualification-corpus RED tests**

Cover incremental invalid UTF-8 replacement; ASCII, wide, combining, and joiner cells; 32-scalar/256-byte per-cell caps; 16 cursor savepoints; colors/styles; alternate-screen exclusion from scrollback; exact accounting of UTF-8 text bytes plus 32 bytes per style run plus 16 bytes per retained line; oldest-first 5,000-line/4 MiB eviction; ignored title/icon/clipboard/hyperlink/notification OSC; and bounded allowlisted device replies. Parameterize every parser corpus fixture named in the qualification artifact now, before implementation, including the assertion that raw ESC/C0/C1 payload never reaches a Rich/Textual renderable and parser failure returns only a content-free category.

```python
def test_osc_clipboard_payload_never_reaches_safe_cells_or_reply_queue() -> None:
    model = TerminalScreenModel(columns=80, rows=24)
    model.feed(b"before\x1b]52;c;SECRET\x07after")
    assert model.visible_text() == "beforeafter"
    assert model.pending_replies() == ()
    assert "SECRET" not in repr(model.snapshot())
```

- [ ] **Step 3: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_protocol_gate.py -q
../../.venv/bin/python -B -m pytest Tests/Terminal/test_screen_model.py -q
```

Expected: each test collects against its minimal skeleton and reaches its own assertion failure: the gate still refuses all sequences, and the model still returns an empty neutral projection. Import, collection, or a failure in the other test file does not count.

- [ ] **Step 4: Implement the pre-parser before pyte**

The gate holds only bounded sequence classification state and never retains discarded payloads. Feed only admitted screen/query operations into the qualified pyte stream. Unknown operations are ignored; an internal invariant failure emits `terminal_protocol_failed` to the owner instead of raising raw terminal data through Textual.

- [ ] **Step 5: Implement safe immutable projections and bounded scrollback**

Use run-compressed `SafeTerminalLine`/`SafeTerminalCell` values. The model owns normal/alternate screens, cursor/modes, scrollback accounting, dirty-line generation, and fixed reply requests; it knows nothing about Textual widgets or subprocesses.

- [ ] **Step 6: Rerun the already-authored qualification corpus**

Replay every parser corpus fixture named in the qualification artifact. Assert no raw ESC/C0/C1 payload reaches a Rich/Textual renderable and parser failure returns a content-free category.

- [ ] **Step 7: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal/test_protocol_gate.py \
  Tests/Terminal/test_screen_model.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Terminal/protocol_gate.py \
  tldw_chatbook/Terminal/screen_model.py \
  Tests/Terminal/test_protocol_gate.py \
  Tests/Terminal/test_screen_model.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/Terminal/protocol_gate.py \
  tldw_chatbook/Terminal/screen_model.py \
  Tests/Terminal/test_protocol_gate.py \
  Tests/Terminal/test_screen_model.py
git commit -m "feat: add bounded terminal screen model"
```

### Task 6: Add bounded input/output actors and priority control

**Files:**
- Create: `tldw_chatbook/Terminal/io_actors.py`
- Create: `Tests/Terminal/test_io_actors.py`

- [ ] **Step 1: Write ordered-input and atomic-paste RED tests**

Prove one actor orders key bytes, complete paste events, fixed replies, and latest-only resize. A paste over 256 KiB, over remaining 512 KiB input credit, or containing NUL/ESC/DEL/prohibited C0/C1 is refused atomically. Tab/CR/LF remain allowed. Refusal reasons contain only the control class and never content. When bracketed paste is active, markers are added only after validation and credit reservation.

```python
def test_prohibited_paste_is_refused_before_any_bytes_are_enqueued() -> None:
    actor = TerminalInputActor(capacity_bytes=MAX_PENDING_INPUT_BYTES)
    result = actor.offer_paste("safe\x1b[201~forged", bracketed=True)
    assert result.reason is PasteRefusalReason.PROHIBITED_CONTROL
    assert actor.pending_bytes == 0
    assert actor.take_nowait() is None
    assert "forged" not in result.safe_message
```

- [ ] **Step 2: Write output-credit, parser-budget, priority-close, race, and flood RED tests**

Prove output chunks are at most 64 KiB, decoded-but-unparsed bytes stop at 512 KiB, full credit pauses the next backend read rather than dropping bytes, one parser turn stops at 256 KiB or injected-clock 8 ms, and only one visible refresh is scheduled per frame. Saturate the input/output paths and prove `request_priority_close()` is still immediate and idempotent. Use barriers rather than sleeps for admission/credit races. Add the synthetic ten-second ANSI flood harness before implementation; developer runs report 100 ms sentinel/p95 measurements, while qualification-host tests apply the approved `<100 ms` threshold.

- [ ] **Step 3: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_io_actors.py -q
```

Expected: the test collects against the actor skeleton and reaches assertions showing that byte limits, credit, priority-close, or parser budgets still return neutral/refusal behavior. Import or collection failure does not count.

- [ ] **Step 4: Implement actors over byte-counted bounded deques**

Do not use an unbounded `asyncio.Queue` or thread queue. Store immutable event envelopes with precomputed encoded size. Resize is a separate latest-only slot debounced one loop turn and no more than 50 ms. Fixed replies are at most 256 bytes and rate-limited to 4 KiB/s. Priority close is an independent event/pipe/handle selected ahead of ordinary input.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_io_actors.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/io_actors.py Tests/Terminal/test_io_actors.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/Terminal/io_actors.py Tests/Terminal/test_io_actors.py
git commit -m "feat: bound terminal io actors"
```

### Task 7: Implement the app-global session manager and cleanup coordinator

**Files:**
- Create: `tldw_chatbook/Terminal/session_manager.py`
- Create: `Tests/Terminal/test_session_manager.py`
- Reference: `tldw_chatbook/Chat/console_raw_cli.py`

- [ ] **Step 1: Write authority truth-table RED tests**

Construct with a strict persisted-unlock reader and injected backend factory. Prove discoverability while locked, false launch arm, independent raw-CLI/Terminal arm bits, no arm when the saved value is string/integer truthy, an explicit first-arm disclosure acknowledgement, immediate launch-local disarm, and fresh-manager reset. The manager remembers only a process-memory `disclosure_acknowledged` bit: after a user has completed the full disclosure once in this Chatbook launch, a later re-arm does not repeat it; a new manager requires it again.

```python
def test_terminal_arm_is_independent_and_resets_per_manager() -> None:
    raw = RawCliRuntime(lambda: True)
    terminal = TerminalSessionManager(read_permitted=lambda: True, backend_factory=fake)
    assert raw.arm().armed is True
    assert terminal.armed is False
    assert terminal.arm(acknowledge_disclosure=True).armed is True
    raw.disarm()
    assert terminal.armed is True
    assert TerminalSessionManager(lambda: True, fake).armed is False
```

Do not share mutable arm state with `RawCliRuntime`, register a tool, or route through the MCP permission store.

- [ ] **Step 2: Write atomic four-record admission and naming RED tests**

Race more than four create calls behind a barrier. Assert exactly four reservations, no backend call before reservation, pre-launch failure releases a slot, and running/exited/closing/cleanup-unproven records retain slots. Prove casefolded normalized names stay unique across retained records.

- [ ] **Step 3: Write lifecycle, shell-exit, cleanup, and Retry RED tests**

Use an injected monotonic clock. Prove shell exit starts draining at one T0; hangup/terminate/force-kill/proof boundaries are absolute; backend waits cannot reset them; Disarm/Shutdown give all sessions one global T0 and join earlier attempts with the earlier deadline; only explicit Retry gets a fresh five seconds. Prove healthy drain continues concurrently, parser invariant failure immediately disables input and starts priority cleanup with `terminal_protocol_failed`, and cleanup uses a content-free raw drain only after process death is otherwise proven.

- [ ] **Step 4: Write ownership and projection RED tests**

Prove one manager owns mutable sessions/backends/models; views receive immutable projections only. `attach_view` returns a generation token; detached/stale tokens cannot resize, focus, repaint, close, or restart a session. Destroy/remount fake views while output continues and assert backend identity is unchanged.

- [ ] **Step 5: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_session_manager.py -q
```

Expected: the test collects against the manager skeleton and reaches lifecycle, deadline, ownership, or stale-generation assertions while the manager still returns neutral/refusal projections. Import or collection failure does not count.

- [ ] **Step 6: Implement the manager without platform branches in UI-facing methods**

The manager owns one lock for authority/reservation mutation, per-session actors and parser state, an injected backend factory, selected session ID, and content-free subscriptions. Use one authoritative backend cleanup task per session. `disarm()` first flips the arm false, then starts one concurrent cleanup cohort; cleanup receipts and Retry remain callable while locked/unarmed.

- [ ] **Step 7: Add shutdown/failure and memory-accounting seams**

Expose a bounded async `shutdown(deadline_seconds=5.0)` and a test-only managed-process inventory for RSS measurement. The inventory may expose PIDs/birth identities to tests but not UI/model projections or diagnostics.

- [ ] **Step 8: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal/test_contracts.py \
  Tests/Terminal/test_io_actors.py \
  Tests/Terminal/test_session_manager.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Terminal Tests/Terminal
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/Terminal/session_manager.py Tests/Terminal/test_session_manager.py
git commit -m "feat: own persistent terminal sessions"
```

### Task 8: Implement and prove the POSIX controlling-PTY backend

**Files:**
- Create: `tldw_chatbook/Terminal/posix_launcher.py`
- Create: `tldw_chatbook/Terminal/posix_backend.py`
- Create: `Tests/Terminal/test_posix_backend.py`
- Create: `Tests/fixtures/terminal/descendant_holds_tty.py`
- Create: `Tests/fixtures/terminal/job_control_tree.py`
- Create: `Tests/fixtures/terminal/terminal_child.py`
- Create: `Tests/fixtures/terminal/posix_app_crash_probe.py`
- Reference: `tldw_chatbook/STT/executor_process_tree.py`
- Reference: `tldw_chatbook/Tools/raw_cli_executor.py`

- [ ] **Step 1: Write launcher-order RED tests with a real child process**

Use an admission pipe/socketpair and a report pipe. First add an AST/source contract that prohibits parent-side `os.fork`, `pty.fork`, and `preexec_fn`. Prove the parent launches a fresh Python executable helper with `subprocess.Popen(..., pass_fds=...)`; only the fresh helper calls `setsid`, reports PID/birth/SID/initial PGID, and remains blocked before controlling-terminal acquisition/`exec`. Refusal closes the slave and exits without executing the profile sentinel. Admission lets that same helper PID acquire the slave, duplicate standard streams, close unrelated descriptors, and `exec` into the shell without changing PID.

Fresh-process tests must construct spawn primitives while `sys.stderr` has a real file descriptor, per `lessons-live-verification.md`.

- [ ] **Step 2: Write interactive PTY RED tests**

Prove controlling TTY, retained `cd` and exported environment state, Unicode round trip, nonblocking input/output, resize plus SIGWINCH, alternate screen, EOF, and exact shell exit. Use real executable fixtures; a Python function double is not enough for descriptor inheritance.

- [ ] **Step 3: Write job-control ownership and cleanup RED tests**

Exercise foreground/background process groups, an exited group leader, numeric PGID reuse simulation, mixed/unrelated membership, enumeration denial, same-session descendants, tracked descendants that change group, and a descendant retaining the slave after shell exit. Group signalling is allowed only with same-birth leader and complete exclusively-owned membership; otherwise only individually revalidated PID+birth identities may be signalled.

- [ ] **Step 4: Write death-proof and crash RED tests**

Require exact shell reap by the sole reaper, PTY EOF, and two zero-owned-process scans at least 50 ms apart inside the proof reserve. Access denial, identity mismatch, or incomplete enumeration returns `cleanup_unproven`. A separate app-crash subprocess proves ordinary master-close SIGHUP cleanup and records the deliberate detached-process limitation without misclassifying it as containment.

- [ ] **Step 5: Verify RED**

Run on POSIX:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q
```

Expected: the test collects against the POSIX backend skeleton and reaches an admission, PTY-control, cleanup, or death-proof assertion while launch remains fail-closed. Import or collection failure does not count. Windows collection must skip the real POSIX tests at module/test marker level without importing unsupported APIs.

- [ ] **Step 6: Implement the admitted launcher and parent backend**

Use `os.openpty`/`pty.openpty`, `fcntl` nonblocking master I/O, `termios`/`ioctl(TIOCSWINSZ)`, and code-owned launcher steps. The multithreaded Chatbook parent must use ordinary `subprocess.Popen([sys.executable, "-m", "tldw_chatbook.Terminal.posix_launcher", ...], pass_fds=...)` with no `preexec_fn`, direct fork, or Python callback between fork and exec. The fresh helper process performs `setsid`, gated identity/admission, `TIOCSCTTY`, descriptor duplication/closure, and `os.execve` of the shell. Reuse `psutil` and narrow process identity concepts where they fit, but extend ownership to SID plus validated descendants/job-control groups. Never call a broad numeric `killpg` without the approved membership proof.

- [ ] **Step 7: Verify GREEN with fresh-process evidence**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal/test_posix_backend.py \
  Tests/Terminal/test_launch.py \
  Tests/Terminal/test_session_manager.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Terminal/posix_backend.py \
  tldw_chatbook/Terminal/posix_launcher.py \
  Tests/Terminal/test_posix_backend.py \
  Tests/fixtures/terminal
git diff --check
```

Expected: PASS on macOS/Linux; supported platform rows and exact commands are appended to the qualification artifact.

- [ ] **Step 8: Commit the POSIX backend**

```bash
git add tldw_chatbook/Terminal/posix_backend.py \
  tldw_chatbook/Terminal/posix_launcher.py \
  Tests/Terminal/test_posix_backend.py \
  Tests/fixtures/terminal/descendant_holds_tty.py \
  Tests/fixtures/terminal/job_control_tree.py \
  Tests/fixtures/terminal/terminal_child.py \
  Tests/fixtures/terminal/posix_app_crash_probe.py \
  Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md
git commit -m "feat: add admitted posix terminal backend"
```

### Task 9: Implement and prove the Windows low-level ConPTY backend

**Files:**
- Create: `tldw_chatbook/Terminal/windows_job.py`
- Create: `tldw_chatbook/Terminal/windows_worker.py`
- Create: `tldw_chatbook/Terminal/windows_backend.py`
- Create: `Tests/Terminal/test_windows_job.py`
- Create: `Tests/Terminal/test_windows_backend_contract.py`
- Create: `Tests/Terminal/test_windows_backend_native.py`
- Create: `Tests/fixtures/terminal/windows_app_crash_probe.py`
- Modify: `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`

- [ ] **Step 1: Write host-independent import/API RED tests**

Parse the Windows modules with AST on every platform. Require lazy platform imports, `multiprocessing.get_context("spawn")`, low-level `winpty.PTY`, explicit `winpty.Backend.ConPTY`, and no `PtyProcess`, legacy Backend, `subprocess.PIPE`, inherited Job handle, or import-time pywinpty side effect.

- [ ] **Step 2: Write injected Job/worker protocol RED tests**

With fake narrow Win32 calls and a spawned worker double, prove create-suspended/gated worker, parent Job creation with kill-on-close, non-inheritable handle, Job assignment before admission release, backend identity refusal, one-credit 64 KiB output, bounded control messages, priority close independent of blocked read, worker crash, and retained cleanup authority.

- [ ] **Step 3: Write native Windows tests before backend implementation**

On a supported Windows host, author `test_windows_backend_native.py` and the crash fixture before product code. Use the real qualified pywinpty pin and real Job APIs to cover explicit ConPTY identity, worker/shell/helper membership, profiles, CMD/PowerShell, Unicode, resize, alternate screen, one-credit backpressure, concurrent blocked-read control, shell exit with descendant-held stream, bounded missing EOF, close ladder, worker failure, descendant cleanup, and app crash. Unsupported OS/version/architecture skips must use the content-free reason recorded in the qualification artifact.

- [ ] **Step 4: Capture host-independent and native Windows RED runs**

Run the host-independent tests on the development host:

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Terminal/test_windows_job.py -q
../../.venv/bin/python -B -m pytest Tests/Terminal/test_windows_backend_contract.py -q
```

Expected: each host-independent test collects against its minimal skeleton and reaches its own Job-admission or backend-protocol assertion while the implementation remains fail-closed. Import or collection failure does not count; these tests must remain runnable on POSIX after implementation.

Run the same pre-implementation slice on a supported Windows host:

```powershell
..\..\.venv\Scripts\python.exe -B -m pytest Tests/Terminal/test_windows_backend_native.py -q
..\..\.venv\Scripts\python.exe -B -m pytest Tests/Terminal/test_windows_backend_contract.py -q
..\..\.venv\Scripts\python.exe -B -m pytest Tests/Terminal/test_windows_job.py -q
```

Expected: each native test collects against the same minimal skeleton and reaches an assertion for missing ConPTY/Job behavior, not an import, collection, absent-pywinpty, or foreign-checkout failure. Run native test files independently if one failure could mask another, and record each exact RED output in the qualification artifact.

- [ ] **Step 5: Implement the narrow Job Object owner and worker protocol**

`windows_job.py` owns only reviewed ctypes declarations and safe-handle closure. The parent retains the Job and waitable process handles. `windows_worker.py` imports pywinpty only after the admitted control message, constructs `PTY(backend=Backend.ConPTY)`, owns blocking read on one dedicated thread, and services write/resize/`cancel_io`/close on an independent bounded path. The worker never owns or inherits the kill-on-close Job handle.

- [ ] **Step 6: Verify host-independent GREEN**

Run the host-independent command from Step 4 again.

Expected: PASS on POSIX and Windows without creating ConPTY in the contract tests.

- [ ] **Step 7: Verify native Windows GREEN**

Run in PowerShell from the worktree:

```powershell
..\..\.venv\Scripts\python.exe -B -m pytest `
  Tests/Terminal/test_windows_backend_native.py `
  Tests/Terminal/test_windows_backend_contract.py `
  Tests/Terminal/test_windows_job.py -q
```

Expected: PASS on every claimed Windows row. Do not merge a Windows-support claim based only on mocks or Wine.

- [ ] **Step 8: Verify fresh import order and packaging**

Spawn a fresh interpreter that imports `windows_worker` before `app`, then the reverse. Prove no circular import and no foreign-worktree import. Confirm the Windows wheel installs from the project marker and POSIX does not install it.

- [ ] **Step 9: Commit the Windows backend and evidence**

```bash
git add tldw_chatbook/Terminal/windows_job.py \
  tldw_chatbook/Terminal/windows_worker.py \
  tldw_chatbook/Terminal/windows_backend.py \
  Tests/Terminal/test_windows_job.py \
  Tests/Terminal/test_windows_backend_contract.py \
  Tests/Terminal/test_windows_backend_native.py \
  Tests/fixtures/terminal/windows_app_crash_probe.py \
  Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md
git commit -m "feat: add admitted windows conpty backend"
```

### Task 10: Make the manager app-owned and shutdown-bounded

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`
- Modify: `Tests/Chat/test_console_runtime_lifetime.py`

- [ ] **Step 1: Write app-ownership and launch-reset RED tests**

Extend the existing ownership tests. Require one `TerminalSessionManager` construction after config load, false initial arm despite a true saved unlock, a live config reader rather than a captured config object, and object identity across Console screen visits.

- [ ] **Step 2: Write shutdown-once/order/deadline RED tests**

Use a barrier to call terminal shutdown concurrently from both app shutdown paths and assert one shared task. Cancel one waiter and prove the underlying cleanup continues to completion for the other waiter. Inspect `_shutdown_app_owned_lifecycles` to require terminal producer shutdown before Console runtime disposal and before later profile/resource teardown. With an injected clock/backend cohort, assert one five-second global deadline and final handle closure at the deadline.

```python
@pytest.mark.asyncio
async def test_terminal_manager_shutdown_is_shared_and_shielded_from_waiter_cancel():
    first = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    second = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    await manager_entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release_manager.set()
    await second
    assert calls == ["terminal"]
    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    assert source.index("_shutdown_terminal_session_manager") < source.index("_shutdown_console_runtime")
```

- [ ] **Step 3: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_runtime_ownership.py -q
../../.venv/bin/python -B -m pytest Tests/Chat/test_console_runtime_lifetime.py -q
```

Expected: each test collects and independently reaches its new terminal ownership or shutdown-order assertion while the skeleton manager remains absent from app lifetime. Import, collection, or a failure in the other file does not count.

- [ ] **Step 4: Wire the app-owned manager**

Construct it beside—but not inside—`RawCliRuntime` and `ConsoleRuntime`. Reuse the strict `_read_app_raw_cli_permitted` reader for the shared persisted unlock, but keep a separate `terminal_armed` field inside the new manager. Platform backend imports must remain lazy so unsupported systems fail closed without app import failure.

- [ ] **Step 5: Add idempotent app shutdown**

Create `_terminal_session_manager_shutdown_task` once and await it through `asyncio.shield`, following the existing app-owned lifecycle pattern so cancellation of one caller cannot cancel cleanup for all callers. Both Textual shutdown paths must share this task. On final deadline, invoke backend final-handle closure without another wait window.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/Chat/test_console_runtime_lifetime.py \
  Tests/Terminal/test_session_manager.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/Terminal Tests/UI/test_console_runtime_ownership.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/app.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_runtime_lifetime.py
git commit -m "feat: own terminal manager at app lifetime"
```

### Task 11: Expose the shared unlock and independent Terminal arm in Settings

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_raw_cli.py`
- Create: `Tests/UI/test_settings_terminal.py`

- [ ] **Step 1: Write authority-copy and independent-control RED tests**

Mount the canonical F9 Settings screen. Require one saved checkbox backed by `[console] raw_cli_permitted` and label it clearly as the shared unlock, for example `Allow raw CLI and Terminal host access`. Under it, require two separately named launch controls:

- `One-shot raw CLI` with its existing arm state/action;
- `Persistent Terminal` with its own `Locked` / `Unlocked, not armed` / `ARMED — HOST TERMINAL` state/action.

Prove saving the checkbox changes both runtimes' eligibility but arms neither; arming one does not arm the other; each resets when a new app instance is built.

- [ ] **Step 2: Write danger-disclosure and confirmation RED tests**

Require Terminal's first-arm-per-launch confirmation to state full OS-user authority, normal profile/startup-file execution, profile ability to restore secrets/agents/proxies, shell history/side effects, active workspace/home as a starting directory only, no sandbox/confinement, and cleanup limitations. The first confirmed arm passes the explicit acknowledgement to the manager. A later re-arm in the same app process does not repeat the full modal, and creating sessions never repeats it. A new app process requires the disclosure again. While armed, the Settings card and Console surface both use persistent red `HOST TERMINAL - FULL USER ACCESS` treatment.

The confirmation body is user-visible static copy. Tests must not derive it from environment values or session output.

- [ ] **Step 3: Write disarm and cleanup-receipt RED tests**

Disarming Terminal confirms once for all live sessions, flips authority off immediately, starts one parallel five-second cleanup cohort, and reports whether cleanup is pending/unproven. Saving the shared persistent unlock false performs the same Terminal revocation/cleanup in addition to the existing raw-CLI revocation. Raw CLI commands are not cancelled by Terminal-only disarm. Terminal cleanup receipts remain visible/actionable in Terminal while the shared unlock is false.

- [ ] **Step 4: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_settings_raw_cli.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_settings_terminal.py -q
```

Expected: the existing raw-CLI test reaches only the newly added shared-unlock assertions, and the Terminal test independently reaches its new control/copy assertion while the minimal screen hooks still refuse. Existing raw-CLI assertions remain green; import, collection, or cross-file masking does not count.

- [ ] **Step 5: Implement the shared card without a new persisted field**

Refactor only enough shared helpers to avoid duplicated draft/save logic. Keep `RAW_CLI_PERMITTED_DRAFT_KEY`; do not add `terminal_armed`, disclosure acknowledgement, session names, paths, or terminal state to config. Add `_terminal_runtime`, `_terminal_is_armed`, `_terminal_state_text`, `_terminal_arm_button_state`, refresh, arm, and disarm handlers parallel to the raw helpers.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_settings_raw_cli.py \
  Tests/UI/test_settings_terminal.py \
  Tests/UI/test_settings_configuration_hub.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_raw_cli.py Tests/UI/test_settings_terminal.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_raw_cli.py Tests/UI/test_settings_terminal.py
git commit -m "feat: add independent terminal launch arm"
```

### Task 12: Build the user-only Terminal workspace widget and controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/terminal.py`
- Create: `tldw_chatbook/Widgets/Console/console_terminal_workspace.py`
- Create: `tldw_chatbook/Widgets/Console/console_terminal_session_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Reference: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Reference: `Tests/UI/test_console_controller_wiring.py`
- Create: `Tests/UI/test_console_terminal_workspace.py`
- Create: `Tests/UI/test_console_terminal_keyboard.py`
- Create: `Tests/UI/test_console_terminal_session_modal.py`

- [ ] **Step 1: Write safe projection/render RED tests**

Construct the workspace only from immutable manager projections. Require locked state plus Settings route, unlocked/unarmed state plus the same first-arm disclosure flow, persistent armed danger banner, selected name/lifecycle/shell/starting-directory, dimensions/clamping, four-record list, New/Rename/Focus/Close/Retry/Jump-live actions by state, content-free refusal/status copy, and safe styled cells. Closing a running/draining session confirms termination; closing an exited session does not confirm. Assert no widget field contains a backend, process handle, environment mapping, raw output bytes, or parser object.

- [ ] **Step 2: Write creation/rename form RED tests**

New defaults to `Terminal N`, discovered `Default` shell, and the selected working-folder binding or real home. Rename exposes only name. Validate names both in the modal for feedback and again in the manager for authority. Validate the launch directory again immediately before backend admission to close mount/modal races.

- [ ] **Step 3: Write keyboard and local scrollback RED tests**

Use explicit key-event mapping tests:

- input-focused terminal forwards terminal-convention keys, including Tab and PageUp/PageDown;
- Ctrl+P, Ctrl+Q, F1, and F6 remain Chatbook globals;
- Ctrl+] is always consumed and enters local navigation mode;
- navigation Up/Down moves one line, Page keys one viewport, Home oldest, End Jump live, Enter returns focus without newline, and Tab moves through visible actions;
- wheel scrolling enters/stays in local navigation mode;
- leaving live bottom freezes the viewed position while hidden/live output continues, increments a new-output count, and Jump live clears that count without dropping screen state;
- alternate-screen local history actions are no-ops with clear status;
- no nested-program mouse sequence is emitted.

Do not add a Screen binding for terminal-convention keys; the focused viewport handles them and lets reserved globals bubble.

- [ ] **Step 4: Write view-generation and repaint RED tests**

Attach/detach/remount the controller around one live manager session. Stale generation callbacks must not resize/focus/repaint. Hidden sessions parse but schedule no widget refresh; when selected again, their accumulated safe state and new-output count project without restarting. A visible session coalesces dirty updates to one Textual frame and sends one debounced resize after remount without clearing screen/scrollback.

- [ ] **Step 5: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_terminal_workspace.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_terminal_keyboard.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_terminal_session_modal.py -q
```

Expected: each test collects against the minimal presentation skeleton and reaches its own safe-projection, keyboard-routing, or modal-state assertion while the widgets render neutral/refusal state. Import, collection, or a failure in another file does not count.

- [ ] **Step 6: Implement a thin controller and projection-only widgets**

`ConsoleTerminalController` owns only screen-local open/closed mode, selected projection subscription, view generation, first-arm confirmation routing, and modal/action routing. Its constructor accepts only named keyword accessors suitable for centralized late binding; do not construct it in `chat_screen.py` yet. It calls app-owned manager methods for arm/create/rename/focus/close/retry/input/resize. `ConsoleTerminalWorkspace` renders safe lines and owns local scrollback offset/focus state; it never launches or reaps. Both Settings and Console render disclosure from the same immutable copy constants and let the manager enforce first-arm acknowledgement. Implement renderer/input mapping fresh; this plan forbids copying or adapting `textual-terminal` source.

- [ ] **Step 7: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Console_Modules/terminal.py \
  tldw_chatbook/Widgets/Console/console_terminal_workspace.py \
  tldw_chatbook/Widgets/Console/console_terminal_session_modal.py \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py
git diff --check
```

Expected: PASS.

```bash
git add tldw_chatbook/UI/Console_Modules/terminal.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/Widgets/Console/console_terminal_workspace.py \
  tldw_chatbook/Widgets/Console/console_terminal_session_modal.py \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py
git commit -m "feat: build console terminal workspace"
```

### Task 13: Integrate Terminal into Console rails, center routing, palette, and CSS

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `tldw_chatbook/UI/console_command_provider.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_console_left_rail.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `Tests/UI/test_console_shell_regions.py`
- Modify: `Tests/UI/test_css_bundle_sync_guard.py`
- Create: `Tests/UI/test_console_terminal_integration.py`

- [ ] **Step 1: Write controller-wiring and visible-entry RED tests**

Extend `test_console_controller_wiring.py` to require `build_console_controllers` to construct `screen._terminal` with late-bound accessors for `app_instance.terminal_session_manager`, the current session ID, `_selected_console_local_root`, account home, Settings routing, center recomposition, and focus. Then add a `TerminalRequested` message to `ConsoleLeftRail` and require one visible `Terminal` button inside the existing Sessions body. Require `Console: Open Terminal` in `ConsoleCommandProvider`. Both routes call one guarded screen action; no global shortcut is added.

- [ ] **Step 2: Write center-routing and rail-preservation RED tests**

When Terminal is closed, preserve the existing `ConsoleTranscriptRegion` IDs/nesting and geometry. When open, replace only the center `#console-main-column` content with `ConsoleTerminalWorkspace`; Context and Inspector rails, handles, header, control bar, and ordinary navigation stay mounted. Opening while locked shows the explicit route to canonical Privacy & Security Settings. Opening while unlocked/unarmed shows the Terminal Arm action and full first-arm disclosure. Neither path launches automatically, and cleanup receipts/Retry remain usable without unlocking or arming.

- [ ] **Step 3: Write live navigation/remount RED tests**

With a fake backend identity and real manager, create a session, emit output, switch conversation and app screen, return/reopen Terminal, and prove process/backend identity, parsed screen, current directory state, and session list survive. Recompose rails and center repeatedly; assert no restart, duplicate read loop, duplicate reaper, or close.

- [ ] **Step 4: Write exact painted-geometry and focus RED tests**

Mount the production hierarchy with app styles at standard, 100-column, and narrow widths. Assert the Terminal center receives the actual allocated pane dimensions capped at 300x120, exposes visible clamp state, and does not overflow the workspace grid. Verify focus transition from rail action to actions/viewport, Ctrl+] release, Tab walk, and return to transcript.

- [ ] **Step 5: Verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_left_rail.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_internals_decomposition.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_controller_wiring.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_workbench_contract.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_shell_regions.py -q
../../.venv/bin/python -B -m pytest Tests/UI/test_console_terminal_integration.py -q
```

Expected: every file collects and independently reaches its new entry, routing, geometry, or remount assertion while the minimal integration hooks still refuse. Existing transcript and rail assertions remain green; import, collection, or cross-file masking does not count.

- [ ] **Step 6: Wire the thin Console integration**

Construct `screen._terminal` in `UI/Console_Modules/wiring.py::build_console_controllers`, alongside the centralized decomposed controller graph, using only late-bound accessors. Generalize the existing raw-CLI selected-root helper there and keep both consumers on that one seam. `chat_screen.py` handles the typed rail message/action and selects one zero-argument center builder only; do not construct the controller, move terminal lifecycle, or add backend branches there. The screen action opens the discoverable workspace and focuses the appropriate control; only explicit New launches.

- [ ] **Step 7: Add source CSS and regenerate the bundle**

Style the center workspace, session strip, safe viewport, clamp/new-output/status rows, receipt actions, and persistent red danger banner in `_agentic_terminal.tcss`. Preserve the existing 3fr/13fr/4fr sibling sizing and narrow-layout waivers. Regenerate—never hand-edit—the bundle:

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
```

- [ ] **Step 8: Verify UI GREEN and CSS sync**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_terminal_integration.py \
  Tests/UI/test_console_narrow_layout.py \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_css_bundle_sync_guard.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/console_command_provider.py \
  Tests/UI/test_console_terminal_integration.py
git diff --check
```

Expected: PASS and the CSS bundle reproduces exactly.

- [ ] **Step 9: Commit the Console integration**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/console_command_provider.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/UI/test_console_terminal_integration.py
git commit -m "feat: expose persistent terminal in console"
```

### Task 14: Prove the user-only privacy, logging, persistence, and tool boundary

**Files:**
- Create: `Tests/Architecture/test_persistent_terminal_privacy_boundary.py`
- Create: `Tests/Terminal/test_terminal_diagnostics.py`
- Modify: `Docs/security/production-diagnostic-inventory.json` only if new production log sites are added
- Reference: `tldw_chatbook/Agents/tool_catalog.py`
- Reference: `tldw_chatbook/Chat/console_chat_controller.py`
- Reference: `tldw_chatbook/Agents/run_log.py`

- [ ] **Step 1: Write mutation-sensitive architectural exclusion characterization**

Scan product imports/registrations and exercise a live fake session. Assert no Terminal package object or projection enters tool schemas/catalogs, MCP permission storage, provider messages/history, Console conversation messages, AgentRuns, run logs, exports, workspace persistence, config, database writes, or persisted app/workspace/conversation/crash-recovery snapshots. Textual/Pilot render snapshots are allowed to contain the safe cells currently displayed to the local user. Add a mutation-sensitive harness that deliberately injects a sentinel projection into each fake persistence/provider/export sink and proves the assertion fails, so a passing negative test is meaningful rather than vacuous.

```python
def test_terminal_package_is_not_registered_as_a_model_tool() -> None:
    catalog_source = Path("tldw_chatbook/Agents/tool_catalog.py").read_text()
    assert "TerminalSessionManager" not in catalog_source
    assert "console_terminal" not in catalog_source
```

Prefer behavior assertions over source scans where a concrete store/provider boundary can be instantiated; use source/AST checks only for negative registration contracts.

- [ ] **Step 2: Write content-free diagnostic RED tests**

Seed unique sentinels into name, starting path, input, output, environment, profile, rejected paste, parser failure, backend failure, and cleanup-unproven paths. Capture Loguru and exported diagnostic payloads. Only opaque session ID, lifecycle, duration/timing, byte counts, dimensions, and stable failure category may appear. Session names and private paths are excluded from generic diagnostics even though the local Terminal UI may show the user-selected name/start path.

- [ ] **Step 3: Run the characterization before any privacy correction**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Architecture/test_persistent_terminal_privacy_boundary.py \
  Tests/Terminal/test_terminal_diagnostics.py -q
```

Expected: PASS if construction already excludes Terminal from every boundary, or FAIL with a specific accidental leak. In either case, the mutation-sensitive control must prove the test would fail if the sentinel reached a forbidden sink; do not claim a synthetic RED result for an already-correct negative boundary.

- [ ] **Step 4: Remove accidental projections; add only content-free diagnostics**

Do not add a redaction pipeline for terminal content—keep content out of diagnostic objects entirely. If production warning/error calls are needed, inventory each stable message with content classification in `production-diagnostic-inventory.json` using the repository's inventory workflow.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Architecture/test_persistent_terminal_privacy_boundary.py \
  Tests/Terminal/test_terminal_diagnostics.py \
  Tests/UI/test_console_raw_cli_transcript.py \
  Tests/Agents/test_raw_shell_integration.py -q
../../.venv/bin/python -m ruff check Tests/Architecture/test_persistent_terminal_privacy_boundary.py Tests/Terminal/test_terminal_diagnostics.py
git diff --check
```

Expected: PASS; existing raw CLI and model shell paths remain unchanged.

```bash
git add Tests/Architecture/test_persistent_terminal_privacy_boundary.py Tests/Terminal/test_terminal_diagnostics.py Docs/security/production-diagnostic-inventory.json
git commit -m "test: pin terminal privacy boundary"
```

If the inventory file did not change, omit it from `git add` rather than creating churn.

### Task 15: Complete performance, real-platform, documentation, and task closeout

**Files:**
- Create: `Tests/Terminal/test_terminal_resource_qualification.py`
- Create: `Tests/integration/test_console_terminal_lifetime.py`
- Create: `Tests/Packaging/test_terminal_distribution.py`
- Modify: `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console.md`
- Modify: `README.md`
- Modify: `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`
- Modify: `backlog/decisions/README.md`
- Modify: `backlog/tasks/task-22512 - Persistent-interactive-PTY-and-ConPTY-terminal-sessions.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if this work produces a genuinely new evidenced lesson

- [ ] **Step 1: Add distribution, four-session memory, and flood qualification**

Build a wheel without network/isolation and assert it contains the `tldw_chatbook.Terminal` package plus metadata for unconditional `pyte==0.8.2` and Windows-only `pywinpty==3.0.5`; assert a POSIX install does not import `winpty`. Then measure empty-manager baseline and four sessions at 300x120 plus 5,000 lines/4 MiB normal scrollback. After five-second quiescence, sum Chatbook parent delta and app-owned worker/helper/IPC RSS while excluding identified user shell/program RSS. Assert `<=256 MiB` only on named qualification hosts; developer runs report measurements. Run ten seconds of synthetic ANSI output with a 100 ms sentinel and assert p95 `<100 ms` on each claimed platform host.

- [ ] **Step 2: Add mounted/live lifetime verification**

Launch the real Textual app hierarchy with fd-backed standard streams and worktree provenance. Exercise arm, create, real shell input, retained `cd`/environment, Unicode, resize, alternate screen, Ctrl+] local navigation, model-turn independence, conversation/screen navigation, recompose/remount, shell exit with final state retained, close, Disarm, Retry cleanup, and app shutdown. Wait for screen state, DOM mount, and compositor paint; Pilot-only key/process evidence is insufficient.

Run the named POSIX test on each claimed macOS/Linux host:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/integration/test_console_terminal_lifetime.py::test_posix_mounted_real_terminal_focus_input_and_navigation -q
```

Run the named Windows test in a real terminal on each claimed Windows host:

```powershell
..\..\.venv\Scripts\python.exe -B -m pytest `
  Tests/integration/test_console_terminal_lifetime.py::test_windows_mounted_real_terminal_focus_input_and_navigation -q
```

Expected: both native tests pass on every claimed platform row; a POSIX Pilot result cannot stand in for Windows terminal input/focus evidence and vice versa.

- [ ] **Step 3: Re-run platform-native cleanup and crash probes**

On macOS/Linux and supported Windows hosts, rerun normal close, parallel Disarm, global Shutdown, exact-shell-exit descendant drain, parser-failure raw cleanup drain, cleanup-unproven Retry, and app-process-failure fixtures. Append exact results and commands to the qualification artifact. Rebase onto latest `origin/dev` before merge, rerun `format_ratchet.py snapshot` against that newly resolved base to replace `format-baseline.json`, then verify `HEAD` against the new stored immutable SHA. Recheck ADR-099's ID/status/index after the rebase.

- [ ] **Step 4: Update user and setup documentation**

Document:

- entry from Sessions rail and command palette;
- New/Rename/Focus/Close/Retry/Jump live;
- four-session and resource limits;
- starting directory convenience versus no confinement;
- normal profiles/history/side effects and full OS-user authority;
- shared saved unlock and independent per-launch raw CLI/Terminal arms;
- Ctrl+] release, local keyboard scrollback, reserved globals, no mouse v1;
- distinction from raw user `!`, model `shell_exec`, and read-only `virtual_cli`;
- no model access, persistence, export, reconnect, or `terminal_armed` config field;
- Windows 10 1809/Server 2019 floor, exact ConPTY dependency, supported wheels, and fail-closed diagnostics.

ADR-094 receives only a cross-reference to accepted ADR-099; do not rewrite its one-shot contracts.

- [ ] **Step 5: Run the focused verification matrix**

First prove provenance:

```bash
../../.venv/bin/python -B -m pytest Tests/test_probe_import_provenance.py -q
```

Then run the reachable focused suites:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Terminal \
  Tests/integration/test_console_terminal_lifetime.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/UI/test_settings_terminal.py \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py \
  Tests/UI/test_console_terminal_integration.py \
  Tests/UI/test_console_narrow_layout.py \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/Architecture/test_persistent_terminal_privacy_boundary.py \
  Tests/Packaging/test_terminal_distribution.py \
  Tests/Chat/test_console_raw_cli_runtime.py \
  Tests/Chat/test_console_raw_cli_persistence.py \
  Tests/Chat/test_console_raw_shell_revocation.py \
  Tests/Agents/test_raw_shell_integration.py -q
```

Expected: PASS. On a non-Windows host, only native Windows tests skip with the reviewed reason; host-independent Windows contract tests pass.

- [ ] **Step 6: Run static and generated-artifact checks**

Run:

```bash
../../.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify \
  --head HEAD \
  --baseline Docs/superpowers/reviews/evidence/task-22512/format-baseline.json
../../.venv/bin/python -m ruff check \
  scripts/terminal_qualification \
  tldw_chatbook/Terminal \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Console_Modules/terminal.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/console_command_provider.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/Widgets/Console/console_terminal_workspace.py \
  tldw_chatbook/Widgets/Console/console_terminal_session_modal.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Terminal \
  Tests/fixtures/terminal \
  Tests/Chat/test_console_runtime_lifetime.py \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/UI/test_settings_terminal.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py \
  Tests/UI/test_console_terminal_integration.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/Architecture/test_persistent_terminal_privacy_boundary.py \
  Tests/Packaging/test_terminal_distribution.py \
  Tests/integration/test_console_terminal_lifetime.py
../../.venv/bin/python -m ruff format --check \
  scripts/terminal_qualification \
  tldw_chatbook/Terminal \
  tldw_chatbook/UI/Console_Modules/terminal.py \
  tldw_chatbook/Widgets/Console/console_terminal_workspace.py \
  tldw_chatbook/Widgets/Console/console_terminal_session_modal.py \
  Tests/Terminal \
  Tests/fixtures/terminal \
  Tests/UI/test_settings_terminal.py \
  Tests/UI/test_console_terminal_workspace.py \
  Tests/UI/test_console_terminal_keyboard.py \
  Tests/UI/test_console_terminal_session_modal.py \
  Tests/UI/test_console_terminal_integration.py \
  Tests/Architecture/test_persistent_terminal_privacy_boundary.py \
  Tests/Packaging/test_terminal_distribution.py \
  Tests/integration/test_console_terminal_lifetime.py
../../.venv/bin/python -B -m pytest Tests/UI/test_css_bundle_sync_guard.py -q
git diff --check
git diff --check origin/dev...HEAD
```

Expected: the normalized base-vs-HEAD formatter ratchet passes for every existing modified Python file; Ruff lint and formatting of every new file, CSS sync, working-tree whitespace, and branch-wide whitespace checks pass. Do not introduce formatter debt on a task-changed line or mass-format unrelated code.

- [ ] **Step 7: Ask whether the user wants a full suite**

Do not infer consent from pre-PR or merge language. If explicitly approved, run the repository's full suite and record the command/result. Otherwise state that focused reachable suites and native platform matrices were run under repository policy.

- [ ] **Step 8: Self-review against every acceptance criterion and tripwire**

Read the complete diff. Confirm:

- raw `!`, `shell_exec`, and `virtual_cli` contracts are unchanged;
- no model/tool/persistence/context projection exists;
- one app owner, one reaper per session, and no widget process ownership;
- no ordinary-pipe/legacy/high-level Windows fallback;
- exact limits, cleanup deadlines, proof reserve, and Retry semantics match ADR-099;
- receipts remain actionable while locked/unarmed;
- all danger and diagnostic copy is honest and content-free;
- dependency versions/API match the qualification artifact;
- no nested-program mouse support slipped into v1;
- no config key for `terminal_armed` exists.

- [ ] **Step 9: Complete task hygiene only after all evidence is green**

Edit the five-digit task file directly: check AC #1-#8 and #10, add concise `## Implementation Notes`, link ADR-099 and the qualification artifact, record focused/native verification, and set frontmatter status to `Done`. Add a lesson only if an incident produced reusable evidence; do not invent one.

- [ ] **Step 10: Commit closeout**

```bash
git add Docs/User_Guide/console/sessions-tabs-workspaces.md \
  Docs/User_Guide/console/agent-runs-and-tools.md \
  Docs/User_Guide/settings.md \
  Docs/User_Guide/console.md \
  README.md \
  backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md \
  backlog/decisions/README.md \
  "backlog/tasks/task-22512 - Persistent-interactive-PTY-and-ConPTY-terminal-sessions.md" \
  Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md \
  Tests/Terminal/test_terminal_resource_qualification.py \
  Tests/integration/test_console_terminal_lifetime.py \
  Tests/Packaging/test_terminal_distribution.py
git commit -m "docs: complete persistent terminal sessions"
git status --short
git diff --check
git diff --check origin/dev...HEAD
```

Expected: commit succeeds, the worktree is clean, and both working-tree and complete branch whitespace checks pass. If a lessons file changed, include its exact path. If unrelated decision/task files changed after rebasing, stage only ADR-094, ADR-099/index, and TASK-22512.

## Review checkpoints

After Tasks 1, 9, 13, and 15, pause for a focused code/design review before proceeding:

1. **Qualification gate:** versions, licenses, hashes, wheel/platform matrix, parser mutable-state audit, and Windows low-level API evidence are complete.
2. **Runtime gate:** admission-before-exec, one-owner/reaper, bounded I/O, identity-checked cleanup, and real platform failure behavior match ADR-099.
3. **UI/privacy gate:** full-host danger is unmistakable, keyboard routing is accurate, widgets own no processes, and no model/persistence boundary changed.
4. **Closeout gate:** focused and native evidence is current, docs are accurate, acceptance criteria are all proven, and task/ADR hygiene is complete.

Any review finding that changes storage, process ownership, dependency/API version, model visibility, persisted authority, cleanup semantics, launch command/environment flexibility, or confinement claims must stop implementation for an ADR-099 amendment or superseding ADR before code continues.
