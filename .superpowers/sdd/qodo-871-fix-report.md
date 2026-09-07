# Qodo review fix report — PR #871

Branch: `worktree-skills-script-execution`
Commit: `fe71e2335bd34eeee59fc5d73d04874b7dfbc3e9`
Pushed: yes (`30e8a6b40..fe71e2335 worktree-skills-script-execution -> worktree-skills-script-execution`)

All three findings were verified as legitimate before fixing (per the task
brief) and are addressed below.

## Finding 1 — missing return type hints

`SkillsScopeService.describe_skill_script()` / `run_skill_script()` in
`tldw_chatbook/Skills_Interop/skills_scope_service.py` had no return type
annotation.

**Fix**: Added a `TYPE_CHECKING`-guarded import block importing `ScriptPlan`
(from `.local_skills_service`) and `ScriptRunResult` (from
`.skill_script_runner`), then annotated the two methods `-> ScriptPlan` and
`-> ScriptRunResult`. This mirrors the exact precedent already in this
package — `local_skills_service.py` does the identical thing for
`ScriptRunResult`/`ScriptRunLimits` with the comment "Deferred at runtime …
to avoid a module-scope import of the subprocess sandbox for every
LocalSkillsService caller". `skills_scope_service.py` already has
`from __future__ import annotations`, so no string-quoting was needed on the
annotations themselves.

Checked for circularity first: neither `local_skills_service.py` nor
`skill_script_runner.py` (nor anything they import) imports
`skills_scope_service` anywhere, so a plain module-scope import would not
actually have been circular — but the `TYPE_CHECKING` guard was used anyway
per the explicit instruction to follow the established precedent (and it
also avoids pulling `subprocess`/`resource`-adjacent code into this
deliberately thin module for every caller, matching the local service's own
stated rationale).

Verified with a standalone import + `__annotations__` inspection:
```
{'skill_name': 'str', 'script_path': 'str', 'mode': 'SkillsBackend | str | None', 'return': 'ScriptPlan'}
{'skill_name': 'str', 'script_path': 'str', 'args': 'Sequence[str]', 'mode': 'SkillsBackend | str | None', 'return': 'ScriptRunResult'}
OK, no circular import
```

Also added a `SandboxUnsupportedError` line to the `Raises:` docstring
section of `SkillsScopeService.run_skill_script` and
`LocalSkillsService.run_skill_script` (see Finding 2) for documentation
completeness, since both now transitively can raise it.

## Finding 2 — POSIX-only runner, Windows claimed as supported

`skill_script_runner.run_script_subprocess` depends on POSIX-only
primitives (`start_new_session=True`, `os.killpg`/`os.getpgid`, and a
trampoline that does `import resource`), none of which exist on Windows,
while `README.md:23` claims "Operating System: Windows, macOS, Linux".

**Fix (fail-closed, no Windows sandbox implementation attempted)**:

- `tldw_chatbook/Skills_Interop/skill_script_runner.py`: added
  `sandbox_supported() -> bool` (`return os.name != "nt"`) with a
  Google-style docstring, and `class SandboxUnsupportedError(RuntimeError)`.
  `run_script_subprocess` now checks `sandbox_supported()` FIRST — before
  `target_argv` validation, before `Popen`, before anything else — and
  raises `SandboxUnsupportedError` if it is False. The `Raises:` docstring
  section was updated accordingly.
- `tldw_chatbook/Chat/console_agent_bridge.py`: the `run_skill_script_tool`
  closure's construction is now gated on `sandbox_supported()` in addition
  to the existing `self._skills_service is not None and
  request_skill_script_confirm is not None` gate — mirroring the existing
  "advertised must equal usable" principle already documented at that same
  call site for the confirm-callback gate (the #847 lesson). The `import` of
  `sandbox_supported` is a lazy, function-scoped import (matching this
  file's existing convention of lazily importing
  `runtime_policy.types.PolicyDeniedError` inside the closure body, and
  `local_skills_service.py`'s own lazy imports of `skill_script_runner`) —
  `Skills_Interop` module-scope imports elsewhere in this file are limited
  to `SkillTrustBlockedError`.
- `Docs/Features/Skills-Script-Execution.md`: added a new "## Platform
  support" section right after the intro, stating plainly that script
  execution is POSIX-only (macOS/Linux) and unavailable on Windows, and
  explaining that the tool is simply never wired up rather than
  auto-failing.
- Did **not** change `README.md`'s general "Operating System: Windows,
  macOS, Linux" line — that describes the whole application's OS support,
  which remains true; only this one feature is platform-gated, and that is
  now documented at the feature level per the task instructions (2e named
  only the feature doc, not the README).

Tests added in `Tests/Skills/test_skill_script_runner.py`:
- `test_sandbox_supported_is_true_on_this_posix_box` — sanity check on the
  real macOS test box.
- `test_sandbox_unsupported_raises_before_any_spawn` — monkeypatches
  `skill_script_runner.os.name = "nt"` (the exact thing the predicate
  inspects), confirms `sandbox_supported()` flips to `False`, then asserts
  `run_script_subprocess` raises `SandboxUnsupportedError` for a
  `target_argv` that is otherwise perfectly valid and runnable (proving the
  platform check runs first, not merely that some other step also happens
  to fail). Never attempts to run any actual Windows code path.

Test added in `Tests/Chat/test_console_skill_script_confirm.py`:
- `test_tool_is_absent_on_an_unsupported_platform` — monkeypatches
  `skill_script_runner.sandbox_supported` to `lambda: False`, then drives
  the REAL `ConsoleAgentBridge.run_reply` (via the existing
  `_capture_run_skill_script_tool` harness that intercepts the real
  `AgentService(...)` construction site) with a skills service AND a
  confirm callback both present, and asserts the captured
  `run_skill_script_tool` kwarg is `None` — i.e. the tool was never built
  despite both of the other two gates being satisfied.

## Finding 3 — error-kind leak in `_resolve_script`

`LocalSkillsService._resolve_script` calls
`validate_supporting_file_path(script_path)` (from
`tldw_api/skills_schemas.py`) without a try/except. That validator raises
its own differently-worded `ValueError`s ("Invalid supporting file path:
…", "Invalid path segment … in …", "Supporting file path too long: …",
"SKILL.md is the skill body, not a supporting file"), which escaped
`_resolve_script` unwrapped — breaking the documented invariant that every
script-path rejection reason (unsafe, missing, symlink, untrusted, reserved
body) surfaces as the identical `local_skill_script_not_found:<script_path>`
error kind.

**Fix**: Wrapped the call —
```python
try:
    validate_supporting_file_path(script_path)
except ValueError as exc:
    raise ValueError(f"{_SCRIPT_NOT_FOUND_ERROR}:{script_path}") from exc
```
— re-raising with the canonical kind and chaining via `from exc` so the
original reason is still available in tracebacks/logs without being part of
the string a caller (or a probing agent) can see.

Test added in `Tests/Skills/test_skill_script_service.py`:
- `test_validator_rejection_is_indistinguishable_from_missing`, parametrized
  over three paths that each trip a *different* branch inside
  `validate_supporting_file_path` (a `..` traversal segment, a segment
  failing `SEGMENT_PATTERN` via an illegal character, and a path exceeding
  `MAX_SUPPORTING_FILE_PATH_LEN`). For each, asserts `_resolve_script`
  raises with the SAME error kind (`local_skill_script_not_found`) as a
  genuinely missing file, and pins the exact echoed string alongside the
  kind check (so a regression to a bare, path-less constant — which would
  make the kind-equality assertion vacuously true — is still caught).

## Constraints respected

- No existing gate weakened: `run_skill_script`/`describe_skill_script`
  still re-enforce policy (`self._enforce("skills.run_script.launch.local")`)
  and re-verify trust (`self._require_trusted_skill`) on every call;
  `_resolve_script`'s containment-before-`is_file()`/`stat()` ordering is
  untouched (the try/except sits entirely BEFORE `skill_dir.is_dir()` and
  every later containment/trust/stat check, at exactly the same point the
  unguarded call used to sit); the trusted-manifest membership check
  (`_script_path_is_trust_material`) is unchanged.
- Google-style docstrings (Args/Returns/Raises) added on the two new public
  callables (`sandbox_supported`, `SandboxUnsupportedError`).

## Verification commands and output tails

```
$ source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
$ python -m pytest Tests/Skills -q
........................................................................ [ 21%]
........................................................................ [ 42%]
........................................................................ [ 64%]
........................................................................ [ 85%]
.................................................                        [100%]
337 passed, 1 warning in 126.38s (0:02:06)
```
(Baseline before this fix, per the task-8 SDD entry and the task brief: 332
passed. +5 new tests = 337, matches exactly: 2 in
`test_skill_script_runner.py`, 3 parametrized cases in
`test_skill_script_service.py`.)

```
$ python -m pytest Tests/Chat/test_console_skill_script_confirm.py Tests/Agents/test_run_skill_script_runtime_tool.py -q
................................                                         [100%]
32 passed in 2.52s
```

Also ran `ruff check` on every touched file: clean, except one
pre-existing, already-documented-in-progress.md `F401 'stat' imported but
unused` in `local_skills_service.py:1242` (`export_skill`), which predates
this fix and was explicitly noted as a known baseline issue in the prior
SDD task-8 entry — not introduced by this change, and not touched here
since it is out of scope for these three findings.

## What was judged NOT to need changing, and why

- **README.md's OS support line** — left as-is. It describes the whole
  application, which genuinely still runs on Windows; only the one
  script-execution feature is gated, and that is now documented at the
  feature-doc level (`Docs/Features/Skills-Script-Execution.md`) per the
  task's own instruction (2e named the feature doc specifically, not the
  top-level README).
- **A Windows sandbox implementation** — explicitly out of scope per the
  task brief ("do NOT attempt a Windows sandbox implementation — that is a
  separate project"); not attempted.
- **Broad-except handling of `SandboxUnsupportedError` inside the bridge
  closure** — no new code needed. The closure already wraps
  `scope.run_skill_script(...)` in a broad `except Exception as exc:` that
  returns a `ToolResult(ok=False, ...)`, so `SandboxUnsupportedError` (a
  `RuntimeError` subclass) is already caught defense-in-depth even though
  the platform gate on the closure's *construction* means this path is
  normally unreachable in practice.
- **The pre-existing `ruff F401` finding** — confirmed out of scope (not
  one of the three Qodo findings; already tracked in `progress.md` as a
  known baseline issue from before this branch).
