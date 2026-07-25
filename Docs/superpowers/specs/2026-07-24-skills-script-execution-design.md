# Skills Script Execution (trust-gated) — Design

Date: 2026-07-24
Program: Skills-install (the "user asks an agent to install and use a skill/pack from a GitHub link" north star). This is the **script-execution layer** — the largest remaining capability after the six merged layers (#762 trust → #784 bundle fidelity → #801 `$`-mention → #814 reachability → #831 remote fetch → #847 agent-callable install). Prior layer specs live alongside this file under `Docs/superpowers/specs/`.

## 1. Purpose

Skills authored to the [Agent Skills](https://agentskills.io) spec routinely ship helper scripts (`scripts/extract.py`, `bin/render.sh`, small compiled helpers) that a skill's instructions tell the agent to run. Today the agent can *read* those files (`skill_file` runtime tool, layer #814) but cannot *run* them — `execute_skill` returns only the rendered `SKILL.md` body and a read-only `reference_files` listing. This layer closes that gap: **a trusted skill's bundled scripts can actually execute, under a per-run human confirmation and best-effort OS containment, exposing captured stdout/stderr back to the agent.**

The security spine already built for the install tool and the skill_file reader is reused wholesale: the same keyed-MAC trust fingerprints, the same in-chat HITL confirm card, the same fail-closed runtime-policy registry, the same name-scoped runtime-tool plumbing.

## 2. Goals / Non-Goals

**Goals**
- A **6th runtime tool** `run_skill_script` the agent can call to execute a script from a trusted skill bundle.
- Every run passes **three independent gates**: runtime policy (fail-closed) → trust re-verification (per-run re-scan) → **human decision** (per-run confirm card, with an "Always allow for this skill" grant that persists and self-invalidates on content change).
- **Best-effort OS containment**: `shell=False` argv, scrubbed environment (no API keys), a fresh per-run scratch working directory that is *never* the skill directory, RLIMIT-based CPU/process/file caps (where the platform supports them), a wall-clock timeout, and capped stdout/stderr.
- Two invocation mechanisms so real bundles work regardless of how they were imported: **interpreter-map** for text scripts and **direct exec** for files fingerprinted executable (incl. compiled binaries).
- Honest documentation of residual risk (network egress, user-level file reads) — parity with the existing Evals sandbox posture.

**Non-Goals (deferred / explicitly out)**
- **Output files.** v1 captures stdout/stderr + exit code only; the scratch dir is deleted after the run. File-producing workflows wait for a consumer story (the Agents runtime has no general file-read tool; `skill_file` is skill-dir-contained by design).
- **Real jail** (macOS `sandbox-exec` profile, Linux seccomp/namespaces, container). Explicit follow-up layer; §8.
- **Network isolation.** Cannot be delivered without a real jail; documented as residual.
- **Frontmatter `allowed-scripts` / declarative permissions.** A skill does not declare which of its files are runnable; any trusted+allowed script may run. Frontmatter conformance is its own (minor) layer.
- **Server-mode skills.** Execution is **local-only**, exactly like `skill_file` reads (§6).

## 3. Scope decisions (locked with the user)

| Decision | Choice |
|---|---|
| Run gate | **Per-run confirm card + "Always allow for this skill"** persistent, revocable grant. No session-wide bypass. |
| Containment | **Best-effort limits** (argv-only, scrubbed env, fresh scratch cwd, RLIMITs where supported, timeout, output caps). Residual risk documented. |
| Tool scope | **Any trusted + allowed skill** by `skill_name` — not name-bound to the turn's `$`-mentions. Gated purely by trust + per-skill allow + per-run confirm. Available to the **primary agent and skill forks** (not arbitrary subagents; §4.3). |
| Runnable set | **Both mechanisms** — text scripts via a fixed extension→interpreter map, and exec-bit-fingerprinted files (including binaries) via direct exec. |
| Outputs | **Stdout/stderr + exit code only**; scratch dir deleted after run. |

## 4. Architecture

Four layers, mirroring the install tool's shape.

```
Agent loop  ──run_skill_script(skill_name, script_path, args[])──▶  LoopDeps.run_skill_script closure
   (Agents/agent_runtime.py dispatch branch)                          (Chat/console_agent_bridge.py)
                                                                          │  order: enforce → resolve+classify
                                                                          │         → grant-check → confirm(HITL)
                                                                          │         → execute → capped ToolResult
                                                                          ▼
                                            SkillsScopeService.run_skill_script(...)         [local-only passthrough + policy]
                                                          │
                                                          ▼
                                            LocalSkillsService.run_skill_script(...)         [trust re-verify + resolve + runner]
                                                          │
                                                          ▼
                                            Skills_Interop/skill_script_runner.py            [the sandboxed subprocess]
```

HITL confirm reuses the install card machinery end-to-end:
```
bridge closure ──request_skill_script_confirm(payload)──▶ console_chat_controller (threading.Event + call_from_thread)
                                                              │
                                                              ▼
                                    chat_screen_state.pending_skill_script  ──▶ ChatTaskCards ──▶ SkillScriptConfirmCard
                                                              ▲                                         │ Allow / Deny / Always-allow
                                                              └──────── resolve_pending_skill_script ◀──┘ (ScriptDecided message)
```

### 4.1 Components

**A. `Skills_Interop/skill_script_runner.py` (new, pure-ish).**
The subprocess execution unit — the one place that spawns a child process. Input: an absolute, already-containment-validated script path, an interpreter argv prefix (possibly empty for direct exec), a caller-supplied `args` list, and a `ScriptRunLimits` config. Output: `ScriptRunResult(exit_code, stdout, stderr, timed_out, duration_seconds, truncated_stdout, truncated_stderr, sandbox_warnings)`. No knowledge of skills, trust, policy, or the agent — testable in isolation with a temp script. Modeled on `Evals/specialized_runners.py` but as a reusable seam, and using `preexec_fn` for the RLIMITs (see §5) rather than prepending source lines (we run arbitrary interpreters/binaries, not just generated Python).

**B. `LocalSkillsService.run_skill_script(skill_name, script_path, args, *, limits) -> ScriptRunResult` (new method).**
The trust + resolution seam. In order:
1. `self._enforce("skills.run_script.launch.local")` — new policy action (§6), fail-closed.
2. `self._require_trusted_skill(skill_name)` — per-run re-scan; a skill revoked or mutated mid-conversation stops being runnable immediately (same discipline as `read_skill_file`, `local_skills_service.py:1273`).
3. Resolve `script_path` against the skill dir: `validate_supporting_file_path(script_path)` (reject `SKILL.md` — it is not a script), then **containment before stat** via `get_safe_relative_path(path, skill_dir)` (returns `None` ⇒ raise `local_skill_script_not_found:` — the symlink-oracle-safe pattern from #814), then reject symlinks / non-regular files with the *same* not-found error.
4. Classify the resolved file and choose the invocation mechanism (§4.2). Unrunnable file types (e.g. a text file with no interpreter mapping and no exec bit) raise a domain error, not a crash.
5. Hand the absolute path + interpreter prefix + args + limits to the runner; return its `ScriptRunResult`.
This method does **not** own the human gate — the confirm is the bridge closure's job, exactly as the install closure owns confirm and `install_skill_from_url` owns the fetch.

**B2. `LocalSkillsService.describe_skill_script(skill_name, script_path) -> ScriptPlan` (new read-only seam).**
Lets the bridge closure pre-resolve a run *for the confirm card* and fail early — with **no prompt** — on trust/path/type errors, without spawning anything. Runs the same first four steps of `run_skill_script` (enforce → trust re-verify → containment-safe resolve → classify) but stops before the runner, returning `ScriptPlan(mechanism, interpreter_display, resolved_relative_path, is_binary)`. Read-only and side-effect-free. `run_skill_script` **re-runs** all of these steps authoritatively at execution time (never trusting a plan computed earlier), so a card built from an at-request `ScriptPlan` that goes stale before the human clicks Allow cannot widen what actually runs — a mid-confirm trust revocation or content change is caught by the authoritative re-verify and denies the run. `SkillsScopeService` gets a matching local-only `describe_skill_script` passthrough.

**C. `SkillsScopeService.run_skill_script(...)` + `enforce_run_script()` (new passthrough + public enforce).**
`run_skill_script` is local-only dispatch mirroring `read_skill_file` (`skills_scope_service.py:319-356`): `_require_service(LOCAL)`, `self._enforce_policy("skills.run_script.launch.local")` (the double-gate with the local service's own enforce is intentional and consistent with the read path), then delegate. Raises `ValueError("skill scripts run local-only")` for any non-local mode. It also exposes a public `enforce_run_script() -> None` passthrough (mirroring `enforce_install_remote`, `skills_scope_service.py:358-368`) that enforces `skills.run_script.launch.local` **standalone**, so the bridge closure can deny on policy *before* showing any confirm card (§7 step 1). Enforcing twice (closure pre-check + in-dispatch) is idempotent and consistent.

**D. Grant store — `SkillTrustService` extension (or a sibling local JSON store).**
The "Always allow for this skill" grant persists across restarts and **self-invalidates when the skill's content changes**. It is pinned to the skill's current fingerprint digest — the same `_fingerprints_digest(snapshot)` value that already invalidates a captured review (`skill_trust_service.py:449, 494`). Store shape: a small local-only, non-synced JSON keyed `{normalized_skill_name: fingerprint_digest}`, written next to the trust store (sibling of the manifest; **not inside the MAC'd manifest**, so granting a run never perturbs trust fingerprints). New trust-service methods:
- `script_grant_digest(skill_name) -> str | None` — the digest currently granted, or `None`.
- `grant_script_execution(skill_name)` — record the skill's *current* fingerprint digest.
- `revoke_script_execution(skill_name)` — drop the entry.
- `current_fingerprint_digest(skill_name) -> str` — expose the live digest for comparison (thin wrapper over `_fingerprints_digest(self._scan_skill(name))`).
The bridge closure treats the grant as satisfied **iff** `script_grant_digest(name) == current_fingerprint_digest(name)`. Any content change (which already forces trust re-review) also drops the standing script grant back to per-run confirm. Revocation is surfaced in the Library ▸ Skills trust panel (§4.4).

### 4.2 Runnable-set classification (`Both mechanisms`)

Given the resolved bundle file (using `bundle_files` metadata already computed at `_read_bundle_manifest`, `local_skills_service.py:459-490`, which carries `executable` and `is_text`):

1. **Direct exec** — if the file is fingerprinted `executable` (owner-exec bit tracked in the manifest, applied on import at `local_skills_service.py:1034-1037`/`1150-1156`): argv = `[abs_path, *args]`, interpreter prefix empty. The OS uses the file's shebang / binary format. Covers compiled helpers and mode-preserving imports.
2. **Interpreter map** — else if the file `is_text` and its extension is in a fixed map: argv = `[interpreter, abs_path, *args]`.
   ```
   .py  → [sys.executable]         .sh  → ["/bin/sh"]
   .bash→ ["/bin/bash"]            .js  → ["node"]   (only if resolvable on the scrubbed PATH)
   ```
   The map is a module constant, deterministic, and **ignores shebangs for text files** (shebang can only pick the interpreter for exec-bit files, which the human reviewed as executable). Extensions not in the map + no exec bit ⇒ `unrunnable_script_type` domain error.
3. A text file **with** the exec bit is treated as direct-exec (case 1) — its shebang was reviewed. This matches how a shell would treat `./scripts/extract.py` with a `#!/usr/bin/env python3` line and mode 0755.

**Security note carried into the plan:** the interpreter must be an absolute path or resolved against the *scrubbed* PATH (`/usr/bin:/bin`), never the user's PATH, so a skill cannot shadow `node`/`sh` via a poisoned PATH. Interpreters that don't resolve on the scrubbed PATH make that mechanism unavailable (surface a clear error), rather than silently falling back to the user environment.

### 4.3 Runtime-tool plumbing (mirror `skill_file`/`install_skill`)

- **Name + schema.** `RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"` joins `RUNTIME_TOOL_NAMES` in `Agents/agent_models.py:31-44`. `RUN_SKILL_SCRIPT_TOOL_SCHEMA` in `Agents/tool_catalog.py` (params: `skill_name: str`, `script_path: str`, `args: list[str]` default `[]`; house-style description mandatory — every other runtime schema has one).
- **Drift guard.** Add `"run_skill_script"` to `Library/library_skills_state.py:_SHADOWED_BUILTIN_NAMES` (the sync test fails otherwise — it did for both prior runtime tools). A skill literally named `run_skill_script` is excluded from `_non_colliding_skill_entries`.
- **LoopDeps.** New field `run_skill_script: Callable[[str, str, list[str]], ToolResult] | None = None` (`Agents/agent_runtime.py`, alongside `read_skill_file`/`install_skill`). Dispatch branch in `run_agent_loop`'s elif chain: `elif call.name == RUN_SKILL_SCRIPT_TOOL_NAME and deps.run_skill_script is not None:` — coerce args defensively (a `Mapping`/list guard, like the reader closure) and return the closure's `ToolResult`.
- **AgentService.** New ctor kwarg `run_skill_script_tool`. Schema pinned in `_run_one` when the tool is wired (`agent_service.py:361-372`). **Availability:** available to the **primary agent and skill forks**, gated the same way `read_skill_file` reaches forks — via presence of the wired callable, not by `agent_kind`. It is *not* install-style primary-only, because a fork acting on its own skill is a legitimate caller; it is not offered to arbitrary spawned subagents (they receive neither the closure nor the schema).

### 4.4 UI

- **`Widgets/Chat_Widgets/skill_script_confirm_card.py` (new)** — `SkillScriptConfirmCard(Container)`, `markup=False` (the install card is markup-safe for the same reason: script paths/args are agent-supplied and must not render Rich markup). Shows: skill name, script path, resolved mechanism (`python scripts/extract.py` / `direct-exec bin/tool`), the `args` list, and the timeout. Buttons: **Allow once**, **Always allow this skill**, **Deny**. Emits `ScriptDecided(allow: bool, remember: bool)`.
- **State + container** — `chat_screen_state.pending_skill_script: Optional[Dict]` + `has_pending_skill_script()` + serde (mirror `pending_skill_install` at `chat_screen_state.py:211/230/244/259`); `ChatTaskCards` yields the new card as a 4th child and extends the display gate (an unextended gate hides the card even though thread mechanics work — the lesson baked into the install card spec); `chat_screen.py` setter + `@on(SkillScriptConfirmCard.ScriptDecided)` → `controller.resolve_pending_skill_script(allow, remember)`.
- **Trust panel** — the Library ▸ Skills trust review/detail panel gains a line showing whether script execution is granted for the skill and a **Revoke script access** affordance calling `revoke_script_execution`. (Discoverability + revocability of the standing grant; the trust panel is where the human already governs this skill.)

### 4.5 Controller HITL

Add to `console_chat_controller.py`, cloning the install-confirm machinery (`:1105-1191`):
- `_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 120.0` (reuse the install value; under the 480 s run budget).
- `request_skill_script_confirm(self, payload: dict) -> dict` — worker-thread blocking call. Returns `{"allow": bool, "remember": bool}`. Fails closed immediately when no UI bridge is wired (early-return `{"allow": False, "remember": False}` — the headless-stall fix the install tool needed post-review). Fresh `threading.Event`, deadline poll breaking on stop/cancel/deadline, `finally` clears state + card.
- `_marshal_pending_skill_script` → `call_from_thread(self.set_pending_skill_script, payload)`.
- `resolve_pending_skill_script(allow, remember)` — UI thread.
- `_deny_pending_skill_script_on_context_change()` — wired into `switch_session` next to the install analogue (`:724`); a conversation switch mid-confirm denies (worker keeps running, card is gone → must not hang the full timeout).
- Wired at the `run_reply` call site as a new kwarg `request_skill_script_confirm=self.request_skill_script_confirm`.

## 5. Sandbox / runner specification

The runner is deliberately conservative and **honest about its limits**. Modeled on `Evals/specialized_runners.py` (PR #851), adapted for arbitrary interpreters/binaries.

**Process launch**
- `subprocess.Popen(argv, cwd=<fresh scratch>, env=<scrubbed>, stdin=DEVNULL, stdout=PIPE, stderr=PIPE, text=True, shell=False, preexec_fn=_apply_limits)` — never `shell=True`, argv always a list.
- **Scratch cwd**: a fresh `tempfile.mkdtemp()` per run (under the OS temp dir, or a `[skills] script_scratch_root` if configured — reachable via the correct `get_cli_setting("skills", "script_scratch_root", default)` 3-arg form, **not** the section-dict form that hits the known unreachable-section bug at `config.py:3965`). **Never the skill directory** — a script cannot mutate its own trusted bundle (which would silently invalidate its fingerprints) and cannot read sibling skills. Deleted in a `finally` (best-effort; `ignore_errors=True`).
- **Env scrub**: start empty, set only `PATH="/usr/bin:/bin"`, `HOME=<scratch>`, `TMPDIR=<scratch>`, `LANG`/`LC_ALL` passthrough for text encoding, and nothing else. **No API keys, no user env** — the app's provider keys never reach a skill script. (A short allowlist of innocuous vars may be added if a real skill needs it; default deny.)

**Resource limits** — applied in `preexec_fn` (child, post-fork, pre-exec) so they bound *any* interpreter/binary:
- `RLIMIT_CPU` — soft/hard CPU seconds (default 10 s), a hard stop distinct from wall-clock.
- `RLIMIT_NPROC` — cap child fork-bombs (default 64; tuned so a Python interpreter's own threads don't trip it).
- `RLIMIT_NOFILE` — cap open descriptors (default 128).
- `RLIMIT_FSIZE` — cap bytes written to any single file (default a few MB) — a scratch-dir belt even though outputs aren't retained.
- `RLIMIT_AS` (address space / memory) — **best-effort**: on Darwin `setrlimit(RLIMIT_AS, ...)` raises `ValueError` and is silently skipped (documented; the macOS reality the Evals runner already faces at `specialized_runners.py:39-48`). A one-time `sandbox_warning` records that memory is uncapped on this platform.
- `preexec_fn` also `os.setsid()` so the whole child process group can be killed on timeout.

**Wall-clock timeout + kill** — `Popen.communicate(timeout=<wall>)` (default 60 s, ≤ the confirm/budget envelope). On `TimeoutExpired`: kill the **process group** (`os.killpg(SIGKILL)`), reap, and return `timed_out=True` with whatever partial output was captured. (Process-group kill avoids orphaned grandchildren that a bare `Popen.kill()` misses — the reason for `setsid`.)

**Output caps** — read/truncate stdout and stderr to a byte cap each (default 64 KiB, i.e. well under the agent loop's `content[:2000]` step truncation but generous enough to be useful; the tool-result formatter decides final trimming). `truncated_stdout`/`truncated_stderr` flags tell the agent output was clipped.

**Result → ToolResult** — the bridge closure formats `ScriptRunResult` into a `ToolResult(ok=..., ...)`: exit code, timed-out flag, and the (capped) stdout/stderr. A non-zero exit or a timeout is a **successful tool call reporting a failed script** (`ok=True` with the failure described), not a tool error — the agent should see and reason about the script's own failure. A *policy/trust/resolution* failure is `ok=False` with the reason (the agent may not retry its way past a trust block).

**`ScriptRunLimits` config** — a dataclass with all the numbers above, defaults as stated, overridable via a greenfield `[skills]` config block (read with the correct 3-arg `get_cli_setting` form). Documented in `rag_config_example.toml`-style example config.

## 6. Policy

New fail-closed registry rows in `runtime_policy/registry.py` under the `skills` domain (alongside `skills.read_file`, `skills.install_remote` at `:1020-1021`):
- `_resource("skills.run_script", actions=(LAUNCH,))` — yields action id `skills.run_script.launch.local` (and `.server`, which the scope service rewrites/rejects since execution is local-only).
Because `SEPARATED_SOURCES` (`registry.py:1013`) generates the `.local`/`.server` suffixes and `validate_registry_completeness()` runs at import (`:1367`), the new resource must satisfy the audited-capability/required-rows checks — the plan verifies the registry still validates. Unknown action ids already fail closed (`get_capability_entry` raises `PolicyDeniedError`, `registry.py:1354-1364`) — so if the row is ever missing, execution denies rather than proceeds.

The scope service exposes a public `enforce_run_script()` passthrough (§4.1-C) so the bridge closure can enforce policy *before* prompting the human (§7 step 1) — mirroring `enforce_install_remote`. The in-dispatch double-gate (scope `_enforce_policy` + local `_enforce`) remains, and re-enforcing in the closure is idempotent.

## 7. Error handling & taxonomy

All messages are user/agent-presentable (the confirm card and `ToolResult` surface them):
- **Policy denied** → `PolicyDeniedError.user_message` → `ToolResult(ok=False)`. No confirm prompt shown (enforce is step 1).
- **Trust blocked / revoked / mutated** → `SkillTrustBlockedError` (reason_code, trust_status) → `ToolResult(ok=False, error="This skill is not trusted to run scripts: <reason>")`. No prompt.
- **Bad path** (traversal, symlink, missing, `SKILL.md`, absolute, depth) → `ValueError("local_skill_script_not_found:<path>")` → `ToolResult(ok=False)`. Symlink-oracle-safe (identical error whether target exists or not — the #814 lesson).
- **Unrunnable type** (text, no interpreter mapping, no exec bit) → `unrunnable_script_type` domain error → `ToolResult(ok=False, error="No way to run <path>: unknown script type and not marked executable")`.
- **User denied** at the card / timeout / context-switch → `ToolResult(ok=False, error="You declined to run this script.")`. Fail-closed on any confirm exception.
- **Script ran, non-zero exit / timeout** → `ToolResult(ok=True, ...)` with exit code, `timed_out`, and captured output — the agent sees the failure and can react.
- **Runner internal error** (spawn failure, interpreter not found on scrubbed PATH) → `ToolResult(ok=False, error=<clear reason>)`; broad-catch wrap so a runner bug never kills the agent turn (the install-closure lesson).

Closure order in the bridge (fail-closed, no wasted prompts), mirroring the install closure at `console_agent_bridge.py:987-1043`:
```
1. scope.enforce_run_script()        →  policy (no prompt on deny)
2. describe_skill_script (B2)         →  trust/bad-path/unrunnable (no prompt on failure); yields ScriptPlan for the card
3. grant-check                        →  if script_grant_digest == current_digest: skip prompt
4. request_skill_script_confirm(plan) →  threading.Event, OUTSIDE asyncio.run; except → deny
   └─ if remember: grant_script_execution(skill_name)
5. run_skill_script (scope→local→runner, re-verifies authoritatively)   →  broad-catch-wrap
6. format ScriptRunResult → ToolResult
```
`describe_skill_script` (step 2) runs **before** confirm so a malformed / untrusted / unrunnable request fails with no prompt (the classify-before-confirm fix from the install spec) and so the card describes a real resolved script. Grant-check (step 3) uses the fresh digest. Step 5 re-verifies enforce+trust+resolve authoritatively — the plan from step 2 is display-only, never load-bearing for what executes.

## 8. Testing strategy

- **Runner unit tests** (`Skills_Interop/skill_script_runner.py`) — real temp scripts: happy stdout, non-zero exit, wall-clock timeout → killed process group (spawn a `sleep`/busy child, assert `timed_out` and no orphan), output truncation past the cap, env scrub (script prints `os.environ` → assert no API key / minimal PATH), scratch cwd is a temp dir not the skill dir, RLIMIT applied where supported + `sandbox_warning` emitted on Darwin for `RLIMIT_AS`. Interpreter-not-on-scrubbed-PATH → clear error.
- **Service tests** (`LocalSkillsService.run_skill_script`) — trust-blocked skill refuses (no spawn); revoked-mid-run refuses; traversal / symlink / `SKILL.md` / absolute path → not-found (symlink-oracle: existing vs missing target yield identical errors); text-with-map runs via interpreter; exec-bit file runs direct; unrunnable type errors; policy-denied (no enforcer wired must be proven a **real** enforcer in e2e, per the non-vacuity lesson — an enforcer-less scope service silently no-ops).
- **Grant store tests** — grant records current digest; grant satisfied only when digest matches; content change (re-fingerprint) drops the grant to per-run; revoke clears; persists across a fresh service instance.
- **Runtime plumbing tests** — schema pinned only when the tool is wired; dispatch branch routes `run_skill_script`; drift-guard test passes with the new name; a spawned subagent does **not** receive the tool (availability scoping non-vacuous — mutate to grant and watch the test fail).
- **HITL tests** — confirm allow/deny/always-allow round-trip (clone `Tests/UI/test_console_mcp_approval.py` / the install-confirm test); no-UI-bridge denies immediately (headless-stall regression); context-switch denies a pending confirm; card `markup=False` renders a hostile path/args literally; display gate shows the card.
- **e2e** (clone `test_e2e_install_skill_from_github_tree_url_real_services`) — real `ServicePolicyEnforcer` + `RuntimeSourceState` through the bridge, a real trusted skill with a tiny `scripts/hello.py`, confirm faked to allow: assert the agent's `ToolResult` carries the script's stdout; then a denied confirm asserts refusal; then a policy-off enforcer asserts denial (mutation-proven non-vacuous — deleting the registry row fails the e2e).

## 9. Residual risk (documented, not mitigated in v1)

- **Network egress** — a script may open sockets. Not blockable without a real jail. Documented; the human confirm is the compensating control (they reviewed the bytes and chose to run it).
- **User-level file reads** — the scrubbed env + scratch cwd stop *casual* access and self-tampering, but a determined script can still `open("/Users/...")`. Same posture as the Evals runner. A real jail (§ follow-up) is the mitigation.
- **Memory (Darwin)** — `RLIMIT_AS` uncapped on macOS; a runaway can OOM. `sandbox_warning` surfaces it; wall-clock + CPU limits bound runtime.
- **Compiled binaries** — direct-exec of an exec-bit binary runs bytes no human meaningfully reviewed (trust review shows binaries as size+sha256 only). This is the user's explicit "both mechanisms" choice; the per-run confirm names it as a direct-exec of an opaque binary so the human decides with that knowledge.

## 10. Follow-up layers (named, out of scope here)

- **Real containment** — macOS `sandbox-exec` profile (deny writes outside scratch, optional deny-network) and/or Linux seccomp/namespace; the honest next security layer.
- **Output files** — retained scratch + a bounded agent read-back path (its own contained read surface).
- **Frontmatter `allowed-scripts`** — let a skill declare which files are runnable (defense in depth on top of trust), folding into the frontmatter-conformance layer.
- **Grant-store consolidation** — if a future layer needs more per-skill local state, fold the script-grant JSON into a single skill-local sidecar.

## 11. Files touched (anticipated)

New: `Skills_Interop/skill_script_runner.py`, `Widgets/Chat_Widgets/skill_script_confirm_card.py`.
Modified: `Skills_Interop/local_skills_service.py` (run_skill_script + describe_skill_script + classify), `Skills_Interop/skills_scope_service.py` (run_skill_script + describe_skill_script passthroughs + enforce_run_script), `Skills_Interop/skill_trust_service.py` (grant store + digest accessor), `Agents/agent_models.py` (name + bindings), `Agents/tool_catalog.py` (schema), `Agents/agent_runtime.py` (LoopDeps field + dispatch), `Agents/agent_service.py` (ctor kwarg + pin + wiring), `Library/library_skills_state.py` (drift guard), `Chat/console_agent_bridge.py` (closure + wiring), `Chat/console_chat_controller.py` (HITL), `UI/Screens/chat_screen_state.py` (pending state), `Widgets/Chat_Widgets/chat_task_cards.py` (4th child + gate), `UI/Screens/chat_screen.py` (setter + @on), `runtime_policy/registry.py` (policy row), Library ▸ Skills trust panel (grant display + revoke), config example doc. Tests across `Tests/Skills/`, `Tests/Agents/`, `Tests/UI/`.
