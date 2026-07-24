# Security hardening cluster (TASK-330 + TASK-331 #1/#2 + TASK-332)

**Date:** 2026-07-24
**Backlog:** TASK-330 (git-clone transport/arg-injection, MEDIUM) + TASK-331 #1/#2 (tool-executor: sandbox root + pickle→json, LOW) + TASK-332 (eval-runner RLIMIT surfacing, LOW). One PR. The final security cluster of the LLM-harness review 2026-07 (327 agent-runtime durability remains after this).
**Branch:** `feat/security-hardening-cluster` (worktree off `origin/dev` @ `5a402cbb7`).
**Explicitly out of scope:** TASK-331 **#3** (the full permission/confirmation gate) — decomposed into its own next spec (a two-call-site, thread-crossing, tag-plumbing, confirmation-UI build ~as large as everything here combined). A follow-up task tracks it.

All findings verified against origin/dev. Confirmed no other branch edits the five target files ahead of dev (clean rebase runway); re-verify line numbers at implementation time regardless — heavy concurrent activity on dev.

## Part A — TASK-330: git-clone RCE hardening

### Vuln (confirmed)
`_clone_git_repository` (`Media/local_media_reading_service.py`, ~L4245) builds
`["git","clone","--depth","1", (…"--branch",str(ref)…), repo_url, str(checkout_path)]`
with `subprocess.run(command)` — argv (not shell), but **no scheme validation, no `--` separator, no env restriction**. `repo_url` and `ref` come from an ingestion-source config dict (`config.get("repo_url")` / `config.get("branch") or config.get("ref")`) that is stored verbatim from user/imported JSON with zero validation. `repo_url = "ext::sh -c '<cmd>'"` → git's `ext::` transport → arbitrary shell RCE; a leading-`-` `repo_url` (positional) → parsed as a git option (argument injection). Zero test coverage of this path today (the one existing git-source test takes the local-filesystem-copy branch and never reaches the clone).

### Fix (all three ACs, defense-in-depth)
1. **New validators in `Utils/input_validation.py`** (next to `validate_url`; `validate_url`/`egress.evaluate_url_policy` can't be reused — both hard-restrict to http/https, but git needs `ssh://`):
   - `validate_git_repo_url(url: str) -> None` — parse with `urlparse`; require **an explicit scheme in the allowlist `{"https", "ssh"}`** (deliberately NOT scp-shorthand `user@host:path` — its colon-before-slash / no-`://` shape is ambiguous with `ext::…`/`-flag:x` and a security-sensitive parser to get right; users needing SSH write `ssh://git@host/path`). Reject: any other/absent scheme (`ext`, `file`, `fd`, `git`, `http`, ``), whitespace/backslash/control chars, a leading `-`, and empty. Raise `ValidationError` with a clear message.
   - `validate_git_ref(ref: str) -> None` — reject a leading `-`, whitespace, control chars, `..`, and shell/ref-dangerous metacharacters; allow normal ref names (alphanumerics, `/`, `-`, `_`, `.`). Raise `ValidationError`.
2. **`_clone_git_repository`**: call the validators on `repo_url` (always) and `ref` (when truthy) **before** building argv → build `["git","clone","--depth","1", …("--branch", ref)…, "--", repo_url, str(checkout_path)]` (the `--` separator, so even a validator gap can't make a positional parse as an option) → `subprocess.run(command, env={**os.environ, "GIT_ALLOW_PROTOCOL": "https:ssh", "GIT_PROTOCOL_FROM_USER": "0"}, …)` so git itself blocks any non-{https,ssh} transport even in redirects/submodules.
3. **Tests** (`Tests/Media/`): `ext::` `repo_url`, `file://` `repo_url`, leading-dash `repo_url`, and leading-dash `ref` each raise (validator) **before** any `subprocess.run` (assert the mock was never called); a valid `https://…` builds argv containing `"--"` immediately before the url and runs with the restricted env (subprocess mocked — assert `command` and `env`).

### Noted follow-up (out of scope)
`_local_git_repository_path` resolves a `file://`/no-scheme `repo_url` that points at a real local dir and reads it directly (skipping the clone) — a secondary local-file-read vector. Out of this task's argv-injection scope; file as a follow-up.

## Part B — TASK-331 #1: real sandbox root

### Vuln (confirmed)
`ReadFileTool`/`ListDirectoryTool`/`WriteFileTool` (`Tools/file_operation_tools.py`, ~L63/168/334) call `validate_path(path, "file")` / `validate_path(path, "directory")` — the literal strings `"file"`/`"directory"` land in `validate_path`'s **`base_directory`** (sandbox root) parameter, resolving to a bogus, CWD-relative `<cwd>/file`. Fails closed only by accident (the root doesn't exist), so the tools are effectively non-functional; no real configured sandbox root exists.

### Fix
- Add a `[tools] file_sandbox_root` config key, **defaulting to `get_user_data_dir() / "tool_sandbox"`** (the real data-dir API is `get_user_data_dir()` in `config.py`, NOT the nonexistent `USER_DATA_DIR`). A shared module helper `_tool_sandbox_root() -> Path` reads the config, expands `~`, resolves, and `mkdir(parents=True, exist_ok=True)`.
- Pass `_tool_sandbox_root()` as `base_directory` at all three call sites instead of the literal strings.
- **Tests:** a traversal payload (`../../etc/passwd`) is still rejected against the real root; an in-sandbox relative path resolves inside the root; the root directory is created.

### Interaction with the deferred gate (on the record)
This makes the tools functional-within-a-sandbox, which removes the accidental confinement that today blocks `write_file` entirely — and TASK-331 **#3 (the confirmation gate) is deferred**. Acceptable posture because: the fs tools remain **default-OFF** (a user must explicitly enable `write_file`), and the sandbox defaults to a **narrow, empty** `<user_data>/tool_sandbox` (small blast radius; the user must explicitly widen `file_sandbox_root` to make the tools broadly useful). Stated here and in the #3 follow-up so it's a conscious tradeoff, not a silent regression.

## Part C — TASK-331 #2: pickle→json + fix the crash

### Vulns (confirmed — two bugs)
1. `ToolResultCache._load_from_disk`/`_save_to_disk` (`Tools/tool_executor.py`, ~L213/241) use `pickle.load`/`pickle.dump` on a local cache file — deserialization anti-pattern (a locally-planted malicious pickle → code execution on next load).
2. `get_tool_executor()` (~L631) does `from ..config import USER_DATA_DIR` — **a name that doesn't exist in `config.py`** (it has `get_user_data_dir()` / `BASE_DATA_DIR_CLI`), UNGUARDED — so `get_tool_executor()` **raises `ImportError` whenever `[tools] cache_enabled = true`**. (This is why the pickle path is currently unreachable, and why no legacy pickle cache files exist in the wild — clean cutover, no migration.)

### Fix
- Replace `pickle.load`/`pickle.dump` with `json.load`/`json.dump`. Cache values are `(result_dict, expiry_float)` tuples; JSON stores each as a 2-element list, reconstructed to a tuple on load. All current tool results are JSON-serializable dicts. Make the save **defensive**: on `TypeError` (a future non-serializable result), log + skip persistence (the in-memory cache still works) rather than crashing the save. The existing load try/except already degrades a corrupt/old file to an empty cache.
- Change `from ..config import USER_DATA_DIR` → `from ..config import get_user_data_dir` and `cache_dir = get_user_data_dir() / "tool_cache"`.
- **Tests** (`Tests/Tools/`): a cache round-trips through the JSON path (`set` → save → new-instance load → `get` returns the value with correct expiry-tuple shape); a corrupt/non-JSON cache file loads to an empty cache without raising; `get_tool_executor()` with cache enabled no longer raises `ImportError` (the persist path computes).

## Part D — TASK-332: surface RLIMIT_AS non-enforcement + document

### Vuln (confirmed, and the finding refined)
The eval code-runner (`Evals/specialized_runners.py`, generated child ~L378-401) sets RLIMITs inside the model-code subprocess. `RLIMIT_AS` (256MB memory) is wrapped in `except (ValueError, OSError): pass` — and on macOS/BSD `setrlimit(RLIMIT_AS, …)` raises `ValueError` (address-space limit aliased to RSS, unsettable), so the memory cap **silently no-ops with no warning/fallback**. (Refinement: `RLIMIT_NPROC` — also flagged — actually *works* on macOS; only `RLIMIT_AS` is the silent gap. The wall-clock `subprocess.run(timeout=…)` still bounds *time* cross-platform, so only *peak memory within ~5s* is unbounded on macOS.) The runner docstring overclaims "memory" as an enforced measure. The one existing memory-exhaustion test is vacuous (the static AST scan blocks its literal payload before the subprocess spawns).

### Fix (AC#1 surface non-enforcement, AC#2 document)
- **Parent-side detection** (robust; the child can't write a report file because `RLIMIT_FSIZE=0`, and grepping child stderr risks colliding with model output): a module-level `_memory_limit_enforced() -> bool` returning `False` on platforms where `RLIMIT_AS` can't apply (`platform.system() == "Darwin"`; extensible). When it returns `False`, the runner logs a **one-time** `WARNING` ("eval sandbox: RLIMIT_AS memory limit not enforced on this platform; model-generated code is bounded by time (timeout) but not peak memory") and records the caveat in each result (`results["sandbox_warnings"]`, a list) so callers/UI see it — no silent gap.
- **Docstring**: correct `_execute_code`'s security-measures docstring to note that on macOS/BSD the `RLIMIT_AS` memory cap is best-effort / not enforced, and that the primary cross-platform bound is the wall-clock timeout.
- **Tests** (`Tests/Evals/`): `_memory_limit_enforced()` returns `False` on Darwin / `True` on Linux (monkeypatch `platform.system`); a run on a non-enforcing platform surfaces a `sandbox_warnings` entry; the warning logs at most once. Platform-aware (no false failure on either OS).

## Cross-cutting

- One PR; ~5 commits: (A) git validators + `_clone_git_repository` wiring + tests; (B) sandbox root + tests; (C) pickle→json + import fix + tests; (D) RLIMIT surfacing + docstring + tests; (E) backlog.
- **Backlog:** TASK-330 → Done; TASK-332 → Done; TASK-331 → check ACs #1/#2, add Implementation Note that **AC #3 (the permission/confirmation gate) is split to a new follow-up task**; file that follow-up task (labels `tools,security`; description = the full-MCP-permission-integration scope from this brainstorm: two call sites A/B, risk-tagging the `Tool` ABC, reusing `MCPPermissionStore`/`resolve_effective_state` + `ChatApprovalCard`). Also file the Part-A `file://` local-read follow-up. IDs via the collision-safe two-namespace scan.
- Fresh worktree; `git add` only listed files (never `-A`); tests via the main checkout `.venv`.

## Out of scope / residual (explicit)
- TASK-331 #3 permission/confirmation gate (own spec).
- The `file://`-resolves-to-local-dir secondary read vector in the git-sync path (follow-up).
- A psutil RSS memory watchdog for the eval sandbox (the chosen approach surfaces non-enforcement rather than adding an enforcement mechanism; watchdog is a possible future enhancement).
- Broadening/validating the ingestion-source `config` dict at creation time (Part A validates at clone time, the security boundary; input-time validation is a defense-in-depth nicety, not required).
