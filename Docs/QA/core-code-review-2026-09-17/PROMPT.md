# Core-runtime code review — tldw_chatbook

You are reviewing `tldw_chatbook`, a Python ≥3.12 / Textual 8.x TUI (SQLite+FTS5, loguru,
httpx, pydantic). This is a **read-only review**. You produce a report and a triaged
candidate list. You do not fix, commit, file tasks, or write ADRs. Read `CLAUDE.md` and
`AGENTS.md` first; they override anything here that conflicts.

## 0. Ground rules (non-negotiable)

- **Isolation.** Work from a fresh worktree of `origin/dev`:
  `git worktree add ../tldw-review origin/dev` (or `EnterWorktree`). Record the SHA. Never
  `git stash`, never switch branches in the main checkout, never commit. Other agents share
  this repo.
- **Interpreter.** `.venv/bin/python` only. System `python3` is 3.9. The venv is
  uv-managed with no pip (`VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` if pytest is
  missing). `timeout(1)` does not exist on this machine.
- **Do not run the app.** Booting it regenerates `css/tldw_cli_modular.tcss` in whatever
  worktree it runs from. If a claim needs a live surface, put it in "Left UNVERIFIED" with
  the `.claude/skills/verify/SKILL.md` recipe as the check to run.
- **Tests.** Targeted runs only (`.venv/bin/python -m pytest Tests/<dir>/<file>.py -q`).
  No full-suite run without asking. `pytest --collect-only -q` over the tree is allowed
  and is the only check that sees through re-exports. Never verify with `-k`.
- **Lint.** `.venv/bin/ruff check --select E9,F63,F7,F82 <paths>` per package, fatal-only.
  Report counts; never `ruff format` anything.
- **Writes.** Only into `qa/core-code-review-2026-09-17/` and your scratchpad.
- **ADRs are settled.** `backlog/decisions/` holds 244 ADRs (max number 165, numbers
  collide — cite by full filename). Before recommending any structural change, grep the
  ADR titles; if one covers it, the finding is "out of scope by ADR <filename>", not a
  recommendation. Before proposing any god-module split, read
  `backlog/docs/library-decomposition-recipe.md` and use its shape.
- **Every finding carries evidence or is labelled unverified.** See §4. A confident
  reading of code is not a finding. The last review of this repo had ~40% of its causal
  attributions retired after tracing to code; the symptoms were real, the causes were not.

## 1. Scope

**Tier 1 (this review, ~0.9M lines).** Review every file in:

| Area | Paths | Lines |
|---|---|---:|
| Entry / config | `app.py`, `config.py`, `Constants.py`, `model_capabilities.py`, `Logging_Config.py`, `emergency_stop.py`, `runtime_policy/` | ~40k |
| Chat | `Chat/` | 205k |
| Agents / tools | `Agents/`, `Tools/`, `MCP/` | 79k |
| Data | `DB/` | 65k |
| Providers | `LLM_Calls/` | 22k |
| Events | `Event_Handlers/` | 17k |
| Shared helpers | `Utils/` (exclude `Utils/Splash_Screens/`) | 23k |
| RAG | `RAG_Search/` | 25k |
| Main screens | `UI/Screens/{chat,library,settings,personas}_screen.py`, `UI/Console_Modules/`, `UI/Library_Modules/`, `UI/Navigation/`, `UI/MCP_Modules/`, `UI/Persona_Modules/` | 226k |
| Main widgets | `Widgets/*.py` (top level), `Widgets/Console/`, `Widgets/Library/`, `Widgets/Persona_Widgets/`, `Widgets/Settings_Widgets/`, `Widgets/Chat_Widgets/` | 162k |

**Excluded, always:** `tldw_chatbook/.venv/` (596k vendored lines inside the package),
`Third_Party/`, `Tests/` (used only as evidence, never reviewed), `Utils/Splash_Screens/`.

**Reviewed once as a class, not per file:** the 31 `*_Interop/` packages (66k lines) and
the ~35 `*_scope_service.py` / `server_*_service.py` files elsewhere. They are one
template (`__init__` + `server_<x>_service.py` + `<x>_scope_service.py`;
`_normalize_mode` / `_require_client` / `_enforce_policy` copied verbatim with only the
enum name changed). Produce one cluster finding with the template, the copy count, and
the outward/downward walk; do not read them individually.

**Tier 2/3 (not this run; same prompt can be pointed at them later):** `Notes/`, `TTS/`,
`Library/`, `tldw_api/`, `Audio/`, `Subscriptions/`, `Character_Chat/`, `Chunking/`,
`Evals/`, `Local_Ingestion/`, `Scheduling/`, remaining `UI/` and `Widgets/` subpackages,
then everything else.

**Legacy / deprecated code (deletion candidates only).** Anything marked `DEPRECATED`,
`legacy`, `retired`, every route in `_SCREEN_ALIASES` (`UI/Navigation/screen_registry.py:231`),
`UI/Tools_Settings_Window.py` (`DEPRECATED (TASK-1346)`), and whatever the enhanced chat
path still reaches: `UI/Chat_Window_Enhanced.py` **no longer exists** (removed in
task-649; `Widgets/Chat_Widgets/chat_approval_card.py:22` still names it and CLAUDE.md
lines 47/51 still describe it — that doc drift is itself a P3). For each: importer count,
reachability from the shipped entry point, pinning tests, and a `delete / keep / unknown`
row. No review of their internals.

## 2. What you are looking for

Four dimensions. Every finding is tagged with exactly one.

**D1 Issues (correctness, safety, data).** Crash paths (unguarded `query_one` in timer
callbacks); thread-safety (module-level `threading.Lock` + raw SQL outside
`db.transaction()`; `run_worker(coroutine)` is NOT a thread, so sync sqlite inside it
blocks the loop); swallowed errors on data paths (`except Exception: pass`); resource
leaks; security at trust boundaries (inline traversal checks bypassing
`Utils/path_validation.py`, URL fetches bypassing `Utils/egress.py`, secrets reaching logs
past `Utils/log_sanitizer.py`); config drift (`get_cli_setting("a.b", key)` dotted-section
lookups are a known silently-broken shape); mutable class attributes on `TldwCli`/widgets
shared across instances; negative-predicate thread offload (`if is_memory: inline` threads
unknown shapes); Textual 8 `.plain`/`str(label)` read-back un-escaping user text;
recompose-lifecycle caches keyed by `id()` in screen dicts.

**D2 Efficiency.** `re.compile` inside function bodies (64 known); `get_cli_setting` in
`compose()`, loops, and retry loops (`app.py` compose has 7; `Summarization_General_Lib`
re-reads config per retry); `fetchall()` with no `LIMIT` on unbounded tables (126 in
`DB/`, 12 in `AgentRuns_DB.py` alone); sqlite on the event loop; O(n²) over conversation
or media lists; whole-screen `recompose=True` where a reactive refresh would do;
guarded-but-not-lazy imports (`try/except ImportError` at module scope still pays full
cost when installed); string `+=` in hot loops; `str(params)` on every query where the
`isEnabledFor` guard is missing.

**D3 Poor patterns.** God modules (list every file >5k lines with its responsibility
count); mixin-as-implementation (`UI/Speech/*_mixin.py` is the known shape); loguru and
stdlib `logging` in the same file (11 known; `Logging_Config.py` is the one legitimate
bridge); private helpers imported across packages; `__init__` side effects; dead shared
helpers; copy-pasted three-method blocks; function-body imports of modules that no longer
exist (mocked tests never catch these — three found in the last audit); `run_worker(
exclusive=True)` without `group=`; per-call `import` inside functions for non-optional
modules.

**D4 Duplication vs shared helpers.** The core question: functionality re-implemented in
N modules where one shared helper exists (in `Utils/`, `DB/base_db.py`,
`DB/sql_validation.py`, `Widgets/form_components.py`, `Widgets/base_components.py`,
`runtime_policy/`) or should. Two sub-cases, report them separately:
- **Helper exists, ignored.** Name the helper, its importer count, the N re-rolls.
- **No helper exists, N copies drifted.** Name the copies, the behavioural drift between
  them (that drift is itself a D1 finding when outputs feed storage or the wire), and the
  proposed canonical home (an existing `Utils` module by preference; a new one only when
  none fits).

### Seed clusters (measured 2026-09-17 at dev d8fb4053f9 — verify, extend, do not re-discover)

| Name | Files | Drift already seen |
|---|---:|---|
| `_maybe_await` | **60 byte-identical** (33 in `*_Interop`, 27 in core: UI 7, Notifications 3, Chat 2, MCP 1, …) | the copy in `Chat/chat_conversation_scope_service.py` was found in the 2026-07 perf audit to never actually defer local mode (sync sqlite on the loop; fixed there under task 283) — check whether the other 59 still carry the original shape |
| `_enforce_policy` | **43 byte-identical** (+ `_enforce` 49 same-name) | |
| `_cancel` / `_cancel_safe` / `_perform_safe_cancel` | 24 / 15 / 39 | verbatim groups — worker/timer cancel boilerplate |
| `_dump` | 11 byte-identical | |
| `_utc_now` / `_utc_now_iso` | 21 | ≥7 output formats: `strftime(...Z)`, `.isoformat()` with `+00:00`, `.replace("+00:00","Z")`, `timespec="milliseconds"`, `timespec="seconds"` no Z, `%f[:-3]+"Z"`, returns `datetime` |
| `_set_status` | 21 | bare `query_one().update()` / `try: ... except: pass` / `if self.is_mounted:` |
| `_get_connection` | 18 | 9 standalone, bypass `DB/base_db.py` (incl. 6 `Scheduling/db/migrations/v*_to_v*.py`) |
| `_normalize_mode` / `_require_client` / `_enforce_policy` | ~35 | verbatim template block, enum name only |
| `_identity` | 17 | |
| `_initialize_schema` | 15 | |
| `_now` / `_now_iso` / `_now_ns` | 15 | |
| `_truncate*` | 15 (+26 inline `[:n] + "..."`) | `Utils/Utils.py:253 truncate_content` has 0 callers; `…` vs `...` for the same 120 budget |
| `_reject_json_constant` + `_strict_json_loads` + `_json_shape_is_safe` | 14 / 4 / 2 | `LLM_Calls/hosted_chat.py:630-670` ≡ `LLM_Calls/qwencloud_streaming.py:61-75` byte-identical; `Chat/thinking_blocks.py:221` ≡ `Chat/provider_continuation.py:204` second identical pair |
| `_coerce_bool*` / `_coerce_int*` | 13 / 12 | |
| `_safe_text` | 11 | |
| `_resolve_api_key` / `_resolve_base_url` | 10 / 10 | provider credential precedence is ADR-governed (`012-provider-credential-settings-boundary.md`) — check each copy honours it |
| byte-size formatter | 10 (+318 raw `1024*1024`) | 4 unit ladders, `.0f` vs `.1f`, TB/PB/YB ceilings; `Widgets/Console/console_transcript.py:639` documents its duplication of `Chat/attachment_core._format_size` on purpose — report as *documented* duplication |
| `_normalize_keywords` | 8 | |
| `_json_safe*` / `_load_json*` | 7 / 3 | `Evals/eval_runner.py:296` ≡ `Evals/dataset_loader.py:163` |
| filename sanitizer | 5 | `Utils/text.py:47`, `Utils/path_validation.py:236`, `Utils/file_extraction.py:614`, two `_safe_filename` |
| token estimate `len(text)/4` | 6 | vs `Utils/token_counter.py:191 _chars_estimate` (12 importers) |

### Seed "helper exists, ignored" (importers at dev d8fb4053f9)

| Helper | Importers | Re-rolls |
|---|---:|---|
| `Utils/ui_helpers.py` (`UIHelpers`) | 0 | — dead |
| `Utils/pagination.py` (`PaginatedResult`, `LazyPaginator`) | 0 | — dead |
| `Utils/widget_helpers.py` | 1 | |
| `Utils/Utils.py:118 ensure_directory_exists` | 0 | 106 raw `mkdir(parents=True, exist_ok=True)` in 73 files |
| `Utils/Utils.py:253 truncate_content` | 0 | 26 files |
| `Utils/secure_temp_files.py` | 7 | 46 files use raw `tempfile.*` |
| `Utils/atomic_file_ops.py` | 9 | 31 files hand-roll tmp + `os.replace` (`MCP/permission_store.py`, `Scheduling/scheduler_heartbeat.py`, `Notes/file_notes_service.py`, `emergency_stop.py`, …) |
| `Utils/optional_deps.py` | 43 | 102 files guard imports with `try/except ImportError` and never import it |
| `Utils/path_validation.py` | 111 | 4 files still inline `is_relative_to` / `commonpath` (`STT/executor_worker.py:503`, `Notes/file_notes_git_service.py:10188`, `Library/library_ingest_state.py:2701`) |
| `Widgets/form_components.py` | 2 | |

### Known-deliberate — do NOT flag (verified in prior audits)

Transcript reconciler is genuinely incremental; Library has no per-keystroke DB search;
browser-search debounce with cancellation token; subscriptions scheduler `thread=True`;
screen registry lazy; config reads are cache-backed; `Local_Ingestion`'s PEP 562 lazy
`__init__`; `load_settings`' decrypt call (defense-in-depth, declined removal);
`UI/Logs_Window.py:459 _compile_pattern` memoizes; hidden-column form
`fts.messages_fts MATCH` is required (bare alias fails in correlated EXISTS+JOIN);
`Logging_Config.py` imports both loggers; `_GATEABLE_BUILTINS` row copy is the
registration contract (see CLAUDE.md "New Tool"). If you believe one of these is wrong,
put it in "Retired / contested" with new evidence; do not file it as a finding.

## 3. Method — five phases, each with an output file

Write `progress.md` after every package (phase, package, files read in full vs sampled,
findings count). The run will outlive one context window; this is how it resumes.

### Phase 0 — Provenance and baseline (write `report.md` §Scope first)
Record: worktree path, `origin/dev` SHA, `git status --porcelain` empty, Python version,
Textual version (`.venv/bin/python -c "import textual; print(textual.__version__)"`).
Run `./scripts/preflight.sh`; record pass/fail. Run the fatal-only ruff per Tier-1 package;
record counts as baseline (do not fix). Grep `backlog/tasks/` titles for `dedup|duplicate|
consolidat|god module|split|Utils|helper` and list open tasks that already cover an area —
this becomes the "Already handled" list and every later finding checks against it.

### Phase 1 — Mechanical candidate generation (stdlib only; outputs in `candidates/`)
These produce CANDIDATES. Nothing here is a finding until Phase 2 or 3 reads it.

1a. **Duplicate-definition census** — write this to your scratchpad and run it over
Tier-1 paths (exclude `.venv`, `Third_Party`, `Tests`):

Run as `.venv/bin/python -W ignore dup_census.py tldw_chatbook candidates` (≈2 min for
the whole package; `-W ignore` silences SyntaxWarnings from files with bad escapes).
Tested 2026-09-17 at dev d8fb4053f9 over the whole package: 1,823 same-name rows (≥3
files), 244 verbatim-clone groups, 665 shape-clone groups. Your Tier-1 counts should be
smaller; if they are larger, your exclusions are wrong.

```python
import ast, copy, hashlib, sys, pathlib, collections
LIFECYCLE = {"compose", "render", "__init__", "on_mount", "on_unmount"}
by_name, by_body, by_shape = (collections.defaultdict(list) for _ in range(3))
class Anon(ast.NodeTransformer):
    def visit_Name(self, n): n.id = "_"; return n
    def visit_arg(self, n): n.arg = "_"; return n
    def visit_Attribute(self, n):
        self.generic_visit(n)
        if not n.attr.startswith("__"): n.attr = "_"
        return n
    def visit_Constant(self, n): n.value = type(n.value).__name__; return n
def _docstring(body):
    return body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
for p in pathlib.Path(sys.argv[1]).rglob("*.py"):
    if any(x in p.parts for x in (".venv", "Third_Party", "Tests", "__pycache__")): continue
    try: tree = ast.parse(p.read_text(encoding="utf-8"))
    except Exception: continue
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        name = node.name
        if name in LIFECYCLE or name.startswith(("on_", "watch_", "action_", "_on_")): continue
        loc = f"{p}:{node.lineno}"
        by_name[name].append(loc)                       # one-liners count here (that is where _utc_now lives)
        body = node.body[1:] if _docstring(node.body) else node.body
        if len(body) < 2: continue                      # but not for clone hashes: too noisy
        raw = hashlib.sha1(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()[:12]
        shape = hashlib.sha1(ast.dump(Anon().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))).encode()).hexdigest()[:12]
        by_body[raw].append(f"{name}@{loc}"); by_shape[shape].append(f"{name}@{loc}")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "candidates"); out.mkdir(exist_ok=True)
def emit(fn, groups, min_files):
    with open(out / fn, "w") as f:
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            files = {x.split("@")[-1].rsplit(":", 1)[0] for x in v}
            if len(files) >= min_files: f.write(f"{len(files)}\t{k}\t{' '.join(v)}\n")
emit("dup_by_name.tsv", by_name, 3)   # same name, >=3 files (weak signal)
emit("dup_verbatim.tsv", by_body, 2)  # byte-identical bodies, docstring stripped (strong)
emit("dup_shape.tsv", by_shape, 2)    # identical after renaming names/args/constants (strong)
```
Verbatim and shape clones are the strong signal; same-name is the weak one. Rows in
`dup_verbatim.tsv` are the first things Phase 3 reads.

1b. **Shared-helper adoption census.** For every public `def`/class in `Utils/*.py`,
`DB/base_db.py`, `DB/sql_validation.py`, `Widgets/form_components.py`,
`Widgets/base_components.py`: count importing files with
`rg -l --glob '!Tests' --glob '!.venv' -e 'from (tldw_chatbook\.|\.+)Utils(\.X| import .*\bX\b)' -e 'Utils\.X\b'`
(substitute X; relative imports `from ..Utils.X`, `from ...Utils import X` both count).
Output `candidates/helper_adoption.tsv`: helper, importers, and for the zero/low ones the
`rg` for the hand-rolled shape (use the seed table's shapes).

1c. **Pattern greps** (one TSV each, `file:line` rows, with counts in `progress.md`):
`re\.compile\(` inside a def (use `ast`, not rg, to know it's inside a function);
`get_cli_setting\(` inside `compose`/`for`/`while`/retry; `fetchall\(\)` where the
preceding `execute` string has no `LIMIT`; `query_one\(` inside a `set_interval`/
`set_timer` callback without an enclosing `try`; `run_worker\(.*exclusive=True` without
`group=`; `except Exception:\s*\n\s*pass`; `threading\.(R)?Lock\(\)` in files that also
call `.execute(`; `^try:\n\s+(from|import) ` in files that do not import `optional_deps`;
`from loguru import logger` AND `^import logging` in the same file; `strftime\(` distinct
format strings; `\[:\d+\]\s*\+\s*["'](\.\.\.|…)`; `mkdir\(parents=True, exist_ok=True\)`;
`os\.replace\(` in files not importing `atomic_file_ops`; `tempfile\.` in files not
importing `secure_temp_files`; `DEPRECATED|legacy|retired` markers; `_SCREEN_ALIASES`
entries.

### Phase 2 — Per-package read (subagents, read-only)
Dispatch one read-only subagent per row of the Tier-1 table; split `Chat/`,
`UI/Screens`, `Widgets/Console` and `DB/` so no subagent holds more than ~60k lines, and
give any file >10k lines its own subagent with line ranges. Run ≤6 concurrently. Each
subagent gets: this file's §0, §2, §4; the candidate rows whose paths fall in its
package; the "Already handled" list; and the output contract in §5. It must return:
files read in full vs sampled (be honest — this goes in the coverage table), findings in
§5 format, and every candidate row it examined marked `confirmed / retired (why) /
unverified (check)`. A subagent that reports zero findings for a 60k-line package must
say what it read; that is a coverage statement, not a clean bill.

### Phase 3 — Cross-module consolidation (you, not a subagent)
Take every D4 candidate that survived Phase 2 and every `dup_verbatim`/`dup_shape` row
spanning ≥2 packages. For each cluster:
1. Read every copy. Table the behavioural differences (formats, guards, error handling,
   return types). Drift that reaches storage, the wire, or a user-visible string is a D1
   finding in its own right.
2. Pick the canonical home: an existing helper if one exists; else the `Utils` module
   whose docstring already owns the concept; else propose a new module name and say why
   none fits. Cite `Utils/fd_protection.py` and `Utils/log_widget_manager.py` as the
   repo's own precedent for "de-duped from N call sites".
3. Walk the graph **both ways**: outward from the copies (what a consolidation may
   delete) and downward from the proposed helper (what it will pull in). Report both LOC
   numbers; a helper that would import a screen module is not a helper.
4. Check for pinning tests: `rg` the copy's name and its output shape (fragments and
   `parametrize` tables, not sentences) in `Tests/`, then `pytest --collect-only -q |
   rg <name>` to see through re-exports. A test whose name states the current behaviour
   as a requirement means the difference is a decision; say so.
5. Size it: S (<100 lines, mechanical swap, no behaviour change), M (one PR, a format or
   guard has to be chosen), L (needs an ADR — cross-module interface, storage format, or
   security boundary).

### Phase 4 — Verify the top findings
For every P0 and P1 and for every S-sized "helper exists, ignored" claim: reproduce it.
A script in the scratchpad against an in-memory DB, a targeted pytest, or an `ast`
check that the helper genuinely has zero importers after re-exports. Record the exact
command and its output in the finding. Anything you could not run stays a finding but
moves to `Confidence: inferred` and gets a row in "Left UNVERIFIED" with the command that
would settle it. Do not promote an inferred finding to P0/P1.

### Phase 5 — Write the deliverables (§5). Then stop.

## 4. Rules of evidence

1. A bare-name grep is a candidate finder, never evidence. `.create(` matches
   `completions.create(`. Resolve what a name is (import, `ast`, `--collect-only`) before
   claiming it is used, unused, or duplicated.
2. Grep cannot see through a re-export. Before "dead", run `pytest --collect-only -q` and
   `rg` the exact module path (not a substring).
3. Any "already handled / out of scope / obviously fine" you state without running it is
   an untested claim. Label it.
4. A pinning test that can go red beats any reading. A green test proves nothing until
   you have seen it fail. A test whose name asserts the current behaviour is the repo
   telling you the "bug" is a decision.
5. Trace to the code path; never infer cause from an aggregate (a counter, a diagnostic
   field, a timer list). The last review blamed footer timers for a stall because a
   diagnostic field named them.
6. Report retired findings as findings. "Symptom real, cause wrong, retired with
   evidence" is a result; silently dropping it is not.
7. State coverage. Which files were read in full, which sampled, which only mechanically
   scanned. A number without its coverage is not comparable to anything.

## 5. Output contract

### `report.md` (house style; see `qa/buddy-uat-2026-09-04/report.md` for the shape)

```
# Core-runtime code review — 2026-09-17
**Verdict:** <one sentence: the single biggest thing and the single biggest risk>

## Scope and provenance
worktree, SHA, tree clean, Python/Textual versions, preflight result, ruff fatal-only
baseline per package, "Already handled" open tasks, coverage table (package | lines |
read in full | sampled | mechanical only).

## Executive summary
Top 10 findings, one line each: [P#][D#] symptom — file:line — Size.

## Findings
One `###` per finding, ordered P0→P3, then by dimension:

### P1 [D4] — <symptom stated as a fact, not a recommendation>
- Where: `path:line` (range if needed; all copies if a cluster)
- Evidence: `<command>` → <result>          (or: read only — see UNVERIFIED)
- Why it matters: one sentence, concrete consequence
- Recommended correction: what, where, and the canonical home for D4
- Size: S/M/L · ADR: no | yes (<full filename or "new">) · Confidence: verified | inferred
- Pinning test: none | `Tests/...::test_x` (+ whether it states the behaviour as a requirement)
- Already covered: none | task-NNNNN

## Duplication clusters (D4)
| Cluster | Copies (files) | Helper exists? (importers) | Drift | Outward / downward LOC | Canonical home | Size | Rec |

## Dead or under-adopted shared helpers
| Helper | Importers | Hand-rolled equivalents (count) | Rec: adopt / delete / keep |

## Legacy reachability
| Symbol / route | Marker | Importers | Reachable from entry? | Pinning tests | delete / keep / unknown |

## Verified-fine (do not "fix")
Things that look like smells and are not, with the evidence. Include the seed list.

## Retired / contested
Findings you or a subagent raised and then retired, with the evidence that retired them.

## Left UNVERIFIED
| Claim | Why not verified | Check to run (literal command) |

## Method appendix
Scripts used (paths in scratchpad, copied into `candidates/`), counts per candidate
category, subagent list with package + lines + files read.
```

Severity: **P0** data loss, security boundary, crash on a shipped path. **P1** wrong
behaviour a user can hit, or a hot-path cost measured or clearly bounded (tick, compose,
per-query). **P2** duplication with behavioural drift, pattern that will produce a P1, a
dead helper with ≥10 re-rolls. **P3** consistency, naming, documented duplication.

### `gap-candidates.md`
Every finding with `Rec` ✅ (file), ➖ (file only for breadth), ❌ (recommend against —
give the design reason), plus Size and the ADR check. Group ✅ into proposed batch tasks
by canonical home (one task per helper being adopted, one per cluster being consolidated,
one per god-module) so each is a single PR. Do not file them. List "Already handled" with
task ids at the top.

### `candidates/*.tsv`, `progress.md` — as described in §3.

## 6. Budget and stopping

Tier 1 is ~0.9M lines. Prefer depth on `Chat/`, `DB/`, `Agents/`, `Utils/`, `app.py`,
`config.py`, and the four main screens over breadth. If you must sample, sample the
largest files by symbol cluster and say so. Stop when Phase 5 is written; do not start
Tier 2. Finish with the verdict line and the path to `report.md`.
