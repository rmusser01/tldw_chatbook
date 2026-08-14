# TASK-15810: Bounded First Library RAG Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and remove the measured CPU spin in the first fresh-profile Library RAG Answer query so the exact 36-note cold fixture renders its first Evidence row in under 30 seconds without freezing the TUI or permitting overlapping stale retrieval work.

**Architecture:** Preserve Library's existing request/state ownership and the process-wide `EnhancedRAGServiceV2` runtime. Reproduce and profile the real Library-to-engine path first; only after the hot Python frame is named may this plan select one lowest shared owner and one mechanism-based regression. Keep CPU work off Textual's event loop, distinguish stale-UI fencing from underlying-work termination, and serialize residual non-cooperative work rather than creating a second retrieval path.

**Tech Stack:** Python 3.11+, asyncio, Textual 8.x workers/Pilot, `EnhancedRAGServiceV2`, Chroma/SQLite FTS5, py-spy or cProfile, pytest, Ruff.

---

## Authority and global constraints

- Design authority: `Docs/superpowers/specs/2026-08-13-task-15810-library-rag-first-query-design.md`.
- Task authority: `backlog/tasks/task-15810 - Library-RAG-Answers-first-query-on-a-fresh-profile-never-returns-CPU-bound-reproduced-twice.md`.
- Existing ownership: ADR-003 (`backlog/decisions/003-settings-library-rag-defaults.md`) and ADR-005 (`backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`).
- Worktree: `.worktrees/task-15810-rag-first-query`; branch `codex/task-15810-rag-first-query`, rebased on `origin/dev` before this plan.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Before every profile/test command, assert `tldw_chatbook.__file__` resolves beneath this worktree by setting `PYTHONPATH` to the worktree root. A main-checkout import is a failed run.
- Baseline at plan time: 143 focused Library RAG service/Pilot tests passed.
- Every production edit is forbidden until Task 1 names the hot frame and an
  independently approved plan amendment names that edit's exact file/function,
  complete RED test, mutation, complete implementation snippet, owner suite,
  Ruff targets, and downstream commands. This includes event-loop offloading,
  cooperative cancellation, and serialization discovered after the first fix.
- Never log or store query text, note bodies, retrieved content, answer prompts, credentials, or secrets in profiler/diagnostic artifacts. The approved fixed query may appear in the fixture description, but profile output remains symbol/timing-only.
- No new dependency, retrieval implementation, runtime, process pool, cache layer, or background warm-up. If profiling demands a new durable/runtime/service boundary, stop and revisit the ADR decision.
- Use @superpowers:systematic-debugging through the profile checkpoint, @superpowers:test-driven-development for every bug fix, @ponytail for the smallest measured correction, @textual-tui for event-loop/worker behavior, and @superpowers:verification-before-completion before any completion claim.

## File map

Files fixed before profiling:

- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/profile-report.md` — fixture provenance, Python hot frame/callers, bounded timing/count evidence, and root-cause statement.
- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/fixture-manifest.sha256` — sorted SHA-256 manifest for the 36 User Guide Markdown files.
- Create at closeout: `Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md` — unprofiled timing, responsive-input check, effective-config/lsof evidence, and real-profile fingerprints.
- Modify: this plan — Task 2's profile checkpoint replaces the conditional owner table with the exact production/test files and complete minimal implementation snippet.
- Modify: the TASK-15810 backlog file — plan/notes/AC/status evidence.

Files selected only by the named profile (choose the smallest applicable row; do not edit the other candidates):

| Profiled owner | Production file | Primary regression file |
|---|---|---|
| Shared runtime construction | `tldw_chatbook/RAG_Search/ingestion_indexing.py` | `Tests/RAG/test_ingestion_indexing.py` |
| Concrete runtime wrapper | `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py` | `Tests/RAG/simplified/test_rag_service_basic.py` or a new focused `Tests/RAG/test_library_first_query_cpu_spin.py` |
| Base search/FTS/vector implementation | the named module under `tldw_chatbook/RAG_Search/simplified/` | the closest existing `Tests/RAG/` mechanism suite or the new focused file above |
| Library runtime/search boundary | `tldw_chatbook/Library/library_local_rag_search_service.py` | `Tests/Library/test_library_local_rag_search_service.py` |
| Textual worker/state boundary | `tldw_chatbook/UI/Screens/library_screen.py` | `Tests/UI/test_product_maturity_gate16_library_search_rag.py` |

The final implementation may modify more than one row only when the profile's
call chain proves both are required (for example, one engine correction plus
one Library offloading/serialization boundary). Record that evidence in the
Task 2 amendment before editing either production file.

---

### Task 1: Freeze the cold fixture and capture the Python hot frame

**Files:**

- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/fixture-manifest.sha256`
- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/profile-report.md`
- Temporary only: a validated scratch root under `/tmp` and profiler outputs under that root

- [ ] **Step 1: Prove the branch and import provenance**

Run:

```bash
git status --short --branch
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import pathlib, tldw_chatbook; p=pathlib.Path(tldw_chatbook.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd().resolve())'
find Docs/User_Guide -type f -name '*.md' | sort | wc -l
```

Expected: clean task branch, import path under this worktree, and exactly `36` Markdown files.

- [ ] **Step 2: Freeze the corpus manifest before seeding**

Run `find Docs/User_Guide -type f -name '*.md' -print0 | sort -z | xargs -0 shasum -a 256`, inspect all 36 lines, then add the exact output to `fixture-manifest.sha256` with `apply_patch`.

Expected: 36 sorted, repository-relative entries and no files outside `Docs/User_Guide`.

- [ ] **Step 3: Create and validate the scratch profile**

Create a root with `mktemp -d /tmp/tldw-task15810.XXXXXX`. Under it, create an effective TOML using `apply_patch` with:

```toml
[paths]
data_dir = "/tmp/<resolved-task-root>/data"

[rag.service]
profile = "hybrid_basic"

[llm_api_settings]
default_api = "custom-openai-api"

[api_settings.custom]
api_url = "http://127.0.0.1:19090/v1/chat/completions"
model = "task15810-loopback"
streaming = false

[model_catalog]
auto_refresh_enabled = false
```

Create a loopback answer stub with `apply_patch` under the scratch root. It
must use `ThreadingHTTPServer(("127.0.0.1", 19090), Handler)`, accept only
`/v1/chat/completions`, read and discard the bounded request body without
parsing or logging it (reject `Content-Length` above 1 MiB), return one static
OpenAI-compatible non-streaming JSON answer, override `log_message()` to do
nothing, and expose a body-free `/health` probe. Refuse to start if port 19090
is already occupied. The stub never writes request bodies to memory beyond the
bounded read, disk, stdout, or logs.

Every seed, profile, and acceptance command starts from `/usr/bin/env -i`, so no
cloud credential or inherited proxy can survive. Add only:

```text
TLDW_TEST_MODE=1
HOME=<scratch>/home
XDG_DATA_HOME=<scratch>/xdg-data
XDG_CONFIG_HOME=<scratch>/xdg-config
TLDW_CONFIG_PATH=<scratch>/config.toml
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/Users/macbook-dev/.cache/huggingface
HF_HUB_CACHE=/Users/macbook-dev/.cache/huggingface/hub
NO_PROXY=127.0.0.1,localhost
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
PYTHONPATH=<this-worktree>
PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
LANG=en_US.UTF-8
```

Do not add `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, any `*_API_KEY`, or any
provider-specific environment variable. Run without filesystem or network
escalation: the real Hugging Face cache is readable but outside writable roots.
Preflight that
`models--sentence-transformers--all-MiniLM-L6-v2/snapshots` exists beneath the
declared cache. `curl -sf http://127.0.0.1:19090/health` must succeed; no other
network destination is configured.

Parse the TOML with `tomllib` before launch and after every boot. Resolve
`[paths].data_dir`; assert it is beneath the scratch root and differs from the
real profile path. In the same clean environment, import
`library_rag_answer_provider_gate()` and assert provider
`custom-openai-api`, no credential recovery, and an enabled Library Run gate.
Hash the real config and data inventory before launch.

Expected: all assertions pass before any app import or DB open.

- [ ] **Step 4: Seed through production APIs in a separate process**

Create a temporary seed harness under the scratch root with `apply_patch`. Its
production construction is exactly:

```python
from tldw_chatbook.config import get_chachanotes_db_path, get_rag_indexing_db_path
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    get_shared_rag_service,
    index_entries,
    note_index_entry,
)

notes_db = CharactersRAGDB(get_chachanotes_db_path(), client_id="task15810-seed")
indexing_db = RAGIndexingDB(get_rag_indexing_db_path())
service = get_shared_rag_service()
assert service is not None
```

It must then:

1. load the 36 manifest paths in sorted order;
2. call `notes_db.add_note(path.stem, path.read_text(encoding="utf-8"))`
   for each and retain every returned note ID;
3. read each row back with `notes_db.get_note_by_id(note_id)`, convert it with
   `note_index_entry()`, and assert 36 non-None `IndexEntry` objects;
4. await `index_entries(service, indexing_db, entries)` and assert the exact
   summary `indexed == 36`, `skipped == 0`, `failed == 0`;
5. assert `service.vector_store.get_collection_stats()["count"] > 0`, then
   fetch metadata only with
   `service.vector_store.collection.get(include=["metadatas"])` and compare
   the resulting `source_id` set exactly with the 36 returned note IDs; and
6. close both DB owners and the service in `finally`, printing only aggregate
   counts, paths, IDs, and timing—never note content, documents, or embeddings.

Run it once and let the seed process exit, guaranteeing the later TUI process has no pre-existing in-process shared runtime or query cache.

Expected: notes `36`, indexed notes `36`, vector chunks `N > 0`, and no failure outcomes.

- [ ] **Step 5: Boot the real TUI and prove handle isolation**

Launch `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m
tldw_chatbook.app` in a true PTY at 100x30, prefixed by the exact
`/usr/bin/env -i` assignment block from Step 3. Resolve and inspect the exact
PID. Run `lsof -p <validated-pid>` and assert:

- zero handles beneath the real config/data/profile roots;
- every database/vector handle is beneath the scratch root; and
- at least one live scratch data handle exists.

In Library, verify `Notes (36)`, RAG Answer mode, Notes-only scope, top_k 15, citations on, and active `hybrid_basic`/hybrid disclosure.

Expected: all preflight UI and isolation conditions match the design fixture.

- [ ] **Step 6: Reproduce the cold query without profiling first**

Submit the fixed query `how do I schedule a watchlist brief`. Record Run time, visible status transitions, CPU use, and time-to-first Evidence. Do not wait silently beyond the point needed to establish the existing >30-second spin and capture a profile.

Expected on the unfixed branch: no Evidence within 30 seconds and sustained CPU-bound work, reproducing the task. If it completes under 30 seconds, repeat in a second new TUI process; if both pass, stop and report non-reproduction rather than inventing a fix.

- [ ] **Step 7: Capture Python-level profile evidence**

Preferred command after validating the PID:

```bash
py-spy record --pid <validated-pid> --duration 15 --rate 100 --format raw --output <scratch>/first-query.raw
```

If attach is unavailable, stop the PTY and use the same scratch profile with a
cProfile harness that calls the production `run_library_rag_search()` →
`LibraryLocalRagSearchService` → resolved `EnhancedRAGServiceV2` path once. The
harness must own `cProfile.Profile`, arm a 15-second `signal.setitimer()` before
`asyncio.run()`, raise a private deadline exception from the signal handler,
catch only that deadline, and always cancel the timer, disable the profiler,
and call `dump_stats()` in `finally`:

Run the harness with the exact `/usr/bin/env -i` assignment block from Step 3
followed by this command, then inspect it with the second command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python <scratch>/profile_first_query.py
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pstats <scratch>/first-query.pstats
```

Sort pstats by cumulative and internal time. Reject the fallback artifact unless
it contains both the Library service boundary and a named hot Python frame.
Capture that frame, immediate callers, call/sample counts, and whether time is
initialization-only or repeated query work.

Expected: a named Python frame and caller chain; a C-only or generic “embedding stack” attribution is insufficient.

- [ ] **Step 8: Write and commit the profiler report**

Using `apply_patch`, write `profile-report.md` with fixture commit/manifest, exact isolation checks, profiler method/command, named frame, callers, sample/call counts, elapsed reproduction, root-cause hypothesis, and rejected alternatives. Include symbols/timing only.

Run:

```bash
git diff --check
git add Docs/superpowers/qa/task-15810-library-rag-first-query/
git commit -m "docs(rag): profile the cold Library query (TASK-15810)"
```

Expected: committed evidence before any production diff exists.

---

### Task 2: Convert the profile into one exact RED regression and amend this plan

**Files:**

- Modify: this plan
- Test: exactly one primary regression file selected from the file map
- Test if engine-owned: one focused test must instantiate the concrete runtime returned by `get_shared_rag_service()` (`EnhancedRAGServiceV2`), not only a fake or base `RAGService`

- [ ] **Step 1: Amend the plan before writing the test**

Replace this Task 2 file list with the exact test path and test name. Add:

- the named production function and call chain from `profile-report.md`;
- the faulty mechanism in one sentence;
- a complete minimal failing test body;
- its exact pytest command and expected failure; and
- the smallest proposed production edit as a complete code snippet;
- complete heartbeat and supersession test bodies at the real changed seam;
- every exact owner-suite/node command and Ruff target used by Tasks 3–5; and
- replacements for every profile-dependent placeholder/candidate command in
  downstream Tasks 3–5, deleting all unselected alternatives.

Delete unselected candidate rows from the amendment. If the proposal changes a runtime/service/storage boundary, stop for an ADR instead.

- [ ] **Step 2: Independently re-review the amended plan**

Dispatch one plan-document reviewer with this plan, the design spec, task, and profile report. Do not proceed until it confirms that the proposed test fails for the profiled mechanism and the fix is the lowest shared owner. Limit the loop to three review passes.

- [ ] **Step 3: Write every regression before production code**

Write the amended mechanism regression, heartbeat regression, and
supersession/serialization regression before production code. The last two are
mandatory even if they already pass on the unfixed implementation; if either
fails and the approved production snippet does not cover its mechanism, stop,
amend, and re-review again. Prefer deterministic counts/state/termination
assertions over a wall-clock timeout. If the owner is in or below the engine
boundary, drive the concrete resolved `EnhancedRAGServiceV2` at least once.

- [ ] **Step 4: Run RED and inspect the reason**

Run every exact amended pytest node.

Expected: FAIL for the named mechanism. A missing dependency, fixture error, empty index, fake-runtime mismatch, or generic timeout is not the required RED.

- [ ] **Step 5: Mutation-check the regression**

Apply each inverse/smallest faulty mutation described in the amendment (or
retain the current faulty implementation), rerun, and prove the corresponding
assertion fails. For event-loop work, move the gated call back onto the event
loop and prove the heartbeat assertion fails without hanging. Restore only
through `apply_patch` and verify the worktree diff afterward.

Expected: the test detects the measured fault, not incidental timing.

---

### Task 3: Apply the minimal measured correction

**Files:**

- Modify: the exact production file(s) approved in the Task 2 amendment
- Test: the exact RED regression file

- [ ] **Step 1: Implement only the amended code snippet**

Keep the request/outcome contracts, active-profile routing, ranking, source filtering, citations, and error shapes unchanged unless the profile explicitly identified one of them as the faulty owner. Do not add speculative abstraction or a second retrieval path.

- [ ] **Step 2: Run the RED node to GREEN**

Run every exact amended pytest node.

Expected: PASS with the measured operation bounded/terminated as asserted.

- [ ] **Step 3: Run the closest owner suite**

Task 2 replaces this candidate list with one exact matching command and deletes
the others before implementation:

```bash
pytest Tests/RAG/test_ingestion_indexing.py -q --tb=short
pytest Tests/RAG/simplified/test_rag_service_basic.py -q --tb=short
pytest Tests/Library/test_library_local_rag_search_service.py -q --tb=short
pytest Tests/UI/test_product_maturity_gate16_library_search_rag.py -q --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit the mechanism fix**

Run Ruff on changed Python files, `git diff --check`, inspect the complete diff,
then stage only the exact amended test/production/plan files and commit the
regression and minimal implementation together:

```bash
git add <approved-plan> <approved-tests> <approved-production-files>
git commit -m "fix(rag): bound the first Library query (TASK-15810)"
```

Expected: no profiler-only, unrelated formatting, or speculative refactor hunks.

---

### Task 4: Verify event-loop responsiveness and non-overlapping supersession

**Files:**

- Modify: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
- Modify if the measured fix needs an offload/serialization owner: the exact approved production file from Task 2
- Modify if service-level serialization is the approved owner: `Tests/Library/test_library_local_rag_search_service.py`

- [ ] **Step 1: Confirm the prewritten deterministic heartbeat regression**

The Task 2 test at the actual changed retrieval seam uses two thread-safe events (`entered`,
`release`) plus an independent fail-safe controller thread. The controller
records its release timestamp and sets `release` after a bounded interval even
if Textual's loop is frozen. Start a real Library RAG Answer run with Pilot,
wait for `entered` without blocking the loop, then schedule a heartbeat and
navigation/cancel input. Assert both were processed before the controller's
release timestamp. Always set `release`, drain workers, and join the controller
thread in `finally`.

Expected before an event-loop offloading fix, when that is the measured fault: heartbeat/input assertion fails. Expected when the engine correction alone already yields control: the test passes without an extra production edit.

- [ ] **Step 2: Confirm the prewritten supersession/serialization regression**

The Task 2 test tracks active underlying CPU calls with a lock-protected counter. Start query A, supersede it with query B, and assert:

```text
max_active_cpu_calls == 1
query A never replaces query B's visible state
all underlying work reaches a terminal state after release
```

If the measured operation supports cooperative cancellation, additionally assert A observes its cancellation token. Otherwise assert B waits for serialized A rather than overlapping it.

- [ ] **Step 3: Run RED where applicable**

Run the two exact nodes with `-q --tb=short`. A test that passes because it gates only a fake coroutine on the event loop is invalid; the gate must sit at the real changed boundary.

- [ ] **Step 4: Implement the smallest missing offload/cancellation rule**

Task 4 may not add production code. If either test exposes a production gap not
already covered by Task 2's approved amendment, stop, amend this plan with its
exact RED/mutation/implementation and downstream commands, independently
re-review, then return to Task 2 before editing. Do not add a general scheduler
or process pool. Preserve Textual generation/stale-result guards.

- [ ] **Step 5: Run GREEN**

Run both exact amended nodes, the exact owner suite, Ruff, and diff check. Task 3
already commits these prewritten tests with the fix; no separate Task 4 commit
is needed unless the re-review loop explicitly adds another approved test.

Expected: heartbeat/input processed, max active CPU calls one, stale A absent, all work terminal.

---

### Task 5: Focused regression battery and cold live acceptance

**Files:**

- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md`
- Modify if user-visible initialization copy was required by profiling: `Docs/User_Guide/library/search-and-rag.md`
- Modify: TASK-15810 backlog file
- Modify only if a genuinely reusable incident was learned: the closest `backlog/docs/lessons-*.md`

- [ ] **Step 1: Run the automated battery**

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_local_rag_search_service.py Tests/Library/test_library_rag_service.py Tests/Library/test_library_rag_mode_resolution.py Tests/UI/test_product_maturity_gate16_library_search_rag.py -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <replaced-by-task2-exact-owner-suite> -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check <replaced-by-task2-exact-python-files>
git diff --check
```

Expected: all pass; no tests collected is failure. Record exact counts and duration.

- [ ] **Step 2: Run the unprofiled 30-second acceptance process**

Reuse the validated persisted scratch index but launch a new TUI process with no in-process runtime/query cache. Revalidate effective TOML and PID handles. Verify the exact fixture controls, activate Run, and record monotonic Run and first-Evidence timestamps.

Expected: the first visible Evidence row appears in less than 30 seconds, with
a final non-searching retrieval status. Record separately whether
`watchlists.md` appears and at which rank; its title is not the timer oracle.
Record any truthful named initialization phase; if none remains, keep ordinary
`searching · Notes…` copy unchanged.

- [ ] **Step 3: Run the separate cold responsiveness process**

Launch another new TUI process against the same scratch index. During its first pending query, send the navigation/cancel action pinned by Task 4 and record acknowledgement within one second. Verify no stale result later overwrites the current screen/query and no overlapping CPU retrieval appears.

Expected: responsive input, no stale overwrite, and terminal residual work.

- [ ] **Step 4: Prove isolation and write live evidence**

After each boot/run:

- parse effective TOML again;
- capture validated-PID `lsof` evidence (zero real-profile handles, scratch handles present);
- compare real config/data fingerprints to the before state; and
- terminate the app cleanly.

Using `apply_patch`, write `live-verification.md` with exact commands, fixture manifest hash, aggregate seed/index counts, timestamps/elapsed time, visible statuses/evidence title, responsiveness result, PID-handle summary, and before/after fingerprints. Do not include note bodies or secrets.

- [ ] **Step 5: Documentation and task hygiene**

Update the User Guide only if profiling left a named one-time initialization phase or changed user-visible behavior. Add concise Implementation Notes to TASK-15810: named frame/root cause, correction, files, automated/live evidence, ADR no, deviations, and lessons decision. Check all three AC boxes only when their evidence exists, then set the task to Done through Backlog CLI if it safely resolves the five-digit task; otherwise edit the exact task file with `apply_patch` and verify with `backlog task 15810 --plain`.

- [ ] **Step 6: Self-review and final commit**

Review `origin/dev...HEAD` for correctness, privacy, cancellation, event-loop behavior, unnecessary complexity, unrelated diffs, and task hygiene. Run the final changed-file battery again after documentation edits. Commit:

```bash
git add Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md "backlog/tasks/task-15810 - Library-RAG-Answers-first-query-on-a-fresh-profile-never-returns-CPU-bound-reproduced-twice.md"
git commit -m "docs(rag): verify bounded first Library query (TASK-15810)"
```

If the conditional User Guide or one exact lessons file changed, add that exact
path in a separate `git add` command before the commit; never use a wildcard.

Expected: clean worktree, complete task notes/AC/status, profiler and live evidence committed, no real-profile mutation.

---

## Plan-time self-review

- AC #1: exact fixture and under-30-second unprofiled run in Tasks 1 and 5.
- AC #2: Python-level named frame/callers committed before production edits in Task 1; Task 2 cannot proceed without it.
- AC #3: actual PTY UI run, Evidence row, isolation proof, and responsive input in Task 5.
- Concrete runtime: mandatory when the measured owner is at/below the engine boundary.
- Event loop: deterministic heartbeat plus separate live acknowledgement, not inferred from final completion.
- Cancellation: stale UI fencing and underlying CPU work are separately asserted; non-cooperative work must serialize.
- Privacy: artifacts allow only symbols, timing, aggregate counts, IDs, paths, and fingerprints.
- ADR: no new decision under current boundaries; explicit stop if the profile requires one.
- YAGNI: one measured owner, one retrieval path, no new dependency or scheduler.
- Known uncertainty is intentional and gated: the exact production edit cannot be written before profiling without violating AC #2. Task 2 requires a complete amendment and fresh independent review before code.
