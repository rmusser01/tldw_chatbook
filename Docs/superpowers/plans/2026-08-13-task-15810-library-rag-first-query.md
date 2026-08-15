# TASK-15810: Bounded First Library RAG Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and remove the measured CPU spin in the first fresh-profile Library RAG Answer query so the exact 36-note cold fixture renders its first Evidence row in under 30 seconds without freezing the TUI or permitting overlapping stale retrieval work.

**Architecture:** Preserve Library's existing request/state ownership and the process-wide `EnhancedRAGServiceV2` runtime. The completed profile names the Textual input-mirror spin: suppress the programmatic sibling event at `_patch_sibling_library_search_input()` with Textual's existing `prevent(Input.Changed)` mechanism. Supersession REDs prove Textual cancellation does not stop an already-admitted `asyncio.to_thread` retrieval and navigation replaces `LibraryScreen`; serialize those admitted calls with one narrow Library-only, app-lifetime async lock exposed by `TldwCli`, while each screen worker retains its canceled shielded retrieval task to settlement before releasing admission and re-raising. Normalize a restored transient `searching` status because no worker is restored with the fresh screen. This adds no broad root state and changes no retrieval runtime or service contract.

**Tech Stack:** Python 3.11+, asyncio, Textual 8.x workers/Pilot, `EnhancedRAGServiceV2`, Chroma/SQLite FTS5, py-spy or cProfile, pytest, Ruff.

---

## Authority and global constraints

- Design authority: `Docs/superpowers/specs/2026-08-13-task-15810-library-rag-first-query-design.md`.
- Task authority: `backlog/tasks/task-15810 - Library-RAG-Answers-first-query-on-a-fresh-profile-never-returns-CPU-bound-reproduced-twice.md`.
- Existing ownership: ADR-003 (`backlog/decisions/003-settings-library-rag-defaults.md`), ADR-005 (`backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`), and ADR-033 (`backlog/decisions/033-application-session-state-ownership.md`).
- Worktree: `.worktrees/task-15810-rag-first-query`; branch `codex/task-15810-rag-first-query`, rebased on `origin/dev` before this plan.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Before every profile/test command, assert `tldw_chatbook.__file__` resolves beneath this worktree by setting `PYTHONPATH` to the worktree root. A main-checkout import is a failed run.
- Baseline at plan time: 143 focused Library RAG service/Pilot tests passed.
- Every production edit is forbidden until Task 1 names the hot frame and an
  independently approved plan amendment names that edit's exact file/function,
  complete RED test, mutation, complete implementation snippet, owner suite,
  Ruff targets, and downstream commands. The profile rejects retrieval
  offloading as the spin fix; independently reproduced same-screen and
  cross-screen supersession REDs still require Library-only admission
  serialization across the app session.
- Never log or store query text, note bodies, retrieved content, answer prompts, credentials, or secrets in profiler/diagnostic artifacts. The approved fixed queries may appear in the fixture description, but profile output remains symbol/timing-only.
- No new dependency, retrieval implementation, runtime, process pool, cache
  layer, or background warm-up. The QA harness may clone the already-cached
  MiniLM artifact into its disposable scratch root solely to preserve read-only
  model isolation when macOS requires the live PTY to share an elevated
  loopback namespace. If profiling demands a new durable/runtime/service
  boundary, stop and revisit the ADR decision.
- Use @superpowers:systematic-debugging through the profile checkpoint, @superpowers:test-driven-development for every bug fix, @ponytail for the smallest measured correction, @textual-tui for event-loop/worker behavior, and @superpowers:verification-before-completion before any completion claim.

## File map

Files fixed before profiling:

- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/profile-report.md` — fixture provenance, Python hot frame/callers, bounded timing/count evidence, and root-cause statement.
- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/fixture-manifest.sha256` — sorted SHA-256 manifest for the 36 User Guide Markdown files.
- Create at closeout: `Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md` — unprofiled timing, responsive-input check, effective-config/lsof evidence, and real-profile fingerprints.
- Modify: this plan — Task 2's completed profile checkpoint records the exact
  UI owner, production/test files, complete regressions, and minimal snippet.
- Modify: the TASK-15810 backlog file — plan/notes/AC/status evidence.

Files selected by the named profile and required supersession RED:

- Modify: `tldw_chatbook/app.py` — expose one lazy, Library-only app-lifetime
  retrieval-admission lock so fresh `LibraryScreen` instances share the same
  admission owner without adding broad root application state.
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — suppress the sibling
  input's programmatic `Input.Changed` in
  `LibraryScreen._patch_sibling_library_search_input()`, use the app-lifetime
  Library admission accessor in `LibraryScreen._execute_library_rag_search()`,
  and normalize a restored transient `searching` status.
- Test: `Tests/UI/test_library_shell.py` — use the real `LibraryHarness`, both
  mounted inputs, real handlers, real Textual message pump, the existing
  Library RAG worker seam, and real `TldwCli` fresh-screen navigation.

No RAG engine, runtime construction, service, or storage file is selected. The
direct service-only run completed and the live profile measured the repeated UI
caller chain below. `LibraryScreen` continues to own requests, workers, stale
suppression, and results; `TldwCli` coordinates only the smallest app-lifetime
Library admission primitive required because navigation replaces that screen.

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
HF_HOME=<resolved-scratch-data>/default_user/models
HF_HUB_CACHE=<resolved-scratch-data>/default_user/models/embeddings
NO_PROXY=127.0.0.1,localhost
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
PYTHONPATH=<this-worktree>
PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
LANG=en_US.UTF-8
```

Do not add `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, any `*_API_KEY`, or any
provider-specific environment variable. The real cache is an un-escalated copy
source only. Parse `[paths].data_dir` without importing the application and
define the production-resolved offline embedding cache as
`<resolved-scratch-data>/default_user/models/embeddings`; assert that path is
beneath the scratch root. Preflight that
`models--sentence-transformers--all-MiniLM-L6-v2/snapshots` exists beneath the
real cache; resolve and assert the source model directory stays beneath the
real cache.
Enumerate every source symlink and assert its resolved target stays beneath that
model directory. Copy that one model directory (refs, blobs, snapshots) into
the production-resolved offline embedding cache without dereferencing symlinks;
resolve and assert the clone stays beneath the scratch root. This exact location
is required because production passes `get_model_cache_dir()` as the embedding
loader's explicit cache directory, which takes precedence over Hugging Face
environment defaults. Before changing permissions, enumerate every clone
symlink and assert its resolved target stays beneath the clone.
Compare source and clone manifests containing each relative path, entry type,
symlink target, and regular-file SHA-256. Only after the manifests match,
recursively remove write permission from the clone and verify a normal write
attempt fails. Explicitly assert the final cloned model directory is
`<embedding-cache>/models--sentence-transformers--all-MiniLM-L6-v2`, remains
beneath the embedding cache, and contains its `snapshots` directory. The
canonical clean environment above points only at this verified scratch clone.

If any seed attempt inserts notes but does not complete exact indexing, abandon
that scratch root and create a new one from the fixture before retrying. If a
listener was started during an earlier attempt, first terminate its captured,
validated PID and prove with `lsof` and a failed health connection that port
19090 is clear. Never repair or reuse a partially seeded/indexed profile.

Create the stub script during this step, but do not launch it until Step 4's
exact seed assertions succeed. Then first attempt the stub, health probe, and
TUI without escalation. If macOS sandboxing rejects the loopback bind or
separates the listener from the TUI's network namespace, record that failure
and request escalation for exactly the discard-only stub, body-free health
probe, and clean-environment TUI launch so those three processes share one
loopback namespace. No seed, DB maintenance, model-cache preparation, config
write, profiler report, or other command may be escalated. The escalated
commands retain `/usr/bin/env -i`, the scratch read-only cache, offline flags,
disabled model catalog, fixed `127.0.0.1:19090` endpoint, no
credentials/proxies, effective-TOML checks, running-PID `lsof`, and real-profile
fingerprints. `curl -sf http://127.0.0.1:19090/health` must succeed in the same
namespace before the TUI starts; no other network destination is configured.
Use one-shot exact escalation commands without a reusable prefix approval.
Capture the stub PID, assert it is the sole listener on strict
`127.0.0.1:19090`, and reuse that exact validated listener across the profiling
and acceptance runs. In final cleanup, terminate the stub PID cleanly, verify
with `lsof` that the port has no listener, and verify the loopback health
connection fails.

Parse the TOML with `tomllib` before launch and after every boot. Resolve
`[paths].data_dir`; assert it is beneath the scratch root and differs from the
real profile path. In the same clean environment, import
`get_model_cache_dir()` and assert its resolved return value is exactly
`<resolved-scratch-data>/default_user/models/embeddings`, stays beneath the
scratch root, and differs from the real cache. Import
`library_rag_answer_provider_gate()` and assert provider `custom-openai-api`, no
credential recovery, and an enabled Library Run gate. Hash the real config and
data inventory before launch.

The real config hash must remain identical. The real data fingerprint must also
remain identical unless an unrelated independently started process is
concurrently writing that profile. In that exceptional case, do not stop or
alter the external process. Record its PID, executable, cwd, and before/after `lsof`; the
task may proceed only when the TASK-15810 TUI PID has zero real-profile handles
and every differing real-profile path is present in the independently
identified external PID's handle inventory. Any unattributed difference is a
hard isolation failure.

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

Repeat this validated-PID handle check for every TUI process used as accepted
reproduction or profile evidence; a run without its own captured handle proof
is rejected even when another run used the same scratch fixture.

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

### Task 2: Convert the profile and supersession probes into exact regressions

**Files:**

- Modify: `Docs/superpowers/plans/2026-08-13-task-15810-library-rag-first-query.md`
- Test: `Tests/UI/test_library_shell.py`
- Later production edit: `tldw_chatbook/app.py`
- Later production edit: `tldw_chatbook/UI/Screens/library_screen.py`

- [ ] **Step 1: Record the profiled chain, mechanism, and ADR decision**

The watchdog-bounded live profile measured this production chain:

```text
LibraryScreen.handle_library_search_changed
  -> LibraryScreen._patch_sibling_library_search_input
  -> sibling.value = value
  -> sibling Input.Changed
  -> LibraryScreen.update_library_rag_query
  -> LibraryScreen._patch_sibling_library_search_input
  -> LibraryScreen._refresh_search_rag_panel_state_widgets
  -> Textual style refresh
  -> textual.css.match._check_selectors
```

In 15.003 seconds the profile recorded 525 rail-handler calls, 1,556 canvas
handler calls, 1,043 sibling-patch calls, 1,568 panel refreshes, and 692,228
`_check_selectors` calls. The direct service-only run completed, rejecting the
retrieval/runtime path as owner.

Faulty mechanism: stale queued A/B `Input.Changed` events alternate shared
state, and each unguarded programmatic `sibling.value = value` emits a fresh
sibling `Input.Changed`, replenishing the queue instead of letting it drain.

Separately, the required real-Run B-to-C-to-D RED discovered a
supersession-contract defect rather than another cause of the profiled spin:
Textual cancels B's worker coroutine, but B's already-admitted
`asyncio.to_thread` call continues while newer Runs can enter underlying
retrievals. D also cancels B a second time while superseding C, so the retained
cancellation loop must survive repeated cancellation until B really settles.

Post-implementation navigation probing exposed a second lifetime hole:
`TldwCli._create_navigation_screen()` constructs a fresh `LibraryScreen` on
every entry. Two real screen instances therefore had distinct per-screen
locks, and a D Run after navigating away entered alongside B (`active=2`,
`max_active=2`, `distinct_locks=True`). The outgoing state snapshot also
persisted `retrieval_status="searching"`, although the new screen restores no
worker capable of settling that transient status.

Lowest mirror-event owner:
`LibraryScreen._patch_sibling_library_search_input()` in
`tldw_chatbook/UI/Screens/library_screen.py`. Both handlers already call it,
and Textual's `prevent(Input.Changed)` is an established repository pattern.

Narrow admission-lifetime owner:
`TldwCli.library_rag_search_execution_lock()` in `tldw_chatbook/app.py`.
`TldwCli` already coordinates fresh-screen construction and owns other lazy
application-lifetime locks such as `_screen_navigation_lock()`. The new
accessor owns only admission for Library retrieval calls; query construction,
worker cancellation retention, stale suppression, outcome application, and
saved Library view state remain in `LibraryScreen`. It is not added to a broad
root `AppState` or to the RAG service.

ADR required: no

ADR path: N/A

Reason: this is direct conformance to existing decisions, not a new boundary.
ADR-003 keeps active Library query execution/source/results in Library;
ADR-005 keeps the existing local RAG runtime/service direction; ADR-033 says
navigation creates fresh screens, app-lifetime coordination belongs in a
narrow owner, and broad root application state is forbidden. `TldwCli` exposes
one Library-specific lock while `LibraryScreen` retains all domain behavior.
Normalizing a transient `searching` snapshot merely prevents a fresh screen
from claiming an operation it does not own. No storage, service, runtime,
security, dependency, or cross-module service contract changes.

- [ ] **Step 2: Add the complete gated-service helper and mechanism RED test**

Add this helper near the existing `_GatedLibraryRagSearchService` fakes in
`Tests/UI/test_library_shell.py`; the mandatory gated execution tests below use
it:

```python
class _SequencedGatedLibraryRagSearchService:
    """Gate B, C, and D independently while recording real thread overlap."""

    def __init__(self):
        self.calls: list[str] = []
        self.entered = {
            query: threading.Event() for query in ("B", "C", "D")
        }
        self.release = {
            query: threading.Event() for query in ("B", "C", "D")
        }
        self.finished = {
            query: threading.Event() for query in ("B", "C", "D")
        }
        self._lock = threading.Lock()
        self._active_calls = 0
        self._max_active_calls = 0

    def snapshot(self) -> tuple[tuple[str, ...], int, int]:
        with self._lock:
            return tuple(self.calls), self._active_calls, self._max_active_calls

    def _block_until_release(self, query: str) -> None:
        with self._lock:
            self._active_calls += 1
            self._max_active_calls = max(
                self._max_active_calls, self._active_calls
            )
        self.entered[query].set()
        try:
            if not self.release[query].wait(_GATED_RELEASE_TIMEOUT_SECONDS):
                raise AssertionError(f"Timed out waiting to release query {query}.")
        finally:
            with self._lock:
                self._active_calls -= 1
            self.finished[query].set()

    async def search(self, query, scope, mode, **kwargs):
        del scope, mode, kwargs
        with self._lock:
            self.calls.append(query)
        await asyncio.to_thread(self._block_until_release, query)
        return {
            "results": [
                {
                    "document_title": query,
                    "snippet": f"Evidence for {query}",
                    "source_id": query,
                }
            ]
        }
```

Add the exact mechanism test. The wrapper calls the real faulty helper for the
first eight mirror attempts, enough to reproduce two full A/B replenishment
cycles, then stops assigning so the unfixed test cannot hang. The expected
faulty sequence is 10 helper calls and four panel refreshes; the fixed helper
produces only the two explicit rail-to-canvas calls and zero refreshes.

```python
@pytest.mark.asyncio
async def test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic(
    monkeypatch,
):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        rail = screen.query_one("#library-search-input", Input)
        canvas = screen.query_one("#library-rag-query-input", Input)
        with rail.prevent(Input.Changed):
            rail.value = "B"
        with canvas.prevent(Input.Changed):
            canvas.value = "B"
        screen._library_rag_query = "B"
        await pilot.pause()

        real_patch = screen._patch_sibling_library_search_input
        mirror_calls: list[tuple[str, str]] = []

        def bounded_patch(selector: str, value: str) -> None:
            mirror_calls.append((selector, value))
            if len(mirror_calls) <= 8:
                real_patch(selector, value)

        monkeypatch.setattr(
            screen, "_patch_sibling_library_search_input", bounded_patch
        )
        real_refresh = screen._refresh_search_rag_panel_state_widgets
        refresh_calls = 0

        async def recording_refresh(*args, **kwargs):
            nonlocal refresh_calls
            refresh_calls += 1
            await real_refresh(*args, **kwargs)

        monkeypatch.setattr(
            screen, "_refresh_search_rag_panel_state_widgets", recording_refresh
        )

        screen.handle_library_search_changed(Input.Changed(rail, "A"))
        screen.handle_library_search_changed(Input.Changed(rail, "B"))
        for _ in range(12):
            await pilot.pause()

        assert mirror_calls == [
            ("#library-rag-query-input", "A"),
            ("#library-rag-query-input", "B"),
        ]
        assert refresh_calls == 0
        assert screen._library_rag_query == "B"
        assert rail.value == "B"
        assert canvas.value == "B"
```

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic -q --tb=short
```

Expected before the fix: FAIL on `mirror_calls`; the bounded wrapper records
10 calls rather than the two explicit calls. It must not fail through timeout,
fixture setup, selector lookup, or missing dependencies.

- [ ] **Step 3: Add the mandatory deterministic heartbeat regression**

This enters through the real Search canvas and `_execute_library_rag_search`
worker. It holds the service pending, schedules a Textual heartbeat, presses a
real navigation control, and asserts both UI events occur while the release
gate is still closed and the one service call remains active. The oracle is
event order/state, not elapsed time; the bounded wait helper only prevents a
broken test from hanging.

```python
@pytest.mark.asyncio
async def test_library_shell_gated_search_keeps_heartbeat_and_navigation_live():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    service = _SequencedGatedLibraryRagSearchService()
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        screen.query_one("#library-rag-query-input", Input).value = "B"
        await _wait_for_library_rag_query_ready(screen, pilot, "B")
        screen.query_one("#library-rag-run-query", Button).press()

        try:
            await _wait_for_condition(
                pilot,
                service.entered["B"].is_set,
                message="The gated Library search never started.",
            )
            heartbeat = asyncio.Event()
            heartbeat_observations: list[
                tuple[bool, tuple[tuple[str, ...], int, int]]
            ] = []

            def record_heartbeat() -> None:
                heartbeat_observations.append(
                    (service.release["B"].is_set(), service.snapshot())
                )
                heartbeat.set()

            screen.call_later(record_heartbeat)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    heartbeat.is_set()
                    and screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
                ),
                message="Heartbeat/navigation did not run while retrieval was gated.",
            )

            assert heartbeat_observations == [(False, (("B",), 1, 1))]
            assert not service.finished["B"].is_set()
        finally:
            service.release["B"].set()
            if service.entered["B"].is_set():
                await _wait_for_condition(
                    pilot,
                    service.finished["B"].is_set,
                    message=(
                        "The admitted Library search did not terminate after release."
                    ),
                )
            await _wait_for_condition(
                pilot,
                lambda: not any(
                    worker.node is screen
                    and worker.group == "library_rag_search"
                    for worker in host.workers
                ),
                message="The gated Library search worker did not terminate.",
            )
            await pilot.pause()
```

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live -q --tb=short
```

Expected before and after the input-mirror fix: PASS. This is a mandatory
regression protecting the already-correct worker/event-loop boundary; the
profile does not authorize changing that boundary. Opening B's gate happens
unconditionally, but teardown waits for `finished["B"]` only if B was actually
admitted; a failure before admission therefore cannot replace the original
assertion with a teardown timeout. The worker-group poll still catches a
scheduled-but-not-yet-admitted worker, which sees the already-open gate.

- [ ] **Step 4: Add the mandatory repeated-supersession/no-overlap RED**

This starts gated B through the real Run button, submits C while B drains, then
submits D while C waits. D supersedes C and cancels B's retained worker a second
time. Before B is released, C and D must remain outside the underlying service
while heartbeat and Browse ▸ Media navigation stay live. After B settles, C
must never enter and D alone may run. The spy on
`_apply_library_rag_search_outcome()` separately proves canceled B and C never
apply, then the test navigates back to Search and asserts mounted D evidence.

```python
@pytest.mark.asyncio
async def test_library_shell_repeated_supersession_serializes_underlying_retrieval(
    monkeypatch,
):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    service = _SequencedGatedLibraryRagSearchService()
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        applied_queries: list[str] = []
        real_apply = screen._apply_library_rag_search_outcome

        async def recording_apply(request, outcome):
            applied_queries.append(request.query)
            await real_apply(request, outcome)

        monkeypatch.setattr(
            screen, "_apply_library_rag_search_outcome", recording_apply
        )
        canvas = screen.query_one("#library-rag-query-input", Input)
        canvas.value = "B"
        await _wait_for_library_rag_query_ready(screen, pilot, "B")
        screen.query_one("#library-rag-run-query", Button).press()

        try:
            await _wait_for_condition(
                pilot,
                service.entered["B"].is_set,
                message="Query B never entered the gated retrieval thread.",
            )
            canvas.value = "C"
            await _wait_for_library_rag_query_ready(screen, pilot, "C")
            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    service.entered["C"].is_set()
                    or sum(
                        worker.node is screen
                        and worker.group == "library_rag_search"
                        for worker in host.workers
                    )
                    >= 2
                ),
                message=(
                    "Query C neither entered nor registered while B drained."
                ),
            )
            workers_before_d = {
                id(worker)
                for worker in host.workers
                if worker.node is screen
                and worker.group == "library_rag_search"
            }

            canvas.value = "D"
            await _wait_for_library_rag_query_ready(screen, pilot, "D")
            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    service.entered["D"].is_set()
                    or any(
                        worker.node is screen
                        and worker.group == "library_rag_search"
                        and id(worker) not in workers_before_d
                        for worker in host.workers
                    )
                ),
                message="Query D never registered after superseding C.",
            )

            heartbeat = asyncio.Event()
            screen.call_later(heartbeat.set)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    heartbeat.is_set()
                    and screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
                ),
                message="Heartbeat/navigation stalled while query D waited for B.",
            )
            for _ in range(8):
                await pilot.pause()

            assert service.snapshot() == (("B",), 1, 1)
            assert not service.entered["C"].is_set()
            assert not service.entered["D"].is_set()
            assert not service.release["B"].is_set()
            assert applied_queries == []

            service.release["B"].set()
            await _wait_for_condition(
                pilot,
                service.entered["D"].is_set,
                message="Query D did not enter after query B settled.",
            )
            assert service.finished["B"].is_set()
            assert not service.entered["C"].is_set()
            assert not service.finished["C"].is_set()
            assert service.snapshot() == (("B", "D"), 1, 1)
            assert applied_queries == []

            service.release["D"].set()
            await _wait_for_condition(
                pilot,
                service.finished["D"].is_set,
                message="Query D did not finish after release.",
            )
            await _wait_for_condition(
                pilot,
                lambda: not any(
                    worker.node is screen
                    and worker.group == "library_rag_search"
                    for worker in host.workers
                ),
                message="Library RAG workers did not terminate after D settled.",
            )
            screen.query_one("#library-row-browse-search", Button).press()
            await _wait_for_selector(screen, pilot, "#library-rag-result-0")
            await pilot.pause()
        finally:
            for query in ("B", "C", "D"):
                service.release[query].set()
            await _wait_for_condition(
                pilot,
                lambda: service.snapshot()[1] == 0,
                message="A gated retrieval thread remained active at teardown.",
            )
            await _wait_for_condition(
                pilot,
                lambda: not any(
                    worker.node is screen
                    and worker.group == "library_rag_search"
                    for worker in host.workers
                ),
                message="A Library RAG worker remained active at teardown.",
            )

        assert service.snapshot() == (("B", "D"), 0, 1)
        assert not service.entered["C"].is_set()
        assert not service.finished["C"].is_set()
        assert applied_queries == ["D"]
        assert [row.title for row in screen._library_rag_results] == ["D"]
        assert screen._library_rag_query == "D"
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH
        assert screen.query_one("#library-rag-query-input", Input).value == "D"
        assert screen.query("#library-rag-result-0")
```

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval -q --tb=short
```

Expected before serialization: FAIL before B is released because C and D enter
underlying calls instead of waiting; calls differ from `("B",)` and max active
exceeds one. The `finally` block opens all three gates, polls real worker
registration without canceling a worker, and does not wait on the unrelated
app-wide worker manager, so the RED is bounded and cannot leak executor
threads.

The mounted-harness prototype of the approved retained loop observed
`(("B",), active=1, max=1)` before release, then `("B", "D")` with
active/max still one, and terminal active zero with only D applied and mounted.

- [ ] **Step 5: Add the real-navigation app-lifetime admission RED**

This test uses the real `TldwCli.handle_screen_navigation()` path, not a
two-screen harness shortcut. B enters on the first `LibraryScreen`; navigation
to Home saves that screen and cancels its worker while its admitted thread
drains. Re-entering Library creates a different screen, where D must wait on
the same narrow app-lifetime admission lock. Add a local import for
`NavigateToScreen` in the test body so the existing module import surface is
not broadened unnecessarily.

```python
@pytest.mark.asyncio
async def test_library_shell_fresh_screen_reentry_serializes_draining_retrieval(
    monkeypatch,
):
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _SequencedGatedLibraryRagSearchService()
    app.library_rag_search_service = service
    applied_queries: list[tuple[int, str]] = []
    real_apply = LibraryScreen._apply_library_rag_search_outcome

    async def recording_apply(screen, request, outcome):
        applied_queries.append((id(screen), request.query))
        await real_apply(screen, request, outcome)

    monkeypatch.setattr(
        LibraryScreen, "_apply_library_rag_search_outcome", recording_apply
    )

    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        first = None
        second = None
        try:
            await _wait_for_condition(
                pilot,
                lambda: getattr(app, "_initial_screen_pushed", False),
                message="The app never finished pushing its initial screen.",
            )
            await app.handle_screen_navigation(NavigateToScreen("library"))
            first = app.screen
            assert isinstance(first, LibraryScreen)
            await _wait_for_library_shell(first, pilot)
            first.query_one("#library-row-browse-search", Button).press()
            await _wait_for_selector(first, pilot, "#library-rag-query-input")
            first.query_one("#library-rag-query-input", Input).value = "B"
            await _wait_for_library_rag_query_ready(first, pilot, "B")
            first.query_one("#library-rag-run-query", Button).press()
            await _wait_for_condition(
                pilot,
                service.entered["B"].is_set,
                message="Query B never entered on the first Library screen.",
            )

            await app.handle_screen_navigation(NavigateToScreen("home"))
            assert app.screen is not first
            await app.handle_screen_navigation(NavigateToScreen("library"))
            second = app.screen
            assert isinstance(second, LibraryScreen)
            assert second is not first
            await _wait_for_library_shell(second, pilot)
            await _wait_for_selector(second, pilot, "#library-rag-query-input")

            canvas = second.query_one("#library-rag-query-input", Input)
            canvas.value = "D"
            await _wait_for_library_rag_query_ready(second, pilot, "D")
            second.query_one("#library-rag-run-query", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    service.entered["D"].is_set()
                    or any(
                        worker.node is second
                        and worker.group == "library_rag_search"
                        for worker in app.workers
                    )
                ),
                message="Query D never entered or registered after re-entry.",
            )
            heartbeat = asyncio.Event()
            second.call_later(heartbeat.set)
            await _wait_for_condition(
                pilot,
                heartbeat.is_set,
                message="The fresh Library screen heartbeat stalled behind B.",
            )
            for _ in range(8):
                await pilot.pause()

            assert service.snapshot() == (("B",), 1, 1)
            assert not service.entered["D"].is_set()
            assert applied_queries == []

            service.release["B"].set()
            await _wait_for_condition(
                pilot,
                service.entered["D"].is_set,
                message="Query D did not enter after query B settled.",
            )
            assert service.finished["B"].is_set()
            assert service.snapshot() == (("B", "D"), 1, 1)
            assert applied_queries == []

            service.release["D"].set()
            await _wait_for_condition(
                pilot,
                service.finished["D"].is_set,
                message="Query D did not finish after release.",
            )
            await _wait_for_condition(
                pilot,
                lambda: not any(
                    (worker.node is first or worker.node is second)
                    and worker.group == "library_rag_search"
                    for worker in app.workers
                ),
                message="A cross-screen Library RAG worker did not terminate.",
            )
            await _wait_for_selector(second, pilot, "#library-rag-result-0")
        finally:
            for query in ("B", "C", "D"):
                service.release[query].set()
            for query in ("B", "C", "D"):
                if service.entered[query].is_set():
                    await _wait_for_condition(
                        pilot,
                        service.finished[query].is_set,
                        message=f"Admitted query {query} did not terminate.",
                    )
            await _wait_for_condition(
                pilot,
                lambda: service.snapshot()[1] == 0,
                message="A cross-screen retrieval thread remained active.",
            )
            await _wait_for_condition(
                pilot,
                lambda: not any(
                    (worker.node is first or worker.node is second)
                    and worker.group == "library_rag_search"
                    for worker in app.workers
                ),
                message="A cross-screen Library RAG worker remained registered.",
            )

        assert second is not None
        assert service.snapshot() == (("B", "D"), 0, 1)
        assert applied_queries == [(id(second), "D")]
        assert second._library_rag_query == "D"
        assert [row.title for row in second._library_rag_results] == ["D"]
        assert second._library_rag_retrieval_status == "ready"
        assert second.query_one("#library-rag-query-input", Input).value == "D"
        assert second.query("#library-rag-result-0")
```

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval -q --tb=short
```

Expected before the app-lifetime fix: FAIL before B release because the fresh
screen's distinct lock admits D; calls are `("B", "D")` and max active is two.
The existing same-screen B-to-C-to-D test remains mandatory and still proves
repeated cancellation on one screen. This new test proves the separate
application-lifetime invariant across real screen replacement.

The real-navigation prototype failed the current per-screen implementation at
the pre-release snapshot assertion and completed teardown. Injecting one shared
lock across both real screen instances made the identical body pass: heartbeat
ran while D waited, calls settled in B/D order, max active stayed one, and only
the fresh screen applied D.

- [ ] **Step 6: Add the transient restored-status RED**

Persisted evidence and settled statuses remain valid destination state, but
`searching` describes a live worker and cannot survive into a fresh screen that
does not restore that worker. Drive the real save/restore pair and mount the
fresh screen to pin both state and control behavior:

```python
@pytest.mark.asyncio
async def test_library_shell_restore_normalizes_transient_rag_searching_state():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    original = LibraryScreen(app)
    original._library_selected_row_id = LIBRARY_ROW_BROWSE_SEARCH
    original._library_rag_query = "B"
    original._library_rag_retrieval_status = "searching"
    saved = original.save_state()
    assert saved["library_rag_retrieval_status"] == "searching"

    restored = LibraryScreen(app)
    restored.restore_state(saved)
    assert restored._library_rag_retrieval_status == ""
    host = LibraryHarness(app, screen=restored)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        await _wait_for_library_rag_query_ready(screen, pilot, "B")

        assert screen._library_rag_retrieval_status == ""
        assert screen.query_one("#library-rag-query-input", Input).value == "B"
        assert screen.query_one("#library-rag-run-query", Button).disabled is False
        assert not screen.query("#library-rag-searching-line")
```

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
```

Expected before normalization: FAIL because the restored screen still reports
`searching` and mounts its searching line despite owning no corresponding
worker. Query text, settled evidence, and non-transient statuses are unchanged.
The current implementation fails immediately on the direct restored-status
assertion (not on a readiness timeout); emulating the exact normalization makes
the complete mounted Run-enabled/status-line body pass.

- [ ] **Step 7: Independently re-review the amended plan**

Dispatch independent spec and quality plan-document reviewers with this plan,
the design spec, task, profile report, ADR-003, ADR-005, and ADR-033. Do not
proceed until both confirm that the current two REDs fail for the named
lifetime/state mechanisms, the prior three regressions remain coherent, and
the app-lifetime Library-only lock is the smallest compliant owner. Limit the
loop to three review passes.

- [ ] **Step 8: Write every regression before production code**

The mirror, heartbeat, and same-screen supersession regressions already exist
and pass on the current post-implementation branch. Write the new fresh-screen-
reentry and restored-status regressions before any further production edit;
both must fail for their newly named reasons. Keep all five nodes as the final
contract. Prefer deterministic counts/state/termination assertions over a
wall-clock timeout. The profile selected UI/app orchestration, so no concrete-
engine fixture is required or permitted for these corrections.

- [ ] **Step 9: Run RED and inspect the reason**

Run the five exact node commands above, then the combined command:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
```

Expected on the current post-implementation branch before this amendment's
production edit: mirror PASS, heartbeat PASS, same-screen supersession PASS,
fresh-screen re-entry FAIL because D overlaps B, and restore FAIL immediately
because dead `searching` survives. Historical RED/mutation evidence remains:
removing `prevent` fails the mirror count and removing same-screen retained
serialization fails the B-to-C-to-D max-active contract. No failure may be a
timeout, fixture error, selector error, dependency failure, or teardown masking
the original assertion.

Post-implementation amendment stop rule: make no further production edit until
the fresh-screen node fails specifically with B/D overlap on the current
per-screen lock, the restore node fails immediately on retained `searching`,
both complete teardown, and independent spec/quality review approves the
app-lifetime Library-only owner. If real navigation requires broader app state,
service locking, a new runtime contract, or a different persistence policy,
stop and revisit the ADR decision instead of expanding this patch.

- [ ] **Step 10: Pin the complete minimal production edit and inverse mutation**

In `LibraryScreen._patch_sibling_library_search_input()`, replace only the
assignment block with:

```python
        if sibling.value != value:
            with sibling.prevent(Input.Changed):
                sibling.value = value
```

Update the helper docstring's final sentence to say that the programmatic
assignment suppresses the sibling Changed event, rather than claiming the
sibling handler makes it harmless.

In `TldwCli`, immediately before `_screen_navigation_lock()`, add the narrow
lazy application-lifetime accessor, matching that existing lock's lifecycle:

```python
    def library_rag_search_execution_lock(self) -> asyncio.Lock:
        """Return the app-lifetime admission lock for Library retrieval calls.

        Returns:
            The shared Library-only admission lock for this app session.
        """
        lock = getattr(self, "_library_rag_search_execution_lock_instance", None)
        if lock is None:
            lock = asyncio.Lock()
            self._library_rag_search_execution_lock_instance = lock
        return lock
```

Do not add this lock to a generic `AppState`, runtime-policy object, or RAG
service. Remove the per-screen `_library_rag_search_execution_lock` constructor
line, then replace `_execute_library_rag_search()`'s body with:

```python
        """Serialize Library admission across screens and retain cancellation.

        The app-lifetime Library-only lock prevents fresh screens from
        overlapping admitted calls. Canceled workers drain admitted retrievals
        under that lock, then re-raise before stale outcomes can apply.
        """
        async with self.app_instance.library_rag_search_execution_lock():
            retrieval_task = asyncio.create_task(
                run_library_rag_search(self.app_instance, request)
            )
            cancellation: asyncio.CancelledError | None = None
            while not retrieval_task.done():
                try:
                    await asyncio.shield(retrieval_task)
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
            outcome = retrieval_task.result()
            if cancellation is not None:
                raise cancellation
            await self._apply_library_rag_search_outcome(request, outcome)
```

In `LibraryScreen.restore_state()`, replace the direct retrieval-status restore
with the exact transient normalization:

```python
        restored_retrieval_status = str(
            state.get("library_rag_retrieval_status") or ""
        )
        self._library_rag_retrieval_status = (
            "" if restored_retrieval_status == "searching"
            else restored_retrieval_status
        )
```

Only `searching` is normalized. Settled `ready`, `empty`, recovery/error state,
results, query, diagnostics, and searched-query state retain the existing
snapshot contract.

Update `save_state()`'s existing RAG snapshot paragraph so it no longer claims
every retrieval status is restored verbatim. Replace that paragraph with:

```python
        The RAG results tuple and settled retrieval/recovery state are safe to
        carry because their rows are frozen dataclasses. A transient
        ``searching`` value may be snapshotted while admitted work drains, but
        ``restore_state`` normalizes it because a fresh screen owns no matching
        worker.
```

The lock acquisition remains an ordinary await: if C is superseded by D while
still waiting for B, C cancels without starting underlying work. Once a request
has acquired the lock and created its retrieval task, cancellation is retained
but not allowed to release admission early; repeated cancellation is absorbed
until the task settles, then re-raised before stale outcome application. This
is the same retained-shield pattern already used by
`_await_library_prompt_durable_call()` in this screen.

Mirror inverse mutation: remove the `with sibling.prevent(Input.Changed):`
context and restore the direct `sibling.value = value`. Run the mechanism node
and prove it fails with the bounded 10-call chain.

Application-lifetime inverse mutation: restore the removed per-screen
`asyncio.Lock()` and make `_execute_library_rag_search()` acquire it instead of
the app accessor. Run the fresh-screen-reentry node. It must fail before B is
released with calls `("B", "D")`, active/max two, while the same-screen
B-to-C-to-D node still passes. This proves the lifetime change, not merely the
presence of a lock.

Transient-state inverse mutation: restore the raw `str(...)` assignment in
`restore_state()`. Run the restored-status node and prove it fails on the dead
`searching` status/searching line while the query remains Run-enabled.

Repeated-cancellation inverse mutation: replace the `while` loop only with this
one-shot catch, leaving the lock and owned retrieval task intact:

```python
            try:
                await asyncio.shield(retrieval_task)
            except asyncio.CancelledError as error:
                cancellation = error
                await asyncio.shield(retrieval_task)
```

Keep `outcome = retrieval_task.result()` immediately after this mutated block.
Run the repeated-supersession node. D's second cancellation escapes the
one-shot catch, releases B's admission lock while B's underlying thread still
runs, and admits D: the prototype observed calls `("B", "D")` with
active/max equal to two before B release. Restore the retained `while` loop,
app-lifetime accessor, status normalization, and mirror snippet only through
`apply_patch`; rerun all five nodes and inspect the diff.

Expected: each mutation fails only its named mechanism; restoring all snippets
makes all five nodes pass.

---

### Task 3: Apply the approved narrow UI/app-lifetime corrections

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Implement only the amended code snippets**

Apply the `prevent(Input.Changed)` mirror fix, the lazy app-lifetime
Library-only lock accessor, removal of the per-screen lock, the concise
worker-orchestration docstring, retained shield loop, and transient restored-
status normalization plus snapshot-docstring correction exactly as amended.
Keep request/outcome contracts,
active-profile routing, ranking, source filtering, citations, and error shapes
unchanged. Do not add a broad app-state field or second retrieval path.

- [ ] **Step 2: Run the RED node to GREEN**

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
```

Expected: PASS with the measured operation bounded/terminated as asserted.

- [ ] **Step 3: Run the closest owner suite**

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_screen_navigation.py -q --tb=short
```

Expected: the complete owning Library shell and real-navigation suites pass; no
tests collected is failure.

- [ ] **Step 4: Commit the mechanism fix**

Run Ruff on changed Python files, `git diff --check`, inspect the complete diff,
then stage only the exact amended test/production/plan files and commit the
regression and minimal implementation together:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py
git diff --check
git add Docs/superpowers/plans/2026-08-13-task-15810-library-rag-first-query.md Tests/UI/test_library_shell.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py
git commit -m "fix(rag): bound the first Library query (TASK-15810)"
```

Expected: no profiler-only, unrelated formatting, or speculative refactor hunks.

---

### Task 4: Verify responsiveness and same-/cross-screen supersession

**Files:**

- Test: `Tests/UI/test_library_shell.py`
- Verify only: `tldw_chatbook/app.py`
- Verify only: `tldw_chatbook/UI/Screens/library_screen.py`

- [ ] **Step 1: Confirm the prewritten deterministic heartbeat regression**

Confirm
`test_library_shell_gated_search_keeps_heartbeat_and_navigation_live` uses the
real Library Run control and worker, holds its gated service active, then
records a scheduled Textual heartbeat and Media-row navigation before opening
the release event. The `finally` block always opens B's gate, waits for
`finished["B"]` only if admission actually occurred, then polls `host.workers`
for terminal RAG-group registration. It deliberately does not call bare
`screen.workers.wait_for_complete()`, which targets the entire app worker
manager. A failure before admission therefore cannot hang in cleanup or mask
the first assertion through unrelated work.

Expected: the observation is exactly `(release_open=False, active=1,
max_active=1, calls=("B",))`, the Media row is selected, and the service has
not finished. This regression is expected to pass before and after the mirror
fix because the profile rejected an event-loop offloading defect.

- [ ] **Step 2: Confirm the prewritten supersession/serialization regression**

Confirm
`test_library_shell_repeated_supersession_serializes_underlying_retrieval` tracks active
underlying service threads with a lock-protected counter. Start B, Run C while
B remains gated, then Run D while C waits, and assert:

```text
calls == ("B",)
active_calls == max_active_calls == 1 before release
C and D have not entered; heartbeat and Browse ▸ Media navigation complete
after B release: C never enters; calls == ("B", "D"), active == max_active == 1
after D release: active == 0, only D applied
after navigating back to Search: mounted input/result and shared query == D
```

The test enters three times through the real Run seam. Its `finally` block
opens all three gates first, waits for active thread count zero, polls the real
worker-group registration to empty, and then makes final mounted-state
assertions. It never performs a bare all-manager worker wait.

- [ ] **Step 3: Confirm cross-screen admission and restored-state regressions**

Confirm
`test_library_shell_fresh_screen_reentry_serializes_draining_retrieval` uses
real app navigation and two distinct `LibraryScreen` instances. B remains an
active admitted call after the first screen unmounts; D registers on the fresh
screen without entering until B settles. Assert app heartbeat, calls/order,
active/max one, terminal worker registration, and only fresh-screen D outcome.

Confirm
`test_library_shell_restore_normalizes_transient_rag_searching_state` drives
the real `save_state()`/`restore_state()` pair. The fresh mounted screen keeps
the query and a usable Run control but neither the transient `searching` value
nor its in-flight status line.

- [ ] **Step 4: Run RED where applicable**

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
```

Expected against the per-screen-lock implementation: heartbeat and same-screen
supersession PASS; cross-screen admission FAIL on calls/max-active before B
release; restore FAIL on dead `searching`. Expected after the amended fixes:
four passes. All execution tests enter through real mounted Library controls
and the real `_execute_library_rag_search` worker; only the service result is
gated.

- [ ] **Step 5: Stop on any unrelated worker defect**

Task 4 may not add production code: Task 3 already applies the independently
reviewed narrow app-lifetime admission owner and screen-owned worker retention.
If any test exposes a different gap, stop and amend/re-review rather than adding
another scheduler, service lock, process, or retrieval implementation.

- [ ] **Step 6: Run GREEN**

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_screen_navigation.py -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py
git diff --check
```

Task 3 already commits these prewritten tests with the fix; Task 4 makes no
separate commit.

Expected: heartbeat/navigation processes while same-screen and fresh-screen D
wait, C never enters, max active calls remains one across both lifetimes, stale
B/C never apply, mounted D evidence appears after B settles, restored dead
`searching` is normalized, all work is terminal, and owner/navigation suites
and Ruff are green.

---

### Task 5: Focused regression battery and cold live acceptance

**Files:**

- Create: `Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md`
- Modify: TASK-15810 backlog file
- Verify unchanged: `Docs/User_Guide/library/search-and-rag.md` — the profile
  found no legitimate initialization phase or user-visible contract change.
- Verify unchanged: `backlog/docs/lessons-testing-evidence.md` — add an
  incident only if implementation/live verification reveals reusable evidence
  beyond the already-recorded bounded-message-pump principle; do not invent one.

- [ ] **Step 1: Run the automated battery**

Run:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic Tests/UI/test_library_shell.py::test_library_shell_gated_search_keeps_heartbeat_and_navigation_live Tests/UI/test_library_shell.py::test_library_shell_repeated_supersession_serializes_underlying_retrieval Tests/UI/test_library_shell.py::test_library_shell_fresh_screen_reentry_serializes_draining_retrieval Tests/UI/test_library_shell.py::test_library_shell_restore_normalizes_transient_rag_searching_state -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_local_rag_search_service.py Tests/Library/test_library_rag_service.py Tests/Library/test_library_rag_mode_resolution.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_library_shell.py Tests/UI/test_screen_navigation.py -q --tb=short
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py
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

Launch another new TUI process against the same scratch index. Label the
already-approved query `how do I schedule a watchlist brief` as B and the
distinct fixture query `how do I schedule recurring watchlist briefs?` as D.
Run B and, while its visible status is still `searching`, navigate to Home and
immediately re-enter Library through the real app navigation. Confirm this is
a fresh destination paint: the retained B query is present, but a dead
`searching` line is absent and Run is enabled. Replace the input with D and
activate the real Run control. While D is waiting/running, activate Browse ▸
Media, the exact in-screen navigation action pinned by Task 4, and record
acknowledgement within one second. After work settles, return to Browse ▸
Search and verify the mounted input, terminal status, and Evidence belong to D;
observe one additional message-pump turn and verify no B outcome overwrites it.
Record only the B/D labels and action/status timestamps in live evidence, not
the query bodies again.

The live process proves real fresh-screen navigation, usable restored controls,
real-control responsiveness, and no stale overwrite; it must not infer
underlying call overlap from CPU shape. The lock-protected same-screen
B-to-C-to-D test and real-navigation B-to-D test in Task 4 are the authoritative
serialization evidence (`max_active_calls == 1`).

Expected: Home and fresh Library re-entry remain responsive, restored Library
does not claim a dead in-flight state, Media acknowledges within one second
while D is pending, terminal mounted Search state belongs only to D, no B
overwrite appears, and all residual work is terminal. Task 4 supplies the
deterministic same-screen repeated-cancellation and cross-screen no-overlap
proofs.

- [ ] **Step 4: Prove isolation and write live evidence**

After each boot/run:

- parse effective TOML again;
- capture validated-PID `lsof` evidence (zero real-profile handles, scratch handles present);
- compare the real config/data fingerprints to the before state, applying the
  narrowly defined external-writer attribution rule from Task 1 when and only
  when every changed path is accounted for; and
- terminate the app cleanly.

After the final run, terminate the previously validated loopback-stub PID
cleanly. Confirm with `lsof` and a failing health connection that nothing is
listening on `127.0.0.1:19090`; do not substitute or terminate any unvalidated
PID.

Using `apply_patch`, write `live-verification.md` with exact commands, fixture manifest hash, aggregate seed/index counts, timestamps/elapsed time, visible statuses/evidence title, responsiveness result, PID-handle summary, and before/after fingerprints. Do not include note bodies or secrets.

- [ ] **Step 5: Documentation and task hygiene**

Do not update the User Guide: profiling found neither a legitimate one-time
initialization phase nor a user-visible contract change. Add concise
Implementation Notes to TASK-15810: named frame/root cause, correction, files,
automated/live evidence, ADR no, deviations, and lessons decision. Check all
three AC boxes only when their evidence exists, then set the task to Done
through Backlog CLI if it safely resolves the five-digit task; otherwise edit
the exact task file with `apply_patch` and verify with
`backlog task 15810 --plain`.

- [ ] **Step 6: Self-review and final commit**

Review `origin/dev...HEAD` for correctness, privacy, cancellation, event-loop
behavior, unnecessary complexity, unrelated diffs, and task hygiene. Run the
final changed-file battery again after documentation edits, then run the full
suite before any PR-readiness or completion claim:

```bash
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=short
```

If the full suite is red, stop, report the exact failing node(s) and output,
and do not commit closeout documentation or claim PR readiness. If it is green,
commit:

```bash
git add Docs/superpowers/qa/task-15810-library-rag-first-query/live-verification.md "backlog/tasks/task-15810 - Library-RAG-Answers-first-query-on-a-fresh-profile-never-returns-CPU-bound-reproduced-twice.md"
git commit -m "docs(rag): verify bounded first Library query (TASK-15810)"
```

The profile selected no User Guide change. If implementation nevertheless
produced a genuinely reusable testing-evidence incident, amend this plan before
editing the exact lessons file, then add that exact path separately; never use
a wildcard.

Expected: clean worktree, complete task notes/AC/status, profiler and live evidence committed, no real-profile mutation.

---

## Plan-time self-review

- AC #1: exact fixture and under-30-second unprofiled run in Tasks 1 and 5.
- AC #2: Python-level named frame/callers committed before production edits in Task 1; Task 2 cannot proceed without it.
- AC #3: actual PTY UI run, Evidence row, isolation proof, and responsive input in Task 5.
- Concrete runtime: not selected; the service-only path completed, the live
  profile named the UI mirror spin, and the supersession REDs named worker
  retention plus fresh-screen admission-lifetime gaps.
- Event loop: deterministic heartbeat/navigation ordering plus separate live
  acknowledgement, not inferred from final completion.
- Supersession: real Runs B, C, then D keep C/D outside admission while B
  drains through repeated cancellation without blocking the UI, never exceed
  one underlying retrieval, never apply stale B/C, and return to mounted
  terminal D evidence. A separate real-navigation B-to-D run proves the same
  admission bound across distinct fresh `LibraryScreen` instances.
- Restore: fresh Library screens keep saved query/results but normalize the
  transient `searching` claim because no matching worker is restored; Run is
  enabled and no dead in-flight line mounts.
- Privacy: artifacts allow only symbols, timing, aggregate counts, IDs, paths, and fingerprints.
- ADR: no new record; the Library-only app-lifetime lock directly implements
  ADR-033's narrow lifecycle owner while preserving ADR-003 Library ownership
  and ADR-005's runtime direction. Explicit stop if implementation requires a
  broader owner or service contract.
- YAGNI: one measured mirror owner, one narrow admission primitive, one
  retrieval path, no broad app state, new dependency, or scheduler.
- Approved corrections: suppress the programmatic sibling `Input.Changed` in
  `_patch_sibling_library_search_input()`; retain admitted retrieval tasks in
  `_execute_library_rag_search()` under the app-lifetime Library-only accessor;
  and normalize restored transient `searching`. No new offload, thread,
  process, runtime, storage, broad state, or service change is proposed.
