# Ingest heavy-lane cap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-library-ingest-heavy-lane-design.md`. Branch `claude/followups-ingest-heavy-lane` off dev `b87fc0e9`. Anchors exact at branch point; grep symbols, line numbers drift.

**Goal:** Cap concurrent audio/video transcription parses (default 1, config-controlled) independently of document parses in the F3 parse pool, with skip-ahead so documents keep filling the pool past a blocked transcription.

**Architecture:** Keep the single parse pool; make the dispatcher heavy-lane-aware. Classify each file once at enqueue (`detect_file_type`) and store `detected_type` on the QUEUED job row. The registry gains a type-filtered `next_queued(skip_types=…)` and a `parsing_count_for_types()`. The dispatcher skips heavy jobs when the heavy lane is full and dispatches lighter jobs past them.

**Tech Stack:** Python ≥3.11, multiprocessing, pytest. No new third-party deps.

## Global Constraints

- **No behavior change with defaults:** `next_queued()` (no args), `submit(...)` without `detected_type`, and an unset config all behave exactly as today. Every existing ingest test stays green unchanged.
- **Heavy set:** `_INGEST_HEAVY_TYPES = frozenset({"audio", "video"})` — a module constant in `app.py`, the exact `detect_file_type` values that transcribe. Not configurable.
- **Config:** `library.ingest_heavy_lane_max_workers`, default **1**, values `<=0` clamp to **1** (never starve heavy work). A cap `> worker_count` is harmless.
- **Classify once:** `detect_file_type` runs at enqueue (`submit_library_ingest_job`), stored on the QUEUED row; the dispatch-time recompute (`app.py:1574`) is removed and `mark_parsing` receives the stored `job.detected_type`.
- **Skip-ahead** ⇒ jobs may complete out of enqueue order (accepted). The `mark_parsing`-rejected `break` guard and one-dispatch-per-iteration structure are preserved (no `continue`).
- **`parsing_count_for_types` mirrors `counts()`** — excludes `superseded`/`dismissed` jobs, so the heavy count aligns with the total-slot accounting.
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: Registry seam — `detected_type` at submit, `next_queued(skip_types)`, `parsing_count_for_types` (pure)

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Test: `Tests/Library/test_library_ingest_jobs.py`

**Interfaces:**
- Produces:
  - `LibraryIngestJobRegistry.submit(..., detected_type: str = "")` — stores it on the created `QUEUED` job.
  - `LibraryIngestJobRegistry.next_queued(*, skip_types: frozenset[str] = frozenset())` — oldest QUEUED job whose `detected_type ∉ skip_types`; `None` if none.
  - `LibraryIngestJobRegistry.parsing_count_for_types(types: frozenset[str]) -> int` — count of visible (non-superseded, non-dismissed) `PARSING` jobs whose `detected_type ∈ types`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Library/test_library_ingest_jobs.py`:
```python
def test_submit_stores_detected_type():
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="a.mp3", detected_type="audio")
    assert job.detected_type == "audio"
    assert registry.next_queued().detected_type == "audio"


def test_next_queued_skip_types_skips_heavy_returns_next_light():
    registry = LibraryIngestJobRegistry()
    a1 = registry.submit(source_path="a1.mp3", detected_type="audio")
    a2 = registry.submit(source_path="a2.mp3", detected_type="audio")
    d1 = registry.submit(source_path="d1.txt", detected_type="plaintext")
    heavy = frozenset({"audio", "video"})
    # default: oldest queued regardless of type
    assert registry.next_queued().job_id == a1.job_id
    # skipping heavy: first non-heavy queued job
    assert registry.next_queued(skip_types=heavy).job_id == d1.job_id


def test_next_queued_skip_types_none_when_only_heavy_left():
    registry = LibraryIngestJobRegistry()
    registry.submit(source_path="a1.mp3", detected_type="audio")
    registry.submit(source_path="a2.mp3", detected_type="video")
    assert registry.next_queued(skip_types=frozenset({"audio", "video"})) is None


def test_parsing_count_for_types_counts_only_inflight_heavy():
    registry = LibraryIngestJobRegistry()
    a1 = registry.submit(source_path="a1.mp3", detected_type="audio")
    a2 = registry.submit(source_path="a2.mp3", detected_type="audio")
    d1 = registry.submit(source_path="d1.txt", detected_type="plaintext")
    heavy = frozenset({"audio", "video"})
    assert registry.parsing_count_for_types(heavy) == 0        # all still QUEUED
    registry.mark_parsing(a1.job_id, detected_type="audio")
    registry.mark_parsing(d1.job_id, detected_type="plaintext")
    assert registry.parsing_count_for_types(heavy) == 1        # only a1 is heavy+parsing
    assert a2.job_id                                            # a2 still QUEUED, not counted
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `submit() got an unexpected keyword argument 'detected_type'` / `next_queued() got an unexpected keyword argument 'skip_types'` / no attribute `parsing_count_for_types`.

- [ ] **Step 3: Add `detected_type` to `submit`**

In `library_ingest_jobs.py`, add `detected_type: str = "",` to `submit`'s keyword-only params (after `chunk_size`), and pass `detected_type=detected_type,` into the `LibraryIngestJob(...)` constructor inside `submit`.

- [ ] **Step 4: Add `skip_types` to `next_queued`**

Replace `next_queued`:
```python
    def next_queued(self, *, skip_types: frozenset[str] = frozenset()) -> LibraryIngestJob | None:
        """Return the oldest still-``QUEUED`` job, or ``None`` if none.

        Args:
            skip_types: ``detected_type`` values to skip. Empty (default)
                returns the oldest queued job of any type; a non-empty set
                returns the oldest queued job whose ``detected_type`` is not
                in the set (skip-ahead for the heavy-lane cap).
        """
        for job in self._jobs:
            if job.state == IngestJobState.QUEUED and job.detected_type not in skip_types:
                return replace(job)
        return None
```

- [ ] **Step 5: Add `parsing_count_for_types`**

Add next to `counts` (mirror its superseded/dismissed exclusion):
```python
    def parsing_count_for_types(self, types: frozenset[str]) -> int:
        """Count visible ``PARSING`` jobs whose ``detected_type`` is in ``types``.

        Excludes ``superseded``/``dismissed`` jobs, matching ``counts()`` so
        the heavy-lane in-flight count aligns with the total-slot accounting.
        """
        return sum(
            1
            for job in self._jobs
            if job.state == IngestJobState.PARSING
            and not job.superseded
            and not job.dismissed
            and job.detected_type in types
        )
```

- [ ] **Step 6: Run to verify it passes + full registry suite**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: PASS (new tests green; the existing `next_queued()`/`submit()` tests unchanged).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Library/library_ingest_jobs.py Tests/Library/test_library_ingest_jobs.py
git commit -m "feat(ingest): registry detected_type at submit + type-filtered queue selection (160)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Config helper + `[library]` template docs

**Files:**
- Modify: `tldw_chatbook/app.py` (add `_ingest_heavy_lane_max_workers` next to `_ingest_parse_worker_count` ~:1371)
- Modify: `tldw_chatbook/config.py` (`CONFIG_TOML_CONTENT` ~:1465)
- Test: `Tests/Library/test_ingest_heavy_lane_config.py` (create)

**Interfaces:**
- Produces: `LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(self) -> int` — configured value, else 1; `<=0` → 1.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_ingest_heavy_lane_config.py`:
```python
import tomllib
from types import SimpleNamespace

import tldw_chatbook.app as app_module
from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_heavy_lane_default_when_unset(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: None)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 1


def test_heavy_lane_uses_configured_value(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: 2)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 2


def test_heavy_lane_clamps_non_positive_to_one(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: 0)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 1


def test_config_template_valid_toml_and_heavy_lane_key_commented():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)   # must not raise
    # The key is documented as a COMMENT, so a fresh template parse must not
    # set it -- keeping the runtime default (1) in force.
    assert "ingest_heavy_lane_max_workers" not in parsed.get("library", {})
    assert "ingest_heavy_lane_max_workers" in CONFIG_TOML_CONTENT  # documented (commented)
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_ingest_heavy_lane_config.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `LibraryIngestQueueMixin` has no `_ingest_heavy_lane_max_workers`; and the `ingest_heavy_lane_max_workers in CONFIG_TOML_CONTENT` assertion fails.

- [ ] **Step 3: Add the config helper**

In `app.py`, directly below `_ingest_parse_worker_count` (~:1393), add:
```python
    def _ingest_heavy_lane_max_workers(self) -> int:
        """Resolve the heavy-lane (audio/video transcription) cap from config.

        UI-thread only. Reads ``library.ingest_heavy_lane_max_workers`` via the
        dotted 1-arg ``get_cli_setting`` form (same reason as
        ``_ingest_parse_worker_count``). Defaults to 1; a missing, invalid, or
        non-positive value clamps to 1 so heavy work is never permanently
        starved.
        """
        try:
            configured = int(get_cli_setting("library.ingest_heavy_lane_max_workers"))
        except (TypeError, ValueError):
            configured = 0
        return configured if configured > 0 else 1
```

- [ ] **Step 4: Document both keys in the template**

In `config.py`, inside the `CONFIG_TOML_CONTENT` string (~:1465), add a `[library]` section near the existing ingest/media config (find a stable spot between two existing top-level sections; place it as its own block):
```toml
[library]
# Parallel ingest parse workers. Default: min(3, cpu-1). Uncomment to override.
# ingest_parse_workers = 3
# Max concurrent heavy (audio/video transcription) parses; document parses fan
# out past this cap to fill the remaining pool workers. Default: 1.
# ingest_heavy_lane_max_workers = 1
```
Keep both keys commented so the shipped defaults stay in force.

- [ ] **Step 5: Run to verify it passes**

Run the Step-1 tests. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/config.py Tests/Library/test_ingest_heavy_lane_config.py
git commit -m "feat(ingest): heavy-lane cap config helper + documented [library] keys (160)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Dispatcher — classify at enqueue + heavy-lane skip-ahead gate

**Files:**
- Modify: `tldw_chatbook/app.py` (`_INGEST_HEAVY_TYPES` constant; `submit_library_ingest_job` ~:1295; `_top_up_ingest_parse_pool` ~:1542)
- Modify: `Tests/Library/test_library_ingest_runner.py` (harness `heavy_lane` override + scenario test)

**Interfaces:**
- Consumes: `next_queued(skip_types=…)`, `parsing_count_for_types(…)`, `submit(..., detected_type=…)` (Task 1); `_ingest_heavy_lane_max_workers()` (Task 2).

- [ ] **Step 1: Write the failing test**

First extend the test harness. In `Tests/Library/test_library_ingest_runner.py`, add a `heavy_lane=None` param to `_IngestRunnerHarness.__init__` (alongside `worker_count`), store `self._heavy_lane_override = heavy_lane`, and add the override method next to `_ingest_parse_worker_count`:
```python
    def _ingest_heavy_lane_max_workers(self) -> int:
        if self._heavy_lane_override is not None:
            return self._heavy_lane_override
        return super()._ingest_heavy_lane_max_workers()
```
Then add the scenario test (mirrors `test_submit_cap_backpressure_*`; uses the fake pool in manual mode). Match the existing test's fixtures/imports (the `db` fixture, `_FakeIngestParsePool`, `tmp_path`):
```python
@pytest.mark.asyncio
async def test_heavy_lane_caps_transcriptions_while_documents_fill_pool(db, tmp_path):
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=3, heavy_lane=1)
    async with app.run_test():
        paths = {}
        for name in ("a1.mp3", "a2.mp3", "d1.txt", "d2.txt", "d3.txt"):
            p = tmp_path / name
            p.write_text("x", encoding="utf-8")
            paths[name] = app.submit_library_ingest_job(source_path=str(p))

        # pool holds exactly 3: audio1 (heavy) + doc1 + doc2. audio2 is
        # skipped (heavy lane full); doc3 waits (pool full).
        assert len(pool.calls) == 3
        states = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states[paths["a1.mp3"].job_id] == IngestJobState.PARSING
        assert states[paths["d1.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["d2.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["a2.mp3"].job_id] == IngestJobState.QUEUED
        assert states[paths["d3.txt"].job_id] == IngestJobState.QUEUED

        # completing audio1 frees the heavy slot -> audio2 is admitted next.
        pool.trigger_success(0, {"ok": True, "payload": {}})
        assert len(pool.calls) == 4
        states_after = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states_after[paths["a2.mp3"].job_id] == IngestJobState.PARSING
```
(Match `trigger_success`'s exact signature + the completion-result shape from `_FakeIngestParsePool`/the sibling `test_submit_cap_backpressure_*` test — grep the file. `IngestJobState` and the `db` fixture are already imported/available in this test module; reuse them.)

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  "Tests/Library/test_library_ingest_runner.py::test_heavy_lane_caps_transcriptions_while_documents_fill_pool" \
  -q -p no:cacheprovider -o addopts="" --timeout=180
```
Expected: FAIL — without the dispatcher gate, all 3 pool slots fill with `[a1, a2, d1]` (a2 not skipped), so `states[a2] == PARSING` (not QUEUED) and the assertions fail. (The harness `heavy_lane` override + `_ingest_heavy_lane_max_workers` you added make the test *run*; the missing dispatcher logic makes it *fail on the assertions* — the intended RED.)

- [ ] **Step 3: Add the heavy-types constant**

In `app.py`, near the other ingest module constants (top of the file or just above `LibraryIngestQueueMixin`), add:
```python
# The detect_file_type() values whose parse worker runs transcription
# (see Local_Ingestion/local_file_ingestion.py audio/video branches). The
# heavy-lane cap limits how many of these parse concurrently.
_INGEST_HEAVY_TYPES = frozenset({"audio", "video"})
```

- [ ] **Step 4: Classify at enqueue, drop the dispatch recompute**

In `submit_library_ingest_job` (~:1295), before the `self.library_ingest_jobs.submit(...)` call, compute the type and pass it in:
```python
        try:
            detected_type = detect_file_type(source_path) or ""
        except Exception:
            detected_type = ""
        job = self.library_ingest_jobs.submit(
            source_path=source_path,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            detected_type=detected_type,
        )
```
In `_top_up_ingest_parse_pool` (~:1567), DELETE the dispatch-time recompute block:
```python
            try:
                detected_type = detect_file_type(job.source_path) or ""
            except Exception:
                detected_type = ""
```
and change the claim to use the stored type:
```python
            claimed = self.library_ingest_jobs.mark_parsing(
                job.job_id, detected_type=job.detected_type
            )
```
(Update the method's docstring paragraph about "detect_file_type is called here" — it now happens at enqueue.)

- [ ] **Step 5: Add the heavy-lane gate to the top-up loop**

Replace the loop header + selection in `_top_up_ingest_parse_pool`:
```python
        worker_count = self._ingest_parse_worker_count()
        heavy_cap = self._ingest_heavy_lane_max_workers()
        while self.library_ingest_jobs.counts().get("parsing", 0) < worker_count:
            heavy_full = (
                self.library_ingest_jobs.parsing_count_for_types(_INGEST_HEAVY_TYPES)
                >= heavy_cap
            )
            job = self.library_ingest_jobs.next_queued(
                skip_types=_INGEST_HEAVY_TYPES if heavy_full else frozenset()
            )
            if job is None:
                return
            # ... existing mark_parsing (now with job.detected_type) + break guard
            #     + options/pool/apply_async body UNCHANGED ...
```
Everything from `mark_parsing` onward (the `claimed is None` break guard, options build, `_ensure_ingest_parse_pool`, `apply_async`) stays exactly as-is.

- [ ] **Step 6: Run to verify it passes + the full runner suite**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_runner.py Tests/Library/test_library_ingest_jobs.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (the new scenario test green; every existing runner/registry test — including `test_submit_cap_backpressure_*` — unchanged, since `heavy_lane` defaults to a real cap of 1 but those tests use light-type or single-job paths). If a pre-existing runner test used an audio/video extension in a way the cap now changes, adjust that test's expectation ONLY if it's a genuine consequence of the cap (note it in the report).

- [ ] **Step 7: Mark backlog Done + commit**

```bash
perl -0pi -e 's/- \[ \] (#\d)/- [x] $1/g' "backlog/tasks/task-160 - Ingest-parallelism-heavy-lane-cap-for-concurrent-transcriptions.md"
perl -0pi -e 's/^status: .*/status: Done/m' "backlog/tasks/task-160 - Ingest-parallelism-heavy-lane-cap-for-concurrent-transcriptions.md"
```
Add a short `## Implementation Notes` section (classify-at-enqueue; registry `next_queued(skip_types)`/`parsing_count_for_types`; skip-ahead dispatcher; config key + template docs).
```bash
git add tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py \
  "backlog/tasks/task-160 - Ingest-parallelism-heavy-lane-cap-for-concurrent-transcriptions.md"
git commit -m "feat(ingest): heavy-lane skip-ahead cap in the parse-pool dispatcher; task 160 done (160)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 3)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_runner.py Tests/Library/test_ingest_heavy_lane_config.py Tests/Local_Ingestion/ \
  -q -p no:cacheprovider -o addopts="" --timeout=600 --timeout-method=thread
```
Plus `python -c "import tldw_chatbook.app"`. Then the whole-branch review (opus) and finishing-a-development-branch. No visual QA (no UI change).
