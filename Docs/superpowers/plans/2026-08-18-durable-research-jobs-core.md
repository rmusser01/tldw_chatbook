# Durable Research Jobs — Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a research run survivable — exactly one executor at a time, a budget that is not re-granted on resume, and phases that do not redo paid work.

**Architecture:** A lease (owner + fencing token + expiry) lives on the `research_runs` row. `LocalResearchService` gains atomic claim/renew/release operations; stale-lease reclaim happens inside claim, on a retry budget. `LocalResearchEngine` claims before executing, runs a keep-alive timer for as long as a phase is in flight, checks the fence before persisting writes, restores its ledger from the last snapshot, and resumes from the last completed phase using a persisted evidence pool.

**Tech Stack:** Python 3.11+, SQLite (`sqlite3`, WAL), pytest, asyncio.

## Global Constraints

- SQLite only, through `LocalResearchService._connect()`; parameterized queries only.
- New columns must be added idempotently — the schema uses `CREATE TABLE IF NOT EXISTS`, so an existing database never re-runs the DDL. Guard every `ALTER TABLE` with a `PRAGMA table_info` check.
- Tests use a real in-memory SQLite service (`LocalResearchService(":memory:")`), never a mock DB.
- No test contacts a network.
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree root.
- Scope: this plan is the durability core only. Scheduler auto-resume and completion surfacing are separate plans.

---

### Task 1: Lease columns and atomic claim

**Files:**
- Modify: `tldw_chatbook/Research_Interop/local_research_service.py` (schema block at ~185, new methods after `_update_run_state` at ~585)
- Test: `Tests/Research/test_research_run_lease.py` (create)

**Interfaces:**
- Consumes: `LocalResearchService._connect()`, `LocalResearchService._require_one(table, id, label)`, `LocalResearchService._now()`
- Produces:
  - `LocalResearchService.claim_run(run_id: str, *, worker_id: str, lease_seconds: float) -> str | None` — returns a new `lease_id` on success, `None` when another live lease holds the run
  - `research_runs` columns: `lease_owner TEXT`, `lease_id TEXT`, `leased_until TEXT`, `lease_attempts INTEGER NOT NULL DEFAULT 0`

- [ ] **Step 1: Write the failing test**

```python
# Tests/Research/test_research_run_lease.py
"""Run leases (task-18060): exactly one executor may hold a run."""

from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService


def _service() -> LocalResearchService:
    return LocalResearchService(":memory:")


def test_first_claim_succeeds_and_returns_a_lease_id():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    lease_id = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert isinstance(lease_id, str) and lease_id


def test_second_claim_is_refused_while_the_lease_is_live():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_run_lease.py -q`
Expected: FAIL with `AttributeError: 'LocalResearchService' object has no attribute 'claim_run'`

- [ ] **Step 3: Add the columns idempotently**

In `local_research_service.py`, immediately after the block that executes the `CREATE TABLE`/`CREATE INDEX` DDL, add:

```python
    #: Columns added after the original schema shipped. CREATE TABLE IF NOT
    #: EXISTS never revisits an existing database, so each one is applied by
    #: an idempotent ALTER guarded on PRAGMA table_info (task-18060).
    _RUN_COLUMN_ADDITIONS = (
        ("lease_owner", "TEXT"),
        ("lease_id", "TEXT"),
        ("leased_until", "TEXT"),
        ("lease_attempts", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _ensure_run_lease_columns(self, conn: sqlite3.Connection) -> None:
        """Add lease columns to research_runs when they are absent.

        Args:
            conn: An open connection inside the caller's transaction.
        """
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()
        }
        for column, declaration in self._RUN_COLUMN_ADDITIONS:
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE research_runs ADD COLUMN {column} {declaration}"
                )
```

Call it from the same place the DDL runs, immediately after the `executescript`/`execute` of the schema, passing the same `conn`.

- [ ] **Step 4: Implement the claim**

Add after `_update_run_state`:

```python
    def claim_run(
        self, run_id: str, *, worker_id: str, lease_seconds: float
    ) -> str | None:
        """Take the execution lease on a run, or decline it.

        The claim is atomic: the UPDATE only matches when no live lease
        exists, so two racing executors cannot both succeed (task-18060).

        Args:
            run_id: The run to claim.
            worker_id: Identifies the claiming executor.
            lease_seconds: How long the lease is valid without renewal.

        Returns:
            A new lease id when the claim succeeded, otherwise None.
        """
        self._require_one("research_runs", run_id, "research run")
        lease_id = uuid.uuid4().hex
        now = self._now()
        expires = self._timestamp_after(lease_seconds)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET lease_owner = ?, lease_id = ?, leased_until = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND (leased_until IS NULL OR leased_until <= ?)
                """,
                (worker_id, lease_id, expires, now, run_id, now),
            )
            if cursor.rowcount != 1:
                return None
        return lease_id
```

Add the timestamp helper beside `_now`:

```python
    def _timestamp_after(self, seconds: float) -> str:
        """An ISO timestamp ``seconds`` in the future, in the same format as
        ``_now`` so string comparison orders correctly."""
        return (
            datetime.now(timezone.utc) + timedelta(seconds=max(0.0, float(seconds)))
        ).isoformat()
```

Ensure `import uuid` and `from datetime import datetime, timedelta, timezone` are present at the top of the module.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_run_lease.py -q`
Expected: PASS, 2 passed

- [ ] **Step 6: Run the existing research suite for regressions**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Research_Interop/local_research_service.py Tests/Research/test_research_run_lease.py
git commit -m "feat(research): lease a run to exactly one executor (TASK-18060)"
```

---

### Task 2: Renewal, release, fencing, and stale reclaim on a retry budget

**Files:**
- Modify: `tldw_chatbook/Research_Interop/local_research_service.py`
- Test: `Tests/Research/test_research_run_lease.py`

**Interfaces:**
- Consumes: `claim_run` from Task 1; the `lease_owner`/`lease_id`/`leased_until`/`lease_attempts` columns
- Produces:
  - `renew_lease(run_id: str, *, lease_id: str, lease_seconds: float) -> bool`
  - `release_lease(run_id: str, *, lease_id: str) -> bool`
  - `holds_lease(run_id: str, *, lease_id: str) -> bool`
  - `claim_run` gains `max_attempts: int = 3`; a reclaim increments `lease_attempts` and refuses past the budget

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/Research/test_research_run_lease.py
# NOTE: a lease of 0 seconds expires the instant it is granted, so these tests
# exercise takeover deterministically. Do NOT use time.sleep() to age a lease:
# a wall-clock dependency makes the suite flaky on a loaded machine, and the
# behaviour under test is the comparison against `leased_until`, not duration.


def test_a_stale_lease_can_be_taken_over():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)


def test_a_displaced_worker_cannot_renew_or_release():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    stale = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)

    assert service.renew_lease(run["id"], lease_id=stale, lease_seconds=60) is False
    assert service.release_lease(run["id"], lease_id=stale) is False
    assert service.holds_lease(run["id"], lease_id=stale) is False


def test_reclaim_stops_at_the_retry_budget():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    for _ in range(3):
        assert service.claim_run(
            run["id"], worker_id="w", lease_seconds=0, max_attempts=3
        )

    assert service.claim_run(
        run["id"], worker_id="w", lease_seconds=0, max_attempts=3
    ) is None


def test_release_frees_the_run_for_the_next_executor():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    lease = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert service.release_lease(run["id"], lease_id=lease) is True
    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_run_lease.py -q`
Expected: FAIL — `renew_lease` undefined, and `claim_run` takes no `max_attempts`

- [ ] **Step 3: Implement renewal, release, fence check, and the attempt budget**

Replace `claim_run`'s signature and body with:

```python
    def claim_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: float,
        max_attempts: int = 3,
    ) -> str | None:
        """Take the execution lease on a run, or decline it.

        The claim is atomic: the UPDATE matches only when no live lease
        exists, so two racing executors cannot both succeed. Taking over an
        EXPIRED lease counts against ``max_attempts`` -- a run whose executor
        keeps dying is broken rather than slow, and must stop being retried
        (task-18060, following the server job manager's retry budget).

        Args:
            run_id: The run to claim.
            worker_id: Identifies the claiming executor.
            lease_seconds: How long the lease is valid without renewal.
            max_attempts: How many times a run may be reclaimed after an
                expired lease before claims are refused.

        Returns:
            A new lease id when the claim succeeded, otherwise None.
        """
        row = self._require_one("research_runs", run_id, "research run")
        previous = row["leased_until"] if "leased_until" in row.keys() else None
        attempts = int(row["lease_attempts"] or 0) if "lease_attempts" in row.keys() else 0
        reclaiming = previous is not None
        if reclaiming and attempts >= int(max_attempts):
            return None
        lease_id = uuid.uuid4().hex
        now = self._now()
        expires = self._timestamp_after(lease_seconds)
        next_attempts = attempts + 1 if reclaiming else attempts
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET lease_owner = ?, lease_id = ?, leased_until = ?,
                       lease_attempts = ?, updated_at = ?
                 WHERE id = ?
                   AND (leased_until IS NULL OR leased_until <= ?)
                """,
                (worker_id, lease_id, expires, next_attempts, now, run_id, now),
            )
            if cursor.rowcount != 1:
                return None
        return lease_id

    def renew_lease(
        self, run_id: str, *, lease_id: str, lease_seconds: float
    ) -> bool:
        """Extend a lease the caller still holds.

        The lease id is a fencing token: a worker that stalled past its lease
        and was taken over still matches on worker id, so matching on the id
        alone would let it act on a run it no longer owns (task-18060).

        Args:
            run_id: The leased run.
            lease_id: The token returned by ``claim_run``.
            lease_seconds: How much longer the lease should be valid.

        Returns:
            True when the lease was extended, False when it was lost.
        """
        expires = self._timestamp_after(lease_seconds)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET leased_until = ?, updated_at = ?
                 WHERE id = ? AND lease_id = ?
                """,
                (expires, self._now(), run_id, lease_id),
            )
            return cursor.rowcount == 1

    def release_lease(self, run_id: str, *, lease_id: str) -> bool:
        """Drop a lease the caller holds so another executor may claim it.

        Args:
            run_id: The leased run.
            lease_id: The token returned by ``claim_run``.

        Returns:
            True when the lease was released, False when it was already lost.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET lease_owner = NULL, lease_id = NULL, leased_until = NULL,
                       updated_at = ?
                 WHERE id = ? AND lease_id = ?
                """,
                (self._now(), run_id, lease_id),
            )
            return cursor.rowcount == 1

    def holds_lease(self, run_id: str, *, lease_id: str) -> bool:
        """Whether ``lease_id`` is still the live lease on the run.

        Args:
            run_id: The run to check.
            lease_id: The token returned by ``claim_run``.

        Returns:
            True when the token matches an unexpired lease.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT lease_id, leased_until FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row["lease_id"] != lease_id:
            return False
        return str(row["leased_until"] or "") > self._now()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_run_lease.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Research_Interop/local_research_service.py Tests/Research/test_research_run_lease.py
git commit -m "feat(research): lease renewal, fencing and a reclaim budget (TASK-18060)"
```

---

### Task 3: The engine claims, keeps alive, and fences its writes

**Files:**
- Modify: `tldw_chatbook/Research_Interop/local_research_engine.py` (`execute_run` at ~398, `_save_ledger`, `service.save_artifact` call sites)
- Test: `Tests/Research/test_local_research_engine.py`

**Interfaces:**
- Consumes: `claim_run`, `renew_lease`, `release_lease`, `holds_lease` from Tasks 1-2
- Produces:
  - `LocalResearchEngine.worker_id: str` (per-instance identity)
  - `LocalResearchEngine._lease_id: str | None` set for the duration of `execute_run`
  - `LocalResearchEngine._require_lease() -> None`, raising `_LeaseLost` when the fence fails

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/Research/test_local_research_engine.py


def test_a_second_engine_declines_a_leased_run():
    """task-18060: two executors must not run one run. The window's
    exclusive-worker guard is per-session and cannot see a second process."""
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    first = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    second = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    service.claim_run(run["id"], worker_id=first.worker_id, lease_seconds=60)
    final = asyncio.run(second.execute_run(run["id"]))

    assert final["status"] != "completed"
    assert "lease" in str(final.get("progress_message") or "").lower()


def test_a_long_silent_phase_keeps_its_lease():
    """The synthesis phase emits no progress for its whole duration, so a
    lease renewed only by progress events would expire inside it."""
    service = _make_service()
    search_fn, _analyze, _calls = _make_pipeline("q")

    async def slow_analyze(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.3)
        return {
            "final_answer": {"text": "Answer citing [1].", "evidence": [],
                             "confidence": 0.5, "chunks": []},
            "relevant_results": {},
        }

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=slow_analyze
    )
    engine.lease_seconds = 0.1
    engine.keepalive_seconds = 0.02
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed", final.get("progress_message")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_local_research_engine.py -q -k "declines_a_leased_run or long_silent_phase"`
Expected: FAIL — `LocalResearchEngine` has no attribute `worker_id`

- [ ] **Step 3: Implement claim, keep-alive and fence**

In `local_research_engine.py`, add near `_RunPaused`:

```python
class _LeaseLost(Exception):
    """Internal control-flow signal: this executor no longer owns the run."""
```

In `LocalResearchEngine.__init__`, after `self.service = local_service`:

```python
        #: Identity of this executor for leasing (task-18060).
        self.worker_id = f"engine-{uuid.uuid4().hex[:12]}"
        #: How long a lease is granted for, and how often it is renewed. The
        #: keep-alive is a TIMER, not a progress hook: the synthesis phase
        #: emits nothing for its whole duration, so a lease renewed only by
        #: progress would expire inside the most expensive phase.
        self.lease_seconds = 120.0
        self.keepalive_seconds = 30.0
        self._lease_id: str | None = None
```

Add `import uuid` to the module imports.

Add the fence and keep-alive helpers beside `_get_run`:

```python
    def _require_lease(self) -> None:
        """Raise when this executor no longer owns the run.

        Checked before every persisting write, not only at completion: a
        displaced executor blocked in a long provider call still returns, and
        would otherwise write artifacts and settle budget on its way out
        (task-18060).
        """
        if self._lease_id is None:
            return
        if not self.service.holds_lease(self._run_id or "", lease_id=self._lease_id):
            raise _LeaseLost("execution lease lost")

    async def _keepalive(self, run_id: str) -> None:
        """Renew the lease on a timer for as long as a phase is in flight."""
        while True:
            await asyncio.sleep(max(0.01, float(self.keepalive_seconds)))
            if self._lease_id is None:
                return
            if not self.service.renew_lease(
                run_id, lease_id=self._lease_id, lease_seconds=self.lease_seconds
            ):
                return
```

Add `import asyncio` if absent, and set `self._run_id: str | None = None` in `__init__`.

In `execute_run`, immediately after `run = self._get_run(run_id)` and the terminal-status check, add:

```python
        self._run_id = run_id
        self._lease_id = self.service.claim_run(
            run_id, worker_id=self.worker_id, lease_seconds=self.lease_seconds
        )
        if self._lease_id is None:
            logger.info(f"Research run {run_id} is leased by another executor")
            return self.service.update_run_progress(
                run_id,
                progress_message="another executor holds this run's lease",
                event="lease_declined",
            )
        keepalive = asyncio.create_task(self._keepalive(run_id))
```

Wrap the existing `try:` body so the keep-alive and lease are always cleaned up, and translate a lost lease into a terminal state:

```python
        except _LeaseLost:
            logger.warning(f"Research run {run_id} lease lost mid-flight")
            return self._get_run(run_id)
        finally:
            keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive
            if self._lease_id is not None:
                self.service.release_lease(run_id, lease_id=self._lease_id)
                self._lease_id = None
            self._run_id = None
```

Add `import contextlib` if absent. Call `self._require_lease()` as the first statement of `_save_ledger`, and immediately before each `self.service.save_artifact(...)` call inside `_execute_phases`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_local_research_engine.py -q`
Expected: PASS, all engine tests green

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Research_Interop/local_research_engine.py Tests/Research/test_local_research_engine.py
git commit -m "feat(research): claim, keep alive, and fence engine writes (TASK-18060)"
```

---

### Task 4: Resume restores the spent budget

**Files:**
- Modify: `tldw_chatbook/Research_Interop/research_budget.py` (add `from_snapshot`), `tldw_chatbook/Research_Interop/local_research_engine.py` (`execute_run` ledger construction)
- Test: `Tests/Research/test_research_budget.py`, `Tests/Research/test_local_research_engine.py`

**Interfaces:**
- Consumes: `BudgetLedger.snapshot()` (existing), `LocalResearchService.get_artifact(run_id, name)` (existing)
- Produces: `BudgetLedger.from_snapshot(snapshot: Mapping[str, Any] | None, limits: Mapping[str, Any] | None) -> BudgetLedger`

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Research/test_research_budget.py


def test_from_snapshot_restores_spend_rather_than_regranting_it():
    """task-18060: execute_run rebuilt the ledger from limits on every entry
    and never read budget_ledger.json back, so a resumed run was granted its
    full budget again. With resume routine rather than exceptional, that is a
    budget leak."""
    from tldw_chatbook.Research_Interop.research_budget import BudgetLedger

    original = BudgetLedger.from_limits({"max_searches": 10})
    original.reserve_searches(4)
    original.settle_searches(4)
    snapshot = original.snapshot()

    restored = BudgetLedger.from_snapshot(snapshot, {"max_searches": 10})

    assert restored.remaining_searches() == 6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_budget.py -q -k from_snapshot`
Expected: FAIL with `AttributeError: type object 'BudgetLedger' has no attribute 'from_snapshot'`

- [ ] **Step 3: Implement `from_snapshot`**

In `research_budget.py`, beside `from_limits`:

```python
    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | None,
        limits: Mapping[str, Any] | None,
    ) -> "BudgetLedger":
        """Rebuild a ledger that has already spent part of its budget.

        Args:
            snapshot: A prior ``snapshot()`` payload, or None for a fresh run.
            limits: The run's limits, used when the snapshot has none.

        Returns:
            A ledger whose used counters continue from the snapshot.
        """
        ledger = cls(dict((snapshot or {}).get("limits") or limits or {}))
        if not snapshot:
            return ledger
        ledger.searches_used = int(snapshot.get("searches_used") or 0)
        ledger.searches_overshoot = int(snapshot.get("searches_overshoot") or 0)
        ledger.docs_used = int(snapshot.get("docs_used") or 0)
        ledger.tokens_settled = int(snapshot.get("tokens_settled") or 0)
        return ledger
```

- [ ] **Step 4: Run it to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_research_budget.py -q -k from_snapshot`
Expected: PASS

- [ ] **Step 5: Use it in the engine**

In `execute_run`, replace `ledger = BudgetLedger.from_limits(limits)` with:

```python
        # task-18060: a resumed run continues its budget rather than being
        # granted it again. The snapshot is the ledger written by the previous
        # executor; its absence means this run has never executed.
        previous_ledger = (
            self.service.get_artifact(run_id, "budget_ledger.json") or {}
        ).get("content")
        ledger = BudgetLedger.from_snapshot(previous_ledger, limits)
```

- [ ] **Step 6: Write the engine-level test**

```python
# append to Tests/Research/test_local_research_engine.py


def test_a_resumed_run_does_not_get_its_budget_back():
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_searches": 10}
    )
    asyncio.run(engine.execute_run(run["id"]))

    snapshot = (service.get_artifact(run["id"], "budget_ledger.json") or {}).get(
        "content"
    ) or {}

    assert int(snapshot.get("searches_used") or 0) > 0
    restored = BudgetLedger.from_snapshot(snapshot, {"max_searches": 10})
    assert restored.remaining_searches() < 10
```

Add `from tldw_chatbook.Research_Interop.research_budget import BudgetLedger` to that test module's imports if absent.

- [ ] **Step 7: Run the suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research -q`
Expected: PASS, no new failures

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Research_Interop/research_budget.py tldw_chatbook/Research_Interop/local_research_engine.py Tests/Research/test_research_budget.py Tests/Research/test_local_research_engine.py
git commit -m "feat(research): resume restores spent budget instead of re-granting it (TASK-18060)"
```

---

### Task 5: Persist the round's evidence and resume from it

**Files:**
- Modify: `tldw_chatbook/Research_Interop/local_research_engine.py` (`_execute_phases` collection loop)
- Test: `Tests/Research/test_local_research_engine.py`

**Interfaces:**
- Consumes: `self.service.save_artifact`, `self.service.get_artifact`, `self._require_lease()` from Task 3
- Produces: artifact `evidence_pool.json` — `{"iteration": int, "results": list[dict], "truncated": bool, "cap_bytes": int}`

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Research/test_local_research_engine.py


def test_a_resumed_run_reuses_persisted_evidence_instead_of_researching():
    """task-18060: collection_summary.json persists counts and sub-questions
    but not the evidence, so a resumed run re-searched everything it had
    already paid for."""
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )
    asyncio.run(engine.execute_run(run["id"]))
    searches_first_pass = calls["search"]

    pool = (service.get_artifact(run["id"], "evidence_pool.json") or {}).get("content")

    assert pool and pool["results"], "the round's evidence must be persisted"
    assert searches_first_pass == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_local_research_engine.py -q -k reuses_persisted_evidence`
Expected: FAIL — `pool` is None, because no such artifact is written

- [ ] **Step 3: Persist the pool under a stated cap**

In `_execute_phases`, immediately after `merged_warnings.extend(round_warnings)` in the collection loop, add:

```python
            # task-18060: the pool itself is persisted so a resumed run does
            # not re-search what it already paid for. Bounded explicitly --
            # 66 sources of scraped text is roughly 0.7-3 MB per round, so
            # beyond the cap the entries persist without their content and the
            # artifact records that it happened.
            self._require_lease()
            self.service.save_artifact(
                run_id,
                artifact_name="evidence_pool.json",
                content_type="application/json",
                content=self._bounded_evidence(merged_results, iteration),
            )
```

Add the helper beside `_save_ledger`:

```python
    #: Bytes of evidence persisted per run before content is dropped.
    EVIDENCE_POOL_CAP_BYTES = 8 * 1024 * 1024

    def _bounded_evidence(
        self, results: list[dict[str, Any]], iteration: int
    ) -> dict[str, Any]:
        """Shape the round's evidence for persistence, under a byte cap.

        Args:
            results: The round's merged evidence records.
            iteration: The round these belong to.

        Returns:
            A payload carrying the evidence, and whether content was dropped.
        """
        kept: list[dict[str, Any]] = []
        used = 0
        truncated = False
        for record in results:
            entry = dict(record)
            size = len(json.dumps(entry, default=str))
            if used + size > self.EVIDENCE_POOL_CAP_BYTES:
                entry.pop("content", None)
                entry.pop("original_content", None)
                truncated = True
                size = len(json.dumps(entry, default=str))
            used += size
            kept.append(entry)
        return {
            "iteration": iteration,
            "results": kept,
            "truncated": truncated,
            "cap_bytes": self.EVIDENCE_POOL_CAP_BYTES,
        }
```

`json` is already imported in this module.

- [ ] **Step 4: Run it to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research/test_local_research_engine.py -q -k reuses_persisted_evidence`
Expected: PASS

- [ ] **Step 5: Write the truncation test**

```python
# append to Tests/Research/test_local_research_engine.py


def test_evidence_beyond_the_cap_persists_without_content():
    service = _make_service()
    engine = LocalResearchEngine(service)
    engine.EVIDENCE_POOL_CAP_BYTES = 200
    bulky = [
        {"url": f"https://e.example/{n}", "content": "x" * 500,
         "original_content": "y" * 500}
        for n in range(4)
    ]

    payload = engine._bounded_evidence(bulky, iteration=1)

    assert payload["truncated"] is True
    assert any("content" not in entry for entry in payload["results"])
    assert payload["cap_bytes"] == 200
```

- [ ] **Step 6: Run the full research suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Research -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Research_Interop/local_research_engine.py Tests/Research/test_local_research_engine.py
git commit -m "feat(research): persist the round's evidence under a stated cap (TASK-18060)"
```

---

## Follow-on plans (not this document)

- **Scheduler integration:** a `research_run` task type whose handler claims and executes, skips `awaiting_*` runs, and is deleted when the run reaches a terminal state. Depends on Tasks 1-3.
- **Surfacing:** a handoff target for window-launched runs with an idempotent delivery marker, and `artifact_source`/`artifact_kind` metadata so reports reach the artifacts screen, with the matching `Docs/User_Guide/` update.
