# Watchlists URLMonitor Off-Event-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the remaining CPU-heavy HTML extraction and text comparison stages in local Watchlists URL monitoring off the asyncio event-loop thread without changing results, persistence, or per-URL ordering.

**Architecture:** Keep the existing async fetch, threshold decision, circuit breaker, database offload, and sequential `url_list`/`sitemap` orchestration intact. Use the standard library's `asyncio.to_thread` at two narrow boundaries: HTML extraction and comparison; group the significant-change calculations in one private synchronous helper so each changed page incurs one worker hop and one pair of segment lists.

**Tech Stack:** Python 3.11+, asyncio, `asyncio.to_thread`, BeautifulSoup, difflib, pytest, pytest-asyncio, SQLite-backed `SubscriptionsDB`.

---

## File map

- Modify `tldw_chatbook/Subscriptions/monitoring_engine.py`: add one private significant-change helper and await CPU work through `asyncio.to_thread`.
- Modify `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`: add thread-identity, threshold short-circuit, cancellation, and real changed-item coverage.
- Modify `Tests/Subscriptions/test_local_watchlists_service.py`: prove every URL in a real `url_list` run takes the off-loop path while order and result accounting remain unchanged.
- Modify `backlog/tasks/task-15764 - Watchlists-URLMonitor-extraction-and-diff-work-off-the-event-loop.md`: align acceptance criteria with the approved `SequenceMatcher` percentage offload and record final evidence.

ADR required: no.

ADR path: N/A.

Reason: TASK-15764 is the direct residual performance fix deferred by TASK-15463. It preserves storage, runtime, service, and public contracts and adds no new dependency or long-lived boundary.

### Task 1: Pin HTML extraction to a worker thread

**Files:**
- Modify: `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py:1448-1525`

- [ ] **Step 1: Write the failing extraction thread-identity test**

Add a real `_fetch_url_content` test that patches only the guarded HTTP response and wraps the real extractor:

```python
@pytest.mark.asyncio
async def test_url_html_extraction_runs_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    extraction_threads: list[int] = []
    real_extract = ContentExtractor.extract_text_from_html

    def recording_extract(html, ignore_selectors=None):
        extraction_threads.append(threading.get_ident())
        return real_extract(html, ignore_selectors)

    _serve(monkeypatch, "<html><body><p>Hello</p></body></html>", content_type="text/html")
    monkeypatch.setattr(ContentExtractor, "extract_text_from_html", recording_extract)

    result = await URLMonitor(SimpleNamespace())._fetch_url_content(
        {"source": "https://example.com/page", "extraction_method": "auto"}
    )

    assert result["text"] == "Hello"
    assert extraction_threads and all(t != loop_thread for t in extraction_threads)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py::test_url_html_extraction_runs_off_the_event_loop
```

Expected: FAIL because the extractor records the event-loop thread.

- [ ] **Step 3: Implement the minimal extraction offload**

Replace only the `full`/`auto` extraction call:

```python
text = await asyncio.to_thread(
    ContentExtractor.extract_text_from_html,
    response.text,
    ignore_selectors,
)
```

Leave the raw-content branch, HTTP client, rate limiting, and response shaping unchanged.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command.

Expected: PASS; extracted text remains `Hello` and the recorded thread differs from the loop thread.

- [ ] **Step 5: Commit the extraction slice**

```bash
git add Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py tldw_chatbook/Subscriptions/monitoring_engine.py
git commit -m "perf(watchlists): offload URL HTML extraction"
```

### Task 2: Offload percentage and significant-change calculations

**Files:**
- Modify: `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py:351-565,1156-1410`

- [ ] **Step 1: Write failing changed-item thread and semantics tests**

Add `test_changed_item_cpu_work_runs_off_the_event_loop_without_changing_semantics`. Use a file-backed `SubscriptionsDB`, run one real `check_url` to store a baseline, then a second with changed HTML. Wrap `ContentExtractor.calculate_change_percentage`, `_segment_for_diff`, `build_change_diff`, `added_and_removed_text`, and `classify_change_type` so each records `threading.get_ident()` before delegating to the real callable.

Assert:

```python
assert disposition["kind"] == "changed"
assert item is not None
assert item["content_kind"] == "change"
assert item["content_format"] == "diff"
assert "Opus 4.5" in item["content"]
assert percentage_threads and all(t != loop_thread for t in percentage_threads)
assert operation_counts == {
    "segment": 2,
    "build": 1,
    "added_removed": 1,
    "classify": 1,
}
assert all(t != loop_thread for t in significant_change_threads)
```

Also capture the existing change percentage, diff summary, added/removed rule text, stored snapshot, and disposition so the test proves the worker boundary did not alter values or write order.

- [ ] **Step 2: Write the failing below-threshold short-circuit test**

Add `test_below_threshold_cpu_work_stops_before_significant_change_details`. Seed a baseline, set `change_threshold` above the real changed-page ratio, wrap `calculate_change_percentage` to record its thread, and replace the new helper seam with a function that raises if called.

Assert the result is `None`, disposition is `withheld`, percentage ran off-loop, the significant-change helper was not called, and no new snapshot was written.

- [ ] **Step 3: Run both tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py -k "changed_item_cpu_work or below_threshold_cpu_work"
```

Expected: FAIL because percentage and significant diff work still execute on the loop and the helper does not exist.

- [ ] **Step 4: Add one private synchronous significant-change helper**

Place the helper beside the existing diff helpers:

```python
def _build_significant_change_details(
    previous_text: str,
    current_text: str,
) -> tuple[str, str, str, str, str]:
    old_segments = _segment_for_diff(previous_text)
    new_segments = _segment_for_diff(current_text)
    diff_body, diff_summary = build_change_diff(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    added_text, removed_text = added_and_removed_text(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    return (
        diff_body,
        diff_summary,
        added_text,
        removed_text,
        classify_change_type(previous_text, current_text),
    )
```

Do not add a class, executor parameter, public export, or configuration switch.

- [ ] **Step 5: Await percentage and significant-change work from `check_url`**

Replace the inline calls with:

```python
change_percentage = await asyncio.to_thread(
    ContentExtractor.calculate_change_percentage,
    previous_text,
    current_content["text"],
)
```

Keep the threshold branch immediately after this await. Only after it passes, run:

```python
(
    diff_body,
    diff_summary,
    added_text,
    removed_text,
    change_type,
) = await asyncio.to_thread(
    _build_significant_change_details,
    previous_text,
    current_content["text"],
)
```

Use `change_type` in `change_info`; leave snapshot persistence and circuit-breaker success after result assembly exactly where they are.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the Step 3 command.

Expected: PASS with percentage in a worker, exactly two segment calls in the single significant-change worker, unchanged disposition and evidence, and no helper call below threshold.

- [ ] **Step 7: Commit the comparison slice**

```bash
git add Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py tldw_chatbook/Subscriptions/monitoring_engine.py
git commit -m "perf(watchlists): offload URL change comparison"
```

### Task 3: Pin cancellation behavior

**Files:**
- Modify: `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`

- [ ] **Step 1: Write the cancellation regression**

Wrap the real extractor with `threading.Event` gates. Start `check_url`, wait until the worker enters extraction without blocking the event loop, cancel the task, assert `CancelledError`, release the worker, and wait for its completion from another worker hop.

Assert after completion:

```python
assert db.conn.execute(
    "SELECT COUNT(*) FROM url_snapshots WHERE subscription_id = ?",
    (source_id,),
).fetchone()[0] == 0
assert monitor.circuit_breakers[source_id].failure_count == 0
```

This pins the truthful contract: the thread may finish, but the cancelled coroutine does not resume to persist or report success/failure.

- [ ] **Step 2: Run the cancellation test**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py::test_cancelled_url_check_does_not_resume_after_extraction_worker_finishes
```

Expected: PASS with the Task 1 implementation. If it fails, fix only the worker-await boundary; do not add shielding, process management, or custom cancellation machinery.

- [ ] **Step 3: Commit the regression**

```bash
git add Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py
git commit -m "test(watchlists): pin URL worker cancellation"
```

### Task 4: Prove every URL in a list follows the worker path

**Files:**
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`

- [ ] **Step 1: Write the real multi-URL regression**

Extend the existing real default-monitor `url_list` coverage rather than introducing a fake monitor. Create a two-URL source, serve baseline bodies for the first run and changed bodies for the second, and wrap the real extractor, percentage function, and significant-change helper to record `(url, thread_id)` through per-request body markers.

Assert:

```python
assert fetched_urls == [url_a, url_b, url_a, url_b]
assert changed_item_urls == [url_a, url_b]
assert extraction_urls == [url_a, url_b, url_a, url_b]
assert percentage_urls == [url_a, url_b]
assert detail_urls == [url_a, url_b]
assert all(thread_id != loop_thread for _, thread_id in cpu_calls)
```

Also assert the second run's disposition counts report two changed URLs and persisted items preserve URL order.

- [ ] **Step 2: Run the new test and verify it passes with the implementation**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Subscriptions/test_local_watchlists_service.py::test_url_list_offloads_cpu_work_for_every_url_in_order
```

Expected: PASS. Temporarily restoring either production offload call to an inline call must make the thread assertion fail; returning after the first URL must make the call-count/order assertions fail.

- [ ] **Step 3: Commit the list regression**

```bash
git add Tests/Subscriptions/test_local_watchlists_service.py
git commit -m "test(watchlists): cover URL-list CPU offload"
```

### Task 5: Run impacted verification and close TASK-15764

**Files:**
- Modify: `backlog/tasks/task-15764 - Watchlists-URLMonitor-extraction-and-diff-work-off-the-event-loop.md`

- [ ] **Step 1: Run the exact impacted test set**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py \
  Tests/Subscriptions/test_watchlist_content_kind_producer.py \
  Tests/Subscriptions/test_local_watchlists_service.py
```

Expected: all tests pass. Do not run the full repository suite; the user explicitly limited verification to touched/modified/impacted functionality.

- [ ] **Step 2: Run scoped static checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Subscriptions/monitoring_engine.py \
  Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py \
  Tests/Subscriptions/test_local_watchlists_service.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Subscriptions/monitoring_engine.py \
  Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py \
  Tests/Subscriptions/test_local_watchlists_service.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Self-review the final diff**

Confirm there is no custom executor, fan-out, database call inside the new plain `to_thread` boundaries, ordering change, threshold move, schema change, or unrelated edit. Confirm the helper segments each side exactly once and every public result field is unchanged.

- [ ] **Step 4: Update task hygiene**

Use Backlog CLI to check all acceptance criteria, add concise Implementation Notes with the impacted test evidence and ADR decision, and set TASK-15764 to Done only after all gates pass:

```bash
backlog task edit 15764 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --notes "Implemented standard-library worker offloads for URL HTML extraction, change percentage, and grouped significant-change details. Preserved sequential URL processing, thresholds, DB offload, persistence, and public results. ADR required: no. Verification: <insert exact impacted pytest and Ruff results>." \
  -s Done
```

- [ ] **Step 5: Commit closeout metadata**

```bash
git add 'backlog/tasks/task-15764 - Watchlists-URLMonitor-extraction-and-diff-work-off-the-event-loop.md'
git commit -m "chore(watchlists): complete URL monitor offload task"
```
