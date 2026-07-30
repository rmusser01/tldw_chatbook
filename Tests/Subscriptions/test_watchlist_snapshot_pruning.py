"""TASK-1393: `url_snapshots` must stop growing without bound.

No live code path ever deleted from `url_snapshots`. The only DELETE in the
repo is `baseline_manager._cleanup_old_baselines`, and that module has zero
importers (TASK-1360) -- so every significant change persisted a full row,
`raw_html` included, for ever. TASK-1362's default `change_threshold` of 0.0
means every real change writes one, and TASK-1361's per-URL baselines multiply
that by a source's URL count.

The fix prunes inside `URLMonitor._store_snapshot`'s existing transaction,
keyed per **(subscription, url)**. These tests drive the REAL producer through
the real service and the real DB wherever the property is end-to-end, and call
`_store_snapshot` / `check_url` directly where the assertion is sharper for it.

The harness is imported, not rebuilt: a hand-built snapshot row would pass
whether or not the live path prunes anything, which is the exact failure mode
`test_watchlist_content_kind_producer` exists to close.
"""

from __future__ import annotations

from typing import Any

import pytest

from Tests.Subscriptions.test_watchlist_content_kind_producer import (
    _check,
    _serve,
    _site_source,
    _stored_items,
)
from Tests.Subscriptions.test_watchlist_noise_not_volume import (
    _counts,
    _direct_check,
    _url_source,
)

pytestmark = pytest.mark.unit

_URL_A = "https://example.com/alpha"
_URL_B = "https://example.com/beta"


# --- helpers ---------------------------------------------------------------


def _page(body: str) -> str:
    return f"<html><body><p>{body}</p></body></html>"


def _serve_by_url(monkeypatch, bodies: dict[str, list[str]]) -> None:
    """Serve pages keyed by the URL fetched, not by global fetch order.

    `_serve` hands out one body per fetch in sequence, which is unambiguous for
    a single-URL source but couples two URLs' timelines on a `url_list`: the
    Nth body belongs to whichever URL happened to be fetched Nth. These tests
    need one URL to churn while another stands still across many runs, so they
    key on the URL and let each URL consume its own list (the last entry
    repeating once exhausted, exactly like `_serve`).

    Args:
        bodies: URL -> the page bodies to serve for it, in order.
    """
    remaining = {url: list(pages) for url, pages in bodies.items()}

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        from types import SimpleNamespace

        pages = remaining[url]
        page = pages.pop(0) if len(pages) > 1 else pages[0]
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "text/html"},
            text=page,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


def _snapshots(db, subscription_id: int, url: str) -> list[dict[str, Any]]:
    """Every surviving snapshot for one (subscription, url), newest first.

    Ordered by the SAME key the production baseline SELECT and the production
    prune both use, so "the newest N" means one thing across the test and the
    code under test.
    """
    rows = db.conn.execute(
        """
        SELECT id, url, extracted_content, content_hash, created_at
        FROM url_snapshots
        WHERE subscription_id = ? AND url = ?
        ORDER BY created_at DESC, id DESC
        """,
        (subscription_id, url),
    ).fetchall()
    return [dict(row) for row in rows]


def _all_snapshots(db, subscription_id: int) -> list[dict[str, Any]]:
    rows = db.conn.execute(
        "SELECT id, url, extracted_content FROM url_snapshots"
        " WHERE subscription_id = ? ORDER BY id ASC",
        (subscription_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _cap() -> int:
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _SNAPSHOTS_KEPT_PER_URL,
    )

    return _SNAPSHOTS_KEPT_PER_URL


async def _monitor_store(db, subscription_id: int, url: str, text: str) -> None:
    """One real `_store_snapshot` call, with a body distinguishable by text."""
    from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor

    await URLMonitor(db)._store_snapshot(
        subscription_id,
        url,
        {"text": text, "html": _page(text), "headers": {}},
        fingerprint="fp",
    )


# --- AC#1: the table stops growing ----------------------------------------


def test_the_cap_is_at_least_two_so_a_previous_snapshot_can_exist():
    """Pins the constant's floor rather than its exact value.

    One would keep only the live baseline and foreclose the spec'd
    `[previous snapshot]` affordance; the rationale for three is in the
    constant's own comment.
    """
    assert _cap() >= 2


@pytest.mark.asyncio
async def test_a_churning_url_keeps_exactly_the_newest_n_snapshots(monkeypatch):
    """AC#1, end to end through the real producer.

    N+3 real changes to one URL. Before this fix the table held one row per
    change plus the baseline and never shrank; now exactly `N` survive, and
    they are the `N` newest -- asserted by row IDENTITY (id and body), not by
    count, because a prune that kept the wrong N would pass a count assertion
    while destroying the baseline.
    """
    n = _cap()
    revisions = [f"Alpha service is at version {i}.0." for i in range(n + 4)]
    db, service, source_id = await _site_source(
        monkeypatch, [_page(body) for body in revisions], change_threshold=0.0
    )

    for _ in revisions:
        result = await _check(service, source_id)
        assert result["status"] == "completed"

    # Precondition: the run really did detect n+3 changes, so n+4 snapshots
    # would exist without pruning. Without this the test could pass on a
    # producer that stored nothing at all.
    assert len(_stored_items(db, source_id)) == n + 3, (
        "each revision after the baseline must have produced one change item"
    )

    survivors = _snapshots(db, source_id, "https://example.com/page")
    assert len(survivors) == n, (
        f"exactly {n} snapshots must survive; got {len(survivors)}"
    )

    kept_bodies = [row["extracted_content"] for row in survivors]
    for expected, actual in zip(reversed(revisions[-n:]), kept_bodies):
        assert expected in actual, (
            "the survivors must be the NEWEST n revisions, newest first; got "
            f"{kept_bodies!r}"
        )
    assert survivors == sorted(survivors, key=lambda r: r["id"], reverse=True), (
        "newest-first by the production ordering must also be newest-first by id"
    )
    for stale in revisions[: -n or None]:
        assert not any(stale in body for body in kept_bodies), (
            f"a superseded revision ({stale!r}) must not have survived"
        )


@pytest.mark.asyncio
async def test_store_snapshot_prunes_within_its_own_transaction(monkeypatch):
    """AC#1 at the write chokepoint, with no fetch in the way.

    Drives `_store_snapshot` directly `n + 5` times. Asserts the cap holds
    after EVERY write, not merely at the end: pruning that ran only
    occasionally (on a counter, say) would leave the table observably over the
    cap in between, and this is the only place a reader could see it.
    """
    n = _cap()
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")

    for i in range(n + 5):
        await _monitor_store(db, source_id, _URL_A, f"revision {i}")
        assert len(_snapshots(db, source_id, _URL_A)) == min(i + 1, n), (
            f"after write {i} the table must hold min(i+1, {n}) rows"
        )

    bodies = [row["extracted_content"] for row in _snapshots(db, source_id, _URL_A)]
    assert bodies == [f"revision {i}" for i in range(n + 4, n + 4 - n, -1)]


@pytest.mark.asyncio
async def test_a_pre_existing_backlog_collapses_on_the_very_next_write(monkeypatch):
    """Every database in the field already holds an unpruned backlog.

    Nothing ever deleted from `url_snapshots`, so a user who has been checking
    one URL for months arrives at this fix with hundreds of rows. The prune is
    unconditional rather than incremental -- it deletes everything past the cap
    in one statement, not one row per write -- so that backlog collapses on the
    first store for that URL and the fix is self-healing with no migration.

    The permanent suite otherwise only ever exercises one-row-over-cap (each
    write prunes exactly one), which would pass just as well for an
    incremental `DELETE ... LIMIT 1`. Fifty seeded rows is far enough past the
    cap to tell the two apart, and the survivors are asserted by identity.
    """
    n = _cap()
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")

    with db.transaction() as conn:
        for i in range(50):
            conn.execute(
                "INSERT INTO url_snapshots (subscription_id, url, content_hash,"
                " extracted_content) VALUES (?, ?, ?, ?)",
                (source_id, _URL_A, f"legacy-{i}", f"legacy {i}"),
            )
    seeded = _snapshots(db, source_id, _URL_A)
    assert len(seeded) == 50, "the precondition: a real backlog exists"

    await _monitor_store(db, source_id, _URL_A, "the first write after the fix")

    survivors = _snapshots(db, source_id, _URL_A)
    assert len(survivors) == n, (
        f"one store must collapse 51 rows to {n}, not shed a single row; got "
        f"{len(survivors)}"
    )
    expected = ["the first write after the fix"] + [
        f"legacy {49 - i}" for i in range(n - 1)
    ]
    assert [row["extracted_content"] for row in survivors] == expected, (
        "the survivors must be the newest by identity, newest first; got "
        f"{[row['extracted_content'] for row in survivors]!r}"
    )
    surviving_ids = {row["id"] for row in survivors}
    assert not surviving_ids & {row["id"] for row in seeded[n - 1 :]}, (
        "every seeded row past the cap must be gone -- by id, so a survivor "
        "list that merely reads right cannot pass"
    )


@pytest.mark.asyncio
async def test_shadow_mode_writes_nothing_and_therefore_prunes_nothing(monkeypatch):
    """A dry run must not delete the user's snapshots.

    `persist_snapshots=False` returns before the INSERT; the prune sits after
    that guard, so a shadow run cannot touch the table. Seeded well over the
    cap so a prune that had been placed BEFORE the guard would be visible.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor

    n = _cap()
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")
    for i in range(n + 4):
        await _monitor_store(db, source_id, _URL_A, f"revision {i}")

    # The cap already applied to those writes; add rows behind its back so the
    # table is genuinely over it when shadow mode runs.
    with db.transaction() as conn:
        for i in range(4):
            conn.execute(
                "INSERT INTO url_snapshots (subscription_id, url, content_hash,"
                " extracted_content) VALUES (?, ?, ?, ?)",
                (source_id, _URL_A, f"legacy-{i}", f"legacy {i}"),
            )
    before = len(_snapshots(db, source_id, _URL_A))
    assert before == n + 4, "the precondition: the table is over the cap"

    shadow = URLMonitor(db, persist_snapshots=False)
    await shadow._store_snapshot(
        source_id, _URL_A, {"text": "shadow", "html": "", "headers": {}}
    )

    after = _snapshots(db, source_id, _URL_A)
    assert len(after) == before, "shadow mode must neither write nor prune"
    assert not any(row["extracted_content"] == "shadow" for row in after)


# --- AC#2: never key the prune per subscription ---------------------------


@pytest.mark.asyncio
async def test_a_quiet_urls_baseline_survives_a_busy_siblings_churn(monkeypatch):
    """AC#2, the review-established constraint, on the real `url_list` arm.

    Two URLs, one `subscription_id`. A churns past the cap; B never changes, so
    B owns exactly one row -- its baseline -- and that row is among the OLDEST
    for the subscription. Pruning keyed per subscription would evict it, and
    B's next check would report `baseline_stored` again, for ever, reporting no
    change each time however much B eventually moved.

    The dispositions of EVERY run are what is asserted, not just the last one
    and not the row counts. Counting rows alone is not enough: under per-
    subscription pruning B is evicted and then immediately re-baselined by its
    own next check, so it still *has* a row at the end -- it just never reports
    a change again, which is the whole defect and is invisible to a count.

    Mutation (a) for this task: re-key the DELETE per subscription (drop the
    two `url` predicates) -> RED here, with B re-baselining.
    """
    n = _cap()
    db, service, source_id = await _url_source(
        monkeypatch,
        [_page("unused")],
        source_type="url_list",
        change_threshold=0.0,
        extraction_rules={"urls": [_URL_A, _URL_B]},
    )
    _serve_by_url(
        monkeypatch,
        {
            # One more revision than the loop consumes, so the FINAL check below
            # still moves A. That is what makes the final counts name B as the
            # unchanged URL rather than merely counting two quiet URLs.
            _URL_A: [_page(f"Alpha is at version {i}.0.") for i in range(n + 5)],
            _URL_B: [_page("Beta has not been touched in months.")],
        },
    )

    first = await _check(service, source_id)
    assert dict(first["stats"]["dispositions"]) == _counts(baseline=2), (
        "the precondition: both URLs start with a baseline of their own"
    )

    for run in range(1, n + 4):
        result = await _check(service, source_id)
        assert result["status"] == "completed"
        counts = dict(result["stats"]["dispositions"])
        assert counts == _counts(changed=1, unchanged=1), (
            f"run {run}: A changed and B did not, so B must report `unchanged` "
            f"against its own surviving baseline; got {counts!r}. A `baseline`/"
            "`rebaselined` here means A's prune evicted B's only snapshot -- "
            "which is what per-subscription keying does, on every rotation, "
            "for ever"
        )

    b_rows = _snapshots(db, source_id, _URL_B)
    assert len(b_rows) == 1, (
        "B never changed, so it owns exactly one snapshot -- its baseline"
    )
    assert "Beta has not been touched" in b_rows[0]["extracted_content"], (
        "B's surviving row must be B's own content, not a sibling URL's"
    )
    assert len(_snapshots(db, source_id, _URL_A)) == n, (
        "A is capped independently of B"
    )

    final = await _check(service, source_id)
    assert dict(final["stats"]["dispositions"]) == _counts(changed=1, unchanged=1), (
        "and still, after the table has been pruned many times over"
    )


@pytest.mark.asyncio
async def test_pruning_one_url_never_touches_another_urls_rows(monkeypatch):
    """AC#2 at the chokepoint: the DELETE's blast radius is one URL.

    Both URLs are pushed over the cap, so *both* prunes fire and each must stop
    at its own URL's rows. Written against `_store_snapshot` directly because
    it isolates the DELETE from every other reason a row might vanish.
    """
    n = _cap()
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")

    for i in range(n + 3):
        await _monitor_store(db, source_id, _URL_A, f"alpha {i}")
        await _monitor_store(db, source_id, _URL_B, f"beta {i}")

    a_rows = _snapshots(db, source_id, _URL_A)
    b_rows = _snapshots(db, source_id, _URL_B)
    assert len(a_rows) == n and len(b_rows) == n, (
        f"each URL is capped at {n} on its own; got {len(a_rows)} / {len(b_rows)}"
    )
    assert len(_all_snapshots(db, source_id)) == 2 * n, (
        "the cap is per URL, not per subscription -- a two-URL source holds 2n"
    )
    assert all("alpha" in r["extracted_content"] for r in a_rows)
    assert all("beta" in r["extracted_content"] for r in b_rows)


# --- AC#3: the [previous snapshot] affordance's data survives -------------


@pytest.mark.asyncio
async def test_the_second_newest_snapshot_survives_heavy_churn(monkeypatch):
    """AC#3: after churn, the URL still has a snapshot BEFORE its baseline.

    The design spec's Content-pane mockup promises the reader a
    `[previous snapshot]` affordance reading from `url_snapshots`. It is not
    built yet (no reference to it anywhere in `UI/`; filed separately), so the
    thing to protect is its data: the second-newest row per URL. Asserted by
    content, so a survivor set of "the newest row twice" could not pass.

    Mutation (b) for this task: `_SNAPSHOTS_KEPT_PER_URL = 1` -> RED.
    """
    revisions = [f"Alpha service is at version {i}.0." for i in range(_cap() + 5)]
    db, service, source_id = await _site_source(
        monkeypatch, [_page(body) for body in revisions], change_threshold=0.0
    )
    for _ in revisions:
        await _check(service, source_id)

    survivors = _snapshots(db, source_id, "https://example.com/page")
    assert len(survivors) >= 2, (
        "at least the baseline and the one before it must survive"
    )
    newest, previous = survivors[0], survivors[1]
    assert revisions[-1] in newest["extracted_content"]
    assert revisions[-2] in previous["extracted_content"], (
        "the second-newest snapshot is what `[previous snapshot]` will read; "
        f"got {previous['extracted_content']!r}"
    )
    assert newest["id"] != previous["id"]
    assert newest["extracted_content"] != previous["extracted_content"]


# --- the survivor set and the baseline SELECT must agree ------------------


@pytest.mark.asyncio
async def test_after_pruning_an_unchanged_page_still_reports_unchanged(monkeypatch):
    """The prune's survivors and the baseline SELECT are the same ordering.

    If they disagreed -- the DELETE keeping the OLDEST rows while the SELECT
    reads the newest, or either dropping the `id` tie-break -- the row the next
    check asks for would be gone, and an untouched page would come back as a
    re-baseline (or, worse, as a phantom change measured against ancient text).
    Runs the churn, then serves the SAME page again and asserts the real
    disposition.

    Mutation (c) for this task: invert the survivor `ORDER BY` to `ASC` -> RED
    (the kept rows are the oldest, so the check re-baselines or diffs against
    stale text).
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        DISPOSITION_UNCHANGED,
    )

    n = _cap()
    revisions = [f"Alpha service is at version {i}.0." for i in range(n + 4)]
    db, service, source_id = await _site_source(
        monkeypatch, [_page(body) for body in revisions], change_threshold=0.0
    )
    for _ in revisions:
        await _check(service, source_id)
    items_before = len(_stored_items(db, source_id))

    # `_serve` repeats its last page once exhausted, so this check refetches
    # the identical body the surviving baseline was captured from.
    result, disposition = await _direct_check(db, source_id)
    assert disposition["kind"] == DISPOSITION_UNCHANGED, (
        "the surviving baseline must be the newest snapshot, so an untouched "
        f"page is unchanged; got {disposition!r}"
    )
    assert result is None
    assert len(_stored_items(db, source_id)) == items_before, (
        "an unchanged check must produce no item"
    )
    assert len(_snapshots(db, source_id, "https://example.com/page")) == n, (
        "an unchanged check writes no snapshot, so the cap still holds"
    )


# --- TASK-1361's same-second tie window -----------------------------------


@pytest.mark.asyncio
async def test_two_snapshots_sharing_a_created_at_both_survive_under_the_cap(
    monkeypatch,
):
    """Under the cap, a `created_at` tie costs nothing.

    `url_snapshots.created_at` is a DATETIME defaulting to CURRENT_TIMESTAMP,
    one-second resolution, so two checks inside one second share it. Written
    directly (the TASK-1361 test pattern) because racing the real clock cannot
    force the tie reliably. Both rows are under the cap, so both must be kept:
    a prune keyed on `created_at` VALUES rather than on row identity would drop
    one of them.
    """
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")

    tied = "2026-07-30 00:00:00"
    with db.transaction() as conn:
        for body in ("tied first", "tied second"):
            conn.execute(
                "INSERT INTO url_snapshots (subscription_id, url, content_hash,"
                " extracted_content, created_at) VALUES (?, ?, ?, ?, ?)",
                (source_id, _URL_A, f"hash-{body}", body, tied),
            )

    assert _cap() >= 2, "the premise: two rows fit under the cap"
    await _monitor_store(db, source_id, _URL_A, "the newest")

    survivors = _snapshots(db, source_id, _URL_A)
    assert len(survivors) == min(3, _cap())
    bodies = [row["extracted_content"] for row in survivors]
    assert bodies[0] == "the newest"
    assert "tied second" in bodies, (
        "both tied rows are under the cap and must both survive"
    )


@pytest.mark.asyncio
async def test_the_tie_break_decides_which_tied_row_is_pruned(monkeypatch):
    """Over the cap, the `id` tie-break makes the outcome deterministic.

    `cap + 1` rows share one `created_at`, so `created_at DESC` alone leaves
    SQLite free to keep any subset -- the same ambiguity TASK-1361 closed on
    the read side, here on the delete side. With `id DESC` the survivors are
    exactly the highest `cap` ids, i.e. true insertion order, and the row the
    baseline SELECT would pick is provably among them.
    """
    n = _cap()
    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    db.conn.execute("DELETE FROM url_snapshots")

    tied = "2026-07-30 00:00:00"
    with db.transaction() as conn:
        for i in range(n + 1):
            conn.execute(
                "INSERT INTO url_snapshots (subscription_id, url, content_hash,"
                " extracted_content, created_at) VALUES (?, ?, ?, ?, ?)",
                (source_id, _URL_A, f"hash-{i}", f"tied {i}", tied),
            )
    seeded = [row["id"] for row in _all_snapshots(db, source_id)]
    assert len(seeded) == n + 1, "the precondition: one row over the cap"

    # A write at the same tied timestamp: now n+2 rows share `created_at`, and
    # only `id DESC` can order them.
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO url_snapshots (subscription_id, url, content_hash,"
            " extracted_content, created_at) VALUES (?, ?, ?, ?, ?)",
            (source_id, _URL_A, "hash-final", "tied final", tied),
        )
        # Reproduce the production prune against this all-tied set by driving
        # the real path: `_store_snapshot` would stamp a fresh `created_at`, so
        # the DELETE is exercised through a real monitor call below instead.
    from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor

    monitor = URLMonitor(db)
    await monitor._store_snapshot(
        source_id, _URL_A, {"text": "newest", "html": "", "headers": {}}
    )

    survivors = _snapshots(db, source_id, _URL_A)
    assert len(survivors) == n
    ids = [row["id"] for row in survivors]
    assert ids == sorted(ids, reverse=True)
    assert survivors[0]["extracted_content"] == "newest", (
        "the newest row is always survivor #0 -- the invariant the baseline "
        "SELECT depends on"
    )
    assert "tied final" == survivors[1]["extracted_content"], (
        "among rows sharing one `created_at`, the highest `id` survives first "
        f"-- the tie-break is missing or reversed; got {[r['extracted_content'] for r in survivors]!r}"
    )
    assert all(row["id"] > min(seeded) for row in survivors), (
        "the oldest tied rows are the ones pruned"
    )
