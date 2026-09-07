# TASK-1393 — `url_snapshots` grows without bound, and no live path prunes it

Branch `fix/task-1393-snapshot-pruning`, off `origin/dev` at `a39b2fc5f`.

## What was wrong

`url_snapshots` had no live deleter. The only `DELETE` in the repo is
`baseline_manager._cleanup_old_baselines` (`tldw_chatbook/Subscriptions/baseline_manager.py:669`),
and that module has zero importers (TASK-1360). Meanwhile `URLMonitor._store_snapshot` writes a
full row — `extracted_content` **and** `raw_html` — on every baseline, every re-baseline and every
significant change.

Two recent changes made that newly load-bearing:

* TASK-1362 set the default `change_threshold` to `0.0`, so **every** real change persists a row.
* TASK-1361 gave each URL of a `url_list`/`sitemap` source its own baseline, multiplying the row
  count by the source's URL count.

Steady state was monotonic growth of raw HTML in the user's private database.

## The fix

One place: `URLMonitor._store_snapshot`
(`tldw_chatbook/Subscriptions/monitoring_engine.py`), a `DELETE` immediately after the `INSERT`
**inside the same `db.transaction()`**, so the two share one commit boundary
(`DB/Subscriptions_DB.py:899-907`) and the table is never observably over the cap.

```sql
DELETE FROM url_snapshots
WHERE subscription_id = ? AND url = ?
  AND id NOT IN (
      SELECT id FROM url_snapshots
      WHERE subscription_id = ? AND url = ?
      ORDER BY created_at DESC, id DESC
      LIMIT ?
  )
```

Three properties, each deliberate:

**Keyed per `(subscription_id, url)`, never per subscription.** This is the review-established
constraint from TASK-1362's whole-branch review, and it is on *both* halves of the statement. A
`url_list`/`sitemap` source gives every one of its URLs one `subscription_id`; per-subscription
pruning lets a busy URL's snapshots evict a quiet URL's only baseline, and that URL then
re-baselines on its next check — reporting no change — for ever. That is exactly the defect in the
orphaned `baseline_manager` pruning.

**Survivors ordered `created_at DESC, id DESC`** — the *same* ordering `check_url`'s baseline
`SELECT` uses (TASK-1361's tie-break). Because the two orderings are identical, the row the next
check will read is by construction the first survivor and can never be pruned, whatever the cap.
The existing index `idx_url_snapshots_lookup(subscription_id, url, created_at)` covers the
subquery.

The invariant is stated **at both ends**, under the greppable token **`TASK-1393 ordering pact`**:
each site names the other by file-relative location and says what breaks if they diverge. It was
one-directional at first — the DELETE named the SELECT, the SELECT said nothing — which left the
pairing invisible to anyone editing the read side, the direction from which the damage is silent.

**After the shadow-mode guard.** `persist_snapshots=False` returns before the `INSERT`, so a dry
run neither writes nor deletes. Pinned by a test that seeds the table *over* the cap first, so a
prune placed before the guard would be visible.

**Crash safety is stronger than "benign".** Because the INSERT and the DELETE share one commit
boundary, a crash before that commit rolls back *both*: the "row inserted, prune not yet run" state
is not merely harmless, it is unrepresentable. Worth stating by contrast — the neighbouring
TASK-1362 fingerprint migration (`DB/Subscriptions_DB.py:626-650`) had to take an explicit
`BEGIN IMMEDIATE`, and is documented there as an exemption, precisely because its partial state
*was* representable and unrepairable.

`N = _SNAPSHOTS_KEPT_PER_URL = 3`, a module constant carrying its rationale:

1. the live baseline — the row the next `check_url` reads;
2. the previous snapshot. The design spec's Content-pane mockup
   (`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md:396`) promises the
   reader a `[previous snapshot]` affordance reading from `url_snapshots`. **It is not built** —
   verified by repo-wide grep, there is no reference to it anywhere in `UI/`; it is being filed
   separately. Pruning must not foreclose it, so the second-newest row per URL survives (AC#3);
3. one row of slack for TASK-1361's same-second tie window (`created_at` is a one-second-resolution
   DATETIME).

**No config surface — deliberate YAGNI.** There is no user question that this number answers, and a
knob would be one more setting to migrate, validate, document and test for a bound nobody has asked
to move. `baseline_manager.retention_days` is orphaned code and stays untouched (TASK-1360).

**Existing over-sized databases self-heal**: the prune is unconditional, so the first write for a
URL collapses whatever backlog that URL had accumulated down to `N`.

## Tests

New: `Tests/Subscriptions/test_watchlist_snapshot_pruning.py`, 10 tests, all
`pytest.mark.unit`. Real bodies, real DB, real service. The end-to-end ones drive the REAL producer
through the imported `_site_source` / `_serve` / `_check` / `_url_source` / `_direct_check`
harness; the sharper ones call `_store_snapshot` and `check_url` directly. One local helper,
`_serve_by_url`, keys served pages on the URL fetched rather than on global fetch order — `_serve`
couples two URLs' timelines on a `url_list`, and AC#2 needs one URL to churn while another stands
still across many runs.

| Test | Pins |
| --- | --- |
| `test_the_cap_is_at_least_two_so_a_previous_snapshot_can_exist` | the constant's floor |
| `test_a_churning_url_keeps_exactly_the_newest_n_snapshots` | **AC#1** end to end: N+3 changes -> exactly N rows, asserted by row *identity* (id + body), plus that no superseded revision survived |
| `test_store_snapshot_prunes_within_its_own_transaction` | the cap holds after *every* write, not only at the end |
| `test_a_pre_existing_backlog_collapses_on_the_very_next_write` | 50 seeded rows -> `N` in ONE store: the fix is self-healing for field databases, and the prune is unconditional rather than incremental |
| `test_shadow_mode_writes_nothing_and_therefore_prunes_nothing` | the guard ordering |
| `test_a_quiet_urls_baseline_survives_a_busy_siblings_churn` | **AC#2** on the real `url_list` arm |
| `test_pruning_one_url_never_touches_another_urls_rows` | **AC#2** at the chokepoint: the DELETE's blast radius is one URL |
| `test_the_second_newest_snapshot_survives_heavy_churn` | **AC#3** |
| `test_after_pruning_an_unchanged_page_still_reports_unchanged` | survivor set and baseline SELECT agree |
| `test_two_snapshots_sharing_a_created_at_both_survive_under_the_cap` | same-second tie, under the cap |
| `test_the_tie_break_decides_which_tied_row_is_pruned` | same-second tie, over the cap |

**AC#2 needed the dispositions, not a row count.** First draft asserted "B ends with exactly one
row". That assertion *passes* under per-subscription pruning: B is evicted and then immediately
re-baselined by its own next check, so it still has a row — it just never reports a change again.
The test now asserts the full disposition-count dict of **every** run, so the failure names the
symptom.

## Mutation testing

| # | Mutation | Result |
| --- | --- | --- |
| a | drop both `url` predicates (prune per subscription) | **RED** — 2 failed / 8 passed. `test_a_quiet_urls_baseline_survives_a_busy_siblings_churn` fails at run 3 with `{'baseline': 1, 'unchanged': 0}`: B re-baselining, the exact defect. `test_pruning_one_url_never_touches_another_urls_rows` also RED. |
| b | `_SNAPSHOTS_KEPT_PER_URL = 1` | **RED** — 4 failed / 6 passed, incl. the AC#3 test ("at least the baseline and the one before it must survive") and both tie tests. |
| c | invert the survivor `ORDER BY` to `ASC` | **RED** — 5 failed / 5 passed. Notably `test_after_pruning_an_unchanged_page_still_reports_unchanged` reports `kind='changed'` — a phantom change measured against the *oldest* kept text, which is the real user-visible harm. |

All three mutations were reverted; the final tree is green.

## Review round (fix round after the first review)

The review verified all three ACs empirically — including dropping each single `url` predicate
separately, both directions red — and confirmed the index already covers all three hot queries.
Three items came back:

1. **(Important) The ordering pact was one-directional.** Fixed at both ends, above.
2. **(Minor) The 50-row collapse was only ever a throwaway probe.** Promoted to the permanent
   suite. This was a real gap, not bookkeeping: the suite otherwise only exercised
   one-row-over-cap, where each write prunes exactly one row. Mutation **(d)** — replace the
   set-based DELETE with an incremental one that sheds the single oldest row past the cap
   (`WHERE id = (SELECT id … LIMIT 1 OFFSET N)`) — leaves the old suite **9 passed / 2 failed**,
   and `test_store_snapshot_prunes_within_its_own_transaction` is among the passers. Only the new
   test (`assert 50 == 3`) and the tie-break test catch it. Reverted.
3. **(Minor) The crash-safety claim was imprecise in the safe direction.** Corrected in the code
   comment and above: the partial state is unrepresentable, not merely benign.

## Suite runs

```
# first round
.venv/bin/python -m pytest Tests/Subscriptions/test_watchlist_snapshot_pruning.py -p no:randomly
  -> 10 passed in 0.90s
.venv/bin/python -m pytest Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/
  -> 601 passed in 193.82s (0:03:13)

# fix round
.venv/bin/python -m pytest Tests/Subscriptions/test_watchlist_snapshot_pruning.py -p no:randomly
  -> 11 passed in 1.30s
.venv/bin/python -m pytest Tests/Subscriptions/
  -> 190 passed in 42.79s
```

## Files

* `tldw_chatbook/Subscriptions/monitoring_engine.py` — `_SNAPSHOTS_KEPT_PER_URL` constant, the
  prune in `_store_snapshot`, and the `TASK-1393 ordering pact` comment at both ends.
* `Tests/Subscriptions/test_watchlist_snapshot_pruning.py` — new, 11 tests.
