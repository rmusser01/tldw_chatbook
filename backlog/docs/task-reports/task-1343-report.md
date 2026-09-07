# TASK-1343 — `content_kind` / `content_format` / `diff_summary` now have producers

Branch `feat/task-1343-content-kind-producer`, worktree `/private/tmp/tldw-1343`.

## What was built

### 1. The change path emits a real diff, not the full page

`URLMonitor.check_url` (`tldw_chatbook/Subscriptions/monitoring_engine.py`) previously put
`current_content["text"]` — the **entire new page** — into the item's `content`, and wrote no
`content_kind`. It now writes:

| field | before | after |
|---|---|---|
| `content` | the whole new page text | a bounded unified-diff body |
| `content_kind` | *(absent)* | `"change"` |
| `content_format` | *(absent)* | `"diff"` |
| `diff_summary` | *(absent — no producer anywhere)* | `"2 line(s) added, 1 removed"` |
| `change_type` | hardcoded `"content"` | `"new"` / `"removed"` / `"content"`, derived |
| `change_percentage` | 0.0–1.0 ratio | the same value × 100 (see §6) |

The full page continues to live in `url_snapshots` — `_store_snapshot` is still called on the very
next line — so nothing was lost by storing the diff instead. A test asserts the snapshot still
holds the whole new text.

### 2. Format choice: a unified-diff body, headers stripped, sentence-segmented

`render_change` colours a line by its leading `+`/`-` (`content_pane.py:190`), so the body is a
`difflib.unified_diff` body with `n=1` context. The `---`/`+++` **file headers are dropped**:
they begin with `-` and `+`, so the renderer would paint them red and green as though the header
itself were the change. `@@` hunk headers are kept — they carry position and colour as ordinary
text.

**The load-bearing finding, and the reason a naive diff would have been useless.**
`ContentExtractor.extract_text_from_html` ends with

```python
text = " ".join(chunk for chunk in chunks if chunk)
```

so **the extracted text of an entire page is ONE line, with no newlines in it at all.** A
line-based diff of two such snapshots is therefore always exactly

```
-<the entire old page>
+<the entire new page>
```

— the full text twice: simultaneously the least readable and the largest possible thing to store.
So both sides are re-segmented before diffing (`_segment_for_diff`):

* real line breaks when the text has any (`extraction_method` other than `full`/`auto`);
* otherwise sentence boundaries, `(?<=[.!?])\s+`. Sentences stay aligned under a local edit;
  fixed-width chunking does not (inserting one word early shifts every boundary and makes every
  chunk differ).
* any segment over **110 characters** is word-wrapped, so no emitted diff line is wider than the
  narrow reader pane. A test asserts this for every line.

The result, from a real check driven through the real producer (pinned in full by
`test_the_whole_stored_diff_body_is_exactly_this`):

```
@@ -1,2 +1,3 @@
 Anthropic status All systems operational.
-Latest release: Opus 4.1 is available.
+Latest release: Opus 4.5 is available.
+Scheduled maintenance on Friday at 02:00 UTC.
```

Four short lines. Readable in a nine-row pane.

### 3. Size cap: **400 lines / 20,000 characters, whichever is hit first**

Why those numbers: the body goes into a `TEXT` column *and* into a pane about nine rows tall, and
the page it was computed from is already archived in `url_snapshots` — so the diff is a summary,
not an archive, and losing its tail loses nothing recoverable. 400 lines is ~40× the pane height
and far more than a reader scrolls; 20,000 characters keeps the item row small next to the
snapshot.

Truncation is stated **in the content**, as the one line allowed past the cap:

```
[diff truncated: showing the first 400 of 1800 diff lines (cap 400 lines / 20000 characters).
This is a partial view of the change; the full page is in this source's snapshot history.]
```

It starts with `[`, not `+`/`-`, so the renderer does not colour it as a change — a test asserts
that. `diff_summary` counts the **whole** diff, not the retained slice, so the headline stays true
when the body is cut.

### 4. `diff_summary`

`"N line(s) added, M removed"` — one line, which is what `render_change`'s `·`-joined headline
needs. A real change renders as e.g. `25% changed · content · 2 line(s) added, 1 removed`.

### 5. `change_type` — improved, not left hardcoded

It was the literal `"content"` for every change ever detected. `classify_change_type` now returns:

* `"new"` — the previous snapshot had no text and the new one does;
* `"removed"` — the page's text disappeared entirely (previously reported to the user as an
  ordinary content edit);
* `"content"` — otherwise.

Those are the only three distinctions two text snapshots can honestly support.
`baseline_manager.ChangeReport`'s richer vocabulary also has `'structural'` and `'semantic'`, but
those need DOM-shape and embedding analysis that `check_url` does not do, so claiming them would
be a guess. **`baseline_manager.py` was not touched** (TASK-1360).

### 6. Feed and API paths → `("article", "text")`

* `FeedMonitor._parse_rss_item`, `_parse_atom_entry`, `_parse_json_feed`
* `LocalWatchlistsService._normalize_api_item`

`"text"` is the honest format, not `"markdown"`: RSS `description`, `atom:content`/`atom:summary`,
JSON Feed `content_html`/`content_text` and an API's mapped JSON field all arrive as the
publisher's plain text or HTML, and **nothing on any of these paths converts anything to
markdown**. Claiming `"markdown"` would make `render_article` hand publisher HTML to a CommonMark
parser. The API path is written unconditionally, even when the source supplied no body: the kind
is still `article` (`render_article` then explains the missing body), and both values are non-`None`
so they survive `_normalize_api_item`'s `if value is not None` filter.

### 7. The vocabulary is named once

`CONTENT_KIND_ARTICLE` / `CONTENT_KIND_CHANGE` / `CONTENT_FORMAT_TEXT` /
`CONTENT_FORMAT_MARKDOWN` / `CONTENT_FORMAT_DIFF` now live in `item_persist.py`, which already
owned `_VALID_PAIRINGS` (now built from them), and both producers import from there. Reason: an
invalid pairing **raises** out of `persist_subscription_item`, `execute_run` catches it into
`record_run_failure`, and **every item the run collected is dropped**. Mutation 6 below
demonstrates exactly that. `item_persist` has no heavy imports, so this is safe at module scope
and does not disturb `monitoring_engine`'s deliberately lazy import in
`_default_run_executor`.

### 8. The stale comment (point 6 of the brief)

`content_pane.py:160-161` claimed `change_percentage` "is always written as a Python float by
`baseline_manager.py`/`monitoring_engine.py`". It now names the single real producer,
`monitoring_engine.URLMonitor.check_url`, states the 0–100 scale, and records that
`baseline_manager` writes nothing because nothing imports it (TASK-1360).

## Findings

**F1 — `change_percentage` was stored as a 0–1 ratio and printed as a percentage.**
`calculate_change_percentage` returns 0.0–1.0. `render_change` prints
`f"{float(pct):.0f}% changed"`. The column is named `change_percentage`. Every fixture in
`Tests/UI/test_watchlists_content_pane.py` passes 12.0 and asserts `"12"` and `"%"`. So the moment
this task made `render_change` reachable, **a real 35% change would have displayed as "0% changed"**
and a total rewrite as "1% changed".

I scaled it at the point the value is handed to the reader (`change_percentage * 100.0` in the
`change_info` dict), and left the threshold comparison on the ratio, where
`change_threshold`'s 0.1 default and `config.py:3609`'s `change_threshold = 0.05  # 5% change
threshold` both live. This is the one change I made that the AC does not name — it is one line
and a comment, and it is reverted by deleting `* 100.0`. Consequence to be aware of: any
`subscription_items` rows written before this branch hold ratios and will render as `0% changed`.
Those rows have never been displayed by anything (the renderer was unreachable), and I did not
migrate them.

**F2 — `check_url`'s "previous snapshot" query orders by a one-second-resolution column.**
`monitoring_engine.py` selects the previous snapshot with
`ORDER BY created_at DESC LIMIT 1`, and `url_snapshots.created_at` has one-second resolution. My
first draft of the diff test failed for exactly this reason: two snapshots written in the same
second, and the query returned the *older* one. In production this means two checks inside one
second can compare the new page against a stale baseline and report a change that already
happened. `ORDER BY id DESC` would fix it. Out of scope here — not filed; my test uses
`ORDER BY id DESC` and says why.

**F3 — the whitespace-only-change hole, closed.** The content hash is taken over the raw extracted
text while segmentation trims and normalizes whitespace, so a whitespace-only or markup-only
change hashes differently and diffs to **nothing**. An empty `content` would make `render_change`
print "no body captured for this item — re-check this source to fetch it": a claim that nothing
was captured, when it was captured and it matched. `build_change_diff` returns an explicit notice
instead.

**F4 — a second live persistence path exists and is now covered.**
`Scheduling/scheduler/handlers/watchlist_check_handler.py` calls the same `check_feed`/`check_url`
and persists via `record_check_result(items=...)` → `_add_subscription_item` →
`persist_subscription_item`. That is the *scheduled* path, i.e. the one that actually runs
unattended. Both paths carry the new fields because the producers changed, not the persisters.
`Tests/Scheduling/` is green (189 passed).

## Verification

```
Tests/Subscriptions/                12 new, 145 passed
Tests/Watchlists/                   189 passed
Tests/DB/ -k subscription           47 passed, 563 deselected
Tests/UI/ -k watchlist              232 passed, 3 failed  (all three pre-existing)
Tests/Scheduling/                   189 passed  (the scheduled persist path, F4)
```

The three `Tests/UI` failures are the known baseline ones named in the brief: two tree-chevron
assertions in `test_destination_visual_parity_correction.py` and the order-dependent
`Select`/`Input` mount race in `test_watchlists_source_create_form.py` (TASK-1345). No new
failures.

New file: `Tests/Subscriptions/test_watchlist_content_kind_producer.py`, 12 tests. All of the
end-to-end ones drive the **real** producers (`URLMonitor.check_url`, `FeedMonitor.check_feed`,
`_normalize_api_item`) through the real service run, the real `persist_subscription_item`, the
real `normalize_watchlist_item`, and the real `render_for` — plus a real `rich.console.Console`,
so ANSI colour is read back rather than assumed. Hand-built item dicts would pass whether or not
any producer writes anything, which is exactly how this shipped: the existing renderer tests are
thorough and every one of their fixtures sets `content_kind` itself.

### Mutation outcomes

| # | Mutation | Result |
|---|---|---|
| 1 | delete `"content_kind": CONTENT_KIND_CHANGE` from `check_url` | **7 failed** |
| 2 | `content` back to `current_content["text"]` | **3 failed** |
| 3 | delete the RSS `content_kind`/`content_format` | **2 failed** |
| 4 | never append the truncation notice (`if False:`) | **1 failed** |
| 5 | `classify_change_type` returns `"content"` always | **1 failed** |
| 6 | change path emits the invalid `("change", "text")` | **7 failed**, run status `failed` |
| 7 | drop the `* 100.0` scaling | **1 failed** |
| 8 | delete the API path's `content_kind`/`content_format` | **2 failed** |
| 9 | typo a literal: `"articl"` instead of `CONTENT_KIND_ARTICLE` | **1 failed** (the AST vocabulary guard) |

Every assignment this task adds is load-bearing. After each mutation the source was diffed back
against a pre-mutation copy and confirmed byte-identical before continuing.

Mutation 6 is worth reading as the production consequence, not just a red test: the run's status
became `failed` and every item it had collected was dropped. That is why the vocabulary is named
constants imported from the module that owns the rule.

---

# Fix round 1

All five points addressed. Baseline before this round: `ff4c10c50`.

## 1. Important — the header filter deleted real change lines (fixed)

Confirmed and reproduced exactly as reported. `line.startswith(("---", "+++"))` cannot distinguish
a file header from content, because a **removed** segment beginning `--` becomes `---…` and an
**added** one beginning `++` becomes `+++…`. A page dropping a literal `--- Deprecated notice`
banner therefore persisted a change whose body showed nothing removed and whose headline read
`0 line(s) added, 0 removed`. The reviewer is right that this is the worst class of the five: the
stored record misrepresented the change, and nothing downstream could tell.

Now dropped **positionally**: `_HEADER_LINES = 2`, applied as `emitted[_HEADER_LINES:]` on the
materialized generator. `difflib.unified_diff` yields `--- <fromfile>` and `+++ <tofile>` together
immediately before the first hunk, or yields nothing at all, so they are always exactly positions 0
and 1 — there is no case where slicing removes content or leaves a header behind.

`test_a_removed_line_that_looks_like_a_diff_header_is_not_deleted` covers both halves (a `--`
removal and a `++` addition), asserts the summary counts are right (`0 added, 1 removed` /
`1 added, 0 removed`), and asserts the body still starts at a `@@` hunk so the real headers are
genuinely gone.

## 2. Important — the truncation notice was unreachable (fixed)

Correct, and a good catch: as line 401 of 401 in a pane about nine rows tall, the notice existed
only for someone who scrolled a diff they did not know had been cut. **Both** remedies applied:

* the notice is now `kept.insert(0, …)` — the **first** line of the body;
* `diff_summary` gains `" (diff truncated)"`, so it appears in `render_change`'s headline, which
  needs no scrolling at all.

The `[`-prefix is unchanged. The test now asserts `"diff truncated" in lines[0]`, that line 0 is
not `+`/`-` prefixed, that the stored `diff_summary` ends with `(diff truncated)`, and — through
the real renderer — that the word "truncated" appears within the **first four rendered rows**.

## 3. Important — rule scope: page-scoped behaviour preserved (fixed)

Confirmed: `_apply_filters_and_alerts` runs before persistence on the raw item, and both services
built their haystack from `item["content"]`, so this branch had silently narrowed every site rule
from "matches anywhere on the page" to "matches a changed segment plus one line of `n=1` context".
Shipping that as a side effect would have looked, to a user, like an alert that had worked for
months just stopping.

Fixed by separating the two concerns rather than choosing between them. `URLMonitor.check_url` now
also carries the full page text as `rule_match_text`, and a new module
`tldw_chatbook/Subscriptions/watchlist_rule_matching.py` owns the single haystack builder that
**both** services now call:

* the body used for matching is `rule_match_text` when present, else `content` (feed and API items
  never set it — their `content` *is* the captured body, so nothing changes for them);
* it **replaces** `content` rather than being appended to it. Appending would have introduced the
  mirror-image bug: text that the change *removed* is in the diff but is no longer on the page, and
  it must not start matching. A test pins that (`"opus 4.1" not in haystack`);
* `rule_match_text` is deliberately **not** a persisted column — a test asserts it is absent from
  `subscription_items` — because that would put a second copy of every page back in the item row
  and undo the point of storing a diff. The full text is already durable in `url_snapshots`.

The two services previously each held their own copy of the same key tuple, which is precisely how
this drifted; the shared module removes that. Three tests cover it: an alert on unchanged page text
still firing (with an explicit precondition that the phrase is *absent* from the stored diff, so it
cannot pass via the context line), an **exclude** filter on unchanged page text still excluding
(the destructive half — a narrowed filter admits items the user told the app to drop), and the
helper in isolation.

**My position, asked for explicitly.** Page-scoped is right as the default and I would not change
it. For *filters* it is not even a close call: include/exclude classifies the item, a site item
represents the page, and a narrowed exclude filter silently admits content. For *content alerts*
there is a real argument that "tell me when this phrase appears" is the more useful semantic — but
that is `changed_to`/`appeared` semantics, which needs to be a **per-rule opt-in with its own UI
affordance** (and probably a distinction between "appeared" and "disappeared", both of which the
diff can support and neither of which a keyword match expresses). It is a feature, not a default,
and it should not arrive as a consequence of a storage change. Worth filing; not worth shipping
silently.

While writing these tests I hit a related detail worth recording: a *small* edit to a *long* page
does not clear the default `change_threshold` of 0.1 at all — `calculate_change_percentage` is
character-level over the whole page, so the filler text alone suppressed the item. The helper now
takes an explicit `change_threshold`, and it is a reminder that the threshold is a whole-page
similarity measure, not a per-region one.

## 4. Minor — `_segment_for_diff` docstring (fixed)

It cited `extraction_method` (full/auto), which the function never reads. It now describes the
actual condition — whether the text already contains newlines — and notes that the extraction
method is a *consequence* of that, not the switch. The module comment above `_SENTENCE_BOUNDARY`
was corrected the same way.

## 5. Minor — mutation table row 9 was wrong (corrected)

Row 9 said "1 failed". Re-run on the full file: **3 failed**
(`test_a_feed_item_is_stored_as_an_article_with_a_legal_format`,
`test_no_producer_emits_a_pairing_persistence_would_reject`,
`test_every_content_kind_literal_in_the_package_is_from_the_vocabulary`). The round-1 number came
from a run I had narrowed with `-k "vocabulary"`, which deselected the other eleven tests — the
number was real for that command and useless as a table row. All mutation runs in this round were
made on the **whole file** with no `-k`, so every row is reproducible as written.

## Round-1 verification

```
Tests/Subscriptions/ + Tests/Scheduling/   339 passed  (17 in the new file, up from 12)
Tests/Watchlists/                          189 passed
Tests/DB/ -k subscription                   47 passed, 563 deselected
Tests/UI/ -k watchlist                     232 passed, 3 failed  (the same three as before)
```

The three `Tests/UI` failures are unchanged in identity and count from the pre-round-1 run: two
tree-chevron assertions in `test_destination_visual_parity_correction.py` and the order-dependent
`Select`/`Input` mount race in `test_watchlists_source_create_form.py` (TASK-1345).

### Round-1 mutation outcomes

| Mutation | Result |
|---|---|
| A — restore the pattern-match header filter | **1 failed** (`test_a_removed_line_that_looks_like_a_diff_header_is_not_deleted`) |
| B — truncation notice back to last line, no summary suffix | **1 failed** (`test_an_oversized_change_is_truncated_and_says_so`) |
| C — haystack body back to `content` (the diff) | **3 failed** (both rule-scope tests + the helper test) |
| D — filter service drifts back to its own local haystack | **1 failed** (the exclude-filter test) |
| 9 (re-run, whole file) — typo `"articl"` literal | **3 failed**, correcting the round-1 table |

Each source file was restored from a pre-mutation copy and `diff`-verified byte-identical before
the next mutation.

### Files changed in this round

* `tldw_chatbook/Subscriptions/monitoring_engine.py` — positional header slice, notice-first +
  summary suffix, docstring/comment corrections, `rule_match_text` on the change item.
* `tldw_chatbook/Subscriptions/watchlist_rule_matching.py` — **new**, the shared haystack.
* `tldw_chatbook/Subscriptions/watchlist_filter_service.py`,
  `watchlist_content_alert_service.py` — both call the shared haystack.
* `Tests/Subscriptions/test_watchlist_content_kind_producer.py` — 12 → 17 tests.

---

# Fix round 2 (PR #1092 / Qodo)

Baseline before this round: `3c3fec30e`.

## 1. Bug — the diff bound protected storage but not memory (fixed)

Correct, and the mechanism is exactly as described: `emitted = list(unified_diff(...))` bounded what
was *stored* while leaving peak allocation proportional to the whole diff, on a path that runs
inside a scheduled fetch over pages the egress layer admits up to `MAX_FETCH_BYTES_PAGE` (10 MB).

`build_change_diff` now consumes the generator **once**, in a single pass:

* the first `_HEADER_LINES` yielded items are skipped by index (the positional header rule from
  round 1 is unchanged, just applied during iteration instead of by slicing);
* `total_lines`, `added` and `removed` accumulate as lines go past;
* `kept` grows only while both caps hold; when a cap is hit, a `truncated` flag is set and
  iteration **continues** so the counters keep describing the whole change rather than the retained
  slice;
* the post-processing (no-textual-change notice, notice-first, summary suffix) is unchanged, and
  `total` in the notice now comes from `total_lines`.

Verified output-preserving: the same input through the shipped code and through a materialising
stand-in produces byte-identical body and summary (asserted in the test, not just observed).

**Measured, differentially, in-process** (4,000 segments per side → 8,001 diff lines):

| | peak allocation |
|---|---|
| streaming (shipped) | 1,289 KiB |
| `list(...)` materialised | 1,945 KiB |
| ratio | **1.51×**, stable at 1.50–1.55× across 4k/12k/20k segments and repeat runs |

### What the tests do and do not prove

Two tests, deliberately separated because they prove different things.

`test_a_diff_far_larger_than_the_cap_is_bounded_with_accurate_counts` — the **behavioural** half.
On a diff twenty times the line cap it proves: the body is exactly `_MAX_DIFF_LINES + 1` lines, the
char bound holds, the summary reports all 4,000 additions and 4,000 removals, the notice's own
total equals `added + removed + 1` (the single `@@` header these disjoint inputs produce), and the
retained body contains **far fewer** `+` lines than the count reports — which is what shows the
counters cannot have come from the retained slice, i.e. that iteration really continued past the
cap. **It proves nothing about memory**; it passes identically under the materialising mutation.

`test_the_diff_generator_is_not_materialised` — the **memory** half, and the honest limits of it:

* it is a **differential** measurement (shipped vs a materialising stand-in, same input, same
  process), not an absolute budget — an absolute threshold would be a machine-specific magic
  number;
* it asserts `peak_streamed < peak_listed * 0.85` against a measured 0.65, deliberately wide so it
  does not become a flaky allocation assertion;
* it does **not** prove "peak memory is proportional to the caps, not the input", and I want to be
  explicit that the stronger claim is **not achievable here**: `_segment_for_diff` must build both
  segment lists in full, because `difflib.SequenceMatcher` needs random access to both sequences,
  and its internal `b2j` index is O(len(b)) as well. Those terms remain proportional to the page.
  What the change removes is the diff-output term *on top* of them — about a third of peak — and
  what it guarantees is that the only term scaling with the **diff** is now the one bounded by
  `_MAX_DIFF_LINES`/`_MAX_DIFF_CHARS`. Making the whole function O(caps) would mean not using
  `difflib`, which is a different task.

## 2. Rule violation — file-backed test DB (fixed)

All four `SubscriptionsDB(tmp_path / "subscriptions.db", "test")` sites are now
`SubscriptionsDB(":memory:", "test")`. The whole file passes (19/19), so nothing depended on the DB
being on disk. `tmp_path` became unused in twelve signatures and the `_site_source` helper, and was
removed rather than left as a dead parameter — which, incidentally, disposes of most of item 3.

Worth recording *why* `:memory:` is safe here rather than leaving it as a compliance change:
`SubscriptionsDB` keeps a **thread-local** connection and builds the schema on the constructing
thread's connection only — `SubscriptionsDB._initialize_schema` documents this, because every
`sqlite3.connect(":memory:")` opens a fresh empty database, so an in-memory instance touched from a
second thread finds zero tables. These tests are single-threaded (the service is awaited directly;
no worker, no `call_from_thread`), so the caveat does not apply. It is now stated in the helper's
docstring, so a future test that adds a thread has the trap in front of it.

## 3. `Args:` in test docstrings — judgment applied, not blanket compliance

I measured the precedent independently and reproduce the coordinator's figure exactly: **50 of
1,392** `.py` files under `Tests/` contain an `Args:` section — 3.6%. (Narrowed to `test_*.py`
files only it is 43 of 1,315, 3.3%.) So the convention exists but is a small minority, and Qodo
itself rates the item "unclear" for tests.

Decision: **`Args:` only where a parameter is genuinely non-obvious; bare pytest fixtures stay
undocumented.** Applied as:

* `_serve` — documented. `pages` has real semantics a reader cannot guess (one body per fetch, in
  order, last entry repeated on exhaustion), and `content_type` selects which parser
  `_fetch_and_parse_feed` uses.
* `_site_source` — documented, and this is the case that justifies the policy: `change_threshold`
  is not merely non-obvious, it is a **trap**. It is a whole-page character-level similarity
  measure, so a small edit to a long page never clears the 0.1 default and the test silently gets
  zero items — which is exactly what happened to the round-1 rule-scope tests on their first run.
  That belongs in an `Args:` entry; "tmp_path: the pytest temporary directory fixture" does not.
* every test function — not documented. After item 2 they take at most `monkeypatch`, and several
  take nothing at all. Twenty repetitions of a boilerplate fixture description would dilute
  docstrings whose value is explaining *why the test exists*.

## Round-2 verification

```
Tests/Subscriptions/ + Tests/Scheduling/   341 passed  (19 in the new file, up from 17)
Tests/Watchlists/                          189 passed
Tests/DB/ -k subscription                   47 passed, 563 deselected
Tests/UI/ -k watchlist                       3 failed, 232 passed   (see below)
```

### A note on the `Tests/UI/ -k watchlist` baseline — it is wider than one test

Three runs of the identical command in this round produced three different failing sets:

| run | failures |
|---|---|
| 1 | 2 × chevron, `test_watchlists_source_create_form::test_clicking_any_row_of_the_name_input_focuses_it[size0]`, `test_watchlists_source_frequency_control::test_frequency_options_are_reachable_when_expanded[size1]` (**4**) |
| 2 | 2 × chevron, `test_watchlists_source_create_form::test_a_source_can_be_created_end_to_end_through_the_form[size1]` (**3**) |
| pre-round-2 | 2 × chevron, `test_watchlists_source_create_form::test_typing_straight_after_opening_the_form_lands_in_name[size1]` (**3**) |

The two chevron failures are constant. The rest is the TASK-1345 order-dependent mount race, and
the finding worth recording is that **it is not confined to one named test**: it moves between tests
within `test_watchlists_source_create_form.py` and has also surfaced once in
`test_watchlists_source_frequency_control.py`. Both files pass **19/19 in isolation**, and neither
can be reached by anything in this round (the source change is confined to `build_change_diff`,
which no UI test in this selection exercises; the test-file change is under `Tests/Subscriptions/`,
which `-k watchlist` over `Tests/UI/` does not collect). Quoting a fixed name for this baseline will
keep producing false regressions.

### Round-2 mutation outcome

| Mutation | Result |
|---|---|
| E — restore `list(unified_diff(...))` materialisation | **1 failed** (`test_the_diff_generator_is_not_materialised`); the other 18 stayed green, which is the point: the behavioural test correctly does not claim to cover memory |

Source restored from a pre-mutation copy and `diff`-verified byte-identical afterwards.

### Files changed in this round

* `tldw_chatbook/Subscriptions/monitoring_engine.py` — single-pass streaming diff with
  continue-past-cap counters.
* `Tests/Subscriptions/test_watchlist_content_kind_producer.py` — 17 → 19 tests; `:memory:` DB;
  `tmp_path` removed throughout; `Args:` added to the two helpers that earn it.
