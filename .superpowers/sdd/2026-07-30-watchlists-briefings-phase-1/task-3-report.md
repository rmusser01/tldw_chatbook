# Task 3 report — the generation service

**Status:** complete. New: `tldw_chatbook/Subscriptions/briefing_service.py`,
`Tests/Subscriptions/test_briefing_service.py` (10 tests). Nothing else touched.

## What shipped

Three public names, the interface task 4 consumes verbatim:

- `build_briefing_prompt(items, featured_ids, overflow_count) -> (system, user)` — pure,
  `content_kind`-aware, featured-first.
- `async generate_briefing(db, watchlist_id, *, chat=chat_api_call, provider=None, model=None,
  now=None) -> dict` — the finished `briefings` row, whatever its status.
- `fail_interrupted_briefings(db, watchlist_id=None) -> int` — zombie recovery.

`now` is the one addition to the brief's signature: an injected clock forwarded to selection
(which already takes one), so the first-briefing 7-day floor is not read from the wall clock
inside the service. Everything else is exactly as briefed.

### The pipeline

`insert_briefing(status='generating')` → read the watchlist's `briefing_selection_mode` →
`select_briefing_items` → branch.

**Empty window** → `status='empty'`, `item_count=0`, and **the chat seam is never invoked**.
The watermark written is the selection's, or — when selection returns `None`, meaning "no line
to record" (empty window, or curated with no prior briefing) — the **prior** watermark. That
choice makes the row self-describing: it states the line it covers through rather than leaving
a NULL a reader has to interpret. `latest_completed_watermark` takes a `MAX`, so echoing the
prior value is a no-op to it either way.

**Items** → prompt → one `chat` call, `streaming=False`. The system prompt travels in
`system_message`, not as a message role, because that is the app's own division of labour
(the `Chat_Functions` "PHILOSOPHY" comment: each provider handler decides whether its API wants
a prepended system turn or a separate top-level field). Response text comes back through
`extract_response_content`, the app's own extractor, so both the bare-string and the
OpenAI-shaped-dict returns of `chat_api_call` are handled.

**Success** → junction rows first (featured flag from `selection.featured_ids`), then the status
flip to `complete`. That order is deliberate: a crash between the two leaves a `generating` row
whose junction rows the selection exclusion already ignores (its allowlist is
`('complete','empty')`) and which recovery then fails honestly. The reverse order would briefly
publish a complete briefing that covered nothing.

**Failure** → `status='failed'`, the exception's message (capped at 1000 chars, never a
traceback — the stack goes to the log), **no junction rows, no `covers_through_item_id`**.

`generate_briefing` never raises for a provider failure. A briefing the user can see failed is
worth more than an exception they cannot.

### Judgement calls worth stating

1. **An empty model response is a failure, not an empty briefing.** Recording it `complete` with
   a blank body would show an artifact with no error to explain it *and* advance the coverage
   window past items nothing ever reported. Covered by
   `test_an_empty_model_response_is_a_failure_not_an_empty_briefing`.
2. **An unknown/NULL `briefing_selection_mode` falls back to `auto_featured` with a warning**
   rather than raising. An unknown mode would otherwise escape `select_briefing_items` *after*
   the `generating` row was inserted — creating exactly the zombie row this design goes out of
   its way to avoid.
3. **The failed row records no counts.** `item_count`/`featured_count` on a failed row read as
   delivered coverage. It records `status`, `error`, `selection_mode`, `model_used` — enough to
   say what was attempted and against which provider.

### Default provider

`config.default_api_endpoint` (config.py:5324 — `settings["llm_api_settings"]["default_api"]`),
read at call time through the module object (`from .. import config as app_config`) rather than
snapshotted at import, so a config reload is picked up. **No provider name is hardcoded in this
module**; config.py owns the `"openai"` fallback. The happy-path test monkeypatches the attribute
to `"local-llama"` and asserts that value reaches the seam — an assertion a hardcoded default
would fail.

## Tests (10, all `pytest.mark.unit`; only the chat seam is faked)

The brief's five, plus five that pin behaviour the five would not have caught:

| Test | Pins |
|---|---|
| `test_generation_happy_path_writes_everything` | body + **service-side** overflow sentence, `item_count`/`featured_count`/`overflow_count`, junction featured flags, `covers_through_item_id == selection`'s and `== max(all ids)`, one non-streaming call to the configured provider |
| `test_llm_failure_is_honest_and_loses_nothing` | **the named invariant** — status+error, no traceback, no junction, watermark unchanged, and a second generate re-selecting the same three item **identities** |
| `test_empty_window_is_a_row_not_an_absence` | `empty` row, `chat.calls == []`, watermark held, self-describing `covers_through_item_id` |
| `test_prompt_labels_diffs_as_diffs` | change section carries the diff + "page change"/"not an article"/`diff_summary`; article section carries its excerpt and is *not* called a page change; featured first from an unfavourable input order |
| `test_interrupted_recovery_only_touches_generating_rows` | scoped and global sweeps, finished rows keep status/body/watermark/**own error text**, re-run returns 0 |
| `test_long_article_excerpt_is_capped_in_the_prompt` | the 800-char cap actually cuts (a tail marker is absent), truncation stated |
| `test_an_empty_model_response_is_a_failure_not_an_empty_briefing` | judgement call 1 |
| `test_explicit_provider_and_model_override_the_default` | preset provider/model win over the app default; `model_used` |
| `test_curated_mode_generation_leaves_the_window_alone` | the service writes task 2's curated **echo** through unchanged instead of recomputing a max from its own item list |
| `test_generation_accepts_an_async_chat_seam` | a coroutine `chat` is awaited, not stored as a coroutine object |

The happy path seeds `DEFAULT_ITEM_CAP + 2` real items so the overflow leg is produced by the
shipped cap rather than simulated, and gives the watchlist a prior completed briefing so
`covers_from_ts` is a property of the items rather than of the clock (and therefore assertable
by equality). Item timestamps are relative to the real clock, so the 7-day first window keeps
containing them however far in the future this suite runs.

**Runs:** `Tests/Subscriptions/test_briefing_service.py` **10 passed in 1.15s**;
`Tests/Subscriptions/` **222 passed in 53.17s** (212 before this task).

## Mutation checks (each restored with an editor; `git diff` clean afterwards)

| # | Mutation | Observed |
|---|---|---|
| 1 | advance the watermark on failure (`covers_through_item_id=covers_through` in the fail path) | RED: `assert 4 is None` in `test_llm_failure_is_honest_and_loses_nothing`. 1 failed, 9 passed. |
| 2 | `_append_overflow` returns the body unchanged | RED: `AssertionError: assert '2 more items arrived in this window and are not covered' in '## This week\n\nAcme shipped a thing [item 1].'`. 1 failed, 9 passed. |
| 3 | recovery `WHERE status IN (?, 'complete')` | RED: `assert 2 == 1` at `fail_interrupted_briefings(db, watchlist_id=watchlist) == 1`. 1 failed, 9 passed. |
| 4 | prompt builder ignores `content_kind` (`if False:` — every item an article) | RED: `assert 'page change' in 'acme pricing\nkind: article\n...\n- free tier: 10 seats\n+ free tier: 3 seats...'`. 1 failed, 9 passed. |

### Mutation 1b — checking the invariant's *consequence* leg has teeth

Mutation 1 trips the direct column assertion, which sits before the consequence assertions — so
on its own it does not prove the "re-selects the same items" leg is anything but decorative.
Worse, mutation 1 alone is survivable at the DB layer: `latest_completed_watermark` already
excludes `failed`, so a failed row carrying a watermark still does not move the line.

So I ran the two together — the service writing the watermark on failure **and**
`latest_completed_watermark`'s allowlist widened to include `'failed'` (task 1's half) — with the
direct column assertion commented out. RED at the consequence:
`assert db.latest_completed_watermark(watchlist) == old_item` → **`assert 4 == 1`**.

The finding: the invariant has **two independent guards**, one per task, and this test asserts
both. Either alone holds the line; neither alone is sufficient documentation of why. Do not
delete either assertion. All three edits were reverted with an editor.

---

# Fix round 1

Four changes, one new test. Runs: `Tests/Subscriptions/test_briefing_service.py` **11 passed in
0.84s**; `Tests/Subscriptions/` **223 passed in 36.00s**.

## 1 (Important) — the failure log leaked prompt content

Upheld, and worse than "a style deviation": my own module docstring claimed "Nothing here is
logged with content" while `logger.opt(exception=True)` at the failure site handed loguru a
traceback whose innermost frame is `_invoke_chat`, whose locals are the prompt. With the file
sink's `diagnose=True` the renderer writes ~120 characters of each local, so the first item's
title and the head of the user prompt land in a local file the user never chose to send
anywhere. Three sibling files (`app.py:7078`, `settings_screen.py:7209`,
`local_llm_provider_catalog_service.py:669`) carry the convention verbatim; I did not follow it.

Now a plain `logger.warning` with `type(exc).__name__` and no message text, carrying the sibling
comment plus the reason specific to this site (the frame here *is* the prompt). The provider's
own message still reaches the user on the row, where they are already looking. The docstring
claim now names the trap and points at the test.

**New test `test_a_failed_generation_logs_no_item_content`** — a loguru sink configured the same
way the file sink is (`diagnose=True, backtrace=True`), a canary item title, and assertions that
the failure IS logged (silence is not the fix), that the exception *type* is logged, and that
neither the canary nor the string `messages_payload` appears.

**The trap inside the trap, worth recording:** the first draft of that test used a 15-character
canary and **passed against the live leak** — loguru truncates each frame-local repr at ~120
characters and the cut landed one character into the title
(`'...### [item 1] ZEBRAFISHCANAR...'`). I only caught it because the mutation refused to RED,
and then probed the captured text instead of trusting the green. A canary must sit near the head
of the value being probed, or it measures the truncation rather than the leak.

## 2 (Important) — the double-guard reasoning moved to the assertions

`test_llm_failure_is_honest_and_loses_nothing` now labels the three legs inline (leg 1 the
service's failure branch, leg 2 the DB's status allowlist, leg 3 the identity consequence) and
states what the review established: each guard alone absorbs a mutation of the other, so only the
composed mutation reaches leg 3, and deleting any assertion as "redundant" deletes the proof that
the surviving guard is doing the work. Future readers read code, not task reports — agreed.

## 3 (Important) — the page label now names the page by identity

Reproduced: genericizing the label to "a monitored page" left all 10 tests green, because
`change_at` was located from the section *heading* (which still carried the title) and the
remaining assertions were all generic wording. The test now extracts the `Kind:` label line
itself and asserts the page title and source name appear in it. Matching the label line needed
`line.startswith("Kind:")` as well — the `Change:` headline contains "page changed" and matched a
bare substring test.

## 4 (Minor) — the excerpt bound is now a checked property

`test_long_article_excerpt_is_capped_in_the_prompt` asserts
`len(contribution) <= EXCERPT_CHAR_CAP + len(marker)` using the real constants, so a cap of 4000
(which would still cut the tail and still say "truncated") fails.

## Mutation checks (restored with an editor; `git diff` clean afterwards)

| # | Mutation | Observed |
|---|---|---|
| 5 | page label genericized to "a diff of a monitored page" | RED: `assert 'Acme Pricing' in 'Kind: page change. This is a diff of a monitored page -- it is not an article...'`. 1 failed, 9 passed (pre-round-1 count). |
| 6 | `logger.opt(exception=True)` restored | First run **GREEN** — the canary was past loguru's repr truncation. With the canary shortened and moved to the head: RED, `assert 'ZEBRACANARY' not in '2026-07-30 ...stream 503\n'`. |

Mutation 6 is the round's real lesson: a mutation that fails to RED is information about the
*test*, not permission to move on.
