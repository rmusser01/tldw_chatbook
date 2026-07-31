# Task 4 — the Artifacts section

Spec #2 phase 1 (`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`,
§UI). Tasks 1-3 built the tables, the selection and the generation service and
gave none of it a surface. This is the surface.

## What shipped

**`tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (new).** A scope
line, a Generate/Refresh toolbar, a `DataTable` of briefings (status, window,
item/featured/overflow counts, created) and a detail area. Sibling
conventions copied verbatim rather than reinvented: `RecomposeCaptureGuard`
ahead of the container (both row reactives are `recompose=True` on a
non-screen widget), `highlight_is_user_driven` (i.e. `table.has_focus`) on
every `RowHighlighted`/`CellHighlighted` so a freshly recomposed table's
row-0 announcement cannot feed back into selection, and the cursor seeded
from the surviving selection so a click on row 2 is not dragged back to row 0.

Every status renders a body of its own: `complete` renders
`Markdown(body, hyperlinks=False)`, `failed` shows the provider's own message,
`empty` says the window was empty, `generating` says it is being written. None
of them is a blank pane — the spec's "silence is never a state", one layer up
from where task 3 applied it.

**`hyperlinks=False`** is the one security decision here, and it is the same
one `content_pane.py` already records for the reader (PR #1091 F3 /
TASK-1348 AC#2): a briefing body is model output written *from remote feed
content*, so both halves of `[label](url)` are attacker-influenced. Rich's
default emits OSC-8, hiding an attacker-chosen destination behind an
attacker-chosen label. With it off, the label renders and the URL renders
beside it as text the reader can judge.

**`watchlists_collections_screen.py`.** Seventh section — `BINDINGS` key `7`,
`_SECTION_DETAIL_TITLE`, the `_build_detail_pane` arm, the loader dispatch,
`BriefingSelected`/`RefreshBriefingsRequested`/`GenerateBriefingRequested`
handlers, `_load_briefings`, and the `wl-briefing` worker. Plus
`SECTIONS` in `watchlists_tab_strip.py` and a `db` property on
`WatchlistBundleService` (the app wires the *service* onto the app instance,
so a caller that legitimately owns its own queries had no honest way to the
store).

The CONTENT gate needed **no** change — it already keys on
`active_section != "items"` — but nothing asserted that, so a test now pins
it: opening Artifacts must not mount the reader.

### The Generate guard, in order

`generate_briefing` neither checks nor recovers, deliberately (its module
docstring: folding either in would make the service both the guarded thing
and the guard). So the order in `handle_generate_briefing_requested` is the
contract:

1. **A generation this screen started is answered from memory**
   (`_briefing_in_flight`). A live worker's row reads `generating` exactly
   like a crashed one's, and only this process knows which it is.
2. **`fail_interrupted_briefings` runs before the generating-check**, so a
   row orphaned by a crash cannot wedge the button shut forever.
3. **Anything the sweep actually recovered is reported, and the press stops
   there.** That row may belong to another live instance of this app against
   the same database file; starting a second generation over the top of one
   still running would spend the user's provider quota twice on the same
   window. Telling them what was found and letting them press again is the
   non-destructive half of an ambiguity the database cannot resolve.
4. A `generating` row that survives the sweep refuses.

Every toast on this path passes `markup=False` (`_notify_watchlists` grew the
parameter): these bodies carry counts, watchlist names and provider error
text, none of which this app authored.

### The worker

`run_worker(..., group="wl-briefing", exclusive=True)` — never exclusive
without a group (TASK-1362); the load path gets its own
`group="wl-briefings-load"` for the same reason, so a repaint cannot cancel a
generation. The body wraps `generate_briefing` in a bare `except` because
that function lets **database** errors propagate on purpose (a database error
is not a briefing outcome) and an exception escaping a Textual worker with
the default `exit_on_error=True` exits the application. The log line names
the exception TYPE only: the file sink runs `diagnose=True`, and the frames
under this call hold the prompt — task 3's review found exactly that leak in
the service, and this is the same rule one layer up.

Completion repaints the **pane**, never the screen: `_load_briefings` pushes
rows into the mounted `ArtifactsPane`, whose own `recompose=True` reactives
rebuild its children while the pane instance survives. A test holds the
instance across a generation and asserts identity.

## Tests

`Tests/Watchlists/test_watchlists_artifacts_pane.py`, 9 tests, marked
`pytest.mark.ui` so CI actually collects them (the unit job selects `-m unit`
and the UI job runs `Tests/UI` plus `Tests -m ui --ignore=Tests/UI`; an
unmarked file in `Tests/Watchlists` is collected by neither).

**Exactly one seam is faked: the chat call**, injected at the service
boundary. `generate_briefing` binds its `chat` default at definition time, so
patching `briefing_service.chat_api_call` would not reach it; the tests wrap
the *screen's* reference instead, leaving selection, statuses, junction rows
and the watermark as shipping code over a real `SubscriptionsDB`.

1. Artifacts is in the strip; opening it mounts the pane at the full detail
   width; CONTENT stays unmounted (its collapsed header still reachable).
2. Generate → one provider call, one `complete` row with `item_count == 2`,
   the pane instance survives, the table paints `complete`, and the body
   renders — with the hostile-link assertions through a real render: no
   OSC-8 anywhere, `https://evil.test/steal` disclosed beside its label,
   `[Anthropic docs](` absent (so the markdown branch really ran), and
   `[click](javascript:alert)` surviving as literal characters. The last of
   those is then confirmed against `_compositor.render_strips()` — what the
   terminal actually painted.
3. A pre-seeded `generating` row: the first press generates nothing and
   toasts (`markup=False`); the second press writes a real briefing, with the
   zombie row left `failed`/`interrupted`. Both halves through the real button.
4. A database error from generation: the app is still running, the screen is
   still on the stack, an error toast fired with `markup=False`, the
   in-flight guard cleared, and a second press is accepted.
5. `failed` and `empty` rows explain themselves in the detail area.
6. Moving the tree scope moves what Artifacts is about — self-review found
   the gap: nothing reloaded the pane on a scope change, so it would have
   kept showing watchlist A's briefings while Generate acted on B. Fixed in
   `watch_tree_scope`, and mutation-checked (below).
7. A bracket-shaped watchlist name paints instead of exploding — the second
   self-review find: `Static` parses Rich markup by **default**
   (`Static(..., markup=True)`), and the scope line names a watchlist the
   user typed. The pane wraps it in a `Text`; the test pins that the name
   paints verbatim, gains no escaping backslashes, and applies no style.
8. Geometry at 160×42 and 180×50 under the production stylesheet
   (`_visual_destination_harness`): list, Generate and body all **placed
   inside the terminal** (not merely `height > 0`), in order, with "Generate"
   actually painted. The fixture generates a 40-paragraph body, which is what
   makes the CSS rule bind.

## Mutation checks

| # | Mutation | Result |
|---|----------|--------|
| a | `fail_interrupted_briefings` call removed (`recovered = 0`) | RED, **only test 3, only its second half**: `AssertionError: after zombie recovery the same button must actually generate — assert 0 == 1`. The first half stays green, via the surviving `still_generating` branch. |
| b | `_MARKDOWN_HYPERLINKS = True` | RED, test 2: `a briefing body must never emit a real terminal hyperlink` — the ANSI shows `]8;id=…;https://evil.test/steal\Anthropic docs]8;;\`. |
| c | worker `try/except/finally` removed | RED, test 4 — and **the application actually dies**: `host.is_running` is False at the first assertion, and the harness then raises `textual.worker.WorkerFailed: Worker raised exception: OperationalError('database is locked')` out of `run_test`'s exit. That is the failure mode the wrapper exists for, observed rather than assumed. |
| d | `#artifacts-detail` → `height: auto` (bundle regenerated) | RED, **both** geometry sizes: `the briefing body runs off the bottom of a 160x42 terminal: Region(x=30, y=30, width=93, height=82)` (and `180x50` likewise). 82 rows laid out below a 42-row terminal — the exact claim the CSS comment makes. |
| e | `watch_tree_scope`'s Artifacts reload disabled | RED, test 6 only: `assert 1 == 0 … DataTable(id='artifacts-table').row_count` — the pane kept the previous watchlist's row. |
| f | scope line back to a bare `str` in `Static` | RED, test 7: `assert '[bold red]Morning [brief' in 'Briefings for Morning [brief · …'` — the tag was **swallowed**, so a real watchlist name silently loses characters. |

(a)-(d) are the four the brief asked for; (e) and (f) cover the two defects
self-review found. All six restored with the editor; the bundle was
regenerated via `build_css.py` after (d) and the suite re-run green.

## Test runs

- `Tests/Watchlists/` — **209 passed** (includes the 9 new).
- `Tests/UI/ -k watchlist` — **246 passed, 2 failed**: only the two known
  `test_watchlists_tree_chevron_shares_a_row_with_its_watchlist` baselines.
- Worth recording: two earlier runs of that selection *also* failed
  `test_the_whole_create_form_fits_inside_the_sources_pane[size0]` with
  `#sources-create-name at Region(0,0,0,0)` — a widget queried before layout.
  Both of those runs had another pytest process running alongside them. Run
  alone it passes, and the run above is the clean one. The test waits a fixed
  `pilot.pause(0.3)` after pressing `New Source`, so it is load-sensitive by
  construction; nothing in this task touches the Sources pane, its CSS, or
  its layout. Flagging it rather than filing it, since "passes on retry" is
  not automatically a flake here — but the retry that passed was the one
  with an idle machine, which is the opposite direction.

## Files

- `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (new)
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `tldw_chatbook/UI/Watchlists_Modules/watchlists_tab_strip.py`
- `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`
- `tldw_chatbook/css/features/_watchlists.tcss` + regenerated
  `tldw_chatbook/css/tldw_cli_modular.tcss`
- `Tests/Watchlists/test_watchlists_artifacts_pane.py` (new),
  `Tests/Watchlists/test_watchlists_tab_strip.py`

---

# Fix round 1

Six findings, all addressed. Two of them changed the shape of the feature;
the rest tightened tests and copy.

## 1 (Important) — the guard no longer runs on the UI thread

`fail_interrupted_briefings` is a transactional UPDATE and `list_briefings`
is a read, and both ran inside the message handler. No busy timeout beyond
SQLite's default is configured, and this feature's own design admits a second
app instance against the same database file, so a contended write froze the
interface.

The handler now does only what the UI thread is entitled to do: answer from
memory (no store / no watchlist / already in flight) and dispatch. Everything
touching the database moved into the worker as one sequence —
`_sweep_and_guard` (sweep, then the generating-check, in that order, still)
runs through `asyncio.to_thread`, and the refuse decision is made and toasted
from inside the worker. `SubscriptionsDB` holds thread-local connections, so
the worker thread gets its own; this is the idiom the rest of the UI already
uses (`mcp_inspector.py`, `ccp_character_handler.py`, …).

## 2 (Important) — the guard is claimed at dispatch

`_briefing_in_flight = True` moved out of the worker body and into the
handler, before `run_worker`. `run_worker` only *schedules*: a check made
inside the worker leaves a window in which two presses both pass, and
`exclusive=True` then cancels the first one mid-generation — the guard
manufacturing the `generating` row it exists to prevent.

**Two tests, and they are not equivalent.** The outcome test the review asked
for (two same-tick presses → exactly one service call, one `complete` row)
**passes under the mutation as well**: `exclusive=True` collapses two
same-tick dispatches into one generation, so the outcome is identical either
way. What the mutation *does* leave behind on that test is a
`PytestUnraisableExceptionWarning: Exception ignored in: <coroutine object
WatchlistsCollectionsScreen._generate_briefing …>` — the discarded first
dispatch, never awaited. So the outcome test is kept for the property, and a
second, deterministic test pins the mechanism: the handler is synchronous and
has no `await`, so when it returns no worker code can have run — the flag
must already be True. That one fails immediately under the mutation.

## 3 (Important) — the re-arm assertion discriminates

`test_a_database_error_during_generation_does_not_exit_the_app` asserted only
`app.notify.called` on the second press, which a refusal satisfies
identically. It now restores a working seam, presses again, and asserts the
service was reached (`len(chat.calls) == 1`) **and** that a `complete` row
exists.

## 4 (Important) — the settle loop waits on state

`_press_generate` now waits, with a 20s bound and no fixed sleep as the
carrier, on: the press being answered (guard claimed / a toast / the rows
changed), then the worker finishing, then the repaint agreeing with the
database.

**Measured, since the round-0 helper's fragility deserves a number rather
than an argument.** With fix 2 in place, the round-0 loop is no longer
vacuous — the flag is set before `press()` returns to the pump, so
`if not _briefing_in_flight: break` now genuinely waits. I could not
construct a failure for it: 0.6s injected into the service, then 0.6s
injected into the sweep, then both *plus* the round-0 flag placement, all
still green (the worker starts within one 0.05s tick on this machine). So the
honest claim is narrower than "it was broken": the round-0 loop depended on
worker-start timing for its correctness and the new one does not, and the new
one additionally covers the pre-dispatch refusal path (which never sets the
flag at all) and the repaint.

## 5 (Minors)

- **(a)** Both refusal toasts now name the row: `briefing 3 (started
  2026-07-30 21:02:11)`, from `_briefing_row_label`. `_sweep_and_guard`
  returns labels rather than counts precisely so they can.
- **(b)** Artifacts joins Notifications in `_LOCAL_ONLY_SECTIONS`: the
  Backend `Select` is disabled with its own tooltip and the header reads
  `Artifacts: local`. `watch_runtime_backend`'s duplicate ternary collapsed
  into the same `_backend_label_text()`. A test pins both halves — Artifacts
  disabled, Sources still live.
- **(c)** The pane comment now states what was measured (`[bold red]Morning
  [brief` paints as `Morning [brief` — the tag is swallowed, Textual did not
  raise on the unclosed one) instead of claiming a compose crash.
- **(d)** The parent-child width assertion is gone. The full-width claim now
  lives in the real-CSS geometry test, against the width **Sources** gets on
  the same terminal (93 at 160×42, 113 at 180×50), which is a claim that can
  fail.
- **(e)** New test: an unfocused table's rebuild announcement must not select,
  and a focused table's cursor move must. **Precise mutation result:**
  dropping the guard from `on_data_table_row_highlighted` alone is **GREEN** —
  this table's cursor is cell-shaped, so the announcement arrives as
  `CellHighlighted`. Dropping both guards REDs two tests: the new one
  (`selected_briefing` is the row-0 announcement instead of `None`) and
  `test_failed_and_empty_briefings_explain_themselves` (the feedback loop
  dragging a selection back). Both handlers are kept, for parity with the
  sibling panes and because the row-shaped path is one `cursor_type` away —
  but the row-handler guard is, today, unexercised, and that is worth knowing
  rather than assuming.

## Round-1 mutation checks

| # | Mutation | Result |
|---|----------|--------|
| g | `_briefing_in_flight = True` moved back inside the worker body | RED, `test_the_guard_is_claimed_before_the_worker_runs`: `the guard must be claimed by the handler, before run_worker has scheduled anything — assert False is True`. The double-press outcome test stays green (see §2) but emits the never-awaited-coroutine warning. |
| h | `has_focus` guard dropped from `on_data_table_row_highlighted` only | **GREEN** — see §5e. |
| h2 | dropped from both highlight handlers | RED ×2: `test_only_a_focused_tables_highlight_selects` (`assert {'id': 2, …} is None`) and `test_failed_and_empty_briefings_explain_themselves`. |

## Round-1 test runs

- `Tests/Watchlists/test_watchlists_artifacts_pane.py` — **13 passed** (9 → 13).
- `Tests/Watchlists/` — **213 passed**.
- `Tests/UI/ -k watchlist`, run with nothing else on the machine —
  **245 passed, 3 failed**: the two known chevron baselines, plus one
  `test_watchlists_source_create_form.py` test.

### That create-form file, now with four data points

It is not a regression from this task, and the evidence is that it is never
the same test twice:

| run | conditions | which create-form test failed |
|-----|-----------|-------------------------------|
| 1 | another pytest running alongside | `test_the_whole_create_form_fits_inside_the_sources_pane[size0]` — `#sources-create-name at Region(0,0,0,0)` |
| 2 | another pytest running alongside | same |
| 3 | machine idle, pre-round-1 | none — file green |
| 4 | machine idle, post-round-1 | `test_a_source_can_be_created_end_to_end_through_the_form[size0]` — `assert 'orning' == 'Morning'` |
| 5 | parity module + this file only | `test_tab_walks_the_create_form_in_visual_order[size1]` — `nothing focused` |

Run alone the file is **15/15 green** (measured after round 1). Three
different tests, three different symptoms, all of the same shape: a fixed
`pilot.pause(0.3)` after `New Source`, asserted against a form whose focus
and layout have not settled — a dropped first keystroke, an unfocused field,
a zero region. Nothing in this task touches the Sources pane, its CSS, or its
layout; `_local_only_section()` returns `None` for Sources, so its header bar
composes exactly as before. Flagging rather than filing: the fix is the same
one applied to `_press_generate` here — wait on observable state, not on a
fixed pause — and it belongs to whoever owns that file.
