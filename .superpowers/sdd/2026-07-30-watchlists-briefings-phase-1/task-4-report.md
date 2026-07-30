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
