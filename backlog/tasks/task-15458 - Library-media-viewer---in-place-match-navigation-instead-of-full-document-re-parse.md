---
id: TASK-15458
title: 'Library media viewer: in-place match navigation instead of full-document re-parse'
status: Done
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit — plausibly the single worst click in the app: the media viewer holds the entire document as one Markdown widget (`Widgets/Library/library_media_viewer.py:148`), and next/prev match (`library_screen.py:23476`), mode toggle (`:23431`), and search submit (`:23362`) each perform a full-document markdown re-parse plus a whole-screen remount — multi-second per click on a long transcript on constrained hardware. Separately, the media panel's content search re-parses the document twice per keystroke via `Markdown.update("")` followed by `Markdown.update(content)` (`Widgets/Media/media_viewer_panel.py:1000-1025`, reached from `:1223/:1266`), with no debounce.

Fix direction: keep the Markdown widget mounted and move match-highlight and scroll-to-match to in-place updates; drop the empty pre-update; debounce the search box. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Match navigation does not re-parse the document or remount the screen (evidence)
- [x] #2 Search-while-typing performs at most one deferred re-render per debounce window and never the double update(\"\")/update(content) parse
- [x] #3 Match highlighting and scroll behavior preserved (tests); click latency before/after on a long document recorded
- [x] #4 Literal async pytest commands run on Windows without mutating the guarded network families, while ordinary same-thread and concurrent-thread application egress remains blocked and recorded
<!-- AC:END -->

## Implementation Plan

1. Add mounted tests and a focused `LibraryMediaContentSearchControls` widget whose match-only updates preserve navigation identity and focus.
2. Add mounted tests and a lazy `LibraryMediaContentBody` that stores Raw search state, mounts each selected mode at most once, and applies latest-request-wins visibility.
3. Integrate the focused children through `LibraryMediaViewer` and replace Library screen recomposes with narrow query, match-index, mode, and post-layout scroll synchronization.
4. Add mounted legacy-panel tests and a generation-guarded 250 ms content-search debounce; remove the empty Markdown cache-busting update.
5. Record deterministic before/after latency and parse/identity evidence, run focused and full verification, complete rendered keyboard UAT, and close the task documentation.

Detailed plan: `Docs/superpowers/plans/2026-08-12-library-media-viewer-inplace.md`

ADR required: yes

ADR path: `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md`

Reason: the media-viewer changes apply existing Textual ownership patterns, but
the review-expanded verification scope changes the repository-wide test network
security boundary and its runtime interception contract.

## Implementation Notes

### Task 3 pre-change product-path evidence

The exact Task 3 focused command selected 16 tests. Before production changes,
five new regressions failed for the intended missing behavior: the viewer was
replaced on search/navigation, the persistent content body was not mounted,
inactive Raw/Rendered children were removed, and Rendered search state did not
reach a later Raw mount. The Enter-only regression already passed.

The deterministic 2,000-line Markdown document contained 101 matching source
lines: 100 generated match lines plus the heading, which also contains
`budget`. The isolated repeat of the identical submit/Next/Prev sequence
recorded `TASK-15458 latency median_ms=1091.689`. The screen stayed at object ID
`2237538817072`, while the viewer IDs were `2235866381104`, `2237674891232`,
`2235874008944`, and `2235896514272`, and the Markdown IDs were `2235871484416`,
`2235878885376`, `2235889419760`, and `2235894721824`. The construction proxy
therefore observed four distinct viewers and Markdown objects, plus four
parse-triggering `Markdown.update` calls (initial mount plus one for every action).
These process-local IDs are evidence of replacement, not stable identifiers
expected to recur across runs. The preceding full-slice run measured 1239.726 ms
and the same four-object/four-parse shape.

ADR required: no additional ADR for Task 3. This slice applies the already
reviewed widget-ownership interface without changing storage, runtime, security,
dependency, or cross-module service boundaries. The parent task's ADR-058 remains
the governing decision for the separate Windows network-guard verification scope.

### Task 3 implementation and post-change evidence

Integrated the reviewed search-controls and persistent-body components through
`LibraryMediaViewer`, then replaced Library screen query, navigation, and mode
recomposes with narrow mounted-viewer synchronization. Search remains
Enter-submitted; navigation wraps and schedules source-line scrolling after a
refresh boundary; Rendered remains the Markdown default; Raw and Rendered remain
mounted after first use; and async mode synchronization delegates to the existing
latest-request-wins body contract. No dependencies, persistence, workers, legacy
Media debounce, or geometry rules changed.

The required focused product-path command passed all 16 selected tests. Its
2,000-line/101-match evidence recorded `TASK-15458 latency median_ms=120.913`,
one stable screen ID (`2247968276448`), one stable viewer ID
(`2247963750576` across all four observations), one stable Markdown ID
(`2247965990096` across all four observations), and one `Markdown.update` call
for the initial mount only. An isolated repeat measured 89.561 ms with the same
one-viewer/one-Markdown/one-parse shape. This compares with the isolated
pre-change 1091.689 ms median, four viewer objects, four Markdown objects, and
four parses.

The full `Tests/Library/test_library_media_content.py` suite passed 9 tests. A
mutation that restored `self.refresh(recompose=True)` in match navigation made
the identity/focus/parse regression fail at the viewer identity assertion; after
restoring the narrow path, that regression passed. Task-scoped Ruff checks passed.
The whole-file Ruff baseline still reports pre-existing `E721` findings in
`library_screen.py` and one pre-existing unused import in `test_library_shell.py`;
the scoped verification excluded only those existing rule findings. `git diff
--check` was clean.

### Task 5 interim verification checkpoint (blocked)

- Extracted scoped Library search controls and a lazy persistent content body so
  match navigation updates status/highlighting/scroll state without remounting
  the screen or reparsing Markdown.
- Preserved Enter-to-search, Rendered-by-default Markdown behavior, Raw
  highlighting, wraparound navigation, and focus continuity in the mounted
  product-path tests. Rendered UAT found that the Library status and navigation
  controls are nevertheless occluded by the content body at 170x48; therefore
  acceptance criterion #3 remains open and this task remains In Progress.
- Debounced legacy content search at 250 ms with monotonic lifecycle invalidation
  and one Markdown update per applied query.
- Repaired Windows async-test bootstrap under ADR-058 with a nested
  current-thread socketpair exemption; literal pytest commands pass while
  ordinary and concurrent-thread egress remain denied and recorded.
- ADR required: yes; followed
  `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md` for the
  review-expanded test security boundary.

#### Fresh focused verification

All required task-focused commands ran literally with the parent checkout's
Python 3.12 virtual environment and default network denial:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py Tests/UI/test_media_viewer_content_search_debounce.py -v
# 16 passed

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/test_network_guard.py -q
# 12 passed, 1 skipped (AF_UNIX unavailable on Windows)

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_library_shell.py -k 'media_content_search or media_viewer_raw_toggle or media_viewer_defaults_markdown or media_viewer_inplace' -v -s
# 17 passed, 546 deselected

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_media_window_v2_parity.py Tests/UI/test_media_handoffs.py -q
# 38 passed
```

The combined focused result is 83 passed and one expected platform skip. The
fresh 2,000-line product-path measurement was `103.287 ms` median for
submit/Next/Previous, compared with the isolated pre-change `1091.689 ms`
median. The four viewer observations had one stable viewer identity; the four
Markdown observations had one stable Markdown identity; and
`markdown_update_count=1` represented initial construction only, so navigation
caused zero Markdown updates. The fixture is 49,288 characters and contains
2,000 lines and 101 matching source lines; earlier notes that called this 100
matches omitted the matching heading.

The parent virtual environment does not contain the Ruff module, so the literal
`python -m ruff` command reports `No module named ruff`. The documented installed
binary, `C:\Python312\Scripts\ruff.exe`, reports eight whole-file baseline
findings on unchanged lines (one `F401` in `test_library_shell.py` and seven
`E721` findings in `library_screen.py`). The complete task surface passes when
only those two pre-existing rule classes are excluded. The branch diff check
initially found two trailing spaces in the design document's Task/Date metadata;
those spaces were removed and the check was rerun.

The full `pytest -q` run stopped during Windows collection after 225.85 seconds:
`Tests/Media_Playback/test_player_pipeline.py` references unavailable
`signal.SIGSTOP`/`SIGCONT`, and
`Tests/TTS/test_profile_reference_materialization.py` imports unavailable
`fcntl`. An exact two-module rerun reproduced both errors in 0.68 seconds, and
both files are byte-identical to `origin/dev`; these are unrelated Windows
baseline failures rather than task regressions. Existing pytest cache-permission,
pytest-asyncio loop-scope, SQLite privacy, optional-dependency, and missing
OpenAI-TTS-mapping warnings also remain attributable to the environment/baseline.

#### Rendered keyboard UAT

No configured real-data Library profile was used. The UAT therefore mounted the
real `LibraryScreen` with the production stylesheet and deterministic media
fixture through the existing product-path harness, plus the real legacy
`MediaViewerPanel`; it drove keyboard events and used
`screen._compositor.render_strips()` as the visual oracle. The ephemeral UAT
driver was removed after capture and is not part of the implementation.

- Library terminal: 170x48. Fixture: 49,288 characters, 2,000 lines, 101
  `budget` matches. Markdown opened Rendered. Typing `budget` left the applied
  query blank until Enter; Enter retained search focus. Previous retained its
  focus, wrapped 1 to 101, and scrolled the content body from y=0 to y=404;
  Next retained its focus, wrapped 101 to 1, and scrolled back to y=0. Raw
  visibly rendered the literal `# Large budget document`, and its selected
  Rich span was `budget`. Four subsequent Raw/Rendered keyboard toggles retained
  the same viewer, Markdown, and Raw identities, retained content, and caused
  no additional Markdown update (one initial update total), so no remount/blank
  frame was observed.
- Library blocker: the status widget has region `(x=35, y=23, width=128,
  height=1)`, is displayed, and contains `Match 1 of 101 matches`, but the
  content body occupies `(x=35, y=20, width=129, height=18)` and paints its
  Markdown heading at terminal row 23. The compositor frame therefore contains
  neither the status text nor visible Previous/Next controls
  (`status_painted=False`, `previous_painted=False`, `next_painted=False`).
  Model-state assertions alone had hidden this overlap. Because visible status
  and navigation behavior are part of criterion #3, that criterion is not
  complete.
- Legacy Media terminal: 120x36. Fixture: 67 characters. A `budget` input burst
  produced no update during the silent window and exactly one highlighted
  repaint after debounce (observed elapsed 0.390 seconds including harness
  pauses). Clearing, typing another burst, and loading a replacement record
  before expiry produced the replacement content with zero stale highlighted
  repaint.

Closeout status: blocked on the rendered Library status/navigation overlap.
Task status remains In Progress; no layout fix was attempted in this verification
checkpoint.

### Task 5 final closeout after rendered-layout correction

The reviewed layout correction at `ae757d8d4` makes the focused search-controls
container size to its content, so the persistent media body starts below the
status and navigation chrome without changing the in-place ownership model. The
new 170x48 compositor regression passed and recorded these non-overlapping
regions:

```text
controls: (x=35, y=19, width=129, height=7)
status:   (x=35, y=23, width=129, height=1)
Previous: (x=36, y=25, width=16,  height=1)
Next:     (x=53, y=25, width=16,  height=1)
content:  (x=35, y=26, width=129, height=18)
heading:  row 29
```

The rendered frame visibly contained `Match 1 of 101 matches`, `◀ Prev`,
`Next ▶`, and `Large budget document`. Every control's bottom is at or above
content row 26, the heading remains within the content region, and the content
body retains its 3/18 row min/max bounds.

#### Final focused verification

The required commands were rerun from the isolated worktree with the parent
Python 3.12 virtual environment:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_library_shell.py::test_library_shell_media_viewer_inplace_search_chrome_paints_above_content -v -s
# 1 passed

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py Tests/UI/test_media_viewer_content_search_debounce.py -v
# 16 passed

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/test_network_guard.py -q
# 12 passed, 1 skipped (AF_UNIX unavailable on Windows)

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_library_shell.py -k 'media_content_search or media_viewer_raw_toggle or media_viewer_defaults_markdown or media_viewer_inplace' -v -s
# 18 passed, 546 deselected

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_media_window_v2_parity.py Tests/UI/test_media_handoffs.py -q
# 38 passed
```

The complete non-duplicated focused matrix is 84 passed with one expected
Windows platform skip. The fresh final 2,000-line submit/Next/Previous median is
`103.962 ms`, compared with `1091.689 ms` before the change. All four viewer
observations and all four Markdown observations retained one identity, with
`markdown_update_count=1` for initial construction only; navigation performed
zero Markdown updates.

The broad/static evidence is reproducible with these literal commands:

```powershell
& 'C:\Python312\Scripts\ruff.exe' check tldw_chatbook/Widgets/Library/library_media_content.py tldw_chatbook/Widgets/Library/library_media_viewer.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Media/media_viewer_panel.py Tests/network_guard.py Tests/test_network_guard.py Tests/Library/test_library_media_content.py Tests/UI/test_library_shell.py Tests/UI/test_media_viewer_content_search_debounce.py --output-format concise
# 8 attributed unchanged-line baseline findings: 1 F401 and 7 E721

& 'C:\Python312\Scripts\ruff.exe' check tldw_chatbook/Widgets/Library/library_media_content.py tldw_chatbook/Widgets/Library/library_media_viewer.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Media/media_viewer_panel.py Tests/network_guard.py Tests/test_network_guard.py Tests/Library/test_library_media_content.py Tests/UI/test_library_shell.py Tests/UI/test_media_viewer_content_search_debounce.py --ignore E721,F401
# All checks passed

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest -q
# Collection stopped after 225.85 seconds on the two Windows baseline errors below

& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Media_Playback/test_player_pipeline.py Tests/TTS/test_profile_reference_materialization.py -q
# Reproduced in 0.68 seconds: unavailable signal.SIGSTOP/SIGCONT and fcntl

git -c safe.directory='C:/Users/GDesktop-1/Working/Github/tldw_tui/.worktrees/task-15458-media-viewer-inplace' rev-parse origin/dev:Tests/Media_Playback/test_player_pipeline.py
git -c safe.directory='C:/Users/GDesktop-1/Working/Github/tldw_tui/.worktrees/task-15458-media-viewer-inplace' hash-object Tests/Media_Playback/test_player_pipeline.py
# Both: d8333dc722565ba4afeaced66e797b30f3729c25

git -c safe.directory='C:/Users/GDesktop-1/Working/Github/tldw_tui/.worktrees/task-15458-media-viewer-inplace' rev-parse origin/dev:Tests/TTS/test_profile_reference_materialization.py
git -c safe.directory='C:/Users/GDesktop-1/Working/Github/tldw_tui/.worktrees/task-15458-media-viewer-inplace' hash-object Tests/TTS/test_profile_reference_materialization.py
# Both: 0dd418c8d4c48ca713529d71e0bb173ef4eded88

git -c safe.directory='C:/Users/GDesktop-1/Working/Github/tldw_tui/.worktrees/task-15458-media-viewer-inplace' diff --check origin/dev...HEAD
# Passed
```

The previous full-suite Windows collection evidence remains applicable because
the layout correction cannot affect collection. No redundant full-suite rerun
was performed after that correction. Cache-permission, pytest-asyncio
loop-scope, SQLite privacy, optional-dependency, and missing mapping warnings
remain environment/baseline warnings.

#### Final rendered keyboard UAT

No configured real-data Library profile was available, so the final UAT used
the same mounted real `LibraryScreen`/production stylesheet and real legacy
`MediaViewerPanel` harnesses, drove keyboard events, and rendered compositor
frames. The ephemeral driver was removed afterward.

- Library at 170x48 used the 49,288-character, 2,000-line fixture with 101
  `budget` matches. Search remained unapplied until Enter and retained focus.
  Previous visibly wrapped `Match 1` to `Match 101`, retained focus, and
  scrolled y=0 to y=404; Next visibly wrapped back to `Match 1`, retained focus,
  and returned to y=0. The status, both buttons, and heading were painted in the
  non-overlapping regions above. Raw visibly showed `# Large budget document`
  with selected span `budget`. Four further Raw/Rendered keyboard toggles kept
  the same viewer, Markdown, Raw, Previous, and Next identities; content stayed
  present and the Markdown update count remained one.
- Legacy Media at 120x36 used the 67-character fixture. A rapid keyboard-event
  `budget` burst caused no update through the 0.10-second silent observation,
  then exactly one highlighted repaint (`1/3` visibly painted; 0.510 seconds
  observed including harness pauses). Loading a replacement during the next
  pending burst visibly painted the replacement and caused zero stale
  highlighted updates.

ADR-058 remains the governing decision for the thread-scoped Windows test
socketpair exemption; the layout correction introduced no new architecture
decision. Final self-review confirmed all four acceptance criteria, component,
product-path, network/default-deny, companion, performance, debounce, static,
diff, and rendered-UAT evidence. The implementation plan was followed, with the
interim compositor finding and subsequent reviewed geometry correction recorded
as the only deviation. The existing testing-evidence lesson already covers the
model-state-versus-painted-frame trap, so no duplicate lesson was added. With
the Windows full-suite baseline limitation explicitly attributed, all
task-scoped Definition of Done items are complete and the task is Done.

### macOS re-verification (2026-08-13) and the open-path double-parse it found

Re-verified this task's shipped behavior at dev `ebf56a763` — the first run of
its evidence on macOS, and the first since task-15457's Library reconciliation
merged on top of it (#1581 landed after #1586, touching `library_screen.py`).

The in-place conversion itself held: with the recorder scoped to the click, the
submit/Next/Prev sequence performs ZERO Markdown parses, keeps one viewer, one
Markdown, one status and both nav buttons, and retains focus. The legacy Media
panel's 250 ms generation-guarded debounce and the removed `update("")`
pre-parse also hold.

One test was red on macOS, deterministically (3/3):
`test_library_shell_media_viewer_inplace_large_document_latency_and_parse_proxy`
asserted a whole-session total of one `Markdown.update` and observed two. It
was reporting a real defect rather than a platform quirk. Opening a media item
issues two whole-screen recomposes — the "Loading media…" one at click time and
the detail worker's arrival one — and Textual's `recompose()` awaits child
teardown BEFORE calling `compose()`. A worker landing inside that await is
already picked up by the in-flight compose, so the arrival recompose parses the
same 49 KB / 2,000-line document a second time. Windows lost that race the
other way (the two refreshes coalesced), which is why the original evidence
recorded one parse.

Fix: `_refresh_library_media_detail` no longer recomposes unconditionally. It
defers one message via `call_next` to
`_recompose_library_media_detail_if_unrendered`, which skips the recompose only
when the composed viewer already rendered the exact current detail Mapping
(identity, recorded in `compose_content`; reset wherever
`_library_media_detail` is cleared, so a cached Mapping can never strand the
viewer on its loading line). Any other state — no detail, list view, a detail
no compose has rendered — still recomposes.

Open-click A/B on the 49 KB fixture, same process, three pairs: 922.970 /
914.755 / 935.161 ms with 2 parses before, 710.736 / 730.087 / 841.238 ms with
1 parse after. The match-navigation median on this hardware is 112.672 ms
(compare the pre-conversion 1091.689 ms recorded above).

Tests added to `Tests/UI/test_library_shell.py`:

- `..._detail_arrival_does_not_reparse_rendered_detail` — born red
  (`AttributeError: no attribute '_library_media_composed_detail'`); pins both
  directions of the guard (skips for an already-rendered detail, still
  recomposes when the compose has not rendered it).
- `..._inplace_navigation_holds_at_compact_size` — 80x24 companion to the
  existing 170x48 chrome test. At that size the nav row sits below the fold
  until focused (the viewer scrolls it into view; verified reachable), so this
  pins identity, focus, the advancing status text and zero reparse rather than
  a painted row.

Mutation evidence: disabling the guard's condition fails both the new arrival
pin and the latency pin; restoring `self.refresh(recompose=True)` in
`_advance_library_media_content_match` fails the new compact test at the viewer
identity assertion.

Not fixed here, recorded instead: at 80x24 the viewer's search status and
Prev/Next sit below the visible fold until focused. That is the viewer's
overall vertical density, not the in-place conversion (the controls container
is `height: auto` and the stack order is unchanged), and needs a layout
decision rather than a perf change.

Ruff on the touched files reports only the pre-existing `F401` in
`test_library_shell.py`. Lesson recorded in
`backlog/docs/lessons-testing-evidence.md` ("An absolute event-count pin
records which side of a race the author's machine won").
