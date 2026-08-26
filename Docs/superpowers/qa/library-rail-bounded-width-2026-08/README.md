# Library rail bounded-width QA evidence

Status: **targeted automated gates and detached-PTY UAT complete; full repository
sweep not run because no user opt-in was received**

This directory records Task 22301 verification against the approved design and
ADR-086. Initial evidence was gathered from `codex/library-rail-bounded-width`
at `bd4da97d1e0bece92f9c4375ed172da7b94f7c86` on 2026-08-26. Final static and
changed-test evidence was gathered at
`b0bb0eb1c64f40a73796e27d7e46022060846b08`. Detached-PTY responsive
acceptance continued through startup repair
`86ad27385c3e3a6af043caa37f3e8f6d5c7ad35a` and keyboard-focus repair
`28ce0a88a0e12c88b8c5975588ba24dad7b18e99`. Final Settings-truth acceptance
ran at `2283283fd89e9bb67a10f33e74d8653ce2111582`. No browser, CUA, iTerm2,
or Windows Terminal acceptance is claimed here.

## Automated verification

The exact Task 8 static-analysis commands were run against the production and
test paths listed in the implementation plan.

| Gate | Result | Evidence |
|---|---|---|
| `ruff check` | PASS | `All checks passed!` (0.04 s real) |
| `ruff format --check` | PASS | 20 files already formatted (0.02 s real) |
| `compileall -q` | PASS | exit 0 (0.02 s real) |
| `git diff --check` | PASS | exit 0 (0.02 s real) |

The first format run exposed five branch-introduced violations:

- `Tests/UI/test_settings_appearance_defaults.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/test_config_library_defaults.py`
- `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`

Each corresponding blob at merge base
`415344e113d99d711dce53035db813840eaa53d2` passed Ruff's stdin format check,
so the initial red gate was branch-introduced. Commit `b0bb0eb1c6` repaired it;
the exact four-command static gate then passed in full.

## Focused regression suite

The exact 13-module command collected 1,505 tests. It was interrupted after
1,158.35 s at 23% because it had become a repeated roughly 30-second selector
timeout cascade (38 visible failures) and did not produce a usable final pytest
summary. The modules were then isolated with fail-fast runs to distinguish
branch failures from merge-base failures.

| Module or group | Result | Elapsed / classification |
|---|---|---|
| three pure `Tests/Library` policy/state modules | 132 passed | 4.37 s pytest; 5.12 s real |
| `Tests/UI/test_library_shell.py` | first failure after 21 passes | 10.14 s pytest; failure reproduced identically at merge base |
| adaptive + media reader shell modules | initial 58 passed, 1 failed | 69.51 s pytest; media failure repaired and exactly rerun |
| `Tests/UI/test_library_notes_reader.py` | 12 passed | 23.70 s pytest; 26.25 s real |
| `Tests/UI/test_library_conversation_reader.py` | 51 passed | 57.61 s pytest; 61.40 s real |
| `Tests/UI/test_library_honesty_accessibility.py` | 30 passed | 15.92 s pytest; 18.12 s real |
| `Tests/UI/test_css_build_integrity.py` | 14 passed | 2.49 s pytest; 3.67 s real |
| `Tests/test_config_library_defaults.py` | 14 passed | 0.65 s pytest; 1.26 s real |
| `Tests/UI/test_settings_appearance_defaults.py` | 17 passed | 1.86 s pytest; 2.76 s real |
| `Tests/UI/test_settings_configuration_hub.py` | first failure after 28 passes | 7.60 s pytest; failure reproduced identically at merge base |

Exact classified failures:

1. **Branch-introduced, repaired and exactly rerun:**
   `Tests/UI/test_library_media_reader_shell.py::test_media_shell_mounts_library_items_reader_and_two_five_column_grips`
   expected `#library-rail-collapse` to be hidden at `170x48`, but its
   `display` property was true. The same node passed at the merge base (1 passed
   in 3.49 s). After `15588d164f`, the node passed in 3.30 s (4.80 s real).
2. **Merge-base baseline:**
   `Tests/UI/test_library_shell.py::test_library_full_lifecycle_landing_preserves_counts_search_and_recents[expanded]`
   raised `NoMatches` for `#library-hub-recents` on both branch and merge base.
3. **Merge-base baseline:**
   `Tests/UI/test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries`
   reported the same ownership tuple mismatch on branch and merge base.

### Consolidated Task 1–7 change selection

After the media and formatting repairs, one bounded consolidated run selected
all test functions whose current AST spans a changed `origin/dev...HEAD` hunk,
plus the exact repaired media-shell regression. This produced 134 explicit
function node IDs and expanded to 257 parameterized cases.

Result: **257 passed in 247.69 s (4:07 pytest; 251.97 s real)**. Every node was
bounded by `--timeout=90`; the slowest node took 10.27 s. This run does not
silently convert the two merge-base failures above into passes and does not
stand in for the optional full repository sweep.

Pytest also repeatedly warned that it could not clean stale, permission-denied
directories below its system temporary root. Using an isolated
`PYTEST_DEBUG_TEMPROOT` removed that output noise and did not change the
classified Settings failure.

## Full repository sweep

Not run. Task 8 requires explicit user opt-in, and this verification assignment
was limited to the focused suite. A full sweep is not a substitute for repairing
and rerunning the red focused gates.

## Detached tmux PTY UAT

**Complete.** Six isolated sessions ran on tmux socket `tldw22301` at 235, 170,
120, 100, 80, and 60 columns by 52 rows. Before
trusting captures, every pane reported the feature
worktree as its current path, `python3.12` as its live command, its requested
pane dimensions, and `dead=0`. All use the shared scratch profile under
`/tmp/tldw-task22301`; no user profile is involved.

The initial automatic-mode landing observations before startup fix
`86ad27385c` were:

- 235: grid 231, rail 34 + canvas 197 — matches the approved projection.
- 170: grid 166, rail 31 + canvas 135 — matches the approved projection.
- 120: grid 116, rail 24 + canvas 92 — matches the approved projection.
- 100: grid 100, **rail-only fills 100; canvas is absent** — FAILED because
  ordinary co-presence is required at every width of 64 or more.
- 80: grid 80, **rail-only fills 80; canvas is absent** — FAILED for the same
  reason.
- 60: emergency rail-only fills 60 before a route is activated.

Both failed captures visibly show the Library landing with no selected rail row,
the `Search Library…` focus treatment, `Library | Local` header, and no canvas.
The scratch config still has
`library.media_reader.custom_widths_enabled=false` and `library_width=31`.
Exact panes and provenance are preserved in `landing-100.txt` and
`landing-80.txt`; neither is accepted as a pass. Those original processes were
left untouched until the failure was diagnosed, then replaced for post-fix
verification as described below.

After `86ad27385c`, only the 100/80 sessions were killed and relaunched with the
same literal environment, scratch profile, width, and 52-row height. Provenance
again showed the exact feature worktree, `python3.12`, requested pane size, and
`dead=0`. Both post-fix landings now keep the 24-cell rail alongside visible
canvas content: 76 cells at width 100 and 56 cells at width 80. As designed for
the compact tier, the outer grid/canvas border is absent, so the grid equals the
full terminal width. Exact accepted panes are preserved in
`landing-100-postfix.txt` and `landing-80-postfix.txt`; the earlier failure files
remain as regression evidence.

The 170-cell route pass successfully activated Collections through rail
Tab/Enter navigation, then recovered focus to the selected rail row with Escape
and opened ordinary Search/RAG with Down/Enter. The 31-cell rail remained stable
with an intact 135-cell canvas. Keyboard-only Media, Chats, and Notes captures
each measured the same adaptive grid: Library 31, first grip 5, Items 40, second
grip 5, and Work 85, for the 166-cell production grid. Exact 170-cell panes are
preserved in `collections-170.txt`, `search-rag-170.txt`, `media-170.txt`,
`chats-170.txt`, and `notes-170.txt`.

### Custom-width production Settings pass

All custom values were changed through F9 Settings > Appearance, never by
editing TOML. The deterministic keyboard route used `/` field search with the
exact indexed labels `Shared Library rail width mode` and
`Preferred Library rail width`, followed by Enter, Ctrl+A, the requested value,
Escape, and `s`. Settings visibly reported `Appearance defaults saved.` and
`No unsaved changes`.

- Saved 35: W74 rendered rail 34 + canvas 40; W75 rendered rail 35 + canvas 40.
- Saved 48: W64 rendered rail 24 + canvas 40; W80 rendered rail 40 + canvas 40;
  W87 rendered rail 47 + canvas 40; W88 rendered rail 48 + canvas 40.
- After every resize boundary, reopening the production Settings field showed
  Custom widths / 48 and `No unsaved changes`; effective compression never
  overwrote the preference.

### Emergency-stage and ASCII pass

At W60, `i` activated the real Import route from the Library landing and
produced a canvas-only stage with one pinned `‹ Library` control and no reserved
rail space. In the safe route state, Escape used the screen-owned guarded action
and returned to rail-only. Resizing 60 -> 64 -> 60 restored the active route at
64 and the prior rail-only emergency stage at 60; the saved width remained 48.

The production ASCII control is not present in Settings' field-search index, so
keyboard navigation used indexed `Animations`, then two Tabs to `ASCII glyphs`.
After enabling and saving it, the W60 canvas visibly rendered `< Library`.
Final Settings verification showed `ASCII glyphs Enabled`, Custom widths,
Preferred rail width 48, and `No unsaved changes`.

Pre-fix detached-PTY failure: on `86ad27385c`, Shift+Tab, F6, and Shift+F6 did
not make Enter activate the pinned return button, and one bounded forward Tab
moved from the path field to Browse. This is preserved as failed evidence, not
silently converted to a pass.

Post-fix detached-PTY pass: after `28ce0a88a0`, only the W60 session was killed
and relaunched with the same literal environment, scratch profile, and size.
Provenance again reported pane `60x52`, the feature worktree, `python3.12`, and
`dead=0`. Palette Library -> `i` opened the real Import canvas and visibly
rendered `< Library`; Shift+Tab followed by Enter returned to rail-only.
Resizing 60 -> 64 restored the active Import route and resizing back to 60
restored rail-only. Settings still showed Custom widths / Preferred rail width
48 / `No unsaved changes`.

Pre-fix Settings-truth failure: the relaunched app at `28ce0a88a0` visibly
rendered `< Library` and the isolated config contained
`appearance.ascii_glyphs = true`, but F9 Settings labeled `ASCII glyphs`
`Disabled`. This mismatch is retained as failed evidence.

Post-fix Settings-truth pass: after `2283283fd8`, W60 was restarted once more
with the same literal profile and environment. In that same process, F9
Settings > Appearance showed `ASCII glyphs Enabled`, Custom widths, Preferred
rail width 48, and `No unsaved changes`. Returning through the palette to
Library and activating Import at W60 visibly rendered `< Library`; Shift+Tab
then Enter returned to rail-only. Provenance was again pane `60x52`, the feature
worktree, `python3.12`, and `dead=0`.

After final capture, `tmux -L tldw22301 kill-server` terminated all six
sessions. A subsequent `has-session` check reported the server/socket absent,
and process inspection found no surviving Task 22301 app or tmux process (only
the inspection command itself).

Pointer activation remains production-mounted Pilot coverage unless a real SGR
mouse event is deliberately injected and documented. Keyboard-driven detached
tmux evidence must not be described as pointer, iTerm2, or Windows Terminal UAT.
