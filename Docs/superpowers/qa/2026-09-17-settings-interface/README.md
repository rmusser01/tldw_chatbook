# Interface Settings — TASK-32757

Base: `11155c15c1ec4ded85d8a8fd63b6f8b2204773b1`, draft PR #2704 against
`dev`. This bounded review covers Appearance drafts, Theme editing and launch
defaults, and Splash instant settings and previews.

ADR required: no. Existing ADR-033 (Settings commit models), ADR-150 (design
tokens) and ADR-161 (component patterns) govern these repairs.

## Confirmed defects and repairs

- Theme and Splash outer containers allocated one fractional height and clipped
  their longer forms. They now grow with their contents so the Settings detail
  pane can scroll every action into view. Compact Theme labels leave enough
  width for the full hex value and swatch, and its tree fits the viewport.
- Appearance and Splash checkboxes inherited three-row chrome inside one-row
  fields. Scoped compact token styles restore visible checkbox state. Compact
  Appearance fields, Splash's Default card field and its gallery stack to preserve usable controls.
- Splash animation speed was read from `splash_screen.effects` but saved in
  `splash_screen`. Reading and writing now use the same owning section.
- Splash previously published success before the write completed. It now keeps
  confirmed state separate from pending values, restores controls after failed
  file writes, and distinguishes saved-file/cache-refresh failure. It preserves
  focus, newer numeric text, latest status ownership and safe completion after
  leaving the view. Checkbox text agrees with its visible state.
- Theme's instant launch-default save left Appearance stale, allowing an
  unrelated Appearance Save to overwrite it. A saved-default message now
  rebases Appearance's baseline while retaining explicit draft values. Matching
  saved/staged choices clear the dirty category indicator. Partial refresh
  failures still publish the successfully saved preference.
- Theme guidance incorrectly said that theme editing never touched config.
  The UI and user guide now distinguish Apply, theme-file Save and the startup
  preference. Splash documentation describes the actual Default card selector.

The source stylesheet adds 72 lines using existing tokens. The generated boot
bundle grows by 317 bytes to **620,598**, below the unchanged 634,050 ceiling
and above the 600,000 floor; see `boot-css.json`. No token values changed.

## Test fixtures and verification

The Theme and Splash tests now select a private profile before imports, using
the existing exact-node subprocess helper. Original behavioral assertions remain.
Theme render fixtures no longer retarget an already-bound storage owner.
The preset keyboard test loads production styles: without them its class-backed
swatch was transparent and read `#00000000`. An intermediate CSS_PATH tuple
was rejected by Textual; a typed list fixed that test fixture.

The final affected selection comprises 98 distinct cases: 18 Interface keyboard
journeys, 18 Splash cases, 41 Theme cases, three Theme render cases and 18
Appearance defaults cases. Receipts are counted by unique case, not summed
across overlapping reruns:

- `affected-initial.txt`: 94 passed and the single CSS_PATH fixture failure
  described above (95 cases before the final three regressions were added).
- `preset-final.txt`: the corrected preset case passes.
- `splash-final.txt`: all 18 final Splash cases pass, including the two newly
  added newer-input success/failure cases.
- `journeys-final.txt`: final keyboard, persistence, launch-default overlap,
  rendered values and resize qualification; includes the new dirty-marker case.
- `default-card-paint-final.txt`: four Splash geometry cases repeated after the
  final compact selector repair.
- `governance.txt`: 32 token, component, generated CSS and boot-budget cases.
- `static.json` and `static-fatal.txt`: zero new scoped Ruff/formatter debt
  against the exact base; fatal checks pass. Existing whole-file debt remains.
- `preflight.txt`: derived-artifact guard results. The diagnostic inventory
  refresh records one added constant shutdown debug message and one reworded
  Splash error retaining the existing key/exception arguments; no new sink or
  user-content argument was introduced.

Initial failures are retained in `geometry-red.txt`, `compact-values-red.txt`,
`appearance-theme-handoff-red.txt` and `pending-input-red.txt`. Independent
review findings and disposition are in `review.txt`.

## Native verification

Final run `/private/tmp/tldw-32757-native-004`, PID 77029, passed all four
journeys starting in dark/light themes at 190×55 and 80×24. It used real
`TldwCli.run`, LinuxDriver, attached tmux streams, a private HOME/config/data
profile and an acquired instance lock. Appearance validation/Preview/Revert,
actual saves, Theme browse/edit/Apply/theme-file Save/launch-default handoff,
and Splash exact-value writes/reload/gallery replay all passed. Apply and
Appearance Preview intentionally change the runtime theme during each journey.

All 28 SVG captures and matching terminal text were inspected, including the
complete compact `minimal_fade` selection. `native-result.json` fingerprints
match the final Settings sources, stylesheets and runner. `lifecycle.json`
records app.run return, exit zero, exact PID absence before terminal closure,
11 healthy private databases, zero conversations/messages, no app ERROR records,
empty faulthandler output and unchanged default-profile file fingerprints.
The owned terminal is closed. `capture-manifest.json` retains original/stored
hashes; only trailing whitespace was removed. Ingestion integration has separate
automated evidence below; this native run exercises the Settings surfaces.

Earlier attempt 001 exited 1 because the runner selected nonexistent card IDs;
its initial terminal resize also made its first wide capture unsuitable. Attempt
002 completed all four cells and exited zero, but predates the newer-input
preservation fix. Attempt 003 was stopped normally after its compact Splash
capture exposed a truncated Default card selection; its pending autopilot then
failed to reach the next Theme field and exited 1. The new compositor regression
reproduced `Random` painting as `Rand`; the field now stacks at full width.
`compact-splash-before.svg` and `default-card-paint-red.txt` retain that evidence.
None substitutes for the final source-matched run.
`earlier-native-*.json` retain their outcomes and confirmed exits before relaunch.

## Incoming dev integration

Dev advanced by four commits through `c97a64eba5` during this review. The sole
text conflict was the Import guide: both the existing resize/retry guidance and
incoming Parakeet/runtime/clipboard guidance are retained. Incoming provider
validation and asynchronous checked clipboard delivery remain present.

Independent review found a semantic interaction in the automatic canvas merge:
new Select errors were created on compose, while this branch updates option
fields in place. Choosing Auto repaired state but left the old invalid-provider
warning visible. Select error lines now remain mounted and update with their
placeholder options; programmatic option replacement suppresses Changed events.
The strengthened incoming test proves repair, invalid/valid snapshot transitions,
retained editor identity and absence of extra edits. `integration-select-red.txt`
records the actual stale-warning assertion; `integration-select-final.txt` passes.

The initial five-file integration run had 51 passes and four known profile-setup
failures. Isolation wrappers let all 55 original cases run. The expanded selection
also exercises existing option identity and event ordering. Its 20 setup failures
all report `raw_source_selection_changed`; four functions now use the same
pre-import private-profile helper without altering their behavioral assertions.
**102 distinct integration cases pass**: `integration-expanded-initial.txt`
contains 82 passes and the 20 setup failures; `options-private-final.txt` passes
all 20 after isolation. Overlapping five-file and single-case reruns are not
added to this total. Together with the 98 Settings and 32 governance cases,
this continuation qualifies **232 distinct targeted cases**.

`integration-static.json` compares both merge parents and reports no additional
lint/format debt. `integration-fatal.txt` passes. All seven merged preflight guards
pass; the diagnostic-inventory and five CSS-sync checks were also repeated after
the Select synchronization repair. No new architecture decision was required.

## Limits

Native checks use private synthetic configuration and only local settings/theme
file operations. Duration and speed persist, but the preview fixture has reduced
motion enabled: static representative `default` and `minimal_fade` cards qualify
gallery selection and replay, not the timing or rendering of every animation.
Failure injection and delayed-write interleavings belong to the mounted tests,
not the native happy-path run. No external provider, speech, network service or
user profile is exercised. The full test suite was not run. Other Settings
categories and destination reviews remain in the completion ledger.
