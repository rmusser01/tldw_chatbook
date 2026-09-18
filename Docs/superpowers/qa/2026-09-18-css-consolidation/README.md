# CSS consolidation recovery — TASK-32813

The 26 declarations found by TASK-32812 are consolidated or proven dead, with
targeted computed-style, performance and native evidence recorded below.
PR #2707 remains draft and requires its own visual review and merge approval.

## Change and boundaries

Twenty-five declarations now use the existing build-time `BUNDLED_CSS` stream,
including modal Screens whose original styles were defaults. They retain the
widget-default tier. The twenty-sixth declaration was a dead alias:
`RoleplayDraftRecoveryDialog` copied selectors naming its sibling
`RoleplayDraftNavigationDialog`, so Textual's owner scope made them match no
nodes. Removing the alias preserves that existing appearance; it does not
redesign the recovery dialog.

Generated widget and screen sheets omit author comments using the existing
quote-aware comment stripper. Python source comments and generated provenance
banners remain. Token comparisons cover the real source sheets and bundled
blocks. This pays for the added styles without increasing the byte ceiling.

Narrow per-owner classes give migrated common control selectors an indexed
subject. Three Backup Restore / Workspace Persona broad selectors remain to
cover inherited and internal controls. Three existing Tool Pack action rules
name their actual button IDs at unchanged specificity. The class additions can
change specificity, so parsed equivalence alone is insufficient: the opt-in
qualification plugin compares computed rules against the recorded original
styles on actual mounted consumers.

The standalone Backup Recovery launcher also registers its screen's styles at
Textual's native default tier. Its real `RecoveryApp` does not load the main
app bundle. The recovery subprocess test continues to patch the native
`textual.app.App.run`, and checks rendered log height and action width on the
real host. This compatibility path introduces no new loader or app imports.

## Evidence method

- [Original declarations](original-defaults.json) and
  [Tool Pack rules](original-toolpack-defaults.json) match commit
  `bc1b1215dd52e63f654e275a5aa3e8a2170bd897`.
- [Consumer selection](consumer-cases.txt) is an explicit targeted list. The
  [serial runner](run_consumer_cases.py) executes each exact collected case in
  a fresh profile selected before collection. It preserves the original test
  bodies and fixtures; no full suite is run.
- [Parity plugin](migration_parity.py) reconstructs the original default CSS,
  then uses upstream Textual cascade resolution at real consumer pause points.
  It compares base computed stylesheet rules for observed target descendants
  and pseudo states. It does not compare virtual component styles or unchanged
  inline styles, and does not claim coverage of unmounted states.
- The source-count probe proves all 15 shell routes mounted before simulating
  first-mount CSS registration for the selected modal classes. Its source
  count is not evidence that those dialogs were opened; mounted consumers and
  native captures supply that separate evidence.
- Native qualification uses real `TldwCli`, LinuxDriver and attached terminal
  streams. Dialogs receive explicit read-only fixture inputs. This verifies
  native paint, focus, disabled styling and cancellation, not product entry
  routes, remote providers or recovery approval effects.

## Failures retained

The [baseline source guard](source-ratchet-red.txt) found 26 declarations and
73 sources after the modal registration probe. The
[first selector attempt](selector-first-red.txt) counted 312 costly subjects;
[the second](selector-second-red.txt) counted 277 against the unchanged 274
ceiling. The final three Tool Pack rewrites address the remaining excess.

[Comment removal red](comment-paydown-red.txt) precedes the
[217-case token comparison](comment-token-parity.txt). Consumer runs also
caught previously bare test hosts that had not loaded consolidated sheets;
the [Buddy rerun](buddy-harness-green.txt) exercises its corrected host.

[Standalone recovery red](standalone-recovery-red.txt) caught the production
host's missing stylesheet. The [initial recovery rerun](standalone-recovery-first-green.txt)
passed both actual styled RecoveryApp cases, with two other cases still
hitting the legacy profile-lifetime guard. Those cases are included in the
isolated consumer matrix rather than excluded from qualification.

The [static guard run](static-guards.txt) passed 230 cases. Its final mounted
header case overlapped the consumer run and is repeated serially in the final
qualification; that original run is not serial app evidence.

## Verified results

- [Consumer results](consumer-final-results.json): 194 exact cases pass after
  isolated reruns; the first run's 14 failures and their corrections remain in
  this directory. These counts are not a full-suite result.
- [Final guards](final-guard-results.json) record the exact targeted cases and
  pytest setup/call/teardown outcomes. Route, source, selector, byte, generated
  artifact and test-host checks pass. Earlier incorrect Research sentinels and
  harness resolver failures are retained beside their passing replacements.
- [Mounted coverage](coverage-summary.json) covers all 26 original owners plus
  the three adjusted Tool Pack owners, with zero observed computed-rule
  mismatches. This is observed-state coverage, not every interaction/state on
  every owner. Pure tests are not counted as mounted evidence.
- [Boot bytes](byte-budget-green.txt): **583,097 / 608,090**, tightening the former
  634,050 ceiling. [Selector cost](selector-budget-green.txt): **274 / 274**.
  [Source probe](modal-source-green.txt): **45** after all 15 proven routes,
  **46** after selected modal registration and all migration targets. The
  source soft limit 56 and cliff 64 are unchanged; no allowlist exceptions added.
- [Native gallery](GALLERY.md): 28 captures, seven dialogs × two themes × two
  sizes, all rendered and visually inspected. Focused and disabled computed
  rules match the original CSS; screenshots precede the disabled check.
- [Native lifecycle](lifecycle-002.json): App.run returns normally, exit 0,
  PID independently absent, lock reacquired, ten healthy SQLite databases,
  zero conversation/message rows, unchanged defaults and empty faulthandler.
- [Independent review](independent-review.txt), [Ruff diagnostic comparison](lint-delta.json)
  and [changed-range formatting](format-check.json) retain their exact scope.
  Existing whole-file lint/format debt is not claimed clean.

The final inventory audit corrected two bare test hosts and taught its path
resolver to expand the actual `LibraryScreen.CSS_PATH`; four already-correct
hosts no longer produce false positives. It does not exempt those hosts.

## Remaining limits and follow-up

- **TASK-32815:** an original archive interaction passed its assertions but
  failed during teardown when Console footer focus queried an empty screen
  stack. Its isolated rerun passed. That rerun does not resolve the lifecycle
  race; the original failure remains in `consumer-red-176.txt`.
- **TASK-32816:** Appearance's original fixed-height layout clips Cancel at
  80×24 in both themes. The first native failure is retained. The final run
  focuses the visible filter, explicitly records the hidden Cancel row and
  checks style parity without claiming compact action visibility.
- Roleplay recovery's original sibling-scoped alias matched nothing. Its
  existing unstyled full-screen appearance is preserved and remains UI debt.
- Native inputs use directly constructed dialogs and read-only fixtures. No
  share/save/approval effect, normal entry route, remote provider or complete
  destination workflow is qualified by this capture run.

## CI assertion alignment — TASK-32814

Saved head `bc1b1215` failed Fast Lane solely on the stale ASCII `...` assertion
(1,151 passed / one failed). TASK-32812 already renders Rich's terminal-cell
ellipsis `…`. The test now expects that glyph; all other content/reachability
assertions remain intact. [Original failure](prior-head-ci-failure.txt) and
[exact passing case](mcp-ellipsis-green.txt) are retained. New-head remote checks
are separate from the local qualification above.

ADR required: no. Existing ADR-150, ADR-161 and ADR-097 govern this mechanical
consolidation, compatibility registration and unchanged performance boundaries.
