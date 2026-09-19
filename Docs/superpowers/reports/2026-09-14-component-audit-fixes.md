# Component audit fixes — 2026-09-14

The five follow-ups from the [first component audit](2026-09-14-component-first-ui-audit.md)
are complete on `feat/component-pattern-library`. This closes the shared-component
repair pass; Library and the remaining feature workflows still need their own audits.

## Changes

| Finding | Result | Backlog |
|---|---|---|
| Current dev integration | Merged `fd30614dcdc` into the design branch in `acfdc333da`; preserved all 89 upstream style declarations and repaired the split-style test harnesses. | TASK-32591 |
| More menu | Keyboard traversal scrolls every destination into view at 80×24. Resizing an open menu preserves visible focus; selection and Escape return focus correctly. Committed in `bb15283147`. | TASK-32592 |
| Compact Settings | Providers Connect and Network stack complete labels above full-width controls at 100 columns or fewer. The default sidebar stays visible; wide layouts keep their existing label column. | TASK-32593 |
| Gallery form | Temperature uses the remaining inline width, preserving its value and every field edge at 80/120 columns in both themes. The catalog demonstrates the same composition. | TASK-32594 |
| Documentation and whitespace | Corrected canonical/scoped heading ownership, Settings ownership, and the earlier verification scope. Source and generated snapshot whitespace now reproduce cleanly. | TASK-32595 |

No existing token value changed. Existing
[ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../backlog/decisions/161-component-pattern-library.md) govern these
repairs; no new architectural decision was needed. Integration also renumbered the
younger component-pattern family to TASK-32596 without changing the older dev task
that already owned TASK-32532; see the [integration report](2026-09-14-component-integration.md).

## Verification

Targeted runs passed; the selections are separate and should not be summed into a
unique-test total.

| Scope | Result | Evidence |
|---|---|---|
| More traversal, routing, return focus, dark/light and resize | 52 passed | [Navigation tests](../qa/2026-09-14-component-fixes/navigation-tests.txt) |
| Compact Settings: readable cells, full policy names, typing, value/focus retention and invalid-path validation | 6 passed | [Compact tests](../qa/2026-09-14-component-fixes/settings-compact-tests.txt) |
| Startup CSS freshness, multi-source splits, missing outputs, partial trees and content manifests | 18 passed | [Startup checks](../qa/2026-09-14-component-fixes/startup-css-tests.txt) |
| Adjacent Settings layout, Network and invalid-field focus | 10 passed | [Adjacent tests](../qa/2026-09-14-component-fixes/settings-adjacent-tests.txt) |
| Gallery containment, value and painted edges in both themes at 80/120 columns, rest/focus | 8 passed | [Gallery tests](../qa/2026-09-14-component-fixes/gallery-layout-tests.txt) |
| Component/token governance, bundle reproduction, boot byte budget and both gallery snapshots | 34 passed | [Shared checks](../qa/2026-09-14-component-fixes/shared-checks.txt) |

The new behavioral regressions failed before their fixes. All generated styles
[reproduce from source](../qa/2026-09-14-component-fixes/bundle-check.txt).
Boot-parsed CSS is **613,053 / 634,050 bytes**, with no budget increase. Fatal Ruff
checks (`E9,F63,F7,F82`) and formatting of the changed tests pass. The full design-branch comparison
against the integrated `origin/dev` passes `git diff --check`; this scope includes
the earlier design work, unlike the original uncommitted-diff check.

The final shared run emitted existing AST escape, temporary-directory cleanup and
byte-headroom warnings. No full suite was run. Independent bounded review found no
remaining actionable issue in these fixes; review and integration findings are
documented in the earlier integration report.

The final native exit-log check exposed a missed integration caller:
`_generated_css_is_stale` still accessed `split.module` after the registry changed
to `split.modules`. Startup caught the exception and rendered committed styles,
so visible UI alone did not reveal the broken automatic freshness check. The
caller now requires all split sources before requiring its generated sheets.
Three new complete/partial-tree cases reproduced the exception before the fix;
the entire 18-test freshness module now passes. A fresh native run rendered and
exited with status 0 without the CSS error; its
[terminal transcript](../qa/2026-09-14-component-fixes/startup-fixed-exit.txt)
retains the unrelated optional-package warnings. TASK-32591 was reopened for this
repair and then closed with the added acceptance criterion verified.

## Live and visual evidence

The actual app ran with an isolated scratch profile and database paths. Native
terminal checks covered More's last destinations at 80×24, Providers Model focus
and Endpoint rendering at 80×24 with restoration at 120×45, and Network policy/path rendering
at 80×24 in dark and light themes. No external model request was sent. All three
policy values, typed-value retention, validation and the full dark/light resize
matrix were exercised by the mounted tests, not by a live model conversation.

- [More final destinations](../qa/2026-09-14-component-fixes/more-last-80.ansi)
- [Providers compact](../qa/2026-09-14-component-fixes/settings-providers-dark-80.ansi)
  and [wide restoration](../qa/2026-09-14-component-fixes/settings-providers-dark-120.ansi)
- [Network dark](../qa/2026-09-14-component-fixes/settings-network-path-dark-80.ansi)
  and [Network light](../qa/2026-09-14-component-fixes/settings-network-light-80.ansi)
- [Gallery geometry](../qa/2026-09-14-component-fixes/gallery-geometry.json),
  [80-column dark SVG](../qa/2026-09-14-component-fixes/gallery-textual-dark-80-focus.svg)
  and [120-column light SVG](../qa/2026-09-14-component-fixes/gallery-textual-light-120-rest.svg).
  Companion PNGs are Quick Look previews of those Textual exports; font fallback
  affects some unrelated glyphs, so containment evidence also checks compositor
  output and geometry. Snapshot blank-line normalization preserves SVG semantics;
  the Temperature edge correction is the intentional visual change. Generated
  terminal IDs change with the snapshot content; this
  [comparison normalizes those IDs](../qa/2026-09-14-component-fixes/gallery-snapshot-visual-diff.txt)
  to expose the small rendering change beneath the larger fixture diff.

## Remaining review

Next is Library: Workspaces → Notes/Media/Conversations → Search/RAG, with populated,
empty, loading, error, cancellation and return-focus states. Subsequent passes cover
remaining Settings categories and the other destinations listed in the first audit.

Two observations remain for dedicated checks: the optional detected-server setup
action below the initial 80×24 Console viewport (integration report), and Network's
default policy label truncating at 120×45 when the Scope Inspector takes horizontal
space. The latter is visible in the
[wide Network capture](../qa/2026-09-14-component-fixes/settings-network-light-120-follow-up.ansi);
the compact repair preserves the existing wide layout. Neither observation is
claimed fixed or fully assessed by this pass.
