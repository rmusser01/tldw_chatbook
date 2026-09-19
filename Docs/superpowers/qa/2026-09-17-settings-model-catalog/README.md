# Automatic model-refresh settings — TASK-32746

Automatic refresh previously ignored false config-save results and launched
independent writers for rapid edits. The regression gates reproduced an older
write overwriting the latest choice, including a return to the persisted value.
At 80 columns the startup checkbox extended beyond the painted viewport.

The existing Settings owner now retains raw form values and drains one latest
pending snapshot at a time. Revision checks keep old completions out of current
feedback. A visible saving/saved/failure receipt and keyboard Retry preserve
choices across category rebuilds; invalid or non-finite intervals explain the
recovery without writing. Fractional intervals are keyboard-editable. Scoped
existing-token styles and shorter labels keep complete control text visible.
No consent field, schema, dependency or global token value changes.

Existing ADR-020/031/150/161 apply; no new architectural decision is required.

## Automated evidence

**223 distinct targeted cases passed**, from three disjoint selections:

- [24 settings cases](targeted.txt): the original 13 new regression cases plus
  11 existing automatic-refresh behavior/layout cases.
- [6 added label-paint cases](labels.txt): all checkboxes traversed with Tab and
  full label text inspected at 170×48, 120×35 and 80×24 in both themes. The
  deselected 13 cases are the ones counted above, not extra coverage.
- [193 adjacent/governance cases](regressions.txt): model discovery, catalog
  settings/defaults, consent/startup offline behavior, token/component governance,
  Python style inventory, bundle sync, gallery layout/snapshots, generated CSS
  comments and boot byte budget.

[Initial red evidence](initial-red.txt): 13 failed before production edits.
Failures established missing recovery/validation receipts, last-write races and
compact clipping. The first intermediate green run (9 non-keyboard cases) is
not added to the total. One initial regression command referenced a nonexistent
filename and ran zero tests; the corrected selection above passed.

The boot census is **616,683 / 634,050 bytes** (17,367 bytes headroom). The
600,000-byte anti-vacuity floor and zero-literal/style ratchets remain unchanged.
Warnings are the existing inventory syntax warnings and the informational budget
warning. [Static checks](static-checks.json) show full Ruff/format/syntax success
for new Python files, and no added Settings diagnostics (118 baseline → 117).
Only changed Settings ranges were formatted; whole-file lint debt remains.
[Independent review](review.json) found no actionable issue. No full suite ran.

## Native evidence

The [runner](native_check.py) drove real `TldwCli.run(auto_pilot=...)` through a
dedicated tmux terminal and validated private profile:
`/private/tmp/tldw-32746-catalog-native-001`.
It asserted LinuxDriver, TTY rendering streams and profile-lock ownership; the
real terminal-capability probe ran before app startup. Dark/light × 170×48/80×24
completed **four journeys**, with eight captures inspected in one batch.

Each cell toggled startup refresh, received one deliberately injected `False`
from the writer, verified the unchanged file, navigated away/back, and activated
Retry using Tab/Enter. Retry and later fractional-interval/provider toggles used
the **real config writer**. The runner compared the full parsed config after
writes: only `model_catalog` changed, and startup consent was unchanged.
24 writes were observed: four injected failures and twenty real successes.
This verifies negative UI handling of the writer result, not an operating-system
permission failure. Mounted tests separately cover exceptions and gated races.

[Result](result.json) includes source/runner hashes matching the verified tree.
[Capture manifest](capture-manifest.json) records raw and stored SVG hashes;
storage trims trailing whitespace only. [Lifecycle](lifecycle.json) confirms
normal return/exit 0, exact PID absence before terminal closure, eleven healthy
private databases, zero conversations/messages, unchanged default config/UI-state/
runtime-policy fingerprints, no error events and an empty faulthandler log.
The owned terminal is closed. No provider generation was requested; startup
refresh network behavior and external provider availability are not qualified by
these editing journeys.

| Theme/size | Failed save / Retry | Saved fractional interval |
| --- | --- | --- |
| Dark 170×48 | [capture](textual-dark-170x48-failed.svg) | [capture](textual-dark-170x48-saved.svg) |
| Dark 80×24 | [capture](textual-dark-80x24-failed.svg) | [capture](textual-dark-80x24-saved.svg) |
| Light 170×48 | [capture](textual-light-170x48-failed.svg) | [capture](textual-light-170x48-saved.svg) |
| Light 80×24 | [capture](textual-light-80x24-failed.svg) | [capture](textual-light-80x24-saved.svg) |

This closes the automatic-refresh slice only. The
[completion ledger](../../reports/2026-09-17-design-system-completion-audit.md)
retains the remaining Settings/destination reviews and current-dev integration.
