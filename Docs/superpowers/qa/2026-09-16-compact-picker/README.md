# Compact folder picker — TASK-32665

Reviewed on `feat/component-pattern-library`, based on `254e9f527f`.

At 80×24 the folder dialog allocated two rows to DirectoryNavigation, both
consumed by its border. Loaded folders therefore had no painted rows. Fixed
chrome also left the typed-path input only 15 columns wide. The compact class
now gives the dialog more terminal space and reduces breadcrumb/input-bar
overhead using existing tokens. Folder rows, the path and Select/Cancel remain
visible. Resizing changes classes while retaining the mounted controls.

Focused review found three validation edges under the same acceptance criterion:
a long missing path consumed every listing row, correcting to the current folder
left a stale error, and an overlong name raised from filesystem metadata probes.
Compact error display now ellipsizes while retaining the full diagnostic in the
widget; the error uses the readable status token in both sizes. Successful path
validation clears the error before assigning the location, including unchanged
locations. Metadata probes now sit inside the existing exception handling.

ADR required: no. Existing
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md),
[160](../../../../backlog/decisions/160-progressive-file-picker-listings.md) and
[161](../../../../backlog/decisions/161-component-pattern-library.md) apply.
No listing authority, token value or picker result contract changed.

## Automated evidence

[Verification commands and log hashes](verification.json) record **154 distinct
passing checks**: 15 new picker cases, 122 neighboring picker/governance checks,
16 Library caller journeys and one CSS budget check. The picker cases load
production app CSS at 80×24 and 170×48 in both themes, browse real temporary
folders with the keyboard, paint empty and scrolled listings, validate paths,
Select/Cancel and retain path, highlight, focus and selection on resize.

The initial four layout failures and five later validation failures are recorded
before their repairs. Empty/scrolling fixtures were moved into a dedicated
listing directory after inherited fixture setup added siblings under `tmp_path`.
The final combined run had 152 passes and one existing Library setup race:
the audio group mounted before its provider Select. Waiting for that actual
control, without increasing the wait budget, passes all 16 caller cases.

[Static comparison](static-comparison.json) adds zero Ruff diagnostics; 38
inherited diagnostics remain in the shared base dialog. New Python and the
adjusted caller test pass full Ruff and formatting; modified production ranges
are [formatted](format-ranges.json). [Sizes](size-comparison.json) record boot CSS
at 615,348 of 634,050 bytes, with 18,702 bytes remaining and no raised budget.
Source CSS was rebuilt and bundle reproducibility passes. The final
[independent review](review.md) found no actionable issue. No full suite was run.
Inherited governance escape-sequence and pytest temporary-directory cleanup
warnings remain; the budget warning reports headroom.

## Native inspection

[native_check.py](native_check.py) ran actual TldwCli/LinuxDriver through
Library → Import → Parakeet Browse in an [exclusive private profile](isolation.json).
The environment lacks `audio_processing` and `parakeet_onnx`; only the widget's
availability probe was simulated to enable the folder controls. The real picker
starts inside the private source directory. Navigation, listing, validation and
selection use the actual implementation. Paths are assigned to inputs and
keyboard actions activate parent/child navigation, Select and Cancel.

Six run-001 captures were rendered and inspected together. After the validation
repairs, four run-002 error captures were inspected together as confirmation.
All linked captures are from the final run-002:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Loaded folders | [capture](listing-170.svg) | [capture](listing-80.svg) |
| Empty folder, parent available | [capture](empty-170.svg) | [capture](empty-80.svg) |
| Missing-folder validation | [capture](error-170.svg) | [capture](error-80.svg) |
| Long missing path | — | [capture](long-error-80.svg) |
| Overlong component | — | [capture](overlong-80.svg) |

[Results](result.json) confirm Select/Cancel preserve the import draft and title
cursor at both sizes. The compact step also resizes the live modal 80×24 →
170×48 → 80×24, preserving typed path, selection, highlighted folder and focus.
The wide step's `resize_preserves_path_selection_and_focus: false` and
`long_path_checks: false` mean those checks run in the compact step only;
they do not represent failures. This evidence covers the macOS terminal path,
not the Windows drive pane.

[Read-only persistence](persistence.json) confirms ten healthy SQLite databases,
zero media/messages/ingest jobs, unchanged synthetic source bytes, the exact
three-entry source tree, empty model folder and no ERROR/CRITICAL log lines.
Normal terminal Quit returned to an observed shell with exit 0; the owned tmux
session was closed. No installation, ingestion, model execution or remote request
was performed, and no production profile was used.

## Remaining review

Continue remaining per-type ingest options, then queue activity and recovery.
Actual import execution, installation, remote/provider actions and restart remain
outside this evidence. No push or integration into `dev`.
