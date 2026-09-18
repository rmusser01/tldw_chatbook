# Workspace lifecycle — TASK-32773

Create, Rename, Set active, Archive, Undo and Restore as keep visible keyboard
context and report truthful outcomes. Create validation appears beside its
origin. A saved workspace with failed folder bindings offers **Retry folders**
and **Keep workspace**, locks already-saved fields and retains its identity.
Rename and Restore feedback stays beside the action; replacement controls keep
focus after activation, Undo and Restore. Archived rows wrap within the pane,
and workspace names render literally in confirmations and receipts. New Workspace
now follows the current theme, including readable error text.

Existing ADR-147 governs lifecycle semantics; ADR-150/161 govern presentation.
Storage, archive refusal, cancellation, delayed completion and receipt ownership
remain unchanged. No new ADR or token was required.

## Targeted checks

**172 distinct cases pass across the final and corrective runs:**
[13 UI journeys](journey-tests.txt), [24 existing create cases](create-modal-tests.txt)
and 135 related cases, including [nine pane cases](pane-tests.txt). The UI tests cover painted feedback,
retained invalid input, partial retry exactly once, Keep/Escape after a partial
create, literal confirmation and cancellation, visible replacement focus, and
actual dark/light surfaces. The [related Settings, async lifecycle, folder and CSS/governance run](related-and-governance.txt)
passed 127 cases and reported eight old pane-harness failures. That run had
collected the old test definitions before their fixture repair; the separate
final pane run passes all nine cases. The [final token/build check](derived-final.txt)
passes 13 overlapping cases. The totals count shared cases once.

The defects were reproduced before repairs: [initial visibility](initial-red.txt),
[partial create](partial-create-red.txt), [completion and literal name](completion-red.txt),
[Undo](undo-red.txt), and [theme surface](theme-red.txt). An initial theme probe
used an unavailable variable-map key; the linked corrected probe compares the
actual surface and fails on the hard-coded black background.

Existing Create/pane tests previously changed profile selection after imports
or clicked below the viewport. They now use the established private-profile
helper and reveal controls before real clicks or keyboard activation. All 24
create-test bodies are AST-identical apart from the visible-click helper; all
nine pane tests retain every original assertion. The Restore ownership fake
now includes the result setter used by production.

A broader CSS check still expected the Notes list introduction to stay two rows.
Its fallback, source and generated declarations match saved HEAD exactly;
TASK-32765 deliberately made the app list introduction grow while retaining the
editor/fallback budget. The [corrected guard](geometry-test.txt) pins that exact
exception. Library production code is unchanged.

[Static checks](static.txt) find no new Ruff diagnostics in changed legacy files;
the new journey and native runner are clean, scoped formatting and diff checks
pass. No full suite or provider request was run.

## Native visual review

The [runner](native_check.py) uses real TldwCli, LinuxDriver and TTY streams with a
fresh private HOME/config/data selected before imports. Four cells drive Create
validation/cancel, a real folder removed between Add and Create then restored for
retry, Rename refusal/success, activation, Archive cancel/confirm, Undo, a real
name collision and Restore as. Registry reads verify exactly one created identity,
one successful binding, literal names, and unchanged activation after recovery.
Controls are focused before real keyboard activation; exhaustive Tab traversal,
Persona auto-creation, imported profiles and project-context interviewing are not
qualified by this run.

| View | Dark compact | Light compact | Dark wide | Light wide |
| --- | --- | --- | --- | --- |
| Create duplicate name | ![dark 80x24](textual-dark-80x24-create-error.svg) | ![light 80x24](textual-light-80x24-create-error.svg) | ![dark 170x48](textual-dark-170x48-create-error.svg) | ![light 170x48](textual-light-170x48-create-error.svg) |
| Partial create recovery | ![dark 80x24](textual-dark-80x24-partial-create.svg) | ![light 80x24](textual-light-80x24-partial-create.svg) | ![dark 170x48](textual-dark-170x48-partial-create.svg) | ![light 170x48](textual-light-170x48-partial-create.svg) |
| Archive confirmation | ![dark 80x24](textual-dark-80x24-archive-confirmation.svg) | ![light 80x24](textual-light-80x24-archive-confirmation.svg) | ![dark 170x48](textual-dark-170x48-archive-confirmation.svg) | ![light 170x48](textual-light-170x48-archive-confirmation.svg) |
| Restore conflict | ![dark 80x24](textual-dark-80x24-restore-conflict.svg) | ![light 80x24](textual-light-80x24-restore-conflict.svg) | ![dark 170x48](textual-dark-170x48-restore-conflict.svg) | ![light 170x48](textual-light-170x48-restore-conflict.svg) |
| Restored workspace | ![dark 80x24](textual-dark-80x24-restored.svg) | ![light 80x24](textual-light-80x24-restored.svg) | ![dark 170x48](textual-dark-170x48-restored.svg) | ![light 170x48](textual-light-170x48-restored.svg) |

All 20 final SVGs were rendered and visually inspected. Run004, PID 33839,
passed four cells and returned after Ctrl+Q with exit 0. The [result](native-result.json),
[capture hashes](capture-manifest.json) and [lifecycle receipt](lifecycle.json)
record matching final source/runner hashes, 11 healthy private databases, zero
durable conversations/messages, no app errors or faulthandler output, a reacquired
instance lock, and unchanged default-profile fingerprints. The exact app PID was
absent before the owned terminal closed.

Earlier attempts remain distinguished: run001 left Create’s default switch
checkbox enabled, so the later Set active probe correctly found no button;
its failed receipt records normal shutdown. Run002 passed behavior but visual
inspection exposed the black light-theme modal; its pre-theme receipts and two
captures retain that evidence. Run003 put token references into a consolidated
widget stylesheet with a separate variable scope and failed before UI-ready.
Its failure receipt records exit 1, absent PID, healthy databases and unchanged
defaults, without claiming a successful native lifecycle. Final token rules live
in the app dialog module, with matching theme-variable standalone fallback.

This qualifies the listed workspace lifecycle flows. Native imported-profile
review, Console agent-turn diff/review/revert and the remaining Settings and
destination reviews remain open in the completion ledger.
