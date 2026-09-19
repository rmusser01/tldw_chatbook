# Workspace Persona identity — TASK-32776

Saved Persona IDs `none` and `auto` now remain distinct from the None and
automatic-create controls. A typed control choice survives Create recomposition;
saved string IDs always pass through authoritative record validation. Both
workspace pickers request the complete local in-memory catalog using its existing
limit argument. A separately readable selected Persona retains its literal label
even if catalog listing fails. Deleted/unavailable records cannot silently replace
the saved default. Profile preservation and explicit read-write confirmation use
the existing registry paths (ADR-079/139); no schema or service contract changed.

## Targeted evidence

**81 distinct targeted cases pass across the final and related runs:** nine new,
38 related, three partial-create cases and [31 governance cases](governance-tests.txt).

[Nine new cases pass](identity-tests.txt): real `none`/`auto` records survive
no-edit Apply and Create folder recomposition; both pickers include all 101
catalog entries; failed listing preserves independently readable selected labels
and refuses deleted identities. [Original identity failures](identity-red.txt)
and [Settings failed-list failure](catalog-red.txt) precede the fix.

The [combined related run](related-tests.txt) passed 35 of 38 cases. Two older
creation/default tests failed before assertions because config roots changed after
imports; all [five creation/default cases pass](creation-tests.txt) after moving
them to the existing private-profile process helper. The remaining first-bind
review case [passed in isolation](review-isolated-tests.txt). Its fixed 0.3-second
wait assumed asynchronous modal readiness; both opening points now use bounded
state waits, and [both variants pass](review-ready-tests.txt). These follow-ups
overlap the 38 cases. [Three partial-create recovery cases](lifecycle-tests.txt)
also pass. No full suite or
external provider requests were run. [Independent source review](review.md)
found no introduced blocker. Settings retains its existing Ruff baseline;
the other changed files are checked directly with scoped formatting;
[static checks](static.txt) also verify backlog IDs and diff hygiene.

## Native scope

The [runner](native_check.py) starts real TldwCli/LinuxDriver on a TTY with
HOME/config/data selected privately before imports. It creates 103 real Personas,
including `none` and `auto`. Settings creates a workspace with the exact selected
Persona, retains it through folder recomposition, and then offers and applies the
oldest catalog entry. The production default modal is mounted directly to verify
its no-edit Apply, memory confirmation and Cancel behavior; Console entry is
covered by the existing targeted test. Each successful mutation is checked by
reopening the workspace database. Tool Profiles is explicitly initialized first;
the separate cold automatic-provisioning defect is outside this slice.

The first run used a project fixture inside the protected app-data root. The app
correctly refused that binding; the corrected runner uses a sibling project folder.
[Failed-run diagnostics](failed-run001-native-result.json) and its
[lifecycle receipt](failed-run001-lifecycle.json) preserve normal exit and profile
isolation. This was a fixture-path error, not a change to folder authority.

## Captures

| View | Dark compact | Light compact | Dark wide | Light wide |
| --- | --- | --- | --- | --- |
| Saved identity during Create | ![dark 80x24](textual-dark-80x24-create.svg) | ![light 80x24](textual-light-80x24-create.svg) | ![dark 170x48](textual-dark-170x48-create.svg) | ![light 170x48](textual-light-170x48-create.svg) |
| No-edit default selection | ![dark 80x24](textual-dark-80x24-saved.svg) | ![light 80x24](textual-light-80x24-saved.svg) | ![dark 170x48](textual-dark-170x48-saved.svg) | ![light 170x48](textual-light-170x48-saved.svg) |
| Required memory confirmation | ![dark 80x24](textual-dark-80x24-confirmation.svg) | ![light 80x24](textual-light-80x24-confirmation.svg) | ![dark 170x48](textual-dark-170x48-confirmation.svg) | ![light 170x48](textual-light-170x48-confirmation.svg) |
| Oldest catalog entry staged | ![dark 80x24](textual-dark-80x24-catalog.svg) | ![light 80x24](textual-light-80x24-catalog.svg) | ![dark 170x48](textual-dark-170x48-catalog.svg) | ![light 170x48](textual-light-170x48-catalog.svg) |
| Exact saved result | ![dark 80x24](textual-dark-80x24-applied.svg) | ![light 80x24](textual-light-80x24-applied.svg) | ![dark 170x48](textual-dark-170x48-applied.svg) | ![light 170x48](textual-light-170x48-applied.svg) |

All 20 captures were rendered and visually inspected. The four-cell
[native receipt](native-result.json) records exact source and runner hashes.
The [capture manifest](capture-manifest.json) records all 40 SVG/text files.
The [lifecycle receipt](lifecycle.json) verifies exit 0, normal keyboard shutdown,
PID 74075 absent before terminal closure, eleven healthy SQLite databases,
instance-lock reacquisition, no error/faulthandler output, zero durable
conversations/messages and unchanged default-profile fingerprints.

Controls are focused before keyboard activation; this is not an exhaustive Tab
traversal or a native failure-injection matrix. Failed listing and deleted records
are covered by the mounted regressions. The Persona store/service is real in both
those tests and the native run. Broader Persona management remains unreviewed.
