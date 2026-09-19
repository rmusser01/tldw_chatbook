# Compact Collections controls — TASK-32662

Reviewed on `feat/component-pattern-library`, based on `72d59ca88f`.
This closes the compact Work-toolbar and manual search-clearing findings left
by TASK-32659.

## Changes

The primary, secondary and mode bars measure the complete intrinsic widths of
their buttons plus CSS gutters and margins. A row that would overflow stacks
vertically; it returns to a horizontal row when space permits. The feature-local
CSS changes layout on the mounted container, preserving button identity, Tab
order and annotation fields. This follows the existing stacked-control pattern
without adding a breakpoint or changing the adaptive shell's pane geometry.

Submitting an empty or whitespace-only text search while relevance is selected
now changes sort to saved desc before constructing the validated request. Other
sort choices and all scope predicates remain intact. This prevents the previous
`relevance_requires_search` exception from leaving the input journey unfinished.

ADR required: no. Existing ADRs
[086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[113](../../../../backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) govern
reader topology, authority and component styling. No storage, service boundary,
dependency, token value or saved layout preference changed. CSS was rebuilt from
source into the generated main bundle; the selector is scoped to Collections Work.

## Automated evidence

[verification.json](verification.json) records exact commands and distinct counts.
**137 distinct targeted checks pass.** The one inherited LibraryScreen size
failure is recorded separately below and in [size-comparison.json](size-comparison.json).
The new production-CSS journeys exercise actual Tab and Shift+Tab through the
nine reader controls, Enter on Favorite with real Local persistence, note-draft
retention and natural focus on wide recovery, and empty/whitespace text clearing
with relevance and title sorting. Both themes are covered; resize journeys span
170×48 and 80×24 with Items open. Wide recovery also verifies a single toolbar row.

The initial eight-case run failed: four painted-control assertions exposed clipped
Archive and invisible Info, three search cases reached the relevance exception,
and one case stopped at transient entry setup. All eight passed after the two
repairs. The expanded run found two immediate focus assertions observing `None`
between recomposition and focus restoration even though result state was correct.
Those assertions now wait for the actual focused field and its painted placeholder;
they do not set focus to manufacture recovery.

Existing browse, reader, controller, geometry, wiring, resize/query-budget, token
and generated-bundle checks are included. The unchanged LibraryScreen ceiling
check remains an inherited failure: 35,210 lines / 1,320 methods against
33,204 / 1,276. Collections controller budgets pass: main 1,689, capture 699,
saved-search 49. No budget was raised.

[Static comparison](static-comparison.json) shows zero new Ruff diagnostics.
New Python files pass full Ruff and formatting; existing changed ranges were
formatted. Changed Python parses and diff whitespace checks pass. Two inherited
pytest garbage-directory cleanup warnings remain. No full suite was run.
Independent read-only review found no actionable correctness or scope finding.

## Native and persistence evidence

[Run-001](result.json) used actual TldwCli and LinuxDriver with a
[private profile](isolation.json), exclusive instance lock, null keyring and two
Local captures seeded with stored article text. No extraction, provider or server
request was needed. [native_check.py](native_check.py) records the journey.

The runner explicitly focuses Mark Read once, then uses Tab through Favorite,
Archive, Open Original, More and the four reader modes. It reverses with
Shift+Tab, then activates Favorite with Enter and checks its persisted value.
Open Original is traversed without activation. It sets a note draft directly to
qualify retention, seeds a relevance query through the normal request method,
then clears it with Home, Shift+End, Backspace, spaces and Enter. Focus after the
write and search is observed without resetting it. Capture selection uses the
screen's normal selection method directly.

All four SVGs were rendered and visually inspected in one batched pass:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Reader controls, focused Info, retained Notes mode | [capture](controls-170.svg) | [capture](controls-80.svg) |
| Cleared relevance search with visible input focus | [capture](search-170.svg) | [capture](search-80.svg) |

At compact size primary and mode bars stack; the shorter secondary bar still
fits horizontally. Every traversed label is complete. The Notes form continues
below the viewport and remains available by vertical scrolling. In the wide
capture the bars are horizontal and the draft is visible. Info is focused but
not activated, so the Notes mode indicator and form correctly remain selected.

[Read-only SQLite checks](persistence.json) confirm two Saved captures, their
original notes, Favorite false after toggling it once in each configuration, ten
healthy databases and zero messages. The log has no ERROR/CRITICAL lines. Normal
terminal C-q returned from app.run with exit 0; the shell was observed and the
owned session closed. Unsaved drafts were never written to the database.

## Next review

Proceed to Library ingestion/import journeys. These checks qualify the repaired
Collections controls at the tested sizes, not every Collections feature or all
possible terminal widths. Full form traversal, remote authorities, extraction,
provider actions and restart remain outside this evidence. Integration into
`dev` is still pending.
