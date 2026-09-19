# TASK-32784 — Tool Profile compact CSS indexing

The [PR guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/35335586309/job/105569547849)
failed deterministically at 275 ancestor-scoped bare-type rules against the
unchanged limit of 274. Timed UI journeys were passing. The compact action rule
introduced by TASK-32781 was the sole added selector in that census.

Profile action buttons now carry `tool-profile-action`. The compact width rule
targets `Button.tool-profile-action` instead of a bare descendant Button.
Specificity remains `(0,2,2)` and declarations are identical. Export, Edit, Bind
and Remove retain the rule; Import is excluded. The CSS bundle was rebuilt from
its source module; the ratchet limit is unchanged.

80 targeted checks pass: the [real parsed-style ratchet](ratchet.txt) and
79 existing compact focus, design/component governance and CSS build/sync
cases ([output](targeted.txt)). [Census evidence](census.txt) shows 275→274.
Panel Ruff and formatting, backlog and diff guards pass. Independent review found
no actionable issue and confirmed exact coverage, specificity and generated CSS.

Four mounted dark/light × 80×24/170×48 comparisons have identical computed button
styles, geometry, visibility and rendered text. The baseline uses the saved
commit stylesheet and original classes; the comparison uses the new stylesheet
and classes. Their identical payload is stored once in
[equivalent-layout.json](equivalent-layout.json), with the shared digest and
source hashes in [verification](verification.json).

The first unconstrained comparison differed by one scroll row in one cell while
styles were already identical. The final comparison settles animations and
explicitly scrolls to the end. Existing focus tests independently qualify
automatic scrolling and replacement focus. No new native captures were taken
for this mechanical selector-index change; this comparison does not claim to
be terminal lifecycle evidence.

ADR-150 applies; no new ADR or token is required. MCP Edit geometry and remaining
component reviews are separate. Remote checks for the new commit must still run.
