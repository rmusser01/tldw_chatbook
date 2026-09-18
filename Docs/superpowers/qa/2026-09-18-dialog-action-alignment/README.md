# Shared dialog action alignment — TASK-32821

Explicit left/right button groups were centered by the later dialog stylesheet.
Compound shared rules now honor the requested horizontal edge while preserving
vertical centering. Plain and explicitly centered dialog rows keep their layout.
The gallery and Roleplay recovery use this contract; the latter's local alignment
override is removed. No token values, handlers or CSS source registrations change.
ADR required: no; this repairs the existing ADR-161 component contract.

## Targeted verification

The [baseline gallery snapshots](baseline-results.json) pass in both themes.
The [red matrix](red-results.json) has twelve intended edge-geometry failures and
eight centered passes. The [qualified case ledger](qualified-cases.json) contains
**53 distinct passing cases**: twenty explicit/default/gallery alignment cases,
six Roleplay layout and Retry/Stay/Escape cases, eight existing gallery layout
cases, two palette command cases, two final snapshots and fifteen token/build/
byte/selector checks. Runs are serial with a fresh isolated profile per case.

The first final-source run completed 36 cases, then its SVG mismatch processing
was [interrupted without a qualified snapshot result](snapshot-first-attempt.txt).
The [remaining run](remaining-results.json) used `--assert=plain`: both snapshot
equalities failed as expected, and all fifteen other checks passed. Assertion
semantics and normalization were unchanged. After a deliberate snapshot update,
[both snapshots pass again in normal mode](snapshot-final-results.json).
The [readable semantic diff](snapshot-semantic-diff.txt) substitutes exporter IDs
and separates SVG elements only for review: changes are confined to the two
button labels and their background row moving 44 columns right. Colors, y
positions, text, control sizes and other gallery elements are unchanged.

CSS bytes are **584,093 / 608,090**; selector candidates remain **274 / 274**.
No limits or allowlist exceptions were raised. [All seven preflight guards](preflight.txt)
pass, including generated sheets, the pinned Mermaid assets, inventories and
backlog IDs. The new test and native runner pass Ruff and formatting.
[Independent read-only review](independent-review.txt) found no actionable issue.

## Native visual and lifecycle evidence

The native runner enters the gallery through Ctrl+P, typed “Pattern Gallery” and
Enter in a real TldwCli terminal. It focuses Cancel directly, tabs to Delete,
checks full painted labels and trailing alignment, and escapes to the host.
It then directly opens the Roleplay modal with all four failed domains and checks
initial Retry focus, keyboard Stay, pointer Retry and Escape callback outcomes.
This does not qualify normal partial-save entry, actual save workers or complete
keyboard traversal from the top of the gallery.

The first run passed its interactions and lifecycle, but its compact dark gallery
capture retained a temporary startup toast below the actions. Its original
[receipt](initial-native/result.json) and [lifecycle](initial-lifecycle.json) remain;
the final runner waits for gallery notifications to clear before capturing.

[All eight final captures](GALLERY.md) were rendered and inspected: Delete and
Stay have readable, non-obscuring focus and follow the trailing edge in dark/light
80×24 and 170×48 views. The [native receipt](native/result.json) records all four
cells passing. [Independent lifecycle checks](lifecycle.json) confirm normal
App.run return, exit 0, absent PID, released profile lock, ten healthy SQLite
files, zero conversations/messages, unchanged default settings and no error or
faulthandler output. All 11 final source hashes match. The owned
terminal exits with its wrapper; no native process remains.

The [fresh-ref/worktree allocation check](allocation-owner-check.json) confirms
this task is the only owner of 32821 across 230 refs and 33 worktrees.

The broader component review remains open. PR2707 stays draft and unmerged,
subject to its own visual review and merge approval. No full suite is claimed.
