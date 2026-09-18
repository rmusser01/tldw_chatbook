# Skills Files and supporting files — TASK-32657

Reviewed on `feat/component-pattern-library`, based on `f3dd181084`.
The existing Files inventory passed the reviewed journeys. No production
repair was needed; this slice adds qualification coverage and user guidance.

## Reviewed behavior

- Empty Skills explain that there are no supporting files. Populated Skills
  show relative paths, actual byte sizes and binary labels without rendering
  file contents or offering an editor in Files.
- A 65-supporting-file bundle includes nested paths, a long wrapped path,
  UTF-8 text (`café\n`, six bytes), an empty text file and a five-byte binary.
  Keyboard End reaches the final file; Home restores visible mode controls.
- Edit → Files → Edit preserves unsaved description and instruction fields.
  The four size/theme tests use Tab/Shift+Tab and Enter for that round trip.
  All 66 bundle files, including `SKILL.md`, retain their exact hashes.
- A real private trust service is bootstrapped in the automated journeys.
  Its trusted status and every trust-file hash, including the generation
  marker, remain unchanged after navigation.

ADR required: no. Existing ADRs [009](../../../../backlog/decisions/009-local-skill-trust-boundary.md),
[076](../../../../backlog/decisions/076-library-lifecycle-progressive-disclosure.md),
[086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) govern
the existing trust boundary, reader layout and design language. No storage,
authority, token or stylesheet changes were made.

## Automated evidence

**64 distinct targeted checks pass.** [verification.json](verification.json)
records commands, counts and observed summaries: 45 reader/state checks,
four new journeys, two selected bundle-service checks and 13 token/bundle
governance checks. The journeys cover 170×48 and 80×24 in both dark and light
themes. Both new Python files pass Ruff and formatter checks.

The first long-path assertion failed at wide size because it joined whole-screen
paint containing text from adjacent panes. Cropping the inventory's visible
region before joining wrapped path segments fixes the assertion; this was a
test defect, not an application defect. These tests qualify existing behavior
and are not claimed as a red/green reproduction of a production fix.

Independent review found two coverage gaps in the first draft: direct focus
did not prove Tab navigation, and a disabled trust fixture did not prove trust
preservation. Both were corrected, all four final journeys passed, and follow-up
review found no remaining issue. Two inherited pytest temporary-directory
cleanup warnings appeared in each targeted invocation. No full suite was run.

## Native and persistence evidence

The passing run is `run-001`, driven by [native_check.py](native_check.py) with
actual `TldwCli`, `LinuxDriver` and an owned tmux terminal. The
[private profile](isolation.json) uses synthetic bundles, private databases and
the null keyring backend. Fixtures are created through the local service;
the binary fixture is written directly into its private bundle.

Native controls are explicitly focused and activated with Enter. Assertions
for focus after Files selection, Home and Discard do not set focus. This native
run supplies real-terminal evidence; the separate automated journeys supply
the Tab/Shift+Tab mode-order coverage. Programmatic fixture creation bypasses
normal creation UI, so the captures do not qualify rail-count refresh.

At 170×48 dark and 80×24 light, the [run result](result.json) confirms empty
and populated inventories, End/Home scrolling, draft preservation and a
natural return to the bundle row after Discard. Native trust starts and remains
uninitialized; no approval or trust manifest is created.

All six SVG captures were rendered through Quick Look and visually inspected:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Empty | [capture](empty-170.svg) | [capture](empty-80.svg) |
| Inventory and wrapped path | [capture](inventory-170.svg) | [capture](inventory-80.svg) |
| Final file after End | [capture](last-file-170.svg) | [capture](last-file-80.svg) |

Normal Ctrl+Q returned from `app.run` with exit code 0; the owned terminal
session was closed. [Persistence checks](persistence.json) verify all 66 exact
SHA-256 values, no trust manifest, ten SQLite integrity results and zero
messages. The log contains no application error marker or unhandled exception.

This review does not qualify provider or script execution, a full app restart,
or performance at the maximum supported file count. Files remains a metadata
inventory. The [user guide](../../../User_Guide/library/skills.md#files) now
explains its contents, scrolling and draft-preserving mode changes.

Next: Library Collections, then the remaining Library destinations.
Integration into `dev` remains pending.
