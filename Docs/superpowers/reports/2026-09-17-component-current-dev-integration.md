# Component workstream current-dev integration — TASK-32749

The reviewed design-system branch now incorporates dev
`1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6` (71 incoming commits).
Draft PR #2704 remains the saved workstream; no merge into dev is performed.

ADR required: no. Existing [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../backlog/decisions/161-component-pattern-library.md) govern
this reconciliation. No new storage, security or application boundary was chosen.

The Console resolution retains incoming sidebar persistence synchronization and
covered-screen reconciliation alongside the reviewed handoff behavior. File Notes
retains incoming path fitting and recovery behavior, using token-backed classes.
Incoming Notes styles live in the decomposed Library owners; Workflow styles use
the current split registry and design tokens. Forty-five changed Notes declarations
and all expanded Workflow CSS preserve incoming values. Generated CSS and the
diagnostic inventory were rebuilt from their reviewed sources.

The incoming config-source lifetime exposed test fixtures that rebound profiles
after import. The failure reproduced at the exact dev commit. Affected cases use
the existing private-process helper and retain production admission checks.
Picker tests now follow dev's intentional folder path-first focus and `Select
folder` copy. Duplicate branch task IDs moved with provenance: Prompt More actions
32628 → 32750 and Search/RAG result focus 32707 → 32751; landed dev tasks keep
their IDs.

Verification: 280 distinct targeted cases, all derived-artifact checks, no new
scoped lint/format debt, and boot CSS 620,062 bytes under the unchanged 634,050
ceiling. Three unrelated fatal lint diagnostics are demonstrated on incoming dev.
Four real terminal theme/size cells have eight inspected captures, ten healthy
private databases, clean shutdown and unchanged default-profile files. Independent
review found no integration-specific blocker.

The native empty Notes view exposes an incoming disabled-action label clipping
defect at 170×48 in both themes. TASK-32752 is the immediate follow-up; this is not
a claim that every incoming feature is visually correct. Remaining Settings and
destination reviews also stay open. The [QA receipt](../qa/2026-09-17-component-current-dev/README.md)
contains failed-attempt history, exact evidence and verification limits; the
[completion ledger](2026-09-17-design-system-completion-audit.md) tracks the broader
workstream.
