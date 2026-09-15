# Component-pattern integration — 2026-09-14

TASK-32591 reconciles the component-pattern branch with dev. The merge parents
are design/audit head `705a39bdb88ef4f64a926eeec169f0d9a02469e4` and
`origin/dev` `fd30614dcdc1e6cbd39b1532769d3e10be9b12b6`; their merge-base is
`2ecf784d6ea41592a9d9b6b915fbe8c8c533470f`. This is local integration on
`feat/component-pattern-library`; it does not merge or publish the branch to dev.

## Resolution

- Preserved both Library harness constants where new split-sheet harnesses and
  the branch's bundled harnesses coexist. Subsequently corrected thirteen
  standalone Library harnesses to load the actual split styles they exercise.
- Kept the decomposed source ownership and removed obsolete
  `screen_agentic_console.tcss`. Rebuilt all generated styles from source.
- Preserved upstream wide Conversation Settings, recent-Library focus shape,
  Console setup button shape, agent preset forms and worktree recovery styles.
  A comparison after token expansion confirms all **89** added/changed upstream
  declarations remain represented in their owning sources.
- Tokenized the new picker width and changed the new Notes action-row inline
  height to the existing `h-auto` class. No token-floor allowance was increased.
- Updated the new harness guard for the retired Console split and corrected
  its multiline-comment parser: filenames in source comments had been treated
  as selectors. A regression failed before the parser fix and passes afterward.
- Refreshed the diagnostic manifest for the five widgets already deleted by the
  design-system migration: ten calls disappear; no new diagnostic sink or
  diagnostic body is introduced. The removed statements were reviewed.
- Added the missing type-checking import for `NoteImportExecutor`, discovered
  by fatal lint in the incoming Notes controller; runtime imports are unchanged.

## Backlog identity

The older dev task `TASK-32532` belongs to ProjectInstructionSetupModal disable
recovery (arrival `e384497095`, 2026-09-12). The younger component-pattern task
(arrival `b855f59ccd`, 2026-09-13) is now **TASK-32596**, with its twelve subtasks
renumbered together and inbound references updated. Allocation followed a fresh
all-ref/worktree sweep through 32595. The older dev task is unchanged. Historical
captures and commits retain their original identities. Backlog uniqueness passes.

## Verification

- Governance/build/token/budget/Backlog run: **68 passed**; its sole initial
  harness-guard failure was fixed and covered by the later successful run below.
- Console Settings geometry and affected Library layout/transitions: **36 passed**.
- Corrected Library harnesses and complete harness guard module: **30 passed**.
  Selections overlap; these counts are not an aggregate unique-test total.
- Fatal Ruff checks (`E9,F63,F7,F82`): **277 changed Python files passed**.
  The edited harness guard passes Ruff formatting; scope-limited whitespace
  checks pass. The merge diff also contains existing incoming dev documentation
  whitespace; no unrelated evidence was rewritten. Historical design-branch
  whitespace remains TASK-32595 work.
- Boot-parsed CSS: **612,733 / 634,050 bytes**, with no budget increase.
- Derived-artifact checks: CSS, profile-owned paths, diagnostic inventory,
  Backlog IDs, table allowlist and index pins pass. Mermaid's sandboxed download
  was unavailable; the same checker with network access verified all six outputs
  against its hash-pinned inputs.
- Native terminal: isolated scratch profile booted to Console at 120×40 and
  80×24; the upstream setup-button focus shape paints. No provider request was
  sent. This is startup/render evidence, not a live model conversation or full
  setup-workflow certification. The optional detected-server action sits below
  the initial 80×24 viewport and needs a later dedicated setup accessibility check.

Two pytest cleanup warnings concerned existing non-empty temporary directories;
no unrelated temporary files were deleted. No full test suite was run.

Evidence, exact targeted selections and review findings are in
[the integration review](../qa/2026-09-14-component-integration/review.md) and
[declaration comparison](../qa/2026-09-14-component-integration/upstream-css-preservation.json).
Both independent review findings were fixed and verified before completion.

ADR required: no new ADR. Existing [ADR-161](../../../backlog/decisions/161-component-pattern-library.md)
and [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md)
govern this integration; their architecture and design contracts are preserved.

Next: TASK-32592 More-menu reachability, TASK-32593 compact Settings,
TASK-32594 gallery containment, TASK-32595 documentation/diff hygiene, then
Library workflow review. These remaining tasks are not completed by this merge.
