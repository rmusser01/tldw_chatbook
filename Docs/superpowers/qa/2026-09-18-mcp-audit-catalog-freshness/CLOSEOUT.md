# PR2726 approved closeout

The owner approved the conflict choices and eight-capture gallery at
`a4158a6a9a103ac27b0a0a825b60951c24c21f03`, including the separately recorded
header follow-up. Qodo reported zero bugs, rule violations or requirement gaps
for that head. Current-head CI exposed an inherited wide-modal selector breach;
this follow-up repairs it without changing approved appearance or thresholds.

## CI cause and repair

The [failure](closeout/selector-red.txt) counted 277 ancestor-scoped bare-type
rules against the unchanged 274 limit. PR2742, already in the merged base,
introduced wide-tier rules ending in `Vertical` for FileExtractionDialog,
DictionaryAttachPicker and ConversationAttachPicker. Those rules became
candidates for every Vertical in the app, even when the modal was absent.
The same 277/274 assertion reproduced locally before edits.

Each existing root Vertical now carries a unique class. Only its corresponding
wide-tier rule uses that class, retaining the original ancestors, child relation,
width percentage and cap tokens. The registry follows the renamed selectors;
the generated app bundle is rebuilt. No handlers, global limits or token values
changed. Existing ADR-097/150/161 apply; no new ADR is required.

## Verification

[203 closeout cases pass](closeout/final-cases.json), including the unchanged
selector ratchet, fast-path style equivalence, CSS byte budget, wide-tier census
and geometry, both picker behaviors, design-token checks, 56 launcher argument
cases and all 14 catalog regressions. Six new dark/light geometry cases preserve
both sides of the 150-column threshold through 120→149→150→170→200→120 resizes;
these also [passed before the selector change](closeout/geometry-before-cases.json).
The earlier 168-case approved-head evidence remains available. No full suite ran.
Three warnings concern existing invalid escape sequences in unrelated source
parsed by the modal census; pytest also reports unrelated old-directory cleanup.

All [seven preflight guards](closeout/preflight.txt) pass. [Static baselines](closeout/static-analysis.json)
are unchanged and new/changed ranges are formatted. [Independent review](closeout/independent-review.txt)
found no blockers in selector equivalence, indexing or QA coverage.

Two real private native journeys open the three affected modals directly with
synthetic in-memory content and activate Cancel. All twelve dark/light compact/
wide [before/after terminal captures](closeout/modal-terminal-comparison.json)
and [rendered PNG comparisons](closeout/modal-pixel-comparison.json) are identical.
The twelve after captures were visually inspected. This verifies appearance and
cancel behavior, not the complete file-save or attachment workflows.
[Before](closeout/modal-before/result.json) and [after](closeout/modal-after/result.json)
receipts preserve source hashes and dimensions; both lifecycle checks pass.
The [initial fixture correction](closeout/fixture-note.txt) is recorded separately.

The final Audit journey repeats the approved eight views after the selector
repair: all [terminal](closeout/audit-terminal-comparison.json) and [pixel](closeout/audit-pixel-comparison.json)
comparisons are identical. [Final Audit receipt](closeout/audit-replay/result.json)
and [lifecycle](closeout/audit-replay/lifecycle.json) pass. All journeys use the
validated private launcher; no file saves, attachments, network connections,
tool executions or permission changes occur. Each process exits, releases its
lock and leaves healthy private databases and unchanged user-default files.
Final application and QA hashes match the saved source.

The owner approval therefore remains applicable to the preserved visuals.
Final-head CI, accumulated review and a fresh check of the live dev tip remain
required before the authorized merge. The intermittent header race stays open
in [HEADER-FOLLOWUP.md](HEADER-FOLLOWUP.md); this closeout does not claim it fixed.
