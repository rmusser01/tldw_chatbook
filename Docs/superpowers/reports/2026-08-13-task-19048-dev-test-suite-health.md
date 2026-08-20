# TASK-19048 Latest-dev Test-Suite Health Report

## Scope

The frozen `baseline-f60b2be72-r1` generation collected 47,572 nodes and
recorded 47,254 passes, 211 skips, 106 failures, and one error. Six additional
chunk processes were retained as structurally red `corrupt_junit` outcomes;
their 1,500 nodes remained fully accounted for by the outcome plugin. The 107
terminal red nodes spanned 54 files.

The user-approved scope amendment keeps the existing Console size ratchet in
TASK-3070.5 through TASK-3070.11. It is a seven-PR controller-decomposition
program, not an atomic test-health correction. This branch does not raise the
budget or hide the node.

## Repair inventory

| Cluster | Classification and repair |
| --- | --- |
| Checkpoint evidence and privacy | Hardened collection/process accounting, bounded JUnit handling, subprocess ownership, and node-ID/output redaction. Replaced private parametrization values with stable semantic IDs without changing test cardinality or assertions. |
| Provider/runtime behavior | Corrected continuation endpoint identity and registered the existing frontmatter optional dependency. |
| Packaging | Included the schema-v40 transcript-annotation migration in source/wheel metadata, the installed-distribution probe, and the release checker’s required-file contract. |
| Notes, Library, Watchlists, and MCP | Reconciled current writer/service seams, preserved the Watchlists compatibility node identity, added the two existing Watchlists artifact tools to the independent distribution expectation, and waited for Prompt import projection before retrying Undo. |
| Console and Settings UI | Updated stale controller/widget seams, mounted-state waits, bounded background completion, current binding normalization, and the real Settings overview selector. No broad Console extraction or budget change was introduced. |
| Speech/TTS tests | Reconciled current dependency/catalog/authentication and delivery contracts. A speculative publication-repair series was removed after latest `dev` proved the opposite accepted contract; final TTS production bytes follow current `dev`. |
| Architecture ratchet | `chat_screen.py` remains 21,292 lines against the 17,727-line ceiling. This unchanged red is delegated to TASK-3070; the ceiling remains intact. |

## Latest-dev comparison

After the initial repair rebase onto `origin/dev` at `a1d6df3f8`, a clean-dev control reproduced
34 failures among 104 directly comparable historical nodes. The rebased branch
initially had 25 failures among 105 runnable historical nodes. Two node names had
changed upstream and were mapped to their exact current semantic replacements;
the Watchlists compatibility alias restored the remaining historical identity.

The final executable candidate was `19cfb24b5`. Running the exact 107 discovered
nodes at that commit produced:

- 106 passed;
- one failed: the unchanged TASK-3070 Console size ratchet;
- zero errors, skips, xfails, deselections, or missing nodes;
- 198.12 seconds wall time.

Before PR closeout, the branch was rebased again onto current `origin/dev` at
`a9b6a6b88`. The executable PR candidate `108c5e672` then passed the exact seven
CI-reported UI failures 21/21 across three fresh two-worker runs, the adjacent
Prompt pagination/lifecycle matrix 8/8, the touched Core/process matrix 17/17,
and the TASK-15103 diagnostic-ledger matrix 48/48. The rebase preserved the new
ADR-067 Library pagination contract; this suite-health task was renumbered to
TASK-19048 because current `dev` now owns TASK-18912 for that pagination series.

## Focused evidence

- Packaging v40 artifact/install/removal contract: 5 passed.
- Review-driven speech, Console completion-bound, Skills selector, and signal
  scanner checks: 13 passed.
- Prompt import/Undo ownership: 2 passed repeatedly; affected import/undo slice
  21 passed.
- Settings/Speech assigned cluster: 8 passed; directly affected slice 30 passed.
- Console/core bounded repairs: five assigned nodes passed; background-signal
  slice 13 passed; affected controller/keyboard slice 63 passed.
- Watchlists restored node identity: full file 22 passed; the 107 historical red
  IDs were present in a 3,957-node collect-only comparison.
- PR #1838 CI artifact review: pre-final run `32383605243` retained five exact
  UI shard-1 failures and two exact shard-0 failures. They reduced to mounted
  projection, live-widget, and app-worker settlement races and are covered by
  the final repeated UI matrix above. The superseded run was cancelled only
  after its structured artifacts were downloaded and classified.
- Qodo's review of the rebased executable head reported zero bugs and zero rule
  violations; CodeRabbit passed and Cubic skipped neutrally.

Ruff lint, targeted format checks, `py_compile`, and `git diff --check` passed for
the changed scopes. Three large pre-existing test files retain unrelated whole-file
formatter drift; their changed hunks are formatted and lint-clean. Independent
correctness/YAGNI re-review approved the packaging, Prompt, Settings/Speech, and
Console/Skills corrections with no remaining Critical, Important, or Minor finding.

## Governance

ADR-072 remains the authority for the checkpoint harness process-ownership
boundary. No new runtime dependency, schema decision, or application architecture
was introduced by the repair work. Raw logs, JUnit, harness files, and manifests
remain permission-restricted in the ignored task evidence root; this report contains
only repository-relative identifiers and sanitized aggregate evidence.
