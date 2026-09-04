# TASK-26000 Current-Dev Ruff Formatter Debt Design

**Status:** approved by the owner after adversarial review on 2026-08-30;
authority-cut closeout amendment approved by the owner on 2026-08-31

**Task:** `TASK-26000`

**Renumbering status:** This approved design was mechanically renumbered from
`TASK-24653` to `TASK-26000` under TASK-19601 after the older Network TLS trust
policy task retained `TASK-24653`; its design semantics are unchanged.

## Goal

Replace the stale, feature-local statement "61 inherited formatter failures" with
an exact census of current `origin/dev`, explain every change from that historical
set, and assign the remaining paths once to conflict-safe atomic cleanup tasks.

This characterization task changes no Python source. It produces the evidence and
Backlog boundaries that later formatter-only tasks execute.

## Pinned Evidence

The current-development pin after the TASK-26000 refresh is:

- Git revision and immutable authority cut `S`:
  `e555df102c950c29beed5e7119f433d35eee1f3c`
- Ruff: `0.15.22`
- Python: `3.12.11`
- Interpreter contract: an explicitly supplied absolute Python 3.12.11 invocation
  executable with Ruff 0.15.22 installed; the generated evidence preserves that
  canonical invocation path without dereferencing a replayable virtual-environment
  symlink or making one developer's absolute path normative
- Universe: Python paths tracked by Git at the pinned revision
- Configuration: the Ruff configuration committed at the pinned revision

The refresh begins with one fetch of `origin/dev`, freezes the resulting commit as
immutable authority cut `S`, and immediately proves the live remote-tracking ref is
still `S`. Detached evidence is recreated from `S`; every census, lineage,
provenance, PR, canonical, mutation, and review gate thereafter addresses immutable
object IDs. A later ordinary fast-forward does not invalidate this point-in-time
artifact or trigger an endless full-repin cycle. The explicit checker
`--require-live-current` switch is retained only for the immediate capture diagnostic
and its self-test, not as a later Task 5/Task 7 closeout invariant.

The owner-approved final authority-cut capture retained common ancestor
`f0e8961222fe1a7a3ac7566f7f78142e717358f3` and historical base,
pre-closeout, and closeout pins `31ed49bb368f54211d6482599e00a5c1340f80b2`,
`1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, and
`642b1c782fe6c066a781314dae669a55b05b62ad`. The refreshed current checkout
contained 5,056 tracked Python entries and 1,966 failures; the common checkout
contained 4,643 entries and 1,746 failures. Both had zero blockers. Historical
arithmetic remains `M=99`, `B=64`, `C=77`, `C-B=16`, `B-C=3`, and `H=61`;
`F_closeout=1,738`. The exhaustive current comparison is
`historical_still_current=44`, `historical_no_longer_current=17`,
`shared_ancestor_debt=1,603`, and `current_line_drift=319`, represented by 2,096
stable identities and 83 conflict-safe batches. At this pre-record authority-cut
capture, cleanup records remained absent.

The refreshed current/common raw SHA-256 values are
`f888cf9351f1c41f66fb98b4ec218c9268beb9b23295037320f725cec567ae10`
and `c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`;
the lineage, replay-cache, pre-record manifest, PR snapshot, materializer, frozen
producer, checker, allocator, and renderer SHA-256 values are respectively
`b9f9876d438b4b6770e84013c515ae54791b14f0e740de67283fb3de20f655a6`,
`0026dce1124fb3e9fc027dca785101c76a77b63882deac9e1951d5ce2d46a1df`,
`0f1a8ca2652e7537628c82885f5d5d0cb4421189c31255bb0f05648991083022`,
`46282d8e81b1bd512263443e97955b1650944684f6c1d0ccd1341f52218bd8d5`,
`69817bd0bac15097f80c6d194b7b27618bc96f494aab806aeb6d009a9c384c5c`,
`fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e`,
`a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79`,
`6d7559449c35cd6db3dca31dbbdb510efbb45d1dc0a96c4f01f59c6a8461403b`,
and `4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a`.
The current post-review checker, renderer, materializer, and allocator SHA-256 values
are `90fe803f28d783feae839b6078ed3244a5881aa62e4f65facfa2a53434bb7ccc`,
`a59bcb7c647927f47e9b858bdcb4329f283f8273138f9ef227381707e6a2ea8e`,
`3804046bd5692ef8ac833193cae84cfc8a4f3c05b4ddd86dad463a547c50931d`, and
`2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432`.
After Task 5, the canonical manifest contains 83 cleanup records allocated as
`TASK-26933` through `TASK-27015`, with `TASK-27015` as the final record. Its
original post-record SHA-256 was
`ded7288d8580367842110dd1a9e79976dc9c00663361251bb9212ca717cea0b9`;
after the review correction to the final gate, the current canonical SHA-256 is
`eadbbabc7e6ba9910ebe086702d2c3ebc9a2b4d97b9a8031f5abff6a96ed75e3`.
The cache-cold temporal replay checked 319 ledgers and 1,272 candidates: 736
failing, 533 clean, and three transient syntax-invalid states. The ownership
capture inspected all 13 open PRs at `2026-08-31T17:40:01Z`; exact
current-failure overlaps were `#2265=6`, `#2264=4`, `#2230=1`, `#2196=12`,
`#2059=1`, `#1903=1`, and `#1655=2`, while six PRs had zero overlap.

The 83 sorted stable labels are `ruff-active-pr-1655`,
`ruff-active-pr-1655-2059`, `ruff-active-pr-1903-2196`,
`ruff-active-pr-2196`, `ruff-active-pr-2230`, `ruff-active-pr-2264`,
`ruff-active-pr-2265`,
`ruff-agents-runtime`, `ruff-api`,
`ruff-character-persona`,
`ruff-chat-agents-tools`, `ruff-chat-citations`, `ruff-chat-console-context`,
`ruff-chat-console-fleet`, `ruff-chat-console-foundation`,
`ruff-chat-console-interaction`, `ruff-chat-console-library`,
`ruff-chat-console-observability`, `ruff-chat-general`, `ruff-chat-media`,
`ruff-chat-metrics`, `ruff-chat-persistence`, `ruff-chat-providers`,
`ruff-chat-retrieval`, `ruff-chat-trajectory`, `ruff-chunking`,
`ruff-console-character-media`, `ruff-console-composer`,
`ruff-console-fleet-ui`, `ruff-console-foundation-ui`,
`ruff-console-inspection`, `ruff-console-knowledge-ui`,
`ruff-console-layout-rails`, `ruff-console-modals`, `ruff-console-runtime`,
`ruff-console-session-send`, `ruff-console-transcript-selection`,
`ruff-console-workspaces`, `ruff-core-runtime`, `ruff-database`, `ruff-evals`,
`ruff-generation-media`, `ruff-ingestion-web-media`, `ruff-integration-live`,
`ruff-library`, `ruff-library-screen-large`, `ruff-mcp-runtime`,
`ruff-model-artifacts-tests`, `ruff-notes`, `ruff-performance`,
`ruff-personas-screen-large`, `ruff-providers-prompts`, `ruff-rag-research`,
`ruff-rag-search-tests`, `ruff-root-ci-architecture-final`,
`ruff-root-test-infrastructure`, `ruff-scheduling-notifications`,
`ruff-skills-runtime`, `ruff-speech-audio`,
`ruff-state-sync-wizards-tests`,
`ruff-tests-misc`, `ruff-tools-runtime`, `ruff-ui-evals`,
`ruff-ui-file-dialogs`, `ruff-ui-library`, `ruff-ui-mcp-tools`,
`ruff-ui-model-management`, `ruff-ui-navigation-shell`, `ruff-ui-personas`,
`ruff-ui-prompts-workbench`, `ruff-ui-remaining-screens`, `ruff-ui-research`,
`ruff-ui-scheduling`, `ruff-ui-settings`, `ruff-ui-speech`,
`ruff-ui-visual-css`, `ruff-ui-watchlists`, `ruff-ui-wizards`,
`ruff-utils-config`, `ruff-watchlists-screen-large`,
`ruff-watchlists-subscriptions`, `ruff-widgets`, and `ruff-workspaces-runtime`.

TASK-22514 supplies the historical comparison, not the current truth. Its scoped
closeout evidence is pinned to local evidence commit
`642b1c782fe6c066a781314dae669a55b05b62ad`, implementation base
`31ed49bb368f54211d6482599e00a5c1340f80b2`, and pre-closeout census
`1f4f72ac5ff02f5237a4946745e82e8932cd41cf`. That closeout's 61 paths
were scoped to its changed-Python manifest rather than the whole repository. This
task reconstructs and persists:

- `M`: the 99 Python path identities changed from implementation base to
  pre-closeout census, with add, delete, and rename state retained;
- `B`: identities in `M` whose base-revision path failed at the implementation
  base;
- `C`: identities in `M` whose pre-closeout path failed at the pre-closeout
  census;
- `H = B ∩ C`: the historical scoped residue, which must have cardinality 61;
- `F_closeout`: a new whole-repository census at the final evidence commit, used
  only to validate TASK-22514's final scoped invariant; and
- `A`: the merge base of the final evidence commit and the current-development
  pin, plus `F_common`, a whole-repository census at `A` used to distinguish debt
  already present on the branches' shared history from current-line drift.

The reconstruction must also reproduce the recorded cardinalities `|M| = 99`,
`|B| = 64`, `|C| = 77`, `|C ∖ B| = 16`, `|B ∖ C| = 3`, and
`|H| = 61`. After projecting stable identities to closeout-revision paths, the
final closeout census must additionally prove
`F_closeout ∩ project(M, closeout) = project(H, closeout)`. A mismatch blocks
characterization rather than redefining TASK-22514's historical claim.

None of the three historical commits is currently reachable from a reviewed
remote-tracking ref. TASK-26000 therefore cannot leave raw SHA references as its
only evidence: before completion it commits the reconstructed sets, per-path
presence and formatter status, Git blob IDs where present, commands, tool versions,
and source revisions into its point-in-time JSON artifact. That committed artifact
is the durable historical provenance; future reruns against the raw objects are
optional corroboration, not a hidden prerequisite for understanding the manifest.

Every census runs in an isolated clean worktree or checkout at its exact revision.
The interpreter must report Python 3.12.11 and the executable must report exactly
Ruff 0.15.22 before any result is retained. The final collision scan still fetches
every mandatory remote branch, paginated open-PR head, and worktree claim source. It
accepts observed `origin/dev` equal to `S` or a verified fast-forward descendant,
records the manifest pin, observed SHA, and exact ancestry result, and fails exact
`E_ORIGIN_DEV_DIVERGED` for a missing pin/tip or non-ancestor/force-pushed state.
PR-head movement and task-ID/worktree collisions remain strict. No fetch occurs after
that final claim scan. The canonical final allocation-audit SHA-256 and its bound
`manifest_pin`, `observed_origin_dev`, and `origin_dev_ancestry` values are required
in Task 7 Implementation Notes. The canonical 155 MB
`raw/allocation-closeout-rescan.json` remains outside the repository but is retained
through review and integration with the other temporary evidence. Its tracked,
review-accessible `allocation-closeout-rescan-summary.json` publishes the complete
83-label allocation, a lossless inclusive-range encoding of all 2,671 occupied IDs,
the open-PR heads, section counts and hashes, and the raw artifact hash. Before cleanup
records exist, a live external-maximum change recomputes the proposed allocation and
forces regeneration. Once all cleanup records exist, the final scan authenticates
their exact manifest-bound self identities, proves they appeared in the live claim
census, rejects any different identity on an active ID, and preserves their allocation
despite unrelated higher task IDs. A fresh collision-recovery scan still allocates
above every observed claim. The evidence never combines path lists, contents, or
configuration from different revisions.

## Chosen Approach

Use owner-aligned batches. Group remaining paths by subsystem and likely test
surface, then separate files under active concurrent ownership. Large or unusually
high-risk files may form a batch by themselves. Every failing path belongs to
exactly one child task.

Rejected alternatives:

- Fixed-size batches are easy to count but split shared test surfaces and ignore
  merge-conflict risk.
- One repository-wide formatting PR is mechanically simple but creates excessive
  review churn and conflicts with active feature branches.

## Census and Comparison

The current census enumerates Git-tracked Python paths with NUL-delimited Git output
inside the exact checkout. A standard-library Python driver consumes those bytes,
prefixes each repository-relative path with `./`, and invokes Ruff once per path.
This avoids shell word splitting, leading-dash ambiguity, and command-length limits.
Each invocation uses `ruff format --check --force-exclude`, so explicit paths retain
the revision's repository exclusions.

Exit zero means the path does not fail the revision's configured formatter check;
it may be clean or excluded. Exit one means Ruff 0.15.22 would reformat the path.
Any other exit is a blocker captured with its repository-relative path, exit code,
and diagnostic category. It is not silently classified as formatter debt. The
aggregate repository command is also run once as a control and must agree on
whether any formatter failure exists, but the per-path exit code is the manifest
source; human output is not parsed to decide membership. Git path bytes are decoded
strictly as UTF-8 for JSON. A non-UTF-8 path is retained losslessly as base64 in the
blocker list rather than silently replaced or omitted.

Set membership is revision-specific even when provenance is stable. Each historical
identity therefore records an optional path projection at base, pre-closeout,
closeout, common-ancestor, and current revisions. Rename and deletion lineage is
resolved before set arithmetic; a path string alone never establishes identity
across the divergent branches. All formulas below operate on the applicable
revision-path projection. The checker rejects a historical identity that maps to
multiple current paths or a current path claimed by multiple identities unless an
explicit copy/split lineage record explains it and assigns the current path to only
one classification.

The plan records these disjoint current-state classifications:

- paths still present in both the historical residue and current failures;
- historical failures no longer current, with deletion or formatting lineage;
- current failures outside the current projection of `H` whose identity already
  failed in `F_common`, identified as shared-ancestor debt; and
- remaining current failures, with current-line failure-introduction, addition, or
  rename lineage from `A` to the current pin.

Deleted paths need no cleanup task. Renames explicitly pair the historical removal
with the current addition and identify the rename lineage; they cannot appear as two
unrelated explanations. Any parse or configuration error is a blocker, not formatter
debt, and is filed separately rather than hidden in a formatting batch.

The sorted `M`, `B`, `C`, `H`, `F_closeout`, `F_common`, current failure set,
revision-path projections, comparison sets, blockers, rename mappings, and stable
batch labels are persisted in one point-in-time JSON evidence file. The artifact
also records a schema version, every source revision, Python and Ruff versions, the
canonical absolute invocation executable, exact commands, and each present path's Git
blob ID. A one-shot
standard-library checker proves the historical cardinalities, the projected
final-closeout invariant, and an exhaustive, pairwise-disjoint current
classification with a lineage record for every moved path.
It also proves that batch unions equal the current failure set, batches are pairwise
disjoint, blocker paths occur in no batch, and every stable batch label resolves to
exactly one newly created cleanup Backlog record without requiring TASK-26000 to
reference higher task IDs.

## Cleanup Task Contract

Each child task is independently mergeable and must:

1. rebase onto current `origin/dev` and reproduce its assigned failures;
2. run Ruff formatting on only its assigned paths;
3. parse each file before and after with the same recorded supported Python
   interpreter and `type_comments=True`, normalize only `TypeIgnore.lineno`, and
   compare `ast.dump(..., include_attributes=False)`; the normalization prevents
   harmless line movement from masquerading as an AST change while retaining every
   type-ignore tag;
4. prove the ordered comment-token text is unchanged; anchor each inline `# noqa`,
   `# type: ignore`, and single-target Ruff directive to the same deepest AST node
   path covering its physical line, using its significant-token position within its
   nearest logical owner as the tie-breaker. For a same-line `ExceptHandler` header,
   that owner is a uniquely, fail-closed validated `except` clause through its unique
   depth-zero colon; otherwise it is the nearest containing `ast.stmt` or decorator
   boundary. Exclude only parentheses independently proven AST-neutral by full-module
   shadow parse/dump equality—never tuple commas or semantic grouping. Preserve
   standalone file directives between the same adjacent statement paths; and prove
   every Ruff-format off/on range encloses the same ordered AST-node interval before
   and after formatting;
5. run Ruff lint and format checks on every touched Python path;
6. record the rationale for its focused test selection plus exact commands and
   results;
7. pass `git diff --check` and the Backlog task-ID guard; and
8. contain no hand-written production behavior change.

After rebasing, an assigned path that is deleted, renamed, modified, or already clean
must be reconciled in that cleanup record before formatting. The record identifies
the upstream lineage, preserves or amends exact path ownership, and reruns its
mechanical proof. It never silently drops a path or absorbs an unassigned one.

The final cleanup record is created after the earlier cleanup records so its
dependencies only point to lower task IDs. It owns the final explicit Git-tracked
repository-wide Ruff format check from a clean Git-tracked checkout after all
dependencies merge. Before that check, Git must report no untracked files via
`git ls-files --others --exclude-standard`, so the command's `.` operand
cannot silently broaden the audited scope. A post-cut unassigned failure blocks that
gate, is never silently added to the pinned counts or existing batches, and requires
a separate correction record. The series either waits for its independently owned
correction or creates a superseding final record after the new correction record
exists; the final task never absorbs the regression as formatting debt.

## Records and Scope

The point-in-time JSON evidence is the mechanically auditable manifest. The
implementation plan lists the pin, comparison counts, stable batch labels, evidence
path, and verification commands. It does not list future cleanup task IDs: later
cleanup records reference TASK-26000, while the final cleanup record may depend only
on earlier-created cleanup records. No generator, runtime dependency, or permanent
formatter-baseline file is introduced.

TASK-26000 itself modifies only its Backlog record, this design, the plan, the
point-in-time evidence, and newly created cleanup Backlog records. Its completion
requires those records and their contracts to exist; it does not require the future
cleanup work to execute. It does not run the local full test suite because it changes
no executable code.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: the owner-approved authority-cut amendment changes only the audit/closeout
process for this point-in-time formatter characterization. It changes no runtime,
dependency, storage, security, privacy, schema, or cross-module ownership decision.
