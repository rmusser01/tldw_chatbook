# TASK-26000 Current-Dev Ruff Formatter Debt Design

**Status:** approved by the owner after adversarial review on 2026-08-30

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

- Git revision: `05c858e87cc1f11c96d6b384b34fdaf914efc51e`
- Ruff: `0.15.22`
- Python: `3.12.11`
- Interpreter contract: an explicitly supplied absolute Python 3.12.11 invocation
  executable with Ruff 0.15.22 installed; the generated evidence preserves that
  canonical invocation path without dereferencing a replayable virtual-environment
  symlink or making one developer's absolute path normative
- Universe: Python paths tracked by Git at the pinned revision
- Configuration: the Ruff configuration committed at the pinned revision

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
Ruff 0.15.22 before any result is retained. If `origin/dev` advances before the
characterization records are committed, the task rebases, updates the pin, and
reruns the current census. It never combines path lists, contents, or configuration
from different revisions.

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
   path covering its physical line, using its position in the logical statement's
   significant-token stream as the tie-breaker; preserve standalone file directives
   between the same adjacent statement paths; and prove every Ruff-format off/on
   range encloses the same ordered AST-node interval before and after formatting;
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
repository-wide Ruff format check. A new unassigned failure blocks that gate. The
series either waits for its independently owned correction or creates a superseding
final record after the new correction record exists; the final task never absorbs
the regression as formatting debt.

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

Reason: this work characterizes and schedules behavior-preserving mechanical
formatting. It changes no runtime, dependency, storage, security, privacy, schema,
or cross-module ownership decision.
