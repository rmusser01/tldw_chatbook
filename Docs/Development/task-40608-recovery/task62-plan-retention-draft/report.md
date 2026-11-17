# Task62: direct plan lookup and terminal control receipts

Artifact-only implementation for the existing source-audit finding. No live
product/test files were edited. No Python imports, native work, formatter,
compiler or tests were run by this helper. Root owns execution and integration.

ADR required: no new ADR. Applicable ADR:
`backlog/decisions/126-complete-local-backup-and-recovery.md`, complete Console
service owner / Task62. This implements the existing positive physical
retirement and finite command-identity contract without changing ownership,
authority, storage checks, protocol or the original journey timing contract.

## Proposed targets

| Artifact under `proposed/` | Live destination |
| --- | --- |
| `console_control_dispatch.py` | `tldw_chatbook/Chat/console_control_dispatch.py` |
| `console_execution_retirement.py` | `tldw_chatbook/Chat/console_execution_retirement.py` |
| `test_console_control_plan_retention.py` | `Tests/Chat/test_console_control_plan_retention.py` (absent) |
| `test_console_execution_retirement.py` | `Tests/Chat/test_console_execution_retirement.py` |

The three existing targets have exact saved preimages. The retirement preimage
includes root's current trace-drain correction and original body-free diagnostic;
this artifact preserves those bytes outside its small change. Central's pending
`_read_projection` readiness-warm insertion is independent and must be reconciled
at adoption rather than overwriting a later live dispatch file.

## Change and preserved behavior

- The existing `ConsoleTurnRetirement` owner retains the keys of plans actually
  registered by its original task. Registration validates that exact owner,
  identity, custody record and task. Its list length supplies the next ordinal
  without scanning service history. Keys remain until positive retirement, so
  abandoned plans cannot cause ordinal reuse during the turn. The existing
  `_turns` membership barrier continues to reject reuse of a retired turn ID.
- Native effect admission finds the exact existing owner by preparation ID,
  follows its existing `_ControlReservation.pending`, and checks exact reservation
  type/receiver/owner and exact keyed pending membership. It retains every prior
  plan identity, task, state, revision, effect-open and fenced-turn check. Original
  source/native preflight and transaction checks are unchanged.
- Publication, source fencing and drain visit only the existing active/unresolved
  retirement owners. A scalar publication order, derived from the still-retained
  map size at registration, preserves the old global insertion ordering across
  interleaved turns; only current waiting plans are sorted.
- Compaction is inline in `_settle` after **all** existing positive physical
  checks, including original trace-worker drain, task completion, logical/source
  closure, qualified reads, admitted trees and finite effects. Under the original
  lock it validates the exact turn entry and every indexed terminal plan before
  changing anything. No additional callable can invoke compaction ahead of that
  proof. Uncertain/started/nonterminal records refuse compaction and retain the
  original unresolved registry entry.
- Successful compaction keeps the original `_PendingPlan` object with its
  fingerprint, received-request identity, ordinal, revision and terminal state,
  while releasing sealed plan, Task, Future and recovery projection. `_turns`
  keeps its original `ControlIdentity` with `None` in the three heavy slots.
  Matching late finished ACK still returns True. A mismatching request,
  fingerprint, session or incarnation still fences the old turn, records conflict
  and refuses; cancelling a positively retired Task is unnecessary. Abandoned,
  conflict and uncertain ACKs retain their original refusal. All command epoch,
  sequence, request, session and source checks remain in their original order.
- Negative/uncertain retirement does not compact, discard or replace its original
  records. Drain still waits for those exact owners and refuses unresolved native
  outcomes. There is no fallback to empty maps or process exit as proof.

The retained fingerprint is the original body-free immutable
`ConsoleDurableAcceptanceFingerprint` in `console_chat_store.py`: IDs, title hash,
origin and digest. `ControlIdentity` contains bounded IDs/epochs and scalar
sequence/revision. The larger recovery projection is removed because late ACK
comparison never reads it. Small identity receipts, command replay IDs and turn
fence IDs intentionally remain proportional to lifetime history; exact indefinite
late-ACK/replay semantics do not permit deleting them. Heavy native/session graphs
are released only after positive retirement. No new active/settled registry is
introduced.

## Complexity

For H lifetime plans, A plans belonging to active or unresolved turns, W waiting
plans, and T active/unresolved turns:

| Operation | Before | Proposed |
| --- | --- | --- |
| Next per-turn ordinal | O(H) | O(1) |
| Native pending-plan lookup | O(H) | O(1) keyed lookup |
| Waiting plan publication | O(H) | O(A + W log W), same publication order |
| Source plan fence | O(H) | O(A) |
| Turn task collection for drain | O(lifetime turns) | O(T) |
| Positive retirement compaction | no heavy release from maps | O(plans in this turn) |

These are source-level complexity facts, not measured contributions to the
Windows journey time. Failed or uncertain owners intentionally remain within A/T.
This artifact does not claim the sub-five-second target or any measured speedup.

## Causal checks authored, not executed

1. New `test_original_turns_use_direct_lookup_and_retain_only_terminal_identity`
   uses the original domain fixture, actual loopback provider, original sealed
   plans/native transactions and positive retirement. It performs two sequential
   turns, matching late ACK, conflicting late ACK, refusal after conflict, late
   cancellation refusal, turn-ID replay refusal, and original source fence/drain.
   Mapping sentinels preserve
   exact original values/keyed operations but reject lifetime enumeration; no
   owner, native check, callback or retirement outcome is fabricated. The baseline
   has a direct causal failure at the old `commit_plans` lifetime scan. Fixture
   cleanup restores ordinary map containers so that baseline refusal does not
   manufacture a second teardown failure. Candidate must complete the actual
   transactions and assert release of the heavy slots. Two loopback requests also
   confirm that late ACK did not start a second provider turn.
2. Existing cancellation test (before original entry / before original ACK) gains
   assertions for the compact turn identity and, where an actual plan existed,
   released heavy slots plus continued late abandoned-ACK refusal. Its original
   actual cancellation and physical retirement assertions remain.
3. Existing actual archive-close uncertainty test gains exact retained
   identity/record/task/source assertions after its original negative physical
   outcome. Its injected native-close failure, original retained lease checks and
   negative drain behavior remain unchanged.

No additional test matrix or synthetic positive proof was added. The root must
run the new regression against baseline for its expected scan failure, then the
candidate and the existing changed retirement tests. The unchanged README/edit
three-pair acceptance remains separate and still required.

## Source-only checks and handoff

Read the audit, current dispatch/retirement paths, original loopback fixtures,
original acceptance fingerprint and applicable ADR. Reviewed both product diffs
and authored test assertions. Git whitespace checks produced no diagnostics.
Manifest records proposed hashes, exact preimages/absence and eleven byte-identical
dependency snapshots; live targets/dependencies were checked against those pins
at handoff. This is not syntax, lint, test or runtime evidence.

## Surgical patch handoff

- `product.patch` contains only dispatch and retirement changes:
  90 insertions / 16 deletions across the two existing files.
  SHA256 `491d380c0efa0457cc354efdfd642238af5980f02f004e3520aa84bd5bcc7993`.
- `tests-only.patch` contains only the 25 additional assertions/import lines in
  the existing retirement test and the new 172-line regression. Its new-file
  hunk uses `--- /dev/null` and `new file mode 100644`.
  SHA256 `c96454e561a2f750e363d0646f644494f49046c6b9b96db76b9ba40e54ddffc5`.

Both patches use repository-relative `a/` and `b/` targets. Only diff text line
endings were normalized to LF; saved preimages, dependencies and proposals were
not rewritten. Separate `git apply --check` calls succeeded against current live
bytes for both patches. `git apply --stat` confirms exactly two intended files
per patch. Neither patch was applied. Tests-only can therefore be installed by
root for the causal red gate before product integration. Raw dependency copies
remain local to this draft; the portable report, proposals and patches can be
backed up at root's next checkpoint.

## Bounded follow-up: late cancellation and exact native map

Original `console_runtime.py:3243` removes the exact turn from `_turn_custody`.
Original task completion's finally at `:3906` invokes that release before the
terminal callback at `:3909`. `ConsoleTurnRetirement.terminal` already checks
that custody is absent, and `_settle` requires logical closure. Consequently a
late `CancelTurn` already refused at original dispatch `:1475` because runtime
custody no longer matched. The compact `None` record refuses earlier with the
same error. The new test now checks both custody absence and that refusal after
an actual positive retired turn.

Original `console_native_commit.py:699` refuses a second native owner for the
same preparation ID. Before worker submission, `:470-488` verifies exact owner
type/issuer/store/session/acceptance/task and both controller/store maps against
the supplied plan. The worker join at `:317-335` survives repeated cancellation;
the outer original caller owns the registration at `:1145` across
`controls.commit_received_turn` at `:1199`. Thus a still-entering original worker
cannot lose its valid owner to a successor with the same preparation ID. The
new direct lookup requires the retained reservation to belong to that owner,
its pending record to be the exact keyed member and its plan to be the supplied
object, then applies every old state/task/revision/fence check. Older completed
or abandoned pending records are not valid admission fallbacks. No additional
native owner lifetime or authority is introduced.
