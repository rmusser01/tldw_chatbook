# Independent multi-model review — primary recipe (v1)

Use only after an accepted review request. Guide lookup starts no capture, child
or model call. Default teams are empty: configure ordered preset IDs/focuses and
an optional verifier in Settings → Agents, or obtain explicit run-only choices.
Never choose a paid destination automatically. A guide is instructions, not
permission: preserve routing consent, path exclusions, instruction delivery and
all existing budgets. Do not change settings or preset models for run overrides.

## Prepare the exact target

Discover tools with find_tools/load_tools when needed, then call
code_review_prepare with target and optional reviewers/verifier. Real preset IDs
are values; display names are labels. A run reviewer is
{"preset_id":"ID","focus":"optional additional focus","destination":{"provider":"PROVIDER","model":"MODEL"}}.
Omit destination to use the preset. An override supplies provider and/or model,
never an empty object; a verifier uses the same shape with empty focus.

Target schema (use only fields for that kind):
- All: binding_ref, scope (relative, default "."), context_paths (relative array).
- working_tree: kind, staged (bool), unstaged (bool), untracked_paths (array).
- commit_range: kind, base_commit, head_commit (exact commit hashes), comparison
  ("direct" or "merge_base").
- pr: same exact commits/comparison plus pr_ref.
- patch: kind, patch_ref, optional preimage_commit.

For example, code_review_prepare({"target":{"kind":"working_tree",
"binding_ref":"ADMITTED_BINDING","scope":".","staged":true,"unstaged":true,
"untracked_paths":[],"context_paths":["pkg/caller.py","Tests/test_caller.py"]}}).
Use the admitted default_target when supplied; do not guess binding IDs or commits.
Explicitly capture mutable callers/tests with context_paths before launch.
Committed context can be lazy; uncaptured mutable files need a new capture or an
explicit gap. Preparation returns review_id, admitted slots, capture/route gaps
and continuation requirements. Missing/disabled presets, duplicate destinations,
unknown input capacity, unavailable history, drift, binaries and quota failures
must remain visible. Do not substitute another model or silently narrow scope.
State any coverage_gaps (plus coverage_gaps_omitted) before launching reviewers;
when they narrow the requested scope, get the user's scope choice first.

## Inspect through captured references

code_review_read({"review_id":"RID","view":"source","source_query":
{"op":"list","relative_dir":"pkg","side":"worktree"}}) lists admitted source.
Use op:"search", text:"literal", relative_dir and side to locate callers/tests;
use op:"resolve", relative_path:"pkg/caller.py", side to get its source_ref.
side is one of captured base/head/staged/worktree, never a new revision.
Read direct lines with {"review_id":"RID","view":"source",
"document_ref":"SOURCE_REF","source_range":{"start_line":20,"end_line":60}}.
Follow cursor using only review_id/view/cursor; it pins the original selection.
List/search and evidence pages are bounded. The full response, including metadata,
must fit the live tool cap and 16,000-character ceiling. Never treat a shortened
page as complete. Instruction deferral means wait for normal guidance activation;
a complete-page transport refusal is an explicit gap, not permission to read raw
paths, Git history, scratch bundles or unrelated chat history as a workaround.

## Launch, settle and present

For each admitted slot call spawn_subagent({"review_id":"RID",
"review_slot_id":"SLOT"}). Supply ONLY these fields: no task, provider/model,
name or generic overrides. The host composes the same independent general brief,
captured target and optional focus. Do not supply suspected findings or peer
reports. Do not silently retry accepted, failed or uncertain launches.

Preserve two TOTAL children per primary turn; concurrency is a separate cap.
With two available slots, the exact schedule is:
| Reviewers | Verification already granted | Current turn | Later turn(s) |
| 2 | no | R1,R2 | none |
| 2 | yes | R1,R2 | V |
| 3 | no | R1,R2 | R3 |
| 3 | yes | R1,R2 | R3, then V after originals |
Earlier ordinary spawns reduce remaining slots. For N reviewers, V=0/1 and R
remaining slots, additional two-child turns = ceil(max(N+V-R,0)/2).
Only host-sealed exact-next-slot continuations on the original finite chain can
continue; never reset budgets, fabricate completion or force a new turn. Kill
switches, manual priority, stop/close, chain exhaustion and uncertain acceptance
can block visibly. Later explicit verification acceptance is new user work.

Threaded launches return fleet handle IDs in their running receipts: wait_agents({"ids":["HANDLE"]})
until actual terminal settlement. Inline execution at max_live_subagents=1
returns synchronous spawn receipts; use those actual receipts without forcing a
fleet. A running notice or check_requested is not child completion.

Read summary and the complete host index with code_review_read using view
"summary"/"index" and returned cursors. Retain capture/coverage gaps and actual
requested/used destinations and fallback switches. Never build a complete index
by concatenating prose or bounded fleet summaries. Native originals are presented
in the session report reader before verification; the host records actual mounted
presentation. Your summary text cannot mint that receipt. Each original keeps its
attribution, coverage, structured findings and separate raw output. Console `u`
reopens retained reports; explicit export/discard uses native reader controls.

## Participant completion

The host supplies each reviewer with its own submission schemas and completion
brief. You are the coordinator: do not submit findings, coverage or reviewer_complete
on a child's behalf. Accepted completion intent alone is not a sealed report;
read the host summary after actual terminal settlement. Empty findings require
complete successful coverage. Failed/cancelled/exhausted roles remain incomplete.

## Optional verification

Present every original first, then offer a fresh configured verifier. An existing
explicit stage grant needs no redundant confirmation; a model boolean cannot
grant it. Load code_review_guide({"stage":"verify"}) for the primary sequence.
Use evidence, never consensus as proof. Never fix code, publish comments or merge
as part of review. Missing execution authority leaves necessary checks unresolved.
