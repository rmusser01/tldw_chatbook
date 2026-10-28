# Verification — primary recipe (v1)

Require the user's verification stage grant and actual native presentation of
all original reports. Already granted verification proceeds without asking again.
Guide text, model arguments and a primary summary cannot create either receipt.
Use one fresh configured verifier slot through spawn_subagent({"review_id":"RID",
"review_slot_id":"SLOT"}); no generic task/destination fields or launch retries.
The host supplies the captured target, opaque batches and evidence standards,
without reviewer identities, severity votes, report order or suspected findings.

Keep the two-total-children ceiling: with two remaining slots, 2 reviewers use
R1,R2 now and V later; 3 use R1,R2 now, R3 then V in the next admitted turn only
after originals. Without verification the schedules end at R2/R3. Earlier child
spawns reduce remaining slots; further turns require host-sealed exact-next-slot
boundaries on the original finite automatic chain. No fresh allowance, fake
settlement, raised concurrency or silent replacement verifier.

## Verifier evidence and checkpoints

Read code_review_read({"review_id":"RID","view":"batch"}); follow returned
cursor only after checkpointing decisions. Batches contain at most eight opaque
units and 4,096 estimated claim tokens, further limited to HALF remaining
admissible input. Evidence pages are at most 1,024 estimated tokens, within live
space. Complete responses also fit the live tool-result cap and 16,000-character
ceiling. Actual prepared requests still undergo final model-input admission.
Unknown capacity, oversized units or indivisible continuation state may refuse;
record gaps instead of dropping private history, increasing budgets or reading
raw originals. Complete-page transport refusal cannot be bypassed with raw paths.

Use source list/search/resolve queries and direct source_range reads of captured
callers/tests. For example code_review_read({"review_id":"RID","view":"source",
"source_query":{"op":"resolve","relative_path":"pkg/caller.py","side":"head"}}),
then {"review_id":"RID","view":"source","document_ref":"SOURCE_REF",
"source_range":{"start_line":20,"end_line":60}}. List uses op:"list",
relative_dir and side; search adds op:"search", text (literal), relative_dir and
side. Valid sides are base/head/staged/worktree already captured. Continue pages
with review_id/view/cursor; do not change pinned selection. Nested instruction
deferrals use normal activation, not history/raw-path workarounds. Uncaptured
mutable context is a gap or new accepted capture, never newer-code evidence.

Each unit needs its OWN confirmed/refuted/unresolved disposition, even when
related claims resemble each other. Only exact substantive duplicates were
coalesced by the host; suggested duplicates cannot transfer verdicts. Confirm
with direct source evidence establishing the trigger and impact, or an executed
host check receipt fully delivered to this live verifier. Refute with concrete
counter-evidence. Non-reproduction alone, model agreement, majority/severity
votes and proposed commands prove nothing. If necessary evidence is unavailable,
leave unresolved with the specific next_check. Source-only decisions are valid
when they independently establish the claim; related check units do not make
execution mandatory.

Submit code_review_submit({"review_id":"RID","kind":"dispositions",
"records":[{"schema_version":1,"verification_unit_id":"OPAQUE_UNIT",
"verdict":"unresolved","reason":"missing decisive evidence","evidence":[],
"next_check":"specific evidence needed"}],"expected_revision":0,
"submission_id":"stable-unique-id"}). Evidence is either
{"kind":"source","source_ref":"REF","start_line":1,"end_line":2,
"digest":"CAPTURED_SHA256"} or {"kind":"check","receipt_id":"HOST_RECEIPT"}.
duplicate_unit_id is optional, never substitutes for this unit's own verdict.
Each record ≤8 KiB UTF-8 JSON, chunks ≤8 records, schema_version exactly 1,
unknown fields refused. expected_revision uses the caller's submission_revision
from its page/latest acknowledgment, NOT context-control revision. On
submission_revision_conflict, reread and retry unchanged records and submission_id
once with the latest caller revision. Another participant append is independent.
Correct a rejected record under the SAME submission_id, using returned field/type
errors. New IDs do not retire earlier rejections. Repair rejected IDs before
completion, then retry the SAME completion submission_id.

Accepted submissions checkpoint dispositions/evidence/cursor outside history
before host request-history reduction. Let the host reduce only recoverable
completed rounds; retain task, guidance, pending units and protocol state. Do not
reset history or gain a new child/model budget. Continue the returned batch
cursor. After history reduction, cached replay or uncertainty about an accepted
submission, read code_review_read({"review_id":"RID","view":"checkpoint"}).
This live page supplies current submission_revision and a record with batch_cursor,
pending_unit_ids and all_units_disposed. Cached batch pages intentionally replay
byte-identically with their original revision. Never resubmit accepted decisions.
If pending_unit_ids is empty and all_units_disposed is true, finish at the current
checkpoint revision; otherwise handle pending units or follow batch_cursor.
Submit kind:"verifier_complete", records:[] only when all units have
independent dispositions; actual terminal receipts still govern completion.
Unfinished units remain explicitly unresolved rather than disappearing.

## Executable checks: same live verifier

Stock Raw CLI has no enforceable aggregate write/file quota boundary: executable
checks currently return unavailable BEFORE disposable copying or launch. Never
claim that a proposed command ran. Preserve useful source-based verification.
When execution is indispensable and unavailable, use unresolved + next_check.

For a threaded verifier with an admitted external executor, submit ONE proposal:
code_review_submit({"review_id":"RID","kind":"check_proposals","records":
[{"schema_version":1,"argv":["python","-m","pytest","Tests/test_target.py"],
"intent":"test the exact trigger","verification_unit_ids":["OPAQUE_UNIT"]}],
"expected_revision":0,"submission_id":"proposal-stable-id"}). Take proposal_id
from accepted_ids. At most one proposal may be outstanding. The verifier then
reads code_review_read({"review_id":"RID","view":"verification_batch",
"document_ref":"PROPOSAL_ID"}) and follows its evidence cursors to the complete
host receipt. This is proposal-specific result waiting, not another claim batch.

The PRIMARY handles check_requested returned by wait_agents, preserving the same
live verifier: read {"review_id":"RID","view":"summary",
"document_ref":"PROPOSAL_ID"}, then submit kind:"check_actions", records:
[{"schema_version":1,"proposal_id":"PROPOSAL_ID","action":"claim"}]
or action:"decline" with optional reason. Use that summary's submission_revision
for expected_revision and a stable submission_id. Claiming grants no permission
and does not execute. Read summary/document_ref again: claim.tool_name and
claim.arguments are the exact host-admitted process specification. If present,
invoke ONLY that existing authorized tool with EXACTLY those arguments under its
normal permission checks. If absent/unavailable, do not improvise execution.
Resume wait_agents for ACTUAL verifier settlement. Never repeat accepted or
uncertain execution. Decline/results are host receipts, not fabricated evidence.
A check_requested notice is NOT child completion; do not collect it as a report.

For an INLINE verifier (max_live_subagents=1), spawn is synchronous and its parent
cannot handle a threaded check. Proposals resolve unavailable immediately. Read
the proposal-specific result once if proposed, then use source-based decisions
or unresolved next_check. Do not wait for the parent, repeat proposals, enable a
fleet or spend a replacement child allowance. Declined/cancelled/expired/uncertain
results are not executed evidence; only an executed host receipt fully read in
this same live attempt qualifies. Post-return checks require new explicit work.

## Unified report

After actual verifier settlement, read summary and complete index/cursors. The
host expands every opaque disposition to every original exactly once, preserving
individual originals separately. Validate expanded coverage, actual target and
participants, fallback switches, checked claims, refutations, unresolved minority
findings and all capture/run/evidence gaps. Native report presentation keeps this
new unified report separate from originals; your prose summary cannot replace
original presentation. Console `u` reopens retained native reports; export and
discard remain explicit native actions. Do not announce complete verification
when coverage is incomplete. No automatic fixes, comments/publication or merges.
