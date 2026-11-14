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

## Coordinate the live verifier

You are the PRIMARY coordinator. The host supplies the verifier's private brief
and tool schemas. Batch/checkpoint reads, dispositions and verifier_complete
belong to that child; do not perform them yourself or duplicate its decisions.
Wait for its actual terminal receipt with wait_agents using its fleet handle.
A check_requested notice is not completion; handle it as described below, then
continue waiting for the same live verifier. Do not launch a replacement.

## Executable checks: same live verifier

Stock Raw CLI has no enforceable aggregate write/file quota boundary: executable
checks return unavailable before copying or launch. Proposed commands are not
execution evidence. The verifier can use source evidence or leave a claim
unresolved with a specific next_check when execution is indispensable.

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

For an inline verifier, spawn is synchronous; proposals resolve unavailable
immediately because the parent cannot service a threaded check. Use its actual
terminal receipt. Never enable a fleet or spend a replacement child allowance.

## Unified report and completion

After actual verifier settlement, fetch fresh pages, starting without a cursor:
code_review_read({"review_id":"RID","view":"summary"}) for each participant's
capture, terminal and seal status, then
code_review_read({"review_id":"RID","view":"report"}) for the complete originals
and their accepted dispositions. Follow each returned cursor with only review_id,
view and cursor. Previously read pages are immutable snapshots and can predate
verification. Each report record identifies original_id, record, disposition and
verifier_report_status. The host expands an exact duplicate unit's decision to
each original separately. Do not ask for verifier-only batches or checkpoints.

Require sealed reports and every original's confirmed/refuted/unresolved decision.
Use accepted structured decisions as the handoff source; raw child prose is
retained separately and can contradict them after history reduction. State any
such discrepancy without replacing accepted decisions, repeating verification,
or calling check_actions with no actual proposal. Report/source gaps and pending
or incomplete seals must remain explicit; a tool refusal is not permission to
perform another role's work. A sealed unresolved verdict remains unresolved.

The native reader presents the unified report separately from the original
reports. Summarize the accepted verdicts and their limits, then RETURN when this
handoff is complete; no additional source reads or submissions are needed merely
to repeat the verifier's work. Console `u` reopens retained native reports;
export/discard remain explicit native actions. Do not claim complete verification
when mapping or seals are incomplete. No automatic fixes, comments or merges.
