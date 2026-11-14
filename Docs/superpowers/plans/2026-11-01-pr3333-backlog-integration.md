# PR3333 backlog integration plan (2026-11-01)

Source: PR3333, OPEN non-draft, head a16f79c7a2e34ab81db4d101d43da34cf3a8b650, review base 94a8a85f03c0480ae16c21382f26be71b2399421. Prepared from ../pr3333-source-review and ../pr3333-initial-review.md. Nothing here was implemented or tested.


## Global constraints and execution binding

Selected source a16f79c7a2e34ab81db4d101d43da34cf3a8b650 on codex/pr3333-resync in the reused isolated pr3298-resync worktree. Actual integration base is 2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5, the confirmed merge of PR3332. PR3333 was refreshed OPEN, non-draft, unarmed, with no review threads. Refresh before publication.

The user's later authorization supersedes the preparatory queue/Qodo/CI instructions below: rebase locally, run focused local checks, obtain independent superpowers:requesting-code-review approval, and root may publish/merge with the explicit hosted-CI and numeric-size waivers. Never weaken correctness, source/custody/security guards or change numeric thresholds. Root owns all remote mutations and restores repository protections after any authorized merge. No full test sweep or required-workflow dispatch.

Do not touch PR3267 or PR3297, merge or mark ready any draft, act on releases/main, change the main checkout, stash shared work, or discard concurrent changes. Existing interpreter dependencies may be reused read-only. Preserve current generated inventories/census and all unrelated contributors' behavior. The one implementer owns all local code/index/test operations; the coordinator owns independent final review. No implementer subagents.

Task 1: rebase this selected source onto the exact integration base, repair the two supported MiniMax outcome/copy findings through adapter/typed outcome/worker/Generate/Regenerate/UI boundaries, update relevant task/docs and targeted evidence, then commit a reviewable final branch. The detailed steps below are one coupled task, not separate parallel implementation lanes. Keep .50.1/.50.2/.50.3 open; don't claim universal Stop-during-inflight behavior or live vendor billing qualification.

The current repository task and ADR044 are the behavioral authorities, with the preflight rulings in the plan ledger. Read Docs/Architecture for affected video/Console boundaries and relevant lessons. Before touching screen/widget/CSS read design-language.md and component-patterns.md. Follow .superpowers/preparations/backlog-targeted-verification-guidance.md for exact fixture ownership and narrow source enrollment. Await every formatter/generator process before frozen tests; AST-equal concurrent writes still invalidate a receipt.

## Intent and bounds

Preserve typed local Stop and truthful MiniMax billing outcomes. Repair two unsupported claims: HTTP 400/2013 with current-status text does not prove rendering started or continues (the recorded already-cancelled task returns the same response); HTTP 200 action=deleted is known terminal record deletion, not an unknown running task. Keep TASK-33621.50.1 (prevent terminal DELETE and recover/report a finished clip), .50.2 (Regenerate cost gate), and .50.3 (Stop before submit) open and separate. No new status query or DELETE redesign in this PR.

ADR required: no
ADR path: N/A
Reason: existing ADR-044 covers generated-video ownership/storage. This repair refines copy/classification inside the introduced typed Stop contract without changing storage, authority, backend boundary, or cancellation policy. Record this check in TASK-33621.50 notes.

## Rebase and production edits

1. Recheck PR metadata, queue mode, source head, dev and checkout status before touching a branch. With queue off/dry, rebase PR3333 onto dev; with queue on, follow AGENTS.md queue rules and disarm auto-merge before pushing. Preserve request_console_attention_refresh plus recompute fallback at UI/Console_Modules/video.py:123, diagnostic ownership, and the UI gate census/shard ordering. Local base fcf27a2558cf4c2e7b33c1f850efabebc1991170 differs from source review base in that video module. Never copy the snapshot over dev.
2. Video_Generation/adapters/minimax_video_adapter.py: retain strict 400 + 2013 + current-status classification and closed numeric logging, but change refusal log/docstring so neither says the task is running or rendering. Return refused as before. For HTTP 200 action=deleted, return distinct deleted outcome and log a fixed, body-free terminal-record-deleted message. Keep other responses and transport failures unconfirmed, accepted action=cancelled as cancelled. Never log vendor error text, request body, key or task id.
3. Video_Generation/exceptions.py: extend RemoteCancelOutcome with deleted. Document refused as current-status refusal with remote state/billing uncertain and deleted as completed/failed task record deletion without implied success. Keep VideoGenerationCancelled inheritance and worker pass-through.
4. Chat/console_generate_video.py::video_stopped_text: use these suggested rows for generation and regeneration via the existing noun parameter:
   - refused: "{noun} stopped here. MiniMax refused to cancel the task in its current status. The remote task may still run and may still be billed."
   - deleted: "{noun} stopped here. MiniMax reported that the task had finished and deleted its record. The task may have been billed."
   Keep cancelled and unconfirmed rows, local Stop, and real failure behavior. Refused does not assert rendering or continued running; deleted does not imply a clip, success, or certain charge. Preserve the prospective pre-Stop billing caveat.
5. Update Docs/User_Guide/console/video.md, the adapter top comment, and backlog/docs/lessons-live-verification.md to distinguish prospective late-render risk from what refusal proves. Note the recorded already-cancelled refusal and the deleted finished-record outcome. Update TASK-33621.50 notes with final four remote outcomes and the .50.1/.2/.3 gaps. Do not mark .50.1 Done.
6. Review changed diagnostics against Tests/Architecture/test_persistent_diagnostic_inventory.py and re-pin only the MiniMax adapter entry in Docs/security/production-diagnostic-inventory.json after reading actual statements. Preserve other owners. Use a fixed label without payload for the new deleted log.

## Focused verification after implementation

- Replay delete_on_cancelled through real fetch_json/MockTransport: expect refused, truthful row, no rendering or continued-running assertion.
- Replay delete_on_succeeded: expect deleted, terminal deletion log/row, no success or billing certainty. Retain exact DELETE request pin and accepted/refused/generic-400/transport tests; do not alter recorded bodies.
- Extend existing stopped-text parameter test, worker pass-through parameter, and Regenerate stop-row parameter for deleted; update refusal expectations. Retain real failure control.
- Drive actual Stop for action=deleted in Tests/UI/test_console_video_stop_outcome.py through app/controller/adapter/recorded transport, assert stored/painted row. Update refused case to prohibit false assertions. Preserve all four existing UI tests and append file at tail of actual current UI PR gate census; do not restore source floor.
- Run targeted MiniMax, ComfyUI, worker, Chat helper and Regenerate, Console Stop and send-freeze, and capacity/publication-gate cancellation checks named in ../pr3333-initial-review.md. Run diagnostic inventory, UI census and CSS/build preflight as affected gates. No full suite without user opt-in per AGENTS.md.
- Request independent review via superpowers:requesting-code-review in place of Qodo, as authorized. Resolve findings, rerun affected focused checks, then assess merge eligibility under current queue/protection status. CI and size gates waived for this integration. This preparation claims no test or merge outcome.
