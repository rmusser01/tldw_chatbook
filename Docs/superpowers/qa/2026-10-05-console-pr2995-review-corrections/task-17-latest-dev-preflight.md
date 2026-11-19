# Task17 latest-dev command integration preflight

**Verdict: clean, bounded composition; root can select this integration scope.** Approved/published head `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036`; previous integrated dev `a78a9a900b4901e33c031f830dd2d80224d5147d`; selected incoming dev `8c4dfe59a243ce0cec8e131aff3935646c64b298`. Incoming PR3006 is 10 commits /19 paths, TASK-33622.16. No tests, collection, live provider calls, installs, subagents, checkout, rebase, tracked-source/index/HEAD writes or Git object creation occurred. Only temporary read-only sketches and these own-SDD report artifacts were produced.

ADR required: no new ADR. Existing ADR219/220 preserve source/start/decision ownership; ADR094 preserves accepted runtime lifetime; ADR097 governs unchanged boot budgets. Read the actual selected-dev TASK-33622.16, relevant testing/live/Textual/Console lessons and design language. No CSS, tokens, bindings, new authority, schema or dependency boundary is required.

## Exact incoming ownership and overlaps

All19 paths, selected-dev blobs/SHA256/bytes, head/base/candidate hashes, changed declarations and incoming AST carry are pinned in `task-17-latest-dev-preflight-selection.json`. **16 paths take exact selected-dev whole-file bytes.** Only three require composition:

- `tldw_chatbook/UI/Console_Modules/wiring.py`: both sides change the top-level `build_console_controllers` function. Incoming changes only the skill controller's append lambda from `lambda message` to `lambda message, **kwargs`, forwarding the supplied `session_id`. Preserve all existing feature Session/compaction/recovery/hook bindings. The clean merge-file sketch is2444 lines, +2 against head2442; all other declarations remain unchanged.
- `tldw_chatbook/UI/Screens/chat_screen.py`: **no shared changed class method**. Incoming changes `_send_console_message_from_visible_action_observed`, `_console_command_doctor` and `_console_command_fewer_permission_prompts`; PR changes different methods, including adjacent `_dispatch_console_draft_send`, pending-handoff draft persistence, and start-aware controls. Preserve the incoming command handoff and origin-output helpers exactly alongside the feature. Both sides also retain their unchanged surrounding methods and incoming comment-only key-handler changes.
- `Docs/security/production-diagnostic-inventory.json`: selected dev adds only the `command_handoff.py` row:3 calls, digest `9cf00a2c9afc387b0ba0`. Its clean merge retains that exact row; removing only the row recovers head JSON exactly, including prior relocated diagnostic owners. No inventory refresh or broader diagnostic scope is needed for this preflight.

All three `git merge-file -p HEAD BASE DEV` sketches exit0, including JSON. They exist under `/private/tmp/pr2995-task17-preflight/merge-sketch-*`; exact input refs, argv and output hashes are in the selection JSON. No sketch was installed as source. A fresh source worker must separately prove its eventual rebase/composition and exact upstream ownership.

Incoming non-overlap production changes: new lazy `command_handoff.py`/`command_draft.py`; image/video generation draft take/restore; prompt/system post-search origin guards and captured system draft take; skills/stream-video output attribution; message method documentation. Incoming tests, task, lessons and user-guide additions retain exact upstream assertions, markers and text.

## Actual boundary intersections

**Visible send and timing.** Only the recognized-command branch changes from an inline awaited dispatch to a screen-owned, non-exclusive `console-command` worker. Enter's app callback, Send/Workbench and spoken routes therefore return without waiting on a modal or generation. Argument-free rewind keeps its existing callback path. The helper sets origin/captured-draft contextvars inside its own worker; it does not rely on inherited Textual worker context or worker-task assignment timing. Same captured revision/session repeats drop until completion; other commands can run concurrently. `WorkerCancelled` from a cancelled modal wait ends quietly; the release remains in `finally`.

**Native accepted turns and automatic starts.** The worker dispatches only recognized slash commands; it never submits an automatic opening prompt or calls ordinary runtime custody. ADR219 automatic starts still use literal native input without composer command parsing. `_dispatch_console_draft_send`, hooks/prompt queue, controller/store/runtime/start/interrupt/compaction authorities retain exact head bytes/ASTs. No allowance, source observation, admission, reservation, launch acceptance or durable commit fence changes.

**Draft scope and origin.** Existing composer stash/edit-serial/generation APIs remain authoritative. `take_command_draft` commits only the captured revision; a failed take still retains the command for a resend hint. Restore requires the same composer, unchanged scope generation and empty current text. The feature's pending handoff writer still reads current composer text and persists empty edits; generation changes on existing switch/load/send paths continue to prevent restoration into a sibling. `/system` and `/prompt` refuse after their awaited search if origin no longer shows; awaited reporting commands append to their origin through `session_id=`. The skill append lambda is the only overlapping injected callback change.

**Typed answers.** The ordinary-text tail of the changed visible-send method remains structurally identical: `_answer_pending_question_with_draft` precedes ordinary dispatch, while slash parsing still precedes it. The actual final Task16 typed-card/bootstrap/adapter source is exact and must stay intact. One exact real-round/card/composer node is proposed because this function is now requalified; all prior typed-adapter and original58 receipts remain historical and unreplayed.

**Close and physical worker lifetime.** Command workers belong to their screen; view removal cancels them. Accepted native/automatic turns still belong to `ConsoleRuntime`; navigation/Close/admission/physical-drain/retention owners are untouched. The command start guard checks both active-session and visible-draft identity. The existing explicit-session output append catches vanished-session `KeyError`, so a reporting command cannot resurrect a closed chat or write into the replacement. Existing generation/session capture and cancel-event ownership remain upstream. No claim is made that these screen commands acquire runtime custody or that this integration newly qualifies every Close race. Carry the actual original Close/source-cutoff/native receipts and owners by exact hashes.

## Cap and loading verdict

| Owner | Candidate actual | Unchanged limit | Headroom |
| --- | ---: | ---: | ---: |
| Console controller |29301|29301|0|
| Console store |22334|22344|10|
| Interrupt owner |6479|6479|0|
| Compaction owner |4185|4185|0|
| ChatScreen |25192 lines /759 methods|25218 /759|26 lines /0 methods|

These are actual parsed/read-only candidate measurements, not test-pass claims. The screen's existing50-line slack rule does not require tightening its26-line remainder. No existing cap/threshold/assertion is edited; controller/store/interrupt/compaction are exact head files.

**No new loading or payload node is justified by this diff.** `command_handoff` and `command_draft` import only from invoked command/generation bodies. Prompts add two constants from the grammar module they already imported. All existing source owners retain module-import nodes except that same-module prompt import expansion. The composed wiring keeps constructor invocation structure and has no eager helper import or new constructor work. ChatScreen/module wiring/image/video/prompt/skill owners are already in the app+Chat prewarm baseline; the two lazy modules never execute during registry import. The marginal preimport module/LOC projection is unchanged. Route/CSS/startup timer/worker owner sources remain exact.

Keep ready1033, preimport557/425347/135111, app686 and CSS608090 limits unchanged. Original35 loading at a688 and Task16's payload557/415370/127527 remain at their original source and warning receipts. This preflight supplies scoped carry, not fresh module census or timing evidence. An eventual new early import or constructor/route change requires root to select its specific loading node before execution.

## Smallest proposed qualification

Selection JSON contains **22 explicit node IDs /22 source-derived cases**, pending root selection and actual bounded collection:

- 8 origin/wiring cases: delayed system switch on Enter/Send; appended/replaced captured drafts; doctor/skills/fewer-permission outputs; command refuses a foreign origin before starting.
- 10 timing/draft cases: both routes under lazy/eager factory; another command during generation; cancelled modal wait; Stop preserves typed text on both routes; switched-to typed draft survives failure; failed image take-left-behind still offers resend.
- 1 real typed-answer round/card/composer control through the changed visible-send function.
- 3 static cases: the two ChatScreen size/slack nodes and the exact production diagnostic inventory/topology node.

No full new test-file owner, command-helper24-case file, generation suite, native/Close suite or historical58/35/61/27 phase replay is proposed. Every selected node keeps its exact upstream/current assertions and canonical private profile; use the actual worktree PYTHONPATH/shared Python3.12 and unchanged300s timeout. Keep actual IDs/argv/source hashes/warnings/failures at their own source. Do not introduce skips/XFAIL, suppress warnings, broaden recovery/interrupt fix scope, or clean inherited formatting to make this integration look green. Root's separately selected recovery/interrupt work follows this integration and has separate provenance.

## Historical preservation

Incoming paths are disjoint from every QA/archive path. Verified all1983 original Task16 QA rows still match exact current Git blobs and original digest `ad92f9666141271109dde626bd96522d969f0dd366da9a0db931e492b3940d04`. Current original-review and latest-dev QA directories contain354 and632 tracked paths respectively; their current path/blob digests are pinned, so all original QA527 history and later additive receipts carry without alteration. The candidate overlays only19 incoming paths and preserves every other head path.

Original ZIP verified directly:63166118 bytes, SHA256 `acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84`; its2118-entry provenance carries from the prior verified receipt without decompression or regeneration. Original58 startup/navigation,35 loading,61 Task15,27 R1 and58 Task16 phases, later exact failed-node/profile/typed-adapter follow-ups, manifests and original warnings remain historical at their own heads. No report or receipt was retargeted.

Final preflight HEAD remains06cfe2f6a30236dcd8f81893ebf49bc3d0b78036 with clean tracked/index status. The selection pins the read index SHA256 and named exact-owner source hashes. Root scope selection precedes all source integration and execution.
