# Live Console qualification — chat destinations and starts

## Environment

Normal `python -m tldw_chatbook.app --focus` entry point in the managed worktree, using Python 3.12.11 and Textual 8.2.8. A dedicated tmux socket and actual OS PTY provided a 235×52 application viewport. Full text and ANSI captures are terminal evidence; no macOS screenshot was obtained because the native driver lacked process permissions.

The disposable profile had explicit config and data paths, no copied credentials or user databases, no folder bindings, and a named source workspace `console-start-verification`. Startup cloud refresh, hooks, MCP, web server, and scheduling were disabled. The genuine local `llama_cpp` endpoint at `http://127.0.0.1:9099` reported the running Gemma 4 26B A4B model. This qualification used real source-agent tool calls, real target provider calls and read-only SQLite receipt inspection.

## Results

| Case | Observed result |
| --- | --- |
| Same workspace / draft | `LIVE_WS_DRAFT`, workspace scope, v2 pending handoff, zero target messages/attempts. Source replied `draft`. |
| Casual / draft | `LIVE_CASUAL_DRAFT`, global scope with SQL NULL workspace, v2 pending handoff, zero target messages/attempts. Source replied `draft`. |
| Same workspace / start | `LIVE_WS_START`, both acceptance receipts, exact machine provenance and direct original allowance membership; one real answer `WS_START_OK`. |
| Casual / start | `LIVE_CASUAL_START`, global/NULL scope, both receipts, one real answer `CASUAL_START_OK`. Opening `/settings @everyone` persisted literally and rendered as Agent handoff, without opening settings or becoming a command. |
| Source custody | Every four-way creation preserved source active chat/workspace and `SOURCE_COMPOSER_SENTINEL`; each scope/mode combination required its own approval despite other session grants. |
| Non-streaming start | `LIVE_WS_NONSTREAM` captured streaming=false destination settings, answered `NONSTREAM_WS_OK`, completed its assistant and settled confirmed usage. A casual count target also completed with confirmed usage. |
| Draft edit / restart | Workspace edit `EDITED_WS_DRAFT_SURVIVES_RESTART` persisted pending revision2 and reopened exactly after a clean restart. Original streaming=true snapshot remained despite new streaming=false defaults. |
| Draft clear / restart | Live verification reproduced clear loss; a mounted Ctrl+U regression drove the composer-handler fix. A subsequent nonblank→clear saved empty pending revision4, and boot5 reopened empty. |
| Refused automatic start | With the private automatic-work switch disabled, `REFUSED_START` returned `not_started` / `runtime_disabled`, retained its pending draft, and created zero target requests or attempts. The toast said Not started. |
| Target Stop | `STOP_FINAL` rendered Generating with Running/Redirect/Stop visible; the ordinary Stop click saved an assistant stopped receipt and completed the attempt. Exactly one generation charge remained committed. Interrupted token usage remained uncertain under the existing conservative policy. |
| Refused start / restart | Re-enabled automatic work and restarted: `REFUSED_START` reopened with its exact draft, zero target messages/attempts and unchanged total attempt count. No automatic resend occurred. |

Streaming starts produced their requested answers but the local adapter omitted confirmed token usage. The shared allowance correctly entered `review_required` / `usage_unknown`; those runs are not evidence of successful usage settlement. `started` confirms Console acceptance rather than provider completion. Non-streaming runs supply the separate confirmed-settlement evidence.

## Defects found by live verification

1. An in-progress syntax error in the initial boot was corrected before the four-way run; that boot does not qualify behavior.
2. The mounted composer did not persist an explicit empty v2 draft at its change event. The worker reproduced and fixed it, then live verification confirmed revision4 and restart behavior.
3. An accepted native start was projected as Preparing instead of an accepted turn, hiding its Stop control. A mounted regression reproduced it; the worker corrected native-start presentation and the action-row width, and the final real Stop click passed.

App startup also reported the existing optional python-frontmatter limitation. Capture-disabled sends emitted the existing `capture_provider_failure` diagnostic while normal dispatch continued. Neither is hidden as a successful configuration check. Early post-action captures sometimes preceded asynchronous UI activation; settled captures confirmed activation. An initial short Stop case completed before navigation, so it supplies completion evidence only.

## Isolation

The real configuration SHA256 matched its pre-run value after boots2,3 and the final clean shutdown. The owned app and PTY client exited; no verification app remains running. All created conversations and charges belong to the disposable database/profile; the local model server was not reconfigured or stopped. All app/client shutdowns use ordinary Ctrl+Q on the owned socket.

## Independent review limits

These live cases qualify the observations above. They did not establish physical provider-thread drain after Stop, required project-instruction decisions before acceptance, manual Send during preparation, preparation-stage refund uncertainty, durable blocked/review row status, or recursive native-start child/wake accounting. Independent review reproduced defects in the first five and identified missing integration evidence for the last. Task 2 remains under review; focused fix evidence will be recorded below before completion. The Stop receipt above proves cancellation visibility and conservative charging, not physical capacity release timing.

## Fix-round live follow-up

Boot7 used the normal app and real local model to create `LIVE_REFUSED_FIXED` (global/NULL target `2bacf615-8da4-4b9e-a0b1-595c9f3b2a75`) with automatic work disabled only in the disposable profile. The source tool returned not_started/runtime_disabled. The browser showed not started and Active showed INPUT NEEDED under Waiting for you. The saved v2 handoff includes bounded launch mode/status/reason.

Boot8 re-enabled automatic work and restarted normally. The browser still showed not started; opening restored the exact draft, and Active again showed INPUT NEEDED. Read-only SQLite proved zero messages, checkpoints and target native attempts, and unchanged total attempt count6. No prompt was sent after restart. The History renderer still showed generic Saved chat without the launch label; the controller returned its full actual capture to the implementer. That render fix and a new mounted regression remain before final live closure.

Both owned app/PTY instances exited normally; the real configuration hash remains unchanged. Ctrl+Q while the switcher was open did not close the app, so the controller closed the modal with Escape and then used normal Ctrl+Q. This observation is separate from the launch-status qualification.

Boot9 loaded the History renderer fix through the normal application. The unchanged saved target visibly shows `Chats · Saved chat · Not started` in History. No prompt was sent; final read-only proof still has zero target messages/checkpoints/attempts and six total attempts. Closing History with Escape and then ordinary Ctrl+Q ended the owned app/PTY cleanly. Real configuration SHA256 remains unchanged. This closes the live status projection follow-up; deterministic physical-thread, preparation and recursive accounting evidence remains in the fix report.

## Round 2 manual-recovery live check

Boot10 loaded the narrow activity fix through the normal app with the same isolated non-streaming local provider. Opening saved LIVE_REFUSED_FIXED restored its exact draft, `Reply only FIXED_SHOULD_NOT_RUN.` Ordinary human Enter/Send produced a User row and the real assistant answer `FIXED_SHOULD_NOT_RUN.` The filtered Active list then showed **CURRENT**, with no Input needed or Waiting for you classification for this target. Its historical **Not started** label remained visible.

Read-only SQLite confirmed the v2 handoff is consumed, launch facts still record not_started/runtime_disabled, the assistant is complete, and the new primary run belongs to an ordinary manual root. The target has zero native-start attempts and the total remains six. See [receipt](fix2-recovery-receipt.json) and the final six frames in [terminal captures](terminal-captures.txt). This qualifies the actual manual-recovery display path; the uncertain-refund Review required variant is covered by deterministic real-controller tests.

Escape closed the activity modal; normal Ctrl+Q ended the owned app and PTY client (exit0). The dedicated app session no longer exists. The real configuration SHA256 remains `15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da`. No local-model server or user profile was changed.

## Final fix-wave approval disclosure live check

Normal boot11 used the existing isolated profile and real local model to request LIVE_FINAL_APPROVAL, a casual draft with explicit standing instructions. The actual card visibly states that Allow for this session permits later requests in the same mode and destination, including supplied opening prompts and instructions without another card. It labels the complete standing body **System prompt (explicit instructions override)**. The title, casual destination, draft mode and exact requesting run were visible.

The ordinary Allow for this session click created target `47374f59-af78-42c6-8320-1ace7e17bf0f`. Read-only SQLite shows global/NULL ownership, generic console assistant, exact system prompt `Use brief plain replies.`, pending v2 draft `Reply only LIVE_DRAFT_TEXT.`, launch draft, and zero messages/checkpoints/native attempts. Total native attempts remains six. The source stayed active. See [receipt](final-fix-approval-receipt.json) and final five [terminal frames](terminal-captures.txt). This verifies actual disclosure and draft creation; unavailable-Persona notices and archive/runtime readiness races are deterministic real-owner coverage, not separate live injections.

Normal Ctrl+Q ended app/PTY94565/client88841 with exit0; dedicated session absent. Real configuration SHA256 remains `15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da`. Final fixer received the closure signal before starting UI-heavy owner checks.
