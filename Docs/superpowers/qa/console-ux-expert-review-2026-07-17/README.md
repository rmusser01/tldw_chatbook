# Console screen — expert UX/HCI review (live-app), 2026-07-20

**Scope:** Console screen (`tldw_chatbook/UI/Screens/chat_screen.py`, `Widgets/Console/`, `Chat/console_*.py`), judged against terminal-native constraints (keyboard-first, no hover, character-cell grid, limited color depth, hard resize).

**Code under review:** detached worktree `.worktrees/console-ux-review` at `origin/dev` **cad9e271d** (2026-07-20) — includes PR #716 "console-screen feedback", the skill-picker and system-prompt modals, and the task-298 transcript-anchor work.

**Method:** six end-to-end journeys against the real served app (textual-serve + Playwright xterm-buffer driver; each journey in an isolated HOME on its own port; live llama-server at :9099 for real streaming). Standard viewport 2050×1240 plus 900×620 and 700×480 degradation passes. Every finding was then adversarially verified against a 114-item prior-art ledger (five shipped UX phase plans, four UAT evidence sets, open backlog) and against the code; only NEW and REGRESSION verdicts are reported below. Raw journey output before screening: 84 findings → **4 regressions, 57 new (8 P1, 24 P2, 25 P3)**; 18 already-known and 5 invalid were screened out (listed at the end).

**Journeys:** J1 cold start/setup · J2 returning power user (12 seeded conversations) · J3 attachments · J4 streaming · J5 settings · J6 keyboard-only + small terminals. Evidence screenshots live in this directory, prefixed `j1-`…`j6-`. (`pre-refresh-run1/` holds captures from an aborted earlier run on older code — ignore for adjudication.)

---

## Executive summary

The Console's shipped UX architecture is holding up well: the setup card, footer hint bar, Ctrl+K switcher fundamentals, draft preservation, pre-send capability gating, and the Stop affordance placement all read as deliberate and mostly work as designed. The serious problems found are concentrated in four clusters, and three of them are *timing/async integrity* problems rather than layout polish:

1. **Typed input is not sacred (P1 cluster).** Three independent mechanisms let the app mangle or misattribute what the user typed: keystrokes reorder in the composer during the post-switch settle window and the mangled text gets sent and persisted (`j2-post-switch-typing-reordered`); Enter snapshots the draft asynchronously, so text typed just after send is folded into the sent message (`j6-send-captures-late-keystrokes`); and the edit modal opens late, leaking waiting keystrokes into the draft (`j6-edit-modal-late-open-keystroke-leak`). For a keyboard-first TUI this is the most corrosive class of defect found: it teaches fast typists not to trust the composer.

2. **Runs without status (P1 cluster).** Regenerate gives literally zero feedback for 75+ seconds (`j6-regenerate-zero-feedback` — code-confirmed: the transcript sync timer only starts on the send path); all status surfaces say "Ready / No active work" while a generation streams (`j4-status-surfaces-say-ready-during-run`); Stop acknowledges nothing and in one of two trials froze the UI in `[streaming]` while generation continued in the background and persisted content diverged from the display (`j4-stop-feedback-unreliable`, suspected regression against the task-227 fix).

3. **Silent persistence lies (P1 cluster).** Three flows accept user intent, confirm it visually, then drop it: tab rename looks like a conversation rename (the transcript header updates!) but evaporates on restart (`j2-rename-tab-only-silently-lost`); "Save as default" writes temperature to a config section the boot path never reads back (`j5-save-as-default-temperature-lost` — the code's own docstring claims otherwise); and a stopped reply persists as an unmarked mid-sentence fragment (`j4-interrupted-reply-unmarked-no-retry`).

4. **Keyboard trust gaps.** Focus is imperceptible (~1.1:1 contrast deltas) across chips, rail rows and message action buttons — measured per-cell, and it caused a real accidental "Save as…" activation during the review (`j6-chip-focus-imperceptible-accidental-activation`); the switcher's arrow keys don't navigate (`j6-switcher-arrows-dead`); F6's rail stop is invisible (`j6-f6-rail-stop-invisible`); and keyboard users cannot scroll the transcript at all (`j4-keyboard-cannot-reach-transcript-scroll`). Where the accent-border focus pattern IS applied (composer, transcript pane), it's excellent — the gap is coverage, not vocabulary.

Also headline-worthy: at 97×30 cells the composer is clipped out entirely with no warning while typing goes nowhere (`j6-small-terminal-composer-clipped`); the rail restructures itself seconds after a switch/send, moving click targets and hiding the chat list (`j2-rail-unpredictable-mode-swap`); wheel scrollback is locked during streaming (`j4-wheel-scroll-locked-during-stream`, suspected regression introduced by the task-298 anchor work); and the agent-runtime send path now formats provider failures as raw `Agent run failed: … 500 … MDN link`, discarding the server's actionable hint and silently poisoning the conversation because the failing image is re-sent forever (`j3-provider-error-discards-detail-poisons-conversation`, regression against the classified-error copy that demonstrably shipped 2026-07-11).

**Recurring motifs worth fixing once, centrally:** boot-time model-catalog toast occludes the composer's Send/Attach/Save and swallows clicks (observed independently in J3/J4/J6); raw internals leak into primary surfaces (conversation UUID as rail "Scope", raw markdown `**`/`###` in the transcript, five spellings of the same provider name); and several affordances are invisible or unrecognizable (star toggle renders as `.`/`*`, "System: none" is a clickable door with zero affordance, Attach morphs into an unlabeled icon pair).

### Suggested fix order

1. Input integrity: synchronous draft snapshot on Enter; block or buffer input during post-switch recompose (J2/J6 P1s).
2. Run-status truthfulness: start the sync timer on all generation paths (regenerate/retry/continue); make chips/status reflect active runs; acknowledge Stop and mark interrupted replies.
3. Persistence honesty: tab-rename scope, save-as-default round-trip, provider-error surfacing with recovery affordance.
4. Focus visibility sweep: apply the existing accent pattern to chips/rail/switcher/message actions; then the small-terminal minimum-size guard.

---
## Regressions (shipped behavior now broken)

#### j3-provider-error-discards-detail-poisons-conversation — API failure shows raw httpx '500' + MDN link, discards the server's actionable message, and silently poisons the whole conversation
*J3 Attachments · P1 · verdict REGRESSION (medium confidence) · heuristic: error recovery / help users diagnose*

**Observed:** Sending an image to llama.cpp produced transcript rows: 'Assistant [failed]' + 'Agent run failed: Server error 500 Internal Server Error for url http://127.0.0.1:9099/v1/chat/completions / For more information check: https://developer.mozilla.org/...Status/500.' The server body actually said 'image input is not supported - hint: you may need to provide the mmproj' (verified by curl) but was thrown away. Worse: a later plain-text message ('line one') in the same conversation failed identically because history re-sends the image forever; nothing tells the user the image is the cause and there is no affordance to drop it from history. On reload the same errors re-render under the role 'Tool' instead of 'System'.

**Expected:** Surface the provider's error body, translate it ('this model can't accept images'), point at the offending attachment, and offer recovery (retry without images / remove from context). Never emit MDN links in a chat transcript.

**Repro:** Mark local-gemma vision-capable in config, attach a PNG, send. Observe the generic 500 + MDN link. Then send any plain text message in the same conversation - it fails with the same 500.

**Evidence:** `j3-63-response-final.png`, `j3-67-shift-enter.png`, `j3-70-reopened-conv.png`

**Verifier note:** Ledger says provider failures render as transcript-only system rows with CLASSIFIED copy (demonstrably worked 2026-07-11). The now-default agent-runtime path (agent_runtime_enabled=True, console_chat_controller.py:278) formats failures at line 2326 as 'Agent run failed: {raw reason}', bypassing describe_stream_failure (line 172) — j3-63 confirms raw httpx text + MDN link, no classification prefix; and the rows come back on reload rendered as 'Tool' (j3-77/79), contradicting 'transcript-only'. The response BODY was discarded even under the old classified copy, and the pointing-at-offending-attachment / drop-from-history recovery never shipped — those sub-asks are new, but the headline copy/persistence behavior contradicts shipped remediation behavior. P1 stands: default path, and the re-sent image poisons every later send with an undiagnosable 500.

#### j4-wheel-scroll-locked-during-stream — Mouse-wheel scrollback over the transcript is inert while a reply is streaming (works when idle)
*J4 Streaming · P1 · verdict REGRESSION (medium confidence) · heuristic: user control & freedom / scroll autonomy vs auto-follow*

**Observed:** With the same overflowing conversation loaded: wheel-up over the transcript scrolls normally when the app is idle (verified: view moved, j4-30a), but during active streaming the identical wheel gesture produced no view change at all - the viewport stayed pinned to the bottom (j4-34; buffer compare 400ms after wheel showed the top region unchanged). Round 1 corroborates: two wheel-up attempts mid-run left the buffer byte-identical (j4-06/j4-07 are identical files). The user cannot read history with the primary terminal scroll gesture while a long reply streams; whether the event is swallowed or instantly yanked back, the observable outcome is a locked viewport.

**Expected:** Wheel-up during streaming should detach auto-follow and scroll back (standard terminal behavior), with auto-follow re-engaging only when the user returns to the bottom.

**Repro:** Load a conversation whose transcript overflows -> confirm wheel-up scrolls while idle -> send a long prompt -> once [streaming] text is growing, hover the transcript and wheel-up -> view stays pinned to bottom.

**Evidence:** `j4-34-midstream-wheel.png`, `j4-30a-idle-wheel.png`, `j4-06-midstream.png`, `j4-07-scrolled-up.png`

**Verifier note:** Contradicts the shipped anchor contract 'released on scroll-up (never yanked)'. Before task-298 there was no auto-follow at all, so wheel scrollback during streams trivially worked — the lock is introduced by the anchor work. task-298's own notes concede AC#2 was pinned only by harness tests because 'xterm can't drive scroll reliably', i.e. the wheel path was never live-verified. Textual 8.2.7 code says pointer scroll-up SHOULD release the anchor (_scroll_up_for_pointer defaults release_anchor=True, unlike _scroll_down_for_pointer), so the live lock's mechanism is unexplained (possible re-anchor via a mid-stream code path or event-ordering under tick load) — but two independent rounds observed it and idle wheel works in the same harness, ruling out a pure tool artifact. Keeping P1: primary terminal scroll gesture dead for minutes-long streams; the click+PageUp workaround is undiscoverable.

#### j4-stop-feedback-unreliable — Stop gives no acknowledgment and behaved inconsistently: one run froze the UI in '[streaming]' state while generation continued in the background
*J4 Streaming · P1 · verdict REGRESSION (medium confidence) · heuristic: visibility of system status / error prevention & recovery*

**Observed:** Two Stop interruptions, two different behaviors, zero feedback in both. Run A (j4-23/24): clicking Stop froze the transcript mid-word at 'A [streaming]'; 3s later the message still carried [streaming] and the Stop button was still active - the UI looked stuck-running with no confirmation; after an app restart the persisted message contained two additional paragraphs that were never displayed, i.e. generation continued after the UI froze and the user's view diverged from what was saved. Run B (j4-36): Stop cleared [streaming] and reverted the button within ~2.5s - but again with no 'stopped' toast, no event line in the 'Transcript / Event Stream', and no state change anywhere else.

**Expected:** Stop should (1) visibly acknowledge immediately (button state change + 'Stopping...'), (2) reliably cancel the provider request, and (3) leave an explicit 'stopped by user' record in the transcript/event stream. What is persisted must match what was shown.

**Repro:** Send a long prompt -> when tokens start streaming, click Stop in the composer bar -> observe no acknowledgment; in one of two trials the message stayed '[streaming]' with the Stop button active, and reloading the conversation later showed extra content generated after the freeze.

**Evidence:** `j4-23-just-after-stop.png`, `j4-24-post-stop-settled.png`, `j4-28-click-partial-message.png`, `j4-36-just-after-stop.png`

**Verifier note:** Run A (frozen '[streaming]', Stop still active, generation continued in background, persisted content diverged from display) is the exact race family task-227 (Done 2026-07-16, in this worktree) claims fixed: 'a stop that lands during an in-flight agent bridge thread always persists the run as cancelled'. Either an unfixed window remains or a later change regressed it — one occurrence in two trials, races are nondeterministic, hence medium confidence. The zero-acknowledgment half is real and previously unexercised (ledger gap-not-exercised-2026-07 lists Stop mid-stream as never verified): _stop_console_generation_from_visible_action (chat_screen.py:9785-9793) emits no success toast/event row, only a warning when nothing is running. P1 stands for the freeze+divergent-persistence behavior.

#### j5-streaming-state-invisible — Streaming state (and its session override) is invisible on the default Console surface
*J5 Settings · P3 · verdict REGRESSION (medium confidence) · heuristic: visibility of system status*

**Observed:** The rail Model section lists Provider/Model/Temperature/Max tokens/System but not Streaming; the chips don't show it either. After toggling Streaming to Off and saving (verified applied: Inspector shows 'Streaming: off', 'Sampling: T 0.70'), nothing on the default screen reflects the change — it's only visible inside the modal or the collapsed-by-default Inspector.

**Expected:** If the rail summarizes the session's model settings, it should include any setting the user can override per-session (or at least surface non-default state); otherwise an override is undetectable at a glance.

**Repro:** 1. Rail > Configure; click the Streaming 'On' button (becomes 'Off'); Save. 2. Scan rail and chips: no streaming mention. 3. Expand Inspector (right edge): 'Streaming: off'.

**Evidence:** `j5-69-rail-after-streaming-save.png`, `j5-70-inspector-streaming.png`

**Verifier note:** True regression against the shipped-behavior ledger item model-section-compact, introduced AFTER the ledger by commit 0c26a8408 'feat(console): split model settings into labeled rows' (2026-07-18, on the PR #716 feature/console-screen-feedback branch): the old line2 built by build_console_model_section_lines joined (sampling, context, transport) where transport was 'Streaming: on/off'; the replacement labeled rows (chat_screen.py:7176-7252) carry only Provider/Model/Temperature/Max tokens — the streaming (and token-budget) readout was silently dropped from the default surface. Confidence medium only on intent: the rail redesign presumably passed the console-approval-gate, so the drop may have been implicitly accepted, but no settled-decision item supersedes the streaming-visible contract. P3 per reviewer: visibility polish, value still shown in modal/Inspector.

## New findings — P1 (major friction / likely user failure)

#### j2-post-switch-typing-reordered — Keystrokes typed shortly after a session switch get reordered in the composer and the mangled text is sent to the model
*J2 Returning power user · P1 · verdict NEW (high confidence) · heuristic: Error prevention / feedback latency*

**Observed:** Reproduced twice. Session 1: after selecting a conversation in the Ctrl+K switcher, clicking the composer and typing 'Follow-up: which vector store...' (15ms/char) produced 'Follow-ch vector store should I pick for a 10k doc corpus? up: whi' — the caret was silently relocated mid-typing — and that mangled string was then sent and persisted as the user message (visible in transcript and after app restart). Controlled retrial: typing 'abcdefghij...' at 15ms/char starting ~0.5s after switcher Enter yielded 'abcdefghijklmnrstuvwxyz0123456789▌opq' (chars opq torn out of sequence). At 100ms/char starting later the text stayed intact, so the hazard window is the post-switch settle period (~0.5-2s here; machine load average was 13-15, which widens it, but the caret-steal is app behavior).

**Expected:** The composer must never reorder buffered keystrokes; recompose/focus work after a switch should preserve caret position and pending input, or input should be blocked with a visible busy state until the switch settles.

**Repro:** Ctrl+K, type a query, Enter to switch. Within ~0.5s click the composer and type a fast burst (15ms/char, i.e. fast typist / key-repeat speed). Composer shows characters out of order; Enter sends the mangled text.

**Evidence:** `j2-57-mangle-trialB.png`, `j2-10-just-sent.png`, `j2-12-reply-later.png`, `j2-24-resumed-long.png`, `j2-56-mangle-trialA.png`

**Verifier note:** Genuine input-integrity defect, nowhere in ledger or backlog. Mangled text is visible persisted in the transcript ('Follow-ch vector store should I pick for a 10k doc corpus?up: whi' in j2-12-reply-later.png), reproduced twice with a controlled retrial. Post-switch activation runs multiple deferred syncs plus draft restore/forced composer focus (_focus_console_composer_if_needed(force=True) after _sync_native_console_chat_ui in chat_screen.py:3342-3344), which can relocate the caret under buffered typing. Load average widened the window but caret-steal is app behavior.

#### j2-rename-tab-only-silently-lost — Renaming a resumed conversation via the tab rename modal looks successful (tab + transcript header update) but is silently lost — the saved conversation keeps its old title
*J2 Returning power user · P1 · verdict NEW (high confidence) · heuristic: Consistency with user expectations / error prevention (silent data loss)*

**Observed:** Resumed 'Websocket reconnect strategy', clicked its active tab to open 'Rename Chat Tab', typed 'Reconnect deep dive', pressed Enter. Tab strip and the transcript header — the same header that identifies the conversation by title ('Transcript / Event Stream | Reconnect deep dive') — updated immediately, signalling the conversation was renamed. On next app start the rail lists 'Websocket reconne...' again; 'Reconnect deep' appears nowhere. The rename applied only to the ephemeral tab label, and tabs do not survive restarts, so the user's rename evaporates without any warning or scope hint beyond the modal title.

**Expected:** Renaming the tab of a saved conversation should rename (or offer to rename) the conversation, or the modal must state that the change is tab-only and temporary.

**Repro:** Open a saved conversation from the rail, click its (active) tab label once or twice until 'Rename Chat Tab' opens, type a new name, Enter. Observe tab + header update. Restart the app: rail still shows the old title.

**Evidence:** `j2-51-rename-modal.png`, `j2-52-rename-typed.png`, `j2-53-after-rename-enter.png`, `j2-55-final-boot-rail.png`

**Verifier note:** Code-verified: ConsoleChatStore.rename_session (console_chat_store.py:253) only mutates the in-memory session title; _apply_rename (chat_screen.py:1080-1100) never writes the persisted conversation title, yet the transcript header (set_session_title from session.title) confirms the rename as if conversation-level. Modal is labeled 'Rename Chat Tab' but the tab IS the conversation under resume-opens-new-tab semantics, and the rename is silently discarded on restart. No ledger item or backlog task covers persisting renames; silent loss of confirmed user input justifies P1.

#### j2-rail-unpredictable-mode-swap — The rail's entire section stack swaps between 'conversation browser' (Starred/Workspaces/Chats) and 'session context' (Context/Model/Agent/Details) on its own, seconds after switches/sends, moving click targets and hiding the chat list
*J2 Returning power user · P1 · verdict NEW (medium confidence) · heuristic: Consistency / predictability; user control*

**Observed:** Buffer dumps show the rail as the browse list at 13:19:30 (j2-10) and as the context stack at 13:19:42 (j2-12) with zero rail interaction in between; the same swap followed the earlier rail-click switch, and end states differ between similar action sequences (after one switch the list stayed, after another it was replaced). The swap displaced a click in practice: a click aimed at a listed conversation row landed on the Context section because the stack flipped between reading and clicking (my scripted C5 click failed exactly the way a human's would, and instead a stray click starred a different row unnoticed). While in context mode the conversation list is simply gone from the rail with no visible way back other than waiting or Ctrl+K.

**Expected:** The rail should be a stable landmark: keep the conversation list persistently reachable (fixed sections or an explicit user-controlled toggle), and never restructure itself asynchronously after the user has stopped acting.

**Repro:** Switch to a saved conversation (rail click or Ctrl+K) and/or send a message; watch the rail for the next ~15s without touching it: Starred/Workspaces/Chats are replaced by Context/Model/Agent/Details (or vice versa) at an unpredictable moment.

**Evidence:** `j2-10-just-sent.png`, `j2-12-reply-later.png`, `j2-06-switcher-open.png`, `j2-44-dot-click.png`, `j2-57-mangle-trialB.png`

**Verifier note:** Phenomenon confirmed by screenshots: j2-10 shows the Session section with the full conversation browser (search row + Starred/Workspaces/Chats); j2-12, 12s later mid-run with zero rail interaction, shows the browser gone (summary-only legacy render, Context/Model/Agent/Details now at the click positions the list occupied). Every sync path routes through _build_console_workspace_context_state which always attaches conversation_browser (chat_screen.py:4702-4738), so this async restructure is unintended, not a designed mode. Not covered by tick-ttl-2s-gating (that blesses content lag, not structural swap) nor task-149 (scroll survival). Displaced a click in practice (accidental star). Mechanism not fully pinned, hence medium confidence on root cause, high on the defect itself.

#### j5-save-as-default-temperature-lost — 'Save as default' writes temperature to a config location the app never reads back — saved default silently reverts on restart
*J5 Settings · P1 · verdict NEW (high confidence) · heuristic: error prevention; persistence expectations; visibility of system status*

**Observed:** With temperature set to 0.88, clicking 'Save as default' closed the modal with no confirmation toast. Config diff of the isolated home: [chat_defaults] got provider="llama_cpp" and streaming=true, while temperature=0.88, model and top_k were written to [api_settings.llama_cpp]; [chat_defaults].temperature stayed 0.6. A fresh app process shows rail 'Temperature 0.60' — the 0.88 the user saved 'as default' never comes back. The modal's help text ('Save as default also writes provider + streaming defaults to config') technically warns, but the button still accepts and writes the temperature edit somewhere inert.

**Expected:** Save as default should either round-trip all shown values, or the dialog should state that sampling values are session-only and not write them to config at all. Silent acceptance followed by silent reversion after restart is the worst combination.

**Repro:** 1. Rail > Configure; set Temperature (e.g. 0.88). 2. Click 'Save as default'. 3. Inspect ~/.config/tldw_cli/config.toml: temperature written under [api_settings.llama_cpp], [chat_defaults].temperature unchanged (0.6). 4. Restart app -> rail shows Temperature 0.60.

**Evidence:** `j5-73-after-save-as-default.png`, `j5-74-rail-after-restart.png`

**Verifier note:** Confirmed real write/read priority bug, not in any ledger item or backlog task. Save-as-default writes temperature (and other PROVIDER_DEFAULT_PERSIST_FIELDS) to [api_settings.<provider>] (console_settings_modal.py:712-746; the docstring at line 718 explicitly claims this is 'the source build_default_console_session_settings reads on the next boot'), but the read path (console_session_settings.py:391-397, _float_setting_from_sources first-hit-wins) checks chat_defaults BEFORE provider settings, and the default config template ships chat_defaults.temperature=0.6 (config.py:2736) — so the saved value is permanently shadowed and silently reverts on restart, exactly as the reviewer's config diff showed. Also no success/failure toast on save. P1 upheld: explicit user save intent silently lost.

#### j6-regenerate-zero-feedback — Regenerate (r / ♻) runs with zero feedback: status stays 'Ready', no placeholder, no Stop for over 75 seconds
*J6 Keyboard-only + small terminal · P1 · verdict NEW (high confidence) · heuristic: Visibility of system status*

**Observed:** With the newest assistant reply selected, pressing r produced no visible change for 75+ seconds of polling: header status stayed 'Ready', the old reply stayed untouched, no spinner/placeholder appeared, and the composer showed Send/Attach/Save (no Stop button — confirmed in j6-a23-after-r.png). The regenerated reply eventually replaced the message minutes later (new content + version '<' '>' chevrons in j6-a26-mystery-state.png). By contrast, the normal send path immediately appends the user message, streams the reply and shows Stop. During the silent window my extra keyboard presses activated unrelated late-arriving UI.

**Expected:** Regeneration shows the same in-flight affordances as send: an immediate placeholder or 'regenerating…' marker on the message, a Stop control, and a busy status.

**Repro:** Open a conversation tab, select the last assistant message (k), press r. Watch header row, composer row and message area: nothing changes for over a minute against a slow local provider; the reply is later swapped wholesale.

**Evidence:** `j6-a23-select-assistant.png`, `j6-a23-after-r.png`, `j6-a24-regen-watch.png`, `j6-a26-mystery-state.png`

**Verifier note:** Code-confirmed root cause: the 0.2s transcript sync timer is started ONLY in _submit_console_native_draft (chat_screen.py:8732, sole call site); _regenerate_console_message/_retry_console_message/_continue_console_message (10789-10820) never start it, so although ConsoleChatController.regenerate_message sets VALIDATING then STREAMING and streams into begin_variant_stream, no UI sync runs until completion — status chip stays Ready, Stop stays display:none, no placeholder. Present since the timer's introduction in PR #359 (git -S), so NEW not REGRESSION. Distinct from task-193 (send-path placeholder) and task-232 (mid-run gating). P1 stands: user cannot stop or even see a multi-minute run.

#### j6-send-captures-late-keystrokes — Send captures the draft asynchronously: text typed after the send keypress is appended into the sent message
*J6 Keyboard-only + small terminal · P1 · verdict NEW (high confidence) · heuristic: Error prevention / feedback latency*

**Observed:** In a fresh Ctrl+T tab, with draft 'line one', a Shift+Enter (delivered by the web terminal as plain Enter = send) was followed ~0.4s later by typing 'line two'. The message that got sent was 'line oneline two' — the post-Enter keystrokes were folded into the sent message. Send side-effects (tab rename to '● line oneline', transcript user row) surfaced only ~3s after the keypress with no interim feedback; Backspace presses during the window did not visibly edit the draft.

**Expected:** Enter snapshots and clears the draft synchronously at the moment of the keypress; anything typed afterwards belongs to the next draft. First-send-in-new-tab latency should show immediate pending feedback.

**Repro:** Ctrl+T for a new tab. Type 'line one'. Press Enter (or Shift+Enter through xterm.js) and immediately type 'line two'. Wait: the sent user message reads 'line oneline two' and the tab is renamed accordingly.

**Evidence:** `j6-a31-shift-enter.png`, `j6-a32-post-shift-enter.png`

**Verifier note:** Code-confirmed: Enter in the composer calls Button.press() on #console-send-message (chat_screen.py:11201-11209); the draft is only read via composer.draft_text() when the bubbled Button.Pressed handler finally runs (_send_console_message_from_visible_action:8866-8871), while printable keys processed meanwhile mutate the draft via on_key→insert_text — so post-Enter typing is folded into the sent message. The settled clear-at-submission-accept decision (decision-failed-sends-system-rows) governs when the composer CLEARS, not when the draft is CAPTURED; capture-at-keypress was never settled. Genuine message-integrity defect, P1 stands.

#### j6-chip-focus-imperceptible-accidental-activation — Chip/button focus styling is imperceptible across the screen; Tab focus moves are invisible and caused an accidental 'Save as...' activation
*J6 Keyboard-only + small terminal · P1 · verdict NEW (medium confidence) · heuristic: Focus visibility / error prevention*

**Observed:** Measured focus styles: rail buttons ('New conversation', section ▾ toggles, conversation rows) bg #1e1e1e -> #272727 on a #242f38 panel; 'Switch' chip #1e1e1e -> #1e262d; palette selected row #141f27 -> #1e1e1e — all ~1.1:1 contrast. Several keyboard stops changed literally zero cells: Enter-to-show-actions on a selected message (focus onto Copy button, no visible change), and 8 consecutive Tab presses from the composer (no visible change at all). Consequence observed live: after Enter on a selected message I pressed Tab 3x + Enter expecting the ♻ button and instead activated the invisible-focused 'Save as...' button, opening an unintended modal.

**Expected:** Focused controls get an unmistakable indicator (inverse video, accent bg, or bracket/underline marker) meeting ~3:1; every Tab press produces a visible focus change so Enter's target is always predictable.

**Repro:** 1) F6 to rail, Tab through items and compare cell attrs (bg deltas ~#090909). 2) Select a message, press Enter (no visible change), Tab 3x (no visible change), Enter — the 'Save as...' modal opens unexpectedly. 3) Focus composer, press Tab 8x — zero visible change anywhere.

**Evidence:** `j6-a08-rail-tab3.png`, `j6-a08-rail-tab10.png`, `j6-a25-enter-actions.png`, `j6-a25-tab3-recycle.png`, `j6-a26-mystery-state.png`, `j6-a31-tab-from-composer.png`

**Verifier note:** Partially code-confirmed: many Console controls (.console-rail-collapse-button, .console-switcher-result, handle buttons) have NO :focus rule → Textual default subtle bg shift; the DS focus vocabulary itself is quiet ($ds-focus-bg = $ds-surface-raised = $surface, _variables.tcss:18). Caveat: several controls DO declare bold-underline focus rules (.console-control-chip:focus, .console-transcript-action-button:focus, .console-rail-section-toggle:focus at _agentic_terminal.tcss:1961/2361/2856), which the journey's cell scans say did not render — that discrepancy is unresolved, hence medium confidence. rail-layout-quiet-focus (task-149) settled only the rail-body boundary, not control focus contrast; no open backlog task covers it. The observed accidental 'Save as…' activation makes P1 defensible; keep P1 as the umbrella focus-visibility finding.

#### j6-small-terminal-composer-clipped — At 97x30 cells the composer is clipped out entirely - no input box, no warning, typing is invisible
*J6 Keyboard-only + small terminal · P1 · verdict NEW (high confidence) · heuristic: Degradation under resize*

**Observed:** Cold start at 700x480 px (97x30 cells): rail, transcript and Inspector chip render, but the composer row does not exist anywhere on screen (workbench boxes are clipped open-ended at the bottom); typing produced no visible echo. No 'terminal too small' notice is shown. At 125x38 (900x620) the composer is present and usable. Footer at small widths truncates the key hints ('Ctrl+K switch se', 'F1 help | E') while memory stats (P:/C\/N:/M:) keep their full space.

**Expected:** Below the minimum viable height, either the layout drops lower-priority panes (rail/inspector) to preserve transcript+composer, or an explicit 'terminal too small' overlay appears. Key hints should win over debug stats in footer truncation.

**Repro:** Serve the app and open it in a 700x480 viewport (97x30 cells). Observe no composer row and no warning; type text and observe nothing change. Compare with 900x620 where the composer renders at row 34.

**Evidence:** `j6-b05-cold-700x480.png`, `j6-b06-700x480-typing.png`, `j6-b01-900x620.png`

**Verifier note:** Evidence j6-b05-cold-700x480.png confirms: no composer row exists at 97x30, boxes clip open-ended, no too-small warning, and footer truncates key hints while P:/C\/N:/M: memory stats keep full width. chat_screen.py has no on_resize/breakpoint handling and no minimum-size overlay. No ledger or backlog coverage (task-226/task-108 are Personas-specific). 97x30 is larger than the 80x24 default terminal, so the core loop is silently broken at common real sizes — P1 stands.

## New findings — P2 (clear friction with workaround)

#### j1-test-provider-contradicts-itself — Test Provider reports 'ready' and 'status=blocked' simultaneously and tests the saved URL, not the one on screen
*J1 New-user cold start · P2 · verdict NEW (high confidence) · heuristic: visibility of system status / error prevention & recovery*

**Observed:** With a deliberately dead draft endpoint (http://127.0.0.1:9098) typed into the Endpoint field, Test Provider returned: 'Provider test | llama_cpp is ready. No API key is required. | model=missing | LLAMA_CPP_API_KEY=<redacted> | Endpoint: api_settings.llama_cpp.api_url=http://localhost:8080/completion | status=blocked'. It (a) says ready AND blocked, (b) cites the SAVED config URL (localhost:8080/completion) while the readiness panel right below shows the draft URL 127.0.0.1:9098, and (c) never flags that the typed URL is unreachable. The stale 'blocked' message also persisted on screen after settings were later saved successfully (j1-34) until Test was manually re-run.

**Expected:** One unambiguous verdict about the endpoint currently shown in the form ('Could not reach http://127.0.0.1:9098 — connection refused'), and stale results cleared or marked outdated when inputs change or are saved.

**Repro:** Settings > Providers & Models -> Provider=llama.cpp -> Endpoint=http://127.0.0.1:9098 (unsaved) -> click Test Provider -> read result line rows 38-39.

**Evidence:** `j1-13-bad-url-typed.png`, `j1-14-test-bad-url-immediate.png`, `j1-16-test-bad-url-full.png`, `j1-34-after-save-category.png`, `j1-35-test-after-save.png`

**Verifier note:** Code-confirmed. _provider_readiness_test_report (settings_screen.py:5257-5311) concatenates readiness.user_message ('llama_cpp is ready…', computed from SAVED app_config) with status=blocked when the model field is empty — the contradictory line is structural. The endpoint line calls _provider_endpoint_summary(provider) with no draft value (5292 → 4785), so it prints the saved api_settings URL while the form shows a different draft. The task-191 live probe (_provider_live_probe_base_url, 5318+) runs only after a PASSING readiness test, so an unreachable draft URL is never flagged when the model is missing. Not covered by provider-test-outcome-toast or task-191 (both shipped the surface, neither covers these defects). Downgraded P1→P2: the toast summary itself renders an unambiguous verdict ('Provider test failed: … Also set a default model.'); the detail row and draft-vs-saved mismatch mislead but do not hard-block.

#### j1-endpoint-v1-guessing — Endpoint URL shape is a guessing game: app's own llama.cpp default fails its own discovery, and a scheme typo is misdiagnosed as a /v1 problem
*J1 New-user cold start · P2 · verdict NEW (high confidence) · heuristic: error prevention / consistency of guidance*

**Observed:** Three contradictory URL shapes in one flow: (1) selecting llama.cpp prefills 'http://localhost:8080/completion'; (2) Discover models with a working server at bare-origin http://127.0.0.1:9099 was refused with 'This endpoint is not OpenAI-compatible for v1 discovery. Configure a /v1 endpoint to discover models.' — meaning the app's own prefilled '/completion' default would also fail discovery; (3) with a realistic typo 'ttp://127.0.0.1:9099/v1' (dropped h), the app returned the SAME '/v1' message even though the URL already ends in /v1 — the real problem (invalid scheme) is never surfaced, and no inline validation flags the malformed URL. Only after typing the exact 'http://.../v1' form did discovery succeed (~0.8s, j1-24).

**Expected:** Prefill a default that works with the app's own features; auto-append or auto-probe /v1 instead of demanding the user reformat; distinguish 'malformed URL' from 'missing /v1 path'; validate the URL inline on blur.

**Repro:** Provider=llama.cpp (note /completion default) -> Endpoint=http://127.0.0.1:9099 -> Discover models -> '/v1' refusal; set Endpoint='ttp://127.0.0.1:9099/v1' -> Discover -> same '/v1' refusal; fix to http://127.0.0.1:9099/v1 -> succeeds.

**Evidence:** `j1-12-after-provider-switch-settle.png`, `j1-21-discover-good-result.png`, `j1-23-endpoint-check.png`, `j1-24-discover-success.png`

**Verifier note:** Code-confirmed on a post-ledger surface (Settings discovery shipped 1fd5f5f0c, 2026-07-11, tasks 188/191-console). _discovery_status_from_error flattens every unsupported_endpoint kind into the single '/v1' copy (settings_screen.py:4958 → MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY:218), discarding the client's distinct messages; supports_openai_compatible_model_discovery (openai_compatible_model_discovery.py:224-249) returns False both for a malformed scheme (parse→None, e.g. 'ttp://…/v1') and for llama.cpp's native /completion path — so the app's own prefilled default fails its own discovery and a scheme typo is misdiagnosed as a /v1-path problem. No inline URL validation on the field. Nothing in the ledger covers discovery UX. P2: first-run guidance actively contradicts itself, but manual model entry and the Console-modal discover path remain.

#### j1-discovered-model-selection-mismatch — Discovered-model checkbox looks checked but 'Save selected' says nothing is selected; toggling produces no visible change
*J1 New-user cold start · P2 · verdict NEW (medium confidence) · heuristic: visibility of state / affordance clarity*

**Observed:** After discovery, the single result renders as '▐X▌ gemma-4-26B-...gguf' — visually checked. Clicking 'Save selected' immediately returned 'Select discovered models to save.' Clicking the model row itself changed NOTHING visually (cell attrs identical, still ▐X▌; the ▐ ▌ cap glyphs even render fg==bg, i.e., invisible), yet after that click 'Save selected' succeeded ('Saved 1 discovered model(s) to Llama_cpp.'). The visual state and the selection model are disconnected; I only recovered by trial and error.

**Expected:** Checked visual == selected state; toggling must produce a visible change; if nothing is selected, the row should render visibly unchecked.

**Repro:** Discover models against http://127.0.0.1:9099/v1 -> result row shows ▐X▌ -> click Save selected -> 'Select discovered models to save.' -> click the model row (no visual change) -> click Save selected -> saves.

**Evidence:** `j1-28-model-list-clean.png`, `j1-29-model-row-after-click.png`, `j1-25-save-selected.png`, `j1-30-save-selected-retry.png`

**Verifier note:** Consistent with code: the SelectionList options are built unselected on first discovery (settings_screen.py:4923-4942, selected only if in the initially-empty _model_discovery_selected_model_ids), so the row the reviewer saw as '▐X▌…' was in fact unselected — the theme leaves Textual ToggleButton component classes (toggle--button) unstyled/mis-contrasted so the X reads as checked in both states and toggling produces no visible change (reviewer measured identical cell attrs; caps render fg==bg). Save-selected honesty path confirmed at 5114-5127. Brand-new surface (2026-07-11 discovery commit), no prior art in ledger or backlog. Downgraded P1→P2: the warning toast ('Select discovered models before saving.') guides recovery within one click, but the control's state display genuinely lies — worth a prompt fix.

#### j1-model-field-pure-recall — Active Model must be re-typed from memory: free-text field, no pick-from-discovered, and the discovered list vanishes after saving
*J1 New-user cold start · P2 · verdict NEW (high confidence) · heuristic: recognition over recall*

**Observed:** Saving the discovered model only appends it to a provider list; the 'Model' field stays an empty free-text input ('Model name' placeholder) and readiness stays 'llama.cpp / not selected'. The discovery list disappears from screen right after 'Save selected', and typing 'gemma' into Model offers no autocomplete or dropdown (j1-32). To reach readiness I had to type the full 56-character 'gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf' exactly, from memory of a string no longer visible anywhere.

**Expected:** After discovery/save, the Model field should offer the discovered/saved models for selection (dropdown or typeahead), or Save selected should offer to set the active model directly.

**Repro:** Discover + Save selected -> observe list disappears, Model still placeholder -> focus Model, type 'gemma' -> no suggestions -> readiness only passes after typing the full gguf filename.

**Evidence:** `j1-30-save-selected-retry.png`, `j1-31-model-field-focused.png`, `j1-32-model-typeahead.png`, `j1-33-category-saved.png`

**Verifier note:** Code-confirmed: the Settings Model field is a bare Input with placeholder 'Model name' (settings_screen.py:6118-6124), no suggester, no Select, and Save selected only appends model ids to the provider list (_save_selected_discovered_provider_models) — nothing offers them for activation, and the discovery list state resets. Not covered by ledger: provider-catalog-display-names covers the provider dropdown, settings-modal-model-prefill covers the Console modal, task-188 added a discovered-models Select to the CONSOLE settings modal and one-click detected-server on the card — but the Settings screen, which is exactly where the setup card routes first-run users, still demands full recall of a 56-char gguf filename. P2: real recognition-over-recall failure on the primary onboarding path; partial mitigations exist on other surfaces.

#### j2-rail-hides-oldest-conversation — Rail Chats list silently caps at 11 saved conversations: the 12th ('Refactor auth middleware plan') never appears, with no 'show more', count, or truncation hint
*J2 Returning power user · P2 · verdict NEW (high confidence) · heuristic: Recognition over recall / visibility of system status*

**Observed:** With 12 seeded conversations, the rail's Chats section listed 11 (Websocket...Debug flaky) in every state, including after wheel-scrolling the rail to its end (blank rows below the last item). 'Refactor auth middleware plan' (the oldest) was absent across all sessions. It IS reachable via Ctrl+K fuzzy search ('refac' finds it), but nothing in the rail discloses that items are hidden, so a user scanning the rail would reasonably conclude the conversation was deleted. There is also no search/filter affordance inside the rail itself.

**Expected:** Either list all conversations (scrollable), or show an explicit 'N more — search with Ctrl+K / Show all' disclosure row at the section end.

**Repro:** Seed 12 conversations (mk_home variant seeded). Open Console, expand/scroll the rail Chats section to its bottom: only 11 saved rows exist, oldest missing, no indicator. Ctrl+K + 'refac' finds the hidden one.

**Evidence:** `j2-04-rail-scrolled.png`, `j2-20-boot2-rail.png`, `j2-36-switcher-refac.png`, `j2-55-final-boot-rail.png`

**Verifier note:** Code-verified silent cap: CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT=12 (conversation_browser_state.py:14), _visible_rows drops overflow and computes hidden_count, but ConsoleWorkspaceContextTray never renders it, and _build_status_copy returns '' unless a search query is active (line 628) — so the default view has zero disclosure ('Showing X of Y' exists only mid-search). task-138 shipped 'cap states with explicit copy' for the old rail's searchable subsection; the newer grouped browser's no-query view is an uncovered gap, not a re-report. 12 seeded + native sessions exceed the cap, hiding the oldest with no hint.

#### j2-switcher-mislabels-saved-chats-in-progress — Switcher labels every idle saved conversation 'Chats - in-progress' while the rail calls the same items 'Chats - saved chat - Xm'
*J2 Returning power user · P2 · verdict NEW (high confidence) · heuristic: Consistency / accuracy of system status*

**Observed:** In the unfiltered Ctrl+K list, all saved conversations (nothing running, no open tabs for them) show the subtitle 'Chats - in-progress'; the rail simultaneously describes them as 'Chats - saved chat - 21m'. 'In-progress' falsely implies activity, and the two surfaces use different vocabulary for identical objects. The switcher also omits the age labels the rail has, weakening recognition where it matters most.

**Expected:** One consistent state vocabulary (saved chat / open session / active session) plus recency shown in both surfaces.

**Repro:** Open Ctrl+K with no query and compare each result's subtitle against the same conversation's subtitle in the rail Chats section.

**Evidence:** `j2-35-switcher-unfiltered.png`, `j2-48-switcher-multi-match.png`, `j2-20-boot2-rail.png`

**Verifier note:** Code-verified: the 'in-progress'→'saved chat' humanization map lives only in console_workspace_context.py (_STATUS_DETAIL_LABELS, lines 38-51); the switcher subtitle joins raw row.status (console_switcher_state.py:90-94), and persisted rows carry status=item.get('state') whose default is the internal lifecycle value 'in-progress' (chat_conversation_service.py:203). Also confirmed: builders set updated_sort but never updated_label, so switcher rows show no ages. The task-179 unification (chats-conversations-unified ledger) covered rail+Library only — the switcher never had the mapping, so NEW not REGRESSION.

#### j2-rail-order-ignores-recency — Rail orders conversations by creation time, not last activity: the chat used minutes ago stays buried mid-list with its creation age
*J2 Returning power user · P2 · verdict NEW (high confidence) · heuristic: Recognition over recall (recency expectations)*

**Observed:** After sending a message into 'Long conversation about embeddings...' at 13:19, a fresh boot at ~13:27 listed it third from top in the same creation order as before, labelled '15m' (creation age) with no sign of the recent activity. Returning power users scan for 'the chat I was just in'; creation-ordered lists defeat that and the age label misrepresents recency of use.

**Expected:** Most-recently-active ordering (or a clear 'Recent' grouping) and age = last activity.

**Repro:** Send a message in an old conversation, restart the app, compare rail order/ages: unchanged from creation order.

**Evidence:** `j2-20-boot2-rail.png`, `j2-12-reply-later.png`

**Verifier note:** Root cause found in code: persisted rows populate updated_sort from item.get('updated_at') or created_at or last_updated (chat_screen.py:4093-4098), but normalize_conversation_row exposes only last_modified/created_at — no 'updated_at' key (chat_conversation_service.py:243-244) — so sorting (recency-first by design, _sort_normal_rows) and age labels silently degrade to creation time for all persisted rows; the Ctrl+K switcher ordering shares the same field. Present since the browser's introduction (commit 995a3f9ae, 2026-06-27), so NEW latent bug, not a REGRESSION of relative-age-labels (which pinned the label derivation and native-session updated_at, both still correct). One-line-class fix: include last_modified in the fallback chain.

#### j2-star-toggle-renders-as-dot — Per-row star toggle renders as a bare '.' (off) / '*' (on) — unrecognizable affordance that I toggled by accident without any feedback
*J2 Returning power user · P2 · verdict NEW (high confidence) · heuristic: Affordance clarity / feedback*

**Observed:** Every rail conversation row ends in a single '.' character on a button-colored cell. Nothing identifies it; during testing a stray click on it starred 'Quick question re keybindings' — no toast, no visible response at click time (the rail had also just mode-swapped) — and the star was only discovered later when the row appeared under Starred with a '*' glyph. A one-cell '.' vs '*' distinction is nearly invisible at a glance in a character grid.

**Expected:** A recognizable glyph pair (☆/★ or [ ]/[*]) plus confirmation feedback ('Starred <title>') when toggled.

**Repro:** Look at any rail conversation row's right edge ('.'), click it: the conversation is starred/unstarred silently; compare Starred section before/after.

**Evidence:** `j2-57-mangle-trialB.png`, `j2-02-initial.png`, `j2-44-dot-click.png`

**Verifier note:** Code-verified: star button label is literally '*' if row.starred else '.' (console_workspace_context.py:950), and the press handler (chat_screen.py:12435-12483) notifies only on failure — success is silent except the eventual rail resync. The '.'/'*' pair is outside the settled console_glyphs vocabulary (glyph-language decision predates this browser and doesn't cover it); the starring feature itself post-dates the ledger. Accidental silent toggle observed in testing supports P2 (unrecognizable affordance + unconfirmed state change on every row).

#### j3-toast-occludes-composer-controls — Feedback toasts cover the staged-attachment chip and Send/Attach/Save buttons for ~5s and swallow clicks
*J3 Attachments · P2 · verdict NEW (high confidence) · heuristic: visibility of system status / affordance clarity*

**Observed:** Every attach/clear/limit event pops a toast anchored bottom-right, directly over the composer's right side. While it is up (measured 5.0s per toast; one persisted >=7s across typing), the staged-chip ('5 files' etc.) and the Send/Attach/Save buttons are invisible and mouse clicks at their coordinates do nothing (verified by blind-clicking the known Attach position during a toast - the picker never opened). During sequential multi-attach the controls are hidden almost continuously; in j3-30 the composer looks completely empty while 5 files are actually staged.

**Expected:** Feedback must not obscure the primary controls it reports on. Place toasts above the composer or in the status bar, keep them short-lived, and never let them intercept clicks aimed at controls beneath.

**Repro:** Console screen (2050x1240, provider home). Click composer Attach, pick any file. Immediately look at the composer's right side: the 'X attached' toast covers the chip strip and buttons for ~5s; clicking where Attach was during that window does nothing. Attach several files in a row to see the controls stay hidden.

**Evidence:** `j3-30-five-images.png`, `j3-44-just-sent.png`, `j3-24-sixth-attempt.png`

**Verifier note:** Confirmed: j3-30 shows the 'img5.png attached' toast sitting exactly over the staged chip and Send/Attach/Save; chat_screen.py has ~95 notify() sites (attach/clear/limit all toast) using Textual's default bottom-right 5s toasts, which do capture clicks. No ledger item or backlog task covers toast placement/occlusion. Downgraded P1→P2: occlusion is transient (~5s), keyboard path unaffected, and a click dismisses the toast so the second click lands — real friction on a core flow, not a hard block.

#### j3-transcript-giant-gap-layout — Transcript spreads messages to the extremes of the pane, leaving a huge dead gap in between
*J3 Attachments · P2 · verdict NEW (high confidence) · heuristic: aesthetic and minimalist design / degradation of scanability*

**Observed:** After the first send, the user message renders at the very top of the transcript pane and the 'Assistant [failed]' + System rows at the very bottom, with ~40 empty rows between them. The pattern persists after more messages, across reloads of the persisted conversation, and in the selected-message state - it is layout, not scroll position or streaming anchoring.

**Expected:** Messages should stack contiguously (top-anchored or bottom-anchored), so the conversation reads as a chronological flow.

**Repro:** Send one message in a fresh Console conversation and let it complete/fail. Observe the user message pinned top and the response pinned bottom with a large void between.

**Evidence:** `j3-63-response-final.png`, `j3-67-shift-enter.png`, `j3-79-view-modal.png`

**Verifier note:** Real, but narrower than reported: j3-63/67 show the void sits immediately after the inline-image row and spans ~36-40 rows — matching the image row widget's max_height 40 clamp (console_transcript.py:932); non-image messages stack contiguously (j3-67 bottom run of 5 rows). So this is an inline-image row-height defect from task-215/PR #626 (post-dates every June-verified transcript capture), not a general transcript layout regression, and no ledger item covers image-row geometry — reviewer's suspected_regression is wrong but the defect is genuine and unrecorded. P2 appropriate: any conversation containing an image becomes unscannable.

#### j4-status-surfaces-say-ready-during-run — All status surfaces report 'Ready' / 'No active work' while a generation is actively streaming
*J4 Streaming · P2 · verdict NEW (high confidence) · heuristic: consistency / visibility of system status*

**Observed:** During an active run (both the silent thinking phase and while tokens were visibly streaming with a '[streaming]' tag in the transcript), the screen-level status chip still reads 'Ready', and the Inspector panel - the dedicated status surface - reads 'Status: Ready', 'Live work: No active work', 'Provider: ready' (j4-33 shows the assistant message mid-stream in the same frame). cell_attrs check confirmed the Ready chip is a plain non-bold gray chip that never changes during a run. The only truthful indicators are the amber Stop button and the per-message [streaming] suffix.

**Expected:** Status surfaces should agree with reality: the chip and Inspector 'Run/Live work' section should switch to a running state (e.g. 'Generating... 12s') for the duration of the run, and return to Ready on completion/stop.

**Repro:** Send any long prompt -> while the reply streams ([streaming] visible in transcript), read the chip under the Console title and open the Inspector panel -> both claim Ready/No active work.

**Evidence:** `j4-33-streaming3.png`, `j4-04d-gap-40s.png`, `j4-36-just-after-stop.png`

**Verifier note:** Confirmed in j4-33 (mid-[streaming] frame shows header chip 'Ready', Inspector 'Status: Ready', 'Live work: No active work', 'Provider: ready'). Code shows these are readiness/launch-context surfaces by construction (console_run_inspector.py:342-347 status line only knows Blocked/Needs-approval/Source-blocked/Ready; console_display_state.py:429 'Live work' comes from pending Library-RAG launch context, never chat runs; ConsoleControlState has no run-active field) — no ledger item settles that they stay 'Ready' during runs. Adjacent to (not covered by) the 2026-07-17 shell-chrome critique's 'Console triple readiness display' finding, which is about redundancy not run-state staleness; inspector-static-streaming-excerpt (task-280) covers only the selected-message excerpt row. Downgrade to P2: truthful run indicators exist (amber Stop, [streaming] suffix, tab dot) — the defect is contradiction, not absence.

#### j4-keyboard-cannot-reach-transcript-scroll — Keyboard-only users cannot scroll the transcript: F6 pane cycling never grants it scroll focus; keys only work after a mouse click on message text
*J4 Streaming · P2 · verdict NEW (medium confidence) · heuristic: keyboard reachability*

**Observed:** With an overflowing transcript loaded (idle): PageUp with composer focus does nothing; cycling panes with F6 four times and pressing PageUp after each cycle never scrolled the transcript; Home never worked. PageUp/ArrowUp scroll only after clicking directly on message text with the mouse - a mouse-only affordance in a footer-advertised keyboard-first UI ('F6 next pane'). In a terminal app this makes scrollback effectively mouse-gated.

**Expected:** F6 pane cycling should be able to land focus on the transcript scroller (with a visible focus indicator), after which PageUp/PageDown/Home/End scroll it.

**Repro:** Load a long conversation -> without touching the mouse press PageUp, then F6+PageUp repeatedly (4 cycles) -> view never moves; click any message text with the mouse -> PageUp now scrolls.

**Evidence:** `j4-30b-idle-pageup-after-click.png`, `j4-29-wal-loaded.png`, `j4-30a-idle-wheel.png`

**Verifier note:** Live observation accepted, but the stated mechanism is wrong: the transcript IS in the F6 cycle (CONSOLE_FOCUS_TARGETS_BY_PANE maps console-transcript-surface → console-native-transcript, chat_screen.py:371-379; _focus_console_workbench_target force-sets can_focus and focuses) and Tests/UI/test_workbench_pane_focus.py::test_console_f6_cycles_between_workbench_panes_and_wraps_backward passes in this worktree (ran it: 1 passed). ConsoleTranscript(VerticalScroll) inherits PageUp/PageDown bindings. So the live failure is most plausibly focus being silently stolen back (sync/refocus between F6 and PageUp) and/or invisible focus state making the landed pane unknowable — a real keyboard-only gap not covered by any ledger item (keyboard-bindings covers c/e/r + selection arrows only). P2 appropriate; needs live-rig reproduction against the passing harness.

#### j4-rail-sections-vanish-during-run — Left-rail Starred/Workspaces/Chats sections disappear entirely while a run is active, then reappear
*J4 Streaming · P2 · verdict NEW (high confidence) · heuristic: consistency / layout stability*

**Observed:** During generation the rail drops its whole conversation area: j4-33 (mid-stream) shows only Session/Context/Model/Agent/Details with the Starred/Workspaces/Chats lists gone; j4-04d shows the same during the thinking gap. Immediately after the run ends the sections return (j4-36). A user glancing at the rail mid-run sees their conversation list apparently deleted, and the whole rail layout jumps twice per send.

**Expected:** The rail should keep its content stable during runs (at most disable interactions), never removing the conversation list wholesale.

**Repro:** Note the rail's Chats list -> send any prompt -> while the run is active, look at the rail: Starred/Workspaces/Chats are gone -> when the run finishes they return.

**Evidence:** `j4-33-streaming3.png`, `j4-04d-gap-40s.png`, `j4-36-just-after-stop.png`, `j4-04b-gap-5s.png`

**Verifier note:** Confirmed real and consistent: j4-04b/04d/33 (Session section reduced to a bare 'Workspace Default' row — Switch, scope, search, Starred/Workspaces/Chats all absent) vs j4-36 (restored while Stop button still visible, i.e. immediately at run end). Not covered: tick-ttl-2s-gating covers ≤2.8s refresh LAG, not wholesale disappearance; rail-layout-quiet-focus/rail-conversations-bounded say nothing about runs. The state builder always attaches the browser (_with_console_conversation_browser_state, chat_screen.py:4702+), so suspect the per-0.2s-tick exclusive console-sync/legacy-alias worker kicks (chat_screen.py:5438-5444) repeatedly cancelling the tray rebuild while ticks run, or run-time recompose height clipping. P2 confirmed severity — twice-per-send layout jump plus apparent data loss.

#### j4-subagent-detour-duplicate-text-truncated-tools — Plain question auto-spawned a sub-agent: answer rendered twice in full, tool entries truncated mid-word, header chips contradict the run
*J4 Streaming · P2 · verdict NEW (high confidence) · heuristic: aesthetic & minimalist design / consistency*

**Observed:** The first ordinary prompt was routed through a 'spawn_subagent' tool call. Consequences in the transcript: (1) the identical 600-word answer appears twice back-to-back - once as the italic tool result ('spawn_subagent -> ### Understanding SQLite...') and once streamed into the assistant bubble; (2) the tool-call args and tool result are hard-truncated mid-word ('the traditional rollba', 'the main database remains in') with no ellipsis and no expand affordance; (3) all this happened while the header chips claimed 'Tools: 0 ready' and 'Approvals: 0 pending' - only the Inspector reveals 'MCP: 10 tools ready'. The hidden sub-agent also made the perceived first-token latency ~100s because its own streaming is invisible.

**Expected:** Tool provenance should be collapsed/summarized (not duplicate the full answer), truncation should be marked with an expand affordance, and the Tools chip should reflect the tools that can actually run.

**Repro:** Fresh provider home -> ask the 600-word WAL question -> model spawns a sub-agent -> compare the tool-result block with the assistant bubble below/above it; read the 'Tools: 0 ready' chip vs Inspector 'MCP: 10 tools ready'.

**Evidence:** `j4-10-reply1-complete.png`, `j4-14-after-stop-settled.png`, `j4-32-inspector-midrun.png`

**Verifier note:** All three sub-claims verified in j4-33 + code: (1) duplicate full answer is structural — format_agent_step_marker renders '⚙ {tool_name} → {result}' with the UNtruncated tool result (console_agent_bridge.py:145-148), so a spawn_subagent whose result IS the answer prints it twice; (2) spawn summary cut mid-word with no ellipsis ('the traditional rollba'); (3) header chip 'Tools: 0 ready' vs Inspector 'MCP: 10 tools ready' — chip counts _console_tool_count only, MCP counted separately. Not prior art: task-231 was an efficiency review (Done, spawned perf tasks 243-245 only); mcp-chat-bridge-deferred no longer applies since the agent runtime made MCP tools live in Console. The auto-spawn routing itself is model behavior under the agent operating prompt, not a UI defect — the transcript/chip consequences are the finding.

#### j4-first-feedback-latency-cluster — Slow/silent first feedback after primary actions: message echo up to 7s, rail conversation clicks that silently do nothing, delayed Inspector toggle
*J4 Streaming · P2 · verdict NEW (medium confidence) · heuristic: feedback latency*

**Observed:** Cluster of measured instances: (a) Enter-to-send: 350ms after Enter the composer still held the full text and the transcript said 'No messages yet' (j4-03); in another session the first transcript change after Enter took 7.3s (echo of the user's own message), though a later send echoed in 1.0s; (b) rail conversation clicks: in one session clicking 'No tools: explain...' produced no visible change within 1.2s and clicking 'Write a detailed...' never opened it at all during ~10s of subsequent interaction (transcript stayed on the empty 'Chat 1'), while in the next session the same click opened in 0.6s - silent intermittent failure with no pressed/loading feedback; (c) clicking 'Inspector' during a run showed no response within 0.7s (j4-32 still collapsed) and the panel was simply found open in a later frame.

**Expected:** Every primary click/submit should acknowledge within ~100ms (pressed state, optimistic echo of the sent message, loading indicator on conversation open).

**Repro:** (a) Type a prompt, press Enter, watch composer/transcript in the first second; (b) right after app start, click a saved conversation in the rail Chats list and wait - sometimes nothing happens; (c) click 'Inspector' during a run.

**Evidence:** `j4-03-sent-immediate.png`, `j4-18-first-change.png`, `j4-26-partial-after-restart.png`, `j4-32-inspector-midrun.png`, `j4-33-streaming3.png`

**Verifier note:** Not covered by the perf ledger: task-280/259 fixed tick/DB-on-loop and transcript derivation, and no item covers first-send echo latency, silently-failing rail conversation clicks, or delayed Inspector toggle. The 7.3s worst-case echo and dead rail clicks are intermittent single-session measurements (hence medium confidence) but three independent sub-symptoms point at on-loop work in the send/resume paths (e.g. first-send agent-bridge/MCP catalog init). No pressed/loading acknowledgment on rail conversation rows is verifiable by design (plain Buttons, no busy state). P2 appropriate.

#### j4-model-catalog-toast-occludes-composer-actions — Boot-time 'Model catalog' toast sits on top of the composer's Send/Attach/Save buttons for tens of seconds
*J4 Streaming · P2 · verdict NEW (high confidence) · heuristic: aesthetic & minimalist design / affordance clarity*

**Observed:** On boot the app auto-refreshes model catalogs and pops a toast ('Model catalog - Model lists updated - OpenRouter: 333 new cached') bottom-right, exactly covering the composer action buttons (Send/Attach/Save are invisible in j4-01 and still covered in j4-03 ~20s later, while the user is typing their first message). No visible dismiss control; it occludes the primary action area during the most likely first interaction.

**Expected:** Toasts should not cover primary controls (place above the composer or in the status bar) and should be dismissible/short-lived.

**Repro:** Launch the app with a provider configured -> wait for the model-catalog toast -> observe it covering Send/Attach/Save while composing the first message.

**Evidence:** `j4-01-initial.png`, `j4-03-sent-immediate.png`, `j4-04b-gap-5s.png`

**Verifier note:** Side effect of the newly-shipped model-catalog auto-refresh (task-301 Done; boot-time notify in app.py/model_auto_refresh.py). Textual toasts render bottom-right — directly over Send/Attach/Save — and j4-01→j4-03 shows coverage persisting ~20s into first compose (likely stacked per-provider toasts). No prior toast-placement/occlusion item in ledger or backlog. P2: occludes primary action affordances during the most likely first interaction, though Enter-send still works.

#### j5-validation-error-low-salience — Save-time validation errors render as unemphasized gray text at the top of a 68-row modal, far from the offending field
*J5 Settings · P2 · verdict NEW (high confidence) · heuristic: error recovery; visibility of system status*

**Observed:** Clicking Save with an empty Temperature keeps the modal open and prints 'Temperature is required.' directly under the intro text. Cell-attrs of the error row: fg #E4E4E5 on bg #32303B, no bold/underline/inverse — nearly the same treatment as the descriptive text above it. The error sits 17 rows above the Temperature field, which itself shows no visible invalid-state highlight. The stale error also remains on screen after the field is fixed, until the next Save. Same pattern for 'Reasoning effort must be one of none, minimal, low, medium, high, or xhigh.'

**Expected:** Errors should be visually distinct (color/bold/inverse) and anchored at or near the offending field (or the field marked invalid), so the user understands why Save 'did nothing' in a modal taller than one screenful of attention.

**Repro:** 1. Rail > Configure. 2. Clear Temperature to empty. 3. Click Save -> modal stays open; single gray line 'Temperature is required.' appears at top; field not highlighted. 4. Re-enter a valid value -> error line remains until next Save.

**Evidence:** `j5-67-error-styling.png`, `j5-63-after-save-9.png`, `j5-64-after-save-maybe.png`

**Verifier note:** Verified in code and not covered by task-178 (which addressed scope labeling, boolean controls, and accepted-values placeholders — not error presentation). .console-settings-error is styled 'background: $ds-status-error 10%; color: $ds-text-primary' (_agentic_terminal.tcss:293-301) — near-body-text salience matching the reviewer's cell-attrs; it is a single banner mounted at the top of the modal (console_settings_modal.py:276), never anchored to or highlighting the offending field, and only updated inside _validated_draft_or_show_errors (lines 699-710) so it stays stale after the field is fixed until the next Save. 'Save did nothing' confusion in a taller-than-viewport modal justifies P2.

#### j5-quick-model-popover-empty — Quick 'Change model…' popover doesn't show the current model and has an unlabeled temperature box
*J5 Settings · P2 · verdict NEW (high confidence) · heuristic: visibility of system status; recognition over recall; consistency*

**Observed:** The palette command 'Console: Change model… — Quick provider/model/temperature switch (Alt+M)' opens a compact 'Model' popover containing: a Provider select showing the raw key 'llama_cpp' (the full modal shows 'llama.cpp'), a Model select showing the placeholder 'Select' instead of the active model (local-gemma), a 'Search all models…' input, a bare unlabeled input containing '0.6', static text 'Streaming: on', and Full settings…/Apply buttons.

**Expected:** A quick-switch surface must reflect the current model (that's what the user is switching FROM), label its temperature input, and use the same provider display names as the full settings modal. As-is a user cannot confirm the active model and could Apply with 'Select'.

**Repro:** 1. Click transcript, Ctrl+P, type 'change model', activate 'Console: Change model…'. 2. Observe popover: Model select = 'Select' placeholder (session model is local-gemma), temperature input has no label, provider shows 'llama_cpp'.

**Evidence:** `j5-76-model-popover.png`, `j5-71-palette-model.png`

**Verifier note:** Real bug with confirmed mechanism, beyond the tracked raw-key issue: ConsoleModelPopover.compose seeds the model Select with the session model (console_model_popover.py:113-115) and build_console_model_options injects the current model into options, BUT the provider Select's mount-time Select.Changed fires _provider_changed (lines 143-159), which rebuilds model options with current_model=None and calls set_options() — resetting the model Select to BLANK and wiping the prefill. Matches j5-76 exactly (model shows 'Select' while rail shows local-gemma). Unlabeled temperature Input (placeholder invisible once a value is present) also verified at lines 122-128. The 'llama_cpp' raw provider key slice is KNOWN (task-194) — the popover ledger item and gap-not-exercised-2026-07 confirm this surface was never live-verified, so NEW not REGRESSION.

#### j5-system-prompt-hidden-door — 'System: none' rail text is the only door to the system-prompt editor but carries zero click affordance; Configure modal omits system prompt entirely
*J5 Settings · P2 · verdict NEW (high confidence) · heuristic: affordance clarity; recognition over recall*

**Observed:** The rail line 'System: none' is styled identically to the static Provider/Model/Temperature lines (cell-attrs: same bg, no underline/bold/inverse), yet clicking it opens the session 'Edit system prompt' modal (textarea, Name, Save to Library, Clear/Cancel/Apply, 'Applies to this session.'). The Console Settings modal opened by 'Configure' — where a user would look for everything model-related — has no system-prompt field, and no other visible entry point exists on the screen.

**Expected:** An interactive row must look interactive (button styling like the adjacent 'Configure', or a chevron/link treatment), and/or the session settings modal should include or link to the system prompt. Otherwise users conclude the system prompt cannot be changed.

**Repro:** 1. Look at rail Model section: 'System: none' renders as inert text (styling identical to labels above). 2. Click it -> 'Edit system prompt' modal opens. 3. Open Configure modal -> no system prompt control anywhere in it.

**Evidence:** `j5-21-rail-model-section.png`, `j5-22-system-none-click.png`, `j5-23-modal-open.png`

**Verifier note:** Code-verified and uncovered by any ledger item or backlog task. #console-rail-system-line is a plain Static with no interactive styling (only color rules at _agentic_terminal.tcss:2428-2435, no hover/underline, no tooltip assigned at compose, chat_screen.py:7264-7281) yet a screen-level on_click at chat_screen.py:11282-11285 opens the system-prompt editor; grep confirms ConsoleSettingsModal contains zero system-prompt controls. One correction to the claim of 'only door': a /system composer command also opens the editor (commit 'feat(console): /system + system prompt modal + rail preview') — but that path is equally undiscoverable, so the P2 discoverability grade stands.

#### j6-f6-rail-stop-invisible — F6 pane cycle has an invisible stop: rail focus shows no pane-level indicator
*J6 Keyboard-only + small terminal · P2 · verdict NEW (high confidence) · heuristic: Focus visibility / consistency*

**Observed:** F6 cycles 3 stops: rail -> transcript -> composer. Transcript and composer stops paint a clear accent (#0178D4) pane border, but the rail stop paints NO accent anywhere (verified by full-screen fg-color scan: accent zones empty at that stop). The only style delta is the 3-cell rail-collapse handle '◂' background changing #1e1e1e -> #272727 (~1.08:1 contrast, imperceptible). The FIRST Tab press after the rail stop also changed zero cells (two consecutive keyboard stops visually identical). The Inspector pane is never part of the F6 cycle.

**Expected:** Every F6 stop is visibly distinguishable with the same pane-border treatment (accent border on the rail like transcript/composer), so the user always knows which pane owns focus.

**Repro:** Open Console (seeded home, 2050x1240). Focus composer. Press F6 once (rail stop): no accent border appears anywhere; compare with a second F6 (transcript: accent rows 14-68) and third (composer: accent rows 71-75). cell_attrs_row on row 13 shows only the '◂' handle bg shifting #1e1e1e->#272727.

**Evidence:** `j6-a06-cycle1.png`, `j6-a06-cycle2.png`, `j6-a06-cycle3.png`, `j6-a07-invisible-stop.png`, `j6-a08-rail-tab1.png`

**Verifier note:** Code-confirmed: F6 rail stop focuses #console-context-rail-collapse (CONSOLE_FOCUS_TARGETS_BY_PANE, chat_screen.py:372-377); .console-rail-collapse-button has no :focus rule (default Button bg shift only) and #console-left-rail:focus paints border $ds-column-line — identical to the region frame color, i.e. invisible by construction (_agentic_terminal.tcss:2170). Adjacent settled decision rail-layout-quiet-focus (task-149) covers only removing the loud focus boundary on the rail BODY scrollable, not the F6 pane-stop indicator (F6 convention is task-103, which never specified visible indication). Downgraded P1→P2: one invisible stop in an otherwise clear cycle; the umbrella focus-visibility finding carries P1.

#### j6-switcher-arrows-dead — Ctrl+K session switcher: arrow keys do not navigate results; focused result highlight nearly invisible
*J6 Keyboard-only + small terminal · P2 · verdict NEW (high confidence) · heuristic: Consistency & standards / focus visibility*

**Observed:** In the Switch Session modal, ArrowDown from the search field (x3) changed nothing but the input cursor blink. After Tab into the results, ArrowDown (x2) still moved nothing — results are plain Buttons in a Vertical (no list widget), so navigation is Tab/Shift+Tab only. The focused result's only distinction is bg #272727 vs #1e1e1e on siblings (~1.08:1). Enter in the search box activates the FIRST result regardless of any focused result.

**Expected:** Quick-switcher idiom: Up/Down moves a clearly highlighted selection through the result list, Enter activates the highlighted item; highlight meets ~3:1 contrast.

**Repro:** Press Ctrl+K. Press ArrowDown 3x (no visible change; buffer diff empty). Press Tab (first result gains bg #272727), then ArrowDown 2x (no change). Tab again moves to second result. Enter activates focused result.

**Evidence:** `j6-a09-ctrlk-open.png`, `j6-a10-ctrlk-down3.png`, `j6-a11-ctrlk-tab.png`, `j6-a11-ctrlk-tab-down2.png`, `j6-a12-ctrlk-tab2.png`

**Verifier note:** Code-confirmed: console_session_switcher_modal.py BINDINGS are only escape/f2; results are plain Buttons in a Vertical, no list widget or arrow handling; phase3 plan (2026-07-04-console-keyboard-layer-phase3.md) is silent on arrow navigation. The Enter-activates-first-result part merely restates shipped design (ledger ctrl-k-switcher) — but the absent up/down navigation and the unfocused-vs-focused Button delta (no :focus rule for .console-switcher-result) are uncovered; ledger gap-not-exercised-2026-07 confirms the switcher was never live-tested before. Not a regression. Downgraded P1→P2: type-to-filter+Enter and Tab navigation still work.

#### j6-edit-modal-late-open-keystroke-leak — Edit modal opens late and silently; keystrokes pressed while waiting are typed into the message draft
*J6 Keyboard-only + small terminal · P2 · verdict NEW (medium confidence) · heuristic: Feedback latency / error prevention*

**Observed:** Pressing e on the selected user message gave no visible response — a full buffer dump 0.9s later showed no modal. Believing it failed, I pressed e again; the Edit Message modal (which had opened in the meantime) received that keypress as text: the textarea read 'eWhat backoff strategy should I use for websocket reconnects?' — the draft was silently corrupted with a stray 'e'. Saving would persist the corruption.

**Expected:** The edit modal opens within one tick of the keypress (or shows immediate busy feedback), and late-arriving modals must not swallow keystrokes typed before they appeared.

**Repro:** Select a user message in the transcript (k until action row sits under it), press e, and press e again about a second later. The modal textarea shows the second 'e' prepended to the message text.

**Evidence:** `j6-a19-edit-modal.png`, `j6-a20-e-retry.png`, `j6-a21-edit-modal-full.png`

**Verifier note:** Mechanism is real: transcript 'e' presses the Edit action Button (keyboard-bindings ledger), the Button.Pressed message hops through widget queues before the async dispatch pushes the modal (chat_screen.py:10166-10171), with zero synchronous feedback at keypress — keys typed in the gap land in the late-opening TextArea. Not covered by any ledger item or task. Medium confidence / downgraded P1→P2 because the 0.9s window is likely amplified by the textual-serve harness latency and the corruption is visible in the modal before saving.

#### j6-live-resize-broken-reflow — Live shrink to 700x480 produces a broken layout different from cold start: transcript/inspector vanish and stale chrome fragments remain
*J6 Keyboard-only + small terminal · P2 · verdict NEW (medium confidence) · heuristic: Degradation under resize*

**Observed:** Live-resizing a healthy 900x620 session down to 700x480 left: rail expanded to full width, transcript and inspector gone entirely (cold start at the same size keeps them), stale inspector text fragments overlaying the header rows, the screen title replaced by a stuck nav tooltip 'Open the live agent Console.', and no composer. Growing to 1400x900 restored panes and composer correctly, but the tooltip fragment kept overpainting the header border through subsequent reflows.

**Expected:** Reflow after a live resize converges to the same layout as a cold start at that size; overlays/tooltips are re-rendered or dismissed on resize instead of leaving artifacts over chrome.

**Repro:** Open at 900x620, hover/click a nav tab label, then resize the browser viewport to 700x480 and wait 2.5s: compare with a fresh 700x480 session. Resize up to 1400x900 and note the persistent tooltip fragment across rows 2-3.

**Evidence:** `j6-b02-live-700x480.png`, `j6-b05-cold-700x480.png`, `j6-b03-live-1400x900.png`, `j6-b04-back-900x620.png`

**Verifier note:** Evidence j6-b02-live-700x480.png vs j6-b05 cold start confirms divergence: rail full-width, transcript/inspector regions gone, no composer, and the nav tooltip 'Open the live agent Console.' stuck over the header — a mounted-overlay leftover surviving full repaint, so app/framework-level rather than a paint artifact. chat_screen.py has no resize handling, so divergence is Textual reflow + stale overlay state. Medium confidence because the only exercisable resize path was browser-viewport via textual-serve (journey's own 'blocked' note); native SIGWINCH may differ. Not covered by any ledger item. P2 appropriate (recoverable by growing the window).

#### j6-help-omits-keyboard-vocabulary — F1 help is a bare full-screen text dump that omits the transcript keyboard vocabulary
*J6 Keyboard-only + small terminal · P2 · verdict NEW (high confidence) · heuristic: Recognition over recall / help & documentation*

**Observed:** F1 replaces the whole 2050x1240 screen with ~15 lines of unstyled top-left text listing 5 actions and 7 shortcuts. It does not mention transcript keys (j/k select, Enter show-actions, c copy, e edit, r regenerate, Escape clear), F2 rename in the switcher, Shift+Enter newline, or Escape-to-composer. The in-transcript 'Guide:' line explains icon meanings (♻ ---> 👍👎 🗑) but not their key bindings either, so the c/e/r keys used by this journey are undiscoverable anywhere in the app.

**Expected:** Help presents the full keyboard map (grouped: panes, transcript, composer, modals) in a styled panel sized to content; the action-row guide teaches the single-key shortcuts.

**Repro:** Press F1 on the Console screen at 2050x1240 and compare the listed shortcuts against the transcript BINDINGS (j/k/enter/escape/c/e/r all missing).

**Evidence:** `j6-a34-f1-help.png`, `j6-a18-select-user.png`

**Verifier note:** Evidence j6-a34-f1-help.png matches code exactly: WorkbenchHelpPanel renders only the 5 actions + the 7 CONSOLE_WORKBENCH_SHORTCUTS (chat_screen.py:381-387) as an unstyled top-left dump on a 2050x1240 blank screen; transcript keys j/k/c/e/r/Enter/Esc (console_transcript.py:378-380), F2, Shift+Enter, Alt+M/Alt+1..9 all absent. Aggravating context: task-264 (settled) folded the old pane-contextual footer (which used to surface C/E/R when the transcript was focused, per contextual-footer ledger item) into these same 7 static shortcuts, so c/e/r are now genuinely undiscoverable — but no ledger item or task owns the help-content gap itself. NEW, P2 stands.

## New findings — P3 (polish)

#### j1-tooltip-occludes-content — Hover tooltips render on top of exactly the content the user needs to read (test results, discovered-model list, tab bar)
*J1 New-user cold start · P3 · verdict NEW (medium confidence) · heuristic: affordance clarity / occlusion; hover reliance in a terminal UI*

**Observed:** The Test Provider tooltip ('Run a local readiness check...') painted over the provider-test result line while it was being read, garbling both (j1-15). The Save-selected tooltip ('Append selected discovered model IDs to the local provider list.') covered the discovered-model checkbox row itself — while the cursor rests on the very button that acts on that row (j1-25). A tab tooltip similarly covered the left half of the tab bar. Tooltips are also the only carrier of key guidance like 'URL-based local providers also get a short live endpoint probe', invisible to keyboard users.

**Expected:** Tooltips should position away from the control's related content (and never over the data a click just produced); essential explanations should exist as static text, not hover-only.

**Repro:** Hover 'Test Provider' after clicking it -> tooltip overlaps result text; hover 'Save selected' after discovery -> tooltip covers the model row; park mouse on the Home tab -> tooltip covers tab labels.

**Evidence:** `j1-15-test-bad-url-result.png`, `j1-25-save-selected.png`, `j1-18-discover-bad-url-result.png`

**Verifier note:** No prior art in the ledger or backlog (searched 'tooltip' across tasks; the phase plans specify tooltip CONTENT, never placement). The observations are credible from the captures (Test Provider tooltip over the result line j1-15, Save-selected tooltip over the model row j1-25) and follow from Textual's default cursor-adjacent tooltip placement — the app does not offset tooltips away from result/target content. The strongest sub-point is hover-only guidance (e.g. the live-probe explanation exists only as tooltip text), invisible to keyboard users. Downgraded P2→P3: transient, mouse-only, self-dismissing on move; a placement/static-copy polish item rather than a flow blocker.

#### j1-internal-jargon-in-ui — Internal identifiers leak throughout first-run surfaces: config paths, env-var dumps, ADR numbers, raw conversation UUID
*J1 New-user cold start · P3 · verdict NEW (high confidence) · heuristic: match between system and the real world*

**Observed:** User-facing copy includes 'api_settings.llama_cpp.api_url=http://...', 'LLAMA_CPP_API_KEY=<redacted>', 'model=missing | status=blocked', discovery rows suffixed '| runtime | runtime_discovered | capability=unknown', a section headed 'Automatic refresh (ADR-020)', and after the first send the Console rail shows 'Scope b7c3bdfb-99c9-462d-a83e-714b8d...' — a raw conversation UUID as the session scope label.

**Expected:** First-run copy in user vocabulary ('Endpoint', 'API key: not needed', 'Test failed: connection refused'); internal decision-record IDs and UUIDs kept out of primary UI or behind a details view.

**Repro:** Read Test Provider output, discovery result rows, the 'Automatic refresh (ADR-020)' section on Providers & Models, and the rail 'Scope' row after sending the first message.

**Evidence:** `j1-16-test-bad-url-full.png`, `j1-28-model-list-clean.png`, `j1-07-setup-surface.png`, `j1-39-send-plus-4s.png`

**Verifier note:** Code-confirmed cluster on new/first-run surfaces: rail Scope renders the raw conversation id — display_state.py:282 scope_label = str(current_conversation or '') → ConsoleWorkspaceStatusPair('Scope', …) (console_workspace_context.py:583); provider-test detail dumps config paths and env-var status (settings_screen.py:5276-5296, 'api_settings.<key>.<url>=', 'LLAMA_CPP_API_KEY=…'); discovery rows are labeled '<id> | runtime | runtime_discovered | capability=unknown' (settings_screen.py:4934); 'Automatic refresh (ADR-020)' is a live Settings heading from the just-shipped task-301 work. Not covered: micro-polish-186 fixed different rail/footer nits and none of the ledger copy items cover these. Downgraded P2→P3: comprehension/trust polish, no task blockage; note the 'Saved as: chat_defaults.model' guide-panel pattern is deliberate Settings design language (the reviewer praised it), so the fix should target the toast/detail/row copy and the Scope UUID, not config-path language wholesale.

#### j1-provider-switch-stale-window — After switching Provider to llama.cpp, OpenAI's model/endpoint/key fields persist on screen for a while before updating
*J1 New-user cold start · P3 · verdict NEW (medium confidence) · heuristic: feedback latency / consistency*

**Observed:** 900ms after selecting llama.cpp, the form still showed Model=gpt-4o, Endpoint=https://api.openai.com/v1, 'set OPENAI_API_KEY or paste a local key' and readiness 'OpenAI / gpt-4o' (j1-10). A few seconds later the dependent fields settled to llama.cpp values (j1-12). During the window the screen asserts a provider/credential combination that never existed.

**Expected:** Dependent fields update atomically with the provider selection, or show a brief loading placeholder instead of the previous provider's values.

**Repro:** Providers & Models -> open Provider select -> choose llama.cpp -> read the form within ~1s, then again after ~3s.

**Evidence:** `j1-10-llamacpp-selected.png`, `j1-12-after-provider-switch-settle.png`

**Verifier note:** Not covered: task-214 is the reverse direction (Settings shows boot-time selection until manual reselect after a config write), settings-input-select-fix is the Input→Select interaction, and task-290 covers the mount-time recompose storm, not provider-switch dependent-field latency. The observation (OpenAI model/endpoint/key text persisting ~1-3s after selecting llama.cpp before dependent widgets settle, j1-10 vs j1-12) is a transient async-recompose staleness window during which the form asserts a provider/credential combination that never existed. Plausible given the screen's dependent-field rebuild pattern; not verified to the exact mechanism. P3: transient, self-corrects, no data written.

#### j1-card-copy-truncated — Setup card step 3 text is clipped mid-sentence: 'Composer unlocks after' (missing 'setup'), no ellipsis, with space remaining
*J1 New-user cold start · P3 · verdict NEW (high confidence) · heuristic: affordance clarity / degradation (truncation without indication)*

**Observed:** The card renders '3. ○ Send your first message  Composer unlocks after      ' — the source constant is 'Composer unlocks after setup' (console_onboarding_state.py CONSOLE_SETUP_STEP_THREE_DETAIL), so the final word is truncated even though ~6 blank cells remain before the card border, and no ellipsis marks the cut.

**Expected:** The full sentence fits or wraps; if truncation is unavoidable, show an ellipsis.

**Repro:** Boot with a bare home at 2050x1240 -> read row 3 of the Get-started card.

**Evidence:** `j1-01-initial-console.png`, `j1-42-fresh-boot.png`

**Verifier note:** Confirmed by arithmetic + capture, but NOT a regression — it never rendered fully. The step line is 59 cells ('3. ○ Send your first message  Composer unlocks after setup'; _step_text, console_setup_modal.py:438-444; constant console_onboarding_state.py:25), while .console-setup-modal-card gives 62 − 2 border − 4 padding = 56 content cells (tcss:4463-4471); 'setup' word-wraps to a second line that .console-setup-step height:1 (tcss:4417) hides — clipped at a word boundary with blank cells left and no ellipsis, exactly as captured (j1-01/j1-42). Timeline: the modal card shipped 2026-07-04 (c50f1dee9/169a6ba04); the 'Composer unlocks after setup' copy landed 2026-07-11 (76a8b1e35, the setup-card-honest-steps remediation) into the too-narrow card, so the shipped remediation itself never displayed its last word. Trivial fix (widen card or shorten detail); P3 cosmetic but on the very first screen users see.

#### j1-footer-cryptic-stats — Footer shows unexplained abbreviations 'Tokens: -- | P: 144.0 KB | C/N: 816.0 KB | M: 376.0 KB' on first run
*J1 New-user cold start · P3 · verdict NEW (medium confidence) · heuristic: match between system and the real world / recognition over recall*

**Observed:** The status bar of the very first screen shows single-letter memory/storage stats (P:, C/N:, M:) with no legend, tooltip access, or context; a new user cannot decode them and they compete with the key hints for footer space.

**Expected:** Spell out or tooltip the abbreviations, or hide diagnostics behind a debug toggle on first-run.

**Repro:** Boot bare home -> read the right side of the bottom status row.

**Evidence:** `j1-01-initial-console.png`

**Verifier note:** Accurate and uncovered. The string is built in db_status_manager.py:67 ('P: {prompts} | C/N: {chachanotes} | M: {media}' — database file sizes) and rendered into the per-screen footer next to 'Tokens: --' (AppFooterStatus.py:85) with no legend or tooltip; confirmed present on the first-run gate in j1-01. No ledger item covers it (open-rail-model-line-and-footer-nits lists other footer nits; per-screen-footer-hints/task-264 covers shortcut hints only) and no backlog task exists for it. Caveat for the filer: a 2026-07-17 shell-chrome critique session flagged AppStatusLine dead-chrome cleanup as in-flight uncommitted work in the main checkout, so check for an in-flight fix before filing a duplicate. P3: cosmetic diagnostics noise competing with key hints.

#### j2-rail-title-truncation-and-boilerplate — Rail truncates titles to ~17 chars + '...' while spending a full second line per row on near-identical boilerplate ('Chats - saved chat - 7m')
*J2 Returning power user · P3 · verdict NEW (high confidence) · heuristic: Recognition over recall / scannability*

**Observed:** In a 48-cell-wide rail, titles render as 'Websocket reconne...', 'Compare SQLite FT...', 'Meeting notes 202...' (17 chars + ellipsis, well short of the ~30 chars the width allows), because ~10 cells right of the title are reserved (star toggle + padding). Meanwhile all 11 rows repeat the same subtitle text, differing only in the age digits — so half the section's vertical space carries almost no information while the distinguishing part of the titles is cut. Long titles ('Long conversation...') become indistinguishable from any other 'Long conversa...' item.

**Expected:** Titles get the available width (25-35 chars) with tighter right-side controls; subtitle should compress to just the differentiator (age, or state only when not the default).

**Repro:** Open Console with the 12 seeded conversations at 2050x1240 and read the rail Chats section.

**Evidence:** `j2-02-initial.png`, `j2-20-boot2-rail.png`

**Verifier note:** Code-verified: _MAX_CONVERSATION_ROW_TITLE=20 is a fixed constant (console_workspace_context.py:52,988-993) regardless of available rail width, and every row repeats 'workspace_label - detail - age' as the second line (line 924-929). Real density/recognition critique, but downgraded to P3: full titles are available via row tooltips (tooltip_label carries untruncated title) and Ctrl+K search; nothing is blocked. Not covered by any ledger item (micro-polish-186 fixed different rail nits).

#### j2-scope-uuid-exposed — Session section shows a raw conversation UUID as 'Scope', wrapped over two rail lines
*J2 Returning power user · P3 · verdict NEW (high confidence) · heuristic: Recognition over recall (no internal jargon)*

**Observed:** After resuming a conversation the rail Session block reads 'Scope d1ebe478-c825-46b6-83b3-d5901d7bb3a1' (wrapped mid-token). The UUID has no user meaning, duplicates nothing useful, and consumes two lines of premium rail space.

**Expected:** Human-readable scope ('This conversation') or omit; keep identifiers in a debug/details view.

**Repro:** Resume any saved conversation and read the rail Session section's Scope row.

**Evidence:** `j2-08-after-switcher-select.png`, `j2-24-resumed-long.png`, `j2-57-mangle-trialB.png`

**Verifier note:** Code-verified: build_console_workspace_state sets scope_label = str(current_conversation or '') — the raw conversation UUID (display_state.py:282) — rendered as the 'Scope' label/value pair (console_workspace_context.py:581-588), wrapping across two rail lines as screenshotted. The setup-state-callout ledger item blesses label/value rows but not raw-identifier content; no task covers humanizing it. Straightforward jargon-leak polish fix.

#### j2-tab-labels-cryptic-fragments — Tab labels truncate long titles to a short fragment with no ellipsis ('Long', 'Terraform'), destroying scent between similar conversations
*J2 Returning power user · P3 · verdict NEW (high confidence) · heuristic: Recognition over recall*

**Observed:** Resumed conversations open tabs labelled with ~9-14 characters of the title and no truncation marker: 'Long conversation about embeddings and vector stores in local RAG' becomes just 'Long'; 'Terraform state migration help' becomes 'Terraform'. Two conversations sharing a first word would be indistinguishable in the strip; the label doesn't even hint it is truncated.

**Expected:** Wider labels with ellipsis, or middle-truncation preserving distinguishing words; at minimum a visible '…'.

**Repro:** Resume a conversation with a long title; read its tab label in the strip.

**Evidence:** `j2-12-reply-later.png`, `j2-08-after-switcher-select.png`

**Verifier note:** Code-verified rendering defect: _display_title truncates to 19 chars and appends '...' (console_session_surface.py:32,138-144), but the fixed 21-cell button (CONSOLE_SESSION_TAB_WIDTH) word-wraps the label at ~16 usable cells and height-1 shows only the first line — i.e. the first word ('Long', 'Terraform'), never the ellipsis. So the intended truncation marker exists in code and is defeated by Button word-wrap. Not covered by tab-strip-symbols (glyphs only) or auto-title-30ch (session titles). Tooltips carry the full title, keeping this P3.

#### j2-markdown-asterisks-raw — Assistant replies show raw markdown markers ('**local RAG**') in the transcript
*J2 Returning power user · P3 · verdict NEW (medium confidence) · heuristic: Consistency / polish*

**Observed:** The persisted assistant reply renders literally with asterisks: 'in a **local RAG** setup, your primary constraints are...' — emphasis markup is neither rendered nor stripped in the resumed transcript view.

**Expected:** Render basic markdown emphasis or strip markers in the plain transcript rendering.

**Repro:** Get any assistant reply containing **bold** from the model; resume the conversation and read the transcript.

**Evidence:** `j2-24-resumed-long.png`

**Verifier note:** Code-verified: transcript bodies render as plain Textual Content with no markdown pass (_message_render_text, console_transcript.py:161-188). The transcript-visual ledger item pins role-label dimming 'plain text unchanged' but is not a decision against ever rendering markdown; no backlog task covers it. Real polish gap for any model that emits **emphasis**; low severity since content remains fully readable.

#### j3-attach-txt-actually-inserts-content — The same Attach affordance stages images but silently pastes text-file CONTENT into the draft, with a size label that matches neither
*J3 Attachments · P3 · verdict NEW (medium confidence) · heuristic: consistency / recognition over recall*

**Observed:** Picking test-image.png staged a right-side chip ('test-image.png - 341 B', button became paperclip+check). Picking notes.txt instead inserted the file's content into the draft as an inline pill '(doc) notes.txt - 115 B' glued to the typed text with no separator; the only differentiator is a transient toast 'notes.txt content inserted'. The pill's '115 B' does not match the 60 B file (it is the wrapped-content size), and the pill is then subject to the unfurl flow (see j3-enter-hijacked). Users reasonably believe they attached a file; they actually pasted its body.

**Expected:** One affordance, one mental model: either stage text files as attachments too, or label the action distinctly at selection time ('Insert as text') and show the real file size.

**Repro:** Attach test-image.png then notes.txt via the composer Attach picker. Compare: image becomes a right-side chip; txt becomes an inline draft pill reading '115 B' for a 60-byte file, announced only by a 5s toast.

**Evidence:** `j3-42-staged-img-and-txt.png`, `j3-43-typed-before-send.png`, `j3-22-two-staged.png`

**Verifier note:** Facts confirmed in code: text files route insert_mode='inline' and splice content as a pill (chat_screen.py:9946-9963), announced only by a 5s toast; the pill size uses processed_size = len(wrapped content) not the file size (attachment_core.py:290 + label property line 218), so '115 B' for a 60 B file is a real mislabel. The dual routing itself is the shipped attachment_core architecture (phase-1 #621) and task-230 adjudicated only excluded-image formats, so part of this restates design — but the wrong size, the no-separator splice, and zero at-selection differentiation are unrecorded defects. P2→P3: confusion/polish, no data loss, toast does differentiate.

#### j3-cap-not-prevented-at-picker — At the 5-attachment cap the picker still opens and lets you pick a 6th file before rejecting it
*J3 Attachments · P3 · verdict NEW (high confidence) · heuristic: error prevention*

**Observed:** With 5 images staged, the attach button still opened the full file picker; I navigated and selected img6.png; the picker closed exactly like a success and only a toast said 'Attachment limit reached (5 per message).' (a toast that itself renders over the composer controls and expires in ~5s). The attach affordance gives no cue that the cap is reached.

**Expected:** Disable or annotate the attach affordance at the cap ('5/5'), or block inside the picker before file selection, so the user never does dead work.

**Repro:** Stage 5 images, click the paperclip button again, select a 6th image. Picker closes normally; only a transient toast reports the rejection.

**Evidence:** `j3-31-sixth-image.png`, `j3-30-five-images.png`

**Verifier note:** Code-confirmed: the attach button always opens the picker; the cap is enforced only after processing, when store.add_pending_attachment returns False and a toast fires (chat_screen.py:9968-9974). No pre-picker gate, no 5/5 annotation; nothing in task-217/222/230 or the ledger records this as decided. Downgraded P2→P3: pure error-prevention polish — the rejection copy itself is precise and the dead work is one picker round-trip.

#### j3-picker-path-entry-drops-filename — Picker's Ctrl+L path bar silently discards the filename part of a typed full file path
*J3 Attachments · P3 · verdict NEW (high confidence) · heuristic: efficiency of use / error prevention*

**Observed:** Ctrl+L opens a path input (good), but entering an absolute FILE path and pressing Enter/Go only navigates to the parent directory and resets the list highlight to '..' - the file is neither selected nor attached, with no message about the dropped filename. A natural second Enter then activates '..' and navigates UP a level. My scripted attaches repeatedly ended stranded in the wrong directory this way - a keyboard user pasting a known path cannot attach without re-finding the file by eye.

**Expected:** A full file path entered in the path bar should select/attach that file (standard file-dialog behavior), or at minimum highlight it in the list and explain.

**Repro:** Composer Attach -> Ctrl+L -> type /full/path/to/test-image.png -> Enter. Listing shows the parent dir with '..' highlighted; nothing attached; Enter again navigates up.

**Evidence:** `j3-p1-after-enter1.png`, `j3-cap4-sixth-ctrl-l.png`, `j3-cap4-sixth-stuck.png`

**Verifier note:** Code-confirmed with a literal TODO: Third_Party/textual_fspicker/base_dialog.py:526-531 — a file path entered in the Ctrl+L bar navigates to path.parent with '# TODO: Ideally, we would also select the file in the list'. Known to the code author, never filed (task-219 was a different FileOpen bug). Downgraded P2→P3: the bottom filename input DOES accept full/~-expanded paths (_confirm_single, enhanced_file_picker.py:1270-1296), so a keyboard workaround exists; affects power users of one picker surface.

#### j3-view-action-inert-preview-hides — The image message's 'View' action does nothing, and selecting the message hides its inline preview
*J3 Attachments · P3 · verdict NEW (medium confidence) · heuristic: affordance clarity / feedback*

**Observed:** Message actions (Copy, Edit, Save as..., regenerate, continue, rate, delete, View, Save Image) only appear after clicking the message - nothing signals messages are clickable. On an image message, clicking 'View' produced no visible result (no modal, no expansion) in repeated attempts; additionally, the moment the message enters its selected/actions state the inline pixel preview disappears, so the 'View' affordance sits next to an image that just vanished.

**Expected:** View should open an enlarged render (or visibly toggle something); selection should not hide the preview; message affordances should be hinted before click.

**Repro:** Open a conversation containing an image message. Click the message: action strip appears and the inline pixels vanish. Click 'View': nothing observable happens.

**Evidence:** `j3-79-view-modal.png`, `j3-77-click-img-label.png`, `j3-70-reopened-conv.png`

**Verifier note:** View is a silent 3-mode cycle (pixels→graphics→hidden, chat_screen.py:2680-2688) with NO feedback of the resulting mode; graphics mode needs a terminal graphics protocol that textual-serve/xterm.js cannot deliver, so the first click legitimately looks inert in the review harness (partly tool artifact), and the graphics-render-failure case has no fallback (only import failure falls back, console_transcript.py:903-914). The 'preview vanishes on selection' is most plausibly the same oversized-image-row/scroll defect as j3-transcript-giant-gap (row derivation keeps image rows when selected, console_transcript.py:738-770 — no hide mechanism exists). The pre-click discoverability sub-point overlaps known-open open-message-action-affordance. Net-new piece: zero state feedback on the View cycle. P2→P3.

#### j3-attach-button-morphs-to-icon — After staging, the labeled 'Attach' button becomes icon-only paperclip+check next to an unlabeled ✕
*J3 Attachments · P3 · verdict NEW (high confidence) · heuristic: recognition over recall / consistency*

**Observed:** Idle composer shows 'Send  Attach  Save'. Once anything is staged the middle button re-labels to '(paperclip)(check)' and an extra '✕' appears. The check-glyph reads as a status ('attached OK') rather than the action 'attach another', and nothing labels ✕ as 'clear all attachments'.

**Expected:** Keep the verb label ('Attach' or 'Attach more (1/5)') and give the clear control an explicit label or count-accurate tooltip.

**Repro:** Stage one file and compare the composer button row to its idle state.

**Evidence:** `j3-16-click-chip.png`, `j3-40-two-images-chip.png`

**Verifier note:** Code-confirmed: attach button relabels to '📎✓' with tooltip 'Attached: … Press to replace.' (console_composer_bar.py:1752-1753) and the ✕ tooltip is 'Remove the pending attachment.' (line 1883) — both tooltips are stale single-attachment-era copy now that staging appends up to 5 (task-217), which reinforces this as an unrecorded polish defect rather than settled design. P3 correct.

#### j3-no-multiline-composer-path — No discoverable way to compose a multiline draft: Enter and Shift+Enter both send, Ctrl+J does nothing, Help documents no alternative
*J3 Attachments · P3 · verdict NEW (medium confidence) · heuristic: keyboard reachability / consistency with terminal constraints*

**Observed:** Shift+Enter sent the draft ('line one' became a sent message - expected, since terminals deliver plain CR for Shift+Enter), Ctrl+J inserted nothing ('aaabbb' stayed one line), and F1 Help lists only 'Enter: send' with no multiline or attachment shortcuts at all. The Help overlay itself is a bare unformatted list.

**Expected:** Provide and document a newline chord that survives terminals (e.g. Ctrl+J) and mention paste behavior; Help should cover the composer's real capabilities (attach, paste-path, cap).

**Repro:** Type text, press Shift+Enter (sends), Ctrl+J (nothing). Press F1 and read the shortcut list.

**Evidence:** `j3-67-shift-enter.png`, `j3-74-ctrl-j.png`, `j3-75-help.png`

**Verifier note:** Nuanced: Shift+Enter→newline IS implemented (chat_screen.py:11194-11200) — the observed send is xterm.js delivering plain CR, the same harness class as the documented Alt limitation, so that sub-claim is a tool artifact. But the actionable gaps are real and unrecorded: no terminal-portable newline chord (ctrl+j falls through unhandled), and CONSOLE_WORKBENCH_SHORTCUTS (chat_screen.py:380-388) — the F1/help and footer source — lists only 'Enter send' with no newline, attach, or paste-behavior coverage. P3 correct.

#### j3-weak-focus-indicator-composer-buttons — Tab focus on composer buttons is a barely visible background shift
*J3 Attachments · P3 · verdict NEW (medium confidence) · heuristic: keyboard reachability / focus visibility*

**Observed:** Cell-attribute inspection during a Tab cycle from the input: Send focus changes bg only from #242F38 to #2D3840 (no bold/underline/inverse); Attach at least gains an underline. At terminal color depth the Send focus state is nearly indistinguishable, so keyboard users lose track of focus in the composer row.

**Expected:** Focused buttons need a high-contrast state (inverse video or accent bg) consistent across all three buttons.

**Repro:** Click the composer input, press Tab repeatedly, watch the Send/Attach/Save buttons; verified via xterm cell attributes (run10 log: bgs 2371384 -> 2963264, no inverse/bold).

**Evidence:** `j3-80-tab-cycle.png`

**Verifier note:** Plausible and consistent with the CSS: Settings and Library panes have explicit Button:focus rules (_agentic_terminal.tcss:3478/3723/3962) but there is no focus rule for the Console composer's destination-action-buttons, leaving Textual's subtle default background shift that the reviewer measured cell-by-cell (#242F38→#2D3840, no inverse/bold). Not covered by any ledger item. P3 correct.

#### j3-tab-tooltip-garbles-header — Tab-bar hover tooltip paints borderless text over the Console header, garbling both
*J3 Attachments · P3 · verdict NEW (high confidence) · heuristic: aesthetic and minimalist design*

**Observed:** With the pointer resting over the '1 Home' tab, its tooltip ('Open dashboard, notifications, and active work.') renders as bare text mixed into the Console header area, producing interleaved fragments like 'Open dashboard, notifications, and / active work. / control actions.' on screen.

**Expected:** Tooltips need an opaque bordered surface that fully covers (or avoids) underlying text.

**Repro:** Hover the mouse over the '1 Home' tab in the top bar and look at the Console header block beneath it.

**Evidence:** `j3-79-view-modal.png`

**Verifier note:** Confirmed in j3-79: the '1 Home' tab tooltip ('Open dashboard, notifications, and active work.') renders as bare borderless text over the Console header, leaving the interleaved fragment 'control actions.' — the Tooltip surface lacks an opaque bordered style in the app theme. Not a harness artifact (it is the app's own tooltip rendering) and not covered by any ledger item or task. P3 correct.

#### j4-interrupted-reply-unmarked-no-retry — A stopped reply persists as a mid-sentence fragment indistinguishable from a finished answer, with no visible retry affordance
*J4 Streaming · P3 · verdict NEW (high confidence) · heuristic: recognition over recall / error recovery*

**Observed:** After stopping, the partial assistant message renders exactly like a completed one: it just ends mid-sentence ('A' in run A; '...search through a potentially massive log file to find' in run B) with no 'stopped/partial' badge, no dimming, nothing. After restart the fragment still looks like a normal answer (j4-28). There is no Retry/Regenerate control anywhere in the transcript; the only hint that Regenerate/Continue exist is a text block buried in the Inspector ('Message actions: Copy, Edit, Save as..., Regenerate, Continue, Feedback, Delete / Keyboard: Tab/Shift+Tab cycle actions; Enter activates') describing an invisible message-selection keyboard flow.

**Expected:** An interrupted message should carry a persistent visible marker ('stopped by user - partial') and offer an in-place Continue/Regenerate affordance right on the fragment.

**Repro:** Stop a streaming reply -> inspect the partial message in the transcript, then reload the conversation from the rail -> fragment has no interruption marker; search the transcript for any retry control.

**Evidence:** `j4-28-click-partial-message.png`, `j4-38-after-tab-flip.png`, `j4-36-just-after-stop.png`

**Verifier note:** Partially overstated: mid-session a stopped message DOES render a ' [stopped]' suffix (console_transcript.py:93-94 appends '[{status}]' for streaming/stopped/failed) — run A never reached 'stopped' (the task-227 race above) and run B's tail was below the fold (j4-36 is scrolled up), so the journey likely never saw it. The genuinely new residue: stopped status is not persisted, so after restart the fragment is indistinguishable from a finished answer (mark_message_stopped flushes content only). 'No retry affordance' is not a gap — Regenerate/Continue exist via click-select action row (shipped, message-actions-save-as-note-e2e ledger item); its discoverability is already tracked as open-message-action-affordance (P3). Downgrade to P3: only the persistent-marker gap survives verification.

#### j4-no-new-content-indicator-when-scrolled-up — When scrolled up mid-stream, new content accumulates below the fold with no 'new content / jump to latest' indicator
*J4 Streaming · P3 · verdict NEW (high confidence) · heuristic: visibility of system status*

**Observed:** PageUp mid-stream did scroll the view up and the position was respected (no yank-back - good). But the stream kept growing below the fold silently: no unread badge, no 'jump to latest' affordance, no signal when generation finished or was stopped. 20 seconds after Stop the viewport still sat mid-message showing a sentence cut by the pane edge (j4-37), with the user unable to tell from this view whether the reply was still streaming, finished, or stopped.

**Expected:** While detached from the bottom during streaming, show a persistent indicator ('▼ streaming below / jump to latest') that also reflects completion or interruption.

**Repro:** During a streaming reply press PageUp (after clicking transcript text once) -> stay scrolled up -> no indicator of ongoing streaming below; stop the run -> still no signal from the scrolled position.

**Evidence:** `j4-36-just-after-stop.png`, `j4-37-stop-plus-20s.png`

**Verifier note:** Real gap but an enhancement on top of deliberate task-298 behavior (no-yank while detached is the shipped contract; nothing in the ledger promises an unread/'jump to latest' pill). Standard chat affordance absent; user can End/PageDown back. Downgrade P2→P3: nothing misbehaves, information is merely unavailable from the scrolled position.

#### j4-raw-markdown-headings-in-transcript — Assistant markdown headings render as literal '###'/'####' text in the transcript
*J4 Streaming · P3 · verdict NEW (high confidence) · heuristic: aesthetic & minimalist design*

**Observed:** Section headings requested in the prompt arrive as markdown and are displayed raw: '### Understanding SQLite Write-Ahead Logging (WAL) Mode', '#### The Checkpointing Process', etc., in both the live-streamed assistant bubble and the persisted transcript. Long replies with many sections read as noisy plain text; other inline markdown (backticks) is also shown literally.

**Expected:** Render headings with terminal-appropriate emphasis (bold/underline/color) or strip the marker characters.

**Repro:** Ask for any answer 'with section headings' -> observe literal #/## characters in the transcript.

**Evidence:** `j4-10-reply1-complete.png`, `j4-33-streaming3.png`, `j4-38-after-tab-flip.png`

**Verifier note:** Confirmed in j4-33/38 (literal ####). The Console transcript deliberately renders markup-off plain text (Content.assemble, never markup-parsed — documented in console_agent_bridge.py's marker docstring and the transcript-visual ledger item's 'plain text unchanged'), but that is an injection-safety implementation choice; no ledger/backlog decision says assistant markdown must render raw, and legacy chat renders markdown. Legit rendering gap, P3 polish.

#### j5-model-chip-degenerate-select — Clicking the 'Model:' header chip turns it into an empty select and floats its option over the action row
*J5 Settings · P3 · verdict NEW (medium confidence) · heuristic: affordance clarity; degradation of layout*

**Observed:** The header chips look like static status text, but clicking 'Model: local-gemma' replaces the chip with an empty blue-outlined box and renders a floating fragment 'Model: local-gemma' below it, overlapping and truncating the 'Settings' button ('Settin') and the rail header. The dropdown offers only the literal chip text as its single option; there is no model list, title, or hint.

**Expected:** Either chips are static status (then they should not react at all), or they are controls (then they need a visible affordance and a well-formed popover that doesn't occlude the action strip). Half-interactive chips that open a degenerate one-item dropdown confuse both readings.

**Repro:** 1. On idle Console, click the 'Model: local-gemma' chip in the status row. 2. Chip becomes an empty outlined box; 'Model: local-gemma' floats below it over 'Settings' (truncated to 'Settin'). 3. Escape restores.

**Evidence:** `j5-41-chip-click-clean.png`, `j5-01-baseline.png`

**Verifier note:** Mechanics misread but a real render defect remains. There is no Select and no dropdown: ConsoleChip is a focusable Static (console_control_bar.py:80-89) whose click focuses it (focus CSS deliberately lifts the 22-cell ellipsis so the full label shows) and whose tooltip carries the full label — the 'floating fragment' is the tooltip. The genuine defect visible in j5-41: the FOCUSED chip renders as an empty outlined box (label not visible at all), defeating the documented focus-expand-to-read-full-label purpose, while the tooltip overlaps and truncates the 'Settings' action button. Not covered by any ledger item (counter-chips-dim covers styling only). Downgraded to P3: Escape recovers, nothing functional lost.

#### j6-rail-crush-letter-wrap — Small widths crush the rail into letter-per-line wraps and ellipsis-free truncation while the rail keeps width priority
*J6 Keyboard-only + small terminal · P3 · verdict NEW (high confidence) · heuristic: Degradation under resize / aesthetic-minimalist design*

**Observed:** At 125x38 and 97x30, the rail renders 'Workspace Default' as 'Def / aul / t' stacked one fragment per line, truncates 'New conversation' to 'New conversati' with no ellipsis, and 'Chat 1 - Chats' to 'Chat 1 -', while the transcript is squeezed to ~57 columns. The inspector also auto-expands at 900x620 (it is collapsed at 2050px), further shrinking the transcript at exactly the sizes where space is scarcest.

**Expected:** Below a min width the rail collapses to its handle (it already has a '◂' collapse affordance) or truncates whole tokens with ellipsis; the transcript/composer get width priority.

**Repro:** Open the Console at 900x620 (125x38 cells) and read the rail Workspace row: 'Def/aul/t' letter stack; compare inspector width vs 2050x1240 baseline.

**Evidence:** `j6-b01-900x620.png`, `j6-b05-cold-700x480.png`

**Verifier note:** Evidence j6-b05-cold-700x480.png confirms 'Def/aul/t' letter-stack and 'New conversati' ellipsis-free truncation. Code: left rail min_width 24 / right rail min-width 34 / main column min-width 56-60 are hard floors with no auto-collapse breakpoint (chat_screen.py:7062/7366/7383); rails already have collapse handles but nothing triggers them on narrow widths. Not covered by ledger (rail phases 2-4 pending item is about IA, not width degradation). P3 appropriate.

#### j6-scope-raw-uuid — Rail 'Scope' field displays the raw conversation UUID wrapped mid-token
*J6 Keyboard-only + small terminal · P3 · verdict NEW (high confidence) · heuristic: Match between system and real world / recognition over recall*

**Observed:** After activating a saved conversation, the rail Session section shows 'Scope 04a57425-d828-4b13-bacd-9e599 617b80f' (and later '7d76a20f-…') — an internal ID, meaningless to users, wrapped across two lines mid-hex. At baseline the field is simply empty.

**Expected:** Scope shows a human-readable label (conversation title / 'This conversation') or is hidden; internal IDs stay in tooltips or debug surfaces.

**Repro:** Ctrl+K, activate any saved conversation, look at the rail Session section 'Scope' row.

**Evidence:** `j6-a18-select-user.png`, `j6-a32-post-shift-enter.png`

**Verifier note:** Code-confirmed: scope_label = str(current_conversation or '') where current_conversation is _current_console_conversation_id() — the raw conversation UUID — rendered as the Session section's Scope value row (Workspaces/display_state.py:282, console_workspace_context.py:581-588, chat_screen.py:3352-3356). Not among the task-186 micro-polish fixes nor the open-rail-model-line-and-footer-nits ledger list. NEW, P3 correct.

#### j6-selection-styling-inconsistent-tool-msgs — Selected tool messages get no underline/panel treatment unlike user/assistant messages
*J6 Keyboard-only + small terminal · P3 · verdict NEW (high confidence) · heuristic: Consistency*

**Observed:** Selecting a user or assistant message shows underline + lighter panel + action row (clearly visible). Selecting a Tool message ('⤷ spawned sub-agent…') shows only the action row; cell scans found no underline/bold/bg change on the message text itself, so with the action row near the fold the selection reads as invisible.

**Expected:** All selectable message kinds share the same selected treatment.

**Repro:** In a conversation whose latest replies include Tool rows, press k repeatedly and compare the selected styling of a Tool message vs a User message.

**Evidence:** `j6-a17-k2-select.png`, `j6-a18-select-user.png`, `j6-a23-after-r.png`

**Verifier note:** Code-confirmed CSS-ordering defect: .console-transcript-message-tool (color $ds-text-muted; text-style dim italic) is declared AFTER .console-transcript-message-selected (bold underline, $ds-focus-fg) at equal specificity (_agentic_terminal.tcss:2800-2810), so a selected tool row loses the underline/fg treatment and keeps only the near-invisible bg delta. transcript-visual and decision-selected-message-accent-border ledger items cover the selected treatment's design, not this per-role inconsistency. NEW, P3 correct.

#### j6-boot-toast-occludes-composer-actions — Model-catalog boot toast covers the composer's Send/Attach/Save buttons for ~8s
*J6 Keyboard-only + small terminal · P3 · verdict NEW (high confidence) · heuristic: Visibility of system status (toast placement)*

**Observed:** On boot the 'Model catalog — Model lists updated — OpenRouter: 333 new cached' toast renders over the right end of the composer box (rows 72-75), hiding the Send/Attach/Save buttons and part of the input area until it auto-dismisses (~8s). The copy-confirmation toast also lands over the composer.

**Expected:** Toasts anchor to a corner that does not overlap the primary input affordances, or the composer keeps a reserved lane.

**Repro:** Cold-start the app with model catalog refresh enabled; compare row 73's right side during the first 8 seconds vs after dismissal.

**Evidence:** `j6-a01-baseline.png`, `j6-a02-toast-state.png`, `j6-a18-after-c.png`

**Verifier note:** The 'Model catalog — Model lists updated' toast is from the brand-new model-catalog auto-refresh feature (app.py:7077, LLM_Provider_Catalog/model_auto_refresh.py:70; spec dated 2026-07-17 — post-dates the entire ledger) and uses Textual's default bottom-right toast anchoring over the composer. No ledger item settles toast placement. NEW, P3 correct.

## Findings screened out as already-known or invalid

Reported live but excluded from the lists above after verification against the prior-art ledger (114 items from the five shipped UX phase plans, four UAT evidence sets, and the open backlog):

- **j1-gate-dead-inputs** (KNOWN, setup-modal-guard): First-run gate silently swallows nearly every advertised input (typing, Enter, F1, Ctrl+P, Esc)
- **j1-setup-focus-invisible** (INVALID): Keyboard focus on the 'Set up provider' button is completely invisible
- **j1-setup-teleports-to-settings** (KNOWN, open-providers-layout-credentials-buried): 'Set up provider' teleports to the full expert Settings screen with OpenAI defaults; checklist context and return path are lost
- **j1-boot-catalog-toast-noise** (KNOWN, task-301): First boot with zero providers pops a 'Model catalog — Model lists updated — OpenRouter: 333 new cached' toast
- **j1-send-echo-latency** (KNOWN, task-193): First sent message does not echo into the transcript immediately; 0.5s after Enter the pane still says 'No messages yet.'
- **j2-switcher-no-selection-highlight** (KNOWN, ctrl-k-switcher): Ctrl+K switcher has no visible selected row; ArrowDown gives no feedback; Enter opens an invisible choice
- **j2-switch-settle-mismatched-state** (KNOWN, tick-ttl-2s-gating): Session switching settles in visible stages with no busy indicator: header, transcript, tab strip and rail disagree about the active conversation for over a second
- **j2-toast-covers-composer-actions** (INVALID): Model-catalog toast pops over the composer's Send/Attach/Save buttons right at boot
- **j3-enter-hijacked-by-unfurl-flow** (KNOWN, paste-unfurl-verified (+ console-attachments-phase1)): With an inserted text-file pill in the draft, Enter no longer sends: it silently steps a hidden 'Unfurl?' state machine (3 Enters to send)
- **j3-no-per-item-remove-clear-all-silent** (KNOWN, backlog/tasks/task-217 AC#1 (+ MEMORY console-multi-attach-217)): Multi-attachment staging collapses to '(clip) 5 files' with no item list and only a silent clear-ALL as removal
- **j3-filename-lost-after-reload** (KNOWN, backlog/tasks/task-217 AC#3 (+ no-bytes-in-screen-state read contract)): Reopened conversations show 'image/png - 341 B' instead of the attached file's name
- **j4-thinking-gap-no-progress** (KNOWN, task-193): 40-100s pre-first-token gap shows zero progress indication - just an empty 'Assistant' bubble
- **j4-tokens-footer-never-updates** (KNOWN, .impeccable/critique/2026-07-17 shell-chrome critique (memory: shell-chrome-critique-2026-07-17)): Footer 'Tokens: --' never updates even though the Inspector tracks live context usage
- **j4-tab-title-truncated-to-7-chars** (KNOWN, .impeccable/critique/2026-07-17 shell-chrome critique (memory: shell-chrome-critique-2026-07-17)): Session tab label truncates the conversation title to '● Write a' despite a nearly empty tab strip
- **j5-settings-button-global-detour** (INVALID): Console 'Settings' button leaves the screen for global Settings; session settings hide behind rail 'Configure'
- **j5-value-provenance-invisible** (KNOWN, open-console-settings-session-scope): No surface indicates whether a settings value is a config default or a session override
- **j5-ctrlp-dead-at-launch** (INVALID): Ctrl+P (advertised palette shortcut) does nothing until the user first clicks a widget
- **j5-enum-fields-free-text** (KNOWN, task-178 / open-console-settings-session-scope): Enum settings (Reasoning/Summary/Verbosity/Thinking) are free-text inputs; invalid values accepted until Save
- **j5-provider-naming-five-variants** (KNOWN, task-194 (+ task-191, open-rail-model-line-and-footer-nits)): Same provider spelled five different ways across the Console's settings surfaces
- **j5-toast-covers-composer** (KNOWN, task-301): Boot-time model-catalog toast covers the composer's Send/Attach/Save buttons
- **j5-streaming-toggle-word-button** (KNOWN, settings-modal-session-scope (task-178)): Streaming control is a bare word-button 'On' — state vs action ambiguous
- **j5-escape-discards-silently** (INVALID): Escape closes the settings modal and silently discards edits
- **j6-inspector-unreachable-keyboard** (KNOWN, task-103): Inspector pane appears unreachable by keyboard (excluded from F6 cycle; no discoverable Tab path)

## What worked well

**J1 New-user cold start**
- Get-started card content is genuinely good: a 3-step checklist with state glyphs (done/active/pending), honest gating copy, and one clear primary action — the next action is obvious for mouse users.
- Settings dirty-state handling: 'State: Unsaved changes | Save or Revert before leaving this category' plus a '*' marker on the category in the sidebar — clear, persistent status visibility.
- Focused-field guide panel in Settings (Purpose / Saved as: chat_defaults.model / 'Validation: model name is required before provider-backed generation can run') is excellent contextual help that updates per focused control.
- Discovery failure for an unreachable host is actionable: 'Model discovery request failed. Check the endpoint URL, server availability, and credentials.'
- Post-save Test Provider runs a staged live probe with visible progression: 'endpoint probe: checking' -> 'endpoint reachable' within ~1s.
- Discovery against a correct /v1 endpoint is fast (~0.8s) and saving confirms explicitly ('Saved 1 discovered model(s) to Llama_cpp.'); the Save action then flips state to 'State: Shared with Console' and 'Provider settings saved.'
- After setup, returning to Console shows the gate fully replaced by the real workbench with a status strip that correctly reflects Provider: Llama_cpp / Model: gemma-4-26B-… / 'Ready' — readiness state is consistent end-to-end.
- First send worked first try once configured: clean User/Assistant transcript blocks, and the session tab auto-titled itself from the first message with a run indicator dot.
- Settings buttons (Test Provider, Discover models) do have a visible focus treatment (underline) when tabbed to — keyboard focus is legible there, unlike on the gate card.
- Provider select list is comprehensive (10 cloud + 10 local engines) and switching provider does eventually reconfigure every dependent field correctly, including 'API key source: not required for this provider' for llama.cpp.

**J2 Returning power user**
- Resume is trustworthy: rail click or Ctrl+K loads the full transcript bottom-anchored at the latest exchange, and all content persisted across app restarts (including a reply that finished streaming just before the previous process closed).
- Ctrl+K switcher fundamentals: opens fast from normal focus (0.32s even on a heavily loaded machine), works while the composer is focused WITHOUT destroying the draft ('my important draft' survived open+Escape intact), Esc dismisses cleanly, and fuzzy matching ('refac', 'embed') finds conversations the rail hides.
- Footer permanently advertises the power-user paths (Ctrl+K switch session, Ctrl+T new tab, F6 pane cycling, Ctrl+P palette) — good discoverability of keyboard flows.
- Once settled, the active conversation is identifiable in three consistent places: rail '▸ + active session - now' marker, underlined tab, and the transcript header carrying the full untruncated title.
- Rail state vocabulary distinguishes 'active session' / 'open session' / 'saved chat' with age labels that are accurate and tick over time (7m → 15m → 21m → 25m verified against seed time).
- Each resumed conversation opens in its own tab, preserving the draft tab and other open sessions — a genuinely good multi-session model for power users; busy tab shows a ● dot and the composer gains a Stop button during generation.
- Rename modal itself is well-built: input pre-focused with current title, Enter submits, updates tab and header instantly (the persistence gap is the P1 above, not the modal UX).
- Message action row (Copy / Edit / Save as... / regenerate / continue / feedback / delete) appears on click-to-select, and the Save as... modal is exemplary: names the selected role, quotes an excerpt, offers clear Chatbook/Note/Media/Prompt destinations plus Close.
- Rail collapse (◂) to a thin labelled strip with a visible re-expand handle works and frees width for the transcript.

**J3 Attachments**
- Pre-send capability gate is excellent: 'Console send blocked: local-gemma can't accept images. Remove the attachment, switch to a vision model, or mark this model as vision-capable under [model_capabilities.models]...' - specific, actionable, and it preserves the draft and staged files for recovery (j3-51-image-only-sent.png).
- Path-paste interception works as designed: pasting an absolute image path into the composer auto-attaches the file ('img3.png attached' toast) instead of leaving raw path text, while plain text pastes normally (j3-49-path-pasted.png, j3-50-plain-pasted.png).
- The file picker has strong keyboard affordances surfaced in its footer (Ctrl+B Bookmarks, Ctrl+R Recent, Ctrl+F Search, Ctrl+L Path, 1-9 Jump, Enter Open, Esc Cancel) and it remembers the last directory across re-opens and even across app sessions (j3-03-attach-picker-initial.png).
- A single staged image shows a clear chip with filename and exact size ('test-image.png - 341 B'), and the sent message representation is good: role header, text, '(img) test-image.png', plus an inline half-block pixel preview scaled sensibly for a small source image (j3-61-sent-with-image.png).
- The 5-item cap message wording is precise and quantified: 'Attachment limit reached (5 per message).' (j3-31-sixth-image.png).
- The Attach button is keyboard-reachable (input -> Tab -> Tab) and the message action strip includes a plain-text 'Guide' legend explaining the icon actions (j3-79-view-modal.png).
- Conversations are auto-titled from the first message and the failure state propagates to the Inspector rail ('failed') (j3-63-response-final.png).

**J4 Streaming**
- Stop affordance appears immediately: on send, the composer bar swaps in an amber 'Stop' button (Send greys out) and reverts on completion - clear mid-run interruptibility affordance in the right place (j4-04b, j4-33; cell-attrs confirmed distinct amber background bg=0x4A3A20).
- Per-message '[streaming]' suffix on the growing assistant bubble is a truthful, always-visible in-transcript state marker while tokens arrive (j4-10, j4-33).
- Session tab shows an activity dot (● prefix) while its run is active (j4-04b).
- Scroll position, once achieved (keyboard PageUp mid-stream), is respected: the stream continues below without yanking the viewport back to the bottom, so reading history during streaming is possible in principle (j4-36, j4-37 - stable across 20s).
- Idle scrollback with the mouse wheel works smoothly on overflowing transcripts (j4-30a).
- Conversations persist and are recoverable across app restarts: the rail lists saved chats with human age labels ('saved chat - 7m'), and the Inspector surfaces 'Resume state: restored from <id>' (j4-25, j4-38).
- The Inspector panel is a genuinely useful consolidation: run recipe, tools/MCP readiness, approvals, selected-message action cheat-sheet, and live context token accounting ('Context: 1,694/4,096' updating during the stream) (j4-33, j4-38).
- Message echo and send flow are usually fast (~1s) with clean User/Assistant separators and comfortable line lengths at 2050x1240 (j4-31b).
- Composer state management around sends is correct: text clears on Enter, placeholder returns, and drafts are not lost to the run lifecycle (j4-17-sent2).

**J5 Settings**
- The Console Settings modal states persistence semantics up front: 'Save applies to this session only. Save as default also writes provider + streaming defaults to config.' — explicit scope disclosure is rare and valuable (even though the temperature round-trip behind it is broken, see finding).
- Valid Save behaves correctly end-to-end: modal closes, the rail summary updates immediately (Temperature 0.60 -> 0.70), and the Inspector's Session Settings reflect the same values ('Sampling: T 0.70', 'Streaming: off') — values agree across rail/modal/Inspector once changed.
- Save-time validation messages are specific and teach the valid values: 'Temperature is required.', 'Reasoning effort must be one of none, minimal, low, medium, high, or xhigh.' (salience/placement is the problem, not the copy).
- Numeric inputs filter alphabetic characters at entry — typing 'abc' into Temperature cannot corrupt the field.
- 'Discover models' queries the live Base URL, reports 'Found 1 model at http://127.0.0.1:9099.', converts the free-text Model input into a dropdown of actually-served models, and its tooltip explains the mechanism ('List models served at the Base URL (/v1/models)').
- The session system-prompt editor (via 'System: none') is well-scoped: 'Applies to this session.', optional Name + 'Save to Library' checkbox, Clear/Cancel/Apply buttons.
- Provider readiness is contextualized in the modal header: 'llama_cpp is ready. No API key is required.' — answers the credential question before the user asks.
- The command palette (once open) surfaces 'Console: Change model… — Quick provider/model/temperature switch (Alt+M)', documenting the keyboard shortcut alongside the mouse path.
- The collapsed Inspector expands to a genuinely useful session-settings readout (endpoint, credential requirement, context budget, sampling line, streaming, persona) — the most complete single view of effective session state.

**J6 Keyboard-only + small terminal**
- Footer hint bar is always visible and accurate for the core loop (F6/Shift+F6, F1, Enter send, Ctrl+K, Ctrl+T, Ctrl+P) - good zero-recall entry point for keyboard use.
- Transcript and composer focus stops paint an unmistakable accent (#0178D4) pane border - where the pattern is applied, focus visibility is excellent.
- Ctrl+K switcher: search input auto-focused with visible block cursor and accent border; Enter-activates-top-result supports a fast type-and-enter flow; F2 rename is offered.
- Picking a conversation lands focus in the composer (verified cursor + accent border) - ideal 'ready to type' landing; Ctrl+T new tab also focuses the composer immediately and typing works with no extra step.
- Send path feedback is good: user message appears within one tick, reply streams progressively, a Stop button appears during streaming, and the tab shows a busy dot then renames to the conversation title.
- Transcript j/k selection auto-scrolls to keep the selected message and its action row in view; selected user/assistant messages get a clear underline + panel + action row with an icon-meaning guide line.
- Escape layering is coherent and safe: dismisses modals (edit modal cancel preserved the transcript selection), and a screen-level Escape returns focus to the composer from anywhere.
- Copy action confirms with a toast even in a headless environment.
- Command palette (Ctrl+P) is searchable, categorized, lists Console commands with their shortcut mnemonics (Ctrl+K, Alt+M, Esc), and arrow keys work there.
- Disabled controls communicate why: 'Switch' workspace chip is dimmed with the caption 'Add another workspace before switching.'; 'Save Chatbook' renders visibly disabled.
- Moderate resize handling is solid: 125x38 cells keeps all core surfaces usable, and growing the viewport restores the full layout correctly (panes, composer, footer).

## Coverage limits (blocked)

- (j1) Alt-key shortcuts could not be tested: xterm.js in this harness does not deliver Alt+<key> to the app (per harness README), so any Alt-based bindings on the gate or Console are unverified.
- (j1) Mid-generation feedback (streaming indicator/'thinking' status) for the first reply was not captured: the local model answered between polling intervals, so I cannot judge in-flight status visibility for slow replies.
- (j1) Exact latency of the 'Set up provider' -> Settings transition was observed once (>0.9s, <~4s) but not re-measured under instrumentation, so the number in j1-setup-teleports-to-settings is a bound, not a measurement.
- (j1) F6/Shift+F6, Ctrl+K and Ctrl+T on the gate screen were not individually tested (F1, Enter, Ctrl+P, Esc and plain typing were); the dead-inputs finding is scoped to the tested keys.
- (j1) Small-terminal resize degradation was not tested (journey spec fixed the viewport at 2050x1240).
- (j1) Whether the setup card ever offers the 'detected local server' one-click action (ConsoleDetectedServerAction exists in source) could not be observed live — no such affordance appeared with the llama-server on the non-default port 9099.
- (j2) Alt-based shortcuts (Alt+1..9 tab jump, Alt+M model popover, Alt+V paste image) — xterm.js in the harness does not deliver Alt+key to the app, so they are unverifiable here.
- (j2) Toast duration/dismissal interaction: the model-catalog toast only appeared on the very first app boot (catalog cached afterwards) and was gone before it could be interacted with; only its placement was captured.
- (j2) Scroll-position behavior when resuming a conversation taller than the viewport (all seeded transcripts fit on one screen; only bottom-anchoring was observable) and any 'load older messages' affordance.
- (j2) Absolute latency measurements are confounded: machine load average was 12-15 during testing (five parallel review sessions + pytest runs), so switch/settle times are reported as ordering/staging evidence, not as timings; one full app session also booted to a permanently black screen and had to be restarted (known harness-era flakiness, not counted as a finding).
- (j2) Whether the rail 'Chats - open session - now' rows offer distinct actions from saved rows, and what the workspace 'Switch/New' flow does with multiple workspaces (single-workspace home).
- (j2) Tab-strip overflow behavior with many tabs (only 3 tabs were ever open) and the second 'New tab' affordance next to the strip vs the control-bar 'New tab' button (apparent duplication, not exercised).
- (j3) Alt+V clipboard-image grab could not be tested: xterm.js does not deliver Alt+key combinations to the app (documented harness limitation).
- (j3) True drag-and-drop could not be tested: terminals convert drops to path pastes; the equivalent path-paste interception was tested instead.
- (j3) The boot-time 'Model catalog - Model lists updated...' toast never fired in any of my ~9 provider-variant sessions, so its timing/placement/dismissal could not be evaluated.
- (j3) The 'hidden' and 'graphics' image render modes (chat.images config) were not exercised - only the default pixels mode; the View action might behave differently there.
- (j3) A real vision reply could not be obtained (the only provider is a text-only llama.cpp model; forcing vision produced the documented 500 path instead), so post-reply attachment presentation with a successful assistant turn is unverified.
- (j4) Could not distinguish whether mid-stream wheel-up is suppressed outright or scrolls-then-instantly-yanks: buffer polling at 400ms only shows the net outcome (viewport still pinned). Either mechanism produces the reported UX.
- (j4) Exact latency of the Inspector toggle could not be measured: no visible response 0.7s after the click (j4-32); the panel was open in the next captured state ~28s later with no transition observable in between.
- (j4) The keyboard message-selection flow described in the Inspector (Tab/Shift+Tab cycle message actions, Enter activates, incl. Regenerate/Continue) was discovered late and not exercised end-to-end; whether Continue can actually resume a stopped fragment is unverified.
- (j4) Alt-based shortcuts are unverifiable in this harness (xterm.js does not deliver Alt+key to the app).
- (j4) Round-1 first-token timing logs were lost to output truncation; the ~100s figure for the sub-agent run is reconstructed from screenshot timestamps (send 13:14:2x -> first tool text ~13:16), so it is approximate.
- (j4) Home/End behavior on a focused transcript could not be fully mapped (Home never scrolled even after mouse-click focus; only PageUp/ArrowUp were verified to work).
- (j4) The 'j4-35-midstream-3s-later' capture was skipped by design because the mid-stream wheel never moved the view (its branch only ran on successful wheel scroll).
- (j5) Alt+M itself could not be pressed: xterm.js in the capture harness does not deliver Alt+<key> chords to the app. The popover was reached and evaluated via its palette command instead ('Console: Change model…').
- (j5) Model-list readability at scale (raw keys vs display names for large catalogs): the local llama-server serves exactly one model, and no cloud provider was credentialed in the isolated home, so 'Discover models' and 'Search all models…' could only ever show one entry; long-list presentation, grouping and display-name quality remain untested.
- (j5) Model-catalog toast dismissal timing: the catalog refresh fires once per home, so the toast appeared only in the first session (before timing instrumentation); its auto-dismiss duration and whether it is click-dismissable could not be measured on later sessions.
- (j5) Harness keystroke fidelity caveat: the web capture path intermittently dropped the first character typed after a mouse click (e.g. '1.5' arriving as '.5'). All values were verified in the terminal buffer before each Save, and per-character verified typing was used for decisive runs, so no finding rests on a mistyped value — but sub-second input-latency judgments were out of scope.
- (j6) Alt-based shortcuts (Alt+M model popover, Alt+V paste image, Alt+1..9 tab jumps) - xterm.js does not deliver Alt chords to the app; unverifiable in this harness.
- (j6) True Shift+Enter soft-newline behavior - the web terminal delivers Shift+Enter as plain Enter (send); the code path (chat_screen.py handler inserting \n) could not be exercised. In any web-served deployment Shift+Enter will always send, which is itself worth noting.
- (j6) Clipboard contents after Copy - only the confirmation toast could be verified in the headless browser.
- (j6) Inspector pane interaction beyond reachability (expand '▸', Configure button) - no keyboard path found and mouse use was out of scope for Part A.
- (j6) OS-level terminal-emulator resize - only browser viewport resize (textual-serve) could be tested; a native terminal's SIGWINCH path may behave differently.
- (j6) Initial DOM-focus quirk: keystrokes are ignored until one click lands on the terminal element (textual-serve/xterm.js focus behavior, not the app's fault) - documented here because it consumed the journey's single allowed click.
