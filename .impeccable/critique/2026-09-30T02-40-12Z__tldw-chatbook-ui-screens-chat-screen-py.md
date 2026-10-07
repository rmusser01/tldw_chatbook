---
target: Console screen (full) on dev 64579cce2c — NN/g + HCI comprehensive review
total_score: 15
max_score: 40
na_heuristics: 
p0_count: 8
p1_count: 42
timestamp: 2026-09-30T02-40-12Z
slug: tldw-chatbook-ui-screens-chat-screen-py
---
Method: dual-agent, multi-reviewer. Assessment A ran 12 reviewers: live journeys J1–J8 (a6220b14, a95173d5, a8adfca6, ad923b07, adadd083, ae54736d, ad0cf5e2, a7a3c69f) and code lenses L1–L4 (aab020e7, ac262ab9, a26e65fd, a9e1e6a3), plus 5 gap reviewers. Assessment B ran in isolation: B1 detector (a3d29779) and B2 measurement census (ae8663db). Each finding was checked against prior art and then put through refute-first verification (workflow `wf_9b1940b7-da2`, 546 agents).

# Console UX review — NN/g heuristics + HCI, latest dev (2026-09-29)

**Target:** the Console screen (`UI/Screens/chat_screen.py`, `UI/Console_Modules/*`, `Widgets/Console/*`) at **`origin/dev @ 64579cce2c`**. The work ran in a clean detached worktree, not the video-generation branch checked out in the main repo.

**How:**
- Every live run used a throwaway profile (disposable HOME, XDG, config and data dir); the real config stayed byte-identical.
- A real LLM (Anthropic claude-haiku-4-5) sat behind a review-only proxy, which injected failures (504, 429, 401, slow, drip, cut, stall).
- Terminal sizes: 235×52, 211×44, 160×45, 120×40, 100×30 and 80×24, plus light themes and ASCII mode.
- Measurements were parsed from the colour codes (SGR) in tmux captures.

**Findings:**
- 300 raw findings consolidated to 184, then 93 more from gap reviewers.
- Of the 277 verified, **275 survived**: 8 P0, 42 P1, 173 P2, 52 P3.
- Refuted: 1. Could not verify: 1.
- Prior art:
  - 157 NEW
  - 53 KNOWN_PARTIAL
  - 34 REGRESSION (a Done task claims it is fixed)
  - 27 BY_DESIGN (a recorded decision, still judged harmful)
  - 3 KNOWN_OPEN
  - 1 IN_FLIGHT_PR

Full ledger: `findings.md` / `findings.json` in this folder.

**Out of scope:** raw speed and backend efficiency, which the in-flight PERF-01..30 PR series covers. Only perceived feedback problems are included.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 1 | Run state is computed in about 9 places that disagree. Healthy runs read "Run: Blocked / Provider setup needed". The header says "Ready" after a 401 or an emergency stop. A false "completed while hidden" toast fires on every turn. Enter is followed by 1–10 s of "No messages yet." Replies are not streamed. |
| 2 | Match System / Real World | 2 | Everyday labels are mostly plain. Recovery copy leaks internals: "delivery status is unknown on the source device", "Saved active leaf lineage…", "STTS Settings". "Agent blocked" actually means the Library tool is off. |
| 3 | User Control and Freedom | 1 | Stop is clipped off-screen during every run at every width, with no key, palette entry or /stop. Tab ✕ does nothing. Delete takes the whole later subtree with no undo. Ctrl+Q discards drafts and Temporary chats. |
| 4 | Consistency and Standards | 1 | The chat object has 5 names. There are 4 state-glyph tables and about 10 modal action-row orders. Single-letter keys change meaning between the transcript and Trace. Styling has two tiers: tokenized .tcss, and Python CSS with 52 named-colour literals. |
| 5 | Error Prevention | 1 | Enter on the focused Menu button sends a paid request. 'y' typed from a chip opens Trace. A prompt typed just after a tab click runs in the previous tab. Shell commands typed under the HOST TERMINAL banner go to the model. |
| 6 | Recognition Rather Than Recall | 2 | The Guide line and slash popup are good. Rewind, Trace and side chat need recall. There are 3 glyph legends. Cost shows only on hover. |
| 7 | Flexibility and Efficiency | 2 | There are many accelerators. Some advertised chords never arrive in real terminals (Alt+1..9; Shift+F3 arrives as f15). There is no stop, close-tab or next-tab key. The palette has 18 actions. |
| 8 | Aesthetic and Minimalist Design | 2 | The palette is calm, but the default screen has 57–75 controls. The chat title appears up to 6×. Inspect runs about 90 rows. At 80×24, chrome takes 12 of 24 rows. |
| 9 | Error Recovery | 1 | A 429 or 5xx is reported as "Trace capture blocked — provider not contacted". Cancel refuses. The tool-schema 400 blames the model. A commit failure says "finish provider setup". |
| 10 | Help and Documentation | 2 | F1 is grouped and partly live but makes false claims: that Esc returns to the composer, and that the palette lists every Alt action. The guide documents Stop, Mic, F3 and F9 that don't exist. |
| **Total** | | **15/40** | **Poor.** The foundations score better than the surface. |

## Design Specificity Verdict

**LLM assessment.** This was authored for this product; it is not a chatbot skin. Several things make sense only for a local-first operator workbench:
- the "What happens if I send now?" authority card (Where / Scope / Run / Sources / Approvals);
- Owner / Problem / Impact recovery callouts;
- approval cards whose scope line changes with the decision;
- the Fork dialog's "What is not copied";
- Temporary chats;
- the dense one-row form convention;
- a glyph registry with ASCII fallbacks.

The .tcss tier follows DESIGN.md: 581 of 595 colour references are `$ds-*`. The authorship leaks at the edges, and there it reads as generic:
- Python-embedded DEFAULT_CSS has 52 named-colour literals, including 20 × `border: tall gray; background: black`, plus 25 side stripes that DESIGN.md bans.
- Emoji and ASCII-art buttons (`--->`, ♻, 🔊, 💬) are the chatbot look PRODUCT.md rejects.
- In the default theme, accent equals warning.
- Only 5 of 59 focus rules meet the DESIGN.md focus contract, and that contract names `$accent` while the focus aliases resolve to `$primary`.

**Deterministic scan.** The impeccable detector is blind here:
- `.py` and `.md` are outside its scannable extensions, so directory targets scanned 0 files and still exited 0.
- Its one hit, `rgb(245,245,245)` at `_console.tcss:2645`, is a false positive; DESIGN.md's Legible Disabled Rule documents that value.

The real instruments were B1's static grep audit and B2's SGR census. Both independently confirmed A's findings:
- the black modal shells;
- 13 banned side stripes;
- the focus-contract gap;
- the placeholder at 2.56:1 when focused, versus 6.77:1 unfocused;
- the switch knob at 1.42:1;
- "Redir" clipped at 235, 120 and 80 columns;
- 'y' from Settings opening Trace;
- the status strip's natural width of about 251 columns with no overflow cue;
- 1,958 unpainted cells.

B also found three modals 28–30 rows tall with no max-height guard, which will clip at 80×24.

**Visual overlays:** not applicable. xterm.js draws to a canvas, so there are no DOM cells to inject into. No overlay was shown.

## Overall Impression

You're right that it is calmer and better organised; the concepts are the right ones. What holds it back is not missing features. **Each region computes its own version of the truth.** Run state, readiness, attention, staged sources, key hints, labels and focus are projected independently, sometimes by matching substrings of display copy. So they contradict each other exactly at the high-stakes moments: stopping, approving, recovering and sending.

More than 30 findings reopen tasks marked Done, and several passed tests that stubbed the exact seam that broke. That makes this as much a verification gap as a design gap.

**Biggest opportunity:** build five shared projections, then re-skin the regions from them:
- one run truth;
- one failure-evidence record;
- one action/key/name registry;
- one attention/glyph registry;
- one width-priority model.

## What's Working

- **The patterns to unify around already exist:**
  - the send-authority card;
  - the Recovery Callout contract;
  - the approval card, with exact args, a scope line, "Denied by you" and a countdown;
  - Fork's "What is not copied";
  - Archive → Undo receipt.
- **Keyboard vocabulary and just-in-time teaching.** F6 skips hidden panes. Alt+A, Alt+I and Alt+C land focus. Ctrl+K groups chats by attention. The Guide line is built from the real action row. Slash descriptions teach accelerators. /help is generated from the live registry.
- **Honest-state groundwork.**
  - Disabled items state their reason at readable contrast.
  - Speak-replies consent names the provider, destination and cost.
  - Per-tab isolation of Stop, drafts and queue holds.
  - Rail collapses are announced.
  - About 40 modals share one Esc/backdrop mixin.

## Priority Issues

**[P0] Default and common setups cannot get a reply, and the reason is hidden or misattributed.**
- *Why:* Four separate failures, all silent or misleading (G4-01, G4-03 and GAP5-01, GAP2-01 and GAP2-02, G1-09):
  - **G4-01:** every default OpenAI or Anthropic send returns HTTP 400. Three built-in tool schemas use top-level `anyOf`/`oneOf`/`allOf`: `watchlists_update_collection_sources`, `watchlists_check_sources` and `todo_update` (`Agents/local_tool_provider.py`, since `6b9fdec12d`, 08-29). The copy says "choose another model".
  - **G4-03, GAP5-01:** any chat with a session system prompt or a character is refused before the provider is contacted, with no rendered reason. This includes every character chat and chats opening with a character greeting or a /generate-image result. It does *not* apply to the template `chat_defaults.system_prompt`; the lead verified that live.
  - **GAP2-01, GAP2-02:** past the compaction trigger, the summarizer bills on every send and never succeeds live.
  - **G1-09:** hydrated messages lack `parent_message_id`, which breaks Fork.
- *Fix:*
  1. Flatten the three schemas and add a conformance test over every LocalToolSpec and MCP schema.
  2. Admit system and image rows in `_build_durable_trace_request`, and surface the existing TRACE_PROVENANCE Retry / Send without capture / Cancel callout.
  3. Set `parent_message_id` during hydration.
  4. Persist the compaction reason, stop auto-retrying, and show a context-limit card.
  5. Add a real-store test per trigger.
- *Command:* `/impeccable harden`

**[P0] During a run, Stop is unreachable and keystrokes land in the wrong place.**
- *Why:*
  - `BASE_ACTIONS_WIDTH` (`console_composer_bar.py:369`) never grew for TASK-28227's Redirect, so the row reads "Queue Dictate Redir" and Stop gets 0 cells at every width. There is no stop key, palette entry or /stop (G1-01).
  - Enter on the focused Menu button sends (G1-07).
  - 'y' from any chip opens Trace (G4-12).
  - Esc never leaves the transcript (G4-13).
  - A prompt typed about 1–2 s after a tab click runs in the previous tab, with its tools (GAP1-01).
  - Shell commands typed under the HOST TERMINAL banner go to the model (GAP4-05).
- *Fix:*
  1. A run-aware action slot where Stop replaces Send and can never be clipped, plus a stop key, a palette entry and /stop.
  2. One input-ownership contract: focused controls own Enter; printables typed on a non-text control go to the composer; bare letters work only in announced list modes.
  3. Bind the send target synchronously at tab activation, and assert it at submit.
- *Command:* `/impeccable harden`

**[P0] Ordinary actions crash, freeze or lose work silently.**
- *Why:*
  - Conversation menu "Copy as ▸ Save .md…" kills the app: `self.push_screen` on a Screen at `chat_screen.py:7149` (G3-02).
  - "Choose folder" for project instructions freezes the app, Ctrl+Q included: `push_screen_wait` from a non-worker handler at `session.py:5420` (GAP4-01).
  - Arrowing onto 14 Cloud providers in the first-run wizard blanks the step, and re-entering the step quits the app (G3-01).
  - Tab ✕ does nothing: an AttributeError is swallowed (G2-04).
  - Ctrl+Q discards drafts and Temporary chats (G4-11, GAP2-15).
  - The queue halts on a false "Turn failed" (GAP1-04, GAP5-09).
- *Fix:*
  1. The one-line fixes: `self.app.push_screen`, `_console_runtime_accessor()`, run the folder picker in a worker, filter the wizard's provider list to persistable providers.
  2. A parametrized live test that dispatches every row-menu action id.
  3. One loss projection shared by close and quit.
  4. Undo for Delete.
- *Command:* `/impeccable harden`

**[P1] Status surfaces contradict each other at the high-stakes moments.**
- *Why:*
  - `active_run` feeds provider readiness, so healthy runs show "Blocked / Provider setup needed" and a "setup" badge (G2-02).
  - The header says "Ready" after a 401 or an emergency stop (G3-05).
  - A 429 or 5xx becomes "Trace capture blocked — provider not contacted", even though the provider was contacted (G4-04, G4-05, G4-06).
  - "Completed while hidden" fires for the visible tab (G2-03).
  - Alt+A doesn't see approvals waiting in other tabs (GAP1-03).
- *Fix:*
  1. One `ConsoleRunPresentation` per session, with a closed state set (Ready · Sending · Running · Waiting for you · Paused · Failed · Stopped · Held) that every surface renders from.
  2. One `ConsoleFailureEvidence` record (phase, provider_contacted, http_status, sanitized provider message) that all recovery copy is derived from, with a lint against branching on display-copy substrings.
  3. An optimistic "Sending…" echo at Enter.
- *Command:* `/impeccable clarify`

**[P1] No single register of actions, keys, names or state markers.**
- *Why:*
  - The footer, F1, palette, slash registry, BINDINGS and the docs are six hand-kept lists, and they already disagree:
    - "Y trace" is advertised, but 'y' types a letter.
    - F1 claims the palette lists every Alt action.
    - The guide documents Stop, Mic, F3 and F9, which don't exist.
  - Chat, tab, conversation, session and agent all name one object.
  - Approval is drawn 8 ways, and a blocked chat never shows on its tab.
- *Fix:*
  1. A `ConsoleActionRegistry` that generates a priority-degrading footer, F1, the palette, /help and the docs' key tables, with an agreement test.
  2. A DESIGN.md Console glossary enforced by a copy lint.
  3. A `ConversationAttention` glyph registry.
- *Command:* `/impeccable clarify`

## Unification themes (the root causes to fix once)

1. **One truth per concept.** Run state and readiness are projected nine ways, some by substring matching (G1-04, G2-02, G1-12, G3-05, GAP1-03 …). *Move:* ConsoleRunPresentation plus typed blocker codes.
2. **Failures are swallowed, then explained with facts the code never had.** This comes from broad excepts and `exit_on_error=False`. Recovery lands in 4–5 places, each with its own grammar (G4-01, G4-04, G2-04, G3-02, GAP2-01 …). *Move:* ConsoleFailureEvidence plus one RecoveryCallout anchored to the affected turn; every catch on a user path must log and render.
3. **Input ownership.** The screen-level `on_key` swallows Enter, bare-letter bindings fire from any control, and session binding is asynchronous (G1-07, G4-12, G4-13, GAP1-01, GAP4-05 …). *Move:* one input-ownership contract pinned with Pilot tests.
4. **No single register** of actions, keys, names and help (G4-17, G4-16, G3-20, G3-23 …). *Move:* ConsoleActionRegistry plus a glossary lint.
5. **State glyphs speak several dialects.** There are 4 tables, and ▸ means five things (G4-52, GAP1-11, GAP3-06 …). *Move:* a ConversationAttention registry plus a controls-grammar table.
6. **Width isn't budgeted by priority,** so safety-critical items clip first: Stop, Deny, Approvals, cost (G1-01, G1-05, GAP4-07, G4-15 …). *Move:* one width-priority model for every horizontal region; never clip a safety control; always show a "+N ▸" overflow; a combined rail budget with hysteresis.
7. **Two styling tiers,** raw hues, and one style used for focus, active and selected. The real focus change is 1.03:1, and about 20 modals turn into black boxes in light themes (GAP3-02, G4-58, G4-14 …). *Move:* move widget CSS into tokenized .tcss, add `$ds-*-fg` text tokens, one shape-changing focus treatment of at least 3:1, and a theme-matrix contrast CI.
8. **Regions have no single job.** The chat title appears 6×, provider/model 3–4×, there are 3 create buttons and 5–6 chat searches, and Inspect runs about 90 rows (G2-17, G2-13, G4-55 …). *Move:* a region-ownership ADR, as an owner decision with a mock.
9. **Consequential actions have no feed-forward or receipt.**
   - Delete takes a subtree.
   - Comment, Continue and Summarize spend tokens silently.
   - /rewind leaves a blank transcript.
   - A character swap relabels history.
   - One Enter on Hands-free starts the microphone pipeline.
   - Exports land in the current working directory. This happened during the review: `trace-export.json` was written into the worktree.

   *Move:* generalise the Archive/Fork receipt pattern.

## Persona Red Flags

**Alex (power user)**
- The run row ends in "Queue Dictate Redir". Esc, Ctrl+C, Ctrl+P "stop" and /stop all fail to stop the run.
- Pasting a log and pressing Enter arms "Expand?" instead of sending.
- Tab ✕ and middle-click do nothing.
- Alt+1..9 type characters.
- A 3-prompt queue halts at a false "Turn failed", and its Retry says nothing is retryable.
- The footer's "Y trace" types a y.

**Jordan (first-timer)**
- Down onto "ByteDance Seed" blanks the wizard's Provider step with "Something went wrong in _ownership_for"; Next then exits the app.
- First view jargon: "source handoffs", "Context rail", "Agent blocked".
- The bold green "Terminal" opens Settings ▸ Privacy & Security.
- After /help, Fork refuses assistant messages.

**Sam (keyboard-only, low vision)**
- Tab to Menu, then Enter, sends the half-written draft.
- Focus moves measure 1.03–2.10:1, and the active tab and selected message reuse the focus look.
- The switches have no On/Off word, a 1.42:1 knob, and a focused state that hides the value.
- The focused placeholder is 2.56:1.
- F1 does nothing inside Trace, Ctrl+K or the Menu.
- After approving, focus goes nowhere.
- At 80×24 the approval card loses Deny.

**Solo builder/operator (PRODUCT.md primary)**
- Default OpenAI and Anthropic sends fail with "choose another model".
- Character chats never reply.
- A prompt typed about 2 s after switching tabs runs in the other tab, with its tools.
- "Choose folder" freezes the app.
- Approved `fs_*` calls fail with "Private scratch space is unavailable".
- Grounded answers cite [S1] but show no Sources row.
- Trace export fails with "dangling parent_event_id".

## Cognitive load and emotional journey

**Cognitive load: high, with 7 of 8 checklist items failing.** Decision points with more than 4 options:
- global nav: 14
- status strip: 8–12 chips
- composer Menu: 7–11
- Inspect: about 15 sections
- message row: 6 actions plus 7 in More
- slash popup: 23 commands
- Trace: 23 single-letter keys
- chat search: 5–6 entry points

**Emotional journey:**
- The peak is arrival: an honest wizard and a numbered setup card.
- Trust drops at the first send: the draft vanishes, the screen says "No messages yet" beside "Ready", the reply lands all at once, then a false toast appears.
- The deepest valley is mid-run: Stop is gone and Inspect calls the healthy run "Blocked".
- The ending is negative: Ctrl+Q discards work, and relaunch opens a blank "Chat 1".

**Design these moments first:** stopping a run, approving file writes, the first reply to a failure, quitting with unsent work, and actions that spend tokens.

## Minor Observations

- Single newlines in assistant Markdown collapse (`breaks=False`), so "one per line" output renders as one line (G1-27).
- Wide tables stretch to about 168 columns, and prose has no line-length cap (G1-42).
- The "jump to latest" pill never clears after End (G1-44).
- Ctrl+J for a newline is taught only in F1 (G1-46).
- Mixed casing and spelling: "Edit Message" vs "Edit system prompt", "Save as..." vs "Rename…", "Favourite" vs "Starred" (G4-61, G4-62).
- The Edit Message modal opens with the caret at (0,0) (G3-43).
- Environment lists untracked files as "A +0 −0" (GAP4-21).
- The Chats list re-sorts on open, and blank tabs push saved chats out of the 12-row cap (G2-21, GAP2-09).
- "Progress: 0 queued" supervisor jargon appears for every user (G2-41).
- In the appearance picker, Tab visits up to 180 emoji before reaching Apply (GAP3-14).
- Perceived-latency note for the PERF series: 75–92% of time-to-first-token passes before the provider is contacted, with 2.3 s event-loop stalls in `private_paths._native_open`.

## Questions to Consider

- If "What happens if I send now?" were computed from the same prepared request the send path uses, could every status surface render from it? What, if anything, would still need a separate source of truth?
- What if Send simply *became* Stop while a run is active: one slot, one key, one place the eye already goes?
- At 80×24 you can't afford a header row, status strip, tab strip, title row and two rail handles at once. If you had to delete one region entirely, which would it be, and what does that say about which region owns each job?
- Is "Temporary" a kind of chat or a capture policy? Today its first send hits an error-styled interstitial whose default action silently makes the chat permanent.
- 34 findings reopen Done tasks, and several stubbed the exact seam that broke. How many would one rule have caught: every user-initiated action gets a live-stack test with no fakes, and every catch on a user path both logs and renders?
