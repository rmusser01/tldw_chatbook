# ADR-210: Console region ownership — one job per region

Every Console region restates facts that another region owns, so the active chat's title appears up to six times, provider and model three or four times, and the chrome takes half of an 80x24 screen. This ADR gives each region one job: the header shows authority, the tab strip shows identity, the Chats rail finds chats, Inspect explains what Enter sends and what the run is doing, the status strip shows this tab's state and cost, and the composer owns the run. It names one new home for every control it moves and ships in nine steps, none of which removes a control before its new home exists.

Date: 2026-09-29
Status: Proposed — pending owner approval (drafted from the 2026-09-29 Console UX review)
Task: [TASK-33627](../tasks/task-33627%20-%20Console-unification-give-each-screen-region-one-job-per-the-region-ownership-ADR.md)
Evidence: [Console UX review 2026-09-29](../../qa/console-ux-review-2026-09-29/report.md), theme 8 "Regions have no single job"; ledger ids G2-36, G2-17, G2-15, G2-13, G2-14, G2-19, G2-18, G4-55, G4-36, G2-34, G2-35, G2-26, G2-28, G2-29, GAP5-08, G2-12, G2-11, G2-30, G2-41, G4-42, GAP5-22, GAP4-13, GAP4-03, GAP4-18
Links: the Task and Evidence links resolve once PR #2926 merges. That PR adds the task file, the review report and its ledger.
Depends on: TASK-33620 (one run truth), TASK-33622 (input ownership), TASK-33623 (action registry and glossary), TASK-33624 (attention and glyph registry), TASK-33625 (width priority, including TASK-33625.1 Stop and TASK-33625.2 Deny), TASK-33626.1 (speech-control state words and consent)

Supersedes in part:

- [ADR-017](017-console-left-rail-usability.md):
  - :10's four-section list (Session/Context/Model/Details) and :18 are replaced by one Chats browser.
  - The :27 rejection of "merge, rename, or reorganize the four rail sections" is reversed.
  - :10's text-only, bordered-section language and :35 remain, and now also govern Inspect's single header grammar. :12 and :36 (the shared workspace-identity helper) remain.
  - :38's `[New]` workspace test retargets to the Workspaces group header.
- [ADR-083](083-console-edge-rails-and-workspace-tree-ownership.md):
  - :20-27: the Sessions, Model, Agent and Details ceilings retire with their sections. Workspaces and the Default list keep 20. Characters keeps :61-66; whether the 2026-09-10 amendment's grow-to-viewport portrait survives is open question 3, because that growth causes GAP5-22.
  - :51-52: the pinned Switch/New/RAG Scope strip is replaced:
    - Switch moves to the header workspace menu, which remains the route to Default.
    - New moves to the Workspaces group header.
    - RAG Scope moves to Next send ▸ Sources.
  - :54-59: the separate search *inputs* become one field. Each projection keeps its own query, debounce, generation, cache, Retry and worker state.
  - :68-70 ("responsive widths … remain unchanged") and :126-128 ("responsive width authority … remain unchanged") are replaced by the combined rail budget (Degradation rules).
  - :111-121: the pinned "What happens if I send now?" card and the More boundary are replaced by the pinned Next send header and three fixed groups. "Lower groups do not repeat the same facts" becomes rule 2 below.
  - Unchanged: :12-18 (edge ownership, one divider owner), :30-50 (Tree ownership, Default outside the Tree, one owner per chat), :81-109 (selection versus activation, layout scopes) and the TASK-20937.4 Tree-move amendment.

Amends:

- [ADR-043](043-console-rail-compact-collapse-yields-to-explicit-toggle.md):
  - :12 "honored at any terminal width" becomes "side by side within the rail budget, otherwise as a pane swap" (open question 4).
  - :14-19 fixed Inspector priority at 100–149 columns becomes "last opened wins" at every width.
  - The :25/:53 118–128-column auto-open band is retired.
  - The 2026-08-19 amendment's 40-column transcript floor becomes 60.
  - Handles no longer hide below `CONSOLE_SINGLE_PANE_COLUMNS` (84).
  - Kept: collapse as a rendering override that never writes the preference, and the :43 rejection of overlays.
- [ADR-077](077-console-bounded-rail-section-scrolling.md):
  - :38-41 the 20-row ceiling now applies per Inspect group, and "Scope remains a compact row" is kept inside Next send.
  - :55-56 `n/p` moves between groups.
  - :76-83 the direct-section ownership map is rewritten as the three-group map. The STRICT/RESILIENT policy is kept.
- [ADR-079](079-console-library-conversation-authority.md) :270-273: the fixed-order chip `Library · Auto {off|on} · Agent {blocked|allowed}` keeps its grammar, order, modal and always-visible guarantee. Its home moves from the status strip to the header, and in focus mode to the strip, where it never folds. :259 and :277 (the Selected turn group) are unchanged.
- [ADR-014](014-retire-legacy-navigation-chrome.md) :12, :21 and :39: on Console the chat title renders in the tab strip, token/cost in the status strip, and database sizes in the header's `Local ▾` details. The Console footer carries key hints only. :41's kept "rail Model readouts" retire with the Model section.
- [ADR-015](015-shell-destination-ia.md):
  - :30-32: Console's DestinationHeader is one row carrying workspace, authority and three actions, with no title or purpose line (a documented DESIGN.md:311 exception; see Consequences). The hidden legacy statics stay until their contract tests migrate (migration step 8).
  - :36's deferred "control bar/chips" step is this ADR.
  - If the owner accepts the 1-row nav for every destination (open question 2), migration step 8 ships it as a shell-wide amendment to ADR-015.
- [ADR-071](071-focus-mode-chrome-free-console.md) :12-16 (Proposed): while focus mode hides the header, the Library chip and a pending Hooks count render in the status strip and never fold, and the active tab carries its workspace and access word.
- [ADR-088](088-console-lightweight-next-send-history-projection.md) :20 and :36: the Next send group renders the asynchronous exact-context preview that Chat details ▸ Context already uses. It is computed off the UI thread only while Inspect is open, debounced, and never executes retrieval. The strip's `next ~$` stays on ADR-088's synchronous projection.
- [ADR-095](095-conversation-owned-console-generation-settings.md):
  - :113-116: the Model section's warning badge and `Retry save` move to the Model slot's warning word and a recovery callout at the top of Next send.
  - :131-137: a failed default save stays an app-level record. Its callout and warning word render in every tab, with its own actions: `Retry default save` and `Discard retry`, or `Refresh running app` and `Dismiss`.
- [ADR-098 (prompt queue)](098-visible-bounded-console-prompt-queue.md) :55-58: while this tab runs, the primary slot is `■ Stop ▾`, and enqueue is `Queue` or Enter. `Preparing…` and `Queue full` render in the consequence line. The queue shelf stays in the deck.
- [ADR-120](120-character-conversation-navigation-and-local-semantic-search.md) :189-191 and ADR-083's 2026-09-03 amendment: Character becomes the Characters group of the Chats browser, after Everyday chats. Model no longer exists in the rail.
- [ADR-132](132-fleet-history-navigation.md) :8-9: the four-row fleet preview and its count move into Run ▸ Sub-agents, and the "View all" tail becomes `All sub-agent runs…`.
- [ADR-171](171-console-conversation-review-and-attention.md):
  - :54-56: the Conversation Inspector modal is renamed **Chat details**, with its tabs (Context, Usage & cost, Exchange history) unchanged.
  - :63-64 ("do not redesign the Inspector sidebar as a side effect") is satisfied, because this is the deliberate redesign.
  - :42-44's right-hand menu control is unchanged, including the chat's own icon.
- [ADR-089 (per-turn change review)](089-console-per-turn-change-review-ownership.md):
  - :23-24: "pinned send authority" now names the pinned Next send header. This is a cross-reference update only.
  - :18-20 stays in force: Run ▸ Changes is the Environment projection of the workspace's git state, not a revived cross-turn Changed Files list.
- [DESIGN.md](../../DESIGN.md):107 and :311: add a Console variant of the screen grammar and the destination header (see Decision 2).

Preserves:

- [ADR-011](011-chatbook-workbench-ui-system.md):46: the palette is never a hiding place.
- [ADR-027 (Default chats)](027-default-workspace-chats-in-chats-section.md) :26-28 and [ADR-082](082-console-per-chat-private-scratch-space.md) :21: everyday chatting uses no workspace vocabulary. Default chats stay outside the Workspaces Tree, labelled "Everyday chats" (the switcher's existing gloss), and the header names Default the same way (open question 5).
- [ADR-028 (workspace folders)](028-settings-workspaces-category-and-folder-roots.md) :19-20: access is set per folder, so the header reads `files: mixed` when bound folders differ.
- [ADR-031 (keybindings)](031-tui-keybinding-and-footer-hint-conventions.md): no `ctrl+c`/`ctrl+w` bindings; footer hints stay truthful.
- [ADR-068 (text selection)](068-console-text-selection-and-annotations.md):157-158: the header hosts only the active-playback lifecycle.
- [ADR-085](085-console-activity-receipts-and-switcher-ownership.md):22-24: the rails do not read the switcher's model.
- [ADR-098 (duplex voice)](098-low-latency-speculative-duplex-voice-pipeline.md) :327-328 and :343-345: Esc discards a provisional voice turn before it returns to the draft (rule 9), and `Dictate ▾` carries the pipeline's state words.
- [ADR-136](136-scoped-child-progress-and-supervisor-relay.md) :229-230: waiting sub-agent reports stay reachable with no live sub-agents.
- [ADR-184 (draft shelf)](184-local-console-prompt-draft-shelf.md) :13-14: the Prompt Workbench stays the sole draft-shelf surface, so the deck has no draft-shelf row.
- [ADR-195](195-console-live-tool-call-presentation.md): the approval card remains the decision surface.
- [ADR-197](197-console-hook-configuration-review.md):28-31: Hooks stays in the visible ConsoleControlBar, now the header's action cluster.

## Context

Measured on `origin/dev @ 64579cce2c` from live captures at 235x52, 211x44, 160x45, 120x40, 100x30 and 80x24 (the review's J7, J8 and B2 evidence):

- **The same facts are restated many times.**
  - The active chat's title appears up to 6 times: the tab, the `Conversation |` row, the rail's current-chat line, the rail list row, Inspect `Where:` and Ctrl+K. It is truncated with both `…` and `...`. On first arrival `Chat 1` appears 4 times (G4-55, G2-15).
  - Provider and model appear 3–4 times: the strip, the Inspect run recipe and the Inspect settings block. TASK-23196 removed the rail copy, and the others survived (G4-55).
  - There are three visible create buttons under four names ("New tab" twice, "New conversation", "New chat"). One is a worker-mounted alias that appears, vanishes and moves between captures (G2-14, G2-13).
  - There are five or six chat-search entry points, each with its own scope, verb and Clear behaviour (G2-19). A query typed into Inspect's Library input is ignored (G2-11).
- **The left rail mixes five jobs across seven sections** (Terminal, Workspaces, Conversations, Character, Model, Agent, Details), in an order that ignores frequency. Its Model section does not show the model (G2-36).
  - It dumps raw step JSON (G2-29) and shows "Progress: 0 queued" to every user (G2-41).
  - A 14-row avatar squeezes the other sections (GAP5-22).
  - The green `Terminal` silently jumps to Settings (G2-12).
- **Inspect is about 15 sections and about 90 rows, in four header grammars (G2-17).**
  - The pinned send card omits the system prompt, persona, lore, tools and history, and misstates sampling. The complete preview hides behind the cost chip or an undocumented key (G2-18, GAP5-08).
  - Raw payload fields are shown (G2-26).
  - `Agents` means sub-agents only, and its drill-in opens in the other rail (G2-28).
  - Environment shows the previous repository for up to 10 s after a workspace switch, and its how-to text is cut off (GAP4-03, GAP4-13).
  - Refresh gives no feedback (G2-30).
- **The header** has a feature-list purpose line and leaves the workspace invisible (G2-34). Its action row has five unrelated buttons, and its "Settings" is not the F4 Settings one row above (G2-35).
- **The status strip** needs about 251 columns and clips mid-token (`Mode`) at 80 (B2 census).
- **The screen overflows at small sizes.** Chrome takes 12 of 24 rows at 80x24. Transcript share is 50%, 60%, 65% and 73% at 80x24, 100x30, 120x40 and 235x52 (G4-36).
- **Focus and workspace switching are unpredictable.**
  - Tab order jumps between the bottom and top of the screen, and most regions are outside the Tab cycle (G4-42).
  - Tabs from every workspace share one strip, and clicking one silently switches workspace (GAP4-18).

Root cause: every region projects its own copy of shared facts. The sibling unifications (TASK-33620, 33623, 33624, 33625) remove the duplicate *computation*. This ADR removes the duplicate *display* by giving each region one job. ADR-197 (accepted 2026-09-27, TASK-33163 in progress) adds a persistent Hooks action to the ConsoleControlBar, and the header below keeps a slot for it.

## Decision

### 1. Rules

1. **One job per region.** A region renders only facts in its own scope (table 2). The one exception is Run ▸ Changes, which shows the header workspace's git state and names that workspace.
2. **One glance, one detail.**
   - A fact appears at most once in persistent chrome (header, tab strip, status strip, composer region, footer) and at most once in its detail owner (an Inspect group or the Chats rail). Both copies render from the same projection, and the glance opens the detail.
   - The next-send cost is the one exception. The strip's `next ~$` is ADR-088's synchronous estimate, and Next send ▸ Total is the exact preview (Decision 3). Both round to the same precision, and when they differ, Total is authoritative.
   - Transcript content and modals are exempt.
   - A list may mark the row the tab strip names, but must not restate it.
   - Other regions may echo state only as one glyph from the TASK-33624 registry.
3. **Scope split for live state:**
   - *This tab:* the status-strip Run slot and the composer's primary slot.
   - *Other tabs:* tab markers, the counted overflow and Alt+A.
   - *Other destinations:* the nav `!`.
   - *Authority* (workspace, file access, storage/server, Library policy, hooks): the header, or the strip and the active tab in focus mode.
4. **Static facts earn no persistent cell.** Tool counts, default scope, the integration inventory and storage paths live in Inspect or Settings. They return to a persistent row only when degraded, as state.
5. **The safety set never clips** (see Degradation rules).
6. **Each action has one visible home in the default view,** plus its key, palette and slash twins (TASK-33623 AC#5).
   - The default view is the Console at the current width, with each rail in its default state.
   - A visible `⋯`/`▾` menu counts as a home, and so does a visible rail handle, because one activation reveals the control. The palette never does.
   - One chat menu reached from a tab or a rail row counts as one home.
7. **No removal before relocation.** A control's new home ships in the same PR as its removal, or earlier.
8. **One vocabulary** (the TASK-33623 glossary):
   - the object is a *chat*, and an open chat is a *tab*;
   - the verb is *New chat*;
   - the modal is *Chat settings…*;
   - the full request view is *Chat details*;
   - *Chats* is the left rail, and *Inspect* means only the right rail.
9. **Focus follows layout.**
   - F6 and Shift+F6 cycle five regions in reading order: Top (the header, then the tab strip) → Chats → Transcript → Inspect → Composer region (deck, status strip, composer row). Hidden rails are skipped.
   - Tab stays in visual order within a region.
   - Esc with no transient surface open first exits hands-free or discards a provisional voice turn when one is active (ADR-098 duplex :327-328; Esc already does this today). Otherwise it returns to the draft (TASK-33622 AC#4).
   - Alt+A focuses the oldest pending decision in any tab, switching tabs and announcing the switch. This is new: today Alt+A reaches only the current tab's card. If a pane swap is showing, Alt+A closes it first.

### 2. Region → job

This table is the Console variant of DESIGN.md:107's screen grammar:

- the destination header is the header;
- the local mode bar is the tab strip;
- the primary list is Chats;
- the optional inspector is Inspect;
- footer status is the status strip plus the footer.

| Region | Job (the question it answers) | Owns | Must not show |
|---|---|---|---|
| Global nav (shell) | Which destination, and does another one need me? | <ul><li>Destination items with their keys</li><li>`More ▾`</li><li>The attention `!`</li></ul> | Console state |
| Header (Console DestinationHeader, 1 row) | Where does this chat's work land, and under what authority? | <ul><li>`Workspace: <name> ▾`, which reads `Everyday chats ▾` for Default. Its menu holds Switch… (Alt+W; the route to Everyday chats), New workspace…, New chat here, Files… and Workspace settings….</li><li>The bound folder and its access word: `files: read-write`, `read-only` or `private scratch`. With several folders the path reads `n folders`, and the word reads `files: mixed` when their access differs.</li><li>`Local ▾`, `Server: <name> ▾` or an explicit unreachable/offline state. It opens the storage, sync, file-tools, server and handoff details, this chat's source and resume state, and the database sizes.</li><li>The ADR-079 Library chip</li><li>ADR-068's playback lifecycle</li><li>ConsoleControlBar with exactly `Chat settings…`, `Hooks` (ADR-197, with its pending/error count) and `Help F1`</li></ul> | This tab's run state; the chat title; a purpose line; create, rail or search buttons; idle speech switches; provider/model |
| Tab strip (top of the centre column) | Which chats are open, which one am I in, and which need me? | <ul><li>A tab per open chat: title (full on focus and tooltip), attention marker, a `Temp ·` prefix, a `<workspace> /` prefix when outside the header's workspace, and ✕. The active tab adds `▾`, which opens the ADR-171 chat menu.</li><li>A counted overflow, `+N ▾`, carrying the highest hidden attention glyph</li><li>A create cluster pinned outside the scroll: `+ New chat`, `+ Temporary`, `Terminal: <state>`. It folds into `+ New ▾`.</li></ul> | A second title row; banners; coachmarks |
| Chats rail (left, formerly Context) | Find and open any saved chat. | <ul><li>One `Find chats…` field over every owner, with results grouped by owner. Ctrl+K is the same component as a modal.</li><li>Groups: Workspaces (the ADR-083 Tree, with `+ New` in its header), Everyday chats (ADR-027's Default and unassigned list) and Characters (ADR-120 bounds, with an `All in Roleplay ↗` footer)</li><li>The active row marked `›` and bold</li><li>The ADR-171 chat menu control at the right of each row, showing the attention glyph or the chat's own icon (custom, or 💬)</li><li>A footer with `Archived n…` and `All in Library ↗`</li></ul> | Terminal; model settings; run monitoring; storage/server details; the current-chat line; create-chat buttons; archive-current-chat; the fleet line; the workspace action strip |
| Transcript | The work, and every decision about it. | <ul><li>Turns and tool rows (one presenter)</li><li>The ADR-195 approval card</li><li>Recovery callouts anchored to the affected turn</li><li>Message actions</li><li>The jump pill</li><li>The first-run empty state</li></ul> | A title row; tips as rows; status summaries |
| Inspect (right, opt-in) | What will Enter send, what is this run doing, and what did the selected turn use? | Three fixed groups (Decision 3) | Identity and `Where:` rows; a second run rollup; the integration inventory; the settings block; raw payload fields; a second search input; Stop |
| Status strip | What is this tab doing, and what will it cost? | Four fixed slots plus `+N ▾` (Decision 4) | Library policy (except in focus mode); tool count; zero or default values; provider as its own chip; a collapse control |
| Composer region | Write the next instruction, and control this tab's run. | <ul><li>The deck: staged evidence, the ADR-098 queue shelf, a pending handoff with `Review…`, and dispatch recovery. Each shows only when non-empty.</li><li>The composer row (Decision 5)</li></ul> | "Send disabled: type a message"; a price suffix on Send; a collapse control |
| Footer | Which keys work here, right now? | <ul><li>Registry-generated hints (TASK-33623), ranked by focused region and state</li><li>The `F1 keys` anchor</li></ul> | State values; token/cost on Console; keys that do nothing in the focused region |
| Overlays, palette, slash | Speed up what is visible, and hold rare or diagnostic views. | <ul><li>The Ctrl+K switcher (ADR-085 data model)</li><li>Chat details, Trace, Library Access, Chat settings, the workspace switcher and the `Local ▾` details</li><li>The Prompt Workbench and its draft shelf (ADR-184)</li><li>Settings ▸ Diagnostics (integrations)</li><li>A twin for every visible action</li></ul> | Being the only home of anything |

### 3. Inspect: three fixed groups

The groups use one header grammar, `▾ Title ──── summary`, with the whole row as the toggle and ADR-034's glyphs.

- **Scrolling:** each body keeps ADR-077's 20-row ceiling. The Next send header row is pinned above the scroll owner, replacing the pinned card.
- **Disclosure** persists through ADR-083's layout scopes. Three group ids replace `inspector_more`, and `environment`/`tasks` survive as Run subsection ids.
- **Defaults:** Next send and Run are open, and Selected turn opens when a turn is selected.

1. **Next send** renders the asynchronous exact-context preview that Chat details ▸ Context already uses (G2-18, GAP5-08; amends ADR-088). It runs off the UI thread only while Inspect is open, is debounced, and never executes retrieval, so auto-search results appear only after the send. Rows:
   - **Model:** provider · model, only the parameters actually sent, and the verification time.
   - **System:** its source, with `Edit…`.
   - **Persona/Character:** the active persona or character, with `Choose…`, which is always enabled and opens the character picker, and `Reaction…`, which is disabled with a stated reason when there is no character.
   - **History:** message count and tokens.
   - **Sources:** readable lines, with raw fields behind `Details`; `Search Library…`, then `Last search: <q>` once one has run; `Scope:`; the auto-search state; and a recovery line only when retrieval is blocked.
   - **Tools:** built-in and MCP counted apart, with a line only when a degraded integration removes tools.
   - **Project:** the instructions file, with `Choose folder…`.
   - **Lore:** attached dictionaries and world books, with `Attach…` and `Detach…`.
   - **Prefill** (ADR-159).
   - **Handoff:** a pending launch, with `Review…`.
   - **Total:** ~tokens · ~$ next, with `Preview…`, which opens Chat details.

   A failed save renders at the top of the group as a recovery callout with ADR-095's actions: `Retry save` for this chat's settings; for a failed default save, `Retry default save` and `Discard retry`, or `Refresh running app` and `Dismiss`. A default-save failure affects future chats, so it shows in every tab.
2. **Run** covers this tab, from ConsoleRunPresentation. Rows:
   - **State line:** the strip's vocabulary.
   - **Waiting decisions:** a pointer to the card, with Alt+A.
   - **Steps:** in the transcript's tool-row grammar.
   - **Queue n:** when n > 0.
   - **Sub-agents:** when there are live sub-agents or waiting reports (ADR-136 :229-230). It holds ADR-132's four-row preview with the fleet count, an in-place drill-in with `Back` and the `Steer this sub-agent…` input, `Stop all`, `n reports waiting` and `All sub-agent runs…`.
   - **Changes:** the header workspace's git state, labelled with that workspace. It shows the branch and files, `Review & commit…`, `Review in Change Review`, `Push ↑n…`, the PR (`Open in browser`, `Add to chat`), the CI result (`Fix — add failure summary to chat`), `Updated HH:MM` after Refresh, and `Checking…` after a workspace switch. It is hidden when no folder is bound.
   - **Tasks.**
   - **Artifacts n:** when n > 0.
   - `Trace…`, `Run log…` (whenever a log is stored) and `All runs…`.
3. **Selected turn** shows:
   - citations;
   - Library activity (ADR-079);
   - usage: tokens, cost and time;
   - `Fork…`, `Trace this turn…` and `Details…`.

### 4. Status strip: four fixed slots

Each slot has a full form and a compact form. A slot with nothing non-zero to say renders nothing and keeps its place in the order. Hidden items fold into a focusable `+N ▾` that opens them.

1. **Run** uses ConsoleRunPresentation's closed vocabulary: Ready · Sending… · Running m:ss · `◆ Waiting for you · n calls · Alt+A` · Paused · `Failed — Retry` · Stopped · `Held — Clear` · `Setup needed…`.
   - It counts calls, not cards.
   - It is never folded.
   - Activating it runs the current state's route.
2. **Model:** `provider/model`, with the short id as its compact form.
   - It shows a warning word when settings failed to save or the provider is not ready. While the warning word shows, activating the slot opens Next send's recovery callout.
   - Alt+M opens its popover.
3. **Context:** `ctx 7% of 200k · $0.38 so far · next ~$0.04`, compact `7% · $0.38`.
   - The percentage is the next send's total against the model's window. A fallback window is marked estimated: `ctx ~7% of ~128k` ([ADR-052 :55-56](052-console-conversation-memory-and-compaction-policy.md)).
   - Near the compaction threshold it states that in words (GAP2-03).
   - It opens Chat details ▸ Usage & cost.
4. **Flags** appear only for non-default settings, in this order:
   - `Hands-free on`
   - `Temporary — not saved · Save`
   - persona or character
   - a custom system prompt
   - a narrowed scope
   - `Speaking replies`

   Flags fold first, except `Hands-free on`, which never folds (ADR-011:46; TASK-33626.1 AC#1).

Library policy lives in the header (the strip in focus mode), staged sources and a pending handoff in the deck, and approvals in Run.

### 5. Composer row

The row holds, in order:

- **`Menu ▾`**, grouped with Add first:
  - Add: Attach file, Paste image, `Search Library…`
  - Draft: Improve…, Prompts… (the Prompt Workbench, which holds the draft shelf), Save draft to shelf, Undo prompt change
  - Create: image, caption, video
  - Roleplay: Impersonate, Buddy…
  - Chat: Save this chat, Save as Chatbook
  - Voice: only when `Dictate ▾` is folded
- **The draft**, with named attachment chips.
- **A consequence line** for real consequences only: "Enter queues after this run", `Preparing…` during provider and skill validation, `Queue full`, or a blocker.
- **The primary slot:** `Send` becomes `■ Stop ▾` while this tab runs.
  - Its width is reserved (TASK-33625.1).
  - `▾` holds "Stop and redirect with this draft" and "Stop all sub-agents".
- **`Queue` and `Redirect`**, only while running with a draft.
- **`Dictate ▾`**, holding Dictate, `Speak replies: Off/On/Paused` and `Hands-free: Off/Starting…/On`, with TASK-33626.1's state words and consent.
  - While a voice turn is active, the control shows the pipeline's state word (`◉ Listening`, `Transcribing…`, `Responding…`, `Speaking…`) and cannot fold.

## Mocks

**Scenario:**

- Workspace `acme-api` is bound read-write to `~/src/acme-api`, Local. The Library policy is `Auto off · Agent allowed`.
- The active tab's agent run is at 1:12 and waiting on one `fs_write` approval. A follow-up is typed, and two sources are staged.
- Other tabs: one running, one finished and unseen, and one in `research-notes`.

**Legend:**

- `▐…▌` is reverse video: the active tab, and the active item of the 1-row nav. In the 3-row nav the active item is boxed (`╭╮` above, `│ │` beside).
- `▎` and `▏` mark the edges of an input field (the draft, `Find chats…`).
- `›` marks the active row in the Chats rail, and nothing else.
- A leading `▾`/`▸` on a group or tree row means expanded/collapsed (ADR-034). A trailing `▾` opens a menu, `…` opens a dialog, and `↗` opens another destination. `▾` keeps both of its current roles; TASK-33624 decides whether to split them.
- At 80 columns the 1-column rail handles are the `▏` and `▕` edges, and each spells its rail's name upright (Settings' existing stacked-letter handle style). An open rail's head shows its toggle key.
- ◆ needs you · ● running · ✓ unseen result · ◉ microphone capturing, from the TASK-33624 registry, which has ASCII fallbacks. 💬 is a chat's own default icon (ADR-171); a custom icon replaces it.
- `Alt+X` in the mocks is the placeholder they were drawn with. The owner chose **Ctrl+G** as the stop chord on 2026-09-30 (open question 1, resolved), so read `Alt+X` as `Ctrl+G`. The mocks are left as drawn because `Ctrl+G` is one cell wider.
- In prose, `▸` separates the steps of a menu path.

Each block's committed width is in its heading (235 or 80 cells), and every row is exactly that wide, so trailing blanks are significant; an editor that strips them breaks the grid. A generator built each row and asserted its width with Rich `cell_len`, Textual's cell measure (💬 counts 2). It also derived each status strip, composer row, 80-column header and footer from the drop orders below. Grid dividers sit at columns 35 and 190 (1-based).

### 235x52, mid-run

Row budget:

- nav 3 · header 1 · rule 1 · tab strip and rail heads 1 · work area 41 · rule 1 · deck 1 · strip 1 · composer 1 · footer 1 = 52
- Transcript: 41 rows, or 42 with nothing staged. Today it gets 38.
- Rails: Chats 34, Inspect 45, transcript 154.

```text
         ╭──────────────╮                                                                                                                                                                                                                  
 ⌃1 Home │ ⌃2 Console   │ ⌃3 Library  ⌃4 Roleplay  ⌃5 Watchlists  ⌃7 Schedules  ⌃8 Workflows  ⌃9 MCP  ⌃0 ACP  F2 Lab  F3 Logs  F4 Settings  F5 Research  F7 Meetings                                                                       
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Workspace: acme-api ▾   ~/src/acme-api · files: read-write   │   Local ▾   │   Library · Auto off · Agent allowed                                                                                     Chat settings…    Hooks    Help  F1 
──────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────
 Chats                      Alt+C │▐◆ Refactor session auth ▾ ✕▌ ● Release notes ✕ │ ✓ Flaky test triage ✕ │ research-notes / Reading list ✕   │ + New chat │ + Temporary │ Terminal: locked │ Inspect                               Alt+I 
 ▎Find chats in every workspace…  │                                                                                                                                                          │ ▾ Next send ────────── what Enter sends now 
 ▾ Workspaces ───────────── + New │  You · 13:52                                                                                                                                             │   Model    anthropic · claude-sonnet-4-5    
  ▾ acme-api       ~/src/acme-api │  Which modules touch session state?                                                                                                                      │            T 0.6 · max 4096 · verified 13:40
    › Refactor session au… now ◆  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │   System   workspace default · 1.2k   Edit… 
      Release notes         2m ●  │  Assistant · claude-sonnet-4-5 · 3.4s · $0.006                                                                                                           │   Persona  Code reviewer            Choose… 
      Flaky test triage    14m ✓  │  Three: src/auth/session.py (SessionManager), src/auth/token_store.py (TokenStore) and src/api/middleware.py, which only reads the                       │   History  14 messages · ~9.8k tok          
      Migrate CI to uv      2d 💬 │  session id from the cookie and calls SessionManager.validate() [S1].                                                                                    │   Sources  2 staged · auto-search off       
      Rate limiter design   5d 💬 │  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │            S1 session.py · file · local     
      9 more…                     │  You · 13:58                                                                                                                                             │            S2 ADR-041 · note · local        
  ▸ research-notes             12 │  Where is session expiry enforced today?                                                                                                                 │            Search Library…  Scope: acme-api 
  ▸ homelab                     4 │  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │   Tools    6 built-in · 3 MCP (github)      
                                  │  Assistant · claude-sonnet-4-5 · 6.2s · $0.011                                                                                                           │   Project  AGENTS.md · sent each turn       
 ▾ Everyday chats ───────────── 3 │  Expiry is enforced in SessionManager.validate() (src/auth/session.py:88), which compares issued_at + ttl with the injected clock [S1].                  │   Lore     none                     Attach… 
      Chat 1                1h 💬 │  The TokenStore refresh path (src/auth/token_store.py:40) never re-checks expiry, so a refreshed token can outlive its session.                          │   Total    ~14.2k tok · ~$0.04     Preview… 
      Quick regex help  Sep 18 💬 │  ADR-041 says a refresh must not extend a session's absolute lifetime [S2].                                                                              │ ▾ Run ───────────────────── this tab · 1:12 
      Packing list      Sep 12 💬 │  Sources   S1 src/auth/session.py · file · local      S2 ADR-041 Token lifetimes · note · local                                                          │   ◆ Waiting for you · fs_write session.py   
                                  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │            decide in transcript · Alt+A     
 ▸ Characters ───────────────── 3 │  You · 14:02                                                                                                                                             │   Steps    4 · 3 done · 1 waiting           
                                  │  Refactor src/auth/session.py to use TokenStore for refresh. Keep the public API stable and add tests for the expiry edge.                               │   Changes  acme-api · main · 3 files +58 −14
                                  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │            Review & commit…   updated 14:03 
                                  │  Assistant · ◆ Waiting for you · 1:12 · step 4                                                                                                           │   Tasks    1 open                           
                                  │                                                                                                                                                          │   Trace…   Run log…   All runs…             
                                  │    ✓ fs_read    src/auth/session.py                                         0.2s                                                                         │ ▸ Selected turn ──────────────── none · j/k 
                                  │    ✓ fs_read    src/auth/token_store.py                                     0.1s                                                                         │                                             
                                  │    ✓ search     "SessionManager(" in src/ · 12 hits                         0.4s                                                                         │                                             
                                  │    ◆ fs_write   src/auth/session.py · +41 −12                   waiting for you                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │    ┌ ◆ Approval needed · fs_write ──────────────────────── acme-api · ~/src/acme-api · read-write ┐                                                      │                                             
                                  │    │ Writes src/auth/session.py (+41 −12). Changes one file inside ~/src/acme-api. Runs nothing.  │                                                      │                                             
                                  │    │ [ Approve once ]   [ Deny ]   Decision: Once ▾                                               │                                                      │                                             
                                  │    │ This call only.                                                                              │                                                      │                                             
                                  │    └──────────────────────────────────────────────────────────────────────────────────────────────┘                                                      │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
                                  │                                                                                                                                                          │                                             
 ──────────────────────────────── │                                                                                                                                                          │                                             
 Archived 7…     All in Library ↗ │                                                                                                                                                          │                                             
──────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────
 Next send · 2 sources    S1 src/auth/session.py · file ✕    S2 ADR-041 Token lifetimes · note ✕                                                                                                                                           
 ◆ Waiting for you · 1 call · Alt+A    │    anthropic/claude-sonnet-4-5    │    ctx 7% of 200k · $0.38 so far · next ~$0.04    │    Persona: Code reviewer                                                                                 
 Menu ▾ ▎also update the validate() docstring once the write lands                                                                                          ▏ Enter queues after this run │ ■ Stop ▾  Alt+X │ Queue │ Redirect │ Dictate ▾ 
 Alt+A approval · Alt+X stop · Enter queue · Ctrl+J newline · / commands · Ctrl+K find chat · Ctrl+T new chat · F6 next pane · Ctrl+P palette · Ctrl+Shift+F focus · F1 keys                                                               
```

### 80x24, mid-run

Row budget:

- compact nav 1 (open question 2) · header 1 · tabs 1 · transcript 17 · deck 1 · strip 1 · composer 1 · footer 1 = 24
- Transcript: 17 rows, or 18 with nothing staged. Today it gets 12. With today's 3-row nav it would get 15 or 16.

What folds at this width, by the drop orders:

- **Rails:** both show 1-column handles, because Chats needs 92 columns beside the transcript and Inspect needs 96.
- **Header:** the folder path, `Help F1` and `Chat settings…` fold (the last two into `⋯`), the label words drop, and the access word compacts to `rw`.
- **Tab strip:** inactive titles are cut to 12 cells, the create cluster folds into `+ New ▾`, and two inactive tabs fold into `+2 ✓ ▾`. The active title stays whole.
- **Status strip:** the persona flag folds into `+1 ▾`, Context compacts and Model shows its short id. Run keeps its full form.
- **Composer:** the consequence text moves to the footer (`Enter queue`), `■ Stop` drops its key label, `Dictate ▾` folds into Menu ▸ Voice and Redirect into `■ Stop ▾`. `Queue` stays, and the draft keeps 52 cells (65%).
- **Footer:** after the state-critical keys and the composer's own keys, `/ commands` does not fit, so fitting stops there. The rail handles stay visible.

```text
 ⌃1 Home ▐⌃2 Console▌ ⌃3 Library  ⌃4 Roleplay  ⌃5 Watchlists  More ▾            
 acme-api ▾ · rw │ Local ▾ │ Library · Auto off · Agent allowed         Hooks ⋯ 
▏▐◆ Refactor session auth ▾ ✕▌ ● Release not… ✕ │ +2 ✓ ▾               + New ▾ ▕
▏  …the injected clock [S1]. The refresh path never re-checks expiry, so a     ▕
▏  refreshed token can outlive its session; ADR-041 forbids that [S2].         ▕
▏  Sources  S1 src/auth/session.py · S2 ADR-041 Token lifetimes                ▕
▏ ──────────────────────────────────────────────────────────────────────────── ▕
▏  You · 14:02                                                                 I
C  Refactor src/auth/session.py to use TokenStore for refresh. Keep the publ…  n
h ──────────────────────────────────────────────────────────────────────────── s
a  Assistant · ◆ Waiting for you · 1:12 · step 4                               p
t    ✓ fs_read   src/auth/session.py                                 0.2s      e
s    ✓ fs_read   src/auth/token_store.py                             0.1s      c
▏    ✓ search    "SessionManager(" · 12 hits                         0.4s      t
▏    ◆ fs_write  src/auth/session.py · +41 −12            waiting for you      ▕
▏  ┌ ◆ Approval needed · fs_write ─────────────────── acme-api · read-write ┐  ▕
▏  │ Writes src/auth/session.py (+41 −12) · 1 file in ~/src/acme-api        │  ▕
▏  │ [ Approve once ]  [ Deny ]  Decision: Once ▾                           │  ▕
▏  │ This call only.                                                        │  ▕
▏  └────────────────────────────────────────────────────────────────────────┘  ▕
 Next send · 2 sources   S1 session.py ✕   S2 ADR-041 ✕                         
 ◆ Waiting for you · 1 call · Alt+A │ sonnet-4-5 │ 7% · $0.38 │ +1 ▾            
 Menu ▾ ▎…pdate the validate() docstring once the write lands▏ ■ Stop ▾ │ Queue 
 Alt+A approval · Alt+X stop · Enter queue · Ctrl+J newline · F1 keys           
```

### 80x24, idle first run

Nothing is staged or queued, so the transcript gets 18 rows.

- The empty state names the first moves and the pane key (G4-42). New tabs reuse the same copy.
- The header names Default `Everyday chats`, with the access word `scratch` because no folder is bound. At 80 columns this reaches header step 7, so its two `▾` markers drop; both segments stay activatable.
- With one tab, the create cluster shows all three of its buttons.
- The strip has no Context slot, because nothing has been spent and the draft is empty.

```text
 ⌃1 Home ▐⌃2 Console▌ ⌃3 Library  ⌃4 Roleplay  ⌃5 Watchlists  More ▾            
 Everyday chats · scratch │ Local │ Library · Auto off · Agent blocked  Hooks ⋯ 
▏▐ Chat 1 ▾ ✕▌                     + New chat │ + Temporary │ Terminal: locked ▕
▏                                                                              ▕
▏                                                                              ▕
▏                                                                              ▕
▏             This chat is saved when you send the first message.              ▕
▏                                                                              ▕
▏              Type a message and press Enter.  / lists commands.              I
C           Search Library… in Menu ▾ stages your own sources first.           n
h                                                                              s
a         Ctrl+K find a chat · Alt+C chats · Alt+I inspect · F6 panes          p
t                                                                              e
s          Chats stay on this computer (Local). A message leaves only          c
▏               when you press Enter, to the model shown below.                t
▏                                                                              ▕
▏                                                                              ▕
▏                                                                              ▕
▏                                                                              ▕
▏                                                                              ▕
▏                                                                              ▕
 Ready │ anthropic/claude-haiku-4-5                                             
 Menu ▾ ▎Ask, command, or paste a task…                      ▏ Send │ Dictate ▾ 
 Enter send · Ctrl+J newline · / commands · Alt+C chats · F1 keys               
```

## Degradation rules

**Never clipped, at every width of 80 columns or more and in every rail configuration (the safety set):**

- the composer primary slot (`Send` / `■ Stop`);
- the approval card's `Approve once` and `Deny`, which wrap by the card's own width;
- the Run slot, including its ◆ count and key;
- the active tab with its ✕, its `Temp ·` prefix and, in focus mode, its workspace prefix;
- `Dictate ▾` while a voice turn is active, and the `Hands-free on` flag;
- in the header:
  - the workspace name (middle ellipsis, at least 8 cells);
  - the file-access word;
  - `Local`/`Server`;
  - the Library chip (ADR-079);
  - `Hooks` (ADR-197);
- in focus mode, the Library chip and a pending Hooks count in the strip;
- in the footer: `F1 keys`, `Alt+A` while anything waits, and the stop chord while running.

Items drop whole and are never cut mid-token. Titles use one shared `truncate_title()` with `…`.

**Drop order per region** (first dropped → last):

- **Header:**
  1. folder path
  2. `Help F1`, into `⋯` (the footer's `F1 keys` remains)
  3. `Chat settings…`, into `⋯`
  4. label words (`Workspace:`, `files:`)
  5. the access word's compact form (`read-write` → `rw`, `read-only` → `ro`, `private scratch` → `scratch`)
  6. the server name (`Server: nas` → `Server`)
  7. the `▾` markers on the workspace and storage segments, which stay activatable
  8. the workspace name, with a middle ellipsis

  The ADR-068 playback lifecycle (`Speech paused · Retry · Resume`, about 30 cells) is recovery state and never folds into `⋯`. When the row cannot hold it after step 8, the header takes a second row until the user resolves it, and the transcript gives up that row.
- **Tab strip:**
  1. key labels on the create buttons
  2. inactive titles, to 12 cells
  3. `Terminal` and `+ Temporary`, into `+ New ▾`
  4. inactive tabs, into the counted overflow, which keeps the highest attention glyph
  5. the active title, to 16 cells
- **Status strip:**
  1. Flags, into `+N ▾` (except `Hands-free on`)
  2. Context, to its compact form
  3. Model, to its short id
  4. Run, which compacts but never drops: `◆ Waiting for you · 1 call · Alt+A` → `◆ Waiting · 1 call · Alt+A` → `◆ 1 · Alt+A`
- **Composer:**
  1. the consequence text, which moves to the footer
  2. the key label on `■ Stop` (the footer keeps the stop chord)
  3. `Dictate ▾`, into Menu ▸ Voice, unless a voice turn is active
  4. `Redirect`, into `■ Stop ▾`
  5. `Queue`, into Enter

  The draft keeps at least 55% of the row or 32 cells, whichever is larger.
- **Footer** ranking, highest first:
  1. state-critical keys (Alt+A, the stop chord, setup Enter)
  2. the focused region's own keys: Enter/Esc, plus Ctrl+J and `/` in the composer
  3. rail keys, while a rail is hidden (an open rail's head shows its key)
  4. Ctrl+K and Ctrl+T
  5. F6
  6. palette
  7. focus mode

  Hints are fitted in rank order, and fitting stops at the first hint that does not fit. `F1 keys` is always kept, last.
- **Inspect,** under height pressure:
  1. Selected turn collapses first.
  2. Run's Changes and Tasks fold to one line.
  3. Next send collapses only by user action.

**Rails** (amends ADR-043; implemented by TASK-33625):

| Condition | Rendering |
|---|---|
| Both rail minimums, a 60-column transcript and two dividers fit (30 + 34 + 60 + 2 = 126 columns) | Both requested rails render beside the transcript, at 34 / 45 by default and shrinking to 30 / 34 |
| Only one rail fits: Chats needs 92 columns (30 + divider + 60 + the Inspect handle), Inspect 96 (34 + divider + 60 + the Chats handle) | The requested rail renders beside the transcript. If both are requested, the most recently opened wins and the other shows its handle. |
| The requested rail does not fit beside the transcript | An explicit open becomes a **pane swap** (open question 4). The rail replaces the transcript view in the centre column, with `‹ Back to chat  Esc`. The tab strip, deck, strip, composer and footer stay. It uses the same widgets and focus model, so it is not ADR-043's rejected overlay. Alt+A and activating the Run slot close the swap first, so a pending approval card is never left behind it. |
| No explicit toggle | Chats is open at ≥ 100 columns when it fits, and Inspect is closed. The 118–128 auto-open band is retired, so a rail never disappears as the terminal widens (G4-35). |
| Every width | ±3-column hysteresis. Collapse never writes the preference. A collapsed rail always shows a handle: labelled at ≥ 100 columns, 1 column wide below that. |

**Height:**

| Rows | Rule |
|---|---|
| ≥ 35 | The nav keeps its 3 rows, with rules above and below the grid. |
| < 35 (`CONSOLE_COMPACT_HEIGHT_ROWS`) | <ul><li>The nav is 1 row (open question 2).</li><li>There are no rule rows; separation is tonal.</li><li>The composer grows to 3 rows, then scrolls internally.</li><li>Tips and coachmarks appear only as toasts.</li><li>The approval card has no blank rows.</li><li>Inspect bodies are at most the viewport minus 4.</li></ul> |

**Transcript targets** with nothing staged:

| Size | Target | Share | Without the compact nav |
|---|---|---|---|
| 80x24 | ≥ 18 rows | 75% | ≥ 16 |
| 100x30 | ≥ 24 rows | 80% | ≥ 22 |
| 120x40 | ≥ 30 rows | 75% | — |
| 235x52 | ≥ 42 rows | 81% | — |

Each non-empty deck row costs one transcript row.

**Focus mode** (ADR-071) hides the nav and the header. While it does:

- the Library chip and a pending Hooks count render in the status strip after Run and never fold; Flags and Context fold into `+N ▾`, and Run compacts, before they would clip;
- the active tab carries its workspace and access word as a prefix (`acme-api · rw /`), so where writes land stays on screen;
- the tab strip, deck, strip, composer and footer remain.

## Where every moved control goes

Twins marked (new) do not exist today and ship with the step that moves the control.

| Control | Old home(s) | New home | Key / palette / slash |
|---|---|---|---|
| Title "Console" and the purpose line | DestinationHeader | The nav's active item; the purpose moves to the F1 intro and the first-run empty state | F1 |
| Ready/Running pill | DestinationHeader | This tab: the Run slot (from step 5). Other tabs: tab markers. | — |
| Speak replies switch | Header | `Dictate ▾` ▸ Speak replies: Off/On/Paused (consent kept); the `Speaking replies` flag | Palette (new) |
| Hands-free switch | Header | `Dictate ▾` ▸ Hands-free: Off/Starting…/On (consent per TASK-33626.1); the `Hands-free on` flag | Ctrl+Shift+H, palette (new) |
| Auto-speak `Retry speech` / `Resume auto-speak` | Header row | Header lifecycle: `Speech paused · Retry · Resume`, on a second header row when it does not fit | Palette (new) |
| New tab | Control bar; tab strip | Tab strip `+ New chat` | Ctrl+T, /new, palette |
| Settings / Conversation settings / Configure | Control bar; rail Model; Inspect | Header `Chat settings…` (in `⋯` below 100 columns) | /settings, palette |
| Context rail | Control bar | Chats head or handle | Alt+C, palette |
| Search Library | Control bar; Inspect button | `Menu ▾` ▸ Add; Next send ▸ Sources | /library (new), palette (new) |
| Help | Control bar | Header `Help F1` (in `⋯` below 100 columns); footer `F1 keys` | F1 |
| Hooks (ADR-197) | Control bar | Unchanged: the header action cluster; in focus mode, a strip item while a review is pending | Palette (new) |
| `Conversation \| <title>` row | Top of the transcript | Removed; the tab owns the title | Ctrl+K |
| Rename chat | Row menu | Chat menu (the active tab's `▾`, or a row) `Rename…` | Palette (new) |
| `New tab` / `Temporary` inside the tab scroll | Tab strip | Pinned cluster; `+ New ▾` when narrow | Ctrl+T, /temp |
| Tab overflow ◂ ▸ | Tab strip | Counted overflow `+N ▾` that carries hidden attention | Ctrl+K Active |
| "Each tab runs its own agent…" banner | Top of the transcript | One-time toast at the second concurrent run; F1 "Tabs" | F1 |
| Fleet line "N other agents running" | Left rail | Tab markers, counted overflow, nav `!`; ADR-132's preview in Run ▸ Sub-agents | Alt+A (cross-tab, new), Ctrl+K Active |
| Terminal | Rail top | Tab strip `Terminal: <state>` (in `+ New ▾` when narrow). When locked it opens the in-Console locked card (G2-12). | /terminal (new), palette |
| Workspace row and Switch | Rail pinned strip | Header workspace `▾` ▸ Switch… (the route to Everyday chats) | Alt+W, /workspace |
| New workspace | Rail pinned strip | Workspaces group header `+ New`; workspace `▾` ▸ New workspace… | Palette |
| RAG scope | Rail pinned strip | Next send ▸ Sources `Scope:` | — |
| Show Files | Rail pinned strip | Workspace `▾` ▸ Files… | Palette (new) |
| Search workspaces; Filter visible titles and Clear; Character "Search chats" | Rail | One `Find chats…` field (Esc clears) | Ctrl+K, /sessions |
| Search all chats… | Rail; Ctrl+K | Ctrl+K History, prefilled; rail footer `All in Library ↗` | Ctrl+K |
| Archived chats | Rail; Ctrl+K | Rail footer `Archived n…` | Ctrl+K |
| Archive this chat | Conversations | Chat menu, disabled with a reason until the chat is saved | Palette (new) |
| Current-chat line | Conversations | Removed; the active row is marked `›` and bold | — |
| `New conversation` (worker alias) | Conversations | Removed; `+ New chat`, or workspace `▾` ▸ New chat here | Ctrl+T |
| Row `💬` icon | Rail rows | Unchanged: ADR-171's menu control, showing the attention glyph or the chat's own icon (custom, or 💬) | `m` on a focused row |
| Export .md (under Copy as ▸) | Row menu | Chat menu `Export…` | — |
| Character groups and search | Character section | Characters group (ADR-120 bounds); search moves to `Find chats…` | Ctrl+K Character chats |
| Character portrait | Character section | Characters group; when it shows and how tall it grows is open question 3 | — |
| Reaction… | Character section | Next send ▸ Persona `Reaction…` | Palette (new) |
| Open Roleplay | Character section | Characters group footer `All in Roleplay ↗` | ⌃4 |
| Temperature, Max tokens | Model section | Next send ▸ Model | Alt+M |
| System: none ▸ | Model section | Next send ▸ System `Edit…`; a strip flag when custom | /system |
| Retry save, Retry default save, Discard retry, Refresh running app, Dismiss | Model section | A recovery callout at the top of Next send (in every tab for a default save); a warning word in the Model slot | — |
| Agent status | Agent section | Run slot; Inspect Run state line | — |
| Agent steps | Agent section | Run ▸ Steps | — |
| Cancel all agents | Agent section | Run ▸ Sub-agents `Stop all`; `■ Stop ▾` ▸ Stop all sub-agents | Palette (new) |
| Back (drill-in in the other rail) | Agent section | In-place drill-in inside Run ▸ Sub-agents, with `Back` | Esc (new) |
| `Steer this sub-agent…` input | Agent section drill-in | Run ▸ Sub-agents drill-in | /steer |
| View full log | Agent section | Run `Run log…` | Palette (new) |
| Progress: n queued | Agent section | Run ▸ Sub-agents `n reports waiting`, only when n > 0, including with no live sub-agents | — |
| Storage, Sync, Local file tools, Server features, Handoff | Details section | Header `Local ▾` details; Settings ▸ Storage | — |
| Default Persona… | Details section | Workspace `▾` ▸ Workspace settings… | — |
| Choose folder · Project | Top of Inspect | Next send ▸ Project `Choose folder…` | — |
| "What happens if I send now?" | Pinned card | The pinned Next send header and its rows. Where moves to the header and tab; Run and Approvals move to the strip. | — |
| Environment, Refresh, Review & commit…, Review in Change Review, Push ↑n…, PR `Open in browser` / `Add to chat`, CI `Fix — add failure summary to chat` | Environment | Run ▸ Changes | — |
| Tasks | Tasks | Run ▸ Tasks | — |
| Agents, Run history ▸, View all runs | Agents | Run ▸ Sub-agents (when there are sub-agents or waiting reports); `All runs…` | — |
| Sources tray (raw fields) | Sources | Deck (✕ per item); Next send ▸ Sources (raw fields behind `Details`) | — |
| Ask Library input and button | Below the tray | `Search Library…`, then `Last search: <q>` | /library (new) |
| Scope row | Inspect | Next send ▸ Sources `Scope:` | — |
| Run recipe, Live work | Run section | Retired in step 4; replaced by the Run state line | — |
| Provider rows | Run section | Retired in step 5; replaced by the Model slot's warning word | — |
| Source Readiness | Run section | A Sources recovery line, only when blocked | — |
| More ▸ Tools / Approvals / Artifacts | More | Next send ▸ Tools; Run waiting line; Run ▸ Artifacts | Alt+A (cross-tab, new) |
| Selected Conversation (source, resume state) | Run section | Header `Local ▾` details; its Prefill rows move to Next send ▸ Prefill | — |
| Prefill | Inspect | Next send ▸ Prefill | /prefill |
| Chat Dictionaries, World Books | Inspect | Next send ▸ Lore `Attach…` / `Detach…` | — |
| Conversation settings block (11 rows) | Inspect | Header `Chat settings…`; verification time on Next send ▸ Model | /settings |
| Selected-turn activity | Inspect | Selected turn group | j/k |
| "Live work sources" card | Bottom of Inspect | Settings ▸ Diagnostics; a Next send ▸ Tools line when degraded | Palette (new) |
| Pending Console launch card | Inspect | A deck row with `Review…`; Next send ▸ Handoff | — |
| Handle badge "setup" | Inspect handle | Replaced by the Run slot's `Setup needed…` | — |
| `Status ▾` / `Status ▴` | Strip | Retired with `status_chips_collapsed`, which step 5 reads once and drops. The collapsed form ("Status hidden") was already one row, the four-slot strip is one row, and hiding it would hide the Run slot (safety set). Placement stays in Settings. | — |
| Temporary chip and Save | Strip | Tab `Temp ·`; flag `Temporary — not saved · Save` | Palette (new) |
| Run chip | Strip | Run slot | — |
| Library chip | Strip | Header, with the same grammar and modal (the strip in focus mode) | Palette (new) |
| Provider and Model chips | Strip | Model slot | Alt+M, /model |
| System Prompt chip | Strip | A flag when custom; Next send ▸ System | /system |
| Assistant chip (opens the character picker) | Strip | A flag when not the default; Next send ▸ Persona `Choose…`, always enabled | Palette (new) |
| Sources chip | Strip | Deck; Next send ▸ Sources | — |
| Tools chip | Strip | Next send ▸ Tools | — |
| Approvals chip | Strip | Run slot `◆ Waiting for you · n calls · Alt+A` | Alt+A (cross-tab, new) |
| Scope chip | Strip | A flag when narrowed; Next send ▸ Sources | — |
| Context/cost chip | Strip | Context slot | Ctrl+Shift+P, /context |
| `Composer ▾` / `Expand ▴` / collapsed Stop | Composer | Retired; focus mode gives the transcript more rows | Ctrl+Shift+F |
| Menu (7–11 tall items) | Composer | `Menu ▾`, grouped, with Add first | — |
| "Send disabled: type a message" | Composer | Removed; the consequence line shows real blockers only | — |
| `Send \| $` | Composer | `Send`; the estimate moves to the Context slot as `next ~$` | Enter |
| `Preparing...` / `Queue full` on the Send button | Composer | The consequence line | — |
| Stop (0 cells today) | Composer | The primary slot while running (TASK-33625.1) | Ctrl+G (decided 2026-09-30), /stop (new), palette (new) |
| Queue | Composer | Shown while running with a draft; folds into Enter | Enter |
| Redirect | Composer | Inline when there is room; otherwise `■ Stop ▾` ▸ Stop and redirect | /redirect |
| Dictate | Composer | `Dictate ▾`; Menu ▸ Voice when folded | Palette (new) |
| 📎 n files and a distant ✕ | Composer | Named attachment chips, each with its own ✕ | — |
| Staged evidence, queue shelf | Above the composer | Unchanged (the deck) | — |
| Draft shelf | Prompt Workbench | Unchanged (ADR-184 :13-14); reached from Menu ▸ Draft | Palette ("Save draft to shelf…", "Insert prompt…") |
| Fixed hints led by Ctrl+Shift+F | Footer | Ranked by region and state from the registry | — |
| `Y trace` | Footer | Shown only while the transcript is focused. Visible homes: Run `Trace…` and Selected turn `Trace this turn…`. | y |
| Conversation Inspector modal | Overlay | Renamed Chat details; opened from `Preview…`, the Context slot and `Details…` | Ctrl+Shift+P, /context |

## Open questions (owner decisions)

1. **The stop chord — resolved 2026-09-30: Ctrl+G.** The owner chose Ctrl+G over Alt+X, which needs Option-as-Meta on macOS and otherwise types `≈`, and over Esc Esc, which collides with Esc's existing jobs. Ctrl+G is BEL (0x07), so it reaches the app in every terminal. Its only other binding is the Speech playground, on a different screen. PR #2934 (TASK-33625.1) ships it as the single constant `STOP_RUN_KEY`. Stop is also reachable by Tab then Enter from the draft, by `/stop`, and from the palette.
2. **A 1-row nav below 35 rows for every destination.** Accepting ships migration step 8 as a shell-wide amendment to ADR-015, and the transcript targets are the "Target" column. Declining keeps the 3-row nav, the targets fall back to the "Without the compact nav" column, and step 8 does only its cleanup.
3. **The character portrait.** Either it shows only while the active chat is a character chat, or it is always present with a fixed ceiling. ADR-083's 2026-09-10 amendment grows the portrait with the viewport at the user's request, and that growth causes GAP5-22. Either option supersedes that amendment for the Characters group; keeping the amendment leaves GAP5-22 open.
4. **Pane swap or side by side.** Below the rail budget, this ADR swaps the rail into the centre column, so the transcript never drops below 60 columns but the two are not visible together. ADR-043 today honours an explicit toggle side by side at any width by waiving the transcript minimum. Which should win?
5. **How the header names Default.** Proposed: `Everyday chats ▾`, which follows ADR-027's rule against workspace vocabulary and matches the switcher's gloss; at 80 columns it drops its `▾` markers (header step 7). The alternatives are keeping `Default ▾`, or omitting the workspace segment in Everyday chats and showing only `private scratch ▾`.

## Consequences

**Positive**

- **Duplicate census, per screen:**
  - chat title: 6 → 2 (the tab plus the rail's row mark)
  - provider/model: 3–4 → one glance plus one detail
  - create controls: 3 buttons → 1 cluster
  - chat search: 5–6 entry points → one field plus Ctrl+K
  - Inspect: about 15 sections and 90 rows → 3 groups and 23 rows in the mid-run mock
  - status strip: fits 80 columns
- **Transcript share** with nothing staged: 80x24 50% → 75%, 100x30 60% → 80%, 120x40 65% → 75%, 235x52 73% → 81%.
- **Authority is readable before any action.** Workspace, file access, storage/server, Library policy and hooks are in one row (PRODUCT.md principles 2 and 7).
- **One place to be wrong.** Every region renders a shared projection, and the fact-placement census fails any restatement.
- **F6 and Tab follow the visual layout** (G4-42).

**Negative**

- **Relearning.**
  - Model, Agent and Details leave the left rail.
  - Terminal moves to the tab strip.
  - Alt+C opens "Chats".
  - F1 carries a one-release "What moved" section.
- **Model settings are one step further away** when Inspect is closed: Alt+M or `Chat settings…`.
- **Below 92 or 96 columns, a rail replaces the transcript** instead of squeezing it, so the two cannot be read side by side (open question 4).
- **The header is dense at 80 columns.** File access compacts to `rw`, `ro` or `scratch`, and in Everyday chats the header drops its `▾` markers.
- **The Console header becomes a documented exception to DESIGN.md:311,** which asks for a title, a one-line purpose, readiness, authority, a primary action, blocked recovery, `border: tall` and `padding: 1 2`. The Console header keeps authority and one recovery case (the playback lifecycle) in one unpadded row. The title is the nav item and the tab, the purpose moves to F1 and the empty state, readiness is the Run slot, the primary action is the composer's primary slot, and run recovery is the Run slot plus the turn's callout.
- **The pinned "What happens if I send now?" card, which the review praised, becomes the pinned Next send header.**
- **Two cost projections.** The strip's `next ~$` and Next send's Total can differ in the last digit; rule 2 makes Total authoritative.
- **Test churn:**
  - task-400's `test_console_live_work_card_swap_keeps_tray_on_top_and_cards_at_bottom`
  - TASK-24611's boundary inventory
  - ADR-077's STRICT ownership map
  - the `CONSOLE_RAIL_SECTION_IDS` preference migration
  - `status_chips_collapsed`
  - `TOP_ACTION_IDS`
  - `CONSOLE_TAB_REGIONS`
- **A shell-wide compact nav** (open question 2) changes every destination below 35 rows.

## Migration plan

**Before step 1:**

- Each step names the sibling tasks it needs, and those land before that step, not before step 1.
- TASK-33163 (Hooks) is independent. It rides in ConsoleControlBar and is unaffected.

**Every step:**

- ships alone;
- moves no control before its new home exists (rule 7);
- flips its own rows of the fact-placement census and never-clip sweep from report-only to enforcing;
- updates the matching `Docs/User_Guide/console/` pages;
- adds geometry tests at 80x24, 120x40 and 235x52;
- adds a live-stack test, with no stubbed seam, for every user action it moves;
- puts new code in `UI/Console_Modules/`, without raising the screen-size ratchet.

0. **Accept and measure.** No visible change.
   - The owner accepts and answers the open questions.
   - Add the fact-placement census and never-clip sweep as report-only tests, recording today's baseline.
   - Mark ADR-017 and ADR-083 "Superseded in part by ADR-210", and the amended ADRs "Amended by ADR-210".
1. **The composer owns the run control.** Needs TASK-33620, TASK-33625.1 and TASK-33626.1.
   - Add the consequence line, including `Preparing…` and `Queue full`; drop "Send disabled" and the `| $` suffix.
   - Add `■ Stop ▾` with Redirect and Stop all sub-agents.
   - Add `Dictate ▾` with the two speech switches and their state words. The header switches go in the same PR. Until step 5 adds the `Hands-free on` flag, `Dictate ▾` does not fold while Hands-free is on.
   - Regroup `Menu ▾` with Add first, including `Search Library…` and a new `/library`.
   - Retire `Composer ▾`.
2. **The header owns authority.** Needs TASK-33620 and TASK-33623.
   - Add the workspace segment and menu, the folder and access word, and the `Local ▾` details: the left rail's Details content, this chat's source and resume state, and the database sizes.
   - Move the Library chip from the strip in the same PR, so ADR-079's guarantee never lapses.
   - Reduce ConsoleControlBar to `Chat settings…` · `Hooks` · `Help F1`, and bring the playback lifecycle into the header row.
   - Move the purpose line to F1.
   - Keep the header status pill. Today's run chip is hidden at idle and terminal states, so until step 5 the pill is the only persistent home of Ready, Failed, Stopped and Held.
3. **The tab strip owns identity.** Needs TASK-33624 and TASK-33625 AC#3.
   - Remove the `Conversation |` row.
   - Add the pinned create cluster with `Terminal: <state>` and a new `/terminal`. The rail's Terminal and fleet line go in the same PR.
   - Add the `Temp ·` prefix, workspace prefixes, the active tab's chat-menu `▾`, and a toast on a workspace-changing tab click (GAP4-18).
   - Make overflow counted and attention-carrying.
   - Turn the banner into a toast.
4. **Inspect gets three groups.** Needs TASK-33620 and step 2, whose `Local ▾` details receive Selected Conversation's source and resume rows.
   - Build Next send (on the exact-context preview Chat details ▸ Context already uses, run only while Inspect is open), Run and Selected turn, with one header grammar and group disclosure ids.
   - Replace the Library input with `Search Library…`.
   - Put raw fields behind `Details`.
   - Retire the Run recipe and Live work rows, which the Run state line replaces. Keep the Provider rows until step 5.
   - In the same PR, move the Live work sources card to Settings ▸ Diagnostics and the pending launch card to the deck and Next send ▸ Handoff.
   - Rename the Conversation Inspector modal to Chat details.
   - The left rail's Model and Agent sections stay until step 6; the PR states the known duplication.
5. **The status strip gets its four slots.** Needs steps 2 and 4.
   - Fold Approvals into Run, and merge Provider and Model. Retire Inspect's Provider rows, now that the Model slot's warning word carries readiness.
   - Remove the Tools, Sources and Scope chips, and make System and Assistant flags. Choose character already lives in Next send ▸ Persona from step 4.
   - Add `+N ▾` and the `Hands-free on` flag.
   - Remove the header status pill, now that the Run slot renders ConsoleRunPresentation in every state.
   - Retire `Status ▾` and `status_chips_collapsed` (read once, then dropped). The placement setting stays.
6. **The left rail becomes Chats.** Needs steps 2–5.
   - Rename the rail and add the single `Find chats…` field.
   - Organise it into the Workspaces, Everyday chats and Characters groups.
   - Remove Model, Agent, Details, the current-chat line, the worker alias, the workspace strip and Archive-this-chat.
   - Add the rail footer and the portrait rule (open question 3).
   - Migrate `CONSOLE_RAIL_SECTION_IDS` losslessly.
7. **Width, height and focus.** Needs TASK-33625 AC#4 and TASK-33622.
   - Add the combined rail budget with pane swap and hysteresis, and retire the auto-open band.
   - Show handles at every width.
   - Apply the height contract below 35 rows.
   - Rewrite `CONSOLE_TAB_REGIONS` as the five-region F6 ring.
8. **Compact nav and cleanup.** The nav needs open question 2; the cleanup ships either way.
   - Build the 1-row nav below 35 rows, if accepted.
   - Remove the hidden `#console-title`, `#console-purpose`, `#console-status-row` and `#console-mode-bar` statics and their dead CSS.
   - Add the Console header variant and region map to DESIGN.md.
   - Make the whole fact-placement census and never-clip sweep enforcing.

## Validation

Measure live, before and after each step, with the review's B2 SGR census on tmux captures:

- **Transcript rows** at 80x24, 100x30, 120x40 and 235x52 against the targets above, with 0 and 1 deck rows.
- **Fact census on the rendered screen:**
  - chat title ≤ 2 (tab plus row mark);
  - provider/model: 1 persistent plus 1 detail;
  - create controls: 1 cluster;
  - chat search: 1 field plus Ctrl+K;
  - run-state words: the Run slot plus Inspect Run only;
  - raw payload fields in Inspect: 0.
- **Never-clip sweep:**
  - every width from 80 to 235 columns;
  - states: idle, running, waiting, failed and held;
  - rail configurations: none, Chats, Inspect and both.

  Every safety-set item must be fully painted, with no mid-token cut in any horizontal region.
- **Contradiction probe** (the G1-04 method): read the tab marker, Run slot, Inspect Run, composer primary slot and footer at the same instant. Cases: a healthy run, an approval, a 401, a 429, a 504 and an emergency stop. Target: 0 disagreements.
- **Inspect:** 40 rows or fewer mid-run; one header grammar (enforced by lint); `Search Library…` within the first 12 rows of Next send.
- **Keyboard only:**
  - stop a run: the stop chord, or Tab and Enter from the draft;
  - approve the oldest pending call: Alt+A, then Enter;
  - reach a waiting background tab: Alt+A;
  - stage Library sources: `/library`, Enter;
  - change model: Alt+M;
  - open Inspect: Alt+I.
- **Focus:** the F6 order matches the visual order; Tab never leaves a region's visual order; no container is a Tab stop.
- **Registry:** no action is palette-only; the footer advertises only bindings that work in the focused region (ADR-031 rule 4).
- **Glance test** with three operators and three first-timers, on the live mid-run and idle first-run screens at 80x24 and 235x52. The five questions:
  - Is anything waiting on you?
  - Where will this write land?
  - What will Enter send?
  - What will the next send cost?
  - Where is last week's chat?

  Target: all five answered correctly within 10 s, without opening a panel. Operators' ten most frequent actions cost at most one extra keystroke each.

## Alternatives considered

**Five-Second Console (first-timer clarity).**

- *Design:*
  - The header owns the run-state word and a state action.
  - The strip shows only cost, fullness and modes that deviate from the default.
  - The composer's Menu splits into `+ Add` and `⋯ More`.
  - The header gains `Workspace ▾`, `Local ▾` and `Chat ▾` menus.
- *Adopted from it:* the empty state, the `Local ▾` details menu, the computed rail budget with pane swap, and the fact-placement census.
- *Why it was not chosen:*
  - Run state in the header is hidden by focus mode and sits farthest from Stop, so it needed a second home in focus mode.
  - Library policy appears only when auto-search is on, which breaks ADR-079's always-visible chip.
  - Its 235x52 mock approves a write inside a folder it binds read-only, which is an impossible authority state.
  - It adds the most new widgets of the three.

**Stable Ring (keyboard and small terminals).**

- *Design:*
  - One session bar merges the header and the tab strip.
  - Run state lives only in the strip.
  - Decision cards dock at the bottom of the transcript.
  - Rails pane-swap below 100 columns.
  - A safety set and a five-region focus ring.
- *Adopted from it:* the focus ring, the safety set (including a capturing microphone and the not-saved state), cross-tab Alt+A, pane swap and the keystroke targets.
- *Why it was not the base:*
  - The merged bar crams workspace, tabs, creation, authority and actions into one row. The active title gets 22 cells at 80 columns, and DESIGN.md's header/mode-bar distinction disappears.
  - It saves no transcript row at 80x24, because it adds a rule.
  - It demotes Library policy to a non-default flag (ADR-079).
  - It folds the queue into a chip at 30 rows or fewer (ADR-098 queue).
  - It moves approval cards, which redesigns ADR-195's decision surface as a side effect.

**Run state in the header** (the synthesis starting point, `Workspace › Chat · Local · state`).

- It restates the title the tab owns.
- It puts this tab's state in the row focus mode hides, farthest from Send and Stop.
- It splits "what is happening" from "what I can do about it".

The strip's Run slot sits directly above the composer's primary slot.

**Deduplicate within today's regions.** TASK-23196 removed one provider/model copy, and the review found two more that survived, because each region kept its own job list. Removing copies one at a time leaves five jobs in the left rail and about fifteen sections in Inspect.

## Restates task decisions

- **TASK-23196:**
  - AC#1 "provider and model appear once" becomes rule 2: one persistent glance (the Model slot) plus one detail (Next send ▸ Model). The run recipe and the 11-row settings block, which kept two more copies (G4-55), are removed.
  - AC#2 is superseded. The Model section leaves the rail, its parameters move to Next send ▸ Model (showing only the parameters actually sent), and Configure becomes `Chat settings…`.
  - The :31 survivor check must be redone at 80 columns, where the Model slot shows the short id.
- **TASK-24611:**
  - AC#1 is realised by the group order Next send → Run → Selected turn.
  - AC#2: the Session Settings block becomes the header's `Chat settings…`. Changed files already retired under ADR-089.
  - AC#3: `Search Library…` sits within Next send's first 12 rows (its 10th row, counting the group header, in the mock).
  - AC#4: group disclosure persists through ADR-083's layout scopes.
  - The owner's option C and task-400's bottom anchoring of the readiness card are superseded. The card moves to Settings ▸ Diagnostics, and `test_console_live_work_card_swap_keeps_tray_on_top_and_cards_at_bottom` retargets to that home.
- **TASK-33620.1:** AC#1 "the header shows a hold state" is delivered by the Run slot's `Held — Clear` once migration step 5 removes the header status pill. Until then the task lands as written.
- **TASK-33626.1:** its state-word and consent ACs follow the speech switches into the composer's `Dictate ▾` in migration step 1. Where `Dictate ▾` folds, the `Hands-free on` flag (step 5) keeps AC#1's state on screen.
