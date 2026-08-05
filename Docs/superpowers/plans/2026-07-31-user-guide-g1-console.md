# User Guide G1 (Console) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the Console section of the User Guide — `console.md` plus six
child pages plus SVG captures — per the approved spec
`Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`, from the
completed live IA survey of dev @ `ff435772c` (2026-07-31).

**Architecture:** Pure-docs change. Branch `claude/user-guide-g1-console`
(worktree `/private/tmp/tldw-guide-g1`), one PR at the user merge gate.
Authoring inputs are the four survey artifacts in the session scratchpad
(`g1_survey.md`, `g1_inventory_shell.md`, `g1_inventory_composer.md`,
`g1_inventory_messages.md`) — code-exact inventories with verbatim on-screen
labels and file:line references, plus the live-walk record. Drafting is done
by subagents from those artifacts; live verification, captures, and stamps
happen in the controller's session (single verification story, single stamp).

## Survey deltas vs the spec (survey wins)

1. **Six children, not five.** The spec's provisional tree folded sessions/
   tabs/workspaces into the parent and voice into nothing. The live IA shows
   the tab strip / Ctrl+K / conversation browser / workspaces cluster is its
   own surface, and dictation belongs with attachments (both are composer
   input paths). Final tree:
   - `console.md` (orientation, layout tour, setup states, session settings
     & model selection, F1 help, leave-guard)
   - `console/chat-basics.md`
   - `console/sessions-tabs-workspaces.md` (NEW vs spec)
   - `console/branching-and-rewind.md`
   - `console/attachments-images-voice.md` (spec's attachments-and-images +
     voice dictation; TTS/speak stays in chat-basics with the 🔊 action)
   - `console/agent-runs-and-tools.md` (spec's agent-runs)
   - `console/context-and-rag.md`
2. **Single PR, not two.** The spec allows a split for review load; declined
   because stacked unmerged PRs would conflict on `index.md` and fork the
   verification stamp. Recorded here as the authorized deviation.
3. **First-run wizard is new since G0** (steps Welcome/Provider/Model/
   Summary; skip lands on Home). G1 mentions it only where Console meets it
   (Get started card); index Quick-Start rewording is G5 scope. An untracked
   `First_Run_Setup.md` exists in the guide tree from another session — G1
   does not touch it.

## Global constraints

- Everything in `_template.md` (section order, authoring rules, capture
  recipe at 200×50, `Verified against dev @ <short-sha>` stamps).
- On-screen labels verbatim from the inventories; no internal jargon.
- Limitations carry backlog refs: 570, 571, 574, 575, 576, plus TASK-222
  (.tiff/.svg picker mismatch). New quirks found during verification get
  fresh backlog tasks (ID collision sweep first, per standing rule).
- Captures: scratch profile `guide_g1` + deterministic stub LLM
  (`stub_llm_server.py`, port 5199, provider "custom", model "guide-demo").
  No personal data; canned demo content only.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Backgrounded shells reset cwd to the main checkout — use absolute paths /
  `git -C /private/tmp/tldw-guide-g1` everywhere.
- Delete `~/.local/share/tldw_cli/guide_g1` when the phase ends.

## Task list

### Task 1: `console.md` (parent) — draft
Author from `g1_inventory_shell.md` + `g1_survey.md`. Sections per template.
Layout tour covers: destination header + status badge, control bar (six
actions), left "Console context" rail (Session / Conversations / Model /
Agent / Details), transcript + tab strip, right "Inspector" rail + handle,
status chips, composer row, footer. Getting there: Ctrl+2 / nav click /
palette. Session settings: "Console Settings" modal (three entry points),
Alt+M popover, "Save as default" semantics. Include the Get-started locked
state and the "Leave Console?" guard + return toast (agent-run scoping is
agent-runs-and-tools.md's deep dive; parent links).

### Task 2: `console/chat-basics.md` — draft
From `g1_inventory_composer.md` (§2 composer keys) + `g1_inventory_messages.md`
(§1 actions) + live walk. Send/stream/stop, [streaming]/[stopped]/[failed]
suffixes, "Response stopped by user.", selection + action row + guide line,
transcript keys, jump-to-latest pill, markdown rendering rule, 🔊/⏹ speak,
Save as… destinations, feedback, two-press delete.

### Task 3: `console/sessions-tabs-workspaces.md` — draft
From `g1_inventory_shell.md` (§2 tab strip, §3a rail, §4 modals). Tabs
(Ctrl+T, ✕ + "Close Tab" confirm, rename via second click, middle-click
close, Alt+1..9), Ctrl+K "Switch Session", conversation browser (search,
Clear, star ☆/★, Starred/Workspaces/Chats groups, "New conversation"),
workspaces (Alt+W "Change Workspace", "New", "Rename Workspace", "Archive
workspace?"), Details section, per-tab draft retention.

### Task 4: `console/branching-and-rewind.md` — draft
From `g1_inventory_messages.md` §2 + live walk. ♻ fork semantics, (n/m)
indicator, < > swipe (deepest-turn landing, pending-selection repeat), Edit
& resend (modal copy verbatim; attachment carry), /rewind (Restore /
Summarize up to here + banner), limitations 570/571/574/575/576 with honest
phrasing.

### Task 5: `console/attachments-images-voice.md` — draft
From `g1_inventory_composer.md` §3-4 + `g1_inventory_messages.md` image
actions. Attach paths (button, paste/drop, Alt+V, rail attach), caps and
config keys, inline-vs-attachment split + toasts, 📎 indicator, ✕ clear-all
(no per-file removal), vision gate, View/Save Image modes, /generate-image
(+ @style picker, variants + keep), Mic dictation (states, chip, first-run
model download toast, 60s cap, extras install copy, no keyboard shortcut).

### Task 6: `console/agent-runs-and-tools.md` — draft
From `g1_inventory_messages.md` §3 + shell §3a Agent section. Runs per tab,
fleet markers + legend + coach-mark, Agent rail (steps, sub-agent
drill-down, Back, View full log + modal), inline ⚙/⤷/⚠ markers (+ task-570
vanish caveat), approval card (all button/option labels), skill install/
script confirm cards, $mention + /skills, MCP surface, parked approvals,
Stop scope, leave-Console cancel semantics ("Nothing is ever
auto-approved").

### Task 7: `console/context-and-rag.md` — draft
From `g1_inventory_messages.md` §4 + composer §1 commands. Ctrl+Shift+P
"Chat Context" (tabs, folds, 1 MiB guard, footer buttons), /prompt, /system
+ editor + rail system line click, /prefill (one-shot vs pin, verbatim
responses, inspector rows, tool-skip note), RAG scope (chip states, row
buttons, picker modal walk), Sources tray, "Run Library RAG" card,
citations ("Sources (N)" modal, repair notices, "Open in Library"),
dictionaries & world books blocks + attach/detach.

### Task 8: Captures + live verification pass + stamps (controller session)
- One overview SVG per page minimum via the `_template.md` recipe (200×50,
  `run_test` pilot): console overview, action-row selected, two tabs +
  coach-mark, sibling (2/2) + Rewind modal, 📎 staged attachment,
  approval card (extend stub to emit a canned tool_call; if that wiring
  fights back, drop this capture and keep prose — do not fake it), Chat
  Context modal or scope picker.
- Execute every page's Common tasks live (tmux session or pilot scripts);
  fix prose to match reality; re-check the two quirk candidates (Edit-modal
  buttons clipped at low heights; /rewind draft residue) and file backlog
  tasks for confirmed ones (collision sweep first).
- Stamp every page `Verified against dev @ <short-sha> — <date>` with the
  sha actually verified against.

### Task 9: `index.md` wiring + link sweep
Console rows/links go live (nav map, Quick Start step 3, legacy "Coding →
Console"), stub notice counts updated if worded, full-guide link sweep
(`BROKEN: []`), template conformance check per page.

### Task 10: Reviews + PR (user gate)
Per-page reviewer subagents (template conformance, jargon, label verbatim
spot-checks against the inventories), then a final whole-branch review
(cross-file seams: parent↔children links, index promises vs pages, spec
done-criteria 1-6). Merge-time drift check: `git -C /private/tmp/tldw-guide-g1
log --oneline origin/dev -- tldw_chatbook/UI/Screens/chat_screen.py
tldw_chatbook/Widgets/Console/ tldw_chatbook/Chat/` since `ff435772c`;
re-verify + re-stamp anything that moved. Push, open PR against dev.
**Do NOT merge — user gate.**
