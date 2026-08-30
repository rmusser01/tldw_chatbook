# tldw_chatbook User Guide

tldw_chatbook is a terminal (TUI) app for working with LLMs: chat with local
or cloud providers, manage conversations/notes/media in a local library,
run roleplay characters with lorebooks, schedule watchlists, and drive
agent/tool workflows — stored locally in SQLite by default (some surfaces
can sync with a tldw server you configure).

> This guide tracks the **dev** branch. Each fully written page carries a
> "Verified against dev @ `<sha>`" stamp (stub pages are marked 🚧 instead);
> if your build is older, screens may differ slightly.

## Quick start — your first five minutes

1. Install and launch — see the [README](../../README.md#installation). On a
   brand-new install the app opens on Home underneath the
   [first-run setup wizard](First_Run_Setup.md), which offers to do step 2
   for you; skipping it leaves you on [Home](home.md), and Console's
   composer stays locked with a "Get started" card until a provider exists.
2. Open **[Settings](settings.md)** — press **F9**, click **F9 Settings** in
   the nav bar, or **Ctrl+P** → "Tab Navigation: Switch to Settings" — and
   set a provider + model (or point at a local server) under **Providers &
   Models**.
3. Open **[Console](console.md)** — press **Ctrl+2**, click **⌃2 Console**
   in the nav bar, or **Ctrl+P** → "Tab Navigation: Switch to Console"; then
   send your first message.
4. Press **F1** anywhere to open the current screen's keyboard-shortcuts
   list; **Ctrl+P** opens the command palette.

## The screens

| Hotkey | Screen | What it's for |
|-----|--------|----------------|
| Ctrl+1 | [Home](home.md) | Triage snapshot: what needs attention, what's running, what's recent, and a suggested next action. |
| Ctrl+2 | [Console](console.md) | Live agent conversations, approvals, tools, RAG, and runs. |
| Ctrl+3 | [Library](library.md) | Source material, imports, notes, media, conversations, prompts, skills, Search/RAG — plus hand-offs to Study for flashcards and quizzes. |
| Ctrl+4 | [Artifacts](artifacts.md) 🚧 | Generated outputs, bundles, reports, datasets, and Chatbooks. |
| Ctrl+5 | [Roleplay](roleplay-chat-dictionaries.md) | Characters, personas, chat dictionaries, and lore/world books. |
| Ctrl+6 | [Watchlists](watchlists.md) 🚧 | Monitored sources, runs, alerts, and recovery. |
| Ctrl+7 | [Schedules](schedules.md) 🚧 | When jobs, watchlists, and workflows run. |
| Ctrl+8 | [Workflows](workflows.md) 🚧 | Reusable procedures, recipes, dry-runs, and outputs. |
| Ctrl+9 | [MCP](mcp.md) 🚧 | MCP servers, tools, permissions, auth, and audit. |
| Ctrl+0 | [ACP](acp.md) 🚧 | Agent Client Protocol agents, sessions, runtimes, diffs, and terminals. |
| F7 | [Lab](lab.md) 🚧 | Models, speech, and evaluation runs. |
| F8 | [Logs](logs.md) 🚧 | Application logs and diagnostics. |
| F9 | [Settings](settings.md) | Global app preferences, appearance, accounts, and storage. |
| F10 | [Research](research_workspace.md) | Authority-explicit research workspaces, plus navigation to durable Research Runs. |

Lab, Logs, Settings, and Research sit past the ten digits, so they get function
keys instead: **F7**, **F8**, **F9**, **F10** — the nav labels say so ("F7 Lab",
"F8 Logs", "F9 Settings", "F10 Research"). The nav bar and the command palette
(**Ctrl+P**) reach them
too.

Two more screens exist with **no nav label and no "Tab Navigation" palette
entry**: **Study** (flashcards and quizzes — reached from
[Library](library.md), e.g. **Continue in Study**) and **Statistics**
("Settings & Preferences: Show Database Stats"). Typing "study", "media", or
"search" into the palette surfaces the **Library** command — those words are
aliases for Library, not entries of their own. The command palette's "Media
& Content: Open Media Library" and "Quick Actions: Search All Content"
entries are deep links into Library's Media and Search/RAG rows, not
separate screens.

## How-to guides

| Guide | What it covers |
|-------|----------------|
| [Turn feeds into a scheduled Watchlist briefing](watchlists-quickstart.md) | A start-to-finish Console walkthrough: create feeds and a Watchlist, follow receipts, generate a briefing, schedule it every 24 hours, and verify the saved result. |
| [Using OpenAI-compatible TTS servers](openai-compatible-tts.md) | Pointing text-to-speech at your own server (e.g. a local, keyless engine like pocket-tts) via Settings ▸ Speech & TTS; also covers the app-wide default voice profile and per-character voices. |

**Note:** The "⌃\<digit\>" (or "F\<n\>") shown before each nav label is
that screen's hotkey: press **Ctrl+digit** (Ctrl+1 … Ctrl+9, Ctrl+0) — or
**F7** / **F8** / **F9** for the last three — to switch to it from anywhere;
the keys work even while a text field has focus. Bare digit keys are not
navigation shortcuts (typing `2` in the composer just types "2"). Clicking
the nav label and **Ctrl+P** work everywhere too.

One screen claims some of these digits for itself: on
[Roleplay](roleplay-chat-dictionaries.md), **Ctrl+1 –
Ctrl+4 switch that screen's four modes** instead of changing screens.
Ctrl+5 … Ctrl+0 still navigate from there, as do the nav bar and
**Ctrl+P**.

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| F1 | Open the current screen's keyboard-shortcuts list (content is screen-specific) |
| Ctrl+P | Open the command palette — search and jump to any screen or command from anywhere |
| Ctrl+Q | Quit the app |
| Ctrl+1 … Ctrl+9, Ctrl+0 | Switch to the screen with that hotkey digit (see the nav map above). Exception: on [Roleplay](roleplay-chat-dictionaries.md), Ctrl+1 – Ctrl+4 switch that screen's modes instead |
| F7 / F8 / F9 | Switch to Lab / Logs / Settings — the three destinations past the digit row; they work while a text field has focus, like the Ctrl+digit chords |
| F6 | Cycle through the current screen's panes; on screens without a pane cycle it only shows a notice |
| Shift+F6 | Cycle panes backward — bound only on [Console](console.md) and [Roleplay](roleplay-chat-dictionaries.md); elsewhere it does nothing |

Everything else (Enter/Ctrl+K/Ctrl+T in Console, and the single-letter
mnemonics like `s`/`r`/`t` on Settings) is screen-specific — see that
screen's own page for its "Keyboard & commands" table.

## Console agent runs are screen-scoped

Agent **turns** you start in Console — and any approval/confirmation
they're waiting on — live only as long as the Console screen itself stays
mounted. Leaving Console for another screen (e.g. Settings, Ctrl+1…Ctrl+0,
or the command palette) cancels every in-flight turn and denies every
pending or parked approval for that visit; coming back starts a fresh
Console. One thing is deliberately **not** screen-scoped: a background
sub-agent that already outlived its spawning turn keeps running through
the leave — its result lands durably in the run log, its completion
raises a toast on whatever screen you're on plus a durable `◈` marker,
and the supervisor's auto-wake is staged and claimed when Console next
mounts (see [Console ▸ Agent runs &
tools](console/agent-runs-and-tools.md)). Guards make all of this visible
instead of silent:

- **Before you leave:** if any run is still in flight or waiting on an
  approval, a confirmation dialog asks "N agent runs will be cancelled if
  you leave Console. Leave anyway?" — **Leave** proceeds, **Stay** keeps
  Console (and the fleet) exactly as it was. An idle Console never shows
  this prompt.
- **After you return:** the next Console mount reports each fate
  truthfully, one-time: "N agent runs were cancelled when you left
  Console." for the turns the teardown killed, and "… sub-agents kept
  running in the background when you left Console — you'll be notified
  as they finish." for the survivors it spared — so neither a lost run
  nor continuing background work is ever silently unexplained.

Nothing is ever auto-approved: an approval that gets caught by this
teardown is always denied, never resolved on your behalf — and an
auto-wake can never resolve one either.

Full detail on runs, approvals, and tools:
[Console ▸ Agent runs & tools](console/agent-runs-and-tools.md).

## Where did … go? (legacy names)

| Old name | Now lives in |
|----------|--------------|
| Notes | [Library ▸ Notes](library/notes.md) |
| Prompts | [Library ▸ Prompts](library/prompts.md) |
| Skills | [Library ▸ Skills](library/skills.md) |
| Subscriptions | [Watchlists](watchlists.md) 🚧 |
| Coding | [Console](console.md) |
| Conversations | [Library ▸ Media & conversations](library/media-and-conversations.md) |
| CCP (Conversations, Characters & Prompts) | [Roleplay](roleplay-chat-dictionaries.md) for characters and personas; prompts moved to [Library ▸ Prompts](library/prompts.md) |
| LLM management | [Lab](lab.md) 🚧 |
| Research | [Research Workspace](research_workspace.md) for the workbench; its **Runs** mode preserves the durable run operator. |
| Ingest | [Library ▸ Import & export](library/import-and-export.md) |
| Writing | [Library](library.md) |
| Chatbooks | [Artifacts](artifacts.md) 🚧 |
| Characters / Roleplay | [Roleplay](roleplay-chat-dictionaries.md) |
| Speech (STTS) / Evals | [Lab](lab.md) 🚧 |
| Tools & Settings | [MCP](mcp.md) 🚧 |
| Stats | [Settings](settings.md) (the palette's "Show Database Stats" opens the separate Statistics screen) |
| Customize | [Settings](settings.md) — the Theme editor specifically |

## Conventions

- Keys are **bold** in prose (**Ctrl+P**, **s**) and bare inside tables;
  slash commands are shown as `/rewind`.
- Pressable controls are **bold**; verbatim on-screen text is in "quotes";
  command-palette entries are quoted with their group prefix
  ("Tab Navigation: Switch to Library").
- Breadcrumbs use ▸ (Settings ▸ Providers & Models); config keys are shown
  as `[section]` + `key`; known defects are cited as (task-NNN).
- What happens to unsaved work when you leave a screen **differs per
  screen** — each page's "Quirks" or save-model section states its own rule;
  do not generalize from one screen to another.
- 🚧 marks stub pages awaiting a full write-up.
- Deep dives live in [Docs/Features](../Features/); pages link out rather
  than duplicate them.

—
*Verified against dev @ 6b6c35a4b — 2026-08-06 (TASK-2851: the legacy Media
Library screen is retired — "Media & Content: Open Media Library" now
deep-links into Library's Media row instead of a separate screen)*
