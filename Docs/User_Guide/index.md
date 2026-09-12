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
2. Open **[Settings](settings.md)** — press **F4**, click **F4 Settings** in
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
| Console → Menu → Buddy | [Buddies](buddies.md) | Manage artwork and Personas; follow and reply to conversations across screens. |
| Ctrl+3 | [Library](library.md) | Source material, imports, notes, media, conversations, prompts, skills, Search/RAG — plus hand-offs to Study for flashcards and quizzes. |
| Ctrl+4 | [Roleplay](roleplay-chat-dictionaries.md) | Characters, personas, chat dictionaries, and lore/world books. |
| Ctrl+5 | [Watchlists](watchlists.md) 🚧 | Monitored sources, runs, alerts, and recovery. |
| Ctrl+6 | [Artifacts](artifacts.md) 🚧 | Generated outputs, bundles, reports, datasets, and Chatbooks. |
| Ctrl+7 | [Schedules](schedules.md) 🚧 | When jobs, watchlists, and workflows run. |
| Ctrl+8 | [Workflows](workflows.md) 🚧 | Reusable procedures, recipes, dry-runs, and outputs. |
| Ctrl+9 | [MCP](mcp.md) 🚧 | MCP servers, tools, permissions, auth, and audit. |
| Ctrl+0 | [ACP](acp.md) 🚧 | Agent Client Protocol agents, sessions, runtimes, diffs, and terminals. |
| F2 | [Lab](lab.md) 🚧 | Models, speech, and evaluation runs. |
| F3 | [Logs](logs.md) 🚧 | Application logs and diagnostics. |
| F4 | [Settings](settings.md) | Global app preferences, appearance, accounts, and storage. |
| F5 | [Research](research_workspace.md) | Authority-explicit research workspaces, plus navigation to durable Research Runs. |
| F7 | [Meetings](meetings.md) | Record a call or a room with a live labelled transcript, then file it in the Library. |

Lab, Logs, Settings, Research, and Meetings sit past the ten digits, so they
continue the walk onto the function-key row from its left end: **F2**, **F3**,
**F4**, **F5**, **F7** (F1 is Help and F6 is Next Pane; F6 is skipped) — the
nav labels say so ("F2 Lab", "F3 Logs", "F4 Settings", "F5 Research",
"F7 Meetings"). The nav bar and the command palette (**Ctrl+P**) reach them
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
| [Set up and use Persona Buddy](buddy.md) | Select Migu, move and resize the companion, use Console voice, handle approvals, and troubleshoot. |
| [Set up and manage your Personal Context Profile](settings/personal-context-profile.md) | Optional interviews, global/workspace context, agent proposals, synchronization boundaries, export, and removal. |
| [Turn feeds into a scheduled Watchlist briefing](watchlists-quickstart.md) | A start-to-finish Console walkthrough: create feeds and a Watchlist, follow receipts, generate a briefing, schedule it every 24 hours, and verify the saved result. |
| [Using OpenAI-compatible TTS servers](openai-compatible-tts.md) | Pointing text-to-speech at your own server (e.g. a local, keyless engine like pocket-tts) via Settings ▸ Speech & TTS; also covers the app-wide default voice profile and per-character voices. |

**Note:** The "⌃\<digit\>" (or "F\<n\>") shown before each nav label is
that screen's hotkey: press **Ctrl+digit** (Ctrl+1 … Ctrl+9, Ctrl+0) — or
**F2**–**F5** / **F7** for the last five — to switch to it from anywhere;
the keys work even while a text field has focus. While a modal dialog
is open it owns the keyboard — destination keys stay inert until it closes
(the Ctrl+K session switcher additionally uses **Shift+F3** for its mode
toggle and **F2** for rename). Bare digit keys are not
navigation shortcuts (typing `2` in the composer just types "2"). Clicking
the nav label and **Ctrl+P** work everywhere too.

No screen borrows these keys for itself: the Ctrl+digit chords and the
F2–F5 / F7 tail navigate from everywhere, including
[Roleplay](roleplay-chat-dictionaries.md) (whose four modes use the single
letters **c / p / d / l**).

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| F1 | Open the current screen's keyboard-shortcuts list (content is screen-specific) |
| Ctrl+P | Open the command palette — search and jump to any screen or command from anywhere |
| Ctrl+Q | Quit the app |
| Ctrl+1 … Ctrl+9, Ctrl+0 | Switch to the screen with that hotkey digit (see the nav map above) — works from every screen, [Roleplay](roleplay-chat-dictionaries.md) included |
| F2 / F3 / F4 / F5 / F7 | Switch to Lab / Logs / Settings / Research / Meetings — the five destinations past the digit row; they work while a text field has focus, like the Ctrl+digit chords (F6 is skipped: Next Pane) |
| F6 | Cycle through the current screen's panes; on screens without a pane cycle it only shows a notice |
| Shift+F6 | Cycle panes backward — bound only on [Console](console.md) and [Roleplay](roleplay-chat-dictionaries.md); elsewhere it does nothing |

Everything else (Enter/Ctrl+K/Ctrl+T in Console, and the single-letter
mnemonics like `s`/`r`/`t` on Settings) is screen-specific — see that
screen's own page for its "Keyboard & commands" table.

<a id="console-agent-runs-are-screen-scoped"></a>
## Console runs continue during navigation

Switching to Settings, Home, Library, or another destination keeps accepted
Console turns, queued prompts, sub-agents, and pending decisions running. Returning
resumes the same Console with its latest results. Opening or closing a modal also
preserves work. Console microphone capture and Console automatic speech stop while
Console is hidden. Optional [Buddy speech](buddies.md) can continue across screens.

A decision that needs your input while Console is hidden raises one notice and a
navigation badge. Answer it in Console or a supported Buddy interaction card.
Configured decision timeouts count only while that session's card is available to
answer; time away from an answerable card does not consume the budget. Nothing is
approved automatically.

**Stop** still interrupts the selected turn and pauses its queue. Closing a session
cancels its work, and confirmed application quit shuts down the runtime. Continuation
after an application exit or crash is not guaranteed. An unsaved queue-manager edit
must still be saved or cancelled before navigation.

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
