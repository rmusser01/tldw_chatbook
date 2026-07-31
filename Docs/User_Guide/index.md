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

1. Install and launch — see the [README](../../README.md#installation). On
   first launch, a "Get started" onboarding card appears and the composer
   stays locked; sending unlocks once you've set up a provider in step 2.
2. Open **Settings** (not yet written) — click **Settings** in the nav bar
   (or **Ctrl+P** → "Switch to Settings") — and set a provider + model (or
   point at a local server).
3. Open **[Console](console.md)** — press **Ctrl+2**, click "Console"
   in the nav bar, or use **Ctrl+P** → "Switch to Console"; then send your
   first message.
4. Press **F1** anywhere to open the current screen's keyboard-shortcuts
   list; **Ctrl+P** opens the command palette.

## The screens

| Hotkey | Screen | What it's for |
|-----|--------|----------------|
| Ctrl+1 | Home (not yet written) | Dashboard, notifications, status, and next actions. |
| Ctrl+2 | [Console](console.md) | Live agent conversations, approvals, tools, RAG, and runs. |
| Ctrl+3 | [Library](library.md) | Source material, imports, notes, media, conversations, prompts, skills, Search/RAG — plus hand-offs to Study for flashcards and quizzes. |
| Ctrl+4 | [Artifacts](artifacts.md) 🚧 | Generated outputs, bundles, reports, datasets, and Chatbooks. |
| Ctrl+5 | Roleplay & Chat Dictionaries (not yet written) | Characters, user profiles, dictionaries, and behavior profiles. |
| Ctrl+6 | [Watchlists](watchlists.md) 🚧 | Monitored sources, runs, alerts, and recovery. |
| Ctrl+7 | [Schedules](schedules.md) 🚧 | When jobs, watchlists, and workflows run. |
| Ctrl+8 | [Workflows](workflows.md) 🚧 | Reusable procedures, recipes, dry-runs, and outputs. |
| Ctrl+9 | [MCP](mcp.md) 🚧 | MCP servers, tools, permissions, auth, and audit. |
| Ctrl+0 | [ACP](acp.md) 🚧 | Agent Client Protocol agents, sessions, runtimes, diffs, and terminals. |
| — | [Lab](lab.md) 🚧 | Models, speech, and evaluation runs. |
| — | [Logs](logs.md) 🚧 | Application logs and diagnostics. |
| — | Settings (not yet written) | Global app preferences, appearance, accounts, and storage. |

Lab, Logs, and Settings have no hotkey — reach them by clicking the nav
label or via the command palette (**Ctrl+P**).

**Note:** The digit shown before each nav label is that screen's hotkey
digit: press **Ctrl+digit** (Ctrl+1 … Ctrl+9, Ctrl+0) to switch to it from
anywhere — the chord works even while a text field has focus. Bare digit
keys are not navigation shortcuts (typing `2` in the composer just types
"2"). Clicking the nav label and **Ctrl+P** work everywhere too.

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| F1 | Open the current screen's keyboard-shortcuts list (content is screen-specific) |
| Ctrl+P | Open the command palette — search and jump to any screen or command from anywhere |
| Ctrl+Q | Quit the app |
| Ctrl+1 … Ctrl+9, Ctrl+0 | Switch to the screen with that hotkey digit (see the nav map above) |
| F6 / Shift+F6 | Cycle forward/backward through the current screen's panes (on screens without multiple panes it only shows a notice) |

Everything else (Enter/Ctrl+K/Ctrl+T in Console, and the single-letter
mnemonics like `s`/`r`/`t` on Settings) is screen-specific — see that
screen's own page for its "Keyboard & commands" table.

## Console agent runs are screen-scoped

Background agent runs and parallel sessions you start in Console — and any
approval/confirmation they're waiting on — live only as long as the Console
screen itself stays mounted. Leaving Console for another screen (e.g.
Settings, Ctrl+1…Ctrl+0, or the command palette) cancels every in-flight
run and denies every pending or parked approval for that visit; coming
back always starts a fresh Console with no memory of what was running
before. Two guards make this visible instead of silent:

- **Before you leave:** if any run is still in flight or waiting on an
  approval, a confirmation dialog asks "N agent runs will be cancelled if
  you leave Console. Leave anyway?" — **Leave** proceeds and cancels them,
  **Stay** keeps Console (and the fleet) exactly as it was. An idle
  Console never shows this prompt.
- **After you return:** if you left anyway (or navigated away some other
  way while runs were active), the next time Console mounts you get a
  one-time toast — "N agent runs were cancelled when you left Console." —
  so a lost run is never silently unexplained.

Nothing is ever auto-approved: an approval that gets caught by this
teardown is always denied, never resolved on your behalf.

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
| Conversations / CCP | Roleplay & Chat Dictionaries (not yet written) |
| LLM management | [Lab](lab.md) 🚧 |
| Research | [Library](library.md) |
| Customize | Settings (not yet written) |

## Conventions

- Keys are shown as `Ctrl+P`; slash commands as `/rewind`.
- 🚧 marks stub pages awaiting a full write-up.
- Deep dives live in [Docs/Features](../Features/); pages link out rather
  than duplicate them.

—
*Verified against dev @ 8975c9b8 — 2026-07-25*
