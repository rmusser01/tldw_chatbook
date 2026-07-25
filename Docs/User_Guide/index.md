# tldw_chatbook User Guide

tldw_chatbook is a terminal (TUI) app for working with LLMs: chat with local
or cloud providers, manage conversations/notes/media in a local library,
run roleplay characters with lorebooks, schedule watchlists, and drive
agent/tool workflows — all stored locally in SQLite.

> This guide tracks the **dev** branch. Each page carries a
> "Verified against dev @ `<sha>`" stamp; if your build is older, screens
> may differ slightly.

## Quick start — your first five minutes

1. Install and launch — see the [README](../../README.md#installation).
2. Open **Settings** (coming in G4) and set a provider + model (or point at
   a local server).
3. Open **Console** (coming in G1) by clicking "Console" in the nav bar, or use **Ctrl+P** → "Switch to Console"; then send your first message. (From most screens, you can also press **2**.)
4. Press **F1** anywhere for contextual help; **Ctrl+P** opens the command
   palette.

## The screens

| Key | Screen | What it's for |
|-----|--------|----------------|
| 1 | Home (coming in G5) | Dashboard, notifications, status, and next actions. |
| 2 | Console (coming in G1) | Live agent conversations, approvals, tools, RAG, and runs. |
| 3 | Library (coming in G2) | Workspaces, source material, imports, notes, media, conversations, Study, flashcards, quizzes, and Search/RAG. |
| 4 | [Artifacts](artifacts.md) 🚧 | Generated outputs, bundles, reports, datasets, and Chatbooks. |
| 5 | Roleplay & Chat Dictionaries (coming in G3) | Characters, user profiles, dictionaries, and behavior profiles. |
| 6 | [Watchlists](watchlists.md) 🚧 | Monitored sources, runs, alerts, and recovery. |
| 7 | [Schedules](schedules.md) 🚧 | When jobs, watchlists, and workflows run. |
| 8 | [Workflows](workflows.md) 🚧 | Reusable procedures, recipes, dry-runs, and outputs. |
| 9 | [MCP](mcp.md) 🚧 | MCP servers, tools, permissions, auth, and audit. |
| 0 | [ACP](acp.md) 🚧 | Agent Client Protocol agents, sessions, runtimes, diffs, and terminals. |
| — | [Lab](lab.md) 🚧 | Models, speech, and evaluation runs. |
| — | [Logs](logs.md) 🚧 | Application logs and diagnostics. |
| — | Settings (coming in G4) | Global app preferences, appearance, accounts, and storage. |

Lab, Logs, and Settings have no dedicated number key — reach them by
clicking the nav label or via the command palette (**Ctrl+P**).

**Note:** Number keys switch tabs from most screens right after launch, but not
once a text field (composer, search box) — or in at least one observed
case, another auto-focused widget — has focus; that keystroke is consumed
by the field instead of the app shell. Click the nav label or use
**Ctrl+P** as the dependable way to switch screens.

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| F1 | Open the current screen's shortcuts help (the list shown is screen-specific) |
| Ctrl+P | Open the command palette — search and jump to any screen or command from anywhere |
| Ctrl+Q | Quit the app |

Everything else (F6/Shift+F6 pane-cycling, Enter/Ctrl+K/Ctrl+T in Console,
and the single-letter mnemonics like `s`/`r`/`t` on Settings) is
screen-specific — see that screen's own page for its "Keyboard &
commands" table.

## Where did … go? (legacy names)

| Old name | Now lives in |
|----------|--------------|
| Notes | Library (coming in G2) |
| Prompts | Library (coming in G2) |
| Skills | Library ▸ Skills (coming in G2) |
| Subscriptions | [Watchlists](watchlists.md) 🚧 |
| Coding | Console (coming in G1) |
| Conversations / CCP | Roleplay & Chat Dictionaries (coming in G3) |
| LLM management | [Lab](lab.md) 🚧 |

## Conventions

- Keys are shown as `Ctrl+P`; slash commands as `/rewind`.
- 🚧 marks stub pages awaiting a full write-up.
- Deep dives live in [Docs/Features](../Features/); pages link out rather
  than duplicate them.
