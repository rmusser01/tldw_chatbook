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
3. Open **Console** (not yet written) by clicking "Console" in the nav bar,
   or use **Ctrl+P** → "Switch to Console"; then send your first message.
   (Pressing **2** can also work when no text field has focus — see the
   number-key note below.)
4. Press **F1** anywhere to open the current screen's keyboard-shortcuts
   list; **Ctrl+P** opens the command palette.

## The screens

| Key | Screen | What it's for |
|-----|--------|----------------|
| 1 | Home (not yet written) | Dashboard, notifications, status, and next actions. |
| 2 | Console (not yet written) | Live agent conversations, approvals, tools, RAG, and runs. |
| 3 | Library (not yet written) | Workspaces, source material, imports, notes, media, conversations, Study, flashcards, quizzes, and Search/RAG. |
| 4 | [Artifacts](artifacts.md) 🚧 | Generated outputs, bundles, reports, datasets, and Chatbooks. |
| 5 | Roleplay & Chat Dictionaries (not yet written) | Characters, user profiles, dictionaries, and behavior profiles. |
| 6 | [Watchlists](watchlists.md) 🚧 | Monitored sources, runs, alerts, and recovery. |
| 7 | [Schedules](schedules.md) 🚧 | When jobs, watchlists, and workflows run. |
| 8 | [Workflows](workflows.md) 🚧 | Reusable procedures, recipes, dry-runs, and outputs. |
| 9 | [MCP](mcp.md) 🚧 | MCP servers, tools, permissions, auth, and audit. |
| 0 | [ACP](acp.md) 🚧 | Agent Client Protocol agents, sessions, runtimes, diffs, and terminals. |
| — | [Lab](lab.md) 🚧 | Models, speech, and evaluation runs. |
| — | [Logs](logs.md) 🚧 | Application logs and diagnostics. |
| — | Settings (not yet written) | Global app preferences, appearance, accounts, and storage. |

Lab, Logs, and Settings have no dedicated number key — reach them by
clicking the nav label or via the command palette (**Ctrl+P**).

**Note:** Number keys can switch tabs, but only when no text field
(composer, search box) — or, in at least one observed case, another
auto-focused widget — has focus; once something else has focus, that
keystroke is consumed by the field instead of the app shell. They're most
likely to work right after launch, before anything has grabbed focus.
Click the nav label or use **Ctrl+P** as the dependable way to switch
screens.

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| F1 | Open the current screen's keyboard-shortcuts list (content is screen-specific) |
| Ctrl+P | Open the command palette — search and jump to any screen or command from anywhere |
| Ctrl+Q | Quit the app |

Everything else (F6/Shift+F6 pane-cycling, Enter/Ctrl+K/Ctrl+T in Console,
and the single-letter mnemonics like `s`/`r`/`t` on Settings) is
screen-specific — see that screen's own page for its "Keyboard &
commands" table.

## Where did … go? (legacy names)

| Old name | Now lives in |
|----------|--------------|
| Notes | Library ▸ Notes (not yet written) |
| Prompts | Library ▸ Prompts (not yet written) |
| Skills | Library ▸ Skills (not yet written) |
| Subscriptions | [Watchlists](watchlists.md) 🚧 |
| Coding | Console (not yet written) |
| Conversations / CCP | Roleplay & Chat Dictionaries (not yet written) |
| LLM management | [Lab](lab.md) 🚧 |
| Research | Library (not yet written) |
| Customize | Settings (not yet written) |

## Conventions

- Keys are shown as `Ctrl+P`; slash commands as `/rewind`.
- 🚧 marks stub pages awaiting a full write-up.
- Deep dives live in [Docs/Features](../Features/); pages link out rather
  than duplicate them.

—
*Verified against dev @ 9af99aba — 2026-07-25*
