# Kanban — a keyboard-first board of columns and cards

The Kanban destination (nav strip: **Kanban**, or `f8`; command palette:
"kanban" / "board") is a keyboard-first work board over the app's local
Kanban store — the same `tldw_chatbook_kanban.db` the Kanban service layer
owns. It is a local, private surface: nothing syncs to a server, and no
other feature reads or writes it yet (Scheduling jobs and agent-run
completions are future card sources; when they grow card emission they
will write through the same store, and the board needs no changes to
surface them).

## The board

Columns are the board's **lists** in order; cards are its cards, each
showing its id, title, priority, and due date. The inspector strip below
the board shows the selected card's full detail (Enter focuses it).

```
┌─ To Do (2) ──┐  ┌─ Doing (0) ─┐  ┌─ Done (1) ──┐
│ #1  first    │  │ No cards    │  │ #3  Done    │
│ #2  second   │  │             │  │     card    │
└──────────────┘  └─────────────┘  └─────────────┘
```

## Keys

All board actions are single-letter htop-style keys (ADR-031); nothing
here needs a mouse, though clicking a card selects it.

| Key | Action |
|---|---|
| `←` / `→` (or `h` / `l`) | Move between columns (empty columns are skipped) |
| `↑` / `↓` (or `k` / `j`) | Move between cards in the current column |
| `Enter` | Focus the selected card's detail in the inspector |
| `Shift+H` / `Shift+L` | Move the selected card to the previous / next column |
| `n` | New card in the focused column (Enter adds, Escape cancels) |
| `/` | Focus the filter box — narrows every column by text (id, title, or description) |
| `r` | Refresh the board from the store |

## Starting honest

A fresh install has no kanban data (the store ships empty and nothing
user-facing has ever written to it), so the board's first screen offers
one action: `n` creates your first board — **Personal**, with **To Do /
Doing / Done** columns — and you build from there. That is the only
seeding the app ever does, and only when you ask for it.

**v1 column scope.** Columns come from the board's lists, and this first
version ships no in-app column management: the bootstrap's To Do / Doing /
Done are the columns you get (lists can still be created through the
Kanban service layer). Duplicate list names are disambiguated on display
("Doing", "Doing (2)") rather than collapsing.

Governance: the board's architecture, data ownership, and keybindings are
decided in
[ADR-237](../../backlog/decisions/237-kanban-board-port-and-data-ownership.md)
— the board UI is ported from backlog-md-py's TUI and owns nothing; the
local Kanban service owns the data.
