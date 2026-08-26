---
id: TASK-19575
title: >-
  CLAUDE.md contains three dead paths and a fictional event list, and AGENTS.md
  is a 175-line fork of it
status: To Do
assignee: []
created_date: '2026-08-21 20:28'
labels:
  - documentation
  - process
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 7 **F7** and Lane 1 **#4, #5**. Two
independent lanes reached the same conclusion from opposite directions (a
process audit and an import-graph audit), which is why this is filed rather
than treated as ordinary doc rot. All re-verified at this branch base.

CLAUDE.md is the operating manual every agent reads before touching this repo.
Its errors propagate into work, and one of them produces a broken feature.

**Hard errors:**

**(a) Three dead paths in a single paragraph.**
`Chat_Window_Enhanced.py` **does not exist anywhere in the tree** (the only
string hit is a historical docstring at
`Widgets/Chat_Widgets/chat_approval_card.py:22`), yet CLAUDE.md describes it at
**lines 47 and 51** — including as "an embedded `Container` widget, not a
Screen", which reads as current architectural guidance.
**`media_ingest_screen.py` also does not exist** — `UI/Screens/` holds only
`media_screen.py` and `media_runtime_state.py` — yet CLAUDE.md:48 names it.

**(b) The uv/pip note is false.** `.venv/bin/python -m pip --version` →
**`pip 25.0.1`**, exit 0. The venv *is* uv-managed (`pyvenv.cfg`:
`uv = 0.8.11`) but ships pip, and `.venv/bin/pytest` is present. Both halves of
the note ("ships NO pip", "if `pytest` is missing") are wrong for the current
checkout; only the bare `pip` shim is absent (`pip3` / `pip3.12` exist). The
note actively misdirects anyone setting up.

**(c) The "Pre-commit Hook" section describes a hook that is not wired.**
`Helper_Scripts/fixed_auto_review.py` exists and its docstring declares itself
a PreToolUse hook for Edit/Write/MultiEdit — but it appears in **no** settings
file. The worktree's hooks block wires two entirely different commands. An
agent reading CLAUDE.md believes its edits are LLM-reviewed before landing.
They are not.

**(d) The "New Tab" procedure omits the step that makes a tab visible.**
CLAUDE.md's four-step recipe never mentions
`UI/Navigation/shell_destinations.py` — the **sole** source of the nav bar, the
overflow menu, and the ⌃1..⌃0 / F7-F9 hotkeys. **Follow the documented
procedure exactly and you ship a nav-invisible tab.** This is the error that
produces broken work, not just confusion.

**(e) The "Key events" list is entirely fictional.** All six documented names
return **zero occurrences in the package**:

| name | package | whole repo |
|---|---|---|
| ChatEvent | 0 | 0 |
| StreamingChunk | 0 | 3 |
| RAGSearchEvent | 0 | 0 |
| SyncEvent | 0 | 0 |
| EvalEvent | 0 | 0 |
| TabEvent | 0 | 0 |

The three `StreamingChunk` hits are tests asserting the class was **retired**
(`Tests/test_application_state_ownership.py:1144` asserts
`"class StreamingChunk" not in worker_source`). The one name with any trace is
pinned as deleted.

**(f) `form_components.py` is documented as the standard and is unreachable.**
CLAUDE.md calls it "Standardized form builders". It has exactly **two
importers, both themselves production-orphans**: `UI/SiteConfigSettings.py:28`
(class referenced nowhere outside its own file) and
`Widgets/media_details_widget.py:24` (referenced only by a test). A third hit
is a comment at `UI/STTS_Window.py:108` explaining why a live screen
*avoids* it. Agents are being pointed at a helper the codebase has abandoned.

**Omissions:** `_SCREEN_ALIASES` has **12** entries (AST-counted — the review
said 13); CLAUDE.md names 5. A fourth lessons file,
`backlog/docs/lessons-textual.md` (5,362 B), is unlisted. Three DB modules
outside `DB/` are unlisted.

**(g) Structural — AGENTS.md is a fork.** `diff -u CLAUDE.md AGENTS.md` →
**exactly 175 diverging lines** (86 removed / 89 added). CLAUDE.md is 489
lines, AGENTS.md 492. **Codex agents and Claude agents read different operating
manuals for the same repo**, and the parent checkout's copy differs again.
Every fix above has to be applied twice, or the fork widens.

Per the owner's standing ruling, the durable fix is to stop maintaining two
copies — a shared source with a thin per-agent preamble — rather than
hand-syncing 175 lines and hoping.

## Acceptance Criteria

- [ ] Every file path named in CLAUDE.md exists — `Chat_Window_Enhanced.py`,
      `media_ingest_screen.py` and any other dead reference are removed or
      replaced with the live equivalent
- [ ] The uv/pip note is corrected to match the actual venv, or deleted
- [ ] The "Pre-commit Hook" section describes what is actually wired, or says
      plainly that `fixed_auto_review.py` is not currently active
- [ ] The "New Tab" procedure includes registering a destination in
      `UI/Navigation/shell_destinations.py`; following the procedure end to end
      produces a tab reachable from the nav bar and its hotkey
- [ ] The "Key events" list names events that exist, or is removed
- [ ] `form_components.py` is no longer presented as the standard unless it is
      made reachable (this overlaps TASK-19571's wire-or-retire decision)
- [ ] The alias count, the fourth lessons file and the out-of-`DB/` modules are
      corrected
- [ ] CLAUDE.md and AGENTS.md stop diverging: one shared source of truth with a
      thin per-agent section, so a correction cannot land in only one
- [ ] A check fails when CLAUDE.md/AGENTS.md reference a path that does not
      exist — this class of drift should be caught mechanically, and it is
      cheap enough to belong in the `derived-artifacts` job from TASK-19572
