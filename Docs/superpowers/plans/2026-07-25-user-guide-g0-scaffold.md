# User Guide G0 (Scaffold) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `Docs/User_Guide/` scaffold — index with Quick Start/nav map/global shortcuts/legacy pointers, the authoring template with a decided capture recipe, eight stub pages, the README link, and the CLAUDE.md maintenance hook — per the approved spec `Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`.

**Architecture:** Pure-docs change plus two bounded investigations (live nav survey; capture-pipeline timebox). No app code changes. Everything lands on branch `claude/user-guide-program` (worktree `.claude/worktrees/console-branching`) and ships as one PR at the user merge gate.

**Tech Stack:** Markdown; tmux-driven live app (`.venv/bin/python -m tldw_chatbook.app` with `TLDW_CONFIG_PATH` scratch profile); Textual SVG export OR textual-serve+Playwright PNG (decided in Task 2).

## Global Constraints

- Spec is authoritative: `Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`.
- On-screen labels quoted verbatim; no internal jargon (no "native id", "store", "recompose").
- No aspirational features; limitations carry backlog refs where they exist.
- Stub pages = first two template sections + "🚧 This page is a stub" banner + Docs/Features links where relevant.
- File names follow visible nav labels, lowercased/kebab-cased.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Run everything from the worktree root `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching` (background Bash resets cwd — re-`cd` in every shell block).
- The live app writes real user data unless `TLDW_CONFIG_PATH` points at a scratch profile — ALWAYS use the scratch profile; delete `~/.local/share/tldw_cli/<scratch users_name>` afterwards.

---

### Task 1: Live nav survey (labels, screen titles, footers, RP&CD naming)

**Files:**
- Create: `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/1d048b3f-6ca0-42a3-99c4-ad0cb7b73ac0/scratchpad/g0_survey.md` (working notes, NOT committed)

**Interfaces:**
- Produces: the verified nav inventory consumed by Tasks 3–5: exact nav-bar labels for all 13 destinations, each screen's on-screen title line, the footer key list per visited screen, the intersection = GLOBAL shortcuts set, and the resolved RP&CD expansion + chosen file naming (`roleplay.md` vs other).

- [ ] **Step 1: Write the scratch profile config**

```bash
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/1d048b3f-6ca0-42a3-99c4-ad0cb7b73ac0/scratchpad
mkdir -p "$S/g0_profile"
cat > "$S/g0_profile/config.toml" <<'EOF'
[general]
users_name = "guide_g0"
default_tab = "chat"

[splash_screen]
enabled = false
EOF
```

- [ ] **Step 2: Launch the app in tmux and confirm boot**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/1d048b3f-6ca0-42a3-99c4-ad0cb7b73ac0/scratchpad
tmux -L g0 kill-server 2>/dev/null
tmux -L g0 new-session -d -x 235 -y 52 "TLDW_CONFIG_PATH=$S/g0_profile/config.toml .venv/bin/python -m tldw_chatbook.app"
sleep 14
tmux -L g0 capture-pane -p | head -3
```

Expected: nav bar line showing `1 Home │ 2 Console │ 3 Library …`.

- [ ] **Step 3: Record the full nav bar and Console footer**

```bash
tmux -L g0 capture-pane -p | sed -n '1,3p'  >> "$S/g0_survey.md"
tmux -L g0 capture-pane -p | tail -3        >> "$S/g0_survey.md"
```

- [ ] **Step 4: Visit each of the 13 destinations; record title + footer for each**

Navigation: click each nav label with an SGR click pair sent to row 2 at the label's column (find COL from a fresh `capture-pane` dump; click = `tmux -L g0 send-keys -l $'\x1b[<0;COL;2M'` then the same with `m`). Locate the label column and click IN ONE Bash invocation (atomic locate+click — separate calls drift). For each screen append to `$S/g0_survey.md`: the nav label, the screen's own title/header line, and the footer line.

Special attention on **RP&CD**: record the full screen title/header text — this resolves the page/directory name. Also record LAB, LOGS, SETTINGS (they have no number key — note how they open).

- [ ] **Step 5: Derive the global-shortcuts set**

In `$S/g0_survey.md`, list footer keys seen on EVERY visited screen (the intersection) as GLOBAL; everything else is screen-specific. Explicitly check: `F6`/`Shift+F6`, `F1`, `Ctrl+P`, number keys `1–0`.

- [ ] **Step 6: Extract the legacy-route alias table from code**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
.venv/bin/python -c "
from tldw_chatbook.UI.Navigation import screen_registry as sr
for alias, target in sorted(sr._SCREEN_ALIASES.items()):
    print(f'{alias} -> {target}')
" >> /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/1d048b3f-6ca0-42a3-99c4-ad0cb7b73ac0/scratchpad/g0_survey.md
```

Expected: at least `notes -> library`, `prompts -> library`, `subscriptions -> watchlists_collections`, plus TAB_CCP/TAB_LLM entries.

- [ ] **Step 7: Quit the app; decide and record the RP&CD file naming**

```bash
tmux -L g0 send-keys C-q; sleep 3; tmux -L g0 kill-server 2>/dev/null
```

In `$S/g0_survey.md` write one line: `RP&CD naming decision: <file/dir name> because on-screen title is "<verbatim>"`. Rule from spec: match what users see, lowercased/kebab-cased; if the title is just "RP&CD", use its expanded form from the screen header; if genuinely only "RP&CD" appears anywhere, use `rp-and-cd`.

(No commit — survey notes are scratchpad-only. The next tasks consume them.)

### Task 2: Capture-recipe timebox (SVG attempt → PNG fallback) + `_template.md`

**Files:**
- Create: `Docs/User_Guide/_template.md`
- Create: `Docs/User_Guide/images/.gitkeep`

**Interfaces:**
- Consumes: nothing from Task 1 (independent).
- Produces: the decided capture recipe (format + exact commands) inside `_template.md`; later phases follow it verbatim.

- [ ] **Step 1: Timeboxed SVG attempt (max ~30 minutes wall clock)**

Try, in order, stopping at the first that yields a faithful full-screen image of the running app:

(a) Textual built-in export — check availability, then drive it:
```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
grep -rn "save_screenshot\|deliver_screenshot" .venv/lib/python3.12/site-packages/textual/app.py | head -5
```
If `save_screenshot` exists (it does in Textual 8.x), test SVG export headlessly with a pilot script:
```bash
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/1d048b3f-6ca0-42a3-99c4-ad0cb7b73ac0/scratchpad
cat > "$S/svg_probe.py" <<'EOF'
"""Probe: boot the real app under run_test and export an SVG screenshot."""
import asyncio, os

if "TLDW_CONFIG_PATH" not in os.environ:
    raise SystemExit("Refusing to run against the real profile: set TLDW_CONFIG_PATH")

async def main() -> None:
    from tldw_chatbook.app import TldwCli
    app = TldwCli()
    async with app.run_test(size=(200, 50)):
        app.save_screenshot(os.environ["G0_PROBE_OUT"])

asyncio.run(main())
EOF
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
TLDW_CONFIG_PATH="$S/g0_profile/config.toml" G0_PROBE_OUT="$S/g0_probe.svg" .venv/bin/python "$S/svg_probe.py" && ls -la "$S/g0_probe.svg"
```
Inspect the SVG (open in browser or read the text) — accept only if the full screen renders faithfully (theme colors, box glyphs intact).

(b) If (a) fails: tmux ANSI → rich SVG:
```bash
tmux -L g0 capture-pane -e -p > "$S/screen.ansi"
.venv/bin/python -c "
from rich.console import Console
c = Console(record=True, width=235)
with open('$S/screen.ansi') as f:
    c.print_ansi = getattr(c, 'print_ansi', None)
    text = f.read()
from rich.text import Text
c.print(Text.from_ansi(text))
c.save_svg('$S/screen.svg', title='tldw_chatbook')
"
```
Accept under the same fidelity bar.

(c) If both fail within the timebox: PNG via the proven textual-serve + Playwright harness (templates in the session scratchpad `harness/` folder per the UX-review program; browser page screenshot at a fixed viewport). Record which option won and why in the recipe.

- [ ] **Step 2: Write `Docs/User_Guide/_template.md`**

Content — the spec's template plus authoring rules plus the WINNING recipe from Step 1 (replace the `<capture recipe>` block with the actual commands that worked, and state the standard size chosen — use the size that rendered best in Step 1, candidate 200×50):

```markdown
# Page template & authoring guide (not user-facing)

Copy everything between the BEGIN/END markers into a new page and fill it in.
Every page is written from a LIVE driving session: execute every claim and
every how-to on-screen before writing it down.

<!-- BEGIN TEMPLATE -->
# <Screen> — <one-line purpose>

## What this screen is for
(2–4 sentences; when to reach for it)

## Getting there
(nav key/number, command palette entry, startup config)

## Layout tour
(capture + region-by-region walk; regions named exactly as labeled on screen)

## Features & controls
(reference table per region: control → what it does)

## Common tasks
(3–8 numbered step-by-step how-tos, imperative voice)

## Keyboard & commands
(table: key / slash command → action; SCREEN-SPECIFIC only — globals live in
the [guide index](index.md))

## Related settings & docs
(Settings panes, config.toml keys, Docs/Features links)

## Quirks & troubleshooting
(honest limitations with backlog refs; common errors and their fixes)

—
*Verified against dev @ <short-sha> — <YYYY-MM-DD>*
<!-- END TEMPLATE -->

## Authoring rules

- On-screen labels verbatim. No internal jargon (no "native id", "store",
  "recompose").
- No aspirational features. Limitations stated honestly with a backlog ref
  where one exists.
- Screen-specific keys only in "Keyboard & commands"; global keys live in
  index.md.
- Stub pages: first two sections + a "🚧 This page is a stub" banner + links
  to any existing Docs/Features deep dive.
- Form-heavy panes (chiefly Settings): self-describing form fields may be
  summarized at field-group level; interactive/behavioral controls are always
  enumerated individually.
- Before the phase PR merges: re-check dev history for the documented
  screen's modules; re-verify and re-stamp affected sections if it moved.

## Capture recipe

<the exact winning commands from Task 2 Step 1: scratch profile setup,
launch, drive-to-state notes, export command, standard size, output path
convention Docs/User_Guide/images/<screen>/<name>.<ext>>

Demo-content rules: scratch profile only (TLDW_CONFIG_PATH), canned demo
text, no personal data; local llama endpoint for live replies; delete the
scratch data dir afterwards.
```

- [ ] **Step 3: Verify and commit**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
ls Docs/User_Guide/_template.md Docs/User_Guide/images/.gitkeep
grep -c "BEGIN TEMPLATE" Docs/User_Guide/_template.md   # expect 1
grep -n "TBD\|TODO\|<the exact winning" Docs/User_Guide/_template.md  # expect NO matches (recipe filled in)
git add Docs/User_Guide/_template.md Docs/User_Guide/images/.gitkeep
git commit -m "docs(guide): authoring template + decided capture recipe (G0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: `index.md`

**Files:**
- Create: `Docs/User_Guide/index.md`

**Interfaces:**
- Consumes: Task 1's survey (nav labels, screen one-liners, global shortcuts, legacy aliases, RP&CD naming).
- Produces: the guide's front door; every other page links back to it as `index.md`.

- [ ] **Step 1: Write `Docs/User_Guide/index.md`**

Structure (fill the bracketed slots from `g0_survey.md` — every table row must match the live survey verbatim; do not invent):

```markdown
# tldw_chatbook User Guide

tldw_chatbook is a terminal (TUI) app for working with LLMs: chat with local
or cloud providers, manage conversations/notes/media in a local library,
run roleplay characters with lorebooks, schedule watchlists, and drive
agent/tool workflows — all stored locally in SQLite.

> This guide tracks the **dev** branch. Each page carries a
> "Verified against dev @ <sha>" stamp; if your build is older, screens may
> differ slightly.

## Quick start — your first five minutes

1. Install and launch — see the [README](../../README.md#installation).
2. Open **Settings** (…as recorded in survey…) and set a provider + model
   (or point at a local server). <link to settings.md#provider-setup>
3. Press **2** to open **Console** and send your first message.
   <link to console.md>
4. Press **F1** anywhere for contextual help; **Ctrl+P** opens the command
   palette.

## The screens

| Key | Screen | What it's for |
|----|--------|----------------|
| 1 | [Home](home.md) | <one-liner from survey> |
| 2 | [Console](console.md) | <one-liner> |
| 3 | [Library](library.md) | <one-liner> |
| … | … (all 13 rows, stub pages linked too, 🚧 marked) | |

## Global keyboard shortcuts

| Key | Action |
|-----|--------|
| <rows = the INTERSECTION set from the survey — globals only> | |

## Where did … go? (legacy names)

| Old name | Now lives in |
|----------|--------------|
| Notes | [Library ▸ Notes](library.md) |
| Prompts | [Library ▸ Prompts](library.md) |
| Subscriptions | [Watchlists](watchlists.md) |
| <remaining rows from the alias table, user-recognizable names only> | |

## Conventions

- Keys are shown as `Ctrl+P`; slash commands as `/rewind`.
- 🚧 marks stub pages awaiting a full write-up.
- Deep dives live in [Docs/Features](../Features/); pages link out rather
  than duplicate them.
```

- [ ] **Step 2: Verify links resolve and commit**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
.venv/bin/python - <<'EOF'
import re, pathlib
page = pathlib.Path("Docs/User_Guide/index.md")
base = page.parent
bad = []
for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text()):
    if target.startswith("http"):
        continue
    if not (base / target).exists():
        bad.append(target)
print("BROKEN:", bad)
EOF
```
Expected: `BROKEN: []` — note deep-page targets (console.md etc.) do NOT exist yet in G0; link the stubs that DO exist, and write not-yet-written deep pages as plain text with "(coming in G1)" instead of links. Re-run until empty.

```bash
git add Docs/User_Guide/index.md
git commit -m "docs(guide): index — quick start, nav map, global shortcuts, legacy pointers (G0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: Eight stub pages

**Files:**
- Create: `Docs/User_Guide/artifacts.md`, `watchlists.md`, `schedules.md`, `workflows.md`, `mcp.md`, `acp.md`, `lab.md`, `logs.md`

**Interfaces:**
- Consumes: Task 1's survey (each screen's one-liner + how to open it); Task 2's stub rules.
- Produces: link targets for index.md's nav map.

- [ ] **Step 1: Write all eight stubs from the shared skeleton**

Skeleton (fill both sections from the survey; keep each file under ~25 lines):

```markdown
# <Screen> — <one-line purpose from survey>

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

<2–4 verified sentences>

## Getting there

- Press **<key>** from anywhere (or: open via **Ctrl+P** → "<palette name>").

<optional: "Deep dive: [<title>](../Features/<file>.md)" where one exists —
e.g. watchlists → SUBSCRIPTION_IMPLEMENTATION_PLAN is internal, skip;
speech/transcription links belong to future media pages, not these stubs>
```

- [ ] **Step 2: Verify and commit**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
ls Docs/User_Guide/{artifacts,watchlists,schedules,workflows,mcp,acp,lab,logs}.md
grep -L "🚧" Docs/User_Guide/{artifacts,watchlists,schedules,workflows,mcp,acp,lab,logs}.md  # expect NO output
git add Docs/User_Guide/*.md
git commit -m "docs(guide): stub pages for the eight non-deep screens (G0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 5: README link + CLAUDE.md maintenance hook

**Files:**
- Modify: `README.md` (immediately after the opening description, before "Project Status")
- Modify: `CLAUDE.md` (in "Development Guidelines")

**Interfaces:**
- Consumes: nothing new.
- Produces: discoverability + the refresh hook required by the spec.

- [ ] **Step 1: Add the README link**

Insert after the first paragraph of `README.md`:

```markdown
> 📖 **New here?** The [User Guide](Docs/User_Guide/index.md) walks through
> every screen — what it does and how to use it.
```

- [ ] **Step 2: Add the CLAUDE.md hook**

Insert as the last bullet of "### Adding Features" in `CLAUDE.md`:

```markdown
- **UI changes:** PRs that change a screen's UI should update the matching
  `Docs/User_Guide/` page (or at least its "Verified against" stamp).
```

- [ ] **Step 3: Verify and commit**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
grep -n "User Guide" README.md | head -2
grep -n "Docs/User_Guide" CLAUDE.md | head -2
git add README.md CLAUDE.md
git commit -m "docs(guide): README entry point + CLAUDE.md maintenance hook (G0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 6: Whole-scaffold check + PR

**Files:**
- None new (verification + PR only).

**Interfaces:**
- Consumes: everything above.
- Produces: PR at the user merge gate.

- [ ] **Step 1: Full link sweep across the guide**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
.venv/bin/python - <<'EOF'
import re, pathlib
bad = []
for page in pathlib.Path("Docs/User_Guide").glob("*.md"):
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text()):
        if target.startswith("http"):
            continue
        if not (page.parent / target).exists():
            bad.append(f"{page.name}: {target}")
print("BROKEN:", bad)
EOF
```
Expected: `BROKEN: []`.

- [ ] **Step 2: Spec-conformance sweep**

Check against the spec's G0 bullet: index (Quick Start + globals) ✓, _template (template + decided recipe) ✓, eight stubs ✓, README link ✓, CLAUDE.md hook ✓, RP&CD naming recorded ✓ (the decision line from Task 1 goes into the PR body). Fix anything missing before the PR.

- [ ] **Step 3: Push and open the PR (leave at user gate)**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-branching
git push -u origin claude/user-guide-program
gh pr create --base dev --title "docs: User Guide scaffold (G0 — index, template, stubs)" --body "<summary of the six deliverables, the RP&CD naming decision + evidence, the capture-recipe decision (SVG vs PNG + why), spec + plan links; footer: 🤖 Generated with [Claude Code](https://claude.com/claude-code)>"
```

Do NOT merge — user gate.
