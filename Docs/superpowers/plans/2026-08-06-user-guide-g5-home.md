# User Guide G5 — Home + index polish (final phase)

**Spec:** `Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`
**Base:** dev @ `84e4b33f0` (the G4 merge). Verified same-day.

## Scope

1. `Docs/User_Guide/home.md` — the last written page of the program. One page,
   no children: Home is the smallest surface yet (831-line screen + 338 lines
   of widgets + a 2,536-line state layer), and its complexity is in what the
   strings claim, not in nested workbenches.
2. **Index polish** — `index.md` was stamped six days and four merged sections
   behind everything else; every global claim predated G1-G4. Re-verified and
   re-stamped, plus the cross-page audit's factual fixes.
3. Three fresh Home captures (populated / idle / details) replacing the three
   stale ones that rode into the G4 merge unreferenced.

## Survey facts the page is built on

- Zero drift: `home_screen.py`, `Widgets/Home/`, `Home/` untouched across the
  ~1,600 commits since the pre-wipe survey — the code-exact inventory carried
  over whole; every load-bearing claim was still re-probed live at the base
  sha.
- The app boots to Console, not Home; Home greets you only under the
  first-run wizard.
- Home declares **no key bindings** — mouse/Tab only; F6 toasts; F1 lists
  only the three inherited bindings.
- Home is a mount-time snapshot: `_sync_home_triage` has exactly three
  callers (two on-mount workers + the row click handler); no timer, watcher,
  or subscription. Live-verified byte-identical after 6 s idle.

## Defects filed (all live-verified at the base sha)

| Task | Defect |
|---|---|
| task-2760 | Primary action navigates from a different `HomeAction` than the one it displays (label/callout promise Console, click lands Library) |
| task-2761 | `mcp_ready`/`rag_ready` are constants no producer sets; "Search your Library" branch is dead code |
| task-2762 | Rail and canvas cannot scroll; overflow work is invisible and unreachable |
| task-2763 | Home never refreshes after mount |
| task-2764 | "Model: Ready" means the `[providers]` table is non-empty, not that a model works |
| task-2765 | "Import Library sources" opens Library without the ingest context it promises |

Documented-not-filed (designed-in, honest toasts): Approve/Reject/Pause/Resume
are adapter fallbacks; Retry is real only for Library ingest; Console
approvals are not counted on Home; "Review notifications" lands on Watchlists.

## Index polish + cross-page fixes (from the dual audit)

- Home row linked; Quick Start rewritten around the real first-run flow
  (wizard first, skip lands on Home) and current nav labels.
- Shift+F6 split out of the global table — it exists only on Console and
  Roleplay (`chat_screen.py` / `personas_screen.py`).
- Four label-less screens (Study, Media, Search, Statistics) documented with
  their real routes; "study/media/search" palette words documented as Library
  aliases — `library.md` and `search-and-rag.md` each recommended a palette
  route that does not exist (fixed).
- Legacy table completed (Ingest, Writing, Chatbooks, Characters/Roleplay,
  Speech/Evals, Tools & Settings, Stats); Customize precision.
- Conventions block rewritten to describe the guide's actual majority style;
  a per-screen unsaved-work warning added.
- Palette entries quoted verbatim guide-wide ("Tab Navigation: Switch to X");
  nav labels quoted as rendered ("⌃2 Console", "F9 Settings") guide-wide.
- `First_Run_Setup.md`: the two Settings categories that do not exist (Tools,
  Notes) repointed at the real owners; Speech step added to its table.
- 🚧 markers added to the nine stub links that lacked them.

## Verification gates

- Live probe battery at the base sha (nav labels, F7/F8/F9, the four bug
  repros, retry/notice strings, rail/canvas text).
- Quoted-string check of `home.md` against the Home modules; template
  conformance; link sweep; backlog duplicate-guard delta vs dev (zero new).
- Captures shot at the base sha via the isolated `g4_profile` scratch config
  (`[paths] data_dir` **and** `[database]` — see the updated
  verification-probes memory for why both).
