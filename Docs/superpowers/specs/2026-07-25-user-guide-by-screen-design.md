# User Guide, screen-by-screen — design

Date: 2026-07-25
Status: approved (brainstorm 2026-07-25, delivery/scope/template/captures/accuracy
decided by the user; Approach A structure + Console-first ordering approved)

## Goal

Give tldw_chatbook a real user guide: a `Docs/User_Guide/` tree where each
screen (and each major sub-screen of the composite screens) has its own page
describing what it is for, what every visible feature/control does, and how to
accomplish common tasks on it. Today no user guide exists — `Docs/` is
developer-facing, and the only user-facing material is the install-centric
README, `FAQs.md`, and eight deep-dive docs in `Docs/Features/`.

## Decisions (user-locked)

- **D1 Delivery:** in-repo Markdown under a new `Docs/User_Guide/` directory,
  one file per screen with an `index.md`, linked prominently from the README.
  No docs site, no in-app Help wiring in this program (both are explicit
  non-goals; the per-screen file layout keeps them cheap to add later).
- **D2 Scope:** deep pages for the five core screens — **Home, Console,
  Library, RP&CD (Roleplay), Settings** — and real-but-tiny stub pages for the
  other eight nav destinations (Artifacts, Watchlists, Schedules, Workflows,
  MCP, ACP, Lab, Logs). Legacy/hidden route names get "now lives in…" pointers
  in the index only.
- **D3 Page shape:** one fixed hybrid template per page (reference AND
  task-oriented; section list below).
- **D4 Screenshots:** fresh captures of current dev taken by the author in the
  same verification session as the prose, stored under
  `Docs/User_Guide/images/<screen>/`. The reproducible capture recipe is
  documented in `_template.md`.
- **D5 Accuracy:** every page is written from a live driving session of the
  real app; every claim and every how-to is executed on-screen before being
  written down; each page footer carries
  `Verified against dev @ <short-sha> — <YYYY-MM-DD>`.

## Structure (Approach A: parent page + child pages for composite screens)

```
Docs/User_Guide/
  index.md                         # app intro, Quick Start (first five minutes:
                                   # launch -> provider setup -> first chat), nav map,
                                   # GLOBAL shortcuts table (F6, Ctrl+P, number-key nav),
                                   # conventions ("guide tracks the dev branch"),
                                   # stub notice, legacy-route pointers
                                   # ("Notes → Library ▸ Notes")
  _template.md                     # canonical page template + capture recipe (authoring aid)
  images/<screen>/*.svg            # fresh captures, one folder per screen
  home.md
  console.md                       # orientation, layout tour, sessions/tabs/workspaces
  console/chat-basics.md           # send/stream/stop, message actions, selection, keyboard
  console/branching-and-rewind.md  # regenerate/swipe siblings, Edit & resend, /rewind menu
  console/attachments-and-images.md# attach/paste, inline images, image generation, speak/TTS
  console/agent-runs.md            # agent mode, tool markers, approvals, skills, MCP tools
  console/context-and-rag.md       # inspector/next-send preview, RAG scoping, prefill,
                                   # dictionaries & world books in the payload
  library.md
  library/notes.md
  library/media.md
  library/skills.md
  library/prompts.md
  library/collections.md
  <roleplay>.md + <roleplay>/{characters,lorebooks,chat-dictionaries}.md
  settings.md                      # one page, one section per pane (split later only if
                                   # a pane outgrows it)
  artifacts.md  watchlists.md  schedules.md  workflows.md
  mcp.md  acp.md  lab.md  logs.md  # stubs with a visible "🚧 stub" banner
```

Open naming item (resolve in G0): the nav label is "RP&CD" — confirm its
on-screen expansion in the live app and name the page/directory to match what
users actually see (candidate: `roleplay.md` + `roleplay/`). File names
otherwise follow visible nav labels, lowercased/kebab-cased.

**Child-page lists above are PROVISIONAL.** They were drafted from program
memory, not a live survey. Every phase (G1–G5) begins with a live IA survey
of the actual screen; the survey's sub-surface inventory wins over this
spec's tree, and the phase plan records any delta.

## Page template (fixed section order)

```markdown
# <Screen> — <one-line purpose>

## What this screen is for      (2–4 sentences; when to reach for it)
## Getting there                (nav key/number, command palette, startup config)
## Layout tour                  (capture + region-by-region walk, regions named
                                 exactly as labeled on screen)
## Features & controls          (reference table per region: control → what it does)
## Common tasks                 (3–8 numbered step-by-step how-tos, imperative voice)
## Keyboard & commands          (table: key / slash command → action;
                                 SCREEN-SPECIFIC only — globals live in index.md)
## Related settings & docs      (Settings panes, config.toml keys, Docs/Features links)
## Quirks & troubleshooting     (honest limitations with backlog refs, common errors
                                 and their fixes)

—
*Verified against dev @ <short-sha> — <date>*
```

Authoring rules:
- On-screen labels quoted verbatim; no internal jargon (no "native id",
  "store", "recompose").
- No aspirational features. Limitations stated honestly with a backlog ref
  where one exists (e.g. `/rewind` restore-to-before-first does not survive an
  app restart — task-574).
- Stub pages use only the first two template sections plus a "🚧 This page is
  a stub" banner and links to any existing `Docs/Features/` deep dive.

## Captures

- Scratch profile (`TLDW_CONFIG_PATH`, throwaway `users_name`), canned demo
  content, no personal data; local llama endpoint when a live reply is needed.
- One standard terminal size for all captures (fixed in `_template.md` during
  G0 after checking what renders best; candidate 200×50).
- Format: SVG preferred (crisp, diffable, text-searchable) — but the SVG
  pipeline is UNPROVEN here. G0 gives it ONE timeboxed attempt (Textual's
  own SVG export, or tmux ANSI -> rich `export_svg`); if it does not work
  cleanly, fall back without ceremony to PNG via the PROVEN
  textual-serve + Playwright harness used by the UX-review program. The
  chosen recipe (launch command, profile, sizing, export mechanism) is
  written into `_template.md` in G0 so refreshing a stale image is minutes,
  not archaeology.

## Cross-linking and de-duplication

- `Docs/Features/*` remain the deep dives. Guide pages give 2–3 orienting
  sentences and link out; they never duplicate deep-dive content.
- `FAQs.md` entries that are really screen how-tos migrate into the relevant
  page; genuine FAQs stay put.
- README gains a prominent "📖 User Guide" link near the top.
- `index.md` carries a legacy-route table (notes, prompts, subscriptions,
  coding, …) pointing at the surviving screen.
- Child pages cross-link laterally where behavior spans pages (e.g.
  branching page ↔ context page for what regenerate does to provider context).

## Phasing (six PRs, each at the user merge gate)

- **G0 — scaffold:** `index.md` (incl. Quick Start + global shortcuts),
  `_template.md` (template + capture recipe, incl. the timeboxed SVG-vs-PNG
  decision), all eight stub pages, README link, RP&CD naming resolved, and a
  one-line maintenance hook in CLAUDE.md ("UI-changing PRs should update the
  matching Docs/User_Guide page or its stamp"). No captures.
- **G1 — Console:** `console.md` + five child pages + captures. (Largest;
  Console is where every new user lands and where the most undocumented
  capability sits.) The phase plan MAY split this into two PRs (parent +
  chat-basics, then the four deeper children) if the single-PR review load
  looks too heavy.
- **G2 — Library:** `library.md` + five child pages.
- **G3 — RP&CD:** parent + three child pages.
- **G4 — Settings:** `settings.md`.
- **G5 — Home** + index polish pass (promote any stub that turned out trivial
  to complete; fix cross-links discovered along the way).

## Done criteria (per deep page)

1. Every control visible on the screen appears in *Features & controls*
   (swept against the live screen during the verification session).
   Exception for form-heavy panes (chiefly Settings): self-describing form
   fields may be summarized at field-group level; interactive/behavioral
   controls are always enumerated individually.
2. Every *Common task* was executed end-to-end live before being written.
3. All links resolve; captures render; the verification stamp is present.
4. No claim contradicts current dev behavior; known limitations carry backlog
   refs.
5. Terminology matches on-screen labels exactly.
6. Merge-time drift check: before the phase PR merges, confirm dev has not
   changed the documented screen since the verification session (diff dev
   history for the screen's modules); re-verify and re-stamp affected
   sections if it has. (Concurrent sessions merge UI changes here daily.)

## Non-goals

- No docs website / mkdocs tooling.
- No in-app Help (F1) wiring to these pages (layout keeps it cheap later).
- No CI drift-checking tooling (CI is intentionally cancelled in this repo;
  the verification stamp is the staleness signal).
- No rewriting of `Docs/Features/*` deep dives beyond adding cross-links.
