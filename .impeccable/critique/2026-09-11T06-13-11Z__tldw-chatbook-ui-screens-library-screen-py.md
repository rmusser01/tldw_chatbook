---
target: "Library screen + all subscreens (critique #10, whole-Library scope, post critique-9 fix wave)"
total_score: 25
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 2
timestamp: 2026-09-11T06-13-11Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets, fresh-eyes, no prior snapshot; parent reconciled the two against code and pinning tests). Target `tldw_chatbook/UI/Screens/library_screen.py` at dev 1f3184655b (2026-09-11, after the critique-9 fix wave: PRs #2568 #2577 #2579 #2580 #2581 #2582 #2583 #2585 #2590 and the parallel Library▸Notes program). Scope = the whole Library. Live at 235x52, 100x30 and 60x24 on a brand-new profile and a seeded profile (11 media, 6 conversations, 7 notes, 5 prompts, 2 skills); no LLM provider configured. Host conditions (not defects): local media Import fails at parse-pool start (leaked POSIX semaphores); the fresh profile's fixture folders arrived empty and were repaired by assessor A. 112 captures.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Status is honest (delete→Undo, Trash→Restore, Export verified at the DB); the Media filter box can hold a draft that differs from the applied filter with no scope line |
| 2 | Match System / Real World | 2 | "Collections" opens "Quick Capture"; "Remove placement"; "Handoff · 0 eligible · 1 blocked"; "Server sync WIP"; "Prompt · Local · System + User" |
| 3 | User Control and Freedom | 3 | Escape ladders, Trash, inline "This cannot be undone." with Cancel first; Export never shows the bundle before writing it |
| 4 | Consistency and Standards | 2 | Viewer tab strip marks the active tab with "(selected)" instead of the legend glyph; three pagers differ; glosses on 4 of 7 Browse rows |
| 5 | Error Prevention | 3 | Paths validated before the queue runs, deletes confirm, blocked actions carry reasons; Export defaults to `quality: thumbnail` |
| 6 | Recognition Rather Than Recall | 2 | The footer switches off whenever an Input has focus: `typing in field | F6 next pane` while `u`, `o`, `/` still work unnamed |
| 7 | Flexibility and Efficiency | 3 | Real keyboard depth; the viewer's Find has no keyboard route |
| 8 | Aesthetic and Minimalist Design | 2 | Landing uses ~10 of 45 rows; Skills at 60 columns spends 4 of 18 rows on single-page pager chrome |
| 9 | Recognize/Diagnose/Recover from Errors | 3 | Import's failure grammar is the best copy in the app; Collections' empty state blames filters nobody set |
| 10 | Help and Documentation | 2 | 14 guide claims contradicted across the two assessors |
| **Total** | | **25/40** | **Fair** |

B filed no P0 and no P1 (84 checks: 55 pass, 8 partial, 6 fail). A scored 25 with four P1s; reconciled to two P1s (footer suppression; age-as-duration, a pinned format), one P2 (filter draft vs applied), one folded into task-32107 (hand-off gate).

## Design Specificity Verdict

Authored in its parts, category-interchangeable in its shell (both assessors). Parts: one glyph legend honoured on the select-mode toolbar, blocked actions that name reason and route, the import queue's cause-plus-fix rows, the Info tab's source authority, a footer that names the right Escape target everywhere, a working below-64-column mode. Shell: the generic three-block hub on a mostly empty canvas, a 20-control rail, internal vocabulary in the Details panel. The detector is vacuous for a Python TUI (exit 0, empty scan); every claim is live-measured.

## What the critique-9 wave fixed, confirmed by fresh eyes

Selected-media export writes a real bundle (the critique-9 P0 is gone). Undo restores DB rows; Trash Restore clears the flag. The glyph legend holds where applied. The blocked RAG panel speaks the Media grammar. The search row no longer clips the rail. Escape leaves the Notes filter box where the footer says. The rail heading ellipsises. Notes autosave persists mid-edit. 100x30 reflows cleanly.

## Overall Impression

Crossed from "acceptable with a P0" to "fair with none". Destructive and failure paths are exemplary; what remains is naming, disclosure and one hidden control. Biggest opportunity: the footer, which switches itself off exactly when a keyboard user is typing.

## What's Working

- Blocked states explain themselves: every `○` carries a reason and a route.
- Destructive-action design: Trash as a real state; "Delete forever" restates the item, says it cannot be undone, focuses Cancel; Undo verifiably restores.
- The context footer names per-key verbs and the correct Escape target on every route.
- The single-stage layout below 64 columns works as documented.

## Priority Issues

- **[P1] Canvas key hints vanish whenever an Input has focus.** After submitting a search the footer reads `typing in field | F6 next pane` while `u`, `o`, `/` still work unnamed (A caps 14/15; B docs table). Not pinned by name. Fix: keep the canvas verbs and prefix the field state; drop `F6 next pane` first when short. /impeccable clarify.
- **[P1] Media row age reads as a duration on audio/video.** `audio · 10m` is age (11m, 14m later) but reads as length (A caps 36/39/47). Pinned format (`test_media_secondary_fallback_when_no_type_no_age`) — a pinned decision I disagree with. Fix: `added 10m ago` or type-aware metadata. /impeccable clarify.
- **[P2] The Media viewer's Find is unreachable, and a failed attempt arms Trash.** Not in the Tab order; the sibling-button click is inert on it; typing afterwards fires `t trash` (B D4/D4a, caps 27/28). No Find binding exists. Fix: a binding, Tab membership, no single-key destructive accelerator while a viewer is open. /impeccable harden.
- **[P2] A new profile lands on the full rail with "Back to Get started" orphaned.** PROVEN (critique #8): a pre-written config makes `coerce_library_lifecycle(raw=None, is_new_profile=False)` resolve EXPANDED; same for a user who completes setup and quits before visiting Library (B D3, A cap 04). Fix: no content yet ⇒ STARTER. /impeccable onboard.
- **[P2] Media filter box and applied filter can disagree with no scope line** (A caps 41/47). Fix: a persistent scope line (`Media · 9 of 11 · filter "x" · [Clear]`). /impeccable clarify.
- **[P2] Import names a batch after its subdirectory; the landing under-reports a failed import** ("nested — 6 files"; "An import needs review." after 4 failed) (B D1/D2). /impeccable clarify.
- **[P2] "Collections" opens "Quick Capture" whose empty state blames unset filters and names an absent action**; its pager is not disabled at 0 of 0 (A 16/58, B D6). Overlaps task-32057. /impeccable clarify.
- **[P2] Export defaults to `quality: thumbnail` and never lists the bundle** (A cap 50). Fix: full fidelity by default; a consequence line with count and size. /impeccable harden.
- **[P2] Single-page pager chrome survives on Skills, Collections and Trash** (4 of 18 rows at 60x24; A caps 60/61). PROVEN: `library_skills_canvas.py` composes its own pager without `library_pager_layout`; pinned by `test_skills_canvas_renders_exact_pager_and_source_wide_trust_count`. /impeccable polish.
- **[P2] The rail collapses to an unlabelled `--->` grip while a note editor is open at 235 columns**; grips unlabelled at every width; the guide promises a "Nav" handle and the rail beside Notes at ≥120 (A cap 11, B D10; PROVEN `library_adaptive_reader_shell.py:151-156`). /impeccable layout.

## Persona Red Flags

**Jordan:** "Back to Get started" for a view never seen; "Draft — not saved yet" on a note already listed; Collections→Quick Capture; the Details panel's "1 blocked"; no in-product definition of RAG, Skill, Collection, workspace.
**Alex:** filter draft split-brain; `Use as source` disabled on all six conversations (task-32107); `]`/`[` typing into the filter box; Conversations footer offers no canvas keys; no export manifest.
**Sam:** footer blanks on Input focus; Find has no keyboard route; active vs focused rail rows visually identical. Focus is otherwise by shape, disabled by glyph, selection by `☐/☑`.
**Riley:** 0-result filters, 5 k note, CJK/emoji rows and live resize all behave; 60x24 clips copy mid-word without an ellipsis.

## Docs vs Live

Fourteen contradictions: landing `n new note` (live `ctrl+n`); Search/RAG `enter select evidence` (absent); a Search⇄RAG mode toggle (absent); Chunking Lab "above" its gloss (below); Notes editor `ctrl+s save note` chip (absent); rail beside Notes at ≥120 (collapsed at 235); "Nav" handle label (`--->`); "Back to Get started never offered after graduation"; bulk-delete banner copy; "cancel delete" footer string; a dirty-edit veto on Escape (autosave instead); the Find bar behaviour; glosses "shown consistently" (4 of 7); empty Browse rows without pager mechanics (Skills/Collections/Trash keep them).

## Minor Observations

- `Loaded ·` leaks into a selected row's title; conversation rows use ` - ` where others use `·`; "New Chat" reads like a button.
- Prompt rows all begin "Prompt · Local ·"; "System + User" is schema-speak.
- Stored analysis renders as raw Markdown; `○ Find` on the Analysis tab has no reason beside it.
- Import footer `enter start`: first Enter validates, second runs.
- A trust-approved seeded skill reads "needs review" under "Skill trust isn't set up" with no precedence explanation.
- Export's disabled reason sits three rows above its button; Conversations gives the reader ~48 of 235 columns; the Search/RAG Sources panel triple-spaces four checkboxes.

## Questions to Consider

- Why does Library have a landing page at all, when every persona's first act is to leave it for the rail?
- Who introduced "workspace" to this user, and should the headline hand-off be gated on a concept Library never defines?
- Is "Collections" a product or an artefact while the local service is read-only and both profiles show zero?
- If the footer is the real UI, why is it the first thing sacrificed to a focused Input or a narrow terminal?

## Environment note

Local media Import remains unexercisable on this Mac until a reboot clears the leaked POSIX semaphores; assessed as failure presentation only. Not reached by either assessor within budget: File Notes + Session Git, Prompts editor/insert, Skills directory import + trust approval, Notes templates + sync panel, Collections CRUD, Study hand-offs, Chunking Lab strip, evidence→Console `o`/`u`.
