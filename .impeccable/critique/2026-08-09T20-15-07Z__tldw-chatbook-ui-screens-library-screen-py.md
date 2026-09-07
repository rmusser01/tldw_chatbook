---
target: Library screen and subscreens (re-critique after 3 fix arcs)
total_score: 22
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 8
timestamp: 2026-08-09T20-15-07Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-director live walkthrough · B: mechanical probes/detector) — isolated tmux instances (`rcritA6729`/`rcritB6729`), scratch profiles, identity-verified on every launch; dev `4d0232358` (all three fix arcs merged), read-only worktree.

> **CORRECTION (2026-08-09, after task-4020's investigation): RC-02 is WITHDRAWN — it was a measurement artifact, not a defect.**
> B's nav probe (V7) used colorless `tmux capture-pane -p`, which cannot distinguish a genuinely
> mid-word-clipped label from a correctly ghosted one: ghosting paints foreground **equal to**
> background, so the characters remain in the buffer while being invisible on screen. Direct ANSI
> decoding at 80/100/120 cols with early and late active tabs shows fg `(18,18,18)` == bg
> `(18,18,18)` for every fragment B quoted (`⌃6 Watc`, `⌃9 M`, the `‹ …` lead-ins). **Nav ghosting
> was never broken.**
> B's own evidence corroborates this in hindsight: its click at col 78 landed on what it called a
> "blank" cell and did not navigate — that cell was a ghosted tab, exactly as designed.
> **Assessment A contradicted B here** ("the nav collapses to `More ▾` rather than hard-cutting
> mid-word") and A was right; the synthesis resolved the conflict toward the mechanical arm without
> flagging the disagreement. Two synthesis lessons: (1) a plain capture is not evidence about a
> colour-based mechanism — use `-e` whenever the mechanism under test IS colour; (2) an explicit
> A-vs-B contradiction must be named and adjudicated in the report, never silently resolved.
> What task-4020 *did* find and fix was real but different: task-3200's tests only ever exercised
> `MainNavigationBar.DEFAULT_CSS`, never the bundled-CSS tier that actually wins live — a genuine
> coverage gap, now closed. The score is unchanged (RC-02 was not scored as its own heuristic
> deduction), but **p1_count should be read as 8, not 9.**

# Re-critique — Library screen + all subscreens (2026-08-09)

Baseline: **22/40** on 2026-08-06 (0 P0 / 8 P1). Three fix arcs merged since: PR #1410 (P1s), PR #1420 (P2 batch), PR #1459 (polish batch).

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Import preflight/queue/receipts exemplary; bulk delete completes in total silence |
| 2 | Match System / Real World | 2 | "opens staging canvas" ×3 in primary nav; "Collection record", "delete records", "0 blocks" |
| 3 | User Control and Freedom | 1 | **Escape from the Notes editor terminates the application**; no undo/trash for bulk delete |
| 4 | Consistency and Standards | 1 | Four footer dialects, three active-state markers, three toolbar layouts, `▸` means two things; empty prompt/skill drafts discard but empty notes persist |
| 5 | Error Prevention | 3 | Disabled-with-reason gating real; accepts `libexport.ziprary export 2026-08-09.zip` without a murmur |
| 6 | Recognition Rather Than Recall | 2 | Cycle-buttons hide their option space; copy names a control that never renders |
| 7 | Flexibility and Efficiency | 3 | Bulk actions real and working; 38 Tab presses to the first canvas control; Enter in rail search doesn't search |
| 8 | Aesthetic and Minimalist Design | 2 | 30-char list column on a 170-col terminal; 7-row viewport for a 33-line document |
| 9 | Error Recovery | 3 | Miss/blocked copy is model-quality; the app's worst error has no message and takes the session |
| 10 | Help and Documentation | 2 | F1 lists Escape 2–3× with contradictory labels; doesn't close on second F1; per-canvas footers carry the real help |
| **Total** | | **22/40** | **Unchanged number, materially changed composition** |

Trend for this slug: 23 → 21 → 27 → 22 → **22** (out of 40).

## The headline: we shipped a P0 fixing the last critique's headline

The 2026-08-06 critique's lead finding was *"Escape is a no-op on every canvas tested."* The remediation wired Escape into `action_library_notes_escape` → `self._back_from_library_note_editor()` — **a method that exists nowhere in the codebase**. `grep -rn "_back_from_library_note_editor" tldw_chatbook/` returns exactly one hit: the call site (`library_screen.py:4371`). Library ▸ Notes ▸ New ▸ Blank note ▸ Escape raises `AttributeError`, the exception is unhandled, and the app terminates (`event=unhandled_exception` → `event=app_stopping`). Reproduced 3/3 by A and 2/2 by B independently; confirmed statically by the controller.

Two aggravators: the footer on that canvas advertises `Esc Notes` as the way out, and the sibling Prompt editor's Escape works correctly — so users learn the gesture is safe, then get punished. B's AST sweep of `LibraryScreen` for undefined `self.X()` calls found **exactly this one** — no sibling landmines. The branch is guarded by `if self._library_notes_view == "editor":`, which is why no mount-time smoke test reached it.

**Fix already in flight** on the residue branch with a state-setting regression test.

## Verified fixed since 2026-08-06 (both arms agreeing)

- **Select mode bulk actions (LIB-05)** — real and working. B's zip listing proves selection-scoped export contains *exactly* the 2 selected items (`total_media_items: 2`); delete drops the rail 3→1; the double-press guard holds (V2: one decrement). The confirm — `Delete 2 selected items? This moves them to trash.` with the footer rewriting to `esc cancel delete` — is now the best destructive-action pattern in the product.
- **Media viewer markdown (LIB-13)** — rendered with a real box-drawn table; `Rendered (selected) | Raw` toggle works both ways and marks state in **text**, not colour.
- **Export receipt + gating (LIB-11/12)** — durable across canvas exit and re-entry; the disabled button carries a visible reason.
- **Study handoff (LIB-06)** — breadcrumb, Escape-back, and an honest tab bar at the destination; provenance carried across.
- **Files mode (LIB-01)** — rebuilt exactly as prescribed, inside the Library frame, with adjacent prompt and button.
- **Folder-notes cross-references (LIB-19)** — all three surfaces now explain their relationship to each other. Genuinely good writing.
- **Import/Export vocabulary (LIB-10)** — "ingest" and "chatbook" gone end to end.
- **Rail truncation (LIB-18)** — B confirms **no mid-word truncation at 120/100/80**; degradation is whole-gloss-drop plus short-title swap, exactly as designed.
- **Focus visibility** — B: 12/12 Tab stops visibly distinct, none byte-identical to unfocused.
- **Search headlines, Conversations title, export quality caption** — all fixed.

## Priority issues

### P0
- **RC-01 — Escape from the Notes editor kills the app.** See headline. Fix in flight.

### P1
- **RC-02 — WITHDRAWN (see correction at the top of this document).** ~~Our own nav ghosting is not producing its intended outcome at dev tip.~~ B measured tab labels cut **mid-word at 80 AND 120** (`⌃6 Watc`, `⌃9 M`, scroll fragments `‹ oleplay…`, `‹ edules…`) and found **no ghosted tabs** — the bar scrolls instead. The ghost machinery IS present (10 references in `main_navigation.py`, 2 in `_navigation.tcss`), so this is a failure of effect, not missing code. Leading hypothesis: dev replaced the in-strip pager with `NavOverflowMenu` during our arc, and the polish batch's rebase reconciliation kept the ghosting while the scroll/paging model beneath it changed. Task-3200's entire four-round arc was about this exact defect. ~~Re-verify and re-root-cause against the new overflow model.~~ (Mitigating: no ghost was clickable-while-invisible — B's blank-cell click did not navigate.) **This was a measurement artifact of colourless `tmux capture-pane -p`: direct ANSI decoding shows the ghosted fragments have fg==bg, i.e. nav ghosting was never broken. Not counted in `p1_count` (read as 8, not 9).**
- **RC-03 — Blank notes still persist; the GC exists but its predicate never fires.** B: opening the blank editor bumps `Notes (2)→(3)` and shows `Saved` before any keystroke; exiting via `‹ Notes` retains it; typing then deleting everything also retains it (`Notes (4)`, four indistinguishable `Untitled` rows). The session-blank GC from the P2 batch IS present and IS wired to ~7 exit paths including this one. Leading hypothesis, worth checking first: the title field carries the **literal string "Untitled"** rather than a placeholder, so `_flush_library_note_save`'s emptiness test (`any(value.strip() for value in (title, content, keywords))`) sees a truthy title and takes the save branch. Contrast: empty **prompt** and **skill** drafts discard correctly — the right behavior exists twice in the same screen.
- **RC-04 — Soft-deleted media becomes permanently un-importable.** B: bulk-delete 2 items → re-import the same files → `≡ matched · Already in Library — matched an existing item; nothing new was imported.` while the items are absent from the list and the count stays down. Dedup matches soft-deleted rows, so the file can never be re-added and the "trash" is unreachable. **Data-loss-shaped**: the user's content is neither present nor restorable through the UI.
- **RC-05 — Bulk delete has no receipt, no undo, and the promised trash does not exist.** The confirm earns consent by promising reversibility; nothing in the rail, the `type:` filter, or any canvas reaches a trash. Creation gets `✓ done · file · 1s` + a jump link; destruction gets silence. (Compounds RC-04.)
- **RC-06 — The Notes canvas advertises a control that never renders.** Copy: *"…for notes that live in a folder on disk, switch to Files, or use Sync to mirror one in."* The `Database | Files` strip (`library_screen.py:7333`) is absent on first paint; A found it appears only after a Sync round-trip, B confirms it is absent in both plain and ANSI captures and unreachable by 18 Tab presses. **LIB-01's fix landed the destination and lost the door** — arguably worse than the old dead end, because the old failure was discoverable.
- **RC-07 — Disabled state is colour-only at 1.08:1–2.30:1, with no reason at the control.** Measured: Select-mode bulk buttons **1.08:1** when 0 selected (the very buttons LIB-05 added — present, focusable, meaningful, unreadable); Media `Select` when empty 1.45:1 (click does nothing, says nothing); Export button ~1.4–1.51:1; Collections' three buttons 2.30:1 *even when enabled*. Violates two stated principles at once. The product's non-colour vocabulary already exists (`☐/☑`, `▸`, `┃…┃`, `(selected)`, `✓/○`) — it simply was never applied to disabled state.
- **RC-08 — Search/RAG buries its own output.** Results land ~30 rows below the fold behind a configuration panel; clicking `Run` leaves the visible half of the canvas pixel-identical. Enter in the rail search navigates and pre-fills but does not run. Two search inputs are live with different values, and navigation silently overwrites one with the other; the never-executed string still enters `Recent searches`.
- **RC-09 — DB sizes are computed once and never refreshed.** B: UI showed `Prompts 148.0KB / Media 476.0KB` while disk (incl. `-shm`) was 180.0KB/508.0KB; a forced recompose with **no disk change** corrected both. `get_formatted_db_size_with_wal` does sum the sidecars — task-2859's fix is correct, and stale. It understates by exactly the sidecar size.
- **RC-10 — F1 contradicts itself and won't close.** Notes list lists Escape three times with two conflicting labels (`- esc: focus rail` / `- escape: Back` / `- escape: Focus rail`); Media lists it twice; Search/RAG omits F6 though the footer advertises it; Collections' panel says nothing about Collections. Second F1 does not dismiss the panel.

### P2 (selected from ~20)
Escape still inert on Export, Collections, and the Study staging canvas; the staging canvas has no back path at all. `Export…` from within Media navigates away with no return. Collections stacks four "nothing here" sentences, still has no "Add to collection" anywhere, and its only enabled control with a collection selected is `Delete Collection` — styled identically to the benign buttons, with a confirm that names nothing and offers no Cancel. "opens staging canvas" printed three times in the primary nav. Four footer dialects; the hub's `i`/`n` shortcuts vanish elsewhere with no statement of whether they still work. `▸` overloaded (disclosure vs silent cycler). Import and Export receipts have entirely different grammars. Export scope "Everything" excludes Prompts, Skills, Collections. Expanding Details silently strips glosses from three unrelated rail rows (scrollbar steals a column). At ≤100 cols the landing canvas vanishes entirely while the rail still says "pick a section on the left". `Type: plaintext` for a `.md` the viewer renders as markdown; extensions stripped from every list title. Media list truncates at 17 chars with ~115 columns blank. Toast overlaps the queue's Clear button.

## Strengths
1. **The Import pipeline** — preflight facts accumulate as you type, the collapsed settings header states its own contents, per-file queue rows carry duration and a jump link, Unicode round-trips perfectly. The best-designed flow in the product.
2. **Bulk delete's confirmation copy** — count, action, and reversibility in nine words, with a context-aware footer. (The follow-through is the problem, not the dialog.)
3. **Cross-boundary honesty** — the Study staging canvas prints `Carries forward: …` before you commit, and the destination re-states its provenance. Nothing is smuggled across.

## Persona red flags
- **Alex (power user):** his reflex Escape kills the app; Enter doesn't search; the bulk buttons he came for are invisible until after he selects; cycle-buttons can't be jumped; 38 Tabs to the first canvas control.
- **Jordan (first-timer):** told to "switch to Files" with no Files control on screen; reads "opens staging canvas" three times; four "nothing here" sentences on Collections; ends up with four indistinguishable `Untitled` notes.
- **Sam (keyboard/a11y):** disabled state is colour-only at 1.08:1; the only enabled control on a populated Collections canvas is the destructive one; the `Database | Files` toggle is not in the focus chain at all; Escape terminates the app. Positives: focus is shape-marked (`┃…┃`), selection is text-marked, 12/12 stops visibly distinct.

## What the arcs actually bought
Eight of ten prior findings are genuinely fixed with evidence, and heuristics 1, 5, 7, 9 improved on real machinery. The score didn't move because **the fixes cast three shadows** (Files mode's lost door, the invisible bulk buttons, jargon replacing a lie) and because the *source* of inconsistency migrated rather than reduced: the old fracture was vocabulary, and we fixed it; the new fracture is grammar — four footer dialects, three active-state markers, three toolbar layouts, one glyph with two meanings. **Fixing the words did not fix the system.**

One of our own shipped fixes (blank-note GC) is present in code but not producing its intended effect at dev tip — worth re-rooting before anything new is built on it. (Nav ghosting is NOT in this category — see the RC-02 withdrawal correction above: it was a colourless-capture measurement artifact, not a broken fix.)

## Questions
1. The last critique's headline demanded Escape work; the remediation wired it to a method that doesn't exist, behind a state branch no mount-time test enters. What test shape would have caught it — and why does a codebase with mutation-tested guards elsewhere have none on the exit path of its most-used editor?
2. Every non-colour vocabulary this product needs already exists and is used well. What is preventing "unavailable, because —" from joining it, and is the honest answer that disabled state has no owner?
3. Import tells you type, size and count before you commit, then hands you a receipt with a jump link. Bulk delete tells you nothing, promises a trash that doesn't exist, and makes the deleted file permanently un-importable. Is the operative principle "show status before action" — or "creation deserves ceremony and destruction is plumbing"?
