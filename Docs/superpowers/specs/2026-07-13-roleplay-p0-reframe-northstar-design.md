# Roleplay (Personas redesign) — P0: reframe + north-star (design)

**Status:** Design approved (brainstorm), pending spec review.
**Program:** Personas workbench redesign, sub-project **P0** of `P0 → P1 → P2 → P3`. P0 is the thin foundation (identity reframe + the binding interaction contract); P1 = Dictionaries mode, P2 = Lore mode, P3 = Characters/Personas flow polish — each its own spec → plan → PR.
**Research inputs (drive P1/P2):** `Docs/superpowers/research/2026-07-13-server-dictionaries-port.md`, `Docs/superpowers/research/2026-07-13-server-worldbooks-port.md`.
**Worktree/branch:** `.claude/worktrees/personas-redesign`, `claude/personas-redesign` off dev `ea88ff95`.

## Problem

The Personas screen is a 3-column workbench (`personas_screen.py`) with a 5-chip mode strip — `Characters · Personas · Prompts · Dictionaries · Lore`. Only **Characters** and **Personas** work; the placeholder literally reads *"This mode is not available yet."* **Prompts** is being moved to Library by a parallel branch; **Dictionaries** and **Lore** are dead chips over real, unused backends (`chat_dictionaries` + services; `world_books` + `WorldBookManager`/`world_info_processor`). And the screen is titled **"Personas"** while its default mode is **Characters** — a container/content mismatch. Once the two dead modes become real (P1/P2), "Personas" is plainly the wrong container name.

## Goal / Acceptance

- **AC1 (reframe, shippable):** The screen presents as **"Roleplay"** with an honest, self-explaining 4-mode strip — every chip says what it *is*, and not-yet-built modes read as an inviting "coming soon," never a dead "not available yet."
- **AC2 (north-star, documented):** This spec defines the binding **List → Detail → Try-it** interaction pattern (below) that P1/P2/P3 implement, so the four modes are consistent instead of each reinventing a layout.
- **AC3 (no collision):** P0 edits only Personas-owned display in `personas_screen.py`. The route id `personas`, the screen registry, and nav/shell files (`screen_registry.py`, `shell_destinations.py`, `route_inventory.py`) are **not touched** — they're owned by the parallel Library-Prompts branch.

## Part A — The north-star interaction pattern (binding contract)

Every Roleplay mode inherits one three-pane skeleton. This is the contract P1/P2/P3 build to; P0 does not build a shared framework speculatively — shared widgets get extracted when the *second* mode needs them, not before.

```
 LIST (left rail)        DETAIL (center)                 TRY-IT (right)
 search / sort / filter  the selected entity             "what does this DO?"
 inline enable toggle    (tabbed where it has sub-parts)  per-mode verify surface
 type/count/used badges  progressive disclosure           
 Duplicate · + New       (Simple ⇄ Advanced)             
 empty → starter templates
```

**Per-mode instantiation:**

| Mode | List | Detail | Try-it (right pane) |
|---|---|---|---|
| **Characters** | character cards | card view / editor | **chat preview** (existing, kept — now framed as the mode's verify surface, not a bolt-on) |
| **Personas** | user personas | persona editor | sample-turn preview |
| **Dictionaries** (P1) | dictionaries + enable toggle, regex/literal + entry-count badge, used-by | **tabbed:** Entries · Attachments · Stats · Settings | **substitution preview** — paste sample text → diff (removed struck / added highlighted) + which entries fired (backed by the already-wired `process_text`) |
| **Lore** (P2) | world books + enable toggle, entry-count, used-by | **tabbed:** Entries · Attachments · Stats · Settings | **Test-Match trigger diagnostics** — paste a sample message → which lore fired, *why* (keyword/secondary/regex/recursion), token cost, budget bar, **+ "near-misses"** (matched-but-skipped: disabled / failed secondary key / dropped by budget) |

**Shared affordances (defined here, realized per mode):**
- **Right pane = "Try it."** The right pane always answers *"what does this thing actually do?"* — a mode-specific verify/test surface. This is the principled replacement for today's over-built chat "preview," and it's the single strongest lesson from the tldw_server UX audits (their #1 finding for both dictionaries and world-books).
- **Progressive disclosure.** The Detail editor shows a **Simple** view (the 2–3 fields most people touch) with an **Advanced** toggle for power fields, using plain-language labels (e.g. "Scan depth" → "Messages to search"). Both server audits independently landed on this — strong signal, and it directly de-clutters the Characters/Personas editor.
- **Honest list + empty states.** Inline enable/disable (never buried in the editor), scannable badges, a Duplicate action, and empty states that offer starter templates instead of a blank void.
- **Structured validation.** Problems surface as a jump-to-entry list of `{code, field, message}` items, not opaque prose.

**Two things Roleplay does *better than the server* (seeded now, built in P1/P2):**
- **"What's in play."** Because Roleplay holds all four modes in one workbench (the server scatters them), it can answer *"for this character / conversation, which persona + lorebooks + dictionaries apply?"* — a cross-mode attachments summary the server never built. Direction: also support **character-level attachment** (not only per-conversation), so one lorebook/dictionary serves all a character's chats.
- **Near-misses in diagnostics.** The Lore Try-it shows not just what fired but what *matched-yet-was-skipped* — answering *"why did X NOT fire,"* which the server's diagnostics can't.

## Part B — P0 shippable scope (the reframe, display-only)

All in `personas_screen.py` (Personas-owned), no behavior change to the working modes:

1. **Identity.** `_title_text()` is a *live* header — currently `base = "Personas | Behavior profiles for chat and agents"` with dynamic ` | New {character|persona}` and ` - unsaved` suffixes appended. Reframe **only the base string** → `"Roleplay | Author the pieces that shape a chat"`, preserving the create-mode/unsaved suffix logic verbatim. The `#personas-purpose` tagline → *"Characters, personas, dictionaries, and lore — the pieces that shape a chat. Attach them to Console."*
2. **Self-explaining mode strip — visible, not tooltip-only.** Tooltips are hover-only and useless to keyboard users, so the active mode's meaning must be **visible on selection**: on mode switch, update a persistent one-line descriptor (extend `#personas-purpose` or add a subtitle under the strip) to what the *active* mode is — *Characters* → "Who the AI plays," *Personas* → "Who you are," *Dictionaries* → "Text find/replace rules," *Lore* → "World facts injected on keywords." Keep the chip `tooltip`s too (as the same copy) for mouse users, but the visible descriptor is the primary affordance and doubles as a "where am I" cue.
3. **Honest coming-soon.** The `PLACEHOLDER_COPY` for not-yet-built modes (Dictionaries, Lore) changes from a flat *"not available yet"* to an inviting, per-mode one-liner — what it will do + "coming soon" — and the chips are visually de-emphasized (dimmed/planned-looking) so they read as *roadmap*, not *broken*. **Fallback:** if P1/P2 slip well past P0, prefer *hiding* the two chips over a stale "coming soon" (don't advertise nav that doesn't work); coming-soon is the default only because the builds are imminent. P1/P2 replace the placeholder with the real workbench.
4. **Keep the three-pane frame.** No structural change to the shell in P0; it already matches the north-star's skeleton.

## Data flow / error handling

None — P0 changes display copy/labels only; the working Characters/Personas flows are untouched. The north-star (Part A) is documentation, not code.

## Collision constraints (parallel Library-Prompts branch)

- **Do not touch** `screen_registry.py` / `shell_destinations.py` / `route_inventory.py` / the route id `personas`. P0 is display-only inside `personas_screen.py`.
- **Exact overlap (verified):** the parallel branch's Task 7 edits the *same three sites* P0 touches — `MODE_CHIP_ORDER` (drops "prompts"), the class docstring, and the `#personas-purpose` copy (drops "prompts"). So a rebase conflict on these lines is *guaranteed but trivial*. P0 should **not fight the prompts removal** — leave the `prompts` entry in `MODE_CHIP_ORDER` for P0 (removing it is the other branch's job), and write P0's new purpose/title copy **prompts-free as its final form**, so the reconcile is simply "take P0's Roleplay copy" regardless of merge order (P0's copy already omits prompts, superseding the other branch's copy edit on that line).
- Name check: **"Roleplay"** was chosen specifically to avoid colliding with the existing **"Prompt Studio"** feature (`Prompt_Studio_Interop/`); do not reintroduce "Studio" for this workbench.

## Testing

- Harness-free / `app.run_test()` smoke on `PersonasScreen`: the header (`_title_text()`/`#personas-title`) leads with **"Roleplay"** and still appends the `New …`/`- unsaved` suffixes in create/dirty states; switching modes updates the **visible mode descriptor** to the active mode's meaning (a direct assertion, not tooltip inspection); selecting a not-yet-built mode shows the "coming soon" copy (not "not available yet"); Characters/Personas modes still compose + switch unchanged (no regression).
- Follow the existing Personas UI test patterns (`Tests/UI/` personas tests) + the project's CSS-presence pin discipline if any class names change.

## Scope / non-goals

- **P0 does NOT** build Dictionaries or Lore (P1/P2), polish Characters/Personas flows (P3), pre-build a shared-widget framework, or change the route/registry/nav.
- **Forward map:** **P1** — Dictionaries mode over the existing backend + the dictionaries research digest (substitution preview, per-entry type/case/enabled/priority, validation codes, inline toggles, import/export). **P2** — Lore mode + the worldbooks digest (Test-Match diagnostics + near-misses, priority-aware budgeting, expose the already-present selective/secondary-keys/position in the editor, character-level attachment). **P3** — Characters/Personas flow polish across all four dimensions (editor progressive disclosure/discard, the Try-it preview, navigation, find), inheriting this pattern.
