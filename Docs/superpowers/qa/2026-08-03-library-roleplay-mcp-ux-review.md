# UX/HCI Review — Library, Roleplay, MCP Screens

Date: 2026-08-03 (updated same day: all 37 findings fixed across PRs #1294/#1318/#1321/#1322)
Method: dual-agent impeccable critique (design review + deterministic/evidence scan per screen, isolated)
Build under review: origin/dev @ b66ab0f8c (worktree `.worktrees/ux-review`)
Evidence: headless real-app captures in `output/ux-review/` at 170x50 and 100x30 (SVG+PNG)
Detector note: `detect.mjs` targets web tech only (`.html/.css/.js/...`); it exits 0 with `[]` on
Python TUI source — no signal. All findings below come from source audit + rendered captures.

## Personas

- **P1 — First-time technical user**: comfortable in a terminal, new to this app; reads docs, tries things methodically.
- **P2 — First-time non-technical user**: researcher/student type; needs plain language, visible guidance, safe defaults; abandons when confused.
- **P3 — Power user (technical)**: daily driver; keyboard-first, wants shortcuts, bulk actions, density, scriptability.
- **P4 — Power user non-technical**: heavy daily use but not a hacker; relies on recognition, consistent patterns, mouse + simple shortcuts; hates jargon and modal dead-ends.

## Scores

| Screen | Nielsen total | Band | Cognitive-load failures (of 8) |
|--------|---------------|------|-------------------------------|
| Library | 21/40 | Acceptable (low) | 6 |
| Roleplay | 27/40 | Acceptable (high) | 4–5 |
| MCP | 22/40 | Acceptable (low) | 5 |

Design specificity: all three are structurally this product (destination-shell idiom, honest-state
engineering) but render as category-interchangeable admin consoles at first paint. The craft is in
the wiring; the first impression is undesigned.

## Running Findings List

Severity: P0 blocking / P1 major / P2 minor / P3 polish. Status: new / confirmed / in-progress / fixed / wontfix.

### Cross-cutting (app chrome, visible on all three screens)

| ID | Severity | Personas | Finding | Evidence | Status |
|----|----------|----------|---------|----------|--------|
| F-001 | P2 | all | Tab bar truncates at ≤100 cols: "8 Workflows" collapses to a bare "8" and Workflows/MCP/ACP/Lab/Logs/Settings become unreachable visually; "More: Ctrl+P" crowds the right edge. No ellipsis or scroll affordance. | library/roleplay/mcp-100x30.png | fixed (TASK-2093, PR #1322) |
| F-002 | P2 | P1, P2 | Nav digit labels ("1 Home … 0 ACP") imply bare-digit keys but the binding is Ctrl+digit — labels lie by omission. | app.py:3493 | fixed (TASK-2093, PR #1322) |
| F-003 | P3 | all | "Tokens: --" footer chip is dead chrome on authoring/config destinations (Roleplay, MCP) — chat-context residue. | roleplay/mcp-170x50.png | fixed (TASK-2094, PR #1322) |

### Library (21/40)

| ID | Severity | Personas | Finding | Evidence | Status |
|----|----------|----------|---------|----------|--------|
| F-010 | P0 | P1, P2 | The landing hub is a dead void: one line of copy in a 130x40 canvas. The real hub — recents, per-source counts, next-action copy — is already implemented (`_hub_state_summary`/`_hub_readiness_summary`, `LIBRARY_EMPTY_NEXT_ACTION_COPY`) and never called. Fix: wire hub summaries into the `canvas_kind == "empty"` branch as clickable next-action rows + recents strip; delete or wire the dead helpers. | library_screen.py:3237,3255,4223 | fixed (TASK-2071, PR #1318) |
| F-011 | P1 | all | Every rail row burns a second line repeating "in Library" (11 rows × 3 lines tall) — pure stutter, and the reason Create is unreachable at 100x30 and the Details status section is clipped even at 170x50. Fix: one-line rows; keep meta line only where it discriminates handoffs. | library_rail.py:221-226 | fixed (TASK-2072, PR #1318) |
| F-012 | P1 | P3, P4, P2 | Zero keyboard affordances on the landing state: footer shows only quit/palette; the `u` (use in Console) hint registers only when the Search row is selected; no `/` focus-search; F6/global keys unadvertised; teaching copy lives in hover-only tooltips. | library_screen.py:1413-1417 | fixed (TASK-2073, PR #1318) |
| F-013 | P1 | P2, P4 | Jargon wall with no plain-language layer: Ingest, RAG, Skills, Collections, Runtime are load-bearing labels with zero gloss; the one guidance sentence presumes "ingest". Fix: per-row dim subtitles in plain language; button → "Add content…". | library_shell_state.py:7 | fixed (TASK-2074, PR #1318) |
| F-014 | P2 | P2 | Count inconsistency + `Prompts: N/A \| Chats/Notes: N/A \| Media: N/A` footer — DB telemetry in user chrome; on a fresh library the whole screen reads "broken". Fix: uniform count policy (dim — while loading); move DB sizes to Details/Logs. | db_status_manager.py:69 | fixed (TASK-2224, PR #1318) |
| F-015 | P2 | P3 | At 100 cols the rail truncates "Conversations (0)" → "Conversations …" (count ellipsized exactly when counts matter) and the search placeholder "Search Library…" truncates to "Search". | library-100x30.png, library_screen.py:15581 | fixed (TASK-2076, PR #1318) |
| F-016 | P2 | P2, P4 | Search input renders as a borderless black void with stray left-edge artifacts — reads as broken, not minimal. | library-170x50.png | fixed (TASK-2076, PR #1318) |
| F-017 | P3 | P3 | "Study decks" sits under Create but is a Study handoff ("Continue in Study") — mis-grouped. | LIBRARY_STUDY_HANDOFF_MODES | fixed (TASK-2077, PR #1318) |
| F-018 | P3 | P3 | Disabled "Export selected" (conversations/media/notes canvases) and skill-trust Unlock/Review/Approve buttons carry no reason tooltips; editor Discard same. Contrast: rail rows and workspace handoff do it well. | library_conversations_canvas.py:100-105, library_skills_canvas.py:1154-1175 | fixed (TASK-2078, PR #1318) |
| F-019 | P3 | P3 | Skill editor `ctrl+s`/`escape` bindings advertised nowhere; no hint line in the editor canvas. | library_screen.py:877-884 | fixed (TASK-2078, PR #1318) |
| F-020 | P3 | P2 | Rail scrolls (`overflow-y: auto`) but shows no scrollbar/affordance — clipped sections (Create, Details) are undiscoverable. | library-100x30.png | fixed (TASK-2079, PR #1318) |
| F-021 | P3 | P2 | Inspector next-action copy is architecture-talk: "Library remains a hub; Notes, Media, Search/RAG, and Study own deeper work." | library_screen.py:324-331 | fixed (TASK-2080, PR #1318) |

### Roleplay (27/40)

| ID | Severity | Personas | Finding | Evidence | Status |
|----|----------|----------|---------|----------|--------|
| F-030 | P0 | P2, P4 | Library toolbar clips at supported widths: at 100x30 only "New" and "Sort: Name" survive — Import, Duplicate, Tag are gone, while the empty-state copy still says "use New or Import". PNG-card import is THE roleplay onboarding path. Fix: wrap toolbar or overflow "⋯" with New pinned; add rendered-layout test at 100x30 and 80x24. (Compact split threshold is 90, so it doesn't engage at 100.) | roleplay-100x30.png, personas_screen.py:368 | fixed (TASK-2081, PR #1321) |
| F-031 | P1 | P1, P2 | First paint is a void + a wall of dead controls: nothing selected on mount → center shows one sentence, Inspector shows 5 disabled buttons + disabled checkbox + a false "Validation: OK". Fix: auto-select first library row; hide action stack when kind is None; suppress Validation line pre-selection. | personas_inspector_pane.py:138 | fixed (TASK-2082, PR #1321) |
| F-032 | P1 | P2, P4 | Three indistinguishable Console CTAs ("Attach to Console" / "Start Chat" / "Open in Console") + gating copy in app-topology speak ("Console blocked: select an item"). Fix: one primary "Chat now" + one secondary "Send to Console draft"; readiness copy in intent language. | personas_inspector_pane.py:346-364 | fixed (TASK-2083, PR #1321) |
| F-033 | P2 | all | Top band stacks five strips (nav, title+subtitle+Ready, purpose line, "Characters: 1", mode strip) and "Characters" appears 3× in 3 lines; library repeats "1 character" at bottom. ~23% of screen at 100x30 before content. | roleplay-170x50.png | fixed (TASK-2084, PR #1321) |
| F-034 | P2 | P2, P4 | Naming undermines the mental model: nav "Roleplay" ≠ header "Roleplay & Chat Dictionaries" ≠ mode "Personas" ("assistant profiles" — genre convention is personas = who YOU play). | personas_screen.py:771,3133 | fixed (TASK-2085, PR #1321) |
| F-035 | P2 | P2 | Empty-state copy is non-adaptive by design: renders "use New or Import" even with 1+ characters in the library; copy is center-aligned in a huge void (reads as broken layout) and right-aligned at some widths. | personas_screen.py:263-269,870-874 | fixed (TASK-2086, PR #1321) |
| F-036 | P2 | P2 | Inspector "Conversations" header dangles with nothing beneath pre-selection — `show_conversations(())` called without empty_copy. | personas_inspector_pane.py:198-209 | fixed (TASK-2087, PR #1321) |
| F-037 | P2 | P4 | Export JSON / Export PNG / Delete / card Edit disabled with no reason tooltip in the no-selection state (Attach/Start Chat do it right — apply the same pattern). | personas_inspector_pane.py:406-414 | fixed (TASK-2088, PR #1321) |
| F-038 | P3 | P3 | Accelerators exist but are undiscoverable: F6 pane cycle, Ctrl+1-4 mode jumps, Space dictionary toggle appear in no footer/chip/tooltip; footer shows 3 of ~10 bindings. | personas_screen.py:425-460 | fixed (TASK-2089, PR #1321) |
| F-039 | P3 | P2 | "Preview conversation" bar stranded at bottom center, visually detached from the canvas it belongs to — the screen's actual delight is two clicks and one expand away. | roleplay-170x50.png | fixed (TASK-2090, PR #1321) |
| F-040 | P3 | P3 | No bulk operations (multi-select delete/export across a large character library); sort is click-to-cycle only (no key, no menu); restore_state round-trips only Characters mode — selection lost on Personas/Dictionaries/Lore after Console→back. | personas_screen.py | fixed (TASK-2091, PR #1321) |
| F-041 | P3 | P2 | Disabled "Include assigned voice profile" checkbox is so dim it reads as a dark gap in the Inspector stack. | roleplay-170x50.png | fixed (TASK-2092, PR #1321) |

### MCP (22/40)

| ID | Severity | Personas | Finding | Evidence | Status |
|----|----------|----------|---------|----------|--------|
| F-050 | P1 | all | Recovery callout clipped mid-sentence at BOTH widths — "○ tldw_chatbook (built-in): Disabled in config ([mcp].enabled =" (170c) / "…: Disabled" (100c). The one sentence explaining the screen's only problem state is unreadable, and it's config-file syntax as user copy. Fix: shorten to actionable fact + make the callout itself perform the fix. | mcp_servers_mode.py:583-591, readiness.py:551 | fixed (TASK-2061, PR #1294) |
| F-051 | P1 | P1, P2 | First run frames an opt-in as failure: built-in server ships disabled → "0 of 1 servers ready — 1 needs setup" + a problem callout on a pristine install. First emotional beat is false alarm. Fix: exclude built-in from readiness math or add a distinct OFF/opt-in state with an Enable action. | readiness.py:208 | fixed (TASK-2062, PR #1294) |
| F-052 | P1 | P2, P4 | Jargon wall, no onramp: "MCP", "scoped tools", "audit readiness", "stdio", "Transport", "Profile id", "legacy control-plane action runner" — not one plain sentence says what MCP is or whether you need it. Fix: one-line explainer; Profile id → Name, Transport → Connection. | mcp_screen.py:120, mcp_profile_form.py:64-96 | fixed (TASK-2063, PR #1294) |
| F-053 | P2 | P4, P2 | "Advanced…" in the inspector is a one-way door: pressing it persists `advanced_visible=True` forever (re-composed on every future visit) with no hide path, revealing jargon content ("Local control plane"). Unlabeled modal dead-end a mouse user will find by curiosity. | mcp_inspector.py:752-774,831-839 | fixed (TASK-2064, PR #1294) |
| F-054 | P2 | P2 | Inspector empty state is dead space (~25% of screen): "Select an item to inspect." teaches nothing; at 100x30 it clips mid-word ("Select an item to inspe", ds-status-badge fixed height 1). Fix: contextual empty copy; pre-select the single problem row on load. | mcp_inspector.py:731 | fixed (TASK-2065, PR #1294) |
| F-055 | P2 | P3 | Footer advertises "space cycle permission" in all four modes but the key only works in Permissions with the matrix focused; `t` from Servers mode force-switches to Tools mode and notifies "Select a tool first." A shortcut bar that lies trains users to ignore it. Fix: context-sensitive footer; make `t` a no-op hint. | mcp_screen.py:30-41 | fixed (TASK-2066, PR #1294) |
| F-056 | P2 | P3, P1 | No Escape-to-cancel on ANY inline form (profile form, import, mutations, delete-confirm, test-tool) — buttons only; and focus is never moved into a form on open (keyboard users must Tab to it). | mcp_profile_form.py:126-127 et al. | fixed (TASK-2067, PR #1294) |
| F-057 | P2 | all | Effectively unusable below ~120 cols: at 100x30 the table loses Tools/Auth columns with no scroll affordance, summary line clips at canvas edge, rail row truncates. No collapse strategy. | mcp-100x30.png | fixed (TASK-2068, PR #1294) |
| F-058 | P3 | P2, P4 | Six readiness glyphs (● ◐ ○ ! ∅ ◌) + ⌂ built-in marker with no legend on the Servers canvas (Permissions has one) — status carried by recall, not recognition; ⌂ renders like an up-arrow artifact. | mcp_rail.py:78 | fixed (TASK-2069, PR #1294) |
| F-059 | P3 | all | Same fact triple-stated for one server (summary line + table row + callout, plus rail duplicates a 4th time); "Auth: none" vs Tools "—" inconsistent empty-cell copy. | mcp_servers_mode.py:562-563 | fixed (TASK-2070, PR #1294) |
| F-060 | P3 | P2 | No "what is MCP / do I need this?" reassurance anywhere; purpose line is a tautology ("Manage MCP servers…" on a screen titled MCP). Rail has no empty state at zero servers; "No scope entities" disabled Select has no reason tooltip. | mcp_rail.py:243-252,333-341 | fixed (TASK-2063, PR #1294) |

## Assessment Synthesis

### Where A (design review) and B (evidence) agree

- Library's "in Library" stutter and landing void (both flagged independently with the same evidence).
- Roleplay toolbar clipping at 100x30 hiding Import — A called it P0 from the screenshot, B confirmed the mechanism (`PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH = 90` never engages at 100 cols; Horizontal toolbar doesn't wrap).
- MCP callout truncation — A read it as the emotional center of the screen, B pinned it to `compact` Button height-1 at `mcp_servers_mode.py:583-591` vs ~73-char text.
- Hidden/lying shortcut disclosure on all three screens (Library's conditional `u`, Roleplay's 3-of-10 footer, MCP's all-modes `space`).

### What B caught that A missed

- No Escape binding on any MCP inline form + focus never moved into forms (A noted dead-ends generally; B enumerated the exact missing bindings).
- "Advanced…" persistence mechanism (`advanced_visible=True` recomposed every visit) — A flagged the trap, B proved it's forever.
- Roleplay's dangling "Conversations" header root cause (`show_conversations(())` without `empty_copy`).
- Library skill-editor ctrl+s/escape unadvertised; skill-trust buttons without reason tooltips.

### False positives (retracted)

- Detector zero-findings on all three files — tool inapplicable to Python, NOT a clean bill.
- Roleplay: row meta/greeting truncation (intentional ellipsis, documented); footer hiding unavailable shortcuts (deliberate task-445 design); "1 character" singularization (already handled).
- MCP: `show=False` bindings (advertised via footer + tooltips — acceptable); the full-width underline under the table row is the DataTable cursor, not a border bug; Select notch glyphs are standard Textual.
- Library: "Export selected" no-tooltip is borderline (adjacent "N selected" counter explains it); "in Library" was designed to discriminate handoff rows — but nearly all rows are "in Library", so the distinction is currently invisible.

### Cross-screen themes (the real story)

1. **First paint is undesigned everywhere.** All three screens open on empty/sparse states that treat "nothing selected/nothing configured" as an afterthought: Library void, Roleplay wall-of-disabled, MCP false-alarm. The empty state IS the product for every first-time user.
2. **Jargon is load-bearing.** Ingest, RAG, Personas, Attach, Console, MCP, stdio, scoped tools — every screen's critical action is named in insider vocabulary with zero inline gloss. P2/P4 personas are screened out at the label level.
3. **Shortcut disclosure is inconsistent and occasionally false.** Four different policies across three screens (conditional, partial, all-modes-static, unadvertised), one of which (MCP `space`) advertises a key that doesn't work in context.
4. **Disabled-state discipline is the app's real strength — but applied unevenly.** MCP's gated-buttons-with-reasons (+ enforcing test) and Roleplay's Attach/Start Chat readiness lines are excellent; Export/Delete/Edit buttons on Roleplay and Library's skill-trust buttons lack the same treatment.
5. **100x30 is a supported size that nobody walks.** Toolbar clipping, count truncation, lost table columns, clipped inspector text, unreachable tab destinations — every screen degrades silently.
6. **Strong engineering sits unwired.** Library's hub summaries, Roleplay's adaptive empty copy, MCP's humanized readiness reasons — the code to fix the worst problems already exists; the composition layer never calls it.

### Top 5 moves (highest leverage, ordered)

1. **MCP: fix the clipped callout + reframe built-in as opt-in** (F-050, F-051) — small code, huge trust payoff; the screen currently opens every relationship with a false alarm and an unreadable explanation.
2. **Library: wire the hub that already exists** (F-010) — dead code renders recents/counts/next-actions; turns the worst first impression into the best.
3. **Roleplay: stop clipping the toolbar** (F-030) — the primary onboarding verb is invisible at a supported terminal size; add a rendered-layout regression test.
4. **All: plain-language subtitles on jargon labels** (F-013, F-034, F-052) — one dim line per destination/mode; the single cheapest fix with the widest persona coverage.
5. **All: one shortcut-disclosure policy** (F-012, F-038, F-055) — context-sensitive footer everywhere; never advertise a key that doesn't work in the current mode.

## Open Questions

1. The code already computes recents, counts, and next-actions for Library's hub — and never renders them. Is the 11-row rail the design, or compensation for a hub that never shipped?
2. MCP sits between Workflows and ACP in primary nav, but a non-developer has no job there. Destination or an Advanced section of Settings?
3. Roleplay is named for play but architected for administration (list/detail/inspector). What if the preview conversation were the center pane and the card/editor lived in a rail?
4. What terminal sizes does the project actually commit to — and which has a human walked end-to-end with a fresh install?

## Suggested Follow-ups

- `$impeccable onboard` — first-run/empty states on all three screens (F-010, F-031, F-051)
- `$impeccable clarify` — jargon layer + copy (F-013, F-032, F-034, F-050, F-052)
- `$impeccable adapt` — 100x30/80x24 layouts (F-001, F-015, F-030, F-057)
- `$impeccable harden` — disabled reasons, escape paths, focus-into-form (F-037, F-053, F-056)
- `$impeccable distill` — header-band and status redundancy (F-011, F-033, F-059)

---

# Post-fix Re-review (2026-08-04, dev @ fd6ff1aa7 — all four PRs merged)

Method: same dual-agent critique, fresh captures in `output/ux-after/` (170x50 + 100x30).

## Score delta

| Screen | Before | After | Delta |
|--------|--------|-------|-------|
| Library | 21/40 | 27/40 | +6 |
| Roleplay | 27/40 | 29/40 | +2 |
| MCP | 22/40 | 31/40 | +9 |

Snapshots: `.impeccable/critique/2026-08-04T16-07-01Z__*` (trend lines now show 21→27, 27→29, 22→31).

## What the re-review confirms

- MCP's first run no longer opens with a false alarm: "Built-in server is off — enable it to let MCP clients use chatbook's tools." + working Enable affordance + glyph legend.
- Library's landing is a real hub (counts, next-action triad), rail rows are one line with subtitles, sections reachable at 100x30, `/` focus-search advertised.
- Roleplay's first paint is designed: auto-selected character, honest inspector, one primary CTA ("Chat now") + one secondary, merged purpose line with count, full accelerator footer.
- Chrome: `⌃`-digit hints, "More ›" pager, no dead Tokens chip.

## Remaining findings (new, post-fix)

Library (27/40):
- [P1] CTA identity crisis: "Add content…" (rail) vs "Import media" (hub + Import/Export row) open the same canvas — pick one canonical label.
- [P1] F-013 subtitles truncate into noise at real widths ("imported…", "saved…") — rewrite to a ≤16-cell budget or drop below a width threshold.
- [P1] Landing keyboard story is one working key (`/`); hub CTAs need single-letter accelerators + a rail-focus key.
- [P2] Canvas void remains: render recents as clickable rows, not one dim line.

Roleplay (29/40):
- [P0] At 100x30 the character card is displaced by empty Dictionaries/World Books panels — collapse empty attachment panels to one line; give the card a real min-height.
- [P1] ~10-line dead void between Dictionaries and bottom-docked World Books at 170x50 — fix the sizing contract instead of the dock workaround.
- [P1] Four names for the Console handoff ("Chat now" / "Send to Console draft" / "Continue this chat in Console" / "ctrl+enter draft") — consolidate to one pair everywhere.
- [P2] Disabled voice-profile checkbox still renders as a dark smear — hide it when no profile is assigned instead of disabling.
- [P2] Preview conversation (best learning feature) is invisible behind a subdued toggle — rename to state the payoff ("▸ Try a test chat (nothing saved)").

MCP (31/40):
- [P1] Fresh-install status contradicts itself: banner "off" (● ready glyph) vs table "○ Needs setup" vs callout "turned off — Enable" — give off/opt-in its own display state and drop the alarm glyph.
- [P1] Inspector dead on a fresh install: pre-select the single built-in row too (its detail is informational, not alarmist).
- [P2] Plain-English explainer should lead the header; expand "MCP" once; demote the jargon purpose line.
- [P2] Kill-switch copy under-sells a global toggle — label its blast radius persistently.
- [P3] Rail glyph legibility; legend is load-bearing but dim/bottom/wrapping — consider word-badges at width or legend under the Servers heading.

These are filed for triage as a possible round 2; not yet backlog tasks.
