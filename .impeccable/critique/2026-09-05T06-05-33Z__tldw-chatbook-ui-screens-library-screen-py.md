---
target: "Library ▸ Media (critique #5, post wave 4)"
total_score: 21
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 3
timestamp: 2026-09-05T06-05-33Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated; parent traced the P0 and select-mode mechanics in source before scoring). Target tldw_chatbook/UI/Screens/library_screen.py at dev c09717a7cb (all of media wave 4 merged). Live at 235x52 and 100x30 under tmux, real config, no analysis provider. Environment: A and B ran concurrently against the same real profile; the app's "Another copy of tldw is already using this profile" guard fired and was bypassed by the orchestration; the host was out of POSIX semaphores (no live import). The P0's trigger is that environment; its presentation is the product's (task-31220 recorded the same wedge once before, single instance).

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 2 | Delete painted "✓ deleted · 1 item · in Trash" while the DB row stayed is_trash=0 (both assessments independently). Receipts, Loaded markers, review footer otherwise strong. |
| 2 | Match system / real world | 3 | "Match 1 of 1 matches"; raw UTC microsecond timestamp on the permanent-delete confirm. |
| 3 | User control and freedom | 1 | After the false receipt: Undo, Retry, every row and the s key inert until the process was killed. The service wall's only control, Continue, leaves Library for Home. |
| 4 | Consistency and standards | 2 | List filter live-as-you-type vs Reader Find on Enter; Find Prev/Next enabled at No matches while the pager's disable; three load-failure sentences. |
| 5 | Error prevention | 2 | Delete confirmation copy excellent. "○ Delete" shifts one cell on enable; "Restore Delete permanently" fused; focus parks on the type filter after a delete. |
| 6 | Recognition rather than recall | 3 | Review footer and "▶ Import behavior · analysis on" carry state in labels. Keyword hit never explains the match; Rendered/Raw silently absent for article/document. |
| 7 | Flexibility and efficiency | 2 | Deep key vocabulary, but after s from the rail F6/Down/Space no-op with no focus painted; only a click on the one-cell ☐ seeds focus. Trash Restore has no key. |
| 8 | Aesthetic and minimalist design | 2 | List 38 cells at 235x52 vs ~47 at 100x30 (titles truncate on the wider terminal); two 5-cell gutters; More displaces ~19 rows. |
| 9 | Error recovery | 1 | "Couldn't load page 1." with Retry 34 rows below; "Library source services unavailable; retry Library later." unbordered, never self-heals, above a bordered card for a lower-stakes notice. |
| 10 | Help and documentation | 3 | ○ Generate tooltip precise but hover-only, no fix named. Field help strong; F1 always in footer. |
| Total | | 21/40 | Acceptable |

## Design Specificity Verdict

LLM assessment: authored, unevenly. Review layer is this product's (review banner + rewriting footer, Use in Console, empty state naming its three searched fields, transcript sniff refusing #intro as a heading). Browse layer interchangeable (title/type·age row that cannot tell analysed from raw; Export/Trash/Select toolbar; Read/Analysis/Highlights/Info tabs; Analysis tab with no model/date/version/regenerate).

Deterministic scan: detect.mjs exit 0, zero findings over the screen and Widgets/Library (.py not scannable — expected null for a Textual app). Measured live instead: every reachable disabled label 7.25:1 (Legible Disabled Rule ×2, ○ glyph carries state); no label clips at either width; Restore + Analyze row inside the pane at 100x30; F6 ring period-3 (Reader, rail search, Items filter). Detector-grade defects: Reader focus indicator colour-only (border 1.01:1 → 6.96:1, identical glyphs; buttons get a heavy ┃ outline); at 235x52 the only review-set exit is the keyboard-only R chip (Back suppressed in the three-pane shell; "Review these" is not a toggle).

Visual overlays: not applicable (terminal application).

## Overall Impression

Wave 4 holds: disabled grammar, Import header state, review-set layer and the 100x30 composition praised by both assessments unprompted. One event sank the score: a destructive action reported success it did not achieve, and the bulk-mutation interlock then swallowed every recovery control. Biggest opportunity: make the bulk-mutation path honest end to end — receipt from the result, interlock released on every path, recovery never gated by what it recovers from.

## What's Working

- Review sets end to end: scope, position, progress, per-item state, all text, at the point of need.
- The disabled convention passes its own rule: 7.25:1 measured live, state carried by a character.
- The narrow layout: rail collapses, list widens to ~47 cells, titles fit, Restore reachable, footer degrades to five bindings.
- The Import queue's failure row ("✗ failed · <file> · <reason>" + Retry/Dismiss on the row) is the model for the rest of the screen.

## Priority Issues

[P0] A bulk delete reports success it did not achieve, then the interlock wedges every recovery control until restart. Both assessments: select one, Delete, confirm → "✓ deleted · 1 item · in Trash" + "○ Undo" + "Media changed; retry to load a current page." with rows/Export/Select/sort/pager disabled; DB row untouched; Retry inert ×2; Dismiss clears the receipt not the gate; s inert; tab round-trip no help; only killing the process recovered. Mechanism (source, CORRECTED after the report): the bulk path treats "no exception" as success, and that IS sound for the local backend — local_media_reading_service.delete_media_item raises ValueError when MediaDatabase.mark_as_trash returns False and KeyError when the row is missing, and mark_as_trash commits the UPDATE inside its transaction before returning True. So a ✓ with an untouched row is NOT explained by the code read; the observation stands, the mechanism is open (candidates: the assessment's DB read on a stale WAL snapshot from a long-lived connection; a write to a different connection/path; a later revert). PR E Task 1 is an instrumented reproduction before any fix. What the code DOES explain is the wedge: the completion seam raises the mutation gate via reconcile_committed_mutation and refreshes only with authority; handle_library_media_retry, handle_library_media_row and _toggle_library_media_select_mode all early-return while _library_media_bulk_delete_in_flight is set — exactly the trio observed inert; Undo is stale-gated by design. This is task-31220's storage wedge, reproduced twice. Fix: receipt from the result (False = failure with reason); release the interlock in a finally on every path; never gate Retry behind the interlock; Undo enabled iff ✓; focus Undo on receipt. Command: harden.

[P1] Select mode unreachable by keyboard, one-cell mouse target. After s from the rail, F6/Down/Space no-op with no focus indicator; only a click on ☐ seeds focus; row-title clicks do nothing; Done takes sort:'s slot. Mechanism: focus sits on the pane grip after the recompose (task-31567, in my wave-4 area). Fix: focus the selected/first row on entering select mode with a visible ring; whole-row toggle; keep Done out of the sort: slot. Command: adapt.

[P1] Failure states unbordered, unreasoned, recovery out of sight. Three load-failure sentences, none with an adjacent fixing control. The service wall is a 5 s asyncio.wait_for on the source snapshot collapsed into a static string by a bare except at library_screen.py:13303 — never self-heals, Continue ejects to Home. Fix: one recovery callout (tinted border, what/why/what-to-do, Retry inside); distinguish timeout from hard failure; retarget Continue. Command: clarify, then shape.

[P1] The wide layout is narrower than the narrow one: 38-cell list at 235x52 (98-char title truncates) vs ~47 at 100x30; two 5-cell gutters; 83 chars in a 145-cell reader; 3 rows per item; More displaces ~19 rows. The Items-pane floor is my wave-3 adaptive-shell decision, set for the collapse case and never told to grow. Fix: list grows with width; close gutters; one-cell row separation; More as an overlay. Command: layout.

[P2] The row cannot answer "processed?" or "why matched?", and Reader focus is colour-only. Analysed items render identically to raw ones; keyword hits show no reason. Row half = design note task-31278; focus half = the heavy outline buttons already get. Commands: typeset (row grammar), audit (focus visibility).

## Persona Red Flags

Alex: / targets the rail's Search Library box, not Title/keyword, and Enter replaces the canvas with Search/RAG; More costs 19 rows; Find keeps the old verdict until Enter; no clickable review-set exit at 235x52.
Sam: cannot start a multi-select and gets no focus signal; ○ Generate's reason is hover-only; Restore/Delete permanently have no key; after a delete Enter opens the type dropdown beside an inert Undo. Holds: text-carried states, 7.25:1 disabled labels.
Riley: ✓ the DB disagreed with; service wall ×3 incl. after waiting; Continue ejects to Home; stale Media (15) + live Reader for an out-of-band-deleted row >20 s; one unreproduced Find "No matches" anomaly.

## Minor Observations

"Match 1 of 1 matches"; raw UTC timestamp on permanent delete; Restore/Delete permanently fused, no keys; Find Prev/Next enabled at No matches; Export/Trash enabled over a failed empty list; Sets tab disappears on zero results; single result auto-loads silently; ] marks reviewed as a side effect of moving; banner readout stale after a click; Reader shows a live item over Trash; failed-load placeholder says "Select a media item"; Rendered/Raw absent for article/document, no note; rendered H1 centred; Import path field vs one-row field idioms; "Open manager" indented one cell further; review-set toast overlaps the Reader border at 100x30 for several seconds.

## Questions to Consider

- I gated Undo, Retry and row opens behind one interlock flag (ADR-055 one-flag rule). Should any recovery control share the interlock it exists to escape?
- I set the Items-pane floor for the collapse case and it caps the wide layout. Grow the list with width, or is there a third pane that earns the columns?
- Should a bulk mutation paint a receipt before its result is known, or show "deleting…" and resolve to ✓/✗?
- What if the default Media view were the review queue rather than the list?
- Two search boxes; / picks the far one. Which owns the universal key, and should the other be a box at all?
- If analysis is a first-class artifact, what should the Analysis tab carry: model, date, version, regenerate?
