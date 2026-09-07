---
target: "Library ▸ Media surface (re-score after critique-fix train #2346/#2350/#2351)"
total_score: 26
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 3
timestamp: 2026-09-03T20-07-51Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: live design review sub-agent · B: detector/evidence sub-agent)

Target: Library ▸ Browse ▸ Media (post critique-fix train #2346/#2350/#2351) — live tmux at 235×52 + 100×30, seeded 6-item profile. A's load-bearing claims re-verified against source before synthesis; no same-profile second instance this session (pgrep-checked).

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Sets picker failed twice then went silently dead; "Read later" gave zero response and persisted nothing; footer advertised walk keys while keystrokes landed in an input |
| 2 | Match System / Real World | 3 | "Match 1 of 1 matches" is line-granular counting; "document · 1m" ambiguous units |
| 3 | User Control and Freedom | 2 | Undo/Cancel/R-exit grammar is strong, but resume/switch/dismiss (the picker) was unreachable all session and Retry on a failed item load never recovered |
| 4 | Consistency and Standards | 3 | Three disclosure patterns on one pane (More expander / chooser strips / modal); four differently named search inputs on one journey |
| 5 | Error Prevention | 3 | Armed confirm + isolated danger row + cap + stale gating; nothing prevents believing a mark saved when it didn't |
| 6 | Recognition Rather Than Recall | 2 | The type chooser renders as an EMPTY bordered box (height math ignores OptionList's 2-row chrome) — selection is blind |
| 7 | Flexibility and Efficiency | 2 | One-press Review these + walk keys are a power dream, but printable-key shortcuts die on invisible focus drift ("s"/"]" typed into inputs) |
| 8 | Aesthetic and Minimalist Design | 3 | Calm and dense, but the Reader caps content at 18 rows while an unstyled 1fr mode container holds ~14 blank rows above it |
| 9 | Error Recovery | 2 | "Couldn't open review sets." has no reason/retry and its except swallows without logging; filter-miss recovery pins "Import media" while Clear filter never rendered |
| 10 | Help and Documentation | 4 | Stamped, accurate user guide; footer micro-help; full-sentence tooltips |
| **Total** | | **26/40** | **Acceptable (high)** |

## Design Specificity Verdict

**LLM assessment (A):** unmistakably authored — the walk grammar (auto-mark-on-leave, m/R, banner + footer progress), the receipt-with-Undo delete flow, ○-prefixed disabled labels with reason tooltips, "(selected)" state-in-text tabs. The armed-delete copy names both recovery paths. The betrayal is execution reliability: in one 25-minute session the same surface produced a dead Sets button, a completion that did not durably persist, an invisible type chooser, and an unresponsive Read later. Authored design, ~26/40 delivery.

**Deterministic scan (B):** detector no-signal on Python (honest null); 0 hardcoded hex; every unicode state marker is word-paired in code AND in the 100×30 render ("○ Export", never bare glyphs — the prior bare-`○ ○ ○` state is confirmed gone); all 18 inline styles are sizing; the 16-notice review-set inventory is specific and severity-tagged, with one mechanical inconsistency ("Review-set storage is unavailable." ships as error twice, warning once). B independently reproduced the multi-row grammar rendering correctly at 100×30, and recorded one unattributed incident: Space in empty select mode at 100×30 blanked the canvas with a pane-grip artifact until palette navigation recovered it.

## Overall Impression

The three P1s and both P2s from the last critique are verifiably gone — the finish line works and announces itself, the action rows are words at every width, documents no longer wear image-failure chrome, and the set has a real banner. What the deeper session exposed underneath is the next layer: a state-dependent storage degradation that makes displayed state diverge from durable state, and two rendering bugs (blind type chooser, Reader wasting its pane) that predate this program. The design language is now consistently excellent; the running system's reliability is the frontier.

## What's Working

1. **The walk model** — auto-mark-on-leave, never-marking Prev, the completion gesture, tombstone-skipping pure logic, banner + footer progress. Original, keyboard-native interaction design.
2. **Destructive-flow grammar** — ○ disabled labels with reason tooltips, danger isolation on its own row, armed confirm naming Undo AND Trash, receipt with Undo. Best-in-class terminal UX writing.
3. **Honest empty states + documentation** — query-echoing zero-match copy, Trash's plain-language empty state, a stamped and accurate user guide.

## Priority Issues

- **[P0] Session-progressive silent storage degradation; displayed state diverges from durable state.** Live sequence: a 6-item set completed in the UI ("All 6 reviewed", clean R-exit) but the DB shows position 5 `done=0`, `active=1`, `completed_at=NULL`; the Sets picker then failed twice ("Couldn't open review sets.") and went silently dead; "Read later" persisted nothing across three presses; a plain row click stuck on "Media item is unavailable." with an ineffective Retry. Each control fails in a different dialect and nothing says "storage is unhealthy". Compounding: the picker worker's `except Exception` (library_screen.py:39972) is the ONE review-set wrapper without traceback logging — the app log contains no trace. No same-profile contention this session (pgrep-verified), so last run's environmental explanation is dead; the uncommitted-transaction signature (app sees its own writes, external reader doesn't) is the lead. **Fix:** log the swallowed exception; one storage-health surface; root-cause the wedged connection/worker. **Suggested command:** /impeccable harden
- **[P1] The type chooser renders zero options.** `choices.styles.height = min(8, max(1, len(options)))` ignores OptionList's 2-row default chrome (the Console popup rule documents this exact cost), so the common 2-option case is an empty bordered band — a core decision made blind. **Fix:** budget +2 rows or strip the inner chrome as `#console-command-popup OptionList` does. **Suggested command:** /impeccable polish
- **[P1] The Reader wastes its pane.** `#library-media-reader-mode-read` has no CSS rule → an unstyled Vertical defaults to 1fr holding ~14 blank rows above the Find bar, while `#library-media-viewer-content` is capped `max-height: 18` regardless of terminal size — content gets ~1/3 of a 45-row pane. **Fix:** `height: auto` on the mode container; let content take remaining height. **Suggested command:** /impeccable layout
- **[P1] The footer advertises keys the focused widget will swallow.** The shortcut branch keys off screen state, not focus: with focus in the rail search the footer showed "] next in set | m | R" while keystrokes were inserted as text (a stray "]" corrupted the filter to a zero-match list). For a keyboard-first brand the footer is the instrument — it must not lie. **Fix:** focus-aware footer context or priority bindings + a visible "typing in <input>" state. **Suggested command:** /impeccable harden
- **[P2] Zero-match filter recovery is wrong and the right one is invisible.** `fresh_zero` doesn't check `canvas.query`, so a filter miss pins "Import media"; meanwhile `#library-media-filter-clear` (no width bound on its row's Input, no shared action class) never rendered at any point live. **Fix:** gate the Import suggestion on an empty query; bound the Input so Clear renders. **Suggested command:** /impeccable onboard
- **[P2] The completion moment has no exit.** After "All N reviewed" the footer keeps advertising "] next in set" (now a silent no-op — the same honest-footer rule, task-28005, that trims Prev/Next at list ends) and offers no next step. **Fix:** at completion swap the footer segment to "R exit review" and stop advertising ]. **Suggested command:** /impeccable polish

## Persona Red Flags

**Alex (impatient power user):** resuming yesterday's batch runs through the one button that failed or silently no-op'd all session; a mid-walk `/` search silently costs him the walk keys and `]` edits his filter; no per-row reviewed ✓ means "what's left" requires walking serially; content confined to an 18-row box forces constant inner scrolling on a large terminal.

**Sam (keyboard-only, no color-only meaning):** genuinely well served by the text idioms ("(selected)", "○ Export", ▸/☑/☐, "✓ reviewed") — the house rule shows. Hard stops: the blank type chooser offers no visible options and no textual highlight feedback; Find's match (line-granular) has no non-color positional marker; invisible focus plus the untruthful footer breaks the one instrument a keyboard-only user navigates by; silent controls give no failure signal at all.

## Minor Observations

"Local Media item" is developer-ese; duplicate "Read" heading under the "Read (selected)" tab; the banner mixes set progress and item state in one line; the More expander shoves tabs+content ~18 rows down; "Open manager" unexplained; Trash header truncates ("Local Trash · 0 i"); storage-unavailable notice ships as error twice / warning once (line 40132); B's unattributed Space-in-empty-select-mode canvas blank at 100×30 (pane-grip artifact suggests Space activated a focused grip — unconfirmed); re-entry resumed the Reader on a different item than last read.

## Questions to Consider

1. Why does a Reader cap its text at 18 rows and spend a third of its pane on a blank band — who is that space for?
2. Is "reviewed" a mode (banner-only) or a property of the items (per-row ✓ in the list)? The current answer forces serial recall of parallel information.
3. Four search boxes with four names on one journey — is search one concept in this product or four?
4. When the data layer wedges, the surface speaks four dialects of failure (toast, silence, error card, dead button). What is this product's single honest voice for "something systemic is wrong"?
