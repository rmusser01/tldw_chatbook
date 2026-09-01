
## Implementation (2026-08-31, branch `perf/task-25812-split-agentic-css`)

Split implemented at BUILD time — `components/_agentic_terminal.tcss` stays
the single source of truth, and `build_css.py` partitions it into the
bundle remainder plus three generated per-screen sheets loaded via
`CSS_PATH` on ChatScreen / LibraryScreen / SettingsScreen (parsed lazily by
`App._load_screen_css` on first visit).

**Mechanism safeguards:**
- The partition is asserted lossless at every build
  (`"".join(units) == text`) — a lossy split silently drops live CSS, the
  incident class the `_settings_splash_theme.tcss` manifest note records.
- Brace counting skips comments; classification strips comments from
  selectors before tokenizing.
- A block moves only when EVERY `#id`/`.class` token belongs to exactly one
  owner; bare-type, multi-owner and unowned blocks stay in the bundle.
- `AGENTIC_SPLIT_PINNED_TOKENS` pins cross-surface tokens found by auditing
  every moved token against Python compose sites: `.settings-input-label`
  is composed by `Widgets/Persona_Widgets/personas_policy_rules_editor.py`,
  so it stays in the bundle (5 of the 6 audit hits were substring/comment
  false positives; this was the one real one).
- `check_bundle_sync.py` extended: all 7 generated sheets must reproduce.

**Measured (same-session, interleaved where it matters):**

| metric | before | after |
|---|---:|---:|
| boot bundle bytes | 672,141 | **470,401** |
| boot rules parsed | 4,231 | **3,304** |
| boot CSS census (ratchet) | 879,439 (BREACHED) | **679,726** |
| bundle cold parse (4 interleaved pairs, median) | ~243 ms | **~156 ms (−36%)** |

Moved cost, paid once each on first visit, all ≥1.5 s after first paint:
console 30.6 ms / library 35.6 ms / settings 16.5 ms.

Ratchet bookkeeping: `MAX_BOOT_PARSED_CSS_BYTES` TIGHTENED 860,000 →
705,000 (measured + standard 25,000 slack, ADR-097 banking convention —
lowering needs no ledger row); anti-vacuity floor re-pinned 700,000 →
600,000 (it tripped on the success case). Only the CSS snapshot was
refreshed; the ui-ready and preimport snapshot rewrites the refresh script
also produced were reverted rather than silently blessing dev's own drift.

Cascade-tie audit: zero exact-selector collisions between moved rules and
later bundle modules.

Cross-day measurement note: yesterday's cold-parse figures are NOT
comparable to today's (machine load differs); the −36% is from
same-session interleaved pairs with non-overlapping ranges.
