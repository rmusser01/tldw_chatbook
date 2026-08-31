---
id: task-25812
title: Boot CSS parse is 191ms and one file is 28% of it
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30'
updated_date: '2026-09-01'
labels:
  - performance
  - css
  - boot
priority: medium
---

## Description (the why)


The boot CSS byte ratchet (ADR-097) is **RED at 878,333 B against a limit
of 860,000 B**, and has been since before this review. Its failure message
says *"Every one of these bytes is parsed before first paint"* — but the
guard prices bytes, and nobody had converted them into time.

Measured: CSS parsing is **one 191 ms hit at boot** (subsequent parses are
`_parse_cache` hits at ~1 ms), which is **≈11–14% of a ~1.7 s cold start**
(module import measured separately at ~1.15 s).

`components/_agentic_terminal.tcss` is **283,119 B — 32% of boot CSS bytes,
1,251 rules (38% of the bundle's rules), ≈54 ms (≈28% of the parse)**. It
also supplied +12,902 B of the +23,613 B growth that pushed the ratchet
red. It was flagged as an owner call in the 08-27 and 08-29 reviews with no
time cost attached; it now has one.

## Evidence

dev `0ef6f3fd4e`. Ablation, three cold interpreters per arm — the file
cannot be parsed standalone because it references variables the bundle
defines earlier, so its cost was measured by removing its segment:

| bundle | rules | cold parse |
|---|---:|---:|
| full | 3,273 | 122.9 / 124.3 / 137.4 ms |
| without `_agentic_terminal.tcss` | 2,022 | 69.0 / 70.0 / 70.0 ms |
| **attributable** | **1,251** | **≈54 ms** |

**Measurement warning, recorded because it cost a round:** in-app timing of
CSS parse is wrong by ~250×. A fresh `Stylesheet` inside a booted app
"parsed" 671 KB in 0.48 ms — 1.4 GB/s, impossible for a Python tokenizer —
and clearing the two module-level caches (`parse_selectors`,
`is_id_selector`) did not change it. The same content in a **cold
subprocess** takes 121 ms (5.5 MB/s). Instrumenting the real
`Stylesheet.parse` independently gives 191 ms and confirms the cold number.

Full method: `Docs/Design/2026-08-30-holistic-perf-review.md` §1.

## Acceptance Criteria (the what)

- [x] Decide, with the owner, whether `_agentic_terminal.tcss` can leave
      the pre-first-paint bundle — the app shows a splash first, so there
      may be a window in which non-initial-screen CSS can parse without
      the user waiting
- [x] If it can be deferred: boot CSS bytes return under the 860,000
      ratchet WITHOUT raising the constant (ADR-097 forbids raising), and
      the parse time saving is measured in a cold subprocess, not in-app
- [x] *(n/a — it could)* If it cannot: record why in the ADR, and shed at least the 18,333 B
      of the current breach from elsewhere on the same path
- [x] Any measurement quoted in the close-out states which method produced
      it; in-app parse timings are not accepted as evidence
- [x] Note: `SCOPED_CSS` does not help — the 08-29 review established
      scoped rules still sit in `self.rules`

## Notes

TASK-24451 covers splitting this file; this task is specifically about
getting it OFF the pre-first-paint parse and clearing the standing ratchet
breach. Coordinate rather than duplicating.

*(This file existed as two copies — the review filing on PR #2258 and the
implementation record on PR #2281 — merged by union here when #2281 landed
on dev first, exactly as both copies' provenance notes anticipated.)*

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

## Implementation Notes (close-out)

Merged to dev as `b62407e258` (PR #2281, squash), owner-directed
("split-by-screen", then "merge it"). Two Qodo review rounds, nine
findings, all fixed or answered with evidence — including a vacuous
cross-surface audit of mine (the worktree NAME matched the path filter), a
comment-brace selector-swallowing bug the demanded unit tests caught, a
staged all-or-nothing build publish, and a build-enforced cross-module
cascade demotion. Final ratchet state: boot CSS census 780,368 B against a
TIGHTENED 806,000 limit (was 860,000; pre-split reality was a breached
879,439). Console sheet boot-loads via `TldwCli.CSS_PATH`; library and
settings parse lazily on first visit.
