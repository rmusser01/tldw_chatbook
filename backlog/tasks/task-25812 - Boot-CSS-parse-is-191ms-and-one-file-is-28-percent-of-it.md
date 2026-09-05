---
id: TASK-25812
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

## Investigation for AC #1 (2026-08-31) — deferral is feasible, but NOT by moving the file

### The timing window exists

Textual loads a screen's own `CSS_PATH` lazily (`App._load_screen_css`, called
from `push_screen`/`switch_screen`), so per-screen CSS is parsed when that
screen is first built. Measured:

| event | at |
|---|---:|
| first paint (`run_test` entered) | 577.9 ms |
| `ChatScreen.__init__` first called | **2049.6 ms** |

ChatScreen is constructed **1.47 s after first paint**, exactly once. So CSS
moved onto it genuinely leaves the pre-first-paint leg, with ample slack.

### But the file is not Console CSS, despite its name

`components/_agentic_terminal.tcss` is a grab-bag. Of 964 distinct id/class
tokens: 411 `console-*`, **259 `library-*`, 91 `settings-*`, 37 `mcp-*`**,
plus personas, home, notes, acp, prompt. **Moving it to ChatScreen would
break five other screens.** That is very likely why the "split this file"
owner call from the 08-27 and 08-29 reviews was never actioned: the obvious
move does not work.

### It is, however, cleanly splittable

Attributing each rule block to the screen its selectors name:

| owner | bytes | rules | share of attributed |
|---|---:|---:|---:|
| console | 72,257 | 452 | 43.0% |
| library | 45,104 | 268 | 26.9% |
| settings | 17,891 | 94 | 10.7% |
| **MIXED (spans screens)** | **6,074** | **15** | **3.6%** |
| mcp / prompt / approval / personas / home / notes / exchange / internal / acp | 15,702 | 102 | 9.4% |
| (unattributed) | 10,841 | 82 | 6.5% |

**Only 15 rules genuinely span screens.** Console + Library + Settings alone
are 81% of attributed bytes and could each move to their own screen's
`CSS_PATH`.

*Method limit, stated because it bounds the confidence:* the regex attributed
167,869 B of the file's 283,119 B. The remainder is comments, whitespace and
blocks a flat rule-block regex does not capture (nested rules). The shares
above are **of attributed bytes, not of the file** — treat them as the shape
of the split, not as the exact byte savings.

### The constraint any split must respect

`app.py`'s `_get_default_css` records TASK-15450: Textual's parse cache is an
`LRUCache(64)` **per stylesheet**, and a destination tour that reached 94
sources made *every* `Stylesheet.parse()` run fully cold (125–380 ms
measured). That is why widget CSS was consolidated into one source in the
first place. A split into ~6 per-screen sheets takes boot sources from 14 to
~20 — well clear of 64 — but the split must be by SCREEN, not per-component,
or it walks back into that cliff.

### Recommendation for the owner call

Split by owning screen and attach each part to that screen's `CSS_PATH`,
keeping the 15 MIXED rules plus the unattributed remainder in the boot
bundle. Coordinate with TASK-24451, which covers the split itself — this task
is specifically about getting the result OFF the pre-first-paint parse and
clearing the standing ratchet breach.

Do not attempt the one-line version (move the whole file to ChatScreen); it
is measurably wrong.

## Owner decision (2026-08-31): split by screen

Owner: "12 split-by-screen" — implement the split recommended in the
investigation. Console / Library / Settings portions move to those screens'
own CSS, MIXED + unattributed stay in the boot bundle. Implementation
proceeds on its own branch; TASK-24451 (the split itself) is satisfied by
the same change.

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
