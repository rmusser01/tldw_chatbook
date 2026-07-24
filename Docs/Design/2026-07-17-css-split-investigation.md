# Per-screen CSS split investigation (task-262)

**Date**: 2026-07-17
**Scope**: Investigation only — no code or CSS changed. Satisfies task-262 AC#1
(selector-dependency audit) and AC#2 (reasoned decision).
**Analyzed against**: current `dev` (worktree at `bbb2d197`; the numbers below
were re-measured there — the perf-audit's "16,064 lines / 2,278 rules" predates
the legacy-UI removals, the stylesheet is now **15,042 lines / ~2,130 top-level
rules / 301 KB**).
**Environment**: Textual 8.2.7, repo venv (`.venv`, Python 3.12), Apple-silicon
macOS. Probe scripts lived in the session scratchpad
(`css-probe/{probe_lazy,probe_vars_cascade,probe_real_app,timing,audit,dead_check}.py`);
they are ephemeral — every load-bearing number and the probe designs are
recorded here.

---

## Verdict: DO NOT split per-screen with `Screen.CSS_PATH`

The naive split is a net loss. Screen CSS **is** parsed lazily (good), but
Textual's screen-CSS load path bypasses its parse cache and re-parses **every
source already loaded — including the remaining global bundle — on the UI
thread at each screen's first visit**. The split converts one ~90–145 ms
startup cost into a ~70 ms startup saving **plus** a ~40–90 ms first-visit jank
on each of ~10–15 screens (aggregate ~400–1,000 ms of extra parse work per full
tour), plus a crash-class `$variable` failure mode and a full-app visual-QA
bill. Two cheaper levers exist (§7): pruning genuinely dead CSS (~15 % of
selector tokens are referenced nowhere in the repo), and — only if the
first-paint budget still misses after tasks 285/257/258 — a **two-phase
deferred-bundle load** that gets the same startup win with exactly one
off-critical-path reparse and no per-screen mechanics.

---

## 1. The "monolith" is generated — a split would be about loading, not authoring

`tldw_chatbook/css/tldw_cli_modular.tcss` is **generated** by
`tldw_chatbook/css/build_css.py`, which concatenates 45 module files
(`core/ → layout/ → components/ → features/ → utilities/`) in a fixed manifest
order with `/* ===== MODULE: … ===== */` banners. Verified: running
`build_css.py` on a copy reproduces the committed monolith **byte-identically**
(modulo the `Generated:` timestamp line), and the git histories of the monolith
and the module sources move in lockstep. So authoring is already modular; the
only question task-262 asks is whether to *load* it differently.

Side findings while verifying:
- Seven on-disk CSS files are **not in the manifest** and are never loaded:
  `components/_unified_sidebar.tcss`, `components/loading_states.css`,
  `features/_chatbooks_improved.tcss`, `features/_ingestion_rebuilt.tcss`,
  `features/_new_ingest.tcss`, `features/_index.tcss`, and root `main.tcss`.
  Zero runtime cost, but they confuse authoring — prune candidates.
- `UI/Chat_Window_Enhanced.py:72` sets `CSS_PATH = "css/features/_chat.tcss"`
  on a **`Container`** — Textual only honors `CSS_PATH` on `App`/`Screen`, so
  this is dead, misleading code (harmless; cleanup candidate).
- `UI/Screens/writing_screen.py:17` sets `CSS_PATH = None` (harmless no-op).

## 2. Textual mechanics (8.2.7, from source + verified empirically)

Source references are into the venv's `textual/app.py` and
`textual/css/stylesheet.py`.

1. **App CSS** parses once inside `App._process_messages` startup
   (`stylesheet.read_all(self.css_path)` + `_get_default_css()`), before first
   paint.
2. **Screen CSS is lazy.** `App._load_screen_css(screen)` (app.py:2822) runs at
   `push_screen`/`switch_screen`/mode-switch time, not app init. Confirmed
   empirically (§3, probe 1).
3. **The lazy load carries a full-reparse tax.** `_load_screen_css` calls
   `Stylesheet.reparse()` (stylesheet.py:409), which builds a **fresh
   `Stylesheet`** — with a fresh, empty `_parse_cache` — re-adds every source,
   and re-parses all of them from raw text. The app-level bundle is re-parsed
   from scratch on the UI thread at every *first* visit of every screen that
   contributes a new CSS source. Second visits are free (`has_source` dedupe;
   the app builds a fresh Screen instance per navigation, which is fine — dedupe
   is by path, not instance).
4. **Today's flow never pays that tax.** Widget `DEFAULT_CSS` added after
   startup (`Widget._post_register` → `add_source` → next `rules` access →
   `parse()`) goes through `_parse_rules`, which **does** hit the LRU parse
   cache — the monolith is not re-tokenized. The asymmetry (`parse()` cached vs
   `reparse()` uncached) is exactly why adding `Screen.CSS_PATH` makes
   navigation strictly worse.
5. **`$variables` do not cross files.** The tokenizer accumulates `$name:`
   definitions per source (`substitute_references` copies the app-theme
   variable map per parse). The `$ds-*` design tokens defined in
   `core/_variables.tcss` inside the bundle would be **undefined** in any split
   file; only Textual theme variables (`$panel`, `$primary`, …) from
   `App.get_css_variables()` are shared. Every split file must have the
   variables preamble prepended by the build step.
6. **Screen CSS_PATH rules are unscoped and permanent.** `stylesheet.read()`
   applies no scope (only inline `Screen.CSS` gets `SCOPED_CSS` treatment), and
   sources are never unloaded. After one visit, a screen's rules match globally
   forever — identical steady-state to today's monolith, i.e. the split never
   reduces long-session rule-matching cost, it only re-orders when parse cost
   is paid.
7. **Cache-thrash watch item (not currently a problem).** `_parse_cache` is
   `LRUCache(64)` per stylesheet. The real app has only **16 sources after
   startup / 27 after further mounts** (measured, §3 probe 4), so no thrash
   today. If mounted-widget-class diversity ever exceeds 64 sources, every
   post-startup `parse()` becomes a sequential-scan cache wipe and first-visits
   start re-parsing the monolith even without any split.

## 3. Empirical probes (all run with the venv Python, real stylesheet)

**Probe 1 — lazy vs eager + reparse tax.** Probe app with the real 15,042-line
monolith as `App.CSS_PATH`; `ScreenB.CSS_PATH` = variables preamble +
`features/_media.tcss` (804 lines). Instrumented
`Stylesheet._parse_rules/parse/reparse` at class level (so the fresh Stylesheet
inside `reparse()` is captured). Result:

| phase | events | wall |
|---|---|---|
| startup (`run_test`) | monolith parsed once, `cached=False`, **72.1 ms**; screen file **not** parsed | — |
| first `push_screen(ScreenB())` | `REPARSE n_sources=7` **71.4 ms** = monolith again `cached=False` 66.9 ms + screen file 3.3 ms | **95.9 ms** |
| pop + second push (fresh instance) | no parse events | **23.3 ms** |

→ **Lazy: confirmed. Reparse tax: confirmed.** The decisive pair of facts.

**Probe 2 — variables.** App CSS defines `$probe-var`; screen CSS references it
without declaring. `push_screen` **fails with
`UnresolvedVariableError: reference to undefined variable '$probe-var'`** — an
exception out of the navigation path, i.e. crash-class, not graceful.

**Probe 3 — cascade + leak.** Equal-specificity `#shared-lab` rules in app CSS
(red) and screen CSS (green): on the screen, green wins (screen source is later
→ wins ties) while non-conflicting app properties still apply. After popping
back, a same-id widget on the **main** screen is now green too — screen-file
rules leak globally and permanently once loaded (parity with today's monolith,
so not a regression — but it kills any "scoping" rationale for the split).

**Probe 4 — the real app.** `TldwCli` under `run_test` (sandboxed `$HOME`;
note: the branch's schema v20 refuses the live user DB at v21 — never point
probes at the real config). Results: startup parse of the monolith
**143.9 ms** cold (squarely confirming the audit's 88–130 ms band; the warm
loop in Probe 5 is faster because the tokenizer/caches are hot); **16 sources
after startup, 27 after additional mounts** (LRU 64 not approached); mounting a
never-seen widget class post-startup produced **no** monolith re-parse (cache
hit) — today's navigation pays no CSS parse cost.

**Probe 5 — parse-time attribution** (direct `Stylesheet.parse()`, median of 5,
warm):

| bundle | lines | median parse |
|---|---|---|
| FULL monolith | 15,042 | **74.6 ms** |
| GLOBAL-only (minus the 17 per-screen candidate modules) | 7,702 | **36.0 ms** |
| all 17 candidates as one file (+vars preamble) | 7,359 | 32.9 ms |
| largest single candidate `_wizards` (+vars) | 1,207 | 5.7 ms |
| `_embeddings` (+vars) | 1,025 | 4.3 ms |
| `_search-rag` (+vars) | 820 | 3.8 ms |

→ Best-case startup saving from a maximal split: **~38 ms warm / ~70 ms cold**
(≈ half the parse). That is the entire ceiling, and the reparse tax charges the
~36 ms-warm global-bundle parse back at every screen's first visit.

## 4. Selector-dependency audit (AC#1)

**Method**: parsed every manifest module's top-level selectors (1,792 distinct
class+id tokens across ~2,130 rules); matched each token against
identifier-level token sets of all non-CSS Python files; attributed referencing
files to screens via an import-reachability graph BFS'd from each
`UI/Screens/*_screen.py` (edges from static `import`/`from` statements,
excluding the `app.py` hub, which TYPE_CHECKING imports would otherwise turn
into "everything reaches everything").

**Totals** (class+id tokens):

| bucket | tokens | share | meaning |
|---|---|---|---|
| EXCLUSIVE (one screen's import cone) | 744 | 41.5 % | could move per-screen |
| SHARED (2+ screens) | 427 | 23.8 % | must stay global |
| CHROME/other (only files no screen reaches: app chrome, dialogs, or orphaned legacy UI) | 265 | 14.8 % | global or dead-in-waiting |
| DEAD-candidate (no Python reference) | 356 | 19.9 % | see refinement below |

**Dead-bucket refinement** (whole-repo grep incl. Tests/docs): 12 are Textual
*component classes* (`.datatable--cursor` etc. — live, styled internals never
named in Python); 73 appear in tests/docs/other; **271 tokens (15.1 %) are
referenced nowhere in the repo**. Heaviest truly-dead modules:
`_search-rag` 42, `_embeddings` 35, `_ingest` 28, `_wizards` 28,
`_evaluation_unified` 27, `_chat` 17, `_splash` 17. Caveat: dynamically built
class names (e.g. `f"status-{state}"` patterns → `.status-blocked` in
`_workbench`) can masquerade as dead — every prune needs a per-token manual
check.

**Per-module picture** (lines/rules; classification):

| module | lines | rules | classification |
|---|---|---|---|
| core/* (4 files) | 88 | 8 | **global** — `$ds-*` token definitions + base |
| layout/* (5 files) | 820 | 116 | **global** — shell/sidebar/tab chrome, shared everywhere |
| components/_agentic_terminal | 4,502 | 537 | **global** — the Console-rail design system; exclusive tokens split across chat:171, library:149, settings:41, mcp:31, personas:15 → multi-screen incl. the **default** screen; unmovable |
| components/_workbench | 214 | 34 | global (chat/stts) |
| components/_buttons/_forms/_lists/_messages/_widgets etc. | ~830 | 130 | **global** — shared form/list/message primitives |
| components/stats_screen.css | 234 | 37 | per-screen candidate (stats; 25/34 tokens exclusive) |
| components/splash_viewer.css | 226 | 30 | mostly orphaned (9 dead, 12 shared-by-name) |
| features/_chat + _chat_tabs | 811 | 116 | **global in practice** — chat/Console is the default first-paint screen |
| features/_splash | 146 | 23 | **global** — splash renders before any navigation |
| features/_conversations | 372 | 58 | candidate, but 18 shared / 13 dead tokens |
| features/_media | 749 | 114 | candidate; 0 provably-exclusive, 41 shared-by-name, 41 chrome — needs rule-level surgery |
| features/_search-rag | 765 | 127 | candidate; 41 exclusive (search), **51 dead-candidates** |
| features/_llm-management | 338 | 52 | candidate (llm_screen) but generic `.action_*` names shared repo-wide |
| features/_tools-settings + config_search | 493 | 77 | candidate (tools_settings: 28 exclusive) |
| features/_ingest + _ingest_tldw_api_tabs | 818 | 129 | candidate; heavy dead/chrome (31 dead) |
| features/_evaluation_unified | 400 | 67 | candidate; 29 dead-candidates |
| features/_embeddings | 970 | 153 | **mostly orphaned** — 80 tokens only in screen-unreachable files, 40 dead; prune target more than split target |
| features/_wizards | 1,152 | 189 | candidate (chatbooks: 53 exclusive) but 41 shared |
| features/_chatbooks, _code_repo, _coding, _metrics, _tab_dropdown, feature_alerts | 949 | 136 | small candidates (code_repo/coding → coding_screen: 37 exclusive) |
| utilities/* (3 files) | 39 | 2 | **global**; `.hidden` is `!important` (order-safe), `.disabled` is not (tie-order-sensitive) |

**Quantification (AC#1 answer)**: the maximal per-screen candidate set is 17
modules ≈ **7,300 lines / ~1,145 rules (≈ 49 % of lines, 54 % of rules)**;
**~7,700 lines must stay global** (design tokens, layout, the 4.5 k-line
Console/agentic-terminal design system used by the default screen, shared
components, chat/splash, utilities). But inside the candidate set only ~200 of
~690 class/id tokens are *provably* single-screen; ~280 are shared-by-name
across screens (genuine sharing — one rule styling `.action-button` used
everywhere) — so most modules cannot move wholesale; a safe split would need
**rule-level** extraction with per-rule audits, multiplying the QA burden.

## 5. Cost/benefit of the naive `Screen.CSS_PATH` split

| | value |
|---|---|
| Startup gain (ceiling, maximal split) | −38 ms warm / ≈ −70 ms cold first paint |
| New cost per screen first-visit | +36–45 ms warm (est. 70–90 ms cold) UI-thread jank (`reparse()` of global bundle + all previously loaded screen files — grows as more screens are visited) |
| Aggregate per full screen tour | ~+400–1,000 ms extra parse work vs today's one-time 144 ms |
| New failure mode | `UnresolvedVariableError` **at navigation time** for any split file missing the `$ds-*` preamble (crash-class; probe 2) |
| Ordering risk | utilities (`.disabled`) currently win equal-specificity ties by coming last; split screen CSS loads after them and flips those ties |
| QA bill | per-screen visual QA on every split screen (regressions are visual, not crashes) — against ~49 % of the stylesheet |
| Steady state | unchanged — screen rules leak globally and permanently anyway (probe 3) |

**Decision (AC#2): no split.**

## 6. What was wrong / right in the original hypothesis

- Right: the parse is real (~90–145 ms) and per-screen lazy loading exists.
- Right: roughly half the stylesheet is per-screen-shaped.
- Wrong: "lazy" does not mean "cheap" — `_load_screen_css` → `reparse()` is
  cache-bypassing and cascade-wide; the split relocates and **multiplies** the
  cost instead of removing it.

## 7. Recommended alternatives (in order)

1. **Dead-CSS prune (do this; no mechanics risk).** 271 truly-unreferenced
   tokens, plus the orphan-heavy modules (`_embeddings` chrome block,
   `_search-rag`/`_ingest`/`_evaluation_unified`/`_splash` dead clusters), plus
   the 7 unbundled orphan files and the dead `Container.CSS_PATH`. Realistic
   yield ~1.5–3 k lines → **~15–30 ms off the cold parse**, smaller
   steady-state matching, and honest authoring. Verification per token
   (dynamic-name caveat) + visual spot-QA per module. Natural continuation of
   the task-169/task-253 legacy sweeps.
2. **Two-phase deferred bundle (only if first-paint budget still misses after
   tasks 285/257/258).** Keep ONE global file at boot (~7.7 k lines, ~74 ms
   cold) and ONE deferred bundle (~7.3 k lines, vars-preamble prepended,
   manifest order preserved) appended in a single `stylesheet.read()` +
   `reparse()` from an idle callback shortly after first paint. Same total
   work, exactly one extra reparse, off the critical path, no per-screen
   mechanics or per-screen variable risk. Required guards: force-load
   synchronously if the user navigates before the idle load lands; audit the
   handful of equal-specificity utility ties (`.disabled`); full-app visual QA
   once (not per stage). Ceiling: **~70 ms cold first paint**, nothing else.
3. **Do nothing further.** After C1–C3 (imports ~1.2 s) land, a 144 ms parse
   may simply not be the next bottleneck; re-measure before spending the QA.

## 8. AC#3

Not applicable — no split executed. If option 2 is ever picked up, AC#3's
measured first-paint delta + per-screen visual QA applies to that work.
