# Component-Pattern Library — Design

Date: 2026-09-13
Status: approved in brainstorming (sections 1–3, with review corrections); pending implementation plan
Extends: ADR-150 (design-token system and design-language constitution); amends ADR-150 §6's opportunistic migration policy (see §3.10)
ADR required: yes — `backlog/decisions/161-component-pattern-library.md` (next free number; the decisions directory runs to 160 and carries three colliding 150s, so the number was verified, not assumed)
Absorbs: TASK-24451 (split the agentic terminal CSS grab-bag) — to be marked superseded-by the parent task filed for this project

## 1. Problem

ADR-150 governs *values* (`$ds-*` tokens) and new-sheet hygiene, and it works: feature
sheets are nearly hex-free (3 stray literals) and the token governance test is strict.
What is not consistent is the *component vocabulary* the app is actually built from:

- **Four competing label conventions** for the same thing (`.form-label` ×162,
  `.settings-input-label` ×164, `.setting-label` ×37, `.field-label`), and
  `.section-title` (87 uses) has four separate definitions; `.form-input` is defined
  twice within its own sheet.
- **The most-used classes live in wrong homes**: the modal vocabulary
  (`.dialog-title`/`.dialog-buttons`) is defined in `features/_media.tcss`;
  `.action-button` (75 uses) in a feature sheet; the `settings-*`, `console-*`,
  `library-*` vocabularies and a genuine `ds-*` primitive layer (`.ds-toolbar` ×111,
  `.ds-field-row` ×69, `.ds-panel` ×17) are buried in
  `components/_agentic_terminal.tcss` — a 12,240-line monolith holding ~80% of all
  component CSS.
- **No catalog**: nothing documents what the canonical form/table/dialog/status
  pattern is; humans and agents discover vocabulary by grepping, so each new screen
  picks a different convention and re-fragmentation continues.
- **No class-level governance**: the token test cannot notice a feature sheet
  defining `.section-title` a fifth time.
- **Dead weight**: `_unified_sidebar.tcss` (474 lines, not in the build manifest),
  `Widgets/base_components.py` (0 importers), six zero-importer widget modules.

An inventory established that the CSS class vocabulary — not the Python builders — is
the de-facto component library: screens hand-compose `classes="..."` strings (~170
screens; 375 `Button` instantiations), while two Python builder libraries
(`form_components.py`, `base_components.py`) are effectively dead (2 and 0 importers).

## 2. Goals

1. A documented, mechanically-enforced catalog of canonical component patterns that
   humans and agents compose from — the layer that turns ADR-150's tokens into a
   design *system*.
2. One winning vocabulary per family, actively consolidated (top offenders renamed
   everywhere), with governance so fragmentation cannot regrow.
3. The `_agentic_terminal.tcss` monolith carved up (absorbing TASK-24451), with every
   promoted class landing in a canonical home.
4. A live pattern gallery that makes "consistent look" verifiable: snapshot tests fail
   when any canonical pattern's rendering changes.
5. Every legacy literal retires — not opportunistically, but as committed,
   ratcheted, project-owned work: the ~6,250 raw dimension declarations across the
   source sheets and the ~580 Python-side ad-hoc `styles.*` visual assignments all
   migrate onto the token system. End state: every color and dimension value in every
   stylesheet traces to a `$ds-*` token (shared or feature-scoped), and Python sets
   visual values only through token-backed classes. This amends ADR-150 §6's
   "migrate opportunistically" policy; ADR-161 records the amendment.

### Non-goals (explicitly out of scope)

- A single big-bang migration pass. The literal migration is in scope (goal 5) but
  ships as ratcheted per-sheet PRs following the `features/_chat.tcss` exemplar —
  ADR-150 rejected the unreviewable one-pass rewrite, and this project keeps that
  shape while committing to its completion.
- Redesigning the visual identity (palette/density changes at the token layer).
- Theme-system extensions and composite (multi-value) tokens — separate futures.
- Reviving Python builder APIs (rejected below).

## 3. Decisions

### 3.1 Form factor: CSS-class catalog + governance

Patterns are documented CSS classes plus structure and state contracts; screens keep
hand-composing class strings exactly as all ~170 do today. Python-side idioms that
already won (modal `SafeModalDismissMixin` — 58 of 66 modals; `ConfirmationDialog`;
`create_status_area`) are documented as part of their pattern rather than replaced.

Rejected alternatives: a typed Python builder API (contradicts 170 screens of
practice; both prior builder libraries died of disuse; every screen would need
rewriting to benefit — the "full sweep" shape ADR-150 rejected); a hybrid (two
expression paths for the same patterns, both must be learned and kept in sync).

### 3.2 The catalog

`backlog/docs/component-patterns.md`, referenced from the constitution
(`backlog/docs/design-language.md` §4) and from AGENTS.md. The constitution stays the
laws; the catalog is the reference. Each entry has a fixed shape:

- **Purpose** and **when-not-to-use** (one paragraph each)
- **Structure**: a minimal `compose()` snippet showing the container/class skeleton
- **Class inventory**: each class, its role, required rest/hover/focus/disabled states
  (state contract per constitution §2.7)
- **Tokens consumed** (`$ds-*` links; never literals)
- **Python idiom** where one exists
- **Lifecycle status**: `Canonical`, `Deprecated` (with rename target + ratchet count),
  or `Draft`

**v1 families (9):**

| Family | Winner codified |
|---|---|
| Forms & fields | `.form-*`; `settings-*` field classes fold in as the compact variant *or* stay as documented variants — decided by measured equivalence (§3.6) |
| Buttons & action rows | `Button` type contract + `.button-group-*`; `.action-button` promoted out of `features/_evaluation_unified.tcss` |
| Lists & tables | `_lists.tcss` cursor/hover/selected contract for `ListView`/`DataTable`/`OptionList` |
| Dialogs & modals | `.dialog-title`/`.dialog-buttons` promoted to `_dialogs.tcss`; `SafeModalDismissMixin` + `ConfirmationDialog` idiom |
| Status, empty & loading states | `.status-area`/`.status-label`/`.status-item` promoted into the (currently empty) `_status.tcss`, mapped to `$ds-status-*`; empty/loading documented as composition recipes |
| Navigation & sidebars | `.sidebar-*` + `$ds-sidebar-*` layout laws |
| Sections & containers | one `.section-title`/`.section-header` definition (today: four) |
| Chat & message rendering | transcript structure, `$ds-chat-*` speaker accents, tool-call/result messages |
| ds-primitives | `.ds-panel`, `.ds-toolbar`, `.ds-field-row`, `.ds-info-callout`, `.ds-approval-card` |

### 3.3 The registry

`tldw_chatbook/css/patterns.json` — the machine-readable twin of the catalog and the
single source governance reads: per family, the owning sheet and per class its
lifecycle status (deprecated classes carry `renames_to` + usage count). Three
consumers, one artifact: the doc sync check, the gallery sync check, and the
governance tests.

### 3.4 The pattern gallery

A dev-facing screen rendering every canonical pattern, reachable via the command
palette only (no keybinding — ADR-031 forbids terminal-convention keys; footer hints
advertise only implemented actions; not in the tab rotation). Its own sheet is a small
in-bundle features sheet, token-clean — the gallery dogfoods the catalog.

Snapshot tests export SVG at a fixed terminal size under two themes: the default dark
theme (`textual-dark`, the config default) **and one light theme** — pinned by name in
the implementation plan from `Themes/themes.py` (polarity coverage is mandatory:
task-31264 documented dark-tuned values silently freezing light themes). Snapshots are stable because
Textual is exact-pinned (`8.2.8`, TASK-1353); dependency bumps are already deliberate
review events. A documented `UPDATE_SNAPSHOTS=1` path regenerates fixtures. The
gallery cannot rot: a sync test fails CI if a `Canonical` class is missing from the
gallery. Known limitation: SVG export does not render the toast rack in this Textual
version, so no pattern verification depends on screenshotting notifications.

### 3.5 Canonical homes & the carve-up

Owning sheets (bundle = `CSS_MODULES` in `build_css.py`; order matters — atomic layer
first so later sheets can override at equal specificity):

```
components/_ds_primitives.tcss  NEW — ds-* primitives out of the monolith
components/_buttons.tcss        + .action-button; .button-group-* moves here
components/_forms.tcss          field vocabulary incl. documented compact variants
components/_lists.tcss          unchanged contract, documented
components/_dialogs.tcss        + .dialog-title/.dialog-buttons from features/_media.tcss
components/_status.tcss         FILLED — status classes promoted
components/_sections.tcss       NEW — the single section-title/header definition
components/_messages.tcss       + shared chat/message grammar
layout/_destination.tcss        NEW — .destination-section (308 uses) consolidated
features/_console.tcss          console-* vocabulary (already bundle-resident today)
features/_library.tcss          library-* vocabulary
```

Carve-up rule: a class promotes to the sheet matching its **consumption scope** —
2+ screens → components/layout; one feature → that feature's sheet. What remains of
the monolith is true Console chrome.

**Two-track rule.** The repo deliberately keeps per-screen stylesheet sources
separate from the bundle (`screen_agentic_*.tcss`, built from `BUNDLED_SCREEN_CSS`)
because Textual accumulates `$variable` definitions per source and concatenating the
screen blocks once let local `$ds-*` fallbacks redefine real tokens app-wide
(TASK-15993/16811; measured: `$ds-focus-bg` silently changed value across the app).
Therefore: **moves stay within a track**. Bundle-internal moves (e.g. console-* out
of the monolith into `features/_console.tcss`) change only intra-bundle cascade
order; nothing crosses between bundle and per-screen sources; new per-screen blocks
carry no local `$ds-*` fallbacks (existing ones get a ratcheted grandfather manifest).

**BUNDLED_CSS rule.** Catalog classes are defined only in `CSS_MODULES` sheets —
never in `BUNDLED_CSS`/`BUNDLED_SCREEN_CSS` blocks. Existing block definitions of
catalog classes (e.g. `ConfirmationDialog`'s `.dialog-title`) migrate out during
consolidation, using the existing `widget_defaults_*` consolidation machinery and its
tie-aware reparse workaround — not a new demotion policy.

**Size ceiling.** ≤2,000 comment-stripped lines per `CSS_MODULES` sheet (data: the
monolith is 12,240; next-largest is 1,755 — the ceiling touches only the monolith,
and counting active declarations means load-bearing comment essays are not punished).
The ceiling test ships in the same PR that completes the split — no temporary
allowance that can outlive the project.

**Tokenization-on-move, precisely bounded.** New sheets fall under the post-ADR-150
ratchet: no raw hex, no raw numeric `padding`/`margin` (width/height literals remain
legal until §3.10 reaches them). So each moved rule converts following the
`features/_chat.tcss` exemplar (TASK-32480) extended to the project's end state:
recurring values map 1:1 to shared `$ds-*` tokens; genuine feature-specific geometry
becomes a feature-scoped `$ds-<feature>-*` token per §3.10 — moved rules arrive
fully on-system, not partially. The monolith's single grandfathered hex does not travel — it is
tokenized. This forced tokenization of ~12K lines is the project's real cost and its
point: nothing re-homes without going on-system. Sheets whose rules do not move are
migrated in place by §3.10.

### 3.10 Legacy-literal migration (in place)

Every remaining legacy sheet — the ~27 feature sheets and anything else that kept its
home — goes through the `features/_chat.tcss` exemplar pass (TASK-32480 is the
canonical end-to-end example), extended to the stricter end state the project
commits to:

- **Recurring values map 1:1 to shared tokens** (`$ds-space-*`, `$ds-control-height`,
  …). A value that appears in two sheets is by definition recurring and must be a
  shared token.
- **Genuinely feature-specific geometry becomes a feature-scoped token** in
  `_variables.tcss` (`$ds-<feature>-<name>`, following the existing
  `$ds-home-followup-row-height` / `$ds-library-source-browser-width` pattern) — not a
  bare literal in the sheet. Feature-scoped tokens promote to layout laws when a
  second consumer appears (existing constitution rule). Sheets reach **zero raw
  numeric dimension and color literals**; the token file is where every value lives.
- **Comments stay load-bearing**: each migrated rule keeps the TASK/defect rationale
  comments the exemplar pattern requires.
- **Python side**: ad-hoc `styles.*` assignments for token-covered values are
  replaced by token-backed CSS classes (the constitution's existing rule for new
  code, now applied to legacy), not by duplicated Python constants.

Ordering: moves and renames first (§3.5, §3.6) — they touch the same lines, and
migrating literals first would double-handle every rule — then one migration PR per
sheet.

Ratchets land with the first migration PR, pinned at then-current counts, and may
only decrease (mirroring the hex ratchet): per-sheet raw numeric
`padding`/`margin`/`width`/`height` declaration counts, and per-file Python
`.styles.<visual-property> = <literal>` assignment counts. The ratchet floor is zero
on both.

### 3.6 Consolidation pipeline (top offenders)

For each competing convention, one PR:

1. **Measure, then decide.** Parse the competitors' declarations; compute
   style-equivalence; winners chosen by usage weight + family fit. Genuine
   differences become documented variants (e.g. standard/compact fields), not forced
   merges. **Equivalence is geometry-aware**: declarations that look equivalent
   "minus a border" are not equivalent in a height-1 row (TASK-13154.1 — a field
   without `.settings-compact-input`'s border suppression spends its only row on the
   border and never paints its text). Equivalence proofs include rendered geometry on
   height-constrained fixtures.
2. **Scripted rename** across Python `classes="..."` sites (~500–800 total),
   `.tcss` compound selectors, and `BUNDLED_CSS` strings — word-boundary-safe.
3. **CSS-side**: delete losing definitions, promote the winner to its owning sheet.
4. **Verify** (targeted; full sweeps only on explicit opt-in per AGENTS.md):
   governance green → zero-remaining grep (`tldw_chatbook/`; `Tests/` rename
   atomically in the same PR, no grandfathering) → existing UI tests for the touched
   files (the rename script knows the file list) → gallery snapshots (dark + light) →
   SVG A/B on the highest-usage screens for the renamed convention (the sample is the
   top screens by occurrence count of the old name, minimum three).

Where today's duplicate definitions *disagree*, screens that were receiving a losing
definition visibly change to the winner. That is the intended consistency, and it is
a visual change: those screens get live verification per
`backlog/docs/lessons-live-verification.md` (real app via the tmux recipe: scratch
`TLDW_CONFIG_PATH`, stderr left attached to the pane, absolute venv python path).
Terminal captures are hints for where to look; the compositor (`render_strips()`) is
the authority when a diff needs investigating.

### 3.7 Governance tests

New `Tests/UI/test_component_pattern_governance.py`, sibling to the token test,
reading `css/patterns.json` and importing `CSS_MODULES` from `build_css.py` (one
source of truth for bundle membership):

1. **Single-definition rule** — parsing selectors, not grepping names: a rule whose
   selector is exactly a `Canonical` class (or the class plus state
   pseudo-selectors — states belong to the owner, per the §2.7 contract) may exist
   only in its owning sheet. Compound/scoped selectors using the class
   (`.compact .form-label`) are legal anywhere — that is composition, and
   `utilities/_overrides.tcss` remains the home for global overrides.
2. **Deprecated-name ratchet** — counts in `tldw_chatbook/` may only decrease.
3. **Sync checks** — every `Canonical` class appears in the catalog doc and in the
   gallery.
4. **Sheet size ceiling** (ships with the split's completion PR).
5. **Fallback-ban ratchet** — existing per-screen blocks carrying local `$ds-*`
   fallbacks are a grandfathered manifest; new blocks may not add them.
6. **Dimension-literal ratchet** — per-sheet counts of raw numeric
   `padding`/`margin`/`width`/`height` declarations (comment-stripped, same regex
   family the token test uses), pinned at migration-start values; may only decrease;
   floor zero.
7. **Python ad-hoc style ratchet** — per-file counts of
   `.styles.<visual-property> = <literal>` assignments, pinned; may only decrease;
   floor zero.

### 3.8 Dead code

Deleted, after grepping for dynamic references (`importlib`, string-built `getattr`):
`components/_unified_sidebar.tcss` (orphaned), `Widgets/base_components.py`, and the
six zero-importer widgets (`empty_state`, `loading_states`, `toast_notification`,
`custom_list_items`, `enhanced_sidebar`, `detailed_progress`). Empty/loading states
are documented as composition recipes; toasts are Textual's built-in `notify()`.

### 3.9 Sequencing with the byte ratchet

The bundle is already over its 806 KB ratchet (TASK-31500: 821,753 B). Two effects
must be sequenced:

- **Dedupe is net-negative** (four `.section-title`s → one; per-widget dialog copies
  → one; double `.form-input` → one; orphaned sheet deleted) and lands **before**
  the gallery sheet lands, so the gallery spends bytes the dedupe freed.
- **Tokenization is net-positive**: `$ds-space-section` is longer than `2`, across
  thousands of declarations plus the new feature-scoped token definitions. The
  migration's byte growth is bounded but real (order: tens of KB), so the
  in-place migration PRs (§3.10) coordinate with TASK-31500's pay-down: either the
  pay-down lands first and creates headroom, or migration PRs land alongside
  pay-down chunks so every ratchet-evaluated commit stays under the limit. The
  implementation plan makes this ordering explicit per PR.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Intra-bundle cascade-order flips when rules move (equal-specificity ties resolve by order) | Move prefixed, disjoint vocabularies in small batches; sampled SVG A/B per batch; compositor as investigation authority |
| Visual changes where dedupe picks a winner among disagreeing definitions | Intended change; affected screens live-verified per the lessons doc |
| Forced tokenization of ~12K moved lines is slow and error-prone | `_chat.tcss` exemplar pattern; per-family PRs; snapshot + pilot verification per batch |
| Catalog/registry/gallery drift | One registry artifact + sync tests fail CI on drift |
| Snapshot flake | Exact-pinned Textual; fixed size; two pinned themes; explicit regen path |
| Project stalls mid-carve-up | Size-ceiling test ships only with completion; governance lands early and independently protects what has shipped |
| Literal migration (~6,250 sheet declarations + ~580 Python sites, ~30 sheets) is the largest workstream | Ratchets pin counts on day one so progress is one-way; per-sheet exemplar-pattern PRs; snapshot + pilot verification per sheet; completion is measurable (ratchet floor zero) |
| Tokenization grows the bundle against an already-breached byte ratchet | Sequencing rule in §3.9 coordinates with TASK-31500's pay-down; per-PR byte accounting in the implementation plan |

## 5. Task structure

One parent backlog task (component-pattern library) with subtasks, each a single PR:
catalog + registry; governance tests; gallery + snapshots; per-family rename
consolidations; monolith carve-up (supersedes TASK-24451, which is closed when the
split lands); dead-code deletions; per-sheet in-place literal-migration PRs plus the
Python `styles.*` cleanup (§3.10); constitution update (`design-language.md` §6
shrinks to reflect the amended migration policy and the zero-literal end state);
ADR-161 + constitution/AGENTS.md linkage. ADR-161
is created before implementation begins and records: the CSS-catalog-over-builders
decision (with the dead-builder evidence), the two-track move rule, the
registry/governance model, and the amendment of ADR-150 §6's opportunistic migration
policy to a committed, ratcheted, project-owned migration with a zero-literal end
state.

## 6. Testing strategy

- Governance tests (§3.7) run in the normal suite; they are the system's teeth.
- Gallery snapshot tests (SVG, fixed size, dark + light) cover every canonical
  pattern's rendering.
- Rename and migration PRs run the touched files' existing UI tests plus targeted
  new assertions; full sweeps only on explicit request.
- Live tmux verification for intentional visual changes (§3.6).
- The repo's evidence rules apply throughout: `backlog/docs/lessons-testing-evidence.md`
  (what counts as evidence) and `lessons-live-verification.md` (the real-app recipe).
