# ADR-212: Share the adaptive pane shell across destinations (Library and Roleplay)

Status: Accepted 2026-10-02 (owner-approved design spec, PR #2960). Numbered 212 at merge: the plan's provisional 211 was taken by open PR #2918 (`211-ephemeral-provider-failure-presentation.md`), found by the 2026-10-03 all-remotes sweep (`backlog/docs/lessons-backlog-hygiene.md`, "ADR numbers collide across concurrent branches").
Date: 2026-10-02
Task: [TASK-33910.1](../tasks/task-33910.1%20-%20Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md)
Spec: [Roleplay on the Library frame (sub-project B)](../../Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md), sections 2.1, 2.12, 4.3, 5.9 (G1, G1a, G1b, G2, G3, G15, G16) and 5.10
Plan: [B0 implementation plan](../../Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md)
Amends: [ADR-086](086-library-adaptive-reader-shell.md) (the shell is no longer Library-local); [ADR-084](084-library-media-reader-ia.md) (its five-column grips become the shared grip grammar)
Supersedes: N/A

## Decision

The adaptive pane shell — a navigation rail, an items list and a work pane, with a full-height grip after each optional pane — is a **shared destination primitive** for the Library's adaptive readers and for Roleplay. Its widgets live in one module, `tldw_chatbook/Widgets/adaptive_pane_shell.py`; its pure layout policy stays in `tldw_chatbook/Utils/adaptive_reader_state.py`.

### 1. What is shared, and where it lives

| Responsibility | Home | Rule |
|---|---|---|
| Resolver, profile, preferences, effective layout, normaliser | `Utils/adaptive_reader_state.py` | Behaviour byte-identical. `AdaptivePaneProfile`, `AdaptivePanePreferences`, `AdaptivePaneLayout` and `resolve_adaptive_pane_layout` are the SAME objects as the reader names; `nav_open`/`nav_width` are read-only properties, never dataclass fields. A golden digest captured before the aliases existed pins the resolver. The normaliser is unchanged: Roleplay maps its persisted `nav_open` onto `library_open` in its own adapter. The pane ids stay `"library"`/`"items"` for every destination. |
| Shell, grip, messages | `Widgets/adaptive_pane_shell.py` | `AdaptivePaneShell`, `AdaptivePaneGrip` (destination class, `painted_names`, equality-guarded `sync_label`), `AdaptivePaneClasses`, `PaneToggleRequested`, `AdaptivePaneShellResized`, `PaneVisibilityChanged`. |
| Rail-row primitives | same | `DestinationRailRow`, `fit_rail_row_label` (spec section 1.4.3 order, measured in terminal cells; the two-cell prefix counts; the count is never clipped; a loading count never paints its key), `rail_row_content` (literal `Content`, never markup), `DestinationRailRowButton` (patched through `sync_row`, never recomposed). |
| Compact search input | same | `SelectAllOnFocusingClickInput` with its "/" re-arm behind `swallow_slash_on_focus`, default `False`; the Library rail search box keeps `True`. |
| Stage return bar | not yet | `StageReturnBar` joins the module only at the post-#2862 shed, when `library_emergency_return.py` folds into it (paying back one pre-import module). |

### 2. The Library keeps every name

- The Library's messages are the same objects as the shared ones (`LibraryPaneVisibilityChanged is PaneVisibilityChanged`, `AdaptiveReaderShellResized is AdaptivePaneShellResized`). Every Library handler binds with `@on(...)`, which matches class identity; a subclass alias would silently stop matching.
- `LibraryAdaptiveReaderShell` and `LibraryAdaptiveReaderPaneGrip` are thin subclasses that only supply `LIBRARY_ADAPTIVE_READER_CLASSES` and the "Nav" painted name; they define no MRO-dispatched lifecycle handler.
- `Widgets/Library/library_rail.py` re-exports the shared input.
- Stays Library-local: the ordinary-route rail contract and its 3-cell handle; `LibraryRail` and `LibraryShellState`; all `library_screen.py` glue (emergency receipt and restore, the pane-toggle if-ladder, the static F6 tuples, the Escape chain); `[library.reader]` and its Settings rows. Copy the contracts, not the code.

### 3. CSS: destination classes, never neutral rules

- Every shell, grip and row rule is keyed to a destination's own `AdaptivePaneClasses` and lives in that destination's lazy split sheet. The build moves a rule to a lazy sheet only when every class token carries the sheet's owner prefix, so a neutral token would pin the rule to the boot bundle (50 B of headroom on 2026-10-02).
- The Library's destination classes are its existing class names, so the Library block in `css/features/_library.tcss` is unchanged and boot bytes did not move. Its one boot-resident member, the 214 B `:hover, .-active` grouped grip rule, is in the bundle because `.-active` has no split owner; it stays. Roleplay's copy (B1) must not add another `.-active` boot rule without an equal boot deletion in the same PR.
- No width rules: widths stay inline from the resolver (`# ds-runtime:`). The shared widgets declare no `DEFAULT_CSS`, `CSS` or `BUNDLED_CSS`.
- The `panes` pattern family is a widget contract with no public classes (`css/patterns.json`, `backlog/docs/component-patterns.md`, `Widgets/pattern_gallery.py`).

### 4. Shared focus contracts

- `BaseAppScreen.TAB_REGION` (default `None`) with `region_focus_next`/`region_focus_previous` and an `arrival_focus_target()` hook (default: the first focusable widget in `#screen-content`). `None` keeps Textual's stock Tab walk on every route; a destination opts in by setting a selector. Library and Console keep their own Tab bindings; Library adoption is FU-1 (TASK-33910.19).
- Grip evacuation: when an open pane closes while it holds focus, focus moves to that pane's grip (`AdaptivePaneShell.sync_layout`).
- Dynamic F6 arrives with the Roleplay frame (B7, TASK-33910.12).

### 5. One collapse grammar, one round border per pane

On adaptive-shell routes a collapsed pane is a full-height grip and each pane has one round border (owner ruling Q1). Recorded exceptions until the Library-adoption follow-up (FU-1): the Library's ordinary routes (a text "Collapse" button, a 3-cell handle with unpersisted state — `LibraryNavigationRailHandle` in `Widgets/Library/library_rail.py` — and three solid border layers in `css/features/_library.tcss`) and the Library's outer frame.

### 6. Boundaries

- Roleplay never imports `Widgets/Library/`: that package's `__init__` eagerly imports every Library widget, and the Library's pre-import route is 176 modules / 125k LOC.
- The shared module imports nothing from `Widgets/Library/`, `Library/` or `UI/Library_Modules/`, contains no Library class literal, and is never imported by a UI-ready-resident module (never `Widgets/destination_rail.py`). Tests pin all three.
- Not authorised: Watchlists convergence, an app-wide workbench, a third shell pane.

### 7. The census rule

Each slice measures the UI-ready census, the pre-import payload and the boot CSS bytes on its paired base arm and its head in the same session, and answers any growth in ADR-097's order: defer, then shed, then an owner-signed exception row in the same commit as the constant change. No blanket re-pin. Boot CSS is net zero or lower in every PR. After each re-key of a counted broad selector, the broad-selector ratchet is lowered to the measured count. B0's own row: +1 pre-import module (`Widgets.adaptive_pane_shell`), ledgered in ADR-097.

### 8. Slice order and the fold-before-rail arithmetic

B0 and B1 may proceed in parallel, except B1's shell-rule copy and parity test, which need B0's Library destination classes. The Inspector folds into the work pane (B5b-2) before the Roleplay rail arrives (B6): adding a 28/35-column rail while the Inspector stays would leave the work pane at 34 columns at 120 (below its 40-column minimum) and 60 at 160; with the fold first the rail costs nothing (interim floors ≥58 / ≥78 / ≥108).

## Exceptions to the design language

### Grips are not slivers (G1a)

`backlog/docs/design-language.md` §2.8 says a shell sidebar collapses "to zero width (never a sliver)". On adaptive-shell routes a collapsed pane keeps a full-height **five-cell grip** at every width. The grip is a labelled, reachable control — it paints the pane's name and its arrow, takes focus, and answers Enter and Space — not a sliver of the pane. Authority: ADR-084's accepted grips ("retains a five-column grip for each pane", ADR-084 lines 15-16; "Both collapsed panes continue to consume five columns each for reachable grips", lines 73-74) and the owner's 2026-09-09 change (PR #2550, `7fc19997e3`), which amended ADR-086 to keep "the intentionally five-cell controls" while widening the columns. Not ADR-148 (Proposed). Below 64 columns a zero-width form arrives only with the single-stage follow-up (spec R38); until then the grips stay at every width.

### Adaptive rails follow the Library policy (G1b)

§2.8 sets `$ds-sidebar-min-width: 35`. An adaptive-shell rail instead follows the resolver's rail policy: `clamp((3W + 8) // 16 + 5, 29, 39)` cells with a 24-cell floor (`Utils/library_rail_width.py`, ADR-086). Roleplay **deliberately shares** the Library's policy — one house rail width, and "Nav" means the same in both destinations. This is a recorded coupling: B7's golden test pins Roleplay's resolved table, so a Library rail-policy change fails a Roleplay test and is reviewed for both destinations.

### Runtime geometry lives in the Python leaf (G15)

§2.8 promotes shared feature geometry to `_variables.tcss`. Geometry a resolver computes from the measured terminal (rail, list and work widths, grip reservations) promotes instead to the config-safe Python leaf `Utils/adaptive_reader_state.py`: a stylesheet token cannot depend on the measured shell, and the widths reach widgets as inline styles marked `# ds-runtime:`.

## Context

ADR-084 kept the first adaptive reader inside Media; ADR-086 shared it across five Library destinations but kept it Library-local and declined an application-wide implementation. The Roleplay redesign (spec, 2026-10-02) adopts the same frame. Copying the shell would duplicate the geometry, focus evacuation and persistence behaviour ADR-086 refused to clone; importing `Widgets/Library/` would pull the Library package onto Roleplay's route. Budgets at the base (`origin/dev @ fccf70d3b0`): boot CSS 608,040 / 608,090 B; broad selectors 273 / 274; UI-ready 1,033 / 1,033 modules; pre-import 557 / 557 modules.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Copy the shell into Roleplay | Two implementations of hard geometry and focus code, ADR-086's own reason to share; they would drift. |
| Roleplay imports `Widgets/Library/` | Pulls the Library package (eager `__init__`) onto Roleplay's route. |
| Put the primitives in `Widgets/destination_rail.py` | UI-ready resident: the module cost lands on every boot, with zero census headroom. |
| A new `Utils` module for the aliases | One more module in both the boot and UI-ready censuses. |
| A neutral boot CSS component for the shell (~1.1 KB) | 50 B of boot headroom; fails the byte budget (spec fresh-lens MF-01). |
| Subclass message aliases | `@on` matches class identity; a subclass would stop matching the shared shell's posts. |
| Gate Tab with `check_action` and fall through to Screen's binding | The binding merge replaces Screen's binding, so a `False` gate kills Tab. |
| Canonical neutral classes for the `panes` family | A neutral class in a rule pins it to boot; without a rule it is nominal; the gallery renders boot CSS only. |

## Consequences

- The Library's behaviour, classes, CSS bytes and Tab order are unchanged; its compatibility names become thin subclasses and aliases.
- Roleplay (B1 onward) builds its frame from the shared module with its own destination classes, its own lazy sheet (`screen_feature_roleplay.tcss`), and `TAB_REGION` (B3).
- The pre-import census carries +1 module until the post-#2862 shed pays it back.
- Rail-row fitting decisions the spec left open are fixed here: the prefix counts toward the width; the key hint is two spaces plus the letter (right alignment is the destination's styling); counts are caller-formatted; the ellipsis is `…`, measured in cells after `resolve_glyph` (Roleplay frame B1 added `…` → `...` to the glyph map, so ASCII mode paints `...`); a style-only row change (a count that starts or stops loading with unchanged text) repaints, because `Content` equality and the `label` reactive compare plain text only.
- `LibraryBrowseReaderShell.apply_route` still relabels a grip by assigning `pane_label` and calling `sync_open` with an unchanged open state, which does not repaint the painted name; `AdaptivePaneGrip.sync_label` is the fix, adopted by the Library in FU-1 (TASK-33910.19).

## Links

- [ADR-084: Library Media reader information architecture](084-library-media-reader-ia.md)
- [ADR-086: Share an adaptive reader shell within Library destinations](086-library-adaptive-reader-shell.md)
- [ADR-097: Boot budget ratchets](097-boot-budget-ratchets.md)
- [Spec: Roleplay on the Library frame (sub-project B)](../../Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md)
- [Design language](../docs/design-language.md) §2.8
- [Component patterns: Panes](../docs/component-patterns.md)
