# Persona Buddy Pet-Only Chrome Design

**Status:** Approved

**Date:** 2026-08-22

**Task:** [TASK-21000](../../../backlog/tasks/task-21000%20-%20Distill-Persona-Buddy-to-pet-only-chrome.md)

**Decision:** [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)

## ADR Check

ADR required: no

ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

Reason: ADR-074 already fixes the local Persona Visual runtime, app-owned Buddy controller, native Textual view, bounded persistence, and runtime authority. This change removes disposable view chrome without changing those ownership or storage contracts.

## Problem

Actual 120x40 application UAT proved that the complete Persona Buddy frame now renders, but the resting widget still reads as a control panel rather than a pet. A three-row `Drag / Fold / Close` header, border title, state label, keyboard hint, one-cell horizontal padding, and a frame region larger than the prepared artwork leave the portrait tiny, dark, and surrounded by dead space. The interface repeats operational state in words even when the user has nothing to address.

The approved direction is ruthless progressive disclosure: the normal widget is the pet. Chrome becomes icon overlays, words appear only for actionable alerts, and the displayed bounds follow the prepared frame rather than forcing the frame to occupy an unrelated rectangle.

## Goals

- Make the resting Buddy visually consist of the pet, not a labelled window.
- Preserve direct Fold/Open, Close, drag, resize, keyboard, focus, and persistence behavior without permanent copy.
- Remove unused horizontal and vertical space without cropping or distorting the artwork.
- Keep alert text explicit when user action is actually required.
- Preserve the existing controller, resolver, renderer, storage, and authority boundaries.

## Non-Goals

- No Persona Visual manifest, renderer, asset, database, preference-schema, state-priority, or provider change.
- No new icon package, image dependency, custom tooltip system, hover-only control, or separate pet window manager.
- No redesign of the underlying pet artwork in this task.
- No raw provider, tool, prompt, assistant, path, or diagnostic text in the Buddy.

## Approved Interaction Contract

### Resting and non-actionable states

The normal widget composes one pet frame plus two overlaid native Textual buttons. It does not compose or paint the current border title, header, drag label, status row, movement hint, or default `Visual pending` copy. Widget and frame padding are zero. The one-cell focus/boundary treatment may remain because it carries focus and hit-region ownership; it must not add an empty inner gutter.

The pet itself communicates `idle`, `thinking`, `speaking`, `listening`, `tool_running`, `wake_armed`, explicit, authored, and custom states through the existing visual resolution and animation contracts. None of those states paints words by default.

### Controls and manipulation

The selected control model is the two-corner overlay:

- `▾` means Fold in the normal state and `▴` means Open in the folded state.
- `×` means Close in both states.
- Both controls use one-glyph labels centered inside compact native Textual buttons at least three cells wide, because Textual's mandatory line padding clips a one-cell button. They retain exact `Fold`/`Open`/`Close` tooltips and the incumbent `c`/`x` keyboard actions.
- Mouse hover uses the native tooltip. In the 10x4-or-larger pet box, keyboard focus uses the repository's non-obscuring button focus treatment—accent background plus bold underline, not a one-row heavy outline—and temporarily exposes the exact `Fold`/`Open`/`Close` label inside the pet. That focus help is interaction feedback, not resting chrome. Below the 10x4 minimum, the two-button compact fallback retains its one-glyph labels under focus because expanded words cannot coexist with both controls and the resize grip; the same focus treatment, exact tooltip, and keyboard action remain available.
- The remaining pet surface is the drag target. Existing lower-right resize authority and `HJKL` size bindings remain available; controls never arm drag or resize.

The controls occupy cells inside the portrait bounds and add no layout row or column. They remain visible at rest; hover-only disclosure is rejected because terminal hover support and keyboard discoverability are not reliable enough.

The operable normal/folded content box has an explicit minimum of 10 columns by 4 rows. That is the smallest box that can host two three-cell native buttons, a transient five-character focus label, and the longest fixed alert wrapped over two rows without collision. Ordinary assets larger than that minimum introduce no decorative gutter. A valid undersized asset is centered in this control-owned minimum; its unavoidable alignment space is accessibility structure, not padding or chrome.

### Actionable alerts

Only `approval_needed`, `error`, and `offline` are actionable word-bearing states. While one is current, the pet region is replaced—not covered or enlarged—by one fixed, path-free label:

- `Approval needed`
- `Error`
- `Offline`

The two corner controls remain available. The alert uses text plus semantic color, so color is never the sole carrier. Alert copy wraps within the same stable content box, using at most two centered rows in the 10x4 minimum; entering an alert does not resize an already accepted widget. When the actionable state clears, the latest current-generation pet frame returns. Raw provider output, tool content, assistant text, prompts, paths, exception details, and arbitrary state strings never enter the label.

### Folded and constrained states

Folded mode is a real reduced pet thumbnail, not a text strip. It retains `▴` and `×` in opposite corners and remains draggable outside those controls. The thumbnail is prepared through the same bounded renderer and retains the same semantic state, Persona Visual graph, and cache provenance as the full pet. It uses a distinct render authority containing folded mode and the fixed folded `(cols, lines)` budget, plus every incumbent controller, preference, profile, selection, viewport, and view-generation fence. Folded and full preparations are never treated as interchangeable. Animation remains frozen consistently with existing collapsed behavior.

If the effective render area—after both preferred-geometry and viewport constraints—cannot fit the 10x4 thumbnail/control minimum, the existing compact/minimal safety path may degrade to the two labelled-by-tooltip compact icon buttons. Those fallback buttons remain glyph-only under focus so neither control nor the lower-right resize grip can be obscured. That is the only normal case in which the pet disappears without an actionable alert.

## Fit-to-Frame Geometry

Saved `PersonaBuddyGeometry.width` and `.height` remain the user's preferred render budget and resize input. They do not force empty painted bounds. Resolution uses a deterministic content budget derived from the current preferred geometry, boundary cost, viewport, and folded mode—not from the last auto-fitted widget region—so fitting cannot create a shrink feedback loop.

After a current-generation visual is accepted, the widget computes one stable animation box from the greater of (a) the 10x4 operable minimum and (b) the maximum prepared cell width and maximum `ceil(pixel_height / 2)` across every frame in that visual. The outer widget fits that stable box plus only its one-cell boundary and the overlaid controls. Every animation frame uses the widget's existing center/middle alignment inside that box. An ordinary single-frame visual at or above the minimum therefore touches every inner content edge; an undersized asset or visual whose source frames genuinely vary in dimensions may show only the unavoidable control/alignment space inside the stable animation box. The widget never reflows per frame, so its controls, clamp position, drag target, and resize corner do not jitter during animation.

The image is never stretched to fill the preferred budget and never cropped merely to preserve a saved rectangle. Auto-fitted width and height are display-only and are not written back to preferences; saved `x/y` remain clamped against the actual displayed bounds.

The widget owns one immutable accepted-render record containing the exact requested resolution authority, semantic/cache/frame identity, and fitted animation box. It replaces that record only with the direct result of `resolve_current_visual()` after every existing post-await fence confirms the active screen, view generation, controller generation, preferences generation, profile generation, selection, requested state, mode, render budget, and viewport are still exact. Ordinary snapshot polling may retain the accepted frame and bounds while a new resolve is pending, but it may not refit from `snapshot.visual` because that visual does not carry its originating size ticket. If the preferred geometry or viewport makes the effective render area smaller than the 10x4 minimum, or the viewport becomes too small for the accepted box before a current replacement arrives, the constrained two-button fallback paints instead of an out-of-date frame. A prior-budget, prior-viewport, or stale-generation result may neither resize nor repaint the current view.

## Component and Data Flow

1. The app-owned controller publishes the current snapshot and resolved visual exactly as today.
2. `PersonaBuddyWidget` derives the preferred normal or fixed folded render budget without reading its previous fitted content size.
3. The controller resolves and prepares bounded frames off the event loop.
4. A direct current-authority result supplies immutable frame dimensions and renderable cells; the widget records its exact resolution authority and stable maximum animation box as the accepted render.
5. The widget fits its display bounds once per accepted visual generation, paints either the pet or a fixed actionable alert, and overlays the two compact native icon buttons.
6. Drag/resize/collapse/close mutations continue through the existing app-owned serialized preference boundary; auto-fit dimensions never enter that write.

No new module or abstraction is required. The shortest correct implementation removes the existing text widgets/header layout, reuses the frame `Static`, uses the existing buttons with icon labels/tooltips, and narrows the current geometry/render-authority methods.

## Accessibility and Failure Behavior

- Icons use the incumbent background-plus-bold-underline focus treatment, exact mouse tooltips, transient focus-visible labels in the 10x4-or-larger pet box, and existing keyboard equivalents; below that minimum the two-button fallback stays glyph-only under focus so both controls and resize remain operable. There is no hover-only action and no one-row outline that can obscure a glyph.
- Actionable alerts use stable text and semantic color.
- Decode, resolver, missing-asset, and stale-authority behavior remains fail-soft and path-free.
- If no current prepared frame exists, the view retains the last current frame or existing fallback behavior; it does not paint speculative prose.
- An unavailable Persona or binding continues through the existing unmount/recovery authority rather than inventing a word-bearing pet state.

## Verification

Focused tests must begin RED and prove:

- the normal mounted widget contains no default words, an ordinary single-frame visual touches every inner content edge, a one-pixel control fixture occupies only the 10x4 operable minimum, and varying frame dimensions retain one non-jittering maximum animation box;
- only `▾`/`▴` and `×` remain as visible glyphs inside operable compact buttons, with tooltips, non-obscuring focus paint, transient focus-visible labels only when the 10x4 pet box fits, hit regions, keyboard behavior, drag exclusion, and resize preservation;
- non-actionable states remain wordless while exact actionable states replace the portrait with fixed labels and restore it afterward;
- folded mode paints a bounded real thumbnail under a distinct folded size authority and only an effective preferred/viewport area below 10x4 reaches the two-button fallback;
- preferred geometry drives render budget while the accepted visual's maximum prepared dimensions drive one stable display box, with no persistence, per-frame jitter, or feedback loop;
- a prior-budget or prior-viewport `snapshot.visual` cannot be accepted or refitted while a current resolve is pending;
- stale resolution, navigation, cancellation, animation, reduced motion, fallback, unavailable recovery, and invalidation contracts remain green.

Mutation proof must remove or weaken the default-text deletion, frame-fit identity, alert allow-list, folded thumbnail render, and control hit fences and show each focused test fail. Verification also includes scoped Ruff/format/compile/diff checks, CSS bundle synchronization, architecture/privacy/governance gates, one Impeccable review, and isolated real-terminal screenshots of normal, alert, folded, and constrained states without `NO_COLOR`.

## Rejected Alternatives

- **Always keep the existing 28x12 panel and stretch the image:** rejected because it distorts artwork.
- **Crop or zoom the image to fill the panel:** rejected because it can remove meaningful pet pixels.
- **Reveal controls only on hover/focus:** rejected because it hides essential terminal controls.
- **Keep a permanent status or hint row:** rejected because normal operational state is already carried by the pet visual.
- **Overlay alert text on the portrait:** rejected because it obscures the pet without improving urgency.
- **Replace folded mode with a generic two-icon strip:** rejected because folded Buddy should still read as the pet whenever the viewport permits.
