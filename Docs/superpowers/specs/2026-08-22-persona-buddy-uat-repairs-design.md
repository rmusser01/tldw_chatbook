# Persona Buddy UAT repairs design

Date: 2026-08-22
Task: [TASK-20938](../../../backlog/tasks/task-20938%20-%20Repair-Persona-Buddy-restart-and-frame-sizing.md)
Decision: [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)
Status: Approved

## Problem

Full-application UAT on current `dev` found two independent defects:

1. The bootstrap loader reads and preserves the raw `[persona_buddy]` TOML table, but `_load_settings_uncached()` does not project that table into the dictionary assigned to `TldwCli.app_config`. `TldwCli` therefore initializes `PersonaBuddyController` from `{}` after every restart. The on-disk preference remains correct until a Workbench action serializes the default in-memory snapshot, at which point valid geometry can be replaced by the never-positioned sentinel.
2. `PersonaBuddyWidget` passes its whole window `content_region` to `resolve_current_visual()`. The image is prepared for that larger height, then painted inside the shorter `#persona-buddy-frame` Static below the three-row header and above status and hints. The Static clips the prepared image, so the user sees only part of the portrait.

The apparent grayscale result was not a third defect. The UAT environment exported `NO_COLOR=1`; Textual correctly installed its `Monochrome` line filter. Removing that variable produced non-grayscale true-color cells without a product change.

## Decision

Repair both root causes in one narrow PR because they are independent regressions of the same completed Buddy surface and share the same full-app UAT fixture. Do not redesign the Buddy, add preferences, change Persona Visual contracts, or special-case the seeded Samira assets.

### Startup preference projection

Project the raw `persona_buddy` table into the existing `config_dict` returned by `_load_settings_uncached()`, alongside the other pass-through application tables. Preserve it as an ordinary mapping for the existing strict `parse_persona_buddy_preferences()` boundary; do not duplicate field parsing in `config.py`.

The regression test must load an isolated real TOML file through `load_settings(force_reload=True)` and prove the exact Buddy table reaches the parser/controller startup seam. A malformed-field control must continue to receive the parser's current field-local defaults. A restart/first-action control must prove saved geometry is not replaced with the sentinel.

### Frame-slot sizing

Use the visible `#persona-buddy-frame` content region as the sole size authority for image resolution. The same exact `(width, height)` must participate in `_resolution_authority()` and be passed to `resolve_current_visual()`, so a frame-slot layout change invalidates the cached resolution while an unchanged slot does not trigger repeated decode work.

If the frame Static is hidden, detached, compact, collapsed, or has a zero-sized content region, resolution must not start. Existing view-generation, controller-generation, cancellation, unavailable-reconcile, and post-await fences remain authoritative and unchanged.

The regression test must mount the production widget with its bundled stylesheet, capture the actual window and frame regions, and assert that the resolver receives the frame slot rather than the container. A production-shaped colored square frame must paint completely inside that slot; the test must discriminate the old vertically cropped result. Controls must cover frame-slot resize invalidation and hidden/collapsed refusal.

## Verification

Implementation follows RED → GREEN independently for configuration and rendering. Each root-cause guard receives a deliberate mutation that restores the old behavior and makes its focused test fail.

After focused preference, widget, app-mount, resolution, Workbench, and architecture gates pass, run one isolated real `TldwCli` UAT with HOME, XDG, config, data, cache, and imports redirected before interpreter startup. The run must explicitly remove `NO_COLOR`, use the real local Persona and Persona Visual repository, and use a disposable local provider for trusted lifecycle transitions. It must verify:

- enabled selection, open/collapsed state, and geometry restore after process restart;
- no geometry rewrite before a user action;
- the complete portrait is recognizable and emits non-grayscale true-color cells;
- idle animation and trusted thinking, speaking, tool, approval, error, and recovery transitions repaint without retargeting the Persona;
- navigation, fold/open, close/show, keyboard geometry, and shutdown remain sound;
- the real user config and unrelated worktrees remain byte-clean.

## Exclusions

- No new ADR, schema, dependency, asset, renderer, or Buddy preference.
- No change to `NO_COLOR` behavior; honoring the standard environment variable is correct.
- No compensating geometry merge or startup write. Correct projection removes the stale-default source.
- No visual redesign or default-size change unless the corrected full-frame UAT still demonstrates an independent legibility defect.
