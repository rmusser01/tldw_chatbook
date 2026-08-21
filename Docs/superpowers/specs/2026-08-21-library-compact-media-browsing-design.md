# Compact Library Media Browsing Design

**Task:** TASK-19579 — Optimize compact Library Media browsing

**Date:** 2026-08-21

**Surface:** Library ▸ Browse Media

**Mode:** Operate

**Source finding:** LIB-UX-19 from the 2026-08-20 Library UX/HCI review

## Goal

Make a populated Media page efficient to scan at an exact 100×30 terminal without weakening the existing authoritative paging, focus, recovery, selection, mutation, or wide-screen contracts.

The primary users are regular technical and non-technical power users. First-time users still benefit from the existing distilled Media empty state, which remains unchanged.

## Problem

At 100×30, each Media row consumes two lines and the automatically selected preview is stacked below the list. Fixed toolbar and pager regions then leave only one record visible even though the page owns twenty records. This makes comparison and keyboard traversal unnecessarily expensive.

The wide 170×48 two-pane presentation is already useful and should not change.

## Approved Direction

Use responsive adaptation inside the existing `LibraryMediaCanvas`. Do not create a second compact widget, a density preference, a new controller, or a new navigation model.

```text
Wide ≥120                         Compact <120
────────────────────────────     ─────────────────────────────
▸ Product demo                   Product demo · video · 2d
    video · 2d                   Research paper · PDF · 3d
                                 Interview · audio · 5d
List           │ Preview         Meeting notes · text · 8d
               │ Type: video     Website · web · 12d
               │ Updated: 2d
```

At compact width, a row activation continues to open the existing full Media viewer. Returning from the viewer restores the same applied page, row focus, and list scroll position.

## Responsive Authority

The screen’s existing measured `<120` compact state remains the only responsive authority. Media must not independently read terminal width or introduce another breakpoint.

The existing responsive transition should update the mounted Media canvas in place:

```text
LibraryScreen measured compact state
                  │
                  ▼
LibraryMediaCanvas.apply_compact_presentation(compact)
                  │
                  ├─ patch mounted row labels and heights
                  ├─ patch stored raw toggle labels
                  └─ toggle preview paint/focus participation
```

No resize-driven Media recompose or service request is allowed. The applied scope, page, type filter, selected identities, and row-scroll offset remain unchanged. Focus remains on the same control except when wide→compact hides the currently focused preview action; that transition moves focus to the row represented by the preview selection.

The incumbent state and CSS class are named for Notes (`_library_notes_compact` and `library-notes-compact`) but already govern the Media wide/compact split. Renaming that cross-surface state is outside this task because it would create broad, unrelated churn.

## Row Presentation

Normal compact rows use one line:

```text
Title · media type · relative age
```

- Title is first and remains the primary scan key.
- Type and relative age remain in the label when space permits.
- Textual ellipsis handles overflow; the existing tooltip retains the full title.
- The wide label stays two lines: title, then type and age.
- The normal compact row omits the `▸` marker and preview-selected background/underline because both identify the wide preview selection, not keyboard focus.
- Select mode retains `☐` and `☑` markers.
- The same pure formatter builds initial labels, resize patches, and raw labels used by in-place checkbox toggles.

## Preview and Viewer

The preview remains mounted to avoid a widget-tree rebuild across the breakpoint, but at compact width it is neither painted nor keyboard-focusable. The last row therefore tabs to the pager controls rather than the hidden `Open in viewer` action.

At wide width the existing side-by-side preview, selected marker, and `Open in viewer` action remain unchanged.

If wide→compact occurs while `Open in viewer` or another preview descendant is focused, focus transfers to the corresponding selected Media row. Other mounted controls retain focus. Compact→wide never moves row focus into the preview automatically. Any deferred fallback must verify that focus still belongs to the disappearing preview (or has become empty because that preview was hidden); a newer user focus move cancels the fallback.

```text
Compact browse
  │
  ├─ focus/activate a row
  ▼
Existing Media viewer
  │
  └─ Back/Escape → same page, row focus, and scroll
```

Before viewer entry, the screen captures the activated row’s canonical Media identity and the row-scroll offset. Viewer Back must feed that snapshot into the existing bounded list-entry focus settlement instead of arming its unconditional first-row target. Repeated background recomposes may retry the semantic target during the existing settlement window, but any keyboard or pointer intent disarms it exactly as today.

If the captured row is still on the authoritative applied page, Back restores that row and the prior scroll offset. If a refresh or bounded page clamp removes it, restoration falls back to the first authoritative visible row and a valid contained scroll offset; it must not forge the missing row, return to a failed requested page, or issue a third read. An empty authoritative page focuses the nearest enabled Media browse control.

## Vertical Budget

At 100×30, a settled normal populated browse must paint at least five rows while keeping the truthful pager visible and contained. The list owns remaining vertical space; the pager stays pinned outside the row scroll.

Required feedback can temporarily reduce the row count:

- loading or stale recovery copy;
- mutation-in-progress reason;
- delete confirmation or receipt;
- type chooser;
- Retry state.

These states remain truthful and visible rather than being hidden to satisfy a geometry target.

## State and Failure Behavior

The task changes presentation only. Existing source-owned controller behavior remains authoritative:

- exact 20-row pages and totals;
- filters and complete type facets;
- loading retention and stale read-only gates;
- Retry and one-clamp shrink recovery;
- Select and mutation interlocks;
- delete receipts and Undo;
- metadata-only diagnostics.

Compact presentation must not fabricate counts, derive types from visible rows, enable unsafe stale actions, or issue a read when the breakpoint changes.

## Accessibility and Keyboard Contract

- Focus remains a visible structural state; color is not the only carrier.
- Disabled actions retain the shared `○` marker and explanatory tooltip/reason.
- Compact rows remain full-width Buttons with their existing activation behavior.
- Normal compact traversal skips the hidden preview action.
- Select-mode traversal and checkbox semantics remain unchanged.
- Wide→compact preview focus transfers to the corresponding row; compact→wide retains row focus.
- A user focus move made after a resize or viewer-Back restoration begins must outrank stale restoration intent.

## Baseline Harness Correction

On merged `dev`, `Tests/UI/test_library_media_side_by_side.py` currently mounts a true new-profile Library. The compact Starter rail correctly hides Browse Media, so all eleven tests fail before Media is opened. The test harness must explicitly set `library_new_profile_admission = False` because these tests model a returning populated Library. This is a test-fixture correction, not product behavior or feature scope.

## Verification

Focused tests only, per user direction:

- exact 100×30 production hierarchy and `TldwCli.CSS_PATH` geometry;
- at least five compositor-painted compact rows plus contained pager;
- exact compact label, marker, tooltip, and Tab behavior;
- wide→compact while the preview action is focused, including the newer-user-focus veto;
- row activation, viewer Back, repeated-recompose settlement, page/focus/scroll restoration, and missing-row/page-clamp fallback;
- compact Select, stale, Retry, paging, mutation receipt, and disabled reasons;
- resize compact→wide→compact with no Media service call or state reset;
- exact 170×48 regression for the existing two-line/two-pane preview;
- CSS source regeneration and bundle parity if component CSS changes;
- Ruff and `git diff --check` on the touched files.

Repository-wide pytest is intentionally excluded.

## Alternatives Rejected

1. **Separate compact Media widget.** Rejected because it duplicates paging, selection, mutation, focus, and recovery presentation and invites state drift.
2. **User-selectable density setting.** Rejected because there is no evidence a persisted manual override is needed; responsive density solves the observed defect directly.
3. **Resize-driven recompose.** Rejected because it needlessly rebuilds the focusable row tree and makes focus/scroll preservation harder.

## ADR Check

**ADR required:** no

**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

**Reason:** ADR-067 already governs authoritative Media paging, source ownership, stale recovery, and mutation refresh. This responsive presentation refinement implements those existing contracts and changes no storage, service, navigation, privacy, or long-lived architectural boundary, so no new ADR is required.
