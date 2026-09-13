---
id: TASK-21150
title: 'Wizard/Console review follow-ups from PR #2101'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-26 00:10'
updated_date: '2026-08-27 04:20'
labels:
  - ux
  - console
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minors and recommendations from the TASK-22281..21149 code review (PR #2101): (a) Summary consent allow-path should schedule the model-catalog refresh in the same session, matching the Console modal's allow behavior; (b) AppearanceStep show-all rebuilds (themes and cards) should re-press the row matching the retained selection; (c) bound the three remaining resolve_for_send awaits outside the send path (continuation replay ~3211, instruction preview ~8066, compaction ~9790 in console_chat_controller.py) — same hang class as UAT H-3; (d) make the composer action-link markup invariant mechanical (escape or assert on the reason literal) instead of comment-enforced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wizard consent allow triggers the catalog refresh that session
- [x] #2 Show-all theme/card rebuilds preserve the pressed row
- [x] #3 No resolve_for_send await can hang unbounded
- [x] #4 Composer markup safety is enforced by code, not comment
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. (c) bound the remaining resolve_for_send awaits via the existing _resolve_for_send_bounded seam\n2. (a) Summary consent allow-path schedules the catalog refresh this session\n3. (b) show-all rebuilds re-press the retained theme row (cards done in the Qodo fix)\n4. (d) enforce composer markup invariant in code\n5. Tests per item; suites; preflight
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
(c) Four unbounded resolve_for_send awaits — not three: dev added a dispatch-retry path (_resolve_dispatch_retry_context) after the review. All four (continuation replay, dispatch retry, instruction preview, compaction) now route through _resolve_for_send_bounded; each site already handled a not-ready resolution, so the timeout stand-in degrades correctly. Guarded twice: an AST test that fails on any raw 'await self.provider_gateway.resolve_for_send(...)' outside the bounded helper (the hazard is a FUTURE await, which only source text can catch), plus a behavioral test that compact_context_now returns actionable copy against a hanging gateway.

(d) The composer action link no longer f-string-interpolates the reason into markup — it passes it as a Content.from_markup template variable, so the value is inserted as text and never re-parsed. New test feeds a hostile reason ('[@click=app.quit]…[bold]') and pins that the metacharacters render literally and no second action span appears; it failed before the change, proving the old comment-enforced invariant was genuinely injectable.

(a) SummaryStep.commit's allow path now calls wizard.request_model_catalog_refresh() -> app.refresh_model_catalogs_now(), which dispatches the same worker in the same exclusive 'model-catalog-refresh' group the Console modal uses (so the two can never race). Failure is non-fatal — the answer is already saved. Parametrized test pins allow=refresh, deny=no network.

(b) Already correct: _theme_buttons always pressed the match, and _card_buttons gained the same during the Qodo pass on PR #2101. No code change; added the test that proves selection survives both show-all rebuilds and locks it in.

Also fixed TASK-22858 in the same branch (dev's own guard breakage): LibraryEmergencyReturn.DEFAULT_CSS -> BUNDLED_CSS + regenerated sheets.

Suites: 1102 passed, 0 failed; preflight green.
<!-- SECTION:NOTES:END -->
