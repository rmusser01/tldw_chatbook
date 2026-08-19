---
id: TASK-18909
title: Reduce Console screen-switch mount cost (1.55s warm on fast hardware)
status: To Do
assignee: []
created_date: '2026-08-19 16:31'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured 2026-08-19 at dev f6ae7d23e (TASK-18908 spike): switching to the Console/ChatScreen costs ~1.55s WARM on an M-series Mac with a scratch config, vs Home 0.76s / Settings 0.89s / Library 1.39s. Screens rebuild fully per switch by design (caching was root-caused to a freeze in July); the cost is construction+compose+CSS-apply of a 21k-line screen plus its Console module chain (transcript+bridge+controller+store grew +9.6k lines in the Aug 15-18 window). On constrained Windows hardware at 3-5x this is 5-8s — the residual of the reported incident. Profile a warm switch (Py-spy or cProfile around handle_screen_navigation) and land the top deferrable items (likely first-visit import already handled by preimport thread; suspect compose-time work that can move to post-first-paint workers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Profile of a warm Console switch exists with named top costs,Top deferrable compose-time work moved off the first paint (or documented why each cannot be),Warm Console switch re-measured and reported in the task,Latency guardrail budgets updated if the improvement moves baseline
<!-- AC:END -->


## Implementation Notes

- Profile (cProfile around a full warm navigation, dev `153a664ce`): the largest app-side cost was NOT screen composition but `build_console_settings_readiness` (~400 calls/switch) each rebuilding the supported + send-capable readiness frozensets by resolving provider identity for all 29 handler keys — 24,484 resolutions / 692k `normalize_provider_config_key` string ops per switch.
- Fix: `functools.cache` on the no-injection path of `_supported_readiness_keys`/`_send_capable_readiness_keys` (pure functions of `CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS`); the `native_provider_keys` test-injection seam stays uncached. PR #1830.
- Result: identity resolution absent from the profile, normalize calls −95%, wall 1.52s → 1.45s (~5%, honest small number).
- "Why not more" (the AC's document-or-defer clause): after the fix, app code is ABSENT from the switch profile — the ~1.4s floor is Textual's mount/CSS/compositor machinery over a ~600-widget screen. The deferrable-work hypothesis was wrong: compose-time app work was already deferral-clean (July's task-15452 memo held). Reducing the floor further requires screen decomposition (splitting ChatScreen's widget tree or upstream Textual caching) — a deliberate architecture project, not a follow-up patch; no task filed because no measured app-side lever remains.
- Guardrail budgets (PR #1827) unchanged: baseline moved ~5%, far inside the 10s budgets.
- ADR: not required — a pure memoization change behind existing function signatures.
