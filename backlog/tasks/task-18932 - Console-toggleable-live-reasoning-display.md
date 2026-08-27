---
id: TASK-18932
title: 'Console: toggleable live reasoning display'
status: Done
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-27 09:40'
labels:
  - console
  - ux
  - streaming
  - persistence
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface actual model-provided reasoning in the Console as durable, dim, collapsible Thinking blocks inside the owning Assistant turn. Displayable thinking streams expanded and auto-collapses when visible answer/tool activity begins. Actual proprietary-reasoning evidence renders a content-free `Thinking · unavailable` disclosure with the exact notice `Proprietary thinking obfuscated - not available`; provider capability alone never creates a block. Keep displayable thinking separate from ADR-063 private continuation, persist it with the selected assistant generation, and offer provider-resolved Auto/Include/Exclude replay for compatible model history. The global Show model thinking setting defaults On and remains presentation-only. Importable conversation exports preserve thinking while human-readable, search, summary, title, usage, speech, and logging surfaces exclude it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Show model thinking` is a device-local persisted canonical Console setting that defaults On, applies immediately without restart, stays out of conversation sync/export, and changes presentation only.
- [x] #2 Actual displayable reasoning streams inside its owning Assistant turn, starts expanded live, auto-collapses once when answer/tool activity begins (or at terminal fallback), restores collapsed, and honors manual disclosure choices.
- [x] #3 A current-turn proprietary-evidence event renders `Thinking · unavailable` with exactly `Proprietary thinking obfuscated - not available`, follows the same live/collapse/manual lifecycle, and exposes or retains no raw private content there.
- [x] #4 Provider/model capability, settings, timing, or absence of visible answer content never fabricates a Thinking block; a turn with no actual displayable or proprietary evidence shows none.
- [x] #5 Displayable/proprietary blocks persist in a bounded versioned envelope separate from answer content and ADR-063 continuation, remain paired with the selected generation through regeneration/edit/delete/recovery/sync, and never transfer between variants or branches.
- [x] #6 Every conversation offers Auto/Include/Exclude optional thinking history plus effective read-only Required when continuation mandates replay; the user can save a default for new conversations.
- [x] #7 Replay is adapter-resolved for the frozen target and exact compatible encoding, uses one owner-atomic serializer/budget projection, never sends proprietary markers, never duplicates reasoning/continuation, and fails before provider contact when an eligible Include block is incompatible.
- [x] #8 Importable round-trip conversation formats and capable persistence/sync backends preserve thinking and replay policy with sensitivity warnings; unsupported persistent backends fail before send instead of silently losing data.
- [x] #9 Human-readable exports, FTS/search, titles, summaries, diagnostics, logs, errors, usage, and speech exclude model thinking by default, with decoded negative tests across every default durable owner.
- [x] #10 Existing safe intermediate model-step summaries are presented as Planning when no actual Thinking block owns that round, preventing duplicate or misleading chain-of-thought claims.
- [x] #11 Focused provider, stream-boundary, migration, persistence, history, import/export, privacy, and painted Textual tests cover positive, negative, stopped, failed, restored, variant, edit, and no-evidence paths.
- [x] #12 The user guide documents visibility, proprietary evidence, replay policy, persistence/backend requirements, export behavior, and provider honesty boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md
Reason: the approved feature changes message/variant storage, schema and migration behavior, sync/conflict ownership, provider streaming contracts, optional history replay, privacy/export boundaries, and long-lived Console interaction structure. ADR-090 extends ADR-063 and ADR-066 without exposing or duplicating private continuation.

1. Finalize and approve the linked Console Thinking Blocks specification and ADR-090.
2. Create dependency-ordered atomic child tasks for persistence/sync, provider/history, UI/settings, and import/export/privacy integration.
3. Produce exact implementation plans with test-first seams and targeted verification commands for each child.
4. Implement and verify the child tasks without widening TASK-18932 beyond its approved acceptance criteria.
5. Complete joined privacy, migration, backend-compatibility, and live Console verification before closing the feature parent.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Completed the four dependency-ordered children for bounded generation-owned storage/sync, typed provider capture and replay, collapsible Textual UI/settings, and exchange/privacy/joined integration. Schema v52 migration and ADR-090 preserve answer, thinking, usage, and separate ADR-063 continuation as one selected-generation owner.
- Console disclosures are created only from actual adapter events. Displayable thinking is expanded live and auto-collapses once at the first answer/tool boundary (terminal fallback); manual interaction wins. Proprietary evidence is structurally text-free and renders exactly `Proprietary thinking obfuscated - not available`. Capability alone creates no row.
- `Show model thinking` defaults On and remains device-local presentation state. Conversation Auto/Include/Exclude and effective Required remain independent, model-specific, persistence-aware replay controls; reasoning and plain models may share one local backend.
- Importable JSON and Chatbook V2 preserve supported thinking/policy with sensitivity warnings. Ordinary exports and answer-oriented search, summary, title, usage, speech, copy, diagnostics, errors, and logs omit thinking and the application notice.
- Serial implementer/spec/code-quality reviews were completed for every child. Final feature review found and red-first fixed one leading-whitespace splitter fail-open that could expose tagged local reasoning; independent re-review returned Ready to merge with no remaining Critical, Important, or Minor issues.
- Post-rebase Qodo review raised five valid public-API documentation findings. The prepared-request, replay-policy/resolver/serializer, and settings-loader helpers now document their exact arguments, results, and contract exceptions; the 68-test focused review matrix and scoped Ruff checks pass.
- Fresh post-rebase evidence: 1,299 passed / 2 expected loopback-permission skips / 2 known warnings across the affected matrix plus latest-dev migration compatibility; all six repository preflight guards pass, including CSS and four generated bundles, with diagnostic inventory at 539 / 1,249 / 7,377 / 8; scoped final Ruff/format, `py_compile`, feature-range diff check, and the earlier post-fix isolated live harness pass. The full repository suite was intentionally not run.
- Live evidence: `Docs/superpowers/qa/2026-08-27-console-thinking-blocks-live-verification/`. User documentation covers actual-turn honesty, exact unavailable copy, collapse/manual behavior, replay/persistence compatibility, sensitive importable exchange, ordinary export omission, and Planning semantics.
- ADR check: `backlog/decisions/090-console-thinking-block-ownership-and-replay.md` is the accepted governing ADR; no additional ADR was required during implementation.
<!-- SECTION:NOTES:END -->

## Child Tasks and Plans

- [TASK-18932.1](task-18932.1%20-%20Persist-selected-generation-thinking-and-replay-policy.md) — [persistence foundation](../../Docs/superpowers/plans/2026-08-26-console-thinking-blocks-foundation.md)
- [TASK-18932.2](task-18932.2%20-%20Normalize-provider-thinking-events-and-history-replay.md) — [provider and history](../../Docs/superpowers/plans/2026-08-26-console-thinking-blocks-provider-history.md)
- [TASK-18932.3](task-18932.3%20-%20Render-collapsible-Console-thinking-and-settings.md) — [UI and settings](../../Docs/superpowers/plans/2026-08-26-console-thinking-blocks-ui-settings.md)
- [TASK-18932.4](task-18932.4%20-%20Complete-thinking-exchange-privacy-and-integration.md) — [exchange and integration](../../Docs/superpowers/plans/2026-08-26-console-thinking-blocks-exchange-integration.md)
- [Master implementation plan](../../Docs/superpowers/plans/2026-08-26-console-thinking-blocks.md)
