---
id: TASK-33165
title: Dreams goal query-angle feedback adjustment
status: Done
assignee: []
created_date: '2026-09-28 20:23'
labels:
  - dreams
  - phase2
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec feedback-loop section: feedback on a goal-derived story adjusts the QUERY ANGLE for that goal, never the goal itself. Phase 2 implemented and pinned the immunity half (goals never decay, never weight-adjusted); the query_angle adjustment mechanism (dream_interest_profile.query_angle column exists, unwritten) was never planned - filed so it is not lost, per the Phase-2 final review recommendation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Feedback on goal-derived stories records a query-angle adjustment consulted by query synthesis,Goals remain immune to weight changes (existing tests stay green)
<!-- AC:END -->

## Implementation Plan

1. TDD: write failing tests for the DB write (`set_goal_query_angle`), the
   cycle feedback pass (angle derivation, replacement, immunity, degrade),
   and synthesis consultation (payload `goal_angles`, preview labels).
2. `DreamsDB.set_goal_query_angle(facet, text, *, angle)`: parameterized
   UPDATE pinned to `facet='goal'`, direct write so `angle=None` clears.
3. `cycle_service`: share one window read between both feedback halves; net
   topics as before; walk rows oldest-first writing goal angles
   (`avoid: {kind}` / `prefer: {kind}`, `unknown`→`content`, `exported`
   neutral); one thread hop; angle failure degrades alone.
4. `query_synthesis`: `goal_angles` in the payload for SEARCHABLE angled
   goals only; system prompt steering instruction; fallback/preview goal
   lines labeled `(angled: ...)`.
5. Guide sentence; task close-out; ADR check (none: mechanism inside
   ADR-196's feedback contract).

## Implementation Notes

- Approach (TDD: 17 new tests written first, confirmed failing on the
  missing mechanism, then implemented to green):
  - `DreamsDB.set_goal_query_angle(facet, text, *, angle)` — parameterized
    UPDATE pinned to `facet='goal'` (non-goal facet raises ValueError);
    direct write, no COALESCE, so `angle=None` is a real clear; never
    touches `weight`; stamps `updated_at`; missing row = benign no-op.
  - `cycle_service`: one shared window read (`_feedback_window_rows`,
    oldest-first via `(created_at, id)`) feeds BOTH feedback halves inside
    ONE `asyncio.to_thread` hop in `_apply_feedback`. Topic netting is
    byte-for-byte the same semantics (extracted into `_net_from_rows`).
    The goal pass (`_write_goal_angles`) joins each story's matched topics
    against `facet='goal'` profile rows on strip+lower normalization
    (writing to the goal's STORED casing) and records `avoid: {story kind}`
    on `less` / `prefer: {story kind}` on any positive kind; `unknown`
    story kind maps to `content`; `exported`/unrecognized reactions are
    neutral. Chronological walk + plain UPDATE = newest non-neutral
    reaction replaces (positive↔negative, and a second `less` with a
    different kind). An angle-write failure degrades with its own note
    (`goal angle feedback failed: ...`) and leaves the topic offset alive.
  - `query_synthesis`: the privacy gate now filters goal ROWS once, then
    derives both `goals` and `goal_angles` from the same filtered list —
    an unsearchable goal's angle never reaches the payload (asserted in
    tests). System prompt tells the model `goal_angles` steer that goal's
    phrasing. `_fallback_queries` takes `(text, angle)` pairs and labels
    angled goal lines `(angled: ...)`, keeping the preview identical to
    what an LLM-exhausted cycle would actually search.
  - `interest_profile.snapshot`: no change needed — goal rows are full row
    dicts, so `query_angle` rides through (pinned by a new test).
- Goals stay immune: the weight path still only offsets `facet='topic'`
  snapshot rows; `test_feedback_adjusts_user_topics_but_never_goals` and
  every other goals-immune test pass UNTOUCHED (the shared seed helper only
  gained an optional `story_kind="content"` parameter).
- Trade-off noted: because the cycle's LLM-exhausted fallback reuses
  `preview_queries`, its degraded search strings carry the
  `(angled: ...)` label too. Keeping the label in the shared builder was
  chosen over diverging the preview from what the degraded cycle searches
  (preview's documented contract).
- Modified files: `tldw_chatbook/DB/Dreams_DB.py`,
  `tldw_chatbook/Dreams/cycle_service.py`,
  `tldw_chatbook/Dreams/query_synthesis.py`,
  `Tests/Dreams/{test_dreams_db,test_cycle_service,test_query_synthesis,
  test_interest_profile}.py`, `Docs/User_Guide/dreams.md`, this task file.
- Gate ×2 green: `Tests/Dreams/ Tests/UI/test_artifacts_dreams_rows.py
  Tests/UI/test_artifacts_dreams_modal.py
  Tests/UI/test_artifacts_dreams_goals_modal.py
  Tests/UI/test_settings_dreams.py` → 259 passed both runs. Ruff on the
  changed files reports only pre-existing findings.
- ADR check: none needed — the mechanism lives inside ADR-196's feedback
  contract (goals never change weight; feedback steers their query angle);
  no schema, migration, or boundary decision was added (the column already
  existed, unwritten).
