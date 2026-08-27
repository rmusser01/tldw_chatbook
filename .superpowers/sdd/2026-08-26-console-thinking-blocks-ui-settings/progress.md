# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-thinking-blocks-ui-settings.md

Setup: isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-thinking-blocks`, branch `codex/console-thinking-blocks`.

Dependencies: TASK-18932.1 complete at `9906b4d4bd`; TASK-18932.2 complete at `7ec7fbc9dc`. This child consumes the reviewed envelope, live capture, replay-policy, and owner-group contracts.

Impeccable: Operate mode; incumbent Neon Workbench vocabulary; required craft floor loaded before UI implementation. Ponytail full: reuse existing disclosure, expansion, Settings, and Context seams.

Task 1: BASE `7ec7fbc9dc`; pure trusted thinking-activity projection and turn ordering only.

Task 1 initial implementation: commit `236b8a448d`; 34 focused and 171 nearest
presentation tests passed; Ruff and `git diff --check` clean.

Task 1 review fix round 1: reproduced generation identity collision and positional
round inference at RED (`3 failed, 34 deselected`). Added mandatory stable
`generation_id` to the projection/ID contract and explicit session-only
`activity_round_ordinal` ownership to activity rows. GREEN: 38 focused and 175 nearest
presentation tests passed. Task 2 must reuse an installed variant ID, frozen live
generation-attempt fact, or durable restored generation identity; the later activity
producer must stamp exact round ordinals and leave post-run rows unowned.

Task 1 review fix round 2 (supersedes round 1's `generation_id` caller contract):
reproduced same-owner capture collision and caller-manufactured presentation identity
at RED (`2 failed`). `ThinkingCapture` now allocates one capture-scoped UUID namespace
and persists it inside every new block ID; the presentation projection hashes that
stored block ID alone, so activity identity remains literal-equal from live unpersisted
capture through variant finalization, first message persistence, variant switch-back,
and fresh-store durable hydration. A second same-Assistant capture differs, existing
durable IDs remain unchanged, and raw imported IDs stay hidden by UUID5. GREEN: 2 new
identity tests, 208 foundation/capture/presentation/grouping tests, and 6 nearest real
store/variant lifecycle tests passed. The explicit `activity_round_ordinal` producer
contract from round 1 remains in force; Task 2 no longer supplies any generation ID.

Task 1 review fix round 2 re-review closed the capture lifecycle but found one Priority-1
legacy/import collision: V1 guarantees block-ID uniqueness only within one envelope,
so hashing the block ID alone coupled different Assistant owners carrying the same
valid existing/imported ID.

Task 1 review fix round 3: RED was 4 failed plus 2 passing ordinary-message controls.
Activity identity now hashes the stable native Assistant owner plus persisted block ID;
thinking-owning Assistant rows pin their native ID on first persistence and the private
thinking hydration pass restores it before indexing. The generic conversation tree
continues to omit private thinking and ordinary messages retain database-allocated
identity. GREEN: 5 round-3 contracts and 210 foundation/capture/presentation/grouping
tests passed. A whole-file store probe was 283 passed with 13 failures on unchanged
fake-persistence/provider-history seams already present at the round-3 base. Ruff/diff
checks passed; `test_console_chat_store.py` retains its base whole-file formatter
failure without unrelated churn. Final independent re-review remains pending.

Task 1 review fix round 4: reproduced the remaining real lifecycle at RED: an ordinary
Assistant persisted before gaining thinking hashes its native owner before restart,
while private generation hydration restores it under the database-allocated durable
owner. Projection now uses `persisted_message_id or id`, the smallest shared fix. New
unpersisted thinking remains stable because first persistence pins native and durable
IDs equal; existing legacy cross-owner, second-generation, temporary-session, ordinary,
capture, presentation, grouping, and variant lifecycle controls remain GREEN in one
fresh 195-test run. Ruff lint passed on all scoped Python files, Ruff format check passed
on production/presentation, and `git diff --check` passed. Included in the round-4 fix
commit.

Task 1 independent re-review: READY at `0f05fcd826`. The reviewer reran the exact
durable-owner lifecycle plus the complete presentation/grouping contracts (39 passed)
and found no ordering, privacy, compatibility, schema, or migration issue. Task 1 is
complete after four bounded review-fix rounds.

Task 2: reused the incumbent activity disclosure for lazy trusted thinking evidence,
one-time manual-win auto-collapse, in-place live updates, selection/navigation, Copy
and Inspector display resolution, session pruning, and semantic status styling. RED was
13 failed/40 passed; focused GREEN is 54 passed. The one required broad target run was
192 passed/4 failed; both Task-2 regressions were fixed and rerun GREEN, while two stale
BASE citation assertions still target pre-Assistant-turn row keys/counters. CSS build
and bundle sync, scoped Ruff lint, focused Ruff format, and `git diff --check` pass.
Task 2 report: `task-2-report.md`. No ADR required.

Task 2 review fix round 1: reproduced all four Priority-1 findings at RED. The
real ChatScreen Inspector now resolves current thinking IDs through the owning
Assistant and renders the full displayable/proprietary body while retaining the
active-session store guard. Lifecycle closure uses newly arrived Tool identity
rather than speculative round ordinals; proprietary blocks use live block identity
while displaying `unavailable`. Thinking activity IDs now map to the Assistant's
causal ownership tuple so selection protects the owner through watermark pruning
and two-sided tail windowing. GREEN: 58 Task-2 disclosure/Assistant tests, 17
Inspector/pruning/windowing tests, and one rendered-Inspector follow-up passed.
Scoped Ruff lint and `git diff --check` pass; whole-file Ruff format remains the
pre-existing legacy-file baseline. No CSS changed. Report updated in
`task-2-report.md`; no ADR required.

Task 2 independent re-review: READY at `3dad2a46a7ca`. The reviewer reran 26
fix-specific ownership/Inspector controls plus 62 disclosure/manual/lazy/ordinary
interaction controls; all passed, with Ruff and `git diff --check` clean. Task 2 is
complete after one bounded review-fix round. Existing ADR-090 remains the governing
decision; this task introduced no additional architectural decision.
