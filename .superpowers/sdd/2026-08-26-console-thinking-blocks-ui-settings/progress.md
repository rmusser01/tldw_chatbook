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

Task 3: canonical F9 now owns a default-on device-local thinking visibility toggle
with immediate refresh and failed-write rollback. The existing Context & memory modal
owns conversation Auto/Include/Exclude replay policy, effective Required copy, and a
bounded new-conversation-default action. Fresh sessions copy the resolved default;
Assistant edit copy explicitly names generation provenance loss. The planned
foundation policy seam was absent at base, so the parent-approved implementation adds
only the minimal conversation-owned store/`ChatPersistenceService` getter/setter.
Focused GREEN evidence: 21 Settings, 66 lifecycle/edit, 3 mounted visibility, and 2
default-copy tests; scoped Ruff lint and `git diff --check` passed. Existing ADR-090
governs; no new ADR is required. The exact plan filter also reached 117/118 with one
unchanged project-instruction DOM-parent assertion that reproduced alone and is outside
Task 3 ownership. Report: `task-3-report.md`.

Task 3 review fix round 1: all three Priority-1 findings reproduced RED and closed.
Effective Required now comes from the existing provider resolver plus the same
continuation-target/group selector used by send, through one shared pure effective-
policy helper; active recovery, incompatibility, changed targets, and unresolved
providers retain the saved optional value. ConsoleChatStore owns one runtime-wired
live thinking-history default provider consulted by every omitted-policy
create_session; explicit/restored values win and legacy NULL hydration remains Auto,
so the UI-only post-create setter/resolver was removed. Device visibility persistence
is one serialized/coalescing drain with desired, confirmed, and in-flight revision
state; latest failure restores checkbox/config/label/transcript to the confirmed disk
baseline and reports the existing error. RED evidence: missing controller/builder seam
(3 AttributeErrors plus one rejected keyword), rejected store provider/runtime Auto
mis-default, and legacy rapid-toggle worker signature/state mismatch. GREEN evidence:
5 effective-policy cases, 2 direct store/runtime-default cases, 2 deterministic
coalescing/restart-equivalent persistence cases, and the final affected filter at
128 passed/460 deselected in 67.30s. Scoped Ruff passed all 12 changed Python/test
files. Existing ADR-090 governs; no new ADR is required. Commit `f23d0f51fe`.

Task 3 review fix round 2: the visibility drain now consumes the structured
ConfigMutationResult from the existing atomic settings mutation primitive rather than
the boolean adapter. A replaced file is the confirmed disk baseline even if cache
publication fails, so partial failure keeps the optimistic UI, reports the refresh
error, and schedules a corrective latest write when needed; no-op success confirms,
while conflict/before-replace failure rolls back. RED was 4 failed/1 passed (the pass
was accidental object truthiness); GREEN was 5 passed. The affected thinking Settings
filter was 17 passed/368 deselected in 10.36s; scoped Ruff and diff-check passed.
Existing ADR-090 governs; no new ADR is required. Round complete.

Task 3 independent re-review: READY at `664f0fc542`. The reviewer reran 21
settings-policy/persistence controls and confirmed the structured mutation outcomes,
one-writer coalescing, continuation-derived Required state, and universal store-level
new-session default together. Ruff and `git diff --check` passed. Task 3 is complete
after two bounded review-fix rounds.

Task 4: renamed the safe intermediate primary model-step activity to Planning and
suppressed it only for exact rounds owned by selected-generation displayable or
proprietary Thinking evidence. Live and resume now stamp session-only activity round
ownership; the existing controller passes selected ThinkingEnvelope round sets by
stable Assistant owner. RED was 10 pure failures and 4 live/resume ownership failures;
GREEN was 10/10 and 4/4. The prescribed bridge/activity/UI run reached 558/560 with
two stale no-summary Planning expectations; both passed after correction and the
complete bridge/activity files finished 403/403. All 157 UI cases in that prescribed
run were green. Scoped Ruff lint and `git diff --check` passed. ADR-090 governs; no new
ADR is required. Report: `task-4-report.md`. Parent retains visual/detector, Backlog,
and export-task ownership.
