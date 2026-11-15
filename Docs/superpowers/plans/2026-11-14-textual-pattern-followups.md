# Textual follow-ups on current remote dev

Baseline: `ecad404bd4567b57139716a6deaa6b37fe833749` (`origin/dev`, fetched 2026-11-14).

Spec: the acceptance criteria in TASK-37360, TASK-40969, and the Speech child of TASK-1378. The user approved these three bounded steps in this order. Preserve existing public behavior except the confirmed input-preservation defects.

## Global constraints

- Work in the existing isolated worktree. Preserve the prior branch and untracked test artifacts.
- Run targeted tests only. Do not run the full test suite.
- Use real mounted Textual controls for event and focus behavior; keep privacy/bootstrap guards intact.
- Do not weaken existing assertions or change production debounce to make a regression pass.
- Use existing category-panel, input-tracking, worker, and design-token contracts. Do not add a parallel framework.
- Keep the existing Settings commit models, persistence ownership, and TTS global/Studio boundary.
- Record red/green evidence and limitations honestly. A task is Done only after its acceptance criteria and relevant checks are satisfied.

## Task 1: Media-filter preservation (TASK-37360)

ADR required: no. ADR path: N/A. Reason: repair retained-control behavior within existing ADR-033/086/161 and applied-scope contracts.

1. Add a mounted regression that types ordinary characters across settled browse publications, asserting the exact query and the same focused Input with retained caret/selection. Record failure before changing production code.
2. Retain the Media filter/chrome; recompose only the results subtree and direct post-recompose callbacks to that owner. Preserve a newer live draft against older loading/results publications.
3. Remove whole-screen replacement from the ordinary clean-Reader zero-result path using existing targeted sync.
4. Verify held older completion, Clear, zero results, applied scope, dirty Reader, sorting/paging, and relevant canvas/governance tests.

## Task 2: Context-memory events (TASK-40969)

ADR required: no. ADR path: N/A. Reason: qualify and repair existing draft/input contracts under ADR-052/161/303; no new commit model.

1. Mount the four Inputs and five Selects, exercise programmatic refresh and queued real edits, and record whether synthetic Changed events dirty the draft or overwrite newer raw text.
2. If confirmed, reuse the existing tracked-input/suppression pattern and protect pending real edits. Preserve percent conversion, invalid raw strings, validation, Save, and confirmed Revert.
3. Run the new event regressions and the focused context-memory/remaining-input tests.

## Task 3: Continue Settings ownership (Speech child of TASK-1378)

ADR required: no. ADR path: N/A. Reason: continue the existing per-category registry migration under ADR-303/304/161, preserving persistence ownership and behavior.

1. Pin current Speech navigation/leave, dirty banner/count, help, and staged Save/Revert behavior with mounted controls, including Cancel, discard, save failure, and audio.cpp cleanup where touched.
2. Move remaining Speech category decisions into the registered panel using the established extension points. Use one authoritative unsaved snapshot for cheap repaint queries; capture live pending fields at consequential action boundaries.
3. Preserve global-versus-Studio scope, callback/monkeypatch seams, and no-op behavior for other panels. Update focused architecture inventory/ratchets only as necessary.
4. Run targeted Settings behavior and affected architecture/style checks. Keep the broader parent open for remaining categories.

## Review and completion

Execute one implementation task at a time. Review each task's diff against its acceptance criteria and tests before proceeding; finish with a combined review. Update task notes, links, and checked criteria only from observed evidence. No merge or publish is part of this request.

## CI integration

Add the verified new UI regression nodes to the existing PR-gate census, preserving the census floor and per-node placement. Keep the recorded 13.3-minute replay target and historical capacity envelope; use the existing TASK-39144/ADR-103 shard-scaling policy for the smallest justified local count adjustment if needed. Record actual timing sources and uncertain projections separately. This is targeted regression routing, not a new CI framework or permission to run the full suite.

## Completion

TASK-37360, TASK-40969 and TASK-1378.3 are Done with checked criteria and implementation notes. The broader TASK-1378 remains In Progress. Independent task reviews and the final whole-branch review approved the implementation. [Verification and limitations](../qa/2026-11-14-textual-pattern-followups/verification.md) include baseline failures, final receipts, scanner qualification limits and CI capacity assumptions. The initial implementation was committed as `6b947be8288bdb4a92e48339a23b98da81f3ba94` and published in PR #3569 at the user's subsequent request.

## Requested rebase (2026-11-15)

Rebase PR #3569 onto fetched `dev` at `2ff0e9a5e5f29566c92ea3f49c82776b57eacfe9`, preserving both sides of each conflict. Run targeted product, architecture, CSS and CI checks, then push with an explicit lease against the published head. No merge is requested.

ADR required: no new ADR. Existing ADR-103 covers census capacity; ADR-303/304/161 continue to govern Settings ownership. The combined census retains all 2076 incoming selectors plus the original 27 cases. Its measured/charged capacity requires 40 shards at the unchanged concurrency limit of 18. TASK-39144 retains actual post-change CI observation. See [rebase verification](../qa/2026-11-14-textual-pattern-followups/rebase-2026-11-15.md) and [current CI evidence](../qa/2026-11-14-textual-pattern-followups/ci/README.md); prior 38-shard evidence remains archived.

## CI feedback and further integration (2026-11-15)

The user subsequently requested CI/review monitoring and fixes. Preflight on
`374207ae0f` found that the Speech migration's two constant debug messages had
moved owners without refreshing the diagnostic inventory. Statement-level review,
regeneration, an ordinary checker run and five focused tests qualified the
metadata correction published as `64e46a5584`; no production code changed.

While CI was queued, `dev` advanced to
`cc6006fcf59328353004206bba8724c399216749`, preventing CI for that new head until
the conflicts are resolved. Rebase both commits while preserving incoming
Library focus/Find/Collections changes, Settings custom-endpoint recovery,
runtime-policy test-factory changes, all CI lanes and all incident/task notes.
Keep the retained Media filter and defer incoming empty-page focus recovery
through the results owner's completion callback with existing focus guards.
Use exact combined ratchets and targeted mounted integration checks.

ADR required: no new ADR. Existing ADR-103 governs the new upstream 13.0-minute
capacity target and concurrency of 19. Reconcile all 2145 incoming selectors with
the same 27 PR cases using the upstream measurement window plus recorded PR
charges. Review the diagnostic artifact again on final integrated source, then
push with a lease and continue current-head CI/review monitoring. No merge or
full local test sweep is requested.
