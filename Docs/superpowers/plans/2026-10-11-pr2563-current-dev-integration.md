# PR2563: native goal runs on current dev

The user authorizes review, fixes, rebase and protected sequential merge of PR2563. The existing accepted native goal design is binding; this integration adds no later milestones. Original source: 50eec6e8e4ec24095a404b1e83dcf757ff17623b. Initial dev: a3c47c6c72d885db2af0971515ddcdcb6cee78fa.

Spec: original Docs/superpowers/specs/2026-09-08-gnhf-inspired-goal-runs-design.md, also preserved in the wave2 evidence directory. Current-dev audit and exact source manifest are in that directory.

ADR required: yes, existing native goal decision with a current-dev integration amendment.
ADR path: backlog/decisions/227-native-console-goal-runs.md (verify global ID ownership before restoration; preserve original provenance if renumbering).
Reason: goal lifecycle/storage is already designed; current schema, backup, custody, allowance and filesystem boundaries require the original decision to describe the actual integration. Governing current ADRs include 069, 131, 134, 135, 150, 158, 208 and 219.

## Global constraints

- Work only in this managed worktree. Do not mutate the primary dirty checkout or any other worktree, stash or shared Git configuration. Never push or merge from the implementation task.
- Use current dev as authority for all shared modules. Carry only goal-specific changes between 77bc58dc171c3dcd2178f4433d19a6ceb59b1e7b and 20bb97cdded8f45cc77fd9848d232e5dc951b24c plus directly relevant initial goal governance. Exclude unrelated prerequisite and scheduling changes.
- No full test sweep or cloud/provider spending. Run targeted meaningful verification using primary .venv Python3.12 with PYTHONPATH set to this worktree and isolated test/profile directories. Historical source evidence remains historical.
- Read actual task files and relevant ADRs/lessons before implementation. Update Backlog status/plan before implementation, using the installed CLI. Preserve existing notes; do not mark unverified acceptance criteria Done. Coordinate any fresh task/ADR ID with root because later queued PRs contain overlapping IDs.
- Preserve validation, permission, resource, recovery, privacy, accounting, runtime custody and shutdown boundaries. No raised size/style/test ratchets or foreign resource closure.
- Every visible control must work through current Console owners and current token-backed design language. Read design-language.md and component-patterns.md before UI changes.

## Task 1: Integrate and qualify native goal runs

This is one cohesive integration task: its storage, runtime and UI contracts are coupled, so perform the following phases sequentially in this worktree, committing reviewable phases without creating competing runtime owners.

1. Inspect the original goal tasks/spec/ADR and current connected code. Restore only the relevant source governance; amend its current-dev integration details before code. Record current plans/ADR checks in the actual goal Backlog tasks. Report proposed fresh ID allocations to root before minting.
2. Adapt goal storage to the actual next AgentRuns schema (initially23 ->24) without relabeling predecessor migrations. Extend exact backup/recovery catalog and domain owners and installed SQL/audit-file policy. Preserve all existing data and supported predecessor catalogs. Test populated upgrade/reopen, archive/staging, rollback/cancellation and unknown/hybrid refusal without automatic replay.
3. Extend current automatic-work models/attempts and policy for goal iteration while retaining progress-only wakes, message acknowledgement, automatic chat starts, immutable allowance roots, conservative unknown usage and manual reserves. Integrate one runtime startup audit and current custody/capacity/decision owners.
4. Port goal service, evidence/verifier, prepared history, launch/provisioning and finite lifetime admission. Fix the confirmed unstarted Pause/Resume trap with safe no-accepted-attempt guards; retain strict checkpoint and uncertainty rules after acceptance. Keep selected source reads under current admitted-root/workspace executor boundaries and primary-only writes; sources never grant instructions or permissions.
5. Add goal setup/history/status, canonical Settings and composer /goal through current command/draft routes, using design tokens and current CSS/size governance. Preserve cancellation/refusal drafts and revalidation after waits.
6. Run goal-specific source suites and affected current migration/backup, automatic-work/chat-start, runtime custody, human decisions, hooks, provider, permission, admitted-root/executor and UI governance regressions. Add actual mounted Pause-before-first-capacity -> Resume -> first dispatch and real isolated CLI nonzero -> correction -> zero plus process-restart/no-duplicate evidence. Fix actual integration defects with tests that demonstrate behavior.
7. Self-review the complete resulting diff, scoped Ruff/format/diff checks. Update tasks/implementation notes with exact current receipts and explicit remaining external qualification; do not promote old failed live-model trials. Commit only task-owned files and return the report path, phase commits and targeted receipts. Root will perform independent review, latest-base rebase, GitHub checks and merge.

Acceptance: the original first-release behavior works on current owners; all identified review defects are addressed; current affected boundaries retain their behavior; new evidence validates actual integrated code. No source stack duplication or unverified completion claims.

## Implementation closeout

The complete implementation is committed through `8bf7c02e51` on the pinned `a3c47c6c72` base. Storage and exact recovery are schema24; current runtime custody, automatic chat-start roots, progress wakes, admitted filesystem roots and backup staging remain the governing owners. ADR227 records the live decision; original source ADR141 and task history remain provenance.

Current owner line-count excess is inherited: controller 29474 -> 29448 (cap29299), store22642 ->22633 (cap22344); ChatScreen25142 ->25135 and756 methods meets25204/756. No ratchet was raised. Root approved narrowing the integration delta instead of repairing unrelated baseline excess.

The final report and byte-bound test receipt are `.superpowers/sdd/2026-10-11-pr2563-current-dev-integration/task-1-report.md` and `pr2563-complete-head-receipt.json` in the external wave2 evidence directory. Full integration review, latest-base reconciliation and publication remain root-owned; TASK34797 AC5 remains pending until that review. No full test sweep or external model qualification was run.

## Independent review fix wave

The complete-head review of `8bf7c02e51` found two blocking integration defects.
Checkpoint and quality-review settlement must include all canonical allowance
root descendants, typed attempts and conservative accounting, including work
that becomes uncertain after a checkpoint. Goal callbacks and native
provider/tool authority checks must retire only the SQLite handles they acquire
using current operation-owned boundaries; borrowed, in-memory and foreign
handles retain their owners. The inherited hook-preflight implementation was
compared to pinned base before extending its finite callback boundary only
when Goal authority is active. No ordinary hook policy or global resource
threshold changes.

The combined fix adds real SQLite descendant and late-uncertainty regressions,
fresh/borrowed actual CLI and mounted paths, cancellation/physical-drain and
maintenance controls. Exact baseline acquisition receipts remain historical;
fixed probe filenames are distinct. All seven Goal/composer/integration tasks
remain In Progress and TASK34797 AC5 stays open for root's scoped re-review.
The task-1 report records the complete fixed commit and byte-bound qualification
when finished. The historical plan's added trailing whitespace is corrected.

Code fix commit: `790d456f213c7c185e756f490162c47887bbae8e`. Supporting fix lanes pass 80 Goal runtime, 96 current owner/budget/hook/maintenance, and 36 mounted UI cases without resource warnings. Exact post-finalizer probes report zero retained AgentRuns, Chat and Workspace handles. Final complete-head qualification uses `pr2563-fixed-head-receipt.json` and `pr2563-fixed-head-verification.txt`; the report preserves both prior failed resource evidence and the baseline JSON overwrite/repeat provenance.


## Task 2: Satisfy actual publication diagnostic and async-UI contracts

This task closes newly observed required local preflight failures after independent complete-head review. It is the same authorized first-release scope, not an additional feature or changed architecture. Initial rebased head e1e2818dfacd5d31887b14178d6a812ffdc16885; actual base9f8533686c6f5d775b98a18bd57c9bee4c518c9f. Frozen pre-rebase review head b919d3ac0a040b7694fe1233497a3a19f2425499 is preserved at codex/wave2-reviewed-pr-2563-b919.

ADR required: no new ADR.
ADR path: existing227 native Goals,097 diagnostics/reference-backed Console and150 UI tokens.
Reason: correct required artifact records and post-await widget lifetime handling within existing ownership; no new lifecycle authority or diagnostics/privacy policy.

1. Read pr2563-fixed-preflight.log in the external wave2 evidence directory. The first restricted-network Mermaid failure is separately superseded by pr2563-fixed-mermaid-network.log, which passed6outputs; do not change that guard or assets.
2. Inspect exact diagnostic statement deltas in console_goal_runs.py, console_chat_store.py and console_conversation_hydration.py using the existing inventory checker's --statements/--since interface, against actual pinned base. Verify moved calls retain metadata-only arguments, no user content/secrets/path leakage and no new persistent sink; only then regenerate the production diagnostic inventory. Record actual statements/dispositions rather than blindly --write.
3. Fix the six actual post-await DOM callback contracts in console_goal_setup_modal.py launch/refresh_tools and console_goal_status.py control_pressed/load_goal/resume_goal/show_iteration. Use existing guarded current-widget patterns; keep service operations/accounting physical ownership and confirmed launch authority unchanged. After awaits, a removed/dismissed subtree must not crash or be retargeted. Do not add blanket exceptions, census exemptions, arbitrary mount assertions or stale side effects to silence the check. If a lookup is truly safe, document its exact owner proof rather than waive by default.
4. Add one focused held-await mounted regression for setup and status demonstrating dismissal/removal during real delayed service/catalog work without NoMatches, late UI publication or incorrect control authority. Cover existing setup/control/navigation/CLI paths affected by changed callbacks. Preserve every current cap; no full sweep or cloud/user profile.
5. Run all actual derived-artifact preflight guards with pinned public Mermaid input network access, scoped Ruff/format/diff/baseline-size checks and the targeted changed-widget behavior lane. Commit only these task-owned product/tests/inventory/task/plan files with normal hooks and gc.auto=0. Root obtains both spec and quality scoped review of this new Task2 diff, then performs final hygiene/publication. Keep all seven tasks In Progress/AC5 unchecked until the required guards/review pass.

Report: this plan's SDD task-2-report.md; exact commands/results/head/source hashes, diagnostics review and held-await proof, commits and concerns. No GitHub writes or child agents. Root owns publication and merge.

## Reviewed current-dev closeout

Complete-head and combined P1 fix reviews passed; publication-contract Task2 passed after its single catalog-control fix round. Latest dev7b7193d647 rebase was clean, including canonical continuation-owner correction. Code head013412725b845c7b860a3e5b3ff04f444f4c661a passed25 affected continuation/startup cases; actual current preflight passed all guards. Previous source-bound and UI/resource receipts remain scoped to their exact bytes. Seven Goal tasks are locally qualified and Done; GitHub publication/required hosted CI/protected merge remain root-owned. No full sweep, external model quality or physical terminal UAT was performed.
