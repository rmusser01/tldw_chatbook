# Cold Sensitive Input Config Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task. The parent owns every native launch and grants a separate production handoff after genuine RED. Do not edit the concurrent scoped RunLogWriter work.

**Goal:** Reduce repeated cold sensitive-input admission while keeping the natural warm-key route, every original getter and every source/privacy check.

**Architecture:** Only an actual `_raw_inputs` memo miss may group exact stock callbacks inside one supported fresh `config_participants.operation(config)`. Existing getters still execute and perform their native checks. The operation is finite; callback metadata supplies only refusal, and memo publication occurs after successful physical scope retirement and final source/key validation. Custom paths retain the preceding ungrouped contract.

**Tech Stack:** Python 3.12+, private SQLite, native Windows facade; original Linux/macOS config and filesystem contracts.

**Spec:** TASK-34404 AC10, existing Console performance spec, and ADR126 finite native/source custody.

ADR required: yes (clarification of the existing finite config interface).
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: Grouping independently guarded readers changes their surrounding native source lifetime, so callback provenance, errors and publication must remain explicit.

## Measured premise

Serial separate-process original-code controls `sensitive-ungrouped-1.json` and `sensitive-grouped-1.json` both PASS with nine stable source hashes, unchanged protected callables, natural cold publication/warm hit, physical seeded SQLite close and zero ordinary/pending/core/raw/retiring resources. Both produce normalized derived pathset `8874f6f6a2b16aa3b01be88aa9d7fd6f447397a00339295dc334027c70d3c1cc`.

Cold native opens: 7,176 ungrouped versus 2,627 under the original supported outer operation, 63.4% lower. All 20 original user-directory calls remain; fresh raw checks rise from 40 to 123. Warm native opens rise from 356 to 377, so always grouping is rejected. These are finite hypothesis receipts, not a full Send budget pass or untouched whole-HEAD baseline. Observed times of 2.814/1.275s are diagnostic only.

## Global constraints

- Leave the initial `_raw_inputs_key` and natural memo-hit return before any added scope; retain exact cache-object identity, generation, source, selected path, verified data directory, whole environment and cwd.
- Keep all public getters, validators, path resolution, original `_same_key` post-build comparison and additive `_debug` failure accounting. No speculative pure path reimplementation or global accessor rewrite.
- No native verdict, lease, operation or resolved sensitive context survives this synchronous build. The existing raw-input memo stores only its preceding unresolved tuple.
- Do not change the 15s Send, 1s UI, 200 main acquisition, 40,000 Windows-open or 16 POSIX-helper limits. The new cold leaf guard is ≤5,000 actual native opens, registered from the 7,176/2,627 causal control before production.
- Preserve the stdlib-only pinned worker bundle: remote/stub config without exact installed stock provenance takes the preceding route without importing Backup_Recovery.
- Ordinary accessor failures remain additive, never memoized; source/callback/pause drift on a qualified new scope refuses and never silently recaptures a different stock source or retries as fallback.

## Review focus

- Custom module functions, bound/borrowed receivers and equality-permissive callable proxies stay outside the added stock operation, with the original no-argument callback contract and calling thread.
- A callback swapped before the helper's first import cannot become an original anchor. Capture config originals at the defining module's completion; use `is` and concrete FunctionType/globals/module identity, never `unwrap` or equality as qualification.
- Mid-read accessor/helper/module/source drift must not invoke a replacement under the newly grouped stock operation or publish its result.
- A scope-exit/final-custody failure must not leave a newly published sensitive memo. Config state rollback does not own `_RAW_INPUTS_MEMO`.
- Preserve environment/cwd and accessor-failure non-memoization, independently fresh calls and actual actor checks; do not keep a settled sensitive-path/permission answer.

### Task 1: qualify native RED and original contracts

**Files:** Create `Tests/Backup_Recovery/test_sensitive_input_config_bundle.py`; retain original `Tests/test_user_data_dir_memo_perf07.py`, `Tests/Utils/test_sensitive_paths.py`, `Tests/Backup_Recovery/test_config_participant_lifetimes.py`, `Tests/Tools/test_remote_worker_bundle.py`.

**Interface:** No new production API. Each new case runs a fresh private subprocess with the same canonical config and actual registered/closed AgentRunsDB owner used by the qualified control. Trace only original Native entries and selected original Python calls/returns, no guarded callable replacement.

- [x] Run the new cold native node against frozen production. Genuine cold failure: the original getters independently reacquired config after literal protected-path checks; the separate qualified finite control measured 7,176 opens above the 5,000 leaf guard.
- [x] Run the original warm and custom/failure positives plus new native source/pause/helper/accessor/final-exit and environment/cwd controls. All 15 cases settled: four expected failures and 11 passes in 151.22s, with no fixture/setup failures. The four failures were independent cold getter scopes, memo publication after the actual mid-read callback replacement, memo publication after the actual private-helper replacement, and the last original whole-key comparison outside any active operation. The last case deliberately fails the active-issued-operation precondition on original ungrouped source; it is a final-custody revocation control, not a simulated native close failure.
- [x] Parent reviewed XML failure tails and granted production release. No production change preceded this native RED.

### Task 2: add only the cold finite operation

**Files:** Modify `Utils/sensitive_paths.py` cold `_raw_inputs` body/local private helper and DB-loop checkpoint; add only defining-module original metadata in `config.py` (and direct owning modules if source qualification requires them). Do not alter public accessor behavior or config/raw/storage guard code.

**Interfaces:**
- Consume original `config_participants.operation(config)` and `checked_config_identity(config, active)` at both admitted edges.
- Private `_sensitive_db_paths` may accept an optional internal refusal checkpoint defaulting to `None`; all preceding callers retain zero arguments. Invoke it before each existing dynamic accessor lookup/call, outside additive exception handling. It checks exact captured metadata only, never substitutes permission.
- A private stock qualification captures strong installed module/function/helper/accessor/source references, cache object/generation/source and exact synchronous actor. Preinstalled custom callbacks return no new scope. Only stock metadata is checked before/after original operation entry, before builders/accessors and after retirement; source metadata drift refuses before publication.

- [ ] Retain config originals at defining module completion before any lazy sensitive helper import. Include the exact direct config getter/helper bindings used by this body, not an unrelated global callback graph. Local builder originals are retained at the sensitive module's own completion. Real FunctionType/globals/module/installed-module identity and exact `is` comparisons reject bound/proxy/borrowed functions. The guarded `get_user_data_dir` wrapper's defining globals are `config_participants.__dict__`, while its published name/module metadata identifies config; retain that actual defining owner rather than incorrectly requiring every callback's globals to be the config dictionary. The original guard still checks its original body/source independently.
- [ ] Leave initial key/memo lookup unchanged. On cold stock miss enter the original operation, obtain fresh checked source generation/path, and build the original tuple through original callbacks/getters. Evaluate the original final whole-key/failure-count test while this operation is active, retain only its eligibility locally, and recheck actual source and metadata before scope exit.
- [ ] Publish a new `(key, inputs)` memo only after successful scope retirement and final metadata fences; no publication in a `finally` or before operation exit. Do not move the original final key comparison outside the operation: the native final-pause control requires the original comparison to run inside the actual issued lifetime, then prohibits publication after that final custody failure. Preserve an earlier memo owned by another completed caller; a failed build cannot clear a successor.
- [ ] Custom/remote paths execute the original body and call shapes outside the added scope. A qualified standard source that drifts never retries under custom fallback. Propagate original unexpected/native errors and preserve additive getter failures.

### Task 3: bounded GREEN and freeze

- [ ] Parent runs the new native module serially and the named original affected controls, including real POSIX memo mutation and stdlib worker bundle tests on native CI. No full sweep.
- [ ] Confirm cold opens ≤5,000 with all 20 getters, warm one getter/two raw checks, protected path coverage, custom contracts, source/pause/mid-read refusal, no memo after failed exit and zero finite native resource counters. Report any retained bootstrap startup owner separately until process exit.
- [ ] Ruff/format only changed leaf/test, source hashes and scope diff check. Report qualification limits and all skipped/error/failed nodes. Freeze source before parent integrated probe; record results in TASK34404/root plan/QA only after actual evidence.

## Concrete callback anchors and test qualification

The actual sensitive module body calls `_raw_inputs_key`, `_same_key`, `_sensitive_single_file_paths`, `_sensitive_skill_trust_dir`, `_sensitive_db_paths`, `_direct_child_rule_container_dirs` and additive `_debug`. Retain their definition-time original references and the original DB-name tuple. Its dynamic config edges are `get_user_data_dir`, `_get_effective_config_path`, and the 13 DB accessors: `get_chachanotes_db_path`, `get_prompts_db_path`, `get_media_db_path`, `get_library_collections_db_path`, `get_library_ingest_jobs_db_path`, `get_workspaces_db_path`, `get_subscriptions_db_path`, `get_notifications_db_path`, `get_research_db_path`, `get_writing_db_path`, `get_scheduled_tasks_db_path`, `get_evals_db_path` and `get_rag_indexing_db_path`.

The DB accessors dynamically call `_get_custom_database_path`, `get_cli_setting`, `load_cli_config_and_ensure_existence`, `lexical_path`, `validate_path_simple` and the profile path module's `custom_database_input`/`database_leaf`. Effective-path selection calls `_resolve_effective_config_path`. The Skills branch dynamically imports `local_skills_service.default_local_skills_store_dir` (which delegates to `Skills_Interop.recovery.default_local_skills_store_dir`) and `skill_trust_store.default_trust_store_dir`. The containers branch imports `simplified.config.default_chroma_persist_directory` and `config_profiles.default_rag_profiles_dir`; their config reader aliases must remain the captured original config functions. Chroma also calls its own `_cleaned_path_setting` and `_explicit_rag_setting`. Keep qualification bounded to these actual supported path-reader edges and their defining owners; preserve the existing guarded config/native protocol.

Draft tests are source-passive with respect to guarded functions. Intentional custom fixtures replace only the named unguarded accessor/private builder, and actual environment/cwd/pause changes occur at original code-object return barriers. The final-pause case requires an active original config operation, so original ungrouped production fails that requirement honestly. It proves final boundary revocation; its barrier must not be moved to a new helper or a replacement guard. Environment/cwd cases preserve the existing additive return, forbid a memo from the changed build, then require a fresh call to protect the newly selected endpoint. The first-import fixture must prove the sensitive module is genuinely absent before its intentional replacement; an already imported fixture is a setup error, never first-import RED evidence. Embedded script syntax and the outer module must both be checked before native launch.

## First native candidate import repair (2026-10-05)

The first fifteen-case GREEN launch settled before collection (exit 4, 4.687s),
with source hashes stable. Config's original `_resolve_effective_config_path` is
an `lru_cache(maxsize=16)` wrapper, not FunctionType, and cannot supply
`__globals__` to the new ordinary-reader tuple. This is a candidate import defect,
not test or performance GREEN. The authorized correction retains the original
concrete wrapper object/type plus its original `__wrapped__` function, code and
globals at config module completion. Qualification compares those exact retained
objects, preserving original wrapper invocation/cache semantics; it does not
unwrap an arbitrary current callback as authority. Unsupported/preinstalled
custom objects keep the ungrouped route; mid-build wrapper/body drift refuses.
Ordinary FunctionType code anchoring remains pending the separate genuine
two-route body-mutation RED. No getters, memo inputs, key barrier, guard, budget
or deadline changes are authorized by this import repair.

## Ordinary reader body anchor completion (2026-10-05)

The separate real native body controls completed with two genuine failures in
8.04s (13.125s driver), source stable: a pre-first-import mutation of the original
unguarded accessor's `__code__` reached literal 13-DB success but invoked its
custom body inside the newly added stock scope; a mutation at the original
single-file builder return accepted the changed body. Function object/globals
remained original in both cases. The fixture cloned the original keyword-only
defaults, restored original body/hooks, physically closed seeded SQLite, and
passed all native census/hash/guard cleanup checks. These are mechanism RED
receipts (`sensitive-body-native-red-1.log/xml/source.json`), not GREEN or an
integrated Send performance result.

Before production completion, retain ordinary FunctionType `__code__` objects at
the same seven existing module-completion tables and direct-owner records.
Compare code identity alongside the existing exact function/globals/module
checks. A changed body before first helper import uses the preceding ungrouped
custom route; drift after qualification refuses before invocation/publication.
No unwrap authority, callback/getter replacement, failure/key/publication
changes, new cached verdict, budget or deadline change is authorized.

## Cold tuple and pause publication refinement (2026-10-05)

ADR required: yes, clarification of existing finite source/publication custody.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: stock callback grouping must distinguish actual original reader names
and current publication availability from permission for admitted work to retire.

The 17-case native candidate completed 15 PASS / 2 FAIL in126.60s with exact
source unchanged. Both unchanged pause/final-pause controls reached their actual
original code-object barriers, then observed a published raw memo. Original raw
custody legitimately lets already admitted native reads finish after a local
pause; retirement alone therefore cannot qualify this leaf publication.

Two additional tuple controls produced genuine native RED in the parent's
four-node source-drift bundle (17.39s total, source unchanged). A preinstalled
tuple addition ran its new custom reader under the added stock scope; an
addition at the actual first-builder return was accepted. Original 13 reader
functions/codes, protected callbacks, actual native/issued barrier, restored
tuple, guards and zero final finite census were verified before final assertions.
The two separate scoped run-log cases in that bundle are setup gaps, not RED.

Before production: retain the original DB-name tuple at sensitive module
completion. Exact stock qualification captures that tuple and its exact original
name-to-reader membership. Preinstalled custom tuples retain the preceding
ungrouped zero-argument route; changed tuples after qualification refuse before
the added reader or memo publication. All original dynamic getters remain.

Capture only refusal metadata for the actual admitted config participant/state,
its original source reference/selection and coordinator lock. Two private config
participant helpers supply metadata capture and publication refusal; their
original functions/codes are retained at config module completion. After the
actual operation retires, recheck that same installed source participant and
current local/participant pause under the captured coordinator lock, and write
the existing unresolved memo in that same interval. Full source/path observation
stays outside this lock; the final in-lock check repeats actor/binding/cache/key
fields without filesystem metadata or a new reader/admission. No lease, native
verdict or operation is cached, and no pre-pause guard/retirement contract changes.

Parent serializes the unchanged original pause controls and new tuple controls
with the stock <=5000 actual-open guard, all20 getters, natural warm one-getter/
two-check route, custom/remote contracts, source/helper/body/environment/cwd/failure
checks and actual finite retirement. No tests/assertions, original15s Send/1s UI,
acquisition/native/worker budgets or deadlines change. Native GREEN remains pending.


Review follow-up before further implementation: preserve definition-time original guarded-reader body/code/globals and exact wrapper closure cells, not only the public guard wrapper. Native TDD will mutate only get_user_data_dir's existing captured body code before qualification and after the original first-builder return. The first must retain the original ungrouped custom route; the latter must refuse before executing the changed body or publishing the memo. All guards/source/thread/operation/physical retirement remain observed and restored. ADR required:no new ADR; existing ADR126 finite-source contract applies. Prior19GREEN is insufficient for this previously uncovered body boundary.
