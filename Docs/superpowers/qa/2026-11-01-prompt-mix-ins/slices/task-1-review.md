### Spec Compliance

- ❌ Issues found: the local adapter does not always fence its read/merge/write operation. An explicit `expected_version=None` bypasses its inferred version check at `tldw_chatbook/Prompt_Management/prompt_scope_service.py:985`, permitting a metadata-only write to replace a concurrently edited canonical definition. This misses the brief's source-write/optimistic-check requirement.
- ✅ The strict literal model and early v3 dispatch preserve whitespace, reject boolean versions/extra fields/blank bodies/unknown placement, and project to empty System plus literal User without block compilation (`prompt_artifact_models.py:90`, `prompt_artifact_codec.py:172`, `prompt_artifact_codec.py:219`). The new local capability and canonical serialization branch are explicit (`prompt_source_capabilities.py:74`, `prompt_source_capabilities.py:252`).
- ✅ Count/list/browse/search push the type filter into the database, including the count query (`tldw_chatbook/DB/Prompts_DB.py:2970`, `:3066`, `:3552`). Exact UUID snapshot reads batch against the live bind limit within one transaction and reject duplicates, missing/deleted records and incompatible definitions (`tldw_chatbook/DB/Prompts_DB.py:2807`). Policy/capability checks and worker offloading exist at the scope entry point (`prompt_scope_service.py:1976`).
- ✅ The file-backed v5→v6 migration is registered and executed transactionally. It preserves the old table columns, IDs and sequence; reinstalls existing indexes/triggers; checks foreign keys; and restores connection-level FK enforcement (`tldw_chatbook/DB/Prompts_DB.py:808`, `:1098`; `tldw_chatbook/DB/migrations/prompts_literal_mixins.sql:1`). Preservation and injected-failure rollback tests compare prior records, keyword links, collection membership and ordered sync rows (`Tests/Prompts_DB/test_prompts_db_retained_history.py:1302`, `:1365`). Exact retained recovery catalogs and authorizer-scoped rebuild support are justified compatibility work, covered by new tests (`tldw_chatbook/DB/recovery_core_schema.py:999`; `Tests/Backup_Recovery/test_sqlite_validation.py:958`).
- ✅ Ordinary Prompt application uses positive eligibility and Recipe flows narrow block definitions (`tldw_chatbook/UI/Console_Modules/prompts.py:1573`, `:1833`; `tldw_chatbook/Widgets/Console/console_prompts_modal.py:947`). The unchanged legacy picker already has its own positive Prompt guard (`tldw_chatbook/Widgets/Console/console_prompt_picker_modal.py:312`, `:345`), with a new stale-System mix-in regression (`Tests/UI/test_console_prompt_picker.py:603`). No `prompt_block_compiler.py` change is necessary for this slice: its input remains a block definition; v3 dispatch bypasses it.
- ⚠️ Cannot verify from diff: source-authority identity checks around awaited runtime reads, captured sends/queues/retries, and portable imports/exports span later slices. This slice establishes only the local artifact/read contract. The reported RED/GREEN execution history and exact-BASE failure runs are supplied evidence, not independently rerun evidence.
- ✅ AC 3's open global migration-guard concern is inherited rather than an introduced defect: the diff adds `Prompts_DB.py` to runtime migration readers (`Tests/DB/test_migration_file_sources.py:49`), and the controller additionally confirmed the new SQL is recognized while BASE and HEAD have the same seven decorative-file failures. The report's exact-BASE comparisons also isolate the existing export/UI failures. Fixing those unrelated failures is not required by this slice. AC 3 can be closed once task-specific blockers and checks are resolved; this review does not require another full guard or suite rerun.

### Strengths

- The model hides the literal body in its representation, and decoding has no interpolation/rendering path (`prompt_artifact_models.py:93`; `prompt_artifact_codec.py:172`). The strict rejection matrix and literal HTML/CSS/SVG test exercise the central contract (`Tests/Prompt_Management/test_prompt_mixins.py:4`, `:35`).
- Migration tests exercise real old-schema stores, high-water sequence state, references and retained history; they do not merely assert the new version label (`Tests/Prompts_DB/test_prompts_db_retained_history.py:1302`). Recovery compatibility was extended rather than relabeling the old catalogs (`tldw_chatbook/DB/recovery_core_schema.py:999`).
- The implementation report identifies exact BASE failures and uses per-file lint diagnostic comparisons instead of hiding existing failures in an aggregate total (`.superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-1-report.md`). No new test-output warning is reported in its final targeted run.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

1. **Fence merged mix-in updates when `expected_version` is explicitly `None`.** `tldw_chatbook/Prompt_Management/prompt_scope_service.py:985` uses `setdefault`, which preserves an existing key with a `None` value. The method has already copied the current definition and compatibility text into `payload` at lines 953–984. If another writer changes the body after that read, `update_prompt_by_id(..., expected_version=None)` at line 995 accepts this stale merged payload and overwrites the new body even when the requested edit only changed metadata. Treat `None` as absent and use the version of the merged record, matching `tldw_chatbook/Prompt_Management/local_prompt_service.py:248`. Add a focused regression for an interleaved write through this direct adapter with explicit `None`, asserting a version conflict and preservation of the intervening definition.

2. **Share the duplicated local mix-in write preparation.** `tldw_chatbook/Prompt_Management/prompt_scope_service.py:937`–985 and `tldw_chatbook/Prompt_Management/local_prompt_service.py:201`–249 repeat the same candidate detection, conversion rejection, field whitelist, merge, canonical validation and projection selection. This is a substantial new logic copy at the same boundary, and finding 1 is already a concrete behavioral divergence between the copies. Extract the shared payload preparation into the existing validation owner and have both existing adapters use it; keep their distinct storage calls. Retain tests for both adapters and the `None` fence case. This is a quality blocker under the task-review rubric's explicit rule for duplicated logic, not a request for a new service layer.

#### Minor (Nice to Have)

- None found.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The local artifact contract, snapshot lookup and migration/recovery work are well covered by the supplied diff and targeted evidence. The merged-update race must be fixed, and its duplicated implementation should be consolidated so both local entry points retain the same validation and version semantics.

**Checks performed:** Read the supplied diff once in bounded chunks. Focused additional reads addressed named risks only: unchanged picker rejection; block compiler's block-only input; history classification and the save tail where hunks cut off mid-function; duplicated adapter version behavior; and generic import forwarding (portability belongs to slice 2). Read the linked ADR's compatibility requirements and the relevant testing/baseline lessons. No git-state mutation, product edit, subagent, full suite, or implementer-suite rerun was performed.

**Verification limit:** Two small focused probes were blocked before their assertions by `Recovery required: recovery_scope_uncertain` during package import. Neither proves the race dynamically. The finding follows directly from the `setdefault`/merge/write path and needs a targeted regression under the repository's pytest isolation fixtures. The reported 472-pass run and controller-supplied baseline attribution remain the available execution evidence.
