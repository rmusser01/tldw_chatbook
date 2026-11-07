# TASK-38538 / Slice 3 implementation report

Status: DONE_WITH_CONCERNS; task remains In Progress for independent review. BASE: `1e5b0c74a7b33000f09c96af635f086d9f3d1ea7`. Commit: `7528f8e3c4df3cf1e2b9f34036b393fada915283` (`feat: retain chat mix-in preferences and attempt snapshots`).

## Delivered and interfaces

- Strict frozen runtime values in `Chat/console_prompt_mixins.py`; content/name/raw fields are repr-hidden. Preferences use duplicate-key-rejecting `strict_json_loads`, exact types, canonical UUIDs, duplicate-source refusal, existing Prompt limits and independent optimistic revisions. Unsupported/corrupt preferences are preserved and cannot be overwritten by ordinary save.
- Caller-cursor repository in `DB/console_prompt_mixins.py`; no independent transaction, commit, rollback, source permission or DB. Four conversation-owned tables retain exact immutable definitions, ordered attempts (including explicit empty), retry links and ownership-checked anchors. Digest lookup compares full values before reuse. Deleted messages null anchors; conversation purge cascades operational content.
- `MixinAttemptContribution(batch)` in the new service module is frozen and implements `write`/`write_exact`. The existing frozen-contribution fingerprint path includes every determining field. `ConsoleTransactionWriter.append_mixin_attempt` is a single narrow operation with the existing lifetime-revocation check; no cursor/general SQL exposure.
- Controller's final ruling preserves fresh acceptance's existing native==durable invariant: exact contributions receive only the identity map; old contributions keep role aliases. The earlier optional acceptance-field proposal was explicitly superseded and was not implemented. Promotion uses the existing real native→durable map; native IDs `user`/`assistant` cannot accidentally select role aliases.
- Assistant metadata has body-free `mixin_attempt_id`/`mixin_snapshot_schema_version=1`. A narrow SQL trigger retains those facts through ordinary/terminal metadata rewrites. Active checkpoints have a nullable schema marker. Missing marked records are unavailable; zero items is recorded-empty; unmarked absent historical attempts are legacy. Legacy `MixinReplayRead` has `snapshot=None`; the runtime can explicitly supply empty for that status. Fork hydration retains markers without enabling selections.
- Existing fork bundle copies referenced snapshots once into destination ownership and leaves preferences absent. Promotion writes temporary contributions once/idempotently. Voice promotion's existing locator census classifies both attempt anchor columns.
- Slice 4 still owns `ConsolePromptMixinService`, composition/reconciliation, capture, memory custody and send/queue orchestration. No placeholders, UI, provider request copies, raw sidecars or diagnostic body fields were added. The pure `console_chat_fork.py` snapshot shape needs no change for this durable storage copy: the transaction uses its existing persisted-ID mapping.

## Schema ownership and approved lifecycle correction

The integrated branch is v81; the descriptive `chachanotes_prompt_mixins.sql` is registered as v81→v82. Current/frozen recovery catalogs, restricted migration allowlists, checkpoint Canvas-validation catalogs and current schema expectations agree. All four v81 historical lineages remain unchanged and migrate through installed SQL. Existing whole-ChaChaNotes backup cohort/inventory/group/lifetime owners already cover the tables; they were inspected and do not require new registrations.

Controller authorized v82 despite unrelated uncommitted portrait migration reservations in the parent checkout. Those files were neither imported nor edited. **Before integrating with newer dev, rebase/renumber and requalify the migration and recovery catalogs.** No future combined schema was invented.

Exact BASE reproduction found `conversations_ad` deleting an already soft-deleted FTS posting twice. Two same-title conversations, first soft-delete→native purge, then second soft-delete raised `sqlite3.DatabaseError: database disk image is malformed`. The approved AC/plan amendment preceded the fix. Only new v82 SQL and its current catalog replace this trigger with an `old.deleted = 0` guard; fresh bootstrap always applies v82. Historical base/v81 SQL and `recovery_operations.py` donor semantics remain historical. Combined native lifecycle now checks surviving search and FTS integrity. The evidenced incident was added to lessons-testing-evidence.md.

## Commands and evidence

All commands ran from the assigned worktree unless marked BASE. Shared interpreter:

```sh
TASK_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

Every run used private platform-temp `--basetemp="$(mktemp -d /private/tmp/<prefix>.XXXXXX)"` and `-p no:cacheprovider`; the prefixes below identify each full output file under [task-3-evidence](task-3-evidence/). Bootstrap-profile fixtures preserve actual product admission. No full suite ran. Counts below overlap and are not aggregated as unique tests.

Initial RED:
```sh
$TASK_PYTHON -m pytest Tests/Chat/test_console_prompt_mixin_values.py Tests/DB/test_console_prompt_mixins.py -q -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-red.XXXXXX)"
```
2 collection errors: missing `console_prompt_mixins` owner. [Full output](task-3-evidence/mixins3-red.log).

Contribution/fork RED:
```sh
$TASK_PYTHON -m pytest Tests/DB/test_console_prompt_mixins.py -k 'writer or acceptance or identity or promotion or integrated' Tests/Backup_Recovery/test_prompt_mixin_restore.py -q -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-seam-red.XXXXXX)"
```
5 failed, 8 deselected: missing contribution owner and fork receipts not copied. The `-k` filter excluded recovery tests; recovery RED was separately observed as unsupported current schema and v81 not migrating, then corrected. [Contribution RED output](task-3-evidence/mixins3-seam-red.log), [recovery RED output](task-3-evidence/mixins3-seam-green.log).

Each of the following single-node RED runs used `-q -p no:cacheprovider` and its named private basetemp prefix:
- `Tests/DB/test_console_prompt_mixins.py::test_checkpoint_marker_survives_message_projection_loss` (`mixins3-marker-red`): missing checkpoint field, 1 failed. [Output](task-3-evidence/mixins3-marker-red.log).
- `Tests/DB/test_console_prompt_mixins.py::test_terminal_metadata_rewrite_retains_marker_and_fork_unavailability` (`mixins3-rewrite-red`): legacy instead of unavailable after rewrite, 1 failed. [Output](task-3-evidence/mixins3-rewrite-red.log).
- `Tests/DB/test_console_prompt_mixins.py::test_foreign_anchors_refused_and_fork_survives_source_purge` (`mixins3-fts-red`, additionally `--tb=short`): real FTS failure, 1 failed. [Output](task-3-evidence/mixins3-fts-red.log).
- `Tests/DB/test_console_prompt_mixins.py::test_legacy_fork_ignores_unrelated_malformed_metadata` (`mixins3-legacy-red`, additionally `--tb=short`): newly strict copy refused unrelated legacy metadata, 1 failed. The correction keeps legacy forks unchanged. [Output](task-3-evidence/mixins3-legacy-red.log).

Exact-BASE qualification:
```sh
mkdir -p /private/tmp/mixins3-base-repro
git archive 1e5b0c74a7b33000f09c96af635f086d9f3d1ea7 tldw_chatbook Tests pyproject.toml packages | tar -x -C /private/tmp/mixins3-base-repro
```
Copied only [baseline_fts_probe.py](task-3-evidence/baseline_fts_probe.py) to `Tests/DB/test_mixin_lifecycle_baseline_probe.py` in that archive, then from there:
```sh
$TASK_PYTHON -m pytest Tests/DB/test_mixin_lifecycle_baseline_probe.py -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-base-fts.XXXXXX)"
```
1 failed with the same disk-image-malformed error at the second owner's soft-delete, schema v81 asserted. [Full output](task-3-evidence/mixins3-base-fts.log).

GREEN lifecycle/recovery correction:
```sh
$TASK_PYTHON -m pytest Tests/DB/test_console_prompt_mixins.py::test_foreign_anchors_refused_and_fork_survives_source_purge Tests/Backup_Recovery/test_prompt_mixin_restore.py -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-fts-green.XXXXXX)"
```
3 passed. [Full output](task-3-evidence/mixins3-fts-green.log).

Latest GREEN new files (after self-review and formatting):
```sh
$TASK_PYTHON -m pytest Tests/Chat/test_console_prompt_mixin_values.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_prompt_mixin_restore.py -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-new-final.XXXXXX)"
```
**29 passed in 7.60s.** Covers strict codecs/repr, preference CAS/rollback/future preservation, explicit empty/lost records, duplicate attempt refusal, immutable rows, actual exact acceptance and rollback, contribution revocation/fingerprints, exact promotion and missing-record marker, integrated fork/hydration, source/final-owner purge, message deletion, retry reuse/FK isolation, terminal settlement then file reopen with missing-record refusal, native maintenance capture and disposable restore. [Full output](task-3-evidence/mixins3-new-final.log).

GREEN schema/migration matrix:
```sh
$TASK_PYTHON -m pytest Tests/Chat/test_console_prompt_mixin_values.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_prompt_mixin_restore.py Tests/Backup_Recovery/test_core_owners.py::test_schema_policy_matches_installed_store Tests/DB/test_response_rules_migration.py Tests/Backup_Recovery/test_response_rules_restore.py::test_restricted_historical_restore_migrates_only_disposable_copy Tests/Backup_Recovery/test_response_rules_restore.py::test_every_accepted_v80_catalog_migrates_to_its_exact_current_variant -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-final.XXXXXX)"
```
**51 passed in 22.25s**, before adding the final legacy-metadata compatibility node. Retains all four accepted historical lineages, v76–80 migrations, v81 candidate migration and real current constructor equality. [Full output](task-3-evidence/mixins3-final.log).

Other targeted coverage:
```sh
$TASK_PYTHON -m pytest Tests/Chat/test_console_chat_fork_persistence.py Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_store_atomic_promotion.py Tests/Chat/test_console_voice_promotion.py Tests/DB/test_migration_file_sources.py Tests/Packaging/test_profile_core_packaging.py -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-regressions.XXXXXX)"
```
207 passed, 2 failed, one existing aggregate FD-growth warning. Exact BASE re-run of the two nodes also failed:
```sh
$TASK_PYTHON -m pytest Tests/Chat/test_console_chat_store_atomic_promotion.py::test_promotion_persists_sparse_context_policy_inside_the_bundle Tests/DB/test_migration_file_sources.py::TestNoDecorativeMigrationFiles::test_every_migration_file_is_load_bearing -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-baseline-regressions.XXXXXX)"
```
The sparse-policy node fails `postcommit context-policy write`; the migration guard flags seven unchanged Dreams/Guardian/Collections scripts. Left unchanged. [Current output](task-3-evidence/mixins3-regressions.log), [BASE output](task-3-evidence/mixins3-baseline-regressions.log).

```sh
$TASK_PYTHON -m pytest Tests/DB/test_console_prompt_mixins.py::test_terminal_settlement_reopen_and_missing_record_refusal Tests/DB/test_console_prompt_mixins.py::test_native_promotion_writes_temporary_attempt_once Tests/Chat/test_console_voice_pair_persistence.py Tests/Backup_Recovery/test_core_owners.py::test_schema_policy_matches_installed_store Tests/Backup_Recovery/test_core_owners.py::test_factory_discovery_and_schema_policy_are_runtime_independent Tests/Packaging/test_installed_distribution.py::test_migration_expectations_are_derived_not_enumerated Tests/Packaging/test_installed_distribution.py::test_built_artifact_ships_the_migrations_its_own_code_opens -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-integration.XXXXXX)"
```
106 passed, one expected current-version-list assertion (v81 omitted) failed; the expectation and catalog-step assertions were updated and superseded by the focused GREEN 51 run. Actual wheel and sdist runtime migration inclusion passed. [Full output](task-3-evidence/mixins3-integration.log).

## Static checks and self-review

New six Python files pass `ruff check --no-cache` and `ruff format --no-cache --check`. Modified existing Python ranges were formatted only; diagnostic code/message multiplicities compared with exact BASE produce **zero introduced diagnostics (753 current / 753 BASE)**. [Comparison](task-3-evidence/ruff-differential.json). `git diff --check` passed.

Self-review checked every new public signature against the ledger, caller-owned transaction boundaries, exact identity propagation and revoked capability, native acceptance fingerprint behavior, snapshot reuse/full equality, item count/order, same-conversation FKs, nullable deleted anchors, retry predecessor retention, future preference preservation, body/name/raw representations, marker persistence through terminal metadata and deleted checkpoint, legacy malformed-metadata fork compatibility, source/fork deletion ownership, no selection activation, runtime-independent exact recovery policies, all retained historical lineages, normal maintenance capture, and packaged migration inclusion. Files are enumerated in [task-3-files.txt](task-3-files.txt).

Skipped: full suite, live UI/provider/server work and later capture/composition orchestration; no push/merge. Risks: known baseline failures above; migration-number rebase/requalification before integration. The FTS fix prevents future double deletion; it does not claim repair of a previously damaged index.

Commit completed successfully. Git repeated its existing auto-gc/unreachable-object warning; no shared-repository cleanup was attempted. Tracked worktree status is clean.


## Review round 1 — I1/I2 corrections

FIX_BASE: `7528f8e3c4df3cf1e2b9f34036b393fada915283`. Both Important findings reproduced before product edits. I1: the strict fork codec now accepts either the original namespace alone or exactly that namespace plus the valid body-free marker pair. Partial, mistyped, future-version and unknown companion fields remain rejected. Native fork persistence/hydration retains stopped status, attachment label/display name, trace alias and snapshot markers, with absent fork preferences; the codec also covers failed status. No execution authority is restored. I2: the v82 trigger uses an object-only CASE input to `json_set`, preserving object fields and normalizing valid nonobjects/SQL NULL before restoring the original marker pair. Only installed new migration/current v82 catalog changed; historical catalogs remain intact. The real acceptance→settlement→metadata update→reopen→lost operational record journey now covers object, SQL NULL, JSON null, array, string, number and boolean; all require unavailable.

All commands below run from the isolated worktree with `TASK_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Existing bootstrap-profile admission seam remains intact.

RED:
```sh
$TASK_PYTHON -m pytest Tests/DB/test_console_prompt_mixins.py -k 'integrated_fork or terminal_settlement or fork_codec' -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-fix-red.XXXXXX)"
```
**7 failed, 9 passed, 14 deselected in 4.64s**: fork hydration reported complete instead of stopped; valid paired markers failed strict codec parsing; five valid nonobject JSON rewrites caused legacy instead of unavailable. [Full RED output](task-3-evidence/mixins3-fix-red.log).

GREEN, direct storage/fork regression coverage (the codec changed, so its owning fork test module and native fork persistence module were included; the prior unrelated 207-test matrix was not repeated):
```sh
$TASK_PYTHON -m pytest Tests/DB/test_console_prompt_mixins.py Tests/Chat/test_console_chat_fork_persistence.py Tests/Chat/test_console_chat_fork.py -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-fix-green.XXXXXX)"
```
**336 passed, 1 warning in 64.37s**. Warning is session FD growth by 330 (14→344, limit 200), not claimed clean or newly attributed. [Full GREEN output](task-3-evidence/mixins3-fix-green.log).

Installed recovery/catalog equality:
```sh
$TASK_PYTHON -m pytest Tests/Backup_Recovery/test_prompt_mixin_restore.py Tests/Backup_Recovery/test_core_owners.py::test_schema_policy_matches_installed_store -q --tb=short -p no:cacheprovider --basetemp="$(mktemp -d /private/tmp/mixins3-fix-schema.XXXXXX)"
```
**7 passed in 4.36s**. [Full recovery output](task-3-evidence/mixins3-fix-schema.log). No aggregate unique count inferred from overlapping runs.

Static:
```sh
$TASK_PYTHON -m ruff check --no-cache Tests/DB/test_console_prompt_mixins.py
$TASK_PYTHON -m ruff format --no-cache --check Tests/DB/test_console_prompt_mixins.py
$TASK_PYTHON -m ruff format --no-cache --check --range 134-153 tldw_chatbook/Chat/console_chat_fork.py
git diff --check
```
All checks passed; each formatter reported 1 file already formatted; diff whitespace check had no output. Initial new-test import ordering was corrected with scoped Ruff before final checks. Existing product-file diagnostics compared to exact FIX_BASE with `ruff check --no-cache --config pyproject.toml --output-format json`: fork module 2→2, recovery catalog 1→1, zero introduced code/message multiplicities. [Differential output](task-3-evidence/mixins3-fix-ruff.json). No unrelated static cleanup.

Self-review: checked strict allowed-key/type predicates, original marker preservation with recursive-trigger termination, native settled checkpoint removal and lost-record distinction, no authority/preferences activation, installed/current catalog agreement and unchanged historical snapshots. Architecture doc and task notes updated; ADR-286 still governs, no new interface/ADR ruling needed. Task remains In Progress. Existing schema-number integration warning and earlier BASE-qualified failures remain as recorded. No full suite, UI/provider checks, push/merge or FD cleanup.

Review fix commit: `e4560ff18054f78733a3e739fef0e6901a13dfa1` (`fix: preserve mix-in markers across fork and metadata rewrites`). Git emitted the existing unreachable-loose-object/disabled-auto-GC warning; no cleanup performed.
