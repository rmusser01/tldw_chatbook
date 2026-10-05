# Task 6 — schema and encrypted-startup integration

## Frozen result

Source commit **`dace3505ab10ce1663a59f0e6edb12d42d0ad5e6`** descends from pinned dev **`057e62eaec5343ffce9445cfb32881e756dd7634`**. Clean input was `392655567f7754a394ea1e7aba8d75e81d2cadb2`; the single rebase of 31 commits rebase produced `07740acdbc4954c594fc6130bb20d0e3730a7207`. [Recovery ref/bundle](task-6-schema-final-checkpoint-recovery.json), [exact 31-commit mapping and final source hashes](task-6-schema-final-preservation.json).

The prior report is retained in [task-6-pre-schema-integration-report.md](task-6-pre-schema-integration-report.md). All 11,572 historical QA/review blobs remain exact. Of 92 incoming paths,81 remain byte-exact; the 11 overlaps/exceptions are enumerated in the manifest. Of 169 previously owned Python files,159 remain byte/strict-AST equal; every changed function or module is accounted for. [Function comparison](task-6-schema-function-accounting.json), [test reversal proofs](task-6-schema-test-preservation.json), [strict formatter proof](task-6-schema-format-proof.json).

Shared `origin/dev` advanced to `df2ba424de63576d36d9c3e38387c84285303f1c` during qualification. The last guard receipt records that observed ref; its tested HEAD still integrates057. No df2 source or qualification is claimed here. Root will scope that subsequent phase separately.

## What changed

- Preserved shipped notes75→76 method and SQL byte-for-byte. Native receipts move to76→77; its SQL changes only the filename and temporary-table suffix. Both run in the real migration chain. [Migration/diagnostic proof](task-6-schema-diagnostic-migration-proof.json).
- Preserved native76 machine receipt IDs, every checkpoint/message field and reopen behavior through exact full 533-entry primary or 584-entry shared Subscription catalogs, including the known dictionary variant. Column presence only selects the runtime gate; it never grants compatibility. The existing SiteConfigManager shared-file caller is separate from primary backup admission.
- Primary staged restore admits only exact native76 catalogs and stamps, then executes one installed metadata UPDATE under the narrowly approved owner/table/column/database/source gate. Final77 catalog/stamp/integrity checks precede commit; cancellation or failed validation rolls back. Ordinary75/shipped76 validate/export but retain unsupported staged DDL migration. Subscription outer2 retains exact embedded75/76/77 pairs.
- Fixed the retained75 Canvas validation omission after immutable BASE reproduced it. Only the two already declared full catalogs join the existing frozen pure-UDF list; their Canvas SQL is byte-identical. [Canvas proof](task-6-schema-canvas-exact-proof.json). Existing ADR126/158/208/219 govern this; the dated ADR219 clarification records the metadata-only exception.
- Kept upstream SQL-file execution, Evals sandbox, data-integrity changes, profile enrollment, encryption lifecycle, Settings/card/widgets, provider readiness and ciphertext refusal. Diagnostic inventory is exact upstream plus the two unchanged feature owners; no pin regeneration occurred.

## Qualification

Each receipt links exact argv, HEAD/source hashes, exit, log and JUnit. [Safe copied outputs and hashes](task-6-schema-safe-evidence-manifest.json) include only closed receipts/logs/XML and the CSS child pytest outputs, never private profiles or databases.

| Evidence | Result |
|---|---|
| [Immutable prior native76 stage](task-6-schema-safe-evidence/task6-schema-prior-native76-staged-profile.json) |1 passed; migrate=True supported at76 |
| [Constructor capture](task-6-schema-safe-evidence/task6-schema-constructor-capture.json) |4 passed; both exact catalogs and dictionary variants |
| [Runtime RED](task-6-schema-safe-evidence/task6-schema-legacy-runtime-red.json) |4 real receipt-copy CHECK failures;3 hybrid byte-comparison diagnostics |
| [Staged RED](task-6-schema-safe-evidence/task6-schema-staged-red.json) |9 failed,11 passed before gate |
| [First combined run, **failed despite green label**](task-6-schema-safe-evidence/task-6-schema-native-and-staged-green.json) |38 passed,4 failed:2 timestamp-converter comparisons,2 Canvas omissions |
| [Immutable75 Canvas RED](task-6-schema-safe-evidence/task-6-schema-prior75-canvas-red.json) |1 failed, confirms prior omission |
| [Final staged owner](task-6-schema-safe-evidence/task-6-schema-staged-final.json) |20 passed: receipts, full catalog/stamps, primary/Subscription separation, denied writes, rollback/cancellation |
| [Upstream migration/backup seams](task-6-schema-safe-evidence/task-6-schema-upstream-seams.json) |35 passed: actual75/76 chains, standalone files, column pin, primary capture, SiteConfigManager, Evals |
| [Encryption and native start](task-6-schema-safe-evidence/task-6-encrypted-entry-provider-seams.json) |14 passed: right/retry/reset entries, readiness/refusal, actual machine start with both receipts |
| [Migration mechanism guards](task-6-schema-safe-evidence/task-6-migration-mechanism-guards.json) |6 passed: entry guard, transaction mechanism, FK and rename scanner |
| [Final startup](task-6-schema-safe-evidence/task-6-final-startup-schema-encryption.json) |26 passed,3 headroom warnings; actual boot worker census included |
| [Diagnostics](task-6-schema-safe-evidence/task-6-schema-diagnostics.json) / [worker contract](task-6-schema-safe-evidence/task-6-worker-contract.json) |643 owners exact;324 DOM lookups/68 wait pushes, none new |
| [Static/source checks](task-6-schema-static.json) |Fatal Ruff, format, whitespace and128-file UI census passed |

The 22 passing v77 runtime-owner cases in the combined run carry through strict-AST formatting and the separately covered Canvas validator repair. Prior CI 34, BaseApp 43, Reader/provider/runtime and 49-case fork census evidence is retained through the relevant exact source mappings; these suites were not replayed.

No startup budget changed: imports681/686, UI-ready1033/1033, preimport557/557, payload415298/425347LOC, Library127527/135111LOC. The three warnings are retained verbatim. The earlier BaseApp FD-growth warning and stronger console_compaction_failure assertion remain as documented in the prior report; this phase does not claim to resolve that warning. Current-head external CI remains a root merge gate for the earlier native timeout.

## Diagnostics and self-review

The initial immutable probe exited 4 because its selected private-profile directory had not been created; the preserved rerun passed. Its unrelated pytest old-temp cleanup warnings remain in the log. Runtime hybrid byte comparisons measured SQLite WAL-close bookkeeping, so corrected controls compare full logical catalog/stamp/data. Staged receipt checks now use raw SQLite types on both sides, retaining all fields.

Automatic approval review rejected three proposed combined scripts before execution: unsafe sequential version replacements; unproven shared Subscription catalog ownership; and perceived historical-version assertion changes. Exact checked resolution, real constructor/caller proof, and explicit current-reopen scope resolved these concerns before narrower edits. No rejected command modified production source. The approved hunks are retained in [conflict evidence](task-6-schema-conflict-exact-edits.json).

Self-review checked exact upstream notes/Evals logic, all original receipt assertions, full-catalog refusal, version-pair matching, unchanged authority outside the metadata gate, final validation before commit and primary/shared-owner separation. Review surface: the two migration steps/runner, recovery catalogs and policies, restricted validator gate/Canvas filter, precise fixture/column deltas, and preserved upstream encrypted startup/provider seams. Original I1 review and PASS re-review remain immutable. Source and qualification are halted; root plan/Backlog edits remain unstaged. No push, merge or next-dev rebase occurred.
