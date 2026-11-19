# K5 fix round2: query exclusion predicate normalization

FIX_BASE: `3820fe11572696f62d27291971f5c8d1dba0faec`.
Commit: `5d927a98ffc695c281465b09a8020d1db196490a`.
Only owned paths: `tldw_chatbook/DB/Workspace_Project_Index_DB.py` and `Tests/DB/test_workspace_project_index_db.py`. Root metadata/QA remain unstaged. Final source-qualified canonical performance PASS; the broader admitted common-query target remains an accepted unmet deviation. Exact evidence follows.

## Change and equivalence

Only lexical_candidates changes in product code: normalize this query's currently admitted exclusion values once with the exact existing replace-backslash/strip-slash/casefold operation, then group immutable exact-file/subtree paths, names, direct parents and subtree prefix tuple. Empty subtree denies all. Each SQL callback casefolds the candidate once and applies these same predicates before unchanged processing prefixes/suffixes, ranking and LIMIT. No cross-call cache, new API/schema, SQL/order change, guard/admission/activation, native owner, current/run exclusion-floor or live validation change. `_path_allowed` remains byte-for-byte unchanged and supplies the behavioral oracle. `k5-fix2-source-scope.json` proves AST equality of all product source outside lexical_candidates.

The new real-store test uses18 file paths and15 successive floors, each queried at limits1 and100. It covers all four rules separately/mixed, Unicode casefold, slash/backslash normalization, empty/root subtree, subtree peers, direct children versus descendants, exact/name paths and a final empty floor after restrictive calls. Expected candidate order is unchanged unfiltered FTS order filtered by `_path_allowed`, so top1 must be selected after exclusion. This checks behavior through canonical publication/query APIs, not implementation counters or raw SQL.

## Evidence

Existing intended performance RED: source3820 final-idle-latency.json nested/common canonical p95577.3054ms exceeds500ms. All six broader admitted cells also failed; original receipts remain unchanged. New semantic tests need not fail for a behavior-preserving optimization.

Before product edit:

```text
.venv/Scripts/python.exe -m pytest Tests/DB/test_workspace_project_index_db.py::test_lexical_exclusion_predicates_match_live_oracle_before_top_k -q --tb=short
```

`k5-fix2-semantic-baseline.txt`: **1passed1warning7.70s**, call6.53s on original3820 product.

After product edit:

```text
.venv/Scripts/python.exe -m pytest Tests/DB/test_workspace_project_index_db.py::test_lexical_exclusion_predicates_match_live_oracle_before_top_k Tests/DB/test_workspace_project_index_db.py::test_prepare_required_and_exclusion_floor_before_top_k Tests/DB/test_workspace_project_index_db.py::test_lexical_policy_filter_matches_path_suffix_runtime_oracle Tests/DB/test_workspace_project_index_db.py::test_narrowed_processing_policy_filters_old_candidates Tests/Agents/test_project_index_tool.py::test_old_run_floor_filters_before_top_k_after_removal_and_refresh Tests/Agents/test_project_index_tool.py::test_filename_only_empty_and_whitespace_use_live_capture -q --tb=short
```

`k5-fix2-scoped-green.txt`: **6passed1warning75.58s**. New equivalence6.39s; before-top-k1.41s; policy/suffix oracle11.15s; narrowed policy1.35s; actual requesting-floor removal/Refresh28.10s; actual filename/empty/whitespace native Capture25.43s. Private TEMP/TMP/TMPDIR use taskdir/native-temp. No broad suite/corpus rebuild.

External installed Ruff `check --no-cache` on the two owned files passes (`k5-fix2-ruff.txt`); `format --check --no-cache` reports2already formatted (`k5-fix2-format.txt`); `git diff --check` clean (`k5-fix2-diff-check.txt`). Self-review verified identical normalization and candidate separator behavior, empty root denial, every rule grouping, unchanged filtering-before-ranking and no mutation of live `_path_allowed`. Exact product diff30lines, test addition114lines.

## Final timing setup (historical launch checkpoint)

Active exec session26210 runs `python -m pytest <taskdir>/test_k5_fix2_idle_latency.py -q --tb=short --timeout=900` with TLDW_TEST_CONFIG_ROOT selecting the SAME retained stable-profile before imports. No tests/builds/renderers compete; light read-only review/coordination and uncontrolled OS background may overlap. `fix2-idle-source.json` records final commit/per-file/probe hashes/start/load. No changes after launch.

Both actual completed20k nested/wide caches:50measured after5warmups for common(top100), rare(1), miss(0), each through the same decorated lexical method with current real registry/root/chain/status/floor execution. Initial guard now precedes fence to match production; pure corpus-count assertions are outside timer. Canonical decorated lexical500ms is the original gate. Expanded guard+fence+status+floor+lexical500ms remains an explicit accepted unmet deviation, not a passing metric or omitted cost. All original failing source/raw receipts stay retained. The final raw probe reports both boundaries.

Separate20public marker calls/cache retain native/pipes/cache/registry retirement. Public invocation has no test ownership wrapper; lookup and candidate owners retire first. Whole-tool metric is instrumented end-to-end and includes post-Capture psutil retirement proof/bookkeeping; Capture's own clock ends before that test proof while including production child/pipes retirement. No sufficiency/pass claim until raw final result. Independent scoped rereview and final whole-branch review remain required.


## Final measured result

The same-cache final query command completed **1passed1warning256.56s**, call255.50s (`k5-fix2-idle-latency.txt`). Source remains `5d927a98ffc695c281465b09a8020d1db196490a`, with no changes after launch. `fix2-idle-source.json` records exact source/probe hashes and load. `fix2-idle-latency.json` retains300 measured samples plus explicit5warmups/workload (330candidate calls total), counts, authority/scope and all public/native proofs. No corpus/cache rebuild or copy.

| Cache/workload | Canonical p50 ms | Canonical p95 ms | Expanded admitted p50 ms | Expanded admitted p95 ms |
|---|---:|---:|---:|---:|
|nested/common|208.1393|222.3257|495.2744|534.4660|
|nested/rare|157.6595|172.1625|439.7766|469.0829|
|nested/miss|156.7933|181.9234|441.0551|493.7689|
|wide/common|212.3573|232.7950|502.7325|561.9087|
|wide/rare|157.2588|177.2331|444.5825|479.2464|
|wide/miss|157.7064|168.1995|443.0204|467.3531|

All six original canonical decorated lexical p95<=500ms gates PASS. Both expanded common cells still FAIL500ms (nested534.4660,wide561.9087); the other four expanded cells are below500ms in this final run. The accepted stronger-target deviation remains visible, with all earlier six-cell misses and nested/common canonical577.3054ms RED untouched. This is not a claim that expanded admission meets its stronger target, nor an exact causal speedup ratio across runs with different uncontrolled OS load.

Public20/cache instrumented end-to-end p50/p95: nested1994.4823/2703.4715ms; wide1909.2825/1957.3418ms. Native Capture p50/p95: nested240.5639/281.9572ms; wide229.7040/257.1561ms. Public lexical-stage p50/p95: nested162.5927/214.5572ms; wide156.0074/168.9987ms. Whole-tool values include probe psutil retirement proof/bookkeeping after Capture; Capture clock excludes that test proof but includes production worker/pipes retirement.

All40public calls returned the fresh marker, began with empty current-thread registry and retired it before return, without an outer test ownership wrapper. Fixture lookup/candidate scopes retired before public calls. All42native Capture children (including two authority warmups) are gone/retired/pipes closed; service closes with0searches. Exact expected100/1/0 candidate counts hold for all samples. `fix2-idle-process-retirement.json` separately records no remaining matching idle/scanner Python processes after completion. Correctness/scoped source-review results pass; root/final whole-branch review still owns acceptance of the expanded-target deviation and final task status.
