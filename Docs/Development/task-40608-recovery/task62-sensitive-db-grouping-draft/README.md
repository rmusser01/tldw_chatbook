# TASK-40608: finite original sensitive DB accessor grouping

Artifact only; root owns live integration, ADR amendment, and actual-user execution.

Measured preparation-a2 attributed 1.346 seconds to `_sensitive_db_paths`. Its 13 original database accessors repeatedly enter CONFIG ownership through `get_user_data_dir`. The candidate groups that existing collector under one existing `config_participants.operation`. It calls every original accessor and guarded directory body, including their original native checks. It adds no path projector, cache, authority, or retirement proof.

`product.patch` changes only `config.py` and `Utils/sensitive_paths.py` (+95/-10). Definition-time records use existing `capture_definitions`. Exact module aliases, original accessor/helper definitions, and the error callback must qualify. Custom or changed sources before entry keep the original ungrouped loop. The operation callable is captured before qualification; definitions are checked after admission, before each accessor lookup outside the additive exception catch, before the error callback, and after scope exit. A source change after admission refuses without retry.

Original per-accessor exceptions stay caught inside the collector. Genuine batch acquisition/retirement failures propagate; there is no empty-tuple substitute, partial result, or retry under uncertain ownership. `_raw_inputs` and its memo code are unchanged, so they cannot publish a newly collected value when this call raises. Existing behavior when called inside an already-active CONFIG owner is unchanged: the existing operation implementation controls nesting.

Pins:

- Product: `2fc734505c5be59b7275983d649c81da95f0cf6fb152e8d2bfbfe68ef322552c`
- Tests: `d7b2db91d738b83120df2dbca7814ae415b5e5fedbb14ba99797ac53a974421e`
- `pins.json` records individual artifacts and raw live preimages. The `.before` copies normalize CRLF to LF for review.

Focused test sequence for root:

1. Apply tests only. Run `test_original_db_getters_share_one_retired_owner` against unchanged product. It observes original code with local `sys.monitoring` events and should fail because the 13 guarded directory bodies retain 13 distinct original owners. No product callable is replaced for measurement.
2. Apply product. Run the seven new tests. The stock comparison requires identical DB results and all 13 original getter/directory body entries, one retired group owner, fewer original acquisitions/native opens than the same collector with grouping disabled, and continued real native opens.
3. The remaining cases cover custom fallback and additive exceptions, an optional accessor exception without new persistence-error poisoning, replacement of a later getter before it can receive the group owner, successful configured overrides and SQLite sidecars, acquisition failure without a partial custom-path denylist/memo, and physically retained raw resources preventing a returned DB tuple. The retirement test releases the retained resources through the original `_retire` before removing its registry entry using the existing scope protocol.
4. If targeted controls pass, root repeats the same actual accepted preparation process and compares the measured DB collector/native admission counts. No broader optimization is justified by this artifact alone.

Verification performed here: read-only source/history review and `git apply --check --whitespace=error` for both patches, both successful. No product imports, native launches, or tests were run by this subagent. No live product/test files were edited. Independent source reviewer `configuration_cost_audit` identified the module alias and admission-time drift checks incorporated above.

Historical test mismatch: `Tests/Performance/test_console_native_pause_probe.py` references `_stock_sensitive_config_bundle`, but that helper is absent from current product. Commit `077e5bfa09` introduced the entire 916-line probe file without a corresponding sensitive-path product change. Historical `f81f756ab2` added 603 sensitive-path lines with that bundle, coupled to other owners/projectors; it is not in the current sensitive-path lineage. This candidate does not resurrect that old bypass or its symbol. The stale test needs reconciliation separately if selected for execution.
