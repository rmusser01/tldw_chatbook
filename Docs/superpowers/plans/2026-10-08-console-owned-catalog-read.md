# Finite Console catalog operation — candidate implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans for this bounded experiment. Root owns integration and every sequential native test; product changes follow the causal RED.

**Goal:** Remove redundant stock helper-entry proofs within one real catalog read, preserving fresh native admission, publication checks and migration effects.

**Architecture:** Use the existing `_checked_read` owner and existing native worker. Factor the original store load/read logic once; an eligible named catalog operation consumes it without entering the public method guards again. Initial configuration and postcommit composition still perform independent reads.

**Tech Stack:** Python 3.12+, existing MCP stores, raw/storage admission, pytest.

**Spec:** `backlog/decisions/126-complete-local-backup-and-recovery.md`; `backlog/decisions/225-console-send-preparation-and-io-ownership.md`.

**Status:** Rejected after final timing confirmation; archived bounded experiment on `81c70f4a10`, under TASK-34601. OPT-31 was rejected after matched timing and fully removed. Root reviewed this plan; causal count RED precedes product implementation. Adoption requires correctness and whole-Send improvement. All 79 ledger entries remain retained.

ADR required: no new ADR.
ADR path: ADR-126 and ADR-225 above.
Reason: consolidate one finite domain operation without changing storage policy, ownership, freshness across phases or public callback contracts.

## Global constraints

- No cache, coordinator, new owner registry, transaction or cross-await lease.
- No JSON-read elimination claim: each actual catalog operation still reads once.
- Keep public method signatures, custom callback ordering/affinity and missing/corrupt behavior.
- Root alone runs targeted tests and sequential native comparisons. No full sweep.
- No implementation unless a small body extraction and existing qualifier machinery suffice; reject a copied permission-eligibility framework for this marginal postcommit gain.

## Review focus

- Custom `_read_payload`: decline the owned body while retaining its current ordinary worker route; do not silently move this previously supported callback to the loop.
- Migration: retain original approval, `_write_payload`, temporary-file/replace gates, and incomplete-persistence marking through final validation/retirement failure.
- Late source or callback drift: refuse publication after actual original read; preserve physical custody until retirement.
- Missing/inactive/corrupt source: defaults, no unreviewed read, and original `LocalMCPStoreLoadError` stay distinct; catalog has no permission-store corrupt-backup policy.
- Separate phases: modifying the source after the maximum must still affect the subsequent composition read, subject to its existing narrowing maximum.

## Smallest proposed interfaces and body split

Only two product files: `tldw_chatbook/MCP/local_store.py` and `tldw_chatbook/MCP/console_snapshot.py`. No changes to generic raw/storage or lifecycle helpers.

In `local_store.py`:

- `_load_from_reader(source: LocalMCPStore, read_payload: Callable[[], Any]) -> LocalMCPStoreState`: move the current `load` body here, including its initial `readable`, schema checks and conditional migration. Public guarded `load()` delegates with `lambda: self._read_payload()`, so even property-backed reader lookup remains after the original readable approval gate. Do not duplicate normalization/migration code.
- `_catalog_bundle_from_state(state: LocalMCPStoreState) -> dict[str, Any]`: the existing three-field projection; both public and owned consumers use it.
- `_catalog_bundle_in_owned_scope(source: LocalMCPStore, operation: object) -> dict[str, Any]`: require the exact live ambient operation, installed source, selected path, route and actor using existing raw owner checks. Use the retained original `_read_payload.__wrapped__` body through `_load_from_reader`, then the projection above. This helper grants no admission. Leave guarded `_read_payload` and `_write_payload` intact, and keep `_write_payload` dynamically invoked by the shared load body.
- Record only the directly bypassed method/helper identities at their defining module. Reuse the existing callback-record/input-check format; no transitive callback graph.

In `console_snapshot.py`:

- `_owned_catalog_callbacks_current(source) -> bool`: consume those defining records with the existing `_controller_inputs_current` machinery and the existing raw guard records. Check exact descriptors before access, and wrapper/body/code/default/closure identity for skipped callbacks. Do not change `standard_local_catalog_sources`, since that would change custom `_read_payload` affinity.
- `_read_owned_catalog_bundle(source, require_current) -> tuple[dict[str, Any], tuple[Any, ...]]`: run the named body through unchanged `_checked_read(source, zero_argument_callback)`. The callback retains its issued raw state locally. Surround the entire checked read, including final proof and scope retirement, with the existing effect/uncertainty error-marking rule; do not alter generic `_checked_read` semantics. Recheck owned callback/source inputs before the body and before publication.
- Add `_CapturedSources.read_catalog_bundle()` for the initial maximum: eligible route uses that helper; otherwise preserve the original `_checked_read(store, catalog_reader(_captured_load=store_loader))` route.
- In `_CapturedLocalCatalog.read_bundle()`, eligible route uses that helper; otherwise retain the current direct `catalog_reader(_captured_load=load_reader)` call. Keep existing metadata receipts/current checks and loop governance/projection placement.

No additional captured DTO is needed: the two existing captured-source owners already retain the exact store and callbacks. No public API or custom-reader call shape changes.

## Causal regression and ownership

Tests-only lane owns new `Tests/Backup_Recovery/test_raw_owned_catalog_load_counts.py`. Reuse `installed_source_case` with `catalog` and `_assert_one_original_read_acquisition` from `test_raw_related_source_preparation.py`; do not alter that shared observer.

- [x] Add `test_owned_catalog_read_keeps_effect_checks_without_nested_scans[maximum|composition]`. On unchanged source, invoke the actual preceding read (`_checked_read` + captured bundle for maximum; existing `_CapturedLocalCatalog.read_bundle` for composition). On new source, maximum uses its new named consumer. Compare the complete three-key bundle before adapting to the existing empty-state helper oracle. Passively count original `_read_payload`/JSON body, `readable`, raw check callers and fresh selections/witnesses. Preserve member order, actual payload, one acquisition owner, all canonical/member leases and descriptor retirement before the count assertion.
- [x] Root runs these two RED nodes alone. Predicted scope/final checks: maximum **4+1 -> 1+1**; composition **3+0 -> 1+1**. Preserve one `_operation`, one `_file`, one `readable` and one payload read in each. Total native-open reduction is measured, not preassigned. If original counts differ, explain the actual path before implementation.
- [ ] Product lane implements only the interfaces above; do not use a callback swap to bypass public methods.

Behavior lane owns new `Tests/MCP/test_console_owned_catalog_load.py` using the existing installed catalog/worker fixtures:

- [ ] Full populated-bundle parity; missing defaults/no creation; malformed bytes unchanged; original legacy migration persists and reopens, including real write-approval refusal.
- [ ] Instance/class `_read_payload` override stays on its original worker with exact zero-argument invocation; custom outer load/catalog routes retain their existing loop fallback.
- [ ] Actual read-return barrier changes source/pauses admission: final publication refuses and issued resources retire. Cover migration followed by final refusal so persistence-error marking is not lost.
- [ ] Actual original held read plus repeated cancellation/close retires before completion; modify catalog between separately completed maximum/composition reads and prove the later read is fresh.

## Final targeted selection and adoption gate

- [ ] Root runs the two new files plus `Tests/MCP/test_external_catalog_worker_ownership.py` and the captured-reader/custom-load/source-drift/descriptor nodes in `test_console_snapshot_source_contracts.py`.
- [ ] Retain `test_local_store.py::test_legacy_schema_migrates_durably_and_reopens`, `::test_unknown_or_malformed_schema_is_not_rewritten`, `::test_migration_refuses_malformed_authoritative_sections`, and `::test_local_store_raises_on_corrupt_payload_instead_of_treating_it_as_empty_state`.
- [ ] Run scoped static checks and independent source review, then quiet matched whole-Send comparison. No acceptance from instrumented elapsed totals alone.

**Savings hypothesis:** one net full proof removed from postcommit catalog, three from the separately checked initial maximum. This removes neither catalog read nor either freshness boundary. In `detail-b1/summary.json`, postcommit catalog inclusive times are 0.300/0.106/0.216 seconds; the entire phase is not removable. Six unpaired starts and 92 ancestry overflows limit that diagnostic. If the implementation needs duplicated large eligibility machinery, or matched whole-Send timing does not improve, reject/defer the candidate and retain the original source.

Native RED receipt: `catalog-red-1`, two intended failures, 7.93 seconds; product source `81c70f4a10` unchanged. Original maximum/composition scope counts 4/3, final counts 1/0; one readable/payload/JSON read each and exact retirement verified before count assertion.

Correctness receipts: catalog-green-1 13 passes; catalog-controls-1 26 passes and six obsolete-barrier failures; catalog-controls-2 all six pass after relocating only the passive observer to the shared original load. Exact held native cancellation/close controls pass. Independent source/barrier review: task artifact catalog-candidate/review.md. Native opens include a +59 acquisition watch-state difference, so no speed claim. Quiet timing is still pending; candidate not adopted.


## Adoption decision (2026-10-08)

Not adopted. Initial quiet ABBA warm mean was 6.433 s baseline versus 5.323 s candidate, but the separate BA confirmation reversed it: candidate 4.273 s versus baseline 3.245 s. Sources stayed stable, every reply persisted and settled, and no native overlap was detected. Host/run variance prevents attributing the initial difference to this change. The 45 passing behavior/count controls remain useful experimental evidence, but do not justify the added eligibility code. Restored all three modified source/test files to 81c70f4a10 and removed the two experimental tests after archiving their exact contents in catalog-candidate/rejected-final. No catalog implementation is shipped.
