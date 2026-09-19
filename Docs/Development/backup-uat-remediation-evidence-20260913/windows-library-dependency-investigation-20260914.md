# Windows Library dependency-scope refusal

Read-only diagnosis of exact d3082ef98211e3ad42273ac2d2dcbbf43b6d93cc evidence. No product or tracked test edits, no remote access, no full-suite rerun.

## What the retained artifact proves

The Library mounted-capture-review.log has two inventories, no truncation, and exactly one changed item/field: db.chachanotes.primary.dependencies. Core status remains included. The three target records notes.file_notes, persona.assets, and persona.visual_identity did **not** become unused: their statuses were already unused in preview and remain unused in capture. The second observation newly lists core dependency edges to those unused records. This corrects the wording of the initial support review.

The capture refuses at capture_service.py185 before maintenance because _review_digest includes inventory.scope_digest. inventory.classify_entries correctly treats an included owner's dependencies on unused records as dependency_unavailable (around261), and correctly includes dependencies in the structural scope digest (around302). Neither comparison should ignore those edges: a real new reference to an unavailable owner must continue to refuse.

## Proven producer mechanism

DB/recovery_core.py, _CoreAdapter.discover (around40): core optional dependency pruning occurs only if validate(path) returns no issue. If validation fails, the adapter keeps status included and the entire default optional-owner list. It thereby represents unknown optional references as existing required dependencies.

_PreviewScope.sqlite_target in storage_admission.py1441–1539 caches one successful stable private main/WAL copy per source. It verifies source metadata before/after copying. If ordinary live writes change that source during the initial copy, the strict check raises preview_sqlite_changed; the failed private copy is removed and not cached. _CoreAdapter.validate converts this to core_validation_unavailable. Later adapters may then successfully copy and validate the source within the same discovery context. Notes and Persona use their own checked reads and can correctly return unused while the initial core record retains guessed optional edges.

This is not a legitimate newly selected scope and is not caused by scope comparison treating ordinary unused records as changed. It is a conservative but misleading inconsistent producer result after an unsuccessful first observation.

## Native disposable reproduction

/private/tmp/test_core_optional_dependency_review.py creates a real private, bound config and actual CharactersRAGDB. It keeps the native validator and SQL unchanged. A real ordinary note write on a separate thread is released between the private-copy before/after state observations. No current UAT profile or installed package is touched.

Initial probe: 1 passed in 2.65s, /private/tmp/uat-core-optional-dependency-native.log. Metadata probe: 1 passed in 2.05s, /private/tmp/uat-core-optional-dependency-native-metadata.log.

Exact output: /private/tmp/uat-core-optional-dependency-native-metadata/test_native_commit_during_firs0/home/dependency-review.json.log. First validation returns core_validation_unavailable; next five validations return no issues. Core status remains included, only dependencies change, and all three relevant target owners remain unused. A later core observation restores exactly the original dependency tuple. The empty-core fixture also adds dictionary/builtin optional edges; the Windows fixture already used those groups, so only its three unused groups were reported unavailable.

The original Windows validation issue itself was not recorded, so this establishes a native mechanism matching all retained structural evidence, not conclusive attribution of that particular Windows occurrence. A narrow future observer can record only the fixed core validation issue tuple plus whether a stable snapshot already existed on that call, without SQL, paths, data, or new timing gates.

## Recommended smallest correction, for review before implementation

When the primary core's validation prevents optional dependency discovery, emit an existing blocking `unavailable` StorageItem for that same exact owner/logical ID/path with only the known config dependency, rather than an `included` item with guessed optional dependencies. Keep successful validation and all native SQL pruning unchanged. Keep _review_digest, dependency_unavailable, native copy before/after checks, schema validation, source authority, and all publication gates unchanged. Existing Notes._FileNotesAdapter already represents failed core validation as unavailable; the subscriptions adapter similarly blocks failed validation before deriving references.

This needs only the existing _CoreAdapter.discover primary branch and focused tests; it does not require snapshot caching changes, a new error protocol, or ignoring unused dependencies. Do not cache a failed validation as success or automatically reuse a previous preview graph. The existing successful snapshot cache is already sufficient for coherent subsequent reads.

The correction intentionally does **not** promise that capture succeeds during an unverified source copy. That capture must still refuse and require a fresh successful review. It makes the unavailable observation explicit instead of manufacturing a different dependency scope. Any automatic retry or user-facing policy change would require separate justification; it is not needed for this bounded correctness repair.

Suggested native tests:
1. Convert this real concurrent-write probe into RED: first core item must be unavailable, no guessed optional edges, overall inventory incomplete even when later owner probes succeed; a fresh stable discovery returns included.
2. Two successful stable discoveries separated only by an ordinary saved note retain the same dependency scope.
3. Real optional references still produce exact required dependencies, and unavailable/missing referenced owners still block; no unused-edge blanket exemption.
4. Invalid schema/domain and permanently unavailable sources remain blocking. If an old pure-declaration test assumes corrupt core bytes are included with every optional edge, correct that precise expectation to the new explicit unavailable result; do not weaken validation.

No broader runtime, timeout, source-enrollment, or SQLite-copy change is recommended from this evidence.
