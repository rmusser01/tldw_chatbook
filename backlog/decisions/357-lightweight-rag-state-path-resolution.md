# ADR-357: Resolve RAG state paths without importing the RAG runtime

Date: 2026-11-07

Renumbered from the RAG draft of ADR-355 on 2026-11-08. The older Ruff lint-gate decision keeps ADR-355. Open PR #3480 claims ADR-356.

## Status

Accepted for implementation under TASK-34564; source and native/browser qualification remain pending.

## Context

Console file-authority preparation calls sensitive_paths._direct_child_rule_container_dirs to protect configuration, vector-store and profile files. Its RAG directory getters currently import config_profiles and simplified.config. The simplified package initializer imports embeddings, vector/RAG services and related optional modules before those directory functions execute.

A guarded fresh-process probe on f3cfffd9, with main config and sensitive_paths already resident, took 1.5782705 seconds and loaded 570 additional dependency names. It returned four private paths with no network or profile violations. This one diagnostic sample does not reproduce or explain the separate full-app 60-second binding wait, and establishes no UI latency percentile.

## Decision

Put the shared RAG state-directory resolution in a lightweight RAG_Search module that depends on configuration and standard-library path handling, without importing optional RAG/ML services. Existing public default_chroma_persist_directory and default_rag_profiles_dir entry points remain compatible wrappers/re-exports; sensitive_paths uses the same authoritative lightweight resolution.

Preserve the current Chroma priority exactly: RAG_PERSIST_DIR, vector_store.persist_directory, legacy chroma.persist_directory, then current user-data/chromadb. Preserve blank handling, conversion of non-null values, expanduser, explicit relative paths, per-setting read failures and the current user-data/rag_profiles default. Keep existing runtime configuration/legacy caller seams where needed. Do not add a second independently maintained directory policy.

Do not change sensitive-path memo keys, failure caching, exclusions, binding validation, permission decisions, storage admission, recovery or worker authority. This decision changes dependency loading, not data locations or permission scope. Keep configuration access live rather than caching directory results.

## Profile-metadata extension (TASK-34569 AC10)

A later genuine received-send diagnostic on 7cf51a captured the preparation worker at console_configuration_preparation.capture -> library_rag_profile_top_k -> simplified.active_config, with the eager simplified facade loading embeddings_wrapper and NumPy. The same instant showed a credential-status thread-start wait on the UI thread and a separate screen-preimport terminal query. It establishes the metadata import edge, not the full wait duration or a deadlock diagnosis; physical timing qualification remains zero.

The depth-only resolver continues to use the SAME active profile manager, profile pointer, storage admission, CRUD, built-in/custom loading, environment override and invalid/missing fallback. Make the simplified facade lazy, preserving its existing exported names and aliases as references to the original objects. Runtime exports load their real modules only when requested; absent optional dependencies still surface at that access. No runtime export is fabricated or made a metadata dependency.

RerankingConfig and ProcessingConfig are plain dataclasses with no post-init/device behavior. Their unchanged definitions belong in lightweight RAG_Search/config_metadata.py; reranker.py and parallel_processor.py reexport those exact class objects, and active_config/config_profiles import the canonical metadata classes. Public names, constructor signatures/defaults, mutable dataclass behavior, asdict and isinstance identity are preserved. The class __module__ intentionally becomes the canonical metadata module so new pickles resolve without loading engines; historical old-module/name pickle references remain loadable through the identical aliases. This is a type-ownership relocation, not a new profile serialization format.

Do not replace or narrow full resolve_active_rag_config. Its existing auto-device probe, optional-dependency decisions, environment and explicit-argument precedence stay as written; depth-only reads continue to avoid that full device layer. Do not add profile/config caches, parallel JSON readers, new defaults or policy/admission paths. Existing patch seams remain live.

Validate the actual Library depth read and original received-configuration capture in fresh protected processes with heavy import-attempt guards before any engine preload or package stub. Test real profile creation/update/resolution, default15, custom/missing/invalid/env cases and full resolution parity, plus all facade aliases, legacy type aliases and old/new pickle paths. Preserve existing import-order, missing-optional-dependency and profile/admission tests. Root regenerates the diagnostic inventory after final code; renewed live qualification follows a reviewed source freeze.

## Alternatives

- Lazily exporting only simplified.__init__ is insufficient: config_profiles also imports NumPy-dependent modules.
- Copying path rules into sensitive_paths creates policy drift between protection and the actual stores.
- Backup RAG discovery has different validation and historical-root semantics and cannot replace these accessors.
- Prewarming or bypassing the original checks would conceal first-use cost or change the qualification contract.

## Validation

Add a fresh-process regression proving real protected-directory resolution returns the correct paths while optional RAG/ML imports are blocked. Test all existing environment/config precedence and dynamic user-data behavior, including legacy public entry points. Preserve the sensitive-path and file-tool refusal tests, canonical remote-worker closure/artifact, diagnostic inventory and boot limits. Repeat the same bounded isolated probe after the change, then qualify the real Console; report the full-app wait separately unless independently attributed.

## Lower-layer ownership correction (TASK-34569 AC12)

The layer-direction gate exposed Utils.sensitive_paths importing the higher RAG_Search package. Canonical directory policy therefore moves to Utils/rag_state_paths.py, below both consumers; RAG_Search/state_paths.py retains the compatible public and patchable wrapper surface. Keep exactly one policy implementation and the same live configuration callbacks, precedence, coercion, failure fallback, privacy and directory choices. This corrects source ownership only; it creates no permission, storage, provider, cache or runtime policy. Validate both public compatibility and low-level protection before renewed qualification.
