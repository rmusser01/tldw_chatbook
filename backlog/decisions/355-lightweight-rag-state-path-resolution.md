# ADR-355: Resolve RAG state paths without importing the RAG runtime

Date: 2026-11-07

## Status

Accepted for implementation under TASK-34564; source and native/browser qualification remain pending.

## Context

Console file-authority preparation calls sensitive_paths._direct_child_rule_container_dirs to protect configuration, vector-store and profile files. Its RAG directory getters currently import config_profiles and simplified.config. The simplified package initializer imports embeddings, vector/RAG services and related optional modules before those directory functions execute.

A guarded fresh-process probe on f3cfffd9, with main config and sensitive_paths already resident, took 1.5782705 seconds and loaded 570 additional dependency names. It returned four private paths with no network or profile violations. This one diagnostic sample does not reproduce or explain the separate full-app 60-second binding wait, and establishes no UI latency percentile.

## Decision

Put the shared RAG state-directory resolution in a lightweight RAG_Search module that depends on configuration and standard-library path handling, without importing optional RAG/ML services. Existing public default_chroma_persist_directory and default_rag_profiles_dir entry points remain compatible wrappers/re-exports; sensitive_paths uses the same authoritative lightweight resolution.

Preserve the current Chroma priority exactly: RAG_PERSIST_DIR, vector_store.persist_directory, legacy chroma.persist_directory, then current user-data/chromadb. Preserve blank handling, conversion of non-null values, expanduser, explicit relative paths, per-setting read failures and the current user-data/rag_profiles default. Keep existing runtime configuration/legacy caller seams where needed. Do not add a second independently maintained directory policy.

Do not change sensitive-path memo keys, failure caching, exclusions, binding validation, permission decisions, storage admission, recovery or worker authority. This decision changes dependency loading, not data locations or permission scope. Keep configuration access live rather than caching directory results.

## Alternatives

- Lazily exporting only simplified.__init__ is insufficient: config_profiles also imports NumPy-dependent modules.
- Copying path rules into sensitive_paths creates policy drift between protection and the actual stores.
- Backup RAG discovery has different validation and historical-root semantics and cannot replace these accessors.
- Prewarming or bypassing the original checks would conceal first-use cost or change the qualification contract.

## Validation

Add a fresh-process regression proving real protected-directory resolution returns the correct paths while optional RAG/ML imports are blocked. Test all existing environment/config precedence and dynamic user-data behavior, including legacy public entry points. Preserve the sensitive-path and file-tool refusal tests, canonical remote-worker closure/artifact, diagnostic inventory and boot limits. Repeat the same bounded isolated probe after the change, then qualify the real Console; report the full-app wait separately unless independently attributed.