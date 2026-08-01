# TASK-1693 Studio TTS Preference Storage Implementation Plan

**Goal:** Add an isolated, versioned, sparse Studio TTS preference store with safe legacy migration, corruption isolation, and stale-writer protection.

**Architecture:** A pure immutable Studio snapshot validates the bounded selection and provider-option surface established by TASK-1692. A small repository reads and writes only the additive `speech_studio` TOML section. Writes replace that one section atomically through the existing configuration owner and use its persisted revision as an optimistic concurrency guard. First-read migration inspects raw saved config only, copies non-default values from the exact proven request-scoped allowlist, and never deletes or rewrites legacy/global keys.

**ADR required:** yes
**ADR path:** `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`
**Reason:** This task directly implements ADR-039's accepted Studio storage, migration, corruption, and compare-before-publish contract; it makes no new architectural decision.

**Scope constraints:** No visible UI, resolver integration, network discovery, generation behavior, provider reconfiguration, character storage, credential mutation, or managed audio.cpp behavior.

---

## 1. Pin the storage contract with failing tests

**Create:** `Tests/TTS/test_studio_preferences.py`

Add deterministic tests that require:

- all selection axes to be optional overrides;
- the seven canonical provider IDs and no others;
- only the TASK-1692-proven provider option keys;
- immutable, defensively copied snapshots;
- bounded model/voice IDs, formats, speeds, and option values;
- unknown keys, endpoints, masked placeholders, credential-shaped fields, runtime paths, character fields, and synthesis text to fail closed;
- a sparse TOML representation that omits inherited values.

Run the new file and confirm import/contract failures before production code exists.

## 2. Implement immutable preference values and serialization

**Create:** `tldw_chatbook/TTS/studio_preferences.py`
**Create:** `tldw_chatbook/TTS/provider_ids.py`
**Modify:** `tldw_chatbook/UI/Speech/speech_settings_contracts.py`

Implement:

- one shared canonical built-in provider tuple used by both ownership and persistence contracts;
- `StudioTTSSelectionOverrides` for optional provider/model/voice/format/speed values;
- `StudioTTSPreferencesSnapshot` with schema version, persisted revision, immutable provider-scoped options, and exact validation;
- strict serialization that emits only non-inherited values and the schema/revision envelope.

Run the new tests until the pure contract and serialization cases pass.

## 3. Add atomic revision-guarded section replacement test-first

**Modify:** `tldw_chatbook/config.py`
**Modify:** `Tests/test_config_delete_settings.py`
**Modify:** `Tests/TTS/test_studio_preferences.py`

First add failing tests proving that:

- a whole revisioned section is replaced in one atomic config mutation;
- a missing or structurally corrupt section is recoverable at revision zero;
- an expected-revision mismatch returns an explicit conflict without replacing the file;
- the next stored revision must be exactly `expected + 1`;
- unrelated config sections remain byte-equivalent after parsing;
- a write failure publishes no partial Studio snapshot.

Then add the smallest public wrapper over the existing locked read/atomic replace/cache-publish lifecycle. Preserve existing `ConfigMutationResult` callers by adding only a defaulted conflict flag and leaving ordinary mutation behavior unchanged.

## 4. Implement round-trip, reset, and stale-writer behavior

**Modify:** `tldw_chatbook/TTS/studio_preferences.py`
**Modify:** `Tests/TTS/test_studio_preferences.py`

Implement a repository that:

- reads the raw `speech_studio` section from a defensive runtime snapshot;
- saves a complete normalized section with optimistic revision comparison;
- returns bounded saved, unchanged, conflict, failed, or cache-reload-failed outcomes;
- restores per-provider option namespaces without cross-provider leakage;
- resets to global by replacing the section with only schema/revision metadata;
- never invokes TTS services or provider reconfiguration.

Cover round trip, sparse inheritance, provider isolation, stale concurrent writers, reset-by-deletion, and atomic write failure.

## 5. Implement versioned migration and corruption isolation

**Modify:** `tldw_chatbook/TTS/studio_preferences.py`
**Modify:** `Tests/TTS/test_studio_preferences.py`

Implement first-read migration from raw saved legacy values only:

- copy Chatterbox exaggeration and CFG weight when valid and non-default;
- copy the active ElevenLabs model or active AllTalk voice/format when valid and non-default;
- never copy the global provider itself, credentials, endpoints, environment values, masked placeholders, runtime resources, safety limits, character fields, or text;
- preserve every legacy/global key;
- parse eligible fields independently and expose only bounded field-name diagnostics;
- write once only when valid overrides or malformed eligible fields require migration;
- make repeated reads idempotent and keep absent/no-op reads write-free;
- treat invalid structural schema/revision data as Studio-only corruption and permit a revision-safe reset without touching other scopes.

## 6. Verify and close TASK-1693

Run:

```bash
python -m pytest Tests/TTS/test_studio_preferences.py Tests/test_config_delete_settings.py Tests/test_config_runtime_snapshot.py Tests/UI/test_speech_settings_contracts.py Tests/TTS/test_tts_preferences.py -q
python -m ruff check tldw_chatbook/TTS/studio_preferences.py tldw_chatbook/TTS/provider_ids.py tldw_chatbook/UI/Speech/speech_settings_contracts.py tldw_chatbook/config.py Tests/TTS/test_studio_preferences.py Tests/test_config_delete_settings.py
git diff --check
```

Request independent code review, address every supported finding, and rerun the final gates. Then update TASK-1693 acceptance criteria and Implementation Notes, document the known repository baseline separately if reproduced, mark the task Done, and commit the atomic slice.
