# ADR-040: Profile-Owned State and Shared Asset Paths

Status: Accepted
Date: 2026-08-01
Related Tasks: [TASK-865](../tasks/task-865%20-%20Sweep-hardcoded-~-.config-tldw_cli-and-~-.local-share-tldw_cli-call-sites-onto-the-real-accessors.md), [TASK-1331](../tasks/task-1331%20-%20Speech-Recognition-history-is-a-shell-switch-records-nothing.md)
Supersedes: N/A

## Decision

Chatbook classifies application-owned local paths by ownership rather than by
their historical directory spelling:

| Class | Canonical resolution | Examples |
| --- | --- | --- |
| Effective-config state | Resolve from `get_cli_config_path().parent` at the call boundary | UI state, prompt and dictionary import roots, custom themes, small user-authored speech presets |
| Active-user data | Resolve from `get_user_data_dir()` at the call boundary | Per-user databases, durable application state, and application-chosen export roots |
| Shared artifacts | Use an explicit configured or documented shared location | Downloaded model weights, reusable voice samples, and tokenizer artifacts |
| User-selected paths | Preserve the user-supplied path after validation | File-picker imports, exports with an explicit destination, workspace roots |
| Legacy probes | Preserve the exact historical path as read-only compatibility evidence | The Evals legacy-database warning probe |
| Persisted defaults | Preserve literal values that are inert configuration data until resolved by their owning accessor | Storage Settings defaults and shipped configuration examples |
| Resolver/default seeds | Retain only inside the canonical resolver that gives the value call-time semantics | `DEFAULT_CONFIG_PATH` and `_default_base_data_dir()` internals |
| Compatibility constants | Retain only when repository evidence identifies a compatibility consumer or bounded failure contract, and normal production resolution does not depend on the frozen value | `BASE_DATA_DIR_CLI` for the standalone prompt-dump helper; derived `DEFAULT_RUNTIME_POLICY_PATH` for its existing compatibility/error fallback |

Executable sites changed by TASK-865 must not construct an effective-config or
active-user path from `Path.home()`,
`expanduser("~/.config/tldw_cli...")`, or
`expanduser("~/.local/share/tldw_cli...")`. They use the public accessors.
Resolution stays lazy for those profile-owned sites: a new module-level
constant, singleton, or cache may not freeze the first effective config path or
user data directory. If a long-lived cache is necessary, its identity includes
the resolved canonical path and it must invalidate when that identity changes.

Resolver/default seeds and explicitly identified compatibility constants are
the bounded exceptions to that lazy rule. A resolver seed is interpreted by a
call-time accessor. A compatibility constant may remain importable; any
production failure fallback that consumes it must be explicitly enumerated and
normal resolution must remain call-time. `BASE_DATA_DIR_CLI` has only the
standalone prompt-dump consumer. The derived
`runtime_policy.DEFAULT_RUNTIME_POLICY_PATH` remains the existing compatibility
and error fallback, while `default_runtime_policy_path()` performs normal
profile-aware resolution. TASK-865 does not change that runtime-policy failure
contract or turn this path-literal sweep into a migration of every existing
consumer of the private `_get_effective_config_path()` helper: sites changed in
this tranche use the public `get_cli_config_path()`, while equivalent
pre-existing helper consumers remain outside scope. In particular, no Notes
Sync path is changed.

Shared artifacts do not become profile-owned merely because old code placed
them beneath `~/.config/tldw_cli`. Large models and reusable voice assets stay
shared in this tranche so changing profiles does not duplicate downloads.
Small user-authored preset definitions are profile state even when they refer
to shared assets. The embedding-cache literal shipped inside the default TOML
is inert configuration data, not a shared runtime owner: normal runtime default
resolution remains active-user scoped through
`get_user_data_dir()/models/embeddings`.

Hardcoded legacy paths are allowed only for read-only detection of data created
by an older defect. A legacy probe must not become a fallback write target or
silently import, copy, move, or delete the discovered data.

Dead persistence implementations for product features that were explicitly
rejected are retired instead of being made profile-aware. In particular,
TASK-1331 dropped persisted Speech Recognition history, so its unreferenced
history store, viewer, and unmounted legacy Dictation window are removed rather
than legitimized.

A source sentinel enforces this boundary. Each executable literal that remains
must be in a small exception registry with its ownership class and reason.
The registry also pins the normalized expression and expected occurrence count
so adding another use in an already excepted file still fails.
Comments, docstrings, tests describing old defects, and non-Python examples are
not production path owners and are outside the sentinel.

## Context

TASK-865 fixed its explicitly named production sites but left its aggregate
inventory incomplete. The remaining source contains several different things
that happen to share the same literal spelling:

- profile-owned state that ignores `TLDW_CONFIG_PATH`;
- application data that omits the active `<user_folder>`;
- intentionally shared model and voice assets;
- inert Settings/configuration defaults;
- a deliberately fixed Evals legacy-data probe; and
- dead code with no production importer.

A mechanical replacement would make model caches profile-specific, mutate the
meaning of storage defaults, and break the Evals warning. Leaving the literals
unclassified would preserve current profile collisions and allow new ones.

The review also found ordinary writes in the swept speech-preset code. Because
TASK-865 causes overridden profiles to create those files at new locations,
their write boundary is part of this tranche: swept effective-config state
uses ADR-029's private atomic-write primitives. Repairing unrelated file
writers remains outside scope.

## Required Boundaries

- Production sites changed by TASK-865 use `get_cli_config_path()`, not the
  private `_get_effective_config_path()` helper. Existing equivalent private
  helper consumers are not hardcoded-path defects and remain outside this
  bounded sweep.
- Active-user paths use `get_user_data_dir()` so configured data roots and the
  active user-folder segment are both preserved.
- Default behavior for users without overrides remains path-compatible unless
  the old path omitted the required user-folder segment.
- No ambiguous global file is automatically adopted into a profile.
- No existing file is copied, moved, or deleted as part of the path sweep.
- A user-requested export is not reclassified as private application state
  merely because Chatbook supplies its default destination.
- Shared-asset exceptions identify the asset class; a generic file-level
  allowlist without a reason is insufficient.
- Rejected, unreachable persistence code is removed rather than added to the
  exception registry.
- Swept effective-config state writers use private atomic replacement; a
  user-requested export retains export semantics.
- `TTSBackendManager` and its cached backend instances belong to one immutable
  application/config session. Another `TLDW_CONFIG_PATH` profile uses another
  application/manager instance; TASK-865 does not promise hot retargeting of a
  live manager.
- Tests exercise production functions or the full application. Reduced test
  applications that imitate only part of `TldwCli` are prohibited.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Replace every literal with the nearest accessor | Would duplicate shared model/voice assets per profile and corrupt the meaning of inert storage defaults and legacy probes. |
| Introduce a central path-policy service | Existing public accessors already own effective config and active-user resolution. A new service would duplicate those contracts for this bounded sweep. |
| Fix only mounted screens | Leaves dormant executable code ready to reintroduce the defect and cannot support a complete regression sentinel. |
| Automatically migrate old global files | Ownership is ambiguous when multiple profiles may have read or written the same legacy path. Silent copying can leak one profile's state into another. |
| Keep rejected transcription history as an allowlisted exception | Preserves an unused plaintext-capable persistence implementation after the product decision explicitly removed the feature. |
| Ban every existing private effective-path helper import | Expands a literal-path sweep into unrelated runtime, security, RAG, and Notes Sync refactors without changing resolved behavior. |

## Consequences

### Benefits

- Effective config overrides and active user folders govern all classified
  profile-owned paths.
- Shared downloads remain reusable and do not multiply across profiles.
- Future hardcoded path regressions fail a focused sentinel.
- Dead transcript persistence can no longer be accidentally re-enabled.
- Legacy data remains untouched until a separately designed, explicit recovery
  flow can establish ownership.

### Accepted Trade-offs

- A profile redirected with `TLDW_CONFIG_PATH` does not silently see state that
  old code wrote to the global default directory.
- Some shared-asset locations remain historical until a separate shared-cache
  placement and migration decision is justified.
- Unrelated file writers are not made atomic or permission-hardened by this
  bounded sweep.

## Links

- [TASK-865 completion design](../../Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md)
- [ADR-004: Settings Storage Defaults Restart Boundary](004-settings-storage-defaults-restart-boundary.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-023: TTS Adapter Registry and Runtime Boundary](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-025: Shared STT Artifacts and Runtime Routing](025-shared-stt-artifacts-and-runtime-routing.md)
- [ADR-028: Character TTS Generation Profile Ownership](028-character-tts-generation-profile-ownership.md)
