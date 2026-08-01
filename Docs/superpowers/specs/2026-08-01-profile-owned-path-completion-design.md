# TASK-865 Profile-Owned Path Completion Design

Date: 2026-08-01
Status: Approved
Backlog task: [TASK-865](../../../backlog/tasks/task-865%20-%20Sweep-hardcoded-~-.config-tldw_cli-and-~-.local-share-tldw_cli-call-sites-onto-the-real-accessors.md)
Architecture decision: [ADR-040](../../../backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md)

## Problem

TASK-865 corrected every explicitly named path site and added regression tests,
but its aggregate inventory remained incomplete. Current `origin/dev` still has
executable code that constructs `~/.config/tldw_cli` or
`~/.local/share/tldw_cli` directly.

Those occurrences are not one homogeneous defect. Some are profile-owned state
that bypasses `TLDW_CONFIG_PATH`; some omit the active user folder; others are
shared binary assets, inert configuration defaults, read-only legacy probes,
or dead code. A mechanical replacement would fix real collisions while also
creating large duplicate downloads and breaking compatibility diagnostics.

The completion tranche must classify every executable occurrence, repair the
profile-owned set, retire rejected dead persistence, and make the classification
enforceable.

## Verified Baseline

The design was checked against commit `af934b83f` on `origin/dev`.

- TASK-865 is `In Progress`; AC #3 and #4 are complete, while AC #1 and #2
  remain open.
- The existing TASK-865 path tests pass: 22 passed across
  `Tests/UI/test_chat_screen_ui_state_path.py` and
  `Tests/Chatbooks/test_chatbook_importer.py`.
- `ImprovedDictationWindow` is mounted by the production `STTSWindow`; its
  default exports currently omit the active user-folder segment.
- Speech preset readers and writers use a global
  `kokoro_voice_blends.json` even when `TLDW_CONFIG_PATH` selects another
  profile.
- Prompt and dictionary import helpers default their validation roots to the
  global config directory.
- Settings displays literal default paths rather than the active config path.
- `Audio/transcription_history.py` is imported only by
  `Widgets/transcription_history_viewer.py`; that viewer has no production
  importer. TASK-1331 explicitly removed persisted speech history.
- `Utils.paths.get_user_database_path()` and its `USER_DB_DIR`/
  `USER_DB_PATH` constants have no production caller.
- `UI/Dictation_Window.py` has no production importer but still implements a
  complete plaintext transcript-history load/append/save path. TASK-1331's
  product decision rejects that persistence just as it rejects the separate
  history store/viewer.
- The production diagnostic inventory is clean at this baseline: its checker
  reports 432 owners, 1,068 TASK-492 calls, 6,660 TASK-494 calls, and four sink
  files.

## Goals

1. Make every swept hardcoded profile-owned config occurrence in the normative
   inventory follow `get_cli_config_path().parent` at its call boundary.
2. Make every swept hardcoded active-user data occurrence in the normative
   inventory follow `get_user_data_dir()` and therefore include the configured
   root and active user folder.
3. Keep large, reusable artifacts shared.
4. Retire rejected or unreferenced persistence code instead of granting it an
   exception.
5. Prevent new hardcoded application-owned roots through a precise source
   sentinel.
6. Preserve existing files and avoid an implicit migration policy.
7. Preserve ADR-029 when a swept effective-config writer creates state in a
   newly selected profile.

## Non-Goals

- Moving, copying, importing, or deleting existing global files.
- Designing a user-facing legacy-data recovery flow.
- Relocating shared model weights, voice samples, or tokenizer artifacts.
- Splitting custom-tokenizer mappings from their currently shared artifact
  directory.
- Consolidating the legacy UI Kokoro blend JSON and the backend
  `voice_blends/voice_blends.json` representation. They have different formats
  and need an explicit compatibility design before either becomes the other's
  authority.
- Repairing non-atomic or permission-weak writers outside the exact
  profile-owned sites swept here.
- Replacing the existing config/data accessors with a service container or new
  path-policy abstraction.
- Migrating existing correct consumers of `_get_effective_config_path()` to its
  public wrapper. That would expand into unrelated runtime, security, RAG, and
  Notes Sync modules without changing resolved behavior.
- Changing `runtime_policy.DEFAULT_RUNTIME_POLICY_PATH`, its documented
  compatibility/error-fallback behavior, or the private helper used by
  `default_runtime_policy_path()`. Normal runtime-policy resolution is already
  call-time/profile-aware and this derived constant is not a remaining literal
  owner.
- Supporting a live `TLDW_CONFIG_PATH` retarget inside an already constructed
  `TTSBackendManager`. The manager and its cached backends belong to one
  `TldwCli` configuration session; selecting another profile requires another
  application/manager instance.
- Changing user-selected import, export, workspace, or model paths.

## Ownership Model

### Effective-config state

Small state and defaults whose identity follows the selected config file use
`get_cli_config_path().parent`. Resolution occurs inside the function or method
that needs the path.

The completion inventory includes:

- dictionary import validation root in
  `Character_Chat/Chat_Dictionary_Lib.py`;
- prompt import validation root in
  `Prompt_Management/Prompts_Interop.py`;
- GitHub token configuration guidance in
  `UI/CodeRepoCopyPasteWindow.py`;
- Kokoro blend definitions in `UI/STTS_Window.py`,
  `UI/Speech/speech_catalog_mixin.py`, and
  `UI/Speech/speech_settings_mixin.py`;
- Kokoro backend blend-directory default in `TTS/backends/kokoro.py`; and
- dynamic Settings theme and internal-prompt save-target copy in
  `UI/Screens/settings_screen.py`.

The implementation may introduce small pure path-formatting helpers to remove
duplication, but must not introduce a second config-path authority.

The legacy single-file Kokoro blend store and the backend blend-directory store
remain distinct persistence formats. Both defaults become profile-aware, but
TASK-865 does not copy between them or declare either format authoritative.
Writes to either swept profile-owned representation use ADR-029's existing
private atomic-write boundary. Shared voice/model artifacts do not.

### Active-user data

Application-chosen data roots use `get_user_data_dir()`.

The completion inventory includes the text and Markdown dictation export roots
in `UI/Dictation_Window_Improved.py`. Both formats must resolve through one
shared function so they cannot diverge again.

### Shared artifacts

The following remain shared and do not follow `TLDW_CONFIG_PATH` or the active
user folder in this tranche:

- Kokoro model weights and packaged voice data;
- Chatterbox and Higgs reusable voice samples/profiles;
- custom tokenizer artifacts.

The sentinel exception registry identifies these by exact executable site and
reason. User-authored Kokoro blend definitions are not shared artifacts; they
are effective-config state.

Custom tokenizer `mappings.json` currently cohabits with shared tokenizer
artifacts. This mixed-ownership directory is not silently split here because a
split needs a compatibility and migration contract.

### Persisted defaults and read-only probes

Literal strings are retained where they are data rather than runtime path
resolution:

- shipped TOML/default values in `config.py`;
- the shipped embedding-cache literal in that TOML, whose runtime default is
  replaced by `get_model_cache_dir()` with
  `get_user_data_dir()/models/embeddings`;
- Storage Settings defaults in
  `UI/Screens/settings_storage_defaults.py` and their editor reset values;
- the RAG configuration example/default that its runtime resolver replaces;
  and
- the exact Evals legacy location used only to warn about stranded data.

The Evals path remains read-only. It cannot be used as a fallback database
target.

Canonical resolver/default seeds are classified separately from inert data:

- `config.DEFAULT_CONFIG_PATH` is the default seed interpreted by
  `get_cli_config_path()` after checking `TLDW_CONFIG_PATH`;
- `config._default_base_data_dir()` is the call-time default interpreted by
  `get_user_data_dir()`; and
- `config.BASE_DATA_DIR_CLI` is a frozen compatibility constant used only by
  the standalone prompt-dump helper, not by production data resolution.

`runtime_policy.DEFAULT_RUNTIME_POLICY_PATH` is a separate frozen compatibility
and error-fallback constant derived from `DEFAULT_CONFIG_PATH`. Normal runtime
resolution uses `default_runtime_policy_path()` and follows the active profile.
The constant contains no independent hardcoded root and its existing failure
policy is outside this literal-ownership tranche; it is documented here so the
sweep cannot accidentally treat it as a second active resolver.

TASK-865 does not ban equivalent existing calls to the private effective-path
helper. Every newly changed consumer uses the public wrapper, but the literal
sentinel does not smuggle a provider-wide accessor refactor into this tranche.

## Normative Disposition Inventory

| Source group | Ownership | Action | Existing-file policy | Writer treatment | Verification |
| --- | --- | --- | --- | --- | --- |
| Character Chat dictionary and Prompt Interop default roots | Effective-config state | Resolve from `get_cli_config_path().parent` | No fallback/import | Read-only | Two-profile function tests |
| Code Repository token guidance and Settings theme/internal-prompt copy | Effective-config state | Format from `get_cli_config_path()` at render/action time | Not applicable | Display-only | Two-profile function/full-app assertions |
| STTS/Speech UI Kokoro blend JSON | Effective-config state | Resolve lazily from the effective config parent | No fallback/import | ADR-029 private atomic replacement | Reader/writer path and atomic-failure tests |
| Kokoro backend blend directory | Effective-config state | Resolve its default at backend construction for that manager's immutable application/config session; preserve explicit configured paths | No fallback/import or format convergence | Secure app-owned child directory plus ADR-029 private atomic replacement | Backend function tests with explicit/default paths and separate managers for two profiles |
| Improved Dictation text/Markdown exports | Active-user data | Resolve both formats through one `get_user_data_dir()/exports/dictation` helper | Earlier exports remain untouched | User-requested export semantics | Two-user-folder function tests and full-app action coverage only if UI behavior changes |
| Separate history store/viewer and unmounted legacy Dictation window | Rejected unreachable persistence | Delete | Existing user files remain untouched | No writer remains | Production-reference, wheel, docs, and inventory checks |
| Legacy `get_user_database_path` and `USER_DB_DIR`/`USER_DB_PATH` | Unused obsolete creation path | Delete and document unsupported-internal removal | No database is touched | No writer remains | SQLite inventory and installed-wheel checks |
| Config/Storage defaults and canonical resolver seeds | Inert data or resolver internal | Retain as exact counted exceptions | Existing semantics preserved | Owning accessor remains authoritative | Multiline/embedded sentinel fixtures |
| Shipped embedding-cache literal | Inert persisted default | Retain as a counted exception; runtime default remains `get_user_data_dir()/models/embeddings` | No runtime fallback to the literal | Active-user data accessor remains authoritative | Sentinel plus two-user-folder `get_model_cache_dir()` assertion |
| `DEFAULT_RUNTIME_POLICY_PATH` | Existing derived compatibility/error fallback | Retain unchanged; normal resolution remains call-time | Existing runtime-policy failure semantics preserved | Not a TASK-865 literal owner | Existing runtime-policy tests plus source census |
| Evals historical database probe | Read-only legacy probe | Retain exact path as a counted exception | Warn only | Writes forbidden | Existing Evals legacy-probe tests plus sentinel |
| TTS models, reusable voice assets, tokenizer artifacts | Shared artifacts | Retain configured/historical shared roots as counted exceptions | No relocation | Existing shared owner | Sentinel plus targeted shared-path assertions |

## Retired Code

The implementation removes:

- `tldw_chatbook/Audio/transcription_history.py`;
- `tldw_chatbook/Widgets/transcription_history_viewer.py`;
- `tldw_chatbook/UI/Dictation_Window.py`;
- tests that exist only to mount or exercise the rejected history viewer; and
- the unused `Utils.paths.get_user_database_path()` path, together with its
  `Utils.Utils.USER_DB_DIR` and `USER_DB_PATH` constants and imports.

This is deletion of unreachable code, not removal of a working feature.
TASK-1331 records the product decision to drop persisted speech history, and
the current source has no production importer for the viewer, store, or legacy
Dictation window.

Deletion collateral is part of the same change:

- remove the legacy-window assertions/imports from
  `Tests/Local_Ingestion/test_dictation_window_provider_ids.py` while
  preserving its production `ImprovedDictationWindow` coverage;
- remove only the history-specific imports and test from
  `Tests/UI/test_disabled_action_recovery_tooltips.py`;
- remove the deleted viewer from
  `Tests/UI/test_file_picker_filters_callable.py`'s source census;
- update `Utils/local_stt_providers.py` copy so it describes the retained
  legacy provider-id compatibility without naming the deleted window as live;
- remove the obsolete user-DB parent creator from the executable SQLite
  inventory and `Tests/DB/test_private_sqlite_inventory.py`, while preserving
  P05 in `backlog/docs/sqlite-private-owner-inventory.md` with a retired
  disposition;
- regenerate the production diagnostic inventory and verify every changed
  record is attributable to a touched or deleted source file rather than
  blessing unrelated drift;
- update or explicitly supersede
  `Docs/Development/TTS/TTS-Dictation-Implementation-Complete.md`,
  `Docs/Development/TTS/TTS-Improve-1.md`, and
  `Docs/Development/TTS/Speech-Recording-1.md` where they still present the
  retired modules as current architecture; and
- record the removal of unsupported direct-module compatibility names in
  `CHANGELOG.md`.

## Resolution and Data Flow

For effective-config state:

1. The caller reaches a production function or method.
2. That boundary calls `get_cli_config_path()`.
3. It derives a child from the returned path's parent.
4. The consumer reads, writes, validates, or displays that exact resolved path.

When the consumer owns swept profile state, serialization completes before
calling `atomic_private_write_text()`. A profile-owned child directory uses the
existing secure application-directory primitive. Failure before replacement
preserves the prior file.

For active-user data:

1. The caller reaches the export or persistence boundary.
2. That boundary calls `get_user_data_dir()`.
3. It derives the feature-specific child path.
4. The existing operation creates or writes only beneath that child.

No profile-owned site changed by TASK-865 resolves either root at module
import. The explicitly classified resolver seeds, compatibility constants, and
shared-asset sites retain their documented semantics. No newly introduced
singleton stores an effective profile path without including that resolved path
in its identity. The existing `TTSBackendManager` is scoped to one
application/config session; a second profile is exercised through a second
manager rather than by mutating a live manager's environment.

## Existing-File Policy

This tranche does not migrate data.

- With no override, effective-config children remain under the same default
  directory.
- With `TLDW_CONFIG_PATH`, newly resolved state belongs to that profile. Global
  files written by old code are not read as a fallback because their owner is
  ambiguous.
- Dictation exports begin using the active user's data tree. Earlier exports
  remain at their historical location and are neither hidden by an internal
  index nor deleted.
- The Evals compatibility warning remains the only approved fixed legacy probe
  in this inventory.

Any future recovery flow must identify the source, destination profile, copy or
move semantics, collision behavior, rollback, and user confirmation before it
mutates files.

## Error Handling

- Accessor failures retain the existing operation's error surface; swept
  callers do not fall back to a global home-directory literal. The separately
  documented runtime-policy compatibility/error fallback is unchanged.
- Read-only import roots do not create directories merely to validate a path.
- Export directory creation remains local to the export action.
- Display-only Settings paths are formatted from the accessor without touching
  the filesystem.
- A private state-write failure preserves the previous file and reports the
  operation as failed; it never retries against the historical global path.
- The sentinel reports the exact file, line, literal/construction, and missing
  classification.

## Regression Sentinel

The sentinel scans executable Python under `tldw_chatbook/` for:

- every occurrence of `~/.config/tldw_cli` or
  `~/.local/share/tldw_cli` inside a non-docstring executable string token,
  including embedded copy and multiline TOML/config constants; and
- normalized path-join suffixes for `.config/tldw_cli` and
  `.local/share/tldw_cli`, including constructions whose home/base expression
  is indirect rather than a direct `Path.home()` call.

It ignores comments and test prose. Remaining executable matches must appear in
an exact exception registry containing:

- source file;
- normalized literal or path-construction shape;
- expected executable occurrence count;
- ownership class;
- reason; and
- its bounded exception kind: persisted default, resolver seed, compatibility
  constant, shared artifact, or read-only legacy probe.

The test fails on an unclassified new match, a changed occurrence count, and a
stale exception whose source match no longer exists. This makes the registry a
census rather than an ever-growing suppression list.

For multiline strings, the scanner reports the physical line of each root
substring rather than only the AST node's first line. Sentinel fixtures prove
detection of embedded strings, multiline shipped configuration, direct and
indirect joins, duplicate occurrences, stale exceptions, and injected
violations; comments and actual docstrings are negative fixtures.

## Testing

Tests follow the repository rule: test the full application or test the actual
function. No reduced or synthetic application is introduced.

- Pure function tests retarget `TLDW_CONFIG_PATH` and assert dictionary,
  prompt, speech-preset, and display paths follow the selected config parent.
- Pure function tests retarget the configured data root/user folder and assert
  both Improved Dictation export formats use the same
  `get_user_data_dir()/exports/dictation` root.
- Pure function tests assert the shipped embedding-cache literal remains inert
  while `get_model_cache_dir()` resolves its runtime default beneath each
  active user's data directory.
- Backend function tests construct separate real `TTSBackendManager` instances
  for two config profiles and prove each Kokoro backend resolves only its own
  blend directory; no reduced application or live-manager retarget is used.
- Import/census tests prove the retired transcription-history and legacy DB
  path symbols have no remaining production references.
- The executable-source sentinel proves all remaining literals are classified
  and that embedded, multiline, indirect-join, duplicate, stale, and injected
  unclassified examples fail appropriately.
- Private-state writer tests prove interrupted serialization/replacement leaves
  the previous Kokoro preset file unchanged and new files satisfy ADR-029's
  platform-specific private-file contract.
- Existing TASK-865 chatbook importer and Chat UI-state tests remain green.
- Targeted suites cover Character Chat, Prompt Management, Speech/STT,
  Settings, Chatbooks, config/path isolation, and the new sentinel.
- The production diagnostic inventory checker and SQLite private-owner
  inventory tests pass with only the reviewed retirement delta.
- Installed-wheel tests prove the production application and retained public
  entry points import without source-checkout fallbacks after dead modules are
  removed.
- The full test suite and repository lint/static checks run before completion.

## Documentation and Task Hygiene

- TASK-865 acceptance criteria are narrowed from every textual occurrence to
  every swept hardcoded profile-owned occurrence in the normative
  inventory/sentinel, with all exceptions classified.
- TASK-865 links ADR-040 and this design.
- The implementation plan links ADR-040 and states that no new ADR is required
  unless implementation changes the approved ownership classes.
- Completion notes list every canonicalized, retired, and excepted production
  site and record verification results.
- Stale Speech/TTS implementation documentation and release notes are updated
  alongside code deletion.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md`

Reason: The work defines persistent data ownership, profile isolation,
shared-artifact boundaries, migration behavior, and a cross-module path
contract.

## Acceptance

The design is complete when implementation can demonstrate:

1. Every swept hardcoded profile-owned config/data occurrence in the normative
   inventory uses the canonical accessor at its call boundary.
2. Every remaining executable literal is an exact, reasoned exception.
3. The rejected transcript-history and unused legacy DB path implementations
   are absent.
4. Two config profiles and two user-data profiles do not resolve the swept
   state to the same path.
5. Shared artifacts retain their existing reusable locations.
6. Existing files are not copied, moved, or deleted.
7. Tests use production functions or the full application only.
8. Swept profile-owned state writers use ADR-029's private atomic-write
   boundary and preserve the previous file on failure.
9. Generated/curated inventories and installed-wheel coverage agree with the
   retired modules and symbols.
