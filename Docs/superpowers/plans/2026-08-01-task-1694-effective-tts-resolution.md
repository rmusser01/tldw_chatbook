# TASK-1694 Effective TTS Resolution Implementation Plan

**Goal:** Resolve every supported TTS selection into one immutable, text-free snapshot with explicit precedence, source metadata, provider constraints, and admission revisions.

**Architecture:** Add one pure provider-neutral resolver between the existing global, character-profile, Studio, and adapter-admission contracts. The resolver consumes already validated immutable layer values, selects each axis without mutating any owner, resolves dynamic models once through an injected accepted-catalog reader, and returns a complete snapshot. The existing request-admission coordinator will use that snapshot to build the same native or temporary-legacy request shapes it uses today. Future Studio UI work will call the same contract; this task does not move controls or add a second persistence path.

**ADR required:** yes
**ADR path:** `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`
**Reason:** TASK-1694 directly implements ADR-039's accepted precedence, dynamic-selection, revision, preview, and fail-closed admission boundary; it makes no new architectural decision.

**Scope constraints:** No visible Settings or Lab changes, no new persistence, no profile or assignment mutation, no provider discovery outside request admission, no native legacy-provider migration, and no managed audio.cpp behavior.

## 1. Pin the immutable resolution contract with failing tests

**Create:** `Tests/TTS/test_effective_settings.py`

Add tests requiring:

- exact bounded source values for every selection axis;
- recursively immutable options and revision metadata;
- no field for text, credentials, endpoints, widgets, character payloads, or adapters;
- canonical providers only and provider-scoped option admission;
- audio.cpp WAV, speed `1.0`, and empty-options enforcement.

Run the new tests first and confirm the resolver module is absent.

## 2. Implement normal and character precedence

**Create:** `tldw_chatbook/TTS/effective_settings.py`

Implement immutable selection-layer, source, revision, and effective-snapshot values. Add a normal-request entry point that resolves provider-compatible values in this order:

1. explicit caller overrides;
2. a previously authority-validated assigned character profile selection;
3. the current immutable global preference snapshot; and
4. a small explicit built-in provider fallback manifest.

Provider changes must prevent model, voice, format, speed, or option values owned by a different provider layer from leaking into the chosen provider. Absence inherits; malformed, unsupported, or incoherent selected values raise one bounded resolution error instead of falling through.

## 3. Implement Studio, preview, and dynamic-mode resolution

**Modify:** `Tests/TTS/test_effective_settings.py`
**Modify:** `tldw_chatbook/TTS/effective_settings.py`

Add the Studio entry point with exact precedence:

1. current validated controls or an explicitly loaded preview;
2. the immutable persisted Studio snapshot;
3. global preferences; and
4. the chosen provider fallback.

The Studio API will not accept a character assignment. Preview inputs remain marked as unsaved Studio draft provenance. A draft based on a stale Studio revision fails closed. `First available` calls the injected catalog reader exactly once and freezes the selected model plus catalog revision; `Server default` freezes an omitted voice. Neither path writes any owner.

## 4. Route existing admission through the shared snapshot

**Modify:** `Tests/TTS/test_tts_request_admission.py`
**Modify:** `tldw_chatbook/TTS/request_admission.py`
**Modify:** `tldw_chatbook/TTS/TTS_Generation.py`

Test first, then make default admission resolve global preferences plus any explicit voice override through the shared resolver while holding the existing admission read gate. Make native exact admission use the same resolver and preserve its existing `TTSRequestedSelectionSnapshot` compatibility projection. Capture the selected provider configuration revision under the gate and pass it to adapter admission unchanged. Keep submitted text separate until the final `TTSRequest` is built.

Do not change the temporary legacy bridge request contract. Its existing default path is built from the effective snapshot, and direct legacy Studio generation remains available for the later Studio-transition task.

## 5. Prove profile, Studio, privacy, and legacy compatibility

**Modify:** `Tests/TTS/test_character_request_resolver.py` only if integration evidence requires it
**Modify:** neighboring focused tests only when behavior is intentionally exercised

Add table-driven coverage for every precedence layer, provider mismatch, missing authoritative character selection, exact/dynamic modes, stale Studio draft revision, invalid higher layers, provider fallback, source metadata, and no owner mutation. Verify current character lookup remains fail closed and non-mutating, global-only requests keep their accepted legacy request shapes, and audio.cpp exact admission keeps complete-WAV provenance.

Run focused resolver/admission/profile/Studio tests, the full TTS suite, Ruff on changed files, compile checks, `git diff --check`, and an independent code review. Record any unchanged repository baseline failure rather than expanding this task to unrelated cleanup.
