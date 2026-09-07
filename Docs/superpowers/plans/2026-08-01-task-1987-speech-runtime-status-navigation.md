# TASK-1987 — Truthful Speech Status and Navigation Plan

**Goal:** Give global Speech & TTS Settings and the Speech Lab the same
revision-aware operational picture while preserving drafts and completed
audio across bounded cross-screen navigation.

**Scope boundary:** Reuse the existing external audio.cpp adapter, accepted
capability observations, Settings persistence path, and Studio preference
editor. Do not add provider discovery on Settings mount, a generic status
framework, managed audio.cpp lifecycle controls, or new persistent status
storage.

ADR required: yes
ADR path: `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`
Reason: TASK-1987 implements ADR-039's accepted revisioned status,
independent capability-row, artifact-independence, and bounded Settings/Lab
navigation contract; it introduces no new storage, provider, or runtime
boundary.

## Implementation steps

1. Add failing pure tests for revision-bound status projection, bounded safe
   diagnostics, independent provider/catalog/local-dependency rows, and exact
   provider/intent navigation context.
2. Implement a small shared Speech status and navigation projection over the
   existing provider configuration revisions and accepted capability
   observations. Reject mismatched provider, configuration, catalog, model,
   request generation, and observation order.
3. Replace the global Settings inspector placeholder and Lab aggregate
   dependency gate with independent configuration, selected-provider runtime,
   catalog/voice freshness, and local STT/TTS dependency rows. Keep Saved
   separate from runtime outcomes and attribute evidence to the saved revision
   while a connection draft is dirty.
4. Route Settings-to-Lab and Lab-to-Settings through the bounded target
   contract, restore provider intent without automatic work, and add
   Save/Discard/Cancel protection for dirty global provider, category,
   deep-link, and dismissal paths while retaining the Studio guards.
5. Preserve completed Playground artifacts across status, catalog, model,
   configuration, and navigation changes while continuing to reject late
   superseded synthesis results.
6. Run focused race, Textual, navigation, persistence/reconfiguration,
   privacy, dependency, and artifact suites; run neighboring regressions and
   static checks; independently review before completion.

## Expected files

- `tldw_chatbook/UI/Speech/speech_settings_contracts.py`
- `tldw_chatbook/UI/Speech/speech_runtime_status.py`
- `tldw_chatbook/UI/Lab_Modules/lab_speech_status.py`
- `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/stts_screen.py`
- `tldw_chatbook/UI/Speech/speech_settings_pane.py`
- `tldw_chatbook/UI/Speech/speech_catalog_mixin.py`
- `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- focused tests under `Tests/UI/` and `Tests/TTS/`
