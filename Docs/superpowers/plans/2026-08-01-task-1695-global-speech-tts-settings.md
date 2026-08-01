# TASK-1695 — Global Speech & TTS Settings Implementation Plan

**Goal:** Add one discoverable global Speech & TTS Settings owner for defaults, provider connection/initialization configuration, and credentials, while making the corresponding Lab controls read-only and preserving existing provider behavior.

**Scope boundary:** This task does not add catalog discovery/freshness UX, the final audio.cpp provider polish, separate Studio editing, revision-safe cross-screen status, or managed audio.cpp lifecycle controls. Those remain in TASK-1696 through TASK-1698.

ADR required: yes
ADR path: `backlog/decisions/039-global-and-studio-tts-settings-ownership.md` and `backlog/decisions/012-provider-credential-settings-boundary.md`
Reason: TASK-1695 directly implements the accepted durable global owner, credential mutation boundary, and provider-runtime handoff. It makes no new architectural decision beyond those ADRs.

## Implementation steps

1. Add failing pure-contract tests for the global field inventory, provider-specific validation, non-secret default restoration, credential Set/Replace/Clear intent, and adapter-affecting mutation classification.
2. Implement a bounded Speech & TTS Settings model that loads current global values, preserves every TASK-1692 global field, validates without provider I/O, builds exact atomic mutations, and never treats masked/environment credential displays as payloads.
3. Add failing Textual tests for the new Settings category, required search vocabulary and provider preselection, keyboard reachability, scope copy, distinct Default/Configure selectors, one mounted provider form, action semantics, path-picker affordances, and absence of managed audio.cpp controls.
4. Add the self-contained global Speech & TTS panel and wire it into the existing Settings category/search/draft shell. Route Save and explicit credential mutations through the existing atomic TTS publication path, report persistence separately from targeted reconfiguration, and keep ordinary mount/search/edit/revert/default actions free of provider calls.
5. Add failing transition tests, then make TASK-1692 global-owned Lab controls read-only while preserving their effective values and excluding them from Lab save payloads; retain runtime and Voice Profile operations and all Studio-owned controls.
6. Run focused model, Textual, persistence, TTS reconfiguration, legacy-provider, and Lab-equivalence suites; run Ruff, compile checks, and `git diff --check`; independently review the completed slice before marking TASK-1695 Done.

## Expected files

- `tldw_chatbook/UI/Screens/settings_config_models.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/settings_speech_tts.py` (new bounded model)
- `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` (new panel)
- `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- `tldw_chatbook/UI/Speech/speech_settings_group.py`
- `tldw_chatbook/UI/Speech/speech_settings_mixin.py`
- focused tests under `Tests/UI/` and `Tests/TTS/`
