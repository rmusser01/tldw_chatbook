---
id: TASK-21108
title: >-
  app.py import diet - 5.6k-line settings panel imported for one dataclass, and
  the import-weight guard is too loose to notice
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 18:30'
labels:
  - performance
  - startup
  - imports
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21108).

app.py's top-level imports grew 194 -> 220 since the 2026-08-11 audit pin. Concrete deferrable
whales verified: `Widgets/Settings_Widgets/speech_tts_settings_panel` (5,618-line widget module
imported for the single `SpeechTTSPanelDraftSnapshot` payload class used in isinstance checks,
app.py:329-331); `TTS/voice_bundle_service` (1,857); the `Notes/notes_sync_runtime` chain;
Notifications package init. None is needed before first paint. Meanwhile
`Tests/Performance/test_app_import_weight.py:85-86` allows 8.0 s / 4,000 modules - far above
any real drift signal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `SpeechTTSPanelDraftSnapshot` lives in a small types module; the 5,618-line panel module is no longer on the app import path (sys.modules assertion)
- [x] #2 voice_bundle_service and the notes_sync_runtime import chain are deferred to first use
- [x] #3 The import-weight guardrail budgets are tightened to sit just above the measured post-diet reality, so the next regression of this class fails a test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census on this base (post-21102/21103/21104/21105/21112): -X importtime + sys.modules diff in an isolated subprocess; record what is STILL eager vs already gone.
2. Move the Speech/TTS panel draft-validation cluster (SpeechTTSPanelDraftSnapshot, _RealtimeSettingsDraft, the bounded validators/constants) into a small speech_tts_panel_types module; the 5,618-line panel re-imports them so class identity (type(x) is ...) is preserved for its own tests, and app.py imports the types module.
3. Defer TTS/voice_bundle_service to first use: function-local import in _ensure_tts_voice_bundle_service plus TYPE_CHECKING + quoted annotations (app.py has no 'from __future__ import annotations', so attribute/return annotations evaluate at import).
4. Defer the notes_sync_runtime chain: move build_notes_sync_legacy_migrator/build_notes_sync_runtime_owner (and the notes_sync_legacy start-evidence import) out of __init__ into a lazy slot+setter property built on first access (on_mount start / library screen read), keeping the single build call the cutover AST test asserts.
5. Probe each deferral for the two sibling-review failure modes: fresh-subprocess submodule-first imports (circular-import unmasking) and the error surface at first use vs boot.
6. Extend the Tests/Packaging closure-guard family with an app-import-diet guard asserting the deferred modules are absent and the anti-vacuity closure members still present.
7. Tighten Tests/Performance/test_app_import_weight.py budgets onto the measured post-diet reality with stated headroom.
8. Re-measure warm+cold, run the Packaging/App/TTS/Notes/notifications suites, --collect-only sweep, diagnostic inventory.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cut 35 modules and ~36 ms of import self-time out of `import tldw_chatbook.app`
(1,700 -> 1,665 total modules; 664 -> 630 `tldw_chatbook.*` modules), added a
Packaging closure guard for each deferral, and replaced the useless 4,000-module
budget with one that a regression of this class actually fails.

### Census first (measured on this base, not inherited from the review)

Re-derived on the post-21102/21103/21104/21105/21112 base with `-X importtime` plus a
`sys.modules` diff in an isolated subprocess (scratch HOME/XDG/`TLDW_CONFIG_PATH`).
All four items the review named were **still eager**, but two were not eager for the
reason the review recorded:

- `voice_bundle_service` is not dragged by `app.py` at all any more; it is dragged by
  the eager `TTS/__init__` (`from tldw_chatbook.TTS import TTSProfileService`,
  app.py:311). Removing the app.py import alone bought **zero** — verified by
  measurement, not reasoning.
- The "`notes_sync_runtime` chain" is 15 modules only if `notes_sync_legacy` goes too:
  most of the subtree (`notes_device_state_store`, `notes_sync_filesystem`,
  `notes_sync_reconciler`, `note_import_*`, `sync_paths`) hangs off the legacy module
  that the TASK-21112 start gate reads, which `app.py` imported separately.
- `app.py` top-level import statements measured 223, not the review's 220 (220 after
  this change).

### The four changes

1. **Panel payload -> `Widgets/Settings_Widgets/speech_tts_panel_types.py` (new).**
   The draft-validation cluster (`SpeechTTSPanelDraftSnapshot`, `_RealtimeSettingsDraft`,
   the bounds, `_detached_draft_data`, both validators) moved verbatim into a pure
   module; the panel re-imports the three names it uses, so `type(x) is
   SpeechTTSPanelDraftSnapshot` still holds whichever module a caller imported from.
   The seam's only dependency, `UI/Screens/settings_speech_tts`, is already on the boot
   path via `STTS_Events/stts_events`, so the replacement costs 0.56 ms.
   **-20 modules, ~-13 ms** (the panel took `Third_Party/textual_fspicker` ×13,
   `rich._emoji_codes`, `lab_speech_status`, `speech_runtime_status`,
   `Chat/console_voice_input` and `Utils/local_stt_providers` with it).
2. **`TTS/voice_bundle_service` -> PEP 562 `__getattr__` on `TTS/__init__` + a
   function-local import in `_ensure_tts_voice_bundle_service`.** A *partial* facade:
   the eager init is untouched, only the five voice-bundle-service exports are deferred,
   so no import-order reshuffle and no cycle risk for the other 11 submodules.
   **-1 module, ~-2 ms.**
3. **Lasting-sync runtime -> lazy `notes_sync_runtime_owner` property.** The
   `__init__` block moved to `_construct_notes_sync_runtime_owner()`, which imports
   `notes_sync_runtime` and `notes_sync_legacy` function-locally; the property builds
   under a lock and keeps a setter so test doubles still assign. `on_mount` is the
   first production reader. `_shutdown_notes_sync_runtime` now returns early when the
   owner was never built rather than constructing one to shut it down.
   **-15 modules, ~-21 ms.**
4. **Budgets.** `MAX_MODULE_COUNT` 4,000 -> 2,200 and a new
   `MAX_TLDW_MODULE_COUNT` = 660 against a measured 630. The split is deliberate: the
   TOTAL closure varies by installed extras (`cryptography` via `subscriptions`,
   plus tokenizers/frontmatter/datasets), so pinning it near 1,665 would fail on a
   fully-extras dev box; the `tldw_chatbook.*` count depends only on this repo's
   import graph, so that is where the tight budget goes. `MAX_IMPORT_SECONDS` stays
   8.0 s on purpose — a genuinely cold run measured 5.6 s, so a tighter time bound
   buys flakes, not signal. Rationale and every measurement are in the module comment.

### Two things the deferrals broke, and why

- **A deferral changes WHICH objects the build binds, not just WHEN.** Two
  `ProductionApp/test_file_notes_session_owner_lifecycle.py` tests replace
  `app.file_notes_session_owner` between construction and mount; the moved-but-verbatim
  builder then read the *replacement* and died on `current_binding`, taking the
  LibraryScreen mount with it. Fixed by capturing `file_notes_binding` and
  `notes_scope_service` in `__init__` — the objects the eager build bound. Recorded in
  `lessons-testing-evidence.md`; no closure probe can see this class.
- **An `endswith()` AST fence counts a wrapper as a second call.**
  `test_notes_sync_cutover.py`'s cutover fence asserts exactly one call whose name ends
  with `build_notes_sync_runtime_owner`; a `_build_...` wrapper made it two. The method
  is named `_construct_notes_sync_runtime_owner` for that reason, documented in place.

Three test call sites that monkeypatched now-function-local names on `app_module` were
repointed to the defining modules (`Tests/TTS/test_tts_app_ownership.py`,
`Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py` ×3), which is also what makes
the substitution actually intercept the deferred import.

### Deliberately not done

**The Notifications package init is left eager, and that is the honest call.** It is
10 modules / 4.0 ms, dragged both by `Home/__init__` -> `active_work_adapter` and by
`app.py:567` — but `TldwCli.__init__` unconditionally calls
`_wire_watchlists_and_notifications_services()` (app.py:5953), which constructs six of
those services. Deferring the init would remove them from the *import* closure and
change real boot cost by ~0, i.e. it would make the new budget look better without the
app booting faster. A real fix belongs with deferring that wiring (21105 family), not
with an import-only shuffle.

### Found while measuring, filed for someone else

`Tests/Packaging/test_persona_buddy_import_closure.py` (TASK-21103's guard) is RED on
dev and was already red before this branch. `Actor_Packs/__init__.py:8` ->
`activation.py:21-22` puts `Persona_Visual.repository` and, through
`Character_Chat/visual_identity.py:24`, PIL back on the boot path — 15 modules
(10 PIL + 5 Persona_Visual) — re-broken by the Actor Pack activation work
(`ae817fefe`). Out of scope here; reclaiming it is worth ~15 more modules of the new
budget's headroom.

### Verification

- Closure: `Tests/Packaging/test_app_import_diet_closure.py` (new, 2 tests) and the two
  new budget/cutover tests are **red on a `git archive HEAD` tree** (naming all 17
  primary modules; 664 > 660) and green here.
- Circular-import probes: 12 fresh-subprocess imports (each deferred module first, the
  package facade first, each consumer first, app-then-module) — all pass. This is the
  TASK-21160 failure mode a lazy facade can unmask.
- Error surfaces traced: `voice_bundle_service` has only stdlib + in-repo imports and
  its one production caller already degrades on `Exception`; the runtime build does no
  I/O and `on_mount` remains its first reader, so a build failure surfaces where the
  eager one did.
- Live: real app launched in tmux on an isolated profile; Library (the
  `notes_sync_runtime_owner` reader) and Settings > Speech & TTS (the deferred panel)
  both mount and render.
- `./scripts/preflight.sh` all green (CSS bundle, profile-owned paths, diagnostic
  inventory 532 owners, duplicate ids, chachanotes allowlist) — no drift from these
  changes.
- Suites, each A/B'd against a `git archive HEAD` tree so no red is inherited blind:
  Notes+TTS 6,932 passed / 1 failed (that one, `test_app_lifecycle_shutdown_drains_all_
  owners_in_authority_order`, fails identically on HEAD — Actor-Pack shutdown, not ours);
  Packaging+Performance+App 315 passed / 6 failed / 58 errors, all 6 and all 58 identical
  on HEAD (the errors are `setuptools.build_meta` missing from this uv venv);
  ProductionApp 63 passed / 6 failed -> after the collaborator-capture fix, back to
  HEAD's 4; speech/panel UI 444 passed / 0 failed; library UI 723 passed / 18 failed vs
  HEAD's 721 / 20 — re-running the exact 18 node ids on both trees gives 18 failed on
  each, so the branch's set is a strict subset of HEAD's.
- `--collect-only` sweep: 56,876 tests collected, 4 collection errors, byte-identical to
  HEAD's (3x missing `playwright`, 1x pre-existing fixture/parametrize mismatch in
  `Tests/UI/test_library_file_notes_workspace.py`).
<!-- SECTION:NOTES:END -->
