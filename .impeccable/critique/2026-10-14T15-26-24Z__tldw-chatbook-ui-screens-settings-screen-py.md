---
target: Settings screen (Providers & Models + whole shell)
total_score: 21
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
target_identity: "file:/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/settings-ux-review-1ef4fcf4ae/tldw_chatbook/UI/Screens/settings_screen.py"
target_fingerprint: "sha256:6396d897cec62a0b816b5dacb85bc09d87c096c7bd2ce4071e67b7190cc69800"
target_path: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/settings-ux-review-1ef4fcf4ae/tldw_chatbook/UI/Screens/settings_screen.py
timestamp: 2026-10-14T15-26-24Z
slug: tldw-chatbook-ui-screens-settings-screen-py
---
# Design Critique — Settings ▸ Providers & Models + whole Settings screen (2026-10-14)

Method: dual-agent (A: a1f40d39594a8fa7b · B: a18cb8a551f19579f). Also: discovery debugger aeee0595c43dc5578, whole-Settings reviewer a04c687c9b3fb545b, context/P7 mapper a1251cdfef28b5770, adversarial verifiers aa41279c8ea214337 / a68643c4da2579896 / ab01f2055579cbcf9, completeness critic abb419ca3ddda9bda.
Target: origin/dev 1ef4fcf4ae (107347734d is +4 backlog-only commits). Live TUI, sandboxed scratch profile, 211x44 + 235x52, fake local OpenAI-compatible server. Mode: Operate.

## Discovery-save complaint: verified root causes
1. C1 (P1, CONFIRMED): rows always "session". Label from never-set `persisted` (settings_screen.py:16281, model_discovery_contracts.py:75); Settings never calls merge_saved_and_discovered_models (model_discovery_merge.py:146). Fixed on P7 #3057.
2. C2 (P1, CONFIRMED): Save selected writes [providers] list only; Model fill (empty-only) is a draft; Ctrl+Q drops it silently (quit guard is theme-editor-only, settings_screen.py:26444; Ctrl+Q asks the topmost screen only). Fixed on P7 (settings_quit_guard.py, DefaultModelPicker).
3. Seeds + top-3 (P2): placeholder seeds (config.py:4021) block the empty-Model fill; appended IDs fall outside Alt+M's TOP_MODELS_PER_PROVIDER=3 (console_model_popover.py:109). NOT fixed on P7.
4. C3 (P2, CONFIRMED live): manual persist appends to the startup snapshot (local_llm_provider_catalog_service.py:1019 → app_service_wiring.py:3457), overwriting later [providers] edits / resurrecting deletions; auto-refresh re-reads the file (:842-849). NOT fixed on P7.
Plus C4 (no-op save toasts "saved", Model stays empty; P7 fixes wording) and C-M2 (static "Capabilities unknown until saved" copy; not on P7).

## Design Health Score
Providers & Models: 20/40 — H1 2, H2 2, H3 3, H4 2, H5 2, H6 1, H7 2, H8 1, H9 3, H10 2.
Whole Settings: 21/40 — H1 2, H2 1, H3 3, H4 1, H5 3, H6 2, H7 3, H8 1, H9 3, H10 2.

## Design Specificity Verdict
Product-specific (banner, scope inspector, readiness vocab, Test Provider), but the screen presents the implementation's model of itself; the key task (choose and keep a model) is the weakest part. Detector: 0 findings (exit 0) over settings_screen.py, 22 settings_*.py and the settings TCSS; canary proved scanning; rules are web-only. No browser overlay (TUI).

## Priority Issues
1. [P1] Discovery doesn't save/show saved (C1, C2, C4) — land #3057 + honest receipts. /impeccable harden
2. [P2] Saved models lost/hidden (C3, seeds, top-3) — re-read file before append, refresh providers_models, ship [] seeds, lift session saves in Alt+M. /impeccable harden
3. [P1] Page structure: ~79 Tabs to Generation defaults behind 60 checkboxes; snapshots above Connect; Goal runs above "Start here" (row 39/44); 33 rail categories misfiled. /impeccable layout, /impeccable distill
4. [P1] Implementation jargon as copy (D4) + false inspector storage line (D5). /impeccable clarify
5. [P2] Instant controls unlabeled on draft pages; inspector says "staged" (D-M1; D1 pinned by TASK-1341). /impeccable harden

Other verified: provider picker clipping + dead Enter + stale "type in the open list" copy (from TASK-33002.3; fixed on P7); invisible Summary duration input (P7); invisible SelectionList cursor (TASK-33003.17, unpushed P9); inert Smooth scrolling / Palette limit + false Splash copy; Tab walks every rail button; red guards on dev (search index 29 unfindable settings; rail size 31 vs 33); Automatic refresh "Saved" on first view (P7); Overview shows draft as Configuration (unverified).
Retired/adjusted: A2, D1 → by design P2; D8 "nothing explains s" refuted (footer shows "Esc, s save category"); D2 placeholder pages 8 → 7; C5, C6 by design.

## Persona Red Flags
Jordan: reads "Discovered models saved." as "model set"; "session" jargon; page opens on Prompt-cache snapshots. Alex: 22 rail Tabs; dead Enter in provider search; ~12 keys to make a discovered model the default; inspector silent on Discover/Save selected/Test. Sam: invisible cursor on unticked rows; glyph-only rail state; ~7 fields per screen. Solo builder: placeholder vllm-model-z prefill; three forms of the endpoint URL; no "use this model".

## Minor Observations
Save selected enabled with nothing ticked; raw "Llama_cpp" in status; save result renders ~40 rows away; blank overrides hide inherited value; category name ×4; clipped label-less checkbox glyphs; User Guide opens with misplaced Goal policy citing F9.

## Questions to Consider
- If choosing a discovered model made it the default (list append as side effect), does Save selected need to exist?
- Two save models (Draft + Instant for one-shot actions) instead of nine badges?
- Do the seven view-only Domain Defaults pages earn rail rows?
- Is Console Behavior one category or four?
