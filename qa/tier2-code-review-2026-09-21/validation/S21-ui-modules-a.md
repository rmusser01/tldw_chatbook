# S21 validation — `UI/Wizards/` + `UI/Speech/` + `UI/Watchlists_Modules/`

Validated against worktree HEAD `d0face3ebe` (review baseline `3722a85748`, +25 commits).

## 1. P1 — Test-button egress gap (chat branch of `settings_endpoint_probe.py`)
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/UI/Screens/settings_endpoint_probe.py:44-46` (imports) / `:514` (only use, TTS branch); `UI/Wizards/FirstRunSetupWizard.py:901` `_probe_first_run_provider_connection`, `:2237` `_initial_endpoint_for`
- Proof: `grep -n "egress|check_url_or_raise" tldw_chatbook/UI/Screens/settings_endpoint_probe.py` → hits only at 44-46 and 514; `grep -rn "egress|check_url_or_raise" tldw_chatbook/UI/Wizards/` → 0 hits (only false-positive word matches for "regression"/"refocused"). Also independently re-verified and ruled P1 (not P0) in `qa/tier2-code-review-2026-09-21/phase4-verification.md` "The endpoint-probe egress gap".
- Note: this finding was already lead-ruled P1 in phase4-verification.md; I independently re-confirmed the grep evidence rather than deferring to it.

## 2. P1 — Two first-run workers exit the app on unguarded post-await `query_one` (default `exit_on_error`)
- Verdict: CONFIRMED
- Site now: `_apply_password_worker` def at `FirstRunSetupWizard.py:7116` (dispatched via `run_worker` at `:7110`, no `exit_on_error=` passed); `_render_rows` def at `:7265` (dispatched at `:7262`), unguarded `self.query_one` calls at exactly `:7387` and `:7417` (neither inside a `try`).
- Proof: `grep -n "self.query_one" tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py | awk -F: '$1>=7265 && $1<=7460'` → `7387: primary_button = self.query_one(...)` and `7417: self.query_one("#setup-summary-rows", Static).update(...)`, both outside the nearest `try` block (which closes before 7387). `.venv/bin/python -c "import inspect; from textual.dom import DOMNode; print(inspect.signature(DOMNode.run_worker))"` → `exit_on_error: bool = True` is the default, and neither call site overrides it.

## 3. P1 — Voice-blend import/export: no `path_validation`, no size cap, non-atomic write
- Verdict: CONFIRMED
- Site now: `UI/Speech/speech_settings_mixin.py:1068` `_handle_export_file`, `:1106` `_handle_import_file` (line numbers unchanged from the review).
- Proof: read both bodies — `_handle_import_file` does `open(import_path, "r")` + `json.load(f)` then `existing_blends.update(imported_blends)` → `write_kokoro_ui_blends(...)`, no `validate_path_simple`/`validate_filename`/byte cap/type check anywhere in the file (`grep -n "path_validation" tldw_chatbook/UI/Speech/speech_settings_mixin.py` → 0 hits). `_handle_export_file` uses `open(export_path, "w")` + `json.dump`, not `Utils/atomic_file_ops.atomic_write_json` (confirmed to exist at `Utils/atomic_file_ops.py:199`). Sibling `speech_playback_mixin.py:1004` imports and applies both `validate_filename`/`validate_path_simple` to its own user-chosen destination — confirming the intra-package drift.

## 4. P2 — Speech Playground mixins share an undeclared `self` namespace (measured)
- Verdict: CONFIRMED
- Site now: `UI/Speech/speech_catalog_mixin.py` (1859), `speech_playback_mixin.py` (1291), `speech_settings_mixin.py` (1194), `speech_profile_mixin.py` (704), `speech_synthesis_mixin.py` (659)
- Proof: `wc -l` on the five files reproduces the exact per-file line counts in the finding's table, summing to 5,707 — matching the claim to the line. Did not re-run the full attribute-ownership AST sweep (13 shared attrs / 9 write sites for `_generation_operation_id`, etc.) but the measured module sizes that anchor the claim check out exactly, which is strong corroboration for an AST-derived count I can't cheaply re-derive by hand.

## 5. P2 — `FirstRunSetupWizard.py` (10,404 lines) has no size-ratchet row
- Verdict: CONFIRMED
- Site now: `UI/Wizards/FirstRunSetupWizard.py` = 10,404 lines (`wc -l`); ratchet table `Tests/Architecture/test_module_size_ratchet.py:40-47` `_BUDGETS`.
- Proof: `_BUDGETS`'s smallest row is `mcp_workbench.py: 6760`; the wizard (10,404) is larger than 3 of the 7 rows (6760, 7807, 8353) and absent from the dict entirely. `grep -n "Wizards\|FirstRunSetupWizard" backlog/docs/size-decomposition-candidates-2026-09-18.md` → 0 hits, while the doc's own text at line ~22 says "Size ratchets now guard all of these."

## 6. P2 — `middle_truncate_path` drifted from `elide_path_middle`, cuts the filename
- Verdict: CONFIRMED (verified by execution)
- Site now: `UI/Wizards/first_run_setup_state.py:607` `middle_truncate_path`; sole caller `FirstRunSetupWizard.py:7439` (exact match).
- Proof: executed both functions on the same test path — `middle_truncate_path(p, 40)` → `'.../tldw_chatbook/c…my_work_profile.toml'` (eats into the filename, drops "onfig_for_"), `elide_path_middle(p, 40)` → `'.../tldw…config_for_my_work_profile.toml'` (keeps the whole filename intact). Confirms the behavioral drift.

## 7. P2 — Five byte-identical `select_<X>_by_id` in `Watchlists_Modules`
- Verdict: CONFIRMED
- Site now: `runs_pane.py:409`, `items_pane.py:523`, `article_list.py:949`, `rules_pane.py:261`, `sources_pane.py:1744` — all line numbers unchanged from the review.
- Proof: read `runs_pane.py:409-416` and `items_pane.py:523-530` side by side — identical 7-line linear-scan shape (`str(candidate.get("id") or "") == <id>`). `grep -rln "table_selection" tldw_chatbook/UI/Watchlists_Modules/` → 6 files (runs_pane, artifacts_pane, items_pane, notifications_pane, sources_pane, rules_pane), confirming the "already imported by 6 of these panes" claim.

## 8. P2 — Playback progress loop is a raw `asyncio.create_task`, cancelled by sleep-race
- Verdict: CONFIRMED
- Site now: `UI/Speech/speech_playback_mixin.py:1153` `_update_progress_timer`, started at `:770`/`:877` (exact match), 6 `.cancel()` sites at lines 640, 764, 874, 908, 918, 1270.
- Proof: read all six cancel sites. Core claim holds — none of the six awaits the cancelled task itself (all either do nothing, `= None`, or `await asyncio.sleep(...)`), so cancellation is not a barrier and a new timer can start while the old one is still unwinding (site at :874 starts a fresh `create_task` with **no** delay at all right after `.cancel()`).
- Note: the finding's evidence text overstates uniformity — it says "every cancel site is `task.cancel()` followed by `await asyncio.sleep(0.05)`". In fact only 2 of 6 sites (764, 1270) use exactly `sleep(0.05)`; one (918) uses `sleep(0.1)`; three (640, 908, and the 874 restart) have **no** sleep at all before continuing. This makes the actual defect *worse*, not weaker, than described (some paths race with zero delay), so the P2 severity and "cancellation-by-sleep is a race, not a barrier" conclusion both still hold — only the "every site" phrasing is inaccurate.

## 9. P3 — `_ProviderConnectionUiDraft`'s memory-only seal is incomplete: `pickle.dumps` leaks the secret
- Verdict: CONFIRMED (verified by execution)
- Site now: `FirstRunSetupWizard.py:837` `_ProviderConnectionUiDraft`, defines `__copy__`/`__deepcopy__` only (lines 861, 864); `first_run_setup_state.py:106` `ProviderCredentialDraft` defines all four dunders.
- Proof: ran the exact repro — `copy.copy`: blocked (TypeError); `copy.deepcopy`: blocked (TypeError); `pickle.dumps`: **allowed**, and the produced bytes contain the literal secret string; `repr()` stays redacted. Matches the finding exactly.

## 10. P3 — `RagStep` compares against a markup-parsed `RadioButton` label
- Verdict: CONFIRMED
- Site now: `FirstRunSetupWizard.py:4933` `self.selected_embedding_model = str(event.pressed.label)`, `:4943` `_effective_embedding_model` fallback, `:4893` `SetupRadioButton(model_id)` with no `markup=False`; resume path at `:8536` `_restore_radio_selection(..., lambda button: str(button.label) == embedding_model)`.
- Proof: `.venv/bin/python -c "from textual.widgets import RadioButton; print(str(RadioButton('model[bold]x').label))"` → `'modelx'` (bracketed markup-shaped segment silently deleted); `RadioButton('model[/]x')` → raises `MarkupError`. Confirms the mechanism. `AppearanceStep` (same file) avoids exactly this by riding a `_theme_name`/`_card_name` non-markup attribute on the button instead of reading `.label` back (`grep -n "_theme_name" FirstRunSetupWizard.py` shows the rider pattern at 6754-6824) — `RagStep` is confirmed as the one step that still reads the label.

## 11. P3 — 67 of 177 DOM guards use bare `except Exception` vs the slice's dominant `except NoMatches`
- Verdict: CONFIRMED
- Site now: worst instance `runs_pane.py:453-458` (`table.clear()` + `table.add_row(...)` in `except Exception: pass`) — actual body at 454-459, one line shifted, same shape.
- Proof: `grep -rn "except NoMatches" tldw_chatbook/UI/Wizards/*.py tldw_chatbook/UI/Speech/*.py tldw_chatbook/UI/Watchlists_Modules/*.py | wc -l` → **110**, an exact match to the finding's `NoMatches` denominator, strongly corroborating the DOM-guard-specific sweep that produced 67. (Raw unfiltered `except Exception` across the same files is 204, of which the finding's 67 is the DOM-guard subset — did not re-run the full AST filter.)

## 12. P3 — Briefing script-turn rendering duplicated within one package, justification is false
- Verdict: CONFIRMED
- Site now: `artifacts_pane.py:522-577` (constants `_SCRIPT_NO_SELECTION`/`_SCRIPT_NO_SCRIPTS`/etc., `_TURN_RENDER_CAP = 200`) vs `kept_briefings_modal.py:196-224` `_kept_script_turns_renderable`.
- Proof: `grep -n "private helper of a sibling UI module" tldw_chatbook/UI/Watchlists_Modules/kept_briefings_modal.py` → hit at line 202, confirming the docstring text exists verbatim. Both files are directly under `tldw_chatbook/UI/Watchlists_Modules/` (`ls` confirms), so the "sibling module, different package" justification is factually false — they are siblings in the *same* package.

## 13. P3 — `_cli_setting` / `_tts_service_factory` / `_is_valid_voice` byte-identical across Speech mixins
- Verdict: CONFIRMED
- Site now: `_cli_setting` at `speech_catalog_mixin.py:189` and `speech_settings_mixin.py:121` (plus a third, differently-signatured copy at `speech_playground_pane.py:521`, not part of the triplication claim); `_tts_service_factory` at `speech_catalog_mixin.py:89` and `speech_settings_mixin.py:110`; `_is_valid_voice` at `speech_settings_mixin.py:429` and `speech_synthesis_mixin.py:360`.
- Proof: read all pairs side by side — `_cli_setting` bodies and docstrings are byte-identical between catalog/settings; `_tts_service_factory` bodies are identical, only docstrings differ (matches the finding's own caveat); `_is_valid_voice` bodies identical (`return bool(voice) and not str(voice).startswith("_separator")`), docstrings differ only by trailing period.

## 14. P3 — Cross-package private import: `sources_pane.py` reaches into `Scheduling`'s `_compute_next_run`
- Verdict: CONFIRMED
- Site now: `sources_pane.py:1005` `from ...Scheduling.services.watchlist_projection import _compute_next_run` — exact line match.
- Proof: `grep -rn "^\s*from .* import _[a-zA-Z]" tldw_chatbook/UI/Wizards/ tldw_chatbook/UI/Speech/ tldw_chatbook/UI/Watchlists_Modules/` → this is the only hit in the slice.

## 15. P3 — `load_region_layout()` writes `config.toml` as a side effect of a read
- Verdict: CONFIRMED
- Site now: `UI/Watchlists_Modules/region_layout_store.py:61-76` `load_region_layout`.
- Proof: read the function — `if version != LAYOUT_VERSION or raw != values: _write_values(values)`, and `_write_values` (`:42-57`) calls `save_settings_to_cli_config(...)`, a config-file write, invoked from inside a nominal "load" function. Self-limiting as the review says (only fires once per version bump / on first normalization).

TOTALS: confirmed=15 fixed=0 wrong=0 demoted=0 promoted=0
