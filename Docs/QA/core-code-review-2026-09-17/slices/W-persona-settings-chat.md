# W-persona-settings-chat — Widgets/Persona_Widgets, Widgets/Settings_Widgets, Widgets/Chat_Widgets — 61 files, 30734 lines

## Coverage

| file | lines | coverage |
|---|---|---|
| `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` | 6014 | read in full 1-3640 + 3768-3930, 3965-4230, 4444-4535, 5245-5500, 5630-5670, 5810-5935; symbol-scanned (def/@on list) 3640-6014; grep-verified for I/O, threading, except, config reads, run_worker over the whole file |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` | 1993 | read in full 1-120, 380-470, 600-760, 1080-1320, 1350-1450, 1450-1530, 1610-1710; symbol-scanned rest; grep-verified over the whole file |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py` | 1529 | read in full (1-1529) |
| `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py` | 1467 | read in full 1-110, 195-262, 795-840, 1025-1300, 1350-1467; symbol-scanned rest; grep-verified over the whole file |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py` | 1287 | read in full 255-300, 390-530, 595-660, 900-935, 1230-1287; symbol-scanned rest; grep-verified over the whole file |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` | 1278 | symbol-scanned (def list) + read in full 1190-1280; grep-verified over the whole file |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py` | 896 | mechanical: clone-normalised diff vs personas_lore_detail.py + candidate-row reads (513, 734-741, 763, 809-853) |
| `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py` | 852 | read in full 218-236, 290-345, 750-805; symbol-scanned rest |
| `tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py` | 807 | mechanical only (candidate rows 27, 222/413/593/791, 423/598/796) + I/O grep |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_message_enhanced.py` | 769 | read in full (1-769) |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` | 725 | candidate-row read 198-225 + grep |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` | 695 | mechanical: clone-normalised diff vs personas_dictionary_detail.py + candidate-row reads (324, 357, 568-573, 635-655) |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` | 638 | read in full 103-200, 555-638 |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py` | 614 | mechanical only (candidate rows) + grep |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_management_modal.py` | 605 | read in full 318-345, 445-475; rest mechanical |
| `tldw_chatbook/Widgets/Persona_Widgets/petdex_import_review.py` | 529 | read in full 230-250, 365-380, 485-525; rest mechanical |
| `tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py` | 524 | mechanical only (I/O + run_worker + except greps) |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py` | 507 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` | 506 | candidate-row reads 226, 267, 345-360 |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_character_review.py` | 459 | read in full 250-270, 380-390, 440-450; rest mechanical |
| `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_panel_types.py` | 427 | mechanical only (class-attr + dataclass scan) |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py` | 421 | mechanical only |
| `tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py` | 384 | mechanical only (candidate row 32) |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_message.py` | 366 | mechanical only + instantiation census (see the dead-widget finding) |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py` | 364 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_conversation_modal.py` | 344 | read in full (1-344) |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` | 315 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/actor_pack_import_review.py` | 315 | read in full 15-45, 200-250; rest mechanical |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py` | 297 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py` | 263 | candidate-row read 140-165 |
| `tldw_chatbook/Widgets/Settings_Widgets/server_switch_modal.py` | 261 | read in full (1-261) |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py` | 258 | candidate-row read 210-230 |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py` | 253 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py` | 234 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_conversation_transcript_widget.py` | 225 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py` | 218 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_dictionaries.py` | 215 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py` | 210 | mechanical only |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py` | 210 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_workspace_modal.py` | 204 | read in full 160-200; rest mechanical |
| `tldw_chatbook/Widgets/Persona_Widgets/character_tts_portability_dialogs.py` | 184 | mechanical only (candidate rows 99, 104-109, 177-182) |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py` | 181 | mechanical only |
| `tldw_chatbook/Widgets/Chat_Widgets/skill_script_confirm_card.py` | 166 | mechanical only |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py` | 139 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py` | 138 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py` | 134 | read in full (clone-normalised diff + class/handler reads) |
| `tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py` | 131 | read in full (clone-normalised diff + class/handler reads) |
| `tldw_chatbook/Widgets/Persona_Widgets/dictionary_attach_picker.py` | 122 | read in full (clone-normalised diff + class/handler reads) |
| `tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py` | 122 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py` | 121 | read in full (clone-normalised diff + class/handler reads) |
| `tldw_chatbook/Widgets/Chat_Widgets/worktree_confirm_card.py` | 119 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/tag_filter_picker.py` | 118 | read in full (clone-normalised diff + class/handler reads) |
| `tldw_chatbook/Widgets/Chat_Widgets/worktree_recovery_dialog.py` | 118 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_validation.py` | 112 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py` | 109 | mechanical only |
| `tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py` | 100 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_speech_controls.py` | 96 | read in full (1-96) |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_resume_panel.py` | 28 | mechanical only |
| `tldw_chatbook/Widgets/Persona_Widgets/__init__.py` | 17 | read in full (trivial) |
| `tldw_chatbook/Widgets/Settings_Widgets/__init__.py` | 1 | read in full (trivial) |
| `tldw_chatbook/Widgets/Chat_Widgets/__init__.py` | 0 | read in full (trivial) |

## Findings   (ordered P0→P3, then D1→D4)

_No P0. 1×P1, 6×P2, 7×P3. Retired candidates and verified-fine reads follow the findings and are repeated in their own contract sections._

### P1 [D2] — opening Settings ▸ Speech & TTS spends 65 ms re-reading cached config on the UI thread
- Where: `Widgets/Settings_Widgets/speech_tts_settings_panel.py:878-882` (`SpeechTTSSettingsPanel.__init__`, the `restored is None` branch) → `_read_realtime_settings_draft()` (:291-320) and `_read_pipeline_voice_settings_draft()` (:322-341). The panel is constructed from a `yield` inside the settings screen's category render: `UI/Screens/settings_screen.py:20063`.
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'PY'
import time
from tldw_chatbook.Widgets.Settings_Widgets import speech_tts_settings_panel as p
from tldw_chatbook.config import get_cli_setting
get_cli_setting("dictation", "response_eagerness_ms", 0)      # warm the cache
t=time.perf_counter(); p._read_realtime_settings_draft();  print(f"realtime {(time.perf_counter()-t)*1000:.2f} ms")
t=time.perf_counter(); p._read_pipeline_voice_settings_draft(); print(f"pipeline {(time.perf_counter()-t)*1000:.2f} ms")
t=time.perf_counter()
for _ in range(20): get_cli_setting("dictation","response_eagerness_ms",0)
print(f"per get_cli_setting {(time.perf_counter()-t)*1000/20:.3f} ms")
PY
```
  → `realtime draft (warm): 54.64 ms` · `pipeline draft (warm): 10.33 ms` · `per get_cli_setting: 5.427 ms` (≈12 reads × 5.4 ms). Corroborates the sibling's "config reads are not free even warm" measurement independently.
- Why it matters: 65 ms of blocking work on the UI thread every time the Speech & TTS category is opened with no draft snapshot, purely to re-read a config the loader has already cached. The panel needs one settings snapshot, not twelve point lookups.
- Recommended correction: read the config once (`load_settings()` / `get_runtime_config_snapshot()` — the settings screen already holds one at `settings_screen.py:20054`) and pass the mapping into both draft readers, the way `_read_pipeline_voice_settings_draft` already does internally (:337 builds `config = {"dictation": section}` and hands it to `response_eagerness_ms(config)` — the helpers already accept a mapping).
- Size: S · ADR: no · Confidence: verified (measurement above)
- Pinning test: none.
- Already covered: none (task-1378/task-31202 cover `settings_screen.py` size, not this panel's config reads).


### P2 [D3] — `chat_message.py` + `chat_message_enhanced.py` (1,135 lines) have no production mount site
- Where: `Widgets/Chat_Widgets/chat_message_enhanced.py` (769), `Widgets/Chat_Widgets/chat_message.py` (366); the app-side consumers `app.py:14666-14690, 14723, 14783, 14812` (`self.query(ChatMessage) + self.query(ChatMessageEnhanced)` in the TTS complete/progress handlers).
- Evidence:
  - `grep -rn "ChatMessageEnhanced(" tldw_chatbook/ Tests/` → exactly one production construction site: `UI/CCP_Modules/ccp_message_manager.py:173`. Everything else is the class definition or `Tests/`.
  - `grep -rn "CCPMessageManager\|ccp_message_manager" --include="*.py" tldw_chatbook Tests` → in `tldw_chatbook/` the name appears only as the class definition (`:26`), its own `logger.bind` (:12) and the package re-export (`UI/CCP_Modules/__init__.py:25,72`). The only construction anywhere is `Tests/UI/test_ccp_handlers.py:415`.
  - `grep -rnE "(^|[^a-zA-Z_])ChatMessage\(" tldw_chatbook --include="*.py" | grep -v ConsoleChatMessage` → only the two class definitions. Nothing mounts the legacy widget either.
  - `grep -rn "CCP_Modules" --include="*.py" tldw_chatbook | grep -v "UI/CCP_Modules/"` → `personas_screen.py:344-348` imports `ccp_character_handler`, `ccp_enhanced_handlers`, `ccp_messages`, `ccp_persona_handler` — **not** `ccp_message_manager`.
- Why it matters: 1,135 lines of widget code that reads as live (it is imported lazily in `app.py` with a `TASK-21103` boot-cost comment, and has two test suites) but can never be mounted; `chat_message_enhanced` is also the reason `PIL`/`rich_pixels`/`textual_image` are kept on a lazy-import leg. The app-side TTS branches that target it are inert.
- Caveat on the "dead" claim: `Tests/UI/test_legacy_entrypoints_retired.py:155-159` lists `ccp_message_manager.py` under `CCP_HANDLER_FILES` — "reused CCP handlers" — and asserts only that its *source text* names `PersonasScreen`. That test does not assert it is wired, and nothing wires it.
- Recommended correction: either wire `CCPMessageManager` into `PersonasScreen` (the intent the test records) or retire both widgets plus the `app.py` query branches through the same `RETIRED_FILES` mechanism `task-577` used.
- Size: M · ADR: no · Confidence: verified (greps above)
- Pinning test: `Tests/Widgets/test_chat_message_enhanced.py`, `Tests/Backup_Recovery/test_recovered_media.py:176` — both construct the widget directly, so they pass whether or not production mounts it.
- Already covered: none


### P2 [D3] — `speech_tts_settings_panel.py:5438` is the package's only `run_worker(exclusive=True)` with no `group=`
- Where: `Widgets/Settings_Widgets/speech_tts_settings_panel.py:5438-5440` — `self.run_worker(self._rebuild_after_custom_id(axis), exclusive=True, exit_on_error=False)`
- Evidence: `grep -rn "run_worker(" Widgets/{Persona_Widgets,Settings_Widgets,Chat_Widgets}` → 28 sites; this is the only one without `group=`. The other 5 in this same file all name a group (`_AUDIO_CPP_PACKAGE_SAVE_VALIDATION_GROUP` :4131, `settings-speech-provider-leave` :5256, `settings-speech-open-lab` :5487/:5925, `_AUDIO_CPP_PACKAGE_SCAN_GROUP` :5648). Textual's default is `group="default"` (`.venv/.../textual/worker_manager.py:87`) and `exclusive` cancels every worker in that group **on the same node** (`worker_manager.py:75-76` → `cancel_group(worker.node, worker.group)`).
- What a collision does, concretely: the two axes share the group. Confirm a custom **model** id, then a custom **voice** id before the first rebuild completes, and the voice worker cancels the model worker mid-`await card.recompose()` (`_replace_card_bodies`, :2211-2247). `_rebuild_after_custom_id`'s `finally` then discards `"model"` from `_custom_id_rebuild_pending` (:5465) — the fence whose documented job (:5411-5419, :3652-3654) is to stop `_collect_visible_state` reading the stale mounted Select. With the fence gone and the Select not yet rebuilt, the next collection reads the old value over the confirmed custom model id.
- Why it matters: the file's only ungrouped exclusive worker is also the one guarding a "never lose the confirmed value" fence.
- Recommended correction: `group="settings-speech-custom-id"` (or per-axis, `group=f"settings-speech-custom-id-{axis}"`, which removes the cross-axis cancel entirely).
- Size: S · ADR: no · Confidence: inferred (the group semantics and the `finally` are verified from source; the end-to-end value loss is not reproduced — settling it needs a Textual `run_test` driving two `_custom_id_modal_result` calls back to back)
- Pinning test: none.


### P2 [D4a+b] — five near-identical list-picker modals in one directory; three of them skip the canonical dismiss mixin
- Where: `Persona_Widgets/dictionary_picker.py` (134), `world_book_picker.py` (131), `dictionary_attach_picker.py` (122), `conversation_attach_picker.py` (121), `tag_filter_picker.py` (118) — 626 lines total, one widget shape.
- Evidence (clone measurement): normalising each file (drop comments/blank lines, rewrite every domain noun — dictionary/world_book/conversation/tag — to `X`) and diffing:
```
dictionary_attach_picker vs conversation_attach_picker : 18 differing lines out of 95
dictionary_picker        vs world_book_picker          : 39 differing lines out of 108
```
  Every one of the 18 differences in the first pair is the class name, the docstring, or a CSS type selector — **no logic differs at all**. `conversation_attach_picker.py`'s own docstring says so: "Generic — used by the Roleplay Lore Attachments flow (P2e); the dictionary flow keeps its own DictionaryAttachPicker."
- The drift that makes this more than tidiness: `grep -n "^class \|SafeModalDismiss\|dismiss" *picker*.py` →
  - `DictionaryPicker` (:28) and `WorldBookPicker` (:29) inherit `SafeModalDismissMixin` and cancel via `dismiss_safe_once(None)`;
  - `DictionaryAttachPicker` (:25), `ConversationAttachPicker` (:24) and `TagFilterPicker` (:27) inherit plain `ModalScreen` and cancel via raw `self.dismiss(None)`.
  `Widgets/modal_dismissal.py` has **82 importers** (`grep -rl SafeModalDismissMixin tldw_chatbook/ | wc -l`). What the three lose is not cosmetic: `dismiss_safe_once` (:270-299) is the one-shot guard (`_safe_dismiss_committed`), the topmost-screen check (`app.screen is not self`), the backdrop-click shield, **and opener-focus restoration** (`_restore_focus_after_dismissal`). A keyboard user cancelling `ConversationAttachPicker` does not get focus back on the button that opened it; cancelling `WorldBookPicker` does.
  Secondary asymmetry inside the two compliant pickers: `_confirm` still calls raw `self.dismiss(self._selected_id())` (`dictionary_picker.py:126`, `world_book_picker.py:123`) while only `_cancel` goes through the mixin — so the double-press guard covers Cancel and not Confirm.
- Recommended correction: one parameterised `ListPickerModal(rows, *, id_key, title, search_placeholder)` in `Widgets/` (next to `modal_dismissal`), inheriting `SafeModalDismissMixin`, with the five call sites passing their id key and copy. Failing that, at minimum put the mixin on the three that lack it and route `_confirm` through `dismiss_safe_once` in all five.
- Size: M · ADR: no · Confidence: verified (diffs + greps above)
- Pinning test: none found asserting focus restoration for these pickers.
- Already covered: none. (The brief's note that the 2-line `_cancel` adapters are P3 holds for the *adapters*; this is the class-level drift it exempts.)

### P2 [D4a] — the Settings "Test" button in `server_switch_modal.py` sends the user's API token to a user-entered URL without the app's egress gate
- Where: `Widgets/Settings_Widgets/server_switch_modal.py:207-232` (`_run_connection_test`): `async with httpx.AsyncClient(timeout=5.0) as client: reach = await client.get(f"{url}/docs")`, then `await client.post(f"{url}/api/v1/sync/send", headers={"X-API-KEY": token}, json={})`.
- The helper that exists and is ignored: `Utils/egress.py` (`check_url_or_raise_async(url, trusted_origins=origin_set(origin))`), whose module docstring states the exact rule for this case — "OR its hostname is in `trusted_origins` (a host the USER explicitly typed/configured) … Metadata endpoints are stricter: blocked even for trusted origins". Its adopter count is 20+ modules; the **sibling settings probe for the same kind of action** does it correctly: `UI/Screens/settings_endpoint_probe.py:43-47, 513-522` calls `check_url_or_raise_async(..., trusted_origins=origin_set(endpoint.origin))` and constructs the client with `follow_redirects=False` (:526-529).
- Evidence: `grep -rln "Utils.egress" tldw_chatbook/` → 20+ modules, `server_switch_modal.py` not among them; `grep -n "egress\|guarded_fetch\|trusted_origin" Widgets/Settings_Widgets/server_switch_modal.py` → no hits.
- What the modal *does* validate (`_validated_root_url`, :162-192): scheme ∈ {http, https}, non-empty netloc, no path/query/fragment, then `Utils/input_validation.validate_url` — which by its own docstring validates *shape* only ("long TLDs, IPv6 literals, IDN/Unicode hosts, IPs, and localhost all validate"). It has no IP-class or metadata-endpoint rule; that is what `egress` is for.
- Why it matters: this is a credential-bearing outbound request (`X-API-KEY: <the user's token>`) to a host the modal never runs past the app's own network policy, from a settings surface whose direct sibling does. Mitigating: httpx defaults to `follow_redirects=False`, so a redirect cannot relocate the token.
- Recommended correction: one `await check_url_or_raise_async(url, trusted_origins=origin_set(origin_of(url)))` before the client block, and construct the client `follow_redirects=False` explicitly, matching `settings_endpoint_probe`.
- Size: S · ADR: no · Confidence: verified (mechanism; no exploit attempted)
- Pinning test: none in `Tests/UI/test_settings_endpoint_probe.py` covers this modal.
- Already covered: task-586 (image-gen) and task-609 (skill remote fetch) name other egress adopters, not this one.


### P2 [D4b] — the "mount a prepared thumbnail renderable" block is copy-pasted 3× in this slice (4× in the repo) with no helper
- Where: `Persona_Widgets/personas_inspector_pane.py:1194-1236` (`set_avatar_thumbnail`), `Persona_Widgets/personas_character_editor_widget.py:1266-1310` (`set_avatar_thumbnail`) and `:1350-1392` (`set_expression_thumbnail`). The fourth is `UI/Screens/chat_screen.py:12749-12785` (`_build_character_avatar_widget`), which the other three name in their own comments ("same fallback as `ChatScreen._build_character_avatar_widget`").
- Evidence: `grep -rn "explicit_cell_size" tldw_chatbook/ | grep -v Utils/mosaic_render.py` → 4 distinct call sites (`chat_screen.py:12777`, `personas_inspector_pane.py:1226`, `personas_character_editor_widget.py:1297` and `:1379`). Each is preceded by the same `holder.remove_children()` → `isinstance(renderable, Widget)` short-circuit → `Static(renderable)` → `explicit_cell_size(...) or (BOX_COLS, BOX_LINES)` sequence, and three of them carry a verbatim copy of the same 4-line comment ("Per explicit_cell_size's documented contract, fall back to the box dimensions…").
- Behavioural drift to note: the two in `personas_character_editor_widget.py` re-import `Widget` and `Static` **inside the function** (`:1284-1285`, `:1368-1369`) even though both are already imported at module scope (`:22`, `:23`); the inspector copy re-imports only `Widget` (:1214) and uses the module-level `Static`. Same behaviour, three spellings.
- Recommended correction: one `mount_thumbnail(holder, renderable, *, box_cols, box_lines)` in `Utils/mosaic_render.py`, which already owns `explicit_cell_size` and its documented `None` contract. All four call sites pass only the holder and the box constants.
- Size: S · ADR: no · Confidence: verified
- Already covered: none.


### P2 [D4b] — three drifted inline tail-truncates in this slice; the shared helper `Utils.Utils.truncate_content` has ZERO importers repo-wide
- Where (my slice): `Persona_Widgets/personas_character_editor_widget.py:1691` (`first[:60] + "…"`), `Persona_Widgets/personas_lore_detail.py:324` (`[:57] + "..."`), `Persona_Widgets/personas_preview_pane.py:267` (`[:39] + "…"`).
- Evidence:
  - `grep -rn "truncate_content" . --exclude-dir=.git` → **one hit**, its own definition at `tldw_chatbook/Utils/Utils.py:253`. No importer in `tldw_chatbook/` or `Tests/`; `Utils/Utils.py` has no `__all__` and `Utils/__init__.py` has no star-import, so nothing can be reaching it indirectly.
  - `grep -rEn "\[: *[0-9]+ *\] *\+ *(\"|')(…|\.\.\.)" tldw_chatbook` → **38** inline re-rolls across the tree.
- The drift: the three copies in my slice disagree on the ellipsis character (`…` vs `...`), on whether the budget includes the ellipsis (`truncate_content` reserves 3 chars, `[:57] + "..."` makes 60, `[:60] + "…"` makes 61), and on the `len <= max` short-circuit. Two of the three are user-visible list previews sitting next to each other in the same workbench.
- Why it matters: the rubric's "dead helper with ≥10 re-rolls" case, at 38. This is a repo-wide finding — I am reporting the three copies inside my slice and the census; whoever owns `Utils/` should decide whether the helper is adopted or deleted.
- Recommended correction: canonical home is the existing `Utils/Utils.py:truncate_content` (fix its off-by-one if the `…` single-char form is preferred), or delete it and stop pretending a helper exists.
- Size: M (repo-wide) · ADR: no · Confidence: verified
- Already covered: none.


### P3 [D1] — `ChatMessageEnhanced.watch_tts_state` calls `self.refresh()` where its own comment says "recompose"
- Where: `Widgets/Chat_Widgets/chat_message_enhanced.py:594-600` (`# Force a recompose of the action buttons` / `self.refresh()`); same shape in `mark_generation_complete` (:704-708).
- Why it matters: `compose()` branches the whole TTS button set on `self.tts_state` (:388-450). `Widget.refresh()` repaints; it does not re-run `compose` (only `refresh(recompose=True)` / a `reactive(..., recompose=True)` does — and `update_variant_info`'s own fallback at :766 uses exactly that spelling). So a TTS state change would never swap 🔊 → ⏳ → ▶️.
- Severity is P3 only because of the finding above: nothing mounts this widget in production, so no user can reach it. If it is ever wired, this becomes a P1.
- Recommended correction: `self.refresh(recompose=True)` in both watchers, or make `tts_state` a `reactive(..., recompose=True)`.
- Size: S · ADR: no · Confidence: inferred (not reproduced — the literal command that would settle it is a Textual `run_test` mounting `ChatMessageEnhanced`, flipping `tts_state`, and asserting `#tts-play` appears)
- Pinning test: none in `Tests/Widgets/test_chat_message_enhanced.py` asserting post-state button ids.


### P3 [D1] — `summarize_arguments` silently skips redaction when `arguments` is not a mapping
- Where: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py:462-467` (`except Exception: return str(arguments or {})[:_ARGS_SUMMARY_LIMIT]`)
- Evidence: `cd $WT && source env.sh && PYTHONPATH=$WT $PY -c` on `summarize_arguments`:
  - `{"api_key": "sk-live-SECRET-123", "path": "/tmp/x"}` → `{"path":"/tmp/x","api_key":"***"}` (redacted, destination hoisted — correct)
  - `'{"api_key": "sk-live-SECRET-123"}'` (a **str** payload) → `{"api_key": "sk-live-SECRET-123"}` — **rendered verbatim, unredacted**, on the card whose whole job is to show the user what they are approving.
  - Same for the collapsed path: `summarize_row_arguments({"all_arguments": ['{"token": "ghp_SECRET"}']})` → `{"token": "ghp_SECRET"}`.
- Why it matters: the `except Exception` fallback exists so "a bad arg must never crash rendering", but it also disables the module's stated invariant ("Redaction runs BEFORE the reordering and clipping below, so neither can expose a secret"). A JSON-string `arguments` payload is the single most common provider shape (OpenAI emits `function.arguments` as a string).
- Reachability (why this is P3, not P0): the only production producer coerces first — `Chat/console_chat_controller.py:2043` `"arguments": dict(call.arguments or {})` — and the only two `set_batch` callers (`Widgets/Chat_Widgets/chat_task_cards.py:220`, `Widgets/Persona_Widgets/buddy_conversation_modal.py:168`) both feed from that payload (`grep -rn "set_batch(" tldw_chatbook/`). So today the string shape cannot reach the card; the gap is defence-in-depth that reads as a guarantee.
- Recommended correction: in the `except` branch, run the fallback text through `tldw_chatbook.MCP.redaction`'s value-level scrubber (or `redact_mapping({"value": text})`) rather than returning the raw text; alternatively `json.loads` a str payload before giving up.
- Size: S · ADR: no · Confidence: verified (probe above)
- Pinning test: none found asserting the fallback's text.
- Already covered: none


### P3 [D1] — every personal-data failure path in the My Profile surfaces is caught, toasted, and never logged
- Where: `Widgets/Settings_Widgets/personal_context_panel.py:203, 247, 811, 976, 987, 1004, 1225, 1229, 1460` and `personal_context_review_modal.py:324, 778` — each `except Exception:` calls `self.notify(...)` / `_unknown_failure()` with deliberately non-specific copy ("The profile change could not be saved. Private details were not displayed.", "Export failed…", "The review action failed…").
- Evidence: `grep -n "logger\|loguru\|logging" Widgets/Settings_Widgets/personal_context_panel.py` → **no hits**; same for `personal_context_review_modal.py`. `grep -c "logger\." Personal_Context/service.py Personal_Context/export_service.py` → **0** and **0**, so the service layer under them does not log either. `Personal_Context/interview_launch.py:195, 201, 225, 234` in the same package *does*, with exactly the shape this needs: `logger.opt(exception=True).warning("…")`.
- Why it matters: a failed profile write, a failed key deletion (`_removal_incomplete`) or a failed **encrypted recovery export** produces a toast and nothing else. Nobody — user or maintainer — can find out why. The non-disclosure in the toast copy is a deliberate privacy decision; it does not require the exception type to be discarded too.
- Recommended correction: `logger.opt(exception=True).warning("<operation> failed")` beside each `call_from_thread(...)` notify, matching `interview_launch.py`. Payload/PII stays out by construction — the exception object, not the record, is what is logged; `Utils/log_sanitizer.py` is available if a message ever carries a value.
- Size: S · ADR: no · Confidence: verified (greps above)
- Pinning test: none.


### P3 [D3] — `chat_message_enhanced.py` uses stdlib `logging` and an `except (ImportError, TypeError, Exception)` optional-dep guard
- Where: `:9` `import logging` (24 call sites in the file); `:28-34` module-scope `try: from textual_image.widget import Image` / `except (ImportError, TypeError, Exception)`.
- Why it matters: the repo logs through loguru (`Widgets/recompose_capture_guard.py:48` is the neighbouring convention); `except Exception` makes the other two members of the tuple dead, and the module never consults `Utils/optional_deps.py`, so `TEXTUAL_IMAGE_AVAILABLE` is a second private availability flag. `PIL` and `rich_pixels` are imported at module scope **unguarded** (:15-16) — they are hard deps in `pyproject.toml:48,59`, so that is correct today, but it is inconsistent with the guarded sibling on the same line of thinking.
- Size: S · ADR: no · Confidence: verified (read + `grep pillow/rich-pixels pyproject.toml`)
- Already covered: task-25704/task-287 (`DEPENDENCIES_AVAILABLE` flags never populated) covers the adjacent flag problem.


### P3 [D3] — `speech_tts_settings_panel.py` is a 6,014-line god module outside every size ratchet
- Where: `Widgets/Settings_Widgets/speech_tts_settings_panel.py` (6,014 lines, 1 panel class + 7 modal/helper classes).
- Responsibilities in the one file: 6 `ModalScreen` dialogs (credential set/clear, OpenAI plaintext consent, custom id, leave-guard) · the panel's 5 compose bodies (global defaults, provider setup, pipeline/conversation, realtime, inspector) · a per-provider form builder covering 8 providers (`_compose_provider_form`, :3012-3610) · draft collection/validation/revision bookkeeping · the audio.cpp guided-package review, scan, lease and cleanup lifecycle · realtime/dictation config persistence · save/revert/restore-defaults orchestration · focus tokens and restore · responsive layout.
- Evidence: `wc -l` = 6014; `grep -n "speech_tts\|Settings_Widgets" Tests/Architecture/test_screen_size_ratchet.py Tests/Architecture/test_library_modules_size_ratchet.py` → **no hits** — the two one-way ratchets cover `UI/Screens/` and `UI/Library_Modules/` only, so nothing stops this file growing.
- Recommended correction: the split shape is settled — `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) and §17 (controller-file size governance). The obvious first seam is the audio.cpp guided-package subsystem (scan/lease/cleanup/review copy, ~1,100 lines) and the 6 modals.
- Size: L · ADR: no (recipe covers it) · Confidence: verified
- Already covered: task-1378 and task-31202 cover `settings_screen.py`, not this panel.


### P3 [D3] — four function-body imports of modules already imported at module scope
- Where: `personas_character_editor_widget.py:1284-1285` and `:1368-1369` — `from textual.widget import Widget as _W` / `from textual.widgets import Static as _S`, while `:22` and `:23` already import both at module scope.
- Why it matters: per-call `import` of a non-optional module, with an alias that hides the duplication from a reader. (The file's *other* function-body imports are legitimate and documented: `...UI.Screens.personas_screen` at :1423 carries a circular-import comment; `...Utils.mosaic_render` at :1290/:1374 is a real deferral.)
- Size: S · ADR: no · Confidence: verified (`grep -n "^from textual"` → :22, :23)


### P3 [D3] — the runtime imports a Textual widget module for a policy constant
- Where: `chat_approval_card.py:386` defines `TOOL_DESCRIPTION_CAPTURE_CAP`; module-scope importers are `Chat/console_chat_controller.py:47`, `Chat/permission_summary_service.py:15`, `Agents/mcp_tool_provider.py:77`, `Agents/local_tool_provider.py:45`.
- Evidence: `grep -rn "TOOL_DESCRIPTION_CAPTURE_CAP" tldw_chatbook/` → 9 hits, 4 of them cross-package imports out of `Widgets/`.
- Why it matters: `Agents/` and `Chat/` now depend on a widget module at import time for a non-UI egress bound.
- Status: **documented and deliberate** — the module's own header (:378-385) records that these helpers lived in `Chat/approval_display.py` and were folded back because the extra module broke the `_ui_ready` census ratchet (ADR-097 / TASK-23029). Cite, do not re-file.
- Size: S · ADR: yes (ADR-097) · Confidence: verified


## Retired / verified-fine reads (detail; summarised again below)

### RETIRED CLUSTER — all 28 `query_one_in_timer_no_try` candidates in this slice
- Where: `Widgets/Persona_Widgets/buddy_conversation_modal.py:109-206` (17), `buddy_speech_controls.py:56-77` (6), `persona_profile_editor_widget.py:593,599` (2), `personas_character_editor_widget.py:1482,1494` (2)
- Evidence (three independent legs):
  1. Textual stops a pump's timers before it can dispatch: `.venv/lib/python3.12/site-packages/textual/message_pump.py:528-535` `_close_messages` → `await Timer._stop_all(self._timers); self._timers.clear()`, and again at `:577-579` in `_process_messages`' `finally`. Every timer here is created with `self.set_interval` / `self.set_timer` on the widget that owns the queried ids, so it cannot fire after that widget unmounts.
  2. Every queried id is composed unconditionally. `grep -n "def compose" personas_character_editor_widget.py` → 400; `awk 'NR>=440&&NR<=710 && /  if |  else:|  elif /'` → **no output** (no conditional in the compose body that yields `#personas-char-editor-name`, `-avatar-status`, `-pack-status`). Same for `persona_profile_editor_widget.py:103-172` (`#personas-editor-name`, `#personas-editor-character-portrait` — the portrait container is `display=False`, which `query_one` does not filter on). `buddy_speech_controls.py:27-33` yields all 5 ids flat. `buddy_conversation_modal.py:66-94` yields all ids flat except `#buddy-mic`, whose only query (`:206`) is already inside `if self.allow_voice:` — the same flag that gates the `yield`.
  3. `grep -n recompose` in all four files → no hits; none of these widgets recompose.
  Additionally `buddy_conversation_modal.refresh_projection` opens with `if not self.is_mounted or not self._visible: return` (:107) and `on_unmount` stops the timer explicitly (:332-337).
- Disposition: **retired, all 28**. Symptom shape real, cause absent here.
- Confidence: verified (Textual source + compose reads).


### RETIRED — the `run_worker(sync callable, no thread=True)` P0 shape does not occur in this slice
- Evidence: `grep -rn "run_worker(" Widgets/{Persona_Widgets,Settings_Widgets,Chat_Widgets}` → 28 call sites. Every one passes either a coroutine object (`self._load(...)`, `self._review(...)`, `self._commit(...)`, `self._rebuild_after_custom_id(...)`, `self._open_lab(...)`, `self._delete_after_confirmation()`, …), a bound `async def` (`petdex_import_review._show_preview/_prepare/_accept` at :373/:428/:492, `buddy_character_review._prepare/_show_preview/_publish` at :265/:335/:384 — all `async def`, confirmed by `grep -n "async def _show_preview"` etc.), a `partial` of an `async def` (`persona_buddy_widget.py:516` → `_resolution_loop` is `async def` at :334), or a **sync** callable *with* `thread=True` (`personal_context_panel.py:1199,1449`; `personal_context_review_modal.py:300,758`).
- Confidence: verified.


### VERIFIED-FINE — the approval card's markup surfaces
- Every Static that renders model- or tool-controlled text is `markup=False`: row header (:936), reason (:945), args (:1006), effects (:1014), raw-shell label/metadata/warning/scope (:986-1000), row scope line (:1063). The raw shell command goes into a read-only `TextArea` (:989), not a markup surface.
- The only two `markup=True` surfaces both escape: `#approval-summary` via `escape(text)` (:1303) and the per-row rationale via `escape(context)` (:1021, and the reuse path at :1243). `rich.markup.escape` is imported at :37.
- `redact_mapping` covers nested mappings, nested sequences and key-name-free values: probe → `{"headers": {"Authorization": "Bearer SECRET"}}` → `{"headers":{"Authorization":"***"}}`; `{"env": ["API_KEY=sk-ant-api03-AAAA…"]}` → `{"env":["***"]}`; `{"note": "sk-ant-api03-AAAA…"}` → `{"note":"***"}`.


### RETIRED — "per-keystroke draft announce is expensive" (measured, it is not)
- Candidate: `handle_draft_field_changed` (:5812-5824) fires on every `Input.Changed` and calls `_announce_draft_state` → `has_unsaved_changes` (which builds a save proposal for **every** built-in provider, :3858-3872) + `draft_snapshot` + 4 `deepcopy`s of the full state (`_draft_revision_values` :4/`DraftModified.__init__` :711-712).
- Evidence:
```
... $PY -c  # load_global_speech_tts_state({}) twice, then time the two halves
providers: 7
deepcopy(state)                  : 0.030 ms
has_unsaved_changes proposal loop: 0.288 ms
=> per keystroke lower bound     : 0.407 ms
```
- Disposition: **retired** — 0.4 ms per keystroke is not a hot-path cost. (Measured against a default-empty config; a populated one grows the deepcopy leg, which is the 0.03 ms half.)


### VERIFIED-FINE — no blocking I/O anywhere in the 6,014-line speech panel
- `grep -n "open(\|read_text\|write_text\|read_bytes\|write_bytes\|os\.(stat|path|listdir|walk)\|\.exists()\|\.iterdir()\|is_dir()\|is_file()\|subprocess\|httpx\|requests\.\|time\.sleep\|sqlite"` → 2 hits, both `lambda path: path.is_file()` inside a file-picker `Filters` (:5993, :5998), evaluated by the picker, not the panel. The only persistence is `save_settings_to_cli_config` on the explicit Save action (:4501-4512), which is the repo-wide pattern.
- Only 3 `except Exception` in 67 handlers (:902, :921, :4510); the first two fail **closed** (return `True`/`None` = "fenced"), the third logs via `logger.exception` and returns False, which both callers surface as "Voice engine settings were not saved." (:4063-4069, :4212-4217). No swallowed persistence failure.


### RETIRED — the 3 `plain_readback` candidates (`str(button.label)` / `str(collapse.label)`)
- Where: `persona_buddy_widget.py:926,927`, `personas_library_pane.py:212`.
- Evidence: all three feed `len(str(...)) + <chrome>` into a **width computation** only (`collapse.styles.width = len(str(collapse.label)) + 2`; `row += len(str(button.label)) + _TOOLBAR_BUTTON_CHROME_COLS`). The labels are literal code strings ("▾", "✕", "Import", …), never user text, and the read-back value is never re-rendered or persisted. `personas_library_pane._required_toolbar_row_width`'s own docstring states the intent ("Derived from labels, not rendered sizes, so toggling the stacked class never changes the measurement").
- Disposition: **retired**.


### VERIFIED-FINE — `str(Static.renderable)` in the character editor round-trips user text exactly
- `personas_character_editor_widget.py:1104-1112` reads the generation preview back with `str(self.query_one(...).renderable)` and `_accept_generation` (:1248-1259) writes that string into the character field — i.e. a read-back on a **save** path, exactly the Textual-8 hazard class.
- It is safe, and for a non-obvious reason: Textual 8.2.8's `Static` has **no** `renderable` attribute (`.venv/.../textual/widgets/_static.py` exposes `content` and `visual`), but `tldw_chatbook/__init__.py:78-89` installs a compatibility shim — `Static.renderable = property(lambda self: self.content, lambda self, v: self.update(v))` — so the read-back returns the original `str` unchanged.
- Evidence (mounted probe, `<SCRATCH>/probe_chared_preview3.py`): four payloads round-tripped through `show_generation_preview` → `generation_preview_text` → Accept → the field:
```
renderable type when mounted: str
in='R&D [draft] report'            field_after_accept='R&D [draft] report'            MATCH=True
in="She said [i]softly[/i], 'no'." field_after_accept="She said [i]softly[/i], 'no'." MATCH=True
in='line one\nline two'            field_after_accept='line one\nline two'            MATCH=True
in='a [bold]literal[/bold] tag'    field_after_accept='a [bold]literal[/bold] tag'    MATCH=True
```
- Residual (P3, not filed separately): the widget reads a shimmed attribute rather than Textual 8's own `Static.content`. If the shim is ever dropped, `except Exception: return ""` at :1111 turns this into a silent "Accept writes an empty field" data-loss bug — the swallow is what makes the dependency invisible.


## Candidate dispositions

| candidate (file:line / pattern) | disposition |
|---|---|
| `query_one_in_timer_no_try` × 18 — `buddy_conversation_modal.py:109-206` (`refresh_projection`, timer :102) | **retired** — callback opens with `if not self.is_mounted or not self._visible: return` (:107); `on_unmount` stops the timer (:335); every id unconditional in `compose` (:66-94); the one conditional id (`#buddy-mic`) is queried inside the same `if self.allow_voice` that yields it; no `recompose` in the file; Textual stops pump timers in `_close_messages` |
| `query_one_in_timer_no_try` × 6 — `buddy_speech_controls.py:56-77` (`refresh_state`, timer :37) | **retired** — all 5 ids yielded flat at :28-33, no conditionals, no recompose, timer owned by the same widget |
| `query_one_in_timer_no_try` × 2 — `persona_profile_editor_widget.py:593,599` (`_run_validation`, timer :608) | **retired** — `_validated_field_ids()` (:570-572) returns two ids both yielded unconditionally in `compose` (:126, :111); `display=False` does not hide a widget from `query_one` |
| `query_one_in_timer_no_try` × 2 — `personas_character_editor_widget.py:1482,1494` (`_run_validation`, timer :1503) | **retired** — `_validated_field_ids()` (:1447-1457) returns 3 ids yielded at :469, :607, :625; `awk` over the compose body (400-710) finds no `if`/`elif`/`else` |
| `run_worker_exclusive_no_group` — `speech_tts_settings_panel.py:5438` | **confirmed** — P2 [D3] above (only ungrouped exclusive worker in the package; cancels the custom-id rebuild fence) |
| `run_worker_coroutine_per_file` × 6 files (buddy_management_modal 2, buddy_workspace_modal 1, persona_buddy_widget 1, petdex_import_review 1, personal_context_panel 5, speech_tts_settings_panel 6) | **retired** — all 28 `run_worker` sites in the slice pass a coroutine/awaitable or an `async def`; the 4 sync callables all pass `thread=True`. No `WorkerError` shape (the P0 found in the neighbouring slice) exists here |
| `except_exception_pass` — `chat_approval_card.py:639` | **retired** — guards `self.focus()` in `focus_first_decision`; no data path |
| `except_exception_pass` — `chat_message_enhanced.py:612` | **retired** — guards a tooltip write in `watch_tts_progress`; cosmetic |
| `except_exception_pass` — `personas_character_editor_widget.py:1187, 1210` | **retired** — :1187 guards `.focus()`; :1210 guards a `query_one` for the name Input during generated-character fill. Neither reaches storage |
| `except_exception_pass` — `personas_policy_rules_editor.py:152` | **retired** — guards `query_one("#personas-policy-status")` in `_set_status`; UI presence only |
| `except_exception_pass` — `personas_preview_pane.py:354` | **retired** — `await widget.remove()` with an explicit comment ("Tolerate a widget already removed by a transcript reseed") |
| `except_exception_return_per_file` × 14 files | **confirmed in part** — `personal_context_panel.py` (2) and `personal_context_review_modal.py` (via `_unknown_failure`) are the P3 [D1] no-diagnostics finding above. The other 12 files' `except Exception: return` sites are `query_one` presence guards or fail-closed fences (`speech_tts_settings_panel.py:902` returns `True` = "fenced"), **retired** |
| `function_body_import_per_file` × 16 files | **confirmed in part** — 4 redundant re-imports in `personas_character_editor_widget.py:1284-1285, 1368-1369` (P3 [D3] above). Every other function-body import in the slice resolves (`ls tldw_chatbook/Utils/mosaic_render.py` → exists; `...UI.Screens.personas_screen`, `...Petdex.conversion`, `...Petdex.registry`, `..enhanced_file_picker`, `...UI.Navigation.buddy_speech`, `...UI.Navigation.buddy_management`, `...Agents.agent_models`, `tldw_chatbook.Utils.input_validation` all import cleanly in the probe runs above) and most carry a circular-import or boot-cost comment — **retired** |
| `inline_truncate` × 3 (`personas_character_editor_widget.py:1691`, `personas_lore_detail.py:324`, `personas_preview_pane.py:267`) + `_bounded_label`/`_bounded_display` | **confirmed** — P2 [D4b] above (dead `Utils.Utils.truncate_content`, 38 re-rolls repo-wide) |
| `plain_readback` × 3 (`persona_buddy_widget.py:926,927`, `personas_library_pane.py:212`) | **retired** — width measurement only, literal labels; see the Retired section |
| `try_import_guard` — `chat_message_enhanced.py:28` | **confirmed** — folded into the P3 [D3] chat_message_enhanced finding (`except (ImportError, TypeError, Exception)`, no `optional_deps`) |
| `try_import_guard` — `actor_pack_import_review.py:23` | **retired** — a correct optional-renderer guard (`except ImportError`, `TerminalImage = None`, `_portrait_widget` returns `None` when absent); `textual-image` is genuinely optional |
| `strftime` — `chat_message_enhanced.py:660` (`%Y%m%d_%H%M%S`) | **retired** — builds a download filename from `datetime.now()`; no timezone contract |
| `strftime` — `speech_tts_settings_panel.py:1626` (`%Y-%m-%d %H:%M UTC`) | **retired** — preceded by `.astimezone(timezone.utc)` (:1624), so the "UTC" literal is true |
| `legacy_markers_per_file` × 9 files (18 in `personas_character_editor_widget.py`) | **retired** — in the character editor all 18 are domain vocabulary ("legacy visual identity" = the pre-pack expression controls, a live UI branch), not debt markers. `chat_approval_card.py`'s 2 are the module docstring's `Chat_Window_Enhanced` reference — already filed as P3 doc drift by a sibling, cited not re-filed |
| `seed_name__cancel` × 14, `seed_name__perform_safe_cancel` × 4, `seed_name__set_status` × 5 | **confirmed in part** — the class-level drift (3 of 5 pickers with no `SafeModalDismissMixin`) is the P2 [D4a+b] finding above. The remaining 2-line `_cancel`/`_set_status` adapters are P3 boilerplate over the canonical mixin, per the brief |
| `dup_verbatim` — `_check_for_files@chat_message.py:350` ↔ `chat_message_enhanced.py:721` | **confirmed, superseded** — both copies live in the two production-dead widgets (P2 [D3] above); de-duplicating them is moot until the wiring question is answered |
| `dup_verbatim` — the 5 picker families (`_filter`, `_apply_filter_debounced`, `_selected_id`, `_confirm`, `_cancel`) | **confirmed** — the P2 [D4a+b] picker finding |
| `dup_verbatim` — `personas_lore_detail.py` ↔ `personas_dictionary_detail.py` (`_row_selected`, `_row_highlighted`, `_up_pressed`, `_down_pressed`, `_selected_attachment_id`) | **retired as a file-level clone, confirmed as P3 boilerplate** — clone-normalised diff of the two files: 755 differing lines out of 1,387, so they are NOT the same widget; only the five 3-8 line row handlers coincide |
| `dup_verbatim` — `speech_tts_settings_panel.py` handle_cancel/handle_confirm/handle_discard/handle_save (×8) ↔ `settings_screen.py:2550-2612` | **retired** — each is a 3-line `event.stop(); self.dismiss(<literal>)` inside a different `ModalScreen[T]` whose `T` differs (`str|None`, `bool`, `LeaveChoice`); there is nothing to share but the two statements |
| `dup_shape` — the 8 large "button handler one-liner" clusters (114/45/28/25/22/20/19/17 copies spanning 49/37/11/12/9/12/8/8 files) | **retired** — these are `@on(Button.Pressed, "#id")` → `event.stop()` → one call. That is Textual's handler contract, not duplication; the mechanical shape-hasher is matching the decorator+stop pair |
| `dup_shape` — `set_avatar_thumbnail` triplet (not in the excerpt rows; found by reading) | **confirmed** — P2 [D4b] above |

## Verified-fine
- **Lint baseline holds.** `ruff check --select E9,F63,F7,F82` over all three packages → `All checks passed!` (0 fatal).
- **The approval card's redaction and markup discipline** — see the dedicated entry above. Every untrusted surface is `markup=False`; the two `markup=True` Statics escape; `redact_mapping` handles nested mappings, nested sequences and key-name-free credential values.
- **No `run_worker(sync_callable)` without `thread=True`** anywhere in the slice (the P0 shape a sibling proved in `audio_troubleshooting_dialog.py`). 28 sites audited individually.
- **No blocking I/O, no sqlite, no `threading.Lock`, no `eval`/`exec`/`pickle`** in the slice. The full sweep (`open(`, `read_text`, `read_bytes`, `write_*`, `json.load`, `sqlite3`, `.execute(`, `threading.`, `time.sleep`, `subprocess`, `httpx`, `requests.`, `urlopen`, `Path.home()`) returns 20 hits, every one accounted for: a `threading.Event` used as a cancel token (`petdex_import_review.py:70`), `json.loads` on a TextArea the user typed into (:155) and on an already-validated pack field (`actor_pack_import_review.py:285`), two `Path.home()` picker start locations, the PIL decodes below, the `httpx` probe (filed above), and `chat_message_enhanced.py:674`'s image save, which is correctly `await asyncio.to_thread(...)`.
- **`actor_pack_import_review._portrait_widget` decodes an imported pack's portrait with PIL and is safe**: the bytes reach it only through `Actor_Packs/importer.py:499-522 read_portrait_preview`, which caps at `MAX_PORTRAIT_BYTES`, after `importer.py:382` ran `validate_actor_portrait` → `contracts.py:704-746`, which checks magic bytes, format-vs-extension, `MAX_PORTRAIT_DIMENSION`, `MAX_PORTRAIT_PIXELS`, `image.verify()` and catches `DecompressionBombError`. The widget's own `except Exception: return None` is then a display fallback, not the security boundary.
- **No mutable class attributes** shared across instances. The 8 `grep` hits for class-level `[]`/`{}`/`set()` are all function-local variables at 4-space indent under a module-level `def` (`personas_dictionary_validation.py:42-43`, `speech_tts_panel_types.py:250,268`, `tool_pack_import_review.py:47`, `chat_approval_card.py:310-311,552`). `chat_message_enhanced.py:210-211` (`_extracted_files`/`_file_extractor = None`) are class-level `None` sentinels that every write rebinds per instance.
- **`speech_tts_settings_panel.request_save` writing config synchronously on the event loop** (`save_settings_to_cli_config`, :4501-4512) is the repo-wide Save pattern, on an explicit one-shot user action. Not flagged.
- **`persona_buddy_widget`'s 10 Hz poll** (`_POLL_SECONDS = 0.10`, `on_mount:284`) is gated: `refresh_from_controller` computes a paint authority and returns early when nothing the view renders moved (:629-635, "TASK-21122"), and stops its own timers plus cancels its resolution worker in `on_unmount` (:288-298).

## Retired
- **All 28 timer `query_one` candidates** — three independent legs of evidence, in the first Findings entry.
- **The per-keystroke draft-announce cost in the TTS panel** — measured at 0.407 ms; entry above.
- **The 3 `plain_readback` sites** — width measurement over literal labels; entry above.
- **"`str(Static.renderable)` un-escapes user text on a save path"** — raised on reading `personas_character_editor_widget.py:1104-1112` + `_accept_generation`, then retired: `tldw_chatbook/__init__.py:78-89` installs `Static.renderable → self.content`, and a mounted-widget probe round-tripped `&`, `[i]…[/i]`, `[bold]` and newlines byte-for-byte through Accept. Entry above keeps the residual P3 note.
- **`personas_lore_detail.py` ↔ `personas_dictionary_detail.py` as a file-level clone** — 755 of 1,387 normalised lines differ; only five short row handlers coincide.
- **The 8 mechanical `dup_shape` button-handler clusters** — Textual's `@on` + `event.stop()` contract, not duplication.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The `speech_tts_settings_panel.py:5438` ungrouped-exclusive collision actually loses a confirmed custom model id | Needs two `_custom_id_modal_result` calls landing inside one `_replace_card_bodies` await window; the panel takes ~15 constructor arguments and the repo's own suite for it cannot start under the isolated profile (below) | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY -m pytest Tests/UI/test_settings_speech_tts_panel.py -q` then a new case that calls `panel._custom_id_modal_result("model", "m-1")` and `panel._custom_id_modal_result("voice", "v-1")` back to back with no `await` between, and asserts `panel.draft_snapshot().state.defaults.model_id == "m-1"` |
| `ChatMessageEnhanced.watch_tts_state`'s `self.refresh()` fails to swap the TTS button set | The widget has no production mount site, so I did not build a harness for a dead path | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY -c` mounting `ChatMessageEnhanced("x","AI")` in a bare `App`, then `w.update_tts_state("generating")`, `await pilot.pause()`, `app.query_one("#tts-generating")` |
| Whether the repo's own UI suites for this slice are green | **Blocked, environment-level**: every async UI test errors at *setup* under the isolated profile with `tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired: raw_source_selection_changed` (`Backup_Recovery/raw_participants.py:127`), before any test body runs. This is the ADR-126 recovery gate, not a slice defect — my own probes mount the same widgets successfully, so it is the conftest fixture chain that trips it | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY -m pytest Tests/UI/test_personas_character_generation_ui.py -q` (currently: `10 errors`, all `ERROR at setup`) |
| Any claim about how these surfaces look or behave on a real terminal | The brief forbids running the app | `tmux -L verify new-session -d -s w 'cd <worktree> && …' ; tmux -L verify capture-pane -p -e` per `.claude/skills/verify/SKILL.md` |
