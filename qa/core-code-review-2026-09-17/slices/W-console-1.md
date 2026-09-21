# W-console-1 — tldw_chatbook/Widgets/Console/ (first half by line count), 60 files, 34,296 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review @ d8fb4053f9

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| console_composer_bar.py | 6499 | **read in full** |
| console_prompts_modal.py | 2194 | **read in full** |
| console_conversation_inspector.py | 2159 | **read in full** |
| console_model_popover.py | 1432 | **read in full** |
| console_auto_speak_consent.py | 1117 | **read in full** |
| console_inspector_section.py | 1079 | **read in full** |
| console_capture_policy_dialog.py | 1039 | **read in full** |
| console_scope_picker_modal.py | 888 | read `:315-330`, `:555-612`, `:730-745` (the escape sites + both debounce timers) |
| console_endpoint_template_modal.py | 875 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_run_inspector.py | 818 | read `:280-300` (the recompose guard) and `:580-680` (action/row build) |
| console_reaction_picker_modal.py | 709 | read `:325-340`, `:440-480`, `:595-610` (both debounce timers + `_row_label`) |
| console_project_instructions.py | 679 | read `:290-470` (status row, `sync_preview`, `sync_state`) and the two `_perform_safe_cancel` rows |
| console_prompt_queue_modal.py | 665 | read `:140-320` (compose, 0.2 s poll, `_apply_snapshot`) + `:470-660` (controller calls); rest by candidate row |
| console_character_context.py | 635 | read in full except the compose tail (`:460-635`); all label/timer/recompose paths read |
| console_selection_menu.py | 630 | read `:340-380` + `:600-630` (the `_NO_RUN_HINT` tooltips and the focus-restore guards) |
| console_bounded_section.py | 587 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_fork_chat_modal.py | 504 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_appearance_picker_modal.py | 492 | read `:250-320` (debounce timer + filter commit) and the two function-body imports |
| console_context_controls.py | 488 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_library_access_modal.py | 486 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_citation_sources_modal.py | 473 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_rag_settings_modal.py | 434 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_control_bar.py | 404 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_provider_picker.py | 402 | read `:130-402` in full (all `query_one`, blur timer, commit paths) |
| console_assistant_turn.py | 389 | read `:95-200` (the `.plain` row, the elapsed timer, `sync_header`) |
| console_review_notes_modal.py | 384 | read `:290-380` (all three defensive guards + the delete confirmation) |
| console_prompt_picker_modal.py | 373 | read `:130-290` (mode titles, the escape site, the debounce timer) |
| console_exchange_export_dialog.py | 366 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_conversation_action_menu.py | 365 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_inspector_ownership.py | 358 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_character_picker_modal.py | 354 | read `:200-300` + `:330-354`; rest by candidate row |
| console_generation_card.py | 349 | read `:200-330` (the image fallback ladder + action buttons) |
| console_rewind_modal.py | 348 | read `:160-240` (compose + the escape site + the disabled-reason tooltips) |
| console_edit_message_modal.py | 318 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_background_effect.py | 298 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_composer_menu_modal.py | 293 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_agent_steering_bar.py | 268 | **read in full** |
| character_expression_avatar.py | 268 | read `:110-200` (the 1/30 s frame timer and the graphics fallback ladder) |
| console_prompts_browse.py | 259 | read `:80-200` (the Text-built result rows and the improve tooltip) |
| console_selection.py | 256 | **read in full** |
| console_canvas_card.py | 220 | read `:1-200` in full (the card + recovery card); `:200-220` action wiring skimmed |
| __init__.py | 214 | **read in full** |
| console_command_popup.py | 208 | read `:95-160` (the trailing re-anchor timer and `reposition`'s guards) |
| console_activity_outcome_notice.py | 207 | **read in full** |
| console_generate_image_modal.py | 203 | read `:155-185` (the preview guard) |
| console_agent_progress_modal.py | 195 | **read in full** |
| console_message_more_menu.py | 193 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_run_log_modal.py | 184 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_prompt_improve_view.py | 183 | read `:130-185` (the analysis-context tooltips and the button build) |
| console_image_viewer_modal.py | 172 | read `:100-172` in full (both try_import_guard rows, the `.plain` row) |
| console_prompts_state.py | 159 | **read in full** |
| console_retrieval_scope_row.py | 159 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_rail_handle.py | 155 | **read in full** |
| console_prompt_comparison_modal.py | 152 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_agent_history_modal.py | 147 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_feedback_comment_modal.py | 146 | read `:25-60` (`_preview_text`) + the submit rows |
| console_save_as_modal.py | 128 | read `:70-110` (the destination buttons and the empty state) |
| console_save_markdown_modal.py | 116 | mechanical only — covered by the AST widget-constructor sweep, the tooltip sweep, the `get_cli_setting`/`.plain`/`escape`/`recompose=True`/`re.compile`/`sqlite`/`threading`/`write_text` greps, the function-body-import resolver, the worker-group check and `ruff --select E9,F63,F7,F82`; no candidate row fell in it |
| console_library_search_modal.py | 113 | read `:1-60` (the query sanitiser + subclass declaration) |
| console_rename_session_modal.py | 108 | **read in full** |

## Findings

### P1 [D1] — User text that contains an uppercase bracketed token (`[WIP] plan`, `[Draft] notes`, `[TODO]`) is silently DELETED from Console labels, and `rich.markup.escape` does not prevent it
- Where (cluster, all in `tldw_chatbook/Widgets/Console/`):
  - `console_character_context.py:348` (`_group_label` → character name), `:386` and `:437` (`_row_label`/`_search_row_label` → conversation title) — **no escape at all**
  - `console_prompt_queue_modal.py:306` — `Button(f"{entry.position}. {entry.preview}")`, queued prompt text
  - `console_rewind_modal.py:181` — `escape_markup(f"{row.index_label}  {row.preview}")`, the user's own prompt text
  - `console_scope_picker_modal.py:567` (media/note title), `:589`, `:605` (tag names)
  - `console_reaction_picker_modal.py:470` (`option.display_label`)
  - `console_prompt_picker_modal.py:273` (prompt name), `console_endpoint_template_modal.py:503` (`candidate.label`), `console_provider_picker.py:242`
  - `console_inspector_section.py:975-988 _refresh_tooltip` — `self.tooltip = escape("\n".join(parts))` at `:988`, whose own docstring says *"Markup is escaped because a Textual tooltip is Rich-parsed content and row text is user-adjacent"*; the tooltip is the ONLY way to read a row's untruncated text, and a changed-file path `src/[WIP]/notes.md` renders in it as `src//notes.md` (verified: `Tooltip().update(escape(s)); str(t.render())`)
  - `console_composer_bar.py:5889` — attachment filename (see the D4a finding below)
  - `console_canvas_card.py:177` — `Static(self.presentation.label)`, where `label` is `f"{card.title} · revision {n} · {status}"` built at `console_transcript.py:237` from a user-authored Canvas document title; **no escape at all**, so this one can raise like `console_character_context.py`
  - upstream helper: `Chat/console_prompt_queue.py:155 make_prompt_preview`, whose docstring claims "Rich-markup-safe"
  - shared sink: `Widgets/confirmation_dialog.py:117` (see the purge-confirmation finding below)
- Evidence (worktree, isolated env):
  ```
  $PY -c 'import rich.markup as rm; from textual.content import Content
  for t in ["[WIP] plan","[TODO]x","[b]b[/b]","[/]"]:
      e=rm.escape(t); print(repr(t), rm.escape(t), repr(str(Content.from_markup(e))))'
  '[WIP] plan'  -> rich.escape '[WIP] plan'  -> Content ' plan'
  '[TODO]x'     -> rich.escape '[TODO]x'     -> Content 'x'
  '[b]b[/b]'    -> rich.escape '\\[b]b\\[/b]' -> Content '[b]b[/b]'   (lowercase IS escaped)
  '[WIP]a[/WIP]' -> Content 'a[/WIP]'   (the open tag vanishes, the close tag prints literally)
  ```
  `rich.markup.escape`'s regex is `(\\*)(\[[a-z#/@][^[]*?])` — it only escapes tags whose first character is `[a-z#/@]`. Textual 8's `Content.from_markup` accepts **any** tag body, uppercase included, and drops it as an unresolvable style span. `textual.markup.escape` is the identical regex, so switching helper does NOT fix it.
  Production constructors reproduced end to end:
  ```
  $PY -c "from types import SimpleNamespace as N
  from tldw_chatbook.Widgets.Console.console_character_context import ConsoleCharacterContext as C, CharacterConversationButton as B
  r=N(title='[WIP] Draft plan', is_current=False); print(repr(str(B(C._row_label(r), row=r).label)))"
  ' Draft plan'                                  # '[WIP] ' gone
  r.title='notes a [/] b'  -> MarkupError: auto closing tag ('[/]') has nothing to close
  ```
  ```
  $PY -c "from tldw_chatbook.Chat.console_prompt_queue import make_prompt_preview as m
  from textual.widgets import Button; print(repr(str(Button('1. '+m('[WIP] summarize this')).label)))"
  '1.  summarize this'
  ```
- Why it matters: a conversation titled `[WIP] Draft plan` shows in the Console Context rail as ` Draft plan`; a queued prompt `[TODO] rerun` shows as ` rerun`; the rewind list loses the same text from the prompt the user is choosing between. Bracketed prefixes are a common titling convention, so this is routine, not adversarial. The `console_character_context.py` sites are unescaped entirely, so a title/character name containing `[/…]` raises `MarkupError` **inside `compose()`** (verified above) — that is an app-level crash from a DB string, and character names can arrive from imported third-party cards.
- Recommended correction: stop routing user text through markup parsing at all — pass `Content(label)` / `Text(label)` (a `Text`/`Content` first arg is not re-parsed; verified: `str(Button(Text("plain [b]x[/b]")).label) == 'plain [b]x[/b]'`), or `markup=False` where the widget supports it. Canonical home: one helper next to the existing escape users, e.g. `Utils/text.py` `as_literal_content(str) -> Content`, then replace every `escape_markup(<user text>)` call in `Widgets/Console/` with it. Deleting the `escape_markup` calls without replacing the mechanism would make it worse (lowercase tags would then also be eaten).
- Size: M · ADR: no · Confidence: verified
- Pinning test: none. `Tests/UI/test_console_character_context.py` has no markup/bracket assertion (`grep -n "markup\|escape\|bracket" Tests/UI/test_console_character_context.py` → only `pilot.press("escape")` lines).
- Already covered: none

### P1 [D1] — Console Context rail passes conversation titles and character names through Textual markup with NO escape: `[/…]` in the string raises `MarkupError` inside `compose()`
- Where: `console_character_context.py:348` (`CharacterGroupButton(self._group_label(...))`), `:386`, `:437` (`CharacterConversationButton(label, ...)`), plus `header.tooltip`/`button.tooltip` set from the same raw strings at `:356`, `:400`, `:451`
- Evidence:
  ```
  $PY -c "from types import SimpleNamespace as N
  from tldw_chatbook.Widgets.Console.console_character_context import ConsoleCharacterContext as C, CharacterConversationButton as B
  for s in ['[WIP] Draft plan','Q3 [budget] review','notes a [/] b']:
      r=N(title=s,is_current=False)
      try: print(repr(s),'->',repr(str(B(C._row_label(r),row=r).label)))
      except Exception as e: print(repr(s),'-> RAISED',type(e).__name__,e)"
  '[WIP] Draft plan'   -> ' Draft plan'
  'Q3 [budget] review' -> 'Q3  review'
  'notes a [/] b'      -> RAISED MarkupError: auto closing tag ('[/]') has nothing to close
  ```
  Every sibling picker in this same package escapes first (`console_rewind_modal.py:181`, `console_scope_picker_modal.py:567`, `console_prompt_picker_modal.py:273`, `console_reaction_picker_modal.py:470`, `console_endpoint_template_modal.py:503`); this file does not, so it is the only one that can raise.
- Why it matters: `compose()` raising takes the Console screen down, and the input is a DB string — a conversation title the user typed, or a `character_label` from an imported V2/V3 character card (a third-party file). The escaped siblings degrade to silent text loss; this one degrades to a crash.
- Recommended correction: the same literal-content fix as the finding above. If only a stop-gap is wanted, `CharacterConversationButton(Text(label), ...)` / `CharacterGroupButton(Text(label))` is a two-line change that removes both the crash and the deletion at these three sites (a `Text` first arg is not re-parsed — verified).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`Tests/UI/test_console_character_context.py` asserts nothing about markup)
- Already covered: none

### P1 [D1] — The irreversible "Delete stored Full captures" confirmation names the WRONG conversation when its title contains a bracketed token, and raises `MarkupError` for `[/…]`
- Where: `console_capture_policy_dialog.py:525-542` (`_purge_confirmation_message`, interpolates `snapshot.conversation_title`) → `:592-597` (`delete_full_captures` calls `_confirm` with it) → `:627-642` `_confirm` → `Widgets/confirmation_dialog.py:117` `yield Label(self.message, classes="dialog-message")` — **no `markup=False`**
- Evidence:
  ```
  $PY -c "import tldw_chatbook.Widgets
  from textual.widgets import Label
  from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import ConsoleCapturePolicyDialog as D
  from types import SimpleNamespace as N
  for title in ('[WIP] Q3 planning','notes a [/] b'):
      snap=N(enabled=False, conversation_title=title, effective=N(detail=N(value='safe')))
      m=D._purge_confirmation_message(snap,12)
      try: print(repr(title),'->',repr(str(Label(m).render())[:90]))
      except Exception as e: print(repr(title),'-> RAISED',type(e).__name__,e)"
  '[WIP] Q3 planning' -> 'Delete 12 stored Full captures from “ Q3 planning”? This irreversible action …'
  'notes a [/] b'     -> RAISED MarkupError: auto closing tag ('[/]') has nothing to close
  ```
- Why it matters: the whole job of this string is to let the user confirm *which* conversation's stored Full captures are about to be irreversibly deleted, and the title it shows is not the title that exists. The `[/…]` case raises inside `ConfirmationDialog.compose()`, so the confirmation for a destructive action fails to appear at all. `ConfirmationDialog` is imported by 45 modules (`grep -rln "confirmation_dialog import" tldw_chatbook | wc -l` → 45), several of which interpolate user-named entities into `message`, so the same hole is open wherever a caller does.
- Recommended correction: `markup=False` on `Widgets/confirmation_dialog.py:117`'s `Label` (and `:116`'s title `Static`). **The repo already does exactly this next door**: `Widgets/Library/prompt_delete_confirmation_modal.py:144` is `yield Static(self._preview_copy(), id="prompt-delete-preview", markup=False)` — same class of dialog, same user-named entity, correct fix. It is the canonical home: one line fixes every one of the 45 callers and cannot be forgotten at a call site. `ConfirmationDialog` messages are all plain prose today, so nothing loses styling.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `ConfirmationDialog` itself has none (`find Tests -iname "*confirmation*"` → only `Tests/UI/test_prompt_delete_confirmation_modal.py`, which covers the *other* modal). That test is worth reading before fixing this: `:306 test_markup_looking_names_render_literally` asserts literal rendering using `name = "[bold magenta]not markup[/bold magenta]"` — an all-lowercase tag, the one case `rich.markup.escape` handles. The uppercase case the Console hits (`"[WIP] not markup"`) is untested there too, so the new test should use an UPPERCASE tag or it proves nothing about this defect class.
- Already covered: none

### P1 [D4a] — The repo already diagnosed the `rich.markup.escape` gap in writing (`console_composer_bar.py:5983-5993`) and fixed it with `Content` in ONE place; the composer's own attachment indicator in the same file still uses the broken escape
- Where: fix + diagnosis at `console_composer_bar.py:5971-6011` (`set_voice_status`'s docstring) and `:6039`/`:6069`/`:6139` (`chip.update(Content(...))`); unfixed sibling at `console_composer_bar.py:5889` — `indicator.update(escape(resolve_glyph_text(f"📎 {normalized}")))`
- Evidence: the docstring states the mechanism verbatim — *"a `Static` parses strings as Textual markup, and `rich.markup.escape` (which used to guard this) only escapes tags opening with `[a-z#/@]`. Whisper's own tokens are uppercase, so `[BLANK_AUDIO]` and `[Music]` survived escaping untouched and were then stripped at paint time … `Content` carries plain text with no markup semantics at all, so it fixes the swallowing and the opposite failure (`[/tmp/x]` raising `MarkupError`) in one move."* Reproduced on the indicator's exact expression:
  ```
  $PY -c "from rich.markup import escape; from textual.content import Content
  print(repr(str(Content.from_markup(escape('\U0001F4CE [WIP]report.pdf · 4 KB')))))"
  '📎 report.pdf · 4 KB'          # lowercase '[budget]' IS escaped and survives; uppercase is not
  ```
  The label is a real filename: `UI/Screens/chat_screen.py:22219` passes `pendings[0].label` ("photo.png · 240 KB") straight through.
- Why it matters: this is the same defect the file's own docstring says was fixed, still live 82 lines earlier in the same file, on a path where the user is being told which file they staged. It also means the "use `escape_markup`" pattern the rest of `Widgets/Console/` follows is documented-here as insufficient, yet 14 files in `Widgets/Console/` still follow it — 8 in this slice plus `console_session_switcher_modal`, `console_session_surface`, `console_settings_modal`, `console_setup_modal`, `console_style_picker_modal`, `console_workspace_context` in the other half (see the first finding).
- Recommended correction: `indicator.update(Content(resolve_glyph_text(f"📎 {normalized}")))` and drop the now-unused `from rich.markup import escape` at `:26` once the other `escape(...)` call sites in the file are checked. Canonical statement of the rule belongs next to the existing `set_voice_status` docstring or in `backlog/docs/lessons-textual.md` so the rest of the package stops reaching for `escape_markup`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: three exist and none can catch it — `Tests/UI/test_console_composer_collapse.py:1334` (`assert "photo.png · 12 B" in str(attachment.renderable)`), `Tests/UI/test_console_voice_chip.py:308` (`assert "2 files" in _painted(indicator)`), `Tests/UI/test_console_dictation_streaming.py:450` — every fixture label is bracket-free, so the substring assertion passes either way
- Already covered: none

### P2 [D4b] — `_maybe_await` is re-implemented 66 times across the package with no shared home
- Where (this slice): `console_prompts_modal.py:153`. Repo-wide: 66 definitions (`grep -rn "async def _maybe_await" tldw_chatbook --include='*.py' | wc -l` → 66).
- Evidence: normalizing each body and de-duplicating shows **no behavioural drift** — every copy is `await value if inspect.isawaitable(value) else value` in one of two spellings (ternary vs `if/return`), some as a `@staticmethod`, some module-level.
- Why it matters: 66 copies of a 2-line helper is the largest single re-roll in the tier; the reason to consolidate is not correctness (there is none to fix) but that the next variant will be the one that drifts — this is exactly the D2 "a pattern that will produce a P1" shape.
- Recommended correction: one `async def maybe_await(value)` in a new `Utils/async_helpers.py` (no such module exists today — `ls tldw_chatbook/Utils/` has no async helper), imported by all 66. Mechanical, no behaviour change.
- Size: M (66 files, one-line each) · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D4b] — "cap this text and append a truncation marker" is re-rolled 14 times with three different markers and two different budget semantics, and the drift reaches user-visible strings and tool results
- Where (this slice owns 2): `console_selection.py:13-14,137-140` (`SELECTION_QUOTE_CAP=4000`, marker `"\n… [truncated]"`, budget-inclusive) and `console_feedback_comment_modal.py:28-37` (`PREVIEW_QUOTE_CAP=600`, marker `"… [truncated]"` — **no leading newline**, budget-inclusive). The other 12: `Agents/run_hooks.py:181,197` (`"…[truncated]"` — no space), `Tools/watchlists_tool_service.py:63` (same no-space form), `Tools/web_tool_impls.py:1171`, `Tools/local_tool_impls.py:355,561`, `Agents/virtual_cli_provider.py:120`, `Agents/local_tool_provider.py:435,521`, `MCP/unified_control_plane_service.py:3840`, `UI/Research_Modules/bundle_rendering.py:131`, `Library/library_rag_answer_service.py:362`.
- Evidence:
  ```
  grep -rhon "…\s*\[truncated\]" tldw_chatbook --include='*.py' | sed 's/^[0-9]*://' | sort | uniq -c
       10 … [truncated]
        2 …[truncated]
  grep -rn "\[truncated\]" tldw_chatbook --include='*.py' | wc -l   → 14
  grep -rn "len(_TRUNCATION_MARKER)\|len(PREVIEW_TRUNCATION_MARKER)\|len(_TRUNCATION_SUFFIX)" … → 4 sites
  ```
  Two distinct budget semantics: 4 sites slice at `CAP - len(marker)` so the **result** is ≤ CAP; the other 10 slice at CAP and then append, so the result is `CAP + len(marker)`.
- Why it matters: this is not cosmetic in every copy. `Tools/local_tool_impls.py:355,561`, `Agents/local_tool_provider.py:435,521`, `Agents/virtual_cli_provider.py:120` and `MCP/unified_control_plane_service.py:3840` truncate **tool results that go on the wire to a model** against a byte budget the appended marker then exceeds — a cap that is documented as N and is actually N+15. The two in this slice are user-visible instead: the same quoted selection shows with a leading newline in one surface and without it in the other.
- Recommended correction: one `truncate_with_marker(text, *, budget)` in `Utils/` (no such helper exists — `ls tldw_chatbook/Utils/` has no text-truncation module), with the budget-inclusive semantics the 4 careful sites already use, and one marker constant. Migrate the 10 append-after-slice sites deliberately, since each one's cap becomes 15 characters tighter.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found for the marker text
- Already covered: none

### P3 [D1] — `ConsoleAutoSpeakCoordinator._observed_completion_generations` grows for the life of the screen and is never pruned
- Where: `console_auto_speak_consent.py:208` (declared), `:384` (the only write). Its two siblings ARE cleaned up: `failed_message_ids` by `_purge_failed_owners` at `:275-286` and again at `:1111`, `_pending_completions` by the `.pop(token, None)` calls at `:928` and `:931`. This one dict has no removal site anywhere in the file.
- Evidence: `grep -n "_observed_completion_generations" console_auto_speak_consent.py` → exactly three lines (`208` declare, `381` read, `384` write); no `del`, no `pop`, no re-assignment. `unmount()` at `:213` releases the store subscription but leaves the dict.
- Why it matters: bounded by messages-per-app-run, not by anything else, so it is a slow leak in a long Console session rather than a bug a user hits. Named because the pruning helper that would fix it already exists two methods away and covers only one of the three dicts.
- Recommended correction: extend `_purge_failed_owners` (`:275`) to filter `_observed_completion_generations` by live session id too — one dict comprehension, next to the one it already runs.
- Size: S · ADR: no · Confidence: inferred (read only; settle with `$PY -m pytest Tests/UI/test_console_auto_speak*.py -q` plus a loop asserting `len(coordinator._observed_completion_generations)` stays bounded after a session is closed)
- Already covered: none

### P3 [D2] — `ConsoleProjectInstructionContextPanel.sync_preview` recomposes unconditionally; its sibling `sync_state` seven lines below has the equality guard
- Where: `console_project_instructions.py:422-459` (`sync_preview`, ends in a bare `self.refresh(recompose=True)` at `:459`) vs `:461-466` (`sync_state`, opens with `if state == self._state: return`)
- Evidence: `sync_preview` rebuilds `self._state` with `replace(...)` and then recomposes with no comparison, so a preview refresh that produces an identical status/sources/warning tuple still tears down and rebuilds the panel's whole subtree. The `console_conversation_inspector` Next Send tab calls it from `_update_view` (`console_conversation_inspector.py:487`) on every `watch_snapshot`, i.e. on every Refresh press and every snapshot load.
- Why it matters: a recompose is the expensive path this class's sibling method deliberately avoids; the fix is the line that already exists next to it.
- Recommended correction: build the candidate into a local at `:452-458`, then `if candidate == self._state: return` before assigning and recomposing — copying `sync_state`'s shape exactly.
- Size: S · ADR: no · Confidence: verified (read; the two methods are adjacent and the asymmetry is literal)
- Pinning test: none
- Already covered: none

### P3 [D3] — `ConsoleComposerBar` is a 6,098-line / 191-method single class, and unlike both decomposed screens it is under no size ratchet
- Where: `console_composer_bar.py:402-6499` (one class; the file is 6,499 lines)
- Evidence:
  ```
  $PY -c "import ast,pathlib
  t=ast.parse(pathlib.Path('tldw_chatbook/Widgets/Console/console_composer_bar.py').read_text())
  n=[x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='ConsoleComposerBar'][0]
  print(n.end_lineno-n.lineno+1, len([m for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef))]))"
  6098 191
  grep -n '"tldw_chatbook/' Tests/Architecture/test_screen_size_ratchet.py
  77:    "tldw_chatbook/UI/Screens/chat_screen.py": ("ChatScreen", 16966, 563),
  887:    "tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", 33204, 1276),
  ```
  Only the two screens are ratcheted; no `Widgets/` file is.
- Why it matters: `test_screen_size_ratchet.py`'s own rationale is that code lands wherever the path of least resistance is. With `chat_screen.py` capped and the composer uncapped, the composer is now that path — and several of its clusters own no region at all, which is exactly what §7 of `DESIGN.md` says should be a controller or a pure module: the HMAC snapshot/projection transaction layer (`_snapshot_fingerprint`, `_validate_snapshot_shape`, `project_snapshot_for_model`, `_validated_apply_parts`, ~500 lines), the undo/redo history store (`_record_undo_snapshot` … `restore_undo_history`, ~300 lines), the cell-wrap/window algorithm (`_cell_wrap_line`, `_wrap_draft_line_slices`, `_visible_draft_line_slices`, ~400 lines, already pure `@classmethod`s), the segment model (`_insert_literal_at_cursor`, `_delete_canonical_range`, boundary handling, ~500 lines), and prompt-history ghost/recall.
- Recommended correction: not a rewrite — add `"tldw_chatbook/Widgets/Console/console_composer_bar.py": ("ConsoleComposerBar", 6098, 191)` to `_BUDGETS` in `Tests/Architecture/test_screen_size_ratchet.py` so the number can only go down, then extract the pure clusters per `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) when someone next touches them. The wrap algorithm and the transaction layer are already `@staticmethod`/`@classmethod` and move without touching the DOM.
- Size: S for the ratchet row · L for the extraction (ADR: no — `library-decomposition-recipe.md` §17 already governs) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` exists and does NOT cover this file
- Already covered: none (task-1378 and task-31202 are `settings_screen.py`)

### P3 [D3] — 13 `query_one` guards in this slice catch bare `Exception` where the package's own idiom (76 sites) is `except NoMatches`
- Where (verified line numbers): `console_review_notes_modal.py:303`, `:331`, `:369`; `console_generate_image_modal.py:173`; `console_character_picker_modal.py:213`, `:340`; `console_image_viewer_modal.py:144`, `:163`; `console_selection_menu.py:371`, `:618`, `:626`; `console_agent_steering_bar.py:193`, `:202`, `:210`, `:237`
- Evidence: `xargs grep -n -B4 "except Exception" | grep -c query_one` → 13 in this slice; `grep -c "except NoMatches"` summed over the same 60 files → 76.
- Why it matters: the bodies these guards wrap are not bare queries — `console_generate_image_modal.py:169-172` wraps `.update(self._preview_text(self._current_command()))`, so a `TypeError` from the formatter is swallowed as "the row is gone" and the preview silently stops updating. Every one of these has a comment saying it is there for a missing row, which `NoMatches`/`QueryError` expresses exactly.
- Recommended correction: narrow each to `except (NoMatches, QueryError)`, matching the 76 sites that already do. Keep the bare `Exception` only at `console_selection_menu.py:618` (`self.screen` can raise `NoScreen`, a different class).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — the graphics→pixels image fallback is duplicated in three Console widgets and one of the three swallows the failure silently
- Where: `console_generation_card.py:219-231` (logs `logger.opt(exception=True).warning("textual-image unavailable; falling back to pixels…")`), `console_image_viewer_modal.py:130-145` (`except Exception: pass`, no log), `character_expression_avatar.py:154-160` (catches 6 named exception types, sets `self._mode = "pixels"`, no log). All three docstrings say they "mirror" `ConsoleTranscript._image_row_widget`.
- Evidence: read; the three bodies are the same ladder (`textual_image.widget.Image` → `fit_image_cell_size` → `Pixels.from_image`) with three different failure policies.
- Why it matters: when the graphics path breaks, two of the three surfaces degrade with no diagnostic at all, so "why is my image a mosaic?" is unanswerable from the log for the viewer and the avatar but answerable for the generation card.
- Recommended correction: one helper (`Chat/console_image_view.py` already owns `fit_image_cell_size` and `resolve_default_mode` — it is the natural home) returning the widget and doing the logging once.
- Size: M · ADR: no · Confidence: verified (read only; the drift is in the source, not inferred)
- Pinning test: none
- Already covered: none

## Candidate dispositions
| candidate (excerpt row) | disposition |
|---|---|
| `query_one_in_timer_no_try` — `console_agent_progress_modal.py:102`, `:136` | **retired** — both ids composed unconditionally (`:64`, `:69`); interval started in `on_mount` after compose, stopped in `on_unmount` (`:90-92`) |
| `plain_readback` — `console_assistant_turn.py:143` (×2) | **retired** — `.plain` is read off `Content(f"…")` values this class builds itself (`_label_content`/`_status_content`, `:125-135`); never markup-parsed, and the property is a documented legacy inspection seam |
| `plain_readback` — `console_character_context.py:60 str(self.label)` | **confirmed, but the cause is upstream** — `str(Content)` is lossless; the loss happened at `Button.__init__` when the title was parsed as markup. Folded into the first finding |
| `plain_readback` — `console_image_viewer_modal.py:149 mosaic.plain` | **retired** — `mosaic` is a `rich.Text` built by `Utils/mosaic_render.mosaic_from_image`, and `.plain` is used only to count rows/columns |
| `except_exception_pass` ×8 (`console_agent_steering_bar:202`, `console_character_picker_modal:213`, `console_generate_image_modal:173`, `console_image_viewer_modal:144`, `console_prompts_modal:1932`, `console_review_notes_modal:331`/`:369`, `console_selection_menu:626`) | **retired as data-path swallows** (every one guards a DOM query or a fallback ladder, each with a comment) / **confirmed as an over-broad-`except` consistency issue** — see the P3 D3 finding |
| `except_exception_return_per_file` ×8 files | **retired** — all are documented fail-closed lifecycle guards (`# noqa: BLE001 - lifecycle access fails closed` in `console_auto_speak_consent.py`, teardown guards elsewhere); none returns a wrong value on a data path |
| `function_body_import_per_file` ×9 files (21 imports) | **retired** — every target resolves (`importlib.import_module` + `hasattr` for all 8 distinct modules and their names → all OK), and a warm function-body import measures **0.06 µs/call** (100k-iteration loop). Three (`console_auto_speak_consent.py:470`, `console_conversation_inspector.py:1644`, `__init__.py:106-128`) carry measured first-paint rationale |
| `run_worker_coroutine_per_file` ×13 files | **retired** — no sync sqlite or file I/O inside any of them (`grep` for `sqlite3`/`.write_text`/`.read_text`/`.execute` over the 60 files → 1 hit, `console_conversation_inspector.py:2102`, which is in a `@on(Button.Pressed)` handler, not a worker). Every `exclusive=True` carries `group=` |
| `raw_1024x1024` — `console_conversation_inspector.py:140` | **retired** — `SIZE_THRESHOLD_BYTES = 1 * 1024 * 1024`, a named constant with a docstring |
| `raw_mkdir` — `console_conversation_inspector.py:2101` | **retired** — preceded by `_validated_export_destination` → `Utils/path_validation.validate_path(path, Downloads, redact_paths=True)` (`:1671-1720`) |
| `strftime` — `console_conversation_inspector.py:2094` | **retired** — export filename timestamp, not a stored/wire format |
| `try_import_guard` ×4 | **retired as import guards** (`textual_image` is optional; every one has a real fallback) / **confirmed as the three-way fallback drift** — see the P3 D4b finding |
| `legacy_markers_per_file` ×8 (32 in `console_conversation_inspector.py`) | **retired** — the 32 hits are `trace_provenance == "legacy_snapshot"`/`"legacy_blob"` (a live data-format discriminator) and prose about the two *retired* modals this file replaced; no dead code |
| `seed_name__perform_safe_cancel` ×15, `seed_name__cancel` ×12, `dup_verbatim` cancel/close clusters | **retired per brief** — `Widgets/modal_dismissal.SafeModalDismissMixin` is the canonical home and every site in this slice routes through `request_safe_cancel`/`dismiss_safe_once`. Checked for drift: `console_prompts_modal.py:1670 _perform_safe_cancel` and `console_model_popover.py:884 action_request_safe_cancel` both add a *guard* step (dirty-guard, Defaults sub-view) before delegating — deliberate, not drift |
| `seed_name__maybe_await` — `console_prompts_modal.py:153` | **confirmed** — see the P2 D4b finding (66 copies repo-wide, no drift) |
| `seed_name__now_iso` — `console_scope_picker_modal.py:237` | **retired** — `_now_iso` there returns `datetime.now(UTC).isoformat()` for an in-memory scope snapshot, not a stored format |
| `seed_name__set_status` ×4 | **retired** — four unrelated 2-4 line widget-local status writers on four different DOM ids; no shared behaviour to extract |
| `dup_shape` — `sync_state@console_retrieval_scope_row:145` / `console_workspace_details:55` / `console_project_instructions:461` | **partly confirmed** — the three are the same equality-guarded-recompose shape, which is the house pattern, not duplication. But `console_project_instructions.sync_preview` (`:452`) is the same shape *without* the guard — see the P3 D2 finding |
| `dup_shape` — `_show_page`/`await_detachment`/`_detach`/`*_menus_on_screen` in `console_conversation_action_menu` vs `console_workspace_action_menu` | **unverified (out of slice)** — `console_workspace_action_menu.py` is in W-console-2's half; the pair looks like a genuine copy of a ~90-line menu-paging block. Check: `diff <(sed -n '230,300p' console_conversation_action_menu.py) <(sed -n '216,290p' console_workspace_action_menu.py)` |
| `dup_shape` — `_move_highlight`/`_filter_submitted`/`_select_highlighted` in `console_prompt_picker_modal` vs `console_style_picker_modal` | **unverified (half out of slice)** — `console_style_picker_modal.py` is in W-console-2's half |
| `dup_shape` — the five large button-handler-shape clusters (114/45/28/25 copies) | **retired** — these are 2-line `@on(Button.Pressed)` adapters whose *shape* (stop the event, call one method) is identical by construction; there is no behaviour to share |
| `dup_verbatim` — `_format_seed@console_video_card:84` / `console_generation_card:137` | **unverified (half out of slice)** — `console_video_card.py` is in W-console-2's half |
| `dup_verbatim` — `_keep@console_video_capacity_modal:223` / `console_prompt_comparison_modal:145` | **unverified (half out of slice)** |
| `dup_shape` — `_truncate@Agents/run_hooks:193` / `_preview_text@console_feedback_comment_modal:32` / `cap_quote@console_selection:136` | **confirmed, and wider than the row says** — see the P2 D4b truncation finding |
| `console_settings_modal.py:5400` unguarded `query_one` in a debounced timer | **not in this slice** — `console_settings_modal.py` (7,601 lines) is in the second half. Left to W-console-2 |

## Verified-fine
- **`console_conversation_inspector.py` Collapsible titles** — every `Collapsible(title=...)` in the file goes through `Content.from_text(..., markup=False)` and the code comment at `:368-378` states the hazard exactly ("a model id containing `[test]` would render mangled and one containing `[/]` raises MarkupError inside compose(), taking the whole modal down with it"). This is the third in-repo statement of the defect class in my first finding and the correct fix; cited there as corroboration, not as a separate defect.
- **`console_conversation_inspector.py:2079 _save_json` / `:2050 _copy_json` do `copy.deepcopy` + `json.dumps(indent=2)` + `write_text` on the message pump** — measured, not a finding. A 400-message payload with 16 KB messages (6.6 MB of JSON) costs `deepcopy 0.3 ms / dumps 9.4 ms / write 0.9 ms` = **10.7 ms total**; a realistic 1.7 MB payload is 3.5 ms. `Tests/Architecture/test_no_blocking_io_on_message_pump.py`'s docstring already states this exclusion as a decision ("`mkdir(exist_ok=True)` plus a small JSON write measures 0.049 ms … flagging them would produce a large noisy baseline"). Retired.
- **Worker-outlives-dismissed-modal in `console_conversation_inspector`** — `textual/widget.py:4849 Widget._on_unmount` calls `self.workers.cancel_node(self)`, so `_load_snapshot`/`_load_turn_captures` are cancelled when the modal is popped. (One residual teardown edge is in Left UNVERIFIED.)
- **`console_prompt_queue_modal.py:208-237`** — the `query_one` block inside the 0.2 s `_poll_snapshot` → `_apply_snapshot` path is wrapped in `try/except NoMatches: return`.
- **`console_character_context.py:134 _sync_progress_counts`** (0.5 s `set_interval`) uses `self.query(CharacterConversationButton)`, which returns an empty `DOMQuery` rather than raising — no crash path.
- **`console_command_popup.py:116 set_timer(0.1, self.reposition)`** — `reposition` guards `self.parent is None`, `NoMatches`, and a catch-all that logs.
- **No `get_cli_setting`/`load_settings` call anywhere in this 60-file slice** (`xargs grep` over the file list → 0 hits), so the 11 ms-per-call D2 class does not apply here.
- **`run_worker(exclusive=True)` always carries `group=`** in this slice — `console_composer_bar.py:1000/1017` (`group="console-prompt-history"`), `console_character_picker_modal.py:274` (`group="console-character-picker-search"`), `console_conversation_inspector.py:121/743/846` (`group=_NEXT_SEND_WORKER_GROUP`) and `:330` (`group="trace-viewer-profile"`).

## Retired
- **`console_model_popover.py:1372` `str(block.renderable)` — symptom real, cause wrong, retired.** `Static.renderable` genuinely does not exist in Textual 8.2.8 (`$PY -c "from textual.widgets import Static; Static('x').renderable"` → `AttributeError: 'Static' object has no attribute 'renderable'`), and the line is reachable (the button can be disabled between the press and the check, because `_validated_draft` → `_rebase_to` → `_sync_default_content` re-evaluates `button.disabled`). But `tldw_chatbook/__init__.py:78 _install_textual_compatibility_shims()` re-adds it as a property, and it IS called — from `tldw_chatbook/UI/__init__.py:5` and `tldw_chatbook/Widgets/__init__.py:5`. Verified: `$PY -c "import tldw_chatbook.Widgets; from textual.widgets import Static; print(repr(Static('hello', markup=False).renderable))"` → `'hello'`. (Importing only the top-level `tldw_chatbook` does **not** install it — that is what made this look broken.) 50 `.renderable` sites repo-wide depend on that package-`__init__` side effect.
- **`console_prompt_queue_modal.py` timer crash via `entry.preview`** — raised, then retired: `Chat/console_prompt_queue.py:171 make_prompt_preview` runs `rich.markup.escape`, which DOES escape a close tag (`[/WIP]` → `\[/WIP]`, first char `/` is in its `[a-z#/@]` class), so no `MarkupError` reaches the 0.2 s poll. The *silent deletion* half survives (`[WIP]a[/WIP]` renders as `a[/WIP]`) and is folded into the first finding.
- **`console_conversation_inspector.py` export on the message pump** — measured at 10.7 ms for a 6.6 MB payload; see Verified-fine.
- **`console_prompts_modal.py:1932 except Exception: pass`** — the swallowed call is a capabilities re-fetch; the very next statement re-checks `_supports_structured_save` and the user gets an explicit "save is unavailable" notify. No data path, no silent success.
- **`console_agent_progress_modal.py:102/136` (mechanical `query_one_in_timer_no_try` rows)** — both ids (`SelectionList` at `:69`, `#agent-progress-count` at `:64`) are composed unconditionally, the interval is started in `on_mount` after compose and stopped in `on_unmount` (`:90-92`). No unguarded-id crash.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The `[/…]` MarkupError in `console_character_context.compose()` / `ConfirmationDialog.compose()` actually takes the app down (rather than being caught by a Textual error boundary) | Requires running the TUI, which the brief forbids for this slice | tmux recipe from `.claude/skills/verify/SKILL.md`: `tmux -L verify new-session -d -s w1 -x 120 -y 40 'cd <worktree> && source <SCRATCH>/env.sh && $PY -m tldw_chatbook.app'`; create a conversation titled `notes a [/] b`, open the Console Context rail, `tmux -L verify capture-pane -p -t w1` |
| `console_conversation_inspector._load_snapshot`'s `finally: self.next_send_loading = False` fires its watcher (`watch_next_send_loading`, unguarded `query_one` at `:1148-1150`) during the worker cancellation that `Widget._on_unmount` triggers, on a screen whose children are already removed | Needs a driven `run_test` pilot that dismisses the Inspector while `snapshot_factory` is still awaiting; reading cannot settle the ordering of `cancel_node` vs child removal | `cd <worktree> && source <SCRATCH>/env.sh && $PY -m pytest Tests/UI/test_console_conversation_inspector.py -q` after adding a case that pushes the modal with a `snapshot_factory` blocked on an `asyncio.Event`, presses escape, then sets the event |
| `console_conversation_action_menu.py` `_show_page`/`await_detachment`/`_detach` are a verbatim copy of `console_workspace_action_menu.py`'s | `console_workspace_action_menu.py` is in W-console-2's half of the package | `diff <(sed -n '230,300p' tldw_chatbook/Widgets/Console/console_conversation_action_menu.py) <(sed -n '216,290p' tldw_chatbook/Widgets/Console/console_workspace_action_menu.py)` |
| `console_terminal_workspace.py:168/199/369` `isinstance(self.renderable, Text)` is dead under the `__init__` shim (which returns `Content`/`str`, never `rich.text.Text`) | That file is in W-console-2's half; the branch may be reached with a value written by `update(Text(...))` rather than read from `content` | `cd <worktree> && source <SCRATCH>/env.sh && $PY -c "import tldw_chatbook.Widgets; from textual.widgets import Static; from rich.text import Text; s=Static(''); s.update(Text('x')); print(type(s.renderable))"` |
