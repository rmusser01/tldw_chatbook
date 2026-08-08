# Toad TUI Study — Potential Improvements for tldw_chatbook

Source studied: https://github.com/batrachianai/toad (batrachian-toad 0.6.20, Textual 8.2.7, Python 3.14), cloned 2026-08-05. Key source paths cited as `src/toad/...` in the toad repo.

**Decision recorded:** tldw_chatbook will pin the latest Textual version (user decision, 2026-08-05), which unlocks the Textual 8.x APIs several items below rely on (`TextArea.suggestion`, `MarkdownStream`, `lazy.Reveal`, `getters`, `var(bindings=True)`).

Items 1–6 are the prioritized set; backlog tasks were created for 1, 2, 3, 4, and 6. Items 7+ are additional candidates awaiting decision.

## Priority items (1–6)

1. **Modal Enter/Shift+Enter chat input + auto-grow box.** Two bindings on the same keys with mutually exclusive `check_action` driven by a `var(bindings=True)` multi-line flag: Enter sends when single-line; once text contains `\n` or ` ``` `, Enter inserts a newline and Shift+Enter sends. Footer/binding state refreshes automatically. Auto-grow via `dock: bottom; height: auto` container + `height: auto; max-height: 50vh` TextArea with `compact = True` and a `:focus-within` accent border that changes color per mode. (toad: `widgets/prompt.py:103-134,235-242,666-675`, `screens/main.tcss:473-483`.)

2. **Streaming rendering overhaul.** Per-turn `MarkdownStream` widgets (incremental tail re-parse instead of full recompose) + global `Markdown { layout: stream; }` + height-based DOM pruning: track `virtual_size.height` against low/high watermarks, batch-`remove_children()` oldest messages (with margin-collapse math), then re-anchor scroll. Bounds layout cost for marathon sessions without real virtualization. Also: mount a `Loading` placeholder on submit, kill it on first fragment, and `await asyncio.sleep(0)` after mounting the user message so it paints before the network worker starts. (toad: `widgets/agent_response.py:70-77`, `widgets/conversation.py:1519-1578,817-818`, `toad.tcss:6-8`.)

3. **Ghost-text suggestions + persistent input history.** Fish-shell-style ghost text via Textual 8.x `TextArea.suggestion`, fed by a ~30-line prefix-map completer over history; JSONL-per-line history with all IO in `asyncio.to_thread`; index convention 0 = live draft buffer (stashed as pseudo-entry so recall never loses your draft) with `validate_*` clamping; Up/Down recall only when cursor is on first/last visual row (`wrapped_document.height` accounts for soft-wrap). (toad: `history.py`, `complete.py`, `widgets/prompt.py:223-300`, `widgets/conversation.py:426-473`.)

4. **Adopt `textual-diff-view` for tool-call edit rendering.** Standalone pip package (McGugan-authored, no toad coupling): `DiffView(path_a, path_b, code_a, code_b, split=..., annotations=..., wrap=...)`, `await diff.prepare()` off-thread before mount. Transport pattern: store before/after contents on tool-call records and diff client-side with `difflib` (`get_grouped_opcodes` gives hunk folding free). Scales via custom `Visual` strip rendering (~6 widgets for a 500-line diff), auto-split on resize, char-level inline diff tinted with `color: transparent` + low-alpha background. Foundation for a future notes-sync conflict diff view (task-97 adjacent). Requires ADR (new dependency). (toad: `widgets/diff_view.py`, `screens/permissions.py`; library: `textual_diff_view/_diff_view.py`.)

5. **Schema-driven settings system.** Settings defined once as nested dicts (`key/title/type/help/choices/default/validate`) generating the whole form recursively; dotted-key store with typed reads and default fallthrough (sparse on-disk JSON); dirty-flag + atomic threaded saves only on dismiss/quit; commit-on-blur for text inputs (validation failure → toast + revert) vs instant apply for toggles; `prevent(Changed)` during form construction; blur-before-dismiss; live-apply via App CSS class toggles + `textual.Signal`; in-form search via container `name` + display toggling; `lazy.Reveal` for instant modal appearance. (toad: `settings.py`, `settings_schema.py`, `screens/settings.py`, `app.py:571-637`.)

6. **Frame-aligned stream batching + timer/coalescing hygiene.** Batch LLM token streams and subprocess reads to ~frame time (first await, then drain ≤1/60s — `shell_read.py` pattern) instead of message-per-token; guard every `set_interval` animation with `screen.is_active` so hidden tabs burn no CPU; coalesce file-sync/watchdog event storms with message `can_replace` instead of custom debounce timers. (toad: `shell_read.py:6-42`, `widgets/future_text.py:61-65`, `directory_watcher.py:27-28`.)

## Additional candidates (7+)

7. **Syntax-highlighted input.** Override `TextArea.get_line` with a cached per-line `Content` pipeline: Pygments markdown (with the `text + "\n```"` fence-closing trick), regex overlays for `@note`/`@media`/command tokens, theme-variable-mapped token styles. (toad: `widgets/highlighted_textarea.py`.)

8. **Cursor-driven popovers + atomic mention tokens.** `watch_selection` diffs cursor positions to trigger pickers and snap selections so `@mention` tokens and commands behave as atomic units (delete/replace whole token). Popover pattern: `display: none` + `overlay: screen` toggled by `var(toggle_class=...)`, priority bindings, `Dismiss` on `on_descendant_blur`. Applicable to note/character/prompt-template pickers. (toad: `widgets/prompt.py:386-427,681-728`, `widgets/path_search.py`.)

9. **Slash-command completion popover.** Fuzzy-scored `OptionList` popover over `/` commands with inline ghost argument hints baked into the highlight pass; built-ins merged with dynamically advertised commands; local-first execution with fallback to the LLM. (toad: `widgets/slash_complete.py`, `widgets/conversation.py:1321-1357,1860+`.)

10. **Danger/safety highlighting for shell-like inputs.** Parse input (bashlex or a simpler classifier) and paint destructive/out-of-root commands red while typing — UX-layer complement to `input_validation.py`. (toad: `danger.py`, `widgets/prompt.py:190-203`.)

11. **Layered escape + double-escape cancel.** Input consumes Escape for suggestion/popover dismissal then `raise SkipAction()` to defer to the screen; double-Escape within 3s aborts streaming. (toad: `widgets/prompt.py:777-786`, `widgets/conversation.py:1720-1729`.)

12. **Paste handling (surpass toad).** Toad has none — no image paste, no large-paste collapsing. Opportunity to add Claude-Code-style `[Pasted text N lines]` placeholders and clipboard-image attachments, which toad lacks.

13. **`data_bind` chains for streaming/agent-ready state.** Screen → container → textarea reactive binding instead of manual setter plumbing; gate submit with bell + flash when not ready. (toad: `widgets/conversation.py:501-509`.)

14. **Time-derived `auto_refresh` animations.** Spinners/throbbers as widgets with `auto_refresh = 1/15` whose `render()` computes frame state from `monotonic()` — no timers, no drift, repaint scoped to the widget, self-terminating. For "agent thinking" and RAG-indexing indicators. (toad: `widgets/throbber.py`, `widgets/strike_text.py`.)

15. **Two-tier fuzzy search.** Trigram (or FTS5) prefilter → off-thread scoring (`asyncio.to_thread` / `@work(exclusive=True)`) → per-query LRU cache, for notes/media search and command palette. (toad: `fuzzy_index.py`, `widgets/path_search.py:228-265`.)

16. **Startup optimizations.** Lazy screen factories with in-function imports, heavy widgets imported inside handlers, non-critical network deferred via `set_timer(1, ...)`, `gc.freeze()` after main screen mount, `PAUSE_GC_ON_SCROLL = True`. (toad: `app.py:211-228,277,664`, `screens/main.py:240-243`.)

17. **Paint sequencing polish.** `await asyncio.sleep(0)` after mounting the user's own message before starting the LLM worker; immediate `Loading` placeholder removed on first fragment; `call_after_refresh` for scroll-sensitive follow-ups. (toad: `widgets/conversation.py:817-818,1511-1512`.)

18. **Reactive class-flip hygiene.** `var(False, toggle_class="-expanded")` for state→style in one assignment; `set_class(..., update=False)` batching when flipping several classes; `check_action` returning `None`/`False` to keep footer bindings honest. (toad: `widgets/tool_call.py:65-66`, `widgets/flash.py:74-75`.)

19. **Pin Textual to latest + API audit (prerequisite).** Pin `textual` to the current 8.x in `pyproject.toml`, remove the `>=3.3` floor, and audit affected call sites (chat input, Markdown streaming, settings) before/with items 1–5.
