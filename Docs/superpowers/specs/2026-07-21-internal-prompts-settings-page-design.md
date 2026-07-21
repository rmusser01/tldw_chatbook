# Internal Prompts Settings Page — Design Spec

**Date:** 2026-07-21
**Status:** Approved design, pending implementation plan
**Owner:** Settings / cross-cutting (new `Internal_Prompts` package)

## Goal

Let users view and modify the internal/system prompts tldw_chatbook uses for its
own tooling — RAG reranking, web-search answer synthesis, sub-agent spawn
instructions, summarization defaults, document generation, subscriptions —
from a Settings page, without source modifications.

## Non-goals

- User-authored chat prompts (Prompts DB / `Prompt_Management/`) — unchanged.
- Character cards, world books, chat dictionaries — user data, not internal prompts.
- Prompt *behavior* changes: every shipped default must render byte-identical
  text to today's inline literals.
- Editing prompts on dead/legacy code paths (see Deferred & excluded).

## Decisions (settled during brainstorming)

| Question | Decision |
|---|---|
| Scope | Curated high-value set (~29 prompts), registry designed for incremental onboarding |
| Storage | Sparse overrides in config.toml; defaults live in code |
| Guardrails | Expert nav group, placeholder validation, format-contract notes, never-crash runtime fallback, per-prompt reset |
| Architecture | Central declarative prompt registry; call sites render through it |

## 1. Registry core — new package `tldw_chatbook/Internal_Prompts/`

### `catalog.py` (pure data — imports nothing from config)

Frozen dataclass `PromptSpec`:

- `id` — dotted, stable, e.g. `websearch.answer_synthesis`. **IDs are public
  config API and frozen once shipped**; a rename requires a legacy-alias entry.
- `subsystem` — grouping key for the UI.
- `title`, `description` — human-readable.
- `used_in` — where the prompt fires (module/feature), shown in the impact pane.
- `default` — the full prompt text, moved here from its current home (original
  modules may re-import the constant for compatibility).
- `required_placeholders` — tuple of `{name}` tokens that must survive an edit.
- `optional_placeholders` — tokens that may appear.
- `contract_note` — optional, e.g. "model output must be TRUE or FALSE —
  downstream code parses it".
- `legacy_config_path` — optional dotted config path honored as a fallback tier.
- `applies` — `"live"` (default) or a note like `"next search"` for
  snapshot-at-init consumers; surfaced in the impact pane.

`CATALOG: dict[str, PromptSpec]` is the single source of truth for the
resolver, the Settings page, and future tooling.

### `resolver.py`

- `get_internal_prompt(prompt_id) -> str` — resolved raw template text.
- `render_internal_prompt(prompt_id, **values) -> str` — resolution + safe
  substitution. **Call sites must never run `.format()`-family calls on
  resolved text** (raw `.format()` on user-edited text is a crash vector —
  rerankers do this today at `reranker.py:325/540/616` and must be migrated
  off it). Prompts that take values render via `render_internal_prompt`;
  zero-placeholder prompts are fetched with `get_internal_prompt` and may be
  plainly concatenated (e.g. `f"{prompt}\n\n{text}"` — concatenation cannot
  crash).

**Precedence vs. existing programmatic channels:** where a call site already
accepts an explicit runtime prompt (subscriptions' per-item `custom_prompt`,
caller-supplied `RerankingConfig.system_prompt`/`scoring_prompt_template`),
that explicit value continues to win. The registry replaces only the
hardcoded-default branch (the "if unset, use literal" fallbacks).

**Resolution precedence:**

1. User override: `get_cli_setting("internal_prompts.<subsystem>", "<key>")` —
   read through the existing config cache. The value may be a table
   (`{text, baseline}`) or a plain string (see §2); the resolver extracts the
   text and treats empty/missing as "no override".
2. Customized legacy key: `legacy_config_path` value, honored **only if it
   differs from the shipped default for that key**. Rationale: first-run config
   writes `CONFIG_TOML_CONTENT` verbatim (config.py:3485-3490), so every user's
   file contains the stub `[Prompts]` one-liners; honoring them unconditionally
   would silently downgrade the real multi-paragraph web-search prompts.
3. Catalog default.

**Safe substitution:** replace only the exact `{name}` tokens declared in the
spec's placeholder lists; every other brace is left untouched. This makes JSON
few-shot examples and Ollama's `{{ .Prompt }}` inert by construction, and
validation reduces to "does the text still contain each required token".
Known accepted edge: a literal `{name}` inside example text that coincides with
a declared placeholder will be substituted.

**Never-raises for user-caused problems:** an invalid override (missing
required placeholder, hand-edited config) logs a warning once per prompt ID and
falls back to the shipped default. Unknown prompt ID raises immediately — that
is a programmer error the test suite catches.

**Import hygiene:** `catalog.py` imports nothing from `config.py`; the resolver
imports `get_cli_setting` lazily inside the function. Keeps the package out of
the cold-start import chain and away from config circulars.

## 2. Override storage format

```toml
[internal_prompts.websearch.answer_synthesis]
text = """..."""
baseline = "ab12cd34"   # short sha256 of the shipped default at save time
```

- Sparse: only customized prompts appear in config.toml.
- `baseline` is UI metadata only (stale-default detection); the resolver reads
  `text` and ignores `baseline`.
- Compatibility: a plain string value (hand-written override without the table
  shape) is accepted as `text`. Empty string means "no override".
- Reset-to-default removes the per-prompt table via the existing
  `delete_settings_from_cli_config("internal_prompts.<subsystem>", ["<key>"])`
  (config.py:3756 — verified to handle dotted paths, atomic write, cache reload).

## 3. Curated set (~29 prompts)

| Subsystem | IDs (under prefix) | Source today | Notes |
|---|---|---|---|
| `websearch` (4) | `sub_question_generation`, `result_relevance_eval`, `result_summarization`, `answer_synthesis` | `Web_Scraping/WebSearch_APIs.py` inline literals (~645, ~789, ~809, ~1001) | Fixes the dead `[Prompts]` keys; those become `legacy_config_path` with the differs-from-default rule. Contract notes: relevance eval must output TRUE/FALSE; synthesis has citation-format rules. |
| `rag_reranker` (6) | `pointwise_system`, `pointwise_template`, `pairwise_system`, `pairwise_template`, `listwise_system`, `listwise_template` | `RAG_Search/reranker.py` `__init__` fallbacks | Snapshot-at-init → `applies="next search"`. Contract notes: numeric score / comparator / ranking output parsed by code. Migrate rendering off raw `.format()`. |
| `agents` (3) | `subagent_system`, `console_agent_operating`, `tool_protocol` | `Agents/agent_service.py:49`, `Chat/console_agent_bridge.py:57`, `Agents/agent_runtime.py:157` | Tool-protocol fence markers and the dynamic tool listing are injected as required placeholders (`{fence_open}`, `{fence_close}`, `{tool_list}`) so edits cannot break the parser contract. Verified templatable: the scaffold is static text; its literal JSON example braces are safe under declared-token substitution; the empty-schemas → `""` early return stays code-side. |
| `summarization` (3) | `analyze_default_system`, `local_summarizer_template`, `rolling_summarize_system` | `LLM_Calls/Summarization_General_Lib.py:528`, `LLM_Calls/Local_Summarization_Lib.py:39`, `Chunking/Chunk_Lib.py:268` | Rolling-summarize already config-backed → `legacy_config_path = chunking_config.summarize_system_prompt`. `local_summarizer_template` has zero placeholders — call sites concatenate (verified); its trailing `</s> {{ .Prompt }}` cruft is part of today's default and ships unchanged (cleanup = behavior change → future work). |
| `document_generation` (6) | `timeline_system`, `timeline_user`, `study_guide_system`, `study_guide_user`, `briefing_system`, `briefing_user` | `Chat/document_generator.py` (system prompts ~219/317/415 hardcoded; user prompts config-backed) | User prompts get `legacy_config_path = prompts.document_generation.<type>.prompt`. |
| `subscriptions` (7) | `analysis_system`, `feed_analysis`, `url_change_analysis`, `podcast_analysis`, `generic_analysis`, `recursive_summarizer_system`, `briefing` | `Subscriptions/content_processor.py:272,344-405`, `recursive_summarizer.py:453`, `briefing_generator.py:312` | Per-type prompts keep their runtime `custom_prompt` override paths; registry supplies the defaults. |

### Deferred & excluded

- **Deferred** (onboard later as one spec each): per-provider one-liner
  fallbacks (`Summarization_*_Lib`, `[api_settings.*].system_prompt`),
  OCR/transcription chat templates (model-specific, brittle), the ~30
  `prompt_selector` UI templates (picker UX, different feature), MCP prompt
  templates, the prompt-engineering metaprompt, media-ingestion one-liners.
- **Excluded — dead code:** character-generation prompts in
  `Event_Handlers/conv_char_events.py` (~3542-3877). The `ccp` route resolves
  to the redesigned `PersonasScreen`, which has no generation prompts;
  `conv_char_events` is only reachable via legacy dispatch. If Personas grows
  AI-assisted generation, onboard then.
- **Excluded — dead key:** `[Prompts].situate_chunk_context_prompt` /
  `CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT` has no consumer anywhere. Do not
  onboard a prompt with no call site. (Candidate for a separate hygiene task.)

## 4. Settings page UX

**Placement:** new `SettingsCategoryId.INTERNAL_PROMPTS` in the **Expert** nav
group. Category summary shows "N customized". Excluded from
`GUIDED_SETTINGS_MUTATION_CATEGORIES` — the panel owns its own persistence
(self-contained editor pattern, like the Theme editor); global `s`/`r` are
informational no-ops.

**Panel:** new self-contained widget `Widgets/settings_internal_prompts_panel.py`
(keeps the ~10k-line `settings_screen.py` from growing; detail-pane branch just
yields it).

- Search `Input` on top; below it a `ListView` grouped under subsystem headers.
- Each row: prompt title + badges — **customized** (active resolved text ≠
  shipped default — this covers both a tier-1 override and a customized legacy
  key, which has no override table) and **default changed** (`baseline` ≠ hash
  of current shipped default; only meaningful when an override table exists).
- **Perf constraint:** filtering performs targeted updates (toggle row
  visibility; config read once per refresh) — no recompose or list rebuild in
  the keystroke path (the task-284 bug class). Note: Textual `mount()` silently
  no-ops during pruning — verify targeted-update paths in the live TUI.
- Keyboard-first: arrow navigation, Enter opens the editor.

**Editor modal** (`InternalPromptEditorModal`, `ModalScreen` — pattern:
`Widgets/Console/console_system_prompt_modal.py`): title/description; contract
callout when present; required-placeholder chips; `TextArea` prefilled with the
active text; collapsible read-only "shipped default" section for comparison;
buttons **Save** / **Reset to default** / **Cancel**. A modal (not in-pane
editing) sidesteps recompose-loses-TextArea-state and the cramped middle pane.
The modal captures the stable prompt ID at open; saves are by-ID, so the
Console edit-modal stale-key hazard does not apply.

**Impact pane:** category stats (override count) plus selected-prompt metadata:
`used_in`, placeholders, and the `applies` note for snapshot-at-init prompts.

**Persistence:**

- Save: validate required placeholders + non-empty inline, then
  `run_worker(thread=True, exclusive=True, group="internal-prompt-save")`
  writes the per-prompt table (with refreshed `baseline`) via
  `save_settings_to_cli_config`. Worker body catches exceptions and marshals
  errors back to the modal via `call_from_thread` — never lets `exit_on_error`
  crash the app. Modal dismisses only on confirmed success.
- Reset: `delete_settings_from_cli_config` in the same worker pattern. Reset
  removes the override table **and** any customized legacy key
  (`legacy_config_path`) — otherwise resetting a prompt whose legacy key was
  customized would silently leave the legacy value active. Reset always lands
  on the shipped default.
- Live-apply: the save path reloads the config cache; the resolver reads
  through it, so overrides apply on next use — except snapshot-at-init
  consumers, which the UI labels via `applies`.
- Re-saving refreshes `baseline`, clearing the "default changed" badge; no
  separate acknowledge action.

**CSS:** new `settings-*` classes go in the partials
(`components/_workbench.tcss` / `features/_tools-settings.tcss`) — never the
generated bundle.

## 5. Error-handling model

| Layer | Behavior |
|---|---|
| Save time (modal) | Missing required placeholder or empty text blocks save with inline error. Only layer that talks to the user. |
| Runtime (resolver) | Never raises for user-caused problems: invalid override → warn once per prompt ID → shipped default. Substitution touches only declared tokens. |
| Programmer error | Unknown prompt ID raises immediately; tests cover every migrated call site. |
| Legacy keys | Honored only when the value differs from the shipped default for that key. |
| Config write failure | Existing atomic-write/backup/file-lock infra; worker surfaces failure in the modal. |

## 6. Testing

- **Resolver unit tests:** precedence chain (override → customized-legacy →
  default), legacy stub-equality skip, placeholder validation, brace safety
  (JSON few-shots, `{{ .Prompt }}`), warn-once, both override shapes
  (table/plain string), empty-string-as-no-override.
- **Golden-parity tests:** every migrated default renders byte-identical text
  to the pre-migration literal for representative inputs (guards the
  f-string→template conversion, including `render_tool_protocol`'s dynamic
  tool listing via `{tool_catalog}`).
- **Integration per subsystem:** set an override in a scratch config
  (`TLDW_CONFIG_PATH`), run the real code path, assert the payload at the LLM
  transport boundary changed. Fakes live only at the transport — no accessor
  mocks (the **kwargs-fakes lesson).
- **UI tests** (`app.run_test()`, scratch config profile): category renders,
  search filters without rebuild, modal round-trips save/reset, badges reflect
  config state, worker-error path shows inline error.

## 7. Accepted limitations

- The TOML writer may serialize long overrides with escaped `\n` rather than
  pretty triple-quoted blocks — functionally correct, ugly for hand-editing;
  the Settings page is the primary editor. Verify at implementation; do not
  drag a TOML-writer change into scope.
- Panel-local state (search text, scroll) is lost on category switch —
  consistent with other categories.
- A declared placeholder token appearing literally in example text will be
  substituted (documented above).

## 8. Future work (not in this program)

- Onboard deferred prompt groups (one `PromptSpec` each).
- Export/import of prompt override sets.
- Diff view (override vs. current default) in the modal.
- Hygiene task: remove dead `CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT` and the
  unconsumed `prompts_strings` loader once the web-search trio migrates.
- Hygiene task (behavior change, needs its own review): drop the stray
  `</s> {{ .Prompt }}` suffix from the local summarizer default — it is
  Ollama-modelfile cruft sent verbatim to models today.
