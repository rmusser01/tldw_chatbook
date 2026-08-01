# Console Prompt Workbench and Improvement Design

**Date:** 2026-08-01
**Status:** Approved design, pending written-spec review
**Scope:** Console prompt discovery, improvement, structured authoring, and
Prompt Library interoperability

## Goal

Give Console users one keyboard-first place to find saved prompts, improve the
current unsent prompt with the active provider and model, and create or follow
reusable structured prompt recipes without silently changing session state.

Success means:

- Prompts are reachable from a new top-row `Prompts` action.
- The current unsent message can be auto-improved, improved for review, or
  mapped into an editable structured recipe.
- The effective current system prompt may be included as analysis context, but
  it is never sent or applied without visible user control.
- Structured prompts and recipes are first-class items in Library > Prompts.
- Every model-dependent action is explicit, cancellable, side-effect-free, and
  honest about provider, model, stale state, and failure.
- Existing legacy prompts and consumers continue to work through compiled
  `system_prompt` and `user_prompt` text.

## Non-goals

- Improving sent transcript messages or full conversation history.
- Sending RAG results, tools, attachments, staged sources, or conversation
  history to the improvement request.
- Showing a quality score, issue list, explanation, or diff in Review mode.
- Switching providers or models inside the modal.
- Automatically saving improvement results to the Prompt Library.
- A drag-and-drop canvas, recipe marketplace, collaborative editing, or
  federated search across local and server sources in one result list.
- Replacing the full Library > Prompts management surface.

## Existing Boundaries

The design follows these existing seams:

- `ConsoleContextModal` supplies the visual lineage for a large Console modal
  and demonstrates how the next-send system prompt and composer draft are
  assembled. The new modal copies its visual language but neither subclasses
  nor reuses the read-only context viewer.
- `ConsolePromptPickerModal` supplies the existing debounced, token-gated,
  keyboard-first search behavior. The new modal replaces neither its command
  compatibility nor its simple picker contract.
- `ConsoleProviderGateway` resolves the active provider, model, credentials,
  endpoint, and generation settings. Improvement calls extend that gateway
  rather than resolving providers independently.
- The Prompts database already stores `prompt_format`,
  `prompt_schema_version`, `prompt_definition`, `system_prompt`, and
  `user_prompt`. No new prompt table or database migration is required.
- Local prompt search already uses FTS. The server adapter exposes prompt
  search, but the current unified scope service rejects server search. This
  mismatch must be resolved before server search is presented as available.

## 1. Console Entry Point and Action-Row Contract

Add `Prompts` immediately after `New tab`. Move `Settings` to the end of the
top action row immediately before `Help`:

1. New tab
2. Prompts
3. Attach context
4. Run Library RAG
5. Save Chatbook
6. Settings
7. Help

This feature amends the one-row action invariant in
`2026-07-21-console-top-area-layout-design.md`. The action set must never be
silently clipped:

- Wide terminals show full labels on one row.
- Medium terminals use `New`, `Prompts`, `Attach`, `RAG`, `Save`, `Settings`,
  and `Help` on one row.
- Narrow terminals use two deterministic rows in the same logical order.
- `Settings` remains immediately before `Help` at every width.

Activating `Prompts` opens `ConsolePromptsModal` in Browse mode. The modal uses
the existing context modal's border, header rhythm, and maximum footprint, but
sizes responsively instead of fixing itself at 95 by 40 cells.

## 2. Unified Modal Navigation

`ConsolePromptsModal` is one stable shell with internal modes rather than a
chain of nested modal screens. Its header contains a compact location line,
for example `Prompts / Improve / Structured recipe`. When a saved artifact is
open, the header also shows `Local` or `Server`, its type, and whether the view
is an unsaved working copy.

The modes are:

- **Browse:** search and inspect saved Prompts and Recipes.
- **Edit:** edit a selected Prompt as blocks without changing its source.
- **Improve:** choose Auto, Review, or Structured recipe.
- **Recipe:** choose, populate, edit, and optionally save a recipe layout.

Back returns to the preceding internal mode and restores its most recent
focus. Escape behaves like Back. If the working copy is dirty, Escape offers
only `Keep editing` and `Discard changes`. Dismissing the modal always returns
focus to the Console composer.

Browse and manual editing remain available when the active provider is
unavailable. Only model-dependent actions are disabled, with the provider
reason and a focused recovery action shown in place.

## 3. Browse and Source-Aware Search

Browse opens with:

- `Improve My Prompt` at the top.
- A Local or Server source selector.
- A search input.
- A bounded, scrollable result list.

Search covers names, tags, and prompt content within the selected source. It
does not merge sources. Each result identifies Prompt or Recipe, source,
System/User/combined content, and last-updated time.

Local search uses the existing FTS seam. Server search must call the server
search endpoint through `PromptScopeService`; client-side filtering of only a
loaded page is forbidden because it presents incomplete results as complete.
If server search is not supported by the connected server or current policy,
the modal says so and continues to offer source switching.

Search uses debounce plus a monotonic search token. A late completion cannot
replace newer results. The UI distinguishes:

- An empty Library.
- No matches for the current query.
- Search failure with Retry.
- Selected source unavailable.
- A selected artifact that was changed or deleted before its detail fetch.

Selecting a result fetches the latest complete record before opening it. The
working copy captures its source identity and optimistic version. Selecting a
Prompt opens its content in the shared block editor. Selecting a Recipe creates
a new unsaved Prompt working copy based on the recipe; a Recipe is never
inserted directly into the composer.

## 4. Structured Artifact Contract

### 4.1 Namespaced definition

Structured artifacts use the existing prompt columns:

- `prompt_format = "structured"`
- `prompt_schema_version = 1`
- `prompt_definition` contains a namespaced block document
- `system_prompt` and `user_prompt` contain compiled compatibility text

The top-level definition has this shape:

```json
{
  "kind": "block_prompt",
  "schema_version": 1,
  "lanes": [
    {
      "id": "system",
      "blocks": []
    },
    {
      "id": "user",
      "blocks": []
    }
  ]
}
```

Recipes use the same shape with `kind = "block_recipe"`. The kind is required
because existing structured definitions are heterogeneous and a version number
alone does not identify their meaning.

Each persisted block contains:

- `id`: stable string identity.
- `title`: editable display and Markdown heading text.
- `syntax`: `freeform`, `markdown`, or `xml`.
- `xml_tag`: required only for XML blocks.
- `content`: prompt content, or optional Recipe starter content.
- `mapping_hint`: optional Recipe guidance describing what belongs here.

Lane membership and order come from the parent lane and block-array order.
They are not duplicated inside the block. The in-memory editor model may expose
lane and position as derived values.

`prompt_definition` is canonical for a structured record. Compiled text is
regenerated on every structured save. If stored compiled text does not match
the definition, the editor shows that compatibility text is stale and uses the
definition; a successful save repairs the compiled fields. It never silently
chooses the stale text.

### 4.2 Compilation

Compilation is deterministic and preserves block content internally:

- Free-form emits its content without adding a heading or wrapper.
- Markdown emits `# {title}`, a blank line, then the content.
- XML emits `<{xml_tag}>`, the content, then `</{xml_tag}>`.
- Blocks are separated by two newlines.
- Empty blocks remain represented in the definition but do not create
  compatibility text unless the user has intentionally entered structural
  content that compiles non-empty.

XML tag names must meet XML name rules. A wrapper tag may not collide with a
matching opening, closing, or self-closing wrapper inside that block's raw
content. Invalid blocks retain their content but cannot be applied or saved.

### 4.3 Legacy prompt decomposition

Opening a legacy Prompt does not make a model call. A conservative,
fence-aware parser recognizes only complete top-level Markdown headings and
complete top-level XML wrappers. Ambiguous or unrecognized material becomes a
Free-form block. The parser never guesses nested structure.

An unchanged legacy lane reapplies its exact original text. Once the user
changes its block content or structure, the lane compiles through the
deterministic structured compiler.

`Structure with AI` is a separate explicit action. It is never triggered by
opening a Prompt.

### 4.4 Markdown import and export

The existing human-readable `### SECTION ###` Markdown grammar remains the
compatibility layer. Structured exports retain the normal `TITLE`, `AUTHOR`,
`SYSTEM`, `USER`, and `KEYWORDS` sections, then append:

````markdown
### STRUCTURE ###
```json
{"kind":"block_prompt","schema_version":1,"lanes":[]}
```
````

Export writes the canonical definition without transforming block content.

The importer extends its exact section map with `STRUCTURE`, parses the fenced
JSON, and validates kind and version before using it. Importers that do not
know `STRUCTURE` still see the preceding compiled System and User sections and
ignore the unknown section under the existing next-section terminator rule.
Exact block restoration requires the new importer. Unknown future kinds or
versions import as legacy compiled text rather than being partially
interpreted.

## 5. Built-In and Saved Recipes

The built-in Outcome-first recipe follows the supplied GPT prompting guidance:

**System lane**

1. Role
2. Personality
3. Collaboration style

**User lane**

1. Goal
2. Success criteria
3. Context and evidence
4. Constraints
5. Output
6. Stop rules

When source content does not fit a named block, Structured mapping appends an
`Additional context` block. Missing information remains blank. The mapper must
not invent requirements, facts, evidence, names, metrics, or constraints.

Structured mode lets the user select Outcome-first, a saved Recipe, or Blank
custom recipe, then choose `Fill from draft with AI` or edit manually. Manual
editing never makes a provider request.

Saving a Recipe stores its lane, order, title, syntax, XML tag, and mapping
hint. `Include current text as starter content` is available and defaults off.
Saving a populated artifact as a Prompt always stores its content.

Prompts and Recipes are labeled distinctly in both Console Browse and Library
> Prompts. The Library structured editor is the same block editor contract;
compiled text may be previewed but is not a second editable source of truth.

## 6. Block Editor Interaction

System and User lanes are stacked vertically so the editor remains usable in
narrow terminals. A non-empty lane opens expanded. Each block exposes:

- Editable title.
- Free-form, Markdown, or XML syntax selector.
- XML tag input when XML is selected.
- Editable multiline content.
- Move up, move down, duplicate, and delete actions.

Each lane ends with `Add block`. Reordering uses explicit actions and keyboard
commands rather than drag and drop. Block IDs remain stable across reorder and
format changes.

Wide block headers place metadata and actions on one line. Narrow headers use
two lines. Controls are never silently removed or replaced only by unexplained
symbols. Editing one block must not recompose other TextAreas or lose their
cursor, selection, scroll, or native undo state.

The fixed modal footer uses two rows when needed:

1. Lane-selection options and validation status.
2. Back, Save as Prompt, Save as Recipe, and the primary Apply action.

For structured application:

- A non-empty User lane is selected for application by default.
- `Apply system prompt to this session` appears when the System lane has
  content and defaults off.
- Empty or unselected lanes are no-ops; they never clear current session
  state.
- Apply is disabled when no selected lane contains content.
- Clearing a draft or session system prompt remains an explicit action outside
  this flow.

Validation appears beside the affected block. Apply and Save remain disabled
until all blocking errors are resolved. The first invalid block receives focus
when either action is attempted.

## 7. Improve My Prompt Modes

The Improve entry presents exactly three choices:

1. Analyze and auto-improve
2. Analyze and user review
3. Create or follow a structured recipe

The effective current session system prompt and current unsent composer
message are copied into a request snapshot. `Include system prompt as analysis
context` defaults on. It controls only request context; it does not grant
permission to change the system prompt.

Auto and Review require a non-empty unsent message. Recipe authoring remains
available with an empty message, but `Fill from draft with AI` is disabled
until a draft exists.

### 7.1 Improvement quality contract

Improvement targets the shortest prompt that preserves intent and materially
improves model steerability. It favors outcome, success criteria, relevant
constraints, available evidence, expected output, and stopping conditions over
legacy step-by-step process narration.

The optimizer must preserve the requested artifact, language, audience,
length, genre, factual claims, safety and business invariants, required output
fields, and explicit side-effect limits. It may remove redundant process
instructions, but it may not weaken a real invariant, invent missing context,
add unsupported claims, or expand a simple prompt into headings that do not
improve comprehension. Personality and collaboration style remain short and
separate when the source actually calls for them.

These are starting patterns, not a demand that every improved prompt use every
Outcome-first block. Auto and Review may remain concise free-form prompts.

Reference:
[GPT-5 Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide).

### 7.2 Auto

Auto requests only a rewritten User prompt. The system prompt, when included,
is analysis context and is never rewritten. A valid result replaces the
composer draft as one explicit draft transaction and closes the modal.

The Console exposes a temporary `Undo improvement` action backed by the exact
pre-apply draft snapshot. The transaction remains valid until the next manual
draft edit, send, session switch, or improvement application. Attachments and
unrelated composer state are untouched. A `Ctrl+Z` binding may be advertised
only if it is proven to invoke this exact transaction without stealing normal
TextArea undo; the design does not assume current Textual behavior provides
that guarantee.

If deterministic preservation checks fail, Auto does not replace the draft.
It opens the result in Review with only the generic status `Review required
before applying`; it does not add issue details, a diff, or an explanation to
Review mode.

### 7.3 Review

Review displays only the rewritten User prompt in one editable TextArea. It
shows no score, issue list, diff, explanation, or hidden analysis. `Apply`
replaces the draft after stale-state validation; `Cancel` changes nothing.

### 7.4 Structured recipe

Structured mode sends the selected Recipe definition and asks the model to map
the source into known block IDs. The response opens in the shared block editor
and is always reviewed before application.

If the optional system context was included, the result may populate both
lanes. Applying the System lane still requires the separate unchecked
`Apply system prompt to this session` control.

If system context was excluded, the mapper cannot reconstruct or infer the
omitted session system prompt. It may populate a System block only from role or
behavior instructions explicitly present in the draft or Recipe starter text.

## 8. Improvement Request Boundary

Introduce a focused `PromptImprovementService`. The UI passes it an immutable
snapshot containing:

- Mode.
- Draft and draft fingerprint.
- Optional system prompt and system fingerprint.
- Session identity.
- Pinned provider resolution and visible provider/model identity.
- Optional Recipe definition.
- Request identity.

The service calls a typed auxiliary-completion method on
`ConsoleProviderGateway`. It never calls normal Console submit, builds a full
context snapshot, or appends to transcript state.

The auxiliary method reuses the active provider, model, credentials, endpoint,
samplers, and compatible reasoning settings resolved at request start. It
overrides only the behavior required for this side-effect-free operation:

- Non-streaming response.
- Tools and tool choice disabled.
- No chat stop sequences.
- Improvement-specific output allowance.
- Provider-native structured response format when supported.

The gateway's current generic `_chat_api_kwargs` does not forward
`response_format`, so this capability belongs in the new typed auxiliary seam,
not as a UI-level provider bypass.

Trusted optimizer instructions occupy the provider's instruction role. Source
fields are untrusted data serialized as escaped JSON values, not interpolated
between closable XML delimiters. The optimizer is told to rewrite rather than
answer, preserve protected material, avoid invented requirements, and return
only the required response envelope.

Auto and Review expect:

```json
{
  "kind": "prompt_rewrite",
  "rewritten_prompt": "..."
}
```

Structured mode expects a validated `block_prompt` document. Providers with
native schemas use them. Other providers receive a JSON-only instruction.
Local transport cleanup may unwrap one outer response fence and parse the
envelope, but it may not trim or normalize the prompt content itself.

There is one provider call per click. Malformed, empty, unsupported, or
answer-like output is not repaired through a hidden second model call. The UI
retains the working state and offers explicit Retry.

## 9. Preservation and Context Guards

Before Auto application, deterministic validators compare source and result
for protected material, including supported template placeholders, fenced code
blocks, URLs, UUID-like identifiers, and XML wrapper names. Removed or renamed
protected material routes the result to Review instead of applying it.

The structured response validator additionally checks kind, schema version,
known lane IDs, known Recipe block IDs plus one permitted Additional-context
block, stable unique block IDs, syntax, XML tags, and protected-material
coverage. The mapping instruction requires unmatched content to appear in
Additional context, but semantic completeness remains a user-review judgment
rather than a deterministic claim.

The service preflights the optimizer instructions, source snapshot, Recipe,
and expected output against the pinned model's known context limit. It never
silently truncates. When the limit is exceeded, the modal offers applicable
recovery such as excluding the system prompt or using a model with a larger
context window. Model selection remains outside this modal, so that recovery
closes or backs out to the existing Console model control rather than adding a
second model selector. When a provider limit is unknown, a documented
conservative application cap is used and reported honestly.

## 10. Concurrency, Cancellation, and Application

Only one improvement request may be active per modal and session. Starting a
request pins the provider resolution and assigns a request ID. Late or detached
completions are ignored unless request ID, session ID, and source fingerprints
still match.

Cancellation shows `Cancelling...`. Native asynchronous HTTP paths abort where
supported. Synchronous provider work already running through `asyncio.to_thread`
may not be interruptible; it is detached and its eventual result is discarded.
Closing the modal during a request follows the same rule.

If the user changes the session, draft, system prompt, provider, or model while
a request is running, the result becomes a reviewable working copy and cannot
auto-apply. Applying a System lane additionally requires the captured system
fingerprint to match.

User and optional System changes are validated and committed together to the
live session. This is not falsely described as a durable cross-store
transaction. Existing Console behavior updates the in-memory system prompt
before conversation persistence. If that persistence fails, the live session
keeps the applied value and the UI reports:

`Applied to this session, but could not save to the conversation.`

The working copy remains available and a persistence Retry is offered. Silent
rollback is forbidden because the durable outcome may be uncertain.

## 11. Saving, Versioning, and Authority

The editor supports `Save as new` for Prompt and Recipe artifacts. `Update
original` is available only when the source backend exposes a real
`expected_version` contract and the captured version still matches.

The local service must check `expected_version` in the same database
transaction as its update. Server updates must use the server's conditional
version contract. A fetch-then-unconditional-write is not optimistic locking.
If the selected backend cannot provide conditional update, Update is disabled
and Save as new remains available.

A conflict preserves the working copy and offers Reload or Save as new. It
never overwrites silently. Merely browsing, decomposing, or improving a saved
artifact does not increment Library usage metadata. Usage is recorded only
when an artifact is actually applied.

## 12. Typed Outcomes and Error Presentation

Improvement service outcomes are typed:

- Success.
- Empty response.
- Unsupported capability.
- Cancelled.
- Provider error.
- Malformed response.
- Preservation veto.
- Context limit exceeded.
- Stale source.

Generic provider fallback UI strings are never accepted as improved content.
Provider errors remain provider errors even when the normal streaming path
would synthesize visible fallback copy.

Errors keep the relevant source or working copy intact. Search, validation,
provider, stale-state, version-conflict, and persistence failures use distinct
copy and distinct recovery actions. Closing or cancelling before a successful
Apply changes no composer, session, transcript, or Library state.

## 13. Privacy and Observability

Before a model-dependent action, the modal visibly identifies the pinned
provider and model and whether the system prompt will be sent. An improvement
request contains only the draft, optional system prompt, selected Recipe, and
trusted optimizer instructions.

It never contains conversation history, RAG results, attachments, tools,
staged sources, or unrelated session state. Tools are disabled and output is
text-only. Improvement results are not persisted unless the user explicitly
saves them.

Logs and telemetry may record provider, model, mode, duration, input/output
sizes, token counts when available, and typed outcome. They must not contain
prompt text, block content, system text, or response content.

## 14. Testing and Quality Gates

### Unit tests

- Structured definition validation and kind/version rejection.
- Free-form, Markdown, and XML compilation.
- Exact structured save/load and Markdown export/import round-trip.
- Conservative, fence-aware legacy parsing and exact unchanged-legacy apply.
- XML collision handling with content preservation.
- Placeholder, fenced-code, URL, ID, and wrapper preservation vetoes.
- Context preflight without truncation.
- One-call response parsing without hidden repair.
- Typed empty, malformed, unsupported, cancelled, provider-error, and stale
  outcomes.
- Local and server optimistic-version behavior.
- Draft transaction application and exact undo invalidation rules.

### Textual pilot tests

- Top action order at wide, medium, and narrow widths, with no silent clipping.
- Modal behavior at wide, medium, and 80 by 24 layouts.
- Search debounce, stale-result rejection, source switching, and distinct empty
  and error states.
- Keyboard navigation, focus restoration, block reorder, and dirty-state
  confirmation.
- TextArea state survives unrelated block edits and validation refreshes.
- Include-system analysis defaults on; Apply-system defaults off.
- Blank lanes are no-ops and all-empty application is blocked.
- Auto success, preservation-veto-to-Review, Review Apply/Cancel, Structured
  Apply, cancellation, detached late result, and temporary Undo.
- In-memory apply plus honest warning on conversation persistence failure.

### Provider and quality fixtures

Gateway tests use fakes and verify the pinned provider configuration, no tools,
non-streaming request, response format routing, one-call boundary, and absence
of transcript mutations.

A compact reviewed corpus covers outcome-first rewriting, over-specified legacy
prompts, code prompts, template placeholders, embedded adversarial
instructions, exact literals, XML/Markdown mixtures, and structured Recipes.
Deterministic preservation invariants must always pass. Qualitative improvement
uses a documented human rubric rather than a nondeterministic unit-test
assertion.

## 15. Delivery Boundaries

Implementation should be planned as three reviewable stages:

1. Block schema, codec, Library editor integration, import/export, version
   contract, and ADR enforcement.
2. Prompts action, Browse, source-aware search, block editor, responsive
   action row, and working-copy application.
3. Auxiliary provider seam, PromptImprovementService, Auto/Review/Recipe
   flows, cancellation, undo, preservation guards, and quality fixtures.

Each stage must preserve existing legacy prompt behavior and pass its focused
tests before the next stage begins.

## Scope and Decision Record

ADR required: yes

ADR path:
`backlog/decisions/028-structured-prompts-and-auxiliary-improvement-calls.md`

Reason: The feature establishes a durable structured prompt/recipe artifact,
canonical-versus-compiled ownership, optimistic update behavior, a new
auxiliary provider-call contract, and a long-lived Console modal boundary.

Related decisions:

- [ADR-005: Console Workspace Server Readiness](../../../backlog/decisions/005-console-workspace-server-readiness.md)
- [ADR-006: Provider-Aware Generation Settings](../../../backlog/decisions/006-provider-aware-generation-settings.md)
- [ADR-011: Chatbook Workbench UI System](../../../backlog/decisions/011-chatbook-workbench-ui-system.md)
