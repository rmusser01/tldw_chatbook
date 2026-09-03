# Console Conversation Settings: Ready-to-Send Design

Date: 2026-09-02
Status: User-approved design direction
Primary surface: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
Related tasks: TASK-145, TASK-178, TASK-191, TASK-366, TASK-14812, TASK-14811.3, TASK-2154.7
Related ADRs: ADR-006, ADR-011, ADR-012, ADR-020

## Summary

Conversation Settings will become a readiness-led configuration flow for both
cloud APIs and locally hosted models. Its primary job is to get the active
conversation to an explicit **Ready to send** state. Durable provider
credentials remain owned by **F9 Settings > Providers & Models**. When a cloud
credential is missing, Conversation Settings deep-links to that exact field and
offers a return path that restores the non-secret conversation draft and focus.

Local-hosting setup remains available in Conversation Settings, but endpoint
persistence becomes an explicit connection action. Model discovery reports
where every choice came from and no longer implies that generation was tested.
Sampling and provider-specific controls move behind progressive disclosure, and
the expert path gains searchable provider selection, compact status, and a
shorter documented keyboard route.

This design covers every issue in the 2026-09-02 Conversation Settings critique:
credential recovery, endpoint false completion, first-run density, model
provenance, naming, test honesty, disabled-action explanations, accessibility,
and minor copy defects.

## Problem

The current modal is mechanically safe but does not reliably complete the setup
task its entry points imply:

- A missing cloud key is described with environment-variable and TOML syntax,
  but the modal has no credential control or targeted recovery action.
- Ordinary **Save** can dismiss after selecting a local endpoint that runtime
  readiness still rejects as not durably saved.
- **Discover models** verifies a listing endpoint, not authentication plus chat
  generation, while stale saved models can remain mixed with current results.
- Provider connection decisions compete with identity, nine sampling controls,
  five provider-specific controls, previews, and long scope prose.
- The product alternates among “Session Settings,” “Console Settings,” and
  conversation-scoped wording.
- Configuration validity, endpoint reachability, credential acceptance, model
  availability, and generation success are collapsed into ambiguous pass/block
  copy.

The resulting peak-end failure is severe: users can close the modal believing
setup succeeded and then discover that sending is still blocked.

## Job And Audience

### First-time cloud user

A person has purchased API access and has a key. They need to connect one
provider, choose the model they bought access to, verify that the configuration
is usable, and begin chatting without editing TOML or understanding environment
variables.

### Local-model operator

A technically experienced user has started Ollama, llama.cpp, vLLM, MLX, or an
OpenAI-compatible server. They need to point Chatbook at its endpoint, discover
served models, distinguish current results from saved/custom IDs, and know
whether listing and generation actually work.

### Established power user

A repeat user needs fast provider/model switching, custom identifiers,
per-conversation tuning, keyboard operation, and explicit control over whether a
change affects only this conversation or future conversations.

## Outcome And Proof

Success is not “the modal accepted the draft.” Success is one of these explicit
states:

1. **Ready to send** — all mandatory local checks pass and no known runtime
   blocker remains.
2. **Ready; credential unverified** — configuration is complete, but the user
   has not run an authenticated network check.
3. **Not ready** — the UI names the single highest-priority blocker, its impact,
   and one direct recovery action.

The active provider, endpoint authority, credential source (never the secret),
selected model provenance, and verification scope remain visible beside the
status. A user returning from canonical Settings sees the exact conversation
draft they left and a refreshed readiness result.

## Ownership And Architectural Boundaries

### Conversation Settings owns

- the active conversation's provider, model, generation, streaming, identity,
  and context-policy drafts;
- readiness presentation and blocked-state recovery;
- local endpoint/model discovery initiated from the conversation;
- the choice between conversation-only application and an existing durable
  provider/default write path;
- creation and consumption of a non-secret return intent for targeted Settings
  recovery.

### Settings > Providers & Models owns

- durable API-key and credential-environment-variable mutation;
- durable provider endpoint and provider-specific API-mode configuration;
- global model-catalog refresh and saved model selection;
- explicit configuration and connectivity tests for provider drafts.

### Shared provider services own

- provider identity/display-name normalization;
- credential-source resolution and redacted readiness facts;
- endpoint normalization and reachability results;
- model-catalog authority and model provenance;
- provider-capability facts used to show only effective controls.

Conversation Settings must not gain a second credential field, secret cache, or
provider registry. Settings must not mutate a conversation draft implicitly.

## ADR Check

ADR required: no new ADR

ADR paths:

- `backlog/decisions/006-provider-aware-generation-settings.md`
- `backlog/decisions/011-chatbook-workbench-ui-system.md`
- `backlog/decisions/012-provider-credential-settings-boundary.md`
- `backlog/decisions/020-automatic-model-catalog-refresh.md`

Reason: ADR-012 already decides that Settings owns durable credentials while
Console owns blocked-send recovery and exact-field navigation. ADR-006 owns the
Settings/Console generation split, ADR-011 requires visible readiness and
recovery, and ADR-020 owns cloud catalog authority. The return intent and modal
restructure implement those accepted boundaries without adding a new storage,
credential, provider, or runtime owner.

## Information Architecture

The modal title is standardized to **Conversation settings**. It retains two
destinations:

1. **Model and generation**
2. **Context and memory**

The first destination is reordered into:

### Connection

1. Provider
2. Credential or endpoint
3. Model
4. Verify/discover
5. Readiness result

This section is always first and remains above the fold at the minimum supported
terminal size. It uses the same compact structure for cloud, local, and custom
providers while changing only the applicable connection control and recovery
action.

### Advanced generation

Collapsed by default. Contains sampling, response limits, streaming, reasoning,
thinking, and provider-specific controls. Controls the selected provider/model
does not consume are omitted. Controls with a constrained vocabulary use a
Select or equivalent enumerated control rather than free text.

### Conversation identity

Collapsed after Advanced generation. It remains conversation-scoped and retains
the global-default explanation.

### Request estimate

Collapsed summary. “0 tokens” and “unknown tokens” are replaced by **Provider
default** and **Not estimated** when that is the actual state.

Context and memory retains its existing separate view, guarded immediate actions,
and save semantics. This design does not redesign memory or compaction policy.

## Readiness Component

The existing prose lines become one compact, text-labeled diagnostic component:

```text
Connection status
OpenAI · GPT-x
Not ready — API key missing
[Configure credential…]
```

or:

```text
Connection status
Custom OpenAI-compatible · qwen-local
Endpoint reachable · Model served now · Generation not tested
Ready; generation unverified
[Test generation]  [Connection details]
```

The component exposes, where applicable:

- provider display name;
- credential source: environment variable, local config, not required, or
  missing;
- endpoint display and whether it is saved, conversation-only, or a draft;
- endpoint reachability: not checked, checking, reachable, refused, timed out,
  or HTTP failure;
- model provenance: served now, current catalog, saved fallback, or
  custom/unverified;
- generation verification: not tested, testing, succeeded, or failed;
- one primary recovery action for the highest-priority blocker.

Internal provider keys, config paths, raw exception prose, headers, credentials,
and credential-bearing URLs never appear in this component.

## Cloud Credential Deep-Link And Return

### Entry

When readiness reports a missing credential, the modal shows **Configure
credential…**. Activating it creates a process-memory return intent containing
only:

- origin Console conversation/session identity;
- the non-secret Conversation Settings draft;
- active modal destination, scroll anchor, and focused logical control;
- provider/model intent;
- an origin revision used to reject stale restoration;
- a single-use opaque return-intent identifier.

The intent must not contain an API key, resolved credential value, request
headers, provider response, prompt, transcript body, or other secret/content.
It is never written to config, database, logs, crash metadata, or navigation
labels.

### Settings destination

The existing Settings navigation context opens:

```text
category = providers-models
provider = active provider
model = active model
field = api_key
return_intent = opaque id
```

Settings focuses the API-key field and presents its existing two credential
paths:

- masked local config key;
- credential environment-variable name, labeled as the safer shell/shared-host
  path.

Settings owns Save/Clear and never pre-fills or echoes a stored secret.

### Return

After successful credential Save, Settings presents **Return to conversation
settings** as the primary continuation. A secondary **Stay in Settings** action
is available. Cancel/back from credential editing also offers return without
claiming success.

On return:

1. Resolve the single-use intent.
2. Verify that the origin conversation still exists and its revision is
   compatible.
3. Reopen Conversation Settings with the non-secret draft restored.
4. Restore the logical focus/anchor when still applicable; otherwise focus the
   Connection section.
5. Reload provider configuration and credential readiness from canonical state.
6. Announce **Credential saved — checking readiness** followed by the resulting
   status.
7. Consume the return intent so refresh/re-entry cannot replay it.

If the origin is missing or stale, do not apply the draft elsewhere. Return to
Console safely, retain the credential change, and report that the original
conversation changed or closed and its settings draft was not restored.

If navigation would cancel an active Console run, the existing leave-Console
guard remains authoritative. The credential action must not bypass it.

## Local Endpoint And Model Flow

### Provider and endpoint

Selecting a URL-based local/custom provider reveals the Base URL input before
the Model control. Endpoint validation occurs inline after a bounded debounce
and distinguishes invalid syntax from an unreachable service.

If the runtime requires a durable endpoint and the draft differs from the saved
endpoint, the primary action becomes **Save endpoint & use model**. This uses the
existing canonical config writer; endpoint persistence is no longer hidden
behind “Save model defaults.”

A conversation-only endpoint action appears only for execution paths that truly
support a session endpoint. Its label is **Use for this conversation**, and its
status explicitly says **Conversation-only endpoint**. The UI must not offer a
conversation-only action that runtime readiness will immediately reject.

### Discovery and provenance

**Discover models** is relabeled or accompanied by: “Lists models reported by
this endpoint; does not test generation.” Results are grouped:

```text
Served now (2)
  qwen2.5:7b
  llama3.2:3b

Saved (1)
  local-old — not reported by this endpoint

Custom
  Enter an unverified model ID…
```

When the existing selection is not reported by the current endpoint, the picker
does not silently present it as verified. Saving requires either choosing a
served model or explicitly confirming **Keep unverified model**.

Singular/plural copy is correct: **1 model available**, **2 models available**.
Changing provider or endpoint invalidates only the corresponding transient
discovery/verification result and preserves per-provider drafts.

## Honest Verification Semantics

The UI treats these as separate claims:

1. **Check configuration** — local validation only; no network request.
2. **Test connection** — explicit bounded endpoint/authentication request.
3. **Discover models** — explicit model-list request.
4. **Test generation** — explicit minimal generation request that may incur
   provider usage.

No result says “Provider test passed” when only local field validation ran. A
cloud provider whose credential is present but untested reads **Configuration
complete; credential not verified**. A local provider with valid fields but a
refused endpoint reads **Configuration valid; endpoint unreachable** and is not
shown as passed.

Generation testing is never automatic. Before the first paid cloud generation
test, copy states that the request may incur API usage. All network tests use
bounded timeouts, support cancellation, redact secrets and credential-bearing
URLs, and preserve the draft on failure.

## Save And Scope Contract

The fixed footer exposes only actions that are valid for the current draft:

- **Cancel** — discard ordinary draft changes; retain only separately confirmed
  immediate context/memory side effects under the existing guards.
- **Use for this conversation** — apply values that runtime supports as session
  overrides.
- **Save as default and use** — persist supported provider/model/generation
  defaults and apply them to the active conversation.
- **Save endpoint & use model** — shown when endpoint persistence is the missing
  completion step.

Exactly one completion action is visually primary. Disabled actions include a
persistent text reason, for example **Available when the current run finishes**.
No action dismisses the modal while knowingly leaving the selected configuration
blocked unless its label and confirmation state explicitly say that sending will
remain blocked.

## Power-User Efficiency

- Provider selection is searchable and groups Cloud, Local, and Custom choices.
- Model selection keeps search, custom IDs, and per-provider draft memory.
- Focus starts on the highest-priority incomplete connection field during setup;
  otherwise it starts on the last logical control or the Provider control.
- Forward traversal follows visual order and excludes hidden, unsupported, and
  no-effect controls.
- A visible footer hint exposes the approved modal-save accelerator selected
  under the repository keybinding convention. It must not shadow terminal
  conventions or global bindings.
- Alt+M remains the quick provider/model/temperature path for configured users;
  it does not duplicate credentials or endpoint persistence.
- Advanced sections remember disclosure state for the current app session, not
  globally per provider unless an existing preference owner is adopted.

## Accessibility And Terminal Behavior

- Important status is always text-labeled; color only reinforces it.
- The two modal destinations expose selected state and an accessible name, and
  behave as one tab set even if implemented with Textual buttons.
- Each editable control has a programmatic label association or equivalent
  accessible description, not only a neighboring `Static`.
- Readiness, validation, discovery, test, and return results are announced once
  without stealing focus.
- Focus remains visibly distinct at all supported terminal sizes.
- The Connection section, readiness, fold hint when needed, and fixed actions
  fit at the minimum supported viewport without horizontal clipping.
- Placeholder/help and disabled text meet the project's measured contrast
  rules in a real terminal; screenshots alone are insufficient proof.
- Long provider/model names truncate with an inspectable full value and never
  displace the completion action.

## Error And Edge States

- Missing credential: direct credential action; no TOML-first copy.
- Credential Save failure: remain in Settings, preserve typed unsaved value in
  the masked control for correction, and do not claim return readiness.
- Credential cleared during return: reopen and show missing credential.
- Endpoint invalid/refused/timeout/HTTP/auth failure: distinct sanitized result
  and direct retry/edit action.
- Empty model list: explain that the endpoint responded but reported no models;
  keep Custom ID available as unverified.
- Discovery result races with provider/endpoint edits: discard stale result.
- Catalog unavailable: show saved fallback with its source and offer Custom ID;
  never call it current.
- Current model absent from an authoritative cloud snapshot: preserve it as
  Current/unverified per ADR-020; do not silently switch.
- Active run: preserve the existing save/leave guards and explain disabled
  actions.
- Temporary conversation: return intent is process-memory only; if navigation
  destroys the origin, fail closed rather than attaching the draft elsewhere.
- Return intent expires, is consumed, or targets a deleted conversation: return
  safely to Console with explicit recovery copy.

## Copy Contract

Use **Conversation settings** everywhere for the full modal. Use **Model** for
the Alt+M quick switcher. Use provider display names, never raw provider keys.

Preferred phrases:

- **API key missing**
- **Configure credential…**
- **Configuration complete; credential not verified**
- **Endpoint reachable** / **Endpoint unreachable**
- **Model served now** / **Saved model — not verified at this endpoint**
- **Ready to send**
- **Use for this conversation**
- **Save as default and use**
- **Save endpoint & use model**

Avoid:

- raw `[api_settings.*]` paths in primary UI;
- “Provider test passed” for local-only checks;
- “Settings” as an instruction while already inside a settings modal without
  naming the owning destination;
- “0 tokens” when the provider owns the default;
- “unknown tokens” when no estimate ran;
- enabled controls annotated as having no effect.

## Implementation Consequences

The implementation should extend existing seams rather than add parallel state:

- `ConsoleSettingsModal` gains the readiness-led composition, disclosure, and
  explicit save variants.
- `ChatScreen` or its existing navigation owner creates/consumes the bounded
  return intent and reopens the origin conversation settings.
- `SettingsScreen` consumes the existing provider/model/field deep-link plus an
  opaque return intent, and emits a return action after save/cancel.
- provider readiness exposes structured facts/reason codes; screens own copy.
- model search receives provenance-aware options from the existing catalog and
  local-discovery owners.
- endpoint and verification services return typed, sanitized outcomes rather
  than composite pass/block prose.

If no existing process-memory navigation-context owner can safely hold the
single-use return intent, implementation planning must stop and either extend
the accepted application-session-state owner or amend the applicable ADR before
code changes begin. It must not create an unreviewed module-level cache.

## Acceptance Criteria

- [ ] A missing cloud credential in Conversation Settings offers a targeted
  **Configure credential…** action rather than TOML-first recovery copy.
- [ ] The action opens Settings > Providers & Models on the selected provider's
  masked API-key field without moving credential ownership into Console.
- [ ] Saving or cancelling credential work offers a return to the originating
  Conversation Settings draft; return restores non-secret values and refreshes
  readiness.
- [ ] Stale, consumed, deleted-origin, temporary-origin, and failed-save return
  states fail closed without applying a draft to another conversation.
- [ ] The primary connection flow fits above the fold and advanced generation is
  collapsed by default.
- [ ] Unsupported/no-effect generation controls are omitted, and enumerated
  values use constrained controls.
- [ ] An endpoint that must be persisted exposes **Save endpoint & use model**;
  ordinary Save cannot silently leave the selected setup blocked.
- [ ] Conversation-only endpoint actions appear only where runtime supports
  them.
- [ ] Discovered, saved, current, and custom models have visible provenance; an
  unreported model requires explicit confirmation.
- [ ] Configuration validation, reachability, model discovery, credential
  verification, and generation verification never overclaim one another.
- [ ] Network verification is explicit, cancellable, bounded, sanitized, and a
  potentially paid generation check is labeled before execution.
- [ ] The full modal is named **Conversation settings** across every entry point,
  status, guide, and test.
- [ ] Provider display names replace internal provider keys in visible copy.
- [ ] Disabled completion actions include a persistent reason.
- [ ] Provider selection is searchable; keyboard traversal excludes hidden
  controls and a compliant save accelerator is discoverable.
- [ ] Status changes are announced accessibly, controls are programmatically
  labeled, and minimum-viewport/contrast checks pass in a real terminal.
- [ ] Minor copy defects are fixed: singular model counts, no orphan Base URL
  label, and honest token-default/estimate labels.
- [ ] API keys, credential values, headers, raw provider error bodies, and
  credential-bearing URLs never enter the return intent, logs, screenshots, or
  persisted test artifacts.

## Testing And Verification

Implementation follows red-green TDD with focused suites only unless the user
separately requests a full sweep.

### Pure and service tests

- structured readiness precedence and copy-independent reason codes;
- endpoint persistence requirements by provider execution path;
- model provenance grouping and stale-result rejection;
- single-use return-intent creation, expiry, consumption, redaction, and stale
  origin rejection;
- verification outcome separation and sanitized failures.

### Textual tests

- cloud missing-key deep-link, Settings focus, Save/Cancel return, draft/focus
  restoration, and refreshed readiness;
- local endpoint invalid, unsaved, save-and-use, refused, timeout, discovery,
  unverified-model confirmation, and successful-ready states;
- first-run progressive disclosure and established-user disclosure memory;
- originating-conversation protection when another conversation becomes active;
- active-run disabled explanations and leave guard;
- keyboard order, modal accelerator, focus restoration, minimum viewport, long
  provider/model values, and status announcements;
- validation retains input and scrolls the highest-priority error into view.

### Security checks

- sentinel fake keys are absent from rendered text, logs, return intents,
  snapshots, and persisted config fixtures outside the isolated expected write;
- Settings masking and clear behavior remain intact;
- endpoint display strips credentials, query strings, and sensitive paths where
  applicable.

### Live verification

Use an isolated pytest/Textual profile first. For final UAT, use a disposable
config and data directory with model-catalog networking controlled explicitly;
do not launch against the developer's real profile. Verify cloud recovery with a
fake/mocked credential boundary and local flow with a disposable localhost
fixture. A real paid provider check is optional, separately authorized, and its
credential remains environment-only.

Capture proof at minimum and typical terminal sizes for:

1. cloud missing credential;
2. targeted credential field;
3. return with refreshed readiness;
4. local endpoint unsaved;
5. provenance-grouped discovery;
6. endpoint failure and recovery;
7. Ready to send;
8. advanced generation collapsed/expanded;
9. keyboard focus and disabled-action explanation.

## Non-Goals

- Moving API-key entry into Conversation Settings.
- A new credential store, keyring migration, or encryption redesign.
- Automatic paid generation tests.
- Replacing the global Settings provider category.
- Redesigning Context and memory behavior, memory persistence, or compaction.
- Replacing Alt+M or making it a credential editor.
- Introducing another provider registry, model catalog, or endpoint cache.
- Broad Console layout changes outside the modal and its targeted return path.
