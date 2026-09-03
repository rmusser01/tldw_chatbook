# Console Conversation Settings: Ready-to-Send Design

Date: 2026-09-02
Status: User-approved design, amended after architecture and UX self-review
Primary surface: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
Related tasks: TASK-145, TASK-178, TASK-191, TASK-366, TASK-14812, TASK-14811.3, TASK-2154.7
Related ADRs: ADR-006, ADR-011, ADR-012, ADR-020, ADR-033

## Summary

Conversation Settings will become a readiness-led configuration flow for both
cloud APIs and locally hosted models. Its primary job is to get the active
conversation to an explicit **Ready to send** state. Durable provider
credentials remain owned by **F9 Settings > Providers & Models**. When a cloud
credential is missing, Conversation Settings deep-links to that exact field and
offers a return path that restores the exact Console-owned conversation draft
and focus without putting prompt content in the navigation handoff.

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

Success is not “the modal accepted the draft.” The UI reports two independent
dimensions instead of treating configuration and network evidence as one pass:

1. **Operability**
   - **Ready to send** — mandatory local requirements pass and Chatbook will
     allow an attempt. This never guarantees that the provider will accept or
     complete the request.
   - **Not ready** — the UI names the single highest-priority blocker, its
     impact, and one direct recovery action.
2. **Verification evidence**
   - configuration checked;
   - credential not verified or authenticated connection checked;
   - models not queried or reported by the current endpoint;
   - generation not tested, succeeded, or failed.

Evidence labels may be combined only when the underlying checks ran. For
example, a cloud provider with a present but untested key reads **Ready to send
— credential not verified**, while a local endpoint that refused a connection
reads **Not ready — endpoint unreachable** even if its fields are syntactically
valid.

The active provider, endpoint authority, credential source (never the secret),
selected model provenance, and verification scope remain visible beside the
operability status. A user returning from canonical Settings sees the exact
Console-owned conversation draft they left and a refreshed readiness result.

## Ownership And Architectural Boundaries

### Conversation Settings owns

- the active conversation's provider, model, generation, streaming, identity,
  and context-policy drafts;
- readiness presentation and blocked-state recovery;
- local endpoint/model discovery initiated from the conversation;
- the choice between conversation-only application and an existing durable
  provider/default write path;
- capturing a suspended modal draft into Console-owned screen state before
  targeted Settings navigation and reopening it after a valid return handoff.

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

### Application navigation owners

- `ScreenStateStore` retains the outgoing Console screen snapshot, including a
  typed suspended Conversation Settings draft, for the normal process-memory
  cross-visit lifetime. Prompt and prefill values remain private Console state;
  they never enter Settings navigation context or diagnostic metadata.
- `PendingHandoffStore` gains one typed, single-slot, consume-once return channel
  following ADR-033. The handoff contains only target conversation/session
  identity, expected Console-settings revision, and logical modal
  destination/focus. The channel revision is the opaque return token.
- `SettingsScreen` retains only that bounded channel revision while it offers
  the return action. It does not copy or own the suspended modal draft.
- A newer return handoff supersedes an older pending one. Successful return and
  terminal rejection acknowledge it; explicit **Stay in Settings** abandons it.
  **Return without saving** still navigates to Console and is acknowledged there
  after restoration or terminal rejection. No module-level cache or second
  token-indexed registry is introduced.

## ADR Check

ADR required: no new ADR

ADR paths:

- `backlog/decisions/006-provider-aware-generation-settings.md`
- `backlog/decisions/011-chatbook-workbench-ui-system.md`
- `backlog/decisions/012-provider-credential-settings-boundary.md`
- `backlog/decisions/020-automatic-model-catalog-refresh.md`
- `backlog/decisions/033-application-session-state-ownership.md`

Reason: ADR-012 already decides that Settings owns durable credentials while
Console owns blocked-send recovery and exact-field navigation. ADR-006 owns the
Settings/Console generation split, ADR-011 requires visible readiness and
recovery, ADR-020 owns cloud catalog authority, and ADR-033 already assigns
cross-visit snapshots and destination handoffs to `ScreenStateStore` and
`PendingHandoffStore`. The return flow uses those accepted lifecycles and does
not add a storage, credential, provider, or runtime owner. If implementation
cannot preserve ADR-033's single-slot, consume-once semantics, planning must
stop and amend that ADR before code changes begin.

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

This section is always first and remains above the fold at 80x24. Cloud and
local providers share the same readiness summary and
visual order, but their Connection controls remain purpose-specific: cloud setup
emphasizes credential ownership and local setup emphasizes endpoint authority.
Uniform structure must not imply identical configuration requirements.

### Advanced generation

Collapsed by default for first-time or blocked setup. It opens automatically
when navigation targets one of its controls and remembers explicit disclosure
state for the current Console session. Contains sampling, response limits,
streaming, reasoning, thinking, and provider-specific controls. A control is
omitted only when an existing authoritative execution/capability mapping says
the selected provider or model cannot consume it. Unknown support remains in
Advanced with neutral explanatory copy; this work does not invent a new model-
capability registry. Controls with a constrained vocabulary use a Select or
equivalent enumerated control rather than free text.

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
Ready to send · Endpoint reachable
Model served now · Generation not tested
[Connection details]
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
- generation verification, when that optional check is supported: not tested,
  testing, succeeded, or failed;
- one primary recovery action for the highest-priority blocker.

Internal provider keys, config paths, raw exception prose, headers, credentials,
and credential-bearing URLs never appear in this component.

## Cloud Credential Deep-Link And Return

### Entry

When readiness reports a missing credential, the modal shows **Configure
credential…**. Activating it first captures a typed suspended modal draft in the
outgoing Console screen snapshot. That Console-owned snapshot may contain the
same private system prompt and pinned prefill already owned by the active
session; it remains process-memory only under ADR-033 and is never copied into
Settings.

Console then stages a typed `PendingHandoffStore` return value containing only:

- origin Console conversation/session identity;
- the active modal destination and focused logical control;
- the exact Console-settings revision captured with the suspended draft; and
- the single-use channel revision used as the opaque return token.

The Console session owner exposes one monotonic `ConsoleSettingsDraftRevision`.
It advances whenever provider, model, endpoint, generation, identity, system
prompt, pinned prefill, or context-policy settings for that session change. A
return is compatible only when the exact session still exists and its current
Console-settings revision equals the captured revision. Reloading credential
readiness after a Settings save does not itself advance this draft revision.

The handoff and Settings navigation context must not contain an API key,
resolved credential value, request headers, provider response, prompt, prefill,
transcript body, raw Base URL, or other secret/content. They are never written
to config, database, logs, crash metadata, or navigation labels.

### Settings destination

The existing Settings navigation context opens:

```text
category = providers-models
provider = active provider
model = active model
field = api_key
return_revision = positive handoff revision
```

Providers & Models parses this through a typed, allowlisted navigation target.
Unknown keys, invalid provider/model/field values, and non-positive return
revisions are rejected rather than retained as loose screen context. The return
navigation to Console adds one allowlisted outcome enum—`credential_saved`,
`provider_settings_saved`, or `without_saving`—so mutation-aware copy does not
require modifying the staged handoff or passing arbitrary text.

### Existing Settings draft conflict

Navigation context is applied after the prior Settings screen snapshot is
restored. If Providers & Models already has unsaved changes:

- for the same provider, preserve the draft, focus its credential field, and
  state which existing fields will also be included by Save;
- for a different provider, do not switch silently. Show **Review existing
  changes**, **Discard changes and configure _Provider_**, and **Return to
  conversation settings**;
- no branch merges, saves, or discards unrelated Settings edits implicitly.

Settings focuses the API-key field and presents its existing two credential
paths:

- masked local config key;
- credential environment-variable name, labeled as the safer shell/shared-host
  path.

Settings owns Save/Clear and never pre-fills or echoes a stored secret.

### Return

After successful provider Save, Settings presents **Return to conversation
settings** as the primary continuation. A secondary **Stay in Settings** action
is available and explicitly abandons the pending return. Cancel/back from
credential editing uses **Return without saving** when a credential draft is
dirty, preserving the normal discard confirmation and never claiming success.

On return:

1. Claim the exact typed return handoff revision.
2. Verify that the origin conversation still exists and its Console-settings
   revision still equals the captured revision.
3. Restore the compatible Console snapshot and reopen its suspended
   Conversation Settings draft.
4. Restore the logical focus/anchor when still applicable; otherwise focus the
   Connection section.
5. Reload provider configuration and credential readiness from canonical state.
6. Announce a mutation-aware result followed by the refreshed status:
   - **Credential saved — checking readiness** only when the credential source
     or value changed;
   - **Provider settings saved — checking readiness** when broader fields also
     changed;
   - **Returned without saving — readiness unchanged** after confirmed discard.
7. Acknowledge the return handoff so refresh/re-entry cannot replay it.

If the origin is missing or stale, do not apply the draft elsewhere. Return to
Console safely, retain the credential change, and report that the original
conversation changed or closed and its settings draft was not restored.

If navigation would cancel an active Console run, the existing leave-Console
guard remains authoritative. The credential action must not bypass it.

Saving a credential-environment-variable name does not prove that the variable
has a value in the running process. If it is absent on return, readiness remains
blocked and explains that the named variable must be exported and Chatbook
relaunched; it never claims that the credential was verified.

## Local Endpoint And Model Flow

### Provider and endpoint

Selecting a URL-based local/custom provider reveals the Base URL input before
the Model control. Endpoint validation occurs inline after a bounded debounce
and distinguishes invalid syntax from an unreachable service.

If the runtime requires a durable endpoint and the draft differs from the saved
endpoint, the primary action becomes **Save endpoint & use model**. This uses the
existing canonical config writer; endpoint persistence is no longer hidden
behind “Save model defaults.” Persistent impact copy names the scope, for
example: **Saves this endpoint for Ollama and future conversations.** The config
write must fully apply before the session selection changes or the modal
dismisses. Failure retains the draft, leaves the current session unchanged, and
offers a sanitized retry; it never produces a partially applied “success.”

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
served model or explicitly confirming **Keep unverified model**. That
confirmation is valid only for the current provider/endpoint/model draft
generation; editing any of those identities invalidates it. Ordinary unrelated
edits do not repeatedly challenge an established user.

Singular/plural copy is correct: **1 model available**, **2 models available**.
Changing provider or endpoint invalidates only the corresponding transient
discovery/verification result and preserves per-provider drafts.

## Honest Verification Semantics

The UI treats these as separate evidence claims:

1. **Check configuration** — local validation only; no network request.
2. **Test connection** — explicit bounded endpoint/authentication request, shown
   only where an existing provider service supports a meaningful non-generating
   probe. Otherwise the UI says **No non-billable live check available**.
3. **Discover models** — explicit model-list request.
4. **Test generation** — optional explicit minimal generation request that may
   incur provider usage. It is a later workstream and never gates the core
   configuration/deep-link release or the ability to attempt a normal send.

No result says “Provider test passed” when only local field validation ran. A
cloud provider whose credential is present but untested reads **Configuration
complete; credential not verified**. A local provider with valid fields but a
refused endpoint reads **Configuration valid; endpoint unreachable** and is not
shown as passed.

Generation testing is never automatic. Before every paid cloud generation test,
copy states that the request may incur API usage and requires explicit
confirmation. It is implemented only for providers covered by a documented
request/response and billing-risk matrix. All network tests use bounded
timeouts, support cancellation, redact secrets and credential-bearing URLs,
and preserve the draft on failure.

## Save And Scope Contract

The fixed footer exposes only actions that are valid for the current draft:

- **Cancel** — discard ordinary draft changes; retain only separately confirmed
  immediate context/memory side effects under the existing guards.
- **Use for this conversation** — apply values that runtime supports as session
  overrides.
- **Save as default and use** — persist supported provider/model/generation
  defaults and apply them to the active conversation.
- **Save endpoint & use model** — shown when endpoint persistence is the missing
  completion step; persistent adjacent text states that it changes the provider
  endpoint used by future conversations.

Exactly one completion action is visually primary. Disabled actions include a
persistent text reason, for example **Available when the current run finishes**.
No action dismisses the modal while knowingly leaving the selected configuration
blocked unless its label and confirmation state explicitly say that sending will
remain blocked.

Footer primacy applies only to completion actions. Immediate Context and memory
operations remain secondary or destructive body actions and never visually
compete with the single primary completion action.

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
- Advanced sections default closed for first-time/blocked setup, open when
  explicitly targeted, and remember disclosure state in the current Console
  session snapshot. No new global preference is introduced.

## Accessibility And Terminal Behavior

- Important status is always text-labeled; color only reinforces it.
- The two modal destinations expose selected state and an accessible name, and
  behave as one tab set even if implemented with Textual buttons.
- Each editable control exposes the best accessible name/description supported
  by the installed Textual version and also retains a persistent visible label;
  tests assert the observable widget name/description contract rather than an
  HTML-style label association that Textual may not implement.
- Readiness, validation, discovery, test, and return results are announced once
  without stealing focus.
- Focus remains visibly distinct at all supported terminal sizes.
- The Connection section, readiness, fold hint when needed, and completion
  actions fit without horizontal clipping at 80x24, 100x30, and 160x40. At
  compact widths, action rows stack or wrap while retaining their full labels;
  the modal never depends on horizontal scrolling.
- Placeholder/help and disabled text meet the project's measured contrast
  rules in a real terminal; screenshots alone are insufficient proof.
- Long provider/model names truncate with an inspectable full value and never
  displace the completion action.

## Error And Edge States

- Missing credential: direct credential action; no TOML-first copy.
- Credential Save failure: remain in Settings, preserve typed unsaved value in
  the masked control for correction, and do not claim return readiness.
- Existing same-provider Settings draft: preserve it, enumerate the additional
  fields Save will include, and target the credential control.
- Existing different-provider Settings draft: require Review, explicit Discard,
  or Return; never replace it because a deep-link arrived.
- Credential environment-variable name saved but value absent: return blocked,
  name the required export/relaunch step, and do not claim verification.
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
- Temporary conversation: the snapshot and handoff are process-memory only; if
  the exact origin cannot be restored, fail closed rather than attaching the
  draft elsewhere.
- Return handoff is superseded, consumed, abandoned, or targets a deleted or
  revision-mismatched conversation: return safely to Console with explicit
  recovery copy.

## Copy Contract

Use **Conversation settings** everywhere for the full modal. Use **Model** for
the Alt+M quick switcher. Use provider display names, never raw provider keys.

Preferred phrases:

- **API key missing**
- **Configure credential…**
- **Configuration complete; credential not verified**
- **Ready to send — credential not verified**
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

**Ready to send** always means “Chatbook found no local blocker and will allow a
request attempt,” not “the provider guarantees successful generation.” Network
evidence remains visible beside it.

## Implementation Consequences

The implementation extends existing seams rather than adding parallel state:

- `ConsoleSettingsModal` gains the readiness-led composition, disclosure,
  explicit save variants, and a typed suspend/restore projection.
- `ChatScreen.save_state()` / `restore_state()` retain the suspended modal draft
  inside the existing native Console snapshot and reopen it only after a valid
  typed return handoff.
- `PendingHandoffStore` gains one ADR-033-conforming Conversation Settings return
  channel with structural detachment, revision checks, claim, acknowledge,
  release, supersession, and explicit abandonment.
- `SettingsScreen` consumes a typed provider/model/field/return-revision target,
  guards restored dirty provider drafts, and emits a return action after a
  successful save or confirmed cancel.
- the active Console session owner exposes one monotonic
  `ConsoleSettingsDraftRevision`; draft restoration compares that revision
  rather than inventing a conversation timestamp or relying only on object
  identity.
- provider readiness exposes structured facts/reason codes; screens own copy.
- model search receives provenance-aware options from the existing catalog and
  local-discovery owners.
- endpoint and verification services return typed, sanitized outcomes rather
  than composite pass/block prose.

No module-level cache, general token registry, duplicate provider registry, or
new root application-state object is permitted. If these requirements cannot be
met with ADR-033's owners and lifecycle, implementation planning stops and ADR-
033 is amended before code changes begin.

## Delivery Workstreams

This programme is intentionally split into atomic, independently reviewable
Backlog tasks/PRs. “All issues” means completing every applicable workstream,
not combining them into one unsafe change.

1. **Return contract and navigation safety** — typed Settings target, Console
   snapshot suspension, `PendingHandoffStore` return channel, exact Console-
   settings revision, dirty-Settings conflict guard, and terminal stale paths.
2. **Structured readiness** — separate operability from evidence, establish
   reason-code precedence, mutation-aware return copy, and environment-variable
   absent/relaunch recovery.
3. **Modal hierarchy and scope** — Connection-first composition, contextual
   progressive disclosure, searchable provider selection, constrained advanced
   controls, explicit save scope, and fixed completion primacy.
4. **Local endpoint and model provenance** — persist-before-apply endpoint
   behavior, durable-impact copy, current/saved/custom grouping, stale-result
   invalidation, and bounded unverified-model confirmation.
5. **Verification and hardening** — supported non-generating connection probes,
   an optional provider-matrix-backed paid generation check, accessibility,
   keyboard behavior, and live 80x24/100x30/160x40 verification.

Workstream 1 is foundational. Workstream 2 follows it; workstream 3 consumes the
typed readiness and return seams from workstreams 1 and 2; workstream 4 builds on
the connection-first structure and readiness evidence; workstream 5 follows the
relevant provider/service contracts. Each task repeats this ADR check and links
ADR-033.

## Acceptance Criteria

- [ ] A missing cloud credential in Conversation Settings offers a targeted
  **Configure credential…** action rather than TOML-first recovery copy.
- [ ] The action opens Settings > Providers & Models on the selected provider's
  masked API-key field without moving credential ownership into Console.
- [ ] Saving or cancelling credential work offers a return to the originating
  Conversation Settings draft; the typed handoff contains no prompt/prefill or
  raw endpoint and Console restores its exact private snapshot before refreshing
  readiness.
- [ ] Superseded, consumed, abandoned, deleted-origin, temporary-origin,
  revision-mismatched, and failed-save return states fail closed without
  applying a draft to another conversation.
- [ ] Existing same-provider Settings changes are preserved and disclosed;
  different-provider changes require Review, explicit Discard, or Return.
- [ ] Provider deep-link context is typed and allowlisted, and the return flow
  uses ADR-033's `ScreenStateStore` and `PendingHandoffStore` rather than a new
  cache or root state owner.
- [ ] The primary connection flow fits above the fold and advanced generation is
  contextually collapsed for first-time/blocked setup while explicit targeting
  and per-Console-session disclosure state are preserved.
- [ ] Unsupported/no-effect generation controls are omitted only with existing
  authoritative capability evidence; unknown support remains in Advanced, and
  enumerated values use constrained controls.
- [ ] An endpoint that must be persisted exposes **Save endpoint & use model**;
  its global/future-conversation impact is visible, persistence precedes session
  application, and failure retains the unchanged session plus modal draft.
- [ ] Conversation-only endpoint actions appear only where runtime supports
  them.
- [ ] Discovered, saved, current, and custom models have visible provenance; an
  unreported model requires explicit confirmation.
- [ ] Operability and verification evidence are separate; configuration
  validation, reachability, model discovery, credential verification, and
  generation verification never overclaim one another.
- [ ] Supported network verification is explicit, cancellable, bounded, and
  sanitized; unsupported non-billable checks say so, and an optional paid
  generation check requires provider-matrix coverage plus confirmation.
- [ ] Saving only an absent environment-variable name returns a blocked state
  with an export/relaunch recovery instruction.
- [ ] The full modal is named **Conversation settings** across every entry point,
  status, guide, and test.
- [ ] Provider display names replace internal provider keys in visible copy.
- [ ] Disabled completion actions include a persistent reason.
- [ ] Provider selection is searchable; keyboard traversal excludes hidden
  controls and a compliant save accelerator is discoverable.
- [ ] Status changes are announced accessibly, controls expose Textual-supported
  accessible names/descriptions plus visible labels, and contrast checks pass in
  a real terminal.
- [ ] At 80x24, 100x30, and 160x40, the connection flow and completion actions
  remain reachable without clipping or horizontal scrolling; compact actions
  stack or wrap with full labels.
- [ ] Minor copy defects are fixed: singular model counts, no orphan Base URL
  label, and honest token-default/estimate labels.
- [ ] API keys, credential values, headers, raw provider error bodies, and
  credential-bearing URLs never enter the return handoff/navigation context,
  logs, screenshots, or persisted test artifacts.
- [ ] The programme is delivered through the five bounded workstreams rather
  than one cross-cutting implementation task.

## Testing And Verification

Implementation follows red-green TDD with focused suites only unless the user
separately requests a full sweep.

### Pure and service tests

- structured readiness precedence and copy-independent reason codes;
- independent operability and verification-evidence projections;
- endpoint persistence requirements by provider execution path;
- persist-before-apply endpoint failure behavior;
- model provenance grouping and stale-result rejection;
- typed Settings-target validation and unknown-key rejection;
- suspended Console snapshot plus single-slot return-handoff staging,
  supersession, claim, release, acknowledgement, abandonment, structural
  detachment, and stale-origin/revision rejection;
- authoritative-supported, authoritative-unsupported, and unknown generation-
  control capability behavior;
- verification outcome separation and sanitized failures.

### Textual tests

- cloud missing-key deep-link, Settings focus, Save/Cancel return, draft/focus
  restoration, and refreshed readiness;
- same-provider dirty Settings disclosure and different-provider
  Review/Discard/Return conflict handling;
- credential-only, broader-provider, and return-without-saving result copy;
- saved-but-absent environment-variable recovery;
- local endpoint invalid, unsaved, save-and-use, refused, timeout, discovery,
  unverified-model confirmation, and successful-ready states;
- first-run progressive disclosure and established-user disclosure memory;
- originating-conversation protection when another conversation becomes active;
- active-run disabled explanations and leave guard;
- keyboard order, modal accelerator, focus restoration, 80x24/100x30/160x40
  geometry, stacked compact actions, long provider/model values, accessible
  names/descriptions, and status announcements;
- validation retains input and scrolls the highest-priority error into view.

### Security checks

- credential-field sentinel fake keys are absent from rendered text, logs,
  return handoffs, navigation context, and persisted config fixtures outside the
  isolated expected write;
- system prompt and pinned prefill survive an exact snapshot/return round trip
  but never enter the handoff, Settings context, or diagnostics;
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

Capture proof at 80x24, 100x30, and 160x40 for:

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
