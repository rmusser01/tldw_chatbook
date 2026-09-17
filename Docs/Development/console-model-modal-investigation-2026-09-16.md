# Console model modal: hang and settings-loss investigation

Investigated 2026-09-16 at `16bf2f0c7b64cac268dd0029b612a14990a45d0b`
(`chore/task-32482-done`), Python 3.12.11 / Textual 8.2.8 on macOS.
The checkout already contained unrelated changes. The initial investigation
did not edit application code; the user then authorized the repairs below.

The reporter described the latest application, a hang after clicking a modal
button other than a save action, and edits apparently not retained by either
Apply or Make default. Their exact build, operating system, provider, button,
and edited fields remain unknown. The defects below are reproducible on this
checkout, but are not yet an attribution of the reporter's specific incident.
Sections 1–4 retain the original findings; the implementation and verification
results are recorded at the end.

## 1. Confirmed: a later Apply/default action resets previously applied values

Reproduction, in either the quick Model popover or full Conversation settings:

1. Start with a configured llama.cpp provider and model. The isolated test's
   default temperature is `0.6`.
2. Change temperature to `0.23` and Apply. Live conversation temperature is
   correctly `0.23`.
3. Reopen settings. The temperature field correctly still shows `0.23`.
4. Change only Streaming.
5. Choose Apply, or Make default for new chats.

Observed: the modal closes successfully, but live temperature becomes `0.6`.
All four combinations (quick/full × apply/default) reproduce the loss.
Both default cases additionally reload the config from disk and resolve a new
chat's target settings: temperature is also `0.6`, not the displayed `0.23`.

Cause:

- `ChatScreen._console_settings_initial_draft` marks all values from the current
  conversation `dirty=False`, even previously applied conversation overrides
  (`tldw_chatbook/UI/Screens/chat_screen.py:2802`).
- `_commit_console_settings_submission_live` unconditionally invokes the
  provider/model rebaser before committing (`chat_screen.py:2862`).
- `ConsoleChatController.rebase_console_settings_draft` rebuilds from defaults
  and carries only dirty fields, including when the provider/model is unchanged
  (`tldw_chatbook/Chat/console_chat_controller.py:12358`).

Consequently, values the user did not edit in this opening are replaced before
the live commit and persistence. This is a value-selection defect upstream of
the disk writer. Both actions share it. It does not mean every first Apply fails.

Fix direction: distinguish preserving the current conversation snapshot from
switching provider/model. Preserve untouched conversation values on a same-target
commit, while respecting explicit Inherit edits and the existing target-change
default/provenance rules. A default action should save the intended displayed
profile. Extend coverage to repeated modal openings and next-chat defaults.
ADR-095 already defines the relevant ownership; a routine corrective fix should
not require a new ADR.

## 2. Confirmed: creating an endpoint throws in the parent modal

Reproduction: full Conversation settings → New endpoint → provide a name and
valid HTTP base URL → Create.

A mounted parent/child-modal test, using the real creation/config-write path and
a simulated connection probe, raises:

```text
_provider_changed
  _switch_provider
    _probe_entry_models
      _begin_model_discovery_identity
ValueError: model discovery requires a valid endpoint
```

Cause: `_probe_entry_models` receives both the new `custom-ep:<slug>` ID and its
family execution key, but passes the registry ID into a URL contract that only
recognizes built-in provider families. The identity constructor rejects it and
the exception escapes the Textual message handler.

Relevant code: `tldw_chatbook/Widgets/Console/console_settings_modal.py:4599`,
`:5238`, and `:5591`. This is an exception/crash path, rather than a proven
deadlock; it is a plausible explanation for a UI reported as hanging.

The existing **TASK-32566**, "Entry-scoped discovery evidence for custom endpoint
drafts", already records the related discovery-identity problem and is In
Progress. Fix the entry identity/evidence flow through the actual parent modal,
not just endpoint creation in isolation. Preserve the entry's identity while
using its family for endpoint interpretation. The evidence store also requires
`ProviderDraftIdentity`, so changing only the provider argument is insufficient.

## 3. Confirmed conditional UI blocking: Anthropic subscription Keychain reads

Conditions: macOS, Anthropic `auth_source = "claude_subscription"`, no usable
credential file, and an uncached/expired Keychain lookup.

Pressing Streaming in the full modal runs readiness synchronously:

```text
_toggle_streaming → _sync_readiness_display
  → build_console_settings_readiness → get_provider_readiness
    → read_claude_code_credential → subprocess.run(security ..., timeout=5)
```

A simulated 300 ms Keychain read ran on the UI thread and delayed a scheduled
20 ms heartbeat until approximately 306 ms after scheduling. This probe does not contact the real
Keychain or claim that the reporter used this provider/platform combination.

A separate deterministic clock test proves the timeout-cache defect: the
five-second cache timestamp is taken **before** a read that may time out after
five seconds. The cached failure is therefore already expired when stored.
Two consecutive reads each invoke `security`, rather than reusing the failure.
Repeated readiness evaluations can consequently cause repeated UI stalls.

Relevant code: `tldw_chatbook/Chat/provider_readiness.py:533` and
`tldw_chatbook/LLM_Calls/anthropic_subscription.py:182` (cache check at `:195`,
subprocess at `:201`, cache publication at `:219`).

Fix direction: remove credential subprocess/file I/O from synchronous UI
readiness evaluation, publish a background-resolved snapshot, and start failure
cache TTL at completion. Correcting TTL alone would still leave one blocking
read on the event loop.

## 4. Requested context-window precedence and confirmed resolution drift

The user added this requirement during the investigation:

1. Use the context size reported by the selected remote server, when available.
2. Otherwise use the default for the selected model/API.
3. If neither exists, use the system default of **32,000 tokens**, exactly.

This is the total context window; output-token reservation, safety margin, and
other request overhead are accounted for afterward. It is not a 32,000-token
output limit.

The current implementation has separate, inconsistent resolution paths:

- `console_session_settings._resolve_token_limit_locally_with_provenance`
  (`:2562`) uses a local table and prefix matching, then provider fallbacks and
  an **8,001** application fallback. It does not consult the selected server or
  the shared model-capability catalog. For example, `gpt-4o` matches the stale
  `gpt-4` prefix and displays **8,192**, even when shared capabilities are set to
  the built-in model catalog and report **128,000**.
- `Utils/token_counter.get_model_token_limit` (`:486`) uses shared capabilities,
  another table, and provider defaults. Its general fallback is **16,384**.
- Console provider request preparation resolves context capacity from shared
  capabilities (`console_provider_gateway.py:3398`), while
  `console_prepared_request.resolve_request_capacity` (`:1082`) can leave the
  window and input ceiling unknown. It does not use the Console display's
  fallback.
- The gateway already fetches bounded optional server metadata for local
  reasoning support (`console_provider_gateway.py:3668`), but that flow retains
  template/tool capabilities rather than server context size.

Implementation direction: resolve one context-window value and its source for
both the display and request budgeting. Server discovery must be asynchronous,
bounded, and scoped to the selected endpoint/model; changing providers/endpoints
must not reuse another server's limit. Missing, invalid, unsupported, or timed-out
metadata falls through to model/API defaults and finally 32,000. Label fallback
values as estimates rather than server-verified limits. A server's actual
configured serving window takes precedence over a model's theoretical maximum.

This changes the existing ADR-052 rule that unknown windows block Automatic
compaction. The implementation plan must explicitly reconcile that rule with
the requested fallback, rather than changing only the displayed number and
leaving the runtime unresolved. No behavior change or ADR amendment has been
implemented during this investigation.

Two additional diagnostic assertions cover the unknown-model 32,000 fallback
and agreement between the Console display and known model capabilities. Both
fail on the intended numeric comparisons in the targeted follow-up run. Remote
server discovery still needs implementation and endpoint/model isolation tests.

## Evidence and limits

The original opt-in probes were promoted into ordinary regression tests during
implementation. Settings retention is covered by
`Tests/UI/test_console_settings_retention.py`, endpoint creation/discovery by
`Tests/UI/test_console_endpoint_discovery.py`, credential responsiveness by
`Tests/LLM_Calls/test_subscription_credential_cache.py` and the Console credential
UI tests, and context resolution by `Tests/Chat/test_console_context_window.py`,
`Tests/UI/test_console_context_window_modal.py`, and
`Tests/UI/test_console_popover_context_window.py`.

It uses the repository's config/DB isolation. The settings-loss tests use the
real ChatScreen/store/controller and config writer. The endpoint test exercises
the mounted child-to-parent event flow. Network and Keychain effects in the
hang investigations are simulated. This is not a real-terminal reproduction on
the reporter's machine.

Original diagnostic run: **7 failed**, at the intended value-retention, event-loop
latency, timeout-cache, and parent-modal exception assertions. Ruff lint and
format checks pass for the diagnostic file.

Existing targeted controls: **16 passed**. These cover initial quick Apply by
mouse/keyboard, persistence/resume/promotion, default-file/runtime publication,
provider switching, connection-test success/failure, and generation-test
confirmation/cancellation. The full suite was not run.

Next reporter details that would distinguish these causes: exact clicked button,
provider/auth mode, operating system, actual version/commit, changed fields,
whether the UI recovered after several seconds, and any traceback. No API keys
or unredacted configuration are needed. For the save symptom, distinguish values
changing immediately on Apply from disappearing after reopening/restarting.

## Implemented repairs and nearby findings

- Same-target Apply and remembered targets preserve the current conversation's
  values. Explicit Inherit still uses defaults. Quick-to-full transfer retains
  newly exposed values in saved profiles. Default writes no longer synthesize
  an unauthorized endpoint patch.
- Named endpoints retain their registry ID while interpreting URLs through the
  configured API family. Connection probes use the entry's credential and
  reject stale results after entry, endpoint, model, or credential changes.
- Subscription readiness uses a bounded background snapshot. Cache lifetime
  begins at completion, concurrent lookups share work, and send-time credential
  reads run off the event loop. Malformed tokens/expiry values fail safely.
- The model window now resolves server → model/API → 32,000, with the same
  result used by settings, provider request preparation, and compaction.
  Estimated fallback capacity remains labeled unverified. Lookup failures,
  cancellation, oversized responses and implausible integers fall back.
- Both model surfaces fence late metadata results across A-B-A switching and
  dismissal. Metadata caches retain no credentials in their keys or repr and
  separate entry, endpoint, model and credential identity.

Server parsing follows the primary contracts: llama.cpp's serving
`default_generation_settings.n_ctx` from
[`/props`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
(selected-model query and `autoload=false` for routers), Ollama's active
[`/api/ps` context_length](https://docs.ollama.com/api/ps), and selected model
records' serving-window fields for compatible APIs. Training-window metadata
is not used as an active serving limit.

Verification includes mounted Textual flows, real config reload, simulated slow
credentials, bounded mock transports, and an owned loopback metadata server
using the real HTTP client. This does not claim a reproduction on the reporter's
machine or validation against their unspecified provider.

Final combined regression run: **277 passed** across 13 targeted modules.
Additional context/token/policy checks: **139 passed**; existing context estimate
and policy selection: **13 passed**; gateway resolution selection: **30 passed**.
The last narrow context/subscription/mounted-modal rerun passed **39 tests**.
These runs overlap and are not a unique-test total. The only warning in the
passing runs was the existing requests dependency-version warning. New files
pass Ruff lint/format, modified regions are formatted, changed Python lines
have no Ruff findings, and `git diff --check` passes.

## Follow-up repairs (TASK-32710 and TASK-32711)

The user authorized all remaining findings. The three pre-existing test failures
are repaired without production changes: identity assertions follow ADR-149,
compaction close/reopen uses the production Console harness and durable memory
records, and first-persistence coverage now checks both character and persona
ownership including the persona system template. The two affected test files
pass all 242 tests; 49 related identity/lifecycle checks also pass.

Slow subscription reads were then reproduced in canonical Settings, first-run
setup and persona handoff. All three surfaces now use the shared bounded
background snapshot and refresh completion, expiry and cache renewal without
another edit. Refreshes use the current provider/selection and stop publishing
after the owning surface is hidden or dismissed. Settings refreshes its status
rows without rebuilding forms or overwriting unsaved edits. First-run pending,
missing and expired subscription messages no longer claim an API key is needed;
borrowed tokens never enter staged or saved credentials. Actual send-time
authentication remains owned by the provider gateway.

Final review also reproduced an adjacent first-run save bug: choosing a
subscription produced an empty API-key decision that deleted inactive stored
keys and environment bindings. The shared config writer now supports a narrowly
validated sparse preservation operation for unchanged subscription setup. It
leaves those fields and their credential-source setting untouched; explicit
Clear and replacement remain authoritative. Issued-mutation and atomic conflict
checks still apply, including an added authentication-mode conflict check.
This contract is recorded in the amended ADR-012. Tests cover actual disk save
and reload, both credential fields together, all subscription states, explicit
edits, invalid preservation requests and concurrent changes.

Two readiness repaint races are also covered: completion before the first
timer tick, and completion between rendering the persona header and inspector.
Each projection converges to the current state without another user action.

Mounted regressions gate simulated credential reads, exercise usable controls
while those reads wait, and check automatic completion, stale selections,
expiry and dismissal without real credential or provider access. First-run
state plus the new readiness tests pass 241 cases; 45 existing provider/auth
wizard checks pass, including stored/environment credential rotation before
save. These counts overlap other runs and are not a unique-test total. No
full-suite pass or reproduction on the reporter's machine is claimed.

Final follow-up verification: **73 passed** in one combined run of the new UI
and preservation regressions, the three repaired failure cases, and shared
subscription-cache/readiness controls. The writer/state/readiness/preservation
run passed **446 tests**. Settings passed **83** draft/readiness checks plus
**10** related controls; persona passed **65** existing action/preview checks,
then **16** focused checks after the header race repair. Runs overlap. All six
new or repaired test files pass Ruff lint and formatting. All changed production
lines are free of Ruff findings; the six production files introduce no new
diagnostics relative to HEAD. Existing unrelated whole-file lint/format debt
remains unchanged, changed source regions are formatted, and scoped diff checks
pass. The existing requests dependency-version warning remains informational.

## PR integration against dev

The user authorized a PR against dev. The repair branch starts at
`24094f23d59c7a9d3cfac964c19fd263bc0393b2` in an isolated worktree; unrelated
working-tree changes and commits from the original checkout are excluded.
Integration retains dev's canonical endpoint identity helpers, sampling enum
ownership, Settings cleanup, and model-popover geometry. Tests that save real
config now use the existing private-profile child-process harness, selecting
their profile before imports. Production recovery/source-selection guards are
unchanged.

Integration review added two lifecycle repairs. Console readiness notices
credential expiry and TTL renewal even without a completion-revision change.
A cancellation-resistant discovery result arriving after dismissal is discarded
before querying removed controls. Existing model-change cancellation tests now
edit actual controls; queued stale-event rejection remains independently covered.

Fresh verification on the dev-based branch:

- **227 passed**, 15 deselected across the focused retention, discovery,
  context-capacity, credential, Settings, persona, and first-run regression set.
  The exclusions are 13 unrelated TTS cases and two loopback-network cases.
- **1 passed** for the new metadata lookup against an owned loopback HTTP server,
  using the real client and a private profile.
- **20 passed** after the final dismissal fix: all four model-change cancellation
  paths, the new dismissed-modal regression, and all 15 endpoint regressions.
- **14 passed** for dev's neighboring model-popover geometry, missing-provider,
  and registry-option controls. Separate retention/close-reopen checks passed
  **22** cases; Settings/credential preservation checks passed **36** cases.

Counts overlap and are not a unique-test total. The earlier counts in this
report describe verification before integration onto current dev.

Broader targeted controls encountered dev's existing profile-selection fixture
failures. All 17 failing provider-readiness/writer cases also failed with original
dev production and test modules loaded from the base commit. The TTS selection
also produced 13 setup errors on unchanged dev with the same
`RecoveryRequired("raw_source_selection_changed")` guard. Those baseline fixtures
are not repaired by this PR, and no full-suite pass is claimed. No external
provider account or reporter-machine reproduction was used.

New Python files pass Ruff lint and formatting. Modified legacy files introduce
no new Ruff diagnostics relative to dev; changed regions are formatted and
scoped diff checks pass. Existing whole-file static debt remains unchanged.

## PR-2703 review and merge verification

Qodo identified three issues, all confirmed and addressed:

- Metadata probes now validate their derived URL through the shared asynchronous
  egress policy before opening a stream. Explicitly configured local origins
  remain usable; denied metadata IPs/hostnames produce the ordinary fallback
  without issuing a request. Four new blocked-destination regressions failed
  before the fix, then passed alongside the existing local-server controls.
- `ContextWindowCache.cached()` and `resolve()` now document parameters,
  cache/fallback results, client ownership, and cancellation behavior.
- Home explicitly uses background credentials during synchronous composition,
  then resolves credentials in its existing content-snapshot worker. The mounted
  valid-subscription regression reproduced the stuck blocked badge before the
  fix; valid, missing, and expired credentials now settle without blocking UI.
  Five related Home tests use the existing private-profile harness with their
  assertions and production recovery guards preserved.

CI also caught an eager import of the metadata module at first paint. Tracing
showed both gateway construction and the visible context summary reached it.
Pure capacity fallback now lives with the existing token-capacity helpers, and
the gateway creates its network cache only on first metadata use. The original
startup budgets remain unchanged: **1022/1022** modules at UI ready and
**669/686** at import. An explicit absence assertion protects the deferred module.

The diagnostic inventory's only delta was removal of the old token-counter
lookup-failure debug statement; statement-level review confirmed no new log
arguments or sinks. The regenerated inventory verifies successfully.

Fresh combined startup/context/token/Home verification: **86 passed**, one
loopback case deselected. That real-client loopback case passed separately.
Home and its existing neighbors passed **8** focused checks; counts overlap.
An additional screen-preload check measured 536 modules against its historical
500 limit; unchanged dev measured the same 536 with original production modules,
so that separate baseline breach was not changed or hidden. New files pass Ruff
lint/format, changed legacy files add no diagnostics, and diff checks pass.
