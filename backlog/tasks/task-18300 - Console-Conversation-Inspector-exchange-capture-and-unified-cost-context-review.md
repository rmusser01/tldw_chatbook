---
id: TASK-18300
title: Console Conversation Inspector — exchange capture + unified cost/context review
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18'
updated_date: '2026-08-27 04:24'
labels:
  - console
  - ui
  - db
  - transparency
dependencies: []
---

## Description (the why)

Clicking the Console token/cost chip today opens a numbers-only breakdown; the
full next-send payload viewer hides behind Ctrl+Shift+P; and nothing records
what actually went over the wire on past turns. Users cannot answer "what
exactly is being sent back and forth in my conversation, and what did each
piece cost." Owner-approved spec:
`Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md`
(six recorded owner decisions, 2026-08-18).

## Acceptance Criteria (the what)

- [x] Every Console provider call (each tool-loop iteration, direct sends, and
      the llama.cpp branch) is captured — request and response — and persisted
      locally per turn, anchored to the turn's assistant message.
      *(Tool-loop and direct-send capture live-verified against a real
      provider — see Implementation Notes scenario (a). The llama.cpp branch
      is covered by the automated suite (T4), not independently
      live-verified — no local llama.cpp server was set up for this pass.)*
- [x] Captured requests are allowlist-sanitized by construction: credentials
      never persist, dropped key names are visible to the user, and binary/
      base64 content is stubbed with mime/size/sha256.
      *(Live-observed "Omitted by capture policy: api_key" on every expanded
      call. Binary/base64 stubbing is covered by the automated property test,
      not live-verified — no image/file attachments were sent in this
      session.)*
- [x] Captures survive Stop with a non-empty partial marked stopped; current
      sibling-branch regeneration preserves both the original and the stopped
      regenerated branch; legacy restored variants keep captures marked
      abandoned; ephemeral sessions never persist captures; deleting a
      conversation removes its captures.
      *(The current sibling-branch Stop path, ephemeral path, and delete
      cascade were live-verified against OpenAI. Legacy `begin_variant_stream`
      compatibility remains covered by focused store/UI regressions.)*
- [x] Clicking the cost chip (and Ctrl+Shift+P, and the command-palette entry)
      opens one Conversation Inspector with Costs, Exchange, and Next Send
      tabs; the old cost and context modals are retired with their behavior
      pins migrated.
      *(The command-palette path was live-driven. Mounted behavior pins cover
      the cost-chip action and raw Ctrl+Shift+P binding, including all three
      tabs; the final source scan finds no retired modal class definitions or
      runtime references.)*
- [x] Per-piece token figures are labeled estimates alongside the provider's
      reported buckets; turns without captures render an explicit
      "no capture recorded" row.
      *(Both halves live-verified: "System prompt (~457 tokens est.)" /
      "Response (~0 tokens est. / reported out:14)" alongside "Reported
      usage -- in:2234 cache_r:0 cache_w:0 out:14"; and the exact string
      "No capture recorded for this turn (recorded before capture existed,
      capture disabled, or capture failed)." with the kill-switch off.)*
- [x] `[console] exchange_capture` kill-switch (default on) disables capture
      end-to-end, read through the resolved settings layer.
      *(The switch mechanism itself is fully live-verified in both
      directions: `true` produces real captures, `false` produces the
      "No capture recorded" row for a turn sent while off, after a full app
      restart with the flipped scratch config. The "(default on)" value was
      not independently tested — the scratch config set the key explicitly
      in both directions rather than omitting it to observe the bare
      default.)*
- [x] ChaChaNotes gains a local-only `message_exchanges` table (schema bump
      with migration); no sync_log or FTS coupling; writes are idempotent.
      *(Structural/DB AC — verified via the already-green automated gate run
      in task 11a (commit 8a34da7f0's predecessor state), not independently
      re-verified by hand in this live-verification pass; not the kind of
      claim a terminal capture can confirm.)*
- [x] Live verification against a real provider covers a multi-call tool turn,
      a non-empty mid-stream Stop, stopped regeneration under the current
      sibling-branch model, and the kill-switch.
      *(The earlier multi-call and kill-switch run is complemented by the
      isolated 2026-08-26 lifecycle replay documented below. The legacy
      abandoned-variant state is not emitted by current regeneration and is
      therefore verified by compatibility tests instead of misreported as a
      live current-path result.)*

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-18-console-conversation-inspector.md`
(11 TDD tasks; re-anchored 2026-08-18 against origin/dev @ 1bdbcac61 —
scoped call signals, PreparedProviderRequest kwargs, schema v40→v41).

ADR required: no

ADR path: N/A

Reason: this closeout verifies the existing Inspector architecture and applies
a narrow OpenAI stream-cancellation cleanup fix without changing storage,
provider, security, or UI boundaries.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Tasks 1-10 (capture core, gateway wiring, llama.cpp branch,
store/persistence lifecycle, schema v41, controller + kill-switch, Costs/
Exchange/Next Send UI, retirement of the old modals) landed across 15
commits (`e0ceda3b0`..`50c65a752`..`dd4921901`) plus two review-round fixups
(`1850ea3dc`, `cee88d074`). Task 11a (docs + targeted gate, commit
`8a34da7f0` and its predecessor) is already done and out of scope here. This
task (11b) completes live verification against a real OpenAI account, fixes
the one cancellation defect that verification exposed, and closes the
backlog/evidence trail.

**Config/data isolation.** Every session used a fully isolated scratch
profile: `HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_CACHE_HOME`/
`TLDW_CONFIG_PATH` all pointed at a session-scratchpad directory (never
`/tmp` directly, per repo convention), with the OpenAI key read from the
repo-root `openai-api-key.txt` into the scratch config's
`api_settings.openai.api_key` — never echoed to any log or transcript.
Verified afterward: the real `~/.config/tldw_cli/config.toml` (mtime
2026-08-17) and `~/.local/share/tldw_cli/` (newest entry 2026-08-18) were
both untouched by this session (2026-08-20).

**Scenario-by-scenario results.** The earlier manual session established the
multi-call tool loop, exact configured-system-prompt visibility, kill-switch,
reported-versus-estimated token labeling, and no-capture state. The redacted
final lifecycle evidence is versioned at
`Docs/superpowers/qa/2026-08-26-task-18300-live-closeout.md`.

- **(a) Multi-call agent turn with a tool — PASS.** "Use the calculator tool
  to compute 47 * 89..." produced a real 4-call agent run
  (`[1] assistant -- 4 calls`). Call 0's request expanded to show
  `Tools (11)` with real OpenAI function-calling JSON schemas (`spawn_
  subagent`, `wait_agents`, etc.). A later call's `Messages (3)` contained a
  `role: tool` message tied to the correct `tool_call_id`. The final call's
  `Response` showed `4183` — the correct answer, matching the transcript.
  One content anomaly noted (not chased further, out of this task's scope):
  the intermediate tool-role message's `content` field held the tool's
  static description string rather than the numeric JSON result; the FINAL
  response was still correct, so this did not block the user-visible
  answer.
- **(b) Stop mid-stream — PASS.** The final direct lifecycle replay consumed
  one real OpenAI delta and then closed the stream. The assistant retained a
  non-empty partial, both native and durable capture status were `stopped`,
  and the durable row retained a non-empty response payload.
- **(c) Regenerate then Stop — PASS under the current branch model.** Current
  regeneration creates a persisted sibling instead of mutating/restoring a
  variant. The original remained complete and non-empty, while the sibling
  remained non-empty with a `stopped` capture. The legacy abandoned tag count
  was correctly zero on this current path; focused tests preserve the older
  restored-variant labeling contract.
- **(d) Captured system prompt matches configured, byte-for-byte — PASS.**
  A marker string (`SYSTEM-PROMPT-VERIFY-9f3d2a1c You are a terse test
  assistant. Keep replies short.`) was set live via the command palette's
  "Console: Edit system prompt" modal, then observed as the exact first line
  of the Exchange tab's expanded `System prompt` section on every
  subsequent call, verbatim. (The agent runtime appends its own tool-use/
  run-log instructions after the user's configured prompt — expected,
  documented behavior, not a mismatch.)
- **(e) Kill-switch — PASS.** With `[console] exchange_capture = false` in
  the scratch config and a full app restart, a fresh turn ("Say hello in
  exactly three words.") rendered the exact string "No capture recorded for
  this turn (recorded before capture existed, capture disabled, or capture
  failed)." in the Exchange tab, with no call/status rows underneath.
- **(f) Ephemeral and deletion lifecycle — PASS.** A real ephemeral call
  retained one governed in-memory capture but created neither a persisted
  conversation/message nor a durable exchange row. Hard-deleting the durable
  control conversation reduced its exchange-row count from one to zero.

**Defect found and fixed.** `chat_with_openai()` yielded its synthetic
`[DONE]` sentinel from `finally`. Closing the generator on Console Stop injects
`GeneratorExit`; yielding while handling that control signal raised
`RuntimeError: generator ignored GeneratorExit` and delayed transport cleanup.
A red-before-green regression now closes the generator after its first mocked
delta and asserts the response closes without another yield. Cleanup remains in
`finally`; the normal/error completion sentinel now follows it. Replaying the
real lifecycle produced no ignored-`GeneratorExit` warning.

**Actual spend:** the earlier session recorded approximately $0.0061. The
final replay added four tiny `gpt-4o-mini` calls and remained well below the
$0.10 task budget. An exact closeout total is not claimed because early Stop
correctly closes the stream before the provider's final usage bucket arrives.

**Technical decisions / trade-offs.**
- DB/schema and legacy-compatibility claims rely on focused automated tests;
  current provider lifecycle claims rely on the isolated real-provider replay.
  The acceptance wording was updated from "abandoned regeneration" to the
  current persisted-sibling model because later controller work superseded
  variant mutation. The legacy path remains pinned rather than being deleted.
- No attempt was made to live-verify the llama.cpp branch (no local
  llama.cpp server configured) or binary/base64 attachment stubbing (no
  attachments sent) — both are covered by the automated suite only.

**Final verification.** The closeout ran targeted checks only, per repository
policy: `Tests/Chat/test_openai_streaming_usage.py` (6 passed); the Inspector,
store, controller, chip, keyboard/palette, and import-provenance gate (110
passed); Ruff on both changed Python files (clean); `git diff --check` (clean);
and a source scan confirming the retired modal classes have no definitions or
runtime references. The two pytest warning classes are incumbent dependency
version/deprecation notices; additional post-summary output was the repository's
known temp-cleanup noise and did not change the zero exit status.

Qodo review follow-up added a gateway integration regression that traverses
`ConsoleProviderGateway` → `chat_api_call` → `chat_with_openai`, cancels after
the first recorded provider delta, and asserts transport cleanup plus a stopped
partial capture. It passes with the fix and was independently proven red by
temporarily restoring the old yield-from-`finally` behavior.

**Files changed in this task.**
- `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (Stop-safe OpenAI stream cleanup)
- `Tests/Chat/test_openai_streaming_usage.py` (GeneratorExit regression)
- `Tests/Chat/test_console_provider_gateway.py` (gateway-boundary Stop
  regression added during Qodo review)
- `backlog/tasks/task-18300 - Console-Conversation-Inspector-exchange-capture-and-unified-cost-context-review.md`
  (final acceptance and implementation evidence)
- `backlog/docs/lessons-live-verification.md` (Stop/GeneratorExit incident)
- `Docs/superpowers/qa/2026-08-26-task-18300-live-closeout.md` (redacted UAT
  evidence)
<!-- SECTION:NOTES:END -->
