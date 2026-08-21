---
id: task-18300
title: Console Conversation Inspector — exchange capture + unified cost/context review
status: In Progress
assignee: ['@claude']
created_date: '2026-08-18'
labels: ['console', 'ui', 'db', 'transparency']
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
- [ ] Captures survive Stop (partial, marked stopped) and abandoned
      regenerations (kept, marked abandoned); ephemeral sessions never persist
      captures; deleting a conversation removes its captures.
      *(Left unticked: Stop-survival was only demonstrated with EMPTY partial
      content (two genuine `[stopped]` captures, `Response ~0 tokens est.`
      both times) — never a non-empty partial. The abandoned-regeneration
      half of this AC was NOT demonstrated at all: four honest attempts to
      stop a regeneration mid-flight all raced past natural completion. See
      scenario (b)/(c) in Implementation Notes and the new lessons-file entry.
      Ephemeral-never-persists and delete-removes-captures were not exercised
      in this live session.)*
- [ ] Clicking the cost chip (and Ctrl+Shift+P, and the command-palette entry)
      opens one Conversation Inspector with Costs, Exchange, and Next Send
      tabs; the old cost and context modals are retired with their behavior
      pins migrated.
      *(Left unticked: only the command-palette entry ("Console: View chat
      context") was live-driven, and it correctly opened one Inspector with
      all three named tabs every time. The literal cost/token chip click and
      a raw Ctrl+Shift+P keypress were not exercised — tmux send-keys cannot
      reliably synthesize Ctrl+Shift+<letter> (documented tmux limitation,
      `verify` skill) and no mouse-hit-test was attempted on the status-bar
      chip. Retirement of the old modals was not independently re-clicked;
      it is evidenced by the git history (commit 69561ac12) rather than by
      this session.)*
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
- [ ] Live verification against a real provider covers a multi-call tool turn,
      a mid-stream Stop, an abandoned regeneration, and the kill-switch.
      *(Left unticked as a whole: multi-call tool turn — PASS; mid-stream
      Stop — PARTIAL (badge renders, non-empty partial content not
      demonstrated); abandoned regeneration — NOT VERIFIED; kill-switch —
      PASS. Full per-scenario detail in Implementation Notes.)*

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-18-console-conversation-inspector.md`
(11 TDD tasks; re-anchored 2026-08-18 against origin/dev @ 1bdbcac61 —
scoped call signals, PreparedProviderRequest kwargs, schema v40→v41).

## Implementation Notes

**Approach.** Tasks 1-10 (capture core, gateway wiring, llama.cpp branch,
store/persistence lifecycle, schema v41, controller + kill-switch, Costs/
Exchange/Next Send UI, retirement of the old modals) landed across 15
commits (`e0ceda3b0`..`50c65a752`..`dd4921901`) plus two review-round fixups
(`1850ea3dc`, `cee88d074`). Task 11a (docs + targeted gate, commit
`8a34da7f0` and its predecessor) is already done and out of scope here. This
task (11b) is Steps 3-5: live verification against a real OpenAI account
(`gpt-4o-mini`, three isolated tmux sessions), backlog close-out, and a
lessons entry.

**Config/data isolation.** Every session used a fully isolated scratch
profile: `HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_CACHE_HOME`/
`TLDW_CONFIG_PATH` all pointed at a session-scratchpad directory (never
`/tmp` directly, per repo convention), with the OpenAI key read from the
repo-root `openai-api-key.txt` into the scratch config's
`api_settings.openai.api_key` — never echoed to any log or transcript.
Verified afterward: the real `~/.config/tldw_cli/config.toml` (mtime
2026-08-17) and `~/.local/share/tldw_cli/` (newest entry 2026-08-18) were
both untouched by this session (2026-08-20).

**Scenario-by-scenario results** (full verbatim observations in
`.superpowers/sdd/2026-08-18-console-conversation-inspector/task-11b-report.md`):

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
- **(b) Stop mid-stream — PARTIAL.** The `[stopped]` status badge and a
  system "Response stopped by user." line render correctly (observed twice,
  genuinely early clicks at ~1.1s and ~4.5s post-send). Both times the
  Exchange tab's `Response` showed `~0 tokens est.` — a legitimate empty-
  partial capture, not a harness miss (confirmed via the capture's own
  status field, not just the transcript). Several further attempts to
  reproduce a **non-empty** stopped capture instead raced past natural
  completion, because Console's transcript pane reveals content in one late
  batch rather than incrementally for several prompt shapes tried (see the
  new lessons-file entry). Non-empty partial content was never observed
  live in this session.
- **(c) Regenerate then Stop the regeneration — NOT VERIFIED.** The
  regenerate mechanism itself works (confirmed via the `Assistant (2/2)` /
  `(3/3)` variant counter and per-generation cost deltas), but four separate
  attempts (spanning three different original turns, including a slower
  essay-style prompt with a ~6s natural completion time) to click Stop
  during the regeneration all landed after the new generation had already
  completed, never producing a genuine abandoned-and-marked regeneration
  run to inspect. Reason: tmux round-trip latency per click attempt (~1-2s)
  exceeded the actual generation time for every prompt shape tried. Not
  inferred as a pass from code reading — reported as not verified.
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

**Actual spend:** ~$0.0061 total across all three tmux sessions (well under
the $0.10 budget) — one accidental extra turn was sent when composer focus
was lost mid-sequence (cost ~$0.0004, self-corrected, documented in the
report) and one command-palette search string was misread as keyboard
shortcuts by the transcript widget when focus was on the transcript instead
of the composer (opened the Trajectory view, sent no message, cost $0).

**Technical decisions / trade-offs.**
- AC ticks in this file reflect a split standard: DB/schema-structural
  claims (the `message_exchanges` table itself) are ticked on the strength
  of the already-green automated gate (11a), since no terminal capture can
  confirm a schema migration; UI/behavior claims are ticked only when this
  session's live session directly observed them, per the task's honesty
  rules. Three ACs stay unticked as a result — see the AC list itself for
  the exact reasoning per criterion.
- No attempt was made to live-verify the llama.cpp branch (no local
  llama.cpp server configured) or binary/base64 attachment stubbing (no
  attachments sent) — both are covered by the automated suite only.

**Files changed in this task.**
- `backlog/tasks/task-18300 - Console-Conversation-Inspector-exchange-capture-and-unified-cost-context-review.md`
  (this file — AC ticks + Implementation Notes)
- `backlog/docs/lessons-live-verification.md` (new entries: four Console-
  inspector implementation traps from tasks 1-10, and the "batched
  transcript reveal vs. stop-click race" finding from this task's own
  scenario (b)/(c) attempts)
- `.superpowers/sdd/2026-08-18-console-conversation-inspector/task-11b-report.md`
  (new — full per-scenario verbatim evidence)

No application code was changed by this task (verification + docs only).
