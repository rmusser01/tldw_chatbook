# Core-loop UAT / sr UX-HCI review — 2026-07-11

Branch: `claude/uat-core-loop-2026-07` (worktree at `origin/dev` @ 75e6987c, incl. PR #601).
Captured live from textual-serve (real app CSS, worktree code), playwright bundled chromium,
viewport 2050×1240 dsf=1 fontsize 12. Two isolated profiles:
**HOME-A** virgin (true first-run, config auto-created by the app) and **HOME-B** seeded
(5 notes, 3 conversations w/ messages via `CharactersRAGDB`, sync folder with 2 `.md` files,
llama.cpp wired, `first_send_completed=true`, splash off).
Live provider throughout: **llama.cpp at `127.0.0.1:9099`** (Qwen3.6-27B gguf) — real streamed
responses, no fixtures. Scope per user decision: **core loop only** (first-run → Home → Console →
Library browse/notes/media/search/ingest → Settings); other destinations await their own redesigns.

## Workflow matrix

| Workflow | Status | Friction / block | Severity |
|---|---|---|---|
| First-run landing (virgin) | Works — lands on **Console** + blocking setup card | Docs/tests expect Home landing on first run; observed Console | P3 (verify intent) |
| Setup card → provider config | Works as blocked state | Card label `Configure API+API Key`; step 2 pre-checked; cloud-key framing for local users | P2 |
| Settings ▸ Providers & Models | Works | Credentials below ~14 advanced fields; raw 27-key catalog; near-duplicate providers | P1/P2 |
| Provider test | Works (probe runs) | Toast says "Provider test finished." — no pass/fail | P2 |
| Save provider settings | Works (persists) | **Console stays blocked until app restart** | **P1** |
| First send (streaming off = default) | **FAILS** | 30s hard client timeout → `Provider stream failed: [failed]` | **P0/P1** |
| First send (streaming on) | Works — real streamed reply | Streaming toggle is session-only, free-text `true` field | P1 |
| Generation feedback | Blocked-ish | Empty Assistant row for 30–90s; composer keeps sent text mid-run | P2 |
| Resume conversation + message actions | Works | Icon-only ♻ / `--->` need guide line; full-underline selection is noisy | P3 |
| Save as… → Note | Works E2E | 3 of 4 destinations are literal "WIP: not wired yet" rows | P2 |
| Console chats ↔ Library conversations | **Inconsistent** | Console "saved workspace" chats never appear in Library `Conversations (0)` | **P1** |
| Notes list/editor/autosave/flush | Works (DB v2 verified) | List eats `[markup]` in titles; toolbar renders as overlapped stack; stale age after edit | P2 |
| Notes sync (real run) | Works — `done · 2 changes` | Rail badge refresh lag (known follow-up, reproduced) | known |
| Ingest local file E2E | **Exemplary** — 5s, honest queue, live counts | Enter doesn't submit valid form; Start button shifts when helper hides | P3 |
| Home feed after ingest | Works (Recent mirror) | Home canvas otherwise near-empty; ignores existing conversations; no "start a chat" action | P2 |
| Library search (keyword) | Works (conversation hit, Open/Select evidence) | Run button shifts on gate-line collapse; `1 messages`; no stemming (`loops`≠`loop`); post-run 0-state copy unchanged | P2/P3 |
| Search no-sources state | Works as gate | 8-line recovery dump + 3 overlapping guidance layers + `Owner: Library source index` jargon | P2 |
| Media in-Library viewer | Works (content/search/analysis/highlights) | Summary button labeled "Open in Media manager" but stays in Library | P3 |
| Settings Overview | Works | Leads with blocked "Manual Sync v2"; internal ownership boilerplate; `Writes allowed: Yes` vs "Writes remain blocked…" contradiction | P1 copy |
| Rapid tab-switch storm (32 switches) | **PASS** — fully interactive after | none (freeze fix #595 holds) | — |

## Finding details (ranked)

### P0/P1 — the new-user-to-first-chat path is broken out of the box for local providers
1. **30s hard timeout kills default first send.** `console_provider_gateway.py:289` builds
   `httpx.AsyncClient(timeout=30.0)`; llama.cpp defaults to `Streaming: off`, so the non-streamed
   completion of any large local model exceeds 30s and every send dies as
   `Provider stream failed: [failed]` (`console-first-send-stream-failed-*.png`). With streaming on,
   the same prompt succeeds (`console-first-chat-success-streaming-*.png`). Recommend: streaming
   default ON for local providers, and/or 120s+ read timeout aligned with `api_timeout`.
2. **Saved readiness doesn't reach Console until restart.** Settings saves
   (`Provider and model settings saved.`) but returning to Console still shows the "Add an API key"
   blocking card; a fresh app instance is unlocked (`console-stale-readiness-after-save` vs
   `console-ready-after-restart`). Classic `app_config`-vs-CLI-config seam; new users cannot
   complete onboarding without restarting.
3. **Console Settings modal is session-scoped and silently non-persistent.** Streaming typed as
   `true` works for the session; config still `streaming = false` after restart; no UI hint of
   scope. Streaming/Reasoning/Verbosity/Thinking are blank free-text fields
   (`console-settings-modal-streaming-freetext-*.png`).
4. **Split-brain conversations.** Console-created chats ("saved workspace - Nm" in the rail) never
   appear in Library, which claims `Conversations (0)` / "No saved conversations yet. Save a
   Console chat and it appears here" — while DB-seeded conversations appear in both. Two
   irreconcilable definitions of "saved" (`library-conversations-zero-bug-*.png`).
5. **Provider catalog is a raw internal-key dump.** `llama_cpp (llama_cpp)` (no display name),
   `local llama.cpp (local_llamacpp)`, `Local Llamafile`, `Local Ollama` **and** `Ollama`,
   `Mistral` **and** `MistralAI`, `Custom_2`, `zai`… unlabeled, ungrouped, undocumented
   (`settings-provider-dropdown-llamacpp-*.png`). A new user cannot pick correctly.
6. **Settings copy contradicts itself and leaks roadmap.** Scope Inspector shows
   `Writes allowed: Yes` above "Mutation replay: disabled / Writes remain blocked until explicit
   review, conflict, rollback, and audit gates are implemented" on the same panel
   (`settings-providers-first-run-*.png`, `settings-overview-wall-*.png`).

### P2
7. **Failed sends pollute history and onboarding state.** The error string is persisted as an
   assistant *message* (feeds future context); `first_send_completed=true` is set by a FAILED
   send; each failure accrues another junk saved conversation (5 junk rows after the HOME-A pass).
8. **No generation progress.** Between send and first token (30–90s on local models) the Assistant
   row is empty — no spinner/status; the composer retains the sent text mid-run, implying "not sent".
9. **Setup card details.** Literal label `Configure API+API Key`; step 2 "Pick a model" is ✓ on a
   virgin profile (default `gpt-4o` counts as "picked"); step 1 "Add an API key" is wrong for
   local providers; card says "Type below, Enter to send" while the composer is blocked.
10. **Provider test toast has no outcome** ("Provider test finished." — pass? fail?).
11. **Notes list issues.** Bracketed title segments are consumed as Rich markup
    (`[draft] Q3 plan [wip]` renders " Q3 plan"; crash-risk class per the history-entry lesson);
    the sort/Sync/Import/Export toolbar renders as an overlapped vertical stack; modified-age
    stays stale after an in-canvas edit (`notes-list-toolbar-stack-markup-eaten-*.png`).
12. **Search no-sources state regressed the quiet-gate principle** (8-line Why/Next/Recovery/Owner
    dump + "Select at least one source." + Evidence hint + carry line, all at once).
13. **Save as… exposes three "WIP: not wired yet" destinations** (Chatbook/Media/Prompt) months on.
14. **Settings Overview** leads with a blocked power-user block (Manual Sync v2) and reads like an
    architecture doc ("Settings owns persisted defaults and validation", boundary lists).
15. **Providers & Models layout**: the only first-run job (credentials) sits below ~14 sampling
    fields; 8 rows of "Unavailable for llama.cpp" noise for gated fields.
16. **Home under-uses its canvas**: one import card; existing conversations/provider-ready state
    not reflected; no "start a conversation" next-best-action.

### P3 / polish
17. Moving primary buttons: gate helper lines collapse and shift `Start ingest` / `Run` ~30–40px —
    hurts muscle memory (and automation). Reserve the helper line's space or fix button position.
18. `1 messages` pluralization in search results; Model rail line `llama_cpp / ` (trailing slash,
    missing model name that the top chip shows); two rail sections both titled "Context";
    `^p Palette Menu` duplicated left+right in footer; Enter doesn't submit the valid ingest form;
    "Open in Media manager" label on an in-Library action; keyword search misses plural forms
    (no stemming); heavy full-underline selection rendering on message/list rows; splash animation
    NameError `ESCAPED_OPEN_BRACKET` in logs; recurring legacy-selector log errors
    (`#chat-api-provider`, `#app-log-display` RichLogHandler) on every boot.

## What works well (keep)
- **Ingest E2E** is the model citizen: honest gate, 5s job, live counts, queue rows with
  Open-in-Library, Clear finished, Home Recent mirror (`ingest-queue-done-5s-*.png`).
- **Tab-storm regression PASS**: 32 rapid switches, app fully interactive (#595 fix holds).
- Notes autosave + flush-on-navigate verified to DB v2; real sync run `done · 2 changes`.
- Save-as-note E2E lands in Library instantly; message-action row + contextual keyboard footer;
  Settings save/revert state machine with honest unsaved-changes banner.
- With streaming on, first chat succeeds end-to-end against a real local model.

## Upgrade opportunities (sr UX recommendations)
- **Auto-detect local servers** during onboarding (probe llama.cpp/Ollama defaults incl.
  `/v1/models`) and offer one-click setup; populate the model field from `/v1/models`.
- Readiness re-probe on Console entry (kills finding 2 without plumbing config invalidation).
- Group the provider list (Cloud / Local / Custom), give every key a display name, collapse the
  llama.cpp triplet, and add type-to-filter in the dropdown.
- Split Providers & Models into "Connect" (provider, model, endpoint, credentials, test) above a
  collapsed "Generation defaults" (sampling) section.
- One conversation model across Console and Library, with one word for "saved".
- Home: reflect real state (provider ready → "Start a conversation"; N conversations; last note).

## Not exercised (honest gaps)
Message edit modal, Stop mid-stream, long-paste unfurl, Ctrl+K switcher and alt+m popover
(xterm.js can't deliver alt; ^k untested this round), theme change persistence, per-result Open
from search, conversations filter, study/flashcards handoff, Artifacts/Personas/Watchlists/
Schedules/Workflows/MCP/ACP/Skills (out of scope per user).

## Verification
- Every matrix row above cites a capture in this folder (24 PNGs, 2050×1240, live app, live model).
- First-chat E2E proven with a real streamed llama.cpp response and persisted DB rows
  (`messages` table checked via sqlite3 during the run).
- Known/deferred items excluded from findings: opaque setup-modal backdrop (user-pref flag),
  selected-message accent border (non-obscuring-focus contract), collection-scoped search/server
  sync/conflict review surface deferrals, rail-badge refresh timing, Settings save_state stub,
  backlog tasks 152-171 and TASK-88/97/70.5/70.6/119.
