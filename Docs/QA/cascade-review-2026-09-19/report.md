# Simplification-cascade review — 2026-09-19

Baseline: `origin/dev` `d6e2a46384`, reviewed read-only in a detached worktree
(`/tmp/tldw-dev-review`). The main checkout was not touched. Method: mined the
2026-09-17 core-runtime review (`qa/core-code-review-2026-09-17/` — report, 29
slices, machine duplication census) for family-level signals, then verified and
quantified each candidate cascade against current dev with five read-only
exploration agents. Every claim below carries file:line evidence verified on
this baseline; LOC deltas are planning estimates, not measurements.

Relationship to the 2026-09-17 review: that review filed per-site work
(helper adoption TASK-32808.x, dead code TASK-32807.x, P3 bundles TASK-32810.x).
This review covers the **family-level collapses** those tasks do not — "one
insight that eliminates multiple components" — and files them as the
`review-cascade` stream, **TASK-32850–TASK-32864** (block claimed 2026-09-19;
concurrent sessions, re-sweep before minting anything adjacent).

## Verified cascades, ranked

| # | Cascade | Est. deletable LOC | ADR status | Task |
|---|---|---|---|---|
| 1 | Every OpenAI-compatible hosted LLM call is one `hosted_chat` engine (chat handlers first, then both summarization libraries) | ~1,900–2,200 conservative; ~5,100–5,400 optimistic of ~10,100 in scope | ADR-062 explicitly names deepseek/groq/mistral/openrouter as awaiting "separately tested migrations" | 32851–32854 |
| 2 | Tool-provider plumbing is one base (ledger + pending-gate + gate helpers + invoke prologue, 9 providers) | ~850–980 total; ~450–550 beyond TASK-32808.7 | ADR-032 mandates the seam; ADR-094 semantics parameterizable | 32857 |
| 3 | Image_Generation and Video_Generation are one modality-parameterized core | 900–1,400 net | ADR-044 Consequences pre-sanction the merge; supersedes decision 2 only | 32856 |
| 4 | TTS engine convergence | ~1,500–2,200 near-term; 2,500–4,500 at full completion | ADR-023 staged bridge governs; near-term quick wins only | 32863 |
| 5 | Dead legacy chat tail in Chat_Functions | ~1,270 now; ~1,950 with MediaWindow_v2 migration | none; reconcile with TASK-32807.5's census | 32858 |
| 6 | Console provider-selection builder written twice | ~230–330 + removes the demonstrated double-fix drift class (PR-2668 fix applied at both copies) | ADR-006/146-compliant if the registry-aware variant wins | 32859 |
| 7 | Riders: recovery-admission guard trio (~300–400); StatusLine ×21 (~150–250); MCP datetime codecs (~110 + a latent naive-tz inconsistency) | ~560–760 combined | none | 32860–32862, 32855 overlap |
| 8 | Strict-JSON parser ×3 drifted families + moonshot/zai ~14 duplicated validators | ~300 | implements ADR-062's direction; acceptance-rule unification is a small contract decision | 32855 |
| 9 | Modal dismissal adoption + confirmation pattern extension (ADR-161-compliant slice only) | pattern adoption; the ~12k universal base stays rejected | ADR-161 explicitly rejected Python builders (two died of disuse) | 32864 |

## Blocked / retired — recorded so nobody re-litigates

- **Universal modal base (~130 ModalScreen classes, ~44k LOC family, ~8–15k
  conservative deletable)**: real, but ADR-161 rejected Python builder layers
  with disuse evidence ("two prior builder libraries died of disuse (0 and 2
  importers)"). Only the compliant slice is filed (32864).
- **MCP JsonStore base**: the atomic-write/backup machinery is already
  centralized in `Backup_Recovery/mcp_source_participants.py`; the residue
  (~150–250 LOC of thin wrappers) fights the recovery framework's
  `cls.__dict__` identity checks and `_SOURCES` pinning. Not worth it.
- **Console state "stored 3 ways"**: misread — `console_display_state.py` is
  pure derived contracts, `session.py` forwarding properties, mirrors
  documented and load-bearing (ADR-006).
- **Subscriptions per-provider special cases**: none exist — one 123-method
  engine, zero provider branches.
- **DB store template, private-sqlite validator, LIKE-escape ×9**: already
  filed by the 2026-09-17 review (TASK-32808.8/32808.11/P3 bundles).

## Execution risks and open issues (the "before continuing" review)

1. **The engine's own contract suite is red.** TASK-19642.10 (To Do, high)
   tracks 25 failing hosted-chat/QwenCloud transport contract tests. Migrating
   four providers onto an engine whose contract suite is red inverts the
   evidence order. The migration tasks (32851/32852) carry a hard dependency on
   TASK-19642.10.
2. **Engine gaps must land as neutral capabilities, not per-provider flags.**
   `owned_json_post` hardcodes Authorization+Content-Type (`hosted_chat.py`'s
   request path) — OpenRouter's `HTTP-Referer`/`X-Title` need an `extra_headers`
   hook. `HostedChatStream` raises if a stream ends without usage; the four
   providers never request `stream_options.include_usage` today, so migrations
   must add it to payloads AND keep the legacy raw-line/raw-dict consumer shape
   via the moonshot `MoonshotStream`/turn-shim pattern. ADR-062 forbids
   "provider-specific flags that weaken its existing contract".
3. **Tests currently pin defects.** `test_summarization_diagnostic_privacy.py`
   freezes per-function log call sites (any consolidation re-keys the ledger —
   the task-17387 constraint) and the kobold/tabby tests consume the broken
   generator contract as correct. The summarization tasks must re-key the
   ledger coherently, not just port tests.
4. **Metric-label fixes change observable output.** groq logs
   `openrouter_api_*` (streaming) / `mistral_api_*` (non-streaming); deepseek
   logs `mistral_api_*`; mistral logs `openrouter_api_*`. If any dashboard or
   analysis keys on these (wrong) names, the migration "fix" is a breaking
   rename — check consumers before merging, note the rename in the PR body.
5. **DeepSeek dual-API interaction.** TASK-15677 (To Do, high; ADR-064) adds a
   Responses-API wire mode to deepseek. Migrating deepseek onto the engine
   first shrinks 15677's surface (it becomes a wire-mode on a profile); doing
   15677 first duplicates the transport work. Sequencing recorded in 32852.
6. **TASK-18313 (gateway bridge tests red)**: any work that touches the
   gateway seams should wait for it; the migrations stay behind `chat_api_call`
   and should be safe, but 32858's MediaWindow_v2 option must not be coupled to
   a moving gateway.
7. **Census reconciliation for the dead tail.** TASK-32807.5's census (~2,350
   dead Chat lines) may already include part of the Chat_Functions tail; 32858
   AC#1 is the reconciliation, so the board never double-claims the same lines.
8. **Image/Video merge polarity choices.** Image carries hardening video lacks
   (config snapshot context + runtime lock in `adapter_registry.py`; ComfyUI
   `_BodyChunkSupervisor`/`_SendSupervisor`/`/object_info` validation). The
   merge must pick image's stricter behavior per site or record the exception —
   silently averaging the two is how the mirror drifted in the first place.
9. **Pin behavior BEFORE refactoring the tool providers.** The pop-vs-peek
   split, raw-shell authority-generation fencing, and builtin's
   never-persist-`always_allow` rule are load-bearing (ADR-094) and, if
   unpinned, a base class could silently homogenize them. Characterization
   tests per provider are a precondition, and 32857 depends on 32808.7 landing
   first to avoid double-touching the same 531 ledger LOC.
10. **Size ratchets.** Several target files carry ratchet rows
    (TASK-32809); big deletions must re-pin their rows in the same PR.
11. **Evidence standard.** Per `backlog/docs/lessons-testing-evidence.md`:
    per-migration targeted runs (provider suites + gateway + sensitive-logging)
    are the bar; full sweeps only on explicit request.

## Filed tasks

Parent TASK-32850; children 32851–32864 (all To Do, `core-review` +
`review-cascade` labels). Ordering recommendation: 19642.10 → 32851 → 32852 →
32853/32854 (summarization) → 32856 (needs its ADR first) → 32857 (after
32808.7) → 32858/32859 → riders 32855/32860–32862/32863/32864 opportunistically.
