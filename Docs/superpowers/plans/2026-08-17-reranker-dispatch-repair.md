# TASK-17065: Reranker Dispatch Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the reranker's bespoke credential path and positional dispatch; call `chat_api_call` the way every other caller in this repo already does. Flip the red-on-repair seam guard, and fix the fakes that hid the defect.

**Architecture:** Two tasks. T1 = the repair + the seam/fake corrections (all ten ACs' code). T2 = closure, the spend release note, and the lesson. No live provider calls anywhere.

**Spec:** `Docs/superpowers/specs/2026-08-17-reranker-dispatch-repair-design.md` — its central decision (delete, don't rewrite) is settled; do not re-litigate.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-17065-reranker-dispatch`, branch `fix/task-17065-reranker-dispatch` (off dev `1c328e1a7`). Venv EXISTS (pinned; provenance verified).
- **EVERY Bash block starts with its own `cd <worktree>` AND echoes `pwd` + branch before any mutating git op.** Two cwd incidents in this programme; the second was caught only by a permission classifier. A cleanup block that leaves the worktree must be the LAST block of its group.
- Never `git stash`; Edit restores; RED-first; do NOT run `Tests/UI/test_library_shell.py`. ruff via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- **NO live provider calls.** Every seam fake must bind through `inspect.signature(chat_api_call).bind(...)` so a mis-ordered call raises instead of passing.
- Commits reference TASK-17065 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified findings (this session — do not re-derive)

- **All 29 handlers resolve their own credential or need none.** Enumerated live from `API_CALL_HANDLERS`: 22 use the `api_key or <config>` idiom; the other 7 were regex false-positives — `qwencloud` (`resolve_qwencloud_api_key`), `custom-openai-api` (`api_key_resolved`), `moonshot`/`zai` (`explicit_api_key=api_key` → `api_settings` → `resolve_provider_api_key`, in `LLM_Calls/moonshot.py`/`zai.py`), and `mlx_lm`/`local_mlx_lm`/`local-llm` (keyless by design). **So the reranker need not pass `api_key` at all.**
- **Every other `chat_api_call` caller already uses keywords and omits `api_key`** — `UI/Tools_Settings_Window.py:4157`/`:4296`, `UI/Screens/evals_screen.py:193`, `Chat/console_provider_gateway.py:2534` (`**kwargs`). The reranker is the sole outlier; the repair makes it match the house pattern.
- Real signature: `chat_api_call(api_endpoint, messages_payload, api_key=None, temp=None, system_message=None, streaming=None, minp=None, maxp=None, model=None, topk=None, topp=None, logprobs=None, ...)` (`Chat/Chat_Functions.py:809+`). Note the existing callers pass `max_tokens=` and `seed=` — confirm those names against the signature when writing the call.
- The reranker's current call: `RAG_Search/reranker.py:224-237` (positional, via `run_in_executor`), credential `if/elif` at `:183-206`, `self._settings = load_settings()` at `:128`.
- **AC#8 is already satisfied on dev** (picker repairability, TASK-3502 Qodo remediation) — T2 verifies and cites; no code.

---

### Task 1: The repair, the guard flip, the fakes

**Files:** Modify `tldw_chatbook/RAG_Search/reranker.py`; `Tests/RAG_Search/test_reranker_degraded_paths.py` (the guard + the fake at `:75`); any other reranker test fake that mirrors the wrong order (grep `fake_chat_api_call` under `Tests/`).

- [ ] **Step 1:** `backlog task edit 17065 -s "In Progress"` + `--plan`. Grep every reranker-seam fake in `Tests/` and list them in the report — each must be corrected, not just the one the task names.
- [ ] **Step 2 (RED — the guard flip is the proof):** rewrite `test_reranker_dispatch_binding_against_the_real_chat_api_call_signature` to assert the CORRECT binding: `api_endpoint` ← `config.model_provider`, `model` ← `config.model_name`, `temp` ← `config.temperature`, and no argument carrying a credential. It must FAIL against today's caller — paste that failure; it is the arc's central evidence.
- [ ] **Step 3 (RED):** a test asserting the reranker no longer reads a settings table: `BaseReranker` has no `_settings` attribute (or it is unused) and `load_settings` is not called during `__init__` (monkeypatch it to raise; construction must succeed).
- [ ] **Step 4 (RED):** a per-provider test over a representative set INCLUDING a keyless local (`ollama`) and a remote (`openai`): with a signature-binding fake, `_call_llm_impl` completes and the fake records `api_endpoint == the provider`; none raises `No API key found for provider:` (AC#4/#5). Parametrize; state which providers are covered.
- [ ] **Step 5:** Implement — delete the `if/elif` credential block and the `load_settings()` read; call `chat_api_call` with KEYWORDS via `functools.partial` (or a small lambda) inside `run_in_executor`, passing `api_endpoint=self.config.model_provider`, `messages_payload=…`, `model=self.config.model_name`, `temp=self.config.temperature`, and the token cap under whatever name the signature accepts (verify — the existing callers use `max_tokens=`). **Do not pass `api_key`.** Remove now-dead imports.
- [ ] **Step 6:** GREEN all four. Fix every other fake found in Step 1 to bind against the real signature. Then the full `Tests/RAG_Search/` + `Tests/Chat/test_chat_functions.py` + the redaction file; counts READ; ruff.
- [ ] **Step 7:** Gate (`RAG_EVAL=1 … Tests/RAG_Eval/`) — run it and report it, while repeating the established caveat that **it is vacuous for the reranker** (no gated cell runs one). Commit `fix(rag): reranker calls chat_api_call the way the rest of the app does (TASK-17065)` + trailer. Push (cd + echo pwd in the same block).

---

### Task 2: Closure, the spend note, the lesson

- [ ] **Step 1:** Close TASK-17065 — all ten ACs against evidence. #8 cites dev's existing repairability test by name (verify it exists and passes). #10 is discharged by the release note below. Implementation Notes: the deletion-over-rewrite decision, the 29/29 verification, and the fact that the sole outlier caller is what broke.
- [ ] **Step 2 (AC#10, the spend note):** add a short "Behaviour change" entry where this repo keeps user-facing release notes (grep `CHANGELOG`/`Docs/User_Guide` release surfaces; if none exists, put it in `Docs/User_Guide/`'s reranking section and say so): reranking-enabled profiles now issue real provider calls — one per candidate up to the configured top-k — where they previously failed silently; the cost disclosure and the skipped/degraded notice shipped in TASK-3502 are the surfaces that make it visible.
- [ ] **Step 3 (the lesson, if earned):** `backlog/docs/lessons-testing-evidence.md` — the incident: a feature module grew its own credential path and its own dispatch convention; every test fake at that seam mirrored the caller's assumption, so ~2,500 green tests could not see that the feature called zero of twenty-nine providers. The rule: a fake at a shared seam must bind against the real signature (`inspect.signature(...).bind(...)`), and a feature that resolves credentials itself is a divergence to justify, not a default. Check for a near-duplicate first and extend rather than duplicate.
- [ ] **Step 4:** Batteries re-read; collection sweep vs merge-base `1c328e1a7`; commit + push (cd + echo pwd in-block).

---

## Self-review (plan time)
- AC coverage: #1/#2/#5/#6 → T1 S3+S5 (deletion); #3 → T1 S2+S5; #4 → T1 S4; #7 → T1 S2+S6 (all fakes); #8 → T2 S1 (cite); #9 → untouched picker, stated in T2; #10 → T2 S2.
- The guard flip is the arc's proof-of-repair and is scheduled RED-first, before the fix.
- No placeholders; every test names its assertion. Ordering: guard/fakes before implementation so the repair is what turns them green.
