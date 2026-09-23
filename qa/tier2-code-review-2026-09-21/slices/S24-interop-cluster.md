# S24 — the 31 `*_Interop/` packages (one cluster finding, per instruction)

**Coverage:** files read in full: 2 | sampled (meaningful ranges): 9 | mechanical only (AST-parsed or grepped across
the whole slice): 167 — of 178. Out-of-slice files read as evidence: `runtime_policy/enforcement.py` (full),
`runtime_policy/engine.py:14-44`, `runtime_policy/registry.py` (parsed), `Chat/chat_conversation_scope_service.py`,
`Notes/notes_scope_service.py`, `Backup_Recovery/participants.py:42-67`,
`backlog/decisions/036-application-service-composition-lifecycle.md`, `scripts/check_timestamp_writers.py`.
The 31 packages were **not** read individually, per instruction.

## Findings

### P2 [D1] — The scope-service security helper is applied per-method, and two shipped sibling-method pairs gate the write and not its inverse
- Where: `Study_Interop/study_scope_service.py:1489` (`import_flashcards_tsv`), `:1609`
  (`import_flashcards_json_file`) vs gated siblings `:1249`, `:1281`, `:1631`;
  `Notes/notes_scope_service.py:2361` (`remove_internal_research_quick_note_owner_proof`) vs the gated `add_` at
  `:1428`. Full census: **24 ungated public methods across 10 of 47 scope services.**
- Evidence: AST intra-class call-graph over all 47 `*_scope_service.py`, marking every public method that
  transitively reaches `_enforce_policy`/`_enforce`/`require_allowed`. **Re-verified by the lead** — see
  `phase4-verification.md` "S24 (a)": the two Study imports have zero `_enforce_policy` in their bodies while three
  siblings *and the preview* all call `_study_flashcard_import_action_id`.
  `Tests/Study_Interop/test_study_scope_service.py:1648` asserts the exact policy-call sequence for the three gated
  ones and never exercises the two ungated ones — **the gap is an omission, not a pinned decision.**
- Why it matters: `ServicePolicyEnforcer.require_allowed` is **fail-closed** — `runtime_policy/engine.py:22-30`
  denies any unregistered `action_id` and `enforcement.py:45` denies when runtime state is unavailable. It is a hard
  stop, not advisory, so a method that never calls it **bypasses the boundary entirely** rather than getting a
  permissive default. The Study pair has no shipped caller today (P2 not P1); the Notes pair does
  (`Research_Workspace/local_adapter.py:497,688`).
- Recommended correction: **not "add two calls".** The defect class is that 51 copies each re-decide per method.
  Route every public method through a single funnel — the shape already exists in-tree at
  `Feedback_Interop/feedback_scope_service.py:116-134` (`_call()` normalises mode → resolves service →
  `_enforce_policy` → dispatch → normalise), and that funnel is why `FeedbackScopeService` shows 0 ungated of 5.
  Fold it into the TASK-32808.6 base.
- Size: M · ADR: yes (rides the .6 ADR) · Confidence: verified
- Already covered: partially — **TASK-32808.6's AC list is four helpers + dispatch and says nothing about the gate
  being uniformly *reached*. This is the AC it is missing.**
- **Triaged out of the 24:** `store_login_tokens`/`clear_login_tokens` (`Auth_Account_Interop`) look alarming (they
  reach `credential_store.set_secret`) but have **zero callers outside the file**; the only reachers are inside
  login/refresh methods the audit marks gated. P3, shrink to private.
  `SyncScopeService`'s five are dry-run/read-only summarisers, each documented as such
  (`:249` "does not dispatch transport calls or drain local outbox entries"; `record_dry_run_mirror_report:461`
  refuses local mode outright). Introspection getters touch no data. **Not gaps.**

### P3 [D4] — Six `*_Interop` modules, 753 lines, have zero production importers; five are test-only, one is fully orphaned
- Where: `Research_Interop/research_models.py` (65), `Sync_Interop/notes_local_store.py` (47),
  `Sync_Interop/notes_m1_flow.py` (126), `Sync_Interop/notes_mirror.py` (112),
  `Sync_Interop/sync_profile_status_state.py` (194), `Writing_Interop/writing_models.py` (209).
- Evidence: AST import graph over all 2,525 modules from `tldw_chatbook.app` + `__main__`, resolving relative
  imports **and string-literal lazy imports**. `research_models` has 0 hits anywhere including `Tests/`.
  `notes_mirror` appears to have importers only because `notes_mirror` is a **parameter name** in
  `Sync_Interop/envelope_builder.py:39`, `envelope_applier.py:49` and `domain_adapters/notes_m1.py:20` —
  duck-typed injection, never the module.
- Why it matters: `notes_local_store` + `notes_m1_flow` + `notes_mirror` form one closed cluster (the M1
  notes-mirror flow) reachable only from `Tests/Sync_Interop/test_notes_m1_flow.py` — a subsystem kept alive by its
  own tests. · Size: S · Confidence: verified
- Already covered: **none of TASK-32807's sub-tasks names an `*_Interop` module.**

## 1. What has changed since TASK-32808.6 was filed
Filed 2026-09-18 16:56. **Zero commits touch any `tldw_chatbook/*_Interop/` path since 2026-09-17**
(`git log --since='2026-09-17' --name-only -- 'tldw_chatbook/*_Interop/'` → empty). No package has diverged from the
template. Everything below corrects the task's *description*, not drift in the code.

| Task / v1-report claim | At this SHA | Delta |
|---|---|---|
| "About 45 scope services" | **47** | +2 |
| report.md:2859 "33 Interop + 12 core" | **31 Interop + 16 elsewhere** | wrong by 2 in the other direction |
| "their own copy of the same four helpers" | **six** (`_maybe_await`, `_enforce_policy`, `_require_client`, `_normalize_mode`, `_identity`, `_dump`) | **AC #1 says "the four helpers" — it will under-scope the PR** |
| "roughly 2,000 lines" | **1,776** | −11% |
| **"the fix for that exists in exactly one of the services"** | **nine services thread, in five mutually incompatible shapes** | **the load-bearing premise of the task is stale** |
| `TLDWAPIClient` runtime imports | **3 runtime / 42 TYPE_CHECKING / 2 none** — AST-verified | confirms the lead's estimate exactly |
| `_require_client` "exactly 1 shape" | **confirmed**: 3 md5 groups (23/22/2) collapse to one — 23 vs 22 is only black's line-wrap of the `raise`; the 2 differ only by `-> Any` vs `-> TLDWAPIClient` | none |

**The stale premise, in detail.** `asyncio.to_thread` now appears in 9 of 47 (3 of 31 in `*_Interop`), in five shapes:
1. `_is_memory_backed(service)` → `service.db.is_memory_db`; **absent attribute ⇒ thread it.**
   `Chat/chat_conversation_scope_service.py:150-163`, `Media/media_reading_scope_service.py:131`.
2. `_call_off_loop` → `service.diagnostics_are_thread_safe()`; **absent/false ⇒ do NOT thread it.**
   `RAG_Admin/rag_admin_scope_service.py:87-124`. **Opposite fail-direction to (1).**
3. `run_finite_local_worker(fn, …)` — **the existing shared helper**, `Backup_Recovery/participants.py:42`,
   9 importers. Used by `Notes/notes_scope_service.py:491,2005,2046`, `Media/…:195`, `Prompt_Management/…:1465`.
4. `_run_on_backend_thread(call, *, lifetime, pending)` — byte-identical pair,
   `Writing_Interop/writing_scope_service.py:90` / `Research_Interop/research_scope_service.py:116`.
5. Unconditional `await asyncio.to_thread(lambda: method(...))`, no predicate at all.
   `Character_Chat/chat_dictionary_scope_service.py:114`, `Skills_Interop/skills_scope_service.py:199`.

**Shape 2 is NEW since the task was filed** — `69eb19f849` (2026-09-19, task-32804.12), the only scope-service commit
in the window. It is the second independent hand-roll of a thread-safety predicate, **landing after the task that
exists to unify them.** Shape 3 shows the repo already owns a generic "run this sync local call on a worker and
retire its thread-local connections" helper; shapes 1, 2, 4 and 5 all reinvent part of it.
**AC #2 ("the threading rule the fixed service already documents") is unimplementable as written — there are two
documented rules with opposite defaults and no stated winner.**

## 2. Does the template hide a real defect?
**(a) Gate reachability — yes, 24 ungated public methods, 3 of them real gaps.** See the P2 above.

**(b) `_normalize_mode` null handling — retired; a bigger divergence found underneath it.**
Every shipped mode source filters empty string before the call (`UI/MediaWindow_v2.py:231`
`str(... or "local")`; `UI/CCP_Modules/ccp_persona_handler.py:41-43` membership test; `rg 'mode=""'` → 8 hits, none a
scope-service call). The `""`→local drift at `Chat/chat_conversation_scope_service.py:64` is **unreachable. Retired.**
**What the count hides instead: the `mode=None` default splits 25 SERVER / 19 LOCAL, and it tracks the filename
prefix, not the domain** — all four `Interop`/`server_*` twin pairs are inverted (lead-verified, 4 for 4; see
`phase4-verification.md`). A shared base **cannot hard-code a default**; it needs a class attribute, and the ADR has
to say which of each pair is right.

**(c) Unreachable packages — none. The "largest deletion candidate in the repo" hypothesis is RETIRED.**
All 31 packages are reachable from the composition root; residue is 753 lines (1.1% of the six packages they sit in,
0.4% of the slice). **Method note worth keeping:** ten of these packages export through a PEP-562
`__getattr__`/`_EXPORTS`/`import_module` table, so a naive import walk reports **7 dead packages and 662 unreachable
modules repo-wide**; the lazy-aware walk (resolving string constants that name a sibling or absolute module) reports
**0 dead packages**. *The naive answer is the trap here.* Spot-checked against `app.py:10650, 10655, 10692, 10153,
11092, 11361` — all six "dead" packages are constructed in the composition root.

## 3. The ADR question
No existing ADR covers this. Grep of all 244 decision files for `scope service|scope_service|ServicePolicyEnforcer|
_enforce_policy` → 7 files; the only one on point is
**`backlog/decisions/036-application-service-composition-lifecycle.md`** (Accepted, 2026-07-28). It governs *where*
scope services are constructed and **explicitly declines the adjacent abstraction**: "This decision does not
introduce a dependency-injection container, mutable service registry, or generic lifecycle manager… A future
extraction may use a typed immutable `ApplicationServices` bundle only after the codebase has more than one real
composition root." A base class is none of the three, so it is **not blocked** — but the new ADR must say so out
loud, because ADR-036 is the standing precedent for "we looked at generalising this layer and chose not to".

**Four things the ADR has to decide:**
1. **Where the base lives.** `runtime_policy/` is the right neighbourhood — `runtime_policy/enforcement.py`
   (`ServicePolicyEnforcer`) is the one shared piece the cluster already imports. Downward cost is ~0: 42 of 45
   `_require_client` definers import `TLDWAPIClient` only under `TYPE_CHECKING`, so the base's imports are
   `inspect`, `asyncio`, `enum`, `typing`. **But** the dispatch method should call
   `Backup_Recovery.participants.run_finite_local_worker`, which imports `Backup_Recovery.storage_admission` at call
   time — the base's only non-stdlib runtime edge, and it has to be an explicit ADR line, not a side effect.
2. **The dispatch predicate and its fail-direction.** Shapes 1 and 2 disagree: one threads when it cannot tell, the
   other refuses to. One has to win, and **the loser's services have to be re-verified, not just re-pointed.**
   Shape 3 already encodes the connection-retirement half both hand-roll; making it the mechanism reduces the
   decision to the predicate alone.
3. **Enum-vs-string mode, and the default — two sub-decisions, not one.** (i) 44 services use a per-domain
   `str, Enum`; 3 use bare strings. (ii) Independently, the `mode=None` default splits 25/19 with four inverted twin
   pairs. A generic base takes the enum type and the default as class attributes; **unifying them silently would
   flip a backend for whichever half loses.** The ADR must state that defaults are preserved as declared and that
   the four inversions are a separate correction.
4. **The 3 runtime `TLDWAPIClient` importers.** They import it at runtime only because request models ride the same
   `..tldw_api` import line — the client name is a passenger, not a need. `_require_client`'s only use of the name
   is a return annotation, and `Character_Chat/server_chat_dictionary_service.py:48` already proves `-> Any` works.
   One ADR sentence settles it; no one gains a runtime dependency.

## 4. A guard — warranted, but not the one the shape suggests
**Not warranted: "no 48th `_maybe_await`."** A guard whose only job is to stop a byte-identical 4-line helper
reappearing is over-engineering; the consolidation removes the reason anyone writes one, and
`git grep -c "def _maybe_await"` in review catches the rest. **Skip it.**

**Warranted: a gate-reachability check** — the one thing in this cluster a reviewer cannot see and a count cannot
answer, and it found two real defects in an afternoon.

`scripts/check_scope_service_contract.py`, wired into `preflight.sh` next to `check_timestamp_writers.py`, asserting
one thing:

> Every public method (no leading `_`) on a class whose name ends `ScopeService` transitively reaches
> `_enforce_policy`, unless its `module<TAB>class<TAB>method` key is present in
> `scripts/scope_service_gate_exemptions.tsv`.

Census shape copied verbatim from `check_timestamp_writers.py`: keyed on **qualname, not line number** (so ordinary
edits do not churn it), counts/keys **only shrink**, `--write` regenerates, and the docstring states the incident
that produced each exemption. Seeded with the 24 current rows minus the 3 the P2 fixes → **21**. A new ungated
public method fails preflight; deleting an exemption is always allowed. Post-consolidation the base's `_call` funnel
makes most services structurally unable to fail it, and the census ratchets toward empty — the trajectory
`timestamp_writer_census.tsv` already completed.

Why a ratchet and not a hard rule: `SyncScopeService`'s five dry-run readers are legitimate and documented; a hard
rule would force five fake gates or five `# noqa`. The exemption TSV makes each a reviewed, named decision.

**Scope discipline — one check, not three.** Considered and **not** recommended: (b) "every constructible action-id
exists in `CAPABILITY_REGISTRY`" — real risk (`registry.py` holds 288 ids against far more
`f"{domain}.{action}.{mode}"` combinations, and unknown ids deny), but the ids are f-strings and a static check
would be guesswork; it belongs with the policy findings. (c) a "no helper copy outside the base" count-ratchet —
cheap, and acceptable **inside the consolidation PR**, never as its own task.

## Candidate triage
`_maybe_await` 65/4/3/59 — **confirmed**, and **no call site in S24 shown to do blocking I/O on the loop**; the
ADDENDUM asked for a real instance and this slice does not have one. *(Lead's note: S12 found it —
`Media/media_reading_scope_service.py:1582`. See `phase4-verification.md` "S12-P1".)*
`_enforce_policy` 51/47 — **confirmed as text; retired as a safety claim.** The bodies being identical is not the
property that matters — 24 public methods never call any of them.
`_require_client` 47 defs / 1 shape — **confirmed by AST.** `_normalize_mode` 46/41 — **confirmed as text;
qualified** by the 25/19 default split. 47 scope services, 31/16 — **confirmed exactly.** 9 of 47 threading —
**confirmed, but in five incompatible shapes with two opposite fail-directions, not 1 fixed + 8 unfixed.**
Outward 1,776 LOC / downward ~0 stdlib — **confirmed**, with the `run_finite_local_worker` edge noted.
"The 31 Interop packages are one template" (v1 report:2872, already ❌) — **confirmed retired, second time**; and
none of the 31 is dead. The slice's `except Exception: pass` ×6, `fetchall_no_limit` ×67, `function_body_import`
×372, `lock_and_execute` ×4 — **not examined**, out of scope under the one-cluster instruction; left for whoever
reviews `Sync_Interop/sync_state_repository.py` (3,384 lines) and `Kanban_Interop/local_kanban_service.py` (2,811)
as god modules.

## D4 observations for repo-wide Phase 3
- **`run_finite_local_worker` — helper exists, partially ignored.** `Backup_Recovery/participants.py:42`, 9
  importers. Adopted by 3 services; **re-rolled** by `chat_conversation_scope_service:186-193` (an inline
  `list_and_close` closure doing the connection-retirement half) and side-stepped by 5 more.
  **This is the helper the TASK-32808.6 base should wrap, and it is named nowhere in the task.**
- **`_run_on_backend_thread` — 2 byte-identical copies** (`writing_scope_service:90` / `research_scope_service:116`),
  with `_maintenance_resume`, `finished`, `is_async_callable` travelling in the same block — 4 verbatim pairs
  between exactly these two files.
- **`_dump` — 39 defs in 3 verbatim groups**, all pydantic `model_dump` coercion, plus `_model`/`_model_dump`/
  `_model_to_dict`/`_to_plain` aliases of the same idea across 6 packages. Overlaps TASK-32808.9.
- **`Writing_Interop/local_writing_service.py` ↔ `Research_Interop/local_research_service.py`**: 6 verbatim pairs
  (`_ensure_schema`, `_discard_connection`, `_end_operation`, `_fetch_one`, `_require_one`, `_run_on_backend_thread`).
  Two sibling local stores forked once — a candidate for **TASK-32808.8**, not .6.
- **`recovery.py` per-package `capture`/`relocate`/`validate`**: 11/9/5 copies. Flagging the shape only — per the
  `_perform_safe_cancel` precedent these may be legitimate per-domain template-method implementations.
  *(Lead's note: S25 and S08 reached opposite readings on this cluster; Phase 3 resolves it — see `report.md`.)*

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The four twin `mode=None` inversions are **reachable** — i.e. some caller actually omits `mode` on those eight services | defaults verified in source; call sites not enumerated for all eight | `rg -n "outputs_scope_service\|sharing_scope_service\|web_clipper_scope_service\|notifications_scope_service" tldw_chatbook/UI tldw_chatbook/app.py -A3 \| rg -v "mode="` |
| Whether `record_sync_mirror_report` (ungated in 4 services) writes to the sync state repository on a shipped path | read only the `sync_scope_service` sibling, which refuses local mode | `rg -n "record_sync_mirror_report" tldw_chatbook/ --type py` then read each body for `_require_state_repository()` |
| Whether any *gated* method constructs an `action_id` absent from `CAPABILITY_REGISTRY` and therefore fail-closed denies on a shipped path | the ids are f-strings; static enumeration unreliable, and the app must not be run | `pytest Tests/Study_Interop Tests/Sync_Interop Tests/Auth_Account -q` with an autouse fixture failing on `PolicyDeniedError(reason_code="authority_denied")` |
| Whether the 5 test-only modules are referenced by a doc, ADR or backlog task justifying their retention | grepped code only | `rg -n "notes_m1_flow\|notes_local_store\|sync_profile_status_state\|writing_models\|research_models" backlog/ Docs/` |
| That the gate audit has no false "gated" — a method reaching `_enforce_policy` on only one branch | the walker marks reachability, not dominance | re-run the audit with the reach test restricted to unconditional calls (skip `ast.If`/`ast.Try` bodies) and diff |
