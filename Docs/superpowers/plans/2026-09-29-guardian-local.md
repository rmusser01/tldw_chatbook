# Guardian × Dreams Local Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers-plans executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the local Guardian — pattern rules over the user's typed Console prompts with humane escalation, crisis resources, trend analysis, and a hard-gated Dreams aggregate feed — off by default with zero footprint.

**Architecture:** A `Guardian/` package + `DB/Guardian_DB.py` following the DreamsDB template, a checker hooked at the `prompt_queue.dispatch` seam (before ADR-148 UserPromptSubmit hooks), a trend analyzer running at visit-end + daily scheduler, and a Dreams interop via a new profile-source reader under the `feeds_discovery` gate.

**Tech Stack:** Python ≥3.12 (stdlib + existing deps), SQLite via `BaseDB`, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-29-guardian-x-dreams-local-design.md` + `backlog/decisions/204-guardian-local-self-monitoring.md` (ADR-204). Binding invariants live in the spec's sections and ADR contracts 1–9 — executors read all three.

**Worktree:** `.worktrees/feat-guardian-local` (branch `feat/guardian-local`, based on `feat/dreams-phase-2` @ the consolidated Dreams stack). Tests: `../../.venv/bin/python -m pytest` from worktree root.

## Global Constraints (from ADR-204's nine contracts — every task implicitly includes these)

- **Zero footprint when off:** `[guardian] enabled=false` (default) → no DB file, no checks, no measurable latency; storage built ONLY by a successful settings-toggle enable (the Dreams R2 pattern, `settings_screen.py:30327-30334` precedent).
- **Crisis rules never silence:** `is_crisis=1` rules can never hold or escalate to `redact`/`block` — write-boundary rejection + test-pinned. Crisis rules surface `notify` + ported resources (988 / Crisis Text Line / SAMHSA / IASP) + the "not a mental health service" disclaimer, always.
- **Digest-only storage:** alerts store sha256(whitespace-stripped prompt) + topic + ids + ts — never text, never matched spans (spans are transient in live notices only).
- **Dedup gates notices only:** every match records + counts toward escalation; `notification_frequency` suppresses surfacing.
- **Seam ordering:** checker runs at dispatch BEFORE user `UserPromptSubmit` hooks; hits count regardless of later hook blocks; Guardian's own `block` short-circuits first; the hook emission point is untouched.
- **Fail-open, loud:** checker errors never eat a send; one `guardian_checker_error` notification per error-signature per visit + loguru context.
- **Slash drafts skipped** (commands divert upstream — verified); conversational drafts only.
- **feeds_discovery gate:** only rules with `feeds_discovery=1` (never crisis rules) reach Dreams' profile reader; crisis-adjacent text may never become an outbound search query (whole-payload absence test).
- **Trend analyzer excludes its own rows** (`rule_id IS NOT NULL` inputs).
- Parameterized SQL; blocking calls off the dispatch/event thread (`asyncio.to_thread`); UTC ISO timestamps; ADR-150 tokens (reused classes only); ADR-031 bindings; targeted test runs; pristine output; NEVER `git commit --amend` (shared-worktree rule); explicit-path commits, `feat(guardian):`/`test(guardian):` prefixes; one commit per task.
- Baseline gate before every commit: `../../.venv/bin/python -m pytest Tests/Guardian/ Tests/Dreams/ Tests/UI/test_settings_dreams.py -q` green + any new files.

## Verified seam reference (recon 2026-09-29 + 2026-09-29 Guardian recon)

- **Dispatch seam:** `ConsolePromptQueueUIController.dispatch` (`UI/Console_Modules/prompt_queue.py:878`); the existing blocked-reason refusal gate (notify + system-row plumbing) at `:887-903`, injected via `UI/Console_Modules/wiring.py:2213-2218` — the checker hooks beside it. Slash commands divert UPSTREAM (`UI/Screens/chat_screen.py:19182-19247` parses before dispatch).
- **Visit:** a lifecycle concept WITHOUT an id (`Chat/console_runtime.py:4890` — `attach_view` opens a visit; `leave_console_runtime` `:4920` ends it; `on_unmount` `UI/Screens/chat_screen.py:17124` triggers it). Visit-id is DERIVED (uuid minted at the checker's first send per mount). Session: `ConsoleChatStore` `session_id` (`Chat/console_chat_store.py:1633`).
- **Inline notice surface:** `ChatScreen._append_native_console_system_message` (`chat_screen.py:18890`), `markup=False`; notifications: `NotificationDispatchService.dispatch(category, title, message, severity, ...)`; post-visit surfacing rides the report-on-next-mount slot (`app.py:3030-3036`).
- **DB template:** `DB/Dreams_DB.py` (BaseDB subclass, `_SCHEMA_DDL` tuple, `schema_version` table, thread-local connections); path helper precedent `config.py:9800` `get_dreams_db_path()`; app-owned builder precedent `TldwCli.get_dreams_db` (`app.py:6295`).
- **Settings precedent:** the Dreams section (`settings_screen.py:20691` render hook; toggle `:30307` → `_persist_dreams_toggle` `:30319`; gate copy `:30102`; `_dreams_db` raw-attr read `:30118-30126`).
- **Dreams reader contract:** `CycleDeps` getter → `profile_sources.read_*` (template: `read_note_topics`, `Dreams/profile_sources.py:48`) → `merge_signals`; `_preferred_sources` (`Dreams/cycle_service.py:366`); profile `source` CHECK at `DB/Dreams_DB.py:84-86` (v3 bump adds `'guardian'`).
- **Scheduler precedent:** `dream_track_handler.py` + `DreamsProjection._track_task` for the daily trend task.
- UI-test conventions: bare-App files carry the module-scope `import Tests.UI.app_factory  # noqa: F401` binding; full-harness tests carry `@pytest.mark.bootstrap_profile`; Dreams settings tests use the `@private_profile_test` child-process seam.

---

### Task 1: GuardianDB + rules engine + check pipeline at the dispatch seam

**Files:**
- Create: `tldw_chatbook/DB/Guardian_DB.py`, `tldw_chatbook/Guardian/__init__.py`, `tldw_chatbook/Guardian/settings.py`, `tldw_chatbook/Guardian/rules_engine.py`, `tldw_chatbook/Guardian/check_pipeline.py`
- Modify: `UI/Console_Modules/prompt_queue.py` (+`wiring.py` only if the injection pattern requires it), `tldw_chatbook/app.py` (app-owned `get_guardian_db` builder via the post-`_ui_ready` seam, ADR-097)
- Test: `Tests/Guardian/test_guardian_db.py`, `Tests/Guardian/test_rules_engine.py`, `Tests/Guardian/test_check_pipeline.py`

**Interfaces:**
- Produces (Tasks 2–3 build on these exactly):
  - `GuardianDB(BaseDB)` schema v1: `guardian_rules`, `guardian_alerts`, `guardian_escalation_state`, `guardian_visit_summaries` per the spec's §Data model (columns verbatim; `rule_id NULL` on alerts for trend rows; the three EXPLAIN-pinned indexes + census rows). Methods: `list_rules(enabled_only=False) -> list[dict]`, `get_rule(rule_id) -> dict|None`, `upsert_rule(**fields) -> int` (**rejects `is_crisis=1` combined with `action != 'notify'` or `feeds_discovery=1` — `GuardianRuleConflict(RuntimeError, reason_code="guardian_rule_conflict")`**), `delete_rule(rule_id)`, `insert_alert(*, rule_id=None, session_id, visit_id, topic, message_digest) -> int`, `count_recent_alerts(rule_id, *, session_id=None, visit_id=None, since_iso) -> int`, `bump_escalation(rule_id, *, now) -> dict` (returns `{session_count, window_count, current_action}` applying the ladder + crisis cap), `set_cooldown(rule_id, until_iso)`, `cooldown_active(rule_id, now) -> bool`, `prune_alerts(cutoff_iso) -> int`.
  - `rules_engine.CompiledRules` — `compile_rules(rows: list[dict]) -> CompiledRules` (regex + except-pattern spans, 60s TTL cache keyed on a rules-version counter bumped by every `upsert_rule`/`delete_rule`); `CompiledRules.match(draft: str) -> list[Match]` where `Match = {rule_row, span_text (≤80 chars), is_crisis}`.
  - `check_pipeline.GuardianChecker` — `__init__(self, *, db_getter, session_id_getter, notify, append_system_row, now)`; `check(draft: str) -> CheckResult` where `CheckResult = {"action": "allow"|"notify"|"redact"|"block", "redacted_draft": str|None, "notice": Notice|None, "recorded": bool}`; **fail-open** (any internal error → `action="allow"`, one error notification per signature per visit); **enabled-gate first** (single cached config read); skips command-kind drafts; visit-id minted per mount (uuid held on the checker, reset when `session_id_getter` reports a fresh mount — Task 1 verifies the anchor: mint at first check after construction, expose `visit_id` property).
  - Pipeline order in `check`: gate → match → dedup (notice-suppression only) → escalation bump (crisis cap applied) → act → record (one `to_thread` hop for alert+escalation).
- The dispatch hook: in `prompt_queue.py`'s dispatch path, immediately BEFORE the existing blocked-reason gate, call the checker (injected via `wiring.py` with a **getter returning None when disabled** — a None checker is a no-op passthrough, keeping the seam zero-cost when off). A `block` result routes through the existing refusal plumbing; `redact` rewrites the draft in-memory before the send continues; `notify` appends the system row (crisis rules: + resources block + disclaimer). Hook runs BEFORE ADR-148 hook emission (verify the hook-emission call site's position relative to dispatch and document in a comment).
- `Guardian/settings.py`: `GUARDIAN_DEFAULTS` (`enabled: False` first, `alert_retention_days: 180`, `fixation_share_threshold: 0.6`, `fixation_window_days: 7`, `fixation_min_hits: 30`, `doomloop_hits_per_day: 20`) + `guardian_setting(key, default=None)`.
- Seed rules: inserted ONLY at first storage build (not at import): the spec's three (crisis-awareness with except-patterns `prevention|hotline|awareness|research|study|clinical|treatment|therapy`, doomscrolling silent_log demo, editable example). Seeding lives in `Guardian_DB.py`'s `_initialize_schema` (version 0→1 transition) or a `seed_defaults()` called once by the builder — pick at implementation, pin with a test that a second build does not re-seed.

**Minimum tests (write first, RED):**
```python
# test_guardian_db.py
def test_upsert_rule_rejects_crisis_with_block_action(tmp_path):  # GuardianRuleConflict, reason_code pinned
def test_upsert_rule_rejects_crisis_with_feeds_discovery(tmp_path)
def test_bump_escalation_applies_ladder_but_caps_crisis_at_notify(tmp_path)  # thresholds hit -> crisis rule current_action stays "notify"
def test_alert_roundtrip_stores_digest_not_text(tmp_path)  # insert_alert then read: no draft text anywhere in row
def test_prune_alerts_cuts_old_only(tmp_path)
# test_rules_engine.py
def test_match_respects_except_patterns()
def test_compiled_cache_invalidates_on_rule_write(tmp_path)  # spy: second match after upsert recompiles; before, no
def test_span_capped_at_80_chars()
# test_check_pipeline.py
def test_gate_off_returns_allow_without_db_touch()  # db_getter raises if called
def test_every_match_records_even_when_notice_deduped()  # once_per_day rule: 2 hits -> 2 alert rows, 1 notice
def test_redact_rewrites_draft_in_memory()
def test_block_short_circuits_before_hooks(dummy_hook_spy)
def test_checker_error_fails_open_with_one_notification_per_signature()
def test_command_kind_draft_skipped()
```
Plus one seam integration test wiring a real `prompt_queue.dispatch` call with an injected checker (block routes through refusal; allow passes; disabled-getter passthrough adds no call).

- [ ] Step 1: failing tests (above) → RED. Step 2: implement GuardianDB + engine + checker + seam hook + app builder (`get_guardian_db`, post-`_ui_ready` deferred import, build-only-after-enable is Task 3's toggle — the builder exists now, nothing calls it until then). Step 3: gate green + import smoke (`../../.venv/bin/python -c "import tldw_chatbook.app"`). Step 4: commit `feat(guardian): GuardianDB, rules engine, and dispatch-seam checker with crisis caps`.

---

### Task 2: Crisis content + trend analyzer + post-visit summaries + daily task + retention

**Files:**
- Create: `tldw_chatbook/Guardian/crisis_resources.py`, `tldw_chatbook/Guardian/trend_analyzer.py`, `Scheduling/scheduler/handlers/guardian_trend_handler.py`; extend `Scheduling/services/dreams_projection.py`? NO — a separate `Scheduling/services/guardian_projection.py` (one daily task `guardian:trends`, the single-definition prefix pattern) — OR fold into an existing daily projection; implementer picks the lighter against the queue's named-parameter rule and documents it.
- Modify: `tldw_chatbook/Guardian/check_pipeline.py` (visit-end hook: `finalize_visit() -> Summary|None`), `UI/Screens/chat_screen.py` (visit-end call at `on_unmount` + summary surfacing via the report-on-next-mount slot), `DB/Guardian_DB.py` (summary write + retention sweep call).
- Test: `Tests/Guardian/test_crisis_resources.py`, `Tests/Guardian/test_trend_analyzer.py`, `Tests/Guardian/test_visit_summary.py`

**Interfaces:**
- `crisis_resources.BLOCK` (the four resources + disclaimer, ported verbatim from tldw_server `Docs/Design/Guardian_Self_Monitoring.md` §Crisis Resources) + `render_crisis_block() -> str` (plain text, markup-safe); shown by the checker's notice path for `is_crisis` matches (wire into Task 1's notice path — it accepted a `Notice` without the block; this task adds it).
- `trend_analyzer.analyze(guardian_db, *, now) -> list[TrendNotice]` — inputs `rule_id IS NOT NULL` alerts only; definitions per spec (fixation share ≥0.6 over 7d with ≥30 hits; doomloop ≥20/day for ≥3 consecutive days; thresholds from `guardian_setting`); each TrendNotice is inserted as an alert row (`rule_id=None`, topic = `fixation:<topic>` / `doomloop:<topic>`) and **frequency-capped: one per topic per visit** (visit-scope check before insert).
- `check_pipeline.finalize_visit()` — computes the summary (`guardian_visit_summaries` payload: per-topic counts + escalated rules + trend notices), stores it ONLY when non-empty (≥1 non-silent alert or trend notice), returns it for the surfacing slot. Called from `ChatScreen.on_unmount` (before `leave_console_runtime`), degrade-never-raise.
- `guardian_trend_handler.GuardianTrendHandler(deps_getter)` — the `dream_track_handler` spawn pattern; projection emits `guardian:trends` daily when `[guardian] enabled`; the handler runs `analyze()` + `NotificationDispatchService.dispatch(category="guardian", ...)` for new trend notices. Budget-free (local analysis, no LLM/network).
- Retention: `prune_alerts(now - alert_retention_days)` called at each daily run + each finalize_visit; logged count.

**Minimum tests:** crisis block content + disclaimer present and markup-safe; analyze() fixation/doomloop detection with boundary cases (29 hits no, 30 yes; 0.59 share no); analyzer-excludes-its-own-rows (seed a `rule_id=None` alert, assert it doesn't count); trend frequency cap (same topic twice in one visit → one notice); finalize_visit empty-visit → None and no row stored; non-empty → row + payload shape; daily handler contract (never raises, spawns nothing when disabled); retention prune called.

- [ ] Step 1: failing tests → RED. Step 2: implement. Step 3: gate + Dreams UI trio (determinism ×2). Step 4: commit `feat(guardian): crisis resources, trend analyzer, visit summaries, daily task, retention`.

---

### Task 3: Dreams interop (feeds_discovery gate + v3 bump) + Settings section + guide

**Files:**
- Modify: `tldw_chatbook/DB/Dreams_DB.py` (v3: `'guardian'` added to the profile `source` CHECK — additive, v1/v2 upgrade in place, faithful-rewind test extended), `tldw_chatbook/Dreams/profile_sources.py` (`read_guardian_topics(guardian_db, *, window_days=14)` — counts per topic over feeds_discovery=1 rules' alerts, `_WEIGHT_PER_TOUCH` weighting), `tldw_chatbook/Dreams/cycle_service.py` (`CycleDeps.guardian_db_getter: Callable|None = None` + wire into `_refresh_profile_signals` + `_preferred_sources` origin), `UI/Screens/settings_screen.py` (Guardian section: enable toggle = the ONLY storage builder, gate copy, rules table + edit modal mirroring DreamsGoalsModal patterns, cooldown status display, read-only trend keys), `tldw_chatbook/app.py` (`guardian_db_getter` into the dreams deps getter; the toggle's build call), `Docs/User_Guide/` (Guardian page: framing, opt-in, rules, trends, crisis resources + disclaimer, the Dreams gate explained; link from dreams.md).
- Test: `Tests/Guardian/test_dreams_interop.py`, `Tests/UI/test_settings_guardian.py`

**Interfaces:**
- `read_guardian_topics(guardian_db, *, window_days=14) -> list[dict]` — `{facet: 'topic', text: topic, weight}` rows from alerts whose rule has `feeds_discovery=1` only (join at the reader; crisis rows excluded by construction AND by the write-boundary rule).
- Settings: `SettingsCategoryId.GUARDIAN` render hook (the Dreams `:20691` pattern); toggle handler = the one storage-builder call (Dreams `:30327-30334` pattern, cooldown-aware: refuse + "available in N min" when any rule's `cooldown_until` is future; the refusal also applies to deactivating a rule in cooldown); rules table rows (`name · topic · action · severity · crisis?`) + edit modal (all §Data model fields; `feeds_discovery` toggle disabled with explanatory copy on crisis rules; crisis rules' action selector fixed to `notify`); cooldown remaining shown when active.
- Cooldown bypass logging: a config-file disable while cooldown active is detected at next settings render and logged once (`guardian_cooldown_bypassed`, loguru).

**Minimum tests:** v3 upgrade (rewound v2 file reopens with `'guardian'` accepted in `upsert_profile_entry` source); `read_guardian_topics` counts only feeds_discovery rules' alerts within the window; **whole-payload absence**: a synthesized payload with guardian topics present contains crisis-rule topic text NOWHERE (goals-gate pattern); settings toggle builds storage on enable / refuses during cooldown / never builds when off; rules editor CRUD + the two write-boundary rejections surfaced as notices; Dreams refresh with guardian_db_getter None degrades silently.

- [ ] Step 1: failing tests → RED. Step 2: implement. Step 3: full gate ×2 + import smoke. Step 4: commit `feat(guardian): Dreams aggregate feed behind the feeds_discovery gate, Settings section, user guide`.

---

## Self-Review (performed during writing)

**Spec coverage:** §Package layout → Tasks 1–3 files; §Data model → Task 1 (indexes + census rows named); §Rules engine + pipeline (visit-id derivation, slash skip, cache, budget, ordering, fail-open) → Task 1; §Seed rules → Task 1; §Disable semantics → Task 3 (toggle + data preserved — the store is never dropped by disable; noted in guide); §Notices (transient span, crisis block, empty visits) → Tasks 1+2; §Trend analyzer (definitions, cadence, caps, loop-exclusion, mitigations) → Task 2; §Dreams interop + gate + v3 → Task 3; §Settings → Task 3; §Error handling → both; §Testing → each task's minimum tests + the interop payload-absence test; §Governance (ADR-204 written; tasks filed at execution start); §Phasing = the three tasks. Response-side monitoring, partner_approval, server port: Non-goals — absent by design.
**Placeholder scan:** one implementation-freedom point (seed placement in schema-init vs builder; projection choice for the daily task) — both are pick-and-document with test pins, not TBDs.
**Type consistency:** `GuardianRuleConflict(reason_code)` used in both Task 1 tests and Task 3 notices; `CheckResult`/`Notice` shapes defined once (Task 1) and consumed by Task 2's finalize; `read_guardian_topics` signature matches the profile_sources template; `guardian_db_getter` field name consistent across Tasks 1/3.
