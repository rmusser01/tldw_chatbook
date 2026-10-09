# 097 — The four boot budgets are ratchets: they never rise

Date: 2026-08-28
Status: Accepted (owner decision, TASK-23029)
Source: `Docs/Design/2026-08-27-holistic-perf-review.md` (the structural
finding), TASK-23029.

## Context — the consumption history that forced this

Four guards budget what the app pays before and just after first paint. All
four were pinned on 2026-08-25 "just above reality". Two days later, the
2026-08-27 holistic review (dev `c6218918d1`) measured every one within 2–4%
of breach:

| guard | constant(s) | limit | 08-27 review | 08-28 (this ADR's base, `b5eaa9cf64`) |
|---|---|---|---|---|
| boot import weight (`Tests/Performance/test_app_import_weight.py`) | `MAX_TLDW_MODULE_COUNT` | 660 | 657 (headroom 3) | **666 — BREACHED** |
| `_ui_ready` census (`Tests/Performance/test_ui_ready_module_census.py`) | `MAX_TLDW_MODULES_AT_UI_READY` | 970 | ~950 | 963 (headroom 7) |
| boot CSS bytes (`Tests/Performance/test_boot_css_byte_budget.py`) | `MAX_BOOT_PARSED_CSS_BYTES` | 860,000 | 842,236 | 854,720 (headroom 5,280) |
| screen pre-import payload (`Tests/Performance/test_screen_preimport_payload_budget.py`) | `MAX_PASS_ADDED_MODULES` / `MAX_PASS_ADDED_LOC` / `MAX_SINGLE_ROUTE_ADDED_LOC` | 500 / 380,000 / 145,000 | 481 / 368,814 | 488 / 374,697 / 137,494 |

**The `limit` column above is as of adoption (2026-08-28) and is not
maintained in place.** `MAX_TLDW_MODULES_AT_UI_READY` has since moved to
**972**; every change to a limit is recorded in the exception ledger below,
which is the authoritative record. The other four limits are unchanged.

The import-weight breach in that last column was repaid by TASK-23112 (646,
headroom 14) — see "Standing breach at adoption" below. Measured on dev
`473e7c9298` at the same time, the other three read 968/970 (headroom **2**),
854,943/860,000 and 488/500 + 375,925/380,000 + 137,783/145,000: the `_ui_ready`
census consumed 5 of its 7 remaining modules in three commits, which is the same
consumption pattern this ADR was written about.

Three holistic reviews in six days (2026-08-22, 08-24, 08-27) each bought
headroom that ordinary merge traffic consumed within days. The sharpest
instance: the guards forbidding `Chat.trajectory_export` on the first-paint
path were written 2026-08-25 and breached within ~24 hours by a PR that
touched neither guard file (TASK-23020). A budget with 0.5% headroom converts
the next ordinary feature into a red build, which trains people to raise the
budget — and between the 08-27 review and this ADR's base, the import-weight
budget was in fact breached (see the ledger below).

## Decision

The constants named in the table above are **ratchets, not budgets: they
never rise.** When one of these guards fails, the legitimate responses are,
in order of preference:

1. **Defer the cost** — take the new import/CSS/payload off the guarded path
   (lazy import, facade, trimmed sheet), in the same PR that would have
   breached.
2. **Shed the cost elsewhere** — if the new cost genuinely must ride the
   guarded path, remove at least as much existing cost from the same path in
   the same PR.
3. **An explicit owner exception** — recorded as a row in the exception
   ledger below, in the same commit that changes the constant. This is loud
   and auditable by construction: a raised constant with no ledger row is a
   defect, and reviewers should reject the PR.

Raising a constant is **not** an option the failure message offers, and every
one of these guards' failure messages says so.

Not covered by this ADR (deliberately): the same files' hang tripwires and
slack catch-alls (`MAX_IMPORT_SECONDS`, `MAX_MODULE_COUNT`) and the
anti-vacuity floors (`MIN_BOOT_PARSED_CSS_BYTES`, the pre-import degeneracy
checks). Those exist to keep the measurements honest, not to price them.

## The tightening convention (re-establishing tension)

A ratchet only ratchets downward. When a PR materially reduces a measured
value — by more than the guard's standard slack below — that PR should also
**lower the constant to `measured + standard slack`**, so the freed headroom
is banked instead of silently re-consumed (the 08-22 → 08-27 history is
three cycles of exactly that silent re-consumption).

| guard | standard slack |
|---|---|
| boot import weight | 30 modules |
| `_ui_ready` census | 30 modules (warm boots wobble ±1) |
| boot CSS bytes | 25,000 bytes |
| pre-import payload | 20 modules / 15,000 LOC / 10,000 single-route LOC |

No automatic tightener is implemented; the per-PR headroom lines (below) make
the opportunity visible, and review applies the convention.

## Instrumentation that makes the policy livable

* **Breaches name the culprit.** Each guard diffs its measurement against a
  pinned snapshot in `Tests/Performance/boot_budget_snapshots/` and prints
  directional `+`/`-` lists (module names, CSS segments, pre-import routes)
  in the failure message — the trace that used to take an import tracer and
  an hour is now the assertion output.
* **Headroom is visible before the breach.** On PASS, each guard emits one
  stable line (e.g. `boot-import-weight: 650/660 modules (headroom 10)`),
  printed and raised as a `UserWarning` so it appears in pytest's warnings
  summary in CI logs.
* **Snapshots are deliberate.** They are written ONLY by
  `.venv/bin/python scripts/update_boot_budget_snapshots.py` (optionally
  `--only import-weight|ui-ready|css|preimport`); the guards never write
  them. The script refuses to pin an over-budget measurement (that would
  hide the culprits behind a blessed baseline) unless `--force` is passed to
  capture breach evidence.

## Exception ledger (append-only)

Every raise of a ratchet constant requires a row here, added in the same
commit, with the owner's explicit sign-off recorded in the PR.

| date | guard | constant | old → new | named cause | owner sign-off |
|---|---|---|---|---|---|
| 2026-08-29 | `_ui_ready` census | `MAX_TLDW_MODULES_AT_UI_READY` | 970 → 972 | `tls_trust` | Owner commit `6fac5dbf95`, "perf: raise ui-ready census ratchet 970->972 for tls_trust (PR #2223, ADR-097 deliberate refresh)" |
| 2026-09-12 | `_ui_ready` census | `MAX_TLDW_MODULES_AT_UI_READY` | 973 → 975 | agent provider routing (`Agents.agent_routing` + `Chat.sampling_params`, pure modules on the already-resident AgentService import path) | Owner directive on PR #2651 ("address all issues... approved for all of it"); commit "perf: raise ui-ready census ratchet 973->975 for agent provider routing (PR #2651, ADR-097 exception)" |
| 2026-09-14 | boot import weight | `MAX_TLDW_MODULE_COUNT` | 660 → 686 | Python backup startup admission/activation: dev `4631b60f8d` measured 643 modules; PR #2642 `9e5914fca1` measured 669 (+26). Preserves dev's existing headroom. | Owner answered "approved" to the explicit 660→686 and 975→1022 exception request in the PR #2642 work session (TASK-32562). Timing limits and backup safety checks remain unchanged. |
| 2026-09-14 | `_ui_ready` census | `MAX_TLDW_MODULES_AT_UI_READY` | 975 → 1022 | Python backup startup admission/activation and registered storage participants: same-probe dev/current warm boot measured 975/1022 (+47). Preserves dev's existing headroom. | Same explicit owner approval for PR #2642 (TASK-32562); optional recovery UI/archive services remain deferred. |
| 2026-09-26 | `_ui_ready` census | `MAX_TLDW_MODULES_AT_UI_READY` | 1031 → 1032 | `provider_registry` (ADR-179 identity + preset records; stdlib-only leaf that `config.py` imports at module scope for `CLOUD_PROVIDER_CONFIG_KEYS`). Same-probe measurement: `origin/dev` 1031, PR #2828 1032; the PR's other new resident, `LLM_Calls.hosted_provider_engine`, was deferred off the path in the same PR rather than budgeted. | Owner chose "Approve exception" when asked explicitly for the 1031→1032 raise in the PR #2828 work session; commit "perf: raise ui-ready census ratchet 1031->1032 for provider_registry (PR #2828, ADR-097 exception)" |
| 2026-09-28 | screen pre-import payload | `MAX_PASS_ADDED_MODULES` / `MAX_PASS_ADDED_LOC` / `MAX_SINGLE_ROUTE_ADDED_LOC` | 500 → 556 / 378,740 → 425,347 / 123,319 → 135,111 | Not one cause: the guard was never in `perf-guard.yml`, so ordinary merges took it red unseen (`9cd9aad65f`: 554 modules, 410,347 LOC, library 125,111 LOC; the audit read 554 / 409,566 at dev `840ed2ca58`; the refreshed snapshot names the routes). Modules pinned at the measurement, LOC at measurement + standard slack, so the guard can run as a PR gate. Paydown owned by TASK-33276 (PERF-17), which lowers all three again. | Directed by the 2026-09-27 perf audit plan (TASK-33260 AC#2: "re-pinned at the measured value with a follow-up task named in the pin"); Owner approved the re-pin in the PR #2888 work session on 2026-09-29 ("2888 approved"), then approved 556 over 554 the same day after the rebase onto dev 6423c4fbd1 measured 556: dev had added `Widgets.Settings_Widgets` and `speech_tts_panel_types` to the Settings route after the first measurement. |
| 2026-09-30 | screen pre-import payload; Console visit storage census | `MAX_PASS_ADDED_MODULES`; `MAX_VISIT_STORAGE_UNITS` (config / storage admissions, os.open calls) | 556 → 557; 39 / 112 / 35,351 → 43 / 134 / 44,712 | PR #2922 (Console hooks) merged to dev before PERF-01's guards could: `settings_screen` imports `settings_hooks` at module level (Settings route 45 → 46 modules), and every Console visit refreshes hook-permission state from disk (four runs at dev `75c06af39a`: 43 config; 127-133 storage, pinned max + 1; 42,830-44,712 opens). Paydown owned by TASK-33642, which lowers all of them again. | Owner chose "re-pin now, fix after" in the PR #2888 work session on 2026-09-30, so the guards land and gate dev from here on. |
| 2026-10-03 | screen pre-import payload; Console visit storage census | `MAX_PASS_ADDED_MODULES`; `MAX_VISIT_STORAGE_UNITS` (config / storage admissions, os.open calls) | 557 → 556; 43 / 134 / 44,712 → 39 / 112 / 35,351 | Paydown of the 2026-09-30 row (TASK-33642): `settings_hooks` is imported only when the Hooks category opens, and a Console visit reuses the hook-permission snapshot while the config file, the permission store and the in-memory sealing state are unchanged. Same-probe census, macOS: visit 8 / 56 / 9 / ~2,160 (dev `2612fc56b2`: 8 / 59–60 / 8–9 / ~2,530); Settings pre-import back within 556. | A lowering needs no sign-off; it meets the condition attached to the owner's 2026-09-30 approval. |
| 2026-10-03 | screen pre-import payload | `MAX_PASS_ADDED_MODULES` | 556 → 557 | Roleplay frame B0 (TASK-33910.1), the shared adaptive-pane-shell ADR (ADR-212): new shared module `tldw_chatbook.Widgets.adaptive_pane_shell`, imported at module scope by `Widgets/Library/library_adaptive_reader_shell.py` (thin subclasses) and `Widgets/Library/library_rail.py` (re-exported input), so the Library route adds it. Same-session paired arms (base `f87153b799`): pass 556 modules / 411,958 LOC -> 557 / 412,492; library route 176 / 125,112 -> 177 / 125,646. Defer is impossible (a base class and a re-export cannot be lazy); nothing can be shed in B0 (`library_emergency_return.py` is reserved for the post-#2862 shed, which pays this back, R17). | Owner, verbatim: "Owner sign-off (2026-10-02, via the controller's question in the Claude Code session): Question: "B0 adds one module (tldw_chatbook.Widgets.adaptive_pane_shell) to the startup import census, plus about +540 lines on the Library route. Deferring it isn't possible, because the Library subclasses it at import time, and the planned offset (folding library_emergency_return.py in) waits for PR #2862. May I raise exactly the ADR-097 limits B0 exceeds, to the values measured on paired arms, with one ledger row quoting them, as long as growth stays within +1 module and +1,000 lines? Anything beyond that comes back to you." Owner answer: "Yes, within those bounds (Rec.)" — Approve raising exactly the exceeded limits (today: MAX_PASS_ADDED_MODULES 557 → 558; if #2862 lands first, the line-count limit instead), with an ADR-097 ledger row quoting the measured numbers." |
| 2026-10-04 | screen pre-import payload | `MAX_PASS_ADDED_MODULES` | 557 → 558 | Roleplay frame B1 (TASK-33910.2): new pure module `tldw_chatbook.UI.Persona_Modules.roleplay_frame_state` (the header state, spec section 5.11), imported at module scope by `UI/Screens/personas_screen.py`, so the Roleplay (`ccp`) route adds it. Same-session paired arms (base `4b1a256e88`): pass 557 modules / 416,217 LOC -> 558 / 416,596; ccp route 62 / 54,812 -> 63 / 55,191. Defer was ruled out by the owner (spec Q10: a lazy import moves the cost onto the first Ctrl+4); nothing on the Roleplay route folds into it in B1. | Owner, 2026-10-04, asked whether the expanded limits also cover B1's one new module (`roleplay_frame_state`); answer, verbatim: "Expand it (Rec.)" |

Row added retroactively on 2026-08-31 (TASK-25813), found while taking the
ratchet baseline for the 2026-08-30 holistic review. **The decision was the
owner's and was made deliberately** — the commit names the cause and cites
this ADR. What was missing is only the row this ADR requires in the same
commit, so the ledger read *"(none granted yet)"* while a raise had in fact
occurred. It is transcribed from the commit, not re-decided here.

The other four constants were audited at the same time and are unchanged
from this ADR's table: `MAX_TLDW_MODULE_COUNT` 660,
`MAX_BOOT_PARSED_CSS_BYTES` 860,000, `MAX_PASS_ADDED_MODULES` 500,
`MAX_PASS_ADDED_LOC` 380,000, `MAX_SINGLE_ROUTE_ADDED_LOC` 145,000.

## Standing breach at adoption (not an exception — a debt) — REPAID

**Repaid 2026-08-28 by TASK-23112: 666 → 646 own modules, ratchet still 660,
no ledger row.** Two deferrals, each re-measured with an import-parent tracer
rather than inferred from the diff:

* `Chat/Chat_Functions.py` now imports `ChatPersistenceService` inside
  `save_chat_history_to_db_wrapper` (its only construction site, never reached
  at import or during `TldwCli.__init__`): **−18 modules** — every module below
  that only the persistence service reached (`attachment_core`,
  `console_chat_fork` + `Event_Handlers.Chat_Events` + `chat_image_events`,
  `video_metadata` + `video_formats` + the package, the console
  context/dispatch/library-policy repositories, `Utils.file_handlers`, …).
* `Chat/console_raw_cli.py` reaches `Tools.raw_cli_executor` through a lazy
  `_raw_cli_executor()` accessor, and builds the default `RawShellExecutor` on
  first `execute()` rather than in `RawCliRuntime.__init__` (which `app.py`
  calls during construction): **−2 modules**.

Two of the traced items below did **not** survive measurement, and the
attribution is why: the import-parent tracer records only the FIRST importer,
so an edge can look load-bearing while a second boot-path importer keeps the
module resident regardless. `Chat.thinking_blocks` is imported at module scope
by `Chat/Chat_Functions.py` as well as by `console_runtime`, so deferring the
`console_runtime` edge buys **0**; `Chat.library_activity` (with
`Chat.trajectory` and `Utils.log_sanitizer`) is also imported by
`Agents/library_tool_provider.py`, reached via the pre-existing
`app -> UI.Tools_Settings_Window -> Agents.local_tool_provider ->
Agents.tool_catalog -> Agents.library_rag_tool_provider` chain, so those three
stayed. `Widgets.pausable_progress` and `Utils.tiktoken_runtime` were verified
genuine, as the trace predicted.

The tightening convention does not fire: the reduction (20) is under the
30-module standard slack, and `646 + 30 = 676` is above 660, so lowering the
constant would be raising it. Per-edge guards:
`Tests/Packaging/test_chat_persistence_import_closure.py`,
`Tests/Packaging/test_raw_cli_import_closure.py`.

The original debt, for the record:

At this ADR's adoption, dev (`b5eaa9cf64`) already breaches the import-weight
ratchet: **666 own modules against 660**. The constant was NOT raised. Vs the
last in-budget state (`c6218918d1`, 657 modules), 17 modules were added and 8
removed (TASK-23023's Research_Workspace diet). The added edges, traced:

* `Chat/chat_persistence_service.py` (+912 lines since the pin) gained
  module-scope imports pulling ~12 modules onto the boot path:
  `Chat.attachment_core` (→ `Utils.file_handlers`), `Chat.console_chat_fork`
  (→ `Event_Handlers.Chat_Events` pkg + `chat_image_events`),
  `Chat.library_activity` (→ `Chat.trajectory`, `Utils.log_sanitizer`), and
  `Video_Generation.video_metadata` (→ `video_formats` + the package).
* `app.py` gained a module-scope `Chat.console_raw_cli` edge
  (→ `Tools.raw_cli_executor` → `Agents.run_log`): 3 modules.
* `Chat.console_runtime` → `Chat.thinking_blocks`: 1 module.
* `Widgets.splash_screen` → `Widgets.pausable_progress` (TASK-23022) and
  `tldw_chatbook/__init__` → `Utils.tiktoken_runtime` (ADR-093): 1 module
  each; both look like genuine boot-path needs.

Under this ADR the debt is repaid by deferral/shedding, not by raising 660.
The `boot_import_modules.txt` snapshot was pinned at the `c6218918d1` set so
the guard's failure message kept naming exactly these modules until the debt
was cleared; it is now re-pinned at the post-repayment 646-module set. The
repayment is **TASK-23112** (see above).

## Component-pattern migration paydown — 2026-09-14

TASK-32596's resumed tree measured 841,903 boot CSS bytes against the 768,000
limit. Generated module payloads now omit authoring comments while editable
sources retain their rationale and generated MODULE markers retain provenance.
Quoted strings, token values and selector whitespace are covered by regressions.
The initial census was **609,050 B**, tightening the limit to **634,050 B**
(measured + 25,000 B standard slack). Completing the active Statistics source
migration brings the final census to **609,446 B**; the tightened limit stays
unchanged with 24,604 B headroom. The snapshot was refreshed by
`scripts/update_boot_budget_snapshots.py --only css`. No exception or budget
increase was used. This pays the CSS-byte debt; it does not claim a measured
startup-time improvement or complete TASK-31500's separate modal deferrals.
