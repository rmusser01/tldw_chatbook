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
| — | — | — | — | *(none granted yet)* | — |

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
