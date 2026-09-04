# ADR-116: Schedules inspector-pane editing

Status: Proposed
Date: 2026-09-03
Related Task: schedules-redesign PR-3 (`.superpowers/sdd/plan-2026-09-03-schedules-redesign-pr3/`)

## Decision

The Schedules workbench's inspector-style detail pane (spec
`backlog/docs/spec-2026-09-02-schedules-screen-redesign.md`) edits
single-value rows in place, through the same `DetailValueRow` grammar
every group already renders with:

1. **Hybrid editing, not a pane-only editor.** Each `Details`/`Frequency`
   row that has a real single-field write target activates in place
   (click/Enter opens a `Select`/`Input`/small mini-editor built from
   `DetailValueRow.begin_edit`; the row's own commit closes it via
   `end_edit`). Question text, custom cron, and full-scope surgery stay
   in the create/edit modal (`ReminderForm`) behind "Edit in full…", and
   the modal remains the fallback at narrow terminal widths where the
   detail pane is hidden. This is the amendment to
   `099-schedule-editor-shape.md` (see below): that ADR closed the
   pane-vs-modal question in favor of the modal for the *whole* form;
   this decision does not reopen that call. It carves out a narrower one
   — single fields only, still backed by the modal everywhere the pane
   itself is unavailable — so the width-cliff argument that made the
   modal durable stays intact for every case that argument covers.
2. **The owner row is the transfer surface.** "Runs on" renders like
   every other row and its dropdown *is* the spec §7 transfer machine:
   opening it lists `This device` / `Server (<id>)`; picking the current
   owner is a no-op; picking the other runs `transfer_refusal` first
   (a refusal renders inline via `DetailValueRow.show_error`, Textual
   `Select` cannot disable one option); an allowed pick opens the PR-5
   confirmation dialog with `transfer_warnings`; confirming drives
   `begin_transfer_to_server`/`begin_transfer_to_local` exactly as the
   existing Move/Retry/Cancel buttons do. Those buttons are a second,
   independently wired surface onto the same facade and are left in
   place through this PR (coexistence, not a refactor) — retiring them
   is PR-4's job, alongside the responsive floor and the old tab bar.
3. **Lifecycle writes need a pull guard, not a migration.** The
   definition pane's header Pause/Resume button is
   `SchedulingService.set_definition_lifecycle`'s first UI caller, and an
   optimistic local toggle races the same server pull that already
   overwrites every other field server-wins. Rather than add
   conflict-resolution machinery, `upsert_automation_definitions_from_
   server` gained a two-layer guard scoped to the `lifecycle` column
   only, ported field-for-field from `upsert_automation_results_from_
   server`'s existing TOCTOU/same-cycle-echo design (schedules-handoff
   program): (1) an in-transaction check for a pending local mutation
   whose `payload["action"]` is `pause`/`resume`/`archive` withholds
   `lifecycle` on that row for that pull; (2) a `skip_lifecycle_server_
   ids` set, threaded from `SyncEngine`'s own lifecycle-replay phase,
   withholds it for ids already pushed *this* sync cycle, whose pending
   mutation is already gone by the time the pull runs. Every other field
   keeps writing server-wins unconditionally; the guard is surgical to
   one column because `automation_definition` also carries ordinary edit
   and transfer mutations that must not freeze alongside a lifecycle one.

## Context

The spec's unified workbench (§3) retired the old Queue/Automations/
Conflicts/Results tab bar in favor of one filtered list (rail, ~40%) plus
an inspector-style detail pane (~60%) whose grouped key-value rows —
`DetailValueRow`, shipped read-only in PR-1 — were dormant placeholders
until this PR. Three rows needed a decision beyond "wire the existing
seam": the owner row, because it doubles as the PR-5 transfer machine
rather than an ordinary field; the header lifecycle toggle, because it
is the first caller of a facade method whose only existing write path
(`SyncEngine`'s definitions pull) is unconditionally server-wins; and
the editing model itself, because `099-schedule-editor-shape.md` had
already closed "should Schedules edit in a pane" against the *whole*
create/edit form, for reasons (the width cliff at narrow terminals, the
need for a discard-guarded lifecycle) that do not disappear just because
this PR only touches single fields.

`099-schedule-editor-shape.md`'s own "Consequences" section anticipated
this: "Reopen only if a future schedule editor must show live queue
context while editing … reconsider all three shapes against the width
floor rather than defaulting to the pane." This program does not reopen
the three-shape comparison — it stays inside shape 1 (modal) for every
case that comparison covers, and adds a narrower, additive carve-out for
rows the modal was never uniquely necessary for in the first place: a
`Select`/`Input` swapped into a fixed-height row commits or cancels
(Escape) without ever needing to scroll, discard-guard across a queue
selection change, or fit at the 80×24 floor — the modal's entire cost
column in that ADR. Where the pane is unavailable (narrow width), Enter
on a list row still opens the modal unchanged.

## Consequences

- `099-schedule-editor-shape.md` gets an "Amendment (2026-09-03)"
  section (this PR, in place, no renumbering) rather than a superseding
  ADR: the original decision and its width-cliff argument still hold for
  the surface it actually decided (create/full-edit/narrow fallback).
- The owner row's dropdown and the legacy Move/Retry/Cancel buttons are
  two live surfaces onto one facade (`SchedulingService.transfer_
  refusal`/`transfer_warnings`/`begin_transfer_to_server`/`begin_
  transfer_to_local`/`cancel_transfer`) until PR-4 removes the legacy
  one; a future change to transfer semantics must update both call
  sites until then.
- The lifecycle pull-guard is intentionally column-scoped, not a general
  per-row conflict system — a future field that needs the same
  optimistic-write protection (e.g. a second lifecycle-shaped write path)
  should extend the existing `_DEFINITION_LIFECYCLE_ACTIONS`/`skip_
  lifecycle_server_ids` mechanism or its `upsert_automation_results_
  from_server` sibling, not invent a third pattern.
- No schema migration: every write in this PR routes through seams that
  already existed (`save_definition`, `update_reminder`, `set_
  definition_lifecycle`, the PR-5 transfer facade); the only DB change is
  the two-layer guard's extra `SELECT`/parameter, not a new column or
  table.
