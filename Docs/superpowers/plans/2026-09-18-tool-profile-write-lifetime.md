# Tool Profile Write Lifetime Implementation Plan

> **For agentic workers:** Use executing-plans to implement this plan with the recorded review checkpoints.

**Goal:** Preserve admitted Tool Profile writes and current UI facts across Settings recreation and normal shutdown.

**Architecture:** A lazy app-owned coordinator holds the three existing write kinds. Settings retains preparation and confirmation; cancellable observers project bounded outcomes through current local receipt/listing paths. Existing services own all mutation authority.

**Tech Stack:** Python 3.12, asyncio, Textual 8, existing Tool Pack services.

**Spec:** `Docs/superpowers/specs/2026-09-18-tool-profile-write-lifetime.md`.

ADR required: yes. ADR path: `backlog/decisions/167-tool-profile-write-lifetime.md`.
Reason: Admitted work and outcome observation cross the Settings/application lifecycle boundary.

## Constraints

- Preserve ADR-107 exact review/profile/revision authority and existing publication cancellation boundaries.
- No schema, dependency, persistent jobs, automatic retries or broad Settings reuse.
- No cancellation-swallowing screen waits; keep the existing shutdown watchdog.
- Use existing token-backed receipt/layout behavior; no new CSS values.
- Targeted verification only; keep PR2707 draft until its visual/merge approval.

## Implementation

1. [x] Add the confirmed recreation regression and coordinator lifecycle tests; record failing evidence.
2. [x] Add `tldw_chatbook/Tool_Packs/operations.py` with synchronous admission, per-kind tasks, bounded outcomes and separate activity/completion revisions. Reuse the existing Settings readiness timer. Keep cancelled observer waits separate from operation ownership.
3. [x] Lazily wire the owner in `app.py` and add its admission-close/drain before other lifecycle owners can fail. Arm the existing process-owned watchdog before the first app drain; retain the unmount fallback.
4. [x] Update Settings to delegate only admitted writes, replay current coordinator state on new visits and refresh facts on completion. Preserve export destination recovery and all review/focus gates.
5. [x] Run coordinator, mounted recreation, existing Tool Profile workflow and app-shutdown ordering tests. Review independently and address confirmed gaps.
6. [x] Qualify real native recreation and shutdown journeys, render/inspect captures and record exact-source/private lifecycle evidence.
7. [x] Run scoped lint/format, governance and inventory checks, update ledgers/task notes, then commit/push to draft PR2707.
