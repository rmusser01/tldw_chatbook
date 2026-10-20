---
id: TASK-35646
title: Defer expired direct Console display until checked preparation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 01:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the measured native configuration entry from direct coalesced Console control-bar refresh while retaining original checked preparation, source refusal and retirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A genuine expired stock display defers without main-thread native config entry and renders only after fresh checked preparation.
- [ ] #2 Direct coalesced refresh, submission refusal and storage/source/lifecycle fences retain their correct behavior; custom readers keep their original route.
- [ ] #3 Serial native original/current profiles retain every result, exact exit, source hash and guard count.
- [ ] #4 Targeted verification, independent review and PR3166 publication are complete with auto-merge off.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: narrow existing disposable display refusal under ADR126/230 without storage, admission, permission, freshness deadline or service boundary changes. 1. Replace the old expiry-falls-back expectation with a genuine-deadline native-I/O regression; add direct coalesced and failed-submission controls. 2. Return refusal for a qualified expired display whether its existing refresh was already pending or just started; remove now-unused pending-at-entry bookkeeping. Preserve every qualification check and unknown/custom fallback. 3. Run targeted display/config/tokenizer/pricing and retirement checks plus separate original/current scalar diagnostics. 4. Run serial473c32b4ed projection-only baseline/current ABBA, independently review, report all outcomes and update PR3166 with auto-merge off.
<!-- SECTION:PLAN:END -->
