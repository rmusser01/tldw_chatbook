---
id: TASK-40331
title: "The User Guide template still says captures are 200x50, while the capture driver and 20 of 22 images use 235x52"
status: Done
assignee: ['@claude']
updated_date: '2026-11-13 03:18'
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/_template.md
- scripts/capture_user_guide.py
dependencies:
- TASK-38970
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream, TASK-38969/38970). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The next person who follows the template's hand recipe makes 200x50 captures that do not match the other images, relying on a nav-bar clipping claim that is no longer true.

**Evidence.** `Docs/User_Guide/_template.md:100` uses `run_test(size=(200, 50))`, and `:138-147` says "Standard size: 200x50 ... 13-item nav bar ... about one column of slack" (the bar now has 15 destinations). The template's own driver pointer (`:65-69`, 814be96ead) contradicts it: `scripts/capture_user_guide.py:44` sets `SIZE = (235, 52)`, and 20 of the 22 User Guide SVGs are 235x52 (the other two are the approval-card harness at 120x20). The driver has no size option (`main` takes one target, `:373-374`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The template names the driver's 235x52 as the standard size and drops the 13-item, 200-column passage
- [x] #2 The hand recipe in the template uses the same size as the driver
- [x] #3 Docs/User_Guide/_template.md no longer tells authors to add a "Verified against dev @ <sha>" footer or to re-verify and re-stamp, and Docs/User_Guide/index.md no longer describes the stamp
- [x] #4 The template's capture recipe points readers at scripts/capture_user_guide.py first and isolates HOME, the XDG dirs and the keyring the way that driver does, not just TLDW_CONFIG_PATH
<!-- AC:END -->

## Implementation Plan

1. Re-check the driver's size (`scripts/capture_user_guide.py` `SIZE`) and the sizes of the checked-in User Guide SVGs at base 26b2e677dd.
2. In `Docs/User_Guide/_template.md`, make the hand recipe call `run_test(size=(235, 52))` and replace the "Standard size: 200×50 ... 13-item nav bar" passage with one naming the driver's 235×52.
3. Grep Tests/ for pins on the old text; run the Docs tests through the wrapper; commit.

## Implementation Notes

The template's hand recipe and its standard-size note now use the driver's 235×52; the 200×50 / 13-item nav-bar passage is gone.

**Re-checked at base 26b2e677dd:** `scripts/capture_user_guide.py:44` `SIZE = (235, 52)`, used at `:396` (`run_test(size=SIZE)`); `main` (`:373-377`) takes exactly one target, no size option. Of the 22 User Guide SVGs, 20 have the 235×52 viewBox (2885 × 1318.8); the other two are `images/console/approval-card.svg` (1482 × 538) and `images/console/context-modal.svg` (1482 × 1026) -- the latter is not an approval-card render, so the page says "nearly every capture" rather than naming an exception.

**Changes.** `Docs/User_Guide/_template.md`: step 2's `run_test(size=(235, 52))`; the "Standard size" paragraph rewritten. No test pins the template (grep of Tests/ for `_template.md`, `200×50`, `13-item` found only unrelated 200x50 UI test sizes); `Tests/Docs/` 32 passed.

**Review round 2** (coordinator ruling, 2026-11-13: AC#3 and AC#4 added to this task's scope).
- AC#3: removed the template block's `*Verified against dev @ <short-sha> — <YYYY-MM-DD>*` footer and the "re-verify and re-stamp" authoring rule, replaced by a rule to keep verification in the task's Implementation Notes (CLAUDE.md's no-"Verified against" rule). `index.md:9-11` no longer describes the stamp. No doc test pins either text (grep of Tests/ for "Verified against"/"re-stamp" finds only unrelated code comments). The index's own closing `*Verified against dev @ 6b6c35a4b …*` footer (`index.md:183-186`) is an existing stamp, not a description of one; left alone and reported.
- AC#4: the old steps 1-3 set only `TLDW_CONFIG_PATH`, so HOME, XDG_* and the OS keyring were the real profile's. The recipe now starts from the driver: add a target, run `PYTHONPATH=. .venv/bin/python scripts/capture_user_guide.py <target>`, check, confirm fonts are stripped. It lists what the driver isolates, each read from `scripts/capture_user_guide.py` `_isolate` (`:64-107`: real-profile guard, HOME/USERPROFILE/XDG_CONFIG_HOME/XDG_DATA_HOME/XDG_CACHE_HOME/TLDW_CONFIG_PATH into the temp profile, `PYTHON_KEYRING_BACKEND` null, proxies and credential-like variables dropped, loopback-only sockets) and `main` (`:391-427`: DB-layer port installs, profile removed afterwards). A one-off script is told to start the way `main()` does (`_isolate()` first, then the ports). The driver docstring's "_template.md, step 5" pointer became step 4. Demo-content rules no longer mention a TLDW_CONFIG_PATH-only scratch profile.
