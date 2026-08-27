---
id: TASK-23022
title: >-
  Invisible ProgressBars run a 15 Hz timer forever - 88% of the Lab screen's idle CPU
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - idle
  - textual
priority: high
---

## Description

`ProgressBar(total=None)` makes Textual's `Bar` *indeterminate*, which arms `auto_refresh = 1/15` -
a **15 Hz `set_interval` that never stops**. Setting `display = False` does not stop it: Textual
gates only the *repaint* on `is_on_screen`, never the timer.

On the Lab screen this is **960 of 1018 timer fires in 15 s, changing zero pixels** - 88% of that
screen's idle CPU, at ~84 us per fire all-in.

This class is **structurally invisible** to the repo's timer census, which parses only
`tldw_chatbook/**.py`; no package file assigns `auto_refresh`, because the `set_interval` lives
inside `textual/dom.py`.

## Acceptance Criteria

- [ ] A hidden or off-screen indeterminate `ProgressBar`/`LoadingIndicator` does not run a timer
- [ ] The six live instances are fixed: `model_curated_view.py:471`, `model_installed_view.py:346` and `:352`, `model_remote_view.py:605`, `library_screen.py:12626`, `UI/CCP_Modules/ccp_loading_indicators.py:71`
- [ ] The remaining 13 `ProgressBar(` and 6 `LoadingIndicator()` sites are audited and each is fixed or recorded as safe
- [ ] A guard prevents a new hidden indeterminate progress widget from arming a permanent clock - the timer census cannot see this class today, so extending it is part of the work
- [ ] Before/after idle CPU measured with interleaved arms

## Evidence

Interleaved A/B on F7, `on/off/off/on` x3, sole change = the three Textual clock arms neutralised:

| arm | idle % of a core (median of 6) | timer fires / 15 s |
|---|---|---|
| shipped | **0.616** (0.574-0.667) | **1018** (67.9/s) |
| framework clocks off | **0.076** (0.071-0.079) | **58** (3.9/s) |

Mechanism: `Widgets/ModelArtifacts/install_progress.py:120-126` composes `ProgressBar(total=None)`
then sets `bar.display = False`. `textual/_progress_bar.py:92-97` arms the 15 Hz refresh; `:287-289`
also arms an unconditional `set_interval(1, self.update)`. `textual/dom.py:554-563` gates the
repaint on `is_on_screen`, not the timer. Personas carries a `LoadingIndicator` at **16 Hz** inside a
permanently `display:none` widget.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
