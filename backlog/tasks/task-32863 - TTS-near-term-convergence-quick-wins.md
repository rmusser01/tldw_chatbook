---
id: TASK-32863
title: TTS near-term convergence — dead isolated engine and voice-manager base adoption
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TTS/ is 92 files / 66k lines around 7 engine families; full convergence is governed by ADR-023's staged bridge (native adapter per provider with explicit removal criteria — it deliberately rejected an immediate rewrite). This task is only the two near-term quick wins that need no staging decision:

1. `TTS/backends/chatterbox_isolated.py` (220 LOC) is dead — zero references outside its own file, absent from the sealed registry and tests. Delete it.
2. `HiggsVoiceProfileManager` (681 LOC) does not inherit `VoiceManagerBase` while `ChatterboxVoiceManager` (451) does; both implement the same ~10-method profile CRUD (create/update/delete/list/get/export/import/validate). Making Higgs adopt the base absorbs ~400 LOC.

Everything else (per-engine credential chains, HTTP error mapping, voice listing over the thin `base_backends.py` 202-line base) stays governed by ADR-023 and is NOT in scope here. ADR required: no — deletion plus base adoption inside an existing ADR's boundary.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.


## Tier-2 scope correction (2026-09-21)

**AC#1 is already satisfied on `origin/dev`** -- verify before doing the work.

**AC#2 as written would remove a safety property.** Converging the two backends on the stated shape
**deletes Higgs's 300 s duration cap**, which has no equivalent on the other side. Re-word AC#2 to
converge *toward* the stricter of the two, not toward either one arbitrarily.

Measured by the tier-2 review against `origin/dev d0face3ebe`; evidence in `qa/tier2-code-review-2026-09-21/` (`report.md`, `validation/`). Tracked as TASK-32903.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `chatterbox_isolated.py` deleted with evidence of zero references recorded in the notes
- [x] #2 `HiggsVoiceProfileManager` adopts `VoiceManagerBase`; the parallel CRUD exists once; higgs voice suites pass
- [x] #3 No native-adapter migrations are forced by this task (ADR-023 staging untouched)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Both quick wins already landed on `dev`; this task was stale as filed and is
ticked here with the evidence, by the tier-2 code review (TASK-32901, slice
S03/S04) rather than by new work.

- **AC#1** — `ls tldw_chatbook/TTS/backends/ | grep -i isolated` → no match.
  `chatterbox_isolated.py` is absent from the worktree; `grep -rn
  chatterbox_isolated tldw_chatbook/ Tests/` → zero hits.
- **AC#2** — `HiggsVoiceProfileManager(VoiceManagerBase)` at
  `TTS/backends/higgs_voice_manager.py:52`, and its class docstring cites this
  task. The parallel CRUD exists once: `load_profiles`, `save_profiles`,
  `get_profile`, `update_profile` and `delete_profile` are defined only on
  `VoiceManagerBase`; Higgs now defines only its engine-specific
  `create_profile`/`list_profiles`/`export_profile`/`import_profile` plus its
  own validator and backup hooks. The review found AC#1 satisfied and filed
  AC#2 as outstanding — **that half of the finding was already stale too**.
- **AC#3** — no adapter-registry or `legacy_bridge` route changed.

**Correction to the review's S03/S04 delta note on AC#2.** It warned that
adopting the base would *drop* Higgs's 300 s duration cap because the base had
no bounds, and proposed `TTS/sample_audio_validation.py` as the adoption
target. Higgs kept its own `_validate_audio_file` override, so no cap was
lost. `sample_audio_validation` is the wrong target regardless: its
`MAX_PLAYABLE_AUDIO_BYTES` is 8 MiB and it bounds audio this app *generated*,
so pointing reference-audio validation at it would reject an ordinary
few-minute user recording. The base instead gained a
`max_reference_audio_bytes` bound (100 MB, matching Higgs's own) under
TASK-32901, which is what `ChatterboxVoiceManager` was missing.
<!-- SECTION:NOTES:END -->
