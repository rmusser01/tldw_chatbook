---
id: task-32916
title: Pin explicit encoding on production text file IO
status: To Do
assignee: []
labels:
  - bug
  - windows
  - data-integrity
created_date: '2026-09-23'
---

## Description

`Path.read_text()` and `Path.write_text()` with no `encoding=` use the **locale**
encoding. That is UTF-8 on macOS and Linux and **cp1252** on a default Windows
install, so the same code path reads and writes different bytes depending on who
runs it.

This is not theoretical. It took down an entire nightly leg: both mermaid fixtures
in `Tests/Canvas/` are valid UTF-8 containing byte `0x81`, cp1252 has no mapping for
it, and because the reads sit at module scope the `windows-latest` leg logged
`collected 104236 items / 2 errors` → `Interrupted` → **zero tests executed**
(run 35838369448, 2026-09-23). Those two test files are fixed in #2818; the
production sites below are the same defect with a user on the other end of it.

The user-visible failure is a crash or silent corruption the moment any non-cp1252
character reaches one of these paths — an emoji, CJK text, a smart quote pasted from
a browser. The files affected are user data: voiceprints, meeting sessions, chat
dictionaries, the RAG context store, subscription site-config import/export.

A file written on Linux and opened on Windows (or a synced config, or a shared
export) breaks even when neither machine alone would.

**Deliberately not solved with `PYTHONUTF8=1`.** That would mask this class rather
than fix it, and the nightly's cp1252 posture is intentional (TASK-25706: exercise
the legacy Windows console, not the UTF-8 environment hosted runners supply).

## Sites

**13 `read_text()` with no encoding** — these are the dangerous half, since they
decode bytes the app did not write:

- `Utils/tls_trust.py:195` — concatenates CA bundles; a non-ASCII byte in a custom
  cert file breaks TLS trust setup
- `Subscriptions/site_config_manager.py:649` — `json.loads` of an imported config
- `Audio/meeting_session.py:259` — `json.loads` of a saved meeting
- `Audio/voiceprint.py:263,363,395` — voiceprint envelope + two label reads
- `TTS/backends/higgs.py:1029` — voice profile json
- `RAG_Search/eval/regression.py:245,387`, `RAG_Search/eval/gating.py:130`
- `Third_Party/aider/repomap.py:297` — vendored, confirm before touching
- (`UI/Evals/library_rail.py:130` and `Utils/paths.py:151` are comments, not calls)

**21 `write_text()` with no encoding.** Severity is uneven here and worth measuring
before fixing: `json.dumps` defaults to `ensure_ascii=True`, so the json writers emit
pure ASCII and round-trip safely under any locale. The ones that can actually emit
non-ASCII are `Evals/specialized_runners.py:362` (`write_text(test_code)`),
`RAG_Search/eval/gating.py:165` (`yaml.dump`), the five
`Utils/ui_responsiveness_artifacts.py` log writers, and
`Workspaces/change_tracking.py:379`.

## Acceptance Criteria

- [ ] Every non-vendored `read_text()` / `write_text()` call in `tldw_chatbook/`
      passes an explicit `encoding`
- [ ] The choice is justified per site rather than blanket `utf-8`: a file the app
      itself wrote is utf-8; a file supplied by the OS or another tool may not be
- [ ] Round-trip test: a string containing an emoji and a CJK character survives
      write→read for voiceprint labels, meeting sessions, and chat dictionaries
- [ ] A guard prevents regression — a check script or lint rule that fails on a new
      encoding-less text read/write in `tldw_chatbook/`, **seen red once on purpose**
      before it is trusted
- [ ] `Third_Party/aider/repomap.py` is explicitly ruled in or out as vendored code
