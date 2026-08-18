---
id: TASK-17963
title: >-
  Shared fixed-name temp files race across app instances in skills store and
  trust store
status: To Do
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - skills
  - reliability
priority: medium
dependencies:
  - TASK-18705
---

## Description (the why)

TASK-18705's live verification and its own new prompt ledger
(`ProjectSkillsPromptLedger.record()` in
`tldw_chatbook/Skills_Interop/project_skills_prompt.py`) had to route around a
pre-existing atomic-write pattern used elsewhere in the skills subsystem: a
fixed, non-writer-unique temp filename for the atomic write-then-`replace()`
sequence. Two writes with the SAME fixed temp name racing across concurrent
writers means one writer's `temp_path.replace(path)` can consume the other's
still-being-written temp file out from under it (or the second writer's own
open/write can clobber the first's in-flight temp file), raising
`FileNotFoundError` or silently corrupting the write.

Two pre-existing sites still have this pattern and were deliberately left
untouched by TASK-18705 (out of that task's scope):

- `tldw_chatbook/Skills_Interop/local_skills_service.py:303`
  (`LocalSkillsService._save_index`, `temp_path =
  self.index_path.with_suffix(".json.tmp")`) — called by every skill
  create/import/edit/delete that touches the shared skills index, so two app
  instances (or two concurrent async callers in the same instance) mutating
  the skills store at close to the same moment can race on this one fixed
  path. The same file's `_write_text_atomic`/`_write_bytes_atomic` (lines
  ~316-329, `temp_path = path.with_name(f"{path.name}.tmp")`) have the
  identical shape, scoped per target file rather than per store, but still
  fixed-name and racy if two writers touch the same skill file concurrently.
- `tldw_chatbook/Skills_Interop/skill_trust_store.py:599` and `:611`
  (`_atomic_write_json`/`_atomic_write_bytes`, `temp_path =
  path.with_name(f".{path.name}.tmp")`) — the trust store's core write
  primitive, used for every trust mutation (bootstrap, approve, generation
  marker updates, encrypted snapshots).

This project's new ledger avoided the bug entirely by including the writer's
PID and thread id in its temp filename
(`project_prompts.json.<pid>.<tid>.tmp`) before the write-and-replace, per
its own inline comment explaining why. The two sites above should get the
same treatment.

## Acceptance Criteria (the what)

- [ ] `LocalSkillsService._save_index`'s temp file name includes a
      writer-unique component (PID + thread id, or equivalent) so two
      concurrent callers never share a temp path
- [ ] `LocalSkillsService._write_text_atomic`/`_write_bytes_atomic` get the
      same writer-unique temp naming
- [ ] `skill_trust_store.py`'s `_atomic_write_json`/`_atomic_write_bytes` get
      the same writer-unique temp naming, preserving the existing
      `_validated_trust_file_path` containment check on the (now
      writer-unique) temp path
- [ ] A test reproduces the race for at least one of the two modules (two
      concurrent writers to the same target path never raise
      `FileNotFoundError` and the final file is one writer's complete,
      valid content) and passes after the fix
- [ ] Existing `Tests/Skills/` suite remains green
