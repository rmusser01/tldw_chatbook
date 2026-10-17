# Defer unused Personal Context interview exports

TASK-34406.9. Baseline actual PR3143 merge d395c8bf5e2d576d8fdb03fa5c3a663ce2f3af47. Bounded existing import closure change, authorized ongoing first-send profiling/reduction. Existing Personal Context architecture and ADR102 remain the behavioral specification; no new spec is necessary.

ADR required: no
ADR path: N/A; existing ADR097 and ADR102/126 apply.
Reason: house PEP562 export deferral preserves service/storage/authority/native boundaries and public identities.

## Global constraints

Only four interview export groups in Personal_Context/__init__.py change. Preserve exact __all__, public object identities, incidental module aliases and dir discoverability. Ordinary bootstrap/context/tool composition remains real and guarded; no change to service/keyring/native admission, cancellation, permissions, timeouts, worker ownership, budgets or dependencies. Root sole writer/runner; targeted tests only. Source/settings frozen during live work. Measurements diagnostic separately from balanced unprofiled first sends; retain failures and slower pairs.

## Review Focus

Public import compatibility before/after direct submodule import, star import, unknown attributes, discoverability without eager import, concurrent first attribute resolution, and import errors. Native bootstrap and enabled/disabled/absent/locked outcomes must remain original. No security check may move behind a lazy export or be bypassed. Matched evidence must retain exact27 ordered schemas/full settings/captured headers and actual observed cleanup, without claims about arbitrary worker retirement, native paint, p95 or Windows.

### Task 1: Defer interview-only initialization

**Files:** tldw_chatbook/Personal_Context/__init__.py; new Tests/Performance/test_personal_context_interview_import_closure.py; Personal Context architecture; task/QA notes.
**Interfaces:** original16 interview public symbols/four submodule aliases and39-name __all__; bootstrap service/core exports unchanged.

- [x] Write fresh-process causal tests using real bootstrap/context/tool composition for locked/absent/disabled/ready outcomes, plus public import compatibility controls. Run via admitted targeted_control.py. Expected: causal tests fail only because interview modules are already resident; compatibility controls pass at original production.
- [x] Defer exactly four interview groups with existing importlib/__getattr__ cached-object pattern and side-effect-free __dir__; all core exports remain eager. Expected: causal and compatibility tests pass.
- [x] Run targeted native/bootstrap/key/service/context/interview/Console tests and touched lint/source checks. Expected: no introduced behavior regressions; preserve original failures with exact-source attribution if encountered.
- [x] Commit and run one fresh whole-branch independent review. Derive/freeze literal-only reuse of the reviewed eight-run unprofiled harness at exact baseline/candidate. Expected: all helper bodies/guards preserved; settle findings before launch.
- [x] Run eight balanced fresh first sends B/C/C/B/B/C/C/B; independently qualify exact payload/settings/trace/cleanup/source equality and report every timing, including adverse pairs. Expected: valid comparison, not assumed speed win. Update notes; keep parent target open unless demonstrated.

Pre-flight: one task; no shared task interfaces. Diagnostic protocol deviation retained: initial system-Python qualification lacked tomllib and missing fail-fast ran the analyzer early; subsequent config comparison confused expanded settings and the plan subset, then sandbox refused ps. Root independent full qualification passed before this design decision. Raw analysis/failed outputs retained; no retry/replay/provider request or artifact overwrite. This instrumented run is not speed acceptance.

Outcome: independent review0/0/0 and all8 functional/source/settings/cleanup packets qualified. Post-save median906.5->933ms (two pairs improve), Enter-toHTTP1.951982->1.808338s (three improve). No post-save improvement established; keep candidate unmerged. Full report Docs/QA/2026-10-17-personal-context-interview-first-send.md. Root qualifier r1 wrongly required user-first database rows; r2 preserves exact role/content equality for timestamp ties and leaves every other gate unchanged, without retry/replay.
