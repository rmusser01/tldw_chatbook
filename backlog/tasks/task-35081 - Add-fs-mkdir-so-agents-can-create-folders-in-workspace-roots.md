---
id: TASK-35081
title: Add fs_mkdir so agents can create folders in workspace roots
status: Done
created_date: 2026-10-16 02:16
priority: high
updated_date: 2026-10-16 03:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report: the model could not create a folder anywhere. The fs_* family has no directory-creation tool: fs_write/fs_edit/fs_patch require an existing parent (spec 2026-08-04 section 2 deliberate divergence), and the only mkdir-capable builtin (family-B write_file create_directories) is gated off by default and scratch/binding-confined. Even a fully-authorized read-write workspace binding cannot create a folder, so the model errors or hallucinates. Add fs_mkdir to the fs_* family across the pinned-worker stack: wire op, local core, worker dispatch, executor admission (local and remote), provider spec with mutates gating, regenerated remote worker bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] fs_mkdir is advertised alongside fs_write for runs with a writable admitted root, and stripped for read-only roots and root-less runs like other mutating path tools
- [x] fs_mkdir creates a directory including missing parents inside the admitted root, confined and denylist-guarded identically to fs_write (sensitive paths, .git metadata write, new-directory-chain refusal applied to the target itself)
- [x] Creating an existing directory is a success no-op; a non-directory at the target path is refused
- [x] Remote (SSH) roots support fs_mkdir via the regenerated worker bundle
- [x] Tests cover core, wire schema, dispatch, executor admission, provider advertising and end-to-end invoke; targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add create_directory core + _mkdir_relative_directory to Tools/local_tool_impls.py (mkdir parents, exist_ok dir no-op, refuse file target, denylist via write-intent choke point)\n2. Add fs_mkdir wire op to Tools/workspace_wire_decode.py (operations, write set, argument schema)\n3. Dispatch branch in Tools/workspace_tool_dispatch.py (directory-classified target check)\n4. Executor admission in Tools/workspace_tool_executor.py local and remote builders (mutation branch)\n5. Provider spec + _PATH_AUTHORITY_LOCAL_NAMES + path-target mapping in Agents/local_tool_provider.py (MUTATES_LOCAL, tags mutates)\n6. Regenerate remote worker bundle and run its drift guard\n7. Tests: core guards, wire schema, dispatch, executor, provider advertising/e2e; spec table row in Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md\n\nADR required: no\nADR path: N/A\nReason: direct implementation of the existing fs_* family (ADR-032) following the fs_patch precedent; spec section 2 table gains a row
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
### Approach

fs_mkdir rides the family's existing machinery end to end rather than a parallel path:

- Wire: `workspace_wire_decode.py` gains the operation in WORKSPACE_OPERATIONS, WORKSPACE_WRITE_OPERATIONS and ARGUMENT_SCHEMAS (path + sensitive_exclusions); `workspace_tool_protocol.py` mirrors the Literal. Both decoders accept it with no protocol-version change (the bundle is laptop-built and shipped per session, so both ends move together).
- Core: `local_tool_impls.create_directory()` (resolve via the write-intent choke point, then `refuses_new_directory_chain` applied to the TARGET itself — mkdir's leaf is a new directory, unlike a file write) and `_mkdir_relative_directory()` (directory-classified exclusion check via `_relative_target_is_safe(is_directory=True)`, non-directory-at-target refusal, exist_ok no-op, `mkdir(parents=True)`).
- Worker: `workspace_tool_dispatch.py` `fs_mkdir` branch + `_request_directory_target` (the directory twin of `_request_mutation_path`).
- Executor: both `_build_request` (local, with the extra chain guard on the resolved target) and `_build_remote_request` (remote, lexical write-intent normalization) admit it in the mutation branch.
- Transport: `remote_workspace_executor.py` adds fs_mkdir to `_EXCLUSION_CARRYING_OPERATIONS` (its wire schema carries sensitive_exclusions).
- Provider: spec with `MUTATES_LOCAL` + `tags=('mutates',)` + CONSOLE_AND_EXTERNAL_MCP exposure; added to `_PATH_AUTHORITY_LOCAL_NAMES` (root_alias routing, stripped with no roots); directory-kind path-target mapping in both the remote-lexical and local-resolved preflight branches. No CAS/read-ledger participation (mkdir never changes file contents the model has read).

### Deliberate contract reversal, with the guard re-sited

The old `test_fs_family_never_creates_directories_on_the_agents_behalf` tripwire existed to force exactly this re-examination ("If a mkdir ever appears here, that decision is being reversed and the guard's placement has to be re-examined"). TASK-35081 reverses the decision (the incident: a model asked to "create a folder" had no honest path to success and confabulated) and re-sites the guard; the tripwire is rewritten to pin the new contract: `_mkdir_relative_directory` is the module's only directory creator, `create_directory` must call `refuses_new_directory_chain`, and `patch_tool_impls` stays mkdir-free. Spec 2026-08-04 section 2's table gains a dated fs_mkdir row; the parent-must-exist rule still governs every file writer.

### Modified files

`tldw_chatbook/Tools/{workspace_wire_decode,workspace_tool_protocol,local_tool_impls,workspace_tool_dispatch,workspace_tool_executor,remote_workspace_executor,remote_worker_bundle}.py`, `tldw_chatbook/Agents/local_tool_provider.py`; tests in `Tests/Tools/{test_local_tool_impls,test_local_tool_sensitive_paths,test_workspace_wire_decode,test_workspace_tool_protocol,test_wire_conformance,test_workspace_tool_executor}.py` and `Tests/Agents/test_local_tool_provider.py`; docs: spec table row + `Docs/Architecture/console-file-authority.md` "Directory creation" section.

### Verification

Tools-layer sweep 599 passed; loopback (real regenerated bundle over the ssh-shaped transport) + note-contract 45 passed; executor module 140 passed. Provider module: the 3 new tests fail only under this machine's pre-existing environment breakage (193 dev-HEAD failures in the same module, `RecoveryRequired('raw_source_selection_changed')` from local backup-recovery state); their exact logic verified green in an inline run through the real worker (create+retry via admitted rw root; stripped for ro-only; blocked on ro alias; stripped root-less). Bundle drift guard 10 passed. Remote bundle regenerated via `python -m tldw_chatbook.Tools.build_remote_worker_bundle`. ADR: none required (documented in plan).

Implemented fs_mkdir across wire/protocol/core/dispatch/executor/transport/provider with the TASK-849 guard re-sited to the mkdir target; regenerated the remote worker bundle; tests and docs updated; the old no-mkdir tripwire rewritten to pin the new single-creator contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
