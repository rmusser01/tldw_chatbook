---
id: TASK-42022
title: "A file root a restore retired stays in the profile binding"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:56
labels:
- follow-up
- backup
priority: low
references:
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/config_participants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42000's review (round 17). Capture registers each config-folder file as its own root; when a Replace retires the file, the root stays registered. TASK-42000 made containment and data-dir association treat an absent root as holding nothing, but a stale root still disables `_selector_evidence` reuse for the profile and reaches containment checks.

**2026-11-20, coordinator decision:** no root retirement. The registry keeps root history by design, and on a separate config folder the measured cost of the stale root is the documented loss of `_selector_evidence` reuse. The criteria below replace the original two: they record why publication keeps the root, pin that storage access still works after the Replace, and settle whether a default-location profile can reach the same state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record why publication keeps a retired file root: the registry and profile-record invariants, and the crash-safety gap
- [x] #2 A test shows that after a Replace retires a config-folder file root on a separate config folder, storage access for that profile still works: admission, containment and a normal read
- [x] #3 Whether a default-location profile (config folder holding the recovery bootstrap) can reach a retired, absent file root through a real Replace is established with evidence. If it can, an absent file root is treated as holding nothing where Admission builds its groups and tokens, so storage access does not raise FileNotFoundError, with a test red before the fix. If it cannot, the notes show the refusal path that prevents it
<!-- AC:END -->

## Implementation Plan

1. Keep the investigation that led to the decision as the notes for #1.
2. #2: in `test_created_owner_rollback.py`'s `[pets-none]` case (separate config folder; the Replace retires the target's `tamagotchi_pets.json`, which has its own root), the post-restore pet start also checks that the retired root is still bound and absent, admits the pet path with `acquire_storage`, checks containment over the profile's declared roots, and reads through the pet store. Show it red under a temporary removal of the TASK-42000 absent-root guard in `_contains_owned_path`.
3. #3: run the same `[pets-none]` Replace with the config at the default location and the default control root (temporary probe), plus a control that retires nothing. Record either the refusal path or, if the state is reachable, fix `Admission._groups`/`_tokens` with a red-first test.

## Implementation Notes

**Summary.** Publication keeps a retired file root (coordinator decision, 2026-11-20). On a separate config folder the profile's storage still works after such a Replace, and a test now pins that. A default-location Replace cannot reach the state: it refuses at activation and rolls back (the backup-then-rotation path is TASK-42114). No product code changed.

**#1: why publication keeps the root.** The source of the root: in the Replace's first binding, `replacement._ensure_first_bindings` calls `capture_service._capture_names`, which registers each existing config-folder file outside `_CONFIG_CAPTURE_LEAVES` as its own `backup.source.<sha>` namespace and binds them. After `[pets-none]`'s Replace, a temporary probe showed the profile's namespaces include the config folder, `config.toml`, `tamagotchi_pets.seeded` and the retired, absent `tamagotchi_pets.json`.
- **Profile-record invariant.** A profile's `roots` must equal the union of its namespaces' registry roots (`bootstrap._binding`: `binding_mapping_changed`). Dropping the root therefore means dropping its namespace from the profile.
- **Registry invariants.**
  - The registry keeps a namespace for good: `Admission` has no unregister, `remap` keeps historical tokens ("no automatic stale-evidence cleanup"), and `_tokens(())` refuses empty roots.
  - An absent root that no profile references is refused: `effective_roots.check_redundant_profiles` raises `FileNotFoundError("redundant_root_profile_required")`, and `Admission._groups` runs that check on every admission while an absent alias is proved (with no pending or proposed entry and no recovery journal). `test_unreferenced_missing_alias_is_not_inferred_owned` pins this.
  - A temporary in-process probe (deleted) bound a directory namespace and a file namespace, then unlinked the file; `authority.normal(("parent",))` still worked. After the profile record was rewritten without the file's namespace, as `_bind_activation` would write it, `check_redundant_profiles` and `authority.normal(("parent",))` both raised `redundant_root_profile_required`.
  - So a binding-only drop would fail `publication._activation_proof` right after the binding write, and every later storage admission of the profile.
- **Crash-safety gap.** A real drop must remove the namespace from the registry in the same crash-safe step as the profile rewrite, and neither order is safe:
  - Registry first: `_binding` raises `binding_authority_missing`, and `_checked_activation_pair` reads `registry[scope]` for the before-namespaces.
  - Profile first: the failure above.
- **Cost of keeping it.** On a separate config folder, the enrolled folder covers the absent root, so `effective_roots` omits it and containment treats it as holding nothing (TASK-42000). `test_deleted_redundant_alias_keeps_raw_binding_and_native_group` pins that shape. The remaining cost is the documented loss of `_selector_evidence` reuse ("absence-proved roots are never reused").

**#2: storage access after the Replace, on a separate config folder.** In `test_created_owner_rollback.py[pets-none]`, the post-restore pet start (a fresh child process) now:
- records the profile's declared roots that are absent before the store runs, and the test asserts they are exactly the retired `tamagotchi_pets.json`;
- checks that the retired root itself holds nothing (`_contains_owned_path`, asserted `False` before the store can recreate the file);
- checks containment of the pet path over the declared roots (`_contains_capture_path`, asserted `True`);
- admits the pet path (`acquire_storage`);
- reads through `ConfigFileStorage().list_pets()` (asserted `[]`, as before).

Evidence (`-p no:xdist --timeout=1500`):
- Green: 1 passed in 133 s at load 23-36, through the later rollback.
- Red under a temporary source edit that removed the TASK-42000 absent-root guard in `storage_admission._contains_owned_path` (reverted with `git checkout` in the same command): the pet start failed with `RecoveryRequired: storage_admission_unavailable` from `raw.scope` → `acquire_storage` (65 s).
- The `_contains_capture_path` check passed even under that edit: the enrolled config folder sorts first, so `any()` never reached the retired root. After review rev1, the child also checks `_contains_owned_path(<retired root>, pets) is False` for each absent root. Under the same temporary edit that check raised `FileNotFoundError` from `resolve(strict=True)` at that line (51-58 s, `-n 4`, load 12-18); with the guard back, `-k pets -n 4` passed all five `pets-*` cases (274 s, load 14-23), and `[pets-none]` alone passed in 100 s (load 12).

**#3: a default-location Replace cannot reach the state; it refuses (the backup-then-rotation path is TASK-42114).** Temporary probes ran the same harness with the config at `~/.config/tldw_cli/config.toml` and `RecoveryService(default_control_root())`, as the F9 default-profile workflow uses.
- `[pets-none]`, which retires the target's own `tamagotchi_pets.json`: the restore failed with `backup_operation_failed`. A wrapper on `publication.finalize_candidate` showed the path:
  - `finalize_candidate` → `activation.bind_activation` → `control_records._bind_activation` raises `ValueError("activation_profile_outside_maintenance")` (line 601).
  - `replacement.replace` catches it after publication, runs `_recover_activation_pairs` and `_rollback_replacement`, and raises `replacement_rolled_back`.
  - Why: the default config folder holds `recovery-bootstrap`, so it is never a root (`bind_profile`: `control_root_overlaps_target`). The retired file's root is absent and no enrolled directory covers it, so `effective_roots` keeps it. In `_bind_activation`, `_contains_capture_path(session._roots, path)` is then False for that root, because an absent root holds nothing.
- Control: the same case with the target's pet file left out (nothing retired) completed, and the profile had no absent root.
- So no default-location profile ends a **Replace** holding a retired, absent file root, and for the Replace path `Admission._groups`/`_tokens` needs no change. The conclusion is limited to Replace. A plain backup registers each captured config-folder file as its own root, dated pet copies included; when the pet store's rotation later deletes such a copy, `_tokens`' `resolve(strict=True)` raises on every admission and the app refuses to start. That backup-then-rotation path is TASK-42114, fixed in another stream. An earlier probe that kept the harness's own control root refused sooner, at the first binding, with `replacement_config_container_unverified`; that probe was not representative.

**Found outside this task** (default location only; listed in the round report):
- the `[pets-none]` Replace itself fails and rolls back;
- `[pets-missing]` and `[pets-dated-only]` fail with `replacement_destination_parent_required`;
- after the control's successful Replace, the pet start fails with `RecoveryRequired: storage_scope_not_enrolled`, because the absent pet file is under no root.

**Files.** `Tests/Backup_Recovery/test_created_owner_rollback.py`; `backlog/docs/lessons-testing-evidence.md` (new entry on testing at the default location).
