---
id: TASK-32901
title: Fix fallback-data-root startup crash and umask-0775 app-directory creation
status: Done
assignee: []
created_date: '2026-09-22 15:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On umask-002 machines (fresh Ubuntu 24.04), the ADR-127 fallback data root engages when ~/.local/share is group-writable; ActorPackImportService then rejects its own profile_root because the fallback root .tldw_cli-data is a dotted base directory (hidden-base anti-bypass rule), crashing TldwCli.__init__. Separately, chat_dicts and the embeddings cache dir are created with umask-derived mkdir, landing 0775 on those machines; the config-participant admission layer then refuses them on every subsequent start. Fix both: app-owned actor-pack roots may live under a dotted base; app-owned directories under the data root are created/hardened via the private lifecycle (0700, self-healing).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ActorPackImportService accepts a profile_root under the dotted ADR-127 fallback root (.tldw_cli-data) without raising actor_pack_import_invalid,User-supplied archive paths still reject hidden bases (allow_hidden only for the two app-owned roots),chat_dicts and the embeddings cache under the data root are created 0700 via secure_private_directory, and a pre-existing 0775 instance no longer breaks the boot: the error log names the exact one-command chmod repair,A fresh umask-002 HOME with group-writable ~/.local/share boots twice with no 'Could not create chat dictionaries folder' error and no actor-pack validation failure,Targeted tests cover the importer flag, the service construction, and the two-run umask-002 regression; existing actor-pack/config/private-path suites still pass
<!-- AC:END -->


## Implementation Plan

ADR check: no new ADR — repairs two integration bugs inside the existing ADR-029/ADR-127 posture; the storage-admission side-effect boundary is respected, not weakened.

1. Reproduce on dev: umask 002 + group-writable ~/.local/share forces the ADR-127 fallback root; boot one creates chat_dicts 0775, boot two refuses it; `_absolute_path(profile_root)` rejects the dotted fallback root.
2. Fix the fatal crash: `_absolute_path` gains `allow_hidden` (default False); the two app-owned service roots (staging_root, profile_root) opt in; user-supplied archive paths stay strict.
3. Fix 0775 creation at the source: chat_dicts and the default embeddings cache use `secure_private_directory(create=True)` (0700); custom cache dirs pin `mkdir(mode=0o700)` without changing user ownership policy.
4. Pre-existing 0775 instances: the storage-admission pre-check refuses the target before any in-operation heal, and healing outside an admitted operation trips `raw_path_outside_scope` — so the error log now names the exact one-command repair (`chmod g-w,o-w -- <offender>`) instead of auto-healing.
5. Tests: unit (strict default preserved; allow_hidden accepts dotted app roots) + a three-boot umask-002 regression (clean create → poisoned survives with repair command → repaired boot fully clean).

## Implementation Notes

**Summary.** On umask-002 machines (fresh Ubuntu 24.04 report), the ADR-127 fallback data root engaged and the app crashed in `TldwCli.__init__`: `ActorPackImportService` validates its `profile_root` via `validate_path(path, path.parent)`, and the fallback root `~/.tldw_cli-data` is a dotted directory, so the hidden-base anti-bypass rule rejected the app's own root (`actor_pack_import_invalid`). Separately, `chat_dicts` and the embeddings cache were created with umask-derived `mkdir` — 0775 on those machines — which the storage-admission pre-check then refuses on every subsequent boot ("Could not create chat dictionaries folder ... shared_writable_parent").

**Files.** `tldw_chatbook/Actor_Packs/importer.py` (allow_hidden for app-owned roots only), `tldw_chatbook/config.py` (hardened creation at both mkdir sites; repair guidance in the PrivatePathError logs), `Tests/Actor_Packs/test_actor_pack_import_hidden_profile_root.py` (new), `Tests/App/test_umask002_fallback_root_boot.py` (new, three-boot regression).

**Deviation from plan.** Auto-healing a pre-existing 0775 instance was dropped deliberately: the admission system observes every private-path operation process-wide and requires them inside admitted operations (`raw_path_outside_scope`), while the admission pre-check refuses the shared-writable target before the operation body runs — a designed boundary this task must not weaken. Shipped behavior for poisoned machines: the boot survives (the error is non-fatal), the log names the offender and the exact `chmod g-w,o-w --` repair, and one manual run heals the machine permanently. The corresponding AC was updated to match.

**Verification.** All four new tests green (strict-default unit, allow-hidden unit, three-boot umask-002 regression incl. the user's exact crash path). Census green. Broader suites: no new failures vs pristine dev in this environment (34 with the change vs 36 baseline — same pre-existing hygiene/caching failures). mypy: importer.py clean; the 4 raw_participants errors pre-exist on dev.

## Review Remediation (PR #2808, Qodo)

All five Qodo findings addressed:

1. **Custom cache paths bypass validation** — the configured `model_cache_dir` now goes through `validate_path_simple` (probe off; the no-follow boundary below owns link handling) and must be absolute; an unusable value refuses loudly instead of silently relocating the cache (silent fallback would diverge from the config-participant binding for the route).
2. **Service wiring regressions evade tests** — added a construction test that builds the real `ActorPackImportService` (real `ActorPackRepository`) with both roots under a dotted fallback directory, pinning the `allow_hidden=True` call sites through the service boundary.
3. **Model cache branches ship untested** — new `Tests/Utils/test_model_cache_dir_lifecycle.py` drives `get_model_cache_dir` through the real config-file boundary in fresh interpreters: default dir 0700 under umask 002, custom dir (with whitespace) validated + 0700, relative value refuses without creating anything, and a poisoned 0775 instance names a shell-safe repair command (`shlex.split` round-trip asserted).
4. **Permission repairs fail for some errors** — the chmod hint is now reason-gated via `startup_errors.private_path_repair_hint` (only the shared-writable family); ownership refusals get a chown hint instead; other reasons keep the diagnostic without a misleading command.
5. **Copied repair commands can run path text** — the hint is emitted with `shlex.join`/`shlex.quote` so hostile paths stay one operand.

Baseline note: Tests/Actor_Packs has 28 pre-existing failures on pristine dev in this environment — identical count with this change; all new tests pass. Census green; mypy clean on the changed modules.

### Diagnostic inventory review (PR CI gate)

`Docs/security/production-diagnostic-inventory.json` regenerated after review: the delta is exactly the two intended new production diagnostics (config.py +2 calls — the chat_dicts and model-cache PrivatePathError handlers; path-privacy candidates +2: `_load_settings_uncached: error paths=[chat_dicts_folder]`, `get_model_cache_dir: error paths=[cache_path]`, both logging offender-derived repair hints). No other topology changes.
