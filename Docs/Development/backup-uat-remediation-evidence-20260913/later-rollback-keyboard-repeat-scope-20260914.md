# Keyboard continuation across a new installation root

Read-only assessment; no application launch, native inventory, live SQLite, recovery, or fixture mutation. The preserved d308 fixture is `/private/tmp/chatbook-backup-final-uat-20260914-ssc_w6dr`.

## Finding

A new immutable wheel at a different private installation root is a storage-location change for this bound profile. Its persisted profile has 23 roots; the two outside its disposable power profile are the original installation's `tldw_chatbook/Evals/config/eval_config.yaml` and `tldw_chatbook/assets`. Neither a new installed Evals file nor new installed assets directory is inside those roots. `storage_admission._scope` (769–827) still requires each operation path inside the verified saved roots, and `_contains_owned_path` (620 onward) requires actual object identity or declared directory descent. There is no implicit enrollment of a newly located installation.

The installed default Evals path derives from `Evals/__init__.py::_default_config_path` and the builtin Persona assets root derives from `Persona_Visual/recovery.py::_Assets._root`, both using the executing module location. Consequently those owner declarations change with the installation root. This is code/metadata evidence, not an executed claim about which ordinary startup operation fails first.

The committed forward plan also preserves the old canonical Evals item without safety-copy coverage. In a later local-snapshot generation, `Evals/recovery.py` reconstructs preserved originals; its exact-current-canonical exception at lines 452–455 will no longer match that old path. Such an uncovered extra is refused at lines 469–470. The preserved-only branch runs for a local snapshot, so this is a subsequent-generation constraint, not proof of an immediate forward-generation exception.

## Recommendation

For a corrected wheel at a new immutable root, initialize a fresh disposable profile there and repeat the approved keyboard journey using the same verified incoming archive. Keep d308 as historical evidence. No binding/activation/plan rewriting, symlink overlays, or path aliases are justified.

A conventional in-place upgrade retaining the identical installed path could retain the canonical locations, but overwriting that directory conflicts with the explicit requirement that the d308 installed bytes remain untouched. It is not the proposed continuation and has not been performed. Starting only a new profile at a different install root while silently inheriting the old authority would not be a valid substitute.
