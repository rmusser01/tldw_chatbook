# TASK-13206 clone voice bundle portability UAT

Date: 2026-08-12
Result: Partial — audible dependency gate pending

This artifact records sanitized engineering evidence only. It contains no
audio, transcript, source/bundle/staging path, checksum, credential, provider
configuration or origin, generated configuration, or private runtime value.

## Environment

- Working-tree base revision: `e160d1792` plus the uncommitted TASK-13206
  verification changes described here.
- Host: macOS 15.6, Darwin 24.6.0, arm64.
- Python: 3.12.11.
- Profile schema: v4.
- Ordinary reference-bearing export: sanitized wire v2.
- Test recipe: model `uat-clone-model`, revision 1.

## Isolation

The deterministic two-launch harness used two fresh, independent sets of
temporary `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`,
`[paths].data_dir`, profile-store, model-package, generated-config, and runtime
roots. Environment variables were set before application imports. The
developer profile database was not opened. The harness removed both temporary
environments after teardown.

## Observations

### Launch A

- Created a schema-v4 clone profile and canonical reference through the real
  profile repository and portability service.
- Ordinary export produced sanitized wire v2 with the reference omitted.
- The warning/acknowledgement gate preceded bundle publication.
- Bundle publication succeeded without creating an assignment.

Sanitized result:

```text
launch=A schema=4 sanitized_wire=2 recipe_revision=1
model=uat-clone-model bundle_published=true assignments=0
```

### Launch B and restart

- Began with an independent empty profile/model/runtime environment and no
  visibility into Launch A's configured dependency.
- Imported the validated bundle with the explicit Create inactive resolution.
- Reopened the independent repository and retained the schema-v4 profile and
  exact recipe/model requirement.
- Dependency state remained `missing`; the stored profile projected
  `Needs compatible model`.
- Import did not create an assignment, change a default, or leave an owned
  staging/output artifact after shutdown.

Sanitized result:

```text
launch=B-restart schema=4 recipe_revision=1 model=uat-clone-model
dependency_before_configuration=missing stored_status="Needs compatible model"
assignments=0 default_changed=false owned_residue=0
```

## Automated verification at partial closeout

- Six focused old-reader, rollback, shutdown, runtime-privacy, UI-privacy, and
  ownership regressions passed.
- The five directly affected test modules produced 406 passes and one existing
  UI mount-timing failure. The failing test passed immediately in isolation
  and together with its preceding lifecycle test; no speculative product or
  test change was made. This is recorded as a non-green broader run, not a
  pass.
- Ruff check, the planned six-file Ruff format check, scoped mypy for 17 source
  files, CSS bundle synchronization, and `git diff --check` passed. The
  normalized legacy mypy comparison contained zero new diagnostics.

## Pending audible gate

This run did **not** configure an exact pre-provisioned audio.cpp clone model,
launch audio.cpp, generate speech, or play audio. No compatible task-owned
model package was provisioned in either isolated environment, and no human was
available to perform and confirm playback. Therefore this evidence does not
satisfy the audible portion of acceptance criterion 7, and TASK-13206 remains
In Progress.

To complete the gate, a user must start the app with the same two-launch
isolation, explicitly configure the exact pre-provisioned dependency in Launch
B after restart, refresh the inactive profile to available, generate and play
speech with that imported profile, and confirm whether the expected voice was
audible. Record only the yes/no human result and the safe metadata above; do
not retain the audio, transcript, local paths, or checksums in this artifact.
