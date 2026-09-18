# Remaining workspace Persona findings

Read-only audit at `3e0c14de96`, followed by TASK-32775 native cold creation.
These are open repairs; the earlier field-edit and lifecycle qualification does
not close them.

1. `WorkspacePersonaPicker` uses valid string IDs `none` and `auto` for control
   choices. The real Persona service accepts both. In a production-CSS mounted
   Default modal, no-edit Apply changed saved Persona `none` into explicit None.
   Use non-string control sentinels and preserve every admitted string ID.
2. The picker requests only the default first page of 100 Personas. With 101 real
   records, the saved oldest Persona remained readable by ID but was labelled
   unavailable; older unselected Personas were omitted. Read every page and
   resolve a selected identity before describing it as unavailable.
3. Cold Settings → Workspaces → Create committed the workspace and opened its
   requested project-context interview, but automatic assistant defaults stayed
   unset. Native run001 logged `Workspace agent defaults could not be persisted`.
   The provisioner is wired after startup while the Tool Profile guard remains
   deferred until first feature use. Verify and repair the first-use dependency
   without weakening profile admission; preserve the committed workspace.

The first two findings used the real local Persona service and registry.
Four keyboard size/theme cells verified the Default modal's Tab ring, expanded
Select Escape and whole-record cancellation preservation. Eight additional
component cells checked acknowledgement, errors and actions. These component
probes are headless, not native-driver qualification.

The third finding is preserved in
[the failed native receipt](../qa/2026-09-18-settings-project-interview/failed-run001-native-result.json)
and its lifecycle receipt (exit 1, normal keyboard shutdown, absent PID, twelve
healthy databases and unchanged default fingerprints). TASK-32775 continues
interview qualification after explicitly opening Tool Profiles; that setup does
not qualify cold automatic Persona creation.

No whole-workstream completion claim is supported until these findings have
regressions, repairs and native evidence.
