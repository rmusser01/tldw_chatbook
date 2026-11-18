# Mounted upstream FD warning

The two BaseAppScreen owners passed 43 cases at 9ce98df2e7. The existing sentinel reported 206 additional descriptors (14→220, limit200). Exact output is in task-6-latestdev-baseapp.log. No filters, thresholds, GC settings or markers changed.

Bounded source inspection: the mounted-route test constructs `_Host(_build_test_app(), screen_class)` and runs the host's lifecycle. It does not run or explicitly dispose the separate TldwCli `app_instance`. That is a plausible owner for retained resources; descriptor types and exact retained handles were not measured, so this is an inference, not a confirmed leak diagnosis. The UI autouse cleanup already unfreezes/collects per test, making a new GC marker unjustified. Lessons from TASK-31392 and TASK-32679 require native-owner diagnosis before any cleanup change.

Both owner files and BaseAppScreen are exact pinned upstream blobs. No functional assertion failed. Per root's scope decision, preserve the warning for independent review; no general leak repair or broad sweep is authorized solely by this warning.
