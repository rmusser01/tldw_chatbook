# Scoped CI review

Read-only review against98bb849eb6 covered only the UI test repair, eval watchdog,
and five-file Windows observer/workflow delta. No remaining actionable findings.

UI waits require the intended row/status and preserve the existing30s deadline and
full recovery assertions. The eval watchdog retains evidence within the unchanged
45s child deadline. The Windows observer uses original code objects, bounds its
metadata and executes the selected case once with the original180s limit and exit
status. Existing60s keyboard defaults remain. The final shell control compares
Bash PWD rather than a host-dependent Python path spelling.

The reviewer did not repeat tests, make edits, run CI or call providers. Windows
causality and native validation remain pending; diagnostic overhead is possible.
These additions collect evidence and are not claimed as Windows production fixes.
