# Lessons: Textual input delivery

Read alongside [the general testing evidence lessons](lessons-testing-evidence.md).

## Scratch UI diagnostics need the existing fixture plugin and an actual input-delivery witness

**Post-PR3207 book audit, 2026-10-22.** Two scratch audiobook probes outside
`Tests/` failed private-profile admission with `unsafe_parent: missing_parent`.
The decorator alone did not load the bootstrap fixtures. A scratch-local
`conftest.py` with `pytest_plugins = ["Tests.conftest"]` admitted the existing
native private profile without weakening any storage or ACL check. Those first
runs are setup failures, not product evidence.

The same audit then treated `Pilot.press()` task completion as key delivery
while a parent widget callback was deliberately held. Both controls failed that
oracle, although their child TextAreas already contained the typed `x`.
Pilot was still waiting for pump settlement. The qualified probe instead
witnessed the actual live child text before releasing the callback: the normal
import control passed and the later-edit retention assertion failed, with real
read/Generate content recorded. When holding a pump, prove the specific child
input was delivered rather than requiring the whole Pilot operation to settle.
