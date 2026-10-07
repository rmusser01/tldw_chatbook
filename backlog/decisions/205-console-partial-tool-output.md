# ADR-205: Session-only partial Console tool output

Date: 2026-09-29
Status: Accepted
Task: TASK-33562
Amends: [ADR-195](195-console-live-tool-call-presentation.md)

## Decision

Primary Console calls may publish optional text while executing. Skill scripts publish stdout/stderr; native MCP stdio calls publish standard progress notifications, labeled as progress rather than result chunks. Other tools keep their final-result behavior. Existing three-line previews and expandable details update in place.

A context-bound, best-effort output observer crosses the existing service worker boundary explicitly, without copying permission or storage contexts. Each execution owns a bounded accumulator (16,000 characters, at most ten refreshes per second). Closing the scope gives pending text at most 100ms of best-effort delivery grace and rejects late output, including output from abandoned workers. The runtime applies the catalog's display projection before delivery and correlates each update with the proposal step, so reused provider call IDs cannot receive earlier execution output.

Partial text is ephemeral: it never enters durable trace steps, trajectory records, provider history, audit metadata, or diagnostic logs. A final result replaces the preview through the existing result path. Run teardown retains any visible partial text with an honest interruption label; it does not assert a worker was killed. Output observation cannot grant permission, change budgets, or alter tool results.

## Alternatives

Writing chunks into the existing trace would retain unreviewed bodies and enlarge durable histories. Extending every provider invocation signature would change a stable provider contract for an optional display feature. Streaming every tool is impossible where a producer only returns a final result. These alternatives are rejected in favor of optional producer support and the existing display surface.
