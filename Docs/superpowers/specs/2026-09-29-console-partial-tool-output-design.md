# Console partial tool output

Task: TASK-33562. Decision: [ADR-205](../../../backlog/decisions/205-console-partial-tool-output.md).

Users see flushed skill-script stdout/stderr and native MCP progress messages before a call completes, in its current primary Console row. Preview remains three wrapped lines; disclosure exposes the retained text and arguments. MCP progress is labeled honestly. Final-only tools behave as before.

Updates are bounded to 16,000 characters and coalesced to ten per second, with a best-effort final flush bounded to 100ms. Per-execution identity rejects stale output even when a provider reuses a call ID. UTF-8 is decoded incrementally, and child pipes keep draining after the retention cap. Observer failures cannot affect execution or cleanup. Existing approval, cancellation, process containment, and final-result authority remain in force.

Partial text is display-projected and session-only. It never reaches durable trace, capture, audit, or model history. Final results replace partial previews; interrupted rows retain captured text with an honest label. There are no new dependencies, configuration, settings, schema changes, or keybindings. UI uses existing ADR-150 tokens and disclosure components.
