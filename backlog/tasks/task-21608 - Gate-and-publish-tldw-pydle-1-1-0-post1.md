---
id: TASK-21608
title: Gate and publish tldw-pydle 1.1.0.post1
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - release
  - ci
  - supply-chain
dependencies:
  - TASK-21601
  - TASK-21602
  - TASK-21603
  - TASK-21604
  - TASK-21605
  - TASK-21606
  - TASK-21607
references:
  - backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md
  - backlog/decisions/149-network-chat-handoff-reliability-amendments.md
documentation:
  - Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md
  - Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md
  - Docs/superpowers/specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md
priority: high
type: task
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the completed fork changes into a reproducible, independently installable
release proven against deterministic protocol, TLS, AgentIRC, and Ergo targets
before any Chatbook task is allowed to pin the real dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pull-request CI runs targeted unit, deterministic protocol, TLS, privacy, resource-bound, lifecycle, namespace, license, wheel, and sdist gates on Python 3.11, 3.12, 3.13, and 3.14.
- [ ] #2 The exact built wheel—not an editable checkout—passes the deterministic oracle and two-client compatibility runs against an exact AgentIRC version and an exact Ergo release whose archive checksum is verified.
- [ ] #3 Normal/cooperative shutdown leaves zero owned work across every compatibility target, while the cancellation-resistant callback case returns the required bounded incomplete outcome with no remaining protocol authority.
- [ ] #4 Release CI builds once from a reviewed tag, independently inspects and installs wheel and sdist, and proves package namespace, Python floor, LICENSE, NOTICE, upstream base, source URLs, and patch inventory.
- [ ] #5 Third-party actions are pinned by full commit SHA, workflow permissions are minimal, pull-request jobs receive no publishing credentials, and no `pull_request_target` path can execute untrusted release code.
- [ ] #6 The release publishes SHA-256 checksums plus provenance or an SBOM, and its notes map every downstream commit to the ADR-148 patch taxonomy and exact Codeberg base.
- [ ] #7 Version `1.1.0.post1` is published through an approved GitHub release and PyPI Trusted Publishing configuration, with externally retrievable artifacts matching the tested hashes. An external blocker is recorded with verified local artifacts but leaves this criterion unchecked and the task unfinished.
- [ ] #8 A separate scheduled compatibility job checks latest stable upstream and Ergo without making those moving targets rewrite the pinned release result.
- [ ] #9 No Chatbook runtime dependency, screen, profile schema, credential store, or transcript persistence is introduced by this fork release task.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
