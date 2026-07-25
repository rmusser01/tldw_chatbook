---
id: TASK-584
title: Retain and expose skill script output files
status: To Do
assignee: []
created_date: '2026-07-25 15:05'
labels:
  - skills
  - agents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Skill script execution returns only stdout, stderr and an exit code. The per-run scratch working directory is deleted afterwards, so any file a script produces is discarded.

That was the deliberate v1 choice: it keeps the run fully bounded and avoids inventing a retention story before there was a consumer for one. But it blocks a whole class of legitimate skills — extract-then-process pipelines, format converters, report or chart generators — whose entire purpose is to produce an artifact.

The reason this is not a small change is that there is no consumer today. The Agents runtime has no general file-read tool, and `skill_file` is deliberately contained to a skill's own bundle (and, after the trust-manifest work, to fingerprinted files only). So "keep the output" immediately raises: who reads it, through what contained seam, and with what lifetime?

Sketch of the options to weigh: persist the scratch directory under the existing tool-sandbox root and merely *list* produced files in the tool result (the user can open them, the agent cannot read them); or add a bounded read-back path so the agent can consume its own output — a new read surface that needs its own containment, size caps, and probably its own trust story. Retention/cleanup policy is required either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on whether the agent can read produced files, or only the user can, with the reasoning
- [ ] #2 Files a script produces survive the run and are reachable at a documented location
- [ ] #3 Retention and cleanup are defined and enforced, so output cannot accumulate without bound
- [ ] #4 If the agent can read output, that read path is contained and size-capped, and cannot reach outside the run's own output directory
- [ ] #5 Produced files cannot be written into a skill's own bundle, so a run cannot invalidate its own fingerprints or plant trust-invisible content
- [ ] #6 The tool result reports what was produced without dumping file contents into the transcript
- [ ] #7 Docs/Features/Skills-Script-Execution.md is updated — it currently states plainly that produced files are discarded
<!-- AC:END -->
