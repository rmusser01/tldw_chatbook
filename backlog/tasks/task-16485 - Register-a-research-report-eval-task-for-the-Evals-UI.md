---
id: TASK-16485
title: Register a research-report eval task for the Evals UI
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:39'
updated_date: '2026-08-16 03:45'
labels:
  - research
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The research self-eval runner is dispatched by category but has no task template or dataset, so it cannot be selected and run from the Evals UI - only the baseline script exercises it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A research_report template registers with the eval template manager pointing at a bundled verification-payload dataset,The task loads via the task loader with task_type research_report and metadata category research,The dataset samples carry verification payloads the runner scores (deterministic metrics on the bundled data),Tests cover template registration, task loading, and end-to-end scoring of the bundled samples through the runner
<!-- AC:END -->

## Implementation Notes

- Survey correction: `eval_templates.py` is shadowed by the `eval_templates/` PACKAGE (Python prefers the package), so the registration landed in dead code first -- reverted, and a proper `ResearchTemplates` category (`eval_templates/research.py`, BaseTemplates subclass) was added as the manager's seventh category instead.
- The `research_report` template points at a bundled dataset (`Evals/eval_datasets/research_report_verification.json`, three synthetic verification payloads: clean, degraded, gate-fallback) via an absolute path resolved from the module (the dataset loader only accepts existing paths). Samples load with `metadata["verification"]` exactly as the runner reads (the JSON loader maps whole items to metadata). Task loads via `TaskLoader.create_task_from_template("research_report")`.
- End-to-end test pins: template -> TaskConfig (task_type research_report, category research) -> EvalRunner dispatch -> ResearchReportRunner -> dataset (3 samples) -> per-sample metrics (clean 1.0 accuracy/gate, degraded 0.75, fallback 0.6 gate). Real runs' payloads from the baseline script can replace the bundled dataset.
