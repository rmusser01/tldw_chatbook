---
id: TASK-17385
title: 'Record repository, research-graph, and biomedical-stress live baselines'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 15:52'
updated_date: '2026-08-17 03:24'
labels:
  - research
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The biomedical lane has a recorded baseline but the repositories lane (Zenodo/Figshare/OSF) and the open_research_graph lane (OpenAlex/Semantic Scholar/Crossref) have none, and the biomedical measurement used general-purpose questions that under-stress the PubMed lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The baseline script supports named question sets with a biomedical set of domain-tuned questions,The repositories and open_research_graph category lanes are run live and scored,The biomedical lane is re-run against the biomedical question set as a stress measurement,The baseline doc records all three results in the comparison table,Questions-per-set and spend bounds stay documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `--question-set {default,biomedical}` added (domain-tuned: CRISPR off-target effects, tau aggregation, gut microbiome); spend bounds unchanged.
- Results: repositories lane 1.00 citation accuracy (73/73 markers) but gate_pass only 0.29 — repository records fail the strict relevance bar most, confirming the gate's known bias against non-paper sources. open_research_graph: 1.00 (85/85), gate 0.72. Biomedical stress: 1.00 (62/62) under domain vocabulary — PubMed held. quote_grounding 0.00 everywhere (model emitted no quotes; untested, not failing).
- Live run exposed two real bugs, both fixed: (1) OSF intermittently 301s and httpx does not follow redirects by default → empty bodies; OSF client now follows redirects + sends the server-parity Accept header. (2) A malformed payload raised a raw JSONDecodeError that ESCAPED the lane's AcademicProviderError degradation catch, killing the OTHER providers' results (zenodo/figshare lost their papers to OSF's 301 HTML); all provider JSON parsing now goes through `_json_or_error`, which raises the typed error so one bad payload degrades that provider only. TDD: 3 new tests (Accept header pin, typed parse failure, good-provider survival).
<!-- SECTION:NOTES:END -->
