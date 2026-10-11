---
id: TASK-34793
title: Discovered models lose their inferred capability status after a restart
status: To Do
created_date: 2026-10-11 16:18
labels:
- providers
- discovery
priority: low
updated_date: 2026-10-11 17:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Model discovery marks a listed model as inferred when its listing says it takes images, which removes the capability unknown label and warning. The on-disk model catalog keeps only model ids and fetch times, so after a restart inside the 24 hour freshness window the models come back without listing metadata and are labelled capability unknown again until the next fetch. Found in the review of TASK-34402: a Vercel record with image input was inferred in the session that fetched it and unknown after a save and reload of the disk store, with no refetch because the entry was still fresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A model inferred as vision-capable in the session that fetched its listing is still inferred after the app restarts, without a new fetch
- [ ] #2 A catalog file written before this change still loads, and its models come back with an unknown status as they do today
- [ ] #3 A catalog entry with a malformed or out-of-bounds status is rejected the same way other malformed entries are
- [ ] #4 A full-size catalog of 4096 models with every model inferred still fits the on-disk size limit
- [ ] #5 Only the status is added to the file: no listing field other than the model id is stored
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
