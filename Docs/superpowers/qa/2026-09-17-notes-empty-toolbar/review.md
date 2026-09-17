# Independent review — TASK-32752

Read-only review by `/root/rag_heading_review` against integration 36164f9ccc
and the final working diff. The measured-width and stable-contract distinction,
actual partition repacking and growth identity behavior were reviewed.

One finding: cached tree labels/partitions survived a transition to a legacy list
without a tree projection. A subsequent shrink could rebuild that list using stale
measurements and drop its filter identity. The regression failed with the cache
reset removed; both caches now clear at the start of `_compose_list`.

Final review confirmed the fix and regression resolve the finding and reported
no remaining blocker. Eighteen affected cases and 44 governance/build/budget cases
passed; four native empty-Notes captures are separately qualified in README.md.
