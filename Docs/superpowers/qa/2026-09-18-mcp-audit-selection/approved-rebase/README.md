# Approved Audit selection: latest-dev rebase

The owner approved PR2720's compact dark/light gallery at `6180cf54c9`.
Dev then gained PR2766's Evals feature, at
`f1a027c12b67e6aa91a52156e3de0fa48c33f354`. Rebase was conflict-free;
all fourteen captured sources, the runner, MCP styles and selection tests remain
identical to the approved version. Upstream Evals and its testing lessons are
preserved. No new conflict or visual choice needs approval.

The rebased tree passes **27 focused selection/startup checks** and all **nine
artifact guards**. The earlier 305-case inventory remains qualified for the
unchanged affected sources; these 27 checks are an overlapping rebase replay,
not an additional distinct-test claim. No full suite ran.

A fresh private native run repeated all four theme/size journeys and twenty
captures. All terminal text matches the approved captures after fixture timestamp
normalization. Seventeen SVGs also match after normalizing Rich's generated IDs;
the remaining three differ only in the filter input's blinking caret (one is
zero-width). Those three SVGs and semantic diffs are retained here, and
`visual-comparison.json` records every original/replay capture hash. There is no
change to the approved layout, selection or drilldown behavior.

`lifecycle.json` verifies normal app exit, absent fixture processes, released
instance lock, ten healthy private databases, unchanged user defaults and
sentinels, no external network or tool execution, and matching source hashes.
Initial preparation used the system Python 3.9, which lacks `tomllib`; it failed
before app launch. The empty owned terminal was closed, and this successful run
used repository Python 3.12.11 with a fresh private profile.

The PR is approved for merge after current-head CI, Qodo review and the final
dev check. TASK-32834 remains In Progress until the merge is confirmed.
