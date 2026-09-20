# PR2730 approved closeout

The owner approved the Console approval gallery and merge at head
`fd3827540e173db190a05d3fa6bb15f1eb926b82`. Dev subsequently merged PR2754,
the Library/Artifacts integration, at `d1a0649cd2eaead3beabdb3d510077a2114f940f`.
The three PR2730 commits rebased cleanly, with **no conflicts**.

The approval card, controller, Console screen, verification runner and both
targeted test modules are byte-identical to the approved head. Upstream dev owns
the navigation change; no new product changes were introduced by this closeout.
[Integration comparison](closeout/integration-review.json).

- **211 targeted passes, 18 existing failures** in [the fresh run](closeout/tests/001.txt).
  [All 18 failures reproduce on unchanged current dev](closeout/baseline-comparison.json),
  comparing names, messages and final lines with only object addresses normalized.
  The previously failing human-input-wait case passes in both current runs; this
  PR does not claim to repair it. The full approval module is not green. No full sweep.
  [Case census](closeout/qualified-cases.json); all prior 210 passing IDs retained.
- [All seven artifact guards pass](closeout/preflight.txt). Approved production,
  tests and runner are unchanged, so their established static checks still apply.
- [Eight fresh native captures](GALLERY.md) cover keyboard Deny all, Approve all
  and Submit through the real controller in dark/light at 120×40 and 170×48.
  [Pixel comparison](closeout/visual-comparison.json) confines every difference
  from the [approved gallery](APPROVED-GALLERY.md) to the main navigation bar:
  Artifacts now lives in Library. Approval controls, choices, focus and all other
  pixels are unchanged. Four final submit views were inspected across all cells.
- [Native result](closeout/native/result.json) and [lifecycle](closeout/lifecycle.json)
  verify current module origins and hashes, normal exit, no network or tool
  dispatch, released lock, ten healthy private databases, zero persisted chats
  or messages and unchanged user defaults. [Export hashes](closeout/export-manifest.json).

The owner’s approval remains applicable to the unchanged Console repair. The
rebased head must pass its own CI and accumulated review before the authorized
merge. Verify the actual merged tree before resuming the saved MCP inspector work.
Existing ADR-032/150 apply; no new ADR. Upstream navigation follows ADR-172.
