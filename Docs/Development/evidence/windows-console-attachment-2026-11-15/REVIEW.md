# Independent review record

Source scope: baseline 7dbeff56aba6032f43aba2f4bbab803d3f133471 to product ea5a104807a53c2ba69c2434f6a4a1b22118b139, including connected fixtures and CI routing. Reviewer: independent read-only agent /root/boot_admission_review, using requesting-code-review. Auto-merge remains off.

Findings addressed:

- Existing synchronous-starter and fake-worker fixtures needed the new deferred admission/exact handle contract.
- The proposed helper initially exceeded an existing startup module ceiling. Restore that module and use a dedicated 83-line boot helper with its own exact ceiling; existing ceilings only decrease.
- Legacy Actor cancellation could hit an earlier project-index drain. Observe the original drain with the exact retained read before cancellation; preserve all ownership/retirement assertions.
- Actual published harness bytes must be hashed, in addition to comparing receipt dictionaries. The validator now does this; staged blob qualification is recorded separately.
- The excluded preliminary cohorts needed normalized-byte equality and complete source-manifest provenance. The witness now verifies both, with names distinguishing executed bytes from the Git source blob.

Final source/evidence review: no remaining Critical or Important finding. Native admission, fresh effect-time scope, actual Worker identity, local-pause retry, physical shutdown ownership and deadlines remain intact. Two child-process test budgets become 120s for startup/seeding/exit; internal safety and profile bounds remain unchanged.

The report retains worse first-return attachments and worst sampled overshoot, and does not claim a uniform or general latency win. Review itself did not execute tests; root verification is recorded in the test/profile receipts. Publication verification passes: all four staged harness blobs exactly match the executed receipt digests, including native CRLF. See harness-git-qualification.json.
