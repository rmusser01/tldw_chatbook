# Ordered authored inputs and pressed admission — incremental artifact

This patch connects the existing composer authored-edit observer to actual
UpdateInputs commands on the selected client. The local route remains intact.
Only the actual Applied revision is published to the composer; no revision is
inferred. Issued input observers remain client-owned through shutdown.

The client retains a single press barrier. Later actual widget edits stay in
FIFO order until the pressed ReceiveTurn receives its original claim or known
refusal. The press retains the actual preceding UpdateInputs receipt and the
already-validated detached input values. Admission uses that receipt's actual
draft revision and the existing closed ReceiveTurn schema. It does not infer
acceptance from retirement. Source/session/visit drift refuses admission.

The separate composition patch adds acknowledgement_idle using the original
activity/run-state/blocked predicate, captured by the actual backend endpoint.
It is the input required by the disjoint original ACK-paint adapter.

The mounted test extension uses actual composer key events and actual process
receipts to verify that a held press prevents subsequent authored input from
overtaking it. It then releases the barrier and reads the real backend draft
and revision. All original mount/navigation/shutdown assertions remain.

Validation: all proposed files parse; undefined-name Ruff checks and portable
patch apply checks pass. No imports, native execution, or test execution here.
Root owns the live mounted test; apply the test hunk only after its current run.

Still incomplete: the actual request_visible_send caller must retain this press,
capture original provider/presentation selection, schedule the original ACK
paint, submit the press, and observe original custody/commit/terminal outcomes.
The new admission API is not yet called by that UI flow. Raw/slash/escaped
classification remains an original caller responsibility; this patch grants
no local-command authority. No whole Send or journey qualification is claimed.

Files and exact preimages are in manifest.json, composition-manifest.json, and
mounted-test-manifest.json. product.patch, composition.patch, and
mounted-test.patch are independently reviewable. Proposed full files are crash
recovery copies; apply patches against matching preimages, not whole-file copies.
