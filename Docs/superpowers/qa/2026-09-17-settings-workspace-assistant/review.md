# TASK-32769 independent review

Read-only review compared this bounded change with bf99fb957d. The reviewer
confirmed saved-field preservation, read-only initialization for a new persona,
and retained read-write/imported-profile confirmation boundaries.

Two introduced presentation gaps were reproduced and repaired:

- A guarded `scroll_visible()` still queued an unguarded later scroll. A small
  Textual probe moved focus to another control before that queued scroll and
  observed its label leave the viewport. The final reveal uses `immediate=True`
  after checking current workspace, result, attached focus, and originating ID.
- A viewport-height OptionList could lose its highlighted first row when the
  receipt scrolled into view. Both owning pickers now use the existing
  `$ds-size-5` maximum. The populated keyboard matrix verifies complete selected
  text, full picker/receipt paint, and internal list scrolling. A focused row uses
  design-system colors; the generic overpainting outline is disabled locally.

The receipt resize hook responds to actual text reflow and reuses the same
guards. Moving the single receipt after its originating control retains widget
identity. Final review found no remaining introduced blocker. The reviewer did
not mutate the checkout or run the final verification commands; their results
are recorded separately in this QA folder.

A pre-existing confirmation arm can survive a roundtrip between workspace cards.
This slice does not claim that transition was repaired or tested. Keep it in the
remaining assistant/navigation review alongside broader workspace lifecycle
flows. It does not change the field-preservation and explicit-press paths
qualified here.
