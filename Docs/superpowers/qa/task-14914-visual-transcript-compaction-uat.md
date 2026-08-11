# TASK-14914 visual transcript compaction UAT

Date: 2026-08-11
Perspective: senior UX/HCI review
Status: implementation UAT complete; live model-quality gate intentionally open

## Step-by-step UAT

1. Open a Console conversation with a text-only model and open Model settings >
   Context.
   - Text summary remains selected by default.
   - Visual transcript and Hybrid remain visible in the dropdown but are disabled.
   - The adjacent status explains that a vision-capable model is required.
2. Switch the current model to a known vision-capable model.
   - Visual transcript and Hybrid become selectable without reopening the modal.
   - The status explains local/request-scoped rendering, recent-text retention, and
     model-specific image-token estimates.
3. Select Visual transcript and save the conversation override.
   - The representation is stored independently from Behavior (Ask/Automatic/Off).
   - Conversation max tokens and Response max tokens remain separately labeled and
     retain their separate values.
4. Reach the compaction threshold with older complete exchanges and send a message.
   - Only the selected oldest complete exchanges become image pages.
   - The active request and recent retained exchanges remain text.
   - The exact image-bearing provider request is re-counted before dispatch.
5. Exercise Compact now in Visual transcript mode.
   - The action reports that the visual transcript fits and will be regenerated for
     each request; it does not claim that durable generated memory was created.
6. Switch that conversation to a text-only model.
   - The saved visual intent remains visible.
   - Effective behavior safely falls back to Text summary without overwriting the
     preference or dropping mandatory context.
7. Select Hybrid on a vision-capable model.
   - Durable text memory is created through the existing guarded transaction.
   - Visual pages are added only when total image count and exact prepared-request
     accounting still fit; otherwise the text summary proceeds alone.
8. Open F9 Settings > Console behavior.
   - The global Representation default exposes all three strategies.
   - Copy explains that unsupported sessions use Text summary and that only visual
     page bytes are request-scoped; Text summary/Hybrid memory remains durable.
9. Inspect the Context UI at 120x42 and Settings at 80x34.
   - Labels remain associated with their controls, the representation explanation
     wraps inside the viewport, and the existing scroll/fold affordance remains.
10. Run the offline benchmark without an OCR/model evaluator.
    - Local render/token-cost fields are populated.
    - OCR fidelity, code/math recovery, instruction recall, adversarial behavior,
      and end-to-end latency report `unknown`; default enablement remains blocked.

## Findings log

1. **Summary-specific labels conflicted with Visual transcript.** Addressed by using
   “Compact at,” “If compaction fails,” and “Keep after compaction.”
2. **Vision-only choices could look broken on a text model.** Addressed with truly
   disabled Select-overlay options plus a visible capability reason.
3. **Capability fallback risked appearing to erase user intent.** Addressed by
   preserving the sparse preference and showing that Text summary is only the
   effective behavior for the current model.
4. **Artifact lifetime was not discoverable.** Addressed with modal and Settings
   copy stating that visual pages are on-device, request-scoped, and not persisted.
5. **“Summary response max” could be mistaken for the conversation maximum or next
   response maximum.** Addressed by keeping all three labels explicit and adding
   that Summary response max applies only to Text summary and Hybrid.
6. **PNG byte compression could be mistaken for token savings.** Addressed by
   labeling image-token cost as model-specific/possibly estimated and by measuring
   provider representation rather than PNG byte size.
7. **Existing attachments could make a nominally valid page count exceed a model's
   total image limit.** Addressed by counting retained request images before adding
   visual pages.
8. **Visual-only Compact now could falsely report durable memory creation.**
   Addressed with representation-specific success copy and no PNG/database write.
9. **Adversarial transcript text could imitate role or exchange headings.**
   Addressed with renderer-owned headings, quoted body prefixes, a fixed untrusted
   data header, explicit role/tool boundaries, and escaped unsupported Unicode.
10. **The first benchmark implementation could initialize user tokenizer storage.**
    Addressed with a pure, side-effect-free conservative text estimate by default
    and an injectable exact model-tokenizer callback.
11. **A renderer/library update could silently change page hashes.** Addressed by
    including the Pillow version in the renderer identity and deterministic hash
    tests over the exact PNG bytes.
12. **An oversized transcript could allocate too many PNGs and block the Textual
    event loop before image-limit rejection.** Addressed by enforcing the remaining
    model image allowance before PNG encoding and running visual planning/rendering
    in a worker thread.

## Offline benchmark sample

Corpus: three repeated code-and-constraint exchanges; provider/model identity
`openai/gpt-4o`; conservative 512 tokens per 1024x1024 image page.

| Metric | Result |
| --- | --- |
| Renderer | `chatbook-visual-transcript-v1-pillow-11.2.1` |
| Pages | 9 |
| Text input cost | 7,355 tokens (estimated) |
| Visual input cost | 4,638 tokens (estimated) |
| Estimated reduction | 36.9% |
| Local render latency | 480 ms |
| OCR fidelity | Unknown — no model/OCR evaluator supplied |
| Code/math recovery | Unknown — no model evaluator supplied |
| Instruction recall | Unknown — no model evaluator supplied |
| Adversarial-text behavior | Unknown — no model evaluator supplied |
| End-to-end latency | Unknown — no provider call supplied |
| Eligible as default | No |

This sample deliberately does not validate the proposed 60–70% token-reduction
claim. Visual transcript and Hybrid remain opt-in until model-specific evaluation
fills the unknown quality and safety fields.
