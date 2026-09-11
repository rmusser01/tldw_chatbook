"""On-demand packaged Canvas documentation and shared offer-first guidance."""

from importlib.resources import files

MAX_CANVAS_GUIDE_RESULT_BYTES = 12 * 1024
CANVAS_GUIDE_PATHS = {
    "basics": "guides/basics.md",
    "controls": "guides/controls.md",
    "mermaid": "static/mermaid-authoring.txt",
    "repair": "guides/repair.md",
}
CANVAS_OFFER_POLICY = (
    "Offer Canvas selectively when a substantial visual, interaction, or revised "
    "single-page result materially helps; ordinary prose, short code, and simple "
    "tables usually stay in chat. Before proactive authoring, offer one short "
    "sentence naming the artifact and benefit, then wait for acceptance. Do not "
    "load detailed guides, delegate authoring, or generate artifact source before "
    "acceptance. Explicit creation requests and requested edits already authorize "
    "that work; do not ask again. Consent covers that artifact and bounded "
    "corrections, not unrelated artifacts or substantial unrequested redesigns. "
    "Respect refusal and do not repeat the same offer unless the user changes "
    "direction. No answer, ambiguity, or elapsed time is not consent. A bare "
    "$canvas invocation without a concrete request needs clarification. "
    "If context does not establish consent, clarify. After "
    "consent, load only needed guide topics and reuse guidance already in context. "
    "Read before edits; preserve exact runtime profiles. Distinguish staged or "
    "saved source from a ready preview. After one failed repair, stop: explain "
    "that the problem remains and ask whether to attempt another repair. A report "
    "that the repair still fails is not permission for a second attempt. Do not "
    "claim to create, update, or repair without a successful matching tool result."
)


def read_canvas_guide(topic: str) -> str:
    """Read one complete UTF-8 guide from the fixed packaged topic set.

    Args:
        topic: One of ``basics``, ``controls``, ``mermaid``, or ``repair``.

    Returns:
        Exact guide text. Callers must also bound their serialized result envelope.

    Raises:
        ValueError: If the topic is invalid or the guide is empty or oversized.
        UnicodeDecodeError: If the resource is not valid UTF-8.
        OSError: If the packaged resource cannot be opened or read.
    """
    if type(topic) is not str or topic not in CANVAS_GUIDE_PATHS:
        raise ValueError("invalid guide topic")
    resource = files("tldw_chatbook.Canvas").joinpath(CANVAS_GUIDE_PATHS[topic])
    with resource.open("rb") as handle:
        raw = handle.read(MAX_CANVAS_GUIDE_RESULT_BYTES + 1)
    if len(raw) > MAX_CANVAS_GUIDE_RESULT_BYTES:
        raise ValueError("guide is oversized")
    guide = raw.decode("utf-8")
    if not guide.strip():
        raise ValueError("guide is empty")
    return guide
