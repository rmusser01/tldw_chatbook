"""Pure data for character probe evals. No I/O, no Textual, no provider calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Probe:
    """One scripted exchange: an ordered list of user turns.

    A "one-off" question is simply a probe with a single turn -- there is no
    separate type for it. Turn text is verbatim, including interior newlines,
    because prompt formatting changes model behaviour.

    Raises:
        ValueError: If the probe has no turns, or if any turn is empty or
            contains only whitespace.
    """

    turns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("A probe needs at least one turn.")
        for turn in self.turns:
            if not turn.strip():
                raise ValueError("A turn cannot be empty or whitespace-only.")


@dataclass(frozen=True)
class ProbeSet:
    """An ordered collection of probes, the unit a bench runs."""

    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class CharacterProbeConfig:
    """A character probe bench definition.

    ``character_ids`` are ``character_cards.id`` INTEGERs, unlike every eval
    id in this slice, which is TEXT. They are deliberately not normalised to
    strings: the cross-database lookup against ``ChaChaNotes_DB`` binds them
    as integers. This is enforced here, not merely documented: a caller
    constructing one directly with string ids (e.g. ``("3", "7")``, easy to
    do by accident from a form field) is rejected at construction rather
    than being accepted here only to come back out as ``int`` after a
    ``storage.save_character_bench``/``load_character_bench`` round trip --
    an asymmetry a future UI caller would otherwise have to discover the
    expensive way.

    Defaults are conservative on purpose: ``samples_per_cell=1`` (no
    surprise fan-out cost), ``seed=None`` (no false sense of determinism
    unless explicitly requested), and ``concurrency=1`` (no surprise
    parallel-request load against a target).

    ``seed`` has THREE meanings, and the distinction is load-bearing rather
    than cosmetic -- ``runner._sample_seed`` branches on it:

    * ``None`` -- unseeded. No seed is sent at all.
    * a NEGATIVE value -- the "pick a random seed" sentinel llama.cpp
      defines (``-1`` conventionally). It is a sentinel, not a number to do
      arithmetic on, so it is sent UNCHANGED for every sample of a cell.
      Offsetting it would make ``-1`` run as ``-1, 0, 1, ...``: sample 0
      randomly seeded, every later sample deterministic and never asked
      for, which destroys exactly the variance multi-sampling exists to
      show.
    * ZERO or POSITIVE -- a real seed. The per-sample seed is
      ``seed + sample_index``, so a seeded run is reproducible *and* its
      samples genuinely differ. ``0`` is a real seed here, not a falsy
      stand-in for "unset" (the same distinction ``storage``'s stored-field
      readers enforce).

    Raises:
        ValueError: If ``samples_per_cell`` or ``concurrency`` is less than
            1, or if ``character_ids``/``target_ids`` is empty -- a bench
            with no characters or no targets can never produce a cell.
        ValueError: If any element of ``character_ids`` is not an ``int``
            (``bool`` included, since it is an ``int`` subclass but never a
            real card id) -- see the type-fidelity note above.
    """

    name: str
    probe_set_id: str
    character_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    description: str = ""
    concurrency: int = 1
    samples_per_cell: int = 1
    seed: Optional[int] = None
    temperature: float = 0.8
    max_tokens: int = 512
    extra_tags: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.samples_per_cell < 1:
            raise ValueError("samples_per_cell must be 1 or more.")
        if self.concurrency < 1:
            raise ValueError("concurrency must be 1 or more.")
        if not self.character_ids:
            raise ValueError("A character probe bench needs at least one character.")
        if not self.target_ids:
            raise ValueError("A character probe bench needs at least one target.")
        for cid in self.character_ids:
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ValueError(
                    f"character_ids must be int (character_cards.id), got "
                    f"{cid!r} of type {type(cid).__name__}."
                )


@dataclass(frozen=True)
class CardSnapshot:
    """A character card's text, copied at run time.

    Cards live in ``ChaChaNotes_DB`` while runs live in ``Evals_DB``, with no
    foreign keys between them. Copying the text into the run means editing or
    deleting a card later never rewrites what a past run shows -- the same
    provenance rule word_bench applies to snippets.

    Every field here is prompting text and every one is composed by
    ``prompt.compose_system_prompt`` (``name`` additionally supplies the
    ``{{char}}`` macro's value). ``description`` is the primary V2 persona
    field: it was absent from this dataclass, from ``cards._SNAPSHOT_FIELDS``,
    and from the design spec's field list until the whole-branch review of
    task-1691 phase 1, so every probe ran against a character stripped of its
    main definition. Adding a field here is only half a change -- compose it
    in ``prompt.py`` too, or the snapshot grows text the model never sees.
    """

    id: int
    name: str
    description: str = ""
    system_prompt: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    post_history_instructions: str = ""
    message_example: str = ""


@dataclass(frozen=True)
class ConversationTurn:
    """One scripted user turn and the model's reply to it.

    A turn is recorded only once its provider call has returned, so a turn
    that never ran (the conversation was cancelled or failed before reaching
    it) is simply absent from its ``Conversation.turns`` tuple. A turn that
    *is* present may still carry ``reply == ""``, and that IS a state this
    runner produces: the provider callable is allowed to return empty text
    (the app's own response extraction yields ``""`` rather than raising for
    a response with no content) and the runner records it verbatim. "The
    model said nothing" is a real observation this eval exists to surface,
    not a malformed row.

    ``error`` is RESERVED and is never populated by the runner today: a
    failed turn ends its whole conversation, and ``Conversation.error`` is
    that failure's only home. The field exists -- and round-trips through
    storage -- so that a future per-turn record (a retried turn, say) has
    somewhere to live without a storage migration. Do not read ``""`` here
    as "this turn succeeded"; read ``Conversation.error`` for that.
    """

    user: str
    reply: str
    error: str = ""


@dataclass(frozen=True)
class Conversation:
    """One cell: a card, a probe, a target, and one sample of the exchange.

    ``turns`` holds only the turns that actually ran -- a conversation that
    failed or was cancelled partway through keeps its completed turns and
    records why it stopped in ``error``, rather than discarding the partial
    transcript. A failed or partial conversation is still evidence and stays
    reviewable.
    """

    card_id: int
    probe_index: int
    sample_index: int
    target_id: str
    turns: tuple[ConversationTurn, ...]
    error: str = ""
