"""Parsing/formatting plus the blocking generation batch for the native
Console ``/generate-image`` command.

``parse_generate_image_args``/``generation_content_marker`` (and the module
docstring's original promise) have no dependency on Textual, the running
app, or any I/O — mirroring ``console_prefill.py``. ``run_generation_batch``
is the one deliberate exception: it drives the blocking, network-calling
``Image_Generation.worker`` entry points, so it must run off the UI loop
(the screen layer offloads it via ``asyncio.to_thread``, exactly like
``run_generation`` itself demands). Its adapter dependencies
(``worker.run_generation``/``worker.build_request``) are imported lazily
inside the function rather than at module scope, so importing this module
for its pure helpers never eagerly pulls in the Image_Generation package.

Grammar: optional leading ``:backend`` and ``@style`` tokens, in any order,
select a non-default backend (``/generate-image :swarmui a dragon``) and/or
a generation-template style (``/generate-image @anime a dragon``). Token
consumption stops at the first token that isn't prefixed with ``:`` or
``@``; everything from there on is the prompt. A bare ``:`` or ``@`` (the
prefix alone, nothing after it) is NOT a token — it stays part of the
prompt. A leading token with no trailing text parses to an empty prompt,
which the caller refuses.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_command_grammar import (
    COMMAND_PREFIX,
    GENERATE_IMAGE_COMMAND_NAME,
)
from tldw_chatbook.Media_Creation.generation_templates import (
    GenerationTemplate,
    apply_template_to_prompt,
    get_all_templates,
    get_template,
)
from tldw_chatbook.Media_Creation.image_generation_service import (
    ImageGenerationService,
)

GENERATION_MARKER_PREFIX = "[image] "
"""Prefix identifying a generation card's content marker in a message row."""

_MARKER_PROMPT_MAX_CHARS = 80

GENERATE_IMAGE_USAGE_TEXT = "Usage: /generate-image [:backend] <prompt>"
"""Status text for a ``/generate-image`` invocation with nothing to work with:

no prompt, no style-resolved template default, and no conversation content
to fall back on.
"""


def clamp_initial_batch(default_batch: int, max_variants: int) -> int:
    """Clamp the initial batch count to not exceed the variant cap.

    Args:
        default_batch: Configured default batch size (≥ 1).
        max_variants: Configured maximum variants per message (≥ 1).

    Returns:
        The minimum of the two values, preserving ≥ 1 semantics.
    """
    return min(default_batch, max_variants)


@dataclass(frozen=True)
class GenerateImageArgs:
    """One parsed ``/generate-image`` invocation.

    Args:
        backend: Backend id from a leading ``:backend`` token, or ``None``
            when the command should use the configured default.
        prompt: Generation prompt text (stripped). Empty when the user
            supplied no prompt — the caller refuses to dispatch then.
        style: Raw text of a leading ``@style`` token (without the ``@``),
            unresolved against the template catalog — the caller passes it
            to `resolve_style_token`. ``None`` when no ``@style`` token was
            present.
    """

    backend: str | None
    prompt: str
    style: str | None = None


def parse_generate_image_args(args: str) -> GenerateImageArgs:
    """Split the args string of one ``/generate-image`` invocation.

    Consumes leading whitespace-delimited tokens in any order/combination:
    a token starting with ``:`` (longer than the bare colon) sets the
    backend override; a token starting with ``@`` (longer than the bare
    ``@``) sets the raw style token. Consumption stops at the first token
    that matches neither shape — that token and everything after it is the
    prompt. Without any leading tokens the whole stripped string is the
    prompt.

    Args:
        args: Raw text after the ``/generate-image`` command word.

    Returns:
        A `GenerateImageArgs` with the optional backend/style overrides and
        the remaining prompt (empty string when no usable prompt was
        given).
    """
    remaining = args.strip()
    backend: str | None = None
    style: str | None = None
    while remaining:
        parts = remaining.split(None, 1)
        token = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if token.startswith(":") and token != ":":
            backend = token[1:]
            remaining = rest
            continue
        if token.startswith("@") and token != "@":
            style = token[1:]
            remaining = rest
            continue
        break
    return GenerateImageArgs(backend=backend, prompt=remaining.strip(), style=style)


GENERATE_IMAGE_COMMAND_WORD = COMMAND_PREFIX + GENERATE_IMAGE_COMMAND_NAME
"""The full leading command word (``"/generate-image"``), as registered."""


def _starts_with_command_word(draft: str) -> bool:
    """Whether `draft` opens with `GENERATE_IMAGE_COMMAND_WORD`, case-sensitive.

    Matches the bare command word, or the command word followed by
    whitespace -- never a string that merely has it as a prefix (e.g.
    ``"/generate-imagex"`` does NOT match).
    """
    word = GENERATE_IMAGE_COMMAND_WORD
    if not draft.startswith(word):
        return False
    return len(draft) == len(word) or draft[len(word)].isspace()


def insert_style_token_into_draft(draft: str, style_id: str) -> str:
    """Compose a valid ``/generate-image`` draft carrying an ``@<style_id>`` token.

    This is the sole grammar-aware seam the Console style picker's "insert"
    callback goes through -- it exists so that callback never has to
    reimplement (or drift from) `parse_generate_image_args`'s leading-token
    rules. Two shapes, both always producing a draft that
    `parse_generate_image_args` resolves with `style` set to `style_id`:

    1. `draft` already opens with `GENERATE_IMAGE_COMMAND_WORD` (the bare
       word, or the word followed by whitespace): the leading `:backend`/
       `@style` tokens are walked the same way `parse_generate_image_args`
       walks them. A leading `:backend` token is kept exactly where it is;
       an existing leading `@style` token is REPLACED (never stacked --
       applying this twice yields exactly one style token, the newest);
       when no style token was present one is inserted immediately before
       the prompt remainder, after any `:backend` token.
    2. Any other `draft`, including the empty string, is treated as plain
       prompt text with no command word yet: the whole thing is prefixed
       with the command word and the new style token.

    Args:
        draft: The current composer draft text, verbatim.
        style_id: The resolved style template's `id` (e.g. ``"style_anime"``),
            without the leading ``@``.

    Returns:
        The new draft text, always opening with `GENERATE_IMAGE_COMMAND_WORD`
        and carrying exactly one ``@<style_id>`` token ahead of the prompt.
    """
    style_token = f"@{style_id}"
    if not _starts_with_command_word(draft):
        prompt = draft.strip()
        if prompt:
            return f"{GENERATE_IMAGE_COMMAND_WORD} {style_token} {draft}"
        return f"{GENERATE_IMAGE_COMMAND_WORD} {style_token} "

    remaining = draft[len(GENERATE_IMAGE_COMMAND_WORD) :].strip()
    leading_tokens: list[str] = []
    style_replaced = False
    while remaining:
        parts = remaining.split(None, 1)
        token = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if token.startswith(":") and token != ":":
            leading_tokens.append(token)
            remaining = rest
            continue
        if token.startswith("@") and token != "@":
            leading_tokens.append(style_token)
            style_replaced = True
            remaining = rest
            continue
        break
    if not style_replaced:
        leading_tokens.append(style_token)

    header = " ".join([GENERATE_IMAGE_COMMAND_WORD, *leading_tokens])
    if remaining:
        return f"{header} {remaining}"
    return f"{header} "


def generation_content_marker(
    prompt: str, limit: int = _MARKER_PROMPT_MAX_CHARS
) -> str:
    """Render the single-line content marker for a generation card.

    Args:
        prompt: Full generation prompt.
        limit: Maximum rendered prompt length, including the ellipsis.

    Returns:
        ``"[image] "`` followed by ``prompt`` with whitespace runs
        collapsed to single spaces, cut to ``limit`` chars with a trailing
        ``…`` when longer.
    """
    flattened = " ".join(prompt.split())
    if len(flattened) > limit:
        flattened = flattened[: limit - 1] + "…"
    return GENERATION_MARKER_PREFIX + flattened


def _normalize_style_text(text: str) -> str:
    """Casefold and collapse underscore/whitespace variance for style matching.

    Underscores and whitespace runs are both treated as word separators and
    reduced to single spaces, so ``"anime_style"``, ``"anime  style"`` and
    ``"Anime Style"`` all normalize identically.

    Args:
        text: Raw text (a token or a template name).

    Returns:
        The normalized, casefolded text.
    """
    return " ".join(text.replace("_", " ").split()).casefold()


@dataclass(frozen=True)
class StyleResolution:
    """Outcome of resolving a raw ``@style`` token against the template catalog.

    Args:
        template: The uniquely resolved template, or ``None`` when the
            token matched nothing or matched more than one candidate.
        ambiguous: Sorted ids of every template a prefix query matched,
            when more than one did. Empty otherwise (including the
            no-match case).
    """

    template: GenerationTemplate | None
    ambiguous: tuple[str, ...] = ()


def resolve_style_token(token: str) -> StyleResolution:
    """Resolve a raw ``@style`` token to a generation template.

    Matches against every builtin AND user-defined style template (see
    `generation_templates.get_all_templates` -- a user template with the
    same id as a builtin overrides it). Case-insensitive, tried in order,
    first hit wins:

    1. Exact template id (e.g. ``style_anime``).
    2. Exact template name, with spaces and underscores interchangeable in
       either direction (e.g. ``anime_style`` or ``anime style`` both match
       the template named ``"Anime Style"``).
    3. A unique prefix over every template id and (normalized) name. When
       the prefix matches more than one template, the result carries every
       matched id (sorted) instead of a template.

    Args:
        token: Raw token text, already stripped of the leading ``@``.

    Returns:
        A `StyleResolution`. Both `StyleResolution.template` and
        `StyleResolution.ambiguous` are empty/``None`` when the token
        matched nothing.
    """
    cleaned = token.strip()
    if not cleaned:
        return StyleResolution(template=None)
    cleaned_cf = cleaned.casefold()
    normalized_token = _normalize_style_text(cleaned)
    templates = get_all_templates()

    for template in templates.values():
        if template.id.casefold() == cleaned_cf:
            return StyleResolution(template=template)

    for template in templates.values():
        if _normalize_style_text(template.name) == normalized_token:
            return StyleResolution(template=template)

    matched: dict[str, GenerationTemplate] = {}
    for template in templates.values():
        if template.id.casefold().startswith(cleaned_cf) or _normalize_style_text(
            template.name
        ).startswith(normalized_token):
            matched[template.id] = template

    if len(matched) == 1:
        return StyleResolution(template=next(iter(matched.values())))
    if len(matched) > 1:
        return StyleResolution(template=None, ambiguous=tuple(sorted(matched)))
    return StyleResolution(template=None)


def compose_styled_request(
    user_prompt: str, template: GenerationTemplate
) -> tuple[str, str, dict[str, Any]]:
    """Compose a styled generation request from a resolved template.

    Builds the template's substitution context by mapping every one of its
    ``context_mappings`` target keys to ``user_prompt`` — so every
    ``{{placeholder}}`` the template's ``base_prompt`` references gets
    filled with the user's own text — then renders it via
    ``apply_template_to_prompt``.

    Invariant: the user's prompt text must appear in the composed prompt.
    A template with no (or effectively unconsumed) context mappings would
    otherwise silently drop what the user typed; when the rendered prompt
    doesn't contain ``user_prompt`` verbatim, this falls back to appending
    it onto whatever the template rendered (stripping stray leading/
    trailing comma-space artifacts left behind by the dropped placeholder).

    Args:
        user_prompt: The user's raw prompt text.
        template: The resolved style template (see `resolve_style_token`).

    Returns:
        A ``(prompt, negative_prompt, params)`` tuple. ``params`` is a copy
        of the template's `GenerationTemplate.default_params`, safe for the
        caller to mutate.
    """
    context = {target: user_prompt for target in template.context_mappings.values()}
    composed, negative, params = apply_template_to_prompt(template.id, context)
    if user_prompt in composed:
        return composed, negative, params
    base = composed.strip(" ,")
    combined = f"{base}, {user_prompt}" if base else user_prompt
    return combined, negative, params


def _apply_template_with_anchor(
    context: dict[str, Any], template: GenerationTemplate
) -> tuple[str, str, dict[str, Any]]:
    """Render ``template`` against ``context`` via `apply_template_to_prompt`,
    then apply the shared "never silently drop the anchor" invariant.

    Shared by `build_context_prompt` (keyword-extracted context) and
    `build_context_prompt_with_llm`'s LLM-composed path -- both shape a
    ``context`` dict with the same keys
    (``last_message``/``mentioned_characters``/``mentioned_settings``/
    ``mood``/``style_hints``) and need the identical anchor-append guard:
    when ``context["last_message"]`` is non-empty and doesn't appear
    verbatim in the rendered prompt (e.g. the template has no context
    mapping consuming it), it is comma-appended so conversation content is
    never silently dropped.

    Args:
        context: Context dict shaped for `apply_template_to_prompt`; must
            carry a ``last_message`` key (may be empty).
        template: The style template to render.

    Returns:
        A ``(prompt, negative_prompt, params)`` tuple.
    """
    composed, negative, params = apply_template_to_prompt(template.id, context)
    anchor = context.get("last_message", "")
    if anchor and anchor not in composed:
        base = composed.strip(" ,")
        composed = f"{base}, {anchor}" if base else anchor
    return composed, negative, params


def build_context_prompt(
    messages: Sequence[tuple[str, str]], template: GenerationTemplate
) -> tuple[str, str, dict[str, Any]] | None:
    """Compose a generation request from conversation context (no user prompt).

    Used by the ``/generate-image`` handler's "generate from conversation"
    path: shapes ``messages`` (chronological ``(role, content)`` pairs) into
    the ``[{"role": ..., "content": ...}]`` form
    ``ImageGenerationService.extract_context_from_messages`` expects,
    extracts context (most recent user message, mood, visual-hint
    fragments), then renders ``template`` against it via
    `_apply_template_with_anchor` — the same engine `compose_styled_request`
    uses for an explicit ``@style`` token.

    ``extract_context_from_messages`` is called as
    ``ImageGenerationService.extract_context_from_messages(None, shaped)``
    rather than on a constructed instance: the method never reads ``self``
    (verified by inspection), while constructing an instance would run
    `ImageGenerationService.__init__`'s side effects (creating the
    generated-images output directory tree, logging). Calling it unbound is
    the honest cheap invocation.

    This is the keyword-shallow extractor (Task-559 AC1's baseline): mood
    via keyword match, ``mentioned_characters``/``mentioned_settings`` never
    populated. `build_context_prompt_with_llm` layers a richer, optional
    LLM-composed alternative on top of this function, which now serves as
    both the direct entry point (unchanged, for callers/tests that want it)
    and the guaranteed fallback when the LLM path is disabled or fails.

    Args:
        messages: Chronological ``(role, content)`` pairs from the session.
        template: The style template to render — an explicit ``@style``
            resolution, or the ``chat_scene_visual`` default.

    Returns:
        A ``(prompt, negative_prompt, params)`` tuple, or ``None`` when
        every message's content is empty/whitespace-only (nothing usable to
        build a prompt from).
    """
    shaped = [
        {"role": role, "content": content}
        for role, content in messages
        if content and content.strip()
    ]
    if not shaped:
        return None
    context = ImageGenerationService.extract_context_from_messages(None, shaped)
    return _apply_template_with_anchor(context, template)


_CONTEXT_LLM_SYSTEM_PROMPT = (
    "You compose a single concise prompt for a text-to-image generator "
    "from a conversation transcript. Read the conversation and write ONE "
    "plain-text paragraph -- no lists, no headings, no preamble, no "
    "surrounding quotes -- describing the subject, setting, mood, and any "
    "style hints suggested by the conversation, suitable to hand directly "
    "to an image generation model. Respond with the prompt text only."
)
"""System instruction for the LLM-composed conversation-context path
(Task-559 AC1). Kept as a module constant so tests can assert on it without
duplicating the literal string."""

_CONTEXT_LLM_MAX_RESPONSE_CHARS = 500
"""Length cap applied to a raw LLM-composed response before use -- a
"concise" prompt per the system instruction; independent of
``[image_generation].max_prompt_length`` (a separate, later-stage cap on
the FINAL composed/templated prompt)."""


@dataclass(frozen=True)
class LLMContextOptions:
    """Resolved inputs for the optional LLM-composed conversation-context path.

    Bundles the ``[image_generation]`` config kill-switch/turns/timeout
    with the session's ALREADY-RESOLVED active provider identity. This
    module has no Textual/app dependency, so it never resolves the active
    provider itself -- the caller (the Console screen) is responsible for
    resolving it through the exact same seam a normal Console chat send
    uses (``ConsoleProviderGateway.resolve_for_send``) and handing the
    result in here.

    Args:
        enabled: Config kill-switch
            (``[image_generation].context_llm_enabled``). ``False`` skips
            the LLM path entirely, unconditionally.
        turns: Max number of trailing ``(role, content)`` pairs sent to the
            LLM as conversation context.
        timeout_seconds: Hard wall-clock cap on the LLM call; a slower
            response is abandoned in place (never joined) and treated as a
            failure.
        provider_ready: Whether the session has a usable, configured
            provider right now (mirrors ``ConsoleProviderResolution.ready``
            -- resolved the same way a normal Console send resolves it:
            config/env for most providers, plus llama.cpp's own bounded
            ``/health`` reachability probe for that one provider).
        api_endpoint: Resolved provider execution key for `chat_api_call`
            (``ConsoleProviderResolution.execution_key``). ``None`` when
            not ready.
        model: Resolved model string, or ``None``.
        api_key: Resolved API key, or ``None`` when the provider needs
            none (e.g. a local backend).
        chat_call: A ``chat_api_call``-shaped callable. ``None`` (the
            default) lazily imports and uses the real
            ``Chat.Chat_Functions.chat_api_call`` -- importing this module
            for its pure helpers never eagerly pulls in every LLM provider
            integration. Overridable for tests.
    """

    enabled: bool
    turns: int
    timeout_seconds: float
    provider_ready: bool
    api_endpoint: str | None = None
    model: str | None = None
    api_key: str | None = None
    chat_call: Callable[..., Any] | None = None


def _shape_llm_context_payload(
    messages: Sequence[tuple[str, str]], turns: int
) -> list[dict[str, str]]:
    """Return the last ``turns`` ``(role, content)`` pairs as an OpenAI-style
    messages payload for `chat_api_call`.

    Any role other than ``user``/``assistant``/``system`` (e.g. Console's
    ``tool`` role) maps to ``user`` -- a safe default most providers accept
    without additional tool-call-id plumbing this context-only call has no
    use for.
    """
    if turns <= 0:
        return []
    recent = list(messages)[-turns:]
    payload: list[dict[str, str]] = []
    for role, content in recent:
        mapped_role = role if role in ("user", "assistant", "system") else "user"
        payload.append({"role": mapped_role, "content": content})
    return payload


def _clean_llm_context_response(text: str) -> str | None:
    """Strip/validate a raw LLM response into a usable composed prompt.

    Collapses all whitespace (including newlines) into single spaces --
    enforcing "plain text, single paragraph" -- strips a single layer of
    wrapping quote characters some providers add despite instructions, then
    caps length at `_CONTEXT_LLM_MAX_RESPONSE_CHARS`. Returns ``None`` for
    an empty/whitespace-only input (refuse-empty).
    """
    collapsed = " ".join(text.split())
    if len(collapsed) >= 2 and collapsed[0] == collapsed[-1] and collapsed[0] in "\"'":
        collapsed = collapsed[1:-1].strip()
    if not collapsed:
        return None
    if len(collapsed) > _CONTEXT_LLM_MAX_RESPONSE_CHARS:
        collapsed = collapsed[:_CONTEXT_LLM_MAX_RESPONSE_CHARS].rstrip()
    return collapsed or None


def _call_chat_api_with_timeout(
    chat_call: Callable[..., Any], timeout_seconds: float, **kwargs: Any
) -> Any:
    """Bound a blocking ``chat_call(**kwargs)`` to ``timeout_seconds`` wall-clock.

    Runs ``chat_call`` on a dedicated single-worker thread pool and waits
    up to ``timeout_seconds`` for a result. On timeout, raises
    ``concurrent.futures.TimeoutError`` -- the orphaned call keeps running
    in the background (``shutdown(wait=False)``, never joined) rather than
    blocking the caller further: a plain ``with ThreadPoolExecutor():``
    block would otherwise wait for the hung call to finish anyway when the
    ``with`` exits, defeating the point of a hard timeout against an
    unresponsive provider/server.

    Args:
        chat_call: The blocking callable to run (typically
            ``chat_api_call``).
        timeout_seconds: Maximum seconds to wait for a result.
        **kwargs: Forwarded to ``chat_call``.

    Returns:
        Whatever ``chat_call`` returns.

    Raises:
        concurrent.futures.TimeoutError: ``chat_call`` did not finish
            within ``timeout_seconds``.
        Exception: Whatever ``chat_call`` itself raised.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(chat_call, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False)


def compose_llm_context_prompt(
    messages: Sequence[tuple[str, str]], options: LLMContextOptions
) -> str | None:
    """Attempt an LLM-composed conversation-context prompt; ``None`` on ANY failure.

    Never raises: kill-switch off, no ready provider, a `chat_call`
    exception, a timeout, or an empty/unusable response all fall through to
    ``None`` so the caller (`build_context_prompt_with_llm`) falls back to
    the existing keyword-shallow `build_context_prompt`. This is the
    load-bearing graceful-degradation seam for Task-559 AC1 — generation
    must never be blocked by this path.

    Only debug-logs failures, and never logs raw conversation content (a
    turn count at most) -- this is user conversation text, not diagnostic
    metadata.

    Args:
        messages: Chronological ``(role, content)`` pairs (already filtered
            to non-empty content by the caller).
        options: Resolved config + provider identity. See `LLMContextOptions`.

    Returns:
        The cleaned, single-paragraph composed prompt text, or ``None``.
    """
    if not options.enabled or not options.provider_ready or not options.api_endpoint:
        return None
    payload = _shape_llm_context_payload(messages, options.turns)
    if not payload:
        return None

    try:
        chat_call = options.chat_call
        if chat_call is None:
            from tldw_chatbook.Chat.Chat_Functions import (
                chat_api_call as _chat_api_call,
            )

            chat_call = _chat_api_call

        response = _call_chat_api_with_timeout(
            chat_call,
            options.timeout_seconds,
            api_endpoint=options.api_endpoint,
            messages_payload=payload,
            api_key=options.api_key,
            model=options.model,
            system_message=_CONTEXT_LLM_SYSTEM_PROMPT,
            streaming=False,
        )
    except Exception as exc:  # noqa: BLE001 - graceful fallback is load-bearing here
        logger.debug(
            f"generate-image: LLM context composition failed "
            f"({len(payload)} turns, provider={options.api_endpoint!r}): {exc!r}"
        )
        return None

    from tldw_chatbook.Chat.Chat_Functions import extract_response_content

    cleaned = _clean_llm_context_response(extract_response_content(response))
    if cleaned is None:
        logger.debug(
            "generate-image: LLM context composition returned an empty/unusable response"
        )
    return cleaned


def build_context_prompt_with_llm(
    messages: Sequence[tuple[str, str]],
    template: GenerationTemplate,
    llm_context: LLMContextOptions | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Compose from conversation context, preferring an LLM-composed prompt.

    Task-559 AC1's richer context builder. When ``llm_context`` is given,
    attempts `compose_llm_context_prompt` first; ANY failure there (kill
    switch off, no ready provider, call error, timeout, empty/garbage
    response) falls back to the pre-existing keyword-shallow
    `build_context_prompt` -- this function never raises and never
    dispatches worse than the original behavior. ``llm_context=None`` (the
    default) skips the LLM attempt entirely and behaves exactly like
    calling `build_context_prompt` directly, so every existing caller/test
    of the keyword path is unaffected.

    On an LLM success, the composed text is threaded through the SAME
    template-rendering + anchor-append pipeline (`_apply_template_with_
    anchor`) that the keyword path uses, standing in for the keyword
    extractor's ``last_message`` — so the "composed prompt visible in card"
    behavior and every template's ``context_mappings`` keep working
    unchanged; only the source of that anchor text gets richer.

    Args:
        messages: Chronological ``(role, content)`` pairs from the session.
        template: The style template to render.
        llm_context: Resolved LLM-path config/provider identity, or
            ``None`` to use the keyword extractor unconditionally.

    Returns:
        A ``(prompt, negative_prompt, params)`` tuple, or ``None`` when
        every message's content is empty/whitespace-only.
    """
    shaped_pairs = [
        (role, content) for role, content in messages if content and content.strip()
    ]
    if not shaped_pairs:
        return None
    if llm_context is not None:
        composed_text = compose_llm_context_prompt(shaped_pairs, llm_context)
        if composed_text is not None:
            context = {
                "last_message": composed_text,
                "mentioned_characters": [],
                "mentioned_settings": [],
                "mood": "",
                "style_hints": [],
            }
            return _apply_template_with_anchor(context, template)
    return build_context_prompt(shaped_pairs, template)


@dataclass(frozen=True)
class PreparedGeneration:
    """A validated, ready-to-dispatch ``/generate-image`` request.

    Args:
        prompt: Final prompt text to generate with — already composed with
            any resolved style template or conversation context. This is
            what both `run_generation_batch` and `generation_content_marker`
            must use.
        negative_prompt: Negative prompt from the applied template, or
            ``None`` for an unstyled, prompt-only request (the command
            grammar has no user-supplied negative-prompt syntax today).
        style_name: Display name of the style template applied
            (`GenerationTemplate.name`), or ``None`` for an unstyled
            request.
        width: Template-provided width override, or ``None``.
        height: Template-provided height override, or ``None``.
        steps: Template-provided sampling steps override, or ``None``.
        cfg_scale: Template-provided CFG scale override, or ``None``.
    """

    prompt: str
    negative_prompt: str | None
    style_name: str | None
    width: int | None
    height: int | None
    steps: int | None
    cfg_scale: float | None


@dataclass(frozen=True)
class GenerationRefusal:
    """A ``/generate-image`` invocation that must not dispatch a batch.

    Args:
        reason: Ready-to-display status text — the caller posts it verbatim
            as the Console system message and must leave the composer draft
            untouched.
    """

    reason: str


def prepare_generation_request(
    args: GenerateImageArgs,
    conversation_pairs: Sequence[tuple[str, str]],
    llm_context: LLMContextOptions | None = None,
) -> PreparedGeneration | GenerationRefusal:
    """Decide what one ``/generate-image`` invocation should generate.

    Pure decision logic extracted from the Console command handler so it is
    independently unit-testable — the handler itself just executes the
    result. Order of operations:

    1. An ``@style`` token (``args.style``), if present, is resolved first
       via `resolve_style_token` regardless of whether a prompt was also
       given. An ambiguous or unknown token refuses immediately — this
       never falls through to generation with no style applied.
    2. A non-empty prompt is composed against the resolved template via
       `compose_styled_request` (styled) or passed through unchanged
       (unstyled).
    3. An empty prompt falls back to the conversation: pairs with
       non-whitespace content are handed to `build_context_prompt_with_llm`
       (Task-559 AC1 — LLM-composed when ``llm_context`` is given and
       ready, else the keyword-shallow extractor) using the resolved style
       template, or ``chat_scene_visual`` when no ``@style`` was given. No
       usable conversation content — or no messages at all — refuses with
       the command's usage text.

    This function itself may block on network I/O when ``llm_context`` is
    given and its LLM path is attempted (`compose_llm_context_prompt`) —
    callers MUST run it off the UI loop (e.g.
    ``await asyncio.to_thread(prepare_generation_request, ...)``), exactly
    like `run_generation_batch`.

    Args:
        args: The parsed ``/generate-image`` invocation.
        conversation_pairs: The session's ``(role, content)`` pairs in
            chronological order. Only consulted on the no-prompt path.
        llm_context: Resolved LLM-composed-context config/provider
            identity (Task-559 AC1), or ``None`` to use the keyword
            extractor unconditionally. Only consulted on the no-prompt
            path.

    Returns:
        A `PreparedGeneration` ready to hand to `run_generation_batch`, or a
        `GenerationRefusal` carrying the status text to display — in which
        case the caller must not dispatch a batch or touch the composer
        draft.
    """
    style_template: GenerationTemplate | None = None
    if args.style:
        resolution = resolve_style_token(args.style)
        if resolution.ambiguous:
            ids = ", ".join(resolution.ambiguous)
            return GenerationRefusal(
                reason=(
                    f"Ambiguous style '@{args.style}' matches: {ids}. "
                    "Use one of these style ids."
                )
            )
        if resolution.template is None:
            valid_ids = ", ".join(sorted(get_all_templates()))
            return GenerationRefusal(
                reason=f"Unknown style '@{args.style}'. Valid styles: {valid_ids}"
            )
        style_template = resolution.template

    prompt = args.prompt.strip()
    if prompt:
        if style_template is not None:
            composed, negative, params = compose_styled_request(
                prompt, style_template
            )
            return PreparedGeneration(
                prompt=composed,
                negative_prompt=negative,
                style_name=style_template.name,
                width=params.get("width"),
                height=params.get("height"),
                steps=params.get("steps"),
                cfg_scale=params.get("cfg_scale"),
            )
        return PreparedGeneration(
            prompt=prompt,
            negative_prompt=None,
            style_name=None,
            width=None,
            height=None,
            steps=None,
            cfg_scale=None,
        )

    usable_pairs = [
        (role, content)
        for role, content in conversation_pairs
        if content and content.strip()
    ]
    if not usable_pairs:
        return GenerationRefusal(reason=GENERATE_IMAGE_USAGE_TEXT)

    template = style_template or get_template("chat_scene_visual")
    built = build_context_prompt_with_llm(usable_pairs, template, llm_context)
    if built is None:
        return GenerationRefusal(reason=GENERATE_IMAGE_USAGE_TEXT)
    composed, negative, params = built
    return PreparedGeneration(
        prompt=composed,
        negative_prompt=negative,
        style_name=template.name,
        width=params.get("width"),
        height=params.get("height"),
        steps=params.get("steps"),
        cfg_scale=params.get("cfg_scale"),
    )


@dataclass(frozen=True)
class BatchResult:
    """Outcome of one ``run_generation_batch`` call.

    Args:
        successes: Ordered ``(data, mime_type, meta)`` tuples for every
            variant that generated successfully, in generation order.
            Shaped exactly for
            ``ConsoleChatStore.append_generation_message(variants=...)``.
        errors: ``str(exception)`` for every variant that raised, in
            generation order. Empty when every variant succeeded.
    """

    successes: list[tuple[bytes, str, GenerationVariantMeta]]
    errors: list[str]


def run_generation_batch(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None,
    seed: int | None,
    count: int,
    style_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    generate: Callable[[Any], Any] | None = None,
    build: Callable[..., Any] | None = None,
) -> BatchResult:
    """Run one blocking batch of ``count`` image-generation variants.

    Blocking: every call to ``generate`` (default
    ``Image_Generation.worker.run_generation``) is synchronous and may hit
    the network or a local subprocess. Callers MUST run this off the UI
    loop (e.g. ``await asyncio.to_thread(run_generation_batch, ...)``) —
    this function performs no offloading itself.

    Applies the identical-image guard: when ``seed`` is an explicit value
    (not ``None``), only the first variant (index 0) uses it — every later
    variant is generated with ``seed=-1`` so a batch of N never produces N
    copies of the same image. A ``None`` seed (the common case — no
    explicit seed configured) is passed through unchanged to every variant,
    since there is nothing to force away from.

    A per-variant failure (``generate`` raising) is caught and recorded in
    ``errors``; it never aborts the remaining variants in the batch.

    Args:
        backend: Backend id to generate with (already resolved/validated
            by the caller).
        prompt: Positive prompt text, shared by every variant.
        negative_prompt: Optional negative prompt, shared by every variant.
        seed: Optional explicit seed for variant 0; ``None`` for no
            explicit seed (every variant generates with ``seed=None``).
        count: Number of variants to generate (``>= 1``).
        style_name: Style label to record on each variant's
            `GenerationVariantMeta.style` (typically a resolved template's
            display name). ``None`` for an unstyled/custom request.
        width: Optional image width, threaded into every `build` call.
        height: Optional image height, threaded into every `build` call.
        steps: Optional sampling steps, threaded into every `build` call.
        cfg_scale: Optional CFG scale, threaded into every `build` call.
        generate: Blocking single-request entry point. Defaults to
            ``Image_Generation.worker.run_generation``, imported lazily.
        build: Request builder. Defaults to
            ``Image_Generation.worker.build_request``, imported lazily.

    Returns:
        A `BatchResult` with every successful variant's
        ``(data, mime_type, meta)`` plus every failure's error string.
    """
    if generate is None or build is None:
        from tldw_chatbook.Image_Generation import worker as _worker

        if generate is None:
            generate = _worker.run_generation
        if build is None:
            build = _worker.build_request

    successes: list[tuple[bytes, str, GenerationVariantMeta]] = []
    errors: list[str] = []
    for index in range(count):
        variant_seed = seed if (index == 0 or seed is None) else -1
        try:
            request = build(
                backend=backend,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=variant_seed,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
            )
            result = generate(request)
        except Exception as exc:  # noqa: BLE001 - collected per-variant, never aborts the batch
            errors.append(str(exc))
            continue
        # task-558: prefer a resolved seed/model the adapter's result
        # reports over the request's own values, when present. `getattr`
        # (not attribute access) tolerates both a real `ImageGenResult`
        # (which always carries these, defaulting to ``None``) and the
        # minimal test doubles used across this module's test suite that
        # predate these fields entirely.
        resolved_seed = getattr(result, "resolved_seed", None)
        resolved_model = getattr(result, "resolved_model", None)
        meta = GenerationVariantMeta(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            backend=backend,
            model=resolved_model,
            seed=resolved_seed if resolved_seed is not None else variant_seed,
            style=style_name,
            params={},
        )
        successes.append((result.content, result.content_type, meta))
    return BatchResult(successes=successes, errors=errors)
