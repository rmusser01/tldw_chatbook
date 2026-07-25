"""Tests for console_generate_image pure helpers."""

import dataclasses
import threading
import time

import pytest
from tldw_chatbook.Chat import console_generate_image as module
from tldw_chatbook.Chat.console_generate_image import (
    GENERATE_IMAGE_USAGE_TEXT,
    GenerateImageArgs,
    GenerationRefusal,
    LLMContextOptions,
    PreparedGeneration,
    build_context_prompt,
    build_context_prompt_with_llm,
    clamp_initial_batch,
    compose_llm_context_prompt,
    compose_styled_request,
    insert_style_token_into_draft,
    parse_generate_image_args,
    generation_content_marker,
    prepare_generation_request,
    resolve_style_token,
    run_generation_batch,
)
from tldw_chatbook.Chat.console_generate_image import (
    LLMContextExecutorSaturatedError,
    _CONTEXT_LLM_COMPOSE_INSTRUCTION,
    _clean_llm_context_response,
    _shape_llm_context_payload,
    reset_llm_context_executor,
)
from tldw_chatbook.Media_Creation.generation_templates import (
    BUILTIN_TEMPLATES,
    get_template,
)


@pytest.fixture(autouse=True)
def _reset_llm_context_executor_state():
    """The shared single-worker LLM-context executor (Qodo PR #867 fix) is
    process-wide/module-level state -- reset it around every test so a
    still-"running" fake-slow-call from one test can never bleed a spurious
    `LLMContextExecutorSaturatedError` into an unrelated later test."""
    reset_llm_context_executor()
    yield
    reset_llm_context_executor()


@pytest.mark.parametrize(
    "args,backend,style,prompt",
    [
        ("a red dragon", None, None, "a red dragon"),
        (":swarmui a red dragon", "swarmui", None, "a red dragon"),
        (":openrouter   spaced  prompt ", "openrouter", None, "spaced  prompt"),
        (":swarmui", "swarmui", None, ""),
        ("", None, None, ""),
        ("   ", None, None, ""),
        (": lonely colon", None, None, ": lonely colon"),  # bare ':' is not a backend token
        ("@anime dragon", None, "anime", "dragon"),
        (":swarmui @anime dragon", "swarmui", "anime", "dragon"),
        ("@anime :swarmui dragon", "swarmui", "anime", "dragon"),
        ("@ lonely at", None, None, "@ lonely at"),  # bare '@' is not a style token
        ("@unknown x", None, "unknown", "x"),  # unresolved style token passes through raw
    ],
)
def test_parse_table(args, backend, style, prompt):
    """Test parse_generate_image_args against spec table."""
    parsed = parse_generate_image_args(args)
    assert (parsed.backend, parsed.style, parsed.prompt) == (backend, style, prompt)


def test_resolve_style_exact_id():
    """An exact (case-insensitive) template id resolves directly."""
    res = resolve_style_token("STYLE_anime")
    assert res.template is not None
    assert res.template.id == "style_anime"
    assert res.ambiguous == ()


def test_resolve_style_exact_name_spaces_and_underscores():
    """Exact name matching treats spaces and underscores as interchangeable."""
    assert resolve_style_token("anime style").template.id == "style_anime"
    assert resolve_style_token("Anime_Style").template.id == "style_anime"


def test_resolve_style_unique_prefix():
    """A prefix unique across ids+names resolves to the one match."""
    res = resolve_style_token("waterc")
    assert res.template is not None
    assert res.template.id == "style_watercolor"
    assert res.ambiguous == ()


def test_resolve_style_ambiguous_prefix_lists_matched_ids():
    """A prefix matching several templates is ambiguous and lists the ids."""
    res = resolve_style_token("style_")
    assert res.template is None
    assert res.ambiguous == ("style_anime", "style_cyberpunk", "style_watercolor")


def test_resolve_style_unknown_token():
    """A token matching nothing resolves to no template and no ambiguity."""
    res = resolve_style_token("totally_not_a_style")
    assert res.template is None
    assert res.ambiguous == ()


def test_compose_styled_request_substitutes_placeholder():
    """The user prompt fills every context-mapped placeholder."""
    template = get_template("quick_simple")
    prompt, negative, params = compose_styled_request("a red dragon", template)
    assert prompt == "a red dragon"
    assert negative == "low quality"
    assert params == {"width": 512, "height": 512, "steps": 15, "cfg_scale": 7.0}


def test_compose_styled_request_appends_when_prompt_unconsumed():
    """When the template doesn't consume the prompt, it is appended (invariant)."""
    base_template = get_template("style_anime")
    template = dataclasses.replace(base_template, context_mappings={})
    prompt, negative, params = compose_styled_request("a dragon", template)
    assert prompt == (
        "anime style, detailed, vibrant colors, high quality anime art, a dragon"
    )
    assert prompt.endswith("a dragon")
    assert negative == base_template.negative_prompt
    assert params == base_template.default_params


@pytest.mark.parametrize(
    "default_batch,max_variants,expected",
    [
        (4, 8, 4),  # default is smaller
        (12, 8, 8),  # cap is smaller
        (5, 5, 5),  # equal
        (1, 1, 1),  # both minimum
    ],
)
def test_clamp_initial_batch(default_batch, max_variants, expected):
    """Test clamp_initial_batch returns minimum of the two values."""
    result = clamp_initial_batch(default_batch, max_variants)
    assert result == expected


def test_content_marker_no_trim():
    """Test generation_content_marker with short prompt."""
    assert generation_content_marker("a red dragon") == "[image] a red dragon"


def test_content_marker_trims():
    """Test generation_content_marker with long prompt."""
    long = "x" * 200
    marker = generation_content_marker(long)
    assert marker.startswith("[image] ")
    assert len(marker) <= 8 + 80 + 1  # [image] (8) + limit (80) + ellipsis (1)
    assert marker.endswith("…")


def test_content_marker_default_limit():
    """Test that default limit is 80."""
    # Prompt of exactly 81 chars should be trimmed
    prompt = "a" * 81
    marker = generation_content_marker(prompt)
    assert marker.endswith("…")


def test_content_marker_exact_80_char_boundary_not_trimmed():
    """task-558: the boundary itself -- a prompt of EXACTLY the 80-char
    default limit -- must render whole, no ellipsis. `len(flattened) >
    limit` is a strict `>`, so 80 must be the last un-trimmed length; only
    `test_content_marker_default_limit` (81, one past the boundary) was
    covered before this."""
    prompt = "a" * 80
    marker = generation_content_marker(prompt)
    assert marker == "[image] " + "a" * 80
    assert not marker.endswith("…")


def test_content_marker_custom_limit():
    """Test generation_content_marker with custom limit."""
    prompt = "a" * 50
    marker = generation_content_marker(prompt, limit=40)
    assert marker == "[image] " + "a" * 39 + "…"


def test_content_marker_whitespace_normalize():
    """Test that newlines are collapsed to spaces."""
    prompt = "a\n\nb\n\nc"
    marker = generation_content_marker(prompt)
    assert marker == "[image] a b c"


class _Res:
    """Minimal ImageGenResult-shaped fake for run_generation_batch tests."""

    def __init__(self, b):
        self.content = b
        self.content_type = "image/png"
        self.bytes_len = len(b)


def test_batch_all_succeed():
    """All variants generate successfully -> no errors, N successes."""
    calls = []

    def gen(req):
        calls.append(req)
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=2, generate=gen,
    )
    assert len(out.successes) == 2 and out.errors == []


def test_batch_partial_failure_keeps_successes():
    """One variant raises -> the rest still succeed and are kept."""
    n = {"i": 0}

    def gen(req):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("boom")
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=3, generate=gen,
    )
    assert len(out.successes) == 2 and len(out.errors) == 1


def test_batch_explicit_seed_only_first_variant():
    """An explicit seed only applies to variant 0; later variants force -1."""
    seeds = []

    def gen(req):
        seeds.append(req.seed)
        return _Res(b"img")

    run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=1234, count=3, generate=gen,
    )
    assert seeds == [1234, -1, -1]  # identical-image guard


def test_batch_build_exception_collected():
    """When build raises on a variant, the error is collected and batch continues."""
    n = {"i": 0}

    def build_fn(*args, **kwargs):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("build boom")
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=3, generate=gen, build=build_fn,
    )
    # First and third variants succeed (build calls 1 and 3).
    # Second fails during build (call 2).
    assert len(out.successes) == 2 and len(out.errors) == 1
    assert "build boom" in out.errors[0]


def test_batch_threads_style_and_template_params():
    """style_name/width/height/steps/cfg_scale reach build() and the variant meta."""
    captured = []

    def build_fn(**kwargs):
        captured.append(kwargs)
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    template = get_template("style_anime")
    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=1, style_name=template.name,
        width=template.default_params["width"],
        height=template.default_params["height"],
        steps=template.default_params["steps"],
        cfg_scale=template.default_params["cfg_scale"],
        generate=gen, build=build_fn,
    )
    assert captured[0]["width"] == template.default_params["width"]
    assert captured[0]["height"] == template.default_params["height"]
    assert captured[0]["steps"] == template.default_params["steps"]
    assert captured[0]["cfg_scale"] == template.default_params["cfg_scale"]
    assert len(out.successes) == 1
    meta = out.successes[0][2]
    assert meta.style == template.name


def test_batch_default_style_and_dims_are_none():
    """Without style/dims arguments, meta.style is None and build() receives None dims."""
    captured = []

    def build_fn(**kwargs):
        captured.append(kwargs)
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=1, generate=gen, build=build_fn,
    )
    assert captured[0]["width"] is None
    assert captured[0]["height"] is None
    assert captured[0]["steps"] is None
    assert captured[0]["cfg_scale"] is None
    assert out.successes[0][2].style is None


class _ResWithResolved(_Res):
    """ImageGenResult-shaped fake also carrying resolved_seed/resolved_model
    (task-558) -- distinct from the plain `_Res` fake every other batch
    test above uses, so those keep exercising the no-such-attribute
    fallback path unmodified."""

    def __init__(self, b, *, resolved_seed=None, resolved_model=None):
        super().__init__(b)
        self.resolved_seed = resolved_seed
        self.resolved_model = resolved_model


def test_batch_uses_resolved_seed_and_model_when_reported():
    """task-558: when the adapter's result reports a resolved seed/model,
    the variant meta uses those instead of the request's own seed/``None``."""

    def gen(req):
        return _ResWithResolved(b"img", resolved_seed=999, resolved_model="sdxl")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=-1, count=1, generate=gen,
    )
    assert len(out.successes) == 1
    meta = out.successes[0][2]
    assert meta.seed == 999
    assert meta.model == "sdxl"


def test_batch_falls_back_to_variant_seed_when_result_has_no_resolved_fields():
    """Regression guard: a plain ImageGenResult-shaped fake with no
    resolved_seed/resolved_model attributes at all (the existing `_Res`
    fake every other batch test above uses) must not raise, and must
    preserve prior behavior -- meta.seed is the request's own variant seed,
    meta.model stays ``None``."""

    def gen(req):
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=42, count=1, generate=gen,
    )
    meta = out.successes[0][2]
    assert meta.seed == 42
    assert meta.model is None


def test_batch_resolved_seed_none_falls_back_to_variant_seed():
    """A result that carries the attribute but leaves it ``None`` (every
    real adapter's default today) still falls back to the variant's own
    seed -- never ``None`` when a real request seed was used."""

    def gen(req):
        return _ResWithResolved(b"img", resolved_seed=None, resolved_model=None)

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=7, count=1, generate=gen,
    )
    meta = out.successes[0][2]
    assert meta.seed == 7
    assert meta.model is None


# --- build_context_prompt ---------------------------------------------


def test_build_context_prompt_composes_from_pairs():
    """The last user message anchors the composed prompt via the template."""
    template = get_template("chat_scene_visual")
    pairs = [
        ("assistant", "Welcome to the tavern."),
        ("user", "a dimly lit tavern with a roaring fireplace"),
    ]
    result = build_context_prompt(pairs, template)
    assert result is not None
    prompt, negative, params = result
    assert "a dimly lit tavern with a roaring fireplace" in prompt
    assert negative == template.negative_prompt
    assert params == template.default_params


def test_build_context_prompt_mood_and_hints_path():
    """Multi-message context still resolves; mood/hint extraction doesn't crash
    and the last user message remains the anchor."""
    template = get_template("chat_scene_visual")
    pairs = [
        ("user", "The forest grows dark and gloomy under the shadow."),
        ("assistant", "A wolf howls somewhere in the distance."),
        ("user", "a mysterious shrine standing in the mist"),
    ]
    result = build_context_prompt(pairs, template)
    assert result is not None
    prompt, _negative, _params = result
    assert "a mysterious shrine standing in the mist" in prompt


def test_build_context_prompt_none_on_no_pairs():
    """No pairs at all -> None (nothing to build a prompt from)."""
    template = get_template("chat_scene_visual")
    assert build_context_prompt([], template) is None


def test_build_context_prompt_none_on_whitespace_only_pairs():
    """Every pair is empty/whitespace-only content -> None."""
    template = get_template("chat_scene_visual")
    pairs = [("user", "   "), ("assistant", "")]
    assert build_context_prompt(pairs, template) is None


def test_build_context_prompt_anchor_guard_appends_last_message():
    """When the template consumes nothing (unrecognized id -> apply_template_to_prompt
    yields an empty base), the last-message anchor is still appended rather than
    silently dropped.

    Note: unlike `compose_styled_request`'s equivalent test,
    `context_mappings={}` alone can't trigger this for `build_context_prompt` --
    `apply_template_to_prompt` re-fetches the REAL registered template by id
    and applies ITS OWN (unmodified) mappings, and every built-in template maps
    something to ``last_message``. An unregistered id is what actually drives
    `apply_template_to_prompt` to return nothing to consume.
    """
    base = get_template("chat_scene_visual")
    template = dataclasses.replace(base, id="not_a_registered_template_id")
    pairs = [("user", "a spooky graveyard at midnight")]
    prompt, negative, params = build_context_prompt(pairs, template)
    assert prompt == "a spooky graveyard at midnight"
    assert negative == ""
    assert params == {}


def test_build_context_prompt_explicit_style_beats_default():
    """An explicit style template is used verbatim, not swapped for chat_scene_visual."""
    template = get_template("style_anime")
    pairs = [("user", "a lone samurai in the rain")]
    prompt, negative, _params = build_context_prompt(pairs, template)
    assert "a lone samurai in the rain" in prompt
    assert "anime style" in prompt
    assert negative == template.negative_prompt


# --- prepare_generation_request -----------------------------------------


def test_prepare_styled_prompt_composes_and_carries_template_params():
    """Prompt + resolved @style -> composed prompt, template negative/params/style name."""
    args = GenerateImageArgs(backend=None, prompt="a red dragon", style="anime")
    result = prepare_generation_request(args, [])
    assert isinstance(result, PreparedGeneration)
    template = get_template("style_anime")
    assert "a red dragon" in result.prompt
    assert result.style_name == template.name
    assert result.negative_prompt == template.negative_prompt
    assert result.width == template.default_params["width"]
    assert result.height == template.default_params["height"]
    assert result.steps == template.default_params["steps"]
    assert result.cfg_scale == template.default_params["cfg_scale"]


def test_prepare_plain_prompt_passthrough():
    """Prompt with no style -> passthrough; every other field is None."""
    args = GenerateImageArgs(backend=None, prompt="a red dragon", style=None)
    result = prepare_generation_request(args, [])
    assert result == PreparedGeneration(
        prompt="a red dragon",
        negative_prompt=None,
        style_name=None,
        width=None,
        height=None,
        steps=None,
        cfg_scale=None,
    )


def test_prepare_unknown_style_refusal_lists_all_ids():
    """An unresolvable @style token refuses and lists every valid style id."""
    args = GenerateImageArgs(backend=None, prompt="a red dragon", style="nope")
    result = prepare_generation_request(args, [])
    assert isinstance(result, GenerationRefusal)
    assert "Unknown style" in result.reason
    for template_id in BUILTIN_TEMPLATES:
        assert template_id in result.reason


def test_prepare_ambiguous_style_refusal_lists_matches():
    """An ambiguous @style prefix refuses and lists only the matched ids."""
    args = GenerateImageArgs(backend=None, prompt="a red dragon", style="style_")
    result = prepare_generation_request(args, [])
    assert isinstance(result, GenerationRefusal)
    assert "style_anime" in result.reason
    assert "style_cyberpunk" in result.reason
    assert "style_watercolor" in result.reason


def test_prepare_empty_prompt_empty_conversation_usage_refusal():
    """No prompt and no usable conversation content -> the usage line."""
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    result = prepare_generation_request(args, [])
    assert result == GenerationRefusal(reason=GENERATE_IMAGE_USAGE_TEXT)


def test_prepare_empty_prompt_whitespace_only_conversation_usage_refusal():
    """Conversation pairs exist but every content is whitespace-only -> usage line."""
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    result = prepare_generation_request(args, [("user", "   "), ("assistant", "")])
    assert result == GenerationRefusal(reason=GENERATE_IMAGE_USAGE_TEXT)


def test_prepare_empty_prompt_with_content_uses_default_context_style():
    """No prompt but usable conversation content -> context path, chat_scene_visual default."""
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    pairs = [("user", "a quiet lakeside cabin at dawn")]
    result = prepare_generation_request(args, pairs)
    assert isinstance(result, PreparedGeneration)
    template = get_template("chat_scene_visual")
    assert result.style_name == template.name
    assert "a quiet lakeside cabin at dawn" in result.prompt


def test_prepare_empty_prompt_with_content_and_style_uses_that_style():
    """No prompt, usable content, AND an explicit @style -> that style's template, not the default."""
    args = GenerateImageArgs(backend=None, prompt="", style="anime")
    pairs = [("user", "a quiet lakeside cabin at dawn")]
    result = prepare_generation_request(args, pairs)
    assert isinstance(result, PreparedGeneration)
    template = get_template("style_anime")
    assert result.style_name == template.name
    assert "a quiet lakeside cabin at dawn" in result.prompt


# ---------------------------------------------------------------------------
# insert_style_token_into_draft (Task 4 style-picker insert, Major review fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft,style_id,expected",
    [
        # Draft already starts with the command word: insert @style right
        # after the command word, in front of the prompt remainder.
        ("/generate-image a dragon", "style_anime", "/generate-image @style_anime a dragon"),
        # A leading :backend token stays exactly where it was; the new
        # style token is inserted right after it.
        (
            "/generate-image :swarmui a dragon",
            "style_anime",
            "/generate-image :swarmui @style_anime a dragon",
        ),
        # An existing leading @style token is REPLACED, not stacked --
        # closes the undisclosed last-wins double-style edge.
        (
            "/generate-image :swarmui @old a dragon",
            "style_anime",
            "/generate-image :swarmui @style_anime a dragon",
        ),
        (
            "/generate-image @old a dragon",
            "style_anime",
            "/generate-image @style_anime a dragon",
        ),
        # Bare command word, no args at all.
        ("/generate-image", "style_anime", "/generate-image @style_anime "),
        # Bare command word with only trailing whitespace.
        ("/generate-image   ", "style_anime", "/generate-image @style_anime "),
        # Any other draft (including empty) is prefixed as the whole prompt.
        ("", "style_anime", "/generate-image @style_anime "),
        ("a dragon", "style_anime", "/generate-image @style_anime a dragon"),
        # A different command word is not recognized -- treated as plain
        # prompt text and prefixed like any other draft.
        (
            "/other-command a dragon",
            "style_anime",
            "/generate-image @style_anime /other-command a dragon",
        ),
        # Command word prefix must be followed by whitespace or end, not
        # just be a string prefix of a longer word.
        (
            "/generate-imagex a dragon",
            "style_anime",
            "/generate-image @style_anime /generate-imagex a dragon",
        ),
    ],
)
def test_insert_style_token_into_draft_table(draft, style_id, expected):
    assert insert_style_token_into_draft(draft, style_id) == expected


def test_insert_style_token_into_draft_applying_twice_yields_one_token():
    """Idempotent-ish: a second insert with a different id replaces, not stacks."""
    once = insert_style_token_into_draft("", "style_anime")
    twice = insert_style_token_into_draft(once, "style_watercolor")
    assert twice == "/generate-image @style_watercolor "
    assert twice.count("@style_") == 1


def test_insert_style_token_into_draft_applying_twice_with_prompt_text():
    once = insert_style_token_into_draft("a dragon", "style_anime")
    twice = insert_style_token_into_draft(once, "style_watercolor")
    assert twice == "/generate-image @style_watercolor a dragon"
    assert twice.count("@style_") == 1


# ---------------------------------------------------------------------------
# Task-559 AC1: LLM-composed conversation-context prompt
# ---------------------------------------------------------------------------


def _llm_options(*, chat_call=None, **overrides) -> LLMContextOptions:
    defaults = dict(
        enabled=True,
        turns=10,
        timeout_seconds=5.0,
        provider_ready=True,
        api_endpoint="openai",
        model="gpt-4o-mini",
        api_key="fake-test-api-key",
    )
    defaults.update(overrides)
    return LLMContextOptions(chat_call=chat_call, **defaults)


def _chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


# --- _shape_llm_context_payload -------------------------------------------


def test_shape_llm_context_payload_keeps_last_n_turns_in_order():
    pairs = [("user", f"turn {i}") for i in range(15)]
    payload = _shape_llm_context_payload(pairs, 3)
    assert payload == [
        {"role": "user", "content": "turn 12"},
        {"role": "user", "content": "turn 13"},
        {"role": "user", "content": "turn 14"},
    ]


def test_shape_llm_context_payload_maps_unknown_role_to_user():
    payload = _shape_llm_context_payload([("tool", "result text")], 10)
    assert payload == [{"role": "user", "content": "result text"}]


def test_shape_llm_context_payload_zero_or_negative_turns_is_empty():
    pairs = [("user", "hello")]
    assert _shape_llm_context_payload(pairs, 0) == []
    assert _shape_llm_context_payload(pairs, -1) == []


# --- _clean_llm_context_response -------------------------------------------


def test_clean_llm_context_response_collapses_whitespace_and_newlines():
    raw = "  A knight   \n  in a cave.\n\nDramatic light.  "
    assert _clean_llm_context_response(raw) == "A knight in a cave. Dramatic light."


def test_clean_llm_context_response_strips_wrapping_quotes():
    assert _clean_llm_context_response('"a red dragon"') == "a red dragon"
    assert _clean_llm_context_response("'a red dragon'") == "a red dragon"


def test_clean_llm_context_response_empty_is_none():
    assert _clean_llm_context_response("") is None
    assert _clean_llm_context_response("   \n  ") is None


def test_clean_llm_context_response_truncates_to_cap():
    raw = "x" * 900
    cleaned = _clean_llm_context_response(raw)
    assert cleaned is not None
    assert len(cleaned) == 500


# --- compose_llm_context_prompt: fallback matrix ---------------------------


def test_compose_llm_context_prompt_disabled_returns_none():
    options = _llm_options(enabled=False, chat_call=lambda **_: _chat_response("x"))
    assert compose_llm_context_prompt([("user", "hi")], options) is None


def test_compose_llm_context_prompt_not_ready_returns_none():
    options = _llm_options(
        provider_ready=False, chat_call=lambda **_: _chat_response("x")
    )
    assert compose_llm_context_prompt([("user", "hi")], options) is None


def test_compose_llm_context_prompt_no_api_endpoint_returns_none():
    options = _llm_options(
        api_endpoint=None, chat_call=lambda **_: _chat_response("x")
    )
    assert compose_llm_context_prompt([("user", "hi")], options) is None


def test_compose_llm_context_prompt_empty_messages_returns_none():
    options = _llm_options(chat_call=lambda **_: _chat_response("x"))
    assert compose_llm_context_prompt([], options) is None


def test_compose_llm_context_prompt_call_raises_returns_none():
    def _raising(**_kwargs):
        raise RuntimeError("provider unreachable")

    options = _llm_options(chat_call=_raising)
    assert compose_llm_context_prompt([("user", "a dragon")], options) is None


def test_compose_llm_context_prompt_timeout_returns_none():
    def _slow(**_kwargs):
        time.sleep(0.3)
        return _chat_response("too late")

    options = _llm_options(chat_call=_slow, timeout_seconds=0.02)
    assert compose_llm_context_prompt([("user", "a dragon")], options) is None


def test_compose_llm_context_prompt_empty_response_returns_none():
    options = _llm_options(chat_call=lambda **_: _chat_response("   "))
    assert compose_llm_context_prompt([("user", "a dragon")], options) is None


def test_compose_llm_context_prompt_garbage_response_shape_returns_none():
    """A response that doesn't match the expected shape yields no content, not a crash."""
    options = _llm_options(chat_call=lambda **_: {"unexpected": "shape"})
    assert compose_llm_context_prompt([("user", "a dragon")], options) is None


def test_compose_llm_context_prompt_success_returns_cleaned_text():
    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _chat_response(" A knight in a glowing cave, dramatic torchlight. ")

    options = _llm_options(chat_call=_fake_call, turns=2)
    pairs = [
        ("user", "A knight enters a cave."),
        ("assistant", "Crystals glow on the walls."),
        ("user", "He raises his torch."),
    ]
    result = compose_llm_context_prompt(pairs, options)
    assert result == "A knight in a glowing cave, dramatic torchlight."
    # Only the last `turns` (2) pairs were sent, non-streaming, to the
    # resolved provider, followed by the task-622 compose instruction as a
    # final user-role message (see the trailing-role invariant tests below).
    assert captured["messages_payload"] == [
        {"role": "assistant", "content": "Crystals glow on the walls."},
        {"role": "user", "content": "He raises his torch."},
        {"role": "user", "content": _CONTEXT_LLM_COMPOSE_INSTRUCTION},
    ]
    assert captured["api_endpoint"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "fake-test-api-key"
    assert captured["streaming"] is False
    assert "system_message" in captured and captured["system_message"]


# --- Task-622: composition payload never ends on an assistant turn --------


def test_compose_llm_context_prompt_trailing_assistant_appends_user_instruction():
    """The normal shape: user asks, assistant answers, user runs
    `/generate-image` -- llama.cpp with `enable_thinking` rejects a payload
    that ends on `assistant` as an invalid response prefill. The payload
    handed to `chat_call` must end with a user-role message carrying the
    compose instruction."""
    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _chat_response("a knight in a cave")

    options = _llm_options(chat_call=_fake_call, turns=10)
    pairs = [
        ("user", "A knight enters a cave."),
        ("assistant", "Crystals glow on the walls."),
    ]
    result = compose_llm_context_prompt(pairs, options)
    assert result == "a knight in a cave"
    assert captured["messages_payload"] == [
        {"role": "user", "content": "A knight enters a cave."},
        {"role": "assistant", "content": "Crystals glow on the walls."},
        {"role": "user", "content": _CONTEXT_LLM_COMPOSE_INSTRUCTION},
    ]
    assert captured["messages_payload"][-1]["role"] == "user"


def test_compose_llm_context_prompt_trailing_user_still_appends_separate_instruction():
    """A conversation that already ends on a user turn (e.g. the user
    typed `/generate-image` right after their own message, no assistant
    reply yet) still gets the instruction appended as its OWN final
    message -- it is never merged into the prior user turn's content."""
    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _chat_response("a dragon over a village")

    options = _llm_options(chat_call=_fake_call, turns=10)
    pairs = [
        ("assistant", "Crystals glow on the walls."),
        ("user", "He raises his torch."),
    ]
    result = compose_llm_context_prompt(pairs, options)
    assert result == "a dragon over a village"
    assert captured["messages_payload"] == [
        {"role": "assistant", "content": "Crystals glow on the walls."},
        {"role": "user", "content": "He raises his torch."},
        {"role": "user", "content": _CONTEXT_LLM_COMPOSE_INSTRUCTION},
    ]
    assert captured["messages_payload"][-1]["role"] == "user"
    # The instruction is a distinct message, not merged into the prior turn.
    assert captured["messages_payload"][-2]["content"] == "He raises his torch."


def test_compose_llm_context_prompt_single_turn_appends_instruction():
    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _chat_response("a lone dragon")

    options = _llm_options(chat_call=_fake_call, turns=10)
    result = compose_llm_context_prompt([("assistant", "A dragon appears.")], options)
    assert result == "a lone dragon"
    assert captured["messages_payload"] == [
        {"role": "assistant", "content": "A dragon appears."},
        {"role": "user", "content": _CONTEXT_LLM_COMPOSE_INSTRUCTION},
    ]


def test_compose_llm_context_prompt_instruction_survives_turns_truncation():
    """The turns-window truncation happens BEFORE appending the
    instruction, so a small `turns` value never truncates the instruction
    itself away."""
    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _chat_response("a single scene")

    options = _llm_options(chat_call=_fake_call, turns=1)
    pairs = [
        ("user", "turn 1"),
        ("assistant", "turn 2"),
        ("assistant", "turn 3 (last raw turn)"),
    ]
    result = compose_llm_context_prompt(pairs, options)
    assert result == "a single scene"
    # turns=1 keeps only the last raw pair, then the instruction is appended
    # -- two messages total, not one, and the instruction was not dropped.
    assert captured["messages_payload"] == [
        {"role": "assistant", "content": "turn 3 (last raw turn)"},
        {"role": "user", "content": _CONTEXT_LLM_COMPOSE_INSTRUCTION},
    ]


def test_compose_llm_context_prompt_empty_after_truncation_still_returns_none():
    """turns<=0 (or an empty conversation) yields an empty truncated window
    -- no LLM call is made at all (nothing changed here by task-622), so
    there's no trailing-role invariant to violate in the first place."""
    options = _llm_options(
        chat_call=lambda **_: _chat_response("should not be called"), turns=0
    )
    assert compose_llm_context_prompt([("user", "hi")], options) is None


def test_compose_llm_context_prompt_never_raises_on_unexpected_executor_error(
    monkeypatch,
):
    """Even a failure inside the timeout-bounding machinery itself degrades to None."""
    import tldw_chatbook.Chat.console_generate_image as module

    def _boom(*_a, **_k):
        raise OSError("thread pool exhausted")

    monkeypatch.setattr(module.concurrent.futures, "ThreadPoolExecutor", _boom)
    options = _llm_options(chat_call=lambda **_: _chat_response("x"))
    assert compose_llm_context_prompt([("user", "a dragon")], options) is None


# --- shared executor: bounded accumulation (Qodo PR #867) -------------------


def test_second_call_while_first_still_hung_fails_fast_without_reinvoking():
    """A fresh-per-call executor that `shutdown(wait=False)` on timeout would
    accumulate one abandoned thread per timed-out attempt, unbounded over a
    session. The shared single-worker executor bounds that at exactly one:
    a second call while the first (timed-out-but-still-running) call still
    occupies that one worker must fail FAST -- no queueing, no re-invoking
    `chat_call`, and no waiting anywhere near either call's own timeout."""
    calls: list = []
    release = threading.Event()

    def _hang(**kwargs):
        calls.append(kwargs)
        release.wait(2.0)
        return _chat_response("finally back")

    options = _llm_options(chat_call=_hang, timeout_seconds=0.05)
    try:
        start = time.monotonic()
        first = compose_llm_context_prompt([("user", "a dragon")], options)
        first_elapsed = time.monotonic() - start
        assert first is None
        assert first_elapsed < 0.5
        assert len(calls) == 1  # the hung call genuinely started running

        start = time.monotonic()
        second = compose_llm_context_prompt([("user", "another dragon")], options)
        second_elapsed = time.monotonic() - start
        assert second is None
        # Fails fast: nowhere near the 0.05s timeout, let alone queued behind
        # the still-hung first call indefinitely.
        assert second_elapsed < 0.02
        assert len(calls) == 1  # chat_call was NOT invoked a second time
    finally:
        release.set()  # let the hung fake finish so nothing leaks past this test


def test_executor_recovers_once_the_stuck_future_finally_completes():
    """Saturation is temporary, not a permanent lockout: once the abandoned
    call actually finishes, the shared executor's one worker is free again
    and a subsequent call proceeds normally."""
    calls: list = []
    release = threading.Event()

    def _hang_then_return(**kwargs):
        calls.append(kwargs)
        release.wait(2.0)
        return _chat_response("finally back")

    options = _llm_options(chat_call=_hang_then_return, timeout_seconds=0.05)
    assert compose_llm_context_prompt([("user", "x")], options) is None  # times out
    assert compose_llm_context_prompt([("user", "y")], options) is None  # saturated
    assert len(calls) == 1

    release.set()
    for _ in range(50):  # bounded poll, not a fixed sleep -- avoids flakiness
        if not module._LLM_CONTEXT_INFLIGHT_FUTURE.done():
            time.sleep(0.02)
            continue
        break

    third = compose_llm_context_prompt([("user", "z")], options)
    assert third == "finally back"
    assert len(calls) == 2


def test_call_chat_api_with_timeout_submit_raises_saturated_error_directly():
    """Unit-level pin on `_submit_llm_context_call`'s contract, independent
    of `compose_llm_context_prompt`'s blanket exception handling."""
    module.reset_llm_context_executor()
    release = threading.Event()

    def _hang(**_kwargs):
        release.wait(2.0)
        return None

    try:
        first_future = module._submit_llm_context_call(_hang)
        assert not first_future.done()
        with pytest.raises(LLMContextExecutorSaturatedError):
            module._submit_llm_context_call(_hang)
    finally:
        release.set()


# --- build_context_prompt_with_llm -----------------------------------------


def test_build_context_prompt_with_llm_none_options_matches_keyword_path():
    """`llm_context=None` behaves exactly like calling `build_context_prompt` directly."""
    template = get_template("chat_scene_visual")
    pairs = [("user", "a dimly lit tavern with a roaring fireplace")]
    assert build_context_prompt_with_llm(pairs, template, None) == build_context_prompt(
        pairs, template
    )


def test_build_context_prompt_with_llm_empty_pairs_returns_none_regardless():
    template = get_template("chat_scene_visual")
    options = _llm_options(chat_call=lambda **_: _chat_response("should not be reached"))
    assert build_context_prompt_with_llm([], template, options) is None
    assert build_context_prompt_with_llm([("user", "   ")], template, options) is None


def test_build_context_prompt_with_llm_success_flows_through_template_pipeline():
    """A successful LLM composition still goes through apply_template_to_prompt +
    the anchor-append invariant, exactly like the keyword path -- so negative_prompt/
    params/style label are unaffected, only the anchor text source changes."""
    template = get_template("chat_scene_visual")
    pairs = [("user", "a knight explores a glowing cave")]
    llm_text = "A knight in a glowing crystal cave, dramatic torchlight."
    options = _llm_options(chat_call=lambda **_: _chat_response(llm_text))
    result = build_context_prompt_with_llm(pairs, template, options)
    assert result is not None
    prompt, negative, params = result
    assert llm_text in prompt
    assert prompt.startswith("scene depicting")
    assert negative == template.negative_prompt
    assert params == template.default_params
    # The raw keyword-extracted last_message must NOT be the anchor when the
    # LLM path succeeded -- it's fully superseded, not merely appended.
    assert "a knight explores a glowing cave" not in prompt


def test_build_context_prompt_with_llm_falls_back_on_llm_failure():
    """Any LLM failure -> identical output to calling build_context_prompt directly."""
    template = get_template("chat_scene_visual")
    pairs = [("user", "a dimly lit tavern with a roaring fireplace")]

    def _raising(**_kwargs):
        raise RuntimeError("boom")

    options = _llm_options(chat_call=_raising)
    result = build_context_prompt_with_llm(pairs, template, options)
    assert result == build_context_prompt(pairs, template)


def test_build_context_prompt_with_llm_falls_back_when_disabled():
    template = get_template("chat_scene_visual")
    pairs = [("user", "a dimly lit tavern with a roaring fireplace")]
    options = _llm_options(
        enabled=False, chat_call=lambda **_: _chat_response("should not be reached")
    )
    result = build_context_prompt_with_llm(pairs, template, options)
    assert result == build_context_prompt(pairs, template)


# --- prepare_generation_request: llm_context threading ----------------------


def test_prepare_empty_prompt_llm_context_success_used_over_keyword_extractor():
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    pairs = [("user", "a quiet lakeside cabin at dawn")]
    llm_text = "A serene lakeside cabin at sunrise, soft mist rising off the water."
    options = _llm_options(chat_call=lambda **_: _chat_response(llm_text))
    result = prepare_generation_request(args, pairs, options)
    assert isinstance(result, PreparedGeneration)
    assert llm_text in result.prompt
    template = get_template("chat_scene_visual")
    assert result.style_name == template.name
    assert result.negative_prompt == template.negative_prompt


def test_prepare_empty_prompt_llm_context_failure_falls_back_to_keyword_result():
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    pairs = [("user", "a quiet lakeside cabin at dawn")]
    options = _llm_options(chat_call=lambda **_: (_ for _ in ()).throw(RuntimeError("x")))
    with_llm = prepare_generation_request(args, pairs, options)
    without_llm = prepare_generation_request(args, pairs, None)
    assert with_llm == without_llm
    assert isinstance(with_llm, PreparedGeneration)
    assert "a quiet lakeside cabin at dawn" in with_llm.prompt


def test_prepare_empty_prompt_llm_context_default_none_matches_pre_ac1_behavior():
    """Omitting llm_context entirely (positional-arg call sites elsewhere in the
    suite) is unaffected by this AC -- default is None, pure keyword path."""
    args = GenerateImageArgs(backend=None, prompt="", style=None)
    pairs = [("user", "a quiet lakeside cabin at dawn")]
    assert prepare_generation_request(args, pairs) == prepare_generation_request(
        args, pairs, None
    )


def test_prepare_nonempty_prompt_ignores_llm_context():
    """llm_context is only ever consulted on the no-prompt path."""
    args = GenerateImageArgs(backend=None, prompt="a red dragon", style=None)
    options = _llm_options(chat_call=lambda **_: (_ for _ in ()).throw(AssertionError("must not be called")))
    result = prepare_generation_request(args, [], options)
    assert result == PreparedGeneration(
        prompt="a red dragon",
        negative_prompt=None,
        style_name=None,
        width=None,
        height=None,
        steps=None,
        cfg_scale=None,
    )
