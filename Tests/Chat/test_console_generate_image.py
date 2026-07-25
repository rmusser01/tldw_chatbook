"""Tests for console_generate_image pure helpers."""

import dataclasses

import pytest
from tldw_chatbook.Chat.console_generate_image import (
    GENERATE_IMAGE_USAGE_TEXT,
    GenerateImageArgs,
    GenerationRefusal,
    PreparedGeneration,
    build_context_prompt,
    clamp_initial_batch,
    compose_styled_request,
    insert_style_token_into_draft,
    parse_generate_image_args,
    generation_content_marker,
    prepare_generation_request,
    resolve_style_token,
    run_generation_batch,
)
from tldw_chatbook.Media_Creation.generation_templates import (
    BUILTIN_TEMPLATES,
    get_template,
)


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
