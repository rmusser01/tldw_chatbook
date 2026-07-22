# Tests/Internal_Prompts/test_summarization_migration.py
"""Overrides must reach the summarization payloads; caller channels win.
Fakes only at the LLM dispatch/transport seams (pipeline code before/after
the seam runs for real).

Adaptations from the brief's skeleton (see task-4-report.md for detail):

- `_dispatch_to_api` was confirmed (by reading Summarization_General_Lib.py)
  as the real seam analyze() calls for the non-chunking path; the skeleton's
  guess was correct as written, no rename needed.
- Reading analyze()'s control flow turned up a pre-existing, unrelated bug:
  when `CHUNKER_AVAILABLE` is True (the normal case) *and* both
  `recursive_summarization` and `chunked_summarization` are False (the
  default/"just summarize" call shape used by the brief's skeleton), the
  `if CHUNKER_AVAILABLE: ... else: final_result = _dispatch_to_api(...)`
  structure means the direct-dispatch `else` branch is paired with
  `CHUNKER_AVAILABLE`, not with the inner recursive/chunked `if`/`elif`.
  So `_dispatch_to_api` is *never called* for that call shape — `analyze()`
  silently returns "Error: Summarization failed unexpectedly." regardless
  of system_message. This is untouched (out of scope for this migration:
  the task is prompt-source migration, not analyze()'s dispatch logic), but
  it means the skeleton's default `sgl.analyze(...)` call cannot reach
  `_dispatch_to_api` as written. The tests below monkeypatch
  `sgl.CHUNKER_AVAILABLE = False` to reach the same `_dispatch_to_api` call
  the brief targets (same call, same args) without depending on the real
  chunking subsystem's behavior on a 4-character input string.
- `_rolling_summarize` is exercised through the public `Chunker.chunk_text(
  ..., method="rolling_summarize", ...)` entry point rather than called
  directly, so the test also proves the `_get_option` wiring between
  chunk_text's "rolling_summarize" branch and `_rolling_summarize` itself.
  `Chunker(options={"summarize_system_prompt": None})` is used to force
  `_get_option`'s fallback branch, because `self.options["summarize_system_
  prompt"]` is populated from the module-level `DEFAULT_CHUNK_OPTIONS` dict
  at *import time* and is virtually never None in real use — see
  `test_rolling_summarize_default_channel_is_frozen_at_import` below, which
  documents that a config change made *after* import does NOT reach a
  freshly constructed `Chunker()` that doesn't pass this override, i.e. the
  import-time channel is not live.
- Local_Summarization_Lib's five call sites are covered by a call-site
  identity/source assertion rather than a real HTTP-payload test: reading
  all five functions turned up two independent, pre-existing obstacles that
  make transport-level tests disproportionate (see docstring on
  `test_local_summarizer_template_call_sites_use_registry` for specifics).
  These are pre-existing issues unrelated to this migration and are left
  untouched, as directed.
"""

from tldw_chatbook.Internal_Prompts import get_internal_prompt


def test_analyze_default_system_uses_registry(scratch_config, monkeypatch):
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl

    scratch_config(
        '[internal_prompts.summarization]\nanalyze_default_system = "CUSTOM ANALYZE SYSTEM"\n'
    )
    captured = {}

    def fake_dispatch(*args, **kwargs):
        captured["kwargs"] = kwargs
        captured["args"] = args
        return "ok"

    # analyze()'s non-chunking path calls _dispatch_to_api(text_content,
    # custom_prompt_arg, api_name, api_key, temp, system_message,
    # streaming=...) positionally/by-kwarg; monkeypatch it in sgl's module
    # namespace (that's the name analyze() actually resolves at call time).
    monkeypatch.setattr(sgl, "_dispatch_to_api", fake_dispatch)
    # See module docstring: reach the direct-dispatch branch without
    # depending on the (unrelated) chunking subsystem.
    monkeypatch.setattr(sgl, "CHUNKER_AVAILABLE", False)
    sgl.analyze(
        input_data="text", custom_prompt_arg="p", api_name="openai",
        api_key=None, temp=0.3, system_message=None, streaming=False,
    )
    assert "CUSTOM ANALYZE SYSTEM" in str(captured)


def test_analyze_caller_system_message_wins(scratch_config, monkeypatch):
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl

    scratch_config(
        '[internal_prompts.summarization]\nanalyze_default_system = "REGISTRY"\n'
    )
    captured = {}

    def fake_dispatch(*args, **kwargs):
        captured["all"] = (args, kwargs)
        return "ok"

    monkeypatch.setattr(sgl, "_dispatch_to_api", fake_dispatch)
    monkeypatch.setattr(sgl, "CHUNKER_AVAILABLE", False)
    sgl.analyze(
        input_data="text", custom_prompt_arg="p", api_name="openai",
        api_key=None, temp=0.3, system_message="CALLER", streaming=False,
    )
    assert "CALLER" in str(captured) and "REGISTRY" not in str(captured)


def test_rolling_summarize_system_override_reaches_llm_payload(scratch_config):
    from tldw_chatbook.Chunking.Chunk_Lib import Chunker

    scratch_config(
        '[internal_prompts.summarization]\nrolling_summarize_system = "CUSTOM ROLLING"\n'
    )
    captured = []

    def fake_llm(payload):
        # _rolling_summarize invokes llm_summarize_step_func(payload_dict)
        # with a single positional dict argument containing "system_message".
        captured.append(payload)
        return "summary"

    # Force _get_option("summarize_system_prompt", ...) to fall through to
    # its dynamic default (now get_internal_prompt(...)), since the
    # import-time dict entry (self.options[key] from DEFAULT_CHUNK_OPTIONS)
    # would otherwise win and mask the live override — see the frozen-at-
    # import test below for why that channel doesn't observe this override.
    chunker = Chunker(options={"summarize_system_prompt": None})
    chunker.chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 20,
        method="rolling_summarize",
        llm_call_function=fake_llm,
        llm_api_config={},
    )
    assert captured, "fake_llm was never invoked"
    assert any(
        "CUSTOM ROLLING" in str(payload.get("system_message", ""))
        for payload in captured
    )


def test_rolling_summarize_caller_option_wins(scratch_config):
    """The per-instance `options={"summarize_system_prompt": ...}` channel
    (the Chunker caller channel) still outranks the registry default."""
    from tldw_chatbook.Chunking.Chunk_Lib import Chunker

    scratch_config(
        '[internal_prompts.summarization]\nrolling_summarize_system = "REGISTRY ROLLING"\n'
    )
    captured = []

    def fake_llm(payload):
        captured.append(payload)
        return "summary"

    chunker = Chunker(options={"summarize_system_prompt": "CALLER ROLLING"})
    chunker.chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 20,
        method="rolling_summarize",
        llm_call_function=fake_llm,
        llm_api_config={},
    )
    assert captured, "fake_llm was never invoked"
    joined = " ".join(str(p.get("system_message", "")) for p in captured)
    assert "CALLER ROLLING" in joined
    assert "REGISTRY ROLLING" not in joined


def test_rolling_summarize_default_channel_is_frozen_at_import(scratch_config):
    """Documents the finding in the task report: the value that actually
    reaches `_rolling_summarize` in the common case (no per-instance
    `options` override) comes from the module-level `DEFAULT_CHUNK_OPTIONS`
    dict, which is built once when Chunking.Chunk_Lib is first imported.
    A config change made after import (as scratch_config does here, well
    after test collection has already imported the module) is NOT observed
    by a freshly constructed `Chunker()` — the override only takes effect
    on the next process start / module (re)import, not live per call."""
    from tldw_chatbook.Chunking.Chunk_Lib import Chunker

    baked_in_default = get_internal_prompt("summarization.rolling_summarize_system")

    scratch_config(
        '[internal_prompts.summarization]\nrolling_summarize_system = "SHOULD NOT APPEAR LIVE"\n'
    )
    captured = []

    def fake_llm(payload):
        captured.append(payload)
        return "summary"

    chunker = Chunker()  # no override -> uses the import-time dict entry
    chunker.chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 20,
        method="rolling_summarize",
        llm_call_function=fake_llm,
        llm_api_config={},
    )
    assert captured, "fake_llm was never invoked"
    joined = " ".join(str(p.get("system_message", "")) for p in captured)
    assert "SHOULD NOT APPEAR LIVE" not in joined
    assert baked_in_default in joined


def test_local_summarizer_template_call_sites_use_registry():
    """Local_Summarization_Lib's five consuming sites (summarize_with_llama,
    summarize_with_kobold, summarize_with_tabbyapi, summarize_with_custom_
    openai, summarize_with_custom_openai_2) all read
    `get_internal_prompt('summarization.local_summarizer_template')` at call
    time rather than the deleted module constant. A real HTTP-payload test
    per the brief's preference turned out disproportionate for two
    independent, pre-existing reasons found while reading these functions
    (neither introduced by this migration, both left untouched):

    1. summarize_with_llama, summarize_with_kobold, and
       summarize_with_tabbyapi read config sections that do not exist under
       those names in config.py's `load_settings()` output (e.g.
       `loaded_config_data["llama_api"]` — the real section is
       `"llama_cpp_api"`; `loaded_config_data["api_keys"]` and
       `["local_api_ip"]` are not produced by load_settings() at all). These
       functions raise KeyError before reaching their HTTP call regardless
       of the prompt migration.
    2. summarize_with_custom_openai and summarize_with_custom_openai_2 (the
       two functions whose config keys DO resolve correctly) reassign the
       migrated value to the local variable `input_data`, which is never
       read again afterward — the outgoing payload's user-message content
       is built from `text` + `custom_prompt_arg` only. The line is dead
       code both before and after this migration, so there is no payload to
       assert against.

    Given that, this test asserts the call-site wiring by source inspection:
    the deleted constant is gone from the module namespace, the five sites
    reference the registry id (not a stray literal), and the unrelated
    `summarize_with_oobabooga` local shadow (a different, hardcoded prompt
    "Please summarize the following text:") is untouched.
    """
    import inspect

    from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as lsl

    assert not hasattr(lsl, "summarizer_prompt")

    source = inspect.getsource(lsl)
    assert (
        source.count("get_internal_prompt('summarization.local_summarizer_template')")
        == 5
    )
    assert "Rewrite this text in summarized form." not in source

    # The oobabooga local shadow (a different prompt) must remain untouched.
    assert '"Please summarize the following text:"' in source


def test_rolling_summarize_skips_resolver_when_caller_option_set(
    scratch_config, monkeypatch
):
    """`Chunker._get_option(key, default_override)` only uses
    `default_override` when `self.options[key]` is None, but Python
    evaluates arguments eagerly — so a bare
    `get_internal_prompt("summarization.rolling_summarize_system")` passed
    positionally as that default runs on *every* rolling_summarize call
    regardless of whether `self.options["summarize_system_prompt"]` is
    already populated (the common case per
    test_rolling_summarize_default_channel_is_frozen_at_import: the
    module-level DEFAULT_CHUNK_OPTIONS entry is virtually never None).
    That's a wasted config lookup on a hot path, and it can trip the
    resolver's warn-once path for a value that's immediately discarded.
    The resolver must not be called at all when the caller option already
    supplies a value."""
    from tldw_chatbook.Chunking import Chunk_Lib

    calls = []
    original = Chunk_Lib.get_internal_prompt

    def spy(prompt_id):
        calls.append(prompt_id)
        return original(prompt_id)

    monkeypatch.setattr(Chunk_Lib, "get_internal_prompt", spy)

    captured = []

    def fake_llm(payload):
        captured.append(payload)
        return "summary"

    chunker = Chunk_Lib.Chunker(options={"summarize_system_prompt": "CALLER SET"})
    chunker.chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 20,
        method="rolling_summarize",
        llm_call_function=fake_llm,
        llm_api_config={},
    )
    assert captured, "fake_llm was never invoked"
    assert calls == [], (
        "get_internal_prompt('summarization.rolling_summarize_system') "
        f"was called even though summarize_system_prompt was already set: {calls}"
    )
