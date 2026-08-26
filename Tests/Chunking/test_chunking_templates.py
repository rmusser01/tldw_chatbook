# test_chunking_templates.py
"""
Template semantics after the file-store deletion (spec §8.1-8.2, ACs 30-32).

The file template store (``Chunking/templates/`` -- 13 JSON + README.md +
``example_usage.py``) and its manager module (``Chunking/chunking_templates.py``
with ``ChunkingTemplateManager``/``ChunkingPipeline``) are deleted. Templates
are DB rows; name resolution lives in ``template_runtime.resolve_template``
at the service layer (spec §8.2 -- resolution needs a Media DB handle, which
the import-light ``Chunk_Lib`` shim must not acquire), and
``Chunker``/``improved_chunking_process`` accept only a PRE-RESOLVED template
dict (the flat spec §4.1 shape). Full template execution (pre + chunk + post
stages) is ``template_runtime.apply_template``'s contract; the shim applies
the chunk-stage options only.
"""

import importlib.util

import pytest

from tldw_chatbook.Chunking import Chunk_Lib
from tldw_chatbook.Chunking.engine.exceptions import TemplateError

# The flat spec §4.1 shape -- the same dict template_runtime.resolve_template
# returns and apply_template consumes.
WORDS_TEMPLATE = {
    "name": "words_small",
    "chunking": {"method": "words", "config": {"max_size": 3, "overlap": 0}},
}
SENTENCES_TEMPLATE = {
    "name": "sentences_small",
    "chunking": {"method": "sentences", "config": {"max_size": 2}},
}
FULL_TEMPLATE = {
    "name": "full",
    "preprocessing": [{"type": "normalize_whitespace", "params": {}}],
    "chunking": {"method": "sentences", "config": {"max_size": 2}},
    "postprocessing": [{"type": "filter_empty", "params": {"min_length": 10}}],
}


class TestBareNameResolutionIsGone:
    """AC 32: a bare name string is a named raise, never silent resolution."""

    def test_bare_name_template_raises_named_error(self):
        # A bare name would need a Media DB handle to resolve; the shim is
        # import-light by contract (AC 33), so the name must point at the
        # service-layer resolver instead of attempting file-store lookup.
        with pytest.raises(TemplateError, match="resolve_template"):
            Chunk_Lib.Chunker(template="academic_paper")

    def test_bare_name_raises_even_with_template_manager(self):
        # template_manager= is accepted-and-ignored: it must NOT be consulted
        # to rescue a bare name (the old file-store path did exactly that).
        with pytest.raises(TemplateError, match="resolve_template"):
            Chunk_Lib.Chunker(template="words", template_manager=object())

    def test_improved_chunking_process_bare_name_raises_named_error(self):
        with pytest.raises(TemplateError, match="resolve_template"):
            Chunk_Lib.improved_chunking_process(
                "Alpha beta gamma. Delta epsilon.", template="conversation"
            )


class TestPreResolvedTemplateDict:
    """AC 32: a pre-resolved dict works and follows the legacy precedence."""

    def test_pre_resolved_dict_works(self):
        pre = {"chunking": {"method": "words", "config": {"max_size": 3}}}
        chunks = Chunk_Lib.Chunker(template=pre).chunk_text("a b c d e")
        assert chunks
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_template_options_beat_defaults_and_lose_to_explicit(self):
        # Legacy precedence: defaults <- template <- explicit options.
        chunker = Chunk_Lib.Chunker(
            options={"max_size": 8},
            template=SENTENCES_TEMPLATE,
        )
        assert chunker.options["method"] == "sentences"
        assert chunker.options["max_size"] == 8

    def test_template_dict_with_pre_post_blocks_accepted(self):
        # resolve_template returns the WHOLE template body, pre/post blocks
        # included. The shim must accept it (applying the chunk-stage
        # options); executing pre/post is apply_template's contract, not
        # the shim's.
        chunker = Chunk_Lib.Chunker(template=FULL_TEMPLATE)
        assert chunker.options["method"] == "sentences"
        chunks = chunker.chunk_text(
            "This is a test document. It has several sentences here."
        )
        assert chunks

    def test_non_dict_non_str_template_raises_named_error(self):
        with pytest.raises(TemplateError):
            Chunk_Lib.Chunker(template=42)

    def test_dict_without_chunking_block_raises_named_error(self):
        with pytest.raises(TemplateError, match="chunking"):
            Chunk_Lib.Chunker(template={"name": "empty"})


class TestTemplateManagerAcceptedAndIgnored:
    """AC 32: template_manager= is accepted and ignored (pinned)."""

    def test_template_manager_accepted_and_ignored(self):
        sentinel = object()
        with_manager = Chunk_Lib.Chunker(
            {"method": "words", "max_size": 3, "overlap": 0},
            template_manager=sentinel,
        )
        without = Chunk_Lib.Chunker({"method": "words", "max_size": 3, "overlap": 0})
        text = "one two three four five six"
        assert with_manager.chunk_text(text) == without.chunk_text(text)
        # The only observable effect is the attribute passthrough; no file
        # store is consulted, no directory is created.
        assert with_manager.template_manager is sentinel

    def test_template_manager_ignored_alongside_dict_template(self):
        chunks = Chunk_Lib.Chunker(
            template=WORDS_TEMPLATE, template_manager=object()
        ).chunk_text("a b c d e")
        assert chunks


class TestPackageRootExportSurface:
    """AC 30 + the §8.1.1 name-collision ruling."""

    DELETED_EXPORTS = (
        "ChunkingTemplateManager",
        "ChunkingPipeline",
        "ChunkingStage",
        "ChunkingOperation",
        "ChunkingTemplate",
    )

    def test_package_root_no_longer_exports_file_store_names(self):
        import tldw_chatbook.Chunking as pkg

        for name in self.DELETED_EXPORTS:
            assert not hasattr(pkg, name), (
                f"tldw_chatbook.Chunking still exports {name!r}"
            )

    def test_file_store_module_is_gone(self):
        assert (
            importlib.util.find_spec("tldw_chatbook.Chunking.chunking_templates")
            is None
        )

    def test_vendored_chunking_template_not_re_exported(self):
        # The vendored engine defines its own ChunkingTemplate (same public
        # name, different class). Only it survives, and it is NOT re-exported
        # at the package root: nothing outside the service layer resolves
        # templates (§8.1.1 collision ruling, §8.2).
        from tldw_chatbook.Chunking.engine.templates import (
            ChunkingTemplate as VendoredTemplate,
        )

        import tldw_chatbook.Chunking as pkg

        assert VendoredTemplate is not None
        assert not hasattr(pkg, "ChunkingTemplate")


class TestTemplateExecutionViaRuntime:
    """Live template behavior (old pipeline/Chunker tests), rewritten
    against ``template_runtime`` + the vendored processor."""

    def test_simple_template_execution(self):
        from tldw_chatbook.Chunking.template_runtime import apply_template

        template = {
            "name": "simple",
            "chunking": {"method": "words", "config": {"max_size": 10, "overlap": 2}},
        }
        text = " ".join(f"word{i}" for i in range(50))
        results = apply_template(template, text)

        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all(r["metadata"].get("offset_basis") for r in results)

    def test_template_with_pre_and_post_operations(self):
        from tldw_chatbook.Chunking.template_runtime import apply_template

        template = {
            "name": "ops_test",
            "preprocessing": [{"type": "normalize_whitespace", "params": {}}],
            "chunking": {"method": "sentences", "config": {"max_size": 2}},
            "postprocessing": [{"type": "filter_empty", "params": {"min_length": 10}}],
        }
        text = (
            "This is   a test.   \n\n  Another sentence.  Short.  "
            "And one more sentence here."
        )
        results = apply_template(template, text)
        assert results
        # filter_empty(min_length=10) removed the short chunks.
        assert all(len(r["text"]) >= 10 for r in results)
        # A pre operation rewrote the text, so offsets are preprocessed-based.
        assert all(
            r["metadata"]["offset_basis"].startswith("preprocessed:")
            for r in results
        )

    def test_chunker_template_options_apply(self):
        chunker = Chunk_Lib.Chunker(template=SENTENCES_TEMPLATE)
        text = "First sentence here. Second sentence follows. Third one now."
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_improved_process_with_pre_resolved_template(self):
        text = "This is a test document. " * 50
        results = Chunk_Lib.improved_chunking_process(
            text, template=SENTENCES_TEMPLATE, chunk_options_dict={"max_size": 3}
        )
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r and "metadata" in r for r in results)
        # The template's method won over the options default.
        assert all(r["metadata"]["chunk_method"] == "sentences" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
