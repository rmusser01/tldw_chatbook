"""B13: the Docling ``DocumentConverter`` is constructed once per process.

Review-B finding B13: ``docling_parse_pdf`` built a fresh ``DocumentConverter``
(and re-initialized its OCR/layout models) for every PDF. Parsing runs inside
process-pool workers in production, so a per-process module singleton is the
correct scope -- each worker constructs once, then reuses.

The construction is config-independent today (``DocumentConverter()`` takes no
arguments; per-call OCR settings ride ``pipeline_options`` into
``converter.convert``), so a single unkeyed singleton preserves behavior. The
optional-dependency error path (``ImportError`` when docling is absent) must be
unchanged.
"""

import sys
import types

import pytest

from tldw_chatbook.Local_Ingestion import PDF_Processing_Lib as pdf_lib


class _FakeDocumentConverter:
    """Counts constructions; returns a stub parse result from ``convert``."""

    instances = 0
    convert_calls = 0
    last_pipeline_options = None

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def convert(self, pdf_path, pipeline_options=None):
        type(self).convert_calls += 1
        type(self).last_pipeline_options = pipeline_options

        class _Doc:
            def export_to_markdown(self):
                return "fake markdown"

        class _Parsed:
            document = _Doc()

        return _Parsed()


class _FakePdfPipelineOptions:
    def __init__(self):
        self.do_ocr = False
        self.do_table_structure = False


@pytest.fixture()
def fake_docling(monkeypatch):
    """Install a fake ``docling`` package tree and reset the singleton."""
    _FakeDocumentConverter.instances = 0
    _FakeDocumentConverter.convert_calls = 0
    _FakeDocumentConverter.last_pipeline_options = None

    fake_converter_mod = types.ModuleType("docling.document_converter")
    fake_converter_mod.DocumentConverter = _FakeDocumentConverter
    fake_pipeline_mod = types.ModuleType("docling.datamodel.pipeline_options")
    fake_pipeline_mod.PdfPipelineOptions = _FakePdfPipelineOptions
    fake_datamodel_mod = types.ModuleType("docling.datamodel")
    fake_datamodel_mod.pipeline_options = fake_pipeline_mod
    fake_docling_mod = types.ModuleType("docling")
    fake_docling_mod.document_converter = fake_converter_mod
    fake_docling_mod.datamodel = fake_datamodel_mod

    monkeypatch.setitem(sys.modules, "docling", fake_docling_mod)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_mod)
    monkeypatch.setitem(sys.modules, "docling.datamodel", fake_datamodel_mod)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake_pipeline_mod)

    pdf_lib._reset_docling_converter_for_tests()
    yield _FakeDocumentConverter
    pdf_lib._reset_docling_converter_for_tests()


def test_two_parses_construct_one_converter(fake_docling):
    """A 2-PDF batch pays for exactly one converter construction."""
    assert (
        pdf_lib.docling_parse_pdf("/tmp/a.pdf") == "fake markdown"
    )
    assert (
        pdf_lib.docling_parse_pdf("/tmp/b.pdf") == "fake markdown"
    )
    assert fake_docling.instances == 1
    assert fake_docling.convert_calls == 2


def test_ocr_flag_variants_still_one_converter(fake_docling):
    """OCR plumbing stays per-call; the converter singleton is unkeyed."""
    pdf_lib.docling_parse_pdf("/tmp/plain.pdf", enable_ocr=False)
    pdf_lib.docling_parse_pdf("/tmp/scanned.pdf", enable_ocr=True, ocr_language="de")
    assert fake_docling.instances == 1
    assert fake_docling.convert_calls == 2
    # The per-call OCR setting reached the convert call both times.
    assert fake_docling.last_pipeline_options is not None
    assert fake_docling.last_pipeline_options.do_ocr is True


def test_reset_helper_forces_reconstruction(fake_docling):
    pdf_lib.docling_parse_pdf("/tmp/a.pdf")
    pdf_lib._reset_docling_converter_for_tests()
    pdf_lib.docling_parse_pdf("/tmp/b.pdf")
    assert fake_docling.instances == 2


def test_missing_docling_still_raises_import_error(monkeypatch):
    """The optional-dependency error path is unchanged (no singleton masking it)."""
    pdf_lib._reset_docling_converter_for_tests()
    # ``None`` in sys.modules makes ``import docling...`` raise ImportError.
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", None)
    with pytest.raises(ImportError, match="Docling library is not installed"):
        pdf_lib.docling_parse_pdf("/tmp/a.pdf")
