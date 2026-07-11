"""Self-healing for OCR backends whose lazy dependency import fails at first use.

The F3 import-chain slimming replaced eager module-level backend imports with
``importlib.util.find_spec`` probes -- which cannot distinguish a MISSING
package from an installed-but-BROKEN one. Pre-change, a broken package tripped
the eager ``except ImportError`` at module load, so the backend simply never
registered and a working lower-priority backend became the default.

These tests pin the replacement contract: when a backend's deferred import
raises at first actual use, that backend is permanently demoted (is_available
becomes False, no re-registration, no import retries) and the manager's
default selection re-resolves to the next available backend -- so the FIRST
OCR call degrades once and succeeds via the fallback, and every subsequent
call behaves exactly as the old startup-time exclusion would have. Runtime
errors after a successful import must NOT demote anything.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Local_Ingestion.OCR_Backends import (
    OCRBackend,
    OCRManager,
    OCRResult,
)


class _BrokenImportBackend(OCRBackend):
    """Backend whose lazy dependency import raises (installed-but-broken)."""

    def __init__(self, config=None):
        super().__init__(config)
        self.import_attempts = 0

    def is_available(self) -> bool:
        if self._import_broken:
            return False
        return True  # find_spec-style probe: package looks installed

    def _import_dependency(self) -> None:
        """Stands in for the deferred `import heavy_package` statement."""
        self.import_attempts += 1
        raise ImportError("simulated broken package: cannot import name 'x'")

    def initialize(self) -> None:
        if not self._initialized:
            try:
                self._import_dependency()
            except Exception as import_err:
                self._mark_import_broken(import_err)
                raise
            self._initialized = True

    def process_image(self, image_path, language="en", **kwargs) -> OCRResult:
        if not self._initialized:
            self.initialize()
        return OCRResult(text="broken", confidence=1.0, language=language, backend="broken")

    def process_pdf(self, pdf_path, language="en", **kwargs):
        if not self._initialized:
            self.initialize()
        return [OCRResult(text="broken", confidence=1.0, language=language, backend="broken")]

    def get_supported_languages(self):
        return ["en"]


class _WorkingBackend(OCRBackend):
    """Backend whose import and processing both succeed."""

    def is_available(self) -> bool:
        return not self._import_broken

    def initialize(self) -> None:
        self._initialized = True

    def process_image(self, image_path, language="en", **kwargs) -> OCRResult:
        return OCRResult(text="ok", confidence=0.9, language=language, backend="working")

    def process_pdf(self, pdf_path, language="en", **kwargs):
        return [OCRResult(text="ok", confidence=0.9, language=language, backend="working")]

    def get_supported_languages(self):
        return ["en"]


class _RuntimeFailureBackend(_WorkingBackend):
    """Import succeeds; the OCR call itself fails (a genuine runtime error)."""

    def process_image(self, image_path, language="en", **kwargs) -> OCRResult:
        raise RuntimeError("OCR engine crashed at runtime")


def _manager_with(backends: dict) -> OCRManager:
    """Build a manager with an injected registry.

    Real backend names ("docling", "tesseract") are used so the manager's
    existing priority order (docext > docling > tesseract > easyocr >
    paddleocr) drives default selection and re-resolution.
    """
    manager = OCRManager()
    manager.backends = dict(backends)
    manager._user_default = False
    manager.default_backend = None
    manager._resolve_default_backend()
    return manager


def test_first_use_import_failure_falls_back_to_next_backend():
    broken = _BrokenImportBackend()
    working = _WorkingBackend()
    manager = _manager_with({"docling": broken, "tesseract": working})
    assert manager.default_backend == "docling"

    result = manager.process_image("/tmp/fake.png")

    # First call degrades once and succeeds via the fallback backend.
    assert result.backend == "working"
    # The broken backend is permanently unavailable afterwards.
    assert broken.is_available() is False
    # Default re-resolved to the next available backend by priority.
    assert manager.default_backend == "tesseract"
    assert manager.get_available_backends() == ["tesseract"]


def test_broken_backend_is_not_retried_on_subsequent_calls():
    broken = _BrokenImportBackend()
    working = _WorkingBackend()
    manager = _manager_with({"docling": broken, "tesseract": working})

    for _ in range(3):
        assert manager.process_image("/tmp/fake.png").backend == "working"

    # Exactly one import attempt: the memoized demotion prevents retries.
    assert broken.import_attempts == 1


def test_process_pdf_first_use_import_failure_also_falls_back():
    broken = _BrokenImportBackend()
    working = _WorkingBackend()
    manager = _manager_with({"docling": broken, "tesseract": working})

    results = manager.process_pdf("/tmp/fake.pdf")

    assert results[0].backend == "working"
    assert broken.import_attempts == 1
    assert manager.default_backend == "tesseract"


def test_explicitly_requested_broken_backend_raises_then_reports_unavailable():
    broken = _BrokenImportBackend()
    working = _WorkingBackend()
    manager = _manager_with({"docling": broken, "tesseract": working})

    # First explicit call surfaces the real import error (no silent fallback
    # when the caller pinned a specific backend).
    with pytest.raises(ImportError):
        manager.process_image("/tmp/fake.png", backend="docling")

    # Subsequent calls behave exactly like the old startup-time exclusion:
    # the backend is simply "not available".
    with pytest.raises(ValueError, match="not available"):
        manager.process_image("/tmp/fake.png", backend="docling")
    assert broken.import_attempts == 1

    # The working backend took over as default and is unaffected.
    assert manager.process_image("/tmp/fake.png").backend == "working"


def test_runtime_failure_after_successful_import_does_not_demote():
    flaky = _RuntimeFailureBackend()
    working = _WorkingBackend()
    manager = _manager_with({"docling": flaky, "tesseract": working})

    with pytest.raises(RuntimeError):
        manager.process_image("/tmp/fake.png")

    # No demotion for runtime errors: still registered, available, default.
    assert flaky.is_available() is True
    assert manager.default_backend == "docling"
    assert "docling" in manager.get_available_backends()


def test_all_backends_broken_raises_no_backend_available():
    b1 = _BrokenImportBackend()
    b2 = _BrokenImportBackend()
    manager = _manager_with({"docling": b1, "tesseract": b2})

    with pytest.raises(ValueError, match="No OCR backend available"):
        manager.process_image("/tmp/fake.png")

    # Each broken backend was tried exactly once before demotion.
    assert b1.import_attempts == 1
    assert b2.import_attempts == 1
    assert manager.get_available_backends() == []
