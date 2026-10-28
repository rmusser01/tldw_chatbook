"""Copyable reports preserve privacy, byte limits, and the reviewed text."""

from markdown_it import MarkdownIt
import pytest

from tldw_chatbook.Utils import bug_report
from tldw_chatbook.Utils.bug_report import (
    BugReportDraft,
    BugReportEvidence,
    BugReportItem,
    BugReportOptions,
    prepare_bug_report_copy,
    render_bug_report,
)

pytestmark = pytest.mark.bootstrap_profile


def test_default_report_excludes_optional_tool_items():
    evidence = BugReportEvidence(
        trace=(BugReportItem("trace", "event-1 tool_result error duration=0.015s"),),
        tools=(BugReportItem("tools", "PRIVATE_TOOL_SENTINEL"),),
    )
    report = render_bug_report(
        BugReportDraft("Observed failure"), evidence, BugReportOptions()
    )
    assert "PRIVATE_TOOL_SENTINEL" not in report
    assert "duration=0.015s" in report
    opted_in = render_bug_report(
        BugReportDraft("Observed failure"),
        evidence,
        BugReportOptions(include_tool_content=True),
    )
    assert "PRIVATE_TOOL_SENTINEL" in opted_in
    assert "other conversations" in opted_in


def test_excluding_trace_excludes_tool_content_even_when_previously_selected():
    report = render_bug_report(
        BugReportDraft("Failure"),
        BugReportEvidence(
            trace=(BugReportItem("trace", "TRACE_SENTINEL"),),
            tools=(BugReportItem("tools", "TOOL_SENTINEL"),),
        ),
        BugReportOptions(include_trace=False, include_tool_content=True),
    )
    assert "TRACE_SENTINEL" not in report and "TOOL_SENTINEL" not in report


def test_utf8_budgets_include_fences_and_notices():
    evidence = BugReportEvidence(
        environment=(("App version", "test"),),
        logs=tuple(
            BugReportItem("logs", f"log-{i:04d} " + "界" * 700) for i in range(700)
        ),
        trace=tuple(
            BugReportItem("trace", f"event-{i:04d} " + "x" * 300) for i in range(700)
        ),
        tools=tuple(
            BugReportItem("tools", f"tool-{i:04d} " + "y" * 400) for i in range(700)
        ),
        inventory=("Application-wide session logs",),
    )
    report = render_bug_report(
        BugReportDraft("Failure with Unicode 界"),
        evidence,
        BugReportOptions(include_tool_content=True),
    )
    assert len(report.encode("utf-8")) <= 60_000
    assert "log-0699" in report and "log-0000" not in report
    assert "event-0699" in report and "event-0000" not in report
    assert "omitted" in report.lower() or "truncated" in report.lower()
    assert "Failure with Unicode 界" in report


def test_multiline_credentials_are_redacted_before_clipping():
    secret = "SYNTHETIC-PRIVATE-MATERIAL"
    text = "-----BEGIN PRIVATE KEY-----\n" + secret + "\n-----END PRIVATE KEY-----"
    evidence = BugReportEvidence(logs=(BugReportItem("logs", text),))
    report = render_bug_report(BugReportDraft("Failure"), evidence, BugReportOptions())
    assert secret not in report
    assert "REDACTED" in report


def test_description_and_environment_use_real_diagnostic_redaction():
    report = render_bug_report(
        BugReportDraft("Contact tester@example.invalid; api_key=synthetic-secret"),
        BugReportEvidence(environment=(("Model", "Bearer synthetic-model-secret"),)),
        BugReportOptions(),
    )
    assert "tester@example.invalid" not in report
    assert "synthetic-secret" not in report and "synthetic-model-secret" not in report


def test_untrusted_markdown_cannot_escape_generated_fence():
    description = "before\n```\n# INJECTED HEADING\n~~~\n" + "`" * 18 + "\nafter"
    report = render_bug_report(
        BugReportDraft(description), BugReportEvidence(), BugReportOptions()
    )
    tokens = MarkdownIt().parse(report)
    assert any(
        token.type == "fence" and description in token.content for token in tokens
    )
    assert not any(
        token.type == "inline" and token.content == "INJECTED HEADING"
        for token in tokens
    )


def test_missing_sources_are_explicit():
    report = render_bug_report(
        BugReportDraft("UI failure"), BugReportEvidence(), BugReportOptions()
    )
    assert "Chatbook bug report v1" in report
    assert "No exception traceback" in report
    assert "unavailable" in report.lower()
    assert "Application-wide session logs" in report


def test_validation_preserves_authored_text_instead_of_silently_trimming():
    with pytest.raises(ValueError, match="[Dd]escription"):
        render_bug_report(BugReportDraft("  "), BugReportEvidence(), BugReportOptions())
    with pytest.raises(ValueError, match="8,000|8000"):
        render_bug_report(
            BugReportDraft("界" * 2667), BugReportEvidence(), BugReportOptions()
        )


def test_final_copy_preserves_oversize_manual_edits():
    text = "界" * 20_001
    check = prepare_bug_report_copy(text)
    assert check.state == "oversize" and check.text == text
    assert check.byte_count == 60_003


def test_repeat_copy_stabilizes_after_redaction():
    check = prepare_bug_report_copy(
        "description\napi_key=synthetic-secret\n" + "x" * 4000
    )
    assert check.state == "review_required"
    assert "synthetic-secret" not in check.text
    assert "x" * 4000 in check.text  # not the 2,000-character log formatter
    second = prepare_bug_report_copy(check.text)
    assert second.state == "ready" and second.text == check.text


def test_sanitizer_failure_never_returns_raw_exception_in_status(monkeypatch):
    def fail(_text):
        raise RuntimeError("UNSAFE_EXCEPTION_SENTINEL")

    monkeypatch.setattr(bug_report, "sanitize_string", fail)
    check = prepare_bug_report_copy("draft")
    assert check.state == "unavailable" and check.text == "draft"
    assert "UNSAFE_EXCEPTION_SENTINEL" not in check.message


def test_report_bodies_are_not_exposed_in_object_repr():
    assert "PRIVATE_DESCRIPTION" not in repr(BugReportDraft("PRIVATE_DESCRIPTION"))
    assert "PRIVATE_LOG" not in repr(BugReportItem("logs", "PRIVATE_LOG"))
    check = prepare_bug_report_copy("PRIVATE_REVIEW")
    assert "PRIVATE_REVIEW" not in repr(check)


def test_long_source_inventory_cannot_evict_scope_and_privacy_notices():
    report = render_bug_report(
        BugReportDraft("Failure"),
        BugReportEvidence(inventory=("noisy source note " * 3000,)),
        BugReportOptions(include_tool_content=True),
    )
    assert "Application-wide session logs" in report
    assert "Recognized credentials and PII" in report
    assert "other conversations" in report
    assert len(report.encode("utf-8")) <= 60_000
