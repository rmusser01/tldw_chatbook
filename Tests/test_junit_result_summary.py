"""Stdlib-only checks for public CI outcome reporting; no app import."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Tests.junit_result_summary import escape_annotation, main


class JUnitResultSummaryTests(unittest.TestCase):
    def test_counts_redaction_escaping_and_incomplete_reports(self):
        self.assertEqual(escape_annotation("%\r\n"), "%25%0D%0A")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "mixed.xml"
            report.write_text(
                """<testsuite>
<testcase classname="Tests.public" name="test_pass"/>
<testcase classname="Tests.public" name="test_skip"><skipped/></testcase>
<testcase classname="Tests.public" name="test_failed[PRIVATE_PARAM]">
<failure message="PRIVATE_DETAIL">PRIVATE_CONFIG_AND_LOG</failure></testcase>
<testcase classname="Tests.public" name="test_failed[OTHER_PRIVATE_PARAM]">
<failure/></testcase>
<testcase classname="Tests.public" name="test_error"/>
<testcase classname="Tests.public" name="test_error"><error/></testcase>
<testcase classname="Tests.public" name="test%escaped&#13;&#10;::error::[PRIVATE_PARAM]">
<failure/></testcase>
</testsuite>""",
                encoding="utf-8",
            )
            malformed = root / "malformed.xml"
            malformed.write_text("<testsuite>PRIVATE_DETAIL", encoding="utf-8")
            summary = root / "summary.md"
            summary.write_text("Existing summary\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch.dict(os.environ, GITHUB_STEP_SUMMARY=str(summary)):
                with contextlib.redirect_stdout(stdout):
                    self.assertIsNone(
                        main([str(report), str(root / "missing.xml"), str(malformed)])
                    )
            emitted = stdout.getvalue()
            text = summary.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("Existing summary\n"))
            self.assertIn("pass=1, fail=3, error=1, skip=1", text)
            self.assertIn("incomplete** (missing report)", text)
            self.assertIn("incomplete** (unreadable or malformed report)", text)
            self.assertEqual(emitted.count("mixed.xml: Tests.public::test_failed\n"), 1)
            self.assertIn("test%25escaped%0D%0A::error::", emitted)
            self.assertEqual(len(emitted.splitlines()), 5)
            for private in (
                "PRIVATE_PARAM",
                "PRIVATE_DETAIL",
                "PRIVATE_CONFIG_AND_LOG",
                str(root),
            ):
                self.assertNotIn(private, emitted + text)


if __name__ == "__main__":
    unittest.main()
