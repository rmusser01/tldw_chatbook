"""Focused checks for extracting executable build inputs without path escapes."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from build import extract_tar, relative_member


class ArchiveBoundaryTests(unittest.TestCase):
    def test_unsafe_archive_paths_are_rejected(self):
        for name in (
            "/absolute",
            "C:/absolute",
            "C:relative",
            "x/../escape",
            r"x\file",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative_member(name)

    def test_safe_source_readme_link_is_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                name = "source/deps/harfbuzz/harfbuzz/README.md"
                member = tarfile.TarInfo(name)
                member.size = 5
                bundle.addfile(member, io.BytesIO(b"hello"))
                link = tarfile.TarInfo(name.removesuffix(".md"))
                link.type = tarfile.SYMTYPE
                link.linkname = "README.md"
                bundle.addfile(link)
            destination = root / "extracted"
            extract_tar(archive, destination)
            readme = destination / "deps/harfbuzz/harfbuzz/README"
            self.assertEqual(readme.read_bytes(), b"hello")
            self.assertFalse(readme.is_symlink())

    def test_other_source_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                link = tarfile.TarInfo("source/unexpected")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                bundle.addfile(link)
            with self.assertRaisesRegex(ValueError, "Unexpected archive link"):
                extract_tar(archive, root / "extracted")


if __name__ == "__main__":
    unittest.main()
