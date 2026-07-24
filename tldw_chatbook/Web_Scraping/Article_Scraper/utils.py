# article_scraper/utils.py
#
# Imports
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
#
# Third-Party Libraries
#
# Local Imports
#
#######################################################################################################################
#
# Functions:


class ContentMetadataHandler:
    """Handles the addition and parsing of metadata for scraped content."""

    METADATA_START = "[METADATA]"
    METADATA_END = "[/METADATA]"

    @staticmethod
    def format_content_with_metadata(
        url: str,
        content: str,
        pipeline: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = {
            "url": url,
            "ingestion_date": datetime.now().isoformat(),
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "scraping_pipeline": pipeline,
            **(additional_metadata or {}),
        }
        metadata_str = json.dumps(metadata, indent=2)
        return f"{ContentMetadataHandler.METADATA_START}\n{metadata_str}\n{ContentMetadataHandler.METADATA_END}\n\n{content}"

    @staticmethod
    def extract_metadata(
        content_with_meta: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Extracts metadata and returns (metadata_dict, clean_content)."""
        try:
            start_idx = content_with_meta.index(ContentMetadataHandler.METADATA_START)
            end_idx = content_with_meta.index(ContentMetadataHandler.METADATA_END)

            metadata_str = content_with_meta[
                start_idx + len(ContentMetadataHandler.METADATA_START) : end_idx
            ].strip()
            metadata = json.loads(metadata_str)

            clean_content = content_with_meta[
                end_idx + len(ContentMetadataHandler.METADATA_END) :
            ].strip()

            return metadata, clean_content
        except (ValueError, json.JSONDecodeError):
            return None, content_with_meta

    # ... other methods from the original class are good ...

    @staticmethod
    def has_metadata(content: str) -> bool:
        """
        Check if content contains metadata.

        Args:
            content: The content to check

        Returns:
            bool: True if metadata is present
        """
        return (
            ContentMetadataHandler.METADATA_START in content
            and ContentMetadataHandler.METADATA_END in content
        )

    @staticmethod
    def strip_metadata(content: str) -> str:
        """
        Remove metadata from content if present.

        Args:
            content: The content to strip metadata from

        Returns:
            Content without metadata
        """
        try:
            metadata_end = content.index(ContentMetadataHandler.METADATA_END)
            return content[
                metadata_end + len(ContentMetadataHandler.METADATA_END) :
            ].strip()
        except ValueError:
            return content

    @staticmethod
    def get_content_hash(content: str) -> str:
        """
        Get hash of content without metadata.

        Args:
            content: The content to hash

        Returns:
            SHA-256 hash of the clean content
        """
        clean_content = ContentMetadataHandler.strip_metadata(content)
        return hashlib.sha256(clean_content.encode("utf-8")).hexdigest()

    @staticmethod
    def content_changed(old_content: str, new_content: str) -> bool:
        """
        Check if content has changed by comparing hashes.

        Args:
            old_content: Previous version of content
            new_content: New version of content

        Returns:
            bool: True if content has changed
        """
        old_hash = ContentMetadataHandler.get_content_hash(old_content)
        new_hash = ContentMetadataHandler.get_content_hash(new_content)
        return old_hash != new_hash


def convert_html_to_markdown(html: str) -> str:
    """A simple HTML to Markdown converter."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n\n").strip()


#
# End of article_scraper/utils.py
#######################################################################################################################
