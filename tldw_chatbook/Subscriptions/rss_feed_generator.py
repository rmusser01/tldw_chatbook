# rss_feed_generator.py
# Description: Generate RSS feeds from scraped website content
#
# This module provides functionality to convert scraped web content into
# RSS/Atom feeds, allowing any website to be treated as a subscription source.
#
# Imports
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import html
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from ..Metrics.metrics_logger import log_counter
#
########################################################################################################################
#
# RSS Feed Generation
#
########################################################################################################################

class RSSFeedGenerator:
    """
    Generate RSS 2.0 feeds from scraped content.
    
    This allows any website to be monitored as if it were an RSS feed,
    enabling subscription-based monitoring of non-RSS sites.
    """
    
    def __init__(self, 
                 feed_title: str,
                 feed_link: str,
                 feed_description: str = "",
                 feed_language: str = "en",
                 feed_generator: str = "tldw-chatbook RSS Generator"):
        """
        Initialize RSS feed generator.
        
        Args:
            feed_title: Title of the generated feed
            feed_link: Link to the source website
            feed_description: Description of the feed
            feed_language: Language code (default: en)
            feed_generator: Generator identification
        """
        self.feed_title = feed_title
        self.feed_link = feed_link
        self.feed_description = feed_description or f"Generated RSS feed for {feed_link}"
        self.feed_language = feed_language
        self.feed_generator = feed_generator
        
        # Create root RSS element
        self.rss = ET.Element('rss', {
            'version': '2.0',
            'xmlns:atom': 'http://www.w3.org/2005/Atom',
            'xmlns:content': 'http://purl.org/rss/1.0/modules/content/',
            'xmlns:dc': 'http://purl.org/dc/elements/1.1/'
        })
        
        # Create channel
        self.channel = ET.SubElement(self.rss, 'channel')
        self._setup_channel()
    
    def _setup_channel(self):
        """Set up RSS channel with metadata."""
        # Required channel elements
        ET.SubElement(self.channel, 'title').text = self.feed_title
        ET.SubElement(self.channel, 'link').text = self.feed_link
        ET.SubElement(self.channel, 'description').text = self.feed_description
        
        # Optional channel elements
        ET.SubElement(self.channel, 'language').text = self.feed_language
        ET.SubElement(self.channel, 'generator').text = self.feed_generator
        ET.SubElement(self.channel, 'lastBuildDate').text = self._format_rfc822_date(datetime.now(timezone.utc))
        
        # Atom self link
        atom_link = ET.SubElement(self.channel, '{http://www.w3.org/2005/Atom}link', {
            'href': self.feed_link,
            'rel': 'self',
            'type': 'application/rss+xml'
        })
    
    def add_item(self, 
                 title: str,
                 link: str,
                 description: str = "",
                 content: Optional[str] = None,
                 author: Optional[str] = None,
                 pub_date: Optional[datetime] = None,
                 categories: Optional[List[str]] = None,
                 guid: Optional[str] = None,
                 enclosure_url: Optional[str] = None,
                 enclosure_type: Optional[str] = None,
                 enclosure_length: Optional[int] = None):
        """
        Add an item to the RSS feed.
        
        Args:
            title: Item title
            link: Item URL
            description: Short description or summary
            content: Full content (uses content:encoded)
            author: Author name/email
            pub_date: Publication date
            categories: List of categories/tags
            guid: Unique identifier (defaults to link)
            enclosure_url: Media attachment URL
            enclosure_type: MIME type of enclosure
            enclosure_length: Size of enclosure in bytes
        """
        item = ET.SubElement(self.channel, 'item')
        
        # Required elements
        ET.SubElement(item, 'title').text = self._escape_text(title)
        ET.SubElement(item, 'link').text = link
        
        # Description (required by RSS spec, but can be empty)
        ET.SubElement(item, 'description').text = self._escape_text(description or title)
        
        # Optional elements
        if content:
            # Use content:encoded for full content
            content_elem = ET.SubElement(item, '{http://purl.org/rss/1.0/modules/content/}encoded')
            content_elem.text = f"<![CDATA[{content}]]>"
        
        if author:
            ET.SubElement(item, 'author').text = self._escape_text(author)
            # Also add dc:creator for better compatibility
            ET.SubElement(item, '{http://purl.org/dc/elements/1.1/}creator').text = self._escape_text(author)
        
        if pub_date:
            ET.SubElement(item, 'pubDate').text = self._format_rfc822_date(pub_date)
        else:
            # Use current time if no pub date provided
            ET.SubElement(item, 'pubDate').text = self._format_rfc822_date(datetime.now(timezone.utc))
        
        # GUID (globally unique identifier)
        guid_elem = ET.SubElement(item, 'guid')
        if guid:
            guid_elem.text = guid
            guid_elem.set('isPermaLink', 'false')
        else:
            guid_elem.text = link
            guid_elem.set('isPermaLink', 'true')
        
        # Categories
        if categories:
            for category in categories:
                ET.SubElement(item, 'category').text = self._escape_text(category)
        
        # Enclosure (for media attachments)
        if enclosure_url and enclosure_type:
            enclosure = ET.SubElement(item, 'enclosure', {
                'url': enclosure_url,
                'type': enclosure_type
            })
            if enclosure_length:
                enclosure.set('length', str(enclosure_length))
        
        log_counter("rss_feed_item_added", labels={
            "has_content": str(bool(content)),
            "has_enclosure": str(bool(enclosure_url))
        })
    
    def add_items_from_scraped_content(self, items: List[Dict[str, Any]]):
        """
        Add multiple items from scraped content.
        
        Args:
            items: List of scraped items with standard fields
        """
        for item in items:
            self.add_item(
                title=item.get('title', 'Untitled'),
                link=item.get('url', ''),
                description=item.get('description', ''),
                content=item.get('content'),
                author=item.get('author'),
                pub_date=item.get('published_date'),
                categories=item.get('categories', []),
                guid=item.get('content_hash') or item.get('url')
            )
    
    def generate_feed(self, pretty_print: bool = True) -> str:
        """
        Generate the RSS feed XML.
        
        Args:
            pretty_print: Format XML with indentation
            
        Returns:
            RSS feed as XML string
        """
        # Update lastBuildDate
        for elem in self.channel:
            if elem.tag == 'lastBuildDate':
                elem.text = self._format_rfc822_date(datetime.now(timezone.utc))
                break
        
        # Convert to string
        if pretty_print:
            self._indent_xml(self.rss)
        
        xml_str = ET.tostring(self.rss, encoding='unicode', method='xml')
        
        # Add XML declaration
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        
        log_counter("rss_feed_generated", labels={
            "item_count": str(len(self.channel.findall('item')))
        })
        
        return xml_declaration + xml_str
    
    def save_feed(self, filepath: str, pretty_print: bool = True):
        """
        Save RSS feed to file.
        
        Args:
            filepath: Path to save the feed
            pretty_print: Format XML with indentation
        """
        feed_content = self.generate_feed(pretty_print)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(feed_content)
        
        logger.info(f"RSS feed saved to: {filepath}")
    
    def _escape_text(self, text: str) -> str:
        """Escape text for XML."""
        if not text:
            return ""
        # Basic HTML escaping
        return html.escape(text, quote=False)
    
    def _format_rfc822_date(self, dt: datetime) -> str:
        """Format datetime as RFC 822 for RSS."""
        # Ensure timezone aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Format as RFC 822
        # Example: Mon, 06 Sep 2009 16:45:00 +0000
        return dt.strftime('%a, %d %b %Y %H:%M:%S %z')
    
    def _indent_xml(self, elem, level=0):
        """Add pretty-printing indentation to XML."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


class AtomFeedGenerator:
    """
    Generate Atom 1.0 feeds from scraped content.
    
    Alternative to RSS for more modern feed format with better
    support for content types and metadata.
    """
    
    def __init__(self,
                 feed_id: str,
                 feed_title: str,
                 feed_link: str,
                 feed_author: Optional[str] = None,
                 feed_subtitle: Optional[str] = None):
        """
        Initialize Atom feed generator.
        
        Args:
            feed_id: Unique feed identifier (usually URL)
            feed_title: Feed title
            feed_link: Feed website URL
            feed_author: Default author name
            feed_subtitle: Feed subtitle/description
        """
        self.feed_id = feed_id
        self.feed_title = feed_title
        self.feed_link = feed_link
        self.feed_author = feed_author
        self.feed_subtitle = feed_subtitle
        
        # Create root feed element
        self.feed = ET.Element('feed', {
            'xmlns': 'http://www.w3.org/2005/Atom'
        })
        
        self._setup_feed()
    
    def _setup_feed(self):
        """Set up Atom feed with metadata."""
        # Required feed elements
        ET.SubElement(self.feed, 'id').text = self.feed_id
        ET.SubElement(self.feed, 'title').text = self.feed_title
        ET.SubElement(self.feed, 'updated').text = self._format_iso8601_date(datetime.now(timezone.utc))
        
        # Link elements
        ET.SubElement(self.feed, 'link', {
            'rel': 'alternate',
            'type': 'text/html',
            'href': self.feed_link
        })
        
        ET.SubElement(self.feed, 'link', {
            'rel': 'self',
            'type': 'application/atom+xml',
            'href': self.feed_id
        })
        
        # Optional elements
        if self.feed_subtitle:
            ET.SubElement(self.feed, 'subtitle').text = self.feed_subtitle
        
        if self.feed_author:
            author = ET.SubElement(self.feed, 'author')
            ET.SubElement(author, 'name').text = self.feed_author
        
        # Generator
        generator = ET.SubElement(self.feed, 'generator', {
            'uri': 'https://github.com/rmusser01/tldw_chatbook',
            'version': '1.0'
        })
        generator.text = 'tldw-chatbook Atom Generator'
    
    def add_entry(self,
                  title: str,
                  link: str,
                  entry_id: Optional[str] = None,
                  summary: Optional[str] = None,
                  content: Optional[str] = None,
                  content_type: str = 'html',
                  author: Optional[str] = None,
                  published: Optional[datetime] = None,
                  updated: Optional[datetime] = None,
                  categories: Optional[List[str]] = None):
        """
        Add an entry to the Atom feed.
        
        Args:
            title: Entry title
            link: Entry URL
            entry_id: Unique entry ID (defaults to link)
            summary: Entry summary
            content: Full content
            content_type: Content MIME type
            author: Author name
            published: Publication date
            updated: Last update date
            categories: List of categories
        """
        entry = ET.SubElement(self.feed, 'entry')
        
        # Required elements
        ET.SubElement(entry, 'id').text = entry_id or link
        ET.SubElement(entry, 'title').text = title
        
        # Link
        ET.SubElement(entry, 'link', {
            'rel': 'alternate',
            'type': 'text/html',
            'href': link
        })
        
        # Dates
        if updated:
            ET.SubElement(entry, 'updated').text = self._format_iso8601_date(updated)
        else:
            ET.SubElement(entry, 'updated').text = self._format_iso8601_date(datetime.now(timezone.utc))
        
        if published:
            ET.SubElement(entry, 'published').text = self._format_iso8601_date(published)
        
        # Content elements
        if summary:
            ET.SubElement(entry, 'summary', {'type': 'text'}).text = summary
        
        if content:
            content_elem = ET.SubElement(entry, 'content', {'type': content_type})
            if content_type == 'html':
                content_elem.text = f"<![CDATA[{content}]]>"
            else:
                content_elem.text = content
        
        # Author
        if author or self.feed_author:
            author_elem = ET.SubElement(entry, 'author')
            ET.SubElement(author_elem, 'name').text = author or self.feed_author
        
        # Categories
        if categories:
            for category in categories:
                ET.SubElement(entry, 'category', {'term': category})
    
    def generate_feed(self, pretty_print: bool = True) -> str:
        """Generate Atom feed XML."""
        # Update feed's updated timestamp
        for elem in self.feed:
            if elem.tag == 'updated':
                elem.text = self._format_iso8601_date(datetime.now(timezone.utc))
                break
        
        if pretty_print:
            self._indent_xml(self.feed)
        
        xml_str = ET.tostring(self.feed, encoding='unicode', method='xml')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        
        return xml_declaration + xml_str
    
    def _format_iso8601_date(self, dt: datetime) -> str:
        """Format datetime as ISO 8601 for Atom."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    
    def _indent_xml(self, elem, level=0):
        """Add pretty-printing indentation to XML."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


class WebsiteToFeedConverter:
    """
    Convert website content to RSS/Atom feeds.
    
    This class bridges the gap between websites without feeds
    and the subscription system's feed-based monitoring.
    """
    
    def __init__(self, 
                 website_url: str,
                 feed_type: str = 'rss',
                 feed_title: Optional[str] = None,
                 feed_description: Optional[str] = None):
        """
        Initialize website to feed converter.
        
        Args:
            website_url: Base URL of the website
            feed_type: 'rss' or 'atom'
            feed_title: Custom feed title
            feed_description: Custom feed description
        """
        self.website_url = website_url
        self.feed_type = feed_type.lower()
        
        # Parse domain for default title
        parsed = urlparse(website_url)
        domain = parsed.netloc
        
        self.feed_title = feed_title or f"{domain} Feed"
        self.feed_description = feed_description or f"Generated feed for {domain}"
        
        # Initialize appropriate generator
        if self.feed_type == 'atom':
            self.generator = AtomFeedGenerator(
                feed_id=website_url,
                feed_title=self.feed_title,
                feed_link=website_url,
                feed_subtitle=self.feed_description
            )
        else:
            self.generator = RSSFeedGenerator(
                feed_title=self.feed_title,
                feed_link=website_url,
                feed_description=self.feed_description
            )
    
    def convert_items_to_feed(self, items: List[Dict[str, Any]]) -> str:
        """
        Convert scraped items to feed format.
        
        Args:
            items: List of scraped items
            
        Returns:
            Generated feed as XML string
        """
        if not items:
            logger.warning("No items to convert to feed")
            return self.generator.generate_feed()
        
        # Sort items by date (newest first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get('published_date', datetime.now(timezone.utc)),
            reverse=True
        )
        
        # Add items to feed
        for item in sorted_items:
            if self.feed_type == 'atom':
                self.generator.add_entry(
                    title=item.get('title', 'Untitled'),
                    link=item.get('url', ''),
                    summary=item.get('description'),
                    content=item.get('content'),
                    author=item.get('author'),
                    published=item.get('published_date'),
                    updated=item.get('updated_date', item.get('published_date')),
                    categories=item.get('categories', []),
                    entry_id=self._generate_entry_id(item)
                )
            else:
                self.generator.add_item(
                    title=item.get('title', 'Untitled'),
                    link=item.get('url', ''),
                    description=item.get('description', ''),
                    content=item.get('content'),
                    author=item.get('author'),
                    pub_date=item.get('published_date'),
                    categories=item.get('categories', []),
                    guid=self._generate_entry_id(item)
                )
        
        return self.generator.generate_feed()
    
    def _generate_entry_id(self, item: Dict[str, Any]) -> str:
        """Generate unique ID for feed entry."""
        # Use content hash if available
        if item.get('content_hash'):
            return f"hash:{item['content_hash']}"
        
        # Otherwise create hash from URL and title
        unique_str = f"{item.get('url', '')}{item.get('title', '')}"
        return hashlib.sha256(unique_str.encode()).hexdigest()
    
    def save_feed(self, items: List[Dict[str, Any]], filepath: str):
        """
        Convert items and save feed to file.
        
        Args:
            items: List of scraped items
            filepath: Path to save feed
        """
        feed_content = self.convert_items_to_feed(items)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(feed_content)
        
        logger.info(f"Feed saved to: {filepath}")


# End of rss_feed_generator.py