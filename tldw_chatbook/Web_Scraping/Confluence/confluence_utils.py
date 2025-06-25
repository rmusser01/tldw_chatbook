# confluence_utils.py
#
# Utility functions for Confluence content processing and conversion
#
# Imports
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import html
#
# Third-party imports
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as md
from loguru import logger
#
#######################################################################################################################
#
# Functions

def convert_confluence_to_markdown(html_content: str) -> str:
    """
    Convert Confluence storage format HTML to clean Markdown
    
    Args:
        html_content: Confluence storage format HTML
        
    Returns:
        Clean markdown text
    """
    if not html_content:
        return ""
        
    try:
        # First, handle Confluence-specific elements
        processed_html = preprocess_confluence_html(html_content)
        
        # Convert to markdown using markdownify with custom settings
        markdown = md(
            processed_html,
            heading_style="atx",
            bullets="-",
            code_language="",
            autolinks=True,
            escape_asterisks=False,
            escape_underscores=False,
            strip=['style', 'script']
        )
        
        # Post-process the markdown
        markdown = postprocess_markdown(markdown)
        
        return markdown.strip()
        
    except Exception as e:
        logger.error(f"Error converting Confluence HTML to Markdown: {str(e)}")
        # Fallback to basic text extraction
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator='\n\n').strip()


def preprocess_confluence_html(html: str) -> str:
    """
    Pre-process Confluence HTML to handle special elements
    
    Args:
        html: Raw Confluence storage format HTML
        
    Returns:
        Processed HTML ready for markdown conversion
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Handle Confluence macros
    handle_confluence_macros(soup)
    
    # Handle structured macros (code blocks, panels, etc.)
    handle_structured_macros(soup)
    
    # Handle tables with Confluence-specific attributes
    handle_confluence_tables(soup)
    
    # Handle confluence links
    handle_confluence_links(soup)
    
    # Handle emoticons
    handle_emoticons(soup)
    
    # Clean up empty paragraphs and excessive whitespace
    cleanup_html(soup)
    
    return str(soup)


def handle_confluence_macros(soup: BeautifulSoup) -> None:
    """Handle various Confluence macros"""
    
    # Code macro
    for macro in soup.find_all('ac:structured-macro', {'ac:name': 'code'}):
        language = ''
        code_content = ''
        
        # Extract language parameter
        for param in macro.find_all('ac:parameter', {'ac:name': 'language'}):
            language = param.get_text(strip=True)
            
        # Extract code content
        plain_text = macro.find('ac:plain-text-body')
        if plain_text:
            code_content = plain_text.get_text()
            
        # Create a pre/code block
        code_block = soup.new_tag('pre')
        code_tag = soup.new_tag('code')
        if language:
            code_tag['class'] = f'language-{language}'
        code_tag.string = code_content
        code_block.append(code_tag)
        macro.replace_with(code_block)
    
    # Info/Warning/Note panels
    panel_types = ['info', 'warning', 'note', 'tip', 'error']
    for panel_type in panel_types:
        for macro in soup.find_all('ac:structured-macro', {'ac:name': panel_type}):
            content = macro.find('ac:rich-text-body')
            if content:
                # Create a blockquote with panel type indicator
                blockquote = soup.new_tag('blockquote')
                indicator = soup.new_tag('strong')
                indicator.string = f'[{panel_type.upper()}] '
                blockquote.append(indicator)
                
                # Move content to blockquote
                for child in list(content.children):
                    blockquote.append(child)
                    
                macro.replace_with(blockquote)
    
    # Expand macro (collapsible content)
    for macro in soup.find_all('ac:structured-macro', {'ac:name': 'expand'}):
        title = 'Click to expand'
        for param in macro.find_all('ac:parameter', {'ac:name': 'title'}):
            title = param.get_text(strip=True)
            
        content = macro.find('ac:rich-text-body')
        if content:
            # Create details/summary element
            details = soup.new_tag('details')
            summary = soup.new_tag('summary')
            summary.string = title
            details.append(summary)
            
            for child in list(content.children):
                details.append(child)
                
            macro.replace_with(details)
    
    # Table of Contents macro - remove it
    for macro in soup.find_all('ac:structured-macro', {'ac:name': 'toc'}):
        macro.decompose()


def handle_structured_macros(soup: BeautifulSoup) -> None:
    """Handle other structured macros that weren't caught by specific handlers"""
    
    for macro in soup.find_all('ac:structured-macro'):
        macro_name = macro.get('ac:name', 'unknown')
        
        # If it has rich text body, preserve the content
        body = macro.find('ac:rich-text-body')
        if body:
            # Add macro name as context
            context = soup.new_tag('p')
            context_text = soup.new_tag('em')
            context_text.string = f'[{macro_name} macro content:]'
            context.append(context_text)
            
            # Create container div
            container = soup.new_tag('div')
            container.append(context)
            
            # Move body content
            for child in list(body.children):
                container.append(child)
                
            macro.replace_with(container)
        else:
            # For macros without rich text body, just indicate their presence
            placeholder = soup.new_tag('p')
            placeholder_text = soup.new_tag('em')
            placeholder_text.string = f'[{macro_name} macro]'
            placeholder.append(placeholder_text)
            macro.replace_with(placeholder)


def handle_confluence_tables(soup: BeautifulSoup) -> None:
    """Process Confluence tables to ensure proper markdown conversion"""
    
    for table in soup.find_all('table'):
        # Remove Confluence-specific attributes
        for attr in ['ac:schema-version', 'class', 'style']:
            if table.has_attr(attr):
                del table[attr]
                
        # Ensure proper table structure
        # Check if table has thead
        if not table.find('thead'):
            # If first row contains th elements, wrap it in thead
            first_row = table.find('tr')
            if first_row and first_row.find('th'):
                thead = soup.new_tag('thead')
                tbody = soup.new_tag('tbody')
                
                # Move first row to thead
                thead.append(first_row.extract())
                
                # Move remaining rows to tbody
                for row in list(table.find_all('tr')):
                    tbody.append(row.extract())
                    
                table.append(thead)
                table.append(tbody)


def handle_confluence_links(soup: BeautifulSoup) -> None:
    """Handle Confluence-specific link formats"""
    
    # Handle ac:link elements
    for link in soup.find_all('ac:link'):
        url = ''
        text = ''
        
        # Try to get page reference
        page_ref = link.find('ri:page')
        if page_ref:
            title = page_ref.get('ri:content-title', '')
            space = page_ref.get('ri:space-key', '')
            if title:
                # Create a relative link format
                url = f"[{space}:{title}]"
                
        # Try to get attachment reference
        attachment_ref = link.find('ri:attachment')
        if attachment_ref:
            filename = attachment_ref.get('ri:filename', '')
            if filename:
                url = f"[attachment:{filename}]"
                
        # Get link text
        link_body = link.find('ac:link-body')
        if link_body:
            text = link_body.get_text(strip=True)
        else:
            text = url
            
        # Create standard anchor tag
        if url:
            a_tag = soup.new_tag('a', href=url)
            a_tag.string = text
            link.replace_with(a_tag)
        else:
            # If we can't determine the link, just use the text
            link.replace_with(text)
            
    # Handle user links
    for user_link in soup.find_all('ri:user'):
        username = user_link.get('ri:username', 'unknown')
        mention = soup.new_tag('span')
        mention.string = f'@{username}'
        user_link.replace_with(mention)


def handle_emoticons(soup: BeautifulSoup) -> None:
    """Convert Confluence emoticons to text equivalents"""
    
    emoticon_map = {
        'smile': ':)',
        'sad': ':(',
        'wink': ';)',
        'cheeky': ':P',
        'laugh': ':D',
        'thumbs-up': '👍',
        'thumbs-down': '👎',
        'information': 'ℹ️',
        'tick': '✓',
        'cross': '✗',
        'warning': '⚠️',
        'star': '⭐',
        'heart': '❤️',
        'broken-heart': '💔'
    }
    
    for emoticon in soup.find_all('ac:emoticon'):
        name = emoticon.get('ac:name', '')
        replacement = emoticon_map.get(name, f':{name}:')
        emoticon.replace_with(replacement)


def cleanup_html(soup: BeautifulSoup) -> None:
    """Clean up the HTML for better markdown conversion"""
    
    # Remove empty paragraphs
    for p in soup.find_all('p'):
        if not p.get_text(strip=True) and not p.find_all(['img', 'br']):
            p.decompose()
            
    # Remove unnecessary divs and spans with no attributes
    for tag in soup.find_all(['div', 'span']):
        if not tag.attrs and tag.string:
            tag.unwrap()
            
    # Remove confluence layout tags
    for layout in soup.find_all(['ac:layout', 'ac:layout-section', 'ac:layout-cell']):
        layout.unwrap()


def postprocess_markdown(markdown: str) -> str:
    """
    Post-process the converted markdown to clean it up
    
    Args:
        markdown: Raw markdown from conversion
        
    Returns:
        Cleaned markdown
    """
    # Remove multiple consecutive blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # Fix code blocks that might have been broken
    markdown = re.sub(r'```\s*\n\s*```', '', markdown)
    
    # Clean up whitespace around headers
    markdown = re.sub(r'\n{2,}(#{1,6}\s)', r'\n\n\1', markdown)
    markdown = re.sub(r'(#{1,6}\s.*?)\n{2,}', r'\1\n\n', markdown)
    
    # Ensure lists have proper spacing
    markdown = re.sub(r'(\n[-*]\s.*?)(\n)([^-*\s])', r'\1\n\n\3', markdown)
    
    # Clean up link formatting
    markdown = re.sub(r'\[([^\]]+)\]\[([^\]]+)\]', r'[\1](\2)', markdown)
    
    # Remove trailing spaces
    lines = markdown.split('\n')
    lines = [line.rstrip() for line in lines]
    markdown = '\n'.join(lines)
    
    return markdown


def extract_confluence_metadata(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from Confluence page data
    
    Args:
        page_data: Raw page data from Confluence API
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        'page_id': page_data.get('id', ''),
        'title': page_data.get('title', ''),
        'type': page_data.get('type', 'page'),
        'status': page_data.get('status', 'current'),
        'version': 1,
        'space_key': '',
        'space_name': '',
        'created_date': '',
        'created_by': '',
        'last_modified': '',
        'modified_by': '',
        'labels': [],
        'ancestors': [],
        'web_url': ''
    }
    
    # Extract version info
    version_info = page_data.get('version', {})
    metadata['version'] = version_info.get('number', 1)
    metadata['modified_by'] = version_info.get('by', {}).get('displayName', '')
    
    # Extract space info
    space_info = page_data.get('space', {})
    metadata['space_key'] = space_info.get('key', '')
    metadata['space_name'] = space_info.get('name', '')
    
    # Extract history info
    history = page_data.get('history', {})
    if history:
        created = history.get('createdDate', '')
        if created:
            metadata['created_date'] = created
            
        created_by = history.get('createdBy', {})
        metadata['created_by'] = created_by.get('displayName', '')
        
        last_updated = history.get('lastUpdated', {})
        if last_updated:
            metadata['last_modified'] = last_updated.get('when', '')
            metadata['modified_by'] = last_updated.get('by', {}).get('displayName', '')
    
    # Extract labels
    labels_data = page_data.get('metadata', {}).get('labels', {}).get('results', [])
    metadata['labels'] = [label.get('name', '') for label in labels_data]
    
    # Extract ancestors (parent pages)
    ancestors_data = page_data.get('ancestors', [])
    metadata['ancestors'] = [
        {
            'id': ancestor.get('id', ''),
            'title': ancestor.get('title', ''),
            'type': ancestor.get('type', '')
        }
        for ancestor in ancestors_data
    ]
    
    # Build web URL
    links = page_data.get('_links', {})
    if 'webui' in links:
        base_url = links.get('base', '')
        web_path = links.get('webui', '')
        metadata['web_url'] = base_url + web_path
    
    return metadata


def parse_confluence_url(url: str) -> Dict[str, Optional[str]]:
    """
    Parse a Confluence URL to extract useful information
    
    Args:
        url: Confluence page URL
        
    Returns:
        Dictionary with parsed URL components
    """
    result = {
        'base_url': None,
        'space_key': None,
        'page_title': None,
        'page_id': None,
        'type': None  # 'display' or 'viewpage'
    }
    
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    # Extract base URL
    result['base_url'] = f"{parsed.scheme}://{parsed.netloc}"
    
    # Handle different URL formats
    if 'display' in path_parts:
        # Legacy format: /display/SPACE/Page+Title
        idx = path_parts.index('display')
        if idx + 1 < len(path_parts):
            result['space_key'] = path_parts[idx + 1]
        if idx + 2 < len(path_parts):
            result['page_title'] = path_parts[idx + 2].replace('+', ' ')
        result['type'] = 'display'
        
    elif 'spaces' in path_parts and 'pages' in path_parts:
        # New format: /spaces/SPACE/pages/12345/Page+Title
        try:
            space_idx = path_parts.index('spaces')
            if space_idx + 1 < len(path_parts):
                result['space_key'] = path_parts[space_idx + 1]
                
            page_idx = path_parts.index('pages')
            if page_idx + 1 < len(path_parts):
                result['page_id'] = path_parts[page_idx + 1]
            if page_idx + 2 < len(path_parts):
                result['page_title'] = path_parts[page_idx + 2].replace('+', ' ')
                
        except ValueError:
            pass
            
        result['type'] = 'viewpage'
    
    # Check query parameters for page ID
    query_params = parse_qs(parsed.query)
    if 'pageId' in query_params:
        result['page_id'] = query_params['pageId'][0]
    
    return result


def format_page_hierarchy(ancestors: List[Dict[str, str]], current_title: str) -> str:
    """
    Format page hierarchy as a breadcrumb string
    
    Args:
        ancestors: List of ancestor pages
        current_title: Title of current page
        
    Returns:
        Formatted hierarchy string
    """
    if not ancestors:
        return current_title
        
    hierarchy = []
    for ancestor in ancestors:
        hierarchy.append(ancestor.get('title', 'Unknown'))
    hierarchy.append(current_title)
    
    return ' > '.join(hierarchy)


def estimate_reading_time(content: str) -> int:
    """
    Estimate reading time for content in minutes
    
    Args:
        content: Text content
        
    Returns:
        Estimated reading time in minutes
    """
    # Average reading speed is 200-250 words per minute
    words = len(content.split())
    minutes = max(1, round(words / 225))
    return minutes

#
# End of confluence_utils.py
#######################################################################################################################