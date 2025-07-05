# test_confluence_utils.py
#
# Unit tests for Confluence utility functions
#
import pytest
from unittest.mock import MagicMock, patch
from tldw_chatbook.Web_Scraping.Confluence.confluence_utils import (
    convert_confluence_to_markdown,
    extract_confluence_metadata,
    parse_confluence_url,
    format_page_hierarchy,
    estimate_reading_time
)


class TestConfluenceUtils:
    """Test suite for Confluence utility functions"""
    
    def test_convert_simple_html_to_markdown(self):
        """Test basic HTML to Markdown conversion"""
        html = "<p>This is a <strong>test</strong> paragraph.</p>"
        markdown = convert_confluence_to_markdown(html)
        assert "This is a **test** paragraph." in markdown
        
    def test_convert_confluence_code_macro(self):
        """Test Confluence code macro conversion"""
        html = '''
        <ac:structured-macro ac:name="code">
            <ac:parameter ac:name="language">python</ac:parameter>
            <ac:plain-text-body><![CDATA[print("Hello, World!")]]></ac:plain-text-body>
        </ac:structured-macro>
        '''
        markdown = convert_confluence_to_markdown(html)
        assert "```" in markdown or "print(\"Hello, World!\")" in markdown
        
    def test_convert_info_panel(self):
        """Test info panel conversion"""
        html = '''
        <ac:structured-macro ac:name="info">
            <ac:rich-text-body>
                <p>This is important information.</p>
            </ac:rich-text-body>
        </ac:structured-macro>
        '''
        markdown = convert_confluence_to_markdown(html)
        assert "[INFO]" in markdown or "This is important information." in markdown
        
    def test_extract_metadata_basic(self):
        """Test basic metadata extraction"""
        page_data = {
            'id': '12345',
            'title': 'Test Page',
            'type': 'page',
            'version': {'number': 5},
            'space': {'key': 'TEST', 'name': 'Test Space'}
        }
        
        metadata = extract_confluence_metadata(page_data)
        
        assert metadata['page_id'] == '12345'
        assert metadata['title'] == 'Test Page'
        assert metadata['version'] == 5
        assert metadata['space_key'] == 'TEST'
        assert metadata['space_name'] == 'Test Space'
        
    def test_extract_metadata_with_history(self):
        """Test metadata extraction with history"""
        page_data = {
            'id': '12345',
            'title': 'Test Page',
            'history': {
                'createdDate': '2024-01-01T10:00:00Z',
                'createdBy': {'displayName': 'John Doe'},
                'lastUpdated': {
                    'when': '2024-01-15T15:30:00Z',
                    'by': {'displayName': 'Jane Smith'}
                }
            }
        }
        
        metadata = extract_confluence_metadata(page_data)
        
        assert metadata['created_date'] == '2024-01-01T10:00:00Z'
        assert metadata['created_by'] == 'John Doe'
        assert metadata['last_modified'] == '2024-01-15T15:30:00Z'
        assert metadata['modified_by'] == 'Jane Smith'
        
    def test_parse_confluence_url_new_format(self):
        """Test parsing new Confluence URL format"""
        url = "https://example.atlassian.net/wiki/spaces/DEV/pages/12345/Page+Title"
        
        result = parse_confluence_url(url)
        
        assert result['base_url'] == 'https://example.atlassian.net'
        assert result['space_key'] == 'DEV'
        assert result['page_id'] == '12345'
        assert result['page_title'] == 'Page Title'
        assert result['type'] == 'viewpage'
        
    def test_parse_confluence_url_legacy_format(self):
        """Test parsing legacy Confluence URL format"""
        url = "https://confluence.example.com/display/PROJ/Project+Documentation"
        
        result = parse_confluence_url(url)
        
        assert result['base_url'] == 'https://confluence.example.com'
        assert result['space_key'] == 'PROJ'
        assert result['page_title'] == 'Project Documentation'
        assert result['type'] == 'display'
        
    def test_parse_confluence_url_with_page_id_param(self):
        """Test parsing URL with pageId parameter"""
        url = "https://confluence.example.com/pages/viewpage.action?pageId=98765"
        
        result = parse_confluence_url(url)
        
        assert result['page_id'] == '98765'
        
    def test_format_page_hierarchy(self):
        """Test page hierarchy formatting"""
        ancestors = [
            {'title': 'Root'},
            {'title': 'Parent'},
            {'title': 'Sub-Parent'}
        ]
        
        hierarchy = format_page_hierarchy(ancestors, 'Current Page')
        
        assert hierarchy == 'Root > Parent > Sub-Parent > Current Page'
        
    def test_format_page_hierarchy_no_ancestors(self):
        """Test hierarchy with no ancestors"""
        hierarchy = format_page_hierarchy([], 'Standalone Page')
        assert hierarchy == 'Standalone Page'
        
    def test_estimate_reading_time(self):
        """Test reading time estimation"""
        # ~225 words should be about 1 minute
        short_text = " ".join(["word"] * 200)
        assert estimate_reading_time(short_text) == 1
        
        # ~450 words should be about 2 minutes
        medium_text = " ".join(["word"] * 450)
        assert estimate_reading_time(medium_text) == 2
        
        # ~900 words should be about 4 minutes
        long_text = " ".join(["word"] * 900)
        assert estimate_reading_time(long_text) == 4
        
    def test_convert_empty_content(self):
        """Test handling of empty content"""
        assert convert_confluence_to_markdown("") == ""
        assert convert_confluence_to_markdown(None) == ""
        
    def test_convert_confluence_table(self):
        """Test table conversion"""
        html = '''
        <table>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
            <tr>
                <td>Cell 1</td>
                <td>Cell 2</td>
            </tr>
        </table>
        '''
        markdown = convert_confluence_to_markdown(html)
        # Should contain table markers
        assert "|" in markdown
        assert "Header 1" in markdown
        assert "Cell 1" in markdown
        
    def test_handle_confluence_emoticons(self):
        """Test emoticon handling"""
        html = '''
        <p>Great work! <ac:emoticon ac:name="smile"/> 
        Warning <ac:emoticon ac:name="warning"/>
        </p>
        '''
        markdown = convert_confluence_to_markdown(html)
        assert ":)" in markdown or "smile" in markdown
        assert "⚠️" in markdown or "warning" in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])