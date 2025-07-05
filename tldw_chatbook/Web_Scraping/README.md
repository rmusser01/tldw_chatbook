# Web Scraping Module

The Web Scraping module provides comprehensive functionality for extracting content from web pages, performing web searches, and managing browser cookies for authenticated scraping. This module is part of the TLDW Chatbook application and integrates seamlessly with its chat and note-taking features.

## Overview

This module consists of several components:

1. **Article_Extractor_Lib.py** - Core article extraction and scraping functionality
2. **WebSearch_APIs.py** - Integration with multiple search engine APIs
3. **Article_Scraper/** - Modular scraping submodule with advanced features
4. **cookie_scraping/** - Browser cookie extraction for authenticated scraping

## Features

### Core Scraping Features
- Asynchronous web scraping using Playwright
- Content extraction with Trafilatura
- HTML to Markdown conversion
- Metadata tracking and content hashing
- Sitemap parsing and site crawling
- Bookmark and CSV URL import support
- Retry logic and error handling
- Stealth mode support to avoid detection

### Search Engine Integration
- Support for multiple search engines:
  - Google Custom Search
  - Bing Search
  - DuckDuckGo
  - Brave Search
  - Kagi
  - Tavily
  - SearX
  - Baidu (partial)
  - Yandex (partial)
- Sub-query generation for comprehensive searches
- Result relevance analysis
- Automatic summarization of search results

### Cookie Management
- Extract cookies from major browsers:
  - Chrome/Chromium
  - Firefox
  - Microsoft Edge
  - Safari (macOS only)
- Cross-platform support (Windows, macOS, Linux)
- Automatic decryption of browser cookie stores

## Installation

### Core Dependencies
```bash
pip install playwright
pip install trafilatura
pip install beautifulsoup4
pip install lxml
pip install pandas
pip install aiohttp
pip install requests
pip install loguru
```

### Optional Dependencies
For cookie extraction:
```bash
pip install pycryptodomex
pip install keyring
```

For stealth mode:
```bash
pip install playwright-stealth
```

### Playwright Setup
After installing playwright, download browser binaries:
```bash
playwright install chromium
```

## Usage

### Basic Article Scraping

```python
import asyncio
from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import scrape_article

# Scrape a single article
async def main():
    article_data = await scrape_article("https://example.com/article")
    print(f"Title: {article_data['title']}")
    print(f"Content: {article_data['content'][:500]}...")

asyncio.run(main())
```

### Using the Article Scraper Submodule

```python
import asyncio
from tldw_chatbook.Web_Scraping.Article_Scraper import Scraper, ScraperConfig, ProcessorConfig, scrape_and_process_urls

async def main():
    urls = ["https://example.com/article1", "https://example.com/article2"]
    
    # Configure scraper
    scraper_config = ScraperConfig(stealth=True, retries=3)
    processor_config = ProcessorConfig(
        api_name="openai",
        api_key="your_key",
        summarize=True,
        custom_prompt="Summarize the key points"
    )
    
    # Use scraper context manager
    async with Scraper(config=scraper_config) as scraper:
        results = await scrape_and_process_urls(
            urls=urls,
            proc_config=processor_config,
            scraper=scraper
        )
        
    for result in results:
        if result.get('extraction_successful'):
            print(f"URL: {result['url']}")
            print(f"Summary: {result.get('summary', 'N/A')}")

asyncio.run(main())
```

### Web Search Integration

```python
from tldw_chatbook.Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate

# Phase 1: Generate sub-queries and search
search_params = {
    "engine": "google",
    "content_country": "US",
    "search_lang": "en",
    "output_lang": "en",
    "result_count": 10,
    "subquery_generation": True,
    "subquery_generation_llm": "openai",
    "relevance_analysis_llm": "openai",
    "final_answer_llm": "openai"
}

phase1_results = generate_and_search("What is quantum computing?", search_params)

# Phase 2: Analyze and aggregate results
import asyncio
phase2_results = asyncio.run(analyze_and_aggregate(
    phase1_results["web_search_results_dict"],
    phase1_results["sub_query_dict"],
    search_params
))

print(phase2_results["final_answer"])
```

### Cookie Extraction

```python
from tldw_chatbook.Web_Scraping.cookie_scraping.cookie_cloner import get_cookies

# Get cookies for a specific domain
cookies = get_cookies("example.com", browser="chrome")

# Get cookies from all browsers
all_cookies = get_cookies("example.com", browser="all")

# Use cookies with scraping
from tldw_chatbook.Web_Scraping.Article_Scraper import Scraper

async with Scraper(custom_cookies=cookies) as scraper:
    result = await scraper.scrape("https://example.com/protected-content")
```

### Crawling and Sitemap Processing

```python
from tldw_chatbook.Web_Scraping.Article_Scraper.crawler import crawl_site, get_urls_from_sitemap

# Crawl a website
async def crawl():
    urls = await crawl_site(
        base_url="https://example.com",
        max_pages=100,
        max_depth=3
    )
    return urls

# Parse sitemap
async def parse_sitemap():
    urls = await get_urls_from_sitemap("https://example.com/sitemap.xml")
    return urls
```

### Bookmark Import

```python
from tldw_chatbook.Web_Scraping.Article_Scraper.importers import collect_urls_from_file

# Import Chrome bookmarks
bookmarks = collect_urls_from_file("/path/to/Chrome/Bookmarks")

# Import Firefox bookmarks
bookmarks = collect_urls_from_file("/path/to/bookmarks.html")

# Import URLs from CSV
urls = collect_urls_from_file("/path/to/urls.csv")
```

## Configuration

### ScraperConfig Options

- `user_agent`: Browser user agent string
- `request_timeout_ms`: Request timeout in milliseconds (default: 60000)
- `retries`: Number of retry attempts (default: 3)
- `stealth`: Enable stealth mode (default: True)
- `stealth_wait_ms`: Wait time after page load in stealth mode (default: 5000)
- `include_comments`: Include comments in extraction (default: False)
- `include_tables`: Include tables in extraction (default: False)
- `include_images`: Include images in extraction (default: False)

### ProcessorConfig Options

- `api_name`: LLM API to use for processing
- `api_key`: API key for the LLM service
- `summarize`: Enable summarization (default: False)
- `custom_prompt`: Custom prompt for summarization
- `system_message`: System message for LLM
- `temperature`: LLM temperature setting (default: 0.7)
- `keywords`: List of keywords for categorization

## Metadata Handling

The module includes a comprehensive metadata system for tracking scraped content:

```python
from tldw_chatbook.Web_Scraping.Article_Scraper.utils import ContentMetadataHandler

# Add metadata to content
content_with_meta = ContentMetadataHandler.format_content_with_metadata(
    url="https://example.com",
    content="Article content here...",
    pipeline="custom-scraper",
    additional_metadata={"author": "John Doe"}
)

# Extract metadata
metadata, clean_content = ContentMetadataHandler.extract_metadata(content_with_meta)

# Check if content has changed
has_changed = ContentMetadataHandler.content_changed(old_content, new_content)
```

## Error Handling

The module implements comprehensive error handling:

- Automatic retries for failed requests
- Graceful degradation when extraction fails
- Detailed logging of errors and warnings
- Timeout handling for long-running requests

## Performance Considerations

- Uses asynchronous operations for concurrent scraping
- Implements connection pooling for efficient network usage
- Includes progress bars for long-running operations
- Supports batch processing of multiple URLs
- Caches search results to avoid duplicate requests

## Security and Ethics

When using this module:

1. **Respect robots.txt**: Check website policies before scraping
2. **Rate limiting**: Add delays between requests to avoid overwhelming servers
3. **User agent**: Use appropriate user agent strings
4. **Authentication**: Only use cookie extraction for sites you have permission to access
5. **Content rights**: Respect copyright and terms of service

## Troubleshooting

### Common Issues

1. **Playwright not installed**
   ```bash
   playwright install chromium
   ```

2. **Cookie extraction fails**
   - Ensure browser is closed before extracting cookies
   - Check file permissions for cookie database access
   - On Linux, may need to install additional keyring dependencies

3. **Stealth mode not working**
   ```bash
   pip install playwright-stealth
   ```

4. **Search API errors**
   - Verify API keys are correctly configured
   - Check rate limits for your API plan
   - Ensure proper network connectivity

## Contributing

When contributing to this module:

1. Follow existing code patterns and style
2. Add appropriate error handling
3. Include docstrings for new functions
4. Update this README for new features
5. Add unit tests for new functionality

## License

This module is part of the TLDW Chatbook project and follows the same AGPLv3+ license.