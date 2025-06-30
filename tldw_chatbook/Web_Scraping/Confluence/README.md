# Confluence Web Scraping Module

This module provides comprehensive functionality for scraping Confluence pages and spaces, building on the existing web scraping infrastructure in tldw_chatbook.

## Features

- **Multiple Authentication Methods**: API token, OAuth, Basic auth, and cookie-based authentication
- **Flexible Scraping Options**: Single pages, entire spaces, or search-based scraping
- **Content Conversion**: Automatic conversion from Confluence storage format to clean Markdown
- **Metadata Preservation**: Maintains page metadata including authors, labels, and hierarchy
- **Smart Crawling**: Configurable depth limits, page filtering, and rate limiting
- **Database Integration**: Direct ingestion to Media_DB_v2 with Confluence-specific metadata

## Configuration

Add the following section to your `config.toml` file:

```toml
[Confluence]
# Base URL of your Confluence instance
base_url = "https://your-domain.atlassian.net/wiki"

# Authentication method: "api_token", "oauth", "basic", or "cookies"
auth_method = "api_token"

# For API token auth (recommended for Atlassian Cloud)
username = "your-email@example.com"
api_token = "your-api-token-here"

# For OAuth authentication
oauth_token = ""

# For basic auth (self-hosted Confluence)
password = ""

# For cookie-based auth
browser = "all"  # or "chrome", "firefox", "edge", "safari"

# Scraping configuration
space_keys = ["SPACE1", "SPACE2"]  # List of default spaces to scrape
max_pages_per_space = 100
max_crawl_depth = 5
include_attachments = false
follow_links = false
rate_limit_delay = 0.5  # Seconds between requests
```

### Environment Variables

You can also use environment variables for sensitive information:

- `CONFLUENCE_BASE_URL`
- `CONFLUENCE_AUTH_METHOD`
- `CONFLUENCE_USERNAME`
- `CONFLUENCE_API_TOKEN`
- `CONFLUENCE_OAUTH_TOKEN`
- `CONFLUENCE_PASSWORD`

## Usage Examples

### Python API

```python
from tldw_chatbook.Web_Scraping.Confluence import (
    scrape_confluence_page,
    scrape_confluence_space,
    scrape_confluence_search
)

# Scrape a single page
result = await scrape_confluence_page(
    "https://example.atlassian.net/wiki/spaces/DEV/pages/12345/Page-Title",
    include_attachments=True,
    ingest_to_db=True,
    keywords="python, documentation"
)

# Scrape an entire space
pages = await scrape_confluence_space(
    "DEV",
    max_pages=50,
    max_depth=3,
    page_filter={
        "include_patterns": ["API", "Guide"],
        "exclude_patterns": ["Draft", "Archive"],
        "labels": ["published"],
        "modified_after": "2024-01-01"
    },
    ingest_to_db=True
)

# Search and scrape
results = await scrape_confluence_search(
    "text ~ 'python' AND space = 'DEV' AND type = page",
    max_pages=20,
    crawl_children=True,
    max_child_depth=2
)
```

### Synchronous Wrappers

For CLI usage, synchronous wrappers are provided:

```python
from tldw_chatbook.Web_Scraping.Confluence import (
    sync_scrape_confluence_page,
    sync_scrape_confluence_space,
    sync_scrape_confluence_search
)

# Same parameters as async versions
page = sync_scrape_confluence_page("https://...")
```

## Authentication Methods

### 1. API Token (Recommended for Atlassian Cloud)
Generate an API token from your Atlassian account settings and use with your email address.

### 2. OAuth
For applications requiring OAuth authentication. Set the `oauth_token` in config.

### 3. Basic Authentication
For self-hosted Confluence instances. Use username and password.

### 4. Cookie-based Authentication
Automatically extracts cookies from your browser. Useful for SSO-protected instances.

## Content Processing

The module handles various Confluence-specific elements:

- **Macros**: Code blocks, info panels, expand sections, etc.
- **Tables**: Proper markdown table conversion
- **Links**: Internal page links, user mentions, attachments
- **Emoticons**: Converted to text equivalents
- **Metadata**: Preserved in structured format

## Database Integration

Scraped content is automatically formatted for ingestion into Media_DB_v2 with:

- Confluence-specific metadata (space, labels, hierarchy)
- Automatic keyword generation from space keys and labels
- Content deduplication via hashing
- Version tracking

## Advanced Features

### Custom Page Filtering

```python
from tldw_chatbook.Web_Scraping.Confluence import create_page_filter

# Create a custom filter
filter_func = create_page_filter(
    include_patterns=["Tutorial", "Guide"],
    exclude_patterns=["WIP", "Deprecated"],
    labels=["official", "reviewed"],
    modified_after="2024-01-01"
)

# Use with crawler
pages = await crawler.crawl_space("DOCS", page_filter=filter_func)
```

### Batch Operations

```python
# Scrape multiple pages concurrently
page_ids = ["12345", "67890", "11111"]
results = await scraper.scrape_many(page_ids)
```

### Progress Tracking

The crawler provides statistics:

```python
stats = crawler.get_crawl_stats()
print(f"Pages visited: {stats['pages_visited']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Error Handling

The module includes comprehensive error handling:

- Automatic retries for transient failures
- Graceful handling of permission errors
- Clear error messages for configuration issues
- Failed pages tracking in crawler statistics

## Performance Considerations

- Rate limiting to avoid overwhelming the server
- Concurrent page processing where appropriate
- Efficient API usage with proper field expansion
- Memory-efficient streaming for large spaces

## Troubleshooting

### Common Issues

1. **Authentication Failed**: Check API token and username
2. **No Pages Found**: Verify space key and permissions
3. **Slow Performance**: Adjust `rate_limit_delay` in config
4. **Missing Content**: Some macros may require special handling

### Debug Mode

Enable debug logging:

```python
from loguru import logger
logger.add("confluence_debug.log", level="DEBUG")
```

## Future Enhancements

- Export to various formats (PDF, DOCX)
- Incremental sync (only changed pages)
- Confluence Cloud API v2 support
- Enhanced macro handling
- Attachment downloading and processing