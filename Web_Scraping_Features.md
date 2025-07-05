# Web Scraping Module - Complete Feature List

This document provides a comprehensive overview of all features available in the tldw_chatbook web scraping module.

## Core Scraping Features

### 1. Browser Automation (Playwright)
- **Headless Chromium browser** for dynamic content rendering
- **Stealth mode** to avoid bot detection (via playwright-stealth)
- **Custom user agent** configuration
- **Viewport settings** (1280x720 default)
- **JavaScript execution** support for SPA/dynamic sites
- **Network state waiting** (networkidle, domcontentloaded)
- **Configurable timeouts** (default 60s, max 600s)

### 2. Content Extraction (Trafilatura)
- **Main content extraction** with automatic boilerplate removal
- **Metadata extraction** (title, author, date, description)
- **Table extraction** (optional flag)
- **Comment extraction** (optional flag)  
- **Image extraction** (optional flag with alt text)
- **Link preservation** in markdown format
- **Language detection** for extracted content

### 3. Authentication & Access
- **Cookie injection** for authenticated scraping
  - Manual cookie string input
  - Browser cookie extraction (Chrome, Firefox, Edge, Safari)
  - Automatic cookie decryption
- **Custom headers** support
- **OAuth token** support (for Confluence)
- **API token** authentication
- **Basic auth** (username/password)

### 4. Advanced Crawling
- **Sitemap parsing** (XML sitemap support)
  - Recursive sitemap index parsing
  - URL filtering from sitemaps
- **Website crawling** with configurable:
  - Maximum pages limit
  - Maximum depth limit
  - URL pattern filtering (include/exclude)
  - Same-domain restriction
- **Link following** with intelligent URL resolution
- **Duplicate URL detection** and removal
- **URL validation** and sanitization

### 5. Batch Processing
- **Concurrent scraping** with semaphore control (default 5 concurrent)
- **Progress tracking** with counters:
  - Success count
  - Failure count  
  - In-progress count
- **Bulk URL import** from multiple sources:
  - Text files (one URL per line)
  - CSV files with URL column
  - Browser bookmarks (HTML export)
  - Browser bookmark databases (Chrome, Firefox)
- **URL deduplication** before processing

### 6. Error Handling & Reliability
- **Automatic retry logic** (configurable attempts, default 3)
- **Exponential backoff** between retries
- **Timeout handling** with graceful failure
- **Network error recovery**
- **Invalid URL detection** and reporting
- **Partial success handling** in batch operations

### 7. Content Processing
- **HTML to Markdown conversion** with clean formatting
- **Content hashing** for change detection
- **Metadata preservation** in structured format
- **Content deduplication** via SHA-256 hashing
- **Text normalization** and cleaning
- **Encoding detection** and conversion

### 8. Performance Optimization
- **Connection pooling** for efficient requests
- **Resource cleanup** via context managers
- **Memory-efficient streaming** for large content
- **Rate limiting** to respect server resources
- **Configurable delays** between requests
- **Browser instance reuse** for multiple pages

### 9. Integration Features
- **Direct database ingestion** to Media_DB_v2
- **Keyword extraction** and tagging
- **LLM summarization** (optional with multiple providers)
- **Custom prompt processing** for content analysis
- **Search engine integration** (Google, Bing, DuckDuckGo, etc.)
- **TLDW API compatibility** for remote processing

### 10. Specialized Scrapers

#### Confluence Integration
- Multiple authentication methods (API token, OAuth, cookies)
- Space-level scraping with hierarchy preservation
- Page filtering by labels, patterns, date
- Confluence-specific macro handling
- Search-based scraping with CQL queries
- Attachment handling

#### MediaWiki Support
- Wiki page extraction with proper formatting
- Category and template handling
- Revision history access

### 11. Output Formats
- **Markdown** (primary format)
- **Plain text** extraction
- **JSON** with structured metadata
- **Database records** with full metadata

### 12. Configuration Options

#### ScraperConfig
- `user_agent`: Custom browser identification
- `request_timeout_ms`: Page load timeout (default 60000)
- `retries`: Retry attempts (default 3)
- `stealth`: Enable stealth mode (default True)
- `stealth_wait_ms`: Wait after load in stealth (default 5000)
- `include_comments`: Extract comments (default False)
- `include_tables`: Extract tables (default False)
- `include_images`: Extract images (default False)

#### ProcessorConfig
- `api_name`: LLM provider for processing
- `api_key`: Authentication for LLM
- `summarize`: Enable summarization
- `custom_prompt`: Custom analysis prompt
- `temperature`: LLM creativity setting
- `keywords`: Categorization keywords

### 13. Security Features
- **Path traversal prevention** for file operations
- **SQL injection protection** for database operations
- **Input validation** for all user inputs
- **Secure temporary file handling**
- **Cookie encryption** support
- **HTTPS upgrade** for HTTP URLs

### 14. Monitoring & Logging
- **Detailed progress reporting** during batch operations
- **Error logging** with context via loguru
- **Performance metrics** collection
- **Success rate tracking**
- **Failed URL reporting** with reasons

### 15. UI Integration Features
- **Real-time progress updates** via Textual events
- **Cancellable operations** with cleanup
- **Status message display** with scrolling
- **URL validation feedback**
- **Drag-and-drop** file support (where applicable)

## Currently Missing from Local Web Scraping UI

Based on the examination of `IngestLocalWebArticleWindow.py`, the following features are available in the backend but not exposed in the UI:

1. **Sitemap/Crawling Features**
   - No option to input sitemap URL
   - No crawl depth configuration
   - No maximum pages limit
   - No URL pattern filtering

2. **Bookmark Import**
   - "Import from File" exists but may not support browser bookmarks
   - No browser selection for cookie/bookmark extraction

3. **Stealth Mode**
   - Backend supports but no UI toggle

4. **Advanced Extraction Options**
   - No toggle for table extraction
   - No toggle for comment extraction

5. **Retry Configuration**
   - No option to set retry attempts
   - No timeout customization beyond wait time

6. **Rate Limiting**
   - No delay configuration between requests
   - No concurrent scraping limit setting

7. **Batch Processing Enhancements**
   - Progress tracking exists but may need enhancement
   - No batch size configuration

8. **Content Processing**
   - No LLM summarization option
   - No custom prompt input for analysis

9. **Confluence/Wiki Support**
   - No specialized options for these platforms

10. **Output Options**
    - No format selection (always uses default)
    - No option to skip database ingestion

## Recommendations

1. Add a "Crawl Website" mode with sitemap/depth options
2. Enhance "Import from File" to explicitly support bookmarks/CSV
3. Add "Advanced Extraction" collapsible with table/comment options
4. Add "Performance" collapsible with retry/timeout/rate limit settings
5. Consider adding LLM processing options as optional feature
6. Add stealth mode toggle in advanced options