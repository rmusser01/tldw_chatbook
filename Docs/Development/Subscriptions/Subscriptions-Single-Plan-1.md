# Single-User Subscription System Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to enhance the tldw_chatbook subscription system for single-user operation. The goal is to create a powerful content monitoring and aggregation system that can track RSS feeds, websites, subreddits, and other sources, automatically generating personalized briefings while preparing for future server synchronization.

## Current State Analysis

### Existing Components
- ✅ **Database Schema**: Comprehensive tables for subscriptions, items, stats, filters
- ✅ **Monitoring Engine**: RSS/Atom feed parsing with security features
- ✅ **Security Layer**: XXE/SSRF protection, rate limiting, circuit breakers
- ✅ **Basic Briefing Generator**: Template-based report generation
- ✅ **Scheduler Framework**: Priority-based task scheduling
- ⚠️ **UI Window**: Stubbed implementation needs completion
- ❌ **Automated Scheduling**: Not integrated with Textual workers
- ❌ **Custom Web Scraping**: No pipeline for arbitrary websites
- ❌ **Advanced Aggregation**: Missing recursive summarization
- ❌ **Narration Pipeline**: Not implemented

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Subscription│  │   Briefing   │  │   Health Monitor      │  │
│  │ Management  │  │Configuration │  │    Dashboard          │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     Core Processing Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Scheduler  │  │  Monitoring  │  │   Aggregation         │  │
│  │   Worker    │  │   Engine     │  │     Engine            │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Scraping   │  │   Content    │  │    Briefing           │  │
│  │  Pipelines  │  │  Processor   │  │   Generator           │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │Subscriptions│  │    Media     │  │      Notes            │  │
│  │     DB      │  │     DB       │  │       DB              │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Web Scraping Pipeline Architecture

**Objective**: Create extensible pipeline for various content sources

**Files to Create**:
- `tldw_chatbook/Subscriptions/web_scraping_pipelines.py`
- `tldw_chatbook/Subscriptions/scrapers/__init__.py`
- `tldw_chatbook/Subscriptions/scrapers/reddit_scraper.py`
- `tldw_chatbook/Subscriptions/scrapers/generic_scraper.py`
- `tldw_chatbook/Subscriptions/scrapers/custom_scraper.py`

**Key Classes**:
```python
class BaseScrapingPipeline(ABC):
    """Abstract base for all scrapers"""
    @abstractmethod
    async def fetch_content(self, url: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def parse_content(self, raw_content: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        pass

class RedditScrapingPipeline(BaseScrapingPipeline):
    """Reddit-specific scraper using RSS feeds"""
    def __init__(self):
        self.base_url = "https://www.reddit.com/r/{subreddit}/.rss"
        self.post_limit = 25
        self.score_threshold = 10
        
class GenericWebScrapingPipeline(BaseScrapingPipeline):
    """BeautifulSoup-based scraper for general websites"""
    def __init__(self, selectors: Dict[str, str]):
        self.selectors = selectors
        self.javascript_enabled = False
        
class CustomScrapingPipeline(BaseScrapingPipeline):
    """User-defined scraping with custom rules"""
    def __init__(self, pipeline_config: Dict[str, Any]):
        self.config = pipeline_config
        self.wait_conditions = []
```

#### 1.2 Baseline Management System

**Objective**: Track content changes intelligently

**Files to Create**:
- `tldw_chatbook/Subscriptions/baseline_manager.py`

**Key Features**:
```python
class BaselineManager:
    """Manage content baselines for change detection"""
    
    def create_baseline(self, subscription_id: int, content: str) -> str:
        """Create initial baseline snapshot"""
        # Store full content on first check
        # Calculate content hash
        # Extract key elements
        
    def detect_changes(self, subscription_id: int, new_content: str) -> ChangeReport:
        """Compare against baseline"""
        # Semantic similarity check
        # Structural diff
        # Token count comparison
        # Return detailed change report
        
    def should_update_baseline(self, change_report: ChangeReport) -> bool:
        """Determine if baseline needs updating"""
        # Based on change percentage
        # Time since last update
        # User preferences
```

**Database Schema Addition**:
```sql
-- Enhanced url_snapshots table
ALTER TABLE url_snapshots ADD COLUMN extracted_text TEXT;
ALTER TABLE url_snapshots ADD COLUMN structural_hash TEXT;
ALTER TABLE url_snapshots ADD COLUMN token_count INTEGER;
ALTER TABLE url_snapshots ADD COLUMN key_elements TEXT; -- JSON
```

#### 1.3 Scheduler Integration

**Objective**: Connect scheduler to Textual's worker system

**Files to Update**:
- `tldw_chatbook/Subscriptions/scheduler.py`
- `tldw_chatbook/Event_Handlers/subscription_events.py`

**Key Components**:
```python
class SubscriptionSchedulerWorker(Worker):
    """Background worker for automated checks"""
    
    def __init__(self, db: SubscriptionsDB):
        super().__init__()
        self.scheduler = SubscriptionScheduler(db)
        self.running = False
        
    async def run(self):
        """Main worker loop"""
        self.running = True
        while self.running:
            # Get next scheduled task
            # Execute check
            # Post results via events
            # Sleep until next task
```

### Phase 2: Advanced Processing (Week 2)

#### 2.1 Aggregation Engine

**Objective**: Intelligent content aggregation with token management

**Files to Create**:
- `tldw_chatbook/Subscriptions/aggregation_engine.py`
- `tldw_chatbook/Subscriptions/token_manager.py`

**Key Classes**:
```python
class AggregationEngine:
    """Aggregate content from multiple sources"""
    
    def __init__(self, token_budget: int = 50000):
        self.token_budget = token_budget
        self.token_manager = TokenBudgetManager()
        
    async def aggregate_items(self, items: List[Dict], config: AggregationConfig) -> AggregatedContent:
        """Main aggregation pipeline"""
        # Group by source/topic
        # Allocate token budgets
        # Process each group
        # Generate final structure
        
class TokenBudgetManager:
    """Manage token allocation across content"""
    
    def allocate_tokens(self, items: List[Dict], total_budget: int) -> Dict[int, int]:
        """Smart token allocation"""
        # Priority-based allocation
        # Ensure minimum per item
        # Reserve for summaries
        
    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        # Use tiktoken or approximation
```

#### 2.2 Recursive Summarization

**Objective**: Handle long content through hierarchical summarization

**Files to Create**:
- `tldw_chatbook/Subscriptions/recursive_summarizer.py`

**Summarization Strategy**:
```python
class RecursiveSummarizer:
    """Multi-level summarization for long content"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm_provider = llm_provider
        self.chunk_size = 2000  # tokens
        self.summary_ratio = 0.2  # 20% of original
        
    async def summarize_content(self, content: str, target_length: int) -> SummarizedContent:
        """Main summarization pipeline"""
        if not self.needs_summarization(content, target_length):
            return SummarizedContent(content, level=0)
            
        # Level 1: Chunk and summarize
        chunks = self.split_into_chunks(content)
        chunk_summaries = await self.summarize_chunks(chunks)
        
        # Level 2: Summarize summaries if needed
        if self.needs_further_reduction(chunk_summaries, target_length):
            return await self.summarize_summaries(chunk_summaries)
            
        return self.combine_summaries(chunk_summaries)
```

#### 2.3 Briefing Templates

**Objective**: Flexible, customizable briefing formats

**Files to Update**:
- `tldw_chatbook/Subscriptions/briefing_generator.py`

**Template System**:
```python
BRIEFING_TEMPLATES = {
    'executive': {
        'name': 'Executive Summary',
        'sections': [
            {'id': 'overview', 'title': 'Executive Overview', 'max_tokens': 500},
            {'id': 'highlights', 'title': 'Key Highlights', 'max_tokens': 1000},
            {'id': 'actions', 'title': 'Recommended Actions', 'max_tokens': 300},
        ],
        'style': 'formal',
        'include_metrics': True
    },
    'technical': {
        'name': 'Technical Digest',
        'sections': [
            {'id': 'summary', 'title': 'Technical Summary', 'max_tokens': 800},
            {'id': 'updates', 'title': 'Code & API Updates', 'max_tokens': 1500},
            {'id': 'security', 'title': 'Security Notices', 'max_tokens': 500},
            {'id': 'issues', 'title': 'Known Issues', 'max_tokens': 700},
        ],
        'style': 'detailed',
        'include_code': True
    },
    'news': {
        'name': 'News Briefing',
        'sections': [
            {'id': 'headlines', 'title': 'Top Headlines', 'max_tokens': 600},
            {'id': 'analysis', 'title': 'In-Depth Analysis', 'max_tokens': 2000},
            {'id': 'trending', 'title': 'Trending Topics', 'max_tokens': 400},
        ],
        'style': 'journalistic',
        'include_images': True
    }
}
```

### Phase 3: UI Implementation (Week 3)

#### 3.1 Complete Subscription Window

**Objective**: Full-featured UI for subscription management

**Files to Update**:
- `tldw_chatbook/UI/Subscription_Window.py`

**UI Components**:
1. **Subscription List**: Tree view with folders/tags
2. **Detail Panel**: Edit subscription settings
3. **New Items View**: Review and process items
4. **Health Dashboard**: Visual monitoring
5. **Briefing Config**: Schedule and template selection
6. **Scraper Builder**: Visual selector tool

#### 3.2 Event Integration

**Objective**: Real-time updates and notifications

**New Events**:
```python
class SubscriptionEvents:
    NewItemsAvailable = "subscription.new_items"
    CheckInProgress = "subscription.check_progress"
    BriefingGenerated = "subscription.briefing_ready"
    HealthAlert = "subscription.health_alert"
    ScraperConfigured = "subscription.scraper_configured"
```

### Phase 4: Advanced Features (Week 4)

#### 4.1 Per-Site Configurations

**Objective**: Custom scraping rules per website

**Database Schema**:
```sql
CREATE TABLE site_configs (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL UNIQUE,
    selectors TEXT, -- JSON
    rate_limit_config TEXT, -- JSON
    authentication_type TEXT,
    javascript_required BOOLEAN DEFAULT 0,
    custom_headers TEXT, -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE scraping_pipelines (
    id INTEGER PRIMARY KEY,
    subscription_id INTEGER NOT NULL,
    pipeline_type TEXT NOT NULL,
    configuration TEXT NOT NULL, -- JSON
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
);
```

#### 4.2 Content Export & Distribution

**Objective**: Multiple output formats and delivery methods

**Export Formats**:
- Markdown (with YAML frontmatter)
- HTML (responsive, styled)
- PDF (via markdown → HTML → PDF)
- JSON (for API consumption)
- EPUB (for e-readers)

**Distribution Methods**:
- Save to Notes (existing integration)
- Export to file
- Copy to clipboard
- Email (optional, via SMTP)
- Webhook delivery

### Phase 5: Future Preparation

#### 5.1 UUID Implementation

**Objective**: Prepare for distributed sync

**Database Updates**:
```sql
-- Add UUID columns to all primary tables
ALTER TABLE subscriptions ADD COLUMN uuid TEXT UNIQUE DEFAULT (lower(hex(randomblob(16))));
ALTER TABLE subscription_items ADD COLUMN uuid TEXT UNIQUE DEFAULT (lower(hex(randomblob(16))));
ALTER TABLE briefing_configs ADD COLUMN uuid TEXT UNIQUE DEFAULT (lower(hex(randomblob(16))));

-- Add sync metadata
ALTER TABLE subscriptions ADD COLUMN sync_version INTEGER DEFAULT 0;
ALTER TABLE subscriptions ADD COLUMN last_synced DATETIME;
ALTER TABLE subscription_items ADD COLUMN sync_status TEXT DEFAULT 'pending';
```

#### 5.2 Narration Pipeline Foundation

**Objective**: Prepare for future audio briefings

**Placeholder Structure**:
```python
class NarrationPipeline:
    """Foundation for future TTS implementation"""
    
    def prepare_for_narration(self, briefing: Briefing) -> NarrationScript:
        """Convert briefing to narration-ready format"""
        # Clean text for speech
        # Add pronunciation hints
        # Structure for audio chapters
        # Generate SSML markup
        
    def generate_podcast_feed(self, briefings: List[Briefing]) -> str:
        """Create podcast RSS feed"""
        # Standard podcast RSS format
        # Episode metadata
        # Placeholder for audio files
```

## Configuration Schema

### config.toml Additions

```toml
[subscriptions]
enabled = true
check_on_startup = true
background_checks = true

[subscriptions.scheduling]
default_check_interval = 3600  # 1 hour
min_check_interval = 300       # 5 minutes
max_concurrent_checks = 10
adaptive_scheduling = true

[subscriptions.scraping]
max_concurrent_scrapers = 5
default_timeout = 30
javascript_renderer = "none"  # or "playwright"
user_agent = "tldw-chatbook/1.0"

[subscriptions.content]
max_content_size_kb = 5000
store_raw_html = false
extract_images = true
preserve_formatting = true

[subscriptions.aggregation]
max_tokens_per_item = 5000
max_tokens_per_briefing = 50000
recursive_summary_threshold = 10000
summary_style = "balanced"  # concise, balanced, detailed

[subscriptions.briefing]
default_template = "executive"
default_schedule = "0 8 * * *"  # cron format
include_original_links = true
include_statistics = true
auto_export_to_notes = false

[subscriptions.reddit]
use_rss = true  # vs API
include_comments = false
score_threshold = 10
post_limit = 25
```

## API Design (Internal)

### Subscription Service API

```python
class SubscriptionService:
    """High-level API for subscription operations"""
    
    async def add_subscription(self, config: SubscriptionConfig) -> Subscription:
        """Add new subscription with validation"""
        
    async def check_subscription(self, subscription_id: int) -> CheckResult:
        """Manually trigger subscription check"""
        
    async def get_new_items(self, filters: ItemFilters) -> List[SubscriptionItem]:
        """Retrieve items with filtering"""
        
    async def generate_briefing(self, config: BriefingConfig) -> Briefing:
        """Generate briefing on-demand"""
        
    async def export_briefing(self, briefing: Briefing, format: ExportFormat) -> bytes:
        """Export briefing to specified format"""
```

## Testing Strategy

### Unit Tests
- Scraping pipeline validation
- Change detection algorithms
- Token counting accuracy
- Summarization quality
- Template rendering

### Integration Tests
- End-to-end subscription checks
- Briefing generation pipeline
- Export format validation
- Database transactions
- Event propagation

### Performance Tests
- Concurrent subscription handling
- Large content summarization
- Database query optimization
- Memory usage monitoring

## Metrics & Monitoring

### Key Metrics to Track
- Subscription check frequency
- Success/failure rates
- Content volume trends
- Processing time per source
- Token usage statistics
- User engagement with briefings

### Health Indicators
- Failed check threshold
- Response time degradation
- Rate limit violations
- Storage usage trends
- Error patterns

## Security Considerations

### Enhanced Protections
- Per-site authentication storage (encrypted)
- JavaScript sandbox for custom scrapers
- Content size limits
- Request timeout enforcement
- IP rotation support (future)

### Privacy Features
- Local-only processing option
- Configurable data retention
- Anonymized metrics
- Secure credential storage

## Migration Path

### From Current State
1. Database schema updates (non-breaking)
2. Add new tables for enhanced features
3. Migrate existing subscriptions
4. Preserve all user data

### To Future Server Sync
1. UUID columns ready
2. Sync metadata in place
3. API-compatible data structures
4. Conflict resolution prepared

## Implementation Priority

### Must Have (P0)
1. Web scraping pipelines
2. Baseline change detection
3. Basic aggregation
4. UI implementation
5. Automated scheduling

### Should Have (P1)
1. Recursive summarization
2. Custom scrapers
3. Multiple templates
4. Health monitoring
5. Reddit integration

### Nice to Have (P2)
1. JavaScript rendering
2. Advanced analytics
3. EPUB export
4. Podcast feed
5. Email delivery

## Success Criteria

### Functional Requirements
- ✓ Monitor RSS/Atom feeds
- ✓ Track website changes
- ✓ Aggregate content intelligently
- ✓ Generate customized briefings
- ✓ Run automated schedules

### Performance Requirements
- Check 100+ subscriptions/hour
- Generate briefing in <30 seconds
- Handle 10MB+ content gracefully
- Maintain <500ms UI responsiveness

### Quality Requirements
- 99% uptime for scheduler
- <1% false positive rate for changes
- Accurate summarization
- Intuitive UI/UX

## Next Steps

1. **Immediate Actions**:
   - Create web scraping pipeline base classes
   - Implement baseline manager
   - Update Subscription Window UI
   - Add scheduler worker integration

2. **Week 1 Deliverables**:
   - Working RSS/URL monitoring
   - Change detection system
   - Basic UI functionality
   - Manual check capability

3. **Week 2 Deliverables**:
   - Aggregation engine
   - Summarization pipeline
   - Template system
   - Automated scheduling

4. **Week 3 Deliverables**:
   - Complete UI
   - Custom scrapers
   - Export formats
   - Health monitoring

5. **Week 4 Deliverables**:
   - Advanced features
   - Performance optimization
   - Documentation
   - Test coverage

## Conclusion

This plan creates a powerful single-user subscription system that can grow into a distributed, multi-user platform. By focusing on clean architecture, extensibility, and user experience, we build a foundation that serves immediate needs while preparing for future enhancements.

The implementation maintains compatibility with existing tldw_chatbook systems while adding significant new capabilities for content monitoring, aggregation, and intelligence extraction.