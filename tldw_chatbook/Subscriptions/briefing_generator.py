# briefing_generator.py
# Description: Generate briefings and newsletters from subscription content
#
# This module provides:
# - Aggregate content from multiple subscriptions
# - LLM-powered summarization
# - Custom report templates
# - Multiple export formats (Markdown, HTML, PDF)
# - Scheduled briefing generation
#
# Imports
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
#
# Third-Party Imports
from jinja2 import Template, Environment, FileSystemLoader
from loguru import logger
import markdown
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..LLM_Calls.LLM_API_Calls import chat_with_provider
from ..Chat.Chat_Functions import get_provider_model_name
from ..Notes.Notes_Library import NotesInteropService
from ..Metrics.metrics_logger import log_histogram, log_counter
from .content_processor import ContentSummarizer
#
########################################################################################################################
#
# Briefing Templates
#
########################################################################################################################

# Default Markdown template
DEFAULT_MARKDOWN_TEMPLATE = """# {title}
*Generated: {date}*

## Executive Summary
{executive_summary}

## New Items by Source
{sources_section}

## Key Insights
{insights_section}

## Trending Topics
{trends_section}

## Recommended Actions
{actions_section}

---
*This briefing contains {total_items} items from {total_sources} sources.*
"""

# Default HTML template
DEFAULT_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .item { margin: 15px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .source { font-weight: bold; color: #0066cc; }
        .date { color: #999; font-size: 0.9em; }
        .summary { margin-top: 5px; }
        .trends { display: flex; flex-wrap: wrap; gap: 10px; }
        .trend-tag { background: #e0e0e0; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="date">Generated: {date}</p>
    
    <h2>Executive Summary</h2>
    <p>{executive_summary}</p>
    
    <h2>New Items by Source</h2>
    {sources_html}
    
    <h2>Key Insights</h2>
    {insights_html}
    
    <h2>Trending Topics</h2>
    <div class="trends">
        {trends_html}
    </div>
    
    <h2>Recommended Actions</h2>
    {actions_html}
    
    <hr>
    <p><em>This briefing contains {total_items} items from {total_sources} sources.</em></p>
</body>
</html>"""


class BriefingGenerator:
    """Generate briefings from subscription content."""
    
    def __init__(self, subscriptions_db: SubscriptionsDB, 
                 llm_provider: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 notes_service: Optional[NotesInteropService] = None):
        """
        Initialize briefing generator.
        
        Args:
            subscriptions_db: Subscriptions database
            llm_provider: LLM provider for analysis
            llm_model: LLM model for analysis
            notes_service: Notes service for saving briefings
        """
        self.db = subscriptions_db
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.notes_service = notes_service
        self.summarizer = ContentSummarizer()
        
    async def generate_briefing(self, 
                               name: str,
                               source_filter: Optional[Dict[str, Any]] = None,
                               time_range: Optional[Tuple[datetime, datetime]] = None,
                               analysis_prompt: Optional[str] = None,
                               format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a briefing from subscription items.
        
        Args:
            name: Briefing name
            source_filter: Filter for selecting sources (tags, folders, subscription IDs)
            time_range: Time range for items (start, end)
            analysis_prompt: Custom analysis prompt
            format: Output format (markdown, html, json)
            
        Returns:
            Briefing data with content and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get items based on filter
            items = await self._get_briefing_items(source_filter, time_range)
            
            if not items:
                logger.warning("No items found for briefing")
                return {
                    'success': False,
                    'message': 'No items found matching criteria',
                    'content': None
                }
            
            # Group items by source
            items_by_source = self._group_by_source(items)
            
            # Generate sections
            if self.llm_provider and self.llm_model:
                # Use LLM for advanced analysis
                sections = await self._generate_sections_with_llm(
                    items, 
                    items_by_source,
                    analysis_prompt
                )
            else:
                # Fallback to basic analysis
                sections = self._generate_sections_basic(items, items_by_source)
            
            # Format briefing
            if format == "markdown":
                content = self._format_markdown(name, sections, items, items_by_source)
            elif format == "html":
                content = self._format_html(name, sections, items, items_by_source)
            elif format == "json":
                content = self._format_json(name, sections, items, items_by_source)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_histogram("briefing_generation_duration", duration, labels={
                "format": format,
                "used_llm": str(bool(self.llm_provider))
            })
            log_counter("briefing_generation_count", labels={
                "format": format,
                "item_count": str(len(items))
            })
            
            return {
                'success': True,
                'name': name,
                'content': content,
                'format': format,
                'item_count': len(items),
                'source_count': len(items_by_source),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error generating briefing: {e}")
            return {
                'success': False,
                'message': str(e),
                'content': None
            }
    
    async def _get_briefing_items(self, source_filter: Optional[Dict[str, Any]], 
                                 time_range: Optional[Tuple[datetime, datetime]]) -> List[Dict[str, Any]]:
        """Get items for briefing based on filters."""
        # Build query
        query = """
            SELECT i.*, s.name as subscription_name, s.type as subscription_type,
                   s.tags as subscription_tags, s.folder as subscription_folder
            FROM subscription_items i
            JOIN subscriptions s ON i.subscription_id = s.id
            WHERE i.status IN ('new', 'reviewed', 'ingested')
        """
        
        params = []
        
        # Apply time filter
        if time_range:
            start, end = time_range
            query += " AND i.created_at BETWEEN ? AND ?"
            params.extend([start.isoformat(), end.isoformat()])
        else:
            # Default to last 24 hours
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            query += " AND i.created_at >= ?"
            params.append(yesterday.isoformat())
        
        # Apply source filter
        if source_filter:
            if 'subscription_ids' in source_filter:
                placeholders = ','.join(['?' for _ in source_filter['subscription_ids']])
                query += f" AND s.id IN ({placeholders})"
                params.extend(source_filter['subscription_ids'])
            
            if 'tags' in source_filter:
                tag_conditions = []
                for tag in source_filter['tags']:
                    tag_conditions.append("s.tags LIKE ?")
                    params.append(f'%{tag}%')
                if tag_conditions:
                    query += f" AND ({' OR '.join(tag_conditions)})"
            
            if 'folders' in source_filter:
                placeholders = ','.join(['?' for _ in source_filter['folders']])
                query += f" AND s.folder IN ({placeholders})"
                params.extend(source_filter['folders'])
        
        query += " ORDER BY i.created_at DESC"
        
        # Execute query
        cursor = self.db.conn.cursor()
        cursor.execute(query, params)
        
        items = []
        for row in cursor.fetchall():
            item = dict(row)
            # Parse JSON fields
            for field in ['categories', 'enclosures', 'extracted_data']:
                if item.get(field):
                    try:
                        item[field] = json.loads(item[field])
                    except:
                        pass
            items.append(item)
        
        return items
    
    def _group_by_source(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group items by subscription source."""
        grouped = {}
        for item in items:
            source = item['subscription_name']
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(item)
        return grouped
    
    async def _generate_sections_with_llm(self, items: List[Dict[str, Any]], 
                                        items_by_source: Dict[str, List[Dict[str, Any]]],
                                        custom_prompt: Optional[str]) -> Dict[str, str]:
        """Generate briefing sections using LLM."""
        # Prepare content for analysis
        content_summary = self._prepare_content_summary(items)
        
        # Build analysis prompt
        if custom_prompt:
            prompt = custom_prompt.replace('{content}', content_summary)
        else:
            prompt = f"""Analyze the following content from various subscriptions and generate a comprehensive briefing:

{content_summary}

Please provide:
1. Executive Summary (2-3 paragraphs highlighting the most important developments)
2. Key Insights (bullet points of significant findings or patterns)
3. Trending Topics (identify common themes across sources)
4. Recommended Actions (actionable items based on the content)

Format each section clearly with appropriate headers."""
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert analyst creating executive briefings from aggregated content."},
            {"role": "user", "content": prompt}
        ]
        
        provider_model = get_provider_model_name(self.llm_provider, self.llm_model)
        
        response = await chat_with_provider(
            messages=messages,
            api_endpoint=self.llm_provider,
            provider=self.llm_provider,
            model_name=provider_model,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse response into sections
        if response and 'content' in response:
            analysis = response['content']
            sections = self._parse_llm_sections(analysis)
        else:
            # Fallback to basic if LLM fails
            sections = self._generate_sections_basic(items, items_by_source)
        
        # Add sources section
        sections['sources_section'] = self._format_sources_section(items_by_source)
        
        return sections
    
    def _generate_sections_basic(self, items: List[Dict[str, Any]], 
                                items_by_source: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Generate basic briefing sections without LLM."""
        sections = {}
        
        # Executive summary
        total_items = len(items)
        total_sources = len(items_by_source)
        newest_time = max(item['created_at'] for item in items)
        
        sections['executive_summary'] = (
            f"This briefing covers {total_items} new items from {total_sources} sources. "
            f"The most recent update was at {newest_time}. "
            "Below you'll find summaries of the latest content organized by source."
        )
        
        # Sources section
        sections['sources_section'] = self._format_sources_section(items_by_source)
        
        # Key insights (basic extraction)
        insights = []
        for source, source_items in items_by_source.items():
            if len(source_items) > 2:
                insights.append(f"• {source} published {len(source_items)} new items")
        
        sections['insights_section'] = '\n'.join(insights) if insights else "• No significant patterns detected"
        
        # Trending topics (keyword extraction)
        all_keywords = []
        for item in items:
            if item.get('categories'):
                all_keywords.extend(item['categories'])
        
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        trends = [f"• {kw} ({count} mentions)" for kw, count in top_keywords]
        sections['trends_section'] = '\n'.join(trends) if trends else "• No trending topics identified"
        
        # Actions
        sections['actions_section'] = "• Review highlighted items for detailed information\n• Consider following up on trending topics"
        
        return sections
    
    def _prepare_content_summary(self, items: List[Dict[str, Any]], max_items: int = 20) -> str:
        """Prepare a summary of items for LLM analysis."""
        summary_parts = []
        
        # Limit items to prevent token overflow
        items_to_analyze = items[:max_items]
        
        for item in items_to_analyze:
            # Extract key info
            title = item.get('title', 'Untitled')
            source = item.get('subscription_name', 'Unknown')
            content = item.get('content', '')[:500]  # Limit content length
            
            if content:
                # Use basic summarizer if no content
                if len(content) > 200:
                    content = self.summarizer.summarize(content, max_sentences=2)
            
            summary_parts.append(f"Source: {source}\nTitle: {title}\nSummary: {content}\n")
        
        if len(items) > max_items:
            summary_parts.append(f"\n... and {len(items) - max_items} more items")
        
        return '\n---\n'.join(summary_parts)
    
    def _parse_llm_sections(self, analysis: str) -> Dict[str, str]:
        """Parse LLM response into sections."""
        sections = {
            'executive_summary': '',
            'insights_section': '',
            'trends_section': '',
            'actions_section': ''
        }
        
        # Simple section extraction based on headers
        current_section = None
        current_content = []
        
        for line in analysis.split('\n'):
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'executive summary' in line_lower or 'summary' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'executive_summary'
                current_content = []
            elif 'key insights' in line_lower or 'insights' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'insights_section'
                current_content = []
            elif 'trending topics' in line_lower or 'trends' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'trends_section'
                current_content = []
            elif 'recommended actions' in line_lower or 'actions' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'actions_section'
                current_content = []
            elif current_section and line.strip():
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Fallback if parsing failed
        if not any(sections.values()):
            sections['executive_summary'] = analysis
        
        return sections
    
    def _format_sources_section(self, items_by_source: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format the sources section."""
        lines = []
        
        for source, items in sorted(items_by_source.items()):
            lines.append(f"\n### {source} ({len(items)} items)")
            
            for item in items[:5]:  # Limit to 5 items per source
                title = item.get('title', 'Untitled')
                url = item.get('url', '')
                published = item.get('published_date', item.get('created_at', ''))
                
                lines.append(f"\n**{title}**")
                if published:
                    lines.append(f"*{published}*")
                if url:
                    lines.append(f"[Read more]({url})")
                
                # Add brief content preview
                content = item.get('content', '')
                if content:
                    preview = content[:200].strip()
                    if len(content) > 200:
                        preview += '...'
                    lines.append(f"\n{preview}")
            
            if len(items) > 5:
                lines.append(f"\n*... and {len(items) - 5} more items*")
        
        return '\n'.join(lines)
    
    def _format_markdown(self, name: str, sections: Dict[str, str], 
                        items: List[Dict[str, Any]], 
                        items_by_source: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format briefing as Markdown."""
        template = Template(DEFAULT_MARKDOWN_TEMPLATE)
        
        return template.render(
            title=name,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=sections.get('executive_summary', ''),
            sources_section=sections.get('sources_section', ''),
            insights_section=sections.get('insights_section', ''),
            trends_section=sections.get('trends_section', ''),
            actions_section=sections.get('actions_section', ''),
            total_items=len(items),
            total_sources=len(items_by_source)
        )
    
    def _format_html(self, name: str, sections: Dict[str, str], 
                    items: List[Dict[str, Any]], 
                    items_by_source: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format briefing as HTML."""
        # Convert markdown sections to HTML
        md = markdown.Markdown(extensions=['extra'])
        
        # Format sources as HTML
        sources_html = self._format_sources_html(items_by_source)
        
        # Format trends as HTML tags
        trends_text = sections.get('trends_section', '')
        trends_items = [line.strip('• ') for line in trends_text.split('\n') if line.strip().startswith('•')]
        trends_html = ''.join([f'<span class="trend-tag">{trend}</span>' for trend in trends_items[:10]])
        
        template = Template(DEFAULT_HTML_TEMPLATE)
        
        return template.render(
            title=name,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=md.convert(sections.get('executive_summary', '')),
            sources_html=sources_html,
            insights_html=md.convert(sections.get('insights_section', '')),
            trends_html=trends_html,
            actions_html=md.convert(sections.get('actions_section', '')),
            total_items=len(items),
            total_sources=len(items_by_source)
        )
    
    def _format_sources_html(self, items_by_source: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format sources section as HTML."""
        html_parts = []
        
        for source, items in sorted(items_by_source.items()):
            html_parts.append(f'<h3>{source} ({len(items)} items)</h3>')
            
            for item in items[:5]:
                title = item.get('title', 'Untitled')
                url = item.get('url', '')
                published = item.get('published_date', item.get('created_at', ''))
                content = item.get('content', '')[:200]
                
                html_parts.append('<div class="item">')
                if url:
                    html_parts.append(f'<a href="{url}" target="_blank"><strong>{title}</strong></a>')
                else:
                    html_parts.append(f'<strong>{title}</strong>')
                
                if published:
                    html_parts.append(f'<span class="date"> - {published}</span>')
                
                if content:
                    html_parts.append(f'<div class="summary">{content}...</div>')
                
                html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_json(self, name: str, sections: Dict[str, str], 
                    items: List[Dict[str, Any]], 
                    items_by_source: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format briefing as JSON."""
        data = {
            'name': name,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_items': len(items),
                'total_sources': len(items_by_source),
                'sources': list(items_by_source.keys())
            },
            'sections': sections,
            'items_by_source': {
                source: [
                    {
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'published_date': item.get('published_date'),
                        'content_preview': item.get('content', '')[:500]
                    }
                    for item in items[:10]  # Limit items
                ]
                for source, items in items_by_source.items()
            }
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    async def save_briefing(self, briefing_data: Dict[str, Any], 
                          save_location: str = "notes") -> bool:
        """
        Save briefing to specified location.
        
        Args:
            briefing_data: Briefing data from generate_briefing
            save_location: Where to save (notes, file path)
            
        Returns:
            Success status
        """
        if not briefing_data.get('success') or not briefing_data.get('content'):
            return False
        
        try:
            if save_location == "notes" and self.notes_service:
                # Save to notes
                note_title = f"Briefing: {briefing_data['name']} - {datetime.now().strftime('%Y-%m-%d')}"
                note_content = briefing_data['content']
                
                # Add metadata
                if briefing_data['format'] == 'markdown':
                    metadata = f"\n\n---\n*Generated: {briefing_data['generated_at']}*\n"
                    metadata += f"*Items: {briefing_data['item_count']} from {briefing_data['source_count']} sources*"
                    note_content += metadata
                
                # Create note
                note_id = await self.notes_service.create_note(
                    title=note_title,
                    content=note_content,
                    tags=['briefing', 'subscription']
                )
                
                logger.info(f"Saved briefing to notes: {note_title}")
                return True
                
            else:
                # Save to file
                path = Path(save_location)
                
                # Determine file extension
                if briefing_data['format'] == 'html':
                    ext = '.html'
                elif briefing_data['format'] == 'json':
                    ext = '.json'
                else:
                    ext = '.md'
                
                # Generate filename if directory provided
                if path.is_dir():
                    filename = f"briefing_{briefing_data['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                    path = path / filename
                
                # Write file
                path.write_text(briefing_data['content'], encoding='utf-8')
                logger.info(f"Saved briefing to file: {path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving briefing: {e}")
            return False


class BriefingSchedule:
    """Manage scheduled briefing generation."""
    
    def __init__(self, generator: BriefingGenerator):
        """
        Initialize scheduler.
        
        Args:
            generator: Briefing generator instance
        """
        self.generator = generator
        self.scheduled_briefings = []
        self._running = False
        self._task = None
        
    def add_briefing(self, config: Dict[str, Any]):
        """
        Add a scheduled briefing.
        
        Args:
            config: Briefing configuration with schedule
        """
        self.scheduled_briefings.append(config)
        logger.info(f"Added scheduled briefing: {config['name']}")
        
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        logger.info("Started briefing scheduler")
        
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped briefing scheduler")
        
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now()
                
                for config in self.scheduled_briefings:
                    if self._should_run_briefing(config, now):
                        asyncio.create_task(self._generate_scheduled_briefing(config))
                
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in briefing scheduler: {e}")
                await asyncio.sleep(60)
    
    def _should_run_briefing(self, config: Dict[str, Any], now: datetime) -> bool:
        """Check if briefing should run now."""
        schedule = config.get('schedule', {})
        
        # Check time
        scheduled_time = schedule.get('time', '06:00')
        hour, minute = map(int, scheduled_time.split(':'))
        
        if now.hour != hour or now.minute != minute:
            return False
        
        # Check if already run today
        last_run = config.get('_last_run')
        if last_run:
            last_run_date = datetime.fromisoformat(last_run).date()
            if last_run_date == now.date():
                return False
        
        # Check day of week if specified
        days_of_week = schedule.get('days_of_week', list(range(7)))
        if now.weekday() not in days_of_week:
            return False
        
        return True
    
    async def _generate_scheduled_briefing(self, config: Dict[str, Any]):
        """Generate a scheduled briefing."""
        try:
            logger.info(f"Generating scheduled briefing: {config['name']}")
            
            # Generate briefing
            briefing = await self.generator.generate_briefing(
                name=config['name'],
                source_filter=config.get('source_filter'),
                time_range=config.get('time_range'),
                analysis_prompt=config.get('analysis_prompt'),
                format=config.get('format', 'markdown')
            )
            
            if briefing['success']:
                # Save briefing
                await self.generator.save_briefing(
                    briefing,
                    config.get('save_location', 'notes')
                )
                
                # Update last run time
                config['_last_run'] = datetime.now().isoformat()
                
                # Send notifications if configured
                # TODO: Implement email/webhook notifications
                
            else:
                logger.error(f"Failed to generate briefing: {briefing.get('message')}")
                
        except Exception as e:
            logger.error(f"Error generating scheduled briefing: {e}")


# End of briefing_generator.py