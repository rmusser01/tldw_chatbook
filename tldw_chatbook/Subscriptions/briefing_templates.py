# briefing_templates.py
# Description: Template system for generating customized briefings
#
# This module provides a flexible template system for creating various
# briefing formats from aggregated subscription content.
#
# Imports
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
#
# Third-Party Imports
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from loguru import logger
import markdown
#
# Local Imports
from .aggregation_engine import AggregatedContent, AggregatedSection
from ..config import get_user_config_dir
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass
class BriefingSection:
    """Represents a section in a briefing template."""
    id: str
    title: str
    token_allocation: int
    priority: str  # 'high', 'medium', 'low'
    required: bool
    template: str  # Jinja2 template string
    metadata: Dict[str, Any]


@dataclass
class BriefingTemplate:
    """Complete briefing template configuration."""
    id: str
    name: str
    description: str
    format: str  # 'markdown', 'html', 'text', 'json'
    sections: List[BriefingSection]
    total_tokens: int
    metadata: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BriefingTemplate':
        """Create template from dictionary."""
        sections = [
            BriefingSection(**section_data)
            for section_data in data.get('sections', [])
        ]
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            format=data.get('format', 'markdown'),
            sections=sections,
            total_tokens=data.get('total_tokens', 4000),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'format': self.format,
            'sections': [
                {
                    'id': s.id,
                    'title': s.title,
                    'token_allocation': s.token_allocation,
                    'priority': s.priority,
                    'required': s.required,
                    'template': s.template,
                    'metadata': s.metadata
                }
                for s in self.sections
            ],
            'total_tokens': self.total_tokens,
            'metadata': self.metadata
        }


########################################################################################################################
#
# Built-in Templates
#
########################################################################################################################

EXECUTIVE_SUMMARY_TEMPLATE = BriefingTemplate(
    id='executive_summary',
    name='Executive Summary',
    description='High-level overview for executives and decision makers',
    format='markdown',
    sections=[
        BriefingSection(
            id='key_highlights',
            title='Key Highlights',
            token_allocation=500,
            priority='high',
            required=True,
            template="""## Key Highlights

{% for item in items[:3] %}
- **{{ item.title }}** ({{ item.source }}): {{ item.summary }}
{% endfor %}""",
            metadata={}
        ),
        BriefingSection(
            id='market_trends',
            title='Market Trends',
            token_allocation=300,
            priority='high',
            required=False,
            template="""## Market Trends

{% if market_items %}
{% for item in market_items %}
- {{ item.summary }}
{% endfor %}
{% else %}
No significant market trends identified.
{% endif %}""",
            metadata={'filter_tags': ['market', 'finance', 'economy']}
        ),
        BriefingSection(
            id='action_items',
            title='Action Items',
            token_allocation=200,
            priority='medium',
            required=True,
            template="""## Recommended Actions

{% if action_items %}
{% for item in action_items %}
1. {{ item }}
{% endfor %}
{% else %}
No immediate actions required.
{% endif %}""",
            metadata={}
        )
    ],
    total_tokens=1000,
    metadata={'audience': 'executives', 'tone': 'professional'}
)


TECHNICAL_DIGEST_TEMPLATE = BriefingTemplate(
    id='technical_digest',
    name='Technical Digest',
    description='Detailed technical updates for developers and engineers',
    format='markdown',
    sections=[
        BriefingSection(
            id='security_updates',
            title='Security Updates',
            token_allocation=600,
            priority='high',
            required=True,
            template="""## 🔒 Security Updates

{% for item in security_items %}
### {{ item.title }}
*Source: {{ item.source }} | {{ item.published_date }}*

{{ item.content }}

{% if item.links %}
**References:**
{% for link in item.links %}
- [{{ link.title }}]({{ link.url }})
{% endfor %}
{% endif %}

---
{% endfor %}""",
            metadata={'filter_tags': ['security', 'vulnerability', 'cve']}
        ),
        BriefingSection(
            id='code_releases',
            title='New Releases & Updates',
            token_allocation=800,
            priority='high',
            required=False,
            template="""## 🚀 New Releases & Updates

{% for item in release_items %}
### {{ item.title }}
{{ item.summary }}

{% if item.version %}Version: `{{ item.version }}`{% endif %}

{% endfor %}""",
            metadata={'filter_tags': ['release', 'update', 'version']}
        ),
        BriefingSection(
            id='technical_articles',
            title='Technical Articles',
            token_allocation=1000,
            priority='medium',
            required=True,
            template="""## 📚 Technical Articles

{% for item in technical_items %}
### [{{ item.title }}]({{ item.url }})
*{{ item.author }} | {{ item.source }}*

{{ item.summary }}

{% if item.tags %}
Tags: {% for tag in item.tags %}`{{ tag }}` {% endfor %}
{% endif %}

{% endfor %}""",
            metadata={}
        )
    ],
    total_tokens=2400,
    metadata={'audience': 'technical', 'tone': 'detailed'}
)


NEWS_BRIEFING_TEMPLATE = BriefingTemplate(
    id='news_briefing',
    name='Daily News Briefing',
    description='Curated news updates organized by category',
    format='markdown',
    sections=[
        BriefingSection(
            id='top_stories',
            title='Top Stories',
            token_allocation=800,
            priority='high',
            required=True,
            template="""## 📰 Top Stories

{% for item in top_stories %}
**{{ loop.index }}. {{ item.title }}**
*{{ item.source }} - {{ item.time_ago }}*

{{ item.summary }}

[Read more →]({{ item.url }})

{% endfor %}""",
            metadata={'max_items': 5}
        ),
        BriefingSection(
            id='by_category',
            title='News by Category',
            token_allocation=1200,
            priority='medium',
            required=True,
            template="""## 📂 By Category

{% for category, items in categorized_items.items() %}
### {{ category|title }}

{% for item in items[:3] %}
- **{{ item.title }}** ({{ item.source }})
  {{ item.summary|truncate(150) }}
{% endfor %}

{% endfor %}""",
            metadata={}
        ),
        BriefingSection(
            id='trending',
            title='Trending Topics',
            token_allocation=400,
            priority='low',
            required=False,
            template="""## 📈 Trending Topics

{% if trending_topics %}
{% for topic in trending_topics %}
- **{{ topic.name }}**: {{ topic.mention_count }} mentions across {{ topic.source_count }} sources
{% endfor %}
{% else %}
No clear trending topics identified today.
{% endif %}""",
            metadata={'analysis_type': 'frequency'}
        )
    ],
    total_tokens=2400,
    metadata={'audience': 'general', 'tone': 'informative'}
)


# Registry of built-in templates
BUILT_IN_TEMPLATES = {
    'executive_summary': EXECUTIVE_SUMMARY_TEMPLATE,
    'technical_digest': TECHNICAL_DIGEST_TEMPLATE,
    'news_briefing': NEWS_BRIEFING_TEMPLATE,
    'default': NEWS_BRIEFING_TEMPLATE  # Default fallback
}


########################################################################################################################
#
# Template Renderer
#
########################################################################################################################

class BriefingRenderer:
    """Renders briefings using templates."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize renderer.
        
        Args:
            template_dir: Directory containing custom templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.env = self._create_jinja_env()
        self._register_filters()
    
    def _get_default_template_dir(self) -> Path:
        """Get default template directory."""
        config_dir = get_user_config_dir()
        template_dir = config_dir / "briefing_templates"
        template_dir.mkdir(exist_ok=True)
        return template_dir
    
    def _create_jinja_env(self) -> Environment:
        """Create Jinja2 environment."""
        return Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def _register_filters(self):
        """Register custom Jinja2 filters."""
        # Time formatting
        self.env.filters['time_ago'] = self._time_ago_filter
        self.env.filters['format_date'] = self._format_date_filter
        
        # Content formatting
        self.env.filters['markdown'] = self._markdown_filter
        self.env.filters['truncate_words'] = self._truncate_words_filter
        self.env.filters['extract_domain'] = self._extract_domain_filter
    
    def render_briefing(self, 
                       template: BriefingTemplate,
                       content: AggregatedContent,
                       context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a briefing using template and content.
        
        Args:
            template: Briefing template
            content: Aggregated content
            context: Additional context variables
            
        Returns:
            Rendered briefing text
        """
        # Prepare context
        render_context = self._prepare_context(template, content, context or {})
        
        # Render sections
        rendered_sections = []
        
        for section in template.sections:
            try:
                # Get section-specific items
                section_items = self._get_section_items(section, content, render_context)
                
                if not section_items and not section.required:
                    continue
                
                # Create section context
                section_context = {
                    **render_context,
                    'items': section_items,
                    'section': section
                }
                
                # Render section template
                section_template = Template(section.template)
                rendered = section_template.render(**section_context)
                
                if rendered.strip():
                    rendered_sections.append(rendered)
                    
            except Exception as e:
                logger.error(f"Error rendering section {section.id}: {str(e)}")
                if section.required:
                    rendered_sections.append(f"## {section.title}\n\n*Error rendering section*")
        
        # Join sections
        briefing = "\n\n".join(rendered_sections)
        
        # Apply format conversion if needed
        if template.format == 'html':
            briefing = self._convert_to_html(briefing)
        elif template.format == 'text':
            briefing = self._convert_to_text(briefing)
        elif template.format == 'json':
            briefing = self._convert_to_json(briefing, rendered_sections)
        
        return briefing
    
    def _prepare_context(self, 
                        template: BriefingTemplate,
                        content: AggregatedContent,
                        extra_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare rendering context."""
        # Basic context
        context = {
            'now': datetime.now(timezone.utc),
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'time': datetime.now(timezone.utc).strftime('%H:%M UTC'),
            'total_items': content.total_items,
            'total_sources': content.total_sources,
            'template': template,
            **extra_context
        }
        
        # Add all items
        all_items = []
        for section in content.sections:
            all_items.extend(section.items)
        context['all_items'] = all_items
        
        # Sort by priority/date
        context['top_stories'] = sorted(
            all_items,
            key=lambda x: (x.get('priority', 3), x.get('published_date', '')),
            reverse=True
        )[:10]
        
        # Categorize items
        categorized = {}
        for item in all_items:
            category = item.get('category', 'Uncategorized')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        context['categorized_items'] = categorized
        
        # Extract trending topics
        context['trending_topics'] = self._extract_trending_topics(all_items)
        
        # Filter by tags
        context['security_items'] = [
            item for item in all_items
            if any(tag in str(item).lower() for tag in ['security', 'vulnerability', 'cve'])
        ]
        
        context['release_items'] = [
            item for item in all_items
            if any(tag in str(item).lower() for tag in ['release', 'version', 'update'])
        ]
        
        context['technical_items'] = [
            item for item in all_items
            if item.get('category', '').lower() in ['technology', 'programming', 'development']
        ]
        
        return context
    
    def _get_section_items(self, 
                          section: BriefingSection,
                          content: AggregatedContent,
                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get items for a specific section."""
        all_items = context.get('all_items', [])
        
        # Apply section filters
        if 'filter_tags' in section.metadata:
            filter_tags = section.metadata['filter_tags']
            filtered_items = []
            for item in all_items:
                item_text = json.dumps(item).lower()
                if any(tag in item_text for tag in filter_tags):
                    filtered_items.append(item)
            return filtered_items
        
        # Return specific context items if they match section ID
        if section.id in context:
            return context[section.id]
        
        # Default to all items
        return all_items
    
    def _extract_trending_topics(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract trending topics from items."""
        # Simple word frequency analysis
        word_counts = {}
        
        for item in items:
            # Extract text
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            
            # Simple tokenization (in production, use proper NLP)
            words = text.split()
            
            # Count significant words
            for word in words:
                if len(word) > 5 and word.isalpha():  # Simple filter
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Format as trending topics
        trending = []
        for word, count in top_words:
            if count > 2:  # Mentioned multiple times
                trending.append({
                    'name': word.title(),
                    'mention_count': count,
                    'source_count': min(count, len(set(item.get('source', '') for item in items)))
                })
        
        return trending[:5]
    
    def _time_ago_filter(self, timestamp: Union[str, datetime]) -> str:
        """Convert timestamp to 'time ago' format."""
        if isinstance(timestamp, str):
            # Try to parse ISO format
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return timestamp
        
        if not isinstance(timestamp, datetime):
            return str(timestamp)
        
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        
        if delta.days > 365:
            return f"{delta.days // 365} year{'s' if delta.days // 365 > 1 else ''} ago"
        elif delta.days > 30:
            return f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
        elif delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60} minute{'s' if delta.seconds // 60 > 1 else ''} ago"
        else:
            return "just now"
    
    def _format_date_filter(self, timestamp: Union[str, datetime], format: str = '%Y-%m-%d') -> str:
        """Format date."""
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return timestamp
        
        if isinstance(timestamp, datetime):
            return timestamp.strftime(format)
        
        return str(timestamp)
    
    def _markdown_filter(self, text: str) -> str:
        """Convert markdown to HTML."""
        return markdown.markdown(text, extensions=['extra', 'sane_lists'])
    
    def _truncate_words_filter(self, text: str, word_count: int = 50) -> str:
        """Truncate text to word count."""
        words = text.split()
        if len(words) <= word_count:
            return text
        
        return ' '.join(words[:word_count]) + '...'
    
    def _extract_domain_filter(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return url
    
    def _convert_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML."""
        html = markdown.markdown(markdown_text, extensions=['extra', 'sane_lists', 'codehilite'])
        
        # Wrap in basic HTML structure
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Briefing</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        a {{ color: #0066cc; }}
        code {{ background: #f4f4f4; padding: 2px 4px; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
    
    def _convert_to_text(self, markdown_text: str) -> str:
        """Convert markdown to plain text."""
        # Simple conversion - in production use proper markdown to text converter
        text = markdown_text
        
        # Remove markdown formatting
        replacements = [
            (r'\*\*(.+?)\*\*', r'\1'),  # Bold
            (r'\*(.+?)\*', r'\1'),      # Italic
            (r'\[(.+?)\]\(.+?\)', r'\1'),  # Links
            (r'^#+\s+', ''),             # Headers
            (r'`(.+?)`', r'\1'),         # Inline code
        ]
        
        import re
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        return text
    
    def _convert_to_json(self, briefing_text: str, sections: List[str]) -> str:
        """Convert to JSON format."""
        data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'format': 'json',
            'sections': []
        }
        
        for i, section_text in enumerate(sections):
            # Extract title (first line)
            lines = section_text.strip().split('\n')
            title = lines[0].strip('#').strip() if lines else f"Section {i+1}"
            
            data['sections'].append({
                'title': title,
                'content': '\n'.join(lines[1:]) if len(lines) > 1 else '',
                'index': i
            })
        
        return json.dumps(data, indent=2, ensure_ascii=False)


########################################################################################################################
#
# Template Manager
#
########################################################################################################################

class BriefingTemplateManager:
    """Manages briefing templates."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize template manager.
        
        Args:
            template_dir: Directory for custom templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.custom_templates = self._load_custom_templates()
    
    def _get_default_template_dir(self) -> Path:
        """Get default template directory."""
        config_dir = get_user_config_dir()
        template_dir = config_dir / "briefing_templates"
        template_dir.mkdir(exist_ok=True)
        return template_dir
    
    def _load_custom_templates(self) -> Dict[str, BriefingTemplate]:
        """Load custom templates from directory."""
        custom = {}
        
        # Look for JSON template files
        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    template = BriefingTemplate.from_dict(data)
                    custom[template.id] = template
                    logger.info(f"Loaded custom template: {template.id}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {str(e)}")
        
        return custom
    
    def get_template(self, template_id: str) -> Optional[BriefingTemplate]:
        """
        Get template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template or None if not found
        """
        # Check custom templates first
        if template_id in self.custom_templates:
            return self.custom_templates[template_id]
        
        # Check built-in templates
        return BUILT_IN_TEMPLATES.get(template_id)
    
    def list_templates(self) -> List[BriefingTemplate]:
        """List all available templates."""
        # Combine built-in and custom
        all_templates = {**BUILT_IN_TEMPLATES, **self.custom_templates}
        return list(all_templates.values())
    
    def save_template(self, template: BriefingTemplate):
        """
        Save custom template.
        
        Args:
            template: Template to save
        """
        # Save to file
        template_file = self.template_dir / f"{template.id}.json"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Update cache
        self.custom_templates[template.id] = template
        logger.info(f"Saved template: {template.id}")
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete custom template.
        
        Args:
            template_id: Template to delete
            
        Returns:
            True if deleted, False if not found or built-in
        """
        # Can't delete built-in templates
        if template_id in BUILT_IN_TEMPLATES:
            logger.warning(f"Cannot delete built-in template: {template_id}")
            return False
        
        # Remove file
        template_file = self.template_dir / f"{template_id}.json"
        if template_file.exists():
            template_file.unlink()
            
            # Update cache
            if template_id in self.custom_templates:
                del self.custom_templates[template_id]
            
            logger.info(f"Deleted template: {template_id}")
            return True
        
        return False


# End of briefing_templates.py