# custom_scraper.py
# Description: Custom scraping pipeline with user-defined rules
#
# This scraper allows users to define complex scraping rules including:
# - Multiple extraction patterns
# - JavaScript rendering support
# - Custom data transformations
# - Conditional logic
#
# Imports
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable
from urllib.parse import urljoin
#
# Third-Party Imports
import httpx
from bs4 import BeautifulSoup
from loguru import logger
#
# Local Imports
from ..web_scraping_pipelines import (
    BaseScrapingPipeline, ScrapedItem, ScrapingConfig, ContentExtractor
)
from ...Metrics.metrics_logger import log_counter
#
########################################################################################################################
#
# Custom Scraping Pipeline
#
########################################################################################################################

class CustomScrapingPipeline(BaseScrapingPipeline):
    """
    Highly customizable scraping pipeline with user-defined rules.
    
    This pipeline supports:
    - Complex CSS/XPath selectors
    - Regular expression extraction
    - Data transformations
    - Conditional logic
    - JavaScript rendering (when enabled)
    """
    
    def __init__(self, config: ScrapingConfig):
        """Initialize custom scraper with configuration."""
        super().__init__(config)
        
        # Parse custom rules from config
        self.rules = config.options.get('rules', {})
        self.transformations = config.options.get('transformations', {})
        self.conditions = config.options.get('conditions', [])
        self.wait_conditions = config.options.get('wait_conditions', [])
        
        # JavaScript settings
        self.wait_time = config.options.get('wait_time', 0)
        self.scroll_to_bottom = config.options.get('scroll_to_bottom', False)
        
        # Extraction settings
        self.use_xpath = config.options.get('use_xpath', False)
        self.extract_json_ld = config.options.get('extract_json_ld', True)
        self.follow_item_links = config.options.get('follow_item_links', False)
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate custom scraper configuration."""
        # Validate rules structure
        if not isinstance(self.rules, dict):
            return False, "Rules must be a dictionary"
        
        # Check required rule fields
        required_fields = ['items']  # At minimum, need to know how to find items
        for field in required_fields:
            if field not in self.rules:
                return False, f"Missing required rule: {field}"
        
        # Validate selector syntax
        for rule_name, rule_config in self.rules.items():
            if isinstance(rule_config, dict) and 'selector' in rule_config:
                selector = rule_config['selector']
                if self.use_xpath:
                    # Basic XPath validation
                    if not selector.startswith('/') and not selector.startswith('./'):
                        return False, f"Invalid XPath in {rule_name}: {selector}"
                else:
                    # Basic CSS selector validation
                    try:
                        # Test parse with BeautifulSoup
                        BeautifulSoup('<div></div>', 'html.parser').select(selector)
                    except Exception as e:
                        return False, f"Invalid CSS selector in {rule_name}: {selector}"
        
        # Validate transformations
        for trans_name, trans_config in self.transformations.items():
            if 'type' not in trans_config:
                return False, f"Transformation {trans_name} missing 'type'"
            
            if trans_config['type'] not in ['regex', 'replace', 'format', 'date_parse']:
                return False, f"Unknown transformation type: {trans_config['type']}"
        
        return True, None
    
    async def fetch_content(self, url: str) -> str:
        """Fetch content with optional JavaScript rendering."""
        if self.config.javascript_enabled:
            return await self._fetch_with_javascript(url)
        else:
            return await self._fetch_static(url)
    
    async def _fetch_static(self, url: str) -> str:
        """Fetch static HTML content."""
        logger.debug(f"Fetching static content from: {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("custom_scraper_fetch", labels={
                "status": "success",
                "javascript": "false",
                "domain": self.get_domain()
            })
            
            return response.text
    
    async def _fetch_with_javascript(self, url: str) -> str:
        """Fetch content with JavaScript rendering."""
        logger.error("JavaScript rendering requested but not implemented")
        raise NotImplementedError(
            "JavaScript rendering is not yet available. "
            "Please disable javascript_enabled in your scraping configuration "
            "or wait for Playwright integration to be implemented."
        )
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse content using custom rules."""
        items = []
        
        try:
            soup = BeautifulSoup(raw_content, 'html.parser')
            
            # Extract JSON-LD data if available
            json_ld_data = {}
            if self.extract_json_ld:
                json_ld_data = self._extract_json_ld(soup)
            
            # Find items using configured selector
            item_rule = self.rules.get('items', {})
            if isinstance(item_rule, str):
                # Simple selector
                item_elements = soup.select(item_rule)
            elif isinstance(item_rule, dict):
                # Complex rule with options
                selector = item_rule.get('selector', 'article')
                item_elements = soup.select(selector)
                
                # Apply conditions if specified
                if 'conditions' in item_rule:
                    item_elements = self._filter_elements(item_elements, item_rule['conditions'])
            else:
                # Fallback to common patterns
                item_elements = soup.select('article, .post, .entry, .item')
            
            # Parse each item
            for element in item_elements:
                item = self._parse_item_with_rules(element, url, json_ld_data)
                if item and self._passes_conditions(item):
                    items.append(item)
            
            # Handle follow item links
            if self.follow_item_links and not items:
                # This page might be a listing, extract links to follow
                # Note: Actual following would need to be handled by caller
                pass
            
            logger.info(f"Parsed {len(items)} items using custom rules")
            log_counter("custom_scraper_items", labels={
                "domain": self.get_domain(),
                "item_count": str(len(items))
            })
            
        except Exception as e:
            logger.error(f"Error parsing with custom rules: {str(e)}")
            log_counter("custom_scraper_errors", labels={
                "error_type": "parse_error",
                "domain": self.get_domain()
            })
            raise
        
        return items
    
    def _parse_item_with_rules(self, element, base_url: str, 
                              json_ld_data: Dict[str, Any]) -> Optional[ScrapedItem]:
        """Parse item using defined extraction rules."""
        try:
            extracted_data = {}
            
            # Extract each field using rules
            for field_name, rule in self.rules.items():
                if field_name == 'items':  # Skip the items selector itself
                    continue
                
                value = self._extract_field(element, rule, base_url)
                
                # Apply transformations if defined
                if value and field_name in self.transformations:
                    value = self._apply_transformation(value, self.transformations[field_name])
                
                extracted_data[field_name] = value
            
            # Map to ScrapedItem fields
            return ScrapedItem(
                url=extracted_data.get('url', '') or self._extract_url(element, base_url),
                title=extracted_data.get('title', ''),
                content=extracted_data.get('content', ''),
                author=extracted_data.get('author'),
                published_date=self._parse_date_field(extracted_data.get('date')),
                categories=self._extract_categories(extracted_data),
                metadata=self._build_metadata(extracted_data, json_ld_data)
            )
            
        except Exception as e:
            logger.error(f"Error parsing item with rules: {str(e)}")
            return None
    
    def _extract_field(self, element, rule: Any, base_url: str) -> Optional[str]:
        """Extract field value using rule definition."""
        if isinstance(rule, str):
            # Simple selector
            return self._extract_with_selector(element, rule)
        
        elif isinstance(rule, dict):
            # Complex rule
            selector = rule.get('selector')
            if not selector:
                return None
            
            # Extract value
            value = self._extract_with_selector(element, selector)
            
            # Apply attribute extraction if specified
            if value and 'attribute' in rule:
                elem = element.select_one(selector)
                if elem:
                    value = elem.get(rule['attribute'], value)
            
            # Apply regex extraction if specified
            if value and 'regex' in rule:
                match = re.search(rule['regex'], value)
                if match:
                    group = rule.get('regex_group', 0)
                    value = match.group(group)
            
            # Handle URL fields
            if rule.get('is_url') and value:
                value = urljoin(base_url, value)
            
            return value
        
        return None
    
    def _extract_with_selector(self, element, selector: str) -> Optional[str]:
        """Extract text using CSS or XPath selector."""
        try:
            if self.use_xpath:
                # TODO: Implement XPath support with lxml
                logger.warning("XPath not implemented, using CSS selector fallback")
            
            # CSS selector
            found = element.select_one(selector)
            if found:
                return found.get_text(strip=True)
            
        except Exception as e:
            logger.error(f"Selector extraction error: {str(e)}")
        
        return None
    
    def _apply_transformation(self, value: str, transformation: Dict[str, Any]) -> str:
        """Apply transformation to extracted value."""
        trans_type = transformation.get('type')
        
        if trans_type == 'regex':
            pattern = transformation.get('pattern', '')
            replacement = transformation.get('replacement', '')
            return re.sub(pattern, replacement, value)
        
        elif trans_type == 'replace':
            old = transformation.get('old', '')
            new = transformation.get('new', '')
            return value.replace(old, new)
        
        elif trans_type == 'format':
            template = transformation.get('template', '{}')
            return template.format(value)
        
        elif trans_type == 'date_parse':
            # Custom date parsing handled elsewhere
            return value
        
        return value
    
    def _filter_elements(self, elements: List, conditions: List[Dict]) -> List:
        """Filter elements based on conditions."""
        filtered = []
        
        for element in elements:
            passes = True
            for condition in conditions:
                if not self._check_element_condition(element, condition):
                    passes = False
                    break
            
            if passes:
                filtered.append(element)
        
        return filtered
    
    def _check_element_condition(self, element, condition: Dict) -> bool:
        """Check if element meets condition."""
        cond_type = condition.get('type')
        
        if cond_type == 'has_class':
            class_name = condition.get('class')
            return class_name in element.get('class', [])
        
        elif cond_type == 'has_text':
            text = condition.get('text')
            return text in element.get_text()
        
        elif cond_type == 'has_child':
            selector = condition.get('selector')
            return element.select_one(selector) is not None
        
        return True
    
    def _passes_conditions(self, item: ScrapedItem) -> bool:
        """Check if item passes global conditions."""
        for condition in self.conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            item_value = getattr(item, field, None)
            if not item_value:
                item_value = item.metadata.get(field)
            
            if not self._check_condition(item_value, operator, value):
                return False
        
        return True
    
    def _check_condition(self, item_value: Any, operator: str, 
                        condition_value: Any) -> bool:
        """Check single condition."""
        if operator == 'equals':
            return item_value == condition_value
        elif operator == 'contains':
            return condition_value in str(item_value)
        elif operator == 'matches':
            return bool(re.search(condition_value, str(item_value)))
        elif operator == 'greater_than':
            return float(item_value) > float(condition_value)
        elif operator == 'less_than':
            return float(item_value) < float(condition_value)
        elif operator == 'not_empty':
            return bool(item_value)
        
        return True
    
    def _extract_url(self, element, base_url: str) -> str:
        """Extract URL from element."""
        # Try URL rule first
        if 'url' in self.rules:
            url = self._extract_field(element, self.rules['url'], base_url)
            if url:
                return url
        
        # Fallback to finding first link
        link = element.find('a', href=True)
        if link:
            return urljoin(base_url, link['href'])
        
        return base_url
    
    def _extract_categories(self, data: Dict[str, Any]) -> List[str]:
        """Extract categories from parsed data."""
        categories = []
        
        # Check direct categories field
        if 'categories' in data:
            cat_value = data['categories']
            if isinstance(cat_value, str):
                # Split by common delimiters
                categories = [c.strip() for c in re.split(r'[,;|]', cat_value)]
            elif isinstance(cat_value, list):
                categories = cat_value
        
        # Check tags field
        if 'tags' in data:
            tags = data['tags']
            if isinstance(tags, str):
                categories.extend([t.strip() for t in re.split(r'[,;|]', tags)])
            elif isinstance(tags, list):
                categories.extend(tags)
        
        return list(set(categories))  # Remove duplicates
    
    def _parse_date_field(self, date_value: Any) -> Optional[datetime]:
        """Parse date from extracted value."""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        # Try common formats
        date_str = str(date_value)
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        
        return None
    
    def _build_metadata(self, extracted_data: Dict[str, Any], 
                       json_ld_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata from all extracted data."""
        metadata = {
            'source': 'custom',
            'domain': self.get_domain(),
            'extraction_rules': len(self.rules)
        }
        
        # Add JSON-LD data if available
        if json_ld_data:
            metadata['structured_data'] = json_ld_data
        
        # Add any extra fields not mapped to ScrapedItem
        standard_fields = {'url', 'title', 'content', 'author', 'date', 
                          'categories', 'tags'}
        for key, value in extracted_data.items():
            if key not in standard_fields and value:
                metadata[key] = value
        
        return metadata
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON-LD structured data."""
        json_ld_data = {}
        
        try:
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        json_ld_data.update(data)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and '@type' in item:
                                json_ld_data[item['@type']] = item
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error extracting JSON-LD: {str(e)}")
        
        return json_ld_data


# End of custom_scraper.py