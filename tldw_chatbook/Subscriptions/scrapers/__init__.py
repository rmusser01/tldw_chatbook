# scrapers/__init__.py
# Description: Scraper module initialization and registration
#
# This module automatically registers all available scrapers with the factory.
#
# Imports
from ..web_scraping_pipelines import ScrapingPipelineFactory
from .reddit_scraper import RedditScrapingPipeline
from .generic_scraper import GenericWebScrapingPipeline
from .custom_scraper import CustomScrapingPipeline
from .github_scraper import GitHubScrapingPipeline
from .hackernews_scraper import HackerNewsScrapingPipeline
from .youtube_scraper import YouTubeScrapingPipeline
#
# Register all scrapers
ScrapingPipelineFactory.register_pipeline('reddit', RedditScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('generic', GenericWebScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('custom', CustomScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('github', GitHubScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('hackernews', HackerNewsScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('hn', HackerNewsScrapingPipeline)  # Alias
ScrapingPipelineFactory.register_pipeline('youtube', YouTubeScrapingPipeline)
ScrapingPipelineFactory.register_pipeline('yt', YouTubeScrapingPipeline)  # Alias
#
# Exports
__all__ = [
    'RedditScrapingPipeline',
    'GenericWebScrapingPipeline', 
    'CustomScrapingPipeline',
    'GitHubScrapingPipeline',
    'HackerNewsScrapingPipeline',
    'YouTubeScrapingPipeline'
]