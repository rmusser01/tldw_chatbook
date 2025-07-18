#!/usr/bin/env python3
"""
Script to add default analysis prompts to the Prompts database.
This can be run to populate the database with pre-configured prompts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.config import get_user_data_dir
from loguru import logger

# Default analysis prompts
DEFAULT_PROMPTS = [
    # Audio Analysis Prompts
    {
        "name": "Audio Analysis Assistant",
        "author": "System",
        "details": "General audio content analysis with focus on transcribed content",
        "system_prompt": "You are an expert audio content analyst. Analyze the transcribed audio and provide insights on the content, speakers, and key discussions.",
        "user_prompt": "",
        "keywords": ["audio", "analysis", "transcription", "general"]
    },
    {
        "name": "Transcription Summarizer",
        "author": "System",
        "details": "Creates concise summaries of transcribed content",
        "system_prompt": "You are a professional transcription summarizer. Create clear, concise summaries of transcribed content while preserving key information.",
        "user_prompt": "",
        "keywords": ["audio", "summary", "transcription"]
    },
    {
        "name": "Meeting Notes Generator",
        "author": "System",
        "details": "Extracts and organizes meeting content from transcriptions",
        "system_prompt": "You are a meeting notes specialist. Extract and organize key points, decisions, and action items from meeting transcriptions.",
        "user_prompt": "",
        "keywords": ["audio", "meeting", "notes", "action-items"]
    },
    {
        "name": "Podcast Summary",
        "author": "System",
        "details": "Summarizes podcast episodes with key insights",
        "system_prompt": "You are a podcast content analyst. Summarize podcast episodes, highlighting key topics, guest insights, and memorable quotes.",
        "user_prompt": "",
        "keywords": ["audio", "podcast", "summary"]
    },
    
    # Video Analysis Prompts
    {
        "name": "Video Content Analyzer",
        "author": "System",
        "details": "Comprehensive video transcript analysis",
        "system_prompt": "You are a video content analyst. Analyze video transcripts and provide comprehensive summaries of visual and audio content.",
        "user_prompt": """Turn the following unorganized text into a well-structured, readable format while retaining EVERY detail, context, and nuance of the original content.
            Refine the text to improve clarity, grammar, and coherence WITHOUT cutting, summarizing, or omitting any information.
            The goal is to make the content easier to read and process by:

            - Organizing the content into logical sections with appropriate subheadings.
            - Using bullet points or numbered lists where applicable to present facts, stats, or comparisons.
            - Highlighting key terms, names, or headings with bold text for emphasis.
            - Preserving the original tone, humor, and narrative style while ensuring readability.
            - Adding clear separators or headings for topic shifts to improve navigation.

            Ensure the text remains informative, capturing the original intent, tone,
            and details while presenting the information in a format optimized for analysis by both humans and AI.
            REMEMBER that Details are important, DO NOT overlook Any details, even small ones.
            All output must be generated entirely in [Language]. Do not use any other language at any point in the response. Do not include this unorganized text into your response.
            Text:""",
        "keywords": ["video", "analysis", "transcript"]
    },
    {
        "name": "Video Content Analyzer",
        "author": "System",
        "details": "Comprehensive video transcript analysis",
        "system_prompt": "You are a video content analyst. Analyze video transcripts and provide comprehensive summaries of visual and audio content.",
        "user_prompt": "",
        "keywords": ["video", "analysis", "transcript"]
    },
    {
        "name": "Educational Analysis",
        "author": "System",
        "details": "Creates study-friendly summaries of educational videos",
        "system_prompt": "You are an educational content specialist. Create study-friendly summaries of educational videos, highlighting key concepts and learning objectives.",
        "user_prompt": """Transform the following transcript into a comprehensive educational text, resembling a textbook chapter. Structure the content with clear headings, subheadings, and bullet points to enhance readability and organization for educational purposes.

Crucially, identify any technical terms, jargon, or concepts that are mentioned but not explicitly explained within the transcript. For each identified term, provide a concise definition (no more than two sentences) formatted as a blockquote.  Integrate these definitions strategically within the text, ideally near the first mention of the term, to enhance understanding without disrupting the flow.

Ensure the text is highly informative, accurate, and retains all the original details and nuances of the transcript. The goal is to create a valuable educational resource that is easy to study and understand.

All output must be generated entirely in [Language]. Do not use any other language at any point in the response. Do not use any other language at any point in the response. Do not include this unorganized text into your response.

Text:""",
        "keywords": ["video", "education", "study", "learning"]
    },
    {
        "name": "Tutorial Breakdown",
        "author": "System",
        "details": "Breaks down tutorial videos into step-by-step instructions",
        "system_prompt": "You are a tutorial content organizer. Break down tutorial videos into clear, step-by-step instructions with key points.",
        "user_prompt": "",
        "keywords": ["video", "tutorial", "instructions", "how-to"]
    },
    {
        "name": "Rewrite Transcript as a Story",
        "author": "System",
        "details": "Breaks down video transcripts into an 'engaging narrative' or story format.",
        "system_prompt": "",
        "user_prompt": """Rewrite the following transcript into an engaging narrative or story format. Transform the factual or conversational content into a more captivating and readable piece, similar to a short story or narrative article.

While rewriting, maintain a close adherence to the original subjects and information presented in the video. Do not deviate significantly from the core topics or introduce unrelated elements.  The goal is to enhance engagement and readability through storytelling techniques without altering the fundamental content or message of the video.  Use narrative elements like descriptive language, scene-setting (if appropriate), and a compelling flow to make the information more accessible and enjoyable.

All output must be generated entirely in [Language]. Do not use any other language at any point in the response. Do not include this unorganized text into your response.

Text:""",
        "keywords": ["video", "story", "engagement"]
    },
    {
        "name": "Rewrite Transcript into a more readable format",
        "author": "System",
        "details": "Breaks down video transcripts into an 'engaging narrative' or story format.",
        "system_prompt": "You are a professional newspaper editor at the New York Times.",
        "user_prompt": """Turn the following unorganized text into a well-structured, readable format while retaining EVERY detail, context, and nuance of the original content.
                Refine the text to improve clarity, grammar, and coherence WITHOUT cutting, summarizing, or omitting any information.
                The goal is to make the content easier to read and process by:

                - Organizing the content into logical sections with appropriate subheadings.
                - Using bullet points or numbered lists where applicable to present facts, stats, or comparisons.
                - Highlighting key terms, names, or headings with bold text for emphasis.
                - Preserving the original tone, humor, and narrative style while ensuring readability.
                - Adding clear separators or headings for topic shifts to improve navigation.

                Ensure the text remains informative, capturing the original intent, tone,
                and details while presenting the information in a format optimized for analysis by both humans and AI.
                REMEMBER that Details are important, DO NOT overlook Any details, even small ones.
                All output must be generated entirely in [Language]. Do not use any other language at any point in the response. Do not include this unorganized text into your response.
                Text:
                """,
        "keywords": ["video", "story", "engagement"]
    },
    {
            "name": "Generate Questions and Answers from Transcript",
            "author": "System",
            "details": "Breaks down video transcripts into a QA Document",
            "system_prompt": "You are a professional newspaper editor at the New York Times.",
            "user_prompt": """Generate a set of questions and answers based on the following transcript for self-assessment or review.  For each question, create a corresponding answer.

Format each question as a level 3 heading using Markdown syntax (### Question Text). Immediately following each question, provide the answer.  This format is designed for foldable sections, allowing users to easily hide and reveal answers for self-testing.

Ensure the questions are relevant to the key information and concepts in the transcript and that the answers are accurate and comprehensive based on the video content.

All output must be generated entirely in [Language]. Do not use any other language at any point in the response. Do not include this unorganized text into your response.

Text:""",
            "keywords": ["video", "story", "engagement"]
        },
    # Document/PDF Analysis Prompts
    {
        "name": "Document Analyzer",
        "author": "System",
        "details": "Thorough document analysis with structured summaries",
        "system_prompt": "You are a document analysis expert. Analyze documents thoroughly and provide structured summaries of content, findings, and implications.",
        "user_prompt": "",
        "keywords": ["document", "pdf", "analysis"]
    },
    {
        "name": "Research Paper Summarizer",
        "author": "System",
        "details": "Summarizes academic papers with focus on methodology and findings",
        "system_prompt": "You are a research paper analyst. Summarize academic papers, highlighting methodology, findings, and conclusions.",
        "user_prompt": "",
        "keywords": ["document", "pdf", "research", "academic"]
    },
    {
        "name": "Technical Document Parser",
        "author": "System",
        "details": "Parses and summarizes technical documentation",
        "system_prompt": "You are a technical documentation specialist. Parse and summarize technical documents, focusing on specifications, procedures, and key technical details.",
        "user_prompt": "",
        "keywords": ["document", "pdf", "technical", "specifications"]
    },
    
    # User Prompts (These work with any system prompt)
    {
        "name": "Summarize Key Points",
        "author": "System",
        "details": "General prompt for extracting key points",
        "system_prompt": "",
        "user_prompt": "Please summarize the key points discussed in this content. Include main topics, important decisions, and any action items mentioned.",
        "keywords": ["summary", "key-points", "general"]
    },
    {
        "name": "Extract Action Items",
        "author": "System",
        "details": "Extracts tasks and commitments from content",
        "system_prompt": "",
        "user_prompt": "Extract all action items, tasks, and commitments mentioned in this content. Organize them by priority if possible.",
        "keywords": ["action-items", "tasks", "commitments"]
    },
    {
        "name": "Create Detailed Summary",
        "author": "System",
        "details": "Provides comprehensive summary with all details",
        "system_prompt": "",
        "user_prompt": "Provide a detailed summary of this content, including main topics, subtopics, key points, and any conclusions or next steps.",
        "keywords": ["summary", "detailed", "comprehensive"]
    },
    {
        "name": "Extract Main Points",
        "author": "System",
        "details": "Extracts and organizes main arguments and points",
        "system_prompt": "",
        "user_prompt": "Extract the main points and arguments from this content. Organize them in a clear, hierarchical structure.",
        "keywords": ["main-points", "arguments", "structure"]
    },
    {
        "name": "Create Executive Summary",
        "author": "System",
        "details": "Creates executive-level summary for decision makers",
        "system_prompt": "",
        "user_prompt": "Create an executive summary of this content, highlighting the most important information for decision-makers.",
        "keywords": ["executive-summary", "decision-making", "high-level"]
    },
    {
        "name": "Create Study Notes",
        "author": "System",
        "details": "Creates comprehensive study notes from content",
        "system_prompt": "",
        "user_prompt": "Create comprehensive study notes from this content, including main topics, definitions, examples, and key takeaways.",
        "keywords": ["study-notes", "education", "learning"]
    }
]

def add_default_prompts():
    """Add default analysis prompts to the database."""
    # Get database path
    data_dir = get_user_data_dir()
    prompts_db_path = data_dir / "prompts.db"
    
    logger.info(f"Initializing Prompts database at: {prompts_db_path}")
    
    # Initialize database
    db = PromptsDatabase(str(prompts_db_path), "default_prompts_script")
    
    added_count = 0
    skipped_count = 0
    
    for prompt_data in DEFAULT_PROMPTS:
        try:
            # Check if prompt already exists
            existing = db.get_prompt_by_name(prompt_data["name"])
            if existing:
                logger.info(f"Prompt '{prompt_data['name']}' already exists, skipping")
                skipped_count += 1
                continue
            
            # Add the prompt
            prompt_id, prompt_uuid, message = db.add_prompt(
                name=prompt_data["name"],
                author=prompt_data["author"],
                details=prompt_data["details"],
                system_prompt=prompt_data["system_prompt"],
                user_prompt=prompt_data["user_prompt"],
                keywords=prompt_data["keywords"]
            )
            
            if prompt_id:
                logger.info(f"✅ Added prompt: {prompt_data['name']} - {message}")
                added_count += 1
            else:
                logger.error(f"❌ Failed to add prompt: {prompt_data['name']} - {message}")
                
        except Exception as e:
            logger.error(f"Error adding prompt '{prompt_data.get('name', 'Unknown')}': {e}")
    
    logger.info(f"\nSummary: Added {added_count} prompts, skipped {skipped_count} existing prompts")
    
    # List all prompts
    all_prompts = db.get_all_prompts()
    logger.info(f"\nTotal prompts in database: {len(all_prompts)}")
    
    db.close()

if __name__ == "__main__":
    add_default_prompts()