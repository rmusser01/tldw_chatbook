"""Golden parity: registry defaults render byte-identical text to the
original WebSearch_APIs.py literals. The f-strings below are verbatim copies
of the pre-migration source."""

from tldw_chatbook.Internal_Prompts import render_internal_prompt


def test_sub_question_generation_parity():
    original_query = "How does climate change affect biodiversity?"
    expected = f"""
            You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. Your goal is to generate queries that are diverse, specific, and highly relevant to the original query, ensuring comprehensive coverage of the topic.

            Important instructions:
            1. Generate between 2 and 6 queries unless a fixed count is specified. Generate more queries for complex or multifaceted topics and fewer for simple or straightforward ones.
            2. Ensure the queries are diverse, covering different aspects or perspectives of the original query, while remaining highly relevant to its core intent.
            3. Prefer specific queries over general ones, as they are more likely to yield targeted and useful results.
            4. If the query involves comparing two topics, generate separate queries for each topic.
            5. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
            6. If the original query is broad or ambiguous, generate queries that explore specific subtopics or clarify the intent.
            7. If the query is too specific or unclear, generate queries that explore related or broader topics to ensure useful results.
            8. Return the queries as a JSON array in the format ["query_1", "query_2", ...].

            Examples:
            1. For the query "What are the benefits of exercise?", generate queries like:
               ["health benefits of physical activity", "mental health benefits of exercise", "long-term effects of regular exercise", "how exercise improves cardiovascular health", "role of exercise in weight management"]

            2. For the query "Compare Python and JavaScript", generate queries like:
               ["key features of Python programming language", "advantages of JavaScript for web development", "use cases for Python vs JavaScript", "performance comparison of Python and JavaScript", "ease of learning Python vs JavaScript"]

            3. For the query "How does climate change affect biodiversity?", generate queries like:
               ["impact of climate change on species extinction", "effects of global warming on ecosystems", "role of climate change in habitat loss", "how rising temperatures affect marine biodiversity", "climate change and its impact on migratory patterns"]

            4. For the query "Best practices for remote work", generate queries like:
               ["tips for staying productive while working from home", "how to maintain work-life balance in remote work", "tools for effective remote team collaboration", "managing communication in remote teams", "ergonomic setup for home offices"]

            5. For the query "What is quantum computing?", generate queries like:
               ["basic principles of quantum computing", "applications of quantum computing in real-world problems", "difference between classical and quantum computing", "key challenges in developing quantum computers", "future prospects of quantum computing"]

            Original query: {original_query}
            """
    assert render_internal_prompt(
        "websearch.sub_question_generation", original_query=original_query
    ) == expected


def test_result_relevance_eval_parity():
    original_question = "What is quantum computing?"
    sub_questions = ["basics of quantum computing", "qubits explained"]
    content = 'Snippet with {braces} and "quotes" to prove safety.'
    expected = f"""
                Given the following search results for the user's question: "{original_question}" and the generated sub-questions: {sub_questions}, evaluate the relevance of the search result to the user's question.
                Explain your reasoning for selection.

                Search Results:
                {content}

                Instructions:
                1. You MUST only answer TRUE or False while providing your reasoning for your answer.
                2. A result is relevant if the result most likely contains comprehensive and relevant information to answer the user's question.
                3. Provide a brief reason for selection.

                You MUST respond using EXACTLY this format and nothing else:

                Selected Answer: [True or False]
                Reasoning: [Your reasoning for the selections]
                """
    assert render_internal_prompt(
        "websearch.result_relevance_eval",
        original_question=original_question,
        sub_questions=sub_questions,
        content=content,
    ) == expected


def test_result_summarization_parity():
    question = "What is quantum computing?"
    content = "Long scraped text with a stray { brace."
    original_template = """
    Summarize the following text in a concise way that captures the key information relevant to this question: "{question}"
    
    Text to summarize:
    {content}
    
    Instructions:
    1. Focus on information relevant to the question
    2. Keep the summary under 2000 characters
    3. Maintain factual accuracy
    4. Include key details and statistics if present
    """
    expected = original_template.format(question=question, content=content)
    assert render_internal_prompt(
        "websearch.result_summarization", question=question, content=content
    ) == expected


def test_answer_synthesis_parity():
    concatenated_texts = "1. Source one summary\n2. Source two summary"
    current_date = "2026-07-21"
    question = "Compare Python and JavaScript"
    expected = rf"""INITIAL_QUERY: Here are some sources {concatenated_texts}. Read these carefully, as you will be asked a Query about them.
        # General Instructions
        
        Write an accurate, detailed, and comprehensive response to the user's query located at INITIAL_QUERY. Additional context is provided as "USER_INPUT" after specific questions. Your answer should be informed by the provided "Search results". Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone. Your answer must be written in the same language as the query, even if language preference is different.
        
        You MUST cite the most relevant search results that answer the query. Do not mention any irrelevant results. You MUST ADHERE to the following instructions for citing search results:
        - to cite a search result, enclose its index located above the summary with brackets at the end of the corresponding sentence, for example "Ice is less dense than water[1][2]." or "Paris is the capital of France[1][4][5]."
        - NO SPACE between the last word and the citation, and ALWAYS use brackets. Only use this format to cite search results. NEVER include a References section at the end of your answer.
        - If you don't know the answer or the premise is incorrect, explain why.
        If the search results are empty or unhelpful, answer the query as well as you can with existing knowledge.
        
        You MUST NEVER use moralization or hedging language. AVOID using the following phrases:
        - "It is important to ..."
        - "It is inappropriate ..."
        - "It is subjective ..."
        
        You MUST ADHERE to the following formatting instructions:
        - Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
        - Use headings level 2 and 3 to separate sections of your response, like "## Header", but NEVER start an answer with a heading or title of any kind.
        - Use single new lines for lists and double new lines for paragraphs.
        - Use markdown to render images given in the search results.
        - NEVER write URLs or links.
        
        # Query type specifications
        
        You must use different instructions to write your answer based on the type of the user's query. However, be sure to also follow the General Instructions, especially if the query doesn't match any of the defined types below. Here are the supported types.
        
        ## Academic Research
        
        You must provide long and detailed answers for academic research queries. Your answer should be formatted as a scientific write-up, with paragraphs and sections, using markdown and headings.
        
        ## Recent News
        
        You need to concisely summarize recent news events based on the provided search results, grouping them by topics. You MUST ALWAYS use lists and highlight the news title at the beginning of each list item. You MUST select news from diverse perspectives while also prioritizing trustworthy sources. If several search results mention the same news event, you must combine them and cite all of the search results. Prioritize more recent events, ensuring to compare timestamps. You MUST NEVER start your answer with a heading of any kind.
        
        ## Weather
        
        Your answer should be very short and only provide the weather forecast. If the search results do not contain relevant weather information, you must state that you don't have the answer.
        
        ## People
        
        You need to write a short biography for the person mentioned in the query. If search results refer to different people, you MUST describe each person individually and AVOID mixing their information together. NEVER start your answer with the person's name as a header.
        
        ## Coding
        
        You MUST use markdown code blocks to write code, specifying the language for syntax highlighting, for example ```bash or ```python If the user's query asks for code, you should write the code first and then explain it.
        
        ## Cooking Recipes
        
        You need to provide step-by-step cooking recipes, clearly specifying the ingredient, the amount, and precise instructions during each step.
        
        ## Translation
        
        If a user asks you to translate something, you must not cite any search results and should just provide the translation.
        
        ## Creative Writing
        
        If the query requires creative writing, you DO NOT need to use or cite search results, and you may ignore General Instructions pertaining only to search. You MUST follow the user's instructions precisely to help the user write exactly what they need.
        
        ## Science and Math
        
        If the user query is about some simple calculation, only answer with the final result. Follow these rules for writing formulas:
        - Always use \( and\) for inline formulas and\[ and\] for blocks, for example\(x^4 = x - 3 \)
        - To cite a formula add citations to the end, for example\[ \sin(x) \] [1][2] or \(x^2-2\) [4].
        - Never use $ or $$ to render LaTeX, even if it is present in the user query.
        - Never use unicode to render math expressions, ALWAYS use LaTeX.
        - Never use the \label instruction for LaTeX.
        
        ## URL Lookup
        
        When the user's query includes a URL, you must rely solely on information from the corresponding search result. DO NOT cite other search results, ALWAYS cite the first result, e.g. you need to end with [1]. If the user's query consists only of a URL without any additional instructions, you should summarize the content of that URL.
        
        ## Shopping
        
        If the user query is about shopping for a product, you MUST follow these rules:
        - Organize the products into distinct sectors. For example, you could group shoes by style (boots, sneakers, etc.)
        - Cite at most 9 search results using the format provided in General Instructions to avoid overwhelming the user with too many options.
        
        The current date is: {current_date}

        The user's query is: {question}
        """
    assert render_internal_prompt(
        "websearch.answer_synthesis",
        concatenated_texts=concatenated_texts,
        current_date=current_date,
        question=question,
    ) == expected
