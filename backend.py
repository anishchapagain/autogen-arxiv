from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient 
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_agentchat.teams import Swarm
# from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
# from autogen_agentchat.messages import HandoffMessage
# from autogen_agentchat.ui import Console

from dotenv import load_dotenv

import os
import asyncio
import arxiv
from typing import List, Dict, AsyncGenerator

load_dotenv() # Load environment variables from .env file

# Librariies
# pip install -U "autogen-ext[ollama]"
# pip install -U "autogen-ext[openai]"

def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search for papers on arXiv using the provided query.
    https://pypi.org/project/arxiv/

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
        
    Returns:
        List[Dict]: A list of dictionaries containing paper information.
    """
    # Construct the default API client.
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        # sort_by=arxiv.SortCriterion.Relevance,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Data Collection
    arxiv_results:List[Dict] = []

    # Perform the search and collect results
    results = client.results(search)
    for result in results:
        arxiv_results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%dT%H:%M:%S"),
            "link": result.entry_id,
            "url": result.pdf_url,
            "categories": result.categories,
        })
    
    return arxiv_results

# Define the system message for the summarizer agent
system_message_arxiv = (
    "Given a user topic, think of the best arxiv query and call the provided tool."
    "Always getch five times the papers related to the topic."
    "When the tool returns the results, summarize each paper in a concise JSON manner."
)

system_message =(
    "You are an expert researcher."
    "You will write a literature review report in Markdown format, based on the JSON list of papers provided."
    "Your task is to generate a concise summary that captures the main contributions and findings of each paper." \
    "Instructions:\n"
    "1. Start with 2-3 sentence introducing the topic\n" \
    "2. Include bullet points for title, authors, problems tackled, key contributions, keywords\n" \
    "3. Title should contain paper link in Markdown format\n" \
    "4. Use clear and concise language, avoiding jargon\n" \
    )

# agent_LLM_OpenAI = OpenAIChatCompletionClient( # On OpenAI API
#     api_key = os.getenv("OPENAI_API_KEY"),
#     model="gpt-4o",
#     temperature=0.1,
#     max_tokens=512,
#     top_p=0.9,
#     top_k=40,
# )

agent_LLM_Ollama = OllamaChatCompletionClient( # On Private Server
    model="llama3.2:latest",
    base_url="http://localhost:11434/api/generate",  # Adjust the base URL as needed
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)

# Define the summarizer agent using the Ollama model
summarizer_agent = AssistantAgent(
    name="SummarizerAgent",
    description="An agent that summarizes text.",
    model_client=agent_LLM_Ollama,
    system_message=system_message,
)

# Define the researcher agent using python library ```arxiv```
arxiv_researcher_agent = AssistantAgent(
    name="Arxiv_ResearcherAgent",
    description="An agent that search, retrieves papers from arxiv.com",
    model_client=agent_LLM_Ollama,
    system_message=system_message_arxiv,
    tools=[arxiv_search]
)

# Define the agents list
agents = [arxiv_researcher_agent, summarizer_agent]
team = RoundRobinGroupChat(
    participants = agents, 
    max_turns=2,
)

task = "Provide a literature review on the topic of machine learning or AI Agent."

async def team_output() -> AsyncGenerator[str, None]:
    """
    Run the team and yield the output as it becomes available.
    
    Returns:
        AsyncGenerator[str, None]: Yields the output of the team.
    """
    async for output in team.run_stream(task = task):
        yield output.


async def main():
    """
    Main function to run the team and print the output.
    """
    async for output in team_output():
        print(output)
        # with open("team_output.md", "a") as f:
        #     print(output, file=f)
        # print(output)   


def test_arxiv_search():
    print("Testing arxiv search function...")
    results = arxiv_search("machine learning", max_results=5)
    if results:
        with open("arxiv_results.json", "w") as f:
            import json
            json.dump(results, f, indent=4) 
    else:
        print("No results found from Arxiv.")
    print("Arxiv search function is working correctly.")


if __name__ == "__main__":
    # Test the arxiv search function
    # test_arxiv_search()

    # Run the team and print the output

    print("Running the team...")
    asyncio.run(main())