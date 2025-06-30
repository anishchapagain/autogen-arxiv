from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaModel
from autogen_ext.models.openai import OpenAIModel

from dotenv import load_dotenv

import os

load_dotenv() # Load environment variables from .env file

system_message = (
    "You are a helpful assistant that summarizes academic papers from arXiv."
    "You will be provided with the title, abstract, and other relevant information of a paper.\n"
    "Your task is to generate a concise summary that captures the main contributions and findings of the paper."
    "Please ensure that your summary is clear, informative, and suitable for readers who may not be familiar with the specific field of study.\n"
    "Avoid jargon and focus on the key points that would help someone understand the significance of the research."
)

system_message =(
    "You are an expert researcher."
    "You will write a literature review report in Markdown format, based on the JSON list of papers provided."
    "Your task is to generate a concise summary that captures the main contributions and findings of each paper." \
    "Instructions:\n"
    "1. Start with 2-3 sentence introducing the topic\n" \
    "2. Include bullet points for title, authors, problems tackled, key contributions\n" \
    "3. Title should contain paper link in Markdown format\n" \
    "4. Use clear and concise language, avoiding jargon\n" \
    )

agent_LLM_OpenAI = OpenAIModel( # On OpenAI API
    api_key = os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0.1,
    max_tokens=512,
    top_p=0.9,
    top_k=40,
)
agent_LLM_Ollama = OllamaModel( # On Private Server
    url="http://localhost:11434/api/generate",
    model="llama3",
    temperature=0.1,
    max_tokens=512,
    top_p=0.9,
    top_k=40,
)

# Define the summarizer agent using the Ollama model
summarizer_agents = AssistantAgent(
    name="SummarizerAgent",
    description="An agent that summarizes text.",
    model_client=agent_LLM_Ollama,
    system_message=system_message,
)

# Define the researcher agent using python library ```arxiv```
researcher_agents = AssistantAgent(
    name="ResearcherAgent",
    description="An agent that writes literature review report in Markdown format.",
    model_client=agent_LLM_Ollama,
    system_message=system_message,
)