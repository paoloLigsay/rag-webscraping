"""
This module is for integrating RAG into LLM for better response.
"""

import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from constants import SYSTEM_PROMPT_TEMPLATE, HUMAN_PROMPT_TEMPLATE
from utils import create_context_from_vector_store, get_vector_store

# Get api_keys for openai and pinecone
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
vector_store = get_vector_store()

# Initial Question of the user after "SYSTEM"
# flow: system -> human/user -> assistant -> human/user -> assistant...
query = input("What is your question about Batangas or Laguna? ")
CONTEXT = create_context_from_vector_store(vector_store, query)
PROMPT = SYSTEM_PROMPT_TEMPLATE.format(context = CONTEXT)

# Instantiate llm and get initial response with the created "message" 
llm = ChatOpenAI(api_key=openai_api_key)
messages = [("system", PROMPT), ("human", query)]
ai_initial_response = llm.invoke(messages)
print(ai_initial_response.content + "\n")

# Initialize memory, we will use this to always pass the whole conversation when invoking llm
# to get a more concise and conversational answer.
memory = messages + [("assistant", ai_initial_response.content)]
SHOULD_CONTINUE = True

while SHOULD_CONTINUE:
    query = input("Ask any questions, type none if you don't have any questions: ").lower()
    if query == "none":
        SHOULD_CONTINUE = False
    else:
        NEW_CONTEXT = create_context_from_vector_store(vector_store, query)
        NEW_PROMPT = HUMAN_PROMPT_TEMPLATE.format(context = NEW_CONTEXT, query = query)
        memory = memory + [("human", NEW_PROMPT)]

        ai_response = llm.invoke(memory)
        memory = messages + [("assistant", ai_response.content)]
        print(ai_response.content + "\n")
