"""
This module contains a list of functions to implement RAG.
"""

import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import PINECONE_INDEX
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# TODO[PAO] If needed, maybe add a param for accepting how many docs to retrieve
def create_context_from_vector_store(vector_store: PineconeVectorStore, query: str):
    """
    This method gets chunks of texts from the document (vector database) and returns
    the augmented context to be added to llm.
    """
    
    retriever = vector_store.as_retriever(search_kwargs = { "k": 2 })
    retrieved_documents = retriever.invoke(query)
    new_context = "\n\n".join(
        retrieved_document.page_content for retrieved_document in retrieved_documents
    )

    return new_context

def get_vector_store():
    """
    This method is checks if the current index exists to use that data and if not, create
    an index + add initial vectors from /data/*.docx
    """

    pc = Pinecone(api_key=pinecone_api_key)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # If index is not yet created, create the index + add initial vectors from /data
    # Else get existing vector index
    if PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name = PINECONE_INDEX,
            # Notes[PAO] Dimension and metric is suggested in Pinecone > indexes > setup by model
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        loader = DirectoryLoader('data', glob="*.docx")
        docs = loader.load()

        # each docs will have 1000 characters: chunksize
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunk_docs = text_splitter.split_documents(docs)

        return PineconeVectorStore.from_documents(chunk_docs, embeddings, index_name=PINECONE_INDEX)
    else:
        return PineconeVectorStore.from_existing_index(embedding=embeddings, index_name="chatbot")
