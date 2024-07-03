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
    the augmented context to be added to llm. The goal of why we retrieve relative 
    information from the vector database is to improve reliable asnwers and to solve the
    following drawbacks in this documentation: https://www.pinecone.io/learn/retrieval-augmented-generation/

    See the diagram/flow of how it works under the hood below:
                                        +-------------------+           
                                        |       Search      |                     
                                        +---------+---------+         
                                                  |                  
                                                  |                     
                                                  v                    
    +-------------------+               +-------------------+               +-------------------+
    |  Embedding Model  | <------------ |      Question     | ------------> |   "How do I..."   |
    +---------+---------+               +---------+---------+               +-------------------+
              |                                   |             
              |                                   |             
              v                                   |
    +-------------------+                         |
    |  Vector Database  |                         |           
    +---------+---------+                         |        
              |                                   |                
              |         (augment context)         |
              v                 |                 v
  +-----------------------+     v     +-----------------------+ 
  | Relevant Data (Top-k) + --------> |   Question+Context    | 
  +-----------------------+           +-----------------------+
                                                  |
                                                  v    
                                      +-----------------------+
                                      |   LLM (Gen AI Model)  | 
                                      +-----------------------+
                                                  |
                                                  v 
                                      +-----------------------+             +-------------------+
                                      |    Reliable Answer    | ----------> |    "You can..."   |
                                      +-----------------------+             +-------------------+
    """
    
    # This is where we get the most relevant documents, based on the documentation
    # here are few things to consider, good to know:
    # lambda_mult - used for balancing relevance and diversity (Default: 0.5)
    #    a. 0 - maximizes diversity, selected documents will be as different as possible from others
    #    b. 1 - minimizes diversity, focused on selecting relevant documents based on similiraty
    # fetch_k - deteremines # of documents to be considered beyond the top-k (Default: 20)
    # k - # of results to be returned (Default: 4)
    # filter - straightforward, adding filter inside search_kwargs. e.g: 'filter': {'author': 'pao'}
    # score_threshold - filters out less relevant documents. higher treshold means more selective results
    
    # retriever = vector_store.as_retriever(search_kwargs = {
    #     "k": 2,
    # })

    # [PAO] From what I understand, similarity is more of focusing on getting the most relevant document
    # to the query. But for mmr, we have more control between diversity and relevance. Setting lambda_mult
    # to 1 would likely behave the same as search_type=similarity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs = {
            "k": 2,
            "lambda_mult": 0.5
        }
    )

    retrieved_documents = retriever.invoke(query)
    new_context = "\n\n".join(
        retrieved_document.page_content for retrieved_document in retrieved_documents
    )

    return new_context

def get_vector_store():
    """
    This method checks if the current index exists to use that data and if not, create
    an index + add initial vectors from /data/*.docx.                           
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

        # first argument is the path or directory, second one is the type of files we're reading through
        loader = DirectoryLoader('data', glob="*.docx")
        docs = loader.load()

        # Each docs will have 1000 characters: chunksize
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunk_docs = text_splitter.split_documents(docs)

        return PineconeVectorStore.from_documents(chunk_docs, embeddings, index_name=PINECONE_INDEX)
    else:
        return PineconeVectorStore.from_existing_index(embedding=embeddings, index_name="chatbot")
