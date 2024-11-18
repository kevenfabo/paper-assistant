GET DATA:
    - define the list of papers to download 
    - define a list of top k papers to retrieve using keywords
    - download and save them in the data/papers folder 
    
    Next steps:
        - define list of papers to download in a paper table

INGESTION PIPELINE:
    - define an setup a vectordb to be used to store paper embeddings: Faiss, Chroma, pgvector
    https://www.langchain.ca/blog/top-5-open-source-vector-databases-2024/
    - define a pipeline to ingest papers into the vectordb
    - define embedding models and vectorization pipeline
    - define ingestion pipeline optimizations

GENERATION PIPELINE:
    - llm: https://ollama.com/
    - define generation steps 
    - define an agentic approach to generation

UI:
    - build a UI to interact with the system: https://docs.chainlit.io/get-started/overview

PROMPT ENGINEERING:
    - optimize prompts with dspy

EVALUATION:
    - define retrieval and generation evaluation metrics: https://github.com/explodinggradients/ragas/tree/main

REFACTORING:
    - create a RAG module that will be used to index and generate answers
    - create config files for the iterations 