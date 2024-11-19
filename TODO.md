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

FINE-TUNING:
    - LLM:
        - full fine-tuning vs parameter efficient fine-tuning:
            - lora
            - qlora
        - training hardware requirements
        - data preparation:
            - case of instruct dataset: https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing
                - blog: https://www.datacamp.com/code-along/fine-tuning-your-own-llama-2-model
        - tools:
            - huggingface transformers
            - unsloth: https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html
            - pytorch fsdp
        - multi-gpu training
        - quantization:
            - https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html
            - https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html
        - instruction fine-tuning:
            - https://github.com/mlabonne/llm-course/blob/main/Fine_tune_Llama_2_in_Google_Colab.ipynb
            - https://www.philschmid.de/fine-tune-llms-in-2024-with-trl

        - case of classification:
            - https://www.datacamp.com/tutorial/fine-tuning-llama-3-1
        - alignment:
            - RLHF
            - DPO: https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html
            - PPO
            - ORPO: https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html
        - serving:
            - ollama
            - exllamav2: https://github.com/mlabonne/llm-course/blob/main/Fine_tune_Llama_2_in_Google_Colab.ipynb
            - TGI
            - vLLM

    - EMBEDDING:
        - sentence transformers:
            - https://www.philschmid.de/fine-tune-embedding-model-for-rag