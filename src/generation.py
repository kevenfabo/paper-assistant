import logging
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# init the logger
logger = logging.getLogger(__name__)

# embeddings definition
EMBEDDINGS_INFO = {
    "mxbai-embed-large": {
        "model_id": "mxbai-embed-large",
        "dimension": 1024
    },
    "bge-large": {
        "model_id": "bge-large",
        "dimension": 1024
    }
}

LLM_INFO = {
    "llama3.1": {
        "model_name": "llama3.1",
    },
    "llama3": {
        "model_name": "llama3",
    },
    "gemma2": {
        "model_name": "gemma2",
    },
    "zephyr": {
        "model_name": "zephyr",
    },
    "phi3.5": {
        "model_name": "phi3.5",
    },
    "mistral7b": {
        "model_name": "mistral",
    },
}

PAPER_INDEXES = {
    "all": {
        "location": "data/vectors/partition/all"
    }
}

def load_indexed_documents(partition_name: str) -> VectorStoreIndex:
    """ Load vector indexes from disk 
    
    Argrs:
        partition_name (str): partition name used as a key to access document vectors
        
    Return:
        VectorStoreIndex
    """
    
    # get the index vectors path 
    partition_path = PAPER_INDEXES[partition_name]["location"]
    vector_store = FaissVectorStore.from_persist_dir(partition_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=partition_path
    )
    
    return load_index_from_storage(
        storage_context=storage_context
    )


def generate_answer(input_message: dict, llm: Ollama, partition_index: VectorStoreIndex):
    """ Generate an answer to a given question
    
    Args:
        input_message (dict): user input message
        llm (Ollama): language model
        partition_index (VectorStoreIndex): indexed documents
        
    Return:
        str: response
    """
    
    # get the user input
    user_input = input_message[-1].content
    
    # get the retrieved contexts
    answer = partition_index.as_query_engine(
        llm=llm,
        top_k=3 # TODO: set the top_k as hyperparam of the script
    ).query(user_input).response

    return answer

if __name__=="__main__":

    INPUT_QUESTIONS = [
        "Explain the attention mechanism in transformers",
        "What is the main goal of Colpali?",
        "What is your name?",
        "What is CONTINUAL MEMORIZATION OF FACTOIDS?"
    ]
    
    # define the llm and embedding models
    llm = Ollama(model="llama3.1")
    Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
    
    # load the indexed documents
    partition_index = load_indexed_documents(partition_name="all")
    
    # generate answers
    responses = []
    for question in INPUT_QUESTIONS:
        # add system message
        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful NLP research assistant. Ask me anything about NLP research papers.",
            )
        ]
        
        # add the user input
        messages.append(
            ChatMessage(
                role="user",
                content=question
            )
        )
        
        # generate the response
        response = generate_answer(
            input_message=messages,
            llm=llm,
            partition_index=partition_index
        )
        
        # add the response
        responses.append(response)
        
    # print the responses
    for i, response in enumerate(responses):
        print(f"Question: {INPUT_QUESTIONS[i]}")
        print(f"Answer: {response}")
        print("\n")
        
        
            