
import faiss
import logging
import argparse
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

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
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "dimension": 384
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dimension": 1024
    }
}

Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha"
)

def load_documents(folder_path: str) -> SimpleDirectoryReader:
    
    logging.info(f"Start loading documents from `{folder_path}` folder")
    # get the documents from papers storage 
    documents = SimpleDirectoryReader(
        input_dir=folder_path,
        encoding="utf-8"
    ).load_data()
    logging.info(f"End loading documents from `{folder_path}` folder")
    
    return documents 

def create_document_nodes(documents, chunk_strategy: str = "token_split"):
    # TODO: add typing and doctring

    # text splitting strategy 
    text_splitter = TokenTextSplitter(chunk_size=512) # TODO: set the token size and token overlap as hyperparam of the script
    
    # create nodes 
    nodes = text_splitter.get_nodes_from_documents(
        documents=documents,
        show_progress=True
    )
    
    return nodes 


def index_documents(document_nodes: object, embedding_name: str) -> VectorStoreIndex:
    
    # create vector store
    embedding_dim = EMBEDDINGS_INFO[embedding_name]["dimension"] # get the embedding dimension from configs
    vector_store = FaissVectorStore(
        faiss_index=faiss.IndexFlatL2(embedding_dim)
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        
    )
    # create indexes 
    index = VectorStoreIndex(
        document_nodes,
        storage_context=storage_context,
        embed_model=HuggingFaceEmbedding(model_name=EMBEDDINGS_INFO[embedding_name]["model_id"]) 
    )

    # save index to disk 
    index.storage_context.persist(
        persist_dir="data/vectors" # TODO: migrate to an env variable
    )
    
    return index

def main(embedding_name: str, chunking_strategy: str) -> None:
    
    # load papers from document store
    documents = load_documents("data/papers") # TODO: set the document as en variable
    
    # create documents node
    document_nodes = create_document_nodes(documents)
    
    # index documents 
    index = index_documents(
        document_nodes,
        embedding_name=embedding_name
    )
    
    return index 

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Launch the ingestion process of documents"
    )
    
    # chuncking strategy 
    parser.add_argument(
        "--chunking_strategy",
        help="Document chunking strategy",
        type=str
    )
    
    # embedding name 
    parser.add_argument(
        "--embedding_name",
        help="Embedding model to be used",
        type=str
    )
    
    # get the args
    args = parser.parse_args()
    
    # init the ingestion proces 
    index = main(
        embedding_name=args.embedding_name,
        chunking_strategy=args.chunking_strategy
    )
    
    # test the created index
    query_engine = index.as_query_engine()
    test_response = query_engine.query("what is your name ?")
    
    test_response