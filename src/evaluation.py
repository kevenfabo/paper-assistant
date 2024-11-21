# Leverage frameworks like ragas to run the evaluation process: https://docs.ragas.io/en/stable/concepts/components/eval_dataset/

import os
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas import EvaluationDataset
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import  HuggingFaceLLM
from ragas.llms import LangchainLLMWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity, faithfulness, answer_correctness, answer_relevancy, context_recall, context_precision
from typing import List, Optional



os.environ["OPENAI_API_KEY"] = ""

LLM_INFO = {
    "smollama":{
        "model_name": "BEE-spoke-data/smol_llama-101M-GQA"
    },
    "tinyllama":{
        "model_name": "tinyllama"
    }

}


def load_generated_answers(file_path: str):
    #eval_dataset = pd.read_csv(file_path)
    # dataset = load_dataset(
    #     "explodinggradients/amnesty_qa",
    #     "english_v3",
    #     trust_remote_code=True
    # )
    #eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])[:10]
    data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts': [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
                 ['The Green Bay Packers...Green Bay, Wisconsin.', 'The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
    eval_dataset = Dataset.from_dict(data_samples)
    
    return eval_dataset

def main(results_path:str, model_name):
    eval_dataset = load_generated_answers(results_path)
    # evaluator_llm = LlamaIndexLLMWrapper(HuggingFaceLLM(model_name=model_name))
    # evaluator_embeddings = LlamaIndexEmbeddingsWrapper(HuggingFaceBgeEmbeddings(model_name=model_name))

    evaluator_llm = LlamaIndexLLMWrapper(Ollama(model=model_name, request_timeout=120.0))
    evaluator_embeddings = LlamaIndexEmbeddingsWrapper(OllamaEmbedding(model_name=model_name))


    metrics=[
           LLMContextRecall(llm=evaluator_llm),
           Faithfulness(llm=evaluator_llm),
           FactualCorrectness(llm=evaluator_llm),
           SemanticSimilarity(embeddings=evaluator_embeddings) 
    ]

    results = evaluate(
        dataset = eval_dataset, 
        metrics = metrics
    )
        
    return results.to_pandas()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers to questions"
    )

    parser.add_argument(
        "--results_path",
        help="Results file path, which contains questions, retrieved contexts and generated answers",
        default="data/results/results.csv",
        type=str
    )

    parser.add_argument(
        "--model_name",
        help="Embedding model to be used",
        default="tinyllama",
        type=str
    )

    args = parser.parse_args()
    model_name = LLM_INFO[args.model_name]["model_name"]
    results = main(args.results_path, model_name)
    print(results.head())

